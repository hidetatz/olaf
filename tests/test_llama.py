"""P1.5 model 2: a small Llama (RMSNorm, rotary embeddings, SwiGLU, grouped-query attention)."""

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import olaf
from olaf.examples import llama

from model_gate import MULTIPLIER, Case, curve, load, one_step, relative

BATCH, SEQ, STEPS = 4, 32, 200
HF = dict(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=SEQ, rms_norm_eps=1e-6, tie_word_embeddings=False)


def hf_model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**HF, attn_implementation="sdpa")).to(dtype=dtype, device=device).eval()


CONFIG = llama.Config.from_huggingface(LlamaConfig(**HF))


def named(tree) -> dict:
    """olaf's parameters (layers stacked) as Hugging Face's per-layer names."""
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        if path.startswith("model.layers."):
            rest = path.removeprefix("model.layers.")
            for i in range(CONFIG.n_layer):
                out[f"model.layers.{i}.{rest}"] = value[i]
        else:
            out[path] = value
    return out


def olaf_loss(model):
    def loss(params, ids):
        logits = model(params, ids)
        logits = olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:])

    return loss


def torch_loss(model, ids):
    p = next(model.parameters())
    t = torch.from_numpy(ids).long().to(p.device)
    logits = model(t, use_cache=False).logits
    logits = logits if p.dtype == torch.float64 else logits.float()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))


def data(steps, seed=0):
    """Repeating short patterns, so the loss falls and each update matters."""
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, HF["vocab_size"], (BATCH, 4)).astype(np.int32)
    return [(np.tile(pattern, (1, SEQ // 4)),) for _ in range(steps)]


SPECS = (olaf.TensorType((BATCH, SEQ), olaf.i32),)


def case(compute=None, seed=1):
    model = llama.Llama(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), lambda dtype, device: hf_model(dtype, device), torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_huggingface_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    (ids,) = data(1)[0]
    traced = olaf.trace(lambda p, ids: olaf.value_and_grad(lambda p: c.olaf_loss(p, ids))(p), specs, SPECS[0])
    (loss, grads), _ = traced.eager(params64, ids)
    model = hf_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, ids)
    reference.backward()
    # Hugging Face computes the rotary tables in f32 even in a double model, and so does olaf;
    # the two f32 cosines may differ by an ulp (NumPy's against PyTorch's), which moves the
    # results by ~1e-8. 1e-6 still catches any wrong term (they are O(1)).
    assert abs(float(loss) - reference.item()) < 1e-6 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-6, name


@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=3e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    c = case(compute)
    worst, ratios = one_step(c, c.params, data(1, seed=7)[0])
    assert worst <= MULTIPLIER, ratios


def test_fusion_at_llama_shapes():
    # The shapes the small gate model does not reach: a 2048-wide hidden state (rows past 1024),
    # head dim 128, four query heads per key/value head. Built only, not run.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    S, I = 256, 5632
    cfg = llama.Config(vocab_size=320, hidden_size=2048, intermediate_size=I, n_layer=1, n_head=16, n_kv_head=4)  # vocab != S: logits are not score-shaped
    model = llama.Llama(cfg, olaf.bf16)
    traced, _ = olaf.train_step(olaf_loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), olaf.TensorType((1, S), olaf.i32))
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    shapes = {b.name: b.type for b in program.buffers}
    names = [launch.task.name for launch in program.launches]
    assert {"attention", "attention_dkv"} <= set(names)
    assert not any(t.shape[-2:] == (S, S) for t in shapes.values()), "attention scores materialized"
    assert names.count("rows") >= 4, "RMSNorm forward and backward fuse into row tasks"
    forward = next(launch for launch in program.launches if launch.task.name == "attention")
    assert any(shapes[n].shape == (1, 4, S, 128) for n in forward.buffers), "the forward reads unrepeated keys"
    assert not any(launch.task.name == "copy" and any(shapes[n].rank == 5 and shapes[n].dtype is olaf.bf16 for n in launch.buffers) for launch in program.launches), "keys or values copied per query head"
    assert not any(t.dtype is olaf.f32 and t.shape == (1, S, I) for t in shapes.values()), "f32 copy of an MLP activation"
