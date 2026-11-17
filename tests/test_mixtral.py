"""P1.5 model 3: a small Mixtral (mixture of experts, top-2 routing by gather and scatter_add)."""

import numpy as np
import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM

import olaf
from olaf.examples import mixtral

import test_llama
from model_gate import MULTIPLIER, Case, curve, load, one_step, relative

BATCH, SEQ, STEPS = 2, 32, 200
HF = dict(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=SEQ, rms_norm_eps=1e-6, tie_word_embeddings=False, router_jitter_noise=0.0)
CONFIG = mixtral.Config.from_huggingface(MixtralConfig(**HF))


def hf_model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    # experts_implementation="eager": the per-expert loop of the module code above. The default
    # grouped matmul has no f64 kernel, and the truth runs in f64.
    return MixtralForCausalLM(MixtralConfig(**HF, attn_implementation="sdpa", experts_implementation="eager")).to(dtype=dtype, device=device).eval()


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        if path.startswith("model.layers."):
            for i in range(CONFIG.n_layer):
                out[f"model.layers.{i}.{path.removeprefix('model.layers.')}"] = value[i]
        else:
            out[path] = value
    return out


def data(steps, seed=0):
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, HF["vocab_size"], (BATCH, 4)).astype(np.int32)
    return [(np.tile(pattern, (1, SEQ // 4)),) for _ in range(steps)]


SPECS = (olaf.TensorType((BATCH, SEQ), olaf.i32),)


def case(compute=None, seed=1):
    model = mixtral.Mixtral(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), test_llama.olaf_loss(model), hf_model, test_llama.torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_top_k_matches_torch():
    probs = np.random.default_rng(0).random((16, 6)).astype(np.float32)
    probs[3, [1, 4]] = probs[3].max()  # a tie: the lower index first
    traced = olaf.trace(lambda p: mixtral.top_k(p, 2), olaf.TensorType((16, 6), olaf.f32))
    (indices, values), _ = traced.eager(probs)
    ref_values, ref_indices = torch.topk(torch.from_numpy(probs), 2, dim=-1)
    np.testing.assert_array_equal(np.stack(indices, 1), ref_indices.numpy())
    np.testing.assert_array_equal(np.stack(values, 1), ref_values.numpy())


def test_eager_forward_and_backward_match_huggingface_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    (ids,) = data(1)[0]
    traced = olaf.trace(lambda p, ids: olaf.value_and_grad(lambda p: c.olaf_loss(p, ids))(p), specs, SPECS[0])
    (loss, grads), _ = traced.eager(params64, ids)
    model = hf_model(torch.float64)
    load(model, named(c.params))
    reference = test_llama.torch_loss(model, ids)
    reference.backward()
    # As for Llama: the rotary tables are f32 in both (an ulp apart), the rest f64.
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
