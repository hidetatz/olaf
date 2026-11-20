"""T5 (relative position biases added to unscaled scores, RMS norms, encoder-decoder)."""

import numpy as np
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

import olaf
from olaf.examples import t5

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SOURCE, TARGET, STEPS = 4, 16, 12, 200
HF = dict(vocab_size=64, d_model=32, d_kv=16, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=2, relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0, feed_forward_proj="relu", tie_word_embeddings=True, decoder_start_token_id=0, pad_token_id=0)
CONFIG = t5.Config.from_huggingface(T5Config(**HF))


def _f64_norm(self, hidden_states):
    # T5LayerNorm with the variance in the input's type: Hugging Face computes it in f32 even in
    # an f64 model, and the f64 reference would carry that f32 rounding (1e-8 of the loss).
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def hf_model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(T5Config(**HF)).to(dtype=dtype, device=device).train()
    if dtype == torch.float64:
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        for module in model.modules():
            if isinstance(module, T5LayerNorm):
                module.forward = _f64_norm.__get__(module)
    return model


def named(tree) -> dict:
    """olaf's paths -> Hugging Face's: stacked blocks split per layer, each stack's position
    bias table in its first block."""
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        stack = path.split(".")[0]
        if path.startswith(f"{stack}.block."):
            rest = path.removeprefix(f"{stack}.block.")
            for i in range(value.shape[0]):
                out[f"{stack}.block.{i}.{rest}"] = value[i]
        elif path == f"{stack}.relative_attention_bias.weight":
            out[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = value
        else:
            out[path] = value
    return out


def torch_loss(model, ids, decoder_ids, labels):
    p = next(model.parameters())
    t = lambda a: torch.from_numpy(a).long().to(p.device)  # noqa: E731
    logits = model(input_ids=t(ids), decoder_input_ids=t(decoder_ids), use_cache=False).logits
    logits = logits if p.dtype == torch.float64 else logits.float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), t(labels).reshape(-1))


def data(steps, seed=0):
    """Four fixed batches, cycled: the target is the source's first TARGET tokens (a repeated
    pattern), the decoder input the target shifted right after the pad token."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(4):
        ids = np.tile(rng.integers(1, HF["vocab_size"], (BATCH, 4)), (1, SOURCE // 4)).astype(np.int32)
        labels = ids[:, :TARGET].copy()
        decoder_ids = np.concatenate([np.zeros((BATCH, 1), np.int32), labels[:, :-1]], axis=1)
        batches.append((ids, decoder_ids, labels))
    return [batches[k % 4] for k in range(steps)]


SPECS = (olaf.TensorType((BATCH, SOURCE), olaf.i32), olaf.TensorType((BATCH, TARGET), olaf.i32), olaf.TensorType((BATCH, TARGET), olaf.i32))


def case(compute=None, seed=1):
    model = t5.T5(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), t5.loss(model), hf_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_huggingface_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    inputs = data(1)[0]
    traced = olaf.trace(lambda p, *xs: olaf.value_and_grad(lambda p: c.olaf_loss(p, *xs))(p), specs, *SPECS)
    (loss, grads), _ = traced.eager(params64, *inputs)
    model = hf_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, *inputs)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=3e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_layer_stacks_are_written_in_place():
    # The unrolled scans stack each layer's weight gradients, bf16 products widened to f32: each
    # product stores its cast result straight into its slice of the stack (no concat, no copy of
    # a stack). The position bias is built once per stack, [heads, S, S], not per batch.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    B, S, T = 2, 64, 32
    config = t5.Config(vocab_size=320, d_model=128, d_kv=32, d_ff=256, num_layers=3, num_decoder_layers=3, num_heads=4)
    model = t5.T5(config, olaf.bf16)
    specs = (olaf.TensorType((B, S), olaf.i32), olaf.TensorType((B, T), olaf.i32), olaf.TensorType((B, T), olaf.i32))
    traced, _ = olaf.train_step(t5.loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    shapes = {b.name: b.type for b in program.buffers}
    assert not any(launch.task.name == "concat" for launch in program.launches)
    assert not any(launch.task.name == "copy" and any(shapes[n].rank == 3 and shapes[n].shape[0] == 3 for n in launch.buffers) for launch in program.launches), "a layer stack copied"
    assert any(t.shape == (4, S, S) for t in shapes.values()) and not any(launch.task.name == "gather" and any(shapes[n].shape == (B, 4, S, S) for n in launch.buffers) for launch in program.launches), "the position bias gathered per batch"
