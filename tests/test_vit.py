"""ViT (patch embedding, class token, sequence lengths off the attention tiles)."""

import numpy as np
import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

import olaf
from olaf.examples import vit

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, STEPS = 16, 100
HF = dict(image_size=16, patch_size=4, num_channels=3, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, num_labels=10)
CONFIG = vit.Config.from_huggingface(ViTConfig(**HF), HF["num_labels"])
STACK = "vit.layers."


def hf_model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    return ViTForImageClassification(ViTConfig(**HF, attn_implementation="sdpa")).to(dtype=dtype, device=device).train()


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        if path.startswith(STACK):
            for i in range(value.shape[0]):
                out[f"{STACK}{i}.{path.removeprefix(STACK)}"] = value[i]
        else:
            out[path] = value
    return out


def olaf_loss(model):
    def loss(params, x, y):
        logits = model(params, x)
        return olaf.cross_entropy(olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits, y)

    return loss


def torch_loss(model, x, y):
    p = next(model.parameters())
    logits = model(pixel_values=torch.from_numpy(x).to(device=p.device, dtype=p.dtype)).logits
    return torch.nn.functional.cross_entropy(logits.float() if p.dtype != torch.float64 else logits, torch.from_numpy(y).long().to(p.device))


def data(steps, seed=0):
    """Four fixed batches in turn, with random labels: memorizing them makes the loss fall."""
    rng = np.random.default_rng(seed)
    size = HF["image_size"]
    batches = [(rng.normal(size=(BATCH, 3, size, size)).astype(np.float32), rng.integers(0, 10, BATCH).astype(np.int32)) for _ in range(4)]
    return [batches[i % 4] for i in range(steps)]


SPECS = (olaf.TensorType((BATCH, 3, HF["image_size"], HF["image_size"]), olaf.f32), olaf.TensorType((BATCH,), olaf.i32))


def case(compute=None, seed=1):
    model = vit.ViT(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), hf_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_huggingface_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, y = data(1)[0]
    traced = olaf.trace(lambda p, x, y: olaf.value_and_grad(lambda p: c.olaf_loss(p, x, y))(p), specs, olaf.TensorType(x.shape, olaf.f64), SPECS[1])
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), y)
    model = hf_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, x.astype(np.float64), y)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        g = p.grad.numpy()
        if np.linalg.norm(g) < 1e-15:
            # The key biases: softmax ignores a constant added to a row of scores.
            assert np.linalg.norm(ours[name]) < 1e-15, name
            continue
        assert relative(ours[name], g) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=1e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.9 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_attention_fuses_over_65_tokens():
    # 64 patches and the class token: a sequence off the 64-wide attention tiles still runs as
    # the fused attention tasks, and no scores are materialized.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    config = vit.Config(image_size=32, patch_size=4, hidden_size=128, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, num_labels=10)
    model = vit.ViT(config, olaf.bf16)
    specs = (olaf.TensorType((8, 3, 32, 32), olaf.f32), olaf.TensorType((8,), olaf.i32))
    traced, _ = olaf.train_step(olaf_loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    names = {launch.task.name for launch in program.launches}
    assert {"attention", "attention_dkv"} <= names
    assert not any(b.type.rank >= 3 and b.type.shape[-2:] == (65, 65) for b in program.buffers), "scores materialized"
