"""P1.5 model 1: an MNIST-shaped MLP (784 -> 256 -> 10) on synthetic data."""

import numpy as np
import pytest
import torch

import olaf
from olaf import nn

from model_gate import MULTIPLIER, Case, curve, load, one_step, relative

BATCH, STEPS = 64, 200


def mlp(compute=None):
    return nn.Sequential(nn.Linear(784, 256, compute=compute), nn.relu, nn.Linear(256, 10, compute=compute))


def olaf_loss(model):
    def loss(params, x, y):
        logits = model(params, x)
        # As the PyTorch twin: autocast's bf16 logits go to the loss in f32.
        return olaf.cross_entropy(olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits, y)

    return loss


def torch_model(dtype, device):
    return torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10)).to(dtype=dtype, device=device)


def torch_loss(model, x, y):
    p = next(model.parameters())
    logits = model(torch.from_numpy(x).to(device=p.device, dtype=p.dtype))
    return torch.nn.functional.cross_entropy(logits.float() if p.dtype != torch.float64 else logits, torch.from_numpy(y).long().to(p.device))


def data(steps, seed=0):
    """Inputs labelled by a fixed random two-layer teacher, so the task is learnable."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=(784, 64)) / 28.0, rng.normal(size=(64, 10))
    out = []
    for _ in range(steps):
        x = rng.normal(size=(BATCH, 784)).astype(np.float32)
        out.append((x, np.argmax(np.maximum(x @ w1, 0.0) @ w2, axis=1).astype(np.int32)))
    return out


SPECS = (olaf.TensorType((BATCH, 784), olaf.f32), olaf.TensorType((BATCH,), olaf.i32))


def case(compute=None, seed=1):
    model = mlp(compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16)


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, y = data(1)[0]
    traced = olaf.trace(lambda p, x, y: olaf.value_and_grad(lambda p: c.olaf_loss(p, x, y))(p), specs, olaf.TensorType((BATCH, 784), olaf.f64), SPECS[1])
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), y)
    model = torch_model(torch.float64, "cpu")
    load(model, c.params)
    reference = torch_loss(model, x.astype(np.float64), y)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    for name, p in model.named_parameters():
        layer, leaf = name.split(".")
        assert relative(grads[layer][leaf], p.grad.numpy()) < 1e-10, name


OPTIMIZERS = {
    "sgd": (lambda t: olaf.SGD(t, lr=0.05, momentum=0.9), lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)),
    "adamw": (lambda t: olaf.AdamW(t, lr=1e-3, weight_decay=0.01), lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)),
}


@pytest.mark.gpu
@pytest.mark.parametrize("compute,optimizer", [(None, "sgd"), (None, "adamw"), (olaf.bf16, "adamw")])
def test_training_gates(compute, optimizer):
    ours, theirs = OPTIMIZERS[optimizer]
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.7 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    c = case(compute)
    worst, ratios = one_step(c, c.params, data(1, seed=7)[0])
    assert worst <= MULTIPLIER, ratios
