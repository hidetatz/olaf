"""ConvNeXt (7x7 depthwise convolutions, channels-last layer norms and MLPs, layer scale)."""

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F

import olaf
from olaf.examples import convnext

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SIZE, STEPS = 16, 16, 100
CONFIG = convnext.Config(num_classes=10, patch=2, depths=(1, 1, 2), dims=(16, 32, 48), layer_scale=0.5)


class TorchBlock(T.Module):
    def __init__(self, dim, c):
        super().__init__()
        self.dwconv = T.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = T.LayerNorm(dim, eps=c.eps)
        self.pwconv1, self.pwconv2 = T.Linear(dim, 4 * dim), T.Linear(4 * dim, dim)
        self.gamma = T.Parameter(torch.full((dim,), c.layer_scale))

    def forward(self, x):
        h = self.dwconv(x).permute(0, 2, 3, 1)
        h = self.gamma * self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + h.permute(0, 3, 1, 2)


def _channel_norm(norm, x):
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TorchConvNeXt(T.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        self.c = c
        self.stem, self.stem_norm = T.Conv2d(3, c.dims[0], c.patch, c.patch), T.LayerNorm(c.dims[0], eps=c.eps)
        for i, (depth, dim) in enumerate(zip(c.depths, c.dims)):
            if i > 0:
                setattr(self, f"down{i}_norm", T.LayerNorm(c.dims[i - 1], eps=c.eps))
                setattr(self, f"down{i}", T.Conv2d(c.dims[i - 1], dim, 2, 2))
            for j in range(depth):
                setattr(self, f"stage{i}_{j}", TorchBlock(dim, c))
        self.norm, self.head = T.LayerNorm(c.dims[-1], eps=c.eps), T.Linear(c.dims[-1], c.num_classes)

    def forward(self, x):
        h = _channel_norm(self.stem_norm, self.stem(x))
        for i, depth in enumerate(self.c.depths):
            if i > 0:
                h = getattr(self, f"down{i}")(_channel_norm(getattr(self, f"down{i}_norm"), h))
            for j in range(depth):
                h = getattr(self, f"stage{i}_{j}")(h)
        return self.head(self.norm(h.mean(dim=(2, 3))))


def torch_model(dtype, device):
    return TorchConvNeXt().to(dtype=dtype, device=device)


def olaf_loss(model):
    def loss(params, x, y):
        logits = model(params, x)
        return olaf.cross_entropy(olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits, y)

    return loss


def torch_loss(model, x, y):
    p = next(model.parameters())
    logits = model(torch.from_numpy(x).to(device=p.device, dtype=p.dtype))
    return F.cross_entropy(logits.float() if p.dtype != torch.float64 else logits, torch.from_numpy(y).long().to(p.device))


def data(steps, seed=0):
    """Four fixed batches in turn, with random labels: memorizing them makes the loss fall."""
    rng = np.random.default_rng(seed)
    batches = [(rng.normal(size=(BATCH, 3, SIZE, SIZE)).astype(np.float32), rng.integers(0, 10, BATCH).astype(np.int32)) for _ in range(4)]
    return [batches[i % 4] for i in range(steps)]


SPECS = (olaf.TensorType((BATCH, 3, SIZE, SIZE), olaf.f32), olaf.TensorType((BATCH,), olaf.i32))


def case(compute=None, seed=1):
    model = convnext.ConvNeXt(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16)


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, y = data(1)[0]
    traced = olaf.trace(lambda p, x, y: olaf.value_and_grad(lambda p: c.olaf_loss(p, x, y))(p), specs, olaf.TensorType(x.shape, olaf.f64), SPECS[1])
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), y)
    twin = torch_model(torch.float64, "cpu")
    load(twin, c.torch_named(c.params))
    reference = torch_loss(twin, x.astype(np.float64), y)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = c.torch_named(grads)
    for name, p in twin.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


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


def test_mlp_rows_over_small_planes_are_one_label():
    # At 4 x 4 positions the block MLP's rows (n, h, w) are 1024 contiguous rows: the products
    # take them as one label (a plain matrix product), not rows spanning three labels.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    config = convnext.Config(num_classes=10, patch=2, depths=(1, 1), dims=(64, 128))
    model = convnext.ConvNeXt(config, olaf.bf16)
    specs = (olaf.TensorType((64, 3, 16, 16), olaf.f32), olaf.TensorType((64,), olaf.i32))
    traced, _ = olaf.train_step(olaf_loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    contracts = [launch.task.origin[1] for launch in program.launches if launch.task.origin and launch.task.origin[0] == "contract"]
    # (Reduced labels stay apart, as weight gradients' split-K moves the outer one.)
    assert not any("abc" in s.split("->")[1] for s in contracts), contracts
