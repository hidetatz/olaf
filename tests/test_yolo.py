"""P1.5 model 7: a small YOLO (v1's loss: responsible box by IoU, chosen with where)."""

import numpy as np
import pytest
import torch
import torch.nn as T

import olaf
from olaf.examples import yolo

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, STEPS, SIZE = 8, 200, 64
CONFIG = yolo.Config()


class TorchYolo(T.Module):
    def __init__(self):
        super().__init__()
        c, cin = CONFIG, CONFIG.in_channels
        for i, w in enumerate(c.widths):
            setattr(self, f"conv{i + 1}", T.Conv2d(cin, w, 3, 1, 1, bias=False))
            setattr(self, f"bn{i + 1}", T.BatchNorm2d(w))
            cin = w
        self.head = T.Conv2d(cin, c.boxes * 5 + c.classes, 1)

    def forward(self, x):
        h = x
        for i in range(len(CONFIG.widths)):
            h = getattr(self, f"bn{i + 1}")(getattr(self, f"conv{i + 1}")(h))
            h = torch.nn.functional.max_pool2d(torch.nn.functional.leaky_relu(h, 0.1), 2)
        return self.head(h).permute(0, 2, 3, 1)


def torch_iou(a, b, S):
    def corners(box):
        x, y, w, h = box
        return x - w * S / 2, y - h * S / 2, x + w * S / 2, y + h * S / 2

    ax0, ay0, ax1, ay1 = corners(a)
    bx0, by0, bx1, by1 = corners(b)
    inter = (torch.minimum(ax1, bx1) - torch.maximum(ax0, bx0)).clamp(min=0) * (torch.minimum(ay1, by1) - torch.maximum(ay0, by0)).clamp(min=0)
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter + 1e-9)


def torch_yolo_loss(raw, obj, box, cls):
    p = raw.float() if raw.dtype == torch.bfloat16 else raw
    S, C = CONFIG.grid, CONFIG.classes
    t = [box[..., i] for i in range(4)]
    preds = [[torch.sigmoid(p[..., 5 * b + i]) for i in range(5)] for b in range(2)]
    with torch.no_grad():
        first = torch_iou(preds[0][:4], t, S) >= torch_iou(preds[1][:4], t, S)
    owners = (torch.where(first, obj, torch.zeros_like(obj)), torch.where(first, torch.zeros_like(obj), obj))
    total = 0.0
    for (x, y, w, h, conf), r in zip(preds, owners):
        coord = (x - t[0]) ** 2 + (y - t[1]) ** 2 + (w.sqrt() - t[2].sqrt()) ** 2 + (h.sqrt() - t[3].sqrt()) ** 2
        total = total + (r * (coord * yolo.COORD + (conf - 1) ** 2)).sum() + ((1 - r) * conf * conf).sum() * yolo.NOOBJ
    ce = torch.nn.functional.cross_entropy(p[..., 10:].reshape(-1, C), cls.reshape(-1), reduction="none").reshape(obj.shape)
    return (total + (obj * ce).sum()) / p.shape[0]


def torch_model(dtype, device):
    return TorchYolo().to(dtype=dtype, device=device)


def torch_loss(model, x, obj, box, cls):
    p = next(model.parameters())
    as_t = lambda a, dt=p.dtype: torch.from_numpy(a).to(device=p.device, dtype=dt)  # noqa: E731
    target_dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
    return torch_yolo_loss(model(as_t(x)), as_t(obj, target_dtype), as_t(box, target_dtype), as_t(cls, torch.long))


def olaf_loss(model):
    def loss(params, buffers, x, obj, box, cls):
        raw, new = model(params, buffers, x)
        return yolo.loss(CONFIG, raw, obj, box, cls), new

    return loss


def data(steps, seed=0):
    """Four fixed batches in turn: noise images, each with one coloured square (its class is
    the colour channel); targets for the cell holding its centre."""
    rng = np.random.default_rng(seed)
    S = CONFIG.grid
    batches = []
    for _ in range(4):
        x = (rng.normal(size=(BATCH, 3, SIZE, SIZE)) * 0.1).astype(np.float32)
        obj = np.zeros((BATCH, S, S), np.float32)
        box = np.full((BATCH, S, S, 4), 0.5, np.float32)
        cls = np.zeros((BATCH, S, S), np.int32)
        for n in range(BATCH):
            c = rng.integers(0, 3)
            w, h = rng.uniform(0.15, 0.4, 2)
            cx, cy = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
            x0, x1, y0, y1 = (int(v * SIZE) for v in (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2))
            x[n, c, y0:y1, x0:x1] += 1.0
            i, j = int(cy * S), int(cx * S)
            obj[n, i, j] = 1.0
            box[n, i, j] = (cx * S - j, cy * S - i, w, h)
            cls[n, i, j] = c
        batches.append((x, obj, box, cls))
    return [batches[k % 4] for k in range(steps)]


SPECS = (olaf.TensorType((BATCH, 3, SIZE, SIZE), olaf.f32), olaf.TensorType((BATCH, 8, 8), olaf.f32), olaf.TensorType((BATCH, 8, 8, 4), olaf.f32), olaf.TensorType((BATCH, 8, 8), olaf.i32))


def case(compute=None, seed=1):
    model = yolo.Yolo(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, buffers=model.init_buffers())


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    model = yolo.Yolo(CONFIG)
    to64 = lambda t: olaf.tree_map(lambda a: a.astype(np.float64), t)  # noqa: E731
    spec = lambda t: olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), t)  # noqa: E731
    x, obj, box, cls = data(1)[0]
    inputs64 = (x.astype(np.float64), obj.astype(np.float64), box.astype(np.float64), cls)
    specs = (*(olaf.TensorType(a.shape, olaf.f64) for a in inputs64[:3]), SPECS[3])

    def step(p, b, *xs):
        (value, _), grads = olaf.value_and_grad(lambda p: olaf_loss(model)(p, b, *xs), has_aux=True)(p)
        return value, grads

    (value, grads), _ = olaf.trace(step, spec(c.params), spec(c.buffers), *specs).eager(to64(c.params), to64(c.buffers), *inputs64)
    twin = torch_model(torch.float64, "cpu")
    load(twin, c.torch_named(c.params))
    reference = torch_loss(twin, *inputs64)
    reference.backward()
    assert abs(float(value) - reference.item()) < 1e-10 * abs(reference.item())
    ours = c.torch_named(grads)
    for name, p in twin.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=1e-3, weight_decay=0.0)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios
