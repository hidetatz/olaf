"""MobileNetV2 (depthwise convolutions, ReLU6, inverted residuals with batch norm)."""

import numpy as np
import pytest
import torch
import torch.nn as T

import olaf
from olaf.examples import mobilenet

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SIZE, STEPS = 16, 16, 100
CONFIG = mobilenet.Config(num_classes=10, stem=8, settings=((1, 8, 1, 1), (6, 12, 2, 2), (6, 16, 2, 2)), head=64)


class TorchBlock(T.Module):
    def __init__(self, cin, cout, t, stride):
        super().__init__()
        hidden = cin * t
        if t != 1:
            self.expand, self.expand_bn = T.Conv2d(cin, hidden, 1, bias=False), T.BatchNorm2d(hidden)
        self.t = t
        self.depthwise, self.depthwise_bn = T.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), T.BatchNorm2d(hidden)
        self.project, self.project_bn = T.Conv2d(hidden, cout, 1, bias=False), T.BatchNorm2d(cout)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        h = F_relu6(self.expand_bn(self.expand(x))) if self.t != 1 else x
        h = self.project_bn(self.project(F_relu6(self.depthwise_bn(self.depthwise(h)))))
        return x + h if self.residual else h


def F_relu6(x):  # noqa: N802
    return torch.nn.functional.relu6(x)


class TorchMobileNet(T.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        self.stem, self.stem_bn = T.Conv2d(3, c.stem, 3, c.stem_stride, 1, bias=False), T.BatchNorm2d(c.stem)
        self.n = len(c.blocks())
        for i, (cin, cout, t, s) in enumerate(c.blocks()):
            setattr(self, f"block{i}", TorchBlock(cin, cout, t, s))
        self.head, self.head_bn = T.Conv2d(c.blocks()[-1][1], c.head, 1, bias=False), T.BatchNorm2d(c.head)
        self.classifier = T.Linear(c.head, c.num_classes)

    def forward(self, x):
        h = F_relu6(self.stem_bn(self.stem(x)))
        for i in range(self.n):
            h = getattr(self, f"block{i}")(h)
        h = F_relu6(self.head_bn(self.head(h)))
        return self.classifier(h.mean(dim=(2, 3)))


def torch_model(dtype, device):
    return TorchMobileNet().to(dtype=dtype, device=device)


def olaf_loss(model):
    def loss(params, buffers, x, y):
        logits, new = model(params, buffers, x)
        return olaf.cross_entropy(olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits, y), new

    return loss


def torch_loss(model, x, y):
    p = next(model.parameters())
    logits = model(torch.from_numpy(x).to(device=p.device, dtype=p.dtype))
    return torch.nn.functional.cross_entropy(logits.float() if p.dtype != torch.float64 else logits, torch.from_numpy(y).long().to(p.device))


def data(steps, seed=0):
    """Four fixed batches in turn, with random labels: memorizing them makes the loss fall."""
    rng = np.random.default_rng(seed)
    batches = [(rng.normal(size=(BATCH, 3, SIZE, SIZE)).astype(np.float32), rng.integers(0, 10, BATCH).astype(np.int32)) for _ in range(4)]
    return [batches[i % 4] for i in range(steps)]


SPECS = (olaf.TensorType((BATCH, 3, SIZE, SIZE), olaf.f32), olaf.TensorType((BATCH,), olaf.i32))


def case(compute=None, seed=1):
    model = mobilenet.MobileNetV2(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, buffers=model.init_buffers())


def test_eager_forward_backward_and_running_statistics_match_torch_f64():
    c = case()
    model = mobilenet.MobileNetV2(CONFIG)
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    buffers64 = olaf.tree_map(lambda a: a.astype(np.float64), c.buffers)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    bspecs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), buffers64)
    x, y = data(1)[0]

    def step(p, b, x, y):
        (loss, new), grads = olaf.value_and_grad(lambda p: olaf_loss(model)(p, b, x, y), has_aux=True)(p)
        return loss, new, grads

    traced = olaf.trace(step, specs, bspecs, olaf.TensorType(x.shape, olaf.f64), SPECS[1])
    (loss, new, grads), _ = traced.eager(params64, buffers64, x.astype(np.float64), y)
    twin = torch_model(torch.float64, "cpu")
    load(twin, c.torch_named(c.params))
    reference = torch_loss(twin, x.astype(np.float64), y)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = c.torch_named(grads)
    for name, p in twin.named_parameters():
        g = p.grad.numpy()
        if np.linalg.norm(g) < 1e-14:
            # A projection's batch-norm bias feeding a block without a residual: the next
            # convolution's batch norm subtracts the batch mean, so its gradient is zero and both
            # sides hold only rounding residue.
            assert np.linalg.norm(ours[name]) < 1e-14, name
            continue
        assert relative(ours[name], g) < 1e-10, name
    running = c.torch_named(new)
    for name, buf in twin.named_buffers():
        if name.endswith(("running_mean", "running_var")):
            if np.linalg.norm(buf.numpy()) < 1e-14:
                # A batch norm right after one whose bias is zero: its input's batch mean is zero.
                assert np.linalg.norm(running[name]) < 1e-14, name
                continue
            assert relative(running[name], buf.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    # lr 1e-3 over 100 steps: at 2e-3 over 200 the loss fell to 0.01 and the runs became
    # chaotic, PyTorch's own f32 curve 2e-4 to 3e-3 from the f64 one depending on the seed, and
    # the gate compared luck (olaf and PyTorch within 0.9x of each other here).
    ours = lambda t: olaf.AdamW(t, lr=1e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.9 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_depthwise_convolutions_are_stencils():
    # Depthwise layers (one channel per group), forward, input and weight gradients: no grouped
    # matrix products (a reduction 9 long per group) and no per-group launches; the weight
    # gradient is one grouped convolution (frontend/autodiff), lowered as a reduction.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    model = mobilenet.MobileNetV2(CONFIG, olaf.bf16)
    traced, _ = olaf.train_step(olaf_loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *SPECS, buffers=model.init_buffers())
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    specs = [launch.task.origin[1] for launch in program.launches if launch.task.origin and launch.task.origin[0] == "contract"]
    assert not any("->go" in s for s in specs), specs
    # Per-group weight gradients were a convolution per channel: 1635 launches for this model,
    # where its 7 blocks now take about 400 (every channel of every depthwise layer adds none).
    hidden = sum(cin * t for cin, _, t, _ in CONFIG.blocks())
    assert len(program.launches) < 2 * hidden, (len(program.launches), hidden)
