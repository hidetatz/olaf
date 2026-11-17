"""P1.5 model 5: a small ResNet (CIFAR stem, basic blocks, batch norm with running statistics)."""

import numpy as np
import pytest
import torch
import torch.nn as T

import olaf
from olaf.examples import resnet

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, STEPS = 16, 200
CONFIG = resnet.Config(blocks=(1, 1, 1), widths=(16, 32, 64), num_classes=10, stem="cifar")


class Block(T.Module):
    """torchvision's BasicBlock."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1, self.bn1 = T.Conv2d(cin, cout, 3, stride, 1, bias=False), T.BatchNorm2d(cout)
        self.conv2, self.bn2 = T.Conv2d(cout, cout, 3, 1, 1, bias=False), T.BatchNorm2d(cout)
        self.downsample = T.Sequential(T.Conv2d(cin, cout, 1, stride, bias=False), T.BatchNorm2d(cout)) if stride != 1 or cin != cout else None

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + (self.downsample(x) if self.downsample is not None else x))


class TorchResNet(T.Module):
    def __init__(self):
        super().__init__()
        c = CONFIG
        self.conv1, self.bn1 = T.Conv2d(3, c.widths[0], 3, 1, 1, bias=False), T.BatchNorm2d(c.widths[0])
        cin = c.widths[0]
        for i, (n, width) in enumerate(zip(c.blocks, c.widths)):
            blocks = []
            for j in range(n):
                blocks.append(Block(cin, width, 2 if j == 0 and i > 0 else 1))
                cin = width
            setattr(self, f"layer{i + 1}", T.Sequential(*blocks))
        self.fc = T.Linear(cin, c.num_classes)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        for i in range(len(CONFIG.blocks)):
            h = getattr(self, f"layer{i + 1}")(h)
        return self.fc(h.mean(dim=(2, 3)))


def torch_model(dtype, device):
    return TorchResNet().to(dtype=dtype, device=device)  # train mode: batch statistics


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
    batches = [(rng.normal(size=(BATCH, 3, 32, 32)).astype(np.float32), rng.integers(0, 10, BATCH).astype(np.int32)) for _ in range(4)]
    return [batches[i % 4] for i in range(steps)]


SPECS = (olaf.TensorType((BATCH, 3, 32, 32), olaf.f32), olaf.TensorType((BATCH,), olaf.i32))


def case(compute=None, seed=1):
    model = resnet.ResNet(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, buffers=model.init_buffers())


def test_eager_forward_backward_and_running_statistics_match_torch_f64():
    c = case()
    model = resnet.ResNet(CONFIG)
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
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name
    running = c.torch_named(new)
    for name, buf in twin.named_buffers():
        if name.endswith(("running_mean", "running_var")):
            assert relative(running[name], buf.numpy()) < 1e-10, name


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


def _eval_case(seed=3):
    model = resnet.ResNet(CONFIG)
    rng = np.random.default_rng(seed)
    params = model.init(rng)
    # Running statistics away from their initial values, so evaluation mode is exercised.
    buffers = olaf.tree_map(lambda a: (rng.normal(size=a.shape) * 0.1 if np.all(a == 0) else rng.uniform(0.5, 1.5, a.shape)).astype(np.float32), model.init_buffers())
    x = rng.normal(size=(4, 3, 32, 32)).astype(np.float32)
    classes = rng.integers(0, 10, 4).astype(np.int32)
    return model, params, buffers, x, classes


def torch_grad_cam(params, buffers, x, classes):
    twin = torch_model(torch.float64, "cpu").eval()
    from olaf.frontend.optim import _paths

    state = {**dict(zip(_paths(params), olaf.tree_flatten(params)[0])), **dict(zip(_paths(buffers), olaf.tree_flatten(buffers)[0]))}
    twin.load_state_dict({k: torch.from_numpy(np.asarray(v, np.float64)) for k, v in state.items()}, strict=False)
    h = torch.relu(twin.bn1(twin.conv1(torch.from_numpy(x).double())))
    for i in range(len(CONFIG.blocks)):
        h = getattr(twin, f"layer{i + 1}")(h)
    logits = twin.fc(h.mean(dim=(2, 3)))
    score = logits[torch.arange(len(classes)), torch.from_numpy(classes).long()].sum()
    (grads,) = torch.autograd.grad(score, h)
    cam = torch.relu((h * grads.mean(dim=(2, 3), keepdim=True)).sum(dim=1, keepdim=True))
    return torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False).detach().numpy()


def test_grad_cam_matches_torch():
    model, params, buffers, x, classes = _eval_case()
    to64 = lambda t: olaf.tree_map(lambda a: a.astype(np.float64), t)  # noqa: E731
    spec = lambda t: olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), t)  # noqa: E731
    traced = olaf.trace(lambda p, b, x, c: resnet.grad_cam(model, p, b, x, c), spec(params), spec(buffers), olaf.TensorType(x.shape, olaf.f64), olaf.TensorType(classes.shape, olaf.i32))
    got, _ = traced.eager(to64(params), to64(buffers), x.astype(np.float64), classes)
    assert relative(got, torch_grad_cam(params, buffers, x, classes)) < 1e-10


@pytest.mark.gpu
def test_grad_cam_on_device_matches_eager():
    from olaf.backend.cuda.runtime import CudaProgram
    from olaf.graph.interp import interpret
    from olaf.pipeline import lower_graph, to_program

    model, params, buffers, x, classes = _eval_case()
    spec = lambda t: olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f32), t)  # noqa: E731
    graph = lower_graph(olaf.trace(lambda p, b, x, c: resnet.grad_cam(model, p, b, x, c), spec(params), spec(buffers), olaf.TensorType(x.shape, olaf.f32), olaf.TensorType(classes.shape, olaf.i32)).graph)
    arrays = [*olaf.tree_flatten(params)[0], *olaf.tree_flatten(buffers)[0], x, classes]
    (want,), _ = interpret(graph, arrays)
    (got,), _ = CudaProgram(to_program(graph)).run(arrays)
    assert relative(got, want) < 1e-5
