"""GAT (edge attention normalized per destination, message passing by gather and scatter_add)."""

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F

import olaf
from olaf.examples import gat

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

NODES, DEGREE, CLASSES, STEPS = 96, 6, 4, 100
CONFIG = gat.Config(in_features=16, hidden=8, heads=4, num_layers=3, num_classes=CLASSES)


class TorchGATConv(T.Module):
    """PyG's GATConv (no dropout, self-loops added by the caller), with index_add_ and the
    per-destination maximum PyG's softmax subtracts."""

    def __init__(self, fan_in, out, heads, concat):
        super().__init__()
        self.heads, self.out, self.concat = heads, out, concat
        self.lin = T.Linear(fan_in, heads * out, bias=False)
        self.att_src, self.att_dst = T.Parameter(torch.zeros(1, heads, out)), T.Parameter(torch.zeros(1, heads, out))
        self.bias = T.Parameter(torch.zeros(heads * out if concat else out))

    def forward(self, x, src, dst):
        N, H, C = x.shape[0], self.heads, self.out
        h = self.lin(x).view(N, H, C)
        e = F.leaky_relu((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst], 0.2)
        top = torch.full((N, H), -torch.inf, dtype=e.dtype, device=e.device).scatter_reduce(0, dst[:, None].expand(-1, H), e, "amax")
        w = torch.exp(e - top[dst])
        alpha = w / torch.zeros(N, H, dtype=w.dtype, device=w.device).index_add(0, dst, w)[dst]
        msg = h[src] * alpha[..., None]
        out = torch.zeros(N, H, C, dtype=msg.dtype, device=msg.device).index_add(0, dst, msg)
        out = out.reshape(N, H * C) if self.concat else out.mean(1)
        return out + self.bias


class TorchGAT(T.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        dims = [c.in_features] + [c.hidden * c.heads] * (c.num_layers - 1)
        self.convs = T.ModuleList([TorchGATConv(d, c.hidden, c.heads, True) for d in dims[:-1]] + [TorchGATConv(dims[-1], c.num_classes, 1, False)])

    def forward(self, x, src, dst):
        loops = torch.arange(x.shape[0], device=x.device)
        src, dst = torch.cat([src, loops]), torch.cat([dst, loops])
        for i, conv in enumerate(self.convs):
            x = conv(x, src, dst)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    return dict(zip(_paths(tree), olaf.tree_flatten(tree)[0]))


def torch_model(dtype=torch.float32, device="cpu"):
    return TorchGAT().to(dtype=dtype, device=device).train()


def torch_loss(model, x, src, dst, labels):
    p = next(model.parameters())
    dev = p.device
    logits = model(torch.from_numpy(x).to(device=dev, dtype=p.dtype), torch.from_numpy(src).long().to(dev), torch.from_numpy(dst).long().to(dev))
    logits = logits if p.dtype == torch.float64 else logits.float()
    return F.cross_entropy(logits, torch.from_numpy(labels).long().to(dev))


def graph(rng, nodes, degree, classes, features):
    """A planted-partition graph: every node gets `degree` incoming edges, 80% from its own
    class; features are a noisy class signature. Edge lists have a fixed length, as specs need."""
    labels = rng.integers(0, classes, nodes)
    signature = rng.normal(size=(classes, features))
    x = (signature[labels] + 1.5 * rng.normal(size=(nodes, features))).astype(np.float32)
    members = [np.flatnonzero(labels == k) for k in range(classes)]
    dst = np.repeat(np.arange(nodes), degree)
    same = rng.random(dst.shape) < 0.8
    src = np.where(same, [rng.choice(members[labels[i]]) for i in dst], rng.integers(0, nodes, dst.shape))
    return x, src.astype(np.int32), dst.astype(np.int32), labels.astype(np.int32)


def data(steps, seed=0):
    """Four fixed graphs, cycled."""
    rng = np.random.default_rng(seed)
    graphs = [graph(rng, NODES, DEGREE, CLASSES, CONFIG.in_features) for _ in range(4)]
    return [graphs[k % 4] for k in range(steps)]


SPECS = (
    olaf.TensorType((NODES, CONFIG.in_features), olaf.f32),
    olaf.TensorType((NODES * DEGREE,), olaf.i32),
    olaf.TensorType((NODES * DEGREE,), olaf.i32),
    olaf.TensorType((NODES,), olaf.i32),
)


def case(compute=None, seed=1):
    model = gat.GAT(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), gat.loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, src, dst, labels = data(1)[0]
    traced = olaf.trace(lambda p, *xs: olaf.value_and_grad(lambda p: c.olaf_loss(p, *xs))(p), specs, olaf.TensorType(x.shape, olaf.f64), *SPECS[1:])
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), src, dst, labels)
    model = torch_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, x.astype(np.float64), src, dst, labels)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=5e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=5e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_edge_messages_are_not_materialized():
    # Gathered source rows are evaluated where they are used (the scatter_add into destinations
    # and the attention gradient's row sums): no [edges, heads, channels] buffer is written.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    config = gat.Config(in_features=32, hidden=16, heads=4, num_layers=2, num_classes=8)
    model = gat.GAT(config, olaf.bf16)
    nodes, edges = 512, 512 * 8
    specs = (olaf.TensorType((nodes, 32), olaf.f32), olaf.TensorType((edges,), olaf.i32), olaf.TensorType((edges,), olaf.i32), olaf.TensorType((nodes,), olaf.i32))
    traced, _ = olaf.train_step(gat.loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    assert not any(b.type.rank == 3 and b.type.shape[0] == edges + nodes for b in program.buffers), "edge messages materialized"
    assert not any(launch.task.name == "gather" and launch.task.args[-1].view.rank == 3 for launch in program.launches)
