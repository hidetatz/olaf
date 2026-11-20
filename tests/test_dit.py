"""DiT (adaLN-Zero modulation from timestep and class conditioning, patchify and unpatchify)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F

import olaf
from olaf.examples import dit

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, STEPS = 8, 100
CONFIG = dit.Config(input_size=8, patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=2, mlp_ratio=4, num_classes=10, frequency_size=32)
STACK = "blocks."


class TorchBlock(T.Module):
    def __init__(self, c):
        super().__init__()
        d = c.hidden_size
        self.heads = c.num_heads
        self.qkv, self.proj = T.Linear(d, 3 * d), T.Linear(d, d)
        self.fc1, self.fc2 = T.Linear(d, c.mlp_ratio * d), T.Linear(c.mlp_ratio * d, d)
        self.adaLN = T.Linear(d, 6 * d)

    def forward(self, x, c):
        B, S, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(F.silu(c)).chunk(6, dim=1)
        h = F.layer_norm(x, (D,), eps=1e-6) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        q, k, v = self.qkv(h).reshape(B, S, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = x + gate_msa[:, None] * self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D))
        h = F.layer_norm(x, (D,), eps=1e-6) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


class TorchDiT(T.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        self.c = c
        d, ps = c.hidden_size, c.patch_size
        self.x_embedder = T.Conv2d(c.in_channels, d, ps, ps)
        self.t_mlp1, self.t_mlp2 = T.Linear(c.frequency_size, d), T.Linear(d, d)
        self.y_embedder = T.Embedding(c.num_classes, d)
        self.blocks = T.ModuleList([TorchBlock(c) for _ in range(c.depth)])
        self.final_adaLN, self.final_linear = T.Linear(d, 2 * d), T.Linear(d, ps * ps * c.in_channels)
        grid = c.input_size // ps
        self.register_buffer("pos", torch.from_numpy(dit.positions(grid, d)), persistent=False)

    def forward(self, x, t, y):
        c = self.c
        N, D, ps, grid = x.shape[0], c.hidden_size, c.patch_size, c.input_size // c.patch_size
        h = self.x_embedder(x).flatten(2).transpose(1, 2)
        h = h + self.pos.to(torch.float64 if h.dtype == torch.float64 else torch.float32)[None]
        half = c.frequency_size // 2
        dtype = self.t_mlp1.weight.dtype
        freqs = torch.from_numpy(np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float64) / half).astype(np.float32)).to(device=x.device, dtype=dtype)
        args = t.to(dtype)[:, None] * freqs[None]
        temb = self.t_mlp2(F.silu(self.t_mlp1(torch.cat([torch.cos(args), torch.sin(args)], dim=1))))
        cond = temb + self.y_embedder(y)
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.final_adaLN(F.silu(cond)).chunk(2, dim=1)
        h = F.layer_norm(h, (D,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        out = self.final_linear(h).reshape(N, grid, grid, ps, ps, c.in_channels)
        return out.permute(0, 5, 1, 3, 2, 4).reshape(N, c.in_channels, grid * ps, grid * ps)


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


def torch_model(dtype=torch.float32, device="cpu"):
    return TorchDiT().to(dtype=dtype, device=device).train()


def torch_loss(model, x, t, y, noise):
    p = next(model.parameters())
    cast = lambda a: torch.from_numpy(a).to(device=p.device, dtype=p.dtype)  # noqa: E731
    pred = model(cast(x), torch.from_numpy(t).to(p.device), torch.from_numpy(y).long().to(p.device))
    pred = pred if p.dtype == torch.float64 else pred.float()
    return F.mse_loss(pred, cast(noise).to(pred.dtype))


def data(steps, seed=0):
    """Four fixed batches, cycled: noisy latents, timesteps, labels and the noises."""
    rng = np.random.default_rng(seed)
    batches = []
    size = CONFIG.input_size
    for _ in range(4):
        clean = np.tanh(rng.normal(size=(BATCH, CONFIG.in_channels, size, size))).astype(np.float32)
        noise = rng.normal(size=clean.shape).astype(np.float32)
        t = rng.integers(0, 1000, BATCH).astype(np.int32)
        alpha = np.cos(t / 1000 * np.pi / 2).astype(np.float32)[:, None, None, None]
        x = (alpha * clean + np.sqrt(1 - alpha**2) * noise).astype(np.float32)
        y = rng.integers(0, CONFIG.num_classes, BATCH).astype(np.int32)
        batches.append((x, t, y, noise))
    return [batches[k % 4] for k in range(steps)]


SPECS = (
    olaf.TensorType((BATCH, CONFIG.in_channels, CONFIG.input_size, CONFIG.input_size), olaf.f32),
    olaf.TensorType((BATCH,), olaf.i32),
    olaf.TensorType((BATCH,), olaf.i32),
    olaf.TensorType((BATCH, CONFIG.in_channels, CONFIG.input_size, CONFIG.input_size), olaf.f32),
)


def case(compute=None, seed=1):
    model = dit.DiT(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), dit.loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, t, y, noise = data(1)[0]
    f64 = lambda a: olaf.TensorType(a.shape, olaf.f64)  # noqa: E731
    traced = olaf.trace(lambda p, *xs: olaf.value_and_grad(lambda p: c.olaf_loss(p, *xs))(p), specs, f64(x), SPECS[1], SPECS[2], f64(noise))
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), t, y, noise.astype(np.float64))
    model = torch_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, x.astype(np.float64), t, y, noise.astype(np.float64))
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        g = p.grad.numpy()
        if np.linalg.norm(g) < 1e-15:
            # The qkv bias's key part: softmax ignores a constant added to a row of scores.
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


def test_modulated_attention_fuses():
    # adaLN modulation around the norms leaves the attention itself plain: it runs as the fused
    # tasks over the 256 tokens, without scores materialized.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    config = dit.Config(input_size=32, patch_size=2, in_channels=4, hidden_size=128, depth=1, num_heads=2, num_classes=10, frequency_size=64)
    model = dit.DiT(config, olaf.bf16)
    specs = (olaf.TensorType((4, 4, 32, 32), olaf.f32), olaf.TensorType((4,), olaf.i32), olaf.TensorType((4,), olaf.i32), olaf.TensorType((4, 4, 32, 32), olaf.f32))
    traced, _ = olaf.train_step(dit.loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    names = {launch.task.name for launch in program.launches}
    assert {"attention", "attention_dkv"} <= names
    assert not any(b.type.rank >= 3 and b.type.shape[-2:] == (256, 256) for b in program.buffers), "scores materialized"
