"""A small text-conditioned diffusion U-Net (group norm, nearest upsampling, skip concats,
self- and cross-attention in a convolutional network)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F

import olaf
from olaf.examples import unet

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SIZE, CONTEXT, STEPS = 4, 16, 6, 150
CONFIG = unet.Config()


class TorchResBlock(T.Module):
    def __init__(self, cin, cout, temb, groups):
        super().__init__()
        self.norm1, self.conv1 = T.GroupNorm(groups, cin), T.Conv2d(cin, cout, 3, 1, 1)
        self.time = T.Linear(temb, cout)
        self.norm2, self.conv2 = T.GroupNorm(groups, cout), T.Conv2d(cout, cout, 3, 1, 1)
        self.skip = T.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (self.skip(x) if self.skip is not None else x) + h


class TorchAttention(T.Module):
    def __init__(self, dim, kv_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q, self.to_k, self.to_v = T.Linear(dim, dim, bias=False), T.Linear(kv_dim, dim, bias=False), T.Linear(kv_dim, dim, bias=False)
        self.to_out = T.Linear(dim, dim)

    def forward(self, x, kv):
        B, Tq, D = x.shape
        split = lambda t: t.reshape(B, t.shape[1], self.heads, D // self.heads).transpose(1, 2)  # noqa: E731
        out = F.scaled_dot_product_attention(split(self.to_q(x)), split(self.to_k(kv)), split(self.to_v(kv)))
        return self.to_out(out.transpose(1, 2).reshape(B, Tq, D))


class TorchTransformer(T.Module):
    def __init__(self, dim, context_dim, heads, groups):
        super().__init__()
        self.norm, self.proj_in = T.GroupNorm(groups, dim), T.Conv2d(dim, dim, 1)
        self.norm1, self.attn1 = T.LayerNorm(dim), TorchAttention(dim, dim, heads)
        self.norm2, self.attn2 = T.LayerNorm(dim), TorchAttention(dim, context_dim, heads)
        self.norm3, self.ff1, self.ff2 = T.LayerNorm(dim), T.Linear(dim, 4 * dim), T.Linear(4 * dim, dim)
        self.proj_out = T.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        t = self.proj_in(self.norm(x)).permute(0, 2, 3, 1).reshape(B, H * W, C)
        n1 = self.norm1(t)
        t = t + self.attn1(n1, n1)
        t = t + self.attn2(self.norm2(t), context)
        t = t + self.ff2(F.gelu(self.ff1(self.norm3(t))))
        return x + self.proj_out(t.reshape(B, H, W, C).permute(0, 3, 1, 2))


class TorchUNet(T.Module):
    def __init__(self, c):
        super().__init__()
        w0, w1 = c.widths
        temb = 4 * c.time_dim
        self.time_dim = c.time_dim
        self.time1, self.time2 = T.Linear(c.time_dim, temb), T.Linear(temb, temb)
        self.conv_in = T.Conv2d(c.channels, w0, 3, 1, 1)
        self.down0 = TorchResBlock(w0, w0, temb, c.groups)
        self.downsample = T.Conv2d(w0, w0, 3, 2, 1)
        self.down1, self.down1_attn = TorchResBlock(w0, w1, temb, c.groups), TorchTransformer(w1, c.context_dim, c.heads, c.groups)
        self.mid1, self.mid_attn, self.mid2 = TorchResBlock(w1, w1, temb, c.groups), TorchTransformer(w1, c.context_dim, c.heads, c.groups), TorchResBlock(w1, w1, temb, c.groups)
        self.up1, self.up1_attn = TorchResBlock(2 * w1, w1, temb, c.groups), TorchTransformer(w1, c.context_dim, c.heads, c.groups)
        self.upsample = T.Conv2d(w1, w1, 3, 1, 1)
        self.up0 = TorchResBlock(w1 + w0, w0, temb, c.groups)
        self.norm_out, self.conv_out = T.GroupNorm(c.groups, w0), T.Conv2d(w0, c.channels, 3, 1, 1)

    def forward(self, x, t, context):
        half = self.time_dim // 2
        dtype = self.time1.weight.dtype
        freqs = torch.from_numpy(np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float64) / half).astype(np.float32)).to(device=x.device, dtype=dtype)
        args = t.to(dtype)[:, None] * freqs[None, :]
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        temb = self.time2(F.silu(self.time1(features)))
        h0 = self.conv_in(x)
        h1 = self.down0(h0, temb)
        h3 = self.down1_attn(self.down1(self.downsample(h1), temb), context)
        m = self.mid2(self.mid_attn(self.mid1(h3, temb), context), temb)
        u = self.up1_attn(self.up1(torch.cat([m, h3], 1), temb), context)
        u = self.upsample(F.interpolate(u, scale_factor=2, mode="nearest"))
        u = self.up0(torch.cat([u, h1], 1), temb)
        return self.conv_out(F.silu(self.norm_out(u)))


def torch_model(dtype=torch.float32, device="cpu"):
    return TorchUNet(CONFIG).to(dtype=dtype, device=device).train()


def torch_loss(model, x, t, context, noise):
    p = next(model.parameters())
    cast = lambda a: torch.from_numpy(a).to(device=p.device, dtype=p.dtype)  # noqa: E731
    pred = model(cast(x), torch.from_numpy(t).to(p.device), cast(context))
    pred = pred if p.dtype == torch.float64 else pred.float()
    return F.mse_loss(pred, cast(noise).to(pred.dtype))


def data(steps, seed=0):
    """Four fixed batches, cycled: noisy samples, their timesteps, contexts and noises."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(4):
        clean = np.tanh(rng.normal(size=(BATCH, CONFIG.channels, SIZE, SIZE))).astype(np.float32)
        noise = rng.normal(size=clean.shape).astype(np.float32)
        t = rng.integers(0, 1000, BATCH).astype(np.int32)
        alpha = np.cos(t / 1000 * np.pi / 2).astype(np.float32)[:, None, None, None]
        x = (alpha * clean + np.sqrt(1 - alpha**2) * noise).astype(np.float32)
        context = rng.normal(size=(BATCH, CONTEXT, CONFIG.context_dim)).astype(np.float32)
        batches.append((x, t, context, noise))
    return [batches[k % 4] for k in range(steps)]


SPECS = (
    olaf.TensorType((BATCH, CONFIG.channels, SIZE, SIZE), olaf.f32),
    olaf.TensorType((BATCH,), olaf.i32),
    olaf.TensorType((BATCH, CONTEXT, CONFIG.context_dim), olaf.f32),
    olaf.TensorType((BATCH, CONFIG.channels, SIZE, SIZE), olaf.f32),
)


def case(compute=None, seed=1):
    model = unet.UNet(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), unet.loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16)


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    x, t, context, noise = data(1)[0]
    f64 = lambda a: olaf.TensorType(a.shape, olaf.f64)  # noqa: E731
    traced = olaf.trace(lambda p, *xs: olaf.value_and_grad(lambda p: c.olaf_loss(p, *xs))(p), specs, f64(x), SPECS[1], f64(context), f64(noise))
    (loss, grads), _ = traced.eager(params64, x.astype(np.float64), t, context.astype(np.float64), noise.astype(np.float64))
    model = torch_model(torch.float64)
    load(model, c.torch_named(c.params))
    reference = torch_loss(model, x.astype(np.float64), t, context.astype(np.float64), noise.astype(np.float64))
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = c.torch_named(grads)
    for name, p in model.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=2e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.7 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_fusion_at_latent_diffusion_shapes():
    # 16 x 16 positions (256 tokens) of 128 channels at the transformer blocks, 77 context
    # tokens: shapes where scores, tokens and contexts differ. Built only, not run.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    B, S, L = 2, 32, 77
    config = unet.Config(channels=4, widths=(64, 128), groups=32, heads=4, context_dim=96, time_dim=64)
    model = unet.UNet(config, olaf.bf16)
    specs = (olaf.TensorType((B, 4, S, S), olaf.f32), olaf.TensorType((B,), olaf.i32), olaf.TensorType((B, L, config.context_dim), olaf.f32), olaf.TensorType((B, 4, S, S), olaf.f32))
    traced, _ = olaf.train_step(unet.loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *specs)
    program = to_program(lower_graph(traced.graph), AMPERE_RTX3060)
    names = [launch.task.name for launch in program.launches]
    shapes = {b.name: b.type for b in program.buffers}
    tokens = (S // 2) ** 2
    assert names.count("attention") == 6 and names.count("attention_dkv") == 6, "self- and cross-attention fused, forward and backward"
    assert not any(t.rank >= 3 and t.shape[-2:] in ((tokens, tokens), (tokens, L)) for t in shapes.values()), "attention scores materialized"
    # The skip concatenations stay channels-last, like their inputs: a channels-first result
    # made every reader (group norm statistics, weight gradients) a transpose.
    concats = [launch for launch in program.launches if launch.task.name == "concat"]
    assert concats and all(launch.task.args[-1].view.strides[1] == 1 for launch in concats if launch.task.args[-1].writes)
