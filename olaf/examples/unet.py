"""A small text-conditioned diffusion U-Net in the shape of Stable Diffusion's UNet: residual
blocks (group norm, SiLU, 3x3 convolutions, the timestep embedding added per channel), a
strided-convolution downsample and a nearest-neighbour upsample, skip connections concatenated
along the channels, and transformer blocks at the low resolution (self-attention over the
positions, cross-attention to a context sequence such as text embeddings, a GELU MLP).

Two levels: `widths[0]` channels at full resolution, `widths[1]` at half, with a transformer
block on the way down, in the middle and on the way up. Trained as a noise predictor
(`loss`): the mean squared error between the predicted and the true noise.

Parameter names follow torch's module paths (tests/test_unet.py builds the same network from
torch.nn modules).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    channels: int = 4  # input and predicted-noise channels
    widths: tuple[int, int] = (32, 64)
    groups: int = 8
    heads: int = 2
    context_dim: int = 32
    time_dim: int = 32  # sinusoidal features; the embedding is 4 x this


def timestep_features(t: olaf.Tensor, dim: int, dtype: DType = DType.f32) -> olaf.Tensor:
    """[N] integer timesteps -> [N, dim]: cos then sin of t at frequencies
    10000^(-i / (dim / 2)), i < dim / 2 (the DDPM / diffusers sinusoids, cosine first),
    computed in dtype (the time embedding's parameters' type)."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float64) / half).astype(np.float32).astype(dtype.numpy)
    n = t.shape[0]
    args = olaf.cast(t, dtype).reshape(n, 1) * olaf.broadcast_to(olaf.constant(freqs, dtype).reshape(1, half), (n, half))
    return olaf.concat([olaf.cos(args), olaf.sin(args)], axis=1)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _add(a, b):
    # Mixed operands meet in the wider type, as PyTorch promotes them (f32 group-norm outputs
    # and bf16 convolutions under autocast).
    if a.dtype is not b.dtype:
        a, b = _to(a, olaf.f32), _to(b, olaf.f32)
    return a + b


@dataclass(frozen=True)
class ResBlock(nn.Module):
    cin: int
    cout: int
    temb: int
    groups: int
    compute: DType | None = None

    def _layers(self):
        k = self.compute
        layers = {
            "norm1": nn.GroupNorm(self.groups, self.cin),
            "conv1": nn.Conv(self.cin, self.cout, (3, 3), 1, 1, compute=k),
            "time": nn.Linear(self.temb, self.cout, compute=k),
            "norm2": nn.GroupNorm(self.groups, self.cout),
            "conv2": nn.Conv(self.cout, self.cout, (3, 3), 1, 1, compute=k),
        }
        if self.cin != self.cout:
            layers["skip"] = nn.Conv(self.cin, self.cout, (1, 1), compute=k)
        return layers

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, temb):
        L = self._layers()
        h = L["conv1"](p["conv1"], olaf.silu(L["norm1"](p["norm1"], x)))
        t = L["time"](p["time"], olaf.silu(temb))
        h = _add(h, olaf.broadcast_to(t.reshape(*t.shape, 1, 1), h.shape))
        h = L["conv2"](p["conv2"], olaf.silu(L["norm2"](p["norm2"], h)))
        skip = L["skip"](p["skip"], x) if "skip" in L else x
        return _add(skip, h)


@dataclass(frozen=True)
class Attention(nn.Module):
    dim: int
    kv_dim: int
    heads: int
    compute: DType | None = None

    def _layers(self):
        k = self.compute
        return {
            "to_q": nn.Linear(self.dim, self.dim, bias=False, compute=k),
            "to_k": nn.Linear(self.kv_dim, self.dim, bias=False, compute=k),
            "to_v": nn.Linear(self.kv_dim, self.dim, bias=False, compute=k),
            "to_out": nn.Linear(self.dim, self.dim, compute=k),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, kv):
        L, H = self._layers(), self.heads
        B, Tq, D = x.shape
        Tk = kv.shape[1]

        def heads(t, n):
            return olaf.transpose(t.reshape(B, n, H, D // H), 1, 2)

        q, k, v = heads(L["to_q"](p["to_q"], x), Tq), heads(L["to_k"](p["to_k"], kv), Tk), heads(L["to_v"](p["to_v"], kv), Tk)
        merged = olaf.transpose(olaf.attention(q, k, v), 1, 2).reshape(B, Tq, D)
        return L["to_out"](p["to_out"], merged)


@dataclass(frozen=True)
class Transformer(nn.Module):
    """Stable Diffusion's spatial transformer: group norm, a 1x1 projection to tokens, then
    self-attention, cross-attention and a GELU MLP (each pre-norm and residual), a 1x1
    projection back, and a residual around the whole block."""

    dim: int
    context_dim: int
    heads: int
    groups: int
    compute: DType | None = None

    def _layers(self):
        k, d = self.compute, self.dim
        return {
            "norm": nn.GroupNorm(self.groups, d),
            "proj_in": nn.Conv(d, d, (1, 1), compute=k),
            "norm1": nn.LayerNorm(d),
            "attn1": Attention(d, d, self.heads, k),
            "norm2": nn.LayerNorm(d),
            "attn2": Attention(d, self.context_dim, self.heads, k),
            "norm3": nn.LayerNorm(d),
            "ff1": nn.Linear(d, 4 * d, compute=k),
            "ff2": nn.Linear(4 * d, d, compute=k),
            "proj_out": nn.Conv(d, d, (1, 1), compute=k),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, context):
        L = self._layers()
        B, C, Hh, W = x.shape
        h = L["proj_in"](p["proj_in"], L["norm"](p["norm"], x))
        tokens = olaf.permute(h, (0, 2, 3, 1)).reshape(B, Hh * W, C)
        def norm(name, v):
            # torch.autocast runs layer_norm in f32 (its fp32 list): b16 tokens are widened.
            return L[name](p[name], _to(v, olaf.f32) if v.dtype is olaf.bf16 else v)

        n1 = norm("norm1", tokens)
        tokens = _add(tokens, L["attn1"](p["attn1"], n1, n1))
        tokens = _add(tokens, L["attn2"](p["attn2"], norm("norm2", tokens), _to(context, self.compute)))
        ff = L["ff2"](p["ff2"], olaf.gelu(L["ff1"](p["ff1"], norm("norm3", tokens))))
        tokens = _add(tokens, ff)
        h = olaf.permute(tokens.reshape(B, Hh, W, C), (0, 3, 1, 2))
        return _add(x, L["proj_out"](p["proj_out"], h))


@dataclass(frozen=True)
class UNet(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        w0, w1 = c.widths
        temb = 4 * c.time_dim
        res = lambda cin, cout: ResBlock(cin, cout, temb, c.groups, k)  # noqa: E731
        block = lambda: Transformer(w1, c.context_dim, c.heads, c.groups, k)  # noqa: E731
        return {
            "time1": nn.Linear(c.time_dim, temb, compute=k),
            "time2": nn.Linear(temb, temb, compute=k),
            "conv_in": nn.Conv(c.channels, w0, (3, 3), 1, 1, compute=k),
            "down0": res(w0, w0),
            "downsample": nn.Conv(w0, w0, (3, 3), 2, 1, compute=k),
            "down1": res(w0, w1),
            "down1_attn": block(),
            "mid1": res(w1, w1),
            "mid_attn": block(),
            "mid2": res(w1, w1),
            "up1": res(2 * w1, w1),
            "up1_attn": block(),
            "upsample": nn.Conv(w1, w1, (3, 3), 1, 1, compute=k),
            "up0": res(w1 + w0, w0),
            "norm_out": nn.GroupNorm(c.groups, w0),
            "conv_out": nn.Conv(w0, c.channels, (3, 3), 1, 1, compute=k),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, t, context):
        """x [N, channels, H, W] noisy samples, t [N] i32 timesteps, context [N, L, context_dim]
        -> the predicted noise, x's shape."""
        L, c = self._layers(), self.config
        features = timestep_features(t, c.time_dim, p["time1"]["weight"].dtype)
        temb = L["time2"](p["time2"], olaf.silu(L["time1"](p["time1"], features)))

        def cat(a, b):
            if a.dtype is not b.dtype:
                a, b = _to(a, olaf.f32), _to(b, olaf.f32)
            return olaf.concat([a, b], axis=1)

        h0 = L["conv_in"](p["conv_in"], x)
        h1 = L["down0"](p["down0"], h0, temb)
        h2 = L["downsample"](p["downsample"], h1)
        h3 = L["down1_attn"](p["down1_attn"], L["down1"](p["down1"], h2, temb), context)
        m = L["mid1"](p["mid1"], h3, temb)
        m = L["mid2"](p["mid2"], L["mid_attn"](p["mid_attn"], m, context), temb)
        u = L["up1_attn"](p["up1_attn"], L["up1"](p["up1"], cat(m, h3), temb), context)
        u = L["upsample"](p["upsample"], nn.upsample_nearest(u, 2))
        u = L["up0"](p["up0"], cat(u, h1), temb)
        return L["conv_out"](p["conv_out"], olaf.silu(L["norm_out"](p["norm_out"], u)))


def loss(model: UNet):
    """Noise-prediction loss: mean((model(x_t, t, context) - noise)^2) in f32."""

    def fn(params, x, t, context, noise):
        pred = model(params, x, t, context)
        pred = olaf.cast(pred, olaf.f32) if pred.dtype is olaf.bf16 else pred
        diff = pred - _to(noise, pred.dtype)
        return olaf.mean(diff * diff)

    return fn
