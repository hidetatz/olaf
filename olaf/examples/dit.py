"""DiT (Peebles and Xie 2023), a diffusion transformer over latent patches: a patch embedding
with fixed 2-D sine-cosine positions, a conditioning vector from the timestep (sinusoids through
an MLP) plus a class embedding, transformer blocks with adaLN-Zero (the conditioning sets a
shift, a scale and a gate for the attention and for the MLP around affine-free layer norms),
and a final adaptive layer norm and linear layer back to patches.

Trained as a noise predictor (`loss`): the mean squared error between the predicted and the
true noise. Parameter names follow tests/test_dit.py's torch.nn twin.
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
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: int = 4
    num_classes: int = 1000
    frequency_size: int = 256


def positions(grid: int, dim: int) -> np.ndarray:
    """DiT's fixed 2-D sine-cosine positions [grid * grid, dim] (half per axis, sin then cos)."""

    def axis(pos):
        omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    h, w = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32))
    return np.concatenate([axis(h), axis(w)], axis=1).astype(np.float32)


def timestep_features(t: olaf.Tensor, dim: int, dtype: DType) -> olaf.Tensor:
    """[N] integer timesteps -> [N, dim]: cos then sin at frequencies 10000^(-i / (dim / 2))."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float64) / half).astype(np.float32).astype(dtype.numpy)
    n = t.shape[0]
    args = olaf.cast(t, dtype).reshape(n, 1) * olaf.broadcast_to(olaf.constant(freqs, dtype).reshape(1, half), (n, half))
    return olaf.concat([olaf.cos(args), olaf.sin(args)], axis=1)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _f32(x):
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


def _promote(a, b):
    # Mixed operands meet in the wider type, as PyTorch promotes them under autocast.
    return (_f32(a), _f32(b)) if a.dtype is not b.dtype else (a, b)


def modulate(x, shift, scale):
    """x * (1 + scale) + shift, scale and shift [N, D] broadcast over the tokens."""
    shift, scale = (olaf.broadcast_to(v.reshape(v.shape[0], 1, v.shape[1]), x.shape) for v in (shift, scale))
    # 1 + scale in scale's own type first (bf16 under autocast), as PyTorch evaluates it.
    x, factor = _promote(x, scale + 1.0)
    y = x * factor
    y, shift = _promote(y, shift)
    return y + shift


def _gated(x, gate, y):
    gate = olaf.broadcast_to(gate.reshape(gate.shape[0], 1, gate.shape[1]), y.shape)
    gate, y = _promote(gate, y)
    x, g = _promote(x, gate * y)
    return x + g


@dataclass(frozen=True)
class Block(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        d = c.hidden_size
        return {
            "qkv": nn.Linear(d, 3 * d, compute=k),
            "proj": nn.Linear(d, d, compute=k),
            "fc1": nn.Linear(d, c.mlp_ratio * d, compute=k),
            "fc2": nn.Linear(c.mlp_ratio * d, d, compute=k),
            "adaLN": nn.Linear(d, 6 * d, compute=k),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, cond):
        L, H = self._layers(), self.config.num_heads
        B, S, D = x.shape
        mods = L["adaLN"](p["adaLN"], olaf.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (mods[:, i * D : (i + 1) * D] for i in range(6))
        h = modulate(olaf.layernorm(_f32(x), eps=1e-6), shift_msa, scale_msa)
        qkv = L["qkv"](p["qkv"], h).reshape(B, S, 3, H, D // H)
        q, k, v = (olaf.transpose(qkv[:, :, i, :, :], 1, 2) for i in range(3))
        attn = L["proj"](p["proj"], olaf.transpose(olaf.attention(q, k, v), 1, 2).reshape(B, S, D))
        x = _gated(x, gate_msa, attn)
        h = modulate(olaf.layernorm(_f32(x), eps=1e-6), shift_mlp, scale_mlp)
        mlp = L["fc2"](p["fc2"], olaf.gelu(L["fc1"](p["fc1"], h), approximate="tanh"))
        return _gated(x, gate_mlp, mlp)


@dataclass(frozen=True)
class DiT(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        d, ps = c.hidden_size, c.patch_size
        return {
            "x_embedder": nn.Conv(c.in_channels, d, (ps, ps), ps, compute=k),
            "t_mlp1": nn.Linear(c.frequency_size, d, compute=k),
            "t_mlp2": nn.Linear(d, d, compute=k),
            "y_embedder": nn.Embedding(c.num_classes, d, 0.02),
            "blocks": nn.Repeat(Block(c, k), c.depth),
            "final_adaLN": nn.Linear(d, 2 * d, compute=k),
            "final_linear": nn.Linear(d, ps * ps * c.in_channels, compute=k),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, params, x, t, y):
        """The predicted noise [N, C, H, W] from noisy latents x, timesteps t [N] and labels y [N]."""
        L, p, c = self._layers(), params, self.config
        N, D, ps, grid = x.shape[0], c.hidden_size, c.patch_size, c.input_size // c.patch_size
        h = L["x_embedder"](p["x_embedder"], x).reshape(N, D, grid * grid)
        h = olaf.transpose(h, 1, 2)
        pos = olaf.constant(positions(grid, D))
        h = _f32(h) + olaf.broadcast_to(_to(pos, _f32(h).dtype).reshape(1, grid * grid, D), h.shape)
        features = timestep_features(t, c.frequency_size, p["t_mlp1"]["weight"].dtype)
        temb = L["t_mlp2"](p["t_mlp2"], olaf.silu(L["t_mlp1"](p["t_mlp1"], features)))
        cond_a, cond_b = _promote(temb, L["y_embedder"](p["y_embedder"], y))
        cond = cond_a + cond_b
        h = L["blocks"](p["blocks"], h, cond)
        mods = L["final_adaLN"](p["final_adaLN"], olaf.silu(cond))
        h = modulate(olaf.layernorm(_f32(h), eps=1e-6), mods[:, :D], mods[:, D:])
        out = L["final_linear"](p["final_linear"], h)  # [N, tokens, ps * ps * C]
        out = out.reshape(N, grid, grid, ps, ps, c.in_channels)
        out = olaf.permute(out, (0, 5, 1, 3, 2, 4))
        return out.reshape(N, c.in_channels, grid * ps, grid * ps)


def loss(model: DiT):
    def fn(params, x, t, y, noise):
        pred = model(params, x, t, y)
        pred = _f32(pred)
        diff = pred - _to(noise, pred.dtype)
        return olaf.mean(diff * diff)

    return fn
