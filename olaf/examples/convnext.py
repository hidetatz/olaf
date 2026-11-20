"""ConvNeXt (Liu et al. 2022) for small images: a patchify stem, stages of ConvNeXt blocks (a
7x7 depthwise convolution, a layer norm over the channels, an inverted MLP with GELU, a learned
per-channel layer scale, a residual), layer norm and a 2x2 stride-2 convolution between stages,
then global average pooling, a layer norm and a classifier.

The block's MLP runs channels-last, as the paper's (and torchvision's) implementation does:
the depthwise output is permuted to [N, H, W, C], so the norm and both linear layers act on
the last axis, and permuted back for the residual. The norms between stages are over the
channels of [N, C, H, W]: computed the same way, channels-last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    num_classes: int = 10
    patch: int = 2
    depths: tuple = (3, 3, 9, 3)
    dims: tuple = (96, 192, 384, 768)
    layer_scale: float = 1e-6
    eps: float = 1e-6


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _f32(x):
    # torch.autocast runs layer_norm in f32: b16 inputs are widened, the result stays f32.
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


def _add(a, b):
    if a.dtype is not b.dtype:
        a, b = _to(a, DType.f32), _to(b, DType.f32)
    return a + b


def channel_norm(norm: nn.LayerNorm, p, x):
    """LayerNorm over the channels of [N, C, H, W], computed channels-last."""
    return olaf.permute(norm(p, _f32(olaf.permute(x, (0, 2, 3, 1)))), (0, 3, 1, 2))


@dataclass(frozen=True)
class Block(nn.Module):
    dim: int
    config: Config
    compute: DType | None = None

    def _layers(self):
        d, k, c = self.dim, self.compute, self.config
        return {
            "dwconv": nn.Conv(d, d, (7, 7), 1, 3, groups=d, compute=k),
            "norm": nn.LayerNorm(d, c.eps),
            "pwconv1": nn.Linear(d, 4 * d, compute=k),
            "pwconv2": nn.Linear(4 * d, d, compute=k),
        }

    def init(self, rng):
        params = {name: layer.init(rng) for name, layer in self._layers().items()}
        params["gamma"] = np.full(self.dim, self.config.layer_scale, np.float32)
        return params

    def __call__(self, p, x):
        L = self._layers()
        h = olaf.permute(L["dwconv"](p["dwconv"], x), (0, 2, 3, 1))
        h = L["pwconv2"](p["pwconv2"], olaf.gelu(L["pwconv1"](p["pwconv1"], L["norm"](p["norm"], _f32(h)))))
        gamma = p["gamma"]
        h, g = (h, _to(gamma, h.dtype)) if gamma.dtype is h.dtype or h.dtype is not DType.bf16 else (_to(h, DType.f32), gamma)
        h = h * olaf.broadcast_to(g, h.shape)
        return _add(x, olaf.permute(h, (0, 3, 1, 2)))


@dataclass(frozen=True)
class ConvNeXt(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        layers = {"stem": nn.Conv(3, c.dims[0], (c.patch, c.patch), c.patch, compute=k), "stem_norm": nn.LayerNorm(c.dims[0], c.eps)}
        for i, (depth, dim) in enumerate(zip(c.depths, c.dims)):
            if i > 0:
                layers[f"down{i}_norm"] = nn.LayerNorm(c.dims[i - 1], c.eps)
                layers[f"down{i}"] = nn.Conv(c.dims[i - 1], dim, (2, 2), 2, compute=k)
            for j in range(depth):
                layers[f"stage{i}_{j}"] = Block(dim, c, k)
        layers["norm"] = nn.LayerNorm(c.dims[-1], c.eps)
        layers["head"] = nn.Linear(c.dims[-1], c.num_classes, compute=k)
        return layers

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, params, x):
        """Logits [N, classes] from images [N, 3, H, W]."""
        L, c = self._layers(), self.config
        h = channel_norm(L["stem_norm"], params["stem_norm"], L["stem"](params["stem"], x))
        for i, depth in enumerate(c.depths):
            if i > 0:
                h = L[f"down{i}"](params[f"down{i}"], channel_norm(L[f"down{i}_norm"], params[f"down{i}_norm"], h))
            for j in range(depth):
                h = L[f"stage{i}_{j}"](params[f"stage{i}_{j}"], h)
        pooled = olaf.mean(_f32(h), axis=(2, 3))
        return L["head"](params["head"], L["norm"](params["norm"], pooled))
