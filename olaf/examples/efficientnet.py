"""EfficientNet (Tan and Le 2019) for small images: MBConv blocks (a 1x1 expansion, a 3x3 or
5x5 depthwise convolution, squeeze-and-excitation, a linear 1x1 projection; batch norm and SiLU),
with a residual where the shape is kept, then a 1x1 head convolution, global average pooling
and a classifier.

`settings` lists the stages as (expansion, kernel, first stride, channels, blocks), as in the
paper's Table 1 for B0; squeeze-and-excitation narrows to a quarter of each block's input
channels. Parameter names follow tests/test_efficientnet.py's torch.nn twin.
"""

from __future__ import annotations

from dataclasses import dataclass

import olaf
from olaf import nn
from olaf.graph.types import DType

B0 = ((1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3), (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1))


@dataclass(frozen=True)
class Config:
    num_classes: int = 10
    stem: int = 32
    stem_stride: int = 1
    # B0 for 32x32 inputs: the second stage keeps the resolution (stride 1), like the stem.
    settings: tuple = ((1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3), (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1))
    head: int = 1280
    se_ratio: float = 0.25

    def blocks(self) -> list[tuple[int, int, int, int, int]]:
        """(in channels, out channels, expansion, kernel, stride) per MBConv block."""
        out, cin = [], self.stem
        for t, k, s, c, n in self.settings:
            for i in range(n):
                out.append((cin, c, t, k, s if i == 0 else 1))
                cin = c
        return out


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def sigmoid(x: olaf.Tensor) -> olaf.Tensor:
    """1 / (1 + exp(-x)), computed in f32 for b16 inputs and rounded once, as PyTorch's kernel."""
    h = _to(x, DType.f32) if x.dtype is DType.bf16 else x
    return _to(olaf.recip(1.0 + olaf.exp(-h)), x.dtype)


@dataclass(frozen=True)
class Block(nn.Module):
    cin: int
    cout: int
    expansion: int
    kernel: int
    stride: int
    se: int
    compute: DType | None = None

    def _layers(self):
        k, hidden = self.compute, self.cin * self.expansion
        layers = {}
        if self.expansion != 1:
            layers["expand"] = nn.Conv(self.cin, hidden, (1, 1), bias=False, compute=k)
            layers["expand_bn"] = nn.BatchNorm(hidden)
        layers["depthwise"] = nn.Conv(hidden, hidden, (self.kernel, self.kernel), self.stride, self.kernel // 2, groups=hidden, bias=False, compute=k)
        layers["depthwise_bn"] = nn.BatchNorm(hidden)
        layers["se_reduce"] = nn.Conv(hidden, self.se, (1, 1), compute=k)
        layers["se_expand"] = nn.Conv(self.se, hidden, (1, 1), compute=k)
        layers["project"] = nn.Conv(hidden, self.cout, (1, 1), bias=False, compute=k)
        layers["project_bn"] = nn.BatchNorm(self.cout)
        return layers

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def init_buffers(self):
        return {name: layer.init_buffers() for name, layer in self._layers().items() if isinstance(layer, nn.BatchNorm)}

    def __call__(self, p, x, buffers, training: bool = True):
        L, new = self._layers(), {}

        def bn(name, h):
            y, new[name] = L[name](p[name], h, buffers[name], training)
            return y

        h = x
        if "expand" in L:
            h = olaf.silu(bn("expand_bn", L["expand"](p["expand"], h)))
        h = olaf.silu(bn("depthwise_bn", L["depthwise"](p["depthwise"], h)))
        # Squeeze-and-excitation: per-channel gates from the positions' mean.
        pooled = olaf.mean(_to(h, DType.f32) if h.dtype is DType.bf16 else h, axis=(2, 3))
        pooled = _to(pooled, h.dtype).reshape(*pooled.shape, 1, 1)
        gate = sigmoid(L["se_expand"](p["se_expand"], olaf.silu(L["se_reduce"](p["se_reduce"], pooled))))
        h = h * olaf.broadcast_to(_to(gate, h.dtype), h.shape)
        h = bn("project_bn", L["project"](p["project"], h))
        if self.stride == 1 and self.cin == self.cout:
            h = x + h
        return h, new


@dataclass(frozen=True)
class EfficientNet(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        layers = {"stem": nn.Conv(3, c.stem, (3, 3), c.stem_stride, 1, bias=False, compute=k), "stem_bn": nn.BatchNorm(c.stem)}
        for i, (cin, cout, t, kernel, s) in enumerate(c.blocks()):
            layers[f"block{i}"] = Block(cin, cout, t, kernel, s, max(1, int(cin * c.se_ratio)), k)
        layers["head"] = nn.Conv(c.blocks()[-1][1], c.head, (1, 1), bias=False, compute=k)
        layers["head_bn"] = nn.BatchNorm(c.head)
        layers["classifier"] = nn.Linear(c.head, c.num_classes, compute=k)
        return layers

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def init_buffers(self):
        return {name: layer.init_buffers() for name, layer in self._layers().items() if isinstance(layer, (nn.BatchNorm, Block))}

    def __call__(self, params, buffers, x, training: bool = True):
        """Logits [N, classes] from images [N, 3, H, W], and the new buffers."""
        L, new = self._layers(), {}
        h, new["stem_bn"] = L["stem_bn"](params["stem_bn"], L["stem"](params["stem"], x), buffers["stem_bn"], training)
        h = olaf.silu(h)
        for i in range(len(self.config.blocks())):
            h, new[f"block{i}"] = L[f"block{i}"](params[f"block{i}"], h, buffers[f"block{i}"], training)
        h, new["head_bn"] = L["head_bn"](params["head_bn"], L["head"](params["head"], h), buffers["head_bn"], training)
        h = olaf.silu(h)
        pooled = olaf.mean(_to(h, DType.f32) if h.dtype is DType.bf16 else h, axis=(2, 3))
        return L["classifier"](params["classifier"], _to(pooled, h.dtype)), new
