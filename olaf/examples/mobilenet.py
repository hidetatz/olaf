"""MobileNetV2 (Sandler et al. 2018) for small images: inverted residual blocks (a 1x1
expansion, a 3x3 depthwise convolution, a linear 1x1 projection, each with batch norm; ReLU6
after the first two), with a residual where the shape is kept, then a 1x1 convolution to the
head width, global average pooling and a classifier.

`settings` lists the stages as (expansion t, channels c, blocks n, first stride s), as in the
paper's Table 2; the stem is a 3x3 convolution with stride `stem_stride` (1 for 32x32 inputs).
Parameter names follow tests/test_mobilenet.py's torch.nn twin.
"""

from __future__ import annotations

from dataclasses import dataclass

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    num_classes: int = 10
    stem: int = 32
    stem_stride: int = 1
    settings: tuple = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))
    head: int = 1280

    def blocks(self) -> list[tuple[int, int, int, int]]:
        """(in channels, out channels, expansion, stride) per inverted residual block."""
        out, cin = [], self.stem
        for t, c, n, s in self.settings:
            for i in range(n):
                out.append((cin, c, t, s if i == 0 else 1))
                cin = c
        return out


def relu6(x: olaf.Tensor) -> olaf.Tensor:
    return olaf.nn.relu6(x)


@dataclass(frozen=True)
class Block(nn.Module):
    cin: int
    cout: int
    expansion: int
    stride: int
    compute: DType | None = None

    def _layers(self):
        k, hidden = self.compute, self.cin * self.expansion
        layers = {}
        if self.expansion != 1:
            layers["expand"] = nn.Conv(self.cin, hidden, (1, 1), bias=False, compute=k)
            layers["expand_bn"] = nn.BatchNorm(hidden)
        layers["depthwise"] = nn.Conv(hidden, hidden, (3, 3), self.stride, 1, groups=hidden, bias=False, compute=k)
        layers["depthwise_bn"] = nn.BatchNorm(hidden)
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
            h = relu6(bn("expand_bn", L["expand"](p["expand"], h)))
        h = relu6(bn("depthwise_bn", L["depthwise"](p["depthwise"], h)))
        h = bn("project_bn", L["project"](p["project"], h))
        if self.stride == 1 and self.cin == self.cout:
            h = x + h
        return h, new


@dataclass(frozen=True)
class MobileNetV2(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        layers = {"stem": nn.Conv(3, c.stem, (3, 3), c.stem_stride, 1, bias=False, compute=k), "stem_bn": nn.BatchNorm(c.stem)}
        for i, (cin, cout, t, s) in enumerate(c.blocks()):
            layers[f"block{i}"] = Block(cin, cout, t, s, k)
        last = c.blocks()[-1][1]
        layers["head"] = nn.Conv(last, c.head, (1, 1), bias=False, compute=k)
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
        h = relu6(h)
        for i in range(len(self.config.blocks())):
            h, new[f"block{i}"] = L[f"block{i}"](params[f"block{i}"], h, buffers[f"block{i}"], training)
        h, new["head_bn"] = L["head_bn"](params["head_bn"], L["head"](params["head"], h), buffers["head_bn"], training)
        h = relu6(h)
        pooled = olaf.mean(olaf.cast(h, DType.f32) if h.dtype is DType.bf16 else h, axis=(2, 3))
        pooled = olaf.cast(pooled, h.dtype) if pooled.dtype is not h.dtype else pooled
        return L["classifier"](params["classifier"], pooled), new
