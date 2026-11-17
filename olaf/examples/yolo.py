"""A small YOLO (Redmon et al. 2016, v1's loss): a convolutional backbone to an S x S grid,
B boxes and C class scores per cell.

Per cell and box: x, y (sigmoid, offset in the cell), w, h (sigmoid, fraction of the image),
confidence (sigmoid). The loss assigns each object to the box of its cell with the larger IoU
(chosen with where; the IoU only selects, it is not differentiated), and sums: coordinates
(x, y, sqrt w, sqrt h) of the responsible boxes, their confidence towards 1, every other
box's confidence towards 0 (weight NOOBJ), and the class cross-entropy of the object cells.
Non-maximum suppression is post-processing and stays outside the graph.

Targets, per cell: obj (1 where an object's centre falls), box (x, y in the cell, w, h of the
image) and the class.
"""

from __future__ import annotations

from dataclasses import dataclass

import olaf
from olaf import nn
from olaf.graph.types import DType

COORD, NOOBJ = 5.0, 0.5


@dataclass(frozen=True)
class Config:
    grid: int = 8
    boxes: int = 2
    classes: int = 3
    widths: tuple[int, ...] = (16, 32, 64)
    in_channels: int = 3


@dataclass(frozen=True)
class Yolo(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, layers, cin = self.config, {}, self.config.in_channels
        for i, w in enumerate(c.widths):
            layers[f"conv{i + 1}"] = nn.Conv(cin, w, (3, 3), 1, 1, bias=False, compute=self.compute)
            layers[f"bn{i + 1}"] = nn.BatchNorm(w)
            cin = w
        layers["head"] = nn.Conv(cin, c.boxes * 5 + c.classes, (1, 1), compute=self.compute)
        return layers

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def init_buffers(self):
        return {name: layer.init_buffers() for name, layer in self._layers().items() if isinstance(layer, nn.BatchNorm)}

    def __call__(self, params, buffers, x, training: bool = True):
        """Raw predictions [N, S, S, 5B + C] and the new buffers."""
        L, new, h = self._layers(), {}, x
        for i in range(len(self.config.widths)):
            h, new[f"bn{i + 1}"] = L[f"bn{i + 1}"](params[f"bn{i + 1}"], L[f"conv{i + 1}"](params[f"conv{i + 1}"], h), buffers[f"bn{i + 1}"], training)
            h = nn.max_pool(nn.leaky_relu(h, 0.1), (2, 2))
        out = L["head"](params["head"], h)
        return olaf.permute(out, (0, 2, 3, 1)), new


def _iou(a, b, grid: int):
    """IoU of boxes (x, y in the cell; w, h of the image), same cell: corners in cell units."""

    def corners(box):
        x, y, w, h = box
        return x - w * grid / 2, y - h * grid / 2, x + w * grid / 2, y + h * grid / 2

    ax0, ay0, ax1, ay1 = corners(a)
    bx0, by0, bx1, by1 = corners(b)
    iw = olaf.maximum(olaf.minimum(ax1, bx1) - olaf.maximum(ax0, bx0), 0.0)
    ih = olaf.maximum(olaf.minimum(ay1, by1) - olaf.maximum(ay0, by0), 0.0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / (union + 1e-9)


def loss(config: Config, raw, obj, box, cls):
    """raw [N, S, S, 5B + C] (any float dtype; computed in f32 for bf16); obj [N, S, S] f32;
    box [N, S, S, 4] f32; cls [N, S, S] i32. Summed over cells, divided by the batch."""
    p = olaf.cast(raw, DType.f32) if raw.dtype is DType.bf16 else raw
    S, B, C = config.grid, config.boxes, config.classes
    if B != 2:
        raise NotImplementedError("the responsible-box choice is written for two boxes per cell")
    N = p.shape[0]
    t = [box[..., i] for i in range(4)]
    preds, ious = [], []
    for b in range(B):
        x, y, w, h, conf = (olaf.tanh(p[..., 5 * b + i] * 0.5) * 0.5 + 0.5 for i in range(5))  # sigmoid
        preds.append((x, y, w, h, conf))
        ious.append(_iou((x, y, w, h), t, S))
    # The responsible box: the larger IoU (the first on a tie), compared without gradient.
    first = ious[0] >= ious[1]
    zero = olaf.zeros(obj.shape, obj.dtype)
    owners = (olaf.where(first, obj, zero), olaf.where(first, zero, obj))
    total = 0.0
    for (x, y, w, h, conf), responsible in zip(preds, owners):
        coord = (x - t[0]) ** 2.0 + (y - t[1]) ** 2.0 + (olaf.sqrt(w) - olaf.sqrt(t[2])) ** 2.0 + (olaf.sqrt(h) - olaf.sqrt(t[3])) ** 2.0
        total = total + olaf.sum(responsible * (coord * COORD + (conf - 1.0) ** 2.0)) + olaf.sum((1.0 - responsible) * conf * conf) * NOOBJ
    logits = p[..., 5 * B :]
    top = olaf.max(logits, axis=-1).reshape(N, S, S, 1)
    lse = olaf.log(olaf.sum(olaf.exp(logits - olaf.broadcast_to(top, logits.shape)), axis=-1)) + top.reshape(N, S, S)
    onehot = olaf.equal(olaf.iota(logits.shape, axis=3), olaf.broadcast_to(cls.reshape(N, S, S, 1), logits.shape))
    picked = olaf.sum(olaf.where(onehot, logits, 0.0), axis=-1)
    total = total + olaf.sum(obj * (lse - picked))
    return total / float(N)
