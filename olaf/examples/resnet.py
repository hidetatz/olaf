"""ResNet (He et al. 2016) with basic blocks, named as torchvision's ResNet (conv1, bn1,
layer{i}.{j}.conv1/bn1/conv2/bn2/downsample.{0,1}, fc), so torchvision's state_dict maps by
name.

stem="imagenet" is torchvision's (7x7 stride-2 convolution, 3x3 stride-2 max pooling);
stem="cifar" is the usual small-image variant (3x3 stride-1 convolution, no pooling).

The model returns (logits, new buffers): batch norm's running statistics are buffers that the
step reads and writes (nn.BatchNorm, olaf.train_step(buffers=...)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    blocks: tuple[int, ...] = (2, 2, 2, 2)  # ResNet-18
    widths: tuple[int, ...] = (64, 128, 256, 512)
    num_classes: int = 1000
    in_channels: int = 3
    stem: str = "imagenet"


def _conv(cin, cout, k, stride, compute):
    return nn.Conv(cin, cout, (k, k), stride, k // 2, bias=False, compute=compute)


@dataclass(frozen=True)
class ResNet(nn.Module):
    config: Config
    compute: DType | None = None

    def _plan(self):
        """(name, cin, cout, stride) for every basic block."""
        c, blocks, cin = self.config, [], self.config.widths[0]
        for i, (n, width) in enumerate(zip(c.blocks, c.widths)):
            for j in range(n):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append((f"layer{i + 1}.{j}", cin, width, stride))
                cin = width
        return blocks

    def _layers(self):
        c, k = self.config, self.compute
        layers = {"conv1": _conv(c.in_channels, c.widths[0], 7 if c.stem == "imagenet" else 3, 2 if c.stem == "imagenet" else 1, k), "bn1": nn.BatchNorm(c.widths[0])}
        for name, cin, cout, stride in self._plan():
            block = {"conv1": _conv(cin, cout, 3, stride, k), "bn1": nn.BatchNorm(cout), "conv2": _conv(cout, cout, 3, 1, k), "bn2": nn.BatchNorm(cout)}
            if stride != 1 or cin != cout:
                block["downsample"] = {"0": _conv(cin, cout, 1, stride, k), "1": nn.BatchNorm(cout)}
            layers[name] = block
        layers["fc"] = nn.Linear(c.widths[-1], c.num_classes, compute=k)
        return layers

    @staticmethod
    def _nest(flat: dict) -> dict:
        tree: dict = {}
        for key, value in flat.items():
            node = tree
            *parents, leaf = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return tree

    def _flat_layers(self) -> dict:
        out = {}

        def walk(prefix, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(f"{prefix}{k}.", v)
            else:
                out[prefix.rstrip(".")] = node

        walk("", self._layers())
        return out

    def init(self, rng):
        return self._nest({name: layer.init(rng) for name, layer in self._flat_layers().items()})

    def init_buffers(self):
        return self._nest({name: layer.init_buffers() for name, layer in self._flat_layers().items() if isinstance(layer, nn.BatchNorm)})

    def __call__(self, params, buffers, x, training: bool = True):
        features, new = self.features(params, buffers, x, training)
        return self.head(params, features), new

    def head(self, params, h):
        """Logits from the last stage's feature map (global average pooling, then fc)."""
        # Global average pooling in f32, rounded once (PyTorch's adaptive_avg_pool2d for bf16).
        pooled = olaf.mean(olaf.cast(h, DType.f32), axis=(2, 3)) if h.dtype is DType.bf16 else olaf.mean(h, axis=(2, 3))
        pooled = olaf.cast(pooled, h.dtype) if pooled.dtype is not h.dtype else pooled
        return self._flat_layers()["fc"](params["fc"], pooled)

    def features(self, params, buffers, x, training: bool = True):
        """The last stage's feature map [N, C, H, W] and the new buffers (Grad-CAM's target)."""
        layers = self._flat_layers()
        new: dict = {}

        def at(tree, name):
            for part in name.split("."):
                tree = tree[part]
            return tree

        def bn(name, h):
            y, new[name] = layers[name](at(params, name), h, at(buffers, name), training)
            return y

        def conv(name, h):
            return layers[name](at(params, name), h)

        h = olaf.nn.relu(bn("bn1", conv("conv1", x)))
        if self.config.stem == "imagenet":
            h = nn.max_pool(h, (3, 3), 2, 1)
        for name, cin, cout, stride in self._plan():
            out = olaf.nn.relu(bn(f"{name}.bn1", conv(f"{name}.conv1", h)))
            out = bn(f"{name}.bn2", conv(f"{name}.conv2", out))
            shortcut = bn(f"{name}.downsample.1", conv(f"{name}.downsample.0", h)) if f"{name}.downsample.0" in layers else h
            h = olaf.nn.relu(out + shortcut)
        return h, self._nest(new)


def grad_cam(model: ResNet, params, buffers, x, classes, size=None):
    """Grad-CAM (Selvaraju et al. 2017) on the last stage: the class score's gradient with
    respect to the feature map, averaged over positions, weighs the channels; the ReLU of the
    weighted sum is the map, bilinearly upsampled to `size` (default: the input's). classes:
    i32 [N]. Evaluation mode (running statistics)."""
    features, _ = model.features(params, buffers, x, training=False)

    def score(a):
        logits = model.head(params, a)
        picked = olaf.where(olaf.equal(olaf.iota(logits.shape, axis=1), olaf.broadcast_to(classes.reshape(logits.shape[0], 1), logits.shape)), logits, 0.0)
        return olaf.sum(picked)

    grads = olaf.grad(score)(features)
    weights = olaf.mean(grads, axis=(2, 3))
    N, C, H, W = features.shape
    cam = olaf.nn.relu(olaf.sum(features * olaf.broadcast_to(weights.reshape(N, C, 1, 1), features.shape), axis=1)).reshape(N, 1, H, W)
    return nn.interpolate_bilinear(cam, size or tuple(x.shape[2:]))
