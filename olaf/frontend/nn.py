"""Model definition: a module is configuration, its parameters are a pytree of arrays.

`init(rng)` returns the parameters (nested dicts of NumPy arrays) and `module(params, x)` traces
the computation. Keeping parameters out of the module, as plain pytrees, is what the step
needs: the training step reads them as resident state, differentiates with respect to the
pytree and writes the update back (frontend/optim.py), and none of that touches Python objects
at run time (principle 1).

Weight layouts and default initializations follow PyTorch (Linear stores [out, in]), so a
PyTorch model's state_dict maps onto the parameters by name, for tests now and the importer
later.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from olaf.frontend.trace import Tensor, abs, broadcast_to, cast, concat, constant, conv, conv_transpose, cos, einsum, exp, log, pad, where, iota, layernorm, maximum, mean, neg, repeat, rsqrt, sin, take, tree_flatten, tree_map, tree_unflatten
from olaf.graph.types import DType

_LETTERS = "abcdefghijklmn"


class Module:
    def init(self, rng: np.random.Generator) -> dict:
        return {}

    def __call__(self, params, *args):
        raise NotImplementedError


def _to(x: Tensor, dtype: DType | None) -> Tensor:
    return x if dtype is None or x.dtype is dtype else cast(x, dtype)


@dataclass(frozen=True)
class Linear(Module):
    """y = x W^T + b over the last axis of x (torch.nn.Linear).

    compute: run the product in this dtype, as torch.autocast does for a linear layer (input,
    weight and bias cast, the output in that dtype). None computes in the parameters' dtype.
    """

    in_features: int
    out_features: int
    bias: bool = True
    compute: DType | None = None

    def init(self, rng: np.random.Generator) -> dict:
        # torch.nn.Linear.reset_parameters: kaiming_uniform_(a=sqrt(5)) on the weight is
        # U(-1/sqrt(in), 1/sqrt(in)), and the bias uses the same bound.
        bound = 1.0 / math.sqrt(self.in_features)
        params = {"weight": rng.uniform(-bound, bound, (self.out_features, self.in_features)).astype(np.float32)}
        if self.bias:
            params["bias"] = rng.uniform(-bound, bound, (self.out_features,)).astype(np.float32)
        return params

    def __call__(self, params, x: Tensor) -> Tensor:
        batch = _LETTERS[: x.rank - 1]
        y = einsum(f"{batch}y,zy->{batch}z", _to(x, self.compute), _to(params["weight"], self.compute))
        if self.bias:
            y = y + _to(params["bias"], self.compute)
        return y


class Sequential(Module):
    """Layers applied in order. Modules get their parameters under their index ("0", "1", ...);
    plain callables (activations) take only the value."""

    def __init__(self, *layers: Module | Callable[[Tensor], Tensor]) -> None:
        self.layers = layers

    def init(self, rng: np.random.Generator) -> dict:
        return {str(i): layer.init(rng) for i, layer in enumerate(self.layers) if isinstance(layer, Module) and layer.init.__func__ is not Module.init}

    def __call__(self, params, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                x = layer(params.get(str(i), {}), x)
            else:
                x = layer(x)
        return x


@dataclass(frozen=True)
class Embedding(Module):
    """Rows of a [num, dim] table (torch.nn.Embedding; initialized N(0, std))."""

    num: int
    dim: int
    std: float = 1.0

    def init(self, rng: np.random.Generator) -> dict:
        return {"weight": (rng.standard_normal((self.num, self.dim)) * self.std).astype(np.float32)}

    def __call__(self, params, ids: Tensor) -> Tensor:
        return take(params["weight"], ids)


@dataclass(frozen=True)
class LayerNorm(Module):
    """Over the last axis, with weight and bias (torch.nn.LayerNorm). Computed in f32 inside
    for b16 inputs, like every composite (docs/design.md, "数値一致の判定")."""

    dim: int
    eps: float = 1e-5

    def init(self, rng: np.random.Generator) -> dict:
        return {"weight": np.ones(self.dim, np.float32), "bias": np.zeros(self.dim, np.float32)}

    def __call__(self, params, x: Tensor) -> Tensor:
        return layernorm(x, params["weight"], params["bias"], eps=self.eps)


@dataclass(frozen=True)
class Repeat(Module):
    """`block` applied n times, each with its own parameters, stacked on a leading axis of
    every leaf. Traced as one scan (frontend/trace.repeat), so the step graph holds one copy of
    the block and the backward differentiates through the scan; unrolled=True traces n copies
    (for comparisons)."""

    block: Module
    n: int

    def init(self, rng: np.random.Generator) -> dict:
        layers = [tree_flatten(self.block.init(rng)) for _ in range(self.n)]
        structure = layers[0][1]
        return tree_unflatten(structure, [np.stack(leaves) for leaves in zip(*(flat for flat, _ in layers))])

    def __call__(self, params, x: Tensor, *args, unrolled: bool = False) -> Tensor:
        if unrolled:
            for i in range(self.n):
                x = self.block(tree_map(lambda leaf: leaf[i], params), x, *args)
            return x
        return repeat(lambda h, p: self.block(p, h, *args), self.n, x, params)


@dataclass(frozen=True)
class RMSNorm(Module):
    """x / sqrt(mean(x^2) + eps) * weight over the last axis, as Hugging Face's LlamaRMSNorm:
    normalized in f32, cast back to the input dtype, then scaled by the weight (in the weight's
    dtype, as PyTorch promotes f32 * bf16). Written with core ops rather than as a composite:
    PyTorch runs these steps as separate ops, so each rounding here is one PyTorch makes too."""

    dim: int
    eps: float = 1e-6

    def init(self, rng: np.random.Generator) -> dict:
        return {"weight": np.ones(self.dim, np.float32)}

    def __call__(self, params, x: Tensor) -> Tensor:
        h = _to(x, DType.f32)
        y = h * rsqrt(mean(h * h, axis=-1).reshape(*h.shape[:-1], 1) + self.eps)
        weight = params["weight"]
        return weight * _to(_to(y, x.dtype), weight.dtype)


def rotary_tables(seq: int, dim: int, theta: float = 10000.0) -> tuple[Tensor, Tensor]:
    """(cos, sin) of rotary position embeddings, [seq, dim] in f32, as Hugging Face computes
    them (the "default" rope type): inverse frequencies in f32, positions times frequencies in
    f32, each frequency used for both halves of the head dimension."""
    inv_freq = (1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))).astype(np.float32)
    positions = iota((seq,), dtype=DType.f32)
    freqs = positions.reshape(seq, 1) * constant(inv_freq).reshape(1, dim // 2)
    emb = concat([freqs, freqs], axis=1)
    return cos(emb), sin(emb)


def rotate(x: Tensor, cos_t: Tensor, sin_t: Tensor) -> Tensor:
    """x * cos + rotate_half(x) * sin over the last axis (x [..., seq, dim], tables [seq, dim])."""
    half = x.shape[-1] // 2
    rotated = concat([neg(x[..., half:]), x[..., :half]], axis=-1)
    # The tables are f32 even for an f64 model, as in Hugging Face, and widened to x's dtype.
    return x * broadcast_to(_to(cos_t, x.dtype), x.shape) + rotated * broadcast_to(_to(sin_t, x.dtype), x.shape)


@dataclass(frozen=True)
class Conv(Module):
    """torch.nn.Conv{1,2,3}d: weight [out, in/groups, *kernel] (the number of spatial dims is
    len(kernel)), optional bias. compute: as Linear (autocast runs convolutions in bf16)."""

    in_channels: int
    out_channels: int
    kernel: tuple[int, ...]
    stride: int | tuple = 1
    padding: int | tuple = 0
    dilation: int | tuple = 1
    groups: int = 1
    bias: bool = True
    compute: DType | None = None

    def init(self, rng: np.random.Generator) -> dict:
        # torch's reset_parameters: kaiming_uniform_(a=sqrt(5)) is U(-1/sqrt(fan_in), +), and
        # the bias uses the same bound.
        fan_in = self.in_channels // self.groups * int(np.prod(self.kernel))
        bound = 1.0 / math.sqrt(fan_in)
        params = {"weight": rng.uniform(-bound, bound, (self.out_channels, self.in_channels // self.groups, *self.kernel)).astype(np.float32)}
        if self.bias:
            params["bias"] = rng.uniform(-bound, bound, (self.out_channels,)).astype(np.float32)
        return params

    def __call__(self, params, x: Tensor) -> Tensor:
        y = conv(_to(x, self.compute), _to(params["weight"], self.compute), self.stride, self.padding, self.dilation, self.groups)
        if self.bias:
            b = _to(params["bias"], self.compute)
            y = y + broadcast_to(b.reshape(self.out_channels, *(1,) * len(self.kernel)), y.shape)
        return y


@dataclass(frozen=True)
class BatchNorm(Module):
    """torch.nn.BatchNorm{1,2,3}d over axis 1, with running statistics as buffers.

    Called as bn(params, x, buffers, training) -> (y, new buffers): olaf modules hold no state,
    so the running mean and variance come in and go out like any value (the step writes them
    back, frontend/optim.train_step). Training normalizes with the batch's biased variance and
    updates the running variance with the unbiased one, as PyTorch. Computed in f32 and rounded
    once to x's dtype, as cuDNN does for bf16 inputs.
    """

    channels: int
    eps: float = 1e-5
    momentum: float = 0.1

    def init(self, rng: np.random.Generator) -> dict:
        return {"weight": np.ones(self.channels, np.float32), "bias": np.zeros(self.channels, np.float32)}

    def init_buffers(self) -> dict:
        return {"running_mean": np.zeros(self.channels, np.float32), "running_var": np.ones(self.channels, np.float32)}

    def __call__(self, params, x: Tensor, buffers, training: bool = True):
        h = _to(x, DType.f32) if x.dtype is DType.bf16 else x
        axes = (0, *range(2, h.rank))
        shape = (1, self.channels, *(1,) * (h.rank - 2))
        if training:
            count = int(np.prod([h.shape[a] for a in axes]))
            mu = mean(h, axis=axes)
            centered = h - broadcast_to(mu.reshape(*shape), h.shape)
            var = mean(centered * centered, axis=axes)
            m = self.momentum
            new = {
                "running_mean": buffers["running_mean"] * (1.0 - m) + mu * m,
                "running_var": buffers["running_var"] * (1.0 - m) + var * (m * count / max(count - 1, 1)),
            }
        else:
            mu, var, new = buffers["running_mean"], buffers["running_var"], buffers
            centered = h - broadcast_to(mu.reshape(*shape), h.shape)
        scale = params["weight"] * rsqrt(var + self.eps)
        y = centered * broadcast_to(scale.reshape(*shape), h.shape) + broadcast_to(params["bias"].reshape(*shape), h.shape)
        return _to(y, x.dtype), new


@dataclass(frozen=True)
class ConvTranspose(Module):
    """torch.nn.ConvTranspose{1,2,3}d: weight [in, out/groups, *kernel], optional bias."""

    in_channels: int
    out_channels: int
    kernel: tuple[int, ...]
    stride: int | tuple = 1
    padding: int | tuple = 0
    output_padding: int | tuple = 0
    groups: int = 1
    bias: bool = True
    dilation: int | tuple = 1
    compute: DType | None = None

    def init(self, rng: np.random.Generator) -> dict:
        # torch computes fan_in from the weight's second axis: out/groups * kernel size.
        fan_in = self.out_channels // self.groups * int(np.prod(self.kernel))
        bound = 1.0 / math.sqrt(fan_in)
        params = {"weight": rng.uniform(-bound, bound, (self.in_channels, self.out_channels // self.groups, *self.kernel)).astype(np.float32)}
        if self.bias:
            params["bias"] = rng.uniform(-bound, bound, (self.out_channels,)).astype(np.float32)
        return params

    def __call__(self, params, x: Tensor) -> Tensor:
        y = conv_transpose(_to(x, self.compute), _to(params["weight"], self.compute), self.stride, self.padding, self.output_padding, self.dilation, self.groups)
        if self.bias:
            b = _to(params["bias"], self.compute)
            y = y + broadcast_to(b.reshape(self.out_channels, *(1,) * len(self.kernel)), y.shape)
        return y


def leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    return where(x > 0.0, x, x * slope)


def binary_cross_entropy_with_logits(logits: Tensor, targets) -> Tensor:
    """Mean of max(x, 0) - x * t + log(1 + exp(-|x|)), PyTorch's stable form."""
    t = targets if isinstance(targets, Tensor) else broadcast_to(constant(float(targets), logits.dtype), logits.shape)
    return mean(maximum(logits, 0.0) - logits * t + log(exp(neg(abs(logits))) + 1.0))


def _strided(x: Tensor, axis: int, start: int, count: int, step: int) -> Tensor:
    """x[..., start : start + count * step : step, ...] along `axis` (no strided slice op: a
    slice, a reshape splitting off the step, and a slice of its first column). x must reach
    start + count * step along the axis."""
    index = [slice(None)] * x.rank
    index[axis] = slice(start, start + count * step)
    piece = x[tuple(index)]
    if step == 1:
        return piece
    shape = list(piece.shape)
    split = shape[:axis] + [count, step] + shape[axis + 1 :]
    first = [slice(None)] * (x.rank + 1)
    first[axis + 1] = slice(0, 1)
    return piece.reshape(*split)[tuple(first)].reshape(*shape[:axis], count, *shape[axis + 1 :])


def _windows(x: Tensor, kernel, stride, padding, fill: float):
    """The kernel's shifted, strided views of x (padded with `fill`), one per kernel offset."""
    n = len(kernel)
    stride = tuple(kernel) if stride is None else (stride,) * n if isinstance(stride, int) else tuple(stride)
    padding = (padding,) * n if isinstance(padding, int) else tuple(padding)
    sizes = x.shape[2:]
    out = [(s + 2 * p - k) // st + 1 for s, p, k, st in zip(sizes, padding, kernel, stride)]
    # Pad so every shifted window reaches start + out * stride (the strided slice needs it).
    high = [max(p, (k - 1) + o * st - s - p) for s, p, k, st, o in zip(sizes, padding, kernel, stride, out)]
    if any(padding) or any(high):
        x = pad(x, (0, 0, *padding), (0, 0, *high), value=fill)
    views = []
    for offset in np.ndindex(*kernel):
        v = x
        for d, (o, size, st) in enumerate(zip(offset, out, stride)):
            v = _strided(v, 2 + d, int(o), size, st)
        views.append(v)
    return views


def max_pool(x: Tensor, kernel, stride=None, padding=0) -> Tensor:
    """torch.nn.functional.max_pool{1,2,3}d (padding counts as -inf). Exact in any dtype.
    Where several entries tie for the max the gradient goes to one of them; after a ReLU the
    ties are zeros, whose gradient the ReLU stops anyway."""
    views = _windows(x, tuple(kernel), stride, padding, -math.inf)
    y = views[0]
    for v in views[1:]:
        y = maximum(y, v)
    return y


def avg_pool(x: Tensor, kernel, stride=None, padding=0) -> Tensor:
    """torch.nn.functional.avg_pool{1,2,3}d with count_include_pad=True. Summed in f32 and
    rounded once for bf16, as PyTorch does."""
    views = _windows(_to(x, DType.f32) if x.dtype is DType.bf16 else x, tuple(kernel), stride, padding, 0.0)
    total = views[0]
    for v in views[1:]:
        total = total + v
    return _to(total * (1.0 / float(np.prod(kernel))), x.dtype)


def flatten(x: Tensor, start: int = 1) -> Tensor:
    return x.reshape(*x.shape[:start], int(np.prod(x.shape[start:])))


def _linear_taps(size_in: int, size_out: int):
    """Source indices and weights of torch's linear interpolation with align_corners=False:
    output i samples (i + 0.5) * in / out - 0.5, clamped at 0, between floor and floor + 1."""
    scale = size_in / size_out
    src = np.maximum((np.arange(size_out) + 0.5) * scale - 0.5, 0.0)
    lo = np.minimum(np.floor(src).astype(np.int64), size_in - 1)
    hi = np.minimum(lo + 1, size_in - 1)
    frac = (src - lo).astype(np.float64)
    return lo.astype(np.int32), hi.astype(np.int32), frac


def interpolate_bilinear(x: Tensor, size: tuple[int, int]) -> Tensor:
    """torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False) for
    x [N, C, H, W]: per axis, a gather of the two neighbours and a blend with fixed weights."""
    for axis, out in ((2, size[0]), (3, size[1])):
        lo, hi, frac = _linear_taps(x.shape[axis], out)
        shape = [1] * x.rank
        shape[axis] = out
        w = broadcast_to(constant(frac.astype(np.float64 if x.dtype is DType.f64 else np.float32), x.dtype).reshape(*shape), (*x.shape[:axis], out, *x.shape[axis + 1 :]))
        a, b = take(x, constant(lo), axis=axis), take(x, constant(hi), axis=axis)
        x = a + (b - a) * w
    return x


def relu(x: Tensor) -> Tensor:
    return maximum(x, 0.0)


__all__ = ["interpolate_bilinear", "BatchNorm", "Conv", "ConvTranspose", "binary_cross_entropy_with_logits", "leaky_relu", "Embedding", "LayerNorm", "Linear", "Module", "RMSNorm", "Repeat", "Sequential", "avg_pool", "flatten", "max_pool", "relu", "rotary_tables", "rotate"]
