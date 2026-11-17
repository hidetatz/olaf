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

from olaf.frontend.trace import Tensor, broadcast_to, cast, concat, constant, cos, einsum, iota, layernorm, maximum, mean, neg, repeat, rsqrt, sin, take, tree_flatten, tree_map, tree_unflatten
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


def relu(x: Tensor) -> Tensor:
    return maximum(x, 0.0)


__all__ = ["Embedding", "LayerNorm", "Linear", "Module", "RMSNorm", "Repeat", "Sequential", "relu", "rotary_tables", "rotate"]
