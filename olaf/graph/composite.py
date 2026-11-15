"""Decomposition rules from composite ops to core ops.

Rules are part of an op's definition, so they live in the graph layer; the pass that rewrites whole graphs is passes/decompose.py.
The helpers without a leading underscore are shared with the composite VJPs in frontend/autodiff.py,
so a VJP recomputes exactly what the forward decomposition computes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from olaf.graph.builder import GraphBuilder
from olaf.graph.dim import Dim
from olaf.graph.ir import Value
from olaf.graph.types import DType

Rule = Callable[[GraphBuilder, Sequence[Value], dict], list[Value]]
RULES: dict[str, Rule] = {}


def _rule(name: str):
    def wrap(fn: Rule) -> Rule:
        RULES[name] = fn
        return fn

    return wrap


def element_count(b: GraphBuilder, dims: Sequence[Dim], dtype: DType) -> Value:
    """Number of elements spanned by `dims`, as a scalar of `dtype`.

    Symbolic dimensions are counted at run time (a sum of ones), so no retrace is needed. The count
    is taken in i32 and cast once: summing float ones stops at 256 in bf16 (256 + 1 rounds to 256),
    and multiplying float counts rounds too. i32 is enough below 2^31 elements per counted span.
    The core op list has no "read a symbol's value" op, so the sum of ones stands in for one; lowering
    to Tile IR must recognise reduce(sum, full(d, 1)) and read the bound value of d instead of
    emitting a reduction task (docs/design.md).
    """
    static = 1
    symbolic: list[Dim] = []
    for d in dims:
        if isinstance(d, int):
            static *= d
        else:
            symbolic.append(d)
    if not symbolic:
        return b.constant(float(static), dtype)
    count = b.constant(static, DType.i32)
    for d in symbolic:
        count = b.op("mul", count, b.op("reduce", b.full((d,), 1, DType.i32), fn="sum", axes=(0,)))
    return b.op("cast", count, dtype=dtype)


def keep_axes(b: GraphBuilder, reduced: Value, like: Value, axes: tuple[int, ...]) -> Value:
    kept = tuple(i for i in range(like.type.rank) if i not in axes)
    return b.op("broadcast", reduced, shape=like.type.shape, dims=kept)


def reduced_mean(b: GraphBuilder, v: Value, axes: tuple[int, ...], denominator: Value) -> Value:
    return b.op("div", b.op("reduce", v, fn="sum", axes=axes), denominator)


def _shifted_by_max(b: GraphBuilder, x: Value, axis: int) -> Value:
    # Subtract the max before exp. GPT-2 logits reach about -100, where fp32 exp would flush
    # everything to 0 (below exp(-104)) and the sum would become 0.
    return b.op("sub", x, keep_axes(b, b.op("reduce", x, fn="max", axes=(axis,)), x, (axis,)))


@_rule("softmax")
def _softmax(b, inputs, attrs):
    x, axis = inputs[0], attrs["axis"]
    e = b.op("exp", _shifted_by_max(b, x, axis))
    total = keep_axes(b, b.op("reduce", e, fn="sum", axes=(axis,)), e, (axis,))
    return [b.op("div", e, total)]


@_rule("gelu")
def _gelu(b, inputs, attrs):
    x = inputs[0]
    half = b.full_like(x, 0.5)
    one = b.full_like(x, 1.0)
    if attrs["approximate"] == "none":
        inner = b.op("erf", b.op("mul", x, b.full_like(x, math.sqrt(0.5))))
    else:
        inner = b.op("tanh", gelu_tanh_argument(b, x))
    return [b.op("mul", b.op("mul", half, x), b.op("add", one, inner))]


def gelu_tanh_argument(b: GraphBuilder, x: Value) -> Value:
    # x*x*x rather than pow(x, 3): pow lowers to exp and log on devices, and its VJP adds a log path.
    cube = b.op("mul", b.op("mul", x, x), x)
    arg = b.op("add", x, b.op("mul", b.full_like(x, 0.044715), cube))
    return b.op("mul", b.full_like(x, math.sqrt(2.0 / math.pi)), arg)


def layernorm_stats(b: GraphBuilder, x: Value, axes: tuple[int, ...], eps: float) -> tuple[Value, Value, Value]:
    """Return (x - mean, rstd, element count). rstd and the count have the reduced shape.

    rsqrt runs on the reduced shape and is broadcast afterwards. Broadcasting first would evaluate
    it once per element (768 times per row in GPT-2) instead of once per row.
    """
    kept = tuple(i for i in range(x.type.rank) if i not in axes)
    reduced_shape = tuple(x.type.shape[i] for i in kept)
    count = element_count(b, [x.type.shape[a] for a in axes], x.type.dtype)
    denominator = b.op("broadcast", count, shape=reduced_shape, dims=()) if reduced_shape else count
    centered = b.op("sub", x, keep_axes(b, reduced_mean(b, x, axes, denominator), x, axes))
    variance = reduced_mean(b, b.op("mul", centered, centered), axes, denominator)
    rstd = b.op("rsqrt", b.op("add", variance, b.full(reduced_shape, eps, x.type.dtype)))
    return centered, rstd, denominator


@_rule("layernorm")
def _layernorm(b, inputs, attrs):
    x, axes = inputs[0], attrs["axes"]
    centered, rstd, _ = layernorm_stats(b, x, axes, attrs["eps"])
    y = b.op("mul", centered, keep_axes(b, rstd, x, axes))
    if len(inputs) == 3:
        aligned = {name: b.op("broadcast", v, shape=x.type.shape, dims=axes) for name, v in zip(("w", "b"), inputs[1:])}
        y = b.op("add", b.op("mul", y, aligned["w"]), aligned["b"])
    return [y]


def attention_scores(b: GraphBuilder, q: Value, k: Value, mask: Value | None) -> tuple[Value, Value]:
    """Return (masked, scaled scores, the scale broadcast to the scores' shape)."""
    batch = "abcdefghijklmnopqrstuvw"[: q.type.rank - 2]
    scores = b.op("contract", q, k, spec=f"{batch}xz,{batch}yz->{batch}xy")
    scale = b.op("rsqrt", element_count(b, [q.type.shape[-1]], q.type.dtype))
    scale = b.op("broadcast", scale, shape=scores.type.shape, dims=())
    scores = b.op("mul", scores, scale)
    if mask is not None:
        scores = b.op("where", mask, scores, b.full_like(scores, -math.inf))
    return scores, scale


@_rule("attention")
def _attention(b, inputs, attrs):
    q, k, v = inputs[:3]
    batch = "abcdefghijklmnopqrstuvw"[: q.type.rank - 2]
    scores, _ = attention_scores(b, q, k, inputs[3] if len(inputs) == 4 else None)
    weights = _softmax(b, [scores], {"axis": scores.type.rank - 1})[0]
    return [b.op("contract", weights, v, spec=f"{batch}xy,{batch}yz->{batch}xz")]


@_rule("cross_entropy")
def _cross_entropy(b, inputs, attrs):
    logits, labels = inputs
    axis = logits.type.rank - 1
    shifted = _shifted_by_max(b, logits, axis)
    total = b.op("reduce", b.op("exp", shifted), fn="sum", axes=(axis,))
    # Gather before subtracting log(total): only N log-probabilities are needed, not N x V.
    picked = b.op("sub", b.op("gather", shifted, labels, axis=axis, batch_dims=axis), b.op("log", total))
    loss = b.op("neg", b.op("reduce", picked, fn="sum", axes=tuple(range(axis))))
    return [b.op("div", loss, element_count(b, labels.type.shape, logits.type.dtype))]
