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


def widen_bf16(b: GraphBuilder, values: Sequence[Value]) -> list[Value]:
    """Cast bf16 values to f32 (others pass through): composites compute bf16 inputs in f32.

    PyTorch (and autocast) compute the inside of layernorm, softmax, gelu and cross entropy in f32
    for bf16 tensors and round only the outputs. Decomposing in bf16 would round every
    intermediate (mean, variance, max, exp sums) instead, which measurably moves the loss: on a
    small layernorm-gelu graph the bf16 loss error grew from below 1e-3 to 9e-3. Matching
    PyTorch's convention is what lets a bf16 training run follow PyTorch's loss curve.
    """
    return [b.op("cast", v, dtype=DType.f32) if v.type.dtype is DType.bf16 else v for v in values]


def narrow_like(b: GraphBuilder, values: Sequence[Value | None], like: Sequence[Value]) -> list[Value | None]:
    """Cast results computed in f32 back to the dtype of the corresponding original value."""
    return [v if v is None or v.type.dtype is ref.type.dtype else b.op("cast", v, dtype=ref.type.dtype) for v, ref in zip(values, like, strict=True)]


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


def sigmoid(b: GraphBuilder, x: Value) -> Value:
    # 1 / (1 + exp(-x)): for x below about -88, exp(-x) overflows to inf in f32 and the result is
    # 0, the correctly rounded value; no branch on the sign is needed.
    return b.op("recip", b.op("add", b.full_like(x, 1.0), b.op("exp", b.op("neg", x))))


@_rule("silu")
def _silu(b, inputs, attrs):
    x = inputs[0]
    return [b.op("mul", x, sigmoid(b, x))]


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


def batchnorm_stats(b: GraphBuilder, x: Value, axes: tuple[int, ...]) -> tuple[Value, Value, Value]:
    """Return (mean, biased variance, element count), all of the kept shape, in one pass over x.

    The two-pass form (layernorm_stats) reads the activation once for the mean and again for
    the variance, and a batch norm runs over every activation of a CNN. Here both sums are of
    d = x - k for a per-channel shift k (the channel's first element), so they read the same
    input and sibling fusion makes them one pass: mean = k + E[d], var = E[d^2] - E[d]^2.
    Without the shift, E[x^2] - E[x]^2 cancels catastrophically in f32 once the mean is large
    against the spread (a channel at mean 100 and deviation 1 keeps about 3 digits); with k a
    value of the channel, E[d] is a few deviations at most and the cancellation costs about
    one digit. The max with 0 covers the rounding when every value is (nearly) equal.
    """
    rank = x.type.rank
    kept = tuple(i for i in range(rank) if i not in axes)
    reduced_shape = tuple(x.type.shape[i] for i in kept)
    count = element_count(b, [x.type.shape[a] for a in axes], x.type.dtype)
    denominator = b.op("broadcast", count, shape=reduced_shape, dims=()) if reduced_shape else count
    first = b.op("slice", x, starts=(0,) * rank, stops=tuple(1 if a in axes else d for a, d in enumerate(x.type.shape)))
    shift = b.op("reshape", first, shape=reduced_shape)
    d = b.op("sub", x, keep_axes(b, shift, x, axes))
    mean_d = reduced_mean(b, d, axes, denominator)
    mean_d2 = reduced_mean(b, b.op("mul", d, d), axes, denominator)
    variance = b.op("max", b.op("sub", mean_d2, b.op("mul", mean_d, mean_d)), b.full(reduced_shape, 0.0, x.type.dtype))
    return b.op("add", shift, mean_d), variance, denominator


def batchnorm_normalized(b: GraphBuilder, x: Value, axes: tuple[int, ...], eps: float) -> tuple[Value, Value, Value]:
    """Return (x - mean, rstd, element count), as layernorm_stats but from batchnorm_stats."""
    mean, variance, denominator = batchnorm_stats(b, x, axes)
    rstd = b.op("rsqrt", b.op("add", variance, b.full(variance.type.shape, eps, x.type.dtype)))
    return b.op("sub", x, keep_axes(b, mean, x, axes)), rstd, denominator


def batchnorm_scale(b: GraphBuilder, rstd: Value, weight: Value) -> Value:
    """weight * rstd per channel. Taken before broadcasting so the normalization costs one
    multiply and one add per element: with a third (xhat * weight), it no longer counts as
    cheap enough to be duplicated into the ReLU that follows (tile.fuse.CHEAP_OPS) and is
    written out in an extra pass instead."""
    return b.op("mul", rstd, weight)


@_rule("batchnorm")
def _batchnorm(b, inputs, attrs):
    x, weight, bias = inputs
    axes = attrs["axes"]
    centered, rstd, _ = batchnorm_normalized(b, x, axes, attrs["eps"])
    y = b.op("mul", centered, keep_axes(b, batchnorm_scale(b, rstd, weight), x, axes))
    return [b.op("add", y, keep_axes(b, bias, x, axes))]


def per_channel(b: GraphBuilder, grouped: Value, channels: int) -> Value:
    """[N, G] -> [N, C], each group's value repeated over its C / G channels."""
    n, g = grouped.type.shape
    wide = b.op("broadcast", grouped, shape=(n, g, channels // g), dims=(0, 1))
    return b.op("reshape", wide, shape=(n, channels))


def over_group(b: GraphBuilder, channels: Value, groups: int) -> Value:
    """[N, C] -> [N, G], each group's channels summed."""
    n, c = channels.type.shape
    return b.op("reduce", b.op("reshape", channels, shape=(n, groups, c // groups)), fn="sum", axes=(2,))


def groupnorm_normalized(b: GraphBuilder, x: Value, groups: int, eps: float) -> tuple[Value, Value, Value]:
    """Return (x - mean [N, C, *spatial], rstd [N, C], elements per group [N, G]).

    One pass over x, as batchnorm_stats (whose shift it uses too: the first element, here of
    each group). The pass sums each channel over the positions, and the channels of a group
    are summed on the [N, C] results: reducing [N, G, C/G, H, W] over (C/G, H, W) instead
    straddles the kept group axis in channels-last memory. The statistics are repeated to [N, C]
    for the same reason: in terms of the groups the normalized tensor has an axis more than its
    readers (the activation, the convolution), no view of [N, G] reads "channel // (C/G)", and
    nothing would be fused across (an f32 copy of the activation per norm and per SiLU).
    """
    n, channels = x.type.shape[:2]
    spatial = tuple(range(2, x.type.rank))
    size = channels // groups
    count = element_count(b, [size, *(x.type.shape[a] for a in spatial)], x.type.dtype)
    denominator = b.op("broadcast", count, shape=(n, groups), dims=())
    grouped = b.op("reshape", x, shape=(n, groups, size, *x.type.shape[2:]))
    first = b.op("slice", grouped, starts=(0,) * grouped.type.rank, stops=(n, groups, *(1,) * (grouped.type.rank - 2)))
    shift = b.op("reshape", first, shape=(n, groups))
    d = b.op("sub", x, keep_axes(b, per_channel(b, shift, channels), x, spatial))
    sum_d, sum_d2 = (b.op("reduce", v, fn="sum", axes=spatial) for v in (d, b.op("mul", d, d)))
    mean_d, mean_d2 = (b.op("div", over_group(b, s, groups), denominator) for s in (sum_d, sum_d2))
    variance = b.op("max", b.op("sub", mean_d2, b.op("mul", mean_d, mean_d)), b.full((n, groups), 0.0, x.type.dtype))
    rstd = b.op("rsqrt", b.op("add", variance, b.full((n, groups), eps, x.type.dtype)))
    mean = per_channel(b, b.op("add", shift, mean_d), channels)
    return b.op("sub", x, keep_axes(b, mean, x, spatial)), per_channel(b, rstd, channels), denominator


def groupnorm_scale(b: GraphBuilder, rstd: Value, weight: Value) -> Value:
    """rstd * weight [N, C], taken before broadcasting for batchnorm_scale's reason."""
    return b.op("mul", rstd, b.op("broadcast", weight, shape=rstd.type.shape, dims=(1,)))


@_rule("groupnorm")
def _groupnorm(b, inputs, attrs):
    x, weight, bias = inputs
    centered, rstd, _ = groupnorm_normalized(b, x, attrs["groups"], attrs["eps"])
    spatial = tuple(range(2, x.type.rank))
    y = b.op("mul", centered, keep_axes(b, groupnorm_scale(b, rstd, weight), x, spatial))
    return [b.op("add", y, b.op("broadcast", bias, shape=x.type.shape, dims=(1,)))]


def attention_inputs(inputs: Sequence[Value], attrs: dict) -> tuple[Value, Value, Value, Value | None, Value | None]:
    """(q, k, v, mask or None, bias or None) of an attention op."""
    biased = attrs.get("bias", False)
    masked = len(inputs) - biased == 4
    return (*inputs[:3], inputs[3] if masked else None, inputs[-1] if biased else None)


def attention_scores(b: GraphBuilder, q: Value, k: Value, mask: Value | None, bias: Value | None = None, scale: float | None = None) -> tuple[Value, Value | None]:
    """Return (masked, scaled, biased scores; the scale broadcast to the scores' shape, None for
    a scale of 1, which is not multiplied: T5's unscaled scores)."""
    batch = "abcdefghijklmnopqrstuvw"[: q.type.rank - 2]
    scores = b.op("contract", q, k, spec=f"{batch}xz,{batch}yz->{batch}xy")
    if scale is None:
        factor = b.op("rsqrt", element_count(b, [q.type.shape[-1]], q.type.dtype))
    elif scale != 1.0:
        factor = b.constant(scale, q.type.dtype)
    factor = b.op("broadcast", factor, shape=scores.type.shape, dims=()) if scale != 1.0 else None
    if factor is not None:
        scores = b.op("mul", scores, factor)
    if bias is not None:
        scores = b.op("add", scores, bias)
    if mask is not None:
        scores = b.op("where", mask, scores, b.full_like(scores, -math.inf))
    return scores, factor


def attention_probabilities(b: GraphBuilder, scores: Value, masked: bool) -> Value:
    """Softmax over the last axis, except that a row whose keys are all masked gets probabilities 0.

    That row's attention output and its gradients are therefore 0, where a plain softmax gives 0/0 =
    NaN (as PyTorch's math SDPA does). Such rows come from left padding, empty cross-attention
    sources, or filler positions in packed sequences; their outputs are discarded, but a NaN would
    still leak into gradients, since NaN * 0 is NaN. The P1 online softmax has to handle tiles whose
    keys are all masked anyway (exp(-inf - -inf) is NaN), and once it does, a fully masked row ends
    with a zero sum, so writing 0 there costs nothing. Without a mask no row can be empty, and this
    is the plain softmax.
    """
    axis = scores.type.rank - 1
    if not masked:
        return _softmax(b, [scores], {"axis": axis})[0]
    peak = b.op("reduce", scores, fn="max", axes=(axis,))
    # A max of -inf means every key is masked. Shifting by 0 instead keeps exp at exp(-inf) = 0
    # rather than exp(-inf - -inf) = NaN. Any row with a live key has a sum of at least exp(0) = 1, so
    # the same flag tells exactly when the sum is 0 and the divisor must be replaced.
    empty = b.op("cmp", peak, b.full_like(peak, -math.inf), fn="eq")
    shift = b.op("where", empty, b.full_like(peak, 0.0), peak)
    e = b.op("exp", b.op("sub", scores, keep_axes(b, shift, scores, (axis,))))
    total = b.op("reduce", e, fn="sum", axes=(axis,))
    total = b.op("where", empty, b.full_like(total, 1.0), total)
    return b.op("div", e, keep_axes(b, total, e, (axis,)))


def round_probabilities(b: GraphBuilder, p: Value, source: DType | None) -> Value:
    """Attention probabilities rounded to the inputs' dtype before they multiply v.

    The one exception to computing a bf16 composite in f32: PyTorch (`.type(value.dtype)` in
    HF's attention, and FlashAttention) multiplies bf16 probabilities with v, and doing the same
    lets that product run on bf16 matrix instructions. The rounded values are exact in f32.
    """
    if source is DType.bf16 and p.type.dtype is DType.f32:
        return b.op("cast", b.op("cast", p, dtype=DType.bf16), dtype=DType.f32)
    return p


@_rule("attention")
def _attention(b, inputs, attrs):
    q, k, v, mask, bias = attention_inputs(inputs, attrs)
    batch = "abcdefghijklmnopqrstuvw"[: q.type.rank - 2]
    scores, _ = attention_scores(b, q, k, mask, bias, attrs.get("scale"))
    weights = round_probabilities(b, attention_probabilities(b, scores, mask is not None), attrs.get("source_dtype"))
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
