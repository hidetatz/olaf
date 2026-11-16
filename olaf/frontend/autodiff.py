"""Reverse-mode automatic differentiation on Graph IR (source transformation).

Whether to save activations is not decided here. Backward ops just reference forward values;
saving, recomputing or offloading is left to the memory planner. scan is the exception: nothing
else carries values across iterations, so residuals used by the body's backward pass become extra
forward scan outputs. Only values that depend on the iteration's carry are stacked that way; see _split_residuals.

Composite ops are differentiated as composites, with their own VJPs, and decomposed afterwards.
Decomposing first would leave no composite in a training graph for dedicated kernels to match
(design.md, IR chapter), and differentiating a decomposed softmax references every intermediate
(the masked scores, exp, the broadcast sum) instead of the single output, and sends gradient
through the max subtracted for stability, which cancels analytically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from olaf.graph.builder import GraphBuilder, dead_code_elimination
from olaf.graph.composite import attention_probabilities, attention_scores, element_count, gelu_tanh_argument, keep_axes, layernorm_stats, narrow_like, reduced_mean, widen_bf16
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import OPS, parse_contract
from olaf.graph.types import DType

Cot = Value | None
VJP = Callable[[GraphBuilder, Op, Value], Sequence[Cot]]
_RULES: dict[str, VJP] = {}
_NOT_DIFFERENTIABLE = frozenset({"cmp", "constant", "iota", "rng", "read_state", "write_state"})
_UNSUPPORTED = frozenset({"cond", "while", "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send", "recv"})


def grad(graph: Graph, wrt: Sequence[int], loss: int = 0) -> Graph:
    """Differentiate `graph.results[loss]` with respect to `graph.params[wrt]`.

    Results are [original results..., gradients in wrt order...]. Composites in `graph` stay composites,
    and their VJPs may emit composites too, so run passes.decompose before interpreting the result.
    """
    loss_value = graph.results[loss]
    if loss_value.type.shape != () or not loss_value.type.dtype.is_float:
        raise TypeError(f"the differentiated value must be a float scalar: {loss_value.type}")
    targets = [graph.params[i] for i in wrt]
    for t in targets:
        if not t.type.dtype.is_float:
            raise TypeError(f"cannot differentiate with respect to a non-float input: {t.type}")

    b = GraphBuilder()
    active = _activity(graph.ops, set(targets))
    records = _augment(b, graph.ops, active)
    cots: dict[Value, Value] = {loss_value: b.full_like(loss_value, 1.0)}
    _reverse(b, graph.ops, cots, active, records)
    grads = [cots[t] if t in cots else b.full_like(t, 0.0) for t in targets]
    return dead_code_elimination(Graph(graph.params, tuple(b.ops), (*graph.results, *grads)))


def _activity(ops: Sequence[Op], seeds: set[Value]) -> set[Value]:
    """Find float values that depend on the differentiation targets."""
    active = set(seeds)
    for op in ops:
        if not any(v in active for v in op.inputs) or op.kind in _NOT_DIFFERENTIABLE:
            continue
        if op.kind in _UNSUPPORTED:
            raise NotImplementedError(f"differentiating {op.kind} is not supported (P0 supports only scan)")
        if op.kind == "scan":
            active.update(op.outputs[i] for i in _scan_activity(op, active).result_indices)
        else:
            active.update(v for v in op.outputs if v.type.dtype.is_float)
    return active


@dataclass(frozen=True)
class _ScanActivity:
    carry: tuple[int, ...]
    xs: tuple[int, ...]
    ys: tuple[int, ...]
    n_carry: int

    @property
    def result_indices(self) -> list[int]:
        return [*self.carry, *(self.n_carry + y for y in self.ys)]


def _scan_activity(op: Op, active: set[Value]) -> _ScanActivity:
    body, n_carry = op.regions[0], op.attr("n_carry")
    xs = tuple(i for i, v in enumerate(op.inputs[n_carry:]) if v in active)
    carry = {i for i, v in enumerate(op.inputs[:n_carry]) if v in active}
    # A carry can become active from a previous iteration's output, so iterate to a fixed point.
    while True:
        seeds = {body.params[i] for i in carry} | {body.params[n_carry + i] for i in xs}
        inner = _activity(body.ops, seeds)
        grown = carry | {i for i, r in enumerate(body.results[:n_carry]) if r in inner}
        if grown == carry:
            break
        carry = grown
    ys = tuple(i for i, r in enumerate(body.results[n_carry:]) if r in inner)
    return _ScanActivity(tuple(sorted(carry)), xs, ys, n_carry)


@dataclass(frozen=True, eq=False)
class _ScanRecord:
    activity: _ScanActivity
    residuals: tuple[Value, ...]  # extra forward scan outputs (residuals stacked per iteration)
    invariants: tuple[Value, ...]  # outer values of pass-through carries, passed to the backward scan as carries
    xs: tuple[Value, ...]  # outer xs, passed to the backward scan as xs
    backward: Graph


def _augment(b: GraphBuilder, ops: Sequence[Op], active: set[Value]) -> dict[Op, _ScanRecord]:
    """Copy forward ops into b, replacing active scans with versions that output residuals."""
    records: dict[Op, _ScanRecord] = {}
    for op in ops:
        if op.kind == "scan" and any(v in active for v in op.inputs):
            activity = _scan_activity(op, active)
            aug_body, backward, residual_types, invariant_carries, xs_indices = _differentiate_body(op, activity)
            length = op.attr("length")
            residuals = tuple(Value(t.with_shape((length, *t.shape))) for t in residual_types)
            b.append(Op("scan", op.inputs, (*op.outputs, *residuals), op.attrs, (aug_body,)))
            invariants = tuple(op.inputs[j] for j in invariant_carries)
            xs = tuple(op.inputs[activity.n_carry + i] for i in xs_indices)
            records[op] = _ScanRecord(activity, residuals, invariants, xs, backward)
        else:
            b.append(op)
    return records


def _differentiate_body(op: Op, activity: _ScanActivity):
    """From a scan body, build a forward body that also returns residuals, and a backward body.

    Backward body inputs are [carry cotangents..., invariant carries..., y cotangents..., residuals..., xs...];
    outputs are [carry cotangents..., invariant carries..., x cotangents...].
    """
    body, n_carry = op.regions[0], activity.n_carry
    seeds = {body.params[i] for i in activity.carry} | {body.params[n_carry + i] for i in activity.xs}
    active = _activity(body.ops, seeds)

    aug = GraphBuilder()
    inner_records = _augment(aug, body.ops, active)

    bwd = GraphBuilder()
    cots: dict[Value, Value] = {}
    cot_params: list[Value] = []
    for i in (*activity.carry, *(n_carry + y for y in activity.ys)):
        param = bwd.param(body.results[i].type)
        cot_params.append(param)
        _accumulate(bwd, cots, body.results[i], param)
    _reverse(bwd, body.ops, cots, active, inner_records)

    wanted = [*(body.params[i] for i in activity.carry), *(body.params[n_carry + i] for i in activity.xs)]
    outs = [cots[p] if p in cots else bwd.full_like(p, 0.0) for p in wanted]
    pruned = dead_code_elimination(Graph(tuple(cot_params), tuple(bwd.ops), tuple(outs)))

    # Collect forward values after DCE; otherwise unused values would be stacked for every iteration.
    defined = set(cot_params)
    needed: list[Value] = []

    def need(value: Value) -> None:
        if value not in defined and value not in needed:
            needed.append(value)

    for inner_op in pruned.ops:
        for value in inner_op.inputs:
            need(value)
        defined.update(inner_op.outputs)
    for value in outs:
        need(value)

    stacked, prologue, subst, invariant_carries, xs_indices = _split_residuals(body, n_carry, aug.ops, needed)
    stacked_params = {r: Value(r.type) for r in stacked}
    subst.update(stacked_params)
    rewritten = [Op(o.kind, tuple(subst.get(v, v) for v in o.inputs), o.outputs, o.attrs, o.regions) for o in pruned.ops]
    n_carry_cots = len(activity.carry)
    invariant_params = [subst[body.params[j]] for j in invariant_carries]
    xs_params = [subst[body.params[n_carry + i]] for i in xs_indices]
    results = [subst.get(v, v) for v in outs]
    backward = Graph(
        (*cot_params[:n_carry_cots], *invariant_params, *cot_params[n_carry_cots:], *stacked_params.values(), *xs_params),
        (*prologue, *rewritten),
        (*results[:n_carry_cots], *invariant_params, *results[n_carry_cots:]),
    )
    augmented = Graph(body.params, tuple(aug.ops), (*body.results, *stacked))
    return augmented, backward, [r.type for r in stacked], invariant_carries, xs_indices


def _split_residuals(body: Graph, n_carry: int, forward_ops: Sequence[Op], needed: Sequence[Value]):
    """Decide which forward values the backward body receives stacked, and which it recomputes.

    A value that does not depend on the iteration's carry is computed from pass-through carries
    (captured weights, masks), the iteration's xs (per-layer weights) and constants only. Stacking it
    stores `length` copies of something the backward scan can rebuild from inputs it already has, so
    it is recomputed at the start of the backward body. For GPT-2 these are the per-layer weights
    themselves, broadcast biases and constants, and the attention mask, all cheap to rebuild. This is
    provisional: an expensive value built from xs (a LoRA A @ B, dequantized weights) trades length
    copies of memory against per-layer recompute, which is the memory planner's call. When it lands
    (P2), only cheap ops (views, broadcasts, constants) are recomputed here and the rest are passed
    on as recomputable residual candidates (docs/design.md, autodiff). Values that depend on the
    carry are activations; whether to keep or recompute those is the memory planner's call (P2),
    so they stay stacked. Measured on GPT-2 small at B=4, S=1024 in f32, this and the composite
    VJPs cut the stacked residuals from 19.4 GiB to 2.25 GiB, ten activations of [B,S,768] or [B,S,3072].

    Returns (stacked values, recomputing ops, value substitution, used pass-through carry indices, used xs indices).
    """
    passthrough = {body.params[j]: j for j in range(n_carry) if body.results[j] is body.params[j]}
    xs_params = {p: i for i, p in enumerate(body.params[n_carry:])}
    producer = {v: o for o in forward_ops for v in o.outputs}
    independent = set(passthrough) | set(xs_params)
    for o in forward_ops:
        # Ops without inputs (constant, iota) are independent too. Scan bodies cannot touch state (verify).
        if not OPS[o.kind].has_effect and all(v in independent for v in o.inputs):
            independent.update(o.outputs)

    stacked = [v for v in needed if v not in independent]
    subst: dict[Value, Value] = {}
    used_carries: list[int] = []
    used_xs: list[int] = []
    cloned: set[Op] = set()
    order: list[Op] = []

    def visit(value: Value) -> None:
        if value in subst:
            return
        if value in passthrough:
            used_carries.append(passthrough[value])
            subst[value] = Value(value.type)
            return
        if value in xs_params:
            used_xs.append(xs_params[value])
            subst[value] = Value(value.type)
            return
        o = producer[value]
        if o in cloned:
            return
        cloned.add(o)
        for v in o.inputs:
            visit(v)
        for v in o.outputs:
            subst[v] = Value(v.type)
        order.append(o)

    for value in needed:
        if value in independent:
            visit(value)
    prologue = [Op(o.kind, tuple(subst[v] for v in o.inputs), tuple(subst[v] for v in o.outputs), o.attrs, o.regions) for o in order]
    return stacked, prologue, subst, sorted(used_carries), sorted(used_xs)


def _reverse(b: GraphBuilder, ops: Sequence[Op], cots: dict[Value, Value], active: set[Value], records: dict[Op, _ScanRecord]) -> None:
    for op in reversed(ops):
        if not any(v in cots for v in op.outputs):
            continue
        if op in records:
            _reverse_scan(b, op, records[op], cots, active)
            continue
        if op.kind in _NOT_DIFFERENTIABLE:
            continue
        inputs_cots = _vjp(b, op, cots[op.outputs[0]])
        for value, cot in zip(op.inputs, inputs_cots, strict=True):
            if cot is not None and value in active:
                _accumulate(b, cots, value, cot)


def _vjp(b: GraphBuilder, op: Op, cot: Value) -> list[Value | None]:
    if not OPS[op.kind].composite or not any(v.type.dtype is DType.bf16 for v in (*op.inputs, *op.outputs)):
        return _RULES[op.kind](b, op, cot)
    # A bf16 composite's backward computes in f32 too, matching its forward (graph.composite.widen_bf16).
    # The rule receives an f32 stand-in for the op. Its outputs are casts of the bf16 outputs
    # (softmax's VJP reads its output), as PyTorch's softmax backward reads the bf16 output.
    wide = Op(op.kind, tuple(widen_bf16(b, op.inputs)), tuple(widen_bf16(b, op.outputs)), op.attrs)
    return narrow_like(b, _RULES[op.kind](b, wide, widen_bf16(b, [cot])[0]), op.inputs)


def _reverse_scan(b: GraphBuilder, op: Op, record: _ScanRecord, cots: dict[Value, Value], active: set[Value]) -> None:
    activity, n_carry = record.activity, record.activity.n_carry

    def seed(i: int) -> Value:
        out = op.outputs[i]
        return cots[out] if out in cots else b.full_like(out, 0.0)

    carry_seeds = [seed(i) for i in activity.carry]
    y_seeds = [seed(n_carry + y) for y in activity.ys]
    n_backward_carry = len(carry_seeds) + len(record.invariants)
    attrs = {
        "n_carry": n_backward_carry,
        "length": op.attr("length"),
        "reverse": not op.attr("reverse", False),
    }
    inputs = [*carry_seeds, *record.invariants, *y_seeds, *record.residuals, *record.xs]
    outputs = b.emit("scan", inputs, attrs, [record.backward])
    targets = [*(op.inputs[i] for i in activity.carry), *(op.inputs[n_carry + i] for i in activity.xs)]
    cot_outputs = [*outputs[: len(carry_seeds)], *outputs[n_backward_carry:]]
    for value, cot in zip(targets, cot_outputs, strict=True):
        # activity.carry also holds carries that only become active inside the body (from active xs);
        # their initial value outside can be inactive, e.g. the output of a frozen scan or a while.
        if value in active:
            _accumulate(b, cots, value, cot)


def _accumulate(b: GraphBuilder, cots: dict[Value, Value], value: Value, cot: Value) -> None:
    cots[value] = b.op("add", cots[value], cot) if value in cots else cot


def _rule(*names: str):
    def wrap(fn: VJP) -> VJP:
        for name in names:
            _RULES[name] = fn
        return fn

    return wrap


def _mul(b, *values: Value) -> Value:
    out = values[0]
    for v in values[1:]:
        out = b.op("mul", out, v)
    return out


_rule("neg")(lambda b, op, c: [b.op("neg", c)])
_rule("exp")(lambda b, op, c: [b.op("mul", c, op.outputs[0])])
_rule("log")(lambda b, op, c: [b.op("div", c, op.inputs[0])])
_rule("sin")(lambda b, op, c: [b.op("mul", c, b.op("cos", op.inputs[0]))])
_rule("cos")(lambda b, op, c: [b.op("neg", b.op("mul", c, b.op("sin", op.inputs[0])))])


@_rule("abs")
def _abs(b, op, c):
    x = op.inputs[0]
    zero = b.full_like(x, 0.0)
    # The subgradient at 0 is 0, matching PyTorch.
    negative = b.op("where", b.op("cmp", x, zero, fn="lt"), b.op("neg", c), zero)
    return [b.op("where", b.op("cmp", zero, x, fn="lt"), c, negative)]


@_rule("sqrt")
def _sqrt(b, op, c):
    return [b.op("mul", c, b.op("div", b.full_like(c, 0.5), op.outputs[0]))]


@_rule("rsqrt")
def _rsqrt(b, op, c):
    return [_mul(b, c, b.full_like(c, -0.5), op.outputs[0], b.op("recip", op.inputs[0]))]


@_rule("recip")
def _recip(b, op, c):
    return [b.op("neg", _mul(b, c, op.outputs[0], op.outputs[0]))]


@_rule("tanh")
def _tanh(b, op, c):
    y = op.outputs[0]
    return [b.op("mul", c, b.op("sub", b.full_like(y, 1.0), b.op("mul", y, y)))]


@_rule("erf")
def _erf(b, op, c):
    x = op.inputs[0]
    return [_mul(b, c, b.full_like(x, 2.0 / math.sqrt(math.pi)), b.op("exp", b.op("neg", b.op("mul", x, x))))]


@_rule("cast")
def _cast(b, op, c):
    # Nothing flows into integer sources. Activity analysis already drops them; the rule states it too.
    source = op.inputs[0].type.dtype
    return [b.op("cast", c, dtype=source) if source.is_float else None]


@_rule("add")
def _add(b, op, c):
    return [c, c]


@_rule("sub")
def _sub(b, op, c):
    return [c, b.op("neg", c)]


@_rule("mul")
def _mul_rule(b, op, c):
    x, y = op.inputs
    return [b.op("mul", c, y), b.op("mul", c, x)]


@_rule("div")
def _div(b, op, c):
    x, y = op.inputs
    return [b.op("div", c, y), b.op("neg", b.op("mul", b.op("div", c, y), op.outputs[0]))]


@_rule("pow")
def _pow(b, op, c):
    x, y = op.inputs
    one = b.full_like(x, 1.0)
    dx = _mul(b, c, y, b.op("pow", x, b.op("sub", y, one)))
    # log diverges for x <= 0, so the exponent's gradient is 0 there (same as PyTorch).
    zero = b.full_like(x, 0.0)
    safe_log = b.op("where", b.op("cmp", zero, x, fn="lt"), b.op("log", b.op("where", b.op("cmp", zero, x, fn="lt"), x, one)), zero)
    return [dx, _mul(b, c, op.outputs[0], safe_log)]


def _max_min_rule(less_side: int):
    def rule(b, op, c):
        # Ties split evenly (same as PyTorch maximum / minimum).
        x, y = op.inputs
        half = b.op("mul", c, b.full_like(c, 0.5))
        zero = b.full_like(c, 0.0)
        equal = b.op("cmp", x, y, fn="eq")
        less = b.op("cmp", x, y, fn="lt")
        winner_is_y = less if less_side == 1 else b.op("cmp", y, x, fn="lt")
        return [
            b.op("where", equal, half, b.op("where", winner_is_y, zero, c)),
            b.op("where", equal, half, b.op("where", winner_is_y, c, zero)),
        ]

    return rule


_rule("max")(_max_min_rule(1))
_rule("min")(_max_min_rule(0))


@_rule("where")
def _where(b, op, c):
    cond = op.inputs[0]
    zero = b.full_like(c, 0.0)
    return [None, b.op("where", cond, c, zero), b.op("where", cond, zero, c)]


@_rule("reduce")
def _reduce(b, op, c):
    x, axes = op.inputs[0], op.attr("axes")
    kept = tuple(i for i in range(x.type.rank) if i not in axes)
    spread = b.op("broadcast", c, shape=x.type.shape, dims=kept)
    if op.attr("fn") == "sum":
        return [spread]
    # Multiple maxima share equally (same as PyTorch amax).
    peak = b.op("broadcast", op.outputs[0], shape=x.type.shape, dims=kept)
    mask = b.op("cast", b.op("cmp", x, peak, fn="eq"), dtype=x.type.dtype)
    count = b.op("broadcast", b.op("reduce", mask, fn="sum", axes=axes), shape=x.type.shape, dims=kept)
    return [b.op("div", b.op("mul", spread, mask), count)]


@_rule("contract")
def _contract(b, op, c):
    terms, out = parse_contract(op.attr("spec"))
    cots: list[Cot] = []
    for i, term in enumerate(terms):
        others = [t for j, t in enumerate(terms) if j != i]
        visible = set(out).union(*others) if others else set(out)
        # A label only in this term and not in the output was summed away, so the cotangent lacks it.
        # The VJP of that sum spreads uniformly along the label.
        kept = "".join(label for label in term if label in visible)
        spec = ",".join([*others, out]) + "->" + kept
        inputs = [v for j, v in enumerate(op.inputs) if j != i] + [c]
        cot = b.op("contract", *inputs, spec=spec)
        if len(kept) != len(term):
            cot = b.op("broadcast", cot, shape=op.inputs[i].type.shape, dims=tuple(term.index(label) for label in kept))
        cots.append(cot)
    return cots


@_rule("reshape")
def _reshape(b, op, c):
    return [b.op("reshape", c, shape=op.inputs[0].type.shape)]


@_rule("permute")
def _permute(b, op, c):
    perm = op.attr("perm")
    return [b.op("permute", c, perm=tuple(perm.index(i) for i in range(len(perm))))]


@_rule("broadcast")
def _broadcast(b, op, c):
    x, shape, dims = op.inputs[0], op.attr("shape"), op.attr("dims")
    axes = [i for i in range(len(shape)) if i not in dims]
    axes += [d for i, d in enumerate(dims) if x.type.shape[i] != shape[d]]
    summed = b.op("reduce", c, fn="sum", axes=tuple(sorted(axes))) if axes else c
    return [b.op("reshape", summed, shape=x.type.shape)]


@_rule("slice")
def _slice(b, op, c):
    x, starts, stops = op.inputs[0], op.attr("starts"), op.attr("stops")
    high = tuple(d - s for d, s in zip(x.type.shape, stops))
    return [b.op("pad", c, low=starts, high=high, value=0.0)]


@_rule("concat")
def _concat(b, op, c):
    axis = op.attr("axis")
    shape = op.outputs[0].type.shape
    cots: list[Cot] = []
    offset = 0
    for value in op.inputs:
        end = offset + value.type.shape[axis]
        starts = tuple(offset if i == axis else 0 for i in range(len(shape)))
        stops = tuple(end if i == axis else d for i, d in enumerate(shape))
        cots.append(b.op("slice", c, starts=starts, stops=stops))
        offset = end
    return cots


@_rule("pad")
def _pad(b, op, c):
    x, low = op.inputs[0], op.attr("low")
    return [b.op("slice", c, starts=low, stops=tuple(l + d for l, d in zip(low, x.type.shape)))]


@_rule("gather")
def _gather(b, op, c):
    x, index = op.inputs
    attrs = {"axis": op.attr("axis"), "batch_dims": op.attr("batch_dims")}
    return [b.op("scatter_add", b.full_like(x, 0.0), index, c, **attrs), None]


@_rule("scatter_add")
def _scatter_add(b, op, c):
    _, index, _ = op.inputs
    attrs = {"axis": op.attr("axis"), "batch_dims": op.attr("batch_dims")}
    return [c, None, b.op("gather", c, index, **attrs)]



# Composite VJPs. Each references the composite's inputs (and softmax its output) and recomputes the
# rest with the same helpers as the decomposition, so the saved set per composite is fixed and small.


def _softmax_backward(b, y: Value, c: Value, axis: int) -> Value:
    dot = keep_axes(b, b.op("reduce", b.op("mul", c, y), fn="sum", axes=(axis,)), y, (axis,))
    return b.op("mul", y, b.op("sub", c, dot))


@_rule("softmax")
def _softmax(b, op, c):
    return [_softmax_backward(b, op.outputs[0], c, op.attr("axis"))]


@_rule("gelu")
def _gelu(b, op, c):
    x = op.inputs[0]
    half, one = b.full_like(x, 0.5), b.full_like(x, 1.0)
    if op.attr("approximate") == "none":
        cdf = b.op("mul", half, b.op("add", one, b.op("erf", b.op("mul", x, b.full_like(x, math.sqrt(0.5))))))
        pdf = b.op("mul", b.full_like(x, 1.0 / math.sqrt(2.0 * math.pi)), b.op("exp", b.op("mul", b.full_like(x, -0.5), b.op("mul", x, x))))
        slope = b.op("add", cdf, b.op("mul", x, pdf))
    else:
        # d/dx 0.5 x (1 + tanh(u)) = 0.5 (1 + t) + 0.5 x (1 - t^2) u', u' = sqrt(2/pi) (1 + 3 * 0.044715 x^2)
        t = b.op("tanh", gelu_tanh_argument(b, x))
        du = b.op("mul", b.full_like(x, math.sqrt(2.0 / math.pi)), b.op("add", one, b.op("mul", b.full_like(x, 3 * 0.044715), b.op("mul", x, x))))
        dt = b.op("mul", b.op("sub", one, b.op("mul", t, t)), du)
        slope = b.op("mul", half, b.op("add", b.op("add", one, t), b.op("mul", x, dt)))
    return [b.op("mul", c, slope)]


@_rule("layernorm")
def _layernorm(b, op, c):
    # dx = rstd * (g - mean(g) - xhat * mean(g * xhat)) with g = c * w, means over the normalized axes.
    # mean and rstd are recomputed from x: two reductions per row, against saving two more tensors.
    x, axes = op.inputs[0], op.attr("axes")
    centered, rstd, denominator = layernorm_stats(b, x, axes, op.attr("eps"))
    spread_rstd = keep_axes(b, rstd, x, axes)
    xhat = b.op("mul", centered, spread_rstd)
    affine = len(op.inputs) == 3
    g = b.op("mul", c, b.op("broadcast", op.inputs[1], shape=x.type.shape, dims=axes)) if affine else c
    mean_g = keep_axes(b, reduced_mean(b, g, axes, denominator), x, axes)
    mean_gx = keep_axes(b, reduced_mean(b, b.op("mul", g, xhat), axes, denominator), x, axes)
    dx = b.op("mul", spread_rstd, b.op("sub", b.op("sub", g, mean_g), b.op("mul", xhat, mean_gx)))
    if not affine:
        return [dx]
    kept = tuple(i for i in range(x.type.rank) if i not in axes)

    def over_kept(v: Value) -> Value:
        return b.op("reduce", v, fn="sum", axes=kept) if kept else v

    return [dx, over_kept(b.op("mul", c, xhat)), over_kept(c)]


@_rule("attention")
def _attention(b, op, c):
    # Saves q, k, v (and the mask) and recomputes the probabilities: they are [..., Tq, Tk] while the
    # inputs are [..., T, D]. Unlike FlashAttention, which also saves the per-row log-sum-exp and so
    # rebuilds P in one pass, this recomputes the row max and sum too, because a composite's VJP sees
    # only its inputs. Exposing the LSE as an auxiliary output is a P1 question (docs/design.md).
    q, k, v = op.inputs[:3]
    mask = op.inputs[3] if len(op.inputs) == 4 else None
    batch = "abcdefghijklmnopqrstuvw"[: q.type.rank - 2]
    scores, scale = attention_scores(b, q, k, mask)
    # Not the softmax composite: the probabilities must match the forward decomposition, which gives
    # rows with every key masked p = 0 instead of NaN.
    p = attention_probabilities(b, scores, mask is not None)
    dv = b.op("contract", p, c, spec=f"{batch}xy,{batch}xz->{batch}yz")
    dp = b.op("contract", c, v, spec=f"{batch}xz,{batch}yz->{batch}xy")
    # Masked positions have p = 0, so their score gradient is 0 without another where. A fully
    # masked row has p = 0 everywhere, so dq for it and its share of dk and dv are 0 too.
    ds = b.op("mul", _softmax_backward(b, p, dp, scores.type.rank - 1), scale)
    dq = b.op("contract", ds, k, spec=f"{batch}xy,{batch}yz->{batch}xz")
    dk = b.op("contract", ds, q, spec=f"{batch}xy,{batch}xz->{batch}yz")
    return [dq, dk, dv] + ([None] if mask is not None else [])


@_rule("cross_entropy")
def _cross_entropy(b, op, c):
    # d mean(-log softmax(z)[label]) / dz = (softmax(z) - onehot(label)) / N
    logits, labels = op.inputs
    axis = logits.type.rank - 1
    per_item = b.op("div", c, element_count(b, labels.type.shape, logits.type.dtype))
    p = b.op("softmax", logits, axis=axis)
    spread = b.op("mul", p, b.op("broadcast", per_item, shape=logits.type.shape, dims=()))
    picked = b.op("broadcast", per_item, shape=labels.type.shape, dims=()) if labels.type.shape else per_item
    # Scatter -1/N into the softmax term directly instead of building a one-hot N x V tensor and subtracting it.
    return [b.op("scatter_add", spread, labels, b.op("neg", picked), axis=axis, batch_dims=axis), None]
