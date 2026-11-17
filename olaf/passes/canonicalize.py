"""View folding.

Rules are a declarative list of pattern -> result functions, so they can move to the P3
e-graph without being rewritten.
"""

from __future__ import annotations

from collections.abc import Callable

from olaf.graph.builder import dead_code_elimination, make_attrs
from olaf.graph.ir import Graph, Op, Value

# A rule returns a replacement Value (identity), a new Op, new Ops (the last one defines the
# result), or None (no match).
Rule = Callable[[Op, dict[Value, Op]], Value | Op | tuple[Op, ...] | None]


def _reshape_identity(op, defs):
    if op.kind == "reshape" and op.attr("shape") == op.inputs[0].type.shape:
        return op.inputs[0]


def _reshape_of_reshape(op, defs):
    inner = defs.get(op.inputs[0]) if op.inputs else None
    if op.kind == "reshape" and inner is not None and inner.kind == "reshape":
        return Op("reshape", inner.inputs, op.outputs, op.attrs)


def _permute_identity(op, defs):
    if op.kind == "permute" and op.attr("perm") == tuple(range(op.inputs[0].type.rank)):
        return op.inputs[0]


def _permute_of_permute(op, defs):
    inner = defs.get(op.inputs[0]) if op.inputs else None
    if op.kind == "permute" and inner is not None and inner.kind == "permute":
        composed = tuple(inner.attr("perm")[p] for p in op.attr("perm"))
        return Op("permute", inner.inputs, op.outputs, make_attrs({"perm": composed}))


def _broadcast_identity(op, defs):
    if op.kind == "broadcast" and op.attr("shape") == op.inputs[0].type.shape:
        return op.inputs[0]


def _broadcast_of_unit_reshape(op, defs):
    """broadcast(reshape(x)), the reshape only inserting size-1 axes, is broadcast(x).

    A row statistic written keepdim-style (mean(...).reshape(..., 1) * x, as RMSNorm usually is)
    otherwise reaches Tile IR through a reshape, whose view has no source map, and row fusion
    cannot see that it is broadcast along the row.
    """
    inner = defs.get(op.inputs[0]) if op.inputs else None
    if op.kind != "broadcast" or inner is None or inner.kind != "reshape":
        return None
    a, b = inner.inputs[0].type.shape, inner.outputs[0].type.shape
    mapping, j = [], 0
    for size in a:
        while j < len(b) and b[j] != size:
            if b[j] != 1:
                return None
            j += 1
        if j == len(b):
            return None
        mapping.append(j)
        j += 1
    if any(b[k] != 1 for k in range(j, len(b))):
        return None
    dims = tuple(op.attr("dims")[k] for k in mapping)
    return Op("broadcast", inner.inputs, op.outputs, make_attrs({"shape": op.attr("shape"), "dims": dims}))


def _slice_identity(op, defs):
    if op.kind == "slice":
        x = op.inputs[0]
        if all(a == 0 for a in op.attr("starts")) and op.attr("stops") == x.type.shape:
            return x


def _slice_of_concat(op, defs):
    """A slice that selects exactly one concat operand is that operand.

    Unrolled scans stack each iteration's outputs with concat, and the backward scan slices
    iteration i back out; without this rule the backward reads a copy, and common
    subexpression elimination cannot see that it is the forward's value.
    """
    inner = defs.get(op.inputs[0]) if op.inputs else None
    if op.kind != "slice" or inner is None or inner.kind != "concat":
        return None
    axis = inner.attr("axis")
    starts, stops = op.attr("starts"), op.attr("stops")
    full = all(a == 0 and b == d for i, (a, b, d) in enumerate(zip(starts, stops, op.inputs[0].type.shape)) if i != axis)
    if not full:
        return None
    position = 0
    for operand in inner.inputs:
        size = operand.type.shape[axis]
        if starts[axis] == position and stops[axis] == position + size:
            return operand
        position = position + size
    return None


def _pad_identity(op, defs):
    if op.kind == "pad" and all(d == 0 for d in (*op.attr("low"), *op.attr("high"))):
        return op.inputs[0]


def _add_of_pads(op, defs):
    """pad(a) + pad(b), zero-padded along one axis to adjacent ranges, is pad(concat(a, b)).

    The gradient of a value sliced into pieces (GPT-2's fused qkv projection split into q, k
    and v) is the sum of each piece's gradient zero-padded back to full size: as adds, two extra
    passes over the whole gradient; as a concat, the pieces' producers store into its slices
    (eliminate_concats). Not bit-identical in one case: -0.0 + 0.0 is +0.0, the concat keeps -0.0.
    """
    if op.kind != "add":
        return None
    pads = [defs.get(v) for v in op.inputs]
    if any(p is None or p.kind != "pad" or p.attr("value") != 0.0 for p in pads):
        return None
    (a, b) = pads
    differ = [i for i, w in enumerate(zip(a.attr("low"), a.attr("high"), b.attr("low"), b.attr("high"))) if any(x != 0 for x in w)]
    if len(differ) != 1 or any(not isinstance(x, int) for x in (*a.attr("low"), *a.attr("high"), *b.attr("low"), *b.attr("high"))):
        return None
    axis = differ[0]
    extent = op.outputs[0].type.shape[axis]
    if a.attr("low")[axis] > b.attr("low")[axis]:
        a, b = b, a
    if extent - a.attr("high")[axis] != b.attr("low")[axis]:
        return None  # not adjacent (overlapping, or a gap between)
    x, y = a.inputs[0], b.inputs[0]
    joined = Value(x.type.with_shape(d + y.type.shape[i] if i == axis else d for i, d in enumerate(x.type.shape)))
    concat = Op("concat", (x, y), (joined,), make_attrs({"axis": axis}))
    concat = _concat_of_concat(concat, defs) or concat  # (dq, concat(dk, dv)) is one concat
    low = tuple(a.attr("low")[i] if i == axis else 0 for i in range(len(a.attr("low"))))
    high = tuple(b.attr("high")[i] if i == axis else 0 for i in range(len(b.attr("high"))))
    return concat, Op("pad", (joined,), op.outputs, make_attrs({"low": low, "high": high, "value": 0.0}))


def _concat_identity(op, defs):
    """A concat of one operand is that operand. An unrolled scan of length 1 (a one-layer model)
    stacks each saved activation with one; left in place, the backward reads the stack instead of
    the forward's value, and attention's backward fusion no longer sees its pattern."""
    if op.kind == "concat" and len(op.inputs) == 1:
        return op.inputs[0]


def _concat_of_concat(op, defs):
    if op.kind != "concat":
        return None
    axis = op.attr("axis")
    flat = []
    for v in op.inputs:
        inner = defs.get(v)
        flat.extend(inner.inputs if inner is not None and inner.kind == "concat" and inner.attr("axis") == axis else (v,))
    if len(flat) == len(op.inputs):
        return None
    return Op("concat", tuple(flat), op.outputs, op.attrs)


RULES: tuple[Rule, ...] = (
    _reshape_identity,
    _reshape_of_reshape,
    _permute_identity,
    _permute_of_permute,
    _broadcast_identity,
    _broadcast_of_unit_reshape,
    _slice_identity,
    _slice_of_concat,
    _pad_identity,
    _add_of_pads,
    _concat_identity,
    _concat_of_concat,
)


def canonicalize(graph: Graph) -> Graph:
    subst: dict[Value, Value] = {}
    defs: dict[Value, Op] = {}
    ops: list[Op] = []
    for op in graph.ops:
        current = Op(op.kind, tuple(subst.get(v, v) for v in op.inputs), op.outputs, op.attrs, tuple(canonicalize(r) for r in op.regions))
        changed = True
        while changed:
            changed = False
            for rule in RULES:
                result = rule(current, defs)
                if isinstance(result, Value):
                    subst[current.outputs[0]] = result
                    current = None
                    break
                if isinstance(result, tuple):
                    *before, result = result
                    ops.extend(before)
                    defs.update((v, o) for o in before for v in o.outputs)
                if result is not None:
                    current = result
                    changed = True
                    break
            if current is None:
                break
        if current is not None:
            ops.append(current)
            defs.update((v, current) for v in current.outputs)
    return dead_code_elimination(Graph(graph.params, tuple(ops), tuple(subst.get(v, v) for v in graph.results)))
