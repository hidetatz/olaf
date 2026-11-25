"""Masks of selections read back from the selected value.

z = where(u < v, v, u) is max(u, v), and u < v holds exactly when u < z (if u < v, z is v;
otherwise z is u and u < u is false, NaN included). Likewise z = where(u < v, u, v) is min(u, v)
and u < v holds exactly when z < v. Later readers of the mask, a ReLU's backward above all
(where(x > 0, x, 0) differentiates to where(x > 0, c, 0)), are given that comparison instead of
the mask, so the step keeps only z alive: z is kept anyway (the next layer reads it and its weight
gradient reads it again), while the mask or the pre-activation x would be one more tensor written
by the forward and read by the backward.

Writing ReLU as maximum(x, 0) instead keeps the same values but not the same gradient at ties:
maximum splits a tie evenly (as torch.maximum does), so its VJP compares x itself with 0 and x had
to be saved next to z. ResNet-18 wrote each residual sum twice in the forward, as x and as z.

A clip is two selections, y = where(lo < x, x, lo) and z = where(y < hi, y, hi), and with lo < hi
the first mask, lo < y, holds exactly when lo < z (y is never NaN: lo < NaN is false, so it is lo):
ReLU6's backward reads z alone.

Only the strict comparison qualifies: with le, z = where(u <= v, v, u) gives u <= z everywhere.
"""

from __future__ import annotations

from olaf.graph.builder import dead_code_elimination
from olaf.graph.ir import Graph, Op, Value


def masks_from_selections(graph: Graph) -> Graph:
    defs: dict[Value, Op] = {}
    subst: dict[Value, Value] = {}
    lower: dict[Value, tuple[Value, Value]] = {}  # y = max(lo, x) -> (lo, its mask)
    ops: list[Op] = []
    for op in graph.ops:
        current = Op(op.kind, tuple(subst.get(v, v) for v in op.inputs), op.outputs, op.attrs, op.regions)
        ops.append(current)
        defs.update((v, current) for v in current.outputs)
        if current.kind != "where" or current.inputs[0] in subst:
            continue
        mask, s, t = current.inputs
        cmp = defs.get(mask)
        if cmp is None or cmp.kind != "cmp" or cmp.attr("fn") != "lt":
            continue
        u, v = cmp.inputs
        z = current.outputs[0]
        if (s, t) == (v, u):
            operands = (u, z)
        elif (s, t) == (u, v):
            operands = (z, v)
        else:
            continue
        recomputed = Value(mask.type)
        ops.append(Op("cmp", operands, (recomputed,), cmp.attrs))
        subst[mask] = recomputed
        if (s, t) == (v, u):
            lower[z] = (u, mask)
        elif u in lower:
            lo, first = lower[u]
            a, b = _constant(lo, defs), _constant(v, defs)
            if a is not None and b is not None and a < b:
                clipped = Value(subst[first].type)
                ops.append(Op("cmp", (lo, z), (clipped,), cmp.attrs))
                subst[first] = clipped
    return dead_code_elimination(Graph(graph.params, tuple(ops), tuple(subst.get(v, v) for v in graph.results)))


def _constant(value: Value, defs: dict[Value, Op]) -> float | None:
    op = defs.get(value)
    while op is not None and op.kind == "broadcast":
        op = defs.get(op.inputs[0])
    if op is None or op.kind != "constant" or op.attr("value").shape != ():
        return None
    return float(op.attr("value").array)
