"""Common subexpression elimination on Graph IR.

Composite VJPs recompute what they need from their inputs (frontend/autodiff.py); when the forward
values are still computed in the same step, the recomputation is the same ops on the same
values, and this pass merges them. For attention that is the whole probability computation of
the backward. Whether a value is then kept alive from the forward or recomputed is the memory
planner's decision (P2); until then, merging is the faster choice whenever memory allows, which
the runtime's arena makes visible.

Ops with effects (write_state, send, recv) and ops with regions are never merged; the rng op is
pure (its counter is an input), so equal rng ops do produce equal values and merge correctly.
"""

from __future__ import annotations

from olaf.graph.builder import dead_code_elimination
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import OPS


def eliminate_common_subexpressions(graph: Graph) -> Graph:
    subst: dict[Value, Value] = {}
    seen: dict[tuple, Op] = {}
    ops: list[Op] = []
    for op in graph.ops:
        inputs = tuple(subst.get(v, v) for v in op.inputs)
        current = Op(op.kind, inputs, op.outputs, op.attrs, tuple(eliminate_common_subexpressions(r) for r in op.regions))
        if OPS[op.kind].has_effect or op.regions:
            ops.append(current)
            continue
        key = (op.kind, tuple(id(v) for v in inputs), _attrs_key(op.attrs), tuple(v.type for v in op.outputs))
        if key in seen:
            subst.update(zip(op.outputs, seen[key].outputs))
            continue
        seen[key] = current
        ops.append(current)
    return dead_code_elimination(Graph(graph.params, tuple(ops), tuple(subst.get(v, v) for v in graph.results)))


def _attrs_key(attrs) -> tuple:
    # Attribute values are hashable dataclasses, tuples and scalars; floats compare by value,
    # except that 0.0 and -0.0 must stay distinct (a negative zero constant is a different value).
    return tuple((k, (v, repr(v)) if isinstance(v, float) else v) for k, v in attrs)
