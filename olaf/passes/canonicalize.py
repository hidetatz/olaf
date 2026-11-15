"""View folding.

Rules are a declarative list of pattern -> result functions, so they can move to the P3
e-graph without being rewritten.
"""

from __future__ import annotations

from collections.abc import Callable

from olaf.graph.builder import dead_code_elimination, make_attrs
from olaf.graph.ir import Graph, Op, Value

# A rule returns a replacement Value (identity), a new Op, or None (no match).
Rule = Callable[[Op, dict[Value, Op]], Value | Op | None]


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


def _slice_identity(op, defs):
    if op.kind == "slice":
        x = op.inputs[0]
        if all(a == 0 for a in op.attr("starts")) and op.attr("stops") == x.type.shape:
            return x


def _pad_identity(op, defs):
    if op.kind == "pad" and all(d == 0 for d in (*op.attr("low"), *op.attr("high"))):
        return op.inputs[0]


RULES: tuple[Rule, ...] = (
    _reshape_identity,
    _reshape_of_reshape,
    _permute_identity,
    _permute_of_permute,
    _broadcast_identity,
    _slice_identity,
    _pad_identity,
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
