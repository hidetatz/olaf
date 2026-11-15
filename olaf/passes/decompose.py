"""Pass that decomposes composite ops into core ops."""

from __future__ import annotations

from olaf.graph.builder import GraphBuilder
from olaf.graph.composite import RULES
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import OPS


def decompose(graph: Graph) -> Graph:
    b = GraphBuilder()
    subst: dict[Value, Value] = {}
    for op in graph.ops:
        inputs = [subst.get(v, v) for v in op.inputs]
        if OPS[op.kind].composite:
            subst.update(zip(op.outputs, RULES[op.kind](b, inputs, op.attr_dict), strict=True))
        else:
            b.append(Op(op.kind, tuple(inputs), op.outputs, op.attrs, tuple(decompose(r) for r in op.regions)))
    return Graph(graph.params, tuple(b.ops), tuple(subst.get(v, v) for v in graph.results))
