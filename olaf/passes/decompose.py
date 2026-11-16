"""Pass that decomposes composite ops into core ops."""

from __future__ import annotations

from olaf.graph.builder import GraphBuilder
from olaf.graph.composite import RULES, narrow_like, widen_bf16
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import OPS


def decompose(graph: Graph) -> Graph:
    b = GraphBuilder()
    subst: dict[Value, Value] = {}
    for op in graph.ops:
        inputs = [subst.get(v, v) for v in op.inputs]
        if OPS[op.kind].composite:
            # source_dtype tells a rule what dtype the composite had before widen_bf16.
            results = RULES[op.kind](b, widen_bf16(b, inputs), {**op.attr_dict, "source_dtype": op.inputs[0].type.dtype})
            subst.update(zip(op.outputs, narrow_like(b, results, op.outputs), strict=True))
        else:
            b.append(Op(op.kind, tuple(inputs), op.outputs, op.attrs, tuple(decompose(r) for r in op.regions)))
    return Graph(graph.params, tuple(b.ops), tuple(subst.get(v, v) for v in graph.results))
