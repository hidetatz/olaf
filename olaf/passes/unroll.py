"""Unroll scans with a static length into straight-line ops.

P1 runs a step by graph replay, which needs a flat launch list, so scan becomes copies of its
body. The copies print identically, so the tile tasks they become share templates (and compile
and search results) by digest; nothing is lost by unrolling. The persistent megakernel (P2) may
keep the loop instead, which is why this is a pass and not part of tile conversion.

xs[i] is a slice of the stacked input, and ys are concatenated along the leading axis. Tile
conversion turns both into address arithmetic, so the unrolled form costs no extra copies.
"""

from __future__ import annotations

from olaf.graph.builder import GraphBuilder, inline
from olaf.graph.ir import Graph, Op, Value


def unroll_scans(graph: Graph) -> Graph:
    b = GraphBuilder()
    subst: dict[Value, Value] = {}
    for op in graph.ops:
        inputs = [subst.get(v, v) for v in op.inputs]
        if op.kind != "scan":
            b.append(Op(op.kind, tuple(inputs), op.outputs, op.attrs, tuple(unroll_scans(r) for r in op.regions)))
            continue
        length = op.attr("length")
        if not isinstance(length, int):
            raise NotImplementedError(f"unrolling a scan of symbolic length {length} is not supported")
        body = unroll_scans(op.regions[0])
        n_carry, reverse = op.attr("n_carry"), op.attr("reverse", False)
        carry, xs = inputs[:n_carry], inputs[n_carry:]
        ys: list[list[Value]] = [[] for _ in op.outputs[n_carry:]]
        for i in reversed(range(length)) if reverse else range(length):
            sliced = []
            for x in xs:
                shape = x.type.shape
                part = b.op("slice", x, starts=(i, *(0 for _ in shape[1:])), stops=(i + 1, *shape[1:]))
                sliced.append(b.op("reshape", part, shape=shape[1:]))
            out = inline(b, body, [*carry, *sliced])
            carry = out[:n_carry]
            for stack, y in zip(ys, out[n_carry:]):
                stack.append(b.op("reshape", y, shape=(1, *y.type.shape)))
        for value, c in zip(op.outputs[:n_carry], carry):
            subst[value] = c
        for value, stack in zip(op.outputs[n_carry:], ys):
            # Each iteration's y goes to its own index, also for a reverse scan.
            ordered = stack[::-1] if reverse else stack
            subst[value] = b.op("concat", *ordered, axis=0) if length else _empty(b, value)
    return Graph(graph.params, tuple(b.ops), tuple(subst.get(v, v) for v in graph.results))


def _empty(b: GraphBuilder, value: Value) -> Value:
    return b.full(value.type.shape, 0, value.type.dtype)
