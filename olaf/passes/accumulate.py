"""Sums of products that differ only in their operands become one product of the stacked operands.

A weight shared across the steps of an unrolled scan (an RNN's W_hh) receives one gradient term
per step: sum_t dpre_t^T h_{t-1}, each term a [B, H] x [B, H] product into [H, H] (K = B, 64
rows), accumulated by a chain of adds. The terms' operands are all there before the last add,
so the sum is one product whose contracted axis runs over every step: K = T * B. cuDNN computes
it this way; as T products it was 27 ms of a 68 ms RNN step (each writing 4 MB to add).

Matched: a chain of adds, each adding one term to the previous partial sum (used by nothing
else), where a term is a contract used only there, possibly through a cast; every term with the
same spec, operand types and cast, and a label summed in both operands. The operands are
concatenated along that label (tile.normalize.eliminate_concats lets their producers store into
the stacks) and the chain becomes base + (cast of) the one product, at the last add.

Rounding: each b16 term was rounded to b16 before its cast and add. When the terms are cast,
the stacks are cast instead (an exact widening, which the lowering folds into a b16 matrix
product with an f32 result), so the merged sum is never rounded to b16. Rounding the merged
product to b16 and then casting would round the whole sum once at b16 precision, and the
chain's independent per-term roundings partly cancel: an RNN's W_hh gradient had 2.8x
PyTorch's error that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from olaf.graph.builder import GraphBuilder, dead_code_elimination
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import parse_contract
from olaf.graph.types import DType

# Fewer terms leave too little to gain to pay for the stacking.
MIN_TERMS = 3
# Only products whose stacked label is shorter than this: those are the inefficient ones (an RNN
# step's batch of 64). Longer ones run well alone, and stacking their operands copies them (the
# encoder output's gradient through each T5 decoder layer's cross-attention, K = 512, stacked
# slices of the layer stack that is otherwise written in place).
MAX_STACKED = 256
_VIEWS = frozenset({"slice", "reshape", "permute", "broadcast", "read_state", "constant", "concat"})


@dataclass(frozen=True)
class _Term:
    value: Value  # what the add reads
    contract: Op
    cast: Op | None

    @property
    def key(self) -> tuple:
        c = self.contract
        return (c.attr("spec"), tuple(v.type for v in c.inputs), None if self.cast is None else self.cast.attr("dtype"))


@dataclass
class _Chain:
    base: Value | None
    terms: list[_Term]
    adds: list[Op]


def merge_accumulated_contracts(graph: Graph) -> Graph:
    uses: dict[Value, int] = {}
    for op in graph.ops:
        for v in op.inputs:
            uses[v] = uses.get(v, 0) + 1
    for v in graph.results:
        uses[v] = uses.get(v, 0) + 1
    producer = {v: op for op in graph.ops for v in op.outputs}

    def term(v: Value) -> _Term | None:
        op = producer.get(v)
        if op is None or uses.get(v) != 1:
            return None
        cast = None
        if op.kind == "cast":
            cast, inner = op, op.inputs[0]
            op = producer.get(inner)
            if op is None or uses.get(inner) != 1:
                return None
        label = _label(op.attr("spec")) if op.kind == "contract" and len(op.inputs) == 2 else None
        if label is None:
            return None
        (ta, _), _ = parse_contract(op.attr("spec"))
        size = op.inputs[0].type.shape[ta.index(label)]
        if not isinstance(size, int) or size >= MAX_STACKED:
            return None
        if not all(producer.get(x) is not None and producer[x].kind not in _VIEWS for x in op.inputs):
            # Its operands must be computed, so their producers can store them into the stacks:
            # views of a stack (a layer's weight in a scan's parameter stack) would be copied.
            return None
        return _Term(v, op, cast)

    chains: dict[Value, _Chain] = {}
    for op in graph.ops:
        if op.kind != "add" or op.regions:
            continue
        a, b = op.inputs
        ta, tb = term(a), term(b)
        out = op.outputs[0]
        for prev, t in ((a, tb), (b, ta)):
            chain = chains.get(prev)
            if t is not None and chain is not None and uses.get(prev) == 1 and t.key == chain.terms[0].key:
                chains[out] = _Chain(chain.base, [*chain.terms, t], [*chain.adds, op])
                del chains[prev]
                break
        else:
            if ta is not None and tb is not None and ta.key == tb.key:
                chains[out] = _Chain(None, [ta, tb], [op])
            elif tb is not None:
                chains[out] = _Chain(a, [tb], [op])
            elif ta is not None:
                chains[out] = _Chain(b, [ta], [op])

    for chain in chains.values():
        # A term with a zero operand adds exactly zero (an RNN's first step multiplies the zero
        # initial state); dropped, the stacks hold no constant slice.
        nonzero = [t for t in chain.terms if not any(_zero(v, producer) for v in t.contract.inputs)]
        if nonzero:
            chain.terms = nonzero
    merged = {id(chain.adds[-1]): chain for chain in chains.values() if len(chain.terms) >= MIN_TERMS}
    if not merged:
        return graph
    skipped = {id(op) for chain in merged.values() for op in (*chain.adds[:-1], *(t.contract for t in chain.terms), *(t.cast for t in chain.terms if t.cast))}
    b = GraphBuilder()
    subst: dict[Value, Value] = {}
    for op in graph.ops:
        if id(op) in skipped:
            continue
        if id(op) in merged:
            subst[op.outputs[0]] = _emit(b, merged[id(op)], subst)
            continue
        b.append(Op(op.kind, tuple(subst.get(v, v) for v in op.inputs), op.outputs, op.attrs, op.regions))
    return dead_code_elimination(b.finish([subst.get(v, v) for v in graph.results], graph.params))


def _zero(v: Value, producer: dict) -> bool:
    """v is a constant zero, possibly cast or broadcast."""
    op = producer.get(v)
    while op is not None and op.kind in ("cast", "broadcast"):
        op = producer.get(op.inputs[0])
    if op is None or op.kind != "constant":
        return False
    data = op.attr("value")
    return not np.any(data.array if hasattr(data, "array") else np.asarray(data))


def _label(spec: str) -> str | None:
    """A label summed in both operands, the one to stack along."""
    (ta, tb), out = parse_contract(spec)
    return next((c for c in ta if c in tb and c not in out), None)


def _emit(b: GraphBuilder, chain: _Chain, subst: dict) -> Value:
    first = chain.terms[0].contract
    spec = first.attr("spec")
    label = _label(spec)
    (ta, tb), _ = parse_contract(spec)
    stacks = [
        b.op("concat", *(subst.get(t.contract.inputs[k], t.contract.inputs[k]) for t in chain.terms), axis=term.index(label))
        for k, term in enumerate((ta, tb))
    ]
    cast = chain.terms[0].cast
    if cast is not None:
        stacks = [b.op("cast", stack, dtype=cast.attr("dtype")) for stack in stacks]
    value = b.op("contract", *stacks, spec=spec)
    if chain.base is not None:
        value = b.op("add", subst.get(chain.base, chain.base), value)
    return value


def scatter_into_sums(graph: Graph) -> Graph:
    """x + scatter_add(zeros, i, u) is scatter_add(x, i, u), x a product (or such a sum itself).

    A tied embedding's gradient is the output product's dense gradient plus one scatter_add per
    use of the table (T5: the encoder's and the decoder's tokens). As separate terms, each
    scatter is a zero fill of the table's size and one more operand of the optimizer's update (66
    MB each of T5-small's 32128 x 512); scattered into the product's output, they touch the rows
    of the batch's tokens alone. Only when the scatter is used nowhere else, and in f32 or f64: a
    bf16 scatter_add is the exact sum rounded once, so it does not accumulate onto a bf16 x.
    """
    uses: dict[Value, int] = {}
    for op in graph.ops:
        for v in op.inputs:
            uses[v] = uses.get(v, 0) + 1
    for v in graph.results:
        uses[v] = uses.get(v, 0) + 1
    producer: dict[Value, Op] = {}

    def scatters(v: Value) -> list[Op] | None:
        """The scatter_adds (inner first) that make v from zeros, each used once."""
        chain: list[Op] = []
        while (op := producer.get(v)) is not None and op.kind == "scatter_add" and uses.get(v) == 1:
            chain.insert(0, op)
            v = op.inputs[0]
        return chain if chain and _zero(v, producer) else None

    def product(v: Value) -> bool:
        while (op := producer.get(v)) is not None and uses.get(v) == 1 and op.kind in ("cast", "scatter_add"):
            v = op.inputs[0]
        return op is not None and op.kind == "contract" and uses.get(v) == 1

    ops: list[Op] = []

    def emit(op: Op) -> None:
        ops.append(op)
        producer.update((v, op) for v in op.outputs)

    changed = False
    for op in graph.ops:
        pick = None
        if op.kind == "add" and op.outputs[0].type.dtype in (DType.f32, DType.f64):
            a, b = op.inputs
            pick = next(((x, chain) for x, y in ((a, b), (b, a)) if x is not y and (chain := scatters(y)) is not None and product(x)), None)
        if pick is None:
            emit(op)
            continue
        value, chain = pick
        for k, s in enumerate(chain):
            out = op.outputs[0] if k == len(chain) - 1 else Value(s.outputs[0].type)
            uses.setdefault(out, 1)
            emit(Op("scatter_add", (value, *s.inputs[1:]), (out,), s.attrs))
            value = out
        changed = True
    return dead_code_elimination(Graph(graph.params, tuple(ops), graph.results)) if changed else graph
