"""Building and verifying Graph IR. Shared by tracing, autodiff and passes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from olaf.graph.ir import ConstantData, Graph, Op, Value, symbol_table
from olaf.graph.ops import OPS, infer_types
from olaf.graph.types import DType, TensorType


def make_attrs(attrs: Mapping) -> tuple:
    # Attribute order must not affect the text (the cache key).
    return tuple(sorted(attrs.items()))


class GraphBuilder:
    def __init__(self) -> None:
        self.params: list[Value] = []
        self.ops: list[Op] = []

    def param(self, type: TensorType) -> Value:
        value = Value(type)
        self.params.append(value)
        return value

    def emit(self, kind: str, inputs: Sequence[Value] = (), attrs: Mapping | None = None, regions: Sequence[Graph] = ()) -> tuple[Value, ...]:
        attrs = dict(attrs or {})
        regions = tuple(regions)
        types = infer_types(kind, [v.type for v in inputs], attrs, regions)
        outputs = tuple(Value(t) for t in types)
        self.ops.append(Op(kind, tuple(inputs), outputs, make_attrs(attrs), regions))
        return outputs

    def op(self, kind: str, *inputs: Value, **attrs) -> Value:
        """Append an op with a single output."""
        return self.emit(kind, inputs, attrs)[0]

    def append(self, op: Op) -> None:
        self.ops.append(op)

    def constant(self, value, dtype: DType) -> Value:
        return self.op("constant", value=ConstantData.of(value, dtype))

    def full(self, shape: Sequence, value, dtype: DType) -> Value:
        scalar = self.constant(value, dtype)
        if not shape:
            return scalar
        return self.op("broadcast", scalar, shape=tuple(shape), dims=())

    def full_like(self, like: Value, value) -> Value:
        return self.full(like.type.shape, value, like.type.dtype)

    def finish(self, results: Iterable[Value], params: Iterable[Value] | None = None) -> Graph:
        return Graph(tuple(self.params if params is None else params), tuple(self.ops), tuple(results))


def inline(builder: GraphBuilder, graph: Graph, inputs: Sequence[Value]) -> list[Value]:
    """Expand `graph` into builder. Output Values are recreated, so a graph can be inlined repeatedly."""
    if [v.type for v in inputs] != [p.type for p in graph.params]:
        raise TypeError("inline input types do not match")
    subst: dict[Value, Value] = dict(zip(graph.params, inputs))
    for op in graph.ops:
        outputs = tuple(Value(v.type) for v in op.outputs)
        subst.update(zip(op.outputs, outputs))
        builder.append(Op(op.kind, tuple(subst[v] for v in op.inputs), outputs, op.attrs, op.regions))
    return [subst[r] for r in graph.results]


def dead_code_elimination(graph: Graph) -> Graph:
    live = set(graph.results)
    kept: list[Op] = []
    for op in reversed(graph.ops):
        if OPS[op.kind].has_effect or any(v in live for v in op.outputs):
            kept.append(op)
            live.update(op.inputs)
    kept.reverse()
    return Graph(graph.params, tuple(kept), graph.results)


def verify(graph: Graph, *, in_region: bool = False) -> None:
    """Check SSA form, types, that regions are closed and the state rules. Also used on pass outputs.

    State rules: a step is a pure function from the old resident state to the new one. read_state
    always returns the value at the start of the step, so a read after a write of the same state and
    a second write are rejected; otherwise the result would depend on op order, which no data
    dependency records and which the tile DAG is free to change. In-place updates (buffer donation)
    then reduce to a data dependency question: the last use of the read value precedes the write.
    Regions (scan, cond, while) cannot touch state at all: scan iterations must stay reorderable, and
    a write under cond would make "written once" depend on run-time values. State goes through inputs.
    """
    symbol_table(graph)
    defined = set(graph.params)
    if len(defined) != len(graph.params):
        raise ValueError("a value appears twice among the params")
    written: set[str] = set()
    for op in graph.ops:
        if any(v in defined for v in op.outputs) or len(set(op.outputs)) != len(op.outputs):
            raise ValueError(f"{op.kind} defines a value that is already defined (SSA values are defined once)")
        missing = [v for v in op.inputs if v not in defined]
        if missing:
            raise ValueError(f"{op.kind} uses an undefined value (possibly one from outside the region)")
        expected = infer_types(op.kind, [v.type for v in op.inputs], op.attr_dict, op.regions)
        if expected != [v.type for v in op.outputs]:
            raise TypeError(f"{op.kind} has wrong output types: {[str(v.type) for v in op.outputs]} / {[str(t) for t in expected]}")
        if op.kind in ("read_state", "write_state"):
            name = op.attr("state").name
            if in_region:
                raise ValueError("a region (scan, cond, while) cannot read or write resident state; pass it as an input")
            if name in written:
                raise ValueError(f"resident state {name} is {'read' if op.kind == 'read_state' else 'written'} after a write in the same step")
            if op.kind == "write_state":
                written.add(name)
        for region in op.regions:
            verify(region, in_region=True)
        defined.update(op.outputs)
    if any(r not in defined for r in graph.results):
        raise ValueError("a result refers to an undefined value")
