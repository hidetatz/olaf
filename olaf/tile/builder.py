"""Tile op definitions (type inference) and the builder used by conversion and passes.

Tile ops mirror the Graph IR core ops where they can, so the elementwise part of a task is the same
op vocabulary on static shapes. What is new is memory access (load, store, the indexed variants)
and the structure a device kernel needs (program ids, for loops, dot).

Scalars are rank-0 tiles. Index arithmetic uses i32: GPT-2 small's largest buffer (the 50257 x 768
embedding gradient, 3.9e7 elements) is far below 2^31, and i64 index math halves integer
throughput on GPUs. A launch whose offsets would pass 2^31 is rejected once the sizes are known
(backend/cuda/runtime.py, check_index_range); i64 lowering for such launches is not implemented.
"""

from __future__ import annotations

from collections.abc import Sequence

from olaf.graph.builder import make_attrs
from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.ir import Body, TileType, TOp, TValue

INDEX = DType.i32

UNARY = frozenset({"neg", "abs", "exp", "log", "sqrt", "rsqrt", "recip", "tanh", "erf", "sin", "cos"})
BINARY = frozenset({"add", "sub", "mul", "div", "max", "min", "pow", "floordiv", "mod"})
# floordiv and mod exist only for index arithmetic on integers; Graph IR has no integer division.


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise TypeError(message)


def infer(kind: str, inputs: Sequence[TileType], attrs: dict, regions: Sequence[Body]) -> list[TileType]:
    if kind in UNARY:
        _check(len(inputs) == 1, f"{kind} takes one input")
        return [inputs[0]]
    if kind in BINARY:
        a, b = inputs
        _check(a == b, f"{kind} input types differ: {a} and {b}")
        _check(kind not in ("floordiv", "mod") or not a.dtype.is_float, f"{kind} is integer-only: {a}")
        return [a]
    if kind == "cmp":
        a, b = inputs
        _check(a == b and attrs["fn"] in ("eq", "lt", "le"), f"invalid cmp: {a}, {b}, {attrs['fn']}")
        return [TileType(a.shape, DType.bool)]
    if kind == "where":
        c, a, b = inputs
        _check(c.dtype is DType.bool and c.shape == a.shape and a == b, f"invalid where: {c}, {a}, {b}")
        return [a]
    if kind == "cast":
        return [TileType(inputs[0].shape, attrs["dtype"])]
    if kind == "full":
        return [TileType(attrs["shape"], attrs["dtype"])]
    if kind == "dim":
        return [TileType((), INDEX)]
    if kind == "iota":
        # iota(start): start + arange along `axis`, constant along the others.
        (start,) = inputs
        _check(start == TileType((), INDEX) and 0 <= attrs["axis"] < len(attrs["shape"]), f"invalid iota: {start}, {attrs}")
        return [TileType(attrs["shape"], INDEX)]
    if kind == "broadcast":
        (x,) = inputs
        shape, dims = attrs["shape"], attrs["dims"]
        _check(len(dims) == x.rank and all(x.shape[i] in (1, shape[d]) for i, d in enumerate(dims)), f"invalid broadcast: {x} -> {shape} {dims}")
        return [TileType(shape, x.dtype)]
    if kind == "reshape":
        (x,) = inputs
        n = 1
        for d in x.shape:
            n *= d
        m = 1
        for d in attrs["shape"]:
            m *= d
        _check(n == m, f"reshape changes the element count: {x} -> {attrs['shape']}")
        return [TileType(attrs["shape"], x.dtype)]
    if kind == "permute":
        (x,) = inputs
        _check(sorted(attrs["perm"]) == list(range(x.rank)), f"invalid permutation {attrs['perm']} / {x}")
        return [TileType(tuple(x.shape[p] for p in attrs["perm"]), x.dtype)]
    if kind == "reduce":
        (x,) = inputs
        axes = attrs["axes"]
        _check(attrs["fn"] in ("sum", "max") and all(0 <= a < x.rank for a in axes), f"invalid reduce: {x}, {attrs}")
        return [TileType(tuple(d for i, d in enumerate(x.shape) if i not in axes), x.dtype)]
    if kind == "dot":
        # The matrix instruction abstraction: acc[M,N] + a[M,K] @ b[K,N]. Which instruction (mma.sync,
        # wgmma, MFMA, FMA) implements it is decided by lowering from the capability descriptor.
        a, b, acc = inputs
        _check(a.rank == b.rank == acc.rank == 2 and a.shape[1] == b.shape[0] and acc.shape == (a.shape[0], b.shape[1]), f"invalid dot: {a}, {b}, {acc}")
        _check(a.dtype == b.dtype and acc.dtype.is_float, f"invalid dot dtypes: {a}, {b}, {acc}")
        return [acc]
    if kind in ("load", "load_indexed"):
        # load(starts...) / load_indexed(index, starts...): one start per tile axis.
        index, starts = (inputs[0], inputs[1:]) if kind == "load_indexed" else (None, inputs)
        shape = attrs["shape"]
        _check(len(starts) == len(shape) and all(s == TileType((), INDEX) for s in starts), f"{kind} needs one i32 start per tile axis")
        if index is not None:
            _check(index.shape == shape and not index.dtype.is_float, f"load_indexed index must be an integer tile of the tile shape: {index}")
        return [TileType(shape, attrs["dtype"])]
    if kind in ("store", "atomic_add"):
        # atomic_add: like store, but adds to memory atomically (several instances may add to
        # one element, in any order).
        value, *starts = inputs
        _check(len(starts) == value.rank and all(s == TileType((), INDEX) for s in starts), f"{kind} needs one i32 start per tile axis")
        _check(kind == "store" or value.dtype in (DType.f32, DType.f64), f"atomic_add of {value.dtype} is not supported")
        return []
    if kind == "atomic_add_indexed":
        index, value, *starts = inputs
        _check(index.shape == value.shape and len(starts) == value.rank, f"invalid atomic_add_indexed: {index}, {value}")
        return []
    if kind == "for":
        # for(lo, hi, step, carries...): region params are (iv, carries...), results the next carries.
        lo, hi, step, *carries = inputs
        _check(lo == hi == step == TileType((), INDEX), "for bounds must be i32 scalars")
        (region,) = regions
        _check([p.type for p in region.params] == [TileType((), INDEX), *carries], "for region params do not match")
        _check([r.type for r in region.results] == list(carries), "for region changes carry types")
        return list(carries)
    raise ValueError(f"unknown tile op: {kind}")


_FOLD = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


class TileBuilder:
    def __init__(self, *, cse: bool = False) -> None:
        """cse: reuse an earlier equal scalar op (folding index arithmetic on constants) and an
        earlier equal load with no store emitted since. A pass that replays several tasks into
        one body (row fusion) otherwise loads the same tile once per replayed task, and each
        load goes to DRAM when the data does not stay in L2. Tile-shaped ops other than loads
        are not merged: two loop carries starting from one shared value would have to agree on
        a register layout in lowering."""
        self.ops: list[TOp] = []
        self.cse = cse
        self._seen: dict[tuple, tuple[TValue, ...]] = {}
        self._constants: dict[TValue, int] = {}

    def emit(self, kind: str, inputs: Sequence[TValue] = (), attrs: dict | None = None, regions: Sequence[Body] = ()) -> tuple[TValue, ...]:
        attrs = dict(attrs or {})
        types = infer(kind, [v.type for v in inputs], attrs, regions)
        key = None
        if self.cse and not regions:
            if kind in ("store", "atomic_add", "atomic_add_indexed"):
                self.forget_loads()
            elif kind in ("load", "load_indexed") or (len(types) == 1 and types[0].shape == ()):
                if kind in _FOLD and types[0] == TileType((), INDEX) and all(v in self._constants for v in inputs):
                    return (self.index(_FOLD[kind](*(self._constants[v] for v in inputs))),)
                key = (kind, tuple(inputs), make_attrs(attrs))
                if key in self._seen:
                    return self._seen[key]
        outputs = tuple(TValue(t) for t in types)
        self.ops.append(TOp(kind, tuple(inputs), outputs, make_attrs(attrs), tuple(regions)))
        if key is not None:
            self._seen[key] = outputs
            if kind == "full" and types[0] == TileType((), INDEX) and isinstance(attrs["value"], int):
                self._constants[outputs[0]] = attrs["value"]
        return outputs

    def forget_loads(self) -> None:
        """For ops appended to self.ops directly: after a store, no earlier load is reused."""
        self._seen = {k: v for k, v in self._seen.items() if k[0] not in ("load", "load_indexed")}

    def op(self, kind: str, *inputs: TValue, **attrs) -> TValue:
        return self.emit(kind, inputs, attrs)[0]

    def index(self, value: int) -> TValue:
        return self.op("full", shape=(), value=value, dtype=INDEX)

    def dim(self, d: Dim) -> TValue:
        return self.index(d) if isinstance(d, int) else self.op("dim", value=d)

    def full(self, shape: tuple[int, ...], value, dtype: DType) -> TValue:
        return self.op("full", shape=tuple(shape), value=value, dtype=dtype)

    def for_range(self, lo: TValue, hi: TValue, step: int, carries: Sequence[TValue], fn, **attrs) -> list[TValue]:
        """fn(builder, iv, carries) -> next carries. Builds the region in a fresh builder.

        attrs: `stages`, the software pipeline depth for the loads in the body (1 = none);
        `stream`, double-buffer the dot operands the body loads at the loop variable (copy
        iteration i+1's while computing iteration i), at the cost of twice their shared memory.
        """
        inner = TileBuilder(cse=self.cse)
        iv = TValue(TileType((), INDEX))
        params = [TValue(c.type) for c in carries]
        results = fn(inner, iv, params)
        region = Body((iv, *params), tuple(inner.ops), tuple(results))
        return list(self.emit("for", (lo, hi, self.index(step), *carries), attrs, (region,)))

    def finish(self, params: Sequence[TValue], results: Sequence[TValue] = ()) -> Body:
        return Body(tuple(params), tuple(self.ops), tuple(results))
