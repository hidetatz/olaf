"""Conversion from Graph IR to Tile IR: one task per compute op, views folded into addressing.

The input must be decomposed (no composite ops) and have no scan (passes.unroll). Every compute
op becomes its own task writing a fresh buffer; reshape, permute, broadcast, slice and pad become
strided Views of an existing buffer and cost nothing. Fusion is a separate Tile IR pass, so this
conversion stays a direct, testable translation and the fusion rules can be checked against it.

Tile sizes here are defaults, not tuned choices: the schedule search replaces them per task.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from olaf.backend.capability import CPU, Capability
from olaf.graph.dim import Dim, product
from olaf.graph.ir import Graph, Op, Value
from olaf.graph.ops import OPS, parse_contract
from olaf.graph.types import DType, TensorType
from olaf.tile.builder import BINARY, INDEX, UNARY, TileBuilder
from olaf.tile.ir import Arg, Buffer, BufferKind, Launch, Program, Task, TileType, TValue, View, contiguous_strides, full_view

ELEMENTWISE_BLOCK = 128
REDUCE_BLOCK = 128
DOT_BLOCK = (32, 32, 32)  # M, N, K, without matrix instructions
# With a 16x8x16 matrix instruction: 128x128 output tiles over 8 warps (64x32 per warp) and a
# 3-stage async copy pipeline over 32-wide K steps. 3 stages of bf16 A and B tiles take 48 KiB of
# shared memory, which leaves room on Ampere (100 KiB per SM) for two blocks per SM.
MATRIX_DOT_BLOCK = (128, 128, 32)
MATRIX_DOT_WARPS = 8
MATRIX_DOT_STAGES = 3


@dataclass(frozen=True)
class Ref:
    buffer: str
    view: View
    dtype: DType

    @property
    def shape(self) -> tuple[Dim, ...]:
        return self.view.shape


def _block(size: Dim, limit: int) -> int:
    if isinstance(size, int):
        return min(limit, 1 << max(size - 1, 0).bit_length())
    return limit


def _ceil_div(d: Dim, t: int) -> Dim:
    if t == 1:
        return d
    if isinstance(d, int):
        return -(-d // t)
    return (d + (t - 1)) // t


def accumulator_dtype(dtype: DType) -> DType:
    return DType.f64 if dtype is DType.f64 else DType.f32 if dtype.is_float else dtype


def _neutral(fn: str, dtype: DType) -> float | int:
    if fn == "sum":
        return 0
    if dtype.is_float:
        return float("-inf")
    return int(np.iinfo(dtype.numpy).min)


def _shifted(source, axis: int, delta: Dim):
    """Source map after view axis `axis` is shifted so that new index = old index - delta."""
    if source is None:
        return None
    return tuple((a, s + delta) if a == axis else (a, s) for a, s in source)


def _placed(ref: Ref, shape, dims):
    """Source map after view axis i moves to dims[i] in `shape` (broadcast semantics)."""
    if ref.view.source is None:
        return None
    moved = {i: d for i, d in enumerate(dims) if ref.shape[i] == shape[d]}
    return tuple((moved.get(a) if a is not None else None, s) for a, s in ref.view.source)


def to_tile(graph: Graph, capability: Capability = CPU) -> Program:
    """Default tile shapes come from the capability: matrix instructions change the dot tiles."""
    return _Converter(graph, capability).run()


class _Converter:
    def __init__(self, graph: Graph, capability: Capability) -> None:
        self.graph = graph
        self.capability = capability
        self.buffers: dict[str, Buffer] = {}
        self.launches: list[Launch] = []
        self.refs: dict[Value, Ref] = {}
        self.states: dict[str, Ref] = {}

    # -- buffers and tasks ----------------------------------------------------------------------

    def new_buffer(self, type: TensorType, kind: BufferKind, **fields) -> Ref:
        name = f"b{len(self.buffers)}"
        self.buffers[name] = Buffer(name, TensorType(type.shape, type.dtype), kind, **fields)
        return Ref(name, full_view(type.shape), type.dtype)

    def launch(self, name: str, grid: Sequence[Dim], refs: Sequence[tuple[Ref, bool]], body, warps: int = 4) -> None:
        args = tuple(Arg(r.view, r.dtype, writes) for r, writes in refs)
        self.launches.append(Launch(Task(name, tuple(grid), args, body, warps), tuple(r.buffer for r, _ in refs)))

    def elementwise(self, name: str, out_type: TensorType, inputs: Sequence[Ref], compute: Callable, out: Ref | None = None, kind: BufferKind = BufferKind.temp) -> Ref:
        """One task over the output, one tile per instance. compute(b, tiles, starts, tile_shape) -> tile."""
        out = out or self.new_buffer(out_type, kind)
        shape = out.shape
        tile = tuple(1 for _ in shape[:-1]) + ((_block(shape[-1], ELEMENTWISE_BLOCK),) if shape else ())
        grid = tuple(_ceil_div(d, t) for d, t in zip(shape, tile))
        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in shape]
        starts = [p if t == 1 else b.op("mul", p, b.index(t)) for p, t in zip(pids, tile)]
        tiles = [b.op("load", *starts, arg=i, shape=tile, dtype=r.dtype, other=0) for i, r in enumerate(inputs)]
        result = compute(b, tiles, starts, tile)
        b.emit("store", (result, *starts), {"arg": len(inputs)})
        self.launch(name, grid, [*((r, False) for r in inputs), (out, True)], b.finish(pids))
        return out

    def materialize(self, ref: Ref, kind: BufferKind = BufferKind.temp, **fields) -> Ref:
        out = self.new_buffer(TensorType(ref.shape, ref.dtype), kind, **fields)
        return self.elementwise("copy", TensorType(ref.shape, ref.dtype), [ref], lambda b, t, s, ts: t[0], out=out)

    def plain(self, ref: Ref) -> Ref:
        """A ref without padding, for view ops that do not compose with pad."""
        return self.materialize(ref) if ref.view.pad is not None else ref

    # -- driver ---------------------------------------------------------------------------------

    def run(self) -> Program:
        inputs = []
        for p in self.graph.params:
            self.refs[p] = self.new_buffer(p.type, BufferKind.input)
            inputs.append(self.refs[p].buffer)
        for op in self.graph.ops:
            if OPS[op.kind].composite:
                raise ValueError(f"composite op {op.kind} must be decomposed before tile conversion")
            pointwise = op.kind in UNARY or op.kind in BINARY or op.kind in ("cast", "cmp", "where")
            handler = self._pointwise if pointwise else getattr(self, f"_op_{op.kind}", None)
            if handler is None:
                raise NotImplementedError(f"tile conversion of {op.kind} is not implemented")
            result = handler(op, [self.refs[v] for v in op.inputs])
            if op.outputs:
                self.refs[op.outputs[0]] = result
        outputs = [self.output(self.refs[r]) for r in self.graph.results]
        return Program(tuple(self.buffers.values()), tuple(self.launches), tuple(inputs), tuple(outputs))

    def output(self, ref: Ref) -> str:
        buffer = self.buffers[ref.buffer]
        whole = ref.view.is_contiguous and ref.view.offset == 0 and ref.shape == buffer.type.shape
        if whole and buffer.kind is BufferKind.temp:
            self.buffers[ref.buffer] = replace(buffer, kind=BufferKind.output)
            return ref.buffer
        return self.materialize(ref, BufferKind.output).buffer

    # -- elementwise ----------------------------------------------------------------------------

    def _pointwise(self, op: Op, inputs: list[Ref]) -> Ref:
        attrs = {k: v for k, v in op.attrs}
        return self.elementwise(op.kind, op.outputs[0].type, inputs, lambda b, t, s, ts: b.op(op.kind, *t, **attrs))

    def _op_iota(self, op: Op, inputs: list[Ref]) -> Ref:
        axis, dtype = op.attr("axis"), op.attr("dtype")

        def compute(b, tiles, starts, tile):
            value = b.op("iota", starts[axis], shape=tile, axis=axis)
            return value if dtype is INDEX else b.op("cast", value, dtype=dtype)

        return self.elementwise("iota", op.outputs[0].type, [], compute)

    def _op_constant(self, op: Op, inputs: list[Ref]) -> Ref:
        data = op.attr("value")
        return self.new_buffer(TensorType(data.shape, data.dtype), BufferKind.constant, constant=data)

    def _op_concat(self, op: Op, inputs: list[Ref]) -> Ref:
        axis = op.attr("axis")
        total = op.outputs[0].type.shape[axis]
        padded, ends = [], []
        start: Dim = 0
        for ref in inputs:
            ref = self.plain(ref)
            stop = start + ref.shape[axis]
            pad = tuple((start, total - stop) if i == axis else (0, 0) for i in range(ref.view.rank))
            shape = tuple(total if i == axis else d for i, d in enumerate(ref.shape))
            padded.append(replace(ref, view=replace(ref.view, shape=shape, pad=pad, source=_shifted(ref.view.source, axis, -start))))
            ends.append(stop)
            start = stop

        def compute(b, tiles, starts, tile):
            coord = b.op("iota", starts[axis], shape=tile, axis=axis)
            result = tiles[-1]
            for x, end in zip(reversed(tiles[:-1]), reversed(ends[:-1])):
                bound = b.op("broadcast", b.dim(end), shape=tile, dims=())
                result = b.op("where", b.op("cmp", coord, bound, fn="lt"), x, result)
            return result

        return self.elementwise("concat", op.outputs[0].type, padded, compute)

    # -- views ----------------------------------------------------------------------------------

    def _op_reshape(self, op: Op, inputs: list[Ref]) -> Ref:
        (ref,) = inputs
        shape = op.attr("shape")
        if shape == ref.shape:
            return ref
        if not ref.view.is_contiguous:
            ref = self.materialize(ref)
        return replace(ref, view=View(shape, contiguous_strides(shape), ref.view.offset))

    def _op_permute(self, op: Op, inputs: list[Ref]) -> Ref:
        (ref,) = inputs
        perm, v = op.attr("perm"), ref.view
        pad = tuple(v.pad[p] for p in perm) if v.pad is not None else None
        source = tuple((None if a is None else perm.index(a), s) for a, s in v.source) if v.source is not None else None
        return replace(ref, view=View(tuple(v.shape[p] for p in perm), tuple(v.strides[p] for p in perm), v.offset, pad, v.fill, source))

    def _op_broadcast(self, op: Op, inputs: list[Ref]) -> Ref:
        ref = self.plain(inputs[0])
        shape, dims = op.attr("shape"), op.attr("dims")
        strides: list[Dim] = [0] * len(shape)
        for i, d in enumerate(dims):
            strides[d] = ref.view.strides[i] if ref.shape[i] == shape[d] else 0
        return replace(ref, view=View(tuple(shape), tuple(strides), ref.view.offset, source=_placed(ref, shape, dims)))

    def _op_slice(self, op: Op, inputs: list[Ref]) -> Ref:
        ref = self.plain(inputs[0])
        starts, stops, v = op.attr("starts"), op.attr("stops"), ref.view
        offset = v.offset + sum((s * st for s, st in zip(starts, v.strides)), 0)
        source = v.source
        for axis, start in enumerate(starts):
            source = _shifted(source, axis, start)
        return replace(ref, view=View(tuple(e - s for s, e in zip(starts, stops)), v.strides, offset, source=source))

    def _op_pad(self, op: Op, inputs: list[Ref]) -> Ref:
        ref = self.plain(inputs[0])
        low, high, v = op.attr("low"), op.attr("high"), ref.view
        shape = tuple(lo + d + hi for lo, d, hi in zip(low, v.shape, high))
        source = v.source
        for axis, lo in enumerate(low):
            source = _shifted(source, axis, -lo)
        return replace(ref, view=View(shape, v.strides, v.offset, tuple(zip(low, high)), float(op.attr("value")), source))

    # -- reductions -----------------------------------------------------------------------------

    def _op_reduce(self, op: Op, inputs: list[Ref]) -> Ref:
        (ref,) = inputs
        fn, axes = op.attr("fn"), op.attr("axes")
        out_type = op.outputs[0].type
        out = self.new_buffer(out_type, BufferKind.temp)
        shape, rank = ref.shape, ref.view.rank
        kept = [a for a in range(rank) if a not in axes]
        inner_kept = rank - 1 if rank - 1 in kept else None
        inner_reduced = max(axes)
        extent = {a: 1 for a in range(rank)}
        if inner_kept is not None:
            extent[inner_kept] = _block(shape[inner_kept], ELEMENTWISE_BLOCK)
        extent[inner_reduced] = _block(shape[inner_reduced], REDUCE_BLOCK)
        out_tile = tuple(extent[a] for a in kept)
        grid = tuple(_ceil_div(shape[a], extent[a]) for a in kept)
        init = _neutral(fn, ref.dtype)
        combine = "add" if fn == "sum" else "max"
        # A bf16 sum accumulates in f32 and rounds once at the store ("exact, then rounded"):
        # adding thousands of bf16 values in bf16 loses everything past the 8th bit. A max is exact in any dtype.
        acc_dtype = accumulator_dtype(ref.dtype) if fn == "sum" else ref.dtype

        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in kept]
        out_starts = [p if extent[a] == 1 else b.op("mul", p, b.index(extent[a])) for p, a in zip(pids, kept)]
        acc = b.full(out_tile, init, acc_dtype)
        reduced = sorted(axes)

        def loop(builder: TileBuilder, depth: int, starts: dict[int, TValue], carry: TValue) -> TValue:
            if depth == len(reduced):
                full_starts = [starts[a] if a in starts else out_starts[kept.index(a)] for a in range(rank)]
                tile = builder.op("load", *full_starts, arg=0, shape=tuple(extent[a] for a in range(rank)), dtype=ref.dtype, other=init)
                if acc_dtype is not ref.dtype:
                    tile = builder.op("cast", tile, dtype=acc_dtype)
                part = builder.op("reduce", tile, fn=fn, axes=tuple(reduced))
                return builder.op(combine, carry, part)
            axis = reduced[depth]

            def body(inner: TileBuilder, iv: TValue, carries: list[TValue]) -> list[TValue]:
                return [loop(inner, depth + 1, {**starts, axis: iv}, carries[0])]

            return builder.for_range(builder.index(0), builder.dim(shape[axis]), extent[axis], [carry], body)[0]

        result = loop(b, 0, {}, acc)
        if acc_dtype is not ref.dtype:
            result = b.op("cast", result, dtype=ref.dtype)
        b.emit("store", (result, *out_starts), {"arg": 1})
        self.launch(f"reduce_{fn}", grid, [(ref, False), (out, True)], b.finish(pids))
        return out

    def _op_contract(self, op: Op, inputs: list[Ref]) -> Ref:
        terms, out_labels = parse_contract(op.attr("spec"))
        if len(terms) != 2:
            raise NotImplementedError(f"tile conversion of a {len(terms)}-operand contract is not implemented")
        (ta, tb), (ra, rb) = terms, inputs
        m_labels = [c for c in out_labels if c in ta and c not in tb]
        n_labels = [c for c in out_labels if c in tb and c not in ta]
        k_labels = [c for c in ta if c in tb and c not in out_labels]
        if any(c not in out_labels and c not in k_labels for c in ta + tb):
            raise NotImplementedError(f"contract labels summed in one operand only: {op.attr('spec')}")
        size = {**dict(zip(ta, ra.shape)), **dict(zip(tb, rb.shape))}
        m_inner = m_labels[-1] if m_labels else None
        n_inner = n_labels[-1] if n_labels else None
        k_inner = [c for c in ta if c in k_labels][-1] if k_labels else None
        matrix = any(i.m == 16 and i.n == 8 and i.k == 16 and i.dtype_a == ra.dtype.value and i.dtype_acc == "f32" for i in self.capability.matrix_instructions)
        bm, bn, bk = MATRIX_DOT_BLOCK if matrix else DOT_BLOCK
        stages, warps = (MATRIX_DOT_STAGES, MATRIX_DOT_WARPS) if matrix else (1, 4)
        extent = {c: 1 for c in size}
        if m_inner:
            extent[m_inner] = _block(size[m_inner], bm)
        if n_inner:
            extent[n_inner] = _block(size[n_inner], bn)
        if k_inner:
            extent[k_inner] = _block(size[k_inner], bk)
        em, en, ek = (extent[c] if c else 1 for c in (m_inner, n_inner, k_inner))
        out_type = op.outputs[0].type
        out = self.new_buffer(out_type, BufferKind.temp)
        acc_dtype = accumulator_dtype(ra.dtype)
        grid = tuple(_ceil_div(size[c], extent[c]) for c in out_labels)

        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in out_labels]
        out_starts = {c: p if extent[c] == 1 else b.op("mul", p, b.index(extent[c])) for c, p in zip(out_labels, pids)}

        def operand(builder: TileBuilder, arg: int, term: str, ref: Ref, starts: dict[str, TValue], rows: str | None, cols: str | None) -> TValue:
            tile = builder.op("load", *(starts[c] for c in term), arg=arg, shape=tuple(extent[c] for c in term), dtype=ref.dtype, other=0)
            # Move the two tiled labels last (rows, then cols) and drop the unit axes around them.
            order = [i for i, c in enumerate(term) if c not in (rows, cols)] + [term.index(c) for c in (rows, cols) if c]
            if order != list(range(len(term))):
                tile = builder.op("permute", tile, perm=tuple(order))
            return builder.op("reshape", tile, shape=(extent[rows] if rows else 1, extent[cols] if cols else 1))

        def step(builder: TileBuilder, starts: dict[str, TValue], acc: TValue) -> TValue:
            a = operand(builder, 0, ta, ra, starts, m_inner, k_inner)
            bt = operand(builder, 1, tb, rb, starts, k_inner, n_inner)
            return builder.op("dot", a, bt, acc)

        def loop(builder: TileBuilder, depth: int, starts: dict[str, TValue], acc: TValue) -> TValue:
            if depth == len(k_labels):
                return step(builder, starts, acc)
            label = k_labels[depth]

            def body(inner: TileBuilder, iv: TValue, carries: list[TValue]) -> list[TValue]:
                return [loop(inner, depth + 1, {**starts, label: iv}, carries[0])]

            pipelined = stages if label == k_inner else 1
            return builder.for_range(builder.index(0), builder.dim(size[label]), extent[label], [acc], body, stages=pipelined)[0]

        acc = loop(b, 0, dict(out_starts), b.full((em, en), 0, acc_dtype))
        result = acc if acc_dtype is out_type.dtype else b.op("cast", acc, dtype=out_type.dtype)
        tiled = [c for c in out_labels if c in (m_inner, n_inner)]
        if tiled == [n_inner, m_inner] and m_inner and n_inner:
            result = b.op("permute", result, perm=(1, 0))
        result = b.op("reshape", result, shape=tuple(extent[c] for c in out_labels))
        b.emit("store", (result, *(out_starts[c] for c in out_labels)), {"arg": 2})
        self.launch("contract", grid, [(ra, False), (rb, False), (out, True)], b.finish(pids), warps)
        return out

    # -- indexing -------------------------------------------------------------------------------

    def _index_over(self, index: Ref, shape: tuple[Dim, ...], axis: int, batch_dims: int) -> tuple[Ref, int]:
        """View the index tensor over the gather output (or scatter update) shape."""
        index = self.plain(index)
        n_index = index.view.rank - batch_dims
        dims = tuple(i if i < batch_dims else axis + i - batch_dims for i in range(index.view.rank))
        strides: list[Dim] = [0] * len(shape)
        for i, d in enumerate(dims):
            strides[d] = index.view.strides[i]
        return replace(index, view=View(shape, tuple(strides), index.view.offset, source=_placed(index, shape, dims))), n_index

    @staticmethod
    def _axis_map(rank: int, axis: int, n_index: int) -> tuple[int, ...]:
        # For each axis of the indexed tensor: the tile axis giving its coordinate, or -1 for the indexed axis.
        return tuple(i if i < axis else -1 if i == axis else i - 1 + n_index for i in range(rank))

    def _op_gather(self, op: Op, inputs: list[Ref]) -> Ref:
        x, index = self.plain(inputs[0]), inputs[1]
        axis, batch_dims = op.attr("axis"), op.attr("batch_dims")
        out_type = op.outputs[0].type
        spread, n_index = self._index_over(index, out_type.shape, axis, batch_dims)
        axis_map = self._axis_map(x.view.rank, axis, n_index)
        out = self.new_buffer(out_type, BufferKind.temp)
        shape = out_type.shape
        tile = tuple(1 for _ in shape[:-1]) + ((_block(shape[-1], ELEMENTWISE_BLOCK),) if shape else ())
        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in shape]
        starts = [p if t == 1 else b.op("mul", p, b.index(t)) for p, t in zip(pids, tile)]
        idx = b.op("load", *starts, arg=1, shape=tile, dtype=spread.dtype, other=0)
        value = b.op("load_indexed", idx, *starts, arg=0, shape=tile, dtype=x.dtype, axis_map=axis_map, other=0)
        b.emit("store", (value, *starts), {"arg": 2})
        grid = tuple(_ceil_div(d, t) for d, t in zip(shape, tile))
        self.launch("gather", grid, [(x, False), (spread, False), (out, True)], b.finish(pids))
        return out

    def _op_scatter_add(self, op: Op, inputs: list[Ref]) -> Ref:
        target, index, updates = inputs
        axis, batch_dims = op.attr("axis"), op.attr("batch_dims")
        out = self.materialize(target)
        spread, n_index = self._index_over(index, updates.shape, axis, batch_dims)
        axis_map = self._axis_map(out.view.rank, axis, n_index)
        shape = updates.shape
        tile = tuple(1 for _ in shape[:-1]) + ((_block(shape[-1], ELEMENTWISE_BLOCK),) if shape else ())
        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in shape]
        starts = [p if t == 1 else b.op("mul", p, b.index(t)) for p, t in zip(pids, tile)]
        idx = b.op("load", *starts, arg=0, shape=tile, dtype=spread.dtype, other=0)
        value = b.op("load", *starts, arg=1, shape=tile, dtype=updates.dtype, other=0)
        # Lanes past the update tensor must not add: their index reads as 0, a valid position.
        b.emit("atomic_add_indexed", (idx, value, *starts), {"arg": 2, "axis_map": axis_map, "bounds": tuple(shape)})
        grid = tuple(_ceil_div(d, t) for d, t in zip(shape, tile))
        self.launch("scatter_add", grid, [(spread, False), (updates, False), (out, True)], b.finish(pids))
        return out

    # -- state ----------------------------------------------------------------------------------

    def _op_read_state(self, op: Op, inputs: list[Ref]) -> Ref:
        decl = op.attr("state")
        if decl.name not in self.states:
            self.states[decl.name] = self.new_buffer(decl.type, BufferKind.state_in, state=decl)
        return self.states[decl.name]

    def _op_write_state(self, op: Op, inputs: list[Ref]) -> None:
        # A separate output buffer, not an in-place write: later launches may still read the step's
        # initial value (read_state semantics). Donating the input buffer is memory planning (P2).
        self.materialize(inputs[0], BufferKind.state_out, state=op.attr("state"))
