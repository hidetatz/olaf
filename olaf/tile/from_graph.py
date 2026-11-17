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
ELEMENTWISE_ELEMENTS = 1024
# Reduced chunk per loop step: wide for row reductions (one value per output, 8 registers per
# thread at 128 threads), narrow when an inner axis is kept, where the accumulator holds
# chunk x 128 values (32 per thread).
ROW_REDUCE_BLOCK = 1024
# Rows up to this long are reduced in one loop step (with more warps, at most 8 elements per
# thread), as tile/rows.py fuses rows up to tile.rows.ROW_LIMIT only when one step covers them.
WHOLE_ROW_LIMIT = 8192
# A split reduction aims at up to this many blocks per compute unit, and a column reduction
# reads this many rows per loop step. Measured on the RTX 3060 by graph replay, column sums of
# 4096 x {768, 2304, 3072} bf16 and 4096 x 768 f32: 4 blocks and 32 rows gave 224 / 149 / 190
# and 303 GB/s; 8 blocks and 16 rows give 255 / 309 / 318 and 311 GB/s. 32 rows is the part
# that hurt, though not monotonically in width (2304 columns was the slowest).
SPLIT_BLOCKS_PER_UNIT = 8
COLUMN_REDUCE_BLOCK = 16
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


def _elementwise_tile(shape) -> tuple[int, ...]:
    """Tile of an elementwise task: up to ELEMENTWISE_BLOCK along the last axis and, when that
    is narrow, rows of the axis before it, up to ELEMENTWISE_ELEMENTS in all.

    One 128-element row per instance (one element per thread) left GPT-2's elementwise passes
    far from memory bandwidth: 24576 tiny blocks for a 4 x 1024 x 768 tensor, too little in
    flight per thread. Rows of the previous axis add work per instance without padding a
    768-wide axis to 1024.
    """
    if not shape:
        return ()
    if isinstance(shape[-1], int) and shape[-1] >= ELEMENTWISE_ELEMENTS:
        # A long row (the 50257-wide logits) is tiled along itself, so the elements a thread
        # owns share their row: a broadcast row statistic is one load for all of them.
        return tuple(1 for _ in shape[:-1]) + (ELEMENTWISE_ELEMENTS,)
    last = _block(shape[-1], ELEMENTWISE_BLOCK)
    rows = 1
    if len(shape) >= 2:
        rows = _block(shape[-2], max(1, ELEMENTWISE_ELEMENTS // last))
    return tuple(1 for _ in shape[:-2]) + ((rows,) if len(shape) >= 2 else ()) + (last,)


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


def _unit_reshape(view: View, shape) -> View | None:
    """`view` reshaped by only inserting or removing size-1 axes, keeping its source map.

    A general reshape view has no source map (axes mix), which hides it from fusion; this kind
    maps axis to axis. RMSNorm written keepdim-style computes its statistic as [batch, seq, 1]
    and broadcasts it along the row: row fusion needs to see that it is a row statistic."""
    old = [i for i, d in enumerate(view.shape) if d != 1]
    new = [j for j, d in enumerate(shape) if d != 1]
    if len(old) != len(new) or any(view.shape[i] != shape[j] for i, j in zip(old, new)):
        return None
    if view.pad is not None and any(view.pad[i] != (0, 0) for i in range(len(view.shape)) if i not in old):
        return None
    to_new = dict(zip(old, new))
    strides = [0] * len(shape)
    pad = [(0, 0)] * len(shape) if view.pad is not None else None
    for i, j in to_new.items():
        strides[j] = view.strides[i]
        if pad is not None:
            pad[j] = view.pad[i]
    source = None
    if view.source is not None:
        source = tuple((None, start) if axis is None or axis not in to_new else (to_new[axis], start) for axis, start in view.source)
    return View(tuple(shape), tuple(strides), view.offset, tuple(pad) if pad is not None else None, view.fill, source)


def _reshaped_strides(view: View, shape) -> tuple[int, ...] | None:
    """Strides that show `view`'s elements in `shape` without copying, or None (as torch.Tensor.view).

    Possible when every group of old axes that merges into, or splits into, new axes is itself
    contiguous: the head split of a q/k/v slice of the fused qkv projection is (only the last
    axis, stride 1, splits), and copying it would cost a pass over q, k and v per layer.
    Static shapes only; the result has no source map (one buffer axis may become several).
    """
    old_shape, old_strides = view.shape, view.strides
    if view.pad is not None or not all(isinstance(d, int) for d in (*old_shape, *old_strides, *shape)):
        return None
    old = [(d, st) for d, st in zip(old_shape, old_strides) if d != 1]
    new = list(shape)
    out: list[int] = []
    i = 0
    j = 0
    while j < len(new):
        if new[j] == 1:
            out.append(0)
            j += 1
            continue
        if i >= len(old):
            return None
        # Take old axes until their product covers the next run of new axes.
        group = [old[i]]
        size = old[i][0]
        i += 1
        run = [j]
        total = new[j]
        j += 1
        while size != total:
            if size < total:
                if i >= len(old):
                    return None
                group.append(old[i])
                size *= old[i][0]
                i += 1
            else:
                while j < len(new) and new[j] == 1:
                    run.append(j)
                    j += 1
                if j >= len(new):
                    return None
                run.append(j)
                total *= new[j]
                j += 1
        for (d0, s0), (d1, s1) in zip(group, group[1:]):
            if s0 != d1 * s1:
                return None  # the merged old axes are not contiguous with each other
        stride = group[-1][1]
        run_strides = {}
        for k in reversed(run):
            run_strides[k] = stride if new[k] != 1 else 0
            stride *= new[k]
        out.extend(run_strides[k] for k in run)
    if i != len(old):
        return None
    return tuple(out)


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


def to_tile(graph: Graph, capability: Capability = CPU, schedules=None) -> Program:
    """Default tile shapes come from the capability: matrix instructions change the dot tiles.

    schedules maps search.schedule.dot_key -> DotSchedule (found by search.tune); dots without
    an entry use the defaults below.
    """
    return _Converter(graph, capability, schedules or {}).run()


class _Converter:
    def __init__(self, graph: Graph, capability: Capability, schedules=None) -> None:
        self.graph = graph
        self.capability = capability
        self.schedules = schedules or {}
        self.buffers: dict[str, Buffer] = {}
        self.launches: list[Launch] = []
        self.refs: dict[Value, Ref] = {}
        self.states: dict[str, Ref] = {}

    # -- buffers and tasks ----------------------------------------------------------------------

    def new_buffer(self, type: TensorType, kind: BufferKind, **fields) -> Ref:
        name = f"b{len(self.buffers)}"
        self.buffers[name] = Buffer(name, TensorType(type.shape, type.dtype), kind, **fields)
        return Ref(name, full_view(type.shape), type.dtype)

    def launch(self, name: str, grid: Sequence[Dim], refs: Sequence[tuple[Ref, bool]], body, warps: int = 4, origin: tuple = ()) -> None:
        args = tuple(Arg(r.view, r.dtype, writes) for r, writes in refs)
        task = Task(name, tuple(grid), args, body, warps, origin or (name,))
        self.launches.append(Launch(task, tuple(r.buffer for r, _ in refs)))

    def elementwise(self, name: str, out_type: TensorType, inputs: Sequence[Ref], compute: Callable, out: Ref | None = None, kind: BufferKind = BufferKind.temp) -> Ref:
        """One task over the output, one tile per instance. compute(b, tiles, starts, tile_shape) -> tile."""
        out = out or self.new_buffer(out_type, kind)
        shape = out.shape
        tile = _elementwise_tile(shape)
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
        self.producers = {v: op for op in self.graph.ops for v in op.outputs}
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
        unit = _unit_reshape(ref.view, tuple(shape))
        if unit is not None:
            return replace(ref, view=unit)
        if not ref.view.is_contiguous:
            strides = _reshaped_strides(ref.view, shape)
            if strides is not None:
                return replace(ref, view=View(tuple(shape), strides, ref.view.offset))
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
        split = self._split(ref, axes)
        if split is not None:
            # Two passes: P slices of one reduced axis in parallel, then the P partials.
            axis, parts = split
            size = ref.shape[axis]
            v = ref.view
            shape = (*v.shape[:axis], parts, size // parts, *v.shape[axis + 1 :])
            strides = (*v.strides[:axis], v.strides[axis] * (size // parts), v.strides[axis], *v.strides[axis + 1 :])
            pad = None if v.pad is None else (*v.pad[:axis], (0, 0), (0, 0), *v.pad[axis + 1 :])
            sliced = replace(ref, view=View(shape, strides, v.offset, pad, v.fill))
            inner_axes = tuple(a if a < axis else a + 1 for a in axes)
            kept_shape = tuple(d for i, d in enumerate(shape) if i not in inner_axes)
            # Partial sums stay in the accumulator dtype: rounding them to bf16 and again after
            # combining them would round twice where the unsplit sum rounds once.
            partial_dtype = accumulator_dtype(ref.dtype) if fn == "sum" else ref.dtype
            partial = self._reduce(sliced, fn, inner_axes, TensorType(kept_shape, partial_dtype))
            position = sum(1 for a in range(axis) if a not in axes)
            return self._reduce(partial, fn, (position,), op.outputs[0].type)
        return self._reduce(ref, fn, axes, op.outputs[0].type)

    def _split(self, ref: Ref, axes) -> tuple[int, int] | None:
        """(axis, parts) when a reduction would leave most of the device idle, else None.

        A column reduction (bias gradients: 4096 rows into 768 columns) runs as 6 blocks of 128
        columns, each walking all rows, on a 28-SM GPU. Splitting the longest reduced axis into
        parts that are reduced in parallel and then summed restores the parallelism.
        """
        units = self.capability.roofline.compute_units
        shape = ref.shape
        if units <= 1 or ref.view.pad is not None or not all(isinstance(d, int) for d in shape):
            return None
        rank = len(shape)
        kept = [a for a in range(rank) if a not in axes]
        inner = rank - 1 if rank - 1 in kept else None
        blocks = 1
        for a in kept:
            blocks *= -(-shape[a] // (_block(shape[a], ELEMENTWISE_BLOCK) if a == inner else 1))
        if blocks >= 2 * units:
            return None
        axis = max(axes, key=lambda a: shape[a])
        parts = 1
        while blocks * parts * 2 <= SPLIT_BLOCKS_PER_UNIT * units and shape[axis] % (parts * 2) == 0 and shape[axis] // (parts * 2) >= 64:
            parts *= 2
        return (axis, parts) if parts > 1 else None

    def _reduce(self, ref: Ref, fn: str, axes: tuple[int, ...], out_type: TensorType) -> Ref:
        out = self.new_buffer(out_type, BufferKind.temp)
        shape, rank = ref.shape, ref.view.rank
        kept = [a for a in range(rank) if a not in axes]
        inner_kept = rank - 1 if rank - 1 in kept else None
        inner_reduced = max(axes)
        extent = {a: 1 for a in range(rank)}
        if inner_kept is not None:
            extent[inner_kept] = _block(shape[inner_kept], ELEMENTWISE_BLOCK)
        warps = 4
        if inner_kept is not None:
            extent[inner_reduced] = _block(shape[inner_reduced], COLUMN_REDUCE_BLOCK)
        elif axes == (rank - 1,) and isinstance(shape[-1], int) and ROW_REDUCE_BLOCK < shape[-1] <= WHOLE_ROW_LIMIT:
            # The whole row in one step, so row fusion (tile/rows.py) can replay it.
            extent[inner_reduced] = _block(shape[-1], WHOLE_ROW_LIMIT)
            warps = max(4, min(32, extent[inner_reduced] // 256))
        else:
            extent[inner_reduced] = _block(shape[inner_reduced], ROW_REDUCE_BLOCK)
        out_tile = tuple(extent[a] for a in kept)
        tile_shape = tuple(extent[a] for a in range(rank))
        grid = tuple(_ceil_div(shape[a], extent[a]) for a in kept)
        init = _neutral(fn, ref.dtype)
        combine = "add" if fn == "sum" else "max"
        # A bf16 sum accumulates in f32 and rounds once at the store ("exact, then rounded"):
        # adding thousands of bf16 values in bf16 loses everything past the 8th bit. A max is exact in any dtype.
        acc_dtype = accumulator_dtype(ref.dtype) if fn == "sum" else ref.dtype

        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in kept]
        out_starts = [p if extent[a] == 1 else b.op("mul", p, b.index(extent[a])) for p, a in zip(pids, kept)]
        # The loops combine whole tiles elementwise and one reduce runs after them: a reduction
        # across threads per loop step would cost a shared-memory round trip every step.
        acc = b.full(tile_shape, init, acc_dtype)
        reduced = sorted(axes)

        def loop(builder: TileBuilder, depth: int, starts: dict[int, TValue], carry: TValue) -> TValue:
            if depth == len(reduced):
                full_starts = [starts[a] if a in starts else out_starts[kept.index(a)] for a in range(rank)]
                tile = builder.op("load", *full_starts, arg=0, shape=tile_shape, dtype=ref.dtype, other=init)
                if acc_dtype is not ref.dtype:
                    tile = builder.op("cast", tile, dtype=acc_dtype)
                return builder.op(combine, carry, tile)
            axis = reduced[depth]

            def body(inner: TileBuilder, iv: TValue, carries: list[TValue]) -> list[TValue]:
                return [loop(inner, depth + 1, {**starts, axis: iv}, carries[0])]

            return builder.for_range(builder.index(0), builder.dim(shape[axis]), extent[axis], [carry], body)[0]

        result = b.op("reduce", loop(b, 0, {}, acc), fn=fn, axes=tuple(reduced))
        if acc_dtype is not out_type.dtype:
            result = b.op("cast", result, dtype=out_type.dtype)
        b.emit("store", (result, *out_starts), {"arg": 1})
        self.launch(f"reduce_{fn}", grid, [(ref, False), (out, True)], b.finish(pids), warps, origin=("reduce", fn, tuple(axes)))
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
        # An f32 operand that is a widened b16 value counts as b16: the dot can run on the b16
        # values with an f32 accumulator, which is exact (products of b16 values fit in f32).
        base = {self._matrix_dtype(v) for v in op.inputs}
        matrix = len(base) == 1 and any(i.m == 16 and i.n == 8 and i.k == 16 and i.dtype_a == next(iter(base)).value and i.dtype_acc == "f32" for i in self.capability.matrix_instructions)
        bm, bn, bk = MATRIX_DOT_BLOCK if matrix else DOT_BLOCK
        stages, warps = (MATRIX_DOT_STAGES, MATRIX_DOT_WARPS) if matrix else (1, 4)
        key = None
        if matrix:
            from olaf.search.schedule import dot_key

            key = dot_key(op.attr("spec"), [v.type.shape for v in op.inputs], next(iter(base)).value, self.capability.digest())
            chosen = self.schedules.get(key)
            if chosen is not None:
                bm, bn, bk, stages, warps = chosen.bm, chosen.bn, chosen.bk, chosen.stages, chosen.warps
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
        # The schedule key is recorded so the search can find this dot's launches in a whole program.
        origin = ("contract", op.attr("spec")) + ((key,) if key else ())
        self.launch("contract", grid, [(ra, False), (rb, False), (out, True)], b.finish(pids), warps, origin=origin)
        return out

    def _matrix_dtype(self, value: Value) -> DType:
        producer = self.producers.get(value)
        if value.type.dtype is DType.f32 and producer is not None and producer.kind == "cast" and producer.inputs[0].type.dtype is DType.bf16:
            return DType.bf16
        return value.type.dtype

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
        tile = _elementwise_tile(shape)
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
        result_type = op.outputs[0].type
        narrow = result_type.dtype is DType.bf16
        if narrow:
            # A b16 scatter_add accumulates in f32 and rounds once at the end: its meaning is
            # the exact sum rounded to the result dtype (docs/design.md), which b16 atomics,
            # rounding at every add, would not give (and Ampere has none for bf16 anyway).
            wide = TensorType(result_type.shape, DType.f32)
            target = self.elementwise("cast", wide, [target], lambda b, tiles, *_: b.op("cast", tiles[0], dtype=DType.f32))
            updates = self.elementwise("cast", TensorType(updates.shape, DType.f32), [updates], lambda b, tiles, *_: b.op("cast", tiles[0], dtype=DType.f32))
        out = self.materialize(target)
        spread, n_index = self._index_over(index, updates.shape, axis, batch_dims)
        axis_map = self._axis_map(out.view.rank, axis, n_index)
        shape = updates.shape
        tile = _elementwise_tile(shape)
        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in shape]
        starts = [p if t == 1 else b.op("mul", p, b.index(t)) for p, t in zip(pids, tile)]
        idx = b.op("load", *starts, arg=0, shape=tile, dtype=spread.dtype, other=0)
        value = b.op("load", *starts, arg=1, shape=tile, dtype=updates.dtype, other=0)
        # Lanes past the update tensor must not add: their index reads as 0, a valid position.
        b.emit("atomic_add_indexed", (idx, value, *starts), {"arg": 2, "axis_map": axis_map, "bounds": tuple(shape)})
        grid = tuple(_ceil_div(d, t) for d, t in zip(shape, tile))
        self.launch("scatter_add", grid, [(spread, False), (updates, False), (out, True)], b.finish(pids))
        if narrow:
            return self.elementwise("cast", result_type, [out], lambda b, tiles, *_: b.op("cast", tiles[0], dtype=result_type.dtype))
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
