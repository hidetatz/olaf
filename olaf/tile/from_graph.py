"""Conversion from Graph IR to Tile IR: one task per compute op, views folded into addressing.

The input must be decomposed (no composite ops) and have no scan (passes.unroll). Every compute
op becomes its own task writing a fresh buffer; reshape, permute, broadcast, slice and pad become
strided Views of an existing buffer and cost nothing. Fusion is a separate Tile IR pass, so this
conversion stays a direct, testable translation and the fusion rules can be checked against it.

Tile sizes here are defaults, not tuned choices: the schedule search replaces them per task.
"""

from __future__ import annotations

import math

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

# Schedule defaults (tile sizes, split targets) are the capability's: backend/capability.Tuning.
# A convolution over fewer padded positions than this runs its rows across images (_conv_grid_rows).
SMALL_PLANE = 1024
# Convolutions whose padded grid (stride included) is more than this times their outputs copy
# their windows out instead (im2col), within a size limit for the copy.
IM2COL_WASTE = 1.3
IM2COL_BYTES = 512 << 20


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


def _elementwise_tile(shape, tuning) -> tuple[int, ...]:
    """Tile of an elementwise task: up to tuning.elementwise_block along the last axis and, when
    that is narrow, rows of the axis before it, up to tuning.elementwise_elements in all.

    One 128-element row per instance (one element per thread) left GPT-2's elementwise passes
    far from memory bandwidth: 24576 tiny blocks for a 4 x 1024 x 768 tensor, too little in
    flight per thread. Rows of the previous axis add work per instance without padding a
    768-wide axis to 1024.
    """
    if not shape:
        return ()
    block, elements = tuning.elementwise_block, tuning.elementwise_elements
    if isinstance(shape[-1], int) and shape[-1] >= elements:
        # A long row (the 50257-wide logits) is tiled along itself, so the elements a thread
        # owns share their row: a broadcast row statistic is one load for all of them.
        return tuple(1 for _ in shape[:-1]) + (elements,)
    tile = [1] * len(shape)
    tile[-1] = _block(shape[-1], block)
    if len(shape) >= 2:
        tile[-2] = _block(shape[-2], max(1, elements // tile[-1]))
    # While the trailing axes fit whole, the tile goes on into the axes before them: a
    # [512, 512, 3, 3] weight otherwise had 4 x 4 tiles, 16 elements for 128 threads (its
    # optimizer update took 2 ms instead of 0.2).
    budget = elements // (tile[-1] * (tile[-2] if len(shape) >= 2 else 1))
    for a in range(len(shape) - 3, -1, -1):
        if budget < 2 or not all(isinstance(shape[i], int) and tile[i] >= shape[i] for i in range(a + 1, len(shape))):
            break
        tile[a] = _block(shape[a], budget)
        budget //= tile[a]
    return tuple(tile)


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


def _physical_order(view: View) -> tuple[int, ...] | None:
    """The view's axes from outermost to innermost in memory, when that is not the logical
    order; None when it is, or when the view has no such order (padding, broadcast or
    overlapping axes, symbolic strides). Size-1 axes go first: they do not move in memory."""
    rank = view.rank
    if view.pad is not None or not all(isinstance(d, int) for d in (*view.shape, *view.strides)):
        return None
    moving = [a for a in range(rank) if view.shape[a] != 1]
    if any(view.strides[a] <= 0 for a in moving):
        return None
    order = sorted(moving, key=lambda a: -view.strides[a])
    if order == moving:
        return None
    for outer, inner in zip(order, order[1:]):
        if view.strides[outer] < view.strides[inner] * view.shape[inner]:
            return None
    return (*(a for a in range(rank) if view.shape[a] == 1), *order)


def _permuted(ref: Ref, perm) -> Ref:
    """ref with new axis i = old axis perm[i] (as the permute op)."""
    v = ref.view
    pad = tuple(v.pad[p] for p in perm) if v.pad is not None else None
    source = tuple((None if a is None else perm.index(a), s) for a, s in v.source) if v.source is not None else None
    return replace(ref, view=View(tuple(v.shape[p] for p in perm), tuple(v.strides[p] for p in perm), v.offset, pad, v.fill, source))


def _inverse(perm) -> tuple[int, ...]:
    return tuple(perm.index(i) for i in range(len(perm)))


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
        self._grids: dict[tuple, Ref] = {}
        self._plain: dict[tuple, Ref] = {}  # one copy per padded view (a pooling's windows share it)

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
        tile = _elementwise_tile(shape, self.capability.tuning)
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
        """A ref without padding, for view ops that do not compose with pad. Materialized in
        the view's memory order: a pooling's padded channels-last input copied channels-first
        was a transpose, and its writer could not store into it directly."""
        if ref.view.pad is None:
            return ref
        key = (ref.buffer, ref.view, ref.dtype)
        if key not in self._plain:
            self._plain[key] = self._unpadded(ref)
        return self._plain[key]

    def _unpadded(self, ref: Ref) -> Ref:
        interior = tuple(d - lo - hi for d, (lo, hi) in zip(ref.shape, ref.view.pad))
        order = _physical_order(replace(ref.view, shape=interior, pad=None, source=None))
        if order is None:
            return self.materialize(ref)
        physical = TensorType(tuple(ref.shape[a] for a in order), ref.dtype)
        out = self.new_buffer(physical, BufferKind.temp)
        self.elementwise("copy", physical, [_permuted(ref, order)], lambda b, t, s, ts: t[0], out=out)
        return _permuted(out, _inverse(order))

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
        out_type = op.outputs[0].type
        # Axes every input broadcasts along: computed once and broadcast again (a global
        # pool's gradient is [N, C] spread over H x W; materialized, its layout also had no
        # input to follow and pulled the backward pass to channels-first).
        spread = [a for a, d in enumerate(out_type.shape) if d != 1 and all(r.view.strides[a] == 0 and (r.view.pad is None or r.view.pad[a] == (0, 0)) for r in inputs)]
        if spread and inputs:
            small = tuple(1 if a in spread else d for a, d in enumerate(out_type.shape))
            narrowed = [replace(r, view=replace(r.view, shape=small, pad=r.view.pad)) for r in inputs]
            out = self._pointwise_ref(op.kind, attrs, TensorType(small, out_type.dtype), narrowed)
            v = out.view
            strides = tuple(0 if a in spread else st for a, st in enumerate(v.strides))
            source = tuple((None, 0) if c in spread else (c, st) for c, st in v.source) if v.source is not None else None
            return replace(out, view=View(tuple(out_type.shape), strides, v.offset, v.pad, v.fill, source))
        return self._pointwise_ref(op.kind, attrs, out_type, inputs)

    def _pointwise_ref(self, kind: str, attrs: dict, out_type: TensorType, inputs: list[Ref]) -> Ref:
        order = self._layout(out_type.shape, inputs)
        if order is None:
            return self.elementwise(kind, out_type, inputs, lambda b, t, s, ts: b.op(kind, *t, **attrs))
        physical = TensorType(tuple(out_type.shape[a] for a in order), out_type.dtype)
        out = self.elementwise(kind, physical, [_permuted(r, order) for r in inputs], lambda b, t, s, ts: b.op(kind, *t, **attrs))
        return _permuted(out, _inverse(order))

    def _layout(self, shape, inputs: list[Ref]) -> tuple[int, ...] | None:
        """The memory order a pointwise result takes: its inputs' (full-shape inputs only, by
        bytes; a tie keeps the logical order), or None for the logical order.

        A result laid out logically from a permuted input reads it with strided loads (an
        elementwise tile runs along the last logical axis) and hands the permutation to its
        reader as a copy: a convolution computes channels-last, and laid out channels-first
        its batch norm and ReLU read it across channels and the next convolution copied the
        result back to channels-last (ResNet-18: two copies per convolution).
        """
        votes: dict[tuple[int, ...] | None, int] = {}
        for r in inputs:
            if tuple(r.shape) != tuple(shape) or r.view.pad is not None or any(st == 0 and d != 1 for st, d in zip(r.view.strides, r.shape)):
                continue
            order = _physical_order(r.view)
            votes[order] = votes.get(order, 0) + np.dtype(r.dtype.numpy).itemsize
        if not votes:
            return None
        best = max(votes.values())
        return None if votes.get(None) == best else max(votes, key=lambda o: votes[o])

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
            if ref.view.pad is not None:
                ref = self.plain(ref)  # in memory order, then maybe viewable as `shape`
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
        order = _physical_order(ref.view)
        if order is None:
            return self._reduce_split(ref, fn, axes, op.outputs[0].type)
        # Reduced in memory order (a channels-last batch norm statistic is a column sum over
        # contiguous channels, not a strided row sum over W), the kept axes in that order too.
        kept = [a for a in order if a not in axes]
        out_type = op.outputs[0].type
        physical = TensorType(tuple(ref.shape[a] for a in kept), out_type.dtype)
        result = self._reduce_split(_permuted(ref, order), fn, tuple(sorted(order.index(a) for a in axes)), physical)
        logical = [a for a in range(ref.view.rank) if a not in axes]
        return _permuted(result, tuple(kept.index(a) for a in logical))

    def _reduce_split(self, ref: Ref, fn: str, axes, out_type: TensorType) -> Ref:
        split = self._split(ref, axes)
        if split is not None:
            # Two passes: P slices of one reduced axis in parallel, then the P partials.
            axis, parts = split
            size = ref.shape[axis]
            v = ref.view
            shape = (*v.shape[:axis], parts, size // parts, *v.shape[axis + 1 :])
            strides = (*v.strides[:axis], v.strides[axis] * (size // parts), v.strides[axis], *v.strides[axis + 1 :])
            pad = None if v.pad is None else (*v.pad[:axis], (0, 0), (0, 0), *v.pad[axis + 1 :])
            # Parts of one element keep a source map (the part index is the old index), so
            # a pointwise producer can still be inlined into the first pass.
            source = None
            if v.source is not None and size // parts == 1:
                source = tuple((None if a is None else a if a <= axis else a + 1, st) for a, st in v.source)
            sliced = replace(ref, view=View(shape, strides, v.offset, pad, v.fill, source))
            inner_axes = tuple(a if a < axis else a + 1 for a in axes)
            kept_shape = tuple(d for i, d in enumerate(shape) if i not in inner_axes)
            # Partial sums stay in the accumulator dtype: rounding them to bf16 and again after
            # combining them would round twice where the unsplit sum rounds once.
            partial_dtype = accumulator_dtype(ref.dtype) if fn == "sum" else ref.dtype
            partial = self._reduce(sliced, fn, inner_axes, TensorType(kept_shape, partial_dtype))
            position = sum(1 for a in range(axis) if a not in axes)
            return self._reduce(partial, fn, (position,), out_type)
        return self._reduce(ref, fn, axes, out_type)

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
            blocks *= -(-shape[a] // (_block(shape[a], self.capability.tuning.elementwise_block) if a == inner else 1))
        if blocks >= 2 * units:
            return None
        axis = max(axes, key=lambda a: shape[a])
        # Each part still reduces the other reduced axes: a channels-last batch norm statistic
        # over [N, H, W, C] splits N into parts of one image each, H x W rows apiece. Counting
        # only the split axis (parts of 64 or more) left it 2 parts of 64 x 32 x 32 rows, and
        # each lane summed thousands of values in sequence (f32 rounding error grows with that
        # count; YOLO's f32 training gate failed at 2.3x PyTorch's error).
        rest = math.prod(shape[a] for a in axes if a != axis)
        parts = 1
        while blocks * parts * 2 <= self.capability.tuning.split_blocks_per_unit * units and shape[axis] % (parts * 2) == 0 and shape[axis] // (parts * 2) * rest >= 64:
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
            extent[inner_kept] = _block(shape[inner_kept], self.capability.tuning.elementwise_block)
        warps = 4
        if inner_kept is not None:
            extent[inner_reduced] = _block(shape[inner_reduced], self.capability.tuning.column_reduce_block)
        elif axes == (rank - 1,) and isinstance(shape[-1], int) and self.capability.tuning.row_reduce_block < shape[-1] <= self.capability.tuning.whole_row_limit:
            # The whole row in one step, so row fusion (tile/rows.py) can replay it.
            extent[inner_reduced] = _block(shape[-1], self.capability.tuning.whole_row_limit)
            warps = max(4, min(32, extent[inner_reduced] // 256))
        else:
            extent[inner_reduced] = _block(shape[inner_reduced], self.capability.tuning.row_reduce_block)
            # Reduced axes that end the shape share one tile: a batch norm's statistic sums
            # [N, C, H, W] over N, H and W, and a 16-wide W alone made 16-element steps (19 GB/s).
            budget = self.capability.tuning.row_reduce_block // extent[inner_reduced]
            for a in range(inner_reduced - 1, -1, -1):
                if a not in axes or budget < 2 or not isinstance(shape[a], int) or extent[a + 1] != shape[a + 1]:
                    break
                extent[a] = _block(shape[a], budget)
                budget //= extent[a]
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
        # The accumulator's columns should run along the output's contiguous label: a dot whose
        # rows are that label (an input gradient x @ W^T written as W's term first) stores
        # through a transposed accumulator, and fused epilogues then read every operand across
        # the register layout (Whisper's fc1 input gradient: 2.95 ms against 0.67 ms unfused).
        # Not when the new rows are shorter than a tile and than the old ones: a weight
        # gradient into 3 output channels would run 16-row tiles (DCGAN's last layer: 0.87 ms
        # against 0.24 ms with the batch as rows and a transposed store).
        values = list(op.inputs)
        size0 = {**dict(zip(ta, ra.shape)), **dict(zip(tb, rb.shape))}
        if (
            m_labels
            and n_labels
            and out_labels.index(m_labels[-1]) > out_labels.index(n_labels[-1])
            and isinstance(size0[n_labels[-1]], int)
            and isinstance(size0[m_labels[-1]], int)
            and size0[n_labels[-1]] >= min(size0[m_labels[-1]], self.capability.tuning.matrix_dot_block[0])
        ):
            (ta, tb), (ra, rb), (m_labels, n_labels) = (tb, ta), (rb, ra), (n_labels, m_labels)
            values.reverse()
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
        bm, bn, bk = self.capability.tuning.matrix_dot_block if matrix else self.capability.tuning.dot_block
        stages, warps = (self.capability.tuning.matrix_dot_stages, self.capability.tuning.matrix_dot_warps) if matrix else (1, 4)
        key = None
        if matrix:
            from olaf.search.schedule import dot_key

            key = dot_key(op.attr("spec"), [v.type.shape for v in op.inputs], next(iter(base)).value, self.capability.digest())
            chosen = self.schedules.get(key)
            if chosen is not None:
                bm, bn, bk, stages, warps = chosen.bm, chosen.bn, chosen.bk, chosen.stages, chosen.warps
            split = self._contract_split(op, (ta, tb), (ra, rb), values, out_labels, k_labels, size, bm, bn)
            if split is not None:
                return split
        extent = {c: 1 for c in size}
        # Matrix instructions need 16-wide tiles; a narrower extent (a first layer's 3 input
        # channels) is padded up and masked rather than falling back to scalar products.
        least = 16 if matrix else 1
        m_tiled = [m_inner] if m_inner else []
        if m_inner:
            extent[m_inner] = max(least, _block(size[m_inner], bm))
            # Rows shorter than a tile (a convolution's 8 output columns, labels n, h, w of
            # implicit GEMM rows) would leave most of it empty; the tile reaches over the labels
            # before them instead, while each inner one is covered whole. The rows are then not
            # one affine axis: the dot's operand copies split a row number into the labels.
            run = {m_inner: _block(size[m_inner], bm)}
            for c in reversed(m_labels[:-1]):
                first = next(iter(run))
                total = math.prod(run.values())
                if not matrix or run[first] != size[first] or total * 2 > bm or not isinstance(size[c], int):
                    break
                run = {c: _block(size[c], bm // total), **run}
            if len(run) > 1 and math.prod(run.values()) % least == 0:
                m_tiled = list(run)
                extent.update(run)
        if n_inner:
            extent[n_inner] = max(least, _block(size[n_inner], bn))
        if k_inner:
            extent[k_inner] = max(least, _block(size[k_inner], bk))
        en, ek = (extent[c] if c else 1 for c in (n_inner, k_inner))
        em = math.prod(extent[c] for c in m_tiled)
        out_type = op.outputs[0].type
        out = self.new_buffer(out_type, BufferKind.temp)
        acc_dtype = accumulator_dtype(ra.dtype)
        grid = tuple(_ceil_div(size[c], extent[c]) for c in out_labels)

        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in out_labels]
        out_starts = {c: p if extent[c] == 1 else b.op("mul", p, b.index(extent[c])) for c, p in zip(out_labels, pids)}

        def operand(builder: TileBuilder, arg: int, term: str, ref: Ref, starts: dict[str, TValue], rows: list[str], cols: str | None) -> TValue:
            tile = builder.op("load", *(starts[c] for c in term), arg=arg, shape=tuple(extent[c] for c in term), dtype=ref.dtype, other=0)
            # Move the tiled labels last (rows, then cols) and drop the unit axes around them.
            tiled = [*rows, *([cols] if cols else [])]
            order = [i for i, c in enumerate(term) if c not in tiled] + [term.index(c) for c in tiled]
            if order != list(range(len(term))):
                tile = builder.op("permute", tile, perm=tuple(order))
            return builder.op("reshape", tile, shape=(math.prod(extent[c] for c in rows), extent[cols] if cols else 1))

        def step(builder: TileBuilder, starts: dict[str, TValue], acc: TValue) -> TValue:
            a = operand(builder, 0, ta, ra, starts, m_tiled, k_inner)
            bt = operand(builder, 1, tb, rb, starts, [k_inner] if k_inner else [], n_inner)
            return builder.op("dot", a, bt, acc)

        def loop(builder: TileBuilder, depth: int, starts: dict[str, TValue], acc: TValue) -> TValue:
            if depth == len(k_labels):
                return step(builder, starts, acc)
            label = k_labels[depth]

            def body(inner: TileBuilder, iv: TValue, carries: list[TValue]) -> list[TValue]:
                return [loop(inner, depth + 1, {**starts, label: iv}, carries[0])]

            pipelined = stages if label == k_inner else 1
            return builder.for_range(builder.index(0), builder.dim(size[label]), extent[label], [acc], body, stages=pipelined)[0]

        def flat_loop(builder: TileBuilder, starts: dict[str, TValue], acc: TValue) -> TValue:
            """All reduced labels in one loop, its step split into each label's position by
            division: nested loops would pipeline only the innermost (a convolution's 64
            channels in 32-wide steps: two iterations between the 3x3 kernel loops)."""
            tiles = -(-size[k_inner] // ek)
            outer = [c for c in k_labels if c != k_inner]
            total = tiles * math.prod(size[c] for c in outer)

            def body(inner: TileBuilder, iv: TValue, carries: list[TValue]) -> list[TValue]:
                here = dict(starts)
                rest = iv
                here[k_inner] = inner.op("mul", inner.op("mod", rest, inner.index(tiles)), inner.index(ek))
                rest = inner.op("floordiv", rest, inner.index(tiles))
                for c in reversed(outer):
                    here[c] = inner.op("mod", rest, inner.index(size[c]))
                    rest = inner.op("floordiv", rest, inner.index(size[c]))
                return [step(inner, here, carries[0])]

            return builder.for_range(builder.index(0), builder.index(total), 1, [acc], body, stages=stages)[0]

        # Only when the innermost loop is too short to pipeline: with long inner loops (a
        # weight gradient over batch x sequence, 32 steps of the sequence) nesting measured
        # faster (Llama's step: 213 ms nested, 239 flattened).
        flatten = len(k_labels) > 1 and k_inner is not None and all(isinstance(size[c], int) for c in k_labels) and -(-size[k_inner] // ek) < 8
        acc0 = b.full((em, en), 0, acc_dtype)
        acc = flat_loop(b, dict(out_starts), acc0) if flatten else loop(b, 0, dict(out_starts), acc0)
        result = acc if acc_dtype is out_type.dtype else b.op("cast", acc, dtype=out_type.dtype)
        if len(m_tiled) > 1:
            held = [*m_tiled, n_inner]
            result = b.op("reshape", result, shape=(*(extent[c] for c in m_tiled), en))
            perm = sorted(range(len(held)), key=lambda i: out_labels.index(held[i]) if held[i] else len(out_labels))
            if perm != list(range(len(held))):
                result = b.op("permute", result, perm=tuple(perm))
        else:
            tiled = [c for c in out_labels if c in (m_inner, n_inner)]
            if tiled == [n_inner, m_inner] and m_inner and n_inner:
                result = b.op("permute", result, perm=(1, 0))
        result = b.op("reshape", result, shape=tuple(extent[c] for c in out_labels))
        b.emit("store", (result, *(out_starts[c] for c in out_labels)), {"arg": 2})
        # The schedule key is recorded so the search can find this dot's launches in a whole program.
        origin = ("contract", op.attr("spec")) + ((key,) if key else ())
        self.launch("contract", grid, [(ra, False), (rb, False), (out, True)], b.finish(pids), warps, origin=origin)
        return out

    def _contract_split(self, op: Op, terms: tuple[str, str], refs: tuple[Ref, Ref], values: list[Value], out_labels: str, k_labels: list[str], size: dict, bm: int, bn: int) -> Ref | None:
        """A product with too few output tiles for the device and a long reduction (a linear
        layer's weight gradient, 384 x 384 over batch x sequence: 9 tiles on 28 SMs) as partial
        products over the outer reduced labels, which become output labels, then summed.

        Only whole labels move (the batch of a weight gradient): their partial products need no
        new views. The partials are kept in the accumulator's type and summed once, so the sum
        rounds to the output type once, as the unsplit product does.
        """
        ta, tb = terms
        if not all(isinstance(size[c], int) for c in size):
            return None
        m_inner = next((c for c in reversed(out_labels) if c in ta and c not in tb), None)
        n_inner = next((c for c in reversed(out_labels) if c in tb and c not in ta), None)
        if m_inner is None or n_inner is None:
            return None
        tiles = math.prod(size[c] for c in out_labels if c not in (m_inner, n_inner)) * _ceil_div(size[m_inner], bm) * _ceil_div(size[n_inner], bn)
        k_inner = [c for c in ta if c in k_labels][-1]
        reduced = math.prod(size[c] for c in k_labels)
        units = self.capability.roofline.compute_units
        moved = []
        # 2 tiles per unit: 128 x 128 tiles with 3 stages fit about 2 blocks on an SM. Each part
        # keeps at least 1024 of the reduction, so the partials' write and sum (an f32 output
        # per part) stay small against the products.
        for c in (c for c in ta if c in k_labels and c != k_inner):
            if tiles >= 2 * units or reduced // size[c] < 1024:
                break
            moved.append(c)
            tiles *= size[c]
            reduced //= size[c]
        if not moved:
            return None
        from olaf.graph.builder import make_attrs

        partial_labels = "".join(moved) + out_labels
        acc = accumulator_dtype(refs[0].dtype)
        partial_type = TensorType(tuple(size[c] for c in partial_labels), acc)
        spec = f"{ta},{tb}->{partial_labels}"
        partial = Op("contract", tuple(values), (Value(partial_type),), make_attrs({"spec": spec}))
        result = self._op_contract(partial, list(refs))
        return self._reduce(result, "sum", tuple(range(len(moved))), op.outputs[0].type)

    def _op_conv(self, op: Op, inputs: list[Ref]) -> Ref:
        """conv as one dense matrix product over the padded grid (see _conv_grid_rows and
        _conv_grid_reduce), or, where neither fits (groups), over window views (_conv_windows)."""
        g = op.attr("groups")
        if any(ld > 1 for ld in op.attr("lhs_dilation")) and all(d == 1 for d in op.attr("dilation")) and all(st == 1 for st in op.attr("stride")):
            return self._conv_phases(op, inputs)
        out_positions = math.prod(op.outputs[0].type.shape[2:])
        kernel_positions = math.prod(inputs[1].shape[2:])
        if g == 1 and out_positions < kernel_positions:
            return self._conv_grid_reduce(op, inputs)
        if g == 1:
            return self._conv_grid_rows(op, inputs)
        return self._conv_windows(op, inputs)

    def _conv_phases(self, op: Op, inputs: list[Ref]) -> Ref:
        """A convolution over an input dilated by s (the input gradient of a stride-s
        convolution, a transposed convolution) as s^n convolutions without the zeros.

        Output position o = m * s + r reads the dilated input only at taps u with
        u = t * s + q_r, q_r = (low - r) mod s, and those taps read input element
        m + (r + q_r - low) / s + t: per phase r a plain stride-1 convolution of the input with
        every s-th tap of the kernel. Computed over the dilated input instead, (s^n - 1) / s^n
        of the products are with zeros. Each phase's result goes to its strided positions of
        the output; a phase without taps is zeros.
        """
        from olaf.graph.builder import make_attrs

        x, w = inputs
        ld, padding, g = op.attr("lhs_dilation"), op.attr("padding"), op.attr("groups")
        n = len(ld)
        out_type = op.outputs[0].type
        out_spatial = out_type.shape[2:]
        kernel = w.shape[2:]
        # Assembled channels-last, like the phases' own results (and the next layer's grid).
        order = (0, *range(2, 2 + n), 1)
        physical = TensorType(tuple(out_type.shape[a] for a in order), out_type.dtype)
        out = _permuted(self.new_buffer(physical, BufferKind.temp), _inverse(order))
        os_ = out.view.strides
        w = self.plain(w)
        for phase in np.ndindex(*ld):
            taps, lows, highs, counts, firsts = [], [], [], [], []
            for i, (r, s_) in enumerate(zip(phase, ld)):
                lo = padding[i][0]
                q = (lo - r) % s_
                t = max(0, -(-(kernel[i] - q) // s_))
                m = max(0, -(-(out_spatial[i] - r) // s_))
                base = (r + q - lo) // s_
                taps.append(t)
                firsts.append(q)
                counts.append(m)
                lows.append(-base)
                highs.append(m - 1 + t - x.shape[2 + i] + base)
            # In memory order ([N, *counts, C]): the copy runs along the channels on both sides.
            target = View((out_type.shape[0], *counts, out_type.shape[1]), (os_[0], *(st * s_ for st, s_ in zip(os_[2:], ld)), os_[1]), sum(r * st for r, st in zip(phase, os_[2:])))
            if 0 in counts:
                continue
            if 0 in taps:
                self.elementwise("fill", TensorType(target.shape, out_type.dtype), [], lambda b, tiles, starts, tile: b.full(tile, 0, out_type.dtype), out=replace(out, view=target))
                continue
            wv = w.view
            sub = Ref(w.buffer, View((*w.shape[:2], *taps), (*wv.strides[:2], *(st * s_ for st, s_ in zip(wv.strides[2:], ld))), wv.offset + sum(q * st for q, st in zip(firsts, wv.strides[2:]))), w.dtype)
            phase_type = TensorType((*out_type.shape[:2], *counts), out_type.dtype)
            attrs = {"stride": (1,) * n, "padding": tuple(zip(lows, highs)), "dilation": (1,) * n, "lhs_dilation": (1,) * n, "groups": g}
            phase_op = Op("conv", (Value(TensorType(x.shape, x.dtype)), Value(TensorType(sub.shape, w.dtype))), (Value(phase_type),), make_attrs(attrs))
            result = _permuted(self._op_conv(phase_op, [x, sub]), order)
            self.elementwise("copy", TensorType(target.shape, out_type.dtype), [result], lambda b, t, s, ts: t[0], out=replace(out, view=target))
        return out

    def _padded_grid(self, x: Ref, order, padding, tail: int) -> Ref:
        """x [N, C, *spatial] (axes listed by `order` as batch, spatial..., channels) copied,
        padded, to a flat channels-last [N, *padded, C] buffer with `tail` unwritten elements
        after it (see _conv_grid_rows). One grid per input and padding: the forward conv and
        its weight gradient's both read the forward's input padded the same way, and copied it
        twice. The source view has a source map, so the copy's writer can store into the grid
        directly (tile.normalize.eliminate_copies)."""
        src = self.plain(x)
        shape = tuple(src.shape[i] for i in order)
        pad = ((0, 0), *((max(lo, 0), max(hi, 0)) for lo, hi in padding), (0, 0))
        padded_shape = tuple(lo + d + hi for d, (lo, hi) in zip(shape, pad))
        permuted = _permuted(src, tuple(order))
        key = (src.buffer, permuted.view, pad)
        known = self._grids.get(key)
        if known is not None and math.prod(self.buffers[known.buffer].type.shape) >= math.prod(padded_shape) + tail:
            return known
        # The tail as whole extra images: the buffer keeps the grid's axes, so views of it have
        # source maps (which moving the copy's writer into it needs).
        extra = -(-tail // math.prod(padded_shape[1:])) if tail else 0
        base = self.new_buffer(TensorType((padded_shape[0] + extra, *padded_shape[1:]), x.dtype), BufferKind.temp)
        source = permuted.view.source
        for axis, (lo, _) in enumerate(pad):
            source = _shifted(source, axis, -lo)
        view = View(padded_shape, permuted.view.strides, permuted.view.offset, pad, 0.0, source)
        into = full_view(padded_shape)
        self.elementwise("copy", TensorType(padded_shape, x.dtype), [replace(src, view=view)], lambda b, t, s, ts: t[0], out=replace(base, view=into))
        self._grids[key] = base
        return base

    def _spread_into(self, x: Ref, shape, strides, offset, padding, lhs_dilation, dtype, order=None) -> Ref:
        """A zeroed buffer of `shape` holding x's elements at `strides` (in x's [N, C, *spatial]
        order) from `offset`: x padded and dilated, in whatever layout the strides say.

        order: the axis order the copy walks (its tiles run along the last), so that its stores
        are the contiguous side: into a channels-last buffer, writes along the channels are
        coalesced and the strided side is the reads, which the caches absorb (the other way
        round, NCHW to NHWC ran at 16 GB/s)."""
        base = self.elementwise("fill", TensorType(tuple(shape), dtype), [], lambda b, tiles, starts, tile: b.full(tile, 0, dtype))
        src = self.plain(x)
        order = tuple(order or range(len(x.shape)))
        src = replace(src, view=View(tuple(src.shape[i] for i in order), tuple(src.view.strides[i] for i in order), src.view.offset))
        into = View(tuple(x.shape[i] for i in order), tuple(strides[i] for i in order), offset)
        self.elementwise("copy", TensorType(src.shape, dtype), [src], lambda b, t, s, ts: t[0], out=replace(base, view=into))
        return base

    def _conv_grid_rows(self, op: Op, inputs: list[Ref]) -> Ref:
        """conv as one matrix product per image, its rows every position of the padded grid.

        The input goes, padded and dilated, to a zeroed channels-last [N, *padded spatial, C]
        buffer. Numbering output positions over the whole padded plane, not just the valid
        ones, row m at kernel offset (u, v) starts at element (m + u * d * Wp + v * d) * C:
        affine in m, so per image the windows are a [plane, KH, KW * C] operand whose
        reduction is contiguous (KW * C when undilated; C stays 16-byte aligned for the async
        copies, which a channels-first layout cannot give: its kernel offsets shift rows by
        single elements). The result is [N, plane, O], channels-last with a row stride of Wp,
        read through a view as [N, O, *spatial]; its pointwise readers keep that layout. The positions outside the valid
        outputs are computed and dropped: (Hp * Wp) / (OH * OW) - 1 extra work, 7% for a padded
        3x3 over 56x56, 56% over 8x8. A stride s is computed at stride 1 and subsampled (s^n the
        work; cuDNN does better there). The buffer's tail is padded so the last rows' windows
        stay inside it.
        """
        from olaf.graph.builder import make_attrs

        x, w = inputs
        stride, padding, dilation, lhs_dilation = (op.attr(k) for k in ("stride", "padding", "dilation", "lhs_dilation"))
        n = len(stride)
        N, C = x.shape[:2]
        O = w.shape[0]
        kernel = tuple(w.shape[2:])
        out_type = op.outputs[0].type
        spread = [(s - 1) * ld + 1 + max(lo, 0) + max(hi, 0) for s, ld, (lo, hi) in zip(x.shape[2:], lhs_dilation, padding)]
        row = [math.prod(spread[i + 1 :]) for i in range(n)]
        plane = math.prod(spread)
        start = sum(max(-lo, 0) * r for (lo, _), r in zip(padding, row))  # negative padding crops
        reach = start + sum((k - 1) * d * r for k, d, r in zip(kernel, dilation, row))
        if all(ld == 1 for ld in lhs_dilation):
            # One pass writes the whole grid: the input read through a padded view (zeros
            # outside it), channels last. The tail past the grid is left unwritten: only rows
            # outside the valid outputs read it, and those are dropped.
            base = self._padded_grid(x, (0, *range(2, 2 + n), 1), padding, reach * C)
        else:
            base = self._spread_into(
                x,
                ((N * plane + reach) * C,),
                (plane * C, 1, *(ld * r * C for ld, r in zip(lhs_dilation, row))),
                sum(max(lo, 0) * r for (lo, _), r in zip(padding, row)) * C,
                padding,
                lhs_dilation,
                x.dtype,
                order=(0, *range(2, 2 + n), 1),
            )
        out_spatial = tuple(out_type.shape[2:])
        waste = plane / math.prod(out_spatial)
        if waste > IM2COL_WASTE and self._implicit_rows(C, out_spatial, dilation):
            return self._conv_implicit(x, w, base, op, spread, row, start)
        if waste > IM2COL_WASTE and N * math.prod(out_spatial) * math.prod(kernel) * C * 2 <= IM2COL_BYTES:
            return self._conv_im2col(x, w, base, op, spread, row, start)
        letters = "uvz"[:n]
        merged = dilation[-1] == 1  # the last kernel axis and the channels are one contiguous run
        # Small planes (a CIFAR network's later stages: 6x6 padded) would leave most of a
        # 128-row tile empty per image; their rows run on across images instead (the grid is
        # channels-last, so image n + 1 starts where image n ends), and the [rows, O] result is
        # copied to the channels-first output.
        across = plane < SMALL_PLANE
        shape, strides, labels = ([N * plane], [C], "m") if across else ([N, plane], [plane * C, C], "nm")
        for i in range(n - 1 if merged else n):
            shape.append(kernel[i])
            strides.append(dilation[i] * row[i] * C)
            labels += letters[i]
        shape.append(kernel[-1] * C if merged else C)
        strides.append(1)
        labels += "c"
        windows = Ref(base.buffer, View(tuple(shape), tuple(strides), start * C), x.dtype)
        # The weight channels-last too: [O, *kernel, C], merged like the windows.
        wt = self.materialize(replace(w, view=View((O, *kernel, w.shape[1]), (w.view.strides[0], *w.view.strides[2:], w.view.strides[1]), w.view.offset)))
        ws = contiguous_strides((O, *kernel, w.shape[1]))
        w_shape = [O, *kernel[: n - 1 if merged else n], kernel[-1] * C if merged else C]
        w_strides = [ws[0], *ws[1 : (n if merged else n + 1)], 1]
        weights = Ref(wt.buffer, View(tuple(w_shape), tuple(w_strides), 0), w.dtype)
        k_order = labels[1:] if across else labels[2:]
        spec = f"{labels},o{k_order}->" + ("mo" if across else "nmo")
        rows_type = TensorType((N * plane, O) if across else (N, plane, O), out_type.dtype)
        contract = Op("contract", (Value(TensorType(windows.shape, x.dtype)), Value(TensorType(weights.shape, w.dtype))), (Value(rows_type),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [windows, weights])
        # Left channels-last: pointwise readers follow that layout (_layout), and the next
        # convolution's grid is channels-last too.
        view = View(tuple(out_type.shape), (plane * O, 1, *(st * r * O for st, r in zip(stride, row))), result.view.offset)
        return replace(result, view=view)

    @staticmethod
    def _implicit_rows(C, out_spatial, dilation) -> bool:
        """Whether _conv_implicit applies: the dot's 16-byte operand copies need channel runs of
        8, and its row tile spans output rows only when each is covered whole (a power of two,
        tile/from_graph._op_contract)."""
        last = out_spatial[-1]
        return isinstance(C, int) and C % 8 == 0 and dilation[-1] == 1 and isinstance(last, int) and last & (last - 1) == 0

    def _conv_implicit(self, x: Ref, w: Ref, base: Ref, op: Op, spread, row, start) -> Ref:
        """Implicit GEMM: the dot reads the windows straight from the channels-last grid, its
        rows the output positions (n, *out), its reduction (*kernel[:-1], KW * C). Where
        _conv_im2col copies KH * KW times the input out first (0.9 GB a ResNet-18 step), the
        rows here are not one affine axis: a row tile spans whole output rows of a few images
        and the operand copies split a row number into n, h, w (backend/cuda/mma.py)."""
        from olaf.graph.builder import make_attrs

        stride, dilation = op.attr("stride"), op.attr("dilation")
        n = len(stride)
        N, C = x.shape[:2]
        O = w.shape[0]
        kernel = tuple(w.shape[2:])
        out_type = op.outputs[0].type
        out_spatial = tuple(out_type.shape[2:])
        plane = math.prod(spread)
        positions, kernels = "hwz"[:n], "uvy"[: n - 1]
        shape = (N, *out_spatial, *kernel[:-1], kernel[-1] * C)
        strides = (plane * C, *(st * r * C for st, r in zip(stride, row)), *(d * r * C for d, r in zip(dilation[:-1], row[:-1])), 1)
        windows = Ref(base.buffer, View(shape, strides, start * C), x.dtype)
        wt = self.materialize(replace(w, view=View((O, *kernel, w.shape[1]), (w.view.strides[0], *w.view.strides[2:], w.view.strides[1]), w.view.offset)))
        K = kernel[-1] * C
        weights = Ref(wt.buffer, View((O, *kernel[:-1], K), (*contiguous_strides((O, *kernel[:-1], K)),), 0), w.dtype)
        rows_type = TensorType((N, *out_spatial, O), out_type.dtype)
        spec = f"n{positions}{kernels}k,o{kernels}k->n{positions}o"
        contract = Op("contract", (Value(TensorType(windows.shape, x.dtype)), Value(TensorType(weights.shape, w.dtype))), (Value(rows_type),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [windows, weights])
        cs = contiguous_strides(rows_type.shape)
        view = View(tuple(out_type.shape), (cs[0], 1, *cs[1:-1]), result.view.offset)
        return replace(result, view=view)

    def _conv_im2col(self, x: Ref, w: Ref, base: Ref, op: Op, spread, row, start) -> Ref:
        """The windows copied out ([N * out positions, kernel * C], channels last, contiguous
        on both sides of the copy), then one dense matrix product. Used where computing every
        grid position would waste more than IM2COL_WASTE: small planes (a padded 6x6 around 4x4
        outputs is 2.25x the work) and strided convolutions (stride s computed at stride 1 is
        s^n the work), where copying k^n times the (small) input costs less."""
        from olaf.graph.builder import make_attrs

        stride, dilation = op.attr("stride"), op.attr("dilation")
        n = len(stride)
        N, C = x.shape[:2]
        O = w.shape[0]
        kernel = tuple(w.shape[2:])
        out_type = op.outputs[0].type
        out_spatial = tuple(out_type.shape[2:])
        plane = math.prod(spread)
        shape = (N, *out_spatial, *kernel, C)
        windows = Ref(base.buffer, View(shape, (plane * C, *(st * r * C for st, r in zip(stride, row)), *(d * r * C for d, r in zip(dilation, row)), 1), start * C), x.dtype)
        cols = self.elementwise("copy", TensorType(shape, x.dtype), [windows], lambda b, t, s, ts: t[0])
        M, K = N * math.prod(out_spatial), math.prod(kernel) * C
        rows = Ref(cols.buffer, View((M, K), (K, 1), 0), x.dtype)
        wt = self.materialize(replace(w, view=View((O, *kernel, w.shape[1]), (w.view.strides[0], *w.view.strides[2:], w.view.strides[1]), w.view.offset)))
        weights = Ref(wt.buffer, View((O, K), (K, 1), 0), w.dtype)
        result_type = TensorType((M, O), out_type.dtype)
        contract = Op("contract", (Value(TensorType((M, K), x.dtype)), Value(TensorType((O, K), w.dtype))), (Value(result_type),), make_attrs({"spec": "mk,ok->mo"}))
        result = self._op_contract(contract, [rows, weights])
        positions = contiguous_strides(out_spatial)
        view = View(tuple(out_type.shape), (math.prod(out_spatial) * O, 1, *(p * O for p in positions)), result.view.offset)
        return replace(result, view=view)

    def _swapped_layout(self, ref: Ref) -> Ref:
        """ref [A, B, ...] copied to memory laid out [B, A, ...] (and still read as [A, B, ...]).
        A weight gradient's conv computes [C, O, *kernel]; autodiff then swaps its first two
        axes into the weight's [O, C, *kernel], and the optimizer reads that along the kernel:
        laid out the other way, every one of its reads was strided (2.2 ms for one 512 x 512 x
        3 x 3 update instead of 0.05)."""
        shape = tuple(ref.shape)
        swapped = (shape[1], shape[0], *shape[2:])
        out = self.new_buffer(TensorType(swapped, ref.dtype), BufferKind.temp)
        cs = contiguous_strides(swapped)
        view = View(shape, (cs[1], cs[0], *cs[2:]), 0)
        return self.elementwise("copy", TensorType(shape, ref.dtype), [ref], lambda b, t, s, ts: t[0], out=replace(out, view=view))

    def _conv_reduce_im2col(self, op: Op, inputs: list[Ref], spread, row, plane) -> Ref:
        """A weight gradient's conv (small output, kernel as large as the input) as the
        transposed product of the forward's im2col columns and the output gradient.

        The conv here is conv(x^T, dy^T): its batch is the forward's input channels, its
        channels the forward's batch, its kernel the forward's output positions and its output
        the forward's kernel positions. In the forward's terms it is dW = dY^T cols, with cols
        [batch, out positions, kernel positions x channels] the windows copied out channels-last
        (aligned and contiguous, unlike reading the grid at each kernel offset). Its result is
        only kernel x channels by O, a few tiles, and its reduction (batch x out positions)
        long, so the batch is split into chunks reduced in parallel, then summed.
        """
        from olaf.graph.builder import make_attrs

        xc, wc = inputs  # xc [Cf, Nf, *in], wc [Of, Nf, *outf] in the forward's names
        stride, padding, dilation = (op.attr(k) for k in ("stride", "padding", "dilation"))
        n = len(stride)
        Cf, Nf = xc.shape[:2]
        Of = wc.shape[0]
        outf = tuple(wc.shape[2:])  # the forward's output positions: this conv's kernel
        kern = tuple(op.outputs[0].type.shape[2:])  # the forward's kernel positions: this conv's output
        out_type = op.outputs[0].type
        # The forward's input, padded, channels-last: [Nf, *spread, Cf].
        start = sum(max(-lo, 0) * r for (lo, _), r in zip(padding, row))
        base = self._padded_grid(xc, (1, *range(2, 2 + n), 0), padding, 0)
        # cols[nf, *outf, *kern, cf] = x[nf, outf * d + kern * s, cf] (this conv's stride s applies
        # to its output, the forward's kernel; its dilation d to its kernel, the forward's output).
        shape = (Nf, *outf, *kern, Cf)
        windows = View(shape, (plane * Cf, *(d * r * Cf for d, r in zip(dilation, row)), *(st * r * Cf for st, r in zip(stride, row)), 1), start * Cf)
        cols = self.elementwise("copy", TensorType(shape, xc.dtype), [replace(base, view=windows)], lambda b, t, s, ts: t[0])
        P, KK = math.prod(outf), math.prod(kern) * Cf
        tiles = -(-KK // 128) * -(-Of // 128)
        chunks = 1
        while tiles * chunks * 2 <= 4 * self.capability.roofline.compute_units and Nf % (chunks * 2) == 0:
            chunks *= 2
        per = Nf // chunks
        a = Ref(cols.buffer, View((chunks, per, P, KK), (per * P * KK, P * KK, KK, 1), 0), xc.dtype)
        dy = wc if wc.view.is_contiguous or all(st != 0 for st in wc.view.strides) else self.materialize(wc)
        ds = dy.view.strides  # [Of, Nf, *outf]; the output positions must merge into one axis
        if not all(ds[2 + i] == ds[3 + i] * outf[i + 1] for i in range(n - 1)):
            dy = self.materialize(dy)
            ds = dy.view.strides
        b = Ref(dy.buffer, View((chunks, per, P, Of), (per * ds[1], ds[1], ds[1 + n], ds[0]), dy.view.offset), dy.dtype)
        acc = accumulator_dtype(out_type.dtype)
        partial_type = TensorType((chunks, KK, Of), acc if chunks > 1 else out_type.dtype)
        contract = Op("contract", (Value(TensorType(a.shape, xc.dtype)), Value(TensorType(b.shape, dy.dtype))), (Value(partial_type),), make_attrs({"spec": "snpk,snpo->sko"}))
        result = self._op_contract(contract, [a, b])
        if chunks > 1:
            result = self._reduce(result, "sum", (0,), TensorType((KK, Of), out_type.dtype))
        else:
            result = replace(result, view=View((KK, Of), (Of, 1), result.view.offset))
        # [(*kern, Cf), Of] read as this conv's output [Cf, Of, *kern].
        ks = contiguous_strides(kern)
        return self._swapped_layout(replace(result, view=View(tuple(out_type.shape), (Of, 1, *(k * Cf * Of for k in ks)), result.view.offset)))

    def _conv_reduce_windows(self, op: Op, inputs: list[Ref], spread, row, plane) -> Ref:
        """A weight gradient's conv read straight from the forward's channels-last grid.

        In the forward's terms (see _conv_reduce_im2col) dW[u, v, c, o] = sum over n, h, w of
        grid[n, h * s + u * d, w * s + v * d, c] * dY[n, h, w, o]: the windows are a view of
        the grid (the one the forward conv read, _padded_grid), its last kernel axis merged
        with the channels (KW * C contiguous), so nothing is copied, where im2col wrote KH *
        KW times the input. The reduction runs over n, h and w, with w the inner, pipelined
        axis: used when the forward's output rows are at least a 16-wide matrix tile (shorter
        rows would pad every k step; im2col merges h and w there). The batch is split into
        chunks reduced in parallel, then summed, as the output is a few dot tiles.
        """
        from olaf.graph.builder import make_attrs

        xc, wc = inputs  # xc [Cf, Nf, *in], wc [Of, Nf, *outf] in the forward's names
        stride, padding, dilation = (op.attr(k) for k in ("stride", "padding", "dilation"))
        n = len(stride)
        Cf, Nf = xc.shape[:2]
        Of = wc.shape[0]
        outf = tuple(wc.shape[2:])
        kern = tuple(op.outputs[0].type.shape[2:])
        out_type = op.outputs[0].type
        start = sum(max(-lo, 0) * r for (lo, _), r in zip(padding, row))
        tail = start + sum(((o - 1) * d + (k - 1) * st) * r for o, k, d, st, r in zip(outf, kern, dilation, stride, row)) + 1 - plane
        base = self._padded_grid(xc, (1, *range(2, 2 + n), 0), padding, max(tail, 0) * Cf)
        KK = kern[-1] * Cf
        tiles = math.prod(kern[:-1]) * -(-KK // 128) * -(-Of // 128)
        chunks = 1
        while tiles * chunks * 2 <= 4 * self.capability.roofline.compute_units and Nf % (chunks * 2) == 0:
            chunks *= 2
        per = Nf // chunks
        a_shape = (chunks, per, *outf, *kern[:-1], KK)
        a_strides = (per * plane * Cf, plane * Cf, *(d * r * Cf for d, r in zip(dilation, row)), *(st * r * Cf for st, r in zip(stride[:-1], row[:-1])), 1)
        a = Ref(base.buffer, View(a_shape, a_strides, start * Cf), xc.dtype)
        ds = wc.view.strides
        b = Ref(wc.buffer, View((chunks, per, *outf, Of), (per * ds[1], ds[1], *ds[2:], ds[0]), wc.view.offset), wc.dtype)
        if wc.view.pad is not None:
            b = replace(self.materialize(wc), dtype=wc.dtype)
            ds = b.view.strides
            b = Ref(b.buffer, View((chunks, per, *outf, Of), (per * ds[1], ds[1], *ds[2:], ds[0]), 0), wc.dtype)
        positions, kernels = "hwz"[:n], "uvy"[: n - 1]
        acc = accumulator_dtype(out_type.dtype)
        partial_type = TensorType((chunks, *kern[:-1], KK, Of), acc if chunks > 1 else out_type.dtype)
        spec = f"sn{positions}{kernels}k,sn{positions}o->s{kernels}ko"
        contract = Op("contract", (Value(TensorType(a.shape, xc.dtype)), Value(TensorType(b.shape, wc.dtype))), (Value(partial_type),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [a, b])
        summed = (*kern[:-1], KK, Of)
        if chunks > 1:
            result = self._reduce(result, "sum", (0,), TensorType(summed, out_type.dtype))
        else:
            result = replace(result, view=View(summed, contiguous_strides(summed), result.view.offset))
        # [*kern, Cf, Of] read as this conv's output [Cf, Of, *kern].
        ks = contiguous_strides(kern)
        return self._swapped_layout(replace(result, view=View(tuple(out_type.shape), (Of, 1, *(k * Cf * Of for k in ks)), result.view.offset)))

    def _conv_grid_reduce(self, op: Op, inputs: list[Ref]) -> Ref:
        """conv with a small output and a large kernel (a weight gradient: the "kernel" is the
        output gradient) as [N * out, K] x [K, O] with K = C * Hp * Wp, the whole padded grid.

        The input stays [N, C, *padded spatial]; the kernel is zero-padded to that grid (placed
        at its dilation). Output position p then reads the input at k + p * stride (elements),
        so the reduction is one contiguous axis of both operands; kernel positions outside the
        real kernel are zeros. Their work, (Hp * Wp) / (KH * KW) - 1, is the price.

        The output (O x 3 x 3 x C for a 3x3 weight gradient) is a handful of dot tiles, too
        few to fill the device, and K is long (N * Hp * Wp), so K is split into chunks that
        are reduced in parallel (an extra label of the product) and then summed.
        """
        from olaf.graph.builder import make_attrs

        x, w = inputs
        stride, padding, dilation, lhs_dilation = (op.attr(k) for k in ("stride", "padding", "dilation", "lhs_dilation"))
        n = len(stride)
        N, C = x.shape[:2]
        O = w.shape[0]
        out_type = op.outputs[0].type
        out_spatial = tuple(out_type.shape[2:])
        spread = [(s - 1) * ld + 1 + max(lo, 0) + max(hi, 0) for s, ld, (lo, hi) in zip(x.shape[2:], lhs_dilation, padding)]
        row = [math.prod(spread[i + 1 :]) for i in range(n)]
        plane = math.prod(spread)
        cols_bytes = C * math.prod(w.shape[2:]) * math.prod(out_spatial) * N * 2
        # In the forward's terms the grid wastes plane / (its output positions, here the kernel).
        if all(ld == 1 for ld in lhs_dilation) and cols_bytes <= IM2COL_BYTES and plane / math.prod(w.shape[2:]) > IM2COL_WASTE:
            return self._conv_reduce_im2col(op, inputs, spread, row, plane)
        # Otherwise the grid itself, through window views, where the output rows fill a matrix
        # tile. Measured against im2col where both apply, the windows lost with few channels
        # (MNIST's first weight gradient, 1 channel and a 5x5 kernel: 0.57 ms against 0.04),
        # so im2col keeps the cases above; against the channels-first grid below they won
        # (YOLO's step: 4.9 ms against 6.5).
        if all(ld == 1 for ld in lhs_dilation) and stride[-1] == 1 and w.shape[-1] >= 16:
            return self._conv_reduce_windows(op, inputs, spread, row, plane)
        K = C * plane
        tiles = math.prod(out_spatial) * -(-N // 128) * -(-O // 128)
        chunks = 1
        while tiles * chunks * 2 <= 4 * self.capability.roofline.compute_units and K // (chunks * 2) >= 1024:
            chunks *= 2
        chunk = -(-K // chunks)
        chunk = -(-chunk // 64) * 64  # whole k tiles per chunk
        start = sum(max(-lo, 0) * r for (lo, _), r in zip(padding, row))
        reach = start + sum((o - 1) * st * r for o, st, r in zip(out_spatial, stride, row)) + chunk * chunks - K
        base = self._spread_into(
            x,
            (N * K + reach,),
            (K, plane, *(ld * r for ld, r in zip(lhs_dilation, row))),
            sum(max(lo, 0) * r for (lo, _), r in zip(padding, row)),
            padding,
            lhs_dilation,
            x.dtype,
        )
        kpad = self._spread_into(w, (O, chunk * chunks), (chunk * chunks, plane, *(d * r for d, r in zip(dilation, row))), 0, [(0, 0)] * n, [1] * n, w.dtype)
        letters = "pqr"[:n]
        windows = Ref(base.buffer, View((N, *out_spatial, chunks, chunk), (K, *(st * r for st, r in zip(stride, row)), chunk, 1), start), x.dtype)
        kernel = Ref(kpad.buffer, View((O, chunks, chunk), (chunk * chunks, chunk, 1), 0), w.dtype)
        acc = accumulator_dtype(out_type.dtype)
        spec = f"n{letters}sk,osk->o{letters}sn"
        partial_type = TensorType((O, *out_spatial, chunks, N), acc if chunks > 1 else out_type.dtype)
        contract = Op("contract", (Value(TensorType(windows.shape, x.dtype)), Value(TensorType(kernel.shape, w.dtype))), (Value(partial_type),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [windows, kernel])
        if chunks > 1:
            result = self._reduce(result, "sum", (1 + n,), TensorType((O, *out_spatial, N), out_type.dtype))
        rs = contiguous_strides((O, *out_spatial, N))
        return self._swapped_layout(replace(result, view=View(tuple(out_type.shape), (rs[-1], rs[0], *rs[1:-1]), result.view.offset)))

    def _conv_windows(self, op: Op, inputs: list[Ref]) -> Ref:
        """conv as a contract over a window view of the input (implicit GEMM: nothing is copied
        per window).

        Output position (n, p...) and kernel offset (u...) address the input at
        p * stride + u * dilation, which is affine, so the windows are one strided view
        [N, *out, groups, *kernel, C/groups] with overlapping strides, and the weight a view
        [groups, O/groups, *kernel, C/groups]. Padding and input dilation are not affine masks
        (a position is in the padding depending on p * stride + u * dilation together), so they
        are made real first: the input is written, spread by its dilation, into a zeroed buffer
        with the padding around it (negative padding crops, by moving the windows' start).
        C/groups is the innermost reduced label, so the dot's k tile runs over channels rather
        than over a kernel width of 3.
        """
        from olaf.graph.builder import make_attrs

        x, w = inputs
        stride, padding, dilation, lhs_dilation, g = (op.attr(k) for k in ("stride", "padding", "dilation", "lhs_dilation", "groups"))
        n = len(stride)
        N, C = x.shape[:2]
        O, Cg = w.shape[:2]
        kernel = tuple(w.shape[2:])
        out_type = op.outputs[0].type
        out_spatial = tuple(out_type.shape[2:])
        if any(ld > 1 for ld in lhs_dilation) or any(lo > 0 or hi > 0 for lo, hi in padding):
            spread = tuple((s - 1) * ld + 1 + max(lo, 0) + max(hi, 0) for s, ld, (lo, hi) in zip(x.shape[2:], lhs_dilation, padding))
            padded_type = TensorType((N, C, *spread), x.dtype)
            base = self.elementwise("fill", padded_type, [], lambda b, tiles, starts, tile: b.full(tile, 0, x.dtype))
            ps = contiguous_strides(padded_type.shape)
            into = View(tuple(x.shape), (ps[0], ps[1], *(st * ld for st, ld in zip(ps[2:], lhs_dilation))), sum(max(lo, 0) * st for (lo, _), st in zip(padding, ps[2:])))
            self.elementwise("copy", TensorType(x.shape, x.dtype), [self.plain(x)], lambda b, t, s, ts: t[0], out=replace(base, view=into))
        else:
            base = x if x.view.is_contiguous else self.materialize(x)
        bs = base.view.strides
        start = base.view.offset + sum(max(-lo, 0) * st for (lo, _), st in zip(padding, bs[2:]))
        win_shape = (N, *out_spatial, g, *kernel, Cg)
        win_strides = (bs[0], *(st * s for st, s in zip(bs[2:], stride)), Cg * bs[1], *(st * d for st, d in zip(bs[2:], dilation)), bs[1])
        windows = Ref(base.buffer, View(win_shape, win_strides, start), x.dtype)
        w = w if w.view.is_contiguous else self.materialize(w)
        ws = w.view.strides
        weights = Ref(w.buffer, View((g, O // g, *kernel, Cg), (O // g * ws[0], ws[0], *ws[2:], ws[1]), w.view.offset), w.dtype)
        out_letters, kernel_letters = "pqr"[:n], "uvz"[:n]
        win_labels = "n" + out_letters + "g" + kernel_letters + "c"
        extent = dict(zip(win_labels, win_shape))
        stride_of = dict(zip(win_labels, win_strides))
        # The dot tiles only the last label of each role, so the role orders matter. Rows (M):
        # the largest of batch and output positions, a whole tile of rows instead of, for a
        # weight gradient (whose "batch" is the input channels and whose "output" is the 3x3
        # kernel), a tile of 3. Reduction (K): the contiguous label when it is long enough for
        # a k tile, so operand copies stay vectorizable; else the longest.
        m_order = sorted("n" + out_letters, key=lambda c: extent[c])
        k_candidates = kernel_letters + "c"
        unit = [c for c in k_candidates if stride_of[c] == 1 and extent[c] >= 16]
        k_inner = unit[0] if unit else max(k_candidates, key=lambda c: extent[c])
        k_order = "".join(c for c in k_candidates if c != k_inner) + k_inner
        a_term = "".join(m_order) + "g" + k_order
        a_perm = [win_labels.index(c) for c in a_term]
        windows = replace(windows, view=View(tuple(win_shape[i] for i in a_perm), tuple(win_strides[i] for i in a_perm), start))
        w_labels = "go" + kernel_letters + "c"
        b_term = "go" + k_order
        b_perm = [w_labels.index(c) for c in b_term]
        weights = replace(weights, view=View(tuple(weights.shape[i] for i in b_perm), tuple(weights.view.strides[i] for i in b_perm), weights.view.offset))
        out_term = "go" + "".join(m_order)
        size = {**extent, "o": O // g}
        grouped = TensorType(tuple(size[c] for c in out_term), out_type.dtype)
        spec = f"{a_term},{b_term}->{out_term}"
        contract = Op("contract", (Value(TensorType(windows.shape, x.dtype)), Value(TensorType(weights.shape, w.dtype))), (Value(grouped),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [windows, weights])
        # The result is [g, O/g, *rows in m_order]; g and O/g are adjacent, so they read as O.
        rs = contiguous_strides(grouped.shape)
        at = {c: rs[i] for i, c in enumerate(out_term)}
        strides = (at["n"], at["o"], *(at[c] for c in out_letters))
        return replace(result, view=View(tuple(out_type.shape), strides, result.view.offset))

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

    def _reversal(self, op: Op) -> bool:
        """gather(x, (K - 1) - iota(K), axis): x reversed along the axis (autodiff's flip of a
        convolution kernel; Graph IR has no reverse op)."""
        if op.attr("batch_dims") != 0 or op.inputs[1].type.rank != 1:
            return False
        size = op.inputs[0].type.shape[op.attr("axis")]
        sub = self.producers.get(op.inputs[1])
        if sub is None or sub.kind != "sub" or op.inputs[1].type.shape != (size,):
            return False
        top, ramp = (self.producers.get(v) for v in sub.inputs)
        if ramp is None or ramp.kind != "iota" or ramp.attr("axis") != 0:
            return False
        while top is not None and top.kind == "broadcast":
            top = self.producers.get(top.inputs[0])
        if top is None or top.kind != "constant":
            return False
        values = np.asarray(top.attr("value").array)
        return values.size == 1 and int(values.reshape(-1)[0]) == size - 1

    def _op_gather(self, op: Op, inputs: list[Ref]) -> Ref:
        if self._reversal(op) and isinstance(inputs[0].shape[op.attr("axis")], int):
            # A view with a negative stride: the copy that consumes it (a convolution's weight
            # going channels-last) does the reversal on the way, instead of a pass of its own.
            x, axis = self.plain(inputs[0]), op.attr("axis")
            v = x.view
            strides = list(v.strides)
            strides[axis] = -v.strides[axis]
            return replace(x, view=View(v.shape, tuple(strides), v.offset + (v.shape[axis] - 1) * v.strides[axis]))
        x, index = self.plain(inputs[0]), inputs[1]
        axis, batch_dims = op.attr("axis"), op.attr("batch_dims")
        out_type = op.outputs[0].type
        spread, n_index = self._index_over(index, out_type.shape, axis, batch_dims)
        axis_map = self._axis_map(x.view.rank, axis, n_index)
        out = self.new_buffer(out_type, BufferKind.temp)
        shape = out_type.shape
        tile = _elementwise_tile(shape, self.capability.tuning)
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
        tile = _elementwise_tile(shape, self.capability.tuning)
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
