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
    last = _block(shape[-1], block)
    rows = 1
    if len(shape) >= 2:
        rows = _block(shape[-2], max(1, elements // last))
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
            blocks *= -(-shape[a] // (_block(shape[a], self.capability.tuning.elementwise_block) if a == inner else 1))
        if blocks >= 2 * units:
            return None
        axis = max(axes, key=lambda a: shape[a])
        parts = 1
        while blocks * parts * 2 <= self.capability.tuning.split_blocks_per_unit * units and shape[axis] % (parts * 2) == 0 and shape[axis] // (parts * 2) >= 64:
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
        bm, bn, bk = self.capability.tuning.matrix_dot_block if matrix else self.capability.tuning.dot_block
        stages, warps = (self.capability.tuning.matrix_dot_stages, self.capability.tuning.matrix_dot_warps) if matrix else (1, 4)
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

        flatten = len(k_labels) > 1 and k_inner is not None and all(isinstance(size[c], int) for c in k_labels)
        acc0 = b.full((em, en), 0, acc_dtype)
        acc = flat_loop(b, dict(out_starts), acc0) if flatten else loop(b, 0, dict(out_starts), acc0)
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

    def _op_conv(self, op: Op, inputs: list[Ref]) -> Ref:
        """conv as one dense matrix product over the padded grid (see _conv_grid_rows and
        _conv_grid_reduce), or, where neither fits (groups), over window views (_conv_windows)."""
        g = op.attr("groups")
        out_positions = math.prod(op.outputs[0].type.shape[2:])
        kernel_positions = math.prod(inputs[1].shape[2:])
        if g == 1 and out_positions < kernel_positions:
            return self._conv_grid_reduce(op, inputs)
        if g == 1:
            return self._conv_grid_rows(op, inputs)
        return self._conv_windows(op, inputs)

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
        single elements). The result is [N, O, plane], the output's channels-first layout with
        a row stride of Wp, so nothing is copied after. The positions outside the valid
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
            base = self.new_buffer(TensorType(((N * plane + reach) * C,), x.dtype), BufferKind.temp)
            src = self.plain(x)
            order = (0, *range(2, 2 + n), 1)
            padded_shape = (N, *spread, C)
            pad = ((0, 0), *((max(lo, 0), max(hi, 0)) for lo, hi in padding), (0, 0))
            source = replace(src, view=View(padded_shape, tuple(src.view.strides[i] for i in order), src.view.offset, pad, 0.0))
            into = View(padded_shape, contiguous_strides(padded_shape), 0)
            self.elementwise("copy", TensorType(padded_shape, x.dtype), [source], lambda b, t, s, ts: t[0], out=replace(base, view=into))
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
        letters = "uvz"[:n]
        merged = dilation[-1] == 1  # the last kernel axis and the channels are one contiguous run
        shape, strides, labels = [N, plane], [plane * C, C], "nm"
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
        k_order = labels[2:]
        spec = f"{labels},o{k_order}->nom"
        rows_type = TensorType((N, O, plane), out_type.dtype)
        contract = Op("contract", (Value(TensorType(windows.shape, x.dtype)), Value(TensorType(weights.shape, w.dtype))), (Value(rows_type),), make_attrs({"spec": spec}))
        result = self._op_contract(contract, [windows, weights])
        view = View(tuple(out_type.shape), (O * plane, plane, *(st * r for st, r in zip(stride, row))), result.view.offset)
        return replace(result, view=view)

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
        return replace(result, view=View(tuple(out_type.shape), (rs[-1], rs[0], *rs[1:-1]), result.view.offset))

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

    def _op_gather(self, op: Op, inputs: list[Ref]) -> Ref:
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
