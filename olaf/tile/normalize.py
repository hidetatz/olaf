"""Final Tile IR normalization before lowering."""

from __future__ import annotations

import math

from dataclasses import replace

from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.ir import Arg, BufferKind, Launch, Program, Task, View, contiguous_strides

_TWO_BYTE = (DType.bf16,)


def hoist_offsets(program: Program) -> Program:
    """Move every view offset out of the task templates into the launches.

    After scan unrolling, the copies of a block read their weights at different offsets into the
    stacked parameters, and with the offsets in the views each copy is its own template: 528
    unique tasks for GPT-2 small's training step instead of one set per block. The offset is just
    the start of the argument's memory, so it belongs with the buffer binding.

    Source maps go too: they serve fusion, which is over by now, and their starts name the
    block's place in the stack (the layer), which would split the templates again.
    """
    launches = []
    for launch in program.launches:
        task = launch.task
        offsets = tuple(launch.offset(i) + a.view.offset for i, a in enumerate(task.args))
        args = tuple(replace(a, view=replace(a.view, offset=0, source=None)) for a in task.args)
        hoisted = Task(task.name, task.grid, args, task.body, task.warps, task.origin)
        launches.append(Launch(hoisted, launch.buffers, offsets if any(o != 0 for o in offsets) else ()))
    return replace(program, launches=tuple(launches))


def align_rows(program: Program, multiple: int = 8) -> Program:
    """Give 2-byte temp buffers with a ragged last axis a row stride rounded up to `multiple`.

    GPT-2's logits gradient is bf16 with rows of 50257 elements: odd, so every row after the
    first starts at an address that is only 2-byte aligned, and no async copy (4, 8 or 16 bytes)
    can read it; the vocabulary products fall back to the general dot. Rounding the row stride
    to 8 elements (16 bytes) makes every row start aligned. Only the strides change, so the
    padding elements are never read (views mask by their logical shape).

    Every view of such a buffer is rebuilt from its source map; a buffer read through a view
    without one (after a reshape) is left as it is.
    """
    views: dict[str, list[View]] = {}
    for launch in program.launches:
        for name, arg in zip(launch.buffers, launch.task.args):
            views.setdefault(name, []).append(arg.view)
    strides: dict[str, tuple[Dim, ...]] = {}
    for b in program.buffers:
        shape = b.type.shape
        if b.kind is not BufferKind.temp or b.type.dtype not in _TWO_BYTE or not shape:
            continue
        last = shape[-1]
        if not isinstance(last, int) or last % multiple == 0:
            continue
        if any(v.source is None for v in views.get(b.name, [])):
            continue
        strides[b.name] = contiguous_strides((*shape[:-1], -(-last // multiple) * multiple))
    if not strides:
        return program
    launches = []
    for launch in program.launches:
        args = []
        for name, arg in zip(launch.buffers, launch.task.args):
            args.append(replace(arg, view=_restrided(arg.view, strides[name])) if name in strides else arg)
        task = launch.task
        launches.append(replace(launch, task=Task(task.name, task.grid, tuple(args), task.body, task.warps, task.origin)))
    buffers = tuple(replace(b, strides=strides[b.name]) if b.name in strides else b for b in program.buffers)
    return replace(program, buffers=buffers, launches=tuple(launches))


def _restrided(view: View, physical: tuple[Dim, ...]) -> View:
    """The same elements of a buffer laid out with `physical` strides."""
    view_strides: list[Dim] = [0] * view.rank
    offset: Dim = 0
    for buffer_axis, (axis, start) in enumerate(view.source):
        if axis is not None:
            view_strides[axis] = view_strides[axis] + physical[buffer_axis]
            if view.pad is not None:
                # source starts include -low for padded axes; the address subtracts low itself.
                start = start + view.pad[axis][0]
        offset = offset + start * physical[buffer_axis]
    return replace(view, strides=tuple(view_strides), offset=offset)


def eliminate_copies(program: Program) -> Program:
    """Remove copy tasks by letting the copied value's writer store into the copy's target.

    write_state and program results materialize their value with a copy (from_graph), and the
    value usually has other readers too, so fusion cannot inline it away. When the copied buffer
    is a temp written once, through its full view, the writer can write the target instead and
    every other reader reads the target. Conditions: the copy reads and writes full views of
    equally typed contiguous buffers, and nothing touches the target before the copy.
    """
    from olaf.tile.ir import full_view

    buffers = {b.name: b for b in program.buffers}
    launches = list(program.launches)
    rename: dict[str, str] = {}
    removed: set[int] = set()
    touches: dict[str, list[int]] = {}
    writers: dict[str, list[int]] = {}
    for i, launch in enumerate(launches):
        for name, arg in zip(launch.buffers, launch.task.args):
            touches.setdefault(name, []).append(i)
            if arg.writes:
                writers.setdefault(name, []).append(i)
    for i, launch in enumerate(launches):
        if launch.task.origin != ("copy",) or launch.offsets or len(launch.buffers) != 2:
            continue
        (src, src_arg), (dst, dst_arg) = zip(launch.buffers, launch.task.args)
        if src_arg.writes or not dst_arg.writes or not _pure_copy(launch.task):
            continue
        a, b = buffers[src], buffers[dst]
        if a.kind is not BufferKind.temp or a.type != b.type or a.strides is not None or b.strides is not None:
            continue
        if src_arg.view != full_view(a.type.shape) or dst_arg.view != full_view(b.type.shape):
            continue
        w = writers.get(src, [])
        if len(w) != 1 or w[0] >= i or writers.get(dst) != [i] or min(touches[dst]) != i:
            continue
        writer = launches[w[0]]
        stores = [arg for name, arg in zip(writer.buffers, writer.task.args) if name == src]
        if len(stores) != 1 or not stores[0].writes or stores[0].view != full_view(a.type.shape):
            continue
        rename[src] = dst
        removed.add(i)
    if not rename:
        return program
    out = []
    for i, launch in enumerate(launches):
        if i in removed:
            continue
        out.append(replace(launch, buffers=tuple(rename.get(n, n) for n in launch.buffers)))
    kept = tuple(b for b in program.buffers if b.name not in rename)
    return replace(program, buffers=kept, launches=tuple(out), outputs=tuple(rename.get(n, n) for n in program.outputs))


def eliminate_pad_copies(program: Program) -> Program:
    """Let a value's writer store straight into the padded (and permuted) copy of it.

    A convolution reads its input from a zero-padded channels-last grid (tile.from_graph,
    _padded_grid), and copying the input there was a pass over every activation, forward and
    backward (ResNet-18: about 30 such copies). When the copied buffer is a temp written once,
    through its full view, and the copy reads all of it through a view with a source map (a
    permutation and padding, nothing cropped or repeated), the writer can store each element
    where the copy would have put it; every other reader reads the grid through its view
    restrided the same way; only the border is left to write, by fill tasks over its slabs.
    """
    from olaf.tile.ir import full_view

    buffers = {b.name: b for b in program.buffers}
    launches = list(program.launches)
    touches: dict[str, list[int]] = {}
    writers: dict[str, list[int]] = {}
    for i, launch in enumerate(launches):
        for name, arg in zip(launch.buffers, launch.task.args):
            touches.setdefault(name, []).append(i)
            if arg.writes:
                writers.setdefault(name, []).append(i)
    replaced: dict[int, list[Launch]] = {}
    moved: dict[str, tuple] = {}  # src -> (grid, the copy's write view, _interior_map, src shape)
    for i, launch in enumerate(launches):
        if launch.task.origin != ("copy",) or launch.offsets or len(launch.buffers) != 2 or not _pure_copy(launch.task):
            continue
        (src, read), (dst, write) = zip(launch.buffers, launch.task.args)
        if read.writes or not write.writes or src in moved or dst in moved or any(m[0] in (src, dst) for m in moved.values()):
            continue
        a, b = buffers[src], buffers[dst]
        if a.kind is not BufferKind.temp or b.kind is not BufferKind.temp or a.strides is not None or b.strides is not None or a.type.dtype != b.type.dtype:
            continue
        w = writers.get(src, [])
        if len(w) != 1 or w[0] >= i or writers.get(dst) != [i] or min(touches[dst]) != i:
            continue
        writer = launches[w[0]]
        if [arg for name, arg in zip(writer.buffers, writer.task.args) if name == src] != [Arg(full_view(a.type.shape), a.type.dtype, True)]:
            continue
        mapped = _interior_map(read.view, a.type.shape)
        if mapped is None:
            continue
        if write.view.source is None or {c for c, _ in write.view.source if c is not None} != set(range(write.view.rank)):
            continue
        # The writer's tiles run along the buffer's last axis: it must stay contiguous in the
        # grid, or every store is strided (a channels-last ReLU into a channels-first pooling
        # grid: the copy, which walks in the grid's order, does that transpose better).
        if _moved_view(full_view(a.type.shape), write.view, mapped, a.type.shape).strides[-1] != 1:
            continue
        views = [arg.view for j in touches[src] if j != i for name, arg in zip(launches[j].buffers, launches[j].task.args) if name == src]
        if any(v.source is None and _axis_steps(v, a.type.shape) is None for v in views):
            continue  # a view through a merging reshape: the grid's gaps break it
        moved[src] = (dst, write.view, mapped, a.type.shape)
        replaced[i] = _border_fills(write.view, read.view, dst, b.type.dtype)
    if not moved:
        return program
    out = []
    for i, launch in enumerate(launches):
        if i in replaced:
            out.extend(replaced[i])
            continue
        if not any(name in moved for name in launch.buffers):
            out.append(launch)
            continue
        args, names = [], []
        for name, arg in zip(launch.buffers, launch.task.args):
            if name in moved:
                dst, into, mapped, shape = moved[name]
                arg = replace(arg, view=_moved_view(arg.view, into, mapped, shape))
                name = dst
            args.append(arg)
            names.append(name)
        task = launch.task
        out.append(replace(launch, task=Task(task.name, task.grid, tuple(args), task.body, task.warps, task.origin), buffers=tuple(names)))
    kept = tuple(b for b in program.buffers if b.name not in moved)
    return replace(program, buffers=kept, launches=tuple(out))


def _axis_steps(view: View, shape):
    """For a view without a source map of a contiguous `shape` buffer: per view axis (buffer
    axis, step) and per buffer axis the start, when every view axis moves along one buffer
    axis and stays inside it (a split of an axis, as a batch cut into chunks, qualifies; a
    merge of two does not). None otherwise."""
    from olaf.tile.ir import contiguous_strides

    if view.pad is not None or not all(isinstance(d, int) for d in (*view.shape, *view.strides, view.offset, *shape)):
        return None
    cs = contiguous_strides(shape)
    rest, starts = view.offset, []
    for st in cs:
        starts.append(rest // st)
        rest %= st
    steps, reach = [], list(starts)
    for d, st in zip(view.shape, view.strides):
        if d == 1 or st == 0:
            steps.append((None, 0))
            continue
        p = next((p for p in range(len(shape)) if st > 0 and st % cs[p] == 0 and st // cs[p] < shape[p]), None)
        if p is None:
            return None
        steps.append((p, st // cs[p]))
        reach[p] += (d - 1) * (st // cs[p])
    if any(r >= n for r, n in zip(reach, shape)):
        return None
    return steps, starts


def _moved_view(view: View, into: View, mapped, shape) -> View:
    """`view` of the moved `shape` buffer, as a view of the grid that `into` (the copy's full
    write view, with a source map) addresses; mapped is _interior_map of the copy's read."""
    placed = {c: (q, q_start) for q, (c, q_start) in enumerate(into.source) if c is not None}
    if view.source is None:
        steps, starts = _axis_steps(view, shape)
        offset: Dim = into.offset + sum(((o + lo) * into.strides[c] for o, (c, lo) in zip(starts, mapped) if c is not None), 0)
        strides = tuple(0 if p is None or mapped[p][0] is None else k * into.strides[mapped[p][0]] for p, k in steps)
        return View(view.shape, strides, offset)
    strides: list[Dim] = [0] * view.rank
    offset = into.offset
    source: list = [(None, 0)] * len(into.source)
    for p, (axis, start) in enumerate(view.source):
        c, lo = mapped[p]
        if c is None:
            continue
        (q, q_start), step = placed[c], into.strides[c]
        if axis is not None:
            strides[axis] = strides[axis] + step
        # Padded axes: the address subtracts low itself, the source start includes -low.
        at = start + (view.pad[axis][0] if view.pad is not None and axis is not None else 0)
        offset = offset + (at + lo) * step
        source[q] = (axis, start + lo + q_start)
    return View(view.shape, tuple(strides), offset, view.pad, view.fill, tuple(source))


def _interior_map(view: View, shape) -> list[tuple[int | None, int]] | None:
    """Per buffer axis of a `shape` buffer, (view axis, its low padding), when `view` shows the
    whole buffer exactly once inside its padding (nothing cropped, broadcast or strided)."""
    if view.source is None or len(view.source) != len(shape):
        return None
    pad = view.pad or tuple((0, 0) for _ in view.shape)
    used = set()
    out = []
    for p, (c, start) in enumerate(view.source):
        if c is None:
            if shape[p] != 1:
                return None
            out.append((None, 0))
            continue
        lo, hi = pad[c]
        if c in used or start != -lo or view.shape[c] - lo - hi != shape[p] or not isinstance(view.shape[c], int):
            return None
        used.add(c)
        out.append((c, lo))
    if any(view.shape[c] != 1 for c in range(view.rank) if c not in used):
        return None
    return out


def _border_fills(into: View, read: View, buffer: str, dtype: DType) -> list[Launch]:
    """One task writing read.fill to the padded slabs of `into` (one per padded side and axis;
    the corners are written twice, with the same value).

    One launch for all of a grid's slabs: a task per slab was five launches of a few
    microseconds' work per padded grid (ResNet-18's step had 125 of them). Program p writes
    tile p of every slab; past a slab's last tile its leading coordinate leaves the slab, and
    the store's bounds drop it."""
    from olaf.tile.builder import INDEX, TileBuilder
    from olaf.tile.ir import TileType, TValue

    slabs = []
    for axis, (lo, hi) in enumerate(read.pad or ()):
        for start, size in ((0, lo), (into.shape[axis] - hi, hi)):
            if size == 0:
                continue
            shape = tuple(size if a == axis else d for a, d in enumerate(into.shape))
            view = View(shape, into.strides, into.offset + start * into.strides[axis])
            tile = [1] * len(shape)
            budget = 1024
            for a in range(len(shape) - 1, -1, -1):
                tile[a] = min(budget, 1 << max(shape[a] - 1, 0).bit_length())
                budget //= tile[a]
                if budget < 2:
                    break
            slabs.append((view, tuple(tile), tuple(-(-d // t) for d, t in zip(shape, tile))))
    if not slabs:
        return []
    count = max(math.prod(grid) for _, _, grid in slabs)
    b = TileBuilder()
    pid = TValue(TileType((), INDEX))
    args = []
    for k, (view, tile, grid) in enumerate(slabs):
        starts = [None] * len(grid)
        rest = pid
        for a in range(len(grid) - 1, -1, -1):
            # The leading coordinate is not reduced: past the slab's tiles it runs past its end.
            at = rest if a == 0 else b.op("mod", rest, b.index(grid[a]))
            if a:
                rest = b.op("floordiv", rest, b.index(grid[a]))
            starts[a] = at if tile[a] == 1 else b.op("mul", at, b.index(tile[a]))
        b.emit("store", (b.full(tile, read.fill, dtype), *starts), {"arg": k})
        args.append(Arg(view, dtype, True))
    return [Launch(Task("fill", (count,), tuple(args), b.finish([pid]), 4, ("fill",)), (buffer,) * len(slabs))]


def _pure_copy(task: Task) -> bool:
    """The body stores what it loads, unchanged (fusion may have inlined a producer into a copy)."""
    stores = [op for op in task.body.ops if op.kind == "store"]
    loads = {op.outputs[0]: op for op in task.body.ops if op.kind == "load"}
    return len(stores) == 1 and len(loads) == 1 and stores[0].inputs[0] in loads and loads[stores[0].inputs[0]].attr("arg") == 0


def eliminate_concats(program: Program) -> Program:
    """Let the producers of a concat's inputs store into their slices of its output.

    Unrolled scans stack each layer's weight gradients with concat (passes.unroll): 12 tensors
    per weight kind, written once by their producers and then copied into the stack; the qkv
    gradient joins dq, dk and dv along its last axis. When input j is a contiguous buffer that
    only the concat reads, and its slice of the output can be shown in j's shape as a strided
    view (an offset block, or rows of a block when the concat axis is inner), its producer
    stores there directly and the concat disappears.
    """
    from olaf.tile.ir import full_view

    buffers = {b.name: b for b in program.buffers}
    readers: dict[str, list[int]] = {}
    writers: dict[str, list[int]] = {}
    for i, launch in enumerate(program.launches):
        for name, arg in zip(launch.buffers, launch.task.args):
            (writers if arg.writes else readers).setdefault(name, []).append(i)
    launches = list(program.launches)
    removed: set[int] = set()
    for i, launch in enumerate(program.launches):
        casts = _select_casts(launch.task) if launch.task.origin == ("concat",) and not launch.offsets else None
        if casts is None:
            continue
        outs = [(n, a) for n, a in zip(launch.buffers, launch.task.args) if a.writes]
        if len(outs) != 1:
            continue
        out, out_arg = outs[0]
        target = buffers[out]
        if out_arg.view != full_view(target.type.shape) or writers.get(out) != [i] or target.strides is not None:
            continue
        if any(r < i for r in readers.get(out, [])):
            continue
        plan = _concat_plan(program, launch, target, buffers, readers, writers, i)
        if plan is None:
            continue
        cast_of = {launch.buffers[k]: dtype for k, dtype in casts.items()}
        if any((cast_of.get(name) or buffers[name].type.dtype) != target.type.dtype for _, name, _ in plan):
            continue  # an input the concat would cast to another type than the output's
        for j, name, view in plan:
            producer = launches[j]
            args = tuple(replace(a, view=view) if n == name and a.writes else a for n, a in zip(producer.buffers, producer.task.args))
            task = producer.task
            task = Task(task.name, task.grid, args, task.body, task.warps, task.origin)
            if name in cast_of and cast_of[name] != buffers[name].type.dtype:
                # The producer stores the cast value itself (a bf16 weight gradient rounded,
                # then widened to the f32 stack, as the concat did).
                task = _store_cast(task, next(k for k, (n, a) in enumerate(zip(producer.buffers, args)) if n == name and a.writes), cast_of[name])
            launches[j] = Launch(task, tuple(out if n == name else n for n in producer.buffers))
        removed.add(i)
    if not removed:
        return program
    kept = [launch for i, launch in enumerate(launches) if i not in removed]
    used = {n for launch in kept for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    return replace(program, launches=tuple(kept), buffers=tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp))


def _pure_select(task: Task) -> bool:
    """A concat body: loads selected by iota bounds, nothing computed from the values."""
    return {op.kind for op in task.body.ops} <= {"load", "store", "full", "mul", "add", "iota", "cmp", "where", "broadcast", "dim"} and not any(
        op.kind in ("mul", "add") and op.outputs[0].type.dtype.is_float for op in task.body.ops
    )


def _select_casts(task: Task) -> dict[int, DType] | None:
    """For a concat body that is a pure selection except for casts applied right to its loads
    (fusion inlines an input's cast: an unrolled scan stacking bf16 weight gradients as f32):
    {arg: the dtype its load is cast to}. None for any other body."""
    loads = {op.outputs[0]: op.attr("arg") for op in task.body.ops if op.kind == "load"}
    casts = {}
    for op in task.body.ops:
        if op.kind == "cast":
            if op.inputs[0] not in loads:
                return None
            casts[loads[op.inputs[0]]] = op.attr("dtype")
    rest = Task(task.name, task.grid, task.args, replace(task.body, ops=tuple(op for op in task.body.ops if op.kind != "cast")), task.warps, task.origin)
    return casts if _pure_select(rest) else None


def _store_cast(task: Task, arg: int, dtype: DType) -> Task:
    """task with its stores to `arg` cast to dtype first (the cast a concat applied to it)."""
    from olaf.tile.ir import TileType, TOp, TValue

    ops = []
    for op in task.body.ops:
        if op.kind == "store" and op.attr("arg") == arg:
            value = op.inputs[0]
            cast = TValue(TileType(value.type.shape, dtype))
            ops.append(TOp("cast", (value,), (cast,), (("dtype", dtype),)))
            op = TOp(op.kind, (cast, *op.inputs[1:]), op.outputs, op.attrs, op.regions)
        ops.append(op)
    args = tuple(replace(a, dtype=dtype) if k == arg else a for k, a in enumerate(task.args))
    return Task(task.name, task.grid, args, replace(task.body, ops=tuple(ops)), task.warps, task.origin)


def _concat_plan(program: Program, launch: Launch, target, buffers, readers, writers, index: int):
    """[(producer launch index, its store view into the concat output)] for every input, or None."""
    from olaf.tile.from_graph import _reshaped_strides
    from olaf.tile.ir import full_view

    shape = target.type.shape
    plan = []
    for name, arg in zip(launch.buffers, launch.task.args):
        if arg.writes:
            continue
        view = arg.view
        if all(s == 0 for s in view.strides):
            continue  # a constant bound
        padded = [k for k, (lo, hi) in enumerate(view.pad or ()) if lo != 0 or hi != 0]
        if view.pad is None or len(padded) != 1 or view.offset != 0:
            return None
        axis = padded[0]
        lo, hi = view.pad[axis]
        logical = tuple(shape[k] - lo - hi if k == axis else d for k, d in enumerate(shape))
        source = buffers[name]
        if source.strides is not None:
            return None
        out_strides = contiguous_strides(shape)
        if not isinstance(lo, int) or not isinstance(out_strides[axis], int):
            return None
        block = View(logical, out_strides, lo * out_strides[axis])
        if view.strides == contiguous_strides(logical):
            strides = _reshaped_strides(block, source.type.shape)  # read as a reshape of the input
        else:
            strides = _permuted_strides(view, logical, source.type.shape, out_strides)  # read permuted
        if strides is None:
            return None
        w = writers.get(name, [])
        if len(w) != 1 or readers.get(name) != [index] or source.kind is not BufferKind.temp or w[0] > index:
            return None
        producer = program.launches[w[0]]
        if producer.offsets:
            return None
        stores = [a for n, a in zip(producer.buffers, producer.task.args) if n == name]
        has_store = any(op.kind == "store" for op in _ops(producer.task.body))
        # Atomics into this input would have to see the other inputs' slices as zeros.
        has_atomic = any(op.kind.startswith("atomic") and producer.buffers[op.attr("arg")] == name for op in _ops(producer.task.body))
        if len(stores) != 1 or not stores[0].writes or not has_store or has_atomic:
            return None
        composed = _compose(stores[0].view, source.type.shape, strides, block.offset)
        if composed is None:
            return None
        plan.append((w[0], name, composed))
    return plan or None


def _permuted_strides(read: View, logical, shape, out_strides) -> tuple | None:
    """Strides placing a contiguous buffer of `shape` where `read` (a permutation of its axes,
    `logical` its unpadded extent) shows it in the concat's coordinates; None when `read` is
    not a permutation."""
    contiguous = contiguous_strides(shape)
    if read.offset != 0 or not all(isinstance(d, int) for d in (*read.strides, *logical, *contiguous, *out_strides)):
        return None
    strides = []
    for k, (size, step) in enumerate(zip(shape, contiguous)):
        if size == 1:
            strides.append(0)
            continue
        axes = [i for i, (d, st) in enumerate(zip(logical, read.strides)) if st == step and d == size]
        if len(axes) != 1:
            return None
        strides.append(out_strides[axes[0]])
    if sorted(d for d in logical if d != 1) != sorted(d for d in shape if d != 1):
        return None
    return tuple(strides)


def _compose(view: View, shape, strides, offset: int) -> View | None:
    """`view` (into a contiguous buffer of `shape`) re-aimed at the same elements laid out with
    `strides` from `offset`. Each axis of the view must step along one buffer axis (a permuted
    or sliced store, as attention's dK and dV are written head-major into a [batch, seq, head,
    dim] buffer); anything else is None."""
    contiguous = contiguous_strides(shape)
    if view.pad is not None or not all(isinstance(d, int) for d in (*view.shape, *view.strides, view.offset, *contiguous)):
        return None
    new = []
    for size, step in zip(view.shape, view.strides):
        if step == 0 or size == 1:
            new.append(0)
            continue
        axes = [k for k, c in enumerate(contiguous) if c == step and shape[k] > 1]
        if len(axes) != 1 or size > shape[axes[0]]:
            return None
        new.append(strides[axes[0]])
    rest, start = view.offset, offset
    for k, c in enumerate(contiguous):
        index, rest = divmod(rest, c) if shape[k] > 1 else (0, rest)
        start += index * strides[k]
    if rest:
        return None
    return View(view.shape, tuple(new), start)


def _ops(body):
    for op in body.ops:
        yield op
        for r in op.regions:
            yield from _ops(r)
