"""Row fusion on Tile IR: row reductions and the pointwise task that uses them become one task.

Layer normalization decomposes into a mean (a row sum), a variance (a row sum that reads the
mean) and the normalization (pointwise, reading both): three passes over the same rows, and its
backward is the same shape. When a row fits in one tile, one task per row can do it all: replay
each reduction's per-element value on the row tile and reduce it in registers, then replay the
pointwise task with those row values in place of its loads of the statistics.

Matched: a pointwise task P whose last axis fits one tile (ROW_LIMIT), reading row statistics
(rank-1 buffers broadcast along the last axis), each written once by a reduction over the last
axis whose loop covers a whole row in one step; recursively, the statistics those reductions
read. Statistics that other tasks read are stored by the new task too; a reduction is kept when
one of those readers runs before P.
"""

from __future__ import annotations

from dataclasses import replace

from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.builder import INDEX, TileBuilder
from olaf.tile.fuse import _pointwise
from olaf.tile.ir import Arg, Body, BufferKind, Launch, Program, Task, TileType, TOp, TValue, View, full_view

# Rows up to this long fuse (Llama's hidden sizes go to 8192). A row tile is spread over
# `_warps(width)` warps so each thread holds at most 8 elements of each row value.
ROW_LIMIT = 8192


def _warps(width: int) -> int:
    return max(4, min(32, width // 256))


def fuse_rows(program: Program) -> Program:
    while True:
        result = _fuse_one(program)
        if result is None:
            return program
        program = result


def _fuse_one(program: Program) -> Program | None:
    buffers = {b.name: b for b in program.buffers}
    writers: dict[str, list[int]] = {}
    readers: dict[str, list[int]] = {}
    for i, launch in enumerate(program.launches):
        for name, arg in zip(launch.buffers, launch.task.args):
            (writers if arg.writes else readers).setdefault(name, []).append(i)
    for i, launch in enumerate(program.launches):
        if launch.offsets:
            continue
        info = _pointwise(launch, buffers)
        if info is None:
            continue
        out = launch.buffers[info.store.attr("arg")]
        shape = buffers[out].type.shape
        if len(shape) < 2 or not isinstance(shape[-1], int) or shape[-1] > ROW_LIMIT:
            continue
        needed = _statistics(program, buffers, writers, launch, shape)
        if not needed:
            continue
        return _build(program, buffers, readers, i, info, needed, shape)
    return None


def _row_view(view: View, rank: int) -> bool:
    """A rank-1 statistic read broadcast along the last axis of a rank-`rank` tile."""
    if view.pad is None and view.source is not None and len(view.source) == rank - 1:
        # A size-1 axis may map to no view axis: its coordinate is 0 either way.
        return all(entry == (j, 0) or (entry == (None, 0) and view.shape[j] == 1) for j, entry in enumerate(view.source))
    return False


def _reduction(program: Program, index: int, shape) -> dict | None:
    """The parts of a row reduction task (see module docstring), or None."""
    launch = program.launches[index]
    origin = launch.task.origin
    rank = len(shape)
    if launch.offsets or not origin or origin[0] != "reduce" or origin[2] != (rank - 1,):
        return None
    body = launch.task.body
    loops = [op for op in body.ops if op.kind == "for"]
    stores = [op for op in body.ops if op.kind == "store"]
    if len(loops) != 1 or len(stores) != 1 or any(op.kind == "for" for op in loops[0].regions[0].ops):
        return None
    region = loops[0].regions[0]
    loads = [op for op in region.ops if op.kind == "load"]
    if not loads or any(op.attr("shape")[-1] < shape[-1] for op in loads):
        return None  # one step must cover the row
    carry = region.params[1]
    combine = [op for op in region.ops if carry in op.inputs]
    if len(combine) != 1 or combine[0].kind not in ("add", "max"):
        return None
    value = next(v for v in combine[0].inputs if v is not carry)
    producers = {v: op for op in body.ops for v in op.outputs}
    stored = stores[0].inputs[0]
    cast = producers[stored].attr("dtype") if producers[stored].kind == "cast" else None
    return {"index": index, "launch": launch, "region": region, "value": value, "fn": origin[1], "cast": cast, "store": stores[0]}


def _statistics(program, buffers, writers, launch: Launch, shape) -> dict:
    """buffer -> reduction parts, for the row statistics P reads (recursively) that row reductions
    produce; statistics written otherwise stay ordinary inputs."""
    rank = len(shape)
    found: dict[str, dict] = {}
    pending = [launch]
    while pending:
        task_launch = pending.pop()
        for name, arg in zip(task_launch.buffers, task_launch.task.args):
            if arg.writes or name in found or buffers[name].type.rank != rank - 1 or not _row_view(arg.view, rank):
                continue
            if buffers[name].type.shape != tuple(shape[:-1]):
                continue
            w = writers.get(name, [])
            parts = _reduction(program, w[0], shape) if len(w) == 1 and buffers[name].kind is BufferKind.temp else None
            if parts is None:
                continue  # computed elsewhere (another fused task, an input): read it as it is
            found[name] = parts
            pending.append(parts["launch"])
    return found


def _order(needed: dict) -> list[str]:
    """Statistics in an order where each comes after those its reduction reads."""
    order: list[str] = []

    def visit(name: str) -> None:
        if name in order:
            return
        for dep, arg in zip(needed[name]["launch"].buffers, needed[name]["launch"].task.args):
            if dep in needed and dep != name and not arg.writes:
                visit(dep)
        order.append(name)

    for name in needed:
        visit(name)
    return order


def _build(program: Program, buffers, readers, p_index: int, info, needed: dict, shape) -> Program:
    rank = len(shape)
    width = 1 << max(shape[-1] - 1, 0).bit_length()
    tile = tuple([1] * (rank - 1) + [width])
    row_tile = tuple([1] * (rank - 1))
    b = TileBuilder(cse=True)  # the replayed reductions and P load the same row tiles
    args: list[tuple[str, Arg]] = []

    def arg_index(name: str, arg: Arg) -> int:
        key = (name, arg)
        if key not in args:
            args.append(key)
        return args.index(key)

    pids = [TValue(TileType((), INDEX)) for _ in range(rank - 1)]
    starts = [*pids, b.index(0)]
    rows: dict[str, TValue] = {}  # statistic -> (1, ..., 1) row value

    def spread(name: str) -> TValue:
        return b.op("broadcast", rows[name], shape=tile, dims=tuple(range(rank - 1)))

    for name in _order(needed):
        parts = needed[name]
        launch, region = parts["launch"], parts["region"]
        body_params = launch.task.body.params
        mapping: dict[TValue, TValue] = dict(zip(body_params, pids))
        mapping[region.params[0]] = b.index(0)
        for op in launch.task.body.ops:
            if op.kind in ("for", "store", "reduce") or op.kind == "cast" and op.outputs[0] is parts["store"].inputs[0]:
                continue
            if op.kind == "full" and op.outputs[0].type.shape != ():
                continue  # the accumulator's initial value
            mapping[op.outputs[0]] = b.op(op.kind, *[mapping.get(v, v) for v in op.inputs], **dict(op.attrs))
        for op in region.ops:
            if op.kind == "store" or region.params[1] in op.inputs:
                continue
            ins = [mapping.get(v, v) for v in op.inputs]
            if op.kind == "load":
                arg = launch.task.args[op.attr("arg")]
                source = launch.buffers[op.attr("arg")]
                if source in rows and _row_view(arg.view, rank):
                    mapping[op.outputs[0]] = spread(source)
                    continue
                out = b.op("load", *ins, arg=arg_index(source, arg), shape=op.attr("shape"), dtype=op.attr("dtype"), other=op.attr("other"))
            else:
                out = b.op(op.kind, *ins, **dict(op.attrs))
            mapping[op.outputs[0]] = out
        value = mapping[parts["value"]]
        row = b.op("reduce", value, fn=parts["fn"], axes=(rank - 1,))
        if parts["cast"] is not None:
            row = b.op("cast", row, dtype=parts["cast"])
        rows[name] = row

    # The pointwise task, on the row tile, with its statistics from registers.
    p_launch = program.launches[p_index]
    task = p_launch.task
    p_tile = info.store.inputs[0].type.shape
    mapped: dict[TValue, TValue] = {}
    for op in task.body.ops:
        if op.kind == "store" or (op.outputs and set(op.outputs) <= info.coordinate):
            continue
        ins = [mapped.get(v, v) for v in op.inputs]
        if op.kind == "load":
            a = op.attr("arg")
            name = p_launch.buffers[a]
            if name in rows and _row_view(task.args[a].view, rank):
                out = spread(name)
            else:
                out = b.op("load", *starts, arg=arg_index(name, task.args[a]), shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
        elif op.kind == "iota":
            axis = info.starts.index(op.inputs[0])
            out = b.op("iota", starts[axis], shape=tile, axis=op.attr("axis"))
        elif op.kind in ("full", "broadcast") and op.outputs[0].type.shape == p_tile and p_tile != ():
            out = b.op(op.kind, *ins, **{**dict(op.attrs), "shape": tile})
        else:
            out = b.op(op.kind, *ins, **dict(op.attrs))
        mapped[op.outputs[0]] = out
    target = p_launch.buffers[info.store.attr("arg")]
    b.emit("store", (mapped[info.store.inputs[0]], *starts), {"arg": arg_index(target, task.args[info.store.attr("arg")])})

    # Statistics read elsewhere are stored too; their reductions go unless read before P.
    merged = {parts["index"] for parts in needed.values()} | {p_index}
    removed = {p_index}
    for name, parts in needed.items():
        outside = [r for r in readers.get(name, []) if r not in merged]
        if any(r < p_index for r in outside):
            continue
        removed.add(parts["index"])
        if outside:
            buffer = buffers[name]
            b.emit("store", (rows[name], *pids), {"arg": arg_index(name, Arg(full_view(buffer.type.shape), buffer.type.dtype, True))})
    grid = tuple(shape[:-1])
    body = b.finish(pids)
    new = Launch(Task("rows", grid, tuple(a for _, a in args), body, _warps(width), ("rows",)), tuple(n for n, _ in args))
    launches = []
    for i, launch in enumerate(program.launches):
        if i == p_index:
            launches.append(new)
        elif i not in removed:
            launches.append(launch)
    used = {n for launch in launches for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    return replace(program, launches=tuple(launches), buffers=tuple(bf for bf in program.buffers if bf.name in used or bf.kind is not BufferKind.temp))
