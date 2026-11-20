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
            elif op.kind == "load_indexed":
                # A gather fuse evaluated in the reduction (GPT-2's statistics over the summed
                # token and position embeddings).
                arg = launch.task.args[op.attr("arg")]
                out = b.op("load_indexed", *ins, **{**dict(op.attrs), "arg": arg_index(launch.buffers[op.attr("arg")], arg)})
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

    # Statistics read elsewhere are stored too; their reductions go unless read before P. A kept
    # reduction is a reader before P of the statistics it reads, so keeping one can keep others
    # (top-k: the reduction finding the max's position is read before P, and it reads the max).
    kept: set[int] = set()
    while True:
        gone = {parts["index"] for parts in needed.values() if parts["index"] not in kept} | {p_index}
        more = {parts["index"] for name, parts in needed.items() if parts["index"] not in kept and any(r < p_index and r not in gone for r in readers.get(name, []))}
        if not more:
            break
        kept |= more
    removed = {p_index}
    for name, parts in needed.items():
        if parts["index"] in kept:
            continue
        outside = [r for r in readers.get(name, []) if r not in gone]
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


# Rows reduced whole in one tile that are shorter than this share a block (pack_short_rows).
SHORT_ROW = 256


def pack_short_rows(program: Program, block: int) -> Program:
    """Reductions over short rows (tile.from_graph._reduce: one row per block) take several rows
    per block, up to `block` elements, along the kept axes from the inner one out. One 64-long
    row per 128-thread block (GAT's attention gradient sums [557056, 4, 64] over 64, its
    gathers evaluated in the loop) ran 2.2M blocks at 6.3 ms, 3x torch.compile.

    It runs after fusion: row fusion and the online softmax match reductions with one row per
    block, and those it fuses are gone by then. Lanes past the tensor along a kept axis compute
    on masked loads (gathers read index 0 there) and are dropped by the store; the reduced axis
    is masked as before.
    """
    launches = [_packed(launch, block) or launch for launch in program.launches]
    return replace(program, launches=tuple(launches))


def _packed(launch: Launch, block: int) -> Launch | None:
    task = launch.task
    origin = task.origin
    if launch.offsets or not origin or origin[0] != "reduce":
        return None
    body = task.body
    rank = len(body.params) + 1
    if origin[2] != (rank - 1,) or len(task.grid) != rank - 1 or not all(isinstance(g, int) for g in task.grid):
        return None
    loads = [op for op in _ops(body) if op.kind in ("load", "load_indexed")]
    if not loads or any(op.kind in ("atomic_add", "atomic_add_indexed") for op in _ops(body)):
        return None
    tile = loads[0].attr("shape")
    if len(tile) != rank or any(d != 1 for d in tile[:-1]) or tile[-1] >= SHORT_ROW or any(op.attr("shape") != tile for op in loads):
        return None
    for op in loads:
        view = task.args[op.attr("arg")].view
        if view.rank != rank or not isinstance(view.shape[-1], int) or view.shape[-1] > tile[-1]:
            return None  # one step covers the row
    for op in _ops(body):
        if op.kind == "store" and tuple(task.args[op.attr("arg")].view.shape) != tuple(task.grid):
            return None  # one row per block: the grid is the output
    extent, budget = [1] * (rank - 1), block // tile[-1]
    for a in reversed(range(rank - 1)):
        if budget < 2:
            break
        extent[a] = min(1 << max(task.grid[a] - 1, 0).bit_length(), budget)
        budget //= extent[a]
        if extent[a] < task.grid[a]:
            break
    if all(e == 1 for e in extent):
        return None
    new_tile = (*extent, tile[-1])
    shapes = {tile: new_tile, tuple(tile[:-1]): tuple(extent), (): ()}
    try:
        b = TileBuilder()
        pids = [TValue(TileType((), INDEX)) for _ in extent]
        starts = [p if e == 1 else b.op("mul", p, b.index(e)) for p, e in zip(pids, extent)]
        rebuilt = _retiled(body, dict(zip(body.params, starts)), shapes)
    except _Unpackable:
        return None
    b.ops.extend(rebuilt.ops)
    grid = tuple(-(-g // e) for g, e in zip(task.grid, extent))
    return replace(launch, task=replace(task, grid=grid, body=b.finish(pids)))


class _Unpackable(Exception):
    pass


def _ops(body: Body):
    for op in body.ops:
        yield op
        for region in op.regions:
            yield from _ops(region)


def _retiled(body: Body, subst: dict, shapes: dict) -> Body:
    def value(v: TValue) -> TValue:
        if v.type.shape not in shapes:
            raise _Unpackable
        return TValue(TileType(shapes[v.type.shape], v.type.dtype))

    params = tuple(subst.setdefault(p, value(p)) for p in body.params)
    ops = []
    for op in body.ops:
        regions = tuple(_retiled(r, subst, shapes) for r in op.regions)
        outputs = tuple(value(v) for v in op.outputs)
        subst.update(zip(op.outputs, outputs))
        attrs = dict(op.attrs)
        if "shape" in attrs:
            if tuple(attrs["shape"]) not in shapes:
                raise _Unpackable
            attrs["shape"] = shapes[tuple(attrs["shape"])]
        ops.append(TOp(op.kind, tuple(subst.get(v, v) for v in op.inputs), outputs, tuple(attrs.items()), regions))
    return Body(params, tuple(ops), tuple(subst.get(v, v) for v in body.results))
