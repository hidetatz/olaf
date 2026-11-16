"""Final Tile IR normalization before lowering."""

from __future__ import annotations

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
    """
    launches = []
    for launch in program.launches:
        task = launch.task
        offsets = tuple(launch.offset(i) + a.view.offset for i, a in enumerate(task.args))
        args = tuple(replace(a, view=replace(a.view, offset=0)) for a in task.args)
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


def _pure_copy(task: Task) -> bool:
    """The body stores what it loads, unchanged (fusion may have inlined a producer into a copy)."""
    stores = [op for op in task.body.ops if op.kind == "store"]
    loads = {op.outputs[0]: op for op in task.body.ops if op.kind == "load"}
    return len(stores) == 1 and len(loads) == 1 and stores[0].inputs[0] in loads and loads[stores[0].inputs[0]].attr("arg") == 0
