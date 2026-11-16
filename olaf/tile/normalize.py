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
        hoisted = Task(task.name, task.grid, args, task.body, task.warps)
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
        launches.append(replace(launch, task=Task(task.name, task.grid, tuple(args), task.body, task.warps)))
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
