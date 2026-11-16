"""Tile IR: a step as a sequence of task launches over global buffers.

A Task is a template: a grid, one strided View per argument, and a body that computes one grid
instance on tiles. A Launch binds a task's arguments to buffers. Buffer names are not part of the
template text, so the 12 blocks of GPT-2 produce the same task texts and share one compile and
search result (docs/design.md, "スケジュール探索とコンパイルキャッシュ").

Tile shapes are static (they become register and shared memory layouts); everything that depends
on the problem size (grid, loop bounds, view shapes and strides) may be symbolic and is evaluated at
launch time. Loads and stores are masked by the view bounds, so a tile may overhang the tensor.

Launches run in order and a launch sees every write of the launches before it. That is the
contract graph replay needs. Tile-granular dependencies for the megakernel (P2) are derived from
the views, which record exactly which elements each instance touches, so no extra field is needed now.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from olaf.graph.dim import Dim, DimExpr, product
from olaf.graph.ir import ConstantData, _attr_text
from olaf.graph.types import DType, StateDecl, TensorType

_MISSING = object()


@dataclass(frozen=True)
class TileType:
    shape: tuple[int, ...]
    dtype: DType

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __str__(self) -> str:
        return f"{self.dtype}<{'x'.join(map(str, self.shape))}>" if self.shape else str(self.dtype)


@dataclass(frozen=True, eq=False)
class TValue:
    type: TileType


@dataclass(frozen=True, eq=False)
class TOp:
    kind: str
    inputs: tuple[TValue, ...]
    outputs: tuple[TValue, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    regions: tuple[Body, ...] = ()

    def attr(self, name: str, default: Any = _MISSING) -> Any:
        for key, value in self.attrs:
            if key == name:
                return value
        if default is _MISSING:
            raise KeyError(f"{self.kind} has no attribute {name}")
        return default


@dataclass(frozen=True, eq=False)
class Body:
    params: tuple[TValue, ...]
    ops: tuple[TOp, ...]
    results: tuple[TValue, ...]


@dataclass(frozen=True)
class View:
    """A strided window onto a buffer's flat storage, in elements.

    `shape` is the logical shape. With `pad`, logical index i on an axis maps to stored index
    i - low, and positions in [0, low) or [shape - high, shape) read `fill` instead of memory. That
    keeps pad and concat as address arithmetic instead of materialized copies.

    `source` says which buffer element a view element is, in the buffer's own coordinates: for
    each buffer axis, (view axis or None, start), meaning buffer index = view index + start, or
    just start when no view axis moves along it (a broadcast or a size-1 axis). Strides alone
    give only the address, and fusion needs coordinates: inlining a producer into a consumer means
    evaluating the producer at the consumer's coordinates. None after a reshape, which mixes axes.
    """

    shape: tuple[Dim, ...]
    strides: tuple[Dim, ...]
    offset: Dim = 0
    pad: tuple[tuple[Dim, Dim], ...] | None = None
    fill: float = 0.0
    source: tuple[tuple[int | None, Dim], ...] | None = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_contiguous(self) -> bool:
        return self.pad is None and self.strides == contiguous_strides(self.shape)

    def __str__(self) -> str:
        text = f"[{','.join(map(str, self.shape))}] strides=[{','.join(map(str, self.strides))}]"
        if self.offset != 0:
            text += f" offset={self.offset}"
        if self.pad is not None:
            text += " pad=[" + ",".join(f"{lo}:{hi}" for lo, hi in self.pad) + f"] fill={self.fill!r}"
        if self.source is not None:
            text += " source=[" + ",".join(f"{'_' if a is None else a}+{s}" for a, s in self.source) + "]"
        return text


def full_view(shape) -> View:
    """The whole of a contiguous buffer."""
    return View(tuple(shape), contiguous_strides(shape), 0, None, 0.0, tuple((i, 0) for i in range(len(shape))))


def contiguous_strides(shape) -> tuple[Dim, ...]:
    return tuple(product(shape[i + 1 :]) for i in range(len(shape)))


@dataclass(frozen=True)
class Arg:
    view: View
    dtype: DType
    writes: bool

    def __str__(self) -> str:
        return f"{'out' if self.writes else 'in'} {self.dtype}{self.view}"


@dataclass(frozen=True, eq=False)
class Task:
    """Body params are one i32 program id per grid axis.

    `warps` is a schedule choice (how many 32-wide execution groups run one instance); like tile
    shapes and the pipeline depth of a loop (the `stages` attribute of `for`), it lives in the IR
    so that the schedule search changes the IR and the cache key covers it.
    """

    name: str
    grid: tuple[Dim, ...]
    args: tuple[Arg, ...]
    body: Body
    warps: int = 4
    # The Graph IR op this task implements, e.g. ("contract", "abxz,abyz->abxy") or
    # ("reduce", "max", (3,)), for passes that match task patterns (tile.online_softmax).
    # Analysis metadata only: it is not part of the template text and never changes code.
    origin: tuple = ()

    def text(self) -> str:
        """Template text: the cache key. The name is a debugging label and is left out."""
        lines = [f"task grid=[{','.join(map(str, self.grid))}] warps={self.warps}"]
        lines += [f"  arg{i}: {a}" for i, a in enumerate(self.args)]
        lines += _print_body(self.body, "body", 1)
        return "\n".join(lines) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()


class BufferKind(StrEnum):
    input = "input"
    output = "output"
    temp = "temp"
    constant = "constant"
    state_in = "state_in"
    state_out = "state_out"


@dataclass(frozen=True)
class Buffer:
    """Storage of a tensor: contiguous, or with the physical `strides` of tile.normalize.align_rows."""

    name: str
    type: TensorType
    kind: BufferKind
    constant: ConstantData | None = None
    state: StateDecl | None = None
    strides: tuple[Dim, ...] | None = None

    def elements(self) -> Dim:
        """Number of elements to allocate."""
        if self.strides is None:
            return product(self.type.shape)
        return self.type.shape[0] * self.strides[0] if self.type.shape else 1

    def __str__(self) -> str:
        text = f"%{self.name}: {self.kind} {self.type}"
        if self.constant is not None:
            text += f" = {self.constant}"
        if self.state is not None:
            text += f" @{self.state.name}"
        if self.strides is not None:
            text += f" strides=[{','.join(map(str, self.strides))}]"
        return text


@dataclass(frozen=True)
class Launch:
    """A task bound to buffers. `offsets` (elements, one per argument, empty for all zero) are
    added to each buffer's base address: per-launch addressing that is not part of the template,
    so the unrolled blocks of a scan, which differ only in where their weights start, share one
    template (tile.normalize.hoist_offsets)."""

    task: Task
    buffers: tuple[str, ...]
    offsets: tuple[Dim, ...] = ()

    def offset(self, i: int) -> Dim:
        return self.offsets[i] if self.offsets else 0


@dataclass(frozen=True, eq=False)
class Program:
    buffers: tuple[Buffer, ...]
    launches: tuple[Launch, ...]
    inputs: tuple[str, ...]  # one per graph param, in order
    outputs: tuple[str, ...]  # one per graph result, in order

    def buffer(self, name: str) -> Buffer:
        for b in self.buffers:
            if b.name == name:
                return b
        raise KeyError(name)

    def tasks(self) -> dict[str, Task]:
        """Unique task templates by digest, in first-use order."""
        found: dict[str, Task] = {}
        for launch in self.launches:
            found.setdefault(launch.task.digest(), launch.task)
        return found

    def text(self) -> str:
        templates = {d: i for i, d in enumerate(self.tasks())}
        lines = [str(b) for b in self.buffers]
        lines += [f"inputs {', '.join('%' + n for n in self.inputs)}", f"outputs {', '.join('%' + n for n in self.outputs)}"]
        for digest, index in templates.items():
            lines.append(f"template t{index}:")
            lines += ["  " + line for line in self.tasks()[digest].text().splitlines()]
        for launch in self.launches:
            bound = ", ".join("%" + n + (f"+{launch.offset(i)}" if launch.offset(i) != 0 else "") for i, n in enumerate(launch.buffers))
            lines.append(f"launch t{templates[launch.task.digest()]}({bound})  # {launch.task.name}")
        return "\n".join(lines) + "\n"


def _tile_attr_text(value: Any) -> str:
    if isinstance(value, View):
        return "{" + str(value) + "}"
    if isinstance(value, DimExpr):
        return str(value)
    if isinstance(value, tuple):
        return "[" + ",".join(_tile_attr_text(v) for v in value) + "]"
    if value is None:
        return "none"
    return _attr_text(value)


def _print_body(body: Body, name: str, depth: int, numbers: dict[TValue, int] | None = None) -> list[str]:
    # Regions see their parent's values (unlike Graph IR regions, which are closed): a loop body
    # reads program ids and loop-invariant tiles, and threading them through would only add noise.
    numbers = {} if numbers is None else numbers

    def define(v: TValue) -> str:
        numbers[v] = len(numbers)
        return f"%{numbers[v]}"

    def use(v: TValue) -> str:
        if v not in numbers:
            raise ValueError("reference to an undefined tile value")
        return f"%{numbers[v]}"

    pad = "  " * depth
    lines = [f"{pad}{name}({', '.join(f'{define(p)}: {p.type}' for p in body.params)}) {{"]
    for op in body.ops:
        outs = ", ".join(define(v) for v in op.outputs)
        args = (" " + ", ".join(use(v) for v in op.inputs)) if op.inputs else ""
        attrs = (" {" + ", ".join(f"{k}={_tile_attr_text(v)}" for k, v in op.attrs) + "}") if op.attrs else ""
        types = (" : " + ", ".join(str(v.type) for v in op.outputs)) if op.outputs else ""
        lines.append(f"{pad}  {outs + ' = ' if outs else ''}{op.kind}{args}{attrs}{types}")
        for i, region in enumerate(op.regions):
            lines.extend(_print_body(region, f"region{i}", depth + 2, numbers))
    lines.append(f"{pad}  yield " + ", ".join(use(v) for v in body.results))
    lines.append(f"{pad}}}")
    return lines
