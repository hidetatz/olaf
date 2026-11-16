"""Rule-based fusion on Tile IR: inline pointwise producers into the loads that read them.

A pointwise task computes each output element from its inputs at the same coordinates (what
from_graph emits for elementwise ops, iota and copies). Inlining it into a consumer replaces the
consumer's load of the intermediate buffer with loads of the producer's inputs, through views
composed with View.source, and a replay of the producer's ops on the consumer's tile. The
intermediate buffer, and the launch that wrote it, disappear when no reader is left.

Policy (the rules a cost model will later replace):
- Nothing is inlined into a dot operand except an exact widening cast of a b16 load (bf16 ->
  f32). Each operand tile is reloaded once per output tile along the other dimension (24 times
  for GPT-2's 768-wide projections with 32-wide tiles), and so would its recomputation be; and a
  matrix-instruction dot copies operands to shared memory with 16-byte async copies, which need
  plain b16 loads. The widening cast is the exception because it is exact: the dot can run on
  the b16 values with an f32 accumulator (backend/cuda/mma.py).
- An intermediate read once is otherwise inlined.
- An intermediate read several times is duplicated into every reader only when it is cheap.
"Cheap" counts arithmetic ops; casts, broadcasts of scalars and index arithmetic are free.

Epilogue fusion (absorbing elementwise consumers into a dot or a reduction) and the online
softmax rewrite for attention are separate rules, added as their own passes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.builder import BINARY, INDEX, UNARY, TileBuilder
from olaf.tile.ir import Arg, Body, Buffer, BufferKind, Launch, Program, Task, TileType, TOp, TValue, View, full_view

CHEAP_OPS = 3

_MEMORY = frozenset({"load", "load_indexed", "store", "atomic_add_indexed"})
_FREE = frozenset({"cast", "full", "dim", "broadcast", "iota"})
_POINTWISE = UNARY | BINARY | {"cmp", "where"} | _FREE


@dataclass(frozen=True)
class _Pointwise:
    """A pointwise launch: its task, buffers, the coordinate values of its body, and its cost."""

    launch: Launch
    starts: tuple[TValue, ...]
    coordinate: frozenset[TValue]
    store: TOp
    cost: int


def fuse(program: Program) -> Program:
    readers = Counter(name for launch in program.launches for name, arg in zip(launch.buffers, launch.task.args) if not arg.writes)
    writers = Counter(name for launch in program.launches for name, arg in zip(launch.buffers, launch.task.args) if arg.writes)
    buffers = {b.name: b for b in program.buffers}
    recipes: dict[str, _Pointwise] = {}
    launches: list[Launch] = []
    for launch in program.launches:
        launch = _inline_into(launch, recipes, readers, buffers)
        info = _pointwise(launch, buffers)
        if info is not None:
            out = launch.buffers[info.store.attr("arg")]
            if buffers[out].kind is BufferKind.temp and writers[out] == 1:
                recipes[out] = info
        launches.append(launch)
    return _without_dead(program, launches)


def _pointwise(launch: Launch, buffers: dict[str, Buffer]) -> _Pointwise | None:
    body = launch.task.body
    stores = [op for op in body.ops if op.kind == "store"]
    if len(stores) != 1 or any(op.regions or op.kind not in _POINTWISE | {"load", "store"} for op in body.ops):
        return None
    store = stores[0]
    starts = store.inputs[1:]
    arg = launch.task.args[store.attr("arg")]
    target = buffers[launch.buffers[store.attr("arg")]]
    if arg.view != full_view(target.type.shape):
        return None
    coordinate = _coordinate_values(body)
    cost = 0
    for op in body.ops:
        if op.kind == "load" and (op.inputs != starts or op.attr("shape") != store.inputs[0].type.shape):
            return None
        if op.kind == "iota" and op.inputs[0] not in starts:
            return None
        if op.kind in ("load", "store") or set(op.outputs) <= coordinate:
            continue
        if any(v in coordinate for v in op.inputs) and op.kind != "iota":
            return None
        if op.kind in ("full", "broadcast") and op.outputs[0].type.shape not in ((), store.inputs[0].type.shape):
            return None
        if op.kind == "broadcast" and op.attr("dims") != ():
            return None
        cost += op.kind not in _FREE
    return _Pointwise(launch, starts, coordinate, store, cost)


def _coordinate_values(body: Body) -> frozenset[TValue]:
    """Program ids and the index scalars computed only from them and constants (the tile starts)."""
    coordinate = set(body.params)
    for op in body.ops:
        if op.kind in ("mul", "add") and op.outputs[0].type == TileType((), INDEX) and any(v in coordinate for v in op.inputs):
            if all(v in coordinate or _is_scalar_constant(body, v) for v in op.inputs):
                coordinate.update(op.outputs)
    return frozenset(coordinate)


def _is_scalar_constant(body: Body, value: TValue) -> bool:
    return any(op.kind == "full" and op.outputs[0] is value and op.outputs[0].type.shape == () for op in body.ops)


class _Rewriter:
    """Rebuilds a consumer body, replacing selected loads by inlined producers."""

    def __init__(self, launch: Launch, recipes: dict[str, _Pointwise], readers: Counter, buffers: dict[str, Buffer]) -> None:
        self.launch = launch
        self.recipes = recipes
        self.readers = readers
        self.buffers = buffers
        self.args: list[tuple[str, Arg]] = list(zip(launch.buffers, launch.task.args))
        self.into_dot = _contains(launch.task.body, "dot")
        self.consumer_pointwise = _pointwise(launch, buffers) is not None
        # A task that only widens a b16 load to f32 stays as it is: inlining its producer would
        # turn it into an f32 computation, and it could no longer be inlined into a dot as an
        # exact widening (which is what lets the dot run on b16 matrix instructions).
        consumer = _pointwise(launch, buffers)
        self.keep = consumer is not None and _widening_cast(consumer)
        self.changed = False

    def arg_index(self, buffer: str, arg: Arg) -> int:
        key = (buffer, arg)
        if key not in self.args:
            self.args.append(key)
        return self.args.index(key)

    def allowed(self, op: TOp) -> _Pointwise | None:
        if op.kind != "load" or self.keep:
            return None
        name = self.launch.buffers[op.attr("arg")]
        recipe = self.recipes.get(name)
        view = self.launch.task.args[op.attr("arg")].view
        if recipe is None or view.source is None:
            return None
        if any(a.view.pad is not None for a in recipe.launch.task.args if not a.writes):
            return None
        cheap = recipe.cost <= CHEAP_OPS
        if self.readers[name] > 1 and not cheap:
            return None
        if self.into_dot and not _widening_cast(recipe):
            return None
        return recipe

    def body(self, body: Body, subst: dict[TValue, TValue]) -> Body:
        b = TileBuilder()
        for op in body.ops:
            recipe = self.allowed(op)
            if recipe is not None:
                subst[op.outputs[0]] = self.inline(b, op, recipe, [subst.get(v, v) for v in op.inputs])
                self.changed = True
                continue
            regions = tuple(self.body(r, subst) for r in op.regions)
            outputs = tuple(TValue(v.type) for v in op.outputs)
            subst.update(zip(op.outputs, outputs))
            attrs = op.attrs
            if op.kind in _MEMORY:
                arg = op.attr("arg")
                attrs = tuple((k, self.arg_index(*self.args[arg]) if k == "arg" else v) for k, v in attrs)
            b.ops.append(TOp(op.kind, tuple(subst.get(v, v) for v in op.inputs), outputs, attrs, regions))
        return Body(body.params, tuple(b.ops), tuple(subst.get(v, v) for v in body.results))

    def inline(self, b: TileBuilder, load: TOp, recipe: _Pointwise, starts: Sequence[TValue]) -> TValue:
        consumer_view = self.launch.task.args[load.attr("arg")].view
        tile = load.attr("shape")
        producer = recipe.launch
        p_tile = recipe.store.inputs[0].type.shape
        mapped: dict[TValue, TValue] = {}
        for op in producer.task.body.ops:
            if op.kind == "store" or (op.outputs and set(op.outputs) <= recipe.coordinate):
                continue
            ins = [mapped.get(v, v) for v in op.inputs]
            if op.kind == "load":
                arg = producer.task.args[op.attr("arg")]
                view = _compose(arg.view, consumer_view)
                index = self.arg_index(producer.buffers[op.attr("arg")], Arg(view, arg.dtype, False))
                out = b.op("load", *starts, arg=index, shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
            elif op.kind == "iota":
                axis = recipe.starts.index(op.inputs[0])
                out = self.coordinate(b, consumer_view, starts, tile, axis)
            elif op.kind in ("full", "broadcast") and op.outputs[0].type.shape == p_tile and p_tile != ():
                attrs = {**dict(op.attrs), "shape": tile}
                out = b.op(op.kind, *ins, **attrs)
            else:
                out = b.op(op.kind, *ins, **dict(op.attrs))
            mapped[op.outputs[0]] = out
        result = mapped[recipe.store.inputs[0]]
        if result.type.shape != tile:
            result = b.op("broadcast", result, shape=tile, dims=())
        if consumer_view.pad is not None:
            # The replaced load read `fill` there; the inlined ops would give f(fill, ...).
            fill = b.full(tile, consumer_view.fill, result.type.dtype)
            for axis, (lo, hi) in enumerate(consumer_view.pad):
                if lo == 0 and hi == 0:
                    continue
                coord = b.op("iota", starts[axis], shape=tile, axis=axis)
                if lo != 0:
                    low = b.op("broadcast", b.dim(lo), shape=tile, dims=())
                    result = b.op("where", b.op("cmp", coord, low, fn="lt"), fill, result)
                if hi != 0:
                    high = b.op("broadcast", b.dim(consumer_view.shape[axis] - hi), shape=tile, dims=())
                    result = b.op("where", b.op("cmp", coord, high, fn="lt"), result, fill)
        exact_zero = _widening_cast(recipe) and load.attr("other") == 0
        if not self.consumer_pointwise and not exact_zero:
            # Lanes outside the consumer's view must read `other` (0 for a sum, -inf for a max), as
            # the replaced load would have; the inlined computation would give f(0, ...) there.
            # A pointwise consumer discards those lanes at its masked store, so it needs no select,
            # and a widening cast reads 0 there already (its own load is masked with 0).
            other = b.full(tile, load.attr("other"), result.type.dtype)
            for axis, (extent, size) in enumerate(zip(tile, consumer_view.shape)):
                if extent > 1:
                    coord = b.op("iota", starts[axis], shape=tile, axis=axis)
                    bound = b.op("broadcast", b.dim(size), shape=tile, dims=())
                    result = b.op("where", b.op("cmp", coord, bound, fn="lt"), result, other)
        return result

    def coordinate(self, b: TileBuilder, view: View, starts: Sequence[TValue], tile: tuple[int, ...], buffer_axis: int) -> TValue:
        """The producer's coordinate along buffer_axis, at each element of the consumer's tile."""
        axis, start = view.source[buffer_axis]
        if axis is None:
            return b.op("broadcast", b.dim(start), shape=tile, dims=())
        origin = starts[axis] if start == 0 else b.op("add", starts[axis], b.dim(start))
        return b.op("iota", origin, shape=tile, axis=axis)

    def finish(self, body: Body) -> Launch:
        used = sorted({op.attr("arg") for op in _walk(body) if op.kind in _MEMORY})
        remap = {old: new for new, old in enumerate(used)}
        body = _renumber(body, remap)
        args = tuple(self.args[i][1] for i in used)
        task = Task(self.launch.task.name, self.launch.task.grid, args, body, self.launch.task.warps, self.launch.task.origin)
        if self.launch.offsets:
            raise ValueError("fuse runs before tile.normalize.hoist_offsets")
        return Launch(task, tuple(self.args[i][0] for i in used))


def _compose(inner: View, outer: View) -> View:
    """inner addresses a buffer in producer coordinates; outer maps consumer coordinates to producer ones."""
    rank = outer.rank
    strides: list[Dim] = [0] * rank
    offset: Dim = inner.offset
    for p_axis, (c_axis, start) in enumerate(outer.source):
        if c_axis is not None:
            strides[c_axis] = strides[c_axis] + inner.strides[p_axis]
            if outer.pad is not None:
                # The kept padding already subtracts low in the address; start includes -low too.
                start = start + outer.pad[c_axis][0]
        offset = offset + start * inner.strides[p_axis]
    source = None
    if inner.source is not None:
        source = []
        for p_axis, start in inner.source:
            if p_axis is None:
                source.append((None, start))
            else:
                c_axis, c_start = outer.source[p_axis]
                source.append((c_axis, start + c_start))
        source = tuple(source)
    # Keeping the consumer's padding means padded positions never touch memory: their producer
    # coordinates lie outside the producer's range, where the address would be out of bounds.
    return View(outer.shape, tuple(strides), offset, outer.pad, outer.fill, source)


def _widening_cast(recipe: _Pointwise) -> bool:
    """The producer only casts one b16 load to f32 (exact)."""
    ops = [op for op in recipe.launch.task.body.ops if op.kind not in ("store",) and not set(op.outputs) <= recipe.coordinate]
    kinds = [op.kind for op in ops if op.kind != "full"]
    if kinds != ["load", "cast"]:
        return False
    load = next(op for op in ops if op.kind == "load")
    cast = next(op for op in ops if op.kind == "cast")
    return load.attr("dtype") is DType.bf16 and cast.attr("dtype") is DType.f32 and recipe.store.inputs[0] is cast.outputs[0]


def _contains(body: Body, kind: str) -> bool:
    return any(op.kind == kind for op in _walk(body))


def _walk(body: Body):
    for op in body.ops:
        yield op
        for region in op.regions:
            yield from _walk(region)


def _renumber(body: Body, remap: dict[int, int]) -> Body:
    ops = []
    for op in body.ops:
        attrs = op.attrs
        if op.kind in _MEMORY:
            attrs = tuple((k, remap[v] if k == "arg" else v) for k, v in attrs)
        ops.append(TOp(op.kind, op.inputs, op.outputs, attrs, tuple(_renumber(r, remap) for r in op.regions)))
    return Body(body.params, tuple(ops), body.results)


def _inline_into(launch: Launch, recipes, readers, buffers) -> Launch:
    rewriter = _Rewriter(launch, recipes, readers, buffers)
    body = rewriter.body(launch.task.body, {})
    if not rewriter.changed:
        return launch
    return rewriter.finish(body)


def _without_dead(program: Program, launches: list[Launch]) -> Program:
    """Drop launches whose every written buffer is a temp nobody reads, then unused buffers."""
    readers = Counter(name for launch in launches for name, arg in zip(launch.buffers, launch.task.args) if not arg.writes)
    kinds = {b.name: b.kind for b in program.buffers}
    kept: list[Launch] = []
    for launch in reversed(launches):
        written = [n for n, a in zip(launch.buffers, launch.task.args) if a.writes]
        if all(kinds[n] is BufferKind.temp and readers[n] == 0 for n in written):
            for name, arg in zip(launch.buffers, launch.task.args):
                if not arg.writes:
                    readers[name] -= 1
            continue
        kept.append(launch)
    kept.reverse()
    used = {n for launch in kept for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    buffers = tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp)
    return replace(program, buffers=buffers, launches=tuple(kept))
