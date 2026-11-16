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
# A producer whose output is wider than its inputs (bf16 in, f32 out) is recomputed in every
# reader up to this cost: reading its inputs again moves fewer bytes than writing and reading
# its output (GPT-2's gelu tanh, shared by forward and backward after CSE: 50 MB of f32 per layer).
RECOMPUTE_OPS = 8

_BYTES = {DType.f32: 4, DType.f64: 8, DType.bf16: 2, DType.i32: 4, DType.i64: 8, DType.u32: 4, DType.bool: 1}
_MEMORY = frozenset({"load", "load_indexed", "store", "atomic_add", "atomic_add_indexed"})
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
    widens: bool = False  # bytes per output element exceed the bytes it reads per element


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
    read = sum(_BYTES[a.dtype] for a in launch.task.args if not a.writes and any(st != 0 for st in a.view.strides))
    return _Pointwise(launch, starts, coordinate, store, cost, _BYTES[store.inputs[0].type.dtype] > read)


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
        if op.kind not in ("load", "load_indexed") or self.keep:
            return None
        if op.kind == "load_indexed" and not self.indexed_ok(op):
            return None
        name = self.launch.buffers[op.attr("arg")]
        recipe = self.recipes.get(name)
        view = self.launch.task.args[op.attr("arg")].view
        if recipe is None or view.source is None:
            return None
        if any(a.view.pad is not None for a in recipe.launch.task.args if not a.writes):
            return None
        cheap = recipe.cost <= CHEAP_OPS or (recipe.widens and recipe.cost <= RECOMPUTE_OPS)
        if self.readers[name] > 1 and not cheap:
            return None
        if self.into_dot and not _widening_cast(recipe):
            return None
        return recipe

    def indexed_ok(self, op: TOp) -> bool:
        """A gather may evaluate its producer at the gathered positions (GPT-2's loss reads 4096
        of 4096 x 50257 shifted logits): no iota in the producer (its coordinates come from the
        index there), and no reduction or dot in the consumer, whose store drops every lane
        outside the tensor, so no mask is needed."""
        name = self.launch.buffers[op.attr("arg")]
        recipe = self.recipes.get(name)
        if recipe is None or any(o.kind == "iota" for o in recipe.launch.task.body.ops):
            return False
        return not (_contains(self.launch.task.body, "reduce") or _contains(self.launch.task.body, "dot"))

    def body(self, body: Body, subst: dict[TValue, TValue]) -> Body:
        b = TileBuilder()
        for op in body.ops:
            recipe = self.allowed(op)
            if recipe is not None and op.kind == "load_indexed":
                index, *starts = [subst.get(v, v) for v in op.inputs]
                subst[op.outputs[0]] = self.inline(b, op, recipe, starts, index=index)
                self.changed = True
                continue
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

    def inline(self, b: TileBuilder, load: TOp, recipe: _Pointwise, starts: Sequence[TValue], index: TValue | None = None) -> TValue:
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
                slot = self.arg_index(producer.buffers[op.attr("arg")], Arg(view, arg.dtype, False))
                if index is not None:
                    out = b.op("load_indexed", index, *starts, arg=slot, shape=tile, dtype=op.attr("dtype"), axis_map=load.attr("axis_map"), other=op.attr("other"))
                else:
                    out = b.op("load", *starts, arg=slot, shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
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
        if index is not None:
            return result
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


def fuse_epilogues(program: Program, *, shared_cheap_only: bool = False) -> Program:
    """Absorb a dot task's only consumer, when pointwise, into the dot's epilogue.

    Producer inlining (fuse) cannot move a dot into its consumer, so a linear layer's output
    takes one more pass over memory per elementwise step after it: bias, activation, residual,
    dtype casts. Here the consumer's body is replayed on the dot's result tile, before the
    store, and the dot stores the consumer's output instead. Repeats along chains.

    Conditions: the dot's output is a temp written only by it through its full view and read
    only by the consumer, through its full view; the consumer writes a buffer of the same shape
    through its full view (so both tasks share one coordinate system).

    shared_cheap_only: absorb only cheap consumers whose output has several readers. Run before
    producer fusion, this keeps a bias add from being duplicated into each of its readers (each
    then materializing its own copy), without hiding from later passes the structure they match
    (the attention pass looks for the masking step after the score dot).
    """
    changed = True
    while changed:
        changed = False
        buffers = {b.name: b for b in program.buffers}
        readers: dict[str, list[int]] = {}
        writers: dict[str, list[int]] = {}
        for i, launch in enumerate(program.launches):
            for name, arg in zip(launch.buffers, launch.task.args):
                (writers if arg.writes else readers).setdefault(name, []).append(i)
        launches = list(program.launches)
        for i, launch in enumerate(launches):
            origin = launch.task.origin if launch is not None else ()
            if launch is None or not origin or not (origin[0] == "contract" or origin[0].startswith("attention")) or launch.offsets:
                continue
            for store in [op for op in launch.task.body.ops if op.kind == "store"]:
                absorbed = _try_absorb(launches, i, store, buffers, readers, writers, shared_cheap_only)
                if absorbed is not None:
                    launches[i], j = absorbed
                    launches[j] = None
                    changed = True
                    break
        program = replace(program, launches=tuple(l for l in launches if l is not None))
    used = {n for launch in program.launches for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    return replace(program, buffers=tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp))


def _try_absorb(launches, i, store, buffers, readers, writers, shared_cheap_only):
    """(new launch i, absorbed launch index) when launch i's `store` can take its consumer."""
    launch = launches[i]
    out = launch.buffers[store.attr("arg")]
    buffer = buffers[out]
    if buffer.kind is not BufferKind.temp or writers.get(out) != [i] or len(readers.get(out, [])) != 1:
        return None
    j = readers[out][0]
    consumer = launches[j]
    if consumer is None or consumer.offsets or j < i:
        return None
    info = _pointwise(consumer, buffers)
    if info is None:
        return None
    if shared_cheap_only:
        consumer_out = consumer.buffers[info.store.attr("arg")]
        if info.cost > CHEAP_OPS or len(readers.get(consumer_out, [])) < 2:
            return None
    reads = [a for a, (n, arg) in enumerate(zip(consumer.buffers, consumer.task.args)) if n == out]
    if len(reads) != 1 or launch.task.args[store.attr("arg")].view != full_view(buffer.type.shape):
        return None
    # The consumer reads the output whole, possibly with its axes permuted: perm[j] is
    # the consumer axis of buffer axis j. Its other views and its output are permuted the
    # same way, so the consumer can be evaluated on the producer's tile.
    perm = _permutation(consumer.task.args[reads[0]].view, buffer.type.shape)
    if perm is None:
        return None
    target = consumer.buffers[info.store.attr("arg")]
    if tuple(buffers[target].type.shape[perm[j]] for j in range(len(perm))) != buffer.type.shape:
        return None
    # The consumer's other inputs must be ready when the producer runs.
    others = {n for n, a in zip(consumer.buffers, consumer.task.args) if not a.writes and n != out}
    if any(w > i for n in others for w in writers.get(n, [])):
        return None
    return _absorb(launch, store, consumer, info, reads[0], perm), j


def _permutation(view: View, shape) -> list[int] | None:
    """perm with view axis perm[j] = buffer axis j, when the view is the whole buffer permuted."""
    if view.pad is not None or view.source is None or len(view.source) != len(shape) or view.rank != len(shape):
        return None
    perm = []
    for j, (axis, start) in enumerate(view.source):
        if axis is None or start != 0 or view.shape[axis] != shape[j]:
            return None
        perm.append(axis)
    return perm if sorted(perm) == list(range(len(shape))) else None


def _permuted(view: View, perm: list[int]) -> View:
    """A consumer view re-indexed by producer axes: producer axis j is consumer axis perm[j]."""
    pad = tuple(view.pad[a] for a in perm) if view.pad is not None else None
    source = tuple((None if a is None else perm.index(a), s) for a, s in view.source) if view.source is not None else None
    return View(tuple(view.shape[a] for a in perm), tuple(view.strides[a] for a in perm), view.offset, pad, view.fill, source)


def _absorb(launch: Launch, store: TOp, consumer: Launch, info: _Pointwise, read_arg: int, perm: list[int]) -> Launch:
    task = launch.task
    args: list[tuple[str, Arg]] = [(n, a) for n, a in zip(launch.buffers, task.args)]

    def arg_index(name: str, arg: Arg) -> int:
        key = (name, arg)
        if key not in args:
            args.append(key)
        return args.index(key)

    b = TileBuilder()
    b.ops = [op for op in task.body.ops if op is not store]
    value, starts = store.inputs[0], list(store.inputs[1:])
    tile = value.type.shape
    p_tile = info.store.inputs[0].type.shape
    mapped: dict[TValue, TValue] = {}
    for op in consumer.task.body.ops:
        if op.kind == "store" or (op.outputs and set(op.outputs) <= info.coordinate):
            continue
        ins = [mapped.get(v, v) for v in op.inputs]
        if op.kind == "load":
            a = op.attr("arg")
            if a == read_arg:
                out = value
            else:
                arg = consumer.task.args[a]
                moved = replace(arg, view=_permuted(arg.view, perm))
                out = b.op("load", *starts, arg=arg_index(consumer.buffers[a], moved), shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
        elif op.kind == "iota":
            axis = perm.index(op.attr("axis"))
            out = b.op("iota", starts[axis], shape=tile, axis=axis)
        elif op.kind in ("full", "broadcast") and op.outputs[0].type.shape == p_tile and p_tile != ():
            out = b.op(op.kind, *ins, **{**dict(op.attrs), "shape": tile})
        else:
            out = b.op(op.kind, *ins, **dict(op.attrs))
        mapped[op.outputs[0]] = out
    result = mapped[info.store.inputs[0]]
    target = consumer.buffers[info.store.attr("arg")]
    target_arg = consumer.task.args[info.store.attr("arg")]
    b.emit("store", (result, *starts), {"arg": arg_index(target, replace(target_arg, view=_permuted(target_arg.view, perm)))})
    body = Body(task.body.params, tuple(b.ops), task.body.results)
    used = sorted({op.attr("arg") for op in _walk(body) if op.kind in _MEMORY})
    remap = {old: new for new, old in enumerate(used)}
    body = _renumber(body, remap)
    new = Task(task.name, task.grid, tuple(args[k][1] for k in used), body, task.warps, task.origin)
    return Launch(new, tuple(args[k][0] for k in used))
