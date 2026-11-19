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
    widens: bool = False  # output in a wider type than its inputs, and no fewer bytes than it reads (see _pointwise)


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


def _pointwise(launch: Launch, buffers: dict[str, Buffer], *, multi: bool = False) -> _Pointwise | None:
    """A pointwise task (see the module docstring). multi: also accept several stores (of
    sibling fusion's merged tasks), all at the same tile; info.store is then the first."""
    body = launch.task.body
    stores = [op for op in body.ops if op.kind == "store"]
    if (len(stores) != 1 and not (multi and stores)) or any(op.regions or op.kind not in _POINTWISE | {"load", "store"} for op in body.ops):
        return None
    store = stores[0]
    starts = store.inputs[1:]
    for other in stores:
        arg = launch.task.args[other.attr("arg")]
        target = buffers[launch.buffers[other.attr("arg")]]
        if arg.view != full_view(target.type.shape) or other.inputs[1:] != starts or other.inputs[0].type.shape != store.inputs[0].type.shape:
            return None
    scalars = _scalar_constants(body)
    coordinate = _coordinate_values(body, scalars)
    args = launch.task.args
    total = _elements(args[store.attr("arg")].view)
    # Values computed only from small broadcast operands (a batch norm's per-channel mean and
    # rstd over a whole activation) vary along few axes; f32 arithmetic on them is not counted:
    # a reader recomputing them per element spends a few instructions where materializing the
    # task's output costs its bytes (BN's backward: 33 MB of f32 per layer, written once and
    # read twice).
    small = {op.outputs[0] for op in body.ops if op.kind == "load" and _small(args[op.attr("arg")].view, total)}
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
        if op.kind not in _FREE and op.inputs and all(v in small or v in coordinate or v in scalars for v in op.inputs):
            small.update(op.outputs)
            if op.outputs[0].type.dtype is not DType.f64:
                continue
        elif op.kind == "cast" and op.inputs[0] in small:
            small.update(op.outputs)
        if op.kind not in _FREE:
            # f64 arithmetic is never cheap: duplicated into a reader it runs once per thread,
            # and consumer GPUs do f64 at 1/64 of the f32 rate (AdamW's f64 bias corrections,
            # inlined into GPT-2's 38M-element embedding update, would be ~1e9 f64 operations).
            cost += CHEAP_OPS + 1 if op.outputs and op.outputs[0].type.dtype is DType.f64 else 1
    # Widening: the output takes no fewer bytes than the inputs it reads, in a wider type than
    # each (bf16 in, f32 out: GELU's input; a residual gradient summing two bf16 tensors in
    # f32). Reading the inputs again in each reader then moves no more than reading the
    # output would, and its write goes away. Same-type chains (f32 attention scores) stay as
    # they are: duplicating them breaks up the patterns later passes match.
    reads = [_BYTES[a.dtype] for a in args if not a.writes and not _small(a.view, total)]
    out_bytes = _BYTES[store.inputs[0].type.dtype]
    return _Pointwise(launch, starts, coordinate, store, cost, out_bytes >= sum(reads) and all(out_bytes > r for r in reads))


def _elements(view: View) -> Dim:
    """Distinct elements a view addresses (axes of stride 0 repeat them)."""
    count: Dim = 1
    for d, st in zip(view.shape, view.strides):
        if st != 0:
            count = count * d
    return count


def _small(view: View, total: Dim) -> bool:
    """A broadcast operand that a task rereads from cache: under 1/64 of its output's elements."""
    n = _elements(view)
    return isinstance(n, int) and isinstance(total, int) and n * 64 <= total


def _coordinate_values(body: Body, scalars: frozenset[TValue] | None = None) -> frozenset[TValue]:
    """Program ids and the index scalars computed only from them and constants (the tile starts)."""
    scalars = _scalar_constants(body) if scalars is None else scalars
    coordinate = set(body.params)
    for op in body.ops:
        if op.kind in ("mul", "add") and op.outputs[0].type == TileType((), INDEX) and any(v in coordinate for v in op.inputs):
            if all(v in coordinate or v in scalars for v in op.inputs):
                coordinate.update(op.outputs)
    return frozenset(coordinate)


def _scalar_constants(body: Body) -> frozenset[TValue]:
    """The body's scalar constants. A set, built once per body: testing each value against
    every op made fusion quadratic in a task's size (a U-Net's sibling fusion: 20 minutes)."""
    return frozenset(op.outputs[0] for op in body.ops if op.kind == "full" and op.outputs[0].type.shape == ())


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
        if recipe is None or (view.source is None and not _same_layout(recipe, name, self.buffers) and _stepped(recipe, view, self.buffers[name]) is None):
            return None
        if view.source is None and self.readers[name] > 1 and _narrows(recipe):
            # Its output stays for the other readers (a bf16 copy for a dot): reading the wider
            # inputs again here would move more bytes than reading that output.
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
                if consumer_view.source is not None:
                    view = _compose(arg.view, consumer_view)
                elif _same_layout(recipe, producer.buffers[recipe.store.attr("arg")], self.buffers):
                    view = _same_layout_view(arg.view, consumer_view)
                else:
                    view = _stepped_view(arg.view, consumer_view, *_stepped(recipe, consumer_view, self.buffers[producer.buffers[recipe.store.attr("arg")]]))
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


def _same_layout(recipe: _Pointwise, name: str, buffers: dict[str, Buffer]) -> bool:
    """The producer reads only buffers laid out like its output (same shape, contiguous, full
    views) and scalars, and computes no coordinates: then a consumer view without a source map
    (a reshape, as a split reduction reads [4, 1024, 768] as [16, 256, 768]) addresses each
    input exactly as it addresses the output. GPT-2's layernorm weight gradient multiplies two
    such buffers only to sum the product over rows."""
    out = buffers[name]
    if out.strides is not None or any(op.kind == "iota" for op in recipe.launch.task.body.ops):
        return False
    for buffer, arg in zip(recipe.launch.buffers, recipe.launch.task.args):
        if arg.writes or all(st == 0 for st in arg.view.strides) and arg.view.pad is None:
            continue
        source = buffers[buffer]
        if source.type.shape != out.type.shape or source.strides is not None or arg.view != full_view(out.type.shape):
            return False
    return True


def _stepped(recipe: _Pointwise, view: View, out: Buffer):
    """For a consumer view without a source map in which every axis steps along one axis of
    the producer's (contiguous) output, without carrying into the next: per view axis
    (producer axis, step) and the producer coordinates of its first element
    (tile.normalize._axis_steps), else None. A split reduction reads a [128, 8, 8, 256]
    activation as [64, 2, 8, 8, 256]: the producer's input views compose with that exactly as
    with a source map, which cannot say "axis 0 is 2a + b"."""
    from olaf.tile.normalize import _axis_steps

    if out.strides is not None or any(op.kind == "iota" for op in recipe.launch.task.body.ops):
        return None
    return _axis_steps(view, out.type.shape)


def _stepped_view(inner: View, outer: View, steps, starts) -> View:
    """inner, a producer input view, seen through outer (see _stepped)."""
    strides = tuple(0 if p is None else k * inner.strides[p] for p, k in steps)
    offset = inner.offset + sum((o * st for o, st in zip(starts, inner.strides)), 0)
    return View(outer.shape, strides, offset)


def _narrows(recipe: _Pointwise) -> bool:
    task = recipe.launch.task
    read = sum(_BYTES[a.dtype] for a in task.args if not a.writes and any(st != 0 for st in a.view.strides))
    return read > _BYTES[recipe.store.inputs[0].type.dtype]


def _same_layout_view(inner: View, outer: View) -> View:
    """inner (see _same_layout) seen through outer, a view of the producer's output."""
    if all(st == 0 for st in inner.strides):
        return View(outer.shape, tuple(0 for _ in outer.shape), inner.offset, outer.pad, outer.fill, None)
    return View(outer.shape, outer.strides, outer.offset, outer.pad, outer.fill, None)


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
        if _widening_cast(info):
            # Each reader widens the b16 value exactly for free; absorbed, the dot would store
            # the f32 copy instead, twice the bytes (Llama's SwiGLU gate, read by the forward
            # and the backward of silu: 23 MB of f32 per layer at 1024 x 5632).
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


def _absorb(launch: Launch, store: TOp, consumer: Launch, info: _Pointwise, read_arg: int, perm: list[int], keep_store: bool = False) -> Launch:
    task = launch.task
    args: list[tuple[str, Arg]] = [(n, a) for n, a in zip(launch.buffers, task.args)]

    def arg_index(name: str, arg: Arg) -> int:
        key = (name, arg)
        if key not in args:
            args.append(key)
        return args.index(key)

    b = TileBuilder()
    b.ops = [op for op in task.body.ops if keep_store or op is not store]
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


def fuse_row_epilogues(program: Program) -> Program:
    """Absorb a pointwise consumer of a row-fused task's output into it (tile/rows.py).

    A layer normalization's output is read twice in GPT-2's training step: in f32 by the next
    residual step or the backward, and through a cast to bf16 (with the affine transform in the
    forward) by the next matrix product. The cast is a pass of its own over 12.6 MB of f32 per
    use; here the row task stores both. Unlike fuse_epilogues the output may have other
    readers: the row task then keeps storing it.
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
            if launch.task.origin != ("rows",) or launch.offsets or any(op.regions for op in launch.task.body.ops):
                continue
            written = {n for n, a in zip(launch.buffers, launch.task.args) if a.writes}
            for store in [op for op in launch.task.body.ops if op.kind == "store"]:
                absorbed = _try_absorb_row(launches, i, store, buffers, readers, writers, written)
                if absorbed is not None:
                    launches[i], j = absorbed
                    del launches[j]
                    changed = True
                    break
            if changed:
                break
        program = replace(program, launches=tuple(launches))
    used = {n for launch in program.launches for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    return replace(program, buffers=tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp))


def _try_absorb_row(launches, i, store, buffers, readers, writers, written):
    launch = launches[i]
    out = launch.buffers[store.attr("arg")]
    buffer = buffers[out]
    if writers.get(out) != [i] or launch.task.args[store.attr("arg")].view != full_view(buffer.type.shape):
        return None
    for j in sorted(set(readers.get(out, []))):
        consumer = launches[j]
        if j < i or consumer.offsets:
            continue
        info = _pointwise(consumer, buffers)
        if info is None:
            continue
        reads = [a for a, n in enumerate(consumer.buffers) if n == out]
        if len(reads) != 1 or consumer.task.args[reads[0]].view != full_view(buffer.type.shape):
            continue
        target = consumer.buffers[info.store.attr("arg")]
        if buffers[target].type.shape != buffer.type.shape or writers.get(target) != [j] or any(r < j for r in readers.get(target, [])):
            continue
        others = {n for n, a in zip(consumer.buffers, consumer.task.args) if not a.writes and n != out}
        if others & written or any(w > i for n in others for w in writers.get(n, [])):
            continue
        keep = buffer.kind is not BufferKind.temp or any(r != j for r in readers.get(out, []))
        return _absorb(launch, store, consumer, info, reads[0], list(range(len(buffer.type.shape))), keep_store=keep), j
    return None


def fuse_sibling_reductions(program: Program, window: int = 4) -> Program:
    """Merge two reductions over the same input into one task: one loop, the input read once.

    The weight and bias gradients of a layer normalization are column sums of dy * xhat and of
    dy: two tasks, each reading dy (12.6 MB of f32 per GPT-2 layer norm). Here they become one
    loop carrying both accumulators, whose loads of dy are shared by value numbering.

    Matched: launches A and B (B within `window` after A), both reductions with one top-level
    loop of the same bounds and attributes and no loop inside it, the same grid and warps,
    sharing a read buffer through the same view. B moves to A's place, so B's inputs must be
    written before A, nothing in between may touch B's output, and B must not read A's.
    """
    changed = True
    while changed:
        changed = False
        launches = list(program.launches)
        for i, a in enumerate(launches):
            if not _single_loop_reduction(a):
                continue
            a_out = {n for n, x in zip(a.buffers, a.task.args) if x.writes}
            a_reads = {(n, x) for n, x in zip(a.buffers, a.task.args) if not x.writes and any(st != 0 for st in x.view.strides)}
            for j in range(i + 1, min(i + 1 + window, len(launches))):
                b = launches[j]
                if not _single_loop_reduction(b) or b.task.grid != a.task.grid or b.task.warps != a.task.warps:
                    continue
                b_reads = {(n, x) for n, x in zip(b.buffers, b.task.args) if not x.writes and any(st != 0 for st in x.view.strides)}
                b_out = {n for n, x in zip(b.buffers, b.task.args) if x.writes}
                if not a_reads & b_reads or {n for n, _ in b_reads} & a_out:
                    continue
                between = launches[i:j]
                if any(n in {m for m, _ in b_reads} for k in between for n, x in zip(k.buffers, k.task.args) if x.writes):
                    continue
                if any(n in b_out for k in launches[i + 1 : j] for n in k.buffers):
                    continue
                merged = _merge_loops(a, b)
                if merged is None:
                    continue
                launches[i] = merged
                del launches[j]
                changed = True
                break
            if changed:
                break
        program = replace(program, launches=tuple(launches))
    return program


def _single_loop_reduction(launch: Launch) -> bool:
    """A reduction whose body is a nest of loops, one per level (one per reduced axis)."""

    def nest(body: Body) -> bool:
        loops = [op for op in body.ops if op.regions]
        return len(loops) <= 1 and all(op.kind == "for" and nest(op.regions[0]) for op in loops)

    body = launch.task.body
    return (
        not launch.offsets
        and launch.task.origin[:1] == ("reduce",)
        and any(op.kind == "for" for op in body.ops)
        and nest(body)
        and not any(op.kind.startswith("atomic") for op in _walk(body))
    )


def _merge_loops(a: Launch, b: Launch) -> Launch | None:
    args: list[tuple[str, Arg]] = []

    def arg_index(name: str, arg: Arg) -> int:
        if (name, arg) not in args:
            args.append((name, arg))
        return args.index((name, arg))

    def replay(target: TileBuilder, launch: Launch, ops, mapping: dict) -> None:
        for op in ops:
            ins = [mapping.get(v, v) for v in op.inputs]
            attrs = dict(op.attrs)
            if op.kind in _MEMORY:
                attrs["arg"] = arg_index(launch.buffers[op.attr("arg")], launch.task.args[op.attr("arg")])
            outs = target.emit(op.kind, ins, attrs)
            mapping.update(zip(op.outputs, outs))

    class Mismatch(Exception):
        pass

    def merge(target: TileBuilder, bodies, maps) -> None:
        """Replay both bodies into target, the loops at this level as one loop."""
        loops = [next((op for op in body.ops if op.kind == "for"), None) for body in bodies]
        if (loops[0] is None) != (loops[1] is None):
            raise Mismatch
        if loops[0] is None:
            for launch, body, mapping in zip((a, b), bodies, maps):
                replay(target, launch, body.ops, mapping)
            return
        if loops[0].attrs != loops[1].attrs:
            raise Mismatch
        for launch, body, loop, mapping in zip((a, b), bodies, loops, maps):
            replay(target, launch, body.ops[: body.ops.index(loop)], mapping)
        bounds = [[maps[k].get(v, v) for v in loops[k].inputs[:3]] for k in range(2)]
        if bounds[0] != bounds[1]:
            raise Mismatch
        n_a = len(loops[0].inputs) - 3
        carries = [maps[k].get(v, v) for k in range(2) for v in loops[k].inputs[3:]]
        iv = TValue(TileType((), INDEX))
        params = [TValue(c.type) for c in carries]
        inner = TileBuilder(cse=True)
        inner_maps = [dict(maps[0]), dict(maps[1])]
        inner_maps[0].update(zip(loops[0].regions[0].params, (iv, *params[:n_a])))
        inner_maps[1].update(zip(loops[1].regions[0].params, (iv, *params[n_a:])))
        merge(inner, [loops[0].regions[0], loops[1].regions[0]], inner_maps)
        results = [inner_maps[k].get(v, v) for k in range(2) for v in loops[k].regions[0].results]
        region = Body((iv, *params), tuple(inner.ops), tuple(results))
        outs = target.emit("for", (*bounds[0], *carries), dict(loops[0].attrs), (region,))
        maps[0].update(zip(loops[0].outputs, outs[:n_a]))
        maps[1].update(zip(loops[1].outputs, outs[n_a:]))
        for launch, body, loop, mapping in zip((a, b), bodies, loops, maps):
            replay(target, launch, body.ops[body.ops.index(loop) + 1 :], mapping)

    builder = TileBuilder(cse=True)
    params = tuple(TValue(v.type) for v in a.task.body.params)
    maps = [dict(zip(launch.task.body.params, params)) for launch in (a, b)]
    try:
        merge(builder, [a.task.body, b.task.body], maps)
    except Mismatch:
        return None
    task = Task(a.task.name, a.task.grid, tuple(x for _, x in args), builder.finish(params), a.task.warps, a.task.origin)
    return Launch(task, tuple(n for n, _ in args))


def fuse_siblings(program: Program, window: int = 32) -> Program:
    """Merge pointwise tasks over the same tile grid that read common inputs (horizontal fusion).

    AdamW updates a parameter's moments and the parameter itself in separate tasks, each
    reading the gradient (the parameter task recomputes the moments from their inputs): 11
    memory passes where one task needs 7. Merged, every distinct load happens once and every
    output is stored. A later task B moves to an earlier task A's place, so: B's inputs are all
    written before A, nothing between them touches B's output, and B reads A's outputs only
    through the views A stores them with, so the merged body passes A's values on directly (the
    parameter update reads the moments just written).
    """
    buffers = {b.name: b for b in program.buffers}
    launches = list(program.launches)
    # Each launch's analysis, kept while the launch is unchanged (keyed by identity, the launch
    # held so its id stays its own), and after a merge the scan resumes a window before it:
    # earlier pairs cannot have changed. Rescanning everything per merge and re-analysing every
    # task was most of a U-Net's compile time (4 million analyses for 467 merges).
    analyses: dict[int, tuple[Launch, _Pointwise | None]] = {}

    def analysis(launch: Launch) -> _Pointwise | None:
        hit = analyses.get(id(launch))
        if hit is None or hit[0] is not launch:
            hit = (launch, _pointwise(launch, buffers, multi=True) if not launch.offsets else None)
            analyses[id(launch)] = hit
        return hit[1]

    start = 0
    changed = True
    while changed:
        changed = False
        for i in range(start, len(launches)):
            a = launches[i]
            info_a = analysis(a)
            if info_a is None or len(launches) <= i + 1:
                continue
            for j in range(i + 1, min(len(launches), i + 1 + window)):
                b = launches[j]
                info_b = analysis(b)
                if info_b is None or b.task.grid != a.task.grid or info_b.store.inputs[0].type.shape != info_a.store.inputs[0].type.shape:
                    continue
                reads_a = {n for n, x in zip(a.buffers, a.task.args) if not x.writes}
                reads_b = {n for n, x in zip(b.buffers, b.task.args) if not x.writes}
                a_out = {n: x for n, x in zip(a.buffers, a.task.args) if x.writes}
                written_between = {n for k in range(i + 1, j) for n, x in zip(launches[k].buffers, launches[k].task.args) if x.writes}
                b_out = {n for n, x in zip(b.buffers, b.task.args) if x.writes}
                touched_between = {n for k in range(i + 1, j) for n in launches[k].buffers}
                forwarded = [x.view == a_out[n].view for n, x in zip(b.buffers, b.task.args) if not x.writes and n in a_out]
                if reads_b & written_between or b_out & touched_between or not all(forwarded):
                    continue
                if not reads_a & reads_b and not forwarded:
                    continue
                launches[i] = _merge(a, info_a, b, info_b)
                del launches[j]
                changed = True
                start = max(0, i - window)
                break
            if changed:
                break
    return replace(program, launches=tuple(launches))


def _merge(a: Launch, info_a: _Pointwise, b: Launch, info_b: _Pointwise) -> Launch:
    """One task computing both pointwise tasks (same grid and tile), with shared loads."""
    args: list[tuple[str, Arg]] = []

    def arg_index(name: str, arg: Arg) -> int:
        key = (name, arg)
        if key not in args:
            args.append(key)
        return args.index(key)

    builder = TileBuilder()
    params = a.task.body.params
    loads: dict[tuple, TValue] = {}
    stored: dict[tuple, TValue] = {}  # (buffer, arg) -> the value a stores there (for b's loads)
    mapping_a: dict[TValue, TValue] = {}
    for launch, info, mapping in ((a, info_a, mapping_a), (b, info_b, dict(zip(b.task.body.params, params)))):
        # b's tile starts are a's (both tasks have the same grid and tile), as a's ops rebuilt here.
        if launch is b:
            mapping.update((sb, mapping_a.get(sa, sa)) for sb, sa in zip(info_b.starts, info_a.starts))
        for op in launch.task.body.ops:
            if op.outputs and set(op.outputs) <= info.coordinate and launch is b:
                continue
            ins = [mapping.get(v, v) for v in op.inputs]
            if op.kind == "load":
                arg = launch.task.args[op.attr("arg")]
                name = launch.buffers[op.attr("arg")]
                if launch is b and (name, replace(arg, writes=True)) in stored:
                    mapping[op.outputs[0]] = stored[(name, replace(arg, writes=True))]
                    continue
                key = (name, arg, tuple(id(v) for v in ins), op.attr("shape"), op.attr("other"))
                if key not in loads:
                    loads[key] = builder.op("load", *ins, arg=arg_index(name, arg), shape=op.attr("shape"), dtype=op.attr("dtype"), other=op.attr("other"))
                mapping[op.outputs[0]] = loads[key]
                continue
            if op.kind == "store":
                arg = launch.task.args[op.attr("arg")]
                builder.emit("store", tuple(ins), {"arg": arg_index(launch.buffers[op.attr("arg")], arg)})
                if launch is a:
                    stored[(launch.buffers[op.attr("arg")], arg)] = ins[0]
                continue
            outs = builder.emit(op.kind, ins, dict(op.attrs), op.regions)
            mapping.update(zip(op.outputs, outs))
    body = Body(params, tuple(builder.ops), ())
    task = Task(a.task.name, a.task.grid, tuple(x for _, x in args), body, a.task.warps, a.task.origin)
    return Launch(task, tuple(n for n, _ in args))
