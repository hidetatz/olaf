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

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from olaf.graph.dim import Dim, product
from olaf.graph.types import DType
from olaf.tile.builder import BINARY, INDEX, UNARY, TileBuilder
from olaf.tile.ir import Arg, Body, Buffer, BufferKind, Launch, Program, Task, TileType, TOp, TValue, View, contiguous_strides, full_view

CHEAP_OPS = 3
# A producer whose output is wider than its inputs (bf16 in, f32 out) is recomputed in every
# reader up to this cost: reading its inputs again moves fewer bytes than writing and reading
# its output (GPT-2's gelu tanh, shared by forward and backward after CSE: 50 MB of f32 per layer).
RECOMPUTE_OPS = 8
# A row task takes the task before it (fuse_row_prologues) when that loads no more operands than
# this: a whole row of every operand is held at once. A depthwise stencil inlined in a residual sum
# (ConvNeXt, 49 taps) made its row tasks 5 to 10 times slower.
PROLOGUE_LOADS = 8
# Concats of more inputs than this take no inlined producers (_Rewriter).
CONCAT_INLINE_INPUTS = 4

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
    gathers: bool = False  # it reads through an index (a gather, possibly with pointwise ops after it)
    read_bytes: float = 0  # per output element, over its operands other than small broadcast ones
    out_bytes: int = 0
    operands: int = 0  # how many of them are full-size


def fuse(program: Program) -> Program:
    readers = Counter(name for launch in program.launches for name, arg in zip(launch.buffers, launch.task.args) if not arg.writes)
    writers = Counter(name for launch in program.launches for name, arg in zip(launch.buffers, launch.task.args) if arg.writes)
    buffers = {b.name: b for b in program.buffers}
    recipes: dict[str, _Pointwise] = {}
    launches: list[Launch] = []
    for launch in program.launches:
        # Inlining merges readers: a ReLU's mask (cmp) read once by its select (where), both
        # reading the pre-activation x, leaves one task reading x where there were two, and x
        # can then go in as well. Counted once, x stayed materialized beside the selected
        # output (ResNet-18 wrote each residual sum twice).
        # Only those merged reads go in on the later rounds; the rest stays as one round left
        # it (a load a widening cast kept from its producer, now inlined here, stays a load).
        only = None
        while (inlined := _inline_into(launch, recipes, readers, buffers, only)) is not launch:
            lowered = _recount(readers, launch, inlined, recipes, buffers)
            launch = inlined
            only = {name for name in lowered if readers[name] == 1 and name in launch.buffers}
            if not only:
                break
        info = _pointwise(launch, buffers, indexed=True)
        if info is not None:
            out = launch.buffers[info.store.attr("arg")]
            if buffers[out].kind is BufferKind.temp and writers[out] == 1:
                recipes[out] = info
        launches.append(launch)
    return _without_dead(program, launches)


def _reads(launch: Launch) -> Counter:
    return Counter(name for name, arg in zip(launch.buffers, launch.task.args) if not arg.writes)


def _recount(readers: Counter, old: Launch, new: Launch, recipes: dict[str, _Pointwise], buffers: dict[str, Buffer]) -> set[str]:
    """Update reader counts after `old` was rewritten as `new`; a recipe left without readers
    is a launch _without_dead drops, so its own reads are not counted either. Returns the
    buffers whose count went down."""
    delta = _reads(new)
    delta.subtract(_reads(old))
    readers.update(delta)
    gone = [name for name, d in delta.items() if d < 0]
    lowered = set(gone)
    dead: set[str] = set()  # a buffer two dead producers read is queued twice
    while gone:
        name = gone.pop()
        recipe = recipes.get(name)
        if name in dead or readers[name] != 0 or recipe is None or buffers[name].kind is not BufferKind.temp:
            continue
        dead.add(name)
        for read, count in _reads(recipe.launch).items():
            readers[read] -= count
            gone.append(read)
            lowered.add(read)
    return lowered


def _pointwise(launch: Launch, buffers: dict[str, Buffer], *, multi: bool = False, indexed: bool = False) -> _Pointwise | None:
    """A pointwise task (see the module docstring). multi: also accept several stores (of
    sibling fusion's merged tasks), all at the same tile; info.store is then the first.
    indexed: also accept gathers (load_indexed at the store's tile, its index loaded there),
    which only fuse's producer inlining knows how to move into a reader."""
    body = launch.task.body
    stores = [op for op in body.ops if op.kind == "store"]
    kinds = _POINTWISE | {"load", "store"} | ({"load_indexed"} if indexed else set())
    if (len(stores) != 1 and not (multi and stores)) or any(op.regions or op.kind not in kinds for op in body.ops):
        return None
    store = stores[0]
    starts = store.inputs[1:]
    for other in stores:
        arg = launch.task.args[other.attr("arg")]
        target = buffers[launch.buffers[other.attr("arg")]]
        # With multi, also each element of an interior once: an activation stored straight into the
        # next convolution's padded grid. (Producer inlining reads a whole buffer at its own coordinates.)
        whole = arg.view == full_view(target.type.shape) or multi and _interior(arg.view, target.type.shape)
        if not whole or other.inputs[1:] != starts or other.inputs[0].type.shape != store.inputs[0].type.shape:
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
    loaded = {op.outputs[0] for op in body.ops if op.kind == "load"}
    for op in body.ops:
        if op.kind == "load" and (op.inputs != starts or op.attr("shape") != store.inputs[0].type.shape):
            return None
        if op.kind == "load_indexed" and (op.inputs[1:] != starts or op.inputs[0] not in loaded or op.attr("shape") != store.inputs[0].type.shape):
            return None
        if op.kind == "load_indexed":
            continue
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
    gathers = any(op.kind == "load_indexed" for op in body.ops)
    # For duplication, an operand counts by its distinct elements: a max pooling's backward
    # selects by its output broadcast over each window, a quarter of the elements it reads.
    distinct = [a for a in args if not a.writes and not _small(a.view, total)]
    weighted = sum(_BYTES[a.dtype] * _fraction(a.view, total) for a in distinct)
    full = sum(1 for a in distinct if _fraction(a.view, total) == 1)
    return _Pointwise(launch, starts, coordinate, store, cost, out_bytes >= sum(reads) and all(out_bytes > r for r in reads), gathers, weighted, out_bytes, full)


def _interior(view: View, shape) -> bool:
    """The view is a block of a contiguous buffer of `shape`, its axes in order."""
    return view.pad is None and view.rank == len(shape) and view.strides == contiguous_strides(shape)


def _fraction(view: View, total: Dim) -> float:
    n = _elements(view)
    return n / total if isinstance(n, int) and isinstance(total, int) and total else 1.0


def _elements(view: View) -> Dim:
    """Distinct elements a view addresses (axes of stride 0 repeat them)."""
    count: Dim = 1
    for d, st in zip(view.shape, view.strides):
        if st != 0:
            count = count * d
    return count


def _duplication_pays(recipe: _Pointwise, readers: int) -> bool:
    """Whether `readers` readers each rereading the producer's operands move no more bytes
    than writing its output once and reading it `readers` times. A ReLU's gradient selecting
    between the sum of two incoming gradients and zero is three cheap ops on three tensors:
    duplicated into a batch norm backward's reduction and its elementwise pass, it read six
    tensors where materializing it wrote one and read two.
    A producer of one operand is left alone: its readers mostly read that operand anyway (an
    attention row's fully-masked test on the row maximum), and narrowing ones have _narrows."""
    if recipe.widens or recipe.operands < 2:
        return True
    return readers * recipe.read_bytes <= (readers + 1) * recipe.out_bytes


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

    def __init__(self, launch: Launch, recipes: dict[str, _Pointwise], readers: Counter, buffers: dict[str, Buffer], only: set[str] | None = None) -> None:
        self.launch = launch
        self.only = only
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
        # Nothing is inlined into a concat of many inputs: tile.normalize.eliminate_concats lets
        # its producers store into their slices, and one inlined producer makes the concat
        # compute, so none can. An RNN stacking its 128 states (each also read by the next
        # step) recomputed them there, every element selecting among 128 inlined steps (167 of
        # 226 ms). A concat of a few (U-Net's skip connections, q, k and v) still takes its
        # producers, which saves their writes when it cannot be eliminated.
        # Casts still go in: the concat stores them through its producers (eliminate_concats'
        # _select_casts; a scan's bf16 weight gradients stacked as f32).
        inputs = sum(1 for a in launch.task.args if not a.writes and a.view.pad is not None)
        self.casts_only = launch.task.origin == ("concat",) and inputs > CONCAT_INLINE_INPUTS
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
        if self.only is not None and name not in self.only:
            return None
        recipe = self.recipes.get(name)
        view = self.launch.task.args[op.attr("arg")].view
        smaller = self.only is not None and recipe is not None and math.prod(recipe.store.inputs[0].type.shape) < math.prod(op.attr("shape"))
        if smaller and any(_contains(self.launch.task.body, k) for k in ("exp", "log", "tanh", "erf", "sin", "cos", "pow")):
            # Found by measuring, not derived: a depthwise stencil (25 taps, made with a smaller
            # tile) inside SiLU's gradient took milliseconds more per task (EfficientNet 52 to
            # 76 ms), while ConvNeXt's 49 taps inside residual adds gain (26.6 to 25.2).
            return None
        if recipe is not None and self.casts_only and [o.kind for o in recipe.launch.task.body.ops if o.kind not in ("store",) and not set(o.outputs) <= recipe.coordinate] != ["load", "cast"]:
            return None
        if recipe is not None and recipe.gathers:
            return self.gather_ok(op, recipe, name, view)
        if recipe is None or (view.source is None and not _same_layout(recipe, name, self.buffers) and _stepped(recipe, view, self.buffers[name]) is None):
            return None
        if view.source is None and self.readers[name] > 1 and _narrows(recipe):
            # Its output stays for the other readers (a bf16 copy for a dot): reading the wider
            # inputs again here would move more bytes than reading that output.
            return None
        if any(a.view.pad is not None for a in recipe.launch.task.args if not a.writes):
            return None
        cheap = recipe.cost <= CHEAP_OPS or (recipe.widens and recipe.cost <= RECOMPUTE_OPS)
        if self.readers[name] > 1 and not (cheap and _duplication_pays(recipe, self.readers[name])):
            return None
        if self.into_dot and not _widening_cast(recipe):
            return None
        return recipe

    def gather_ok(self, op: TOp, recipe: _Pointwise, name: str, view: View) -> _Pointwise | None:
        """A gather is evaluated in its reader instead of materialized (GAT's messages: rows of
        a [32768, 256] table at 557056 edge sources, 570 MB in f32, written once and read by a
        scatter_add and a reduction). The reader reads it whole and as laid out (its view is the
        output's full view), so the gather's own views and coordinates apply unchanged; what
        it rereads is the index once per row and the table's rows, from cache when they repeat.
        Not into a gather (indices of indices) nor into a dot (whose operand loads are staged)."""
        if op.kind != "load" or self.into_dot or self.buffers[name].strides is not None or view != full_view(self.buffers[name].type.shape):
            return None
        # Only gathers that expand their table: then materializing moves more bytes than
        # rereading the table does. A language model's embedding (fewer tokens than vocabulary
        # rows) stays written once; inlined, it went down the residual adds into every layer's
        # norm (Llama's RMSNorm then no longer matched row fusion).
        tables = [recipe.launch.buffers[o.attr("arg")] for o in _walk(recipe.launch.task.body) if o.kind == "load_indexed"]
        sizes = [_elements(full_view(self.buffers[n].type.shape)) for n in (name, *tables)]
        if not all(isinstance(n, int) for n in sizes) or not all(sizes[0] > n for n in sizes[1:]):
            return None  # symbolic sizes: not known to expand
        if any(a.view.pad is not None for a in recipe.launch.task.args if not a.writes):
            return None
        if self.readers[name] > 1 and not (recipe.cost <= CHEAP_OPS or (recipe.widens and recipe.cost <= RECOMPUTE_OPS)):
            return None
        return recipe

    def indexed_ok(self, op: TOp) -> bool:
        """A gather may evaluate its producer at the gathered positions (GPT-2's loss reads 4096
        of 4096 x 50257 shifted logits): no iota in the producer (its coordinates come from the
        index there), and no reduction or dot in the consumer, whose store drops every lane
        outside the tensor, so no mask is needed."""
        name = self.launch.buffers[op.attr("arg")]
        recipe = self.recipes.get(name)
        if recipe is None or recipe.gathers or any(o.kind == "iota" for o in recipe.launch.task.body.ops):
            return False
        return not (_contains(self.launch.task.body, "reduce") or _contains(self.launch.task.body, "dot"))

    def body(self, body: Body, subst: dict[TValue, TValue]) -> Body:
        # Loads are merged (TileBuilder's cse): a producer can arrive twice, inlined in a reader
        # that was itself inlined here and inlined directly. A ReLU's select and its mask both
        # read the batch-normalized pre-activation, cheap enough to go into both; once they
        # are one task, each copy loaded the convolution output again.
        b = TileBuilder(cse=True)
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
            if op.kind in ("store", "atomic_add", "atomic_add_indexed") or op.regions:
                b.forget_loads()
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
            if op.kind == "load_indexed":
                arg = producer.task.args[op.attr("arg")]
                slot = self.arg_index(producer.buffers[op.attr("arg")], arg)
                out = b.op("load_indexed", *ins[:1], *starts, arg=slot, shape=tile, dtype=op.attr("dtype"), axis_map=op.attr("axis_map"), other=op.attr("other"))
            elif op.kind == "load" and recipe.gathers:
                arg = producer.task.args[op.attr("arg")]
                slot = self.arg_index(producer.buffers[op.attr("arg")], arg)
                out = b.op("load", *starts, arg=slot, shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
            elif op.kind == "load":
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


def _cast_recipe(launch: Launch, buffers: dict[str, Buffer]) -> _Pointwise | None:
    """The task as a pointwise recipe, also when it writes its whole output through a permuted
    view (a cast that merges attention heads): only an epilogue can take that one, as it stores
    through the consumer's own view; producer inlining reads a pointwise output at its
    logical coordinates."""
    info = _pointwise(launch, buffers)
    if info is not None:
        return info
    body = launch.task.body
    stores = [op for op in body.ops if op.kind == "store"]
    if len(stores) != 1 or any(op.regions or op.kind not in _POINTWISE | {"load", "store"} for op in body.ops):
        return None
    store = stores[0]
    view = launch.task.args[store.attr("arg")].view
    target = buffers[launch.buffers[store.attr("arg")]]
    # Every element of the target exactly once: its contiguous axes, reordered.
    axes = sorted((st, d) for st, d in zip(view.strides, view.shape) if d != 1)
    whole = sorted((st, d) for st, d in zip(contiguous_strides(target.type.shape), target.type.shape) if d != 1)
    if view.pad is not None or view.offset != 0 or axes != whole:
        return None
    loads = [op for op in body.ops if op.kind == "load"]
    if any(op.inputs != store.inputs[1:] or op.attr("shape") != store.inputs[0].type.shape for op in loads):
        return None
    scalars = _scalar_constants(body)
    return _Pointwise(launch, tuple(store.inputs[1:]), _coordinate_values(body, scalars), store, 0)


def _narrowing_cast(recipe: _Pointwise) -> bool:
    """The task only casts one f32 load to bf16."""
    ops = [op for op in recipe.launch.task.body.ops if op.kind not in ("store",) and not set(op.outputs) <= recipe.coordinate]
    kinds = [op.kind for op in ops if op.kind != "full"]
    if kinds != ["load", "cast"]:
        return False
    load = next(op for op in ops if op.kind == "load")
    cast = next(op for op in ops if op.kind == "cast")
    return load.attr("dtype") is DType.f32 and cast.attr("dtype") is DType.bf16 and recipe.store.inputs[0] is cast.outputs[0]


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


def _inline_into(launch: Launch, recipes, readers, buffers, only: set[str] | None = None) -> Launch:
    rewriter = _Rewriter(launch, recipes, readers, buffers, only=only)
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
            if launch is None or not origin or not (origin[0] in ("contract", "depthwise_rows") or origin[0].startswith("attention")) or launch.offsets:
                continue
            stores = [op for op in launch.task.body.ops if op.kind == "store"]
            if origin[0] in ("attention_dkv", "depthwise_rows"):
                # dQ, stored row tile by row tile inside the loop over them; a stencil's rows.
                stores += [op for loop in launch.task.body.ops for region in loop.regions for op in region.ops if op.kind == "store"]
            for store in stores:
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
    if buffer.kind is not BufferKind.temp or writers.get(out) != [i] or not readers.get(out):
        return None
    # With other readers, only a narrowing cast is absorbed, and the original store stays: the
    # task then writes both copies. A fused attention's f32 output is also read by its
    # backward's D = rowsum(dO * O) (which keeps the f32 values: in bf16, the rounding would
    # reach dS = P (dP - D) through the cancellation), and the separate cast pass read the f32
    # copy again to write the bf16 one in the heads-merged layout (T5: 2.5 ms per step).
    keep = len(readers[out]) > 1
    if keep:
        if shared_cheap_only:
            return None
        casts = [j for j in readers[out] if launches[j] is not None and j > i and (r := _cast_recipe(launches[j], buffers)) is not None and _narrowing_cast(r)]
        if not casts:
            return None
        j = casts[0]
    else:
        j = readers[out][0]
    consumer = launches[j]
    if consumer is None or consumer.offsets or j < i:
        return None
    info = _pointwise(consumer, buffers)
    if info is None and not shared_cheap_only:
        # An attention's gradients are cast straight into the heads-merged layout.
        info = _cast_recipe(consumer, buffers)
        if info is not None and not _narrowing_cast(info):
            info = None
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
    if len(reads) != 1:
        return None
    written = launch.task.args[store.attr("arg")].view
    read = consumer.task.args[reads[0]].view
    target = consumer.buffers[info.store.attr("arg")]
    target_view = consumer.task.args[info.store.attr("arg")].view
    perm = _permutation(read, buffer.type.shape) if written == full_view(buffer.type.shape) else None
    if perm is not None:
        # The consumer reads the output whole, possibly with its axes permuted: perm[j] is
        # the consumer axis of buffer axis j. Its other views and its output are permuted the
        # same way, so the consumer can be evaluated on the producer's tile.
        # The consumer writes its whole target in its own coordinates, possibly through a permuted
        # view (a cast of attention heads [B, H, S, D] into [B, S, H, D] for the output projection).
        if tuple(target_view.shape[perm[j]] for j in range(len(perm))) != buffer.type.shape or target_view.pad is not None or product(target_view.shape) != product(buffers[target].type.shape):
            return None
        views = {a: _permuted(consumer.task.args[a].view, perm) for a in range(len(consumer.task.args))}
        iota_axes = {a: j for j, a in enumerate(perm)}
    else:
        # A dot over merged row labels (from_graph._merged_contract) writes [8320, 384] where the
        # consumer sees [128, 65, 384]: the consumer's axes are merged the same way. ViT's
        # projections otherwise each took their bias in a pass of their own.
        groups = None if keep else _merge_groups(read, written)
        if groups is None:
            return None
        views = {a: _merged_view(consumer.task.args[a].view, groups) for a in range(len(consumer.task.args))}
        if any(v is None for v in views.values()):
            return None
        # As in _merged_contract: without a source map the target is never row-aligned by
        # normalize.align_rows, so it must not need to be.
        if not views[info.store.attr("arg")].is_contiguous or target_view.offset != 0 or product(target_view.shape) != product(buffers[target].type.shape) or not _aligned_rows(target_view):
            return None
        if any(op.kind == "iota" for op in consumer.task.body.ops):
            return None
        iota_axes = {}
    # The consumer's other inputs must be ready when the producer runs.
    others = {n for n, a in zip(consumer.buffers, consumer.task.args) if not a.writes and n != out}
    if any(w > i for n in others for w in writers.get(n, [])):
        return None
    # The consumer's store moves up to launch i: nothing up to the consumer may read the target's
    # old value (a parameter updated in place and still read by the backward), nor write it.
    if any(i <= r < j for r in readers.get(target, [])) or any(i <= w < j for w in writers.get(target, [])):
        return None
    return _absorb(launch, store, consumer, info, reads[0], views, iota_axes, keep_store=keep), j


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


def _merge_groups(view: View, merged: View) -> list[int] | None:
    """How many consecutive axes of `view` make up each axis of `merged`, when both are the whole
    of one contiguous buffer, else None."""
    if not view.is_contiguous or not merged.is_contiguous or view.offset != 0 or merged.offset != 0 or not all(isinstance(d, int) for d in view.shape + merged.shape):
        return None
    groups, axis = [], 0
    for d in merged.shape:
        size, first = 1, axis
        while axis < view.rank and (size < d or axis == first):
            size *= view.shape[axis]
            axis += 1
        if size != d:
            return None
        groups.append(axis - first)
    return groups if axis == view.rank else None


def _merged_view(view: View, groups: list[int]) -> View | None:
    """`view` with each group of consecutive axes as one, when they are one run in memory."""
    if view.pad is not None:
        return None
    shape, strides, axis = [], [], 0
    for n in groups:
        sizes, steps = view.shape[axis : axis + n], view.strides[axis : axis + n]
        if any(steps[k] != steps[k + 1] * sizes[k + 1] for k in range(n - 1)):
            return None
        shape.append(math.prod(sizes))
        strides.append(steps[-1])
        axis += n
    return View(tuple(shape), tuple(strides), view.offset)


def _aligned_rows(view: View) -> bool:
    return isinstance(view.shape[-1], int) and view.shape[-1] % 8 == 0


def _absorb(launch: Launch, store: TOp, consumer: Launch, info: _Pointwise, read_arg: int, views: dict[int, View], iota_axes: dict[int, int], keep_store: bool = False) -> Launch:
    """views: the consumer's views over the producer's axes; iota_axes: consumer axis -> producer axis."""
    task = launch.task
    args: list[tuple[str, Arg]] = [(n, a) for n, a in zip(launch.buffers, task.args)]

    def arg_index(name: str, arg: Arg) -> int:
        key = (name, arg)
        if key not in args:
            args.append(key)
        return args.index(key)

    home = next(body for body in (task.body, *(r for op in task.body.ops for r in op.regions)) if any(op is store for op in body.ops))
    b = TileBuilder()
    b.ops = [op for op in home.ops if keep_store or op is not store]
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
                moved = replace(arg, view=views[a])
                out = b.op("load", *starts, arg=arg_index(consumer.buffers[a], moved), shape=tile, dtype=op.attr("dtype"), other=op.attr("other"))
        elif op.kind == "iota":
            axis = iota_axes[op.attr("axis")]
            out = b.op("iota", starts[axis], shape=tile, axis=axis)
        elif op.kind in ("full", "broadcast") and op.outputs[0].type.shape == p_tile and p_tile != ():
            out = b.op(op.kind, *ins, **{**dict(op.attrs), "shape": tile})
        else:
            out = b.op(op.kind, *ins, **dict(op.attrs))
        mapped[op.outputs[0]] = out
    result = mapped[info.store.inputs[0]]
    target = consumer.buffers[info.store.attr("arg")]
    target_arg = consumer.task.args[info.store.attr("arg")]
    b.emit("store", (result, *starts), {"arg": arg_index(target, replace(target_arg, view=views[info.store.attr("arg")]))})
    body = Body(home.params, tuple(b.ops), home.results)
    if home is not task.body:
        ops = tuple(replace(op, regions=tuple(body if r is home else r for r in op.regions)) for op in task.body.ops)
        body = Body(task.body.params, ops, task.body.results)
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
        return _absorb(launch, store, consumer, info, reads[0], {a: arg.view for a, arg in enumerate(consumer.task.args)}, {a: a for a in range(len(buffer.type.shape))}, keep_store=keep), j
    return None


class _Prologue(_Rewriter):
    """Replaces a row task's loads of `name` by the task that computed it, keeping the value."""

    def __init__(self, launch: Launch, name: str, recipe: _Pointwise, buffers: dict[str, Buffer]) -> None:
        super().__init__(launch, {name: recipe}, Counter(), buffers)
        self.name = name
        self.value: TValue | None = None
        self.at: tuple[TValue, ...] = ()

    def allowed(self, op: TOp) -> _Pointwise | None:
        return self.recipes[self.name] if op.kind == "load" and self.launch.buffers[op.attr("arg")] == self.name else None

    def inline(self, b, load, recipe, starts, index=None) -> TValue:
        if self.value is None:
            self.value, self.at = super().inline(b, load, recipe, starts, index), tuple(starts)
        return self.value


def fuse_row_prologues(program: Program, window: int = 16) -> Program:
    """Merge the pointwise task whose output a row-fused task reads into it, output included.

    A residual sum is read by the layer normalization that follows and by much else (the next
    residual sum, the backward), so it is written; the norm, a pass of its own, then read all of
    it back (12.6 MB of f32 per norm of a DiT). The row task computes the sum itself, stores it,
    and normalizes the row it holds. The mirror image of fuse_row_epilogues, and run after
    fuse_epilogues: a task that a dot can absorb is better there.

    The merged task takes the row task's place, so nothing between the two reads the sum or
    writes what it is computed from; or the pointwise task's, where the row task's other
    inputs must have been written.
    """
    launches = list(program.launches)
    buffers = {b.name: b for b in program.buffers}
    written_by: dict[str, int] = Counter(n for launch in launches for n, a in zip(launch.buffers, launch.task.args) if a.writes)
    i = 0
    while i < len(launches):
        merged = _try_prologue(launches, i, buffers, written_by, window)
        if merged is None:
            i += 1
            continue
        new, j, place = merged
        launches[place] = new
        del launches[i if place == j else j]
        # The merged task sits at j, or at i - 1 once j is gone; it may take another producer.
        i = j if place == j else i - 1
    return replace(program, launches=tuple(launches))


def _try_prologue(launches, i, buffers, written_by, window) -> tuple[Launch, int, int] | None:
    launch = launches[i]
    if launch.task.origin != ("rows",) or launch.offsets or any(op.regions for op in launch.task.body.ops):
        return None
    for name, arg in zip(launch.buffers, launch.task.args):
        if arg.writes or written_by[name] != 1 or arg.view != full_view(buffers[name].type.shape):
            continue
        if sum(n == name for n in launch.buffers) != 1:
            continue
        loads = [op for op in launch.task.body.ops if op.kind == "load" and launch.buffers[op.attr("arg")] == name]
        if any(op.kind == "load_indexed" and launch.buffers[op.attr("arg")] == name for op in launch.task.body.ops):
            continue
        if not loads or any(op.inputs != loads[0].inputs or op.attr("shape") != loads[0].attr("shape") for op in loads):
            continue
        j = next((k for k in range(i - 1, max(-1, i - 1 - window), -1) if any(n == name and a.writes for n, a in zip(launches[k].buffers, launches[k].task.args))), None)
        if j is None or launches[j].offsets:
            continue
        producer = launches[j]
        info = _pointwise(producer, buffers)
        if info is None or any(a.view.pad is not None for a in producer.task.args):
            continue
        if sum(op.kind == "load" for op in producer.task.body.ops) > PROLOGUE_LOADS:
            continue
        p_reads = {n for n, a in zip(producer.buffers, producer.task.args) if not a.writes}
        reads = {n for n, a in zip(launch.buffers, launch.task.args) if not a.writes} - {name}
        writes = {n for n, a in zip(launch.buffers, launch.task.args) if a.writes}
        between = launches[j + 1 : i]
        touched = {n for k in between for n in k.buffers}
        written = {n for k in between for n, a in zip(k.buffers, k.task.args) if a.writes}
        if name in p_reads or writes & p_reads:
            continue
        if name not in touched and not p_reads & written:
            place = i
        elif not reads & written and not writes & touched:
            place = j
        else:
            continue
        rewriter = _Prologue(launch, name, info, buffers)
        body = rewriter.body(launch.task.body, {})
        slot = rewriter.arg_index(name, replace(arg, writes=True))
        store = TOp("store", (rewriter.value, *rewriter.at), (), (("arg", slot),))
        return rewriter.finish(Body(body.params, (*body.ops, store), body.results)), j, place
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
    parameter update reads the moments just written). Or A moves down to B's, under the same
    conditions the other way around: the bias corrections of T5's AdamW come after the moments of
    its embedding and before its update.

    A merged task's stores that only it read are dropped at the end (a tied embedding's gradient,
    summed from three parts and read by the moments: 66 MB).
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
                if not all(forwarded) or not reads_a & reads_b and not forwarded:
                    continue
                if not (reads_b & written_between or b_out & touched_between):
                    launches[i] = _merge(a, info_a, b, info_b)
                    del launches[j]
                elif not (reads_a & written_between or set(a_out) & touched_between):
                    # A moves down to B instead.
                    launches[j] = _merge(a, info_a, b, info_b)
                    del launches[i]
                else:
                    continue
                changed = True
                start = max(0, i - window)
                break
            if changed:
                break
    return _without_dead_stores(replace(program, launches=tuple(launches)))


def _without_dead_stores(program: Program) -> Program:
    """Drop pointwise tasks' stores to temps that nothing reads (other stores stay in the task)."""
    read = {n for launch in program.launches for n, x in zip(launch.buffers, launch.task.args) if not x.writes} | set(program.outputs)
    kinds = {b.name: b.kind for b in program.buffers}
    launches = []
    for launch in program.launches:
        task = launch.task
        dead = {k for k, (n, x) in enumerate(zip(launch.buffers, task.args)) if x.writes and kinds[n] is BufferKind.temp and n not in read}
        ops = tuple(op for op in task.body.ops if not (op.kind == "store" and op.attr("arg") in dead))
        if not dead or any(op.regions for op in task.body.ops) or not any(op.kind == "store" for op in ops):
            launches.append(launch)
            continue
        used = sorted({op.attr("arg") for op in ops if op.kind in _MEMORY})
        body = _renumber(Body(task.body.params, ops, task.body.results), {old: new for new, old in enumerate(used)})
        launches.append(replace(launch, task=replace(task, args=tuple(task.args[k] for k in used), body=body), buffers=tuple(launch.buffers[k] for k in used)))
    used = {n for launch in launches for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    return replace(program, launches=tuple(launches), buffers=tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp))


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
