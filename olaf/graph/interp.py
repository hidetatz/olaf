"""NumPy eager interpreter: the numerical reference for every transformation.

Speed is not a goal. Each output is checked against its declared type, catching drift between type rules and semantics.
Composite ops are not executed. With only decomposed core ops as the reference, the
decomposition rules themselves get tested against the reference implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.special import erf

from olaf.graph.dim import as_symbol, bind, evaluate
from olaf.graph.ir import Graph, Op, Value, bf16_round, bf16_values, symbol_table
from olaf.graph.ops import OPS
from olaf.graph.types import DType, TensorType

Observer = Callable[[Value, np.ndarray], None]


def interpret(
    graph: Graph,
    inputs: Sequence,
    state: Mapping[str, np.ndarray] | None = None,
    *,
    dims: Mapping[str, int] | None = None,
    observer: Observer | None = None,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Run the graph and return (results, updated resident state). `state` is not modified.

    read_state returns the value at the start of the step regardless of op order (see builder.verify).
    State that is only written (an initialization step) needs no value in `state`.
    `observer` is called for every computed value, to extract intermediates.
    """
    if len(inputs) != len(graph.params):
        raise ValueError(f"wrong number of inputs: {len(inputs)} / {len(graph.params)}")
    arrays = [np.asarray(a) for a in inputs]
    initial = dict(state or {})
    env: dict[str, int] = {}
    table = symbol_table(graph)
    for name, value in (dims or {}).items():
        # An unknown name is almost always a typo, and accepting it would leave the real symbol unbound.
        if name not in table:
            raise ValueError(f"the graph has no dimension named {name}")
        bind(env, table[name], value)
    read = {op.attr("state").name for op in graph.ops if op.kind == "read_state"}
    typed = [*zip((p.type for p in graph.params), arrays)]
    for decl in graph.states():
        if decl.name in initial:
            typed.append((decl.type, np.asarray(initial[decl.name])))
        elif decl.name in read:
            raise ValueError(f"no value for resident state {decl.name}")
    _bind_dims(env, typed)
    run = _Run(env, initial, observer)
    results = run.graph(graph, arrays)
    return results, {**initial, **run.written}


def _bind_dims(env: dict[str, int], typed: Sequence[tuple[TensorType, np.ndarray]]) -> None:
    for t, array in typed:
        if array.ndim != t.rank:
            raise ValueError(f"wrong rank: {array.shape} / {t}")
        for d, n in zip(t.shape, array.shape):
            # Symbols are not solved from expressions like S//2; _Run.check verifies those values later.
            if (s := as_symbol(d)) is not None:
                bind(env, s, n)


class _Run:
    def __init__(self, env: dict[str, int], state: dict[str, np.ndarray], observer: Observer | None) -> None:
        self.env = env
        self.state = state
        self.written: dict[str, np.ndarray] = {}
        self.observer = observer

    def shape(self, dims) -> tuple[int, ...]:
        return tuple(evaluate(d, self.env) for d in dims)

    def check(self, value: Value, array: np.ndarray) -> np.ndarray:
        """Check an op result against its type. bf16 results are rounded here, once per op.

        Elementwise ops compute in f32 on bf16 inputs and are then rounded to bf16, as a device
        does. Ops whose definition is "exact, then rounded" (contract, reduce sum, cast) produce
        the f64 value and are rounded from it directly: rounding to f32 first would round twice.
        """
        expected = self.shape(value.type.shape)
        if value.type.dtype is DType.bf16 and array.dtype in (np.float32, np.float64):
            array = bf16_round(array)
        if array.shape != expected or array.dtype != value.type.dtype.numpy:
            raise TypeError(f"value does not match type: {array.dtype}{array.shape} / {value.type} = {expected}")
        return array

    def graph(self, graph: Graph, inputs: Sequence[np.ndarray]) -> list[np.ndarray]:
        values: dict[Value, np.ndarray] = {}
        for param, array in zip(graph.params, inputs):
            values[param] = self.check(param, array)
            if self.observer:
                self.observer(param, array)
        for op in graph.ops:
            if OPS[op.kind].composite:
                raise ValueError(f"cannot execute composite op {op.kind}; run passes.decompose first")
            handler = _HANDLERS.get(op.kind)
            if handler is None:
                raise NotImplementedError(f"eager execution of {op.kind} is not implemented")
            outputs = handler(self, op, [values[v] for v in op.inputs])
            for value, array in zip(op.outputs, outputs, strict=True):
                values[value] = self.check(value, array)
                if self.observer:
                    self.observer(value, array)
        return [values[r] for r in graph.results]


_HANDLERS: dict[str, Callable[[_Run, Op, list[np.ndarray]], list[np.ndarray]]] = {}


def _handler(*names: str):
    def wrap(fn):
        for name in names:
            _HANDLERS[name] = fn
        return fn

    return wrap


def _elementwise(name: str, fn) -> None:
    _HANDLERS[name] = lambda run, op, args: [fn(*args)]


_elementwise("neg", np.negative)
_elementwise("abs", np.abs)
_elementwise("exp", np.exp)
_elementwise("log", np.log)
_elementwise("sqrt", np.sqrt)
_elementwise("rsqrt", lambda x: 1 / np.sqrt(x))
_elementwise("recip", lambda x: 1 / x)
_elementwise("tanh", np.tanh)
_elementwise("erf", erf)
_elementwise("sin", np.sin)
_elementwise("cos", np.cos)
_elementwise("add", np.add)
_elementwise("sub", np.subtract)
_elementwise("mul", np.multiply)
_elementwise("div", np.divide)
_elementwise("max", np.maximum)
_elementwise("min", np.minimum)
_elementwise("pow", np.power)
_elementwise("where", np.where)


@_handler("cast")
def _cast(run, op, args):
    if op.attr("dtype") is DType.bf16:
        return [args[0].astype(np.float64)]  # rounded once by check
    return [args[0].astype(op.attr("dtype").numpy)]


@_handler("cmp")
def _cmp(run, op, args):
    fn = {"eq": np.equal, "lt": np.less, "le": np.less_equal}[op.attr("fn")]
    return [fn(*args)]


@_handler("reduce")
def _reduce(run, op, args):
    x, axes = args[0], op.attr("axes")
    if op.attr("fn") == "sum":
        # Without an explicit dtype, integer sums widen to i64 and disagree with the declared type.
        if op.inputs[0].type.dtype is DType.bf16:
            return [np.sum(x, axis=axes, dtype=np.float64)]  # rounded once by check
        if x.dtype == np.float32:
            return [np.sum(x, axis=axes, dtype=np.float64).astype(np.float32)]
        return [np.sum(x, axis=axes, dtype=x.dtype)]
    return [np.max(x, axis=axes)]


@_handler("contract")
def _contract(run, op, args):
    # optimize=True sends binary products to BLAS via tensordot. A naive einsum loop is too
    # slow to verify anything at GPT-2 scale.
    #
    # f32 accumulates in f64 and then rounds: a reference must not depend on BLAS summation order.
    # GPT-2 logits sit near -100 and differ by only about 3 across the vocabulary. Accumulating in
    # f32 grows the error in those differences, which drive the loss, to about 1e-4 (a norm ratio shows only 3e-7).
    if op.inputs[0].type.dtype is DType.bf16:
        return [np.einsum(op.attr("spec"), *(a.astype(np.float64) for a in args), optimize=True)]  # rounded once by check
    if args[0].dtype == np.float32:
        wide = np.einsum(op.attr("spec"), *(a.astype(np.float64) for a in args), optimize=True)
        return [wide.astype(np.float32)]
    return [np.einsum(op.attr("spec"), *args, optimize=True)]


def conv_reference(x: np.ndarray, w: np.ndarray, stride, padding, dilation, lhs_dilation, groups) -> np.ndarray:
    """conv in NumPy, in x's dtype: dilate the input, pad (or crop), then one product per kernel
    offset over strided windows."""
    n = x.ndim - 2
    if any(d > 1 for d in lhs_dilation):
        shape = (*x.shape[:2], *((s - 1) * d + 1 for s, d in zip(x.shape[2:], lhs_dilation)))
        dilated = np.zeros(shape, x.dtype)
        dilated[(slice(None), slice(None), *(slice(None, None, d) for d in lhs_dilation))] = x
        x = dilated
    crop = tuple(slice(max(-lo, 0), s - max(-hi, 0)) for s, (lo, hi) in zip(x.shape[2:], padding))
    x = x[(slice(None), slice(None), *crop)]
    x = np.pad(x, [(0, 0), (0, 0), *((max(lo, 0), max(hi, 0)) for lo, hi in padding)])
    kernel = w.shape[2:]
    out = [(s - ((k - 1) * d + 1)) // st + 1 for s, k, d, st in zip(x.shape[2:], kernel, dilation, stride)]
    N, C, O = x.shape[0], x.shape[1], w.shape[0]
    cg, og = C // groups, O // groups
    y = np.zeros((N, O, *out), x.dtype)
    for offset in np.ndindex(*kernel):
        window = tuple(slice(o * d, o * d + (size - 1) * st + 1, st) for o, d, size, st in zip(offset, dilation, out, stride))
        patch = x[(slice(None), slice(None), *window)]
        for g in range(groups):
            tap = w[(slice(g * og, (g + 1) * og), slice(None), *offset)]
            y[:, g * og : (g + 1) * og] += np.einsum("nc...,oc->no...", patch[:, g * cg : (g + 1) * cg], tap)
    return y


@_handler("conv")
def _conv(run, op, args):
    attrs = [op.attr(k) for k in ("stride", "padding", "dilation", "lhs_dilation", "groups")]
    # As contract: accumulate in f64, round once (bf16 by check).
    wide = conv_reference(*(a.astype(np.float64) for a in args), *attrs)
    return [wide if op.inputs[0].type.dtype in (DType.bf16, DType.f64) else wide.astype(args[0].dtype)]


@_handler("reshape")
def _reshape(run, op, args):
    return [args[0].reshape(run.shape(op.attr("shape")))]


@_handler("permute")
def _permute(run, op, args):
    return [args[0].transpose(op.attr("perm"))]


@_handler("broadcast")
def _broadcast(run, op, args):
    x, shape, dims = args[0], run.shape(op.attr("shape")), op.attr("dims")
    aligned = [1] * len(shape)
    for i, d in enumerate(dims):
        aligned[d] = x.shape[i]
    return [np.broadcast_to(x.reshape(aligned), shape)]


@_handler("slice")
def _slice(run, op, args):
    starts, stops = run.shape(op.attr("starts")), run.shape(op.attr("stops"))
    # NumPy clamps out-of-range slices; a device kernel would read out of bounds instead.
    if not all(0 <= a <= b <= d for a, b, d in zip(starts, stops, args[0].shape)):
        raise IndexError(f"slice out of range: {starts}, {stops} / {args[0].shape}")
    return [args[0][tuple(slice(a, b) for a, b in zip(starts, stops))]]


@_handler("concat")
def _concat(run, op, args):
    return [np.concatenate(args, axis=op.attr("axis"))]


@_handler("pad")
def _pad(run, op, args):
    low, high = run.shape(op.attr("low")), run.shape(op.attr("high"))
    if min((*low, *high), default=0) < 0:
        raise IndexError(f"negative pad width: {low}, {high}")
    return [np.pad(args[0], list(zip(low, high)), constant_values=op.attr("value"))]


def _gather_index(shape_out: tuple[int, ...], index: np.ndarray, axis: int, batch_dims: int) -> tuple:
    """Build the index tuple mapping each gather output element to its input position."""
    grid = np.indices(shape_out, sparse=True)
    n_index = index.ndim - batch_dims
    aligned = index.reshape(
        (*index.shape[:batch_dims], *(1 for _ in range(axis - batch_dims)), *index.shape[batch_dims:], *(1 for _ in shape_out[axis + n_index :]))
    )
    return (*grid[:axis], aligned, *grid[axis + n_index :])


def _check_bounds(index: np.ndarray, size: int) -> None:
    # NumPy wraps negative indices but device implementations will not, so the reference rejects them.
    if index.size and (index.min() < 0 or index.max() >= size):
        raise IndexError(f"index out of range: [{index.min()}, {index.max()}] / {size}")


@_handler("gather")
def _gather(run, op, args):
    x, index = args
    axis, batch_dims = op.attr("axis"), op.attr("batch_dims")
    _check_bounds(index, x.shape[axis])
    return [x[_gather_index(run.shape(op.outputs[0].type.shape), index, axis, batch_dims)]]


@_handler("scatter_add")
def _scatter_add(run, op, args):
    target, index, updates = args
    axis, batch_dims = op.attr("axis"), op.attr("batch_dims")
    _check_bounds(index, target.shape[axis])
    if op.outputs[0].type.dtype in (DType.bf16, DType.f32):
        # As reduce: the exact sum (approximated in f64), rounded once (bf16 by check).
        out = target.astype(np.float64)
        np.add.at(out, _gather_index(updates.shape, index, axis, batch_dims), updates.astype(np.float64))
        return [out if op.outputs[0].type.dtype is DType.bf16 else out.astype(np.float32)]
    out = target.copy()
    np.add.at(out, _gather_index(updates.shape, index, axis, batch_dims), updates)
    return [out]


@_handler("constant")
def _constant(run, op, args):
    data = op.attr("value")
    return [bf16_values(data.array).reshape(data.shape) if data.dtype is DType.bf16 else data.array]


@_handler("iota")
def _iota(run, op, args):
    shape, axis = run.shape(op.attr("shape")), op.attr("axis")
    aligned = [1] * len(shape)
    aligned[axis] = shape[axis]
    line = np.arange(shape[axis], dtype=op.attr("dtype").numpy).reshape(aligned)
    return [np.broadcast_to(line, shape)]


_PHILOX_M0 = np.uint64(0xD2511F53)
_PHILOX_M1 = np.uint64(0xCD9E8D57)
_PHILOX_W0 = np.uint64(0x9E3779B9)
_PHILOX_W1 = np.uint64(0xBB67AE85)
_MASK32 = np.uint64(0xFFFFFFFF)


def philox4x32(counter: np.ndarray, key: np.ndarray, rounds: int = 10) -> np.ndarray:
    """Philox4x32-10. `counter` is [..., 4] and `key` is [2], both u32.

    NumPy's `random.Philox` is 4x64 and its output order is NumPy's own. GPUs multiply 32-bit
    integers fast, so we use 4x32 and define the sequence ourselves to reproduce it on devices later.
    """
    r = [counter[..., i].astype(np.uint64) for i in range(4)]
    k0, k1 = np.uint64(key[0]), np.uint64(key[1])
    for round_index in range(rounds):
        if round_index > 0:
            k0, k1 = (k0 + _PHILOX_W0) & _MASK32, (k1 + _PHILOX_W1) & _MASK32
        p0, p1 = _PHILOX_M0 * r[0], _PHILOX_M1 * r[2]
        hi0, lo0, hi1, lo1 = p0 >> np.uint64(32), p0 & _MASK32, p1 >> np.uint64(32), p1 & _MASK32
        r = [hi1 ^ r[1] ^ k0, lo1, hi0 ^ r[3] ^ k1, lo0]
    return np.stack([x.astype(np.uint32) for x in r], axis=-1)


@_handler("rng")
def _rng(run, op, args):
    # Element i is lane i%4 of block counter + i//4. Each value depends only on its position,
    # so tiling and execution order cannot change the sequence.
    key, counter = args
    shape = run.shape(op.attr("shape"))
    n = int(np.prod(shape, dtype=np.int64))
    blocks = np.arange((n + 3) // 4, dtype=np.uint64)
    carry = blocks
    lanes = []
    for lane in counter:
        total = carry + np.uint64(lane)
        lanes.append(total & _MASK32)
        carry = total >> np.uint64(32)
    words = philox4x32(np.stack(lanes, axis=-1).astype(np.uint32), key).reshape(-1)[:n]
    dtype = op.attr("dtype").numpy
    # Use only the top 24 bits: they fit the f32 significand, so rounding cannot reach 1.0 and [0, 1) holds.
    uniform = (words >> np.uint32(8)).astype(dtype) * dtype.type(2.0**-24)
    return [uniform.reshape(shape)]


@_handler("scan")
def _scan(run, op, args):
    body = op.regions[0]
    n_carry, reverse = op.attr("n_carry"), op.attr("reverse", False)
    length = evaluate(op.attr("length"), run.env)
    carry, xs = list(args[:n_carry]), args[n_carry:]
    ys = [np.empty(run.shape(v.type.shape), v.type.dtype.numpy) for v in op.outputs[n_carry:]]
    for i in reversed(range(length)) if reverse else range(length):
        out = run.graph(body, [*carry, *(x[i] for x in xs)])
        carry = out[:n_carry]
        for buffer, y in zip(ys, out[n_carry:]):
            buffer[i] = y
    return [*carry, *ys]


@_handler("cond")
def _cond(run, op, args):
    n_then = op.attr("n_then")
    if args[0]:
        return run.graph(op.regions[0], args[1 : 1 + n_then])
    return run.graph(op.regions[1], args[1 + n_then :])


@_handler("while")
def _while(run, op, args):
    cond, body = op.regions
    state = list(args)
    while run.graph(cond, state)[0]:
        state = run.graph(body, state)
    return state


@_handler("read_state")
def _read_state(run, op, args):
    return [np.asarray(run.state[op.attr("state").name])]


@_handler("write_state")
def _write_state(run, op, args):
    run.written[op.attr("state").name] = args[0]
    return []
