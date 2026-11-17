"""Tracing. Python runs only to capture the graph and is never called at run time.

Context is a stack. Functions inside scan or value_and_grad are captured as child traces in their
own graphs, and parent values they use are lifted to inputs (a closed Graph allows no implicit references).
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from olaf.frontend.autodiff import grad as graph_grad
from olaf.graph.builder import GraphBuilder, inline, verify
from olaf.graph.composite import element_count
from olaf.graph.dim import Dim
from olaf.graph.interp import interpret
from olaf.graph.ir import Graph, Value
from olaf.graph.types import DType, StateDecl, StateKind, TensorType
from olaf.passes.decompose import decompose


class _Trace:
    def __init__(self, parent: _Trace | None) -> None:
        self.parent = parent
        self.builder = GraphBuilder()
        self.captures: dict[Value, Value] = {}


_stack: list[_Trace] = []


def _current() -> _Trace:
    if not _stack:
        raise RuntimeError("operations need an active trace; call this inside a @olaf.step function")
    return _stack[-1]


def _resolve(tensor: Tensor, trace: _Trace | None = None) -> Value:
    """Make `tensor` usable in `trace`. Values from ancestor traces are lifted to inputs level by level."""
    trace = trace or _current()
    if tensor.trace is trace:
        return tensor.value
    if trace.parent is None:
        raise RuntimeError("value belongs to another trace or to a finished trace")
    outer = _resolve(tensor, trace.parent)
    if outer not in trace.captures:
        trace.captures[outer] = trace.builder.param(outer.type)
    return trace.captures[outer]


@dataclass(frozen=True, eq=False)
class Tensor:
    value: Value
    trace: _Trace = field(repr=False)

    @property
    def type(self) -> TensorType:
        return self.value.type

    @property
    def shape(self) -> tuple[Dim, ...]:
        return self.value.type.shape

    @property
    def dtype(self) -> DType:
        return self.value.type.dtype

    @property
    def rank(self) -> int:
        return self.value.type.rank

    def __add__(self, other):
        return _binary("add", self, other)

    def __radd__(self, other):
        return _binary("add", other, self)

    def __sub__(self, other):
        return _binary("sub", self, other)

    def __rsub__(self, other):
        return _binary("sub", other, self)

    def __mul__(self, other):
        return _binary("mul", self, other)

    def __rmul__(self, other):
        return _binary("mul", other, self)

    def __truediv__(self, other):
        return _binary("div", self, other)

    def __rtruediv__(self, other):
        return _binary("div", other, self)

    def __pow__(self, other):
        return _binary("pow", self, other)

    def __rpow__(self, other):
        return _binary("pow", other, self)

    def __neg__(self):
        return _emit("neg", self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __lt__(self, other):
        return _compare("lt", self, other)

    def __le__(self, other):
        return _compare("le", self, other)

    def __gt__(self, other):
        return _compare("lt", other, self)

    def __ge__(self, other):
        return _compare("le", other, self)

    def __bool__(self):
        raise TypeError("the truth value of a Tensor is unknown until run time; use olaf.cond or olaf.where")

    # With eq=False, == and != fell back to identity and returned a Python bool, so `if x == 0:` traced
    # one branch silently, slipping past __bool__ above. Emitting cmp instead was not chosen: != has
    # no core op (no "ne" comparison, no logical not), and an == that works while != does not is a
    # worse surprise than a clear error. Hashing stays by identity.
    def __eq__(self, other):
        raise TypeError("== on a Tensor is not traced; use olaf.equal (and olaf.where to negate it)")

    def __ne__(self, other):
        raise TypeError("!= on a Tensor is not traced; use olaf.where(olaf.equal(a, b), olaf.constant(False), True)")

    __hash__ = object.__hash__

    def __getitem__(self, key):
        return _getitem(self, key)

    def reshape(self, *shape):
        return reshape(self, shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)


def _wrap(value: Value) -> Tensor:
    return Tensor(value, _current())


def _emit(kind: str, *inputs: Tensor, **attrs) -> Tensor:
    return _wrap(_current().builder.op(kind, *(_resolve(t) for t in inputs), **attrs))


def as_tensor(value, like: Tensor | None = None, dtype: DType | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value
    resolved = dtype or (like.dtype if like is not None else None)
    if resolved is None:
        array = np.asarray(value)
        names = {"float32": DType.f32, "float64": DType.f64, "int32": DType.i32, "int64": DType.i64, "uint32": DType.u32, "bool": DType.bool}
        if array.dtype.name not in names:
            raise TypeError(f"cannot determine the constant dtype: {array.dtype}")
        resolved = names[array.dtype.name]
    if not resolved.is_float:
        # Python numbers take the Tensor's dtype, so `x_i32 * 0.5` would silently become `x * 0`.
        # Rounding into a float dtype is expected; changing the value of an integer constant is not.
        exact = np.asarray(value)
        if not np.array_equal(exact.astype(resolved.numpy), exact):
            raise TypeError(f"{value!r} is not exactly representable as {resolved}; cast the tensor first")
    return _wrap(_current().builder.constant(value, resolved))


def broadcast_shapes(a: Sequence[Dim], b: Sequence[Dim]) -> tuple[Dim, ...]:
    out: list[Dim] = []
    # max in this module is olaf.max for Tensors, so name the builtin explicitly.
    for i in range(1, builtins.max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1
        y = b[-i] if i <= len(b) else 1
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise TypeError(f"cannot broadcast: {tuple(a)} and {tuple(b)}")
    return tuple(reversed(out))


def broadcast_to(x, shape: Sequence[Dim]) -> Tensor:
    x = as_tensor(x)
    shape = tuple(shape)
    if x.shape == shape:
        return x
    offset = len(shape) - x.rank
    return _emit("broadcast", x, shape=shape, dims=tuple(range(offset, len(shape))))


def _binary(kind: str, a, b) -> Tensor:
    a, b = _coerce(a, b)
    shape = broadcast_shapes(a.shape, b.shape)
    return _emit(kind, broadcast_to(a, shape), broadcast_to(b, shape))


def _compare(fn: str, a, b) -> Tensor:
    a, b = _coerce(a, b)
    shape = broadcast_shapes(a.shape, b.shape)
    return _emit("cmp", broadcast_to(a, shape), broadcast_to(b, shape), fn=fn)


def _coerce(a, b) -> tuple[Tensor, Tensor]:
    # Python numbers follow the Tensor's dtype. Tensors are never implicitly promoted, because that
    # would change dtypes before the precision assignment pass sees them.
    if isinstance(a, Tensor):
        return a, as_tensor(b, like=a)
    if isinstance(b, Tensor):
        return as_tensor(a, like=b), b
    raise TypeError("at least one operand must be a Tensor")


def equal(a, b) -> Tensor:
    return _compare("eq", a, b)


def neg(x) -> Tensor:
    return _emit("neg", x)


def abs(x) -> Tensor:  # noqa: A001
    return _emit("abs", x)


def exp(x) -> Tensor:
    return _emit("exp", x)


def log(x) -> Tensor:
    return _emit("log", x)


def sqrt(x) -> Tensor:
    return _emit("sqrt", x)


def rsqrt(x) -> Tensor:
    return _emit("rsqrt", x)


def recip(x) -> Tensor:
    return _emit("recip", x)


def tanh(x) -> Tensor:
    return _emit("tanh", x)


def erf(x) -> Tensor:
    return _emit("erf", x)


def sin(x) -> Tensor:
    return _emit("sin", x)


def cos(x) -> Tensor:
    return _emit("cos", x)


def cast(x, dtype: DType) -> Tensor:
    return x if x.dtype is dtype else _emit("cast", x, dtype=dtype)


def maximum(a, b) -> Tensor:
    return _binary("max", a, b)


def minimum(a, b) -> Tensor:
    return _binary("min", a, b)


def where(cond: Tensor, a, b) -> Tensor:
    a, b = _coerce(a, b)
    shape = broadcast_shapes(broadcast_shapes(cond.shape, a.shape), b.shape)
    return _emit("where", broadcast_to(cond, shape), broadcast_to(a, shape), broadcast_to(b, shape))


def _axes(x: Tensor, axis) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(x.rank))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(sorted(a % x.rank if x.rank else a for a in raw))


def sum(x: Tensor, axis=None) -> Tensor:  # noqa: A001
    return _emit("reduce", x, fn="sum", axes=_axes(x, axis))


def max(x: Tensor, axis=None) -> Tensor:  # noqa: A001
    return _emit("reduce", x, fn="max", axes=_axes(x, axis))


def mean(x: Tensor, axis=None) -> Tensor:
    axes = _axes(x, axis)
    return sum(x, axes) / _wrap(element_count(_current().builder, [x.shape[a] for a in axes], x.dtype))


def einsum(spec: str, *inputs: Tensor) -> Tensor:
    return _emit("contract", *inputs, spec=spec)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.rank < 2 or b.rank != a.rank:
        raise TypeError(f"matmul needs tensors of equal rank >= 2: {a.type}, {b.type}")
    batch = "abcdefghijklmnopqrstuvw"[: a.rank - 2]
    return einsum(f"{batch}xz,{batch}zy->{batch}xy", a, b)


def reshape(x: Tensor, shape: Sequence[Dim]) -> Tensor:
    return _emit("reshape", x, shape=tuple(shape))


def permute(x: Tensor, perm: Sequence[int]) -> Tensor:
    return _emit("permute", x, perm=tuple(perm))


def transpose(x: Tensor, a: int, b: int) -> Tensor:
    perm = list(range(x.rank))
    perm[a], perm[b] = perm[b], perm[a]
    return permute(x, perm)


def concat(inputs: Sequence[Tensor], axis: int = 0) -> Tensor:
    return _emit("concat", *inputs, axis=axis % inputs[0].rank)


def pad(x: Tensor, low: Sequence[Dim], high: Sequence[Dim], value: float = 0.0) -> Tensor:
    return _emit("pad", x, low=tuple(low), high=tuple(high), value=float(value))


def _getitem(x: Tensor, key) -> Tensor:
    key = key if isinstance(key, tuple) else (key,)
    if key.count(Ellipsis) > 1:
        raise IndexError("at most one ... is allowed")
    if Ellipsis in key:
        at = key.index(Ellipsis)
        key = (*key[:at], *(slice(None) for _ in range(x.rank - len(key) + 1)), *key[at + 1 :])
    if len(key) > x.rank:
        raise IndexError(f"too many indices: {key} / {x.type}")
    key = (*key, *(slice(None) for _ in range(x.rank - len(key))))
    starts: list[Dim] = []
    stops: list[Dim] = []
    drop: list[int] = []
    for i, (k, d) in enumerate(zip(key, x.shape)):
        if isinstance(k, slice):
            if k.step not in (None, 1):
                raise IndexError("slice step must be 1")
            start = 0 if k.start is None else k.start
            stop = d if k.stop is None else k.stop
            starts.append(d + start if isinstance(start, int) and start < 0 else start)
            stops.append(d + stop if isinstance(stop, int) and stop < 0 else stop)
        elif isinstance(k, int):
            start = d + k if k < 0 else k
            starts.append(start)
            stops.append(start + 1)
            drop.append(i)
        else:
            raise IndexError(f"unsupported index: {k!r}; use olaf.take to index with a tensor")
    out = _emit("slice", x, starts=tuple(starts), stops=tuple(stops))
    if drop:
        out = reshape(out, [d for i, d in enumerate(out.shape) if i not in drop])
    return out


def take(x: Tensor, index: Tensor, axis: int = 0, batch_dims: int = 0) -> Tensor:
    return _emit("gather", x, index, axis=axis % x.rank, batch_dims=batch_dims)


def scatter_add(target: Tensor, index: Tensor, updates: Tensor, axis: int = 0, batch_dims: int = 0) -> Tensor:
    return _emit("scatter_add", target, index, updates, axis=axis % target.rank, batch_dims=batch_dims)


def constant(value, dtype: DType | None = None) -> Tensor:
    return as_tensor(value, dtype=dtype)


def full(shape: Sequence[Dim], value, dtype: DType = DType.f32) -> Tensor:
    return broadcast_to(as_tensor(value, dtype=dtype), shape)


def zeros(shape: Sequence[Dim], dtype: DType = DType.f32) -> Tensor:
    return full(shape, 0, dtype)


def ones(shape: Sequence[Dim], dtype: DType = DType.f32) -> Tensor:
    return full(shape, 1, dtype)


def iota(shape, axis: int = 0, dtype: DType = DType.i32) -> Tensor:
    shape = (shape,) if not isinstance(shape, (tuple, list)) else tuple(shape)
    return _wrap(_current().builder.op("iota", shape=shape, axis=axis, dtype=dtype))


def rng_uniform(key: Tensor, counter: Tensor, shape: Sequence[Dim], dtype: DType = DType.f32) -> Tensor:
    return _emit("rng", key, counter, shape=tuple(shape), dtype=dtype)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return _emit("softmax", x, axis=axis % x.rank)


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """approximate="none" is the exact erf form; "tanh" is the approximation used by GPT-2."""
    return _emit("gelu", x, approximate=approximate)


def silu(x: Tensor) -> Tensor:
    """x * sigmoid(x) (torch.nn.functional.silu; computed in f32 for bf16 inputs, rounded once)."""
    return _emit("silu", x)


def layernorm(x: Tensor, weight: Tensor | None = None, bias: Tensor | None = None, axes=-1, eps: float = 1e-5) -> Tensor:
    inputs = (x,) if weight is None else (x, weight, bias)
    return _emit("layernorm", *inputs, axes=_axes(x, axes), eps=float(eps))


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    return _emit("attention", *((q, k, v) if mask is None else (q, k, v, mask)))


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    return _emit("cross_entropy", logits, labels)


# pytree


def tree_flatten(tree) -> tuple[list, Any]:
    if isinstance(tree, dict):
        keys = sorted(tree)
        pairs = [tree_flatten(tree[k]) for k in keys]
        return [leaf for leaves, _ in pairs for leaf in leaves], ("dict", tuple(keys), tuple(s for _, s in pairs))
    if isinstance(tree, (tuple, list)):
        pairs = [tree_flatten(v) for v in tree]
        return [leaf for leaves, _ in pairs for leaf in leaves], (type(tree).__name__, None, tuple(s for _, s in pairs))
    if tree is None:
        return [], ("none", None, ())
    return [tree], ("leaf", None, ())


def tree_unflatten(structure, leaves: Sequence):
    iterator = iter(leaves)

    def build(s):
        kind, keys, children = s
        if kind == "leaf":
            return next(iterator)
        if kind == "none":
            return None
        built = [build(c) for c in children]
        if kind == "dict":
            return dict(zip(keys, built))
        return tuple(built) if kind == "tuple" else built

    return build(structure)


def tree_map(fn: Callable, tree):
    leaves, structure = tree_flatten(tree)
    return tree_unflatten(structure, [fn(leaf) for leaf in leaves])


def _subtrace(types: Sequence[TensorType]) -> tuple[_Trace, list[Tensor]]:
    sub = _Trace(_current())
    params = [Tensor(sub.builder.param(t), sub) for t in types]
    return sub, params


def _results_in(sub: _Trace, leaves: Sequence) -> list[Value]:
    out: list[Value] = []
    for leaf in leaves:
        tensor = leaf if isinstance(leaf, Tensor) else as_tensor(leaf)
        out.append(_resolve(tensor, sub))
    return out


def _tensors(leaves: Sequence) -> list[Tensor]:
    return [as_tensor(leaf) for leaf in leaves]


def _outer_values(sub: _Trace) -> list[Value]:
    # Sources of the lifted values, as seen from the parent, so they can be parent op inputs directly.
    return list(sub.captures)


def scan(fn: Callable, init, xs=None, length: Dim | None = None, reverse: bool = False):
    """Iterate fn(carry, x) -> (carry, y). This expresses repeated layers without unrolling them.

    Outer values used by the body pass through as carry. Every iteration reads the same value, so
    semantics are unchanged and scan inputs stay in the form [carry..., xs...].
    """
    init_leaves, init_structure = tree_flatten(init)
    xs_leaves, xs_structure = tree_flatten(xs)
    init_tensors, xs_tensors = _tensors(init_leaves), _tensors(xs_leaves)
    if length is None:
        if not xs_tensors:
            raise ValueError("length is required when there are no xs")
        length = xs_tensors[0].shape[0]

    types = [t.type for t in init_tensors] + [t.type.with_shape(t.shape[1:]) for t in xs_tensors]
    sub, params = _subtrace(types)
    n_carry = len(init_tensors)
    carry_in = tree_unflatten(init_structure, params[:n_carry])
    x_in = tree_unflatten(xs_structure, params[n_carry:])

    _stack.append(sub)
    try:
        carry_out, y_out = fn(carry_in, x_in)
        carry_leaves, carry_structure = tree_flatten(carry_out)
        if carry_structure != init_structure:
            raise TypeError(f"scan changed the carry structure: {carry_structure} / {init_structure}")
        y_leaves, y_structure = tree_flatten(y_out)
        carry_values = _results_in(sub, carry_leaves)
        y_values = _results_in(sub, y_leaves)
    finally:
        _stack.pop()

    outer, inner = _outer_values(sub), list(sub.captures.values())
    body = sub.builder.finish(
        [*carry_values, *inner, *y_values],
        [*(p.value for p in params[:n_carry]), *inner, *(p.value for p in params[n_carry:])],
    )
    inputs = [*(_resolve(t) for t in init_tensors), *outer, *(_resolve(t) for t in xs_tensors)]
    attrs = {"n_carry": n_carry + len(inner), "length": length, "reverse": reverse}
    outputs = _current().builder.emit("scan", inputs, attrs, [body])
    carry_result = [_wrap(v) for v in outputs[:n_carry]]
    y_result = [_wrap(v) for v in outputs[n_carry + len(inner) :]]
    return tree_unflatten(init_structure, carry_result), tree_unflatten(y_structure, y_result)


def repeat(block: Callable, n: Dim, carry, xs=None):
    """Apply a layer of the same structure n times.

    block(carry) or block(carry, x) returns the next carry. xs holds per-layer weights stacked on the first dimension.
    """
    if xs is None:
        out, _ = scan(lambda c, _: (block(c), None), carry, None, length=n)
    else:
        out, _ = scan(lambda c, x: (block(c, x), None), carry, xs, length=n)
    return out


def cond(pred: Tensor, then_fn: Callable, else_fn: Callable, *args):
    """Data-dependent branch. Python's if is evaluated at trace time, so use this instead."""
    leaves, structure = tree_flatten(args)
    tensors = _tensors(leaves)
    types = [t.type for t in tensors]
    regions: list[Graph] = []
    structures: list = []
    inputs: list[Value] = [_resolve(pred)]
    n_then = 0
    for i, fn in enumerate((then_fn, else_fn)):
        sub, params = _subtrace(types)
        _stack.append(sub)
        try:
            out_leaves, out_structure = tree_flatten(fn(*tree_unflatten(structure, params)))
            results = _results_in(sub, out_leaves)
        finally:
            _stack.pop()
        inner = list(sub.captures.values())
        regions.append(sub.builder.finish(results, [*(p.value for p in params), *inner]))
        structures.append(out_structure)
        inputs += [*(_resolve(t) for t in tensors), *_outer_values(sub)]
        if i == 0:
            n_then = len(tensors) + len(inner)
    if structures[0] != structures[1]:
        raise TypeError(f"cond branch outputs have different structures: {structures}")
    outputs = _current().builder.emit("cond", inputs, {"n_then": n_then}, regions)
    return tree_unflatten(structures[0], [_wrap(v) for v in outputs])


def while_loop(cond_fn: Callable, body_fn: Callable, init):
    """Data-dependent loop. Outer values are added to the state and passed through."""
    leaves, structure = tree_flatten(init)
    tensors = _tensors(leaves)
    types = [t.type for t in tensors]

    cond_sub, cond_params = _subtrace(types)
    _stack.append(cond_sub)
    try:
        cond_results = _results_in(cond_sub, [cond_fn(tree_unflatten(structure, cond_params))])
    finally:
        _stack.pop()

    body_sub, body_params = _subtrace(types)
    _stack.append(body_sub)
    try:
        out_leaves, out_structure = tree_flatten(body_fn(tree_unflatten(structure, body_params)))
        if out_structure != structure:
            raise TypeError(f"while changed the state structure: {out_structure} / {structure}")
        body_results = _results_in(body_sub, out_leaves)
    finally:
        _stack.pop()

    cond_caps, body_caps = list(cond_sub.captures.items()), list(body_sub.captures.items())
    # Condition and body receive the same state, so values used by only one still need an input.
    cond_filler = [Value(o.type) for o, _ in body_caps]
    body_filler = [Value(o.type) for o, _ in cond_caps]
    cond_inner = [i for _, i in cond_caps]
    body_inner = [i for _, i in body_caps]
    cond_region = cond_sub.builder.finish(cond_results, [*(p.value for p in cond_params), *cond_inner, *cond_filler])
    body_region = body_sub.builder.finish(
        [*body_results, *body_filler, *body_inner],
        [*(p.value for p in body_params), *body_filler, *body_inner],
    )
    inputs = [*(_resolve(t) for t in tensors), *(o for o, _ in cond_caps), *(o for o, _ in body_caps)]
    outputs = _current().builder.emit("while", inputs, {}, [cond_region, body_region])
    return tree_unflatten(structure, [_wrap(v) for v in outputs[: len(tensors)]])


def value_and_grad(fn: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False):
    """Build a function returning (value, gradient). With has_aux=True, fn returns (loss, aux) and the result is ((loss, aux), gradient).

    fn is captured as a child trace, differentiated on the graph, then inlined into the parent.
    """
    single = isinstance(argnums, int)
    positions = (argnums,) if single else tuple(argnums)

    def wrapped(*args):
        leaves, structure = tree_flatten(args)
        tensors = _tensors(leaves)
        sub, params = _subtrace([t.type for t in tensors])
        _stack.append(sub)
        try:
            out = fn(*tree_unflatten(structure, params))
            loss, aux = out if has_aux else (out, None)
            aux_leaves, aux_structure = tree_flatten(aux)
            results = _results_in(sub, [loss, *aux_leaves])
        finally:
            _stack.pop()

        # Find each argument's range in the flattened inputs.
        wrt: list[int] = []
        spans: dict[int, tuple[int, Any]] = {}
        offset = 0
        for i, arg in enumerate(args):
            arg_leaves, arg_structure = tree_flatten(arg)
            if i in positions:
                spans[i] = (len(arg_leaves), arg_structure)
                wrt.extend(range(offset, offset + len(arg_leaves)))
            offset += len(arg_leaves)

        inner = list(sub.captures.values())
        graph = sub.builder.finish(results, [*(p.value for p in params), *inner])
        differentiated = graph_grad(graph, wrt)
        outer_inputs = [*(_resolve(t) for t in tensors), *_outer_values(sub)]
        outputs = [_wrap(v) for v in inline(_current().builder, differentiated, outer_inputs)]

        grads_flat = outputs[len(results) :]
        grads = []
        cursor = 0
        for i in positions:
            count, arg_structure = spans[i]
            grads.append(tree_unflatten(arg_structure, grads_flat[cursor : cursor + count]))
            cursor += count
        grad_out = grads[0] if single else tuple(grads)
        value = outputs[0]
        if has_aux:
            return (value, tree_unflatten(aux_structure, outputs[1 : len(results)])), grad_out
        return value, grad_out

    return wrapped


def grad(fn: Callable, argnums: int | Sequence[int] = 0):
    transformed = value_and_grad(fn, argnums)
    return lambda *args: transformed(*args)[1]


@dataclass(frozen=True)
class State:
    """Reference to resident state. Reads and writes become IR read_state / write_state."""

    decl: StateDecl

    def read(self) -> Tensor:
        return _wrap(_current().builder.op("read_state", state=self.decl))

    def write(self, value: Tensor) -> None:
        _current().builder.emit("write_state", [_resolve(value)], {"state": self.decl})


def state(name: str, type: TensorType, kind: StateKind = StateKind.param) -> State:
    return State(StateDecl(name, type, kind))


@dataclass(frozen=True, eq=False)
class Traced:
    """A captured function. Eager execution exists only for debugging and verification."""

    graph: Graph
    input_structure: Any
    output_structure: Any

    @cached_property
    def decomposed(self) -> Graph:
        return decompose(self.graph)

    def eager(self, *args, state: dict | None = None, dims: dict | None = None, observer=None):
        """Return (results, updated resident state)."""
        leaves, structure = tree_flatten(args)
        if structure != self.input_structure:
            raise TypeError(f"input structure differs: {structure} / {self.input_structure}")
        results, new_state = interpret(self.decomposed, leaves, state, dims=dims, observer=observer)
        return tree_unflatten(self.output_structure, results), new_state


def trace(fn: Callable, *specs) -> Traced:
    """Capture `fn`. `specs` gives the argument types as a pytree of TensorType."""
    leaves, structure = tree_flatten(specs)
    root = _Trace(None)
    params = [Tensor(root.builder.param(t), root) for t in leaves]
    if _stack:
        raise RuntimeError("trace cannot be called inside a trace")
    _stack.append(root)
    try:
        out = fn(*tree_unflatten(structure, params))
        out_leaves, out_structure = tree_flatten(out)
        results = _results_in(root, out_leaves)
    finally:
        _stack.pop()
    graph = root.builder.finish(results)
    verify(graph)
    return Traced(graph, structure, out_structure)


@dataclass(frozen=True, eq=False)
class Step:
    """A step function captured with @olaf.step."""

    fn: Callable

    def trace(self, *specs) -> Traced:
        return trace(self.fn, *specs)


def step(fn: Callable) -> Step:
    return Step(fn)

