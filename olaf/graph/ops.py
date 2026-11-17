"""Definitions of core and composite ops (the closed list in the IR chapter of docs/design.md).

An op definition only derives output types from inputs and attributes. Semantics live in the
interpreter, decomposition in composite.py, and differentiation in frontend/autodiff.py.

Binary ops require identical shapes; broadcasting is an explicit broadcast op. Implicit
broadcasting would force fusion and autodiff to reason about restoring shapes everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from olaf.graph.dim import product
from olaf.graph.ir import ConstantData, Graph
from olaf.graph.types import DType, StateDecl, TensorType

Infer = Callable[[list[TensorType], dict, tuple[Graph, ...]], list[TensorType]]


@dataclass(frozen=True)
class OpDef:
    name: str
    infer: Infer
    composite: bool = False
    has_effect: bool = False


OPS: dict[str, OpDef] = {}


def _register(*names: str, composite: bool = False, has_effect: bool = False):
    def wrap(fn: Infer) -> Infer:
        for name in names:
            OPS[name] = OpDef(name, fn, composite, has_effect)
        return fn

    return wrap


def infer_types(kind: str, inputs: list[TensorType], attrs: dict, regions: tuple[Graph, ...] = ()) -> list[TensorType]:
    if kind not in OPS:
        raise ValueError(f"unknown op: {kind}")
    return OPS[kind].infer(inputs, attrs, regions)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise TypeError(message)


def _arity(inputs: list, n: int, kind: str) -> None:
    _check(len(inputs) == n, f"{kind} takes {n} inputs (got {len(inputs)})")


def _maybe_nonnegative(d) -> bool:
    """False only when d is provably negative. Symbolic bounds are checked when the interpreter binds them."""
    return not isinstance(d, int) or d >= 0


def _axes_ok(axes: tuple[int, ...], rank: int) -> bool:
    return len(set(axes)) == len(axes) and all(0 <= a < rank for a in axes)


def _unary(kind: str, *, float_only: bool):
    def infer(inputs, attrs, regions):
        _arity(inputs, 1, kind)
        _check(not float_only or inputs[0].dtype.is_float, f"{kind} is float-only: {inputs[0]}")
        return [inputs[0]]

    return infer


def _binary(kind: str, *, float_only: bool):
    def infer(inputs, attrs, regions):
        _arity(inputs, 2, kind)
        a, b = inputs
        _check(a.shape == b.shape and a.dtype == b.dtype, f"{kind} input types differ: {a} and {b}")
        _check(not float_only or a.dtype.is_float, f"{kind} is float-only: {a}")
        return [a]

    return infer


for _name in ("exp", "log", "sqrt", "rsqrt", "recip", "tanh", "erf", "sin", "cos"):
    OPS[_name] = OpDef(_name, _unary(_name, float_only=True))
for _name in ("neg", "abs"):
    OPS[_name] = OpDef(_name, _unary(_name, float_only=False))
for _name in ("div", "pow"):
    OPS[_name] = OpDef(_name, _binary(_name, float_only=True))
for _name in ("add", "sub", "mul", "max", "min"):
    OPS[_name] = OpDef(_name, _binary(_name, float_only=False))


@_register("cast")
def _infer_cast(inputs, attrs, regions):
    _arity(inputs, 1, "cast")
    return [inputs[0].with_dtype(attrs["dtype"])]


@_register("cmp")
def _infer_cmp(inputs, attrs, regions):
    _arity(inputs, 2, "cmp")
    a, b = inputs
    _check(a.shape == b.shape and a.dtype == b.dtype, f"cmp input types differ: {a} and {b}")
    _check(attrs["fn"] in ("eq", "lt", "le"), f"unknown comparison: {attrs['fn']}")
    return [a.with_dtype(DType.bool)]


@_register("where")
def _infer_where(inputs, attrs, regions):
    _arity(inputs, 3, "where")
    c, a, b = inputs
    _check(c.dtype is DType.bool, f"where condition must be bool: {c}")
    _check(c.shape == a.shape == b.shape and a.dtype == b.dtype, f"where types differ: {c}, {a}, {b}")
    return [a]


@_register("reduce")
def _infer_reduce(inputs, attrs, regions):
    _arity(inputs, 1, "reduce")
    x = inputs[0]
    axes = attrs["axes"]
    _check(attrs["fn"] in ("sum", "max"), f"unknown reduction: {attrs['fn']}")
    _check(_axes_ok(axes, x.rank), f"invalid reduce axes: {axes} / {x}")
    # A bool sum would be a logical or (NumPy sums in the input dtype), which nobody means by "sum".
    _check(x.dtype is not DType.bool, f"cannot reduce bool; cast to an integer first: {x}")
    return [x.with_shape(d for i, d in enumerate(x.shape) if i not in axes)]


def parse_contract(spec: str) -> tuple[list[str], str]:
    """Split `ab,bc->ac` into input and output label strings."""
    _check("->" in spec, f"contract must state output labels: {spec}")
    lhs, out = spec.split("->")
    terms = lhs.split(",")
    labels = [*out, *"".join(terms)]
    _check(all(c.isascii() and c.isalpha() for c in labels), f"labels must be letters: {spec}")
    # Repeated labels in one term (taking a diagonal) are rejected. Their VJP writes back to the
    # diagonal, which needs scatter, and no use from GPT-2 to matrix multiply requires them.
    for term in [*terms, out]:
        _check(len(set(term)) == len(term), f"a term cannot repeat a label: {spec}")
    _check(set(out) <= set("".join(terms)), f"output label missing from inputs: {spec}")
    return terms, out


@_register("contract")
def _infer_contract(inputs, attrs, regions):
    terms, out = parse_contract(attrs["spec"])
    _arity(inputs, len(terms), "contract")
    sizes: dict[str, object] = {}
    for term, t in zip(terms, inputs):
        _check(t.dtype == inputs[0].dtype, f"contract dtypes differ: {t} and {inputs[0]}")
        _check(len(term) == t.rank, f"contract labels {term} do not match the rank of {t}")
        for label, d in zip(term, t.shape):
            _check(sizes.setdefault(label, d) == d, f"contract label {label} has differing sizes: {sizes[label]} and {d}")
    return [inputs[0].with_shape(sizes[c] for c in out)]


@_register("reshape")
def _infer_reshape(inputs, attrs, regions):
    _arity(inputs, 1, "reshape")
    x, shape = inputs[0], attrs["shape"]
    _check(product(x.shape) == product(shape), f"reshape changes the element count: {x.shape} -> {shape}")
    return [x.with_shape(shape)]


@_register("permute")
def _infer_permute(inputs, attrs, regions):
    _arity(inputs, 1, "permute")
    x, perm = inputs[0], attrs["perm"]
    _check(sorted(perm) == list(range(x.rank)), f"invalid permutation: {perm} / {x}")
    return [x.with_shape(x.shape[p] for p in perm)]


@_register("broadcast")
def _infer_broadcast(inputs, attrs, regions):
    _arity(inputs, 1, "broadcast")
    x, shape, dims = inputs[0], attrs["shape"], attrs["dims"]
    _check(len(dims) == x.rank and list(dims) == sorted(set(dims)), f"invalid broadcast dims: {dims}")
    _check(all(0 <= d < len(shape) for d in dims), f"broadcast dims out of range: {dims} / {shape}")
    for i, d in enumerate(dims):
        _check(x.shape[i] in (1, shape[d]), f"cannot broadcast: {x.shape} -> {shape} (dims={dims})")
    return [x.with_shape(shape)]


@_register("slice")
def _infer_slice(inputs, attrs, regions):
    _arity(inputs, 1, "slice")
    x, starts, stops = inputs[0], attrs["starts"], attrs["stops"]
    _check(len(starts) == len(stops) == x.rank, f"slice bounds have the wrong rank: {starts}, {stops} / {x}")
    for start, stop, d in zip(starts, stops, x.shape):
        _check(_maybe_nonnegative(start) and _maybe_nonnegative(stop - start) and _maybe_nonnegative(d - stop), f"slice out of range: {starts}, {stops} / {x}")
    return [x.with_shape(stop - start for start, stop in zip(starts, stops))]


@_register("concat")
def _infer_concat(inputs, attrs, regions):
    _check(len(inputs) >= 1, "concat needs at least one input")
    first, axis = inputs[0], attrs["axis"]
    _check(0 <= axis < first.rank, f"invalid concat axis: {axis} / {first}")
    size = 0
    for t in inputs:
        same = all(a == b for i, (a, b) in enumerate(zip(t.shape, first.shape)) if i != axis)
        _check(t.rank == first.rank and same and t.dtype == first.dtype, f"concat types differ: {t} and {first}")
        size = size + t.shape[axis]
    return [first.with_shape(size if i == axis else d for i, d in enumerate(first.shape))]


def conv_output_size(size, kernel, stride, low, high, dilation, lhs_dilation):
    """One spatial dimension of conv's output: the input dilated by lhs_dilation (zeros between
    elements) and padded by low/high (negative crops), windows of the kernel dilated by
    dilation, every stride-th window."""
    dilated_in = (size - 1) * lhs_dilation + 1 + low + high
    dilated_kernel = (kernel - 1) * dilation + 1
    return (dilated_in - dilated_kernel) // stride + 1


@_register("conv")
def _infer_conv(inputs, attrs, regions):
    # x [N, C, *spatial], w [O, C // groups, *kernel] (PyTorch's layout) -> [N, O, *out].
    _arity(inputs, 2, "conv")
    x, w = inputs
    n = x.rank - 2
    _check(n >= 1 and w.rank == x.rank, f"conv needs x [N, C, *spatial] and w [O, C/groups, *kernel]: {x}, {w}")
    _check(x.dtype == w.dtype and x.dtype.is_float, f"conv dtypes: {x}, {w}")
    stride, padding, dilation, lhs_dilation, groups = (attrs[k] for k in ("stride", "padding", "dilation", "lhs_dilation", "groups"))
    _check(len(stride) == len(padding) == len(dilation) == len(lhs_dilation) == n, f"conv attributes need {n} spatial entries: {attrs}")
    _check(all(isinstance(v, int) and v >= 1 for v in (*stride, *dilation, *lhs_dilation, groups)), f"conv strides, dilations and groups are positive ints: {attrs}")
    spatial = (*x.shape[2:], *w.shape[2:])
    _check(all(isinstance(d, int) for d in (*spatial, x.shape[1], w.shape[0], w.shape[1])), "conv needs static channel and spatial sizes")
    _check(x.shape[1] == w.shape[1] * groups and w.shape[0] % groups == 0, f"conv channels do not match groups={groups}: {x}, {w}")
    out = [conv_output_size(s, k, st, lo, hi, d, ld) for s, k, st, (lo, hi), d, ld in zip(x.shape[2:], w.shape[2:], stride, padding, dilation, lhs_dilation)]
    _check(all(o >= 1 for o in out), f"conv output would be empty: {x}, {w}, {attrs}")
    return [x.with_shape((x.shape[0], w.shape[0], *out))]


@_register("pad")
def _infer_pad(inputs, attrs, regions):
    _arity(inputs, 1, "pad")
    x, low, high = inputs[0], attrs["low"], attrs["high"]
    _check(len(low) == len(high) == x.rank, f"pad widths have the wrong rank: {low}, {high} / {x}")
    _check(all(_maybe_nonnegative(w) for w in (*low, *high)), f"pad widths must be non-negative: {low}, {high}")
    return [x.with_shape(l + d + h for l, d, h in zip(low, x.shape, high))]


def _check_gather_common(x: TensorType, index: TensorType, axis: int, batch_dims: int) -> None:
    _check(index.dtype in (DType.i32, DType.i64), f"index must be an integer: {index}")
    _check(0 <= batch_dims <= axis < x.rank, f"invalid gather axis: axis={axis}, batch_dims={batch_dims} / {x}")
    _check(batch_dims <= index.rank and x.shape[:batch_dims] == index.shape[:batch_dims], f"batch dimensions differ: {x} and {index}")


def gather_shape(x: TensorType, index: TensorType, axis: int, batch_dims: int) -> tuple:
    return (*x.shape[:axis], *index.shape[batch_dims:], *x.shape[axis + 1 :])


@_register("gather")
def _infer_gather(inputs, attrs, regions):
    # batch_dims lets one op express both embedding lookup (0) and cross entropy (like
    # take_along_axis). Without it, the latter would need iota and concat to build indices.
    _arity(inputs, 2, "gather")
    x, index = inputs
    _check_gather_common(x, index, attrs["axis"], attrs["batch_dims"])
    return [x.with_shape(gather_shape(x, index, attrs["axis"], attrs["batch_dims"]))]


@_register("scatter_add")
def _infer_scatter_add(inputs, attrs, regions):
    _arity(inputs, 3, "scatter_add")
    target, index, updates = inputs
    _check_gather_common(target, index, attrs["axis"], attrs["batch_dims"])
    expected = gather_shape(target, index, attrs["axis"], attrs["batch_dims"])
    _check(updates.shape == expected and updates.dtype == target.dtype, f"wrong scatter_add update type: {updates} / {expected}")
    return [target]


@_register("constant")
def _infer_constant(inputs, attrs, regions):
    _arity(inputs, 0, "constant")
    data: ConstantData = attrs["value"]
    return [TensorType(data.shape, data.dtype)]


@_register("iota")
def _infer_iota(inputs, attrs, regions):
    _arity(inputs, 0, "iota")
    _check(0 <= attrs["axis"] < len(attrs["shape"]), f"invalid iota axis: {attrs['axis']}")
    return [TensorType(attrs["shape"], attrs["dtype"])]


@_register("rng")
def _infer_rng(inputs, attrs, regions):
    _arity(inputs, 2, "rng")
    key, counter = inputs
    _check(key == TensorType((2,), DType.u32), f"rng key must be u32[2]: {key}")
    _check(counter == TensorType((4,), DType.u32), f"rng counter must be u32[4]: {counter}")
    _check(attrs["dtype"].is_float, f"rng only produces uniform floats: {attrs['dtype']}")
    return [TensorType(attrs["shape"], attrs["dtype"])]


def _region_signature(graph: Graph) -> tuple[list[TensorType], list[TensorType]]:
    return [p.type for p in graph.params], [r.type for r in graph.results]


@_register("scan")
def _infer_scan(inputs, attrs, regions):
    # Inputs are [carry..., xs...] and outputs [carry..., ys...]. ys stack at the same positions as xs.
    # Even with reverse=True, result i goes to position i (keeps it aligned with xs in backprop).
    _check(len(regions) == 1, "scan has one region")
    n_carry, length = attrs["n_carry"], attrs["length"]
    carry, xs = inputs[:n_carry], inputs[n_carry:]
    params, results = _region_signature(regions[0])
    for x in xs:
        _check(x.rank >= 1 and x.shape[0] == length, f"scan xs leading dimension differs from length={length}: {x}")
    expected = [*carry, *(x.with_shape(x.shape[1:]) for x in xs)]
    _check(params == expected, f"scan body inputs do not match: {params} / {expected}")
    _check(results[:n_carry] == carry, f"scan changes carry types: {results[:n_carry]} / {carry}")
    ys = [y.with_shape((length, *y.shape)) for y in results[n_carry:]]
    return [*carry, *ys]


@_register("cond")
def _infer_cond(inputs, attrs, regions):
    # Inputs are [pred, then..., else...]: each branch captures different outer values.
    _check(len(regions) == 2, "cond has two regions")
    pred, n_then = inputs[0], attrs["n_then"]
    _check(pred == TensorType((), DType.bool), f"cond predicate must be a bool scalar: {pred}")
    then_params, then_results = _region_signature(regions[0])
    else_params, else_results = _region_signature(regions[1])
    _check(then_params == inputs[1 : 1 + n_then], "cond then-branch inputs do not match")
    _check(else_params == inputs[1 + n_then :], "cond else-branch inputs do not match")
    _check(then_results == else_results, f"cond branch outputs differ: {then_results} / {else_results}")
    return then_results


@_register("while")
def _infer_while(inputs, attrs, regions):
    _check(len(regions) == 2, "while has two regions (condition and body)")
    cond_params, cond_results = _region_signature(regions[0])
    body_params, body_results = _region_signature(regions[1])
    _check(cond_params == inputs and body_params == inputs, "while region inputs do not match")
    _check(cond_results == [TensorType((), DType.bool)], f"while condition must be a bool scalar: {cond_results}")
    _check(body_results == inputs, f"while body must preserve state types: {body_results} / {inputs}")
    return inputs


@_register("read_state")
def _infer_read_state(inputs, attrs, regions):
    _arity(inputs, 0, "read_state")
    decl: StateDecl = attrs["state"]
    return [decl.type]


@_register("write_state", has_effect=True)
def _infer_write_state(inputs, attrs, regions):
    _arity(inputs, 1, "write_state")
    decl: StateDecl = attrs["state"]
    _check(inputs[0] == decl.type, f"wrong type for resident state {decl.name}: {inputs[0]} / {decl.type}")
    return []


# Communication ops. P0 only defines types: no eager execution or differentiation until P3 (simulated mesh).


@_register("all_reduce")
def _infer_all_reduce(inputs, attrs, regions):
    _arity(inputs, 1, "all_reduce")
    _check(attrs["fn"] in ("sum", "max"), f"unknown reduction: {attrs['fn']}")
    return [inputs[0]]


def _resized(x: TensorType, axis: int, size) -> TensorType:
    _check(0 <= axis < x.rank, f"invalid axis: {axis} / {x}")
    return x.with_shape(size if i == axis else d for i, d in enumerate(x.shape))


@_register("all_gather")
def _infer_all_gather(inputs, attrs, regions):
    _arity(inputs, 1, "all_gather")
    x, axis = inputs[0], attrs["axis"]
    return [_resized(x, axis, x.shape[axis] * attrs["group_size"])]


@_register("reduce_scatter")
def _infer_reduce_scatter(inputs, attrs, regions):
    _arity(inputs, 1, "reduce_scatter")
    x, axis = inputs[0], attrs["axis"]
    _check(x.shape[axis] % attrs["group_size"] == 0, f"reduce_scatter size is not divisible: {x} / {attrs['group_size']}")
    return [_resized(x, axis, x.shape[axis] // attrs["group_size"])]


@_register("all_to_all")
def _infer_all_to_all(inputs, attrs, regions):
    _arity(inputs, 1, "all_to_all")
    x, split, concat, n = inputs[0], attrs["split_axis"], attrs["concat_axis"], attrs["group_size"]
    _check(x.shape[split] % n == 0, f"all_to_all size is not divisible: {x} / {n}")
    y = _resized(x, split, x.shape[split] // n)
    return [_resized(y, concat, y.shape[concat] * n)]


@_register("send", has_effect=True)
def _infer_send(inputs, attrs, regions):
    _arity(inputs, 1, "send")
    return []


@_register("recv", has_effect=True)
def _infer_recv(inputs, attrs, regions):
    _arity(inputs, 0, "recv")
    return [attrs["type"]]


# Composite ops. Decomposition rules are in composite.py.


@_register("softmax", composite=True)
def _infer_softmax(inputs, attrs, regions):
    _arity(inputs, 1, "softmax")
    _check(inputs[0].dtype.is_float and 0 <= attrs["axis"] < inputs[0].rank, f"invalid softmax input: {inputs[0]}")
    return [inputs[0]]


@_register("gelu", composite=True)
def _infer_gelu(inputs, attrs, regions):
    _arity(inputs, 1, "gelu")
    _check(inputs[0].dtype.is_float, f"gelu is float-only: {inputs[0]}")
    _check(attrs["approximate"] in ("none", "tanh"), f"invalid gelu approximation: {attrs['approximate']}")
    return [inputs[0]]


@_register("silu", composite=True)
def _infer_silu(inputs, attrs, regions):
    _arity(inputs, 1, "silu")
    _check(inputs[0].dtype.is_float, f"silu is float-only: {inputs[0]}")
    return [inputs[0]]


@_register("layernorm", composite=True)
def _infer_layernorm(inputs, attrs, regions):
    # Inputs are x, or x, weight, bias. weight and bias match the normalized dimensions.
    _check(len(inputs) in (1, 3), f"layernorm takes 1 or 3 inputs: {len(inputs)}")
    x, axes = inputs[0], attrs["axes"]
    _check(x.dtype.is_float and _axes_ok(axes, x.rank), f"invalid layernorm input: {x}, axes={axes}")
    expected = TensorType(tuple(x.shape[a] for a in axes), x.dtype)
    for t in inputs[1:]:
        _check(t == expected, f"wrong layernorm weight/bias type: {t} / {expected}")
    return [x]


@_register("attention", composite=True)
def _infer_attention(inputs, attrs, regions):
    # Inputs are q[...,Tq,D], k[...,Tk,D], v[...,Tk,E] and an optional mask bool[...,Tq,Tk].
    _check(len(inputs) in (3, 4), f"attention takes 3 or 4 inputs: {len(inputs)}")
    q, k, v = inputs[:3]
    _check(q.rank >= 2 and q.rank == k.rank == v.rank, f"invalid attention ranks: {q}, {k}, {v}")
    _check(q.dtype == k.dtype == v.dtype and q.dtype.is_float, f"invalid attention dtypes: {q}, {k}, {v}")
    batch = q.shape[:-2]
    _check(k.shape == (*batch, k.shape[-2], q.shape[-1]), f"attention q and k do not match: {q}, {k}")
    _check(v.shape[:-1] == k.shape[:-1], f"attention k and v do not match: {k}, {v}")
    if len(inputs) == 4:
        expected = TensorType((*batch, q.shape[-2], k.shape[-2]), DType.bool)
        _check(inputs[3] == expected, f"wrong attention mask type: {inputs[3]} / {expected}")
    return [v.with_shape((*batch, q.shape[-2], v.shape[-1]))]


@_register("cross_entropy", composite=True)
def _infer_cross_entropy(inputs, attrs, regions):
    # Returns the mean loss (a scalar) from logits[...,V] and labels[...].
    _arity(inputs, 2, "cross_entropy")
    logits, labels = inputs
    _check(logits.dtype.is_float and logits.rank >= 1, f"invalid cross_entropy logits: {logits}")
    _check(labels.shape == logits.shape[:-1] and labels.dtype in (DType.i32, DType.i64), f"invalid cross_entropy labels: {labels} / {logits}")
    return [logits.with_shape(())]
