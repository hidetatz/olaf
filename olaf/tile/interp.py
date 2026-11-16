"""NumPy interpreter for Tile IR.

It runs every grid instance one after another with the exact tile shapes, masks and loop
structure of the program, so it checks the conversion and the tile passes (fusion, tiling
changes) against the eager interpreter before any device code exists. Device lowerings are then
compared with this interpreter task by task, which localizes a failure to one task.

Accumulation follows the program: a dot accumulates in its accumulator dtype tile by tile and a
reduction combines chunk partials in the tile dtype, as a device would. So results differ from
eager (which accumulates f32 in f64) by f32 rounding, and tests compare with the fp32 tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import erf

from olaf.graph.dim import as_symbol, bind, evaluate
from olaf.graph.ir import Graph, Value, symbol_table
from olaf.graph.types import DType
from olaf.tile.ir import Body, BufferKind, Program, Task, TOp, TValue, View

_UNARY = {
    "neg": np.negative,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "recip": lambda x: 1 / x,
    "tanh": np.tanh,
    "erf": erf,
    "sin": np.sin,
    "cos": np.cos,
}
_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
    "floordiv": np.floor_divide,
    "mod": np.mod,
}


def run_program(program: Program, inputs: Sequence, state: Mapping[str, np.ndarray] | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Run the program and return (outputs, updated resident state), like graph.interp.interpret."""
    if len(inputs) != len(program.inputs):
        raise ValueError(f"wrong number of inputs: {len(inputs)} / {len(program.inputs)}")
    state = dict(state or {})
    env: dict[str, int] = {}
    table = _symbols(program)
    storage: dict[str, np.ndarray] = {}
    given = dict(zip(program.inputs, (np.asarray(a) for a in inputs)))
    for b in program.buffers:
        if b.kind is BufferKind.state_in:
            given[b.name] = np.asarray(state[b.state.name])
    for name, array in given.items():
        t = program.buffer(name).type
        if array.ndim != t.rank:
            raise ValueError(f"wrong rank for {name}: {array.shape} / {t}")
        for d, n in zip(t.shape, array.shape):
            if (s := as_symbol(d)) is not None:
                bind(env, table.get(s.name, s), n)
    for b in program.buffers:
        shape = tuple(evaluate(d, env) for d in b.type.shape)
        if b.name in given:
            array = given[b.name]
            if array.shape != shape or array.dtype != b.type.dtype.numpy:
                raise TypeError(f"buffer {b.name} expects {b.type} = {shape}, got {array.dtype}{array.shape}")
            storage[b.name] = np.ascontiguousarray(array).reshape(-1).copy()
        elif b.kind is BufferKind.constant:
            storage[b.name] = np.asarray(b.constant.array).reshape(-1).copy()
        else:
            # NaN (or an extreme integer) instead of zeros, so a task that reads an unwritten element shows up in the result.
            fill = np.nan if b.type.dtype.is_float else 0
            storage[b.name] = np.full(int(np.prod(shape, dtype=np.int64)), fill, b.type.dtype.numpy)
    for launch in program.launches:
        _run_task(launch.task, [storage[n] for n in launch.buffers], env)

    def shaped(name: str) -> np.ndarray:
        b = program.buffer(name)
        return storage[name].reshape(tuple(evaluate(d, env) for d in b.type.shape))

    new_state = {**state, **{b.state.name: shaped(b.name) for b in program.buffers if b.kind is BufferKind.state_out}}
    return [shaped(n) for n in program.outputs], new_state


def _symbols(program: Program) -> dict:
    # Reuse the graph symbol table on a dummy graph holding every buffer type, so constraints on
    # symbols (multiple_of, min, max) are checked at bind time exactly as eager does.
    return symbol_table(Graph(tuple(Value(b.type) for b in program.buffers), (), ()))


def _run_task(task: Task, arrays: list[np.ndarray], env: dict[str, int]) -> None:
    views = [_Bound(a.view, env) for a in task.args]
    grid = tuple(evaluate(d, env) for d in task.grid)
    with np.errstate(all="ignore"):
        for pid in np.ndindex(*grid):
            values = {p: np.int32(i) for p, i in zip(task.body.params, pid)}
            _Run(views, arrays, env, values).body(task.body)


class _Bound:
    """A view with its dimensions evaluated."""

    def __init__(self, view: View, env: dict[str, int]) -> None:
        self.shape = tuple(evaluate(d, env) for d in view.shape)
        self.strides = tuple(evaluate(d, env) for d in view.strides)
        self.offset = evaluate(view.offset, env)
        self.pad = tuple((evaluate(lo, env), evaluate(hi, env)) for lo, hi in view.pad) if view.pad is not None else None
        self.fill = view.fill

    def address(self, coords: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (flat offsets, in-bounds mask, in-padding mask) for logical coordinates."""
        inside = np.ones(np.broadcast_shapes(*(c.shape for c in coords)), bool)
        padded = np.zeros_like(inside)
        offset = np.full(inside.shape, self.offset, np.int64)
        for axis, c in enumerate(coords):
            c = c.astype(np.int64)
            inside &= (c >= 0) & (c < self.shape[axis])
            stored = c
            if self.pad is not None:
                lo, hi = self.pad[axis]
                padded |= (c < lo) | (c >= self.shape[axis] - hi)
                stored = c - lo
            offset = offset + stored * self.strides[axis]
        return offset, inside, padded & inside


def _tile_coords(starts: Sequence[np.ndarray], shape: tuple[int, ...]) -> list[np.ndarray]:
    grids = np.indices(shape, sparse=True) if shape else []
    return [int(s) + g for s, g in zip(starts, grids)]


class _Run:
    def __init__(self, views: list[_Bound], arrays: list[np.ndarray], env: dict[str, int], values: dict[TValue, np.ndarray]) -> None:
        self.views = views
        self.arrays = arrays
        self.env = env
        self.values = values

    def body(self, body: Body) -> list[np.ndarray]:
        for op in body.ops:
            outputs = self.op(op, [self.values[v] for v in op.inputs])
            for v, array in zip(op.outputs, outputs, strict=True):
                array = np.asarray(array)
                if array.shape != v.type.shape or array.dtype != v.type.dtype.numpy:
                    raise TypeError(f"{op.kind} produced {array.dtype}{array.shape}, declared {v.type}")
                self.values[v] = array
        return [self.values[r] for r in body.results]

    def read(self, arg: int, coords: list[np.ndarray], other, dtype: DType) -> np.ndarray:
        view, flat = self.views[arg], self.arrays[arg]
        offset, inside, padded = view.address(coords)
        valid = inside & ~padded
        data = flat[np.where(valid, offset, 0)] if flat.size else np.zeros(offset.shape, flat.dtype)
        out = np.where(valid, data, np.asarray(other).astype(dtype.numpy))
        return np.where(padded, np.asarray(view.fill).astype(dtype.numpy), out).astype(dtype.numpy)

    def indexed_coords(self, op: TOp, index: np.ndarray, starts) -> list[np.ndarray]:
        tile = _tile_coords(starts, index.shape)
        return [index if m == -1 else np.broadcast_to(tile[m], index.shape) for m in op.attr("axis_map")]

    def op(self, op: TOp, args: list[np.ndarray]) -> list[np.ndarray]:
        kind, out_types = op.kind, [v.type for v in op.outputs]
        if kind in _UNARY:
            return [_UNARY[kind](args[0])]
        if kind in _BINARY:
            return [_BINARY[kind](*args).astype(out_types[0].dtype.numpy)]
        if kind == "cmp":
            return [{"eq": np.equal, "lt": np.less, "le": np.less_equal}[op.attr("fn")](*args)]
        if kind == "where":
            return [np.where(*args)]
        if kind == "cast":
            return [args[0].astype(op.attr("dtype").numpy)]
        if kind == "full":
            return [np.full(op.attr("shape"), op.attr("value"), op.attr("dtype").numpy)]
        if kind == "dim":
            return [np.int32(evaluate(op.attr("value"), self.env))]
        if kind == "iota":
            shape, axis = op.attr("shape"), op.attr("axis")
            aligned = [1] * len(shape)
            aligned[axis] = shape[axis]
            line = (np.arange(shape[axis], dtype=np.int32) + args[0]).astype(np.int32).reshape(aligned)
            return [np.broadcast_to(line, shape)]
        if kind == "broadcast":
            shape, dims = op.attr("shape"), op.attr("dims")
            aligned = [1] * len(shape)
            for i, d in enumerate(dims):
                aligned[d] = args[0].shape[i]
            return [np.broadcast_to(args[0].reshape(aligned), shape)]
        if kind == "reshape":
            return [args[0].reshape(op.attr("shape"))]
        if kind == "permute":
            return [args[0].transpose(op.attr("perm"))]
        if kind == "reduce":
            fn = np.sum if op.attr("fn") == "sum" else np.max
            return [fn(args[0], axis=op.attr("axes")).astype(args[0].dtype)]
        if kind == "dot":
            a, b, acc = args
            return [(acc + a.astype(acc.dtype) @ b.astype(acc.dtype)).astype(acc.dtype)]
        if kind == "load":
            return [self.read(op.attr("arg"), _tile_coords(args, op.attr("shape")), op.attr("other"), op.attr("dtype"))]
        if kind == "load_indexed":
            index, *starts = args
            return [self.read(op.attr("arg"), self.indexed_coords(op, index, starts), op.attr("other"), op.attr("dtype"))]
        if kind == "store":
            value, *starts = args
            view, flat = self.views[op.attr("arg")], self.arrays[op.attr("arg")]
            offset, inside, _ = view.address(_tile_coords(starts, value.shape))
            flat[offset[inside]] = np.broadcast_to(value, inside.shape)[inside]
            return []
        if kind == "atomic_add_indexed":
            index, value, *starts = args
            view, flat = self.views[op.attr("arg")], self.arrays[op.attr("arg")]
            offset, inside, _ = view.address(self.indexed_coords(op, index, starts))
            for c, bound in zip(_tile_coords(starts, value.shape), op.attr("bounds")):
                inside = inside & (c < evaluate(bound, self.env))
            np.add.at(flat, offset[inside], value[inside])
            return []
        if kind == "for":
            lo, hi, step, *carries = args
            region = op.regions[0]
            for i in range(int(lo), int(hi), int(step)):
                self.values.update(zip(region.params, [np.int32(i), *carries]))
                carries = self.body(region)
            return carries
        raise NotImplementedError(f"tile interpretation of {kind} is not implemented")
