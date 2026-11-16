"""Lowering of one Tile IR task to an NVVM-dialect MLIR module (per-thread code).

Thread mapping. One CUDA block of T = 32 * task.warps threads runs one grid instance. A tile of
N > 1 elements is spread cyclically: element e (row-major) lives in thread e % T, register slot
e // T. Every tile with the same element count therefore has the same layout, so reshape is
free and elementwise ops work slot by slot; consecutive threads own consecutive elements, so loads
along the last axis coalesce. A tile of one element (scalars, tile starts, row results) is
replicated in every thread instead: it is what every thread needs to address memory.

Ops that move elements between threads (broadcast, permute, reduce, dot) go through shared memory.
This is the general, correct path. Tasks that matter for speed get specialized lowerings on top
of it: matrix-instruction dots (mma.sync with ldmatrix and cp.async pipelining on Ampere) and
shuffle-based reductions. Those decisions come from the capability descriptor, not from here.

Numerics follow the reference interpreters: bf16 values are widened to f32 for each op and
rounded once (round to nearest even); exp, log, tanh, erf, sin, cos and pow call libdevice, whose
f32 results are within 2 ulp. PTX's approximate instructions (tanh.approx has about 2^-11
relative error) would not meet the fp32 tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.builder import INDEX
from olaf.tile.ir import Body, Task, TileType, TOp, TValue, View

DEFAULT_WARPS = 4

_LLVM = {DType.f32: "f32", DType.f64: "f64", DType.bf16: "bf16", DType.i32: "i32", DType.i64: "i64", DType.u32: "i32", DType.bool: "i1"}
_MEMORY_TYPE = {**_LLVM, DType.bool: "i8"}
_BYTES = {DType.f32: 4, DType.f64: 8, DType.bf16: 2, DType.i32: 4, DType.i64: 8, DType.u32: 4, DType.bool: 1}
_LIBDEVICE = {"exp": "exp", "log": "log", "tanh": "tanh", "erf": "erf", "sin": "sin", "cos": "cos", "pow": "pow"}


def element_bytes(dtype: DType) -> int:
    return _BYTES[dtype]


@dataclass
class Lowered:
    """MLIR text of a kernel and how to call it."""

    text: str
    dims: list[Dim]  # runtime i32 parameters after the buffer pointers, in order
    shared_bytes: int  # dynamic shared memory per block
    threads: int
    # Dimensions that must be multiples of 8 for this kernel to be valid (16-byte async copies of
    # bf16). The host checks them per launch and uses `fallback` when one is not.
    aligned: list[Dim] = field(default_factory=list)
    fallback: Lowered | None = None


def _compute_type(dtype: DType) -> str:
    return "f32" if dtype is DType.bf16 else _LLVM[dtype]


@dataclass
class _Value:
    """A lowered tile: one SSA name per register slot, or one replicated name.

    The layout is the cyclic one of _Emitter.coords unless `coords` gives each slot's tile
    coordinates explicitly (with `valid`, the optional per-slot guard). Explicit layouts come from
    matrix instructions, whose accumulator fragments put elements in fixed (row, col) positions;
    elementwise ops, cast, permute and unit reshapes then work on them without moving data.
    """

    regs: list[str]
    replicated: bool
    type: TileType
    coords: list[list[str]] | None = None
    valid: list[str | None] | None = None


@dataclass
class _Emitter:
    task: Task
    use_matrix: bool = True
    lines: list[str] = field(default_factory=list)
    counter: int = 0
    dims: dict[str, tuple[int, Dim]] = field(default_factory=dict)
    externs: set[str] = field(default_factory=set)
    shared_bytes: int = 0
    constants: dict[tuple[str, str], str] = field(default_factory=dict)
    values: dict[TValue, _Value] = field(default_factory=dict)
    coords_cache: dict[tuple[int, ...], list[list[str]]] = field(default_factory=dict)
    blocks: int = 0
    entry_constants: list[str] = field(default_factory=list)
    aligned: list[Dim] = field(default_factory=list)

    @property
    def threads(self) -> int:
        return 32 * self.task.warps

    # -- basic emission -------------------------------------------------------------------------

    def tmp(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    def emit(self, rhs: str) -> str:
        name = self.tmp()
        self.lines.append(f"    {name} = {rhs}")
        return name

    def raw(self, line: str) -> None:
        self.lines.append(f"    {line}")

    def label(self) -> str:
        self.blocks += 1
        return f"^bb{self.blocks}"

    def const(self, value, type: str) -> str:
        if type in ("f32", "f64", "bf16"):
            v = float(value)
            if math.isnan(v):
                literal = {"f32": "0x7FC00000", "f64": "0x7FF8000000000000", "bf16": "0x7FC0"}[type]
            elif math.isinf(v):
                positive = {"f32": "0x7F800000", "f64": "0x7FF0000000000000", "bf16": "0x7F80"}
                negative = {"f32": "0xFF800000", "f64": "0xFFF0000000000000", "bf16": "0xFF80"}
                literal = (positive if v > 0 else negative)[type]
            else:
                # MLIR float literals need a decimal point; 17 digits round-trip any f64.
                literal = f"{v:.17e}"
            text = f"{literal} : {type}"
        elif type == "i1":
            text = f"{'true' if value else 'false'}"
        else:
            text = f"{int(value)} : {type}"
        key = (text, type)
        if key not in self.constants:
            # Constants go to the entry block so every later block can use them.
            name = self.tmp()
            self.constants[key] = name
            self.entry_constants.append(f"    {name} = llvm.mlir.constant({text}) : {type}")
        return self.constants[key]

    def entry(self, rhs: str) -> str:
        """Emit into the entry block: for values that depend only on the thread id."""
        name = self.tmp()
        self.entry_constants.append(f"    {name} = {rhs}")
        return name

    def conj(self, a: str | None, b: str | None) -> str | None:
        if a is None:
            return b
        if b is None:
            return a
        return self.emit(f"llvm.and {a}, {b} : i1")

    def i32(self, value: int) -> str:
        return self.const(value, "i32")

    def dim(self, d: Dim) -> str:
        if isinstance(d, int):
            return self.i32(d)
        key = str(d)
        if key not in self.dims:
            self.dims[key] = (len(self.dims), d)
        return f"%dim{self.dims[key][0]}"

    def barrier(self) -> None:
        self.raw("nvvm.barrier")

    # -- layout -----------------------------------------------------------------------------------

    @staticmethod
    def count(shape) -> int:
        return math.prod(shape)

    def slots(self, shape) -> int:
        n = self.count(shape)
        return 1 if n == 1 else -(-n // self.threads)

    def flat_index(self, slot: int) -> str:
        key = ("flat", slot)
        if key not in self.constants:
            self.constants[key] = self.entry(f"llvm.add %tid, {self.i32(slot * self.threads)} : i32") if slot else "%tid"
        return self.constants[key]

    def coords(self, shape: tuple[int, ...]) -> list[list[str]]:
        """coords[slot][axis]: this thread's coordinate of each element it owns (entry block)."""
        if shape in self.coords_cache:
            return self.coords_cache[shape]
        result = []
        if self.count(shape) == 1:
            result = [[self.i32(0) for _ in shape]]
        else:
            strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
            for slot in range(self.slots(shape)):
                e = self.flat_index(slot)
                axes = []
                for axis, (size, stride) in enumerate(zip(shape, strides)):
                    q = self.entry(f"llvm.udiv {e}, {self.i32(stride)} : i32") if stride != 1 else e
                    # The leading axis needs no remainder: e < count, so e // stride < size.
                    axes.append(q if axis == 0 else self.entry(f"llvm.urem {q}, {self.i32(size)} : i32"))
                result.append(axes)
        self.coords_cache[shape] = result
        return result

    def valid(self, shape, slot: int) -> str | None:
        """i1 'this slot holds an element' when the last slot is partial, else None."""
        n = self.count(shape)
        if n == 1 or n % self.threads == 0 or slot < self.slots(shape) - 1:
            return None
        key = ("valid", n, slot)
        if key not in self.constants:
            self.constants[key] = self.entry(f"llvm.icmp \"ult\" {self.flat_index(slot)}, {self.i32(n)} : i32")
        return self.constants[key]

    def make(self, t: TileType, regs: list[str], like: _Value | None = None) -> _Value:
        """A value in the cyclic layout of t, or in the explicit layout of `like`."""
        if like is not None and like.coords is not None:
            return _Value(regs, False, t, like.coords, like.valid)
        return _Value(regs, self.count(t.shape) == 1, t)

    def layout_of(self, v: _Value) -> list[list[str]]:
        return v.coords if v.coords is not None else self.coords(v.type.shape)

    def valid_of(self, v: _Value, slot: int) -> str | None:
        if v.coords is not None:
            return v.valid[slot] if v.valid is not None else None
        return self.valid(v.type.shape, slot)

    def relayout(self, v: _Value, like: _Value) -> _Value:
        """v's elements in the layout of `like` (same shape), through shared memory."""
        if v.coords is like.coords or (v.replicated and like.coords is None):
            return v
        if v.replicated:
            return _Value([v.regs[0]] * len(like.regs), False, v.type, like.coords, like.valid)
        self.to_shared(v)
        regs = []
        for slot, coords in enumerate(self.layout_of(like)):
            index = self.linear(coords, v.type.shape)
            guard = self.valid_of(like, slot)
            if guard is not None:
                index = self.emit(f"llvm.select {guard}, {index}, {self.i32(0)} : i1, i32")
            regs.append(self.load_shared(index, v.type.dtype))
        if like.coords is None:
            return self.make(v.type, regs)
        return _Value(regs, False, v.type, like.coords, like.valid)

    def cyclic(self, v: _Value) -> _Value:
        if v.coords is None:
            return v
        return self.relayout(v, _Value([], self.count(v.type.shape) == 1, v.type))

    def unify(self, args: list[_Value]) -> list[_Value]:
        explicit = next((a for a in args if a.coords is not None), None)
        if explicit is None:
            return args
        return [self.relayout(a, explicit) for a in args]

    def spread(self, v: _Value, t: TileType) -> list[str]:
        """Registers of v in the layout of shape t (only for replicated v)."""
        return [v.regs[0]] * self.slots(t.shape)

    # -- shared memory exchange -------------------------------------------------------------------

    def smem_ptr(self, index: str, dtype: DType) -> str:
        base = self.emit("llvm.mlir.addressof @smem : !llvm.ptr<3>")
        return self.emit(f"llvm.getelementptr {base}[{index}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, {_MEMORY_TYPE[dtype]}")

    def to_shared(self, v: _Value) -> None:
        """Write every element of v to shared memory at its row-major index."""
        self.shared_bytes = max(self.shared_bytes, self.count(v.type.shape) * element_bytes(v.type.dtype))
        self.barrier()
        if v.replicated:
            self.store_shared(self.i32(0), v.regs[0], v.type.dtype)
        else:
            layout = self.layout_of(v)
            for slot, reg in enumerate(v.regs):
                index = self.flat_index(slot) if v.coords is None else self.linear(layout[slot], v.type.shape)
                guard = self.valid_of(v, slot)
                self.guarded(guard, lambda index=index, reg=reg: self.store_shared(index, reg, v.type.dtype))
        self.barrier()

    def store_shared(self, index: str, reg: str, dtype: DType) -> None:
        value = self.emit(f"llvm.zext {reg} : i1 to i8") if dtype is DType.bool else reg
        self.raw(f"llvm.store {value}, {self.smem_ptr(index, dtype)} : {_MEMORY_TYPE[dtype]}, !llvm.ptr<3>")

    def load_shared(self, index: str, dtype: DType) -> str:
        value = self.emit(f"llvm.load {self.smem_ptr(index, dtype)} : !llvm.ptr<3> -> {_MEMORY_TYPE[dtype]}")
        return self.emit(f"llvm.trunc {value} : i8 to i1") if dtype is DType.bool else value

    def linear(self, coords: list[str], shape: tuple[int, ...]) -> str:
        index = self.i32(0)
        for c, stride in zip(coords, (math.prod(shape[i + 1 :]) for i in range(len(shape)))):
            term = c if stride == 1 else self.emit(f"llvm.mul {c}, {self.i32(stride)} : i32")
            index = term if index == self.i32(0) else self.emit(f"llvm.add {index}, {term} : i32")
        return index

    def guarded(self, condition: str | None, body) -> None:
        if condition is None:
            body()
            return
        then, done = self.label(), self.label()
        self.raw(f"llvm.cond_br {condition}, {then}, {done}")
        self.lines.append(f"  {then}:")
        body()
        self.raw(f"llvm.br {done}")
        self.lines.append(f"  {done}:")

    # -- arithmetic on registers ------------------------------------------------------------------

    def widen(self, reg: str, dtype: DType) -> str:
        return self.emit(f"llvm.fpext {reg} : bf16 to f32") if dtype is DType.bf16 else reg

    def narrow(self, reg: str, dtype: DType) -> str:
        return self.emit(f"llvm.fptrunc {reg} : f32 to bf16") if dtype is DType.bf16 else reg

    def libdevice(self, name: str, args: list[str], type: str) -> str:
        fn = f"__nv_{name}{'f' if type == 'f32' else ''}"
        self.externs.add(f"  llvm.func @{fn}({', '.join([type] * len(args))}) -> {type}")
        return self.emit(f"llvm.call @{fn}({', '.join(args)}) : ({', '.join([type] * len(args))}) -> {type}")

    def scalar_op(self, kind: str, regs: list[str], dtype: DType, attrs: dict) -> str:
        if dtype.is_float:
            t = _compute_type(dtype)
            xs = [self.widen(r, dtype) for r in regs]
            result = self.float_op(kind, xs, t)
            return self.narrow(result, dtype)
        return self.int_op(kind, regs, _LLVM[dtype], dtype)

    def float_op(self, kind: str, xs: list[str], t: str) -> str:
        binary = {"add": "llvm.fadd", "sub": "llvm.fsub", "mul": "llvm.fmul", "div": "llvm.fdiv"}
        if kind in binary:
            return self.emit(f"{binary[kind]} {xs[0]}, {xs[1]} : {t}")
        if kind in ("max", "min"):
            # NaN-propagating, like NumPy's maximum/minimum (PTX max.NaN, available from sm_80).
            name = "maximum" if kind == "max" else "minimum"
            return self.emit(f"llvm.intr.{name}({xs[0]}, {xs[1]}) : ({t}, {t}) -> {t}")
        if kind == "neg":
            return self.emit(f"llvm.fneg {xs[0]} : {t}")
        if kind == "abs":
            return self.emit(f"llvm.intr.fabs({xs[0]}) : ({t}) -> {t}")
        if kind == "sqrt":
            return self.emit(f"llvm.intr.sqrt({xs[0]}) : ({t}) -> {t}")
        if kind == "rsqrt":
            # 1/sqrt with both steps correctly rounded; rsqrt.approx is not accurate enough.
            root = self.emit(f"llvm.intr.sqrt({xs[0]}) : ({t}) -> {t}")
            return self.emit(f"llvm.fdiv {self.const(1.0, t)}, {root} : {t}")
        if kind == "recip":
            return self.emit(f"llvm.fdiv {self.const(1.0, t)}, {xs[0]} : {t}")
        if kind in _LIBDEVICE:
            return self.libdevice(_LIBDEVICE[kind], xs, t)
        raise NotImplementedError(f"CUDA lowering of float {kind}")

    def int_op(self, kind: str, xs: list[str], t: str, dtype: DType) -> str:
        unsigned = dtype is DType.u32
        binary = {"add": "llvm.add", "sub": "llvm.sub", "mul": "llvm.mul", "floordiv": "llvm.udiv" if unsigned else "llvm.sdiv", "mod": "llvm.urem" if unsigned else "llvm.srem"}
        if kind in binary:
            return self.emit(f"{binary[kind]} {xs[0]}, {xs[1]} : {t}")
        if kind in ("max", "min"):
            name = ("u" if unsigned else "s") + kind
            return self.emit(f"llvm.intr.{name}({xs[0]}, {xs[1]}) : ({t}, {t}) -> {t}")
        if kind == "neg":
            return self.emit(f"llvm.sub {self.const(0, t)}, {xs[0]} : {t}")
        if kind == "abs":
            return self.emit(f"\"llvm.intr.abs\"({xs[0]}) <{{is_int_min_poison = false}}> : ({t}) -> {t}")
        raise NotImplementedError(f"CUDA lowering of integer {kind}")

    def cast(self, reg: str, src: DType, dst: DType) -> str:
        if src == dst or (_LLVM[src] == _LLVM[dst]):
            return reg
        if src is DType.bool:
            if dst.is_float:
                return self.narrow(self.emit(f"llvm.uitofp {reg} : i1 to {_compute_type(dst)}"), dst)
            return self.emit(f"llvm.zext {reg} : i1 to {_LLVM[dst]}")
        if dst is DType.bool:
            if src.is_float:
                return self.emit(f"llvm.fcmp \"une\" {self.widen(reg, src)}, {self.const(0.0, _compute_type(src))} : {_compute_type(src)}")
            return self.emit(f"llvm.icmp \"ne\" {reg}, {self.const(0, _LLVM[src])} : {_LLVM[src]}")
        if src.is_float and dst.is_float:
            if src is DType.bf16 or dst is DType.bf16:
                # bf16 <-> f64 goes through f32: exact from bf16, and from f64 it rounds twice. The
                # reference rounds f64 -> bf16 once; this path is accepted only because no
                # graph casts f64 to bf16 (f64 runs are verification runs in f64 throughout).
                wide = self.widen(reg, src)
                if dst is DType.bf16:
                    if src is DType.f64:
                        wide = self.emit(f"llvm.fptrunc {wide} : f64 to f32")
                    return self.narrow(wide, dst)
                return self.emit(f"llvm.fpext {wide} : f32 to f64") if dst is DType.f64 else wide
            op = "llvm.fpext" if _BYTES[dst] > _BYTES[src] else "llvm.fptrunc"
            return self.emit(f"{op} {reg} : {_LLVM[src]} to {_LLVM[dst]}")
        if src.is_float:
            op = "llvm.fptoui" if dst is DType.u32 else "llvm.fptosi"
            return self.emit(f"{op} {self.widen(reg, src)} : {_compute_type(src)} to {_LLVM[dst]}")
        if dst.is_float:
            op = "llvm.uitofp" if src is DType.u32 else "llvm.sitofp"
            return self.narrow(self.emit(f"{op} {reg} : {_LLVM[src]} to {_compute_type(dst)}"), dst)
        if _BYTES[dst] > _BYTES[src]:
            return self.emit(f"llvm.{'zext' if src is DType.u32 else 'sext'} {reg} : {_LLVM[src]} to {_LLVM[dst]}")
        return self.emit(f"llvm.trunc {reg} : {_LLVM[src]} to {_LLVM[dst]}")

    def compare(self, fn: str, a: str, b: str, dtype: DType) -> str:
        if dtype.is_float:
            t = _compute_type(dtype)
            pred = {"eq": "oeq", "lt": "olt", "le": "ole"}[fn]
            return self.emit(f"llvm.fcmp \"{pred}\" {self.widen(a, dtype)}, {self.widen(b, dtype)} : {t}")
        if dtype is DType.bool:
            a, b = (self.emit(f"llvm.zext {x} : i1 to i8") for x in (a, b))
            t = "i8"
        else:
            t = _LLVM[dtype]
        unsigned = dtype in (DType.u32, DType.bool)
        pred = {"eq": "eq", "lt": "ult" if unsigned else "slt", "le": "ule" if unsigned else "sle"}[fn]
        return self.emit(f"llvm.icmp \"{pred}\" {a}, {b} : {t}")

    # -- global memory ----------------------------------------------------------------------------

    def view_address(self, view: View, coords: list[str]) -> tuple[str, str | None, str | None]:
        """Element offset, in-bounds condition and in-padding condition for logical coordinates."""
        inside: str | None = None
        padded: str | None = None
        offset = self.dim(view.offset)
        for axis, c in enumerate(coords):
            ok = self.emit(f"llvm.icmp \"ult\" {c}, {self.dim(view.shape[axis])} : i32")
            inside = self.conj(inside, ok)
            stored = c
            if view.pad is not None:
                lo, hi = view.pad[axis]
                if lo != 0 or hi != 0:
                    below = self.emit(f"llvm.icmp \"slt\" {c}, {self.dim(lo)} : i32")
                    upper = self.emit(f"llvm.sub {self.dim(view.shape[axis])}, {self.dim(hi)} : i32")
                    above = self.emit(f"llvm.icmp \"sge\" {c}, {upper} : i32")
                    p = self.emit(f"llvm.or {below}, {above} : i1")
                    padded = p if padded is None else self.emit(f"llvm.or {padded}, {p} : i1")
                    stored = self.emit(f"llvm.sub {c}, {self.dim(lo)} : i32")
            stride = view.strides[axis]
            if stride != 0:
                term = stored if stride == 1 else self.emit(f"llvm.mul {stored}, {self.dim(stride)} : i32")
                offset = self.emit(f"llvm.add {offset}, {term} : i32")
        return offset, inside, padded

    def global_ptr(self, arg: int, offset: str, dtype: DType) -> str:
        return self.emit(f"llvm.getelementptr %arg{arg}[{offset}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, {_MEMORY_TYPE[dtype]}")

    def masked_load(self, arg: int, view: View, coords: list[str], dtype: DType, other, fill, slot_valid: str | None) -> str:
        offset, inside, padded = self.view_address(view, coords)
        ok = inside
        if padded is not None:
            ok = self.conj(ok, self.emit(f"llvm.xor {padded}, {self.const(True, 'i1')} : i1"))
        ok = self.conj(ok, slot_valid) or self.const(True, "i1")
        # Masked lanes read element 0 instead (buffers are never empty), then select `other`:
        # branch-free, and the hardware cannot fault on a masked address.
        safe = self.emit(f"llvm.select {ok}, {offset}, {self.i32(0)} : i1, i32")
        value = self.emit(f"llvm.load {self.global_ptr(arg, safe, dtype)} : !llvm.ptr<1> -> {_MEMORY_TYPE[dtype]}")
        if dtype is DType.bool:
            value = self.emit(f"llvm.icmp \"ne\" {value}, {self.const(0, 'i8')} : i8")
        t = _LLVM[dtype]
        value = self.emit(f"llvm.select {ok}, {value}, {self.typed_const(other, dtype)} : i1, {t}")
        if padded is not None:
            value = self.emit(f"llvm.select {padded}, {self.typed_const(fill, dtype)}, {value} : i1, {t}")
        return value

    def typed_const(self, value, dtype: DType) -> str:
        if dtype is DType.bool:
            return self.const(bool(value), "i1")
        return self.const(value, _LLVM[dtype])

    # -- ops ------------------------------------------------------------------------------------

    def body(self, body: Body) -> list[_Value]:
        for op in body.ops:
            outs = self.op(op, [self.values[v] for v in op.inputs])
            for v, lowered in zip(op.outputs, outs, strict=True):
                self.values[v] = lowered
        return [self.values[r] for r in body.results]

    def op(self, op: TOp, args: list[_Value]) -> list[_Value]:
        kind = op.kind
        out_t = op.outputs[0].type if op.outputs else None
        attrs = dict(op.attrs)
        if kind == "full":
            reg = self.typed_const(attrs["value"], out_t.dtype)
            return [self.make(out_t, [reg] * self.slots(out_t.shape))]
        if kind == "dim":
            return [self.make(out_t, [self.dim(attrs["value"])])]
        if kind == "iota":
            start = args[0].regs[0]
            coords = self.coords(out_t.shape)
            axis = attrs["axis"]
            return [self.make(out_t, [self.emit(f"llvm.add {start}, {c[axis]} : i32") for c in coords])]
        if kind == "reshape":
            src = args[0]
            if src.coords is None:
                return [self.make(out_t, list(src.regs))]
            if [d for d in src.type.shape if d != 1] != [d for d in out_t.shape if d != 1]:
                return [self.make(out_t, list(self.cyclic(src).regs))]
            # Only unit axes appear or disappear: relabel the coordinates.
            zero = self.i32(0)
            coords = []
            for c in src.coords:
                big = [x for x, d in zip(c, src.type.shape) if d != 1]
                coords.append([big.pop(0) if d != 1 else zero for d in out_t.shape])
            return [_Value(list(src.regs), False, out_t, coords, src.valid)]
        if kind == "permute" and args[0].coords is not None:
            perm = attrs["perm"]
            src = args[0]
            return [_Value(list(src.regs), False, out_t, [[c[p] for p in perm] for c in src.coords], src.valid)]
        if kind in ("broadcast", "permute"):
            return [self.exchange(op, self.cyclic(args[0]), out_t)]
        if kind in ("cmp", "where") or (kind not in ("full", "dim", "iota", "cast", "reduce", "dot", "load", "load_indexed", "store", "atomic_add_indexed", "for")):
            args = self.unify(args)
        if kind == "cmp":
            a, b = args
            return [self.make(out_t, [self.compare(attrs["fn"], x, y, a.type.dtype) for x, y in zip(a.regs, b.regs)], a)]
        if kind == "where":
            c, a, b = args
            t = _LLVM[a.type.dtype]
            return [self.make(out_t, [self.emit(f"llvm.select {p}, {x}, {y} : i1, {t}") for p, x, y in zip(c.regs, a.regs, b.regs)], a)]
        if kind == "cast":
            return [self.make(out_t, [self.cast(r, args[0].type.dtype, out_t.dtype) for r in args[0].regs], args[0])]
        if kind == "reduce":
            return [self.reduce(op, self.cyclic(args[0]), out_t)]
        if kind == "dot":
            a, b, acc = args
            return [self.dot(self.cyclic(a), self.cyclic(b), self.cyclic(acc))]
        if kind == "load":
            return [self.load(op, args, out_t)]
        if kind == "load_indexed":
            return [self.load(op, args[1:], out_t, index=args[0])]
        if kind == "store":
            self.store(op, args)
            return []
        if kind == "atomic_add_indexed":
            self.atomic_add(op, args)
            return []
        if kind == "for":
            if self.use_matrix and op.attr("stages", 1) > 1:
                from olaf.backend.cuda.mma import lower_matrix_loop

                lowered = lower_matrix_loop(self, op, args)
                if lowered is not None:
                    return lowered
            return self.loop(op, args)
        # Elementwise.
        dtype = args[0].type.dtype
        n = max(len(a.regs) for a in args)
        regs = [self.scalar_op(kind, [a.regs[i] if len(a.regs) > 1 else a.regs[0] for a in args], dtype, attrs) for i in range(n)]
        return [self.make(out_t, regs, next((a for a in args if a.coords is not None), None))]

    def exchange(self, op: TOp, src: _Value, out_t: TileType) -> _Value:
        attrs = dict(op.attrs)
        if src.replicated and op.kind == "broadcast":
            return self.make(out_t, self.spread(src, out_t))
        if self.count(out_t.shape) == 1 and src.replicated:
            return self.make(out_t, list(src.regs))
        self.to_shared(src)
        regs = []
        for slot, coords in enumerate(self.coords(out_t.shape)):
            if op.kind == "broadcast":
                source = [self.i32(0) if src.type.shape[i] == 1 else coords[d] for i, d in enumerate(attrs["dims"])]
            else:
                perm = attrs["perm"]
                source = [coords[perm.index(i)] for i in range(src.type.rank)]
            index = self.linear(source, src.type.shape)
            if self.valid(out_t.shape, slot) is not None:
                index = self.emit(f"llvm.select {self.valid(out_t.shape, slot)}, {index}, {self.i32(0)} : i1, i32")
            regs.append(self.load_shared(index, src.type.dtype))
        return self.make(out_t, regs)

    def reduce(self, op: TOp, src: _Value, out_t: TileType) -> _Value:
        # Every thread computes the output elements it owns by walking the reduced elements in
        # shared memory in row-major order. Correct for any shape; fast reductions use shuffles.
        axes = op.attr("axes")
        fn = op.attr("fn")
        dtype = src.type.dtype
        self.to_shared(src)
        kept = [i for i in range(src.type.rank) if i not in axes]
        reduced_shape = [src.type.shape[a] for a in axes]
        regs = []
        for slot, coords in enumerate(self.coords(out_t.shape)):
            valid = self.valid(out_t.shape, slot)
            acc = None
            for position in range(math.prod(reduced_shape)):
                full = [None] * src.type.rank
                for i, a in enumerate(kept):
                    full[a] = coords[i]
                rest = position
                for a, size in reversed(list(zip(axes, reduced_shape))):
                    full[a] = self.i32(rest % size)
                    rest //= size
                index = self.linear(full, src.type.shape)
                if valid is not None:
                    index = self.emit(f"llvm.select {valid}, {index}, {self.i32(0)} : i1, i32")
                x = self.load_shared(index, dtype)
                acc = x if acc is None else self.scalar_op("add" if fn == "sum" else "max", [acc, x], dtype, {})
            regs.append(acc)
        return self.make(out_t, regs)

    def dot(self, a: _Value, b: _Value, acc: _Value) -> _Value:
        # a and b go to shared memory one after the other; each thread accumulates its own acc
        # elements over k. Products and sums are in the accumulator type.
        m, k = a.type.shape
        n = b.type.shape[1]
        acc_t = _LLVM[acc.type.dtype]
        a_bytes = m * k * element_bytes(a.type.dtype)
        self.shared_bytes = max(self.shared_bytes, a_bytes + k * n * element_bytes(b.type.dtype))
        self.barrier()
        self.write_tile(a, 0)
        self.write_tile(b, a_bytes)
        self.barrier()
        accs = list(acc.regs)
        out = []
        for slot, coords in enumerate(self.coords(acc.type.shape)):
            row, col = coords
            total = accs[slot]
            for kk in range(k):
                ai = self.emit(f"llvm.add {self.emit(f'llvm.mul {row}, {self.i32(k)} : i32')}, {self.i32(kk)} : i32")
                bi = self.emit(f"llvm.add {self.i32(kk * n)}, {col} : i32")
                x = self.to_acc(self.load_shared_at(ai, a.type.dtype, 0), a.type.dtype, acc.type.dtype)
                y = self.to_acc(self.load_shared_at(bi, b.type.dtype, a_bytes), b.type.dtype, acc.type.dtype)
                total = self.emit(f"llvm.intr.fma({x}, {y}, {total}) : ({acc_t}, {acc_t}, {acc_t}) -> {acc_t}")
            out.append(total)
        return self.make(acc.type, out)

    def to_acc(self, reg: str, src: DType, acc: DType) -> str:
        return self.cast(reg, src, acc)

    def write_tile(self, v: _Value, byte_offset: int) -> None:
        if v.replicated:
            self.raw(f"llvm.store {v.regs[0]}, {self.smem_at(self.i32(0), v.type.dtype, byte_offset)} : {_MEMORY_TYPE[v.type.dtype]}, !llvm.ptr<3>")
            return
        for slot, reg in enumerate(v.regs):
            index = self.flat_index(slot)
            guard = self.valid(v.type.shape, slot)
            self.guarded(guard, lambda index=index, reg=reg: self.raw(f"llvm.store {reg}, {self.smem_at(index, v.type.dtype, byte_offset)} : {_MEMORY_TYPE[v.type.dtype]}, !llvm.ptr<3>"))

    def smem_at(self, index: str, dtype: DType, byte_offset: int) -> str:
        base = self.emit("llvm.mlir.addressof @smem : !llvm.ptr<3>")
        if byte_offset:
            base = self.emit(f"llvm.getelementptr {base}[{self.i32(byte_offset)}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        return self.emit(f"llvm.getelementptr {base}[{index}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, {_MEMORY_TYPE[dtype]}")

    def load_shared_at(self, index: str, dtype: DType, byte_offset: int) -> str:
        return self.emit(f"llvm.load {self.smem_at(index, dtype, byte_offset)} : !llvm.ptr<3> -> {_MEMORY_TYPE[dtype]}")

    def tile_coords(self, starts: list[_Value], shape: tuple[int, ...], slot_coords: list[str]) -> list[str]:
        return [c if s.regs[0] == self.i32(0) else self.emit(f"llvm.add {s.regs[0]}, {c} : i32") for s, c in zip(starts, slot_coords)]

    def load(self, op: TOp, starts: list[_Value], out_t: TileType, index: _Value | None = None) -> _Value:
        arg = op.attr("arg")
        view = self.task.args[arg].view
        dtype = op.attr("dtype")
        regs = []
        for slot, coords in enumerate(self.coords(out_t.shape)):
            tile = self.tile_coords(starts, out_t.shape, coords)
            if index is not None:
                idx = index.regs[slot] if not index.replicated else index.regs[0]
                idx = self.cast(idx, index.type.dtype, INDEX)
                tile = [idx if m == -1 else tile[m] for m in op.attr("axis_map")]
            regs.append(self.masked_load(arg, view, tile, dtype, op.attr("other"), view.fill, self.valid(out_t.shape, slot)))
        return self.make(out_t, regs)

    def store(self, op: TOp, args: list[_Value]) -> None:
        value, *starts = args
        arg = op.attr("arg")
        view = self.task.args[arg].view
        dtype = value.type.dtype
        coords_all = self.layout_of(value)
        # A replicated tile is stored by thread 0 only: every thread holds the same value, and
        # one store per element keeps atomics-free reductions into it well defined.
        slots = [0] if value.replicated else range(len(value.regs))
        for slot in slots:
            tile = self.tile_coords(starts, value.type.shape, coords_all[slot])
            offset, inside, _ = self.view_address(view, tile)
            ok = self.conj(inside, self.valid_of(value, slot))
            if value.replicated:
                ok = self.conj(ok, self.emit(f"llvm.icmp \"eq\" %tid, {self.i32(0)} : i32"))
            reg = value.regs[slot]
            stored = self.emit(f"llvm.zext {reg} : i1 to i8") if dtype is DType.bool else reg
            ptr = self.global_ptr(arg, offset, dtype)
            self.guarded(ok, lambda ptr=ptr, stored=stored: self.raw(f"llvm.store {stored}, {ptr} : {_MEMORY_TYPE[dtype]}, !llvm.ptr<1>"))

    def atomic_add(self, op: TOp, args: list[_Value]) -> None:
        index, value, *starts = args
        index, value = self.cyclic(index), self.cyclic(value)
        arg = op.attr("arg")
        view = self.task.args[arg].view
        dtype = value.type.dtype
        if dtype not in (DType.f32, DType.f64):
            raise NotImplementedError(f"atomic add of {dtype} (needs a compare-and-swap loop)")
        coords_all = self.coords(value.type.shape)
        slots = [0] if value.replicated else range(len(value.regs))
        for slot in slots:
            tile = self.tile_coords(starts, value.type.shape, coords_all[slot])
            idx = self.cast(index.regs[slot] if not index.replicated else index.regs[0], index.type.dtype, INDEX)
            target = [idx if m == -1 else tile[m] for m in op.attr("axis_map")]
            offset, ok, _ = self.view_address(view, target)
            for c, bound in zip(tile, op.attr("bounds")):
                ok = self.conj(ok, self.emit(f"llvm.icmp \"ult\" {c}, {self.dim(bound)} : i32"))
            ok = self.conj(ok, self.valid(value.type.shape, slot))
            if value.replicated:
                ok = self.conj(ok, self.emit(f"llvm.icmp \"eq\" %tid, {self.i32(0)} : i32"))
            ptr = self.global_ptr(arg, offset, dtype)
            reg = value.regs[slot]
            self.guarded(ok, lambda ptr=ptr, reg=reg: self.raw(f"llvm.atomicrmw fadd {ptr}, {reg} monotonic : !llvm.ptr<1>, {_LLVM[dtype]}"))

    def loop(self, op: TOp, args: list[_Value]) -> list[_Value]:
        lo, hi, step, *carries = args
        region = op.regions[0]
        header, body, done = self.label(), self.label(), self.label()
        flat = [r for c in carries for r in c.regs]
        types = [_LLVM[c.type.dtype] for c in carries for _ in c.regs]
        self.raw(f"llvm.br {header}({', '.join([lo.regs[0], *flat])} : {', '.join(['i32', *types])})")
        iv = self.tmp()
        params = [self.tmp() for _ in flat]
        sig = ", ".join(f"{n}: {t}" for n, t in zip([iv, *params], ["i32", *types]))
        self.lines.append(f"  {header}({sig}):")
        cond = self.emit(f"llvm.icmp \"slt\" {iv}, {hi.regs[0]} : i32")
        self.raw(f"llvm.cond_br {cond}, {body}, {done}")
        self.lines.append(f"  {body}:")
        self.values[region.params[0]] = _Value([iv], True, region.params[0].type)
        position = 0
        carried = []
        for p, c in zip(region.params[1:], carries):
            v = _Value(params[position : position + len(c.regs)], c.replicated, c.type, c.coords, c.valid)
            position += len(c.regs)
            self.values[p] = v
            carried.append(v)
        results = [self.relayout(r, c) for r, c in zip(self.body(region), carries)]
        nxt = self.emit(f"llvm.add {iv}, {step.regs[0]} : i32")
        out = [r for v in results for r in v.regs]
        self.raw(f"llvm.br {header}({', '.join([nxt, *out])} : {', '.join(['i32', *types])})")
        self.lines.append(f"  {done}:")
        return carried

    # -- driver ---------------------------------------------------------------------------------

    def run(self) -> Lowered:
        prologue = ["    %tid = nvvm.read.ptx.sreg.tid.x : i32", "    %block = nvvm.read.ptx.sreg.ctaid.x : i32"]
        # Recover one program id per grid axis from the flat block index.
        task = self.task
        pids = []
        rest = "%block"
        for axis in reversed(range(len(task.grid))):
            size = self.dim(task.grid[axis])
            if axis == 0:
                pids.append(rest)
            else:
                pids.append(self.emit(f"llvm.urem {rest}, {size} : i32"))
                rest = self.emit(f"llvm.udiv {rest}, {size} : i32")
        for param, pid in zip(task.body.params, reversed(pids)):
            self.values[param] = _Value([pid], True, param.type)
        self.body(task.body)
        self.raw("llvm.return")

        pointer_args = [f"%arg{i}: !llvm.ptr<1>" for i in range(len(task.args))]
        dim_args = [f"%dim{i}: i32" for i in range(len(self.dims))]
        header = [
            "module attributes {gpu.container_module} {",
            "gpu.module @kernels {",
            # Dynamic shared memory (sized at launch): static allocations stop at 48 KiB.
            "  llvm.mlir.global external @smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>",
            *sorted(self.externs),
            f"  llvm.func @task({', '.join(pointer_args + dim_args)}) attributes {{nvvm.kernel, nvvm.maxntid = array<i32: {self.threads}, 1, 1>}} {{",
        ]
        text = "\n".join(header + prologue + self.entry_constants + self.lines + ["  }", "}", "}"]) + "\n"
        dims = [d for _, d in sorted(self.dims.values(), key=lambda t: t[0])]
        lowered = Lowered(text, dims, self.shared_bytes, self.threads, list(self.aligned))
        if self.aligned:
            lowered.fallback = _Emitter(self.task, use_matrix=False).run()
        return lowered


def lower_task(task: Task) -> Lowered:
    return _Emitter(task).run()
