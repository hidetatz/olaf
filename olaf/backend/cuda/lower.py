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
_ELEMENTWISE = frozenset({"neg", "abs", "exp", "log", "sqrt", "rsqrt", "recip", "tanh", "erf", "sin", "cos", "add", "sub", "mul", "div", "max", "min", "pow", "floordiv", "mod"})
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
    aligned_args: list[int] = field(default_factory=list)  # arguments whose launch offset must be a multiple of 8
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
    # MmaAcc / MmaRows when the explicit layout is a matrix instruction's (see those classes).
    structure: object = None
    # rebuild(like) -> registers of this value in like's layout, re-emitting its computation:
    # set for values computed only from constants, iota and read-only loads. Changing such a
    # value's layout then costs its recomputation instead of a round trip through shared memory.
    rebuild: object = None


@dataclass(frozen=True)
class MmaAcc:
    """Accumulator fragments of m16n8 matrix instructions over a wm x wn warp grid.

    Slot (i, j, q) holds row warp_m * em/wm + 16 i + lane/4 + 8 (q // 2) and column
    warp_n * en/wn + 8 j + 2 (lane % 4) + q % 2 of an em x en tile; row_axis and col_axis say
    which tile axes those are (unit axes may surround them).
    """

    em: int
    en: int
    wm: int
    wn: int
    row_axis: int
    col_axis: int
    rank: int

    @property
    def mi(self) -> int:
        return self.em // (16 * self.wm)

    @property
    def nj(self) -> int:
        return self.en // (8 * self.wn)


@dataclass(frozen=True)
class MmaRows:
    """One value per accumulator row of an MmaAcc: slot (i, h) is row 16 i + lane/4 + 8 h of the
    warp's rows, held by the 4 lanes of a quad (and by every warp along n)."""

    acc: MmaAcc
    row_axis: int
    rank: int


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
    aligned_args: set[int] = field(default_factory=set)
    memo: dict = field(default_factory=dict)
    keep: list = field(default_factory=list)
    # Shared memory: [resident operands of the enclosing loops | scratch]. Scratch users (operand
    # staging, layout exchanges, reductions) address from scratch_base, which moves past the
    # operands a loop keeps resident while its body is emitted.
    scratch_base: int = 0
    resident: dict = field(default_factory=dict)  # load op -> byte offset of its resident copy
    streamed: dict = field(default_factory=dict)  # load op -> this iteration's buffer (scratch-relative SSA)

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

    def reserve(self, nbytes: int) -> None:
        """Record that scratch needs nbytes from scratch_base."""
        self.shared_bytes = max(self.shared_bytes, self.scratch_base + nbytes)

    def smem_base(self) -> str:
        """Pointer to the start of scratch shared memory."""
        base = self.emit("llvm.mlir.addressof @smem : !llvm.ptr<3>")
        if self.scratch_base:
            base = self.emit(f"llvm.getelementptr {base}[{self.i32(self.scratch_base)}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        return base

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
            return _Value(regs, False, t, like.coords, like.valid, like.structure)
        return _Value(regs, self.count(t.shape) == 1, t)

    def template(self, t: TileType, like: _Value) -> _Value:
        """An empty value with like's layout and t's type, for rebuild()."""
        return _Value([], like.coords is None and self.count(t.shape) == 1, t, like.coords, like.valid, like.structure)

    def n_slots(self, like: _Value) -> int:
        return len(like.coords) if like.coords is not None else self.slots(like.type.shape)

    def rebuilt(self, v: _Value, like: _Value) -> _Value:
        """v recomputed in like's layout (v.rebuild must be set). Memoized per layout."""
        key = ("rebuilt", id(v), id(like.coords) if like.coords is not None else ("cyclic", like.type.shape))
        if key not in self.memo:
            regs = v.rebuild(self.template(v.type, like))
            self.memo[key] = _Value(regs, like.coords is None and self.count(v.type.shape) == 1, v.type, like.coords, like.valid, like.structure, v.rebuild)
            self.keep.append(v)  # id(v) must stay unique while the memo refers to it
        return self.memo[key]

    def layout_of(self, v: _Value) -> list[list[str]]:
        return v.coords if v.coords is not None else self.coords(v.type.shape)

    def valid_of(self, v: _Value, slot: int) -> str | None:
        if v.coords is not None:
            return v.valid[slot] if v.valid is not None else None
        return self.valid(v.type.shape, slot)

    def relayout(self, v: _Value, like: _Value) -> _Value:
        """v's elements in the layout of `like` (same shape): recomputed if v can be, else through shared memory."""
        if v.coords is like.coords or (v.replicated and like.coords is None):
            return v
        if v.replicated:
            return _Value([v.regs[0]] * self.n_slots(like), False, v.type, like.coords, like.valid, like.structure)
        if v.rebuild is not None:
            return self.rebuilt(v, like)
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
        return _Value(regs, False, v.type, like.coords, like.valid, like.structure)

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
        base = self.smem_base()
        return self.emit(f"llvm.getelementptr {base}[{index}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, {_MEMORY_TYPE[dtype]}")

    def to_shared(self, v: _Value) -> None:
        """Write every element of v to shared memory at its row-major index."""
        self.reserve(self.count(v.type.shape) * element_bytes(v.type.dtype))
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

    def view_address(self, view: View, coords: list[str], addressing: list | None = None) -> tuple[str, str | None, str | None]:
        """Element offset, in-bounds condition and in-padding condition for logical coordinates.

        addressing: if given, receives the bounds checks of the axes the address depends on
        (a nonzero stride or padding); see masked_load."""
        inside: str | None = None
        padded: str | None = None
        offset = self.dim(view.offset)
        for axis, c in enumerate(coords):
            ok = self.emit(f"llvm.icmp \"ult\" {c}, {self.dim(view.shape[axis])} : i32")
            inside = self.conj(inside, ok)
            if addressing is not None and (view.strides[axis] != 0 or (view.pad is not None and view.pad[axis] != (0, 0))):
                addressing.append(ok)
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
        addressing: list[str] = []
        offset, inside, padded = self.view_address(view, coords, addressing)
        ok = inside
        if padded is not None:
            ok = self.conj(ok, self.emit(f"llvm.xor {padded}, {self.const(True, 'i1')} : i1"))
        ok = self.conj(ok, slot_valid) or self.const(True, "i1")
        # Masked lanes read element 0 instead (buffers are never empty), then select `other`:
        # branch-free, and the hardware cannot fault on a masked address. Only the axes the
        # address depends on decide whether it is safe: a row statistic broadcast along a tile's
        # row then has the same address in every slot of the row, and LLVM loads it once (the
        # bounds of the broadcast axis still select `other` below).
        safe_ok = None
        for c in addressing:
            safe_ok = self.conj(safe_ok, c)
        if padded is not None:
            safe_ok = self.conj(safe_ok, self.emit(f"llvm.xor {padded}, {self.const(True, 'i1')} : i1"))
        safe = offset if safe_ok is None else self.emit(f"llvm.select {safe_ok}, {offset}, {self.i32(0)} : i1, i32")
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
            value = self.make(out_t, [reg] * self.slots(out_t.shape))
            value.rebuild = lambda like, reg=reg: [reg] * self.n_slots(like)
            return [value]
        if kind == "dim":
            return [self.make(out_t, [self.dim(attrs["value"])])]
        if kind == "iota":
            start = args[0].regs[0]
            axis = attrs["axis"]

            def iota_at(like, start=start, axis=axis):
                return [self.emit(f"llvm.add {start}, {c[axis]} : i32") for c in self.layout_of(like)]

            value = self.make(out_t, iota_at(_Value([], False, out_t)))
            value.rebuild = iota_at
            return [value]
        if kind == "reshape":
            src = args[0]
            if src.coords is None:
                return [self.make(out_t, list(src.regs))]
            if [d for d in src.type.shape if d != 1] != [d for d in out_t.shape if d != 1]:
                return [self.make(out_t, list(self.cyclic(src).regs))]
            # Only unit axes appear or disappear: relabel the coordinates.
            return [self.relabel(src, out_t, self.unit_map(src.type.shape, out_t.shape))]
        if kind == "permute" and args[0].coords is not None:
            perm = attrs["perm"]
            return [self.relabel(args[0], out_t, list(perm))]
        if kind == "broadcast" and (args[0].rebuild is not None or isinstance(args[0].structure, MmaRows)) and not args[0].replicated:
            return [self.lazy_broadcast(args[0], out_t, attrs["dims"])]
        if kind == "broadcast" and args[0].replicated:
            # A splat: free in any layout.
            value = self.make(out_t, self.spread(args[0], out_t))
            value.rebuild = lambda like, reg=args[0].regs[0]: [reg] * self.n_slots(like)
            return [value]
        if kind == "permute" and args[0].rebuild is not None and args[0].coords is None:
            # Recompute at permuted coordinates instead of exchanging through shared memory; if
            # nothing reads the registers (a dot that copies its operand from memory), the
            # recomputation is dead code, where an exchange would leave its stores and barriers.
            return [self.lazy_permute(args[0], out_t, attrs["perm"])]
        if kind in ("broadcast", "permute"):
            return [self.exchange(op, self.cyclic(args[0]), out_t)]
        if kind in ("cmp", "where") or (kind not in ("full", "dim", "iota", "cast", "reduce", "dot", "load", "load_indexed", "store", "atomic_add", "atomic_add_indexed", "for")):
            args = self.unify(args)
        if kind in ("cmp", "where", "cast") or kind in _ELEMENTWISE:
            value = self.pointwise(op, args, out_t)
            sources = [a for a in args if not a.replicated]
            if sources and all(a.rebuild is not None for a in sources):
                value.rebuild = lambda like, op=op, args=args, out_t=out_t: self.pointwise(op, [a if a.replicated else self.rebuilt(a, self.template(a.type, like)) for a in args], out_t).regs
            return [value]
        if kind == "reduce":
            if isinstance(args[0].structure, MmaAcc):
                rows = self.mma_row_reduce(op, args[0], out_t)
                if rows is not None:
                    return [rows]
            return [self.reduce(op, self.cyclic(args[0]), out_t)]
        if kind == "dot":
            a, b, acc = args
            if self.use_matrix:
                from olaf.backend.cuda.mma import lower_register_dot

                lowered = lower_register_dot(self, op, a, b, acc)
                if lowered is not None:
                    return [lowered]
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
        if kind == "atomic_add":
            self.atomic_store(op, args)
            return []
        if kind == "for":
            if self.use_matrix and op.attr("stages", 1) > 1:
                from olaf.backend.cuda.mma import lower_matrix_loop

                lowered = lower_matrix_loop(self, op, args)
                if lowered is not None:
                    return lowered
            return self.loop(op, args)
        raise NotImplementedError(f"CUDA lowering of tile op {kind}")

    def pointwise(self, op: TOp, args: list[_Value], out_t: TileType) -> _Value:
        """cmp, where, cast and the elementwise ops, slot by slot (args already in one layout)."""
        kind, attrs = op.kind, dict(op.attrs)
        like = next((a for a in args if a.coords is not None), None)
        if kind == "cmp":
            a, b = args
            return self.make(out_t, [self.compare(attrs["fn"], x, y, a.type.dtype) for x, y in zip(a.regs, b.regs)], a)
        if kind == "where":
            c, a, b = args
            t = _LLVM[a.type.dtype]
            n = max(len(c.regs), len(a.regs), len(b.regs))
            pick = lambda v, i: v.regs[i] if len(v.regs) > 1 else v.regs[0]  # noqa: E731
            return self.make(out_t, [self.emit(f"llvm.select {pick(c, i)}, {pick(a, i)}, {pick(b, i)} : i1, {t}") for i in range(n)], like)
        if kind == "cast":
            return self.make(out_t, [self.cast(r, args[0].type.dtype, out_t.dtype) for r in args[0].regs], args[0])
        dtype = args[0].type.dtype
        n = max(len(a.regs) for a in args)
        regs = [self.scalar_op(kind, [a.regs[i] if len(a.regs) > 1 else a.regs[0] for a in args], dtype, attrs) for i in range(n)]
        return self.make(out_t, regs, like)

    @staticmethod
    def unit_map(src_shape, out_shape) -> list[int | None]:
        """For each output axis of a unit-only reshape: the source axis it is, or None (a new unit axis)."""
        big = [i for i, d in enumerate(src_shape) if d != 1]
        out = []
        for d in out_shape:
            out.append(big.pop(0) if d != 1 else None)
        return out

    def relabel(self, src: _Value, out_t: TileType, mapping: list[int | None]) -> _Value:
        """src with output axis k taken from source axis mapping[k] (None: a unit axis). No data moves."""
        zero = self.i32(0)
        coords = [[c[m] if m is not None else zero for m in mapping] for c in src.coords]
        structure = src.structure
        if structure is not None:
            inverse = {m: k for k, m in enumerate(mapping) if m is not None}
            if isinstance(structure, MmaAcc):
                structure = MmaAcc(structure.em, structure.en, structure.wm, structure.wn, inverse[structure.row_axis], inverse[structure.col_axis], out_t.rank)
            else:
                structure = MmaRows(structure.acc, inverse[structure.row_axis], out_t.rank)
        # Relabeled coordinates are entry-block names; caching them persistently keeps one list
        # object per layout, which is how layouts are compared (by identity).
        key = ("relabel", id(src.coords), tuple(mapping))
        if key not in self.coords_cache:
            self.coords_cache[key] = coords
            self.keep.append(src.coords)
        value = _Value(list(src.regs), False, out_t, self.coords_cache[key], src.valid, structure)
        if src.rebuild is not None:
            # like's coordinates in the output -> the source's axes.
            def back(like, src=src, mapping=mapping):
                source_coords = []
                for c in self.layout_of(like):
                    sc = [self.i32(0)] * src.type.rank
                    for k, m in enumerate(mapping):
                        if m is not None:
                            sc[m] = c[k]
                    source_coords.append(sc)
                pseudo = _Value([], False, src.type, source_coords, like.valid, None)
                return src.rebuild(pseudo)

            value.rebuild = back
        return value

    def lazy_permute(self, src: _Value, out_t: TileType, perm) -> _Value:
        def at(like, src=src, perm=perm):
            source_coords = []
            for c in self.layout_of(like):
                sc = [None] * src.type.rank
                for k, p in enumerate(perm):
                    sc[p] = c[k]
                source_coords.append(sc)
            return src.rebuild(_Value([], False, src.type, source_coords, like.valid, None))

        regs = at(_Value([], self.count(out_t.shape) == 1, out_t))
        return _Value(regs, self.count(out_t.shape) == 1, out_t, rebuild=at)

    def rows_broadcast_layout(self, rows: MmaRows, out_t: TileType, dims) -> _Value | None:
        """The accumulator layout a broadcast of MmaRows along a new column axis lands in.

        Built with the same cache key as a unit reshape of the accumulator (relabel), so it is
        the same layout object as the tiles it will be combined with."""
        acc = rows.acc
        new_axes = [k for k in range(out_t.rank) if k not in dims]
        cols = [k for k in new_axes if out_t.shape[k] == acc.en]
        if len(cols) != 1 or any(out_t.shape[k] != 1 for k in new_axes if k != cols[0]):
            return None
        base = self.coords_cache.get(("mma-acc", acc.em, acc.en, acc.wm, acc.wn, self.threads))
        if base is None:
            return None
        out_row = dims[rows.row_axis]
        mapping = [0 if k == out_row else 1 if k == cols[0] else None for k in range(out_t.rank)]
        template = _Value([], False, TileType((acc.em, acc.en), out_t.dtype), base, None, MmaAcc(acc.em, acc.en, acc.wm, acc.wn, 0, 1, 2))
        return self.relabel(template, out_t, mapping)

    def rebuild_rows(self, src: _Value, dims, like: _Value) -> list[str] | None:
        """A row-constant value (src varies only along the axis that becomes like's rows),
        recomputed once per accumulator row a thread holds instead of once per element: four
        loads per thread for a 64-row tile where per-element recomputation issues sixteen."""
        acc = like.structure
        mapped = [i for i, d in enumerate(dims) if d == acc.row_axis]
        if len(mapped) != 1 or any(src.type.shape[i] != 1 for i in range(src.type.rank) if i != mapped[0]):
            return None
        row_axis = mapped[0]
        base = self.coords_cache.get(("mma-acc", acc.em, acc.en, acc.wm, acc.wn, self.threads))
        if base is None:
            return None
        zero = self.i32(0)
        coords = []
        for i in range(acc.mi):
            for h in range(2):
                row = base[(i * acc.nj) * 4 + 2 * h][0]
                coords.append([row if k == row_axis else zero for k in range(src.type.rank)])
        return src.rebuild(_Value([], False, src.type, coords, None, None))

    def lazy_broadcast(self, src: _Value, out_t: TileType, dims) -> _Value:
        """A broadcast emitted in its consumer's layout (see relayout); cyclic by default, except
        that rows of an accumulator land in the accumulator's layout (register-local)."""

        def at(like, src=src, dims=dims):
            if isinstance(src.structure, MmaRows) and isinstance(like.structure, MmaAcc) and like.structure == _acc_like(src.structure, like.structure):
                acc = like.structure
                return [src.regs[i * 2 + q // 2] for i in range(acc.mi) for j in range(acc.nj) for q in range(4)]
            if src.rebuild is not None and isinstance(like.structure, MmaAcc):
                rows = self.rebuild_rows(src, dims, like)
                if rows is not None:
                    acc = like.structure
                    return [rows[i * 2 + q // 2] for i in range(acc.mi) for j in range(acc.nj) for q in range(4)]
            if src.rebuild is None:
                return self.exchange_into(src, out_t, dims, like)
            source_coords = [[c[d] if src.type.shape[i] != 1 else self.i32(0) for i, d in enumerate(dims)] for c in self.layout_of(like)]
            return src.rebuild(_Value([], False, src.type, source_coords, like.valid, None))

        if isinstance(src.structure, MmaRows):
            target = self.rows_broadcast_layout(src.structure, out_t, dims)
            if target is not None:
                regs = at(target)
                return _Value(regs, False, out_t, target.coords, None, target.structure, rebuild=at)
        regs = at(_Value([], self.count(out_t.shape) == 1, out_t))
        return _Value(regs, self.count(out_t.shape) == 1, out_t, rebuild=at)

    def exchange_into(self, src: _Value, out_t: TileType, dims, like: _Value) -> list[str]:
        """A broadcast through shared memory, read in like's layout."""
        self.to_shared(src)
        regs = []
        for slot, coords in enumerate(self.layout_of(like)):
            source = [self.i32(0) if src.type.shape[i] == 1 else coords[d] for i, d in enumerate(dims)]
            index = self.linear(source, src.type.shape)
            guard = self.valid_of(like, slot)
            if guard is not None:
                index = self.emit(f"llvm.select {guard}, {index}, {self.i32(0)} : i1, i32")
            regs.append(self.load_shared(index, src.type.dtype))
        return regs

    def mma_row_reduce(self, op: TOp, src: _Value, out_t: TileType) -> _Value | None:
        """Reduce an accumulator tile over its columns without leaving registers when possible.

        Each thread combines its own columns of a row, the 4 lanes of a quad (which hold one row)
        combine with shuffles, and warps along n (if any) through shared memory.
        """
        acc = src.structure
        axes = op.attr("axes")
        fn = op.attr("fn")
        dtype = src.type.dtype
        others = [i for i in range(src.type.rank) if i not in (acc.row_axis, acc.col_axis)]
        if tuple(axes) != (acc.col_axis,) or any(src.type.shape[i] != 1 for i in others) or _BYTES[dtype] != 4:
            return None
        combine = "add" if fn == "sum" else "max"
        rows = []
        for i in range(acc.mi):
            for h in range(2):
                parts = [src.regs[(i * acc.nj + j) * 4 + 2 * h + c] for j in range(acc.nj) for c in range(2)]
                value = parts[0]
                for x in parts[1:]:
                    value = self.scalar_op(combine, [value, x], dtype, {})
                rows.append(self.butterfly(value, 4, dtype, combine))
        if acc.wn > 1:
            lane = self.entry(f"llvm.urem %tid, {self.i32(32)} : i32")
            warp = self.entry(f"llvm.udiv %tid, {self.i32(32)} : i32")
            warp_n = self.entry(f"llvm.udiv {warp}, {self.i32(acc.wm)} : i32")
            row_coords = self.rows_layout(acc, out_t, op)
            self.reserve(acc.wn * acc.em * 4)
            self.barrier()
            first = self.emit(f"llvm.icmp \"eq\" {self.emit(f'llvm.urem {lane}, {self.i32(4)} : i32')}, {self.i32(0)} : i32")
            for slot, value in enumerate(rows):
                row = row_coords[slot][-1] if False else row_coords[slot][self.row_position(acc, out_t)]
                index = self.emit(f"llvm.add {self.emit(f'llvm.mul {warp_n}, {self.i32(acc.em)} : i32')}, {row} : i32")
                self.guarded(first, lambda index=index, value=value: self.store_shared(index, value, dtype))
            self.barrier()
            combined = []
            for slot in range(len(rows)):
                row = row_coords[slot][self.row_position(acc, out_t)]
                parts = [self.load_shared(row if w == 0 else self.emit(f"llvm.add {row}, {self.i32(w * acc.em)} : i32"), dtype) for w in range(acc.wn)]
                value = parts[0]
                for x in parts[1:]:
                    value = self.scalar_op(combine, [value, x], dtype, {})
                combined.append(value)
            rows = combined
        position = self.row_position(acc, out_t)
        return _Value(rows, False, out_t, self.rows_layout(acc, out_t, op), None, MmaRows(acc, position, out_t.rank))

    @staticmethod
    def row_position(acc: MmaAcc, out_t: TileType) -> int:
        return acc.row_axis - (1 if acc.col_axis < acc.row_axis else 0)

    def rows_layout(self, acc: MmaAcc, out_t: TileType, op: TOp) -> list[list[str]]:
        """Coordinates of MmaRows slots in out_t (cached per accumulator geometry and rank)."""
        key = ("mma-rows", acc.em, acc.en, acc.wm, acc.wn, out_t.rank, self.row_position(acc, out_t))
        if key in self.coords_cache:
            return self.coords_cache[key]
        base = self.coords_cache[("mma-acc", acc.em, acc.en, acc.wm, acc.wn, self.threads)]
        position = self.row_position(acc, out_t)
        zero = self.i32(0)
        coords = []
        for i in range(acc.mi):
            for h in range(2):
                row = base[(i * acc.nj) * 4 + 2 * h][0]
                coords.append([row if k == position else zero for k in range(out_t.rank)])
        self.coords_cache[key] = coords
        return coords

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
        fast = self.fast_reduce(op, src, out_t)
        if fast is not None:
            return fast
        return self.general_reduce(op, src, out_t)

    def fast_reduce(self, op: TOp, src: _Value, out_t: TileType) -> _Value | None:
        """Reductions whose thread layout is known statically (the cyclic layout of src).

        (A) every element a thread owns belongs to one output (a reduction over leading axes):
            each thread combines its own slots, then threads holding parts of one output are
            combined through shared memory;
        (B) the reduced axes are trailing (output = element // R): with R a multiple of the
            thread count, a butterfly within each warp and the warps' partials through shared
            memory; with R <= 32, a butterfly within groups of R lanes.
        Anything else returns None.
        """
        import numpy as np

        if src.replicated or src.coords is not None:
            return None
        axes = op.attr("axes")
        fn = op.attr("fn")
        dtype = src.type.dtype
        shape = src.type.shape
        n, t = self.count(shape), self.threads
        kept = [i for i in range(len(shape)) if i not in axes]
        out_shape = out_t.shape
        n_out = self.count(out_shape)
        coords = np.unravel_index(np.arange(n), shape)
        out_index = np.ravel_multi_index([coords[i] for i in kept], out_shape) if kept else np.zeros(n, np.int64)
        combine = "add" if fn == "sum" else "max"
        neutral = 0 if fn == "sum" else (float("-inf") if dtype.is_float else -(2**31))
        thirty_two = _BYTES[dtype] == 4

        def fold(values: list[str]) -> str:
            acc = values[0]
            for v in values[1:]:
                acc = self.scalar_op(combine, [acc, v], dtype, {})
            return acc

        def slot_value(slot: int) -> str:
            guard = self.valid(shape, slot)
            reg = src.regs[slot]
            return reg if guard is None else self.emit(f"llvm.select {guard}, {reg}, {self.typed_const(neutral, dtype)} : i1, {_LLVM[dtype]}")

        # Trailing reductions first: a butterfly beats any shared-memory pass when it applies.
        r = n // n_out
        trailing = np.array_equal(out_index, np.arange(n) // r) and thirty_two
        if trailing and r % t == 0:
            per = r // t
            warps = t // 32
            self.reserve(n_out * warps * _BYTES[dtype])
            lane = self.entry(f"llvm.urem %tid, {self.i32(32)} : i32")
            warp = self.entry(f"llvm.udiv %tid, {self.i32(32)} : i32")
            totals = [self.butterfly(fold([src.regs[o * per + k] for k in range(per)]), 32, dtype, combine) for o in range(n_out)]
            self.barrier()
            first = self.emit(f"llvm.icmp \"eq\" {lane}, {self.i32(0)} : i32")
            for o, total in enumerate(totals):
                index = self.emit(f"llvm.add {warp}, {self.i32(o * warps)} : i32")
                self.guarded(first, lambda index=index, total=total: self.store_shared(index, total, dtype))
            self.barrier()
            finals = [fold([self.load_shared(self.i32(o * warps + w), dtype) for w in range(warps)]) for o in range(n_out)]
            if n_out == 1:
                return self.make(out_t, finals)
            regs = []
            for slot in range(self.slots(out_shape)):
                value = finals[0]
                index = self.flat_index(slot)
                for o in range(1, n_out):
                    is_o = self.emit(f"llvm.icmp \"eq\" {index}, {self.i32(o)} : i32")
                    value = self.emit(f"llvm.select {is_o}, {finals[o]}, {value} : i1, {_LLVM[dtype]}")
                regs.append(value)
            return self.make(out_t, regs)
        if trailing and r <= 32 and t % r == 0 and r & (r - 1) == 0:
            self.reserve(n_out * _BYTES[dtype])
            self.barrier()
            leader = self.emit(f"llvm.icmp \"eq\" {self.emit(f'llvm.urem %tid, {self.i32(r)} : i32')}, {self.i32(0)} : i32")
            for slot in range(len(src.regs)):
                total = self.butterfly(slot_value(slot), r, dtype, combine)
                index = self.emit(f"llvm.udiv {self.flat_index(slot)}, {self.i32(r)} : i32")
                guard = self.conj(leader, self.valid(shape, slot))
                self.guarded(guard, lambda index=index, total=total: self.store_shared(index, total, dtype))
            self.barrier()
            if n_out == 1:
                # A single output is replicated: every thread reads it, not "its own" element.
                return self.make(out_t, [self.load_shared(self.i32(0), dtype)])
            regs = []
            for slot in range(self.slots(out_shape)):
                index = self.flat_index(slot)
                guard = self.valid(out_shape, slot)
                if guard is not None:
                    index = self.emit(f"llvm.select {guard}, {index}, {self.i32(0)} : i1, i32")
                regs.append(self.load_shared(index, dtype))
            return self.make(out_t, regs)
        lanes = np.arange(min(n, t))
        if all(out_index[e] == out_index[e % t] for e in range(n)) and n_out <= t and t % n_out == 0 and np.array_equal(out_index[lanes], lanes % n_out):
            partial = fold([slot_value(s) for s in range(len(src.regs))])
            if n_out == t:
                return self.make(out_t, [partial])
            contributors = min(n, t) // n_out if n >= t else -(-n // n_out)
            self.reserve(t * _BYTES[dtype])
            self.barrier()
            self.store_shared("%tid", partial, dtype) if n >= t else self.guarded(self.emit(f"llvm.icmp \"ult\" %tid, {self.i32(n)} : i32"), lambda: self.store_shared("%tid", partial, dtype))
            self.barrier()
            owner = "%tid" if n_out > 1 else self.i32(0)
            if n_out > 1:
                owner = self.emit(f"llvm.select {self.valid(out_shape, 0) or self.const(True, 'i1')}, %tid, {self.i32(0)} : i1, i32")
            parts = [self.load_shared(owner if k == 0 else self.emit(f"llvm.add {owner}, {self.i32(k * n_out)} : i32"), dtype) for k in range(contributors)]
            return self.make(out_t, [fold(parts)])

        return None

    def butterfly(self, value: str, width: int, dtype: DType, combine: str) -> str:
        """Combine `value` across groups of `width` consecutive lanes; every lane gets the result."""
        t = _LLVM[dtype]
        offset = width // 2
        while offset:
            other = self.emit(f"nvvm.shfl.sync bfly {self.const(-1, 'i32')}, {value}, {self.i32(offset)}, {self.i32(31)} : {t} -> {t}")
            value = self.scalar_op(combine, [value, other], dtype, {})
            offset //= 2
        return value

    def general_reduce(self, op: TOp, src: _Value, out_t: TileType) -> _Value:
        # Every thread computes the output elements it owns by walking the reduced elements in
        # shared memory in row-major order. Correct for any shape, and slow.
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
        self.reserve(a_bytes + k * n * element_bytes(b.type.dtype))
        self.barrier()
        self.write_tile(a, 0)
        self.write_tile(b, a_bytes)
        self.barrier()
        accs = list(acc.regs)
        out = []
        for slot, coords in enumerate(self.coords(acc.type.shape)):
            row, col = coords
            # This k step's products go into a partial sum of their own, added to acc once: one
            # chain of fma over all of K (784 for an MNIST layer) has a 3.5x larger error than
            # cuBLAS (5.1e-7 against 1.4e-7 relative), and the partials give cuBLAS's. One add
            # per k step of 32. The first product starts the partial, so no zero is needed.
            total = None
            for kk in range(k):
                ai = self.emit(f"llvm.add {self.emit(f'llvm.mul {row}, {self.i32(k)} : i32')}, {self.i32(kk)} : i32")
                bi = self.emit(f"llvm.add {self.i32(kk * n)}, {col} : i32")
                x = self.to_acc(self.load_shared_at(ai, a.type.dtype, 0), a.type.dtype, acc.type.dtype)
                y = self.to_acc(self.load_shared_at(bi, b.type.dtype, a_bytes), b.type.dtype, acc.type.dtype)
                if total is None:
                    total = self.emit(f"llvm.fmul {x}, {y} : {acc_t}")
                else:
                    total = self.emit(f"llvm.intr.fma({x}, {y}, {total}) : ({acc_t}, {acc_t}, {acc_t}) -> {acc_t}")
            out.append(accs[slot] if total is None else self.emit(f"llvm.fadd {accs[slot]}, {total} : {acc_t}"))
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
        base = self.smem_base()
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

        broadcast_scalar = index is None and view.pad is None and all(st == 0 for st in view.strides)
        scalar = None
        if broadcast_scalar:
            # Every element is the same address: read it once. Lanes outside the view still get
            # `other` (reductions rely on it), which is a compare and a select instead of a load.
            zeros = [self.i32(0)] * len(view.shape)
            scalar = self.masked_load(arg, view, zeros, dtype, 0, view.fill, None)

        def at(like):
            if scalar is not None:
                regs = []
                other = self.typed_const(op.attr("other"), dtype)
                for slot, coords in enumerate(self.layout_of(like)):
                    tile = self.tile_coords(starts, out_t.shape, coords)
                    inside = self.valid_of(like, slot)
                    for axis, c in enumerate(tile):
                        if out_t.shape[axis] != 1 or not isinstance(view.shape[axis], int) or view.shape[axis] != 1:
                            inside = self.conj(inside, self.emit(f"llvm.icmp \"ult\" {c}, {self.dim(view.shape[axis])} : i32"))
                    regs.append(scalar if inside is None else self.emit(f"llvm.select {inside}, {scalar}, {other} : i1, {_LLVM[dtype]}"))
                return regs
            regs = []
            for slot, coords in enumerate(self.layout_of(like)):
                tile = self.tile_coords(starts, out_t.shape, coords)
                if index is not None:
                    idx = index.regs[slot] if not index.replicated else index.regs[0]
                    idx = self.cast(idx, index.type.dtype, INDEX)
                    tile = [idx if m == -1 else tile[m] for m in op.attr("axis_map")]
                regs.append(self.masked_load(arg, view, tile, dtype, op.attr("other"), view.fill, self.valid_of(like, slot)))
            return regs

        value = self.make(out_t, at(_Value([], self.count(out_t.shape) == 1, out_t)))
        # Re-reading in another layout is safe when this task never writes the argument.
        if index is None and not self.task.args[arg].writes:
            value.rebuild = at
        return value

    def store(self, op: TOp, args: list[_Value]) -> None:
        value, *starts = args
        if self.paired_store(op, value, starts):
            return
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

    def paired_store(self, op: TOp, value: _Value, starts: list[_Value]) -> bool:
        """Store an accumulator tile two adjacent columns at a time (one 4- or 8-byte store).

        Fragment slots q and q+1 are columns c and c+1 of one row, with c even. When the store's
        column axis is the view's contiguous axis and every other stride and the offset are
        even, the pair is one aligned vector store; the launch offset of the argument is checked
        at launch (Lowered.aligned_args). A pair that straddles the end of a ragged row is
        stored element by element.
        """
        acc = value.structure
        arg = op.attr("arg")
        view = self.task.args[arg].view
        dtype = value.type.dtype
        if not isinstance(acc, MmaAcc) or dtype not in (DType.bf16, DType.f32) or view.pad is not None or self.task.args[arg].dtype is not dtype:
            return False
        col = acc.col_axis
        if view.strides[col] != 1:
            return False
        for d in [view.offset, *(st for i, st in enumerate(view.strides) if i != col)]:
            if not isinstance(d, int) or d % 2:
                return False
        self.aligned_args.add(arg)
        t = _MEMORY_TYPE[dtype]
        coords_all = self.layout_of(value)
        for slot in range(0, len(value.regs), 2):
            tile = self.tile_coords(starts, value.type.shape, coords_all[slot])
            offset, inside, _ = self.view_address(view, tile)
            inside = inside or self.const(True, "i1")
            nxt = self.emit(f"llvm.add {tile[col]}, {self.i32(1)} : i32")
            second = self.emit(f"llvm.icmp \"ult\" {nxt}, {self.dim(view.shape[col])} : i32")
            both = self.emit(f"llvm.and {inside}, {second} : i1")
            past = self.emit(f"llvm.xor {second}, {self.const(True, 'i1')} : i1")
            only = self.emit(f"llvm.and {inside}, {past} : i1")
            vec = self.emit(f"llvm.mlir.undef : vector<2x{t}>")
            vec = self.emit(f"llvm.insertelement {value.regs[slot]}, {vec}[{self.i32(0)} : i32] : vector<2x{t}>")
            vec = self.emit(f"llvm.insertelement {value.regs[slot + 1]}, {vec}[{self.i32(1)} : i32] : vector<2x{t}>")
            ptr = self.global_ptr(arg, offset, dtype)
            self.guarded(both, lambda ptr=ptr, vec=vec: self.raw(f"llvm.store {vec}, {ptr} {{alignment = {2 * _BYTES[dtype]} : i64}} : vector<2x{t}>, !llvm.ptr<1>"))
            self.guarded(only, lambda ptr=ptr, reg=value.regs[slot]: self.raw(f"llvm.store {reg}, {ptr} : {t}, !llvm.ptr<1>"))
        return True

    def atomic_store(self, op: TOp, args: list[_Value]) -> None:
        """atomic_add: a store whose writes are atomic additions (any layout)."""
        value, *starts = args
        arg = op.attr("arg")
        view = self.task.args[arg].view
        dtype = value.type.dtype
        coords_all = self.layout_of(value)
        slots = [0] if value.replicated else range(len(value.regs))
        for slot in slots:
            tile = self.tile_coords(starts, value.type.shape, coords_all[slot])
            offset, inside, _ = self.view_address(view, tile)
            ok = self.conj(inside, self.valid_of(value, slot))
            if value.replicated:
                ok = self.conj(ok, self.emit(f"llvm.icmp \"eq\" %tid, {self.i32(0)} : i32"))
            ptr = self.global_ptr(arg, offset, dtype)
            reg = value.regs[slot]
            self.guarded(ok, lambda ptr=ptr, reg=reg: self.raw(f"llvm.atomicrmw fadd {ptr}, {reg} monotonic : !llvm.ptr<1>, {_LLVM[dtype]}"))

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
        if any(c.rebuild is not None for c in carries):
            # Trial emission: find the layouts the body produces, then start the carries in them
            # (constant initial values cost nothing to lay out), so no iteration converts layouts.
            saved = self.snapshot()
            probe = self.emit_loop(op, lo, hi, step, carries)[1]
            self.restore(saved)
            carries = [
                self.rebuilt(c, r) if c.rebuild is not None and r.coords is not None and r.coords is not c.coords else c
                for c, r in zip(carries, probe)
            ]
        return self.emit_loop(op, lo, hi, step, carries)[0]

    def snapshot(self):
        # Entry-block code (constants, coordinates) is pure and stays: probed layouts refer to it.
        return (len(self.lines), dict(self.values), dict(self.memo))

    def restore(self, saved) -> None:
        lines, values, memo = saved
        del self.lines[lines:]
        self.values, self.memo = values, memo

    def emit_loop(self, op: TOp, lo, hi, step, carries) -> tuple[list[_Value], list[_Value]]:
        region = op.regions[0]
        saved_base, saved_resident, saved_streamed = self.scratch_base, dict(self.resident), dict(self.streamed)
        stream_plan = []
        if self.use_matrix:
            from olaf.backend.cuda.mma import make_resident, make_streamed, prefetch_streamed

            make_resident(self, region)
            # Double buffering is a schedule choice made in Tile IR (the loop's `stream` attribute).
            stream_plan = make_streamed(self, region, lo) if op.attr("stream", False) else []
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
        if stream_plan:
            prefetch_streamed(self, stream_plan, iv, lo, step, region.params[0])
        position = 0
        carried = []
        for p, c in zip(region.params[1:], carries):
            v = _Value(params[position : position + len(c.regs)], c.replicated, c.type, c.coords, c.valid, c.structure)
            position += len(c.regs)
            self.values[p] = v
            carried.append(v)
        memo = dict(self.memo)
        raw = self.body(region)
        results = [self.relayout(r, c) for r, c in zip(raw, carries)]
        # Values rebuilt inside the body do not dominate the code after the loop.
        self.memo = memo
        self.scratch_base, self.resident, self.streamed = saved_base, saved_resident, saved_streamed
        nxt = self.emit(f"llvm.add {iv}, {step.regs[0]} : i32")
        out = [r for v in results for r in v.regs]
        self.raw(f"llvm.br {header}({', '.join([nxt, *out])} : {', '.join(['i32', *types])})")
        self.lines.append(f"  {done}:")
        if stream_plan:
            # The last prefetch (past the end) may still be in flight into memory that scratch
            # is about to reuse.
            self.raw("nvvm.cp.async.wait.group 0")
            self.barrier()
        return carried, raw

    # -- driver ---------------------------------------------------------------------------------

    def run(self) -> Lowered:
        self.producers = {}

        def walk(body: Body) -> None:
            for o in body.ops:
                self.producers.update((v, o) for v in o.outputs)
                for r in o.regions:
                    walk(r)

        walk(self.task.body)
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
        lowered = Lowered(text, dims, self.shared_bytes, self.threads, list(self.aligned), sorted(self.aligned_args))
        if self.aligned or self.aligned_args:
            lowered.fallback = _Emitter(self.task, use_matrix=False).run()
        return lowered


def lower_task(task: Task) -> Lowered:
    return _Emitter(task).run()


def _acc_like(rows: MmaRows, target: MmaAcc) -> MmaAcc:
    """The MmaAcc a broadcast of `rows` along the columns must have to be register-local."""
    acc = rows.acc
    return MmaAcc(acc.em, acc.en, acc.wm, acc.wn, target.row_axis, target.col_axis, target.rank)
