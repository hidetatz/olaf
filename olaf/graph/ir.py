"""Graph IR core: immutable SSA and the canonical text used as a cache key.

Values are distinguished by identity. Without stored numbers, the same computation prints
identically however it was built, so the text works as a cache key (numbers are assigned when printing).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from olaf.graph.dim import DimExpr, Sym, symbols
from olaf.graph.types import DType, StateDecl, TensorType

_MISSING = object()


@dataclass(frozen=True, eq=False)
class Value:
    type: TensorType


@dataclass(frozen=True)
class ConstantData:
    """Constant payload. Comparison and printing use the digest, so huge weights never expand into text."""

    dtype: DType
    shape: tuple[int, ...]
    digest: str
    array: np.ndarray = field(compare=False, repr=False)

    @staticmethod
    def of(value, dtype: DType) -> ConstantData:
        """For bf16, `array` holds the uint16 bit patterns (NumPy has no bf16 dtype)."""
        array = bf16_bits(value) if dtype is DType.bf16 else np.array(value, dtype=dtype.numpy)
        array.flags.writeable = False
        digest = hashlib.sha256(array.tobytes()).hexdigest()[:16]
        return ConstantData(dtype, array.shape, digest, array)

    def __str__(self) -> str:
        if self.array.ndim == 0:
            scalar = bf16_values(self.array) if self.dtype is DType.bf16 else self.array
            return f"{scalar.item()!r}"
        return f"dense<{self.digest}>"


_BF16_MIN_QUANTUM_EXP = -133  # subnormal spacing: 2^-126 (smallest normal) * 2^-7 (7 fraction bits)


def bf16_bits(value) -> np.ndarray:
    """Round to bf16 (to nearest, ties to even) and return the bit patterns as uint16.

    Rounds from f64 directly. Going through f32 first rounds twice, which can land on the wrong
    neighbour when the f32 result sits exactly halfway between two bf16 values. Scaling by a power
    of two and rint are exact in f64, and the result has at most 8 significant bits, so it is
    exactly representable in f32, whose upper 16 bits are then the bf16 encoding. Values at or
    above the overflow threshold round to an exponent of 128, which the f32 conversion turns into inf.
    """
    x = np.asarray(value, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        _, exponent = np.frexp(x)
        quantum = np.ldexp(1.0, np.maximum(exponent - 8, _BF16_MIN_QUANTUM_EXP))
        rounded = np.where(np.isfinite(x), np.rint(x / quantum) * quantum, x).astype(np.float32)
    bits = (rounded.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    # Truncating a NaN's payload can leave only zero fraction bits, which would encode inf.
    return np.where(np.isnan(x), np.uint16(0x7FC0), bits).astype(np.uint16)


def bf16_values(bits: np.ndarray) -> np.ndarray:
    """Decode bf16 bit patterns to f32, exactly."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


@dataclass(frozen=True, eq=False)
class Op:
    kind: str
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    regions: tuple[Graph, ...] = ()

    def attr(self, name: str, default: Any = _MISSING) -> Any:
        for key, value in self.attrs:
            if key == name:
                return value
        if default is _MISSING:
            raise KeyError(f"{self.kind} has no attribute {name}")
        return default

    @property
    def attr_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


@dataclass(frozen=True, eq=False)
class Graph:
    """A closed function. Regions do not reference parent values either.

    Being closed, a region can be printed and cached alone, and optimizing or searching one block
    does not depend on outer context. Outer values must be passed explicitly as inputs.
    """

    params: tuple[Value, ...]
    ops: tuple[Op, ...]
    results: tuple[Value, ...]

    def text(self) -> str:
        # Constraints change simplification and bind-time checks (and later tiling), so they are part
        # of the key; the shapes alone print only symbol names.
        table = symbol_table(self)
        header = ["symbols " + ", ".join(_symbol_text(s) for s in table.values())] if table else []
        return "\n".join([*header, *_print_graph(self, "main", 0)]) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()

    def states(self) -> tuple[StateDecl, ...]:
        """Resident state touched by this function, including in regions, sorted by name."""
        found: dict[str, StateDecl] = {}

        def visit(graph: Graph) -> None:
            for op in graph.ops:
                if op.kind in ("read_state", "write_state"):
                    decl = op.attr("state")
                    if found.setdefault(decl.name, decl) != decl:
                        raise ValueError(f"conflicting declarations of resident state {decl.name}")
                for region in op.regions:
                    visit(region)

        visit(self)
        return tuple(found[k] for k in sorted(found))


def symbol_table(graph: Graph) -> dict[str, Sym]:
    """Every symbol in the graph and its regions, sorted by name.

    Dimensions are bound by name, so two symbols with one name and different constraints are an error.
    """
    found: dict[str, Sym] = {}

    def add_dims(dims) -> None:
        for d in dims:
            for s in symbols(d):
                if found.setdefault(s.name, s) != s:
                    raise ValueError(f"symbol {s.name} has conflicting constraints: {found[s.name]!r} and {s!r}")

    def add_attr(value: Any) -> None:
        if isinstance(value, DimExpr):
            add_dims([value])
        elif isinstance(value, tuple):
            for v in value:
                add_attr(v)
        elif isinstance(value, TensorType):
            add_dims(value.shape)
        elif isinstance(value, StateDecl):
            add_dims(value.type.shape)

    def visit(g: Graph) -> None:
        for p in g.params:
            add_dims(p.type.shape)
        for op in g.ops:
            for v in op.outputs:
                add_dims(v.type.shape)
            for _, value in op.attrs:
                add_attr(value)
            for region in op.regions:
                visit(region)

    visit(graph)
    return {name: found[name] for name in sorted(found)}


def _symbol_text(s: Sym) -> str:
    constraints = [f"multiple_of={s.multiple_of}"] if s.multiple_of != 1 else []
    constraints += [f"min={s.min}"] if s.min != 0 else []
    constraints += [f"max={s.max}"] if s.max is not None else []
    return s.name + ("{" + ", ".join(constraints) + "}" if constraints else "")


def _attr_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and not isinstance(value, DType):
        return f'"{value}"'
    if isinstance(value, tuple):
        return "[" + ",".join(_attr_text(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _print_graph(graph: Graph, name: str, depth: int) -> list[str]:
    numbers: dict[Value, int] = {}

    def define(value: Value) -> str:
        numbers[value] = len(numbers)
        return f"%{numbers[value]}"

    def use(value: Value) -> str:
        if value not in numbers:
            raise ValueError("reference to an undefined value")
        return f"%{numbers[value]}"

    pad = "  " * depth
    params = ", ".join(f"{define(p)}: {p.type}" for p in graph.params)
    lines = [f"{pad}graph @{name}({params}) {{"]
    for op in graph.ops:
        inputs = ", ".join(use(v) for v in op.inputs)
        attrs = ""
        if op.attrs:
            attrs = " {" + ", ".join(f"{k}={_attr_text(v)}" for k, v in op.attrs) + "}"
        outs = ", ".join(define(v) for v in op.outputs)
        types = ", ".join(str(v.type) for v in op.outputs)
        lhs = f"{outs} = " if outs else ""
        rhs = f" : {types}" if types else ""
        args = f" {inputs}" if inputs else ""
        lines.append(f"{pad}  {lhs}{op.kind}{args}{attrs}{rhs}")
        for i, region in enumerate(op.regions):
            lines.extend(_print_graph(region, f"region{i}", depth + 2))
    lines.append(f"{pad}  return " + ", ".join(use(v) for v in graph.results))
    lines.append(f"{pad}}}")
    return lines
