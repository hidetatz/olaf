"""Graph IR types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum

import numpy as np

from olaf.graph.dim import Dim


class DType(StrEnum):
    f32 = "f32"
    f64 = "f64"
    bf16 = "bf16"
    i32 = "i32"
    i64 = "i64"
    u32 = "u32"
    bool = "bool"

    @property
    def is_float(self) -> bool:
        return self in (DType.f32, DType.f64, DType.bf16)

    @property
    def numpy(self) -> np.dtype:
        """The NumPy dtype that holds values of this type in the reference interpreters.

        NumPy has no bf16, so bf16 values are held in float32 arrays, restricted to the bf16 grid:
        the interpreters round every bf16 result (graph.ir.bf16_round). Constants keep the bf16 bit
        patterns instead (ConstantData), and device buffers store the 16-bit encoding.
        """
        return np.dtype({"f32": "float32", "bf16": "float32", "f64": "float64", "i32": "int32", "i64": "int64", "u32": "uint32", "bool": "bool"}[self.value])


class StateKind(Enum):
    param = "param"
    optim = "optim"
    kv_cache = "kv_cache"
    rng = "rng"
    buffer = "buffer"  # model state that is not trained (batch norm's running statistics)


@dataclass(frozen=True)
class Sharding:
    """Mesh axis name for each dimension. In P0-P2 this is only a field: no propagation or communication insertion."""

    axes: tuple[str | None, ...]


@dataclass(frozen=True)
class TensorType:
    shape: tuple[Dim, ...]
    dtype: DType
    # Excluded from type equality until P3 defines propagation rules. Ops drop the field (with_shape,
    # with_dtype) and binary ops keep the left operand's, so comparing it would turn `2.0 * x` versus
    # `x * 2.0` into a type error at write_state and scan. It is still printed, so it stays in the cache key.
    sharding: Sharding | None = field(default=None, compare=False)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def with_shape(self, shape) -> TensorType:
        return TensorType(tuple(shape), self.dtype)

    def with_dtype(self, dtype: DType) -> TensorType:
        return TensorType(self.shape, dtype)

    def __str__(self) -> str:
        text = f"{self.dtype}[{','.join(map(str, self.shape))}]"
        if self.sharding is not None:
            text += "{" + ",".join(a or "_" for a in self.sharding.axes) + "}"
        return text


@dataclass(frozen=True)
class StateDecl:
    """Resident state. The kind is in the type because later stages choose storage and placement by kind."""

    name: str
    type: TensorType
    kind: StateKind = StateKind.param

    def __str__(self) -> str:
        return f"@{self.name}:{self.kind.value}:{self.type}"
