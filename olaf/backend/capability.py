"""Capability descriptors: the only place that hides vendor differences from upper layers.

Upper layers ask "what kind of async copy exists", never "is this Ampere". Supporting new
hardware starts by adding a descriptor here, with no change to upper layers.

P0 is only the framework and nothing reads it yet. Other hardware is not described until its values can be checked on real devices.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum


class AsyncCopy(Enum):
    hardware_bulk = "hardware_bulk"  # TMA-like
    software = "software"  # cp.async-like
    none = "none"


class ForwardProgress(Enum):
    """Forward progress guarantee between blocks."""

    concurrent = "concurrent"  # all blocks are guaranteed to be resident at once
    best_effort = "best_effort"
    none = "none"


class GraphReplay(Enum):
    cuda_graphs = "cuda_graphs"
    hip_graphs = "hip_graphs"
    metal_icb = "metal_icb"
    none = "none"


@dataclass(frozen=True)
class MatrixInstruction:
    m: int
    n: int
    k: int
    dtype_a: str
    dtype_b: str
    dtype_acc: str
    scaled: bool = False


@dataclass(frozen=True)
class ExecutionUnit:
    widths: tuple[int, ...]  # warp / wavefront / subgroup width
    variable_width: bool = False


@dataclass(frozen=True)
class OnChipMemory:
    shared_bytes: int  # largest allocation one block can get (including any opt-in above the default)
    registers_per_thread: int
    dedicated_matrix_accumulator: bool = False
    shared_bytes_per_unit: int = 0  # total per compute unit; bounds how many blocks can be resident


@dataclass(frozen=True)
class Cooperation:
    clusters: bool
    distributed_shared_memory: bool
    forward_progress: ForwardProgress


@dataclass(frozen=True)
class MemoryModel:
    device_scope_atomics: bool
    acquire_release: bool


@dataclass(frozen=True)
class Communication:
    device_initiated: bool
    intra_node_gbps: float | None = None
    inter_node_gbps: float | None = None


@dataclass(frozen=True)
class Roofline:
    """Unmeasured values stay None: a guess in the cost model does more harm than a gap."""

    compute_units: int
    memory_gbytes_per_s: float | None = None
    peak_flops_by_dtype: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Capability:
    name: str
    matrix_instructions: tuple[MatrixInstruction, ...]
    execution: ExecutionUnit
    on_chip: OnChipMemory
    async_copy: AsyncCopy
    cooperation: Cooperation
    memory_model: MemoryModel
    communication: Communication
    graph_replay: GraphReplay
    roofline: Roofline

    @property
    def supports_megakernel(self) -> bool:
        # A persistent kernel needs all blocks resident at once and deadlocks without that guarantee.
        # Device-scope atomics are needed for task dependency counters shared between workers.
        return self.cooperation.forward_progress is ForwardProgress.concurrent and self.memory_model.device_scope_atomics

    def matrix_shapes(self, dtype: str) -> tuple[MatrixInstruction, ...]:
        return tuple(i for i in self.matrix_instructions if i.dtype_a == dtype)

    def digest(self) -> str:
        """Part of the compile cache key: a changed descriptor must not reuse search results."""
        text = repr(asdict(self))
        return hashlib.sha256(text.encode()).hexdigest()


# Development machine (RTX 3060 12GB, GA106). No TMA, wgmma, clusters or FP8 matrix multiply.
# Forward progress is best_effort: the GPU is shared with the display, and assuming all SMs stay
# resident gives no guarantee when display work preempts them. Revisit with P2 measurements.
# Peak compute is left out until measured in P1.
# On-chip and SM values were read from the device (cudaDeviceProp, 2026-11): 101376 bytes of shared memory
# per block with opt-in (the default is 49152, so lowering must raise
# CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES to use more), 102400 per SM, 28 SMs.
# Memory bandwidth stays None: 360 GB/s is the datasheet figure (192-bit bus x 15 Gbps), not a measurement.
# tf32 and int8 mma shapes exist on sm_86 but are omitted until a phase uses them.
AMPERE_RTX3060 = Capability(
    name="nvidia-ampere-sm86",
    matrix_instructions=(
        MatrixInstruction(16, 8, 8, "bf16", "bf16", "f32"),
        MatrixInstruction(16, 8, 16, "bf16", "bf16", "f32"),
        MatrixInstruction(16, 8, 8, "f16", "f16", "f32"),
        MatrixInstruction(16, 8, 16, "f16", "f16", "f32"),
    ),
    execution=ExecutionUnit(widths=(32,)),
    on_chip=OnChipMemory(shared_bytes=101376, registers_per_thread=255, shared_bytes_per_unit=102400),
    async_copy=AsyncCopy.software,
    cooperation=Cooperation(clusters=False, distributed_shared_memory=False, forward_progress=ForwardProgress.best_effort),
    memory_model=MemoryModel(device_scope_atomics=True, acquire_release=True),
    communication=Communication(device_initiated=False),
    graph_replay=GraphReplay.cuda_graphs,
    roofline=Roofline(compute_units=28),
)

# For tests and the simulated mesh: no matrix instructions and no on-chip memory limits.
CPU = Capability(
    name="cpu",
    matrix_instructions=(),
    execution=ExecutionUnit(widths=(1,)),
    on_chip=OnChipMemory(shared_bytes=0, registers_per_thread=0),
    async_copy=AsyncCopy.none,
    cooperation=Cooperation(clusters=False, distributed_shared_memory=False, forward_progress=ForwardProgress.none),
    memory_model=MemoryModel(device_scope_atomics=True, acquire_release=True),
    communication=Communication(device_initiated=False),
    graph_replay=GraphReplay.none,
    roofline=Roofline(compute_units=1),
)
