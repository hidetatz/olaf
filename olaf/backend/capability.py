"""Capability descriptors: the only place that hides vendor differences from upper layers.

Upper layers ask "what kind of async copy exists", never "is this Ampere". Supporting new
hardware starts by adding a descriptor here, with no change to upper layers.

The development machine's descriptor (AMPERE_RTX3060) holds values read from the device and
tuning measured on it. The others (A100, RTX 4090, H100) hold values from NVIDIA's
documentation, unchecked on a device, and the RTX 3060's tuning until they are measured: they
exist so that programs can be built and compiled for those targets (PTX) and so that a
descriptor change shows where upper layers would decide differently. Features they list that
lowering does not implement (TMA, clusters) are not used.
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
class Tuning:
    """Default schedule parameters of the conversion to Tile IR (tile/from_graph.py), where
    the schedule search has not chosen (it chooses dot schedules only; search/tune.py).
    measured_on names the device the values were measured on."""

    measured_on: str = "nvidia-ampere-sm86"
    # Elementwise tiles: up to elementwise_block along the last axis, up to
    # elementwise_elements in all (8 per thread at 128 threads).
    elementwise_block: int = 128
    elementwise_elements: int = 1024
    # Reduced chunk per loop step of a row reduction (one value per output: 8 registers per
    # thread at 128 threads); rows up to whole_row_limit are reduced in one step (more warps,
    # at most 8 elements per thread), as row fusion (tile/rows.py) needs.
    row_reduce_block: int = 1024
    whole_row_limit: int = 8192
    # A split reduction aims at up to split_blocks_per_unit blocks per compute unit, and a
    # column reduction reads column_reduce_block rows per loop step. Measured on the RTX 3060
    # by graph replay, column sums of 4096 x {768, 2304, 3072} bf16 and 4096 x 768 f32: 4
    # blocks and 32 rows gave 224 / 149 / 190 and 303 GB/s; 8 blocks and 16 rows give
    # 255 / 309 / 318 and 311 GB/s. 32 rows is the part that hurt, though not monotonically in
    # width (2304 columns was the slowest).
    split_blocks_per_unit: int = 8
    column_reduce_block: int = 16
    # Dots without matrix instructions: M, N, K tile.
    dot_block: tuple[int, int, int] = (32, 32, 32)
    # With a 16x8x16 matrix instruction: 128x128 output tiles over 8 warps (64x32 per warp)
    # and a 3-stage async copy pipeline over 32-wide K steps. 3 stages of bf16 A and B tiles
    # take 48 KiB of shared memory, which leaves room on Ampere (100 KiB per SM) for two
    # blocks per SM.
    matrix_dot_block: tuple[int, int, int] = (128, 128, 32)
    matrix_dot_warps: int = 8
    matrix_dot_stages: int = 3


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
    target: str = ""  # code generation target (an NVIDIA "sm_XY"); empty: the running device's
    tuning: Tuning = Tuning()

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
    target="sm_86",
)

_AMPERE_MMA = AMPERE_RTX3060.matrix_instructions

# From NVIDIA's documentation, not checked on a device (see the module docstring). A100 (GA100,
# sm_80): 108 SMs, 164 KiB of shared memory per SM, up to 163 KiB per block with opt-in.
AMPERE_A100 = Capability(
    name="nvidia-ampere-sm80",
    matrix_instructions=_AMPERE_MMA,
    execution=ExecutionUnit(widths=(32,)),
    on_chip=OnChipMemory(shared_bytes=166912, registers_per_thread=255, shared_bytes_per_unit=167936),
    async_copy=AsyncCopy.software,
    cooperation=Cooperation(clusters=False, distributed_shared_memory=False, forward_progress=ForwardProgress.best_effort),
    memory_model=MemoryModel(device_scope_atomics=True, acquire_release=True),
    communication=Communication(device_initiated=False),
    graph_replay=GraphReplay.cuda_graphs,
    roofline=Roofline(compute_units=108),
    target="sm_80",
)

# RTX 4090 (AD102, sm_89): 128 SMs, 100 KiB of shared memory per SM (99 KiB per block). FP8
# matrix instructions exist on sm_89 but are left out until a phase uses them.
ADA_RTX4090 = Capability(
    name="nvidia-ada-sm89",
    matrix_instructions=_AMPERE_MMA,
    execution=ExecutionUnit(widths=(32,)),
    on_chip=OnChipMemory(shared_bytes=101376, registers_per_thread=255, shared_bytes_per_unit=102400),
    async_copy=AsyncCopy.software,
    cooperation=Cooperation(clusters=False, distributed_shared_memory=False, forward_progress=ForwardProgress.best_effort),
    memory_model=MemoryModel(device_scope_atomics=True, acquire_release=True),
    communication=Communication(device_initiated=False),
    graph_replay=GraphReplay.cuda_graphs,
    roofline=Roofline(compute_units=128),
    target="sm_89",
)

# H100 SXM (GH100, sm_90): 132 SMs, 228 KiB of shared memory per SM (227 KiB per block),
# TMA, thread block clusters with distributed shared memory. Lowering uses none of these yet
# (CLAUDE.md: Hopper features are described, not implemented), nor wgmma: mma.sync still runs.
HOPPER_H100 = Capability(
    name="nvidia-hopper-sm90",
    matrix_instructions=_AMPERE_MMA,
    execution=ExecutionUnit(widths=(32,)),
    on_chip=OnChipMemory(shared_bytes=232448, registers_per_thread=255, shared_bytes_per_unit=233472),
    async_copy=AsyncCopy.hardware_bulk,
    cooperation=Cooperation(clusters=True, distributed_shared_memory=True, forward_progress=ForwardProgress.best_effort),
    memory_model=MemoryModel(device_scope_atomics=True, acquire_release=True),
    communication=Communication(device_initiated=False),
    graph_replay=GraphReplay.cuda_graphs,
    roofline=Roofline(compute_units=132),
    target="sm_90",
)

NVIDIA = (AMPERE_RTX3060, AMPERE_A100, ADA_RTX4090, HOPPER_H100)

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
