"""Running a Tile IR program on a CUDA device: one kernel per launch, in order.

Compilation happens once per unique task template (Program.tasks), when the program is loaded.
run() binds dimensions, allocates buffers and enqueues the launches; capture() records the same
launches into a CUDA graph for replay (graph replay mode), after which a step is one call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from olaf.backend.cuda import driver
from olaf.backend.cuda.compile import chip, compile_ptx
from olaf.backend.cuda.lower import Lowered, element_bytes, lower_task
from olaf.graph.dim import as_symbol, bind, evaluate
from olaf.graph.ir import bf16_bits, bf16_values
from olaf.graph.types import DType
from olaf.tile.interp import _symbols
from olaf.tile.ir import BufferKind, Program


def to_device_bytes(array: np.ndarray, dtype: DType) -> np.ndarray:
    if dtype is DType.bf16:
        return bf16_bits(array)
    if dtype is DType.bool:
        return np.asarray(array, np.bool_).astype(np.uint8)
    return np.ascontiguousarray(array, dtype.numpy)


def from_device(buffer: driver.DeviceBuffer, dtype: DType, shape) -> np.ndarray:
    if dtype is DType.bf16:
        return bf16_values(buffer.download(np.uint16, shape)).reshape(shape)
    if dtype is DType.bool:
        return buffer.download(np.uint8, shape).astype(bool)
    return buffer.download(dtype.numpy, shape)


@dataclass
class _Kernel:
    lowered: Lowered
    function: driver.Kernel
    fallback: _Kernel | None = None

    def select(self, env: dict[str, int]) -> _Kernel:
        """The specialized kernel when its alignment assumptions hold for these sizes, else the general one."""
        if self.fallback is not None and any(evaluate(d, env) % 8 for d in self.lowered.aligned):
            return self.fallback
        return self


def _load(lowered: Lowered, target: str) -> _Kernel:
    function = driver.Module(compile_ptx(lowered.text, target)).function("task")
    if lowered.shared_bytes > 48 * 1024:
        function.set_shared_bytes(lowered.shared_bytes)
    fallback = _load(lowered.fallback, target) if lowered.fallback is not None else None
    return _Kernel(lowered, function, fallback)


class CudaProgram:
    def __init__(self, program: Program) -> None:
        self.program = program
        dev = driver.device()
        target = chip(dev.compute_capability)
        self.kernels: dict[str, _Kernel] = {}
        for key, task in program.tasks().items():
            self.kernels[key] = _load(lower_task(task), target)
        self.symbols = _symbols(program)
        self.stream = driver.Stream()

    def bind(self, inputs: Sequence[np.ndarray], state: Mapping[str, np.ndarray]) -> dict[str, int]:
        env: dict[str, int] = {}
        given = dict(zip(self.program.inputs, inputs))
        for b in self.program.buffers:
            if b.kind is BufferKind.state_in:
                given[b.name] = state[b.state.name]
        for name, array in given.items():
            t = self.program.buffer(name).type
            if np.ndim(array) != t.rank:
                raise ValueError(f"wrong rank for {name}: {np.shape(array)} / {t}")
            for d, n in zip(t.shape, np.shape(array)):
                if (s := as_symbol(d)) is not None:
                    bind(env, self.symbols.get(s.name, s), n)
        return env

    def allocate(self, env: dict[str, int]) -> dict[str, driver.DeviceBuffer]:
        buffers = {}
        for b in self.program.buffers:
            count = int(np.prod([evaluate(d, env) for d in b.type.shape], dtype=np.int64))
            # At least one element: masked lanes read element 0 of every argument.
            buffers[b.name] = driver.DeviceBuffer(max(count, 1) * element_bytes(b.type.dtype))
        return buffers

    def enqueue(self, buffers: dict[str, driver.DeviceBuffer], env: dict[str, int]) -> None:
        for launch in self.program.launches:
            kernel = self.kernels[launch.task.digest()].select(env)
            blocks = int(np.prod([evaluate(d, env) for d in launch.task.grid], dtype=np.int64))
            if blocks == 0:
                continue
            args = [(buffers[name].ptr, driver.POINTER) for name in launch.buffers]
            args += [(evaluate(d, env), driver.I32) for d in kernel.lowered.dims]
            kernel.function.launch((blocks, 1, 1), (kernel.lowered.threads, 1, 1), args, shared_bytes=kernel.lowered.shared_bytes, stream=self.stream)

    def run(self, inputs: Sequence, state: Mapping[str, np.ndarray] | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        state = dict(state or {})
        inputs = [np.asarray(a) for a in inputs]
        env = self.bind(inputs, state)
        buffers = self.allocate(env)
        for name, array in zip(self.program.inputs, inputs):
            buffers[name].upload(to_device_bytes(array, self.program.buffer(name).type.dtype), self.stream)
        for b in self.program.buffers:
            if b.kind is BufferKind.constant:
                data = b.constant.array if b.type.dtype is not DType.bool else b.constant.array.astype(np.uint8)
                buffers[b.name].upload(np.ascontiguousarray(data), self.stream)
            elif b.kind is BufferKind.state_in:
                buffers[b.name].upload(to_device_bytes(np.asarray(state[b.state.name]), b.type.dtype), self.stream)
        self.enqueue(buffers, env)
        self.stream.synchronize()

        def shaped(name: str) -> np.ndarray:
            b = self.program.buffer(name)
            return from_device(buffers[name], b.type.dtype, tuple(evaluate(d, env) for d in b.type.shape))

        new_state = {**state, **{b.state.name: shaped(b.name) for b in self.program.buffers if b.kind is BufferKind.state_out}}
        return [shaped(n) for n in self.program.outputs], new_state
