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

    def select(self, env: dict[str, int], offsets: list[int]) -> _Kernel:
        """The specialized kernel when its alignment assumptions hold for this launch, else the general one."""
        if self.fallback is None:
            return self
        if any(evaluate(d, env) % 8 for d in self.lowered.aligned) or any(offsets[i] % 8 for i in self.lowered.aligned_args):
            return self.fallback
        return self


def _load(lowered: Lowered, ptx: dict[str, str]) -> _Kernel:
    function = driver.Module(ptx[lowered.text]).function("task")
    if lowered.shared_bytes > 48 * 1024:
        function.set_shared_bytes(lowered.shared_bytes)
    fallback = _load(lowered.fallback, ptx) if lowered.fallback is not None else None
    return _Kernel(lowered, function, fallback)


def _compiled(task, target: str) -> tuple[Lowered, dict[str, str]]:
    """Lowered task and the PTX of it (and of its fallback), from the disk cache when possible.

    The key is the task's template text, the target and a fingerprint of the compiler's own
    source, so any change to olaf invalidates every entry. Lowering in Python is most of the
    cost (GPT-2 small's step takes about 30 s cold), so the cache is consulted before lowering.
    """
    import hashlib
    import pickle

    key = hashlib.sha256(f"{task.text()}\n{target}\n{_fingerprint()}".encode()).hexdigest()
    path = _cache_dir() / f"{key}.pickle"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass  # a damaged or incompatible entry is recompiled
    lowered = lower_task(task)
    ptx = {}
    current = lowered
    while current is not None:
        ptx[current.text] = compile_ptx(current.text, target)
        current = current.fallback
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps((lowered, ptx)))
        tmp.replace(path)
    except OSError:
        pass  # the cache is an optimization only
    return lowered, ptx


def _cache_dir():
    import os
    from pathlib import Path

    return Path(os.environ.get("OLAF_CACHE_DIR", Path.home() / ".cache" / "olaf")) / "kernels"


_FINGERPRINT: list[str] = []


def _fingerprint() -> str:
    if not _FINGERPRINT:
        import hashlib
        from pathlib import Path

        import olaf

        root = Path(olaf.__file__).resolve().parent
        digest = hashlib.sha256()
        for source in sorted(root.rglob("*.py")):
            digest.update(source.relative_to(root).as_posix().encode())
            digest.update(source.read_bytes())
        _FINGERPRINT.append(digest.hexdigest())
    return _FINGERPRINT[0]


INDEX_LIMIT = 2**31  # kernels address in i32 (tile/builder.py); see check_index_range


def check_index_range(program: Program, env: dict[str, int]) -> None:
    """Fail when a kernel would compute an element offset or block index past i32.

    Kernels add view offsets and coordinate * stride terms in i32 on top of the argument's base
    pointer (the launch offset is added on the host, in 64 bits). Past 2^31 the arithmetic wraps
    and the kernel reads and writes the wrong elements without any error, so the sizes are
    checked once they are known. Masked lanes may compute wrapped offsets; they are replaced by
    0 before the access, so only in-bounds coordinates count here. Lowering to i64 offsets for
    such launches is not implemented: it costs integer throughput on every access.
    """
    for launch in program.launches:
        blocks = 1
        for d in launch.task.grid:
            blocks *= evaluate(d, env)
        if blocks >= INDEX_LIMIT:
            raise OverflowError(f"{launch.task.name}: {blocks} blocks exceed the i32 block index")
        for name, arg in zip(launch.buffers, launch.task.args):
            view = arg.view
            low = high = evaluate(view.offset, env)
            for axis, (size, stride) in enumerate(zip(view.shape, view.strides)):
                extent = evaluate(size, env)
                if view.pad is not None:
                    extent -= evaluate(view.pad[axis][0], env) + evaluate(view.pad[axis][1], env)
                step = evaluate(stride, env)
                if extent > 0:
                    span = (extent - 1) * step
                    low, high = low + min(span, 0), high + max(span, 0)
            if high >= INDEX_LIMIT or low <= -INDEX_LIMIT:
                raise OverflowError(
                    f"{launch.task.name}: argument {name} ({program.buffer(name).type}) reaches element offset {high}, past the i32 offsets kernels compute"
                )


def compile_for(program: Program, capability) -> dict[str, dict[str, str]]:
    """PTX of every task of `program` for capability.target, without a device: for targets
    there is no hardware to run on (tests/test_targets.py). {task digest: {kernel: PTX}}."""
    if not capability.target:
        raise ValueError(f"{capability.name} names no code generation target")
    return {key: _compiled(task, capability.target)[1] for key, task in program.tasks().items()}


class CudaProgram:
    def __init__(self, program: Program) -> None:
        self.program = program
        dev = driver.device()
        target = chip(dev.compute_capability)
        self.kernels: dict[str, _Kernel] = {}
        for key, task in program.tasks().items():
            self.kernels[key] = _load(*_compiled(task, target))
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
            count = evaluate(b.elements(), env)
            # At least one element: masked lanes read element 0 of every argument.
            buffers[b.name] = driver.DeviceBuffer(max(count, 1) * element_bytes(b.type.dtype))
        return buffers

    def enqueue(self, buffers: dict[str, driver.DeviceBuffer], env: dict[str, int]) -> None:
        self.enqueue_at({name: b.ptr for name, b in buffers.items()}, env)

    def enqueue_at(self, addresses: dict[str, int], env: dict[str, int]) -> None:
        """Enqueue every launch with buffers at the given device addresses."""
        for launch in self.program.launches:
            offsets = [evaluate(launch.offset(i), env) for i in range(len(launch.buffers))]
            kernel = self.kernels[launch.task.digest()].select(env, offsets)
            blocks = int(np.prod([evaluate(d, env) for d in launch.task.grid], dtype=np.int64))
            if blocks == 0:
                continue
            args = []
            for name, offset, arg in zip(launch.buffers, offsets, launch.task.args):
                args.append((addresses[name] + offset * element_bytes(arg.dtype), driver.POINTER))
            args += [(evaluate(d, env), driver.I32) for d in kernel.lowered.dims]
            kernel.function.launch((blocks, 1, 1), (kernel.lowered.threads, 1, 1), args, shared_bytes=kernel.lowered.shared_bytes, stream=self.stream)

    def run(self, inputs: Sequence, state: Mapping[str, np.ndarray] | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        state = dict(state or {})
        inputs = [np.asarray(a) for a in inputs]
        env = self.bind(inputs, state)
        check_index_range(self.program, env)
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


def plan_arena(program: Program, env: dict[str, int], alignment: int = 256) -> tuple[dict[str, int], int]:
    """Byte offsets for temp buffers in one arena, reusing memory between disjoint lifetimes.

    A temp lives from the first launch that touches it to the last. Buffers are placed largest
    first at the lowest offset that overlaps no placed buffer with an overlapping lifetime. This
    is plain reuse, enough to fit a training step; choosing what to save, recompute or offload is
    the memory planner of P2.
    """
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for i, launch in enumerate(program.launches):
        for name in launch.buffers:
            first.setdefault(name, i)
            last[name] = i
    sizes = {}
    for b in program.buffers:
        if b.kind is BufferKind.temp and b.name in first:
            count = evaluate(b.elements(), env)
            sizes[b.name] = -(-max(count, 1) * element_bytes(b.type.dtype) // alignment) * alignment
    placed: list[tuple[int, int, int, int]] = []  # (offset, end, first, last)
    offsets: dict[str, int] = {}
    for name in sorted(sizes, key=lambda n: (-sizes[n], first[n])):
        busy = sorted((o, e) for o, e, f, l in placed if not (l < first[name] or last[name] < f))
        offset = 0
        for o, e in busy:
            if offset + sizes[name] <= o:
                break
            offset = max(offset, e)
        offsets[name] = offset
        placed.append((offset, offset + sizes[name], first[name], last[name]))
    return offsets, max((e for _, e, _, _ in placed), default=0)


class Session:
    """A program bound to fixed sizes, with device-resident state, run step after step.

    Resident state that a step both reads and writes gets two buffers that swap roles every
    step (no copy). Each of the two parities is captured once into a CUDA graph and replayed.
    """

    def __init__(self, compiled: CudaProgram, dims: dict[str, int], *, graphs: bool = True) -> None:
        self.compiled = compiled
        program = compiled.program
        self.env = dict(dims)
        check_index_range(program, self.env)
        offsets, arena_bytes = plan_arena(program, self.env)
        self.arena_bytes = arena_bytes
        self.arena = driver.DeviceBuffer(max(arena_bytes, 1))
        self.fixed: dict[str, driver.DeviceBuffer] = {}
        by_state: dict[str, dict[BufferKind, str]] = {}
        for b in program.buffers:
            if b.kind in (BufferKind.state_in, BufferKind.state_out):
                by_state.setdefault(b.state.name, {})[b.kind] = b.name
        self.states = by_state
        self.views: list[dict[str, int]] = [{}, {}]  # buffer name -> device address, per parity
        for b in program.buffers:
            if b.name in offsets:
                for parity in (0, 1):
                    self.views[parity][b.name] = self.arena.ptr + offsets[b.name]
            elif b.kind in (BufferKind.state_in, BufferKind.state_out):
                continue
            else:
                buffer = self._alloc(b)
                self.fixed[b.name] = buffer
                for parity in (0, 1):
                    self.views[parity][b.name] = buffer.ptr
        self.pairs: dict[str, tuple[driver.DeviceBuffer, driver.DeviceBuffer]] = {}
        for name, kinds in by_state.items():
            b = program.buffer(next(iter(kinds.values())))
            pair = (self._alloc(b), self._alloc(b))
            self.pairs[name] = pair
            for parity in (0, 1):
                if BufferKind.state_in in kinds:
                    self.views[parity][kinds[BufferKind.state_in]] = pair[parity].ptr
                if BufferKind.state_out in kinds:
                    # Read and written: write the other buffer. Only written: always the same one.
                    target = pair[1 - parity] if BufferKind.state_in in kinds else pair[0]
                    self.views[parity][kinds[BufferKind.state_out]] = target.ptr
        for b in program.buffers:
            if b.kind is BufferKind.constant:
                data = b.constant.array if b.type.dtype is not DType.bool else b.constant.array.astype(np.uint8)
                self.fixed[b.name].upload(np.ascontiguousarray(data), compiled.stream)
        self.parity = 0
        self.use_graphs = graphs
        self.graphs: list[driver.Graph | None] = [None, None]

    def _alloc(self, b) -> driver.DeviceBuffer:
        count = evaluate(b.elements(), self.env)
        return driver.DeviceBuffer(max(count, 1) * element_bytes(b.type.dtype))

    def _shape(self, name: str) -> tuple[int, ...]:
        return tuple(evaluate(d, self.env) for d in self.compiled.program.buffer(name).type.shape)

    def load_state(self, state: Mapping[str, np.ndarray]) -> None:
        program = self.compiled.program
        for name, kinds in self.states.items():
            if name in state:
                dtype = program.buffer(next(iter(kinds.values()))).type.dtype
                self.pairs[name][self.parity].upload(to_device_bytes(np.asarray(state[name]), dtype), self.compiled.stream)

    def read_state(self) -> dict[str, np.ndarray]:
        self.compiled.stream.synchronize()
        program = self.compiled.program
        out = {}
        for name, kinds in self.states.items():
            b = program.buffer(next(iter(kinds.values())))
            current = self.pairs[name][self.parity] if BufferKind.state_in in kinds else self.pairs[name][0]
            out[name] = from_device(current, b.type.dtype, self._shape(b.name))
        return out

    def _enqueue(self, parity: int) -> None:
        self.compiled.enqueue_at(self.views[parity], self.env)

    def step(self, inputs: Sequence) -> None:
        """Run one step asynchronously; outputs stay on the device until outputs() is called."""
        program = self.compiled.program
        for name, array in zip(program.inputs, inputs):
            self.fixed[name].upload(to_device_bytes(np.asarray(array), program.buffer(name).type.dtype), self.compiled.stream)
        if not self.use_graphs:
            self._enqueue(self.parity)
        else:
            if self.graphs[self.parity] is None:
                parity = self.parity
                self.graphs[parity] = driver.Graph(self.compiled.stream, lambda: self._enqueue(parity))
            self.graphs[self.parity].replay()
        self.parity = 1 - self.parity

    def outputs(self) -> list[np.ndarray]:
        self.compiled.stream.synchronize()
        program = self.compiled.program
        return [from_device(self.fixed[n], program.buffer(n).type.dtype, self._shape(n)) for n in program.outputs]


def profile(session: Session, inputs: Sequence, *, with_tasks: bool = False) -> list:
    """Time every launch of one step on its own: (total ms, launch count, label) per task template.

    Launches are timed one at a time with device events, so the sum exceeds a replayed step's
    time by the launch gaps a graph removes; the ranking is what this is for.
    with_tasks: instead, one (ms, 1, label, task) per launch, in launch order.
    """
    compiled, program = session.compiled, session.compiled.program
    for name, array in zip(program.inputs, inputs):
        session.fixed[name].upload(to_device_bytes(np.asarray(array), program.buffer(name).type.dtype), compiled.stream)
    addresses = session.views[session.parity]
    totals: dict[str, list] = {}
    per_launch: list = []
    for launch in program.launches:
        offsets = [evaluate(launch.offset(i), session.env) for i in range(len(launch.buffers))]
        kernel = compiled.kernels[launch.task.digest()].select(session.env, offsets)
        blocks = int(np.prod([evaluate(d, session.env) for d in launch.task.grid], dtype=np.int64))
        if blocks == 0:
            continue
        args = [(addresses[n] + o * element_bytes(a.dtype), driver.POINTER) for n, o, a in zip(launch.buffers, offsets, launch.task.args)]
        args += [(evaluate(d, session.env), driver.I32) for d in kernel.lowered.dims]
        with driver.Timer(compiled.stream) as t:
            kernel.function.launch((blocks, 1, 1), (kernel.lowered.threads, 1, 1), args, shared_bytes=kernel.lowered.shared_bytes, stream=compiled.stream)
        shapes = "x".join(str(evaluate(d, session.env)) for d in launch.task.grid)
        general = " (general)" if kernel is not compiled.kernels[launch.task.digest()] or "mma.sync" not in kernel.lowered.text else ""
        label = f"{launch.task.name} grid={shapes} warps={launch.task.warps}{general if launch.task.name == 'contract' else ''}"
        if with_tasks:
            per_launch.append((t.ms, 1, label, launch.task))
            continue
        entry = totals.setdefault(label, [0.0, 0])
        entry[0] += t.ms
        entry[1] += 1
    if with_tasks:
        return per_launch
    return sorted(((ms, n, label) for label, (ms, n) in totals.items()), reverse=True)
