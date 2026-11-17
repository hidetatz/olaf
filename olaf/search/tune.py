"""Schedule search for matrix-instruction dots: measure candidates on the device, cache the best.

Each distinct dot (search.schedule.dot_key) is timed alone, as a one-op program on random
inputs of its shapes, under each candidate schedule; the fastest is kept. Results are cached in
a JSON file keyed by dot_key (which includes the capability digest), so a later run of the same
model, or another model with the same shapes, measures nothing. This is the "tuned" mode of
docs/design.md; without a cache entry, conversion uses its defaults ("fast").
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np

from olaf.backend.capability import Capability
from olaf.graph.ir import Graph
from olaf.search.schedule import DotSchedule, dot_key

SHAPES = [(128, 128), (128, 256), (256, 128), (64, 128), (128, 64)]


def candidates(capability: Capability) -> list[DotSchedule]:
    out = []
    limit = min(capability.on_chip.shared_bytes, 96 * 1024)
    for bm, bn in SHAPES:
        # Two-stage pipelines and 4-warp 128x128-or-larger tiles never won in measurements on the
        # RTX 3060; leaving them out halves the search.
        warps = 8 if bm * bn >= 128 * 128 else 4
        threads = 32 * warps
        for bk in (32, 64):
            for stages in (3, 4):
                s = DotSchedule(bm, bn, bk, stages, warps)
                if s.shared_bytes() > limit or bm * bn // threads > 128:
                    continue  # shared memory, or accumulator registers per thread
                if (bm * bk // 8) % threads or (bk * bn // 8) % threads:
                    continue  # 16-byte copies must split evenly over the threads
                out.append(s)
    return out


def cache_path() -> Path:
    root = Path(os.environ.get("OLAF_CACHE_DIR", Path.home() / ".cache" / "olaf"))
    return root / "schedules.json"


def load() -> dict[str, DotSchedule]:
    path = cache_path()
    if not path.exists():
        return {}
    try:
        return {k: DotSchedule(**v) for k, v in json.loads(path.read_text()).items()}
    except Exception:
        return {}


def save(schedules: dict[str, DotSchedule]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: vars(v) for k, v in sorted(schedules.items())}, indent=1))
    tmp.replace(path)


def dots(graph: Graph, capability: Capability) -> dict[str, tuple[str, list, str]]:
    """dot_key -> (spec, operand shapes, b16 dtype) for the matrix-instruction dots of a lowered graph."""
    from olaf.tile.from_graph import _Converter

    converter = _Converter(graph, capability)
    converter.producers = {v: op for op in graph.ops for v in op.outputs}
    found = {}
    for op in graph.ops:
        if op.kind != "contract" or len(op.inputs) != 2:
            continue
        base = {converter._matrix_dtype(v) for v in op.inputs}
        if len(base) != 1 or not any(i.m == 16 and i.dtype_a == next(iter(base)).value for i in capability.matrix_instructions):
            continue
        shapes = [v.type.shape for v in op.inputs]
        if not all(isinstance(d, int) for s in shapes for d in s):
            continue
        dtype = next(iter(base)).value
        found[dot_key(op.attr("spec"), shapes, dtype, capability.digest())] = (op.attr("spec"), shapes, dtype)
    return found


def measure(spec: str, shapes, dtype: str, capability: Capability, schedule: DotSchedule | None, repeats: int = 5) -> float | None:
    """Median milliseconds of the dot alone under `schedule` (None: defaults), or None if invalid."""
    import olaf
    from olaf.backend.cuda import driver
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram, to_device_bytes
    from olaf.graph.types import DType, TensorType
    from olaf.pipeline import lower_graph, to_program

    dt = DType(dtype)
    graph = lower_graph(olaf.trace(lambda a, b: olaf.einsum(spec, a, b), TensorType(tuple(shapes[0]), dt), TensorType(tuple(shapes[1]), dt)).graph)
    key = dot_key(spec, shapes, dtype, capability.digest())
    program = to_program(graph, capability, schedules={key: schedule} if schedule else None)
    if schedule is not None and "nvvm.mma.sync" not in lower_task(program.launches[0].task).text:
        return None  # the schedule does not fit the matrix-instruction lowering
    try:
        compiled = CudaProgram(program)
    except Exception:
        return None
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=s).astype(np.float32) for s in shapes]
    env = compiled.bind(arrays, {})
    buffers = compiled.allocate(env)
    for name, array in zip(compiled.program.inputs, arrays):
        buffers[name].upload(to_device_bytes(array, dt), compiled.stream)
    for _ in range(2):
        compiled.enqueue(buffers, env)
    times = []
    for _ in range(repeats):
        with driver.Timer(compiled.stream) as t:
            for _ in range(3):
                compiled.enqueue(buffers, env)
        times.append(t.ms / 3)
    return statistics.median(times)


def tune_in_program(graph: Graph, capability: Capability, inputs, state, *, top: int = 5, log=print) -> dict[str, DotSchedule]:
    """Refine the schedules in the context they run in: the whole program, its real operands.

    A dot alone, on fresh contiguous inputs, is not the dot in a training step: there its
    operands are views of earlier results and its epilogue carries fused work, and the
    standalone ranking does not carry over (GPT-2 small's dots took the same total time with
    standalone-tuned and default schedules). For each dot, the `top` standalone candidates are
    compiled into the whole program in turn and the dot's own launches timed; the fastest stays.
    Results replace the standalone ones in the cache, marked so they are not searched again.
    """
    from olaf.backend.cuda.runtime import CudaProgram, Session, profile
    from olaf.pipeline import to_program

    schedules = tune(graph, capability, log=log)  # standalone results and the cache
    done = _load_marks()

    def timed(assignment: dict[str, DotSchedule], key: str) -> float:
        compiled = CudaProgram(to_program(graph, capability, schedules=assignment))
        session = Session(compiled, {}, graphs=False)
        session.load_state(state)
        runs = []
        for _ in range(3):
            total = 0.0
            for ms, n, label, task in profile(session, inputs, with_tasks=True):
                if task.origin[2:3] == (key,):
                    total += ms
            runs.append(total)
        return statistics.median(runs)

    for key, (spec, shapes, dtype) in dots(graph, capability).items():
        if key in done:
            continue
        ranked = sorted(((ms, c) for c in candidates(capability) if (ms := measure(spec, shapes, dtype, capability, c)) is not None), key=lambda mc: mc[0])
        trial = [schedules[key], *[c for _, c in ranked if c != schedules[key]][: top - 1]]
        best, best_ms = None, None
        for candidate in trial:
            ms = timed({**schedules, key: candidate}, key)
            if best_ms is None or ms < best_ms:
                best, best_ms = candidate, ms
        if best_ms == 0.0:
            done.add(key)  # no launch carries this dot (fused away, e.g. into attention)
            _save_marks(done)
            continue
        schedules[key] = best
        save(schedules)
        done.add(key)
        _save_marks(done)
        log(f"in program: {spec} {shapes}: {best} ({best_ms:.3f} ms)")
    return schedules


def _marks_path() -> Path:
    return cache_path().with_name("schedules_in_program.json")


def _load_marks() -> set[str]:
    try:
        return set(json.loads(_marks_path().read_text()))
    except Exception:
        return set()


def _save_marks(done: set[str]) -> None:
    _marks_path().write_text(json.dumps(sorted(done)))


def tune(graph: Graph, capability: Capability, *, log=print) -> dict[str, DotSchedule]:
    """Schedules for every dot of the lowered graph: cached ones, and newly searched ones."""
    schedules = load()
    for key, (spec, shapes, dtype) in dots(graph, capability).items():
        if key in schedules:
            continue
        best, best_ms = None, measure(spec, shapes, dtype, capability, None)
        for candidate in candidates(capability):
            ms = measure(spec, shapes, dtype, capability, candidate)
            if ms is not None and (best_ms is None or ms < best_ms):
                best, best_ms = candidate, ms
        if best is None:
            from olaf.tile.from_graph import MATRIX_DOT_BLOCK, MATRIX_DOT_STAGES, MATRIX_DOT_WARPS

            best = DotSchedule(*MATRIX_DOT_BLOCK, MATRIX_DOT_STAGES, MATRIX_DOT_WARPS)
        schedules[key] = best
        save(schedules)
        log(f"tuned {spec} {shapes}: {best} ({best_ms:.3f} ms)")
    return schedules
