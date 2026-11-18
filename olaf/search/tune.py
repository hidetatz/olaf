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
    """dot_key -> (spec, operand shapes, b16 dtype) for the matrix-instruction dots of a lowered graph.

    Read from the converted program's launches (their origins carry the key) rather than the
    graph's contract ops: a convolution becomes contracts of its own during conversion (over
    padded grids or im2col columns), with shapes no graph op has.
    """
    from olaf.tile.from_graph import to_tile

    found = {}
    for launch in to_tile(graph, capability).launches:
        origin = launch.task.origin
        if origin[:1] != ("contract",) or len(origin) < 3:
            continue
        spec, a, b, dtype, _ = origin[2].split("|")
        shapes = [[int(d) for d in a.split("x")], [int(d) for d in b.split("x")]]
        found[origin[2]] = (spec, shapes, dtype)
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
    pending = {key: v for key, v in dots(graph, capability).items() if key not in done}
    trials = {}
    for key, (spec, shapes, dtype) in pending.items():
        ranked = sorted(((ms, c) for c in candidates(capability) if (ms := measure(spec, shapes, dtype, capability, c)) is not None), key=lambda mc: mc[0])
        trials[key] = [schedules[key], *[c for _, c in ranked if c != schedules[key]][: top - 1]]

    def timed(assignment: dict[str, DotSchedule]) -> dict[str, float]:
        compiled = CudaProgram(to_program(graph, capability, schedules=assignment))
        session = Session(compiled, {}, graphs=False)
        session.load_state(state)
        runs = []
        for _ in range(3):
            total = dict.fromkeys(pending, 0.0)
            for ms, n, label, task in profile(session, inputs, with_tasks=True):
                if len(task.origin) > 2 and task.origin[2] in total:
                    total[task.origin[2]] += ms
            runs.append(total)
        return {key: statistics.median(r[key] for r in runs) for key in pending}

    # Every dot's launches are timed on their own, so one program per round tries the round's
    # candidate of every dot at once: `top` whole-program compiles instead of `top` per dot (a
    # ResNet-18 step has 37 distinct dots, most of them inside convolutions).
    best = {}
    for r in range(top):
        assignment = {**schedules, **{key: t[min(r, len(t) - 1)] for key, t in trials.items()}}
        for key, ms in timed(assignment).items():
            if key not in best or ms < best[key][1]:
                best[key] = (assignment[key], ms)
    for key, (chosen, ms) in best.items():
        if ms > 0.0:  # 0: no launch carries this dot (fused away, e.g. into attention)
            schedules[key] = chosen
            spec, shapes, _ = pending[key]
            log(f"in program: {spec} {shapes}: {chosen} ({ms:.3f} ms)")
        done.add(key)
    save(schedules)
    _save_marks(done)
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
            t = capability.tuning
            best = DotSchedule(*t.matrix_dot_block, t.matrix_dot_stages, t.matrix_dot_warps)
        schedules[key] = best
        save(schedules)
        log(f"tuned {spec} {shapes}: {best} ({best_ms:.3f} ms)")
    return schedules
