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
from dataclasses import replace
from pathlib import Path

from olaf.backend.capability import Capability
from olaf.graph.ir import Graph
from olaf.search.schedule import DotSchedule, dot_key

# Launches of a dot timed together (measure).
REPEATS = 16

SHAPES = [(64, 64), (128, 128), (128, 256), (256, 128), (64, 128), (128, 64)]


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
    return root / "schedules2.json"  # 1: timed launches one by one (see measure)


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


def dots(graph: Graph, capability: Capability, schedules=None) -> dict[str, tuple[str, list, str]]:
    """dot_key -> (spec, operand shapes, b16 dtype) for the matrix-instruction dots of a lowered graph.

    Collected during conversion rather than from the graph's contract ops: a convolution becomes
    contracts of its own (over padded grids or im2col columns), with shapes no graph op has. A
    dot that the default schedule splits along its reduction is here as well as its parts: whether
    to split follows from the tile, so it is the whole dot's schedule that decides it.
    """
    from olaf.tile.from_graph import matrix_dots

    return {key: dot for key, dot in matrix_dots(graph, capability, schedules).items() if all(isinstance(d, int) for shape in dot[1] for d in shape)}


def measure(spec: str, shapes, dtype: str, capability: Capability, schedule: DotSchedule | None, repeats: int = 5) -> float | None:
    """Median milliseconds of the dot alone under `schedule` (None: defaults), or None if invalid."""
    import olaf
    from olaf.backend.cuda import driver
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram, Session
    from olaf.graph.types import DType, TensorType
    from olaf.pipeline import lower_graph, to_program
    from olaf.tile.from_graph import matrix_dots

    dt = DType(dtype)
    graph = lower_graph(olaf.trace(lambda a, b: olaf.einsum(spec, a, b), TensorType(tuple(shapes[0]), dt), TensorType(tuple(shapes[1]), dt)).graph)
    key = dot_key(spec, shapes, dtype, capability.digest())
    # Under a split, its parts too.
    keys = {key, *matrix_dots(graph, capability, {key: schedule})} if schedule else ()
    program = to_program(graph, capability, schedules=dict.fromkeys(keys, schedule))
    if schedule is not None and "nvvm.mma.sync" not in lower_task(program.launches[0].task).text:
        return None  # the schedule does not fit the matrix-instruction lowering
    # The launches repeated in one CUDA graph: launched one by one, a kernel takes the host about
    # 100 us to launch, and every candidate of a shorter dot measured that.
    program = replace(program, launches=program.launches * REPEATS)
    try:
        compiled = CudaProgram(program)
    except Exception:
        return None
    # The inputs are whatever the buffers hold: the time of a product does not depend on it, and
    # making and uploading random ones was a third of the search.
    session = Session(compiled, {})
    graph = driver.Graph(compiled.stream, lambda: session._enqueue(0))
    graph.replay()
    times = []
    for _ in range(repeats):
        with driver.Timer(compiled.stream) as t:
            graph.replay()
        times.append(t.ms / REPEATS)
    return statistics.median(times)


_RANKINGS: dict[tuple, list[tuple[float, DotSchedule]]] = {}


def ranking(spec: str, shapes, dtype: str, capability: Capability) -> list[tuple[float, DotSchedule]]:
    """(ms, candidate) of a dot alone, fastest first; measured once per process."""
    key = (spec, str(shapes), dtype, capability.digest())
    if key not in _RANKINGS:
        timed = [(measure(spec, shapes, dtype, capability, c), c) for c in candidates(capability)]
        _RANKINGS[key] = sorted(((ms, c) for ms, c in timed if ms is not None), key=lambda mc: mc[0])
    return _RANKINGS[key]


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
    pending = {key: v for key, v in dots(graph, capability, schedules).items() if key not in done}
    trials = {}
    for key, (spec, shapes, dtype) in pending.items():
        ranked = ranking(spec, shapes, dtype, capability)
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
    absent = set()  # dots some candidate leaves without a launch of their own (split, or not)
    for r in range(top):
        assignment = {**schedules, **{key: t[min(r, len(t) - 1)] for key, t in trials.items()}}
        for key, ms in timed(assignment).items():
            if ms == 0.0:
                absent.add(key)
            elif key not in best or ms < best[key][1]:
                best[key] = (assignment[key], ms)
    for key, (chosen, ms) in best.items():
        # absent: fused away (into attention), or split under some candidates; their own times
        # are then not the dot's, and the standalone choice, which times the sum too, stays.
        if key not in absent:
            schedules[key] = chosen
            spec, shapes, _ = pending[key]
            log(f"in program: {spec} {shapes}: {chosen} ({ms:.3f} ms)")
    done.update(pending)
    save(schedules)
    _save_marks(done)
    return schedules


def _marks_path() -> Path:
    return cache_path().with_name("schedules2_in_program.json")


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
    # Twice: the parts of a split under the chosen schedules may differ from the defaults' parts.
    for known in ({}, schedules):
        for key, (spec, shapes, dtype) in dots(graph, capability, known).items():
            if key in schedules:
                continue
            best, best_ms = None, measure(spec, shapes, dtype, capability, None)
            for ms, candidate in ranking(spec, shapes, dtype, capability)[:1]:
                if best_ms is None or ms < best_ms:
                    best, best_ms = candidate, ms
            if best is None:
                t = capability.tuning
                best = DotSchedule(*t.matrix_dot_block, t.matrix_dot_stages, t.matrix_dot_warps)
            schedules[key] = best
            save(schedules)
            log(f"tuned {spec} {shapes}: {best} ({best_ms} ms)")
    return schedules
