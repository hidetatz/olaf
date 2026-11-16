"""bf16 GEMM: olaf's generated kernel against cuBLAS (through PyTorch), the P1 matmul gate.

Each side is warmed up, then timed 5 times (10 launches each) and the median is compared, as
docs/design.md prescribes. The GPU is shared with the display, so run both back to back.

    python bench/gemm.py [n] [spec]      e.g. python bench/gemm.py 4096 ik,kj->ij
"""

import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import olaf  # noqa: E402
from olaf.backend.capability import AMPERE_RTX3060  # noqa: E402
from olaf.backend.cuda import driver  # noqa: E402
from olaf.backend.cuda.runtime import CudaProgram, to_device_bytes  # noqa: E402
from olaf.graph.types import TensorType  # noqa: E402
from olaf.passes.canonicalize import canonicalize  # noqa: E402
from olaf.passes.decompose import decompose  # noqa: E402
from olaf.tile.from_graph import to_tile  # noqa: E402


def median_ms(run, synchronize, timer) -> float:
    for _ in range(3):
        run()
    synchronize()
    times = []
    for _ in range(5):
        times.append(timer(lambda: [run() for _ in range(10)]) / 10)
    return statistics.median(times)


def olaf_ms(n: int, spec: str) -> float:
    t = TensorType((n, n), olaf.bf16)
    graph = canonicalize(decompose(olaf.trace(lambda x, y: olaf.einsum(spec, x, y), t, t).graph))
    program = CudaProgram(to_tile(graph, AMPERE_RTX3060))
    a = np.random.default_rng(0).normal(size=(n, n)).astype(np.float32)
    env = program.bind([a, a], {})
    buffers = program.allocate(env)
    for name in program.program.inputs:
        buffers[name].upload(to_device_bytes(a, olaf.bf16))

    def timer(fn):
        with driver.Timer(program.stream) as t:
            fn()
        return t.ms

    return median_ms(lambda: program.enqueue(buffers, env), program.stream.synchronize, timer)


def cublas_ms(n: int, spec: str) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    fn = {"ik,kj->ij": lambda: a @ b, "ik,jk->ij": lambda: a @ b.t(), "ki,kj->ij": lambda: a.t() @ b, "ki,jk->ij": lambda: a.t() @ b.t()}[spec]

    def timer(run):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop)

    return median_ms(fn, torch.cuda.synchronize, timer)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    spec = sys.argv[2] if len(sys.argv) > 2 else "ik,kj->ij"
    reference = cublas_ms(n, spec)
    ours = olaf_ms(n, spec)
    flops = 2 * n**3
    print(f"{spec} n={n}: cuBLAS {reference:.3f} ms ({flops / reference / 1e9:.1f} TFLOPS), olaf {ours:.3f} ms ({flops / ours / 1e9:.1f} TFLOPS), olaf/cuBLAS = {reference / ours:.1%}")
