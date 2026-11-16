"""NVVM-dialect MLIR -> PTX, in process, with the pinned LLVM/MLIR build.

The pipeline is attach target (sm_86, linking libdevice) then gpu-module-to-binary with
format=isa, which runs LLVM's NVPTX backend and returns PTX text. The driver JIT turns the PTX
into SASS at module load, so no CUDA toolkit binary (nvcc, ptxas) is involved at run time; only
libdevice's bitcode is read from the toolkit.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import cache
from pathlib import Path

PTX_FEATURES = "+ptx80"  # PTX ISA 8.0 is enough for sm_86 and accepted by every driver since CUDA 12.0


def libdevice_path() -> str:
    explicit = os.environ.get("OLAF_LIBDEVICE")
    if explicit:
        return explicit
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root and (path := Path(root) / "nvvm" / "libdevice" / "libdevice.10.bc").exists():
            return str(path)
    raise FileNotFoundError("libdevice.10.bc not found; set OLAF_LIBDEVICE or CUDA_HOME")


def chip(compute_capability: tuple[int, int]) -> str:
    return f"sm_{compute_capability[0]}{compute_capability[1]}"


_ESCAPE = re.compile(r"\\([0-9A-Fa-f]{2}|\\|\")")


def _unescape(text: str) -> str:
    return _ESCAPE.sub(lambda m: m.group(1) if m.group(1) in ('\\', '"') else chr(int(m.group(1), 16)), text)


@cache
def compile_ptx(mlir_text: str, target: str) -> str:
    """PTX for a module holding one gpu.module. Cached by text: equal tasks compile once."""
    from mlir import ir
    from mlir.passmanager import PassManager

    with ir.Context():
        module = ir.Module.parse(mlir_text)
        pipeline = f"builtin.module(nvvm-attach-target{{chip={target} features={PTX_FEATURES} O=3 l={libdevice_path()}}},gpu-module-to-binary{{format=isa}})"
        PassManager.parse(pipeline).run(module.operation)
        text = str(module)
    match = re.search(r'assembly = "((?:[^"\\]|\\.)*)"', text)
    if match is None:
        raise RuntimeError("gpu-module-to-binary produced no PTX")
    return _unescape(match.group(1))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
