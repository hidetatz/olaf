"""Thin layer over the CUDA Driver API (cuda-python): device memory, modules, launches, graphs.

Only the host side of compiled programs talks to this module, and only outside the hot loop:
a step is captured once into a CUDA graph and then replayed with one call (principle 1).

The Driver API rather than the runtime API: modules are loaded from PTX produced by our own
lowering, with no nvcc involved, and the driver's JIT turns PTX into SASS for the local GPU.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cache

import numpy as np
from cuda.bindings import driver as cu


class CudaError(RuntimeError):
    pass


def check(result):
    """Unwrap a cuda-python result tuple, raising on any error code."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        _, text = cu.cuGetErrorString(err)
        raise CudaError(f"{name.decode() if name else err}: {text.decode() if text else ''}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


@dataclass(frozen=True)
class Device:
    handle: object
    context: object
    compute_capability: tuple[int, int]
    name: str

    def attribute(self, attr: cu.CUdevice_attribute) -> int:
        return check(cu.cuDeviceGetAttribute(attr, self.handle))

    def make_current(self) -> None:
        check(cu.cuCtxSetCurrent(self.context))


@cache
def device(ordinal: int = 0) -> Device:
    # The primary context is shared with any other user of the device in this process (PyTorch in
    # tests), so both see the same allocations and nothing tears the context down under the other.
    check(cu.cuInit(0))
    handle = check(cu.cuDeviceGet(ordinal))
    context = check(cu.cuDevicePrimaryCtxRetain(handle))
    check(cu.cuCtxSetCurrent(context))
    major = check(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle))
    minor = check(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle))
    name = check(cu.cuDeviceGetName(128, handle)).split(b"\0")[0].decode()
    return Device(handle, context, (major, minor), name)


def available() -> bool:
    try:
        device()
        return True
    except Exception:
        return False


class DeviceBuffer:
    """An owned device allocation."""

    def __init__(self, nbytes: int) -> None:
        device().make_current()
        self.nbytes = nbytes
        # cuMemAlloc rejects 0 bytes; a 1-byte allocation keeps empty tensors addressable.
        self.ptr = int(check(cu.cuMemAlloc(max(nbytes, 1))))

    def __del__(self) -> None:
        ptr = getattr(self, "ptr", None)
        if ptr:
            try:
                cu.cuMemFree(ptr)
            except Exception:
                pass
            self.ptr = 0

    def upload(self, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        if array.nbytes != self.nbytes:
            raise ValueError(f"size mismatch: {array.nbytes} bytes for a {self.nbytes}-byte buffer")
        if self.nbytes:
            check(cu.cuMemcpyHtoD(self.ptr, array.ctypes.data, self.nbytes))

    def download(self, dtype, shape) -> np.ndarray:
        out = np.empty(shape, dtype)
        if out.nbytes != self.nbytes:
            raise ValueError(f"size mismatch: {out.nbytes} bytes from a {self.nbytes}-byte buffer")
        if self.nbytes:
            check(cu.cuMemcpyDtoH(out.ctypes.data, self.ptr, self.nbytes))
        return out

    def fill_bytes(self, value: int) -> None:
        if self.nbytes:
            check(cu.cuMemsetD8(self.ptr, value, self.nbytes))


class Stream:
    def __init__(self) -> None:
        device().make_current()
        self.handle = check(cu.cuStreamCreate(cu.CUstream_flags.CU_STREAM_NON_BLOCKING))

    def synchronize(self) -> None:
        check(cu.cuStreamSynchronize(self.handle))

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle is not None:
            try:
                cu.cuStreamDestroy(handle)
            except Exception:
                pass


class Module:
    def __init__(self, ptx: str) -> None:
        device().make_current()
        data = ptx.encode() + b"\0"
        log_size = 1 << 14
        # cuda-python takes the log buffer as bytes and the driver writes the log into it.
        error_log = b"\0" * log_size
        options = [cu.CUjit_option.CU_JIT_ERROR_LOG_BUFFER, cu.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES]
        values = [error_log, log_size]
        try:
            self.handle = check(cu.cuModuleLoadDataEx(data, len(options), options, values))
        except CudaError as e:
            raise CudaError(f"{e}\n{error_log.split(b'\0')[0].decode(errors='replace')}") from None
        self._functions: dict[str, Kernel] = {}

    def function(self, name: str) -> Kernel:
        if name not in self._functions:
            self._functions[name] = Kernel(check(cu.cuModuleGetFunction(self.handle, name.encode())), self)
        return self._functions[name]


# Kernel parameter kinds: pointers are 64-bit, symbolic dimensions are passed as i32 (see tile.builder.INDEX).
POINTER = ctypes.c_void_p
I32 = ctypes.c_int32


class Kernel:
    def __init__(self, handle, module: Module) -> None:
        self.handle = handle
        self.module = module  # keeps the module loaded while the kernel is alive

    def set_shared_bytes(self, nbytes: int) -> None:
        # Above 48 KiB the dynamic shared memory size must be opted into per function.
        attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        check(cu.cuFuncSetAttribute(self.handle, attr, nbytes))

    def launch(self, grid: tuple[int, int, int], block: tuple[int, int, int], args: list[tuple[int, type]], *, shared_bytes: int = 0, stream: Stream | None = None) -> None:
        values = tuple(v for v, _ in args)
        types = tuple(t for _, t in args)
        handle = stream.handle if stream is not None else 0
        check(cu.cuLaunchKernel(self.handle, *grid, *block, shared_bytes, handle, (values, types), 0))


class Graph:
    """A captured sequence of launches on one stream, replayed with a single call."""

    def __init__(self, stream: Stream, record) -> None:
        check(cu.cuStreamBeginCapture(stream.handle, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
        try:
            record()
        finally:
            graph = check(cu.cuStreamEndCapture(stream.handle))
        self.graph = graph
        self.exec = check(cu.cuGraphInstantiate(graph, 0))
        self.stream = stream

    def replay(self) -> None:
        check(cu.cuGraphLaunch(self.exec, self.stream.handle))


class Timer:
    """Device-side elapsed time between two points on a stream, in milliseconds."""

    def __init__(self, stream: Stream) -> None:
        self.stream = stream
        self.start = check(cu.cuEventCreate(0))
        self.stop = check(cu.cuEventCreate(0))

    def __enter__(self) -> Timer:
        check(cu.cuEventRecord(self.start, self.stream.handle))
        return self

    def __exit__(self, *exc) -> None:
        check(cu.cuEventRecord(self.stop, self.stream.handle))
        check(cu.cuEventSynchronize(self.stop))
        self.ms = check(cu.cuEventElapsedTime(self.start, self.stop))
