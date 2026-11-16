"""The Driver API layer, checked with a hand-written PTX kernel (independent of our lowering)."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

ADD_PTX = """
.version 7.0
.target sm_80
.address_size 64
.visible .entry add(.param .u64 a, .param .u64 b, .param .u64 out, .param .u32 n)
{
  .reg .pred %p;
  .reg .b32 %r<5>;
  .reg .f32 %f<4>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [a];
  ld.param.u64 %rd2, [b];
  ld.param.u64 %rd3, [out];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r2, %r2, %r3, %r4;
  setp.ge.u32 %p, %r2, %r1;
  @%p bra DONE;
  mul.wide.u32 %rd4, %r2, 4;
  add.s64 %rd5, %rd1, %rd4;
  add.s64 %rd6, %rd2, %rd4;
  add.s64 %rd7, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  ld.global.f32 %f2, [%rd6];
  add.f32 %f3, %f1, %f2;
  st.global.f32 [%rd7], %f3;
DONE:
  ret;
}
"""


def test_launch_and_graph_replay():
    from olaf.backend.cuda.driver import I32, POINTER, DeviceBuffer, Graph, Module, Stream, device

    assert device().compute_capability >= (8, 0)
    n = 1000
    a, b = np.arange(n, dtype=np.float32), np.full(n, 0.5, np.float32)
    bufs = [DeviceBuffer(n * 4) for _ in range(3)]
    bufs[0].upload(a)
    bufs[1].upload(b)
    kernel = Module(ADD_PTX).function("add")
    stream = Stream()
    args = [(bufs[0].ptr, POINTER), (bufs[1].ptr, POINTER), (bufs[2].ptr, POINTER), (n, I32)]
    kernel.launch(((n + 255) // 256, 1, 1), (256, 1, 1), args, stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(bufs[2].download(np.float32, n), a + b)

    # Replay out = a + out twice through a captured graph: the launch arguments are fixed at
    # capture, so the second replay must see the first one's result.
    bufs[2].upload(np.zeros(n, np.float32))
    graph = Graph(stream, lambda: kernel.launch(((n + 255) // 256, 1, 1), (256, 1, 1), [(bufs[0].ptr, POINTER), (bufs[2].ptr, POINTER), (bufs[2].ptr, POINTER), (n, I32)], stream=stream))
    graph.replay()
    graph.replay()
    stream.synchronize()
    np.testing.assert_array_equal(bufs[2].download(np.float32, n), 2 * a)


def test_bad_ptx_reports_the_jit_log():
    from olaf.backend.cuda.driver import CudaError, Module

    with pytest.raises(CudaError, match="(?i)ptx|syntax|error"):
        Module(".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry broken() { bogus; }\n")
