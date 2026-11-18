"""conv (core op): forward and gradients against PyTorch, eager (f64) and on the device."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import olaf

R = np.random.default_rng(0)

# (x shape, w shape, stride, padding, dilation, groups)
CONVS = [
    ((2, 3, 9, 9), (4, 3, 3, 3), 1, 1, 1, 1),
    ((2, 3, 10, 11), (4, 3, 3, 3), 2, 1, 1, 1),
    ((1, 4, 9, 9), (6, 2, 3, 2), (2, 1), (1, 0), (1, 2), 2),
    ((2, 4, 8, 8), (4, 1, 3, 3), 1, 1, 1, 4),  # depthwise
    ((2, 3, 7, 7), (5, 3, 1, 1), 1, 0, 1, 1),  # 1x1
    ((2, 3, 12, 12), (4, 3, 5, 5), 3, 2, 1, 1),  # stride not dividing the input
    ((2, 2, 16), (3, 2, 3), 2, 1, 1, 1),  # 1-D (Whisper's front end)
    ((1, 2, 5, 6, 6), (3, 2, 3, 3, 3), 1, 1, 1, 1),  # 3-D
    # Weight gradients over large grids: read through window views of the channels-last grid
    # (the first and the 1-D case), or through im2col (the strided and the dilated case).
    ((2, 16, 18, 18), (4, 16, 3, 3), 1, 1, 1, 1),
    ((2, 16, 34, 34), (4, 16, 3, 3), 2, 1, 1, 1),
    ((1, 16, 20, 20), (4, 16, 3, 3), 1, 2, (2, 1), 1),
    ((2, 16, 40), (3, 16, 3), 1, 1, 1, 1),
]

# (x shape, w shape, stride, padding, output_padding, groups)
TRANSPOSED = [
    ((2, 4, 5, 5), (4, 3, 4, 4), 2, 1, 0, 1),  # DCGAN's upsampling
    ((2, 4, 5, 6), (4, 3, 3, 3), 2, 1, 1, 1),
    ((1, 4, 4, 4), (4, 2, 3, 3), 1, 0, 0, 2),
]


def relative(a, b):
    return float(np.linalg.norm(np.ravel(a) - np.ravel(b)) / np.linalg.norm(np.ravel(b)))


def torch_reference(fn, x, w, r):
    xt, wt = (torch.from_numpy(a).requires_grad_() for a in (x, w))
    y = fn(xt, wt)
    (y * torch.from_numpy(r)).sum().backward()
    return y.detach().numpy(), xt.grad.numpy(), wt.grad.numpy()


def olaf_eager(fn, x, w, r):
    spec = [olaf.TensorType(a.shape, olaf.f64) for a in (x, w, r)]

    def step(x, w, r):
        y = fn(x, w)
        grads = olaf.grad(lambda x, w: olaf.sum(fn(x, w) * r), argnums=(0, 1))(x, w)
        return y, grads

    (y, (dx, dw)), _ = olaf.trace(step, *spec).eager(x, w, r)
    return y, dx, dw


def _torch_conv(shape):
    return {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[len(shape)]


def _torch_conv_transpose(shape):
    return {3: F.conv_transpose1d, 4: F.conv_transpose2d, 5: F.conv_transpose3d}[len(shape)]


def _out_shape(fn, x, w):
    with torch.no_grad():
        return tuple(fn(torch.from_numpy(x), torch.from_numpy(w)).shape)


@pytest.mark.parametrize("xs,ws,stride,padding,dilation,groups", CONVS)
def test_conv_eager_matches_torch(xs, ws, stride, padding, dilation, groups):
    x, w = R.normal(size=xs), R.normal(size=ws)
    torch_fn = lambda x, w: _torch_conv(xs)(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)  # noqa: E731
    r = R.normal(size=_out_shape(torch_fn, x, w))
    want = torch_reference(torch_fn, x, w, r)
    got = olaf_eager(lambda x, w: olaf.conv(x, w, stride, padding, dilation, groups), x, w, r)
    for name, a, b in zip(("y", "dx", "dw"), got, want):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert relative(a, b) < 1e-12, name


@pytest.mark.parametrize("xs,ws,stride,padding,output_padding,groups", TRANSPOSED)
def test_conv_transpose_eager_matches_torch(xs, ws, stride, padding, output_padding, groups):
    x, w = R.normal(size=xs), R.normal(size=ws)
    torch_fn = lambda x, w: _torch_conv_transpose(xs)(x, w, stride=stride, padding=padding, output_padding=output_padding, groups=groups)  # noqa: E731
    r = R.normal(size=_out_shape(torch_fn, x, w))
    want = torch_reference(torch_fn, x, w, r)
    got = olaf_eager(lambda x, w: olaf.conv_transpose(x, w, stride, padding, output_padding, groups=groups), x, w, r)
    for name, a, b in zip(("y", "dx", "dw"), got, want):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert relative(a, b) < 1e-12, name


def _tile_check(fn, x, w, r):
    from olaf.pipeline import lower_graph, to_program
    from olaf.tile.interp import run_program

    spec = [olaf.TensorType(a.shape, olaf.f32) for a in (x, w, r)]
    graph = lower_graph(olaf.trace(lambda x, w, r: (fn(x, w), olaf.grad(lambda x, w: olaf.sum(fn(x, w) * r), argnums=(0, 1))(x, w)), *spec).graph)
    arrays = [a.astype(np.float32) for a in (x, w, r)]
    from olaf.graph.interp import interpret

    want, _ = interpret(graph, arrays)
    return to_program(graph), arrays, want


@pytest.mark.parametrize("xs,ws,stride,padding,dilation,groups", CONVS)
def test_conv_tile_program_matches_eager(xs, ws, stride, padding, dilation, groups):
    from olaf.tile.interp import run_program

    x, w = R.normal(size=xs), R.normal(size=ws)
    fn = lambda x, w: olaf.conv(x, w, stride, padding, dilation, groups)  # noqa: E731
    r = R.normal(size=_out_shape(lambda x, w: _torch_conv(xs)(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups), x, w))
    program, arrays, want = _tile_check(fn, x, w, r)
    got, _ = run_program(program, arrays)
    for a, b in zip(got, want):
        assert relative(a, b) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("xs,ws,stride,padding,dilation,groups", CONVS)
def test_conv_on_device_matches_eager(xs, ws, stride, padding, dilation, groups):
    from olaf.backend.cuda.runtime import CudaProgram

    x, w = R.normal(size=xs), R.normal(size=ws)
    fn = lambda x, w: olaf.conv(x, w, stride, padding, dilation, groups)  # noqa: E731
    r = R.normal(size=_out_shape(lambda x, w: _torch_conv(xs)(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups), x, w))
    program, arrays, want = _tile_check(fn, x, w, r)
    got, _ = CudaProgram(program).run(arrays)
    for a, b in zip(got, want):
        assert relative(a, b) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("xs,ws,stride,padding,output_padding,groups", TRANSPOSED)
def test_conv_transpose_on_device_matches_eager(xs, ws, stride, padding, output_padding, groups):
    from olaf.backend.cuda.runtime import CudaProgram

    x, w = R.normal(size=xs), R.normal(size=ws)
    fn = lambda x, w: olaf.conv_transpose(x, w, stride, padding, output_padding, groups=groups)  # noqa: E731
    r = R.normal(size=_out_shape(lambda x, w: _torch_conv_transpose(xs)(x, w, stride=stride, padding=padding, output_padding=output_padding, groups=groups), x, w))
    program, arrays, want = _tile_check(fn, x, w, r)
    got, _ = CudaProgram(program).run(arrays)
    for a, b in zip(got, want):
        assert relative(a, b) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize(
    "xs,ws,stride,padding,dilation,groups",
    [
        CONVS[1],
        CONVS[2],
        ((4, 32, 16, 16), (64, 32, 3, 3), 1, 1, 1, 1),
        # Small planes, plain and strided: implicit GEMM, row tiles over n, h and w.
        ((6, 32, 8, 8), (64, 32, 3, 3), 1, 1, 1, 1),
        ((6, 32, 16, 16), (64, 32, 3, 3), 2, 1, 1, 1),
    ],
)
def test_bf16_conv_on_device_matches_eager(xs, ws, stride, padding, dilation, groups):
    # bf16 operands go to the matrix instructions through strided window views.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram
    from olaf.graph.interp import interpret
    from olaf.graph.ir import bf16_round
    from olaf.pipeline import lower_graph, to_program

    x, w = (bf16_round(R.normal(size=s).astype(np.float32)) for s in (xs, ws))
    spec = [olaf.TensorType(a.shape, olaf.bf16) for a in (x, w)]
    fn = lambda x, w: olaf.conv(x, w, stride, padding, dilation, groups)  # noqa: E731
    graph = lower_graph(olaf.trace(lambda x, w: (fn(x, w), olaf.grad(lambda x, w: olaf.sum(olaf.cast(fn(x, w), olaf.f32) ** 2.0), argnums=(0, 1))(x, w)), *spec).graph)
    want, _ = interpret(graph, [x, w])
    # The device's capability, so the dots take the matrix instructions' tiles and lowering.
    got, _ = CudaProgram(to_program(graph, AMPERE_RTX3060)).run([x, w])
    for a, b in zip(got, want):
        assert relative(a, b) < 1e-2
