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
