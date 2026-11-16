"""CUDA lowering of Tile IR: device results must match the eager interpreter (fp32 1e-5, bf16 1e-2)."""

import numpy as np
import pytest

import olaf
from olaf.graph.interp import interpret
from olaf.graph.ir import bf16_round
from olaf.tile.from_graph import to_tile
from olaf.tile.fuse import fuse

from test_tile import FUSION_CASES, R, X, Y, lower, rel, tt

pytestmark = pytest.mark.gpu

F32 = olaf.f32


def run_both(fn, arrays, specs=None, fused=False, tol=1e-5):
    from olaf.backend.cuda.runtime import CudaProgram

    specs = specs or [tt(*a.shape, dtype=_dtype(a)) for a in arrays]
    graph = lower(olaf.trace(fn, *specs).graph)
    program = to_tile(graph)
    if fused:
        program = fuse(program)
    expected, _ = interpret(graph, list(arrays))
    actual, _ = CudaProgram(program).run(list(arrays))
    for a, e in zip(actual, expected):
        assert a.shape == e.shape, (a.shape, e.shape)
        if e.dtype.kind in "biu":
            np.testing.assert_array_equal(a, e)
        else:
            assert rel(a, e) < tol, rel(a, e)
    return program


def _dtype(a):
    return {"float32": F32, "float64": olaf.f64, "int32": olaf.i32, "bool": olaf.DType.bool}[a.dtype.name]


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, y: x + y,
        lambda x, y: olaf.exp(x) * olaf.tanh(y) - x / (olaf.abs(y) + 1.0),
        lambda x, y: olaf.where(x < y, olaf.erf(x), olaf.sqrt(olaf.abs(y))),
        lambda x, y: olaf.maximum(x, y) ** 2.0 + olaf.rsqrt(olaf.abs(x) + 1.0),
        lambda x, y: olaf.cast(x > y, olaf.i32),
        lambda x, y: olaf.sin(x) * olaf.cos(y) + olaf.log(olaf.abs(x) + 1.0),
    ],
)
def test_elementwise(fn):
    run_both(fn, [X, Y])


def test_views_and_iota():
    run_both(lambda x: olaf.transpose(x, 0, 1)[1:4] * 2.0 + olaf.reshape(x, (5, 3))[1:4], [X])
    run_both(lambda x: x * olaf.cast(olaf.iota((3, 5), axis=1), F32) + olaf.broadcast_to(olaf.reshape(olaf.sum(x, axis=1), (3, 1)), (3, 5)), [X])


@pytest.mark.parametrize("axes", [(0,), (1,), (2,), (0, 1)])
@pytest.mark.parametrize("fn", ["sum", "max"])
def test_reduce(axes, fn):
    x = R.normal(size=(4, 6, 300)).astype(np.float32)
    run_both(lambda a: (olaf.sum if fn == "sum" else olaf.max)(a, axis=axes), [x])


@pytest.mark.parametrize(
    "spec,a,b",
    [
        ("ik,kj->ij", (5, 70), (70, 9)),
        ("ik,jk->ij", (37, 7), (9, 7)),
        ("ik,kj->ji", (40, 7), (7, 33)),
        ("bsd,de->bse", (2, 3, 4), (4, 6)),
        ("bhqd,bhkd->bhqk", (2, 2, 3, 4), (2, 2, 5, 4)),
        ("bsd,bse->de", (2, 3, 4), (2, 3, 6)),
        ("i,j->ij", (5,), (6,)),
        ("i,i->", (40,), (40,)),
    ],
)
def test_contract(spec, a, b):
    run_both(lambda x, y: olaf.einsum(spec, x, y), [R.normal(size=a).astype(np.float32), R.normal(size=b).astype(np.float32)])


def test_concat_pad_gather_scatter():
    run_both(lambda x, y: olaf.concat([x, y, x], axis=1), [X, Y])
    run_both(lambda x: olaf.pad(x, (1, 0), (2, 3), value=-1.5) * 2.0, [X])
    table = R.normal(size=(10, 6)).astype(np.float32)
    run_both(lambda t, i: olaf.take(t, i), [table, R.integers(0, 10, (2, 7)).astype(np.int32)])
    run_both(lambda x, l: olaf.cross_entropy(x, l), [X, R.integers(0, 5, (3,)).astype(np.int32)])
    run_both(lambda t, i: olaf.grad(lambda t: olaf.sum(olaf.take(t, i) ** 2.0))(t), [table, np.array([[1, 1, 3, 1]], np.int32)])


def test_composites_and_symbolic_shapes():
    run_both(lambda x: olaf.softmax(x, axis=1), [X])
    run_both(lambda x: olaf.layernorm(x), [X])
    q, k, v = (R.normal(size=(2, 4, 8)).astype(np.float32) for _ in range(3))
    run_both(lambda a, b, c: olaf.attention(a, b, c, olaf.iota((2, 4, 4), axis=1) >= olaf.iota((2, 4, 4), axis=2)), [q, k, v])
    from olaf.backend.cuda.runtime import CudaProgram

    B, D = olaf.sym("B"), olaf.sym("D")
    graph = lower(olaf.trace(lambda x: olaf.softmax(x * 2.0, axis=1) @ olaf.transpose(x, 0, 1), tt(B, D)).graph)
    device = CudaProgram(fuse(to_tile(graph)))
    for shape in [(3, 5), (7, 130)]:
        x = R.normal(size=shape).astype(np.float32)
        assert rel(device.run([x])[0][0], interpret(graph, [x])[0][0]) < 1e-5


@pytest.mark.parametrize("index", range(len(FUSION_CASES)))
def test_fused(index):
    fn, arrays = FUSION_CASES[index]
    run_both(fn, list(arrays), fused=True)


def test_f64_and_bf16():
    x64, y64 = X.astype(np.float64), Y.astype(np.float64)
    run_both(lambda x, y: olaf.softmax(x * y, axis=1) @ olaf.transpose(y, 0, 1), [x64, y64], fused=True, tol=1e-12)
    xb, yb = bf16_round(X), bf16_round(Y)
    specs = [tt(3, 5, dtype=olaf.bf16)] * 2
    run_both(lambda x, y: olaf.layernorm(olaf.exp(x) * y) @ olaf.transpose(y, 0, 1), [xb, yb], specs=specs, fused=True, tol=1e-2)


def test_gpt2_small_config_step():
    from test_gpt2 import IDS_SMALL, small_model, split_params, trace_step

    from olaf.backend.cuda.runtime import CudaProgram

    model = small_model()
    params = split_params(model)
    graph = lower(trace_step(params, model.config).graph)
    leaves = olaf.tree_flatten((params, IDS_SMALL))[0]
    expected, _ = interpret(graph, leaves)
    actual, _ = CudaProgram(fuse(to_tile(graph))).run(leaves)
    errors = [rel(a, e) for a, e in zip(actual, expected)]
    assert max(errors) < 1e-5, max(errors)
