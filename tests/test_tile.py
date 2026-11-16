"""Graph IR -> Tile IR conversion: the tile interpreter must match the eager interpreter."""

import numpy as np
import pytest

import olaf
from olaf.graph.interp import interpret
from olaf.graph.types import DType, TensorType
from olaf.passes.canonicalize import canonicalize
from olaf.passes.decompose import decompose
from olaf.passes.unroll import unroll_scans
from olaf.tile.from_graph import to_tile
from olaf.tile.interp import run_program

R = np.random.default_rng(0)
F32 = olaf.f32


def tt(*shape, dtype=F32):
    return TensorType(tuple(shape), dtype)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(np.ravel(a - b)) / max(np.linalg.norm(np.ravel(b)), 1e-30))


def lower(graph):
    return canonicalize(unroll_scans(decompose(graph)))


def check(fn, *arrays, specs=None, tol=1e-5):
    specs = specs or [tt(*a.shape, dtype=DType(str(a.dtype).replace("float", "f").replace("int", "i"))) for a in arrays]
    graph = lower(olaf.trace(fn, *specs).graph)
    program = to_tile(graph)
    expected, _ = interpret(graph, arrays)
    actual, _ = run_program(program, arrays)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype, (a.shape, a.dtype, e.shape, e.dtype)
        if e.dtype.kind in "biu":
            np.testing.assert_array_equal(a, e)
        else:
            assert rel(a, e) < tol, rel(a, e)
    return program


X = R.normal(size=(3, 5)).astype(np.float32)
Y = R.normal(size=(3, 5)).astype(np.float32)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, y: x + y,
        lambda x, y: olaf.exp(x) * olaf.tanh(y) - x / (olaf.abs(y) + 1.0),
        lambda x, y: olaf.where(x < y, x, y),
        lambda x, y: olaf.maximum(x, y) ** 2.0,
        lambda x, y: olaf.cast(x > y, olaf.i32),
    ],
)
def test_elementwise(fn):
    check(fn, X, Y)


def test_views_fold_into_addressing():
    program = check(lambda x: olaf.transpose(x, 0, 1)[1:4, ::1] * 2.0 + olaf.reshape(x, (5, 3))[1:4], X)
    # Only the multiply and the add materialize; transpose, slice and reshape are views.
    assert len(program.launches) == 2


def test_broadcast_and_iota():
    check(lambda x: x + olaf.broadcast_to(olaf.reshape(olaf.sum(x, axis=1), (3, 1)), (3, 5)), X)
    check(lambda x: x * olaf.cast(olaf.iota((3, 5), axis=1), F32), X)


@pytest.mark.parametrize("axes", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("fn", ["sum", "max"])
def test_reduce(axes, fn):
    x = R.normal(size=(4, 6, 300)).astype(np.float32)
    reduce = olaf.sum if fn == "sum" else olaf.max
    check(lambda a: reduce(a, axis=axes), x)
    check(lambda a: reduce(a, axis=(2,)), x)


def test_integer_sum():
    check(lambda a: olaf.sum(a, axis=0), R.integers(0, 9, (5, 7)).astype(np.int32))


@pytest.mark.parametrize(
    "spec,a,b",
    [
        ("ik,kj->ij", (5, 70), (70, 9)),
        ("ik,jk->ij", (5, 7), (9, 7)),
        ("ik,kj->ji", (40, 7), (7, 33)),
        ("bsd,de->bse", (2, 3, 4), (4, 6)),
        ("bhqd,bhkd->bhqk", (2, 2, 3, 4), (2, 2, 5, 4)),
        ("bsd,bse->de", (2, 3, 4), (2, 3, 6)),
        ("i,j->ij", (5,), (6,)),
        ("i,i->", (40,), (40,)),
    ],
)
def test_contract(spec, a, b):
    check(lambda x, y: olaf.einsum(spec, x, y), R.normal(size=a).astype(np.float32), R.normal(size=b).astype(np.float32))


def test_concat_and_pad():
    check(lambda x, y: olaf.concat([x, y, x], axis=1), X, Y)
    check(lambda x: olaf.pad(x, (1, 0), (2, 3), value=-1.5) * 2.0, X)
    check(lambda x: olaf.concat([olaf.pad(x, (0, 1), (0, 0)), x[:, :1]], axis=1), X)


def test_gather_and_scatter_add():
    table = R.normal(size=(10, 6)).astype(np.float32)
    ids = R.integers(0, 10, (2, 7)).astype(np.int32)
    check(lambda t, i: olaf.take(t, i), table, ids)
    labels = R.integers(0, 5, (3,)).astype(np.int32)
    check(lambda x, l: olaf.cross_entropy(x, l), X, labels)
    # take's VJP is a scatter_add into the table, with repeated ids accumulating.
    repeated = np.array([[1, 1, 3, 1]], np.int32)
    check(lambda t, i: olaf.grad(lambda t: olaf.sum(olaf.take(t, i) ** 2.0))(t), table, repeated)


def test_composites():
    check(lambda x: olaf.softmax(x, axis=1), X)
    check(lambda x: olaf.layernorm(x), X)
    q, k, v = (R.normal(size=(2, 4, 8)).astype(np.float32) for _ in range(3))
    check(lambda a, b, c: olaf.attention(a, b, c, olaf.iota((2, 4, 4), axis=1) >= olaf.iota((2, 4, 4), axis=2)), q, k, v)


def test_symbolic_shapes():
    B, D = olaf.sym("B"), olaf.sym("D")
    graph = lower(olaf.trace(lambda x: olaf.softmax(x * 2.0, axis=1) @ olaf.transpose(x, 0, 1), tt(B, D)).graph)
    program = to_tile(graph)
    for shape in [(3, 5), (7, 130)]:
        x = R.normal(size=shape).astype(np.float32)
        assert rel(run_program(program, [x])[0][0], interpret(graph, [x])[0][0]) < 1e-5


def test_scan_unrolled():
    w = R.normal(size=(3, 5, 5)).astype(np.float32) * 0.3

    def fn(x, ws):
        return olaf.scan(lambda h, w: (olaf.tanh(h @ w), h * 2.0), x, ws)

    check(fn, X, w)
    check(lambda x, ws: olaf.grad(lambda x, ws: olaf.sum(fn(x, ws)[0]) + olaf.sum(fn(x, ws)[1]), argnums=(0, 1))(x, ws), X, w)


def test_state_is_double_buffered():
    w_state = olaf.state("w", tt(3, 5))

    def fn(x):
        w = w_state.read()
        w_state.write(w + x)
        return w * 2.0

    traced = olaf.trace(fn, tt(3, 5))
    graph = lower(traced.graph)
    w0 = R.normal(size=(3, 5)).astype(np.float32)
    expected, expected_state = interpret(graph, [X], {"w": w0})
    actual, actual_state = run_program(to_tile(graph), [X], {"w": w0})
    # The result reads the step's initial w even though the write is emitted before it.
    assert rel(actual[0], expected[0]) < 1e-6
    assert rel(actual_state["w"], expected_state["w"]) < 1e-6


def test_template_text_ignores_buffer_names():
    program = check(lambda x, y: olaf.exp(x) + olaf.exp(y), X, Y)
    exps = [launch for launch in program.launches if launch.task.name == "exp"]
    assert len(exps) == 2 and exps[0].buffers != exps[1].buffers
    assert exps[0].task.digest() == exps[1].task.digest()
    assert len(program.tasks()) == 2


def test_gpt2_small_config_step_matches_eager():
    """The whole training step (forward, backward, every gradient) of a 2-layer GPT-2."""
    from test_gpt2 import IDS_SMALL, small_model, split_params, trace_step

    model = small_model()
    params = split_params(model)
    traced = trace_step(params, model.config)
    graph = lower(traced.graph)
    program = to_tile(graph)
    leaves = olaf.tree_flatten((params, IDS_SMALL))[0]
    expected, _ = interpret(graph, leaves)
    actual, _ = run_program(program, leaves)
    errors = [rel(a, e) for a, e in zip(actual, expected)]
    assert max(errors) < 1e-5, max(errors)
    # Both layers' tasks share templates: unrolling must not multiply the unique task count.
    assert len(program.tasks()) < len(program.launches) / 2
