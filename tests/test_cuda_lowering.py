"""CUDA lowering of Tile IR: device results must match the eager interpreter (fp32 1e-5, bf16 1e-2)."""

import numpy as np
import pytest

import olaf
from olaf.graph.interp import interpret
from olaf.graph.ir import bf16_round
from olaf.tile.from_graph import to_tile
from olaf.pipeline import to_program
from olaf.tile.fuse import fuse

from test_tile import FUSION_CASES, R, X, Y, lower, rel, tt

pytestmark = pytest.mark.gpu

F32 = olaf.f32


def run_both(fn, arrays, specs=None, fused=False, tol=1e-5):
    from olaf.backend.cuda.runtime import CudaProgram

    specs = specs or [tt(*a.shape, dtype=_dtype(a)) for a in arrays]
    graph = lower(olaf.trace(fn, *specs).graph)
    program = to_program(graph, fused=fused)
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
    actual, _ = CudaProgram(to_program(graph)).run(leaves)
    errors = [rel(a, e) for a, e in zip(actual, expected)]
    assert max(errors) < 1e-5, max(errors)


@pytest.mark.parametrize("spec", ["ik,kj->ij", "ik,jk->ij", "ki,kj->ij", "ki,jk->ij"])
def test_matrix_instruction_dot_all_orientations(spec):
    """mma.sync path (Ampere tiles): every operand orientation, with edges that cut tiles."""
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram

    m, n, k = 200, 136, 96
    shape = {"i": m, "j": n, "k": k}
    ta, tb = spec.split("->")[0].split(",")
    a = bf16_round(R.normal(size=[shape[c] for c in ta]))
    b = bf16_round(R.normal(size=[shape[c] for c in tb]))
    graph = lower(olaf.trace(lambda x, y: olaf.einsum(spec, x, y), tt(*a.shape, dtype=olaf.bf16), tt(*b.shape, dtype=olaf.bf16)).graph)
    program = to_tile(graph, AMPERE_RTX3060)
    assert "nvvm.mma.sync" in lower_task(program.launches[0].task).text
    expected = interpret(graph, [a, b])[0][0]
    actual = CudaProgram(program).run([a, b])[0][0]
    # Both sides round the same f32-accurate sums to bf16; they differ only where f32 summation
    # order moves a value across a bf16 rounding boundary.
    assert rel(actual, expected) < 1e-2


def test_matrix_instruction_dot_falls_back_when_unaligned():
    # K = S is symbolic: 16-byte copies need S % 8 == 0, so S = 13 must take the general kernel
    # chosen at launch, and S = 64 the mma kernel.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram

    S = olaf.sym("S")
    graph = lower(olaf.trace(lambda x, y: x @ y, tt(64, S, dtype=olaf.bf16), tt(S, 48, dtype=olaf.bf16)).graph)
    program = to_tile(graph, AMPERE_RTX3060)
    lowered = lower_task(program.launches[0].task)
    assert lowered.aligned and lowered.fallback is not None
    device = CudaProgram(program)
    for s in (13, 64):
        a, b = bf16_round(R.normal(size=(64, s))), bf16_round(R.normal(size=(s, 48)))
        assert rel(device.run([a, b])[0][0], interpret(graph, [a, b])[0][0]) < 1e-2


def test_matrix_instruction_dot_with_ragged_rows_and_padding():
    # K = 50 is ragged in 16-byte chunks (partial zero-filled copies); the A operand is a padded
    # view (as the gradient of logits[:, :-1] is), whose pad rows are zero-filled.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram

    a = bf16_round(R.normal(size=(140, 56)))
    b = bf16_round(R.normal(size=(56, 72)))

    def fn(x, y):
        return olaf.pad(x[:, :50], (0, 0), (3, 0)) @ olaf.pad(y[:50], (0, 0), (0, 0))

    specs = [tt(140, 56, dtype=olaf.bf16), tt(56, 72, dtype=olaf.bf16)]
    graph = lower(olaf.trace(fn, *specs).graph)
    program = to_program(graph, AMPERE_RTX3060)
    assert any("nvvm.mma.sync" in lower_task(launch.task).text for launch in program.launches)
    assert rel(CudaProgram(program).run([a, b])[0][0], interpret(graph, [a, b])[0][0]) < 1e-2


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((4, 6, 20), (0,)),  # kept inner extent 32: partials of 4 threads combined in shared memory
        ((4, 6, 300), (0, 1)),  # 128 outputs, each thread's own: no exchange
        ((5, 16), (1,)),  # 16-wide rows: butterfly within groups of 16 lanes
        ((3, 8), (1,)),  # 8-wide rows, most lanes empty
        ((2, 3000), (1,)),  # long rows: warp butterflies and warp partials
        ((7,), (0,)),
    ],
)
@pytest.mark.parametrize("fn", ["sum", "max"])
def test_reduce_layouts(shape, axes, fn):
    x = R.normal(size=shape).astype(np.float32)
    reduce = olaf.sum if fn == "sum" else olaf.max
    run_both(lambda a: reduce(a, axis=axes), [x])
    run_both(lambda a: reduce(a, axis=axes), [R.integers(-50, 50, shape).astype(np.int32)])


def test_split_column_reduction_on_device():
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    x = R.normal(size=(4, 1024, 768)).astype(np.float32)
    graph = lower(olaf.trace(lambda a: olaf.sum(a, axis=(0, 1)), tt(4, 1024, 768)).graph)
    program = to_program(graph, AMPERE_RTX3060)
    assert len(program.launches) == 2
    assert rel(CudaProgram(program).run([x])[0][0], interpret(graph, [x])[0][0]) < 1e-5


@pytest.mark.parametrize("spec,a,b", [("ik,kj->ij", (70, 48), (48, 40)), ("ik,jk->ij", (33, 32), (64, 32)), ("ki,kj->ij", (32, 50), (32, 30))])
def test_register_operand_dot_uses_matrix_instructions(spec, a, b):
    # Default (non-pipelined) tiles: the dot's operands are register tiles staged through shared
    # memory for mma.sync, including f32 values that are widened bf16.
    from olaf.backend.cuda.lower import lower_task
    from olaf.backend.cuda.runtime import CudaProgram

    x, y = bf16_round(R.normal(size=a)), bf16_round(R.normal(size=b))

    def fn(p, q):
        return olaf.einsum(spec, olaf.cast(p, olaf.f32), olaf.cast(q, olaf.f32))

    graph = lower(olaf.trace(fn, tt(*a, dtype=olaf.bf16), tt(*b, dtype=olaf.bf16)).graph)
    program = to_program(graph)
    assert any("nvvm.mma.sync" in lower_task(launch.task).text for launch in program.launches)
    assert rel(CudaProgram(program).run([x, y])[0][0], interpret(graph, [x, y])[0][0]) < 1e-5


@pytest.mark.parametrize("backward", [False, True])
def test_online_softmax_fusion_on_device(backward):
    from test_tile import _attention_case

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    graph, args = _attention_case(backward)
    program = to_program(graph, AMPERE_RTX3060)
    assert "attention" in {launch.task.name for launch in program.launches}
    for a, e in zip(CudaProgram(program).run(args)[0], interpret(graph, args)[0]):
        assert rel(a, e) < 1e-2, rel(a, e)


def test_short_row_reduction_result_is_used_by_every_thread():
    # Rows of 5 reduce with a butterfly in groups of 8 lanes; the single result is then
    # broadcast back over the row, so every thread must hold it, not only thread 0.
    x = R.normal(size=(4, 5)).astype(np.float32)
    run_both(lambda a: a - olaf.broadcast_to(olaf.reshape(olaf.max(a, axis=1), (4, 1)), (4, 5)), [x], fused=True)


def test_kernel_cache_skips_lowering_on_the_second_compile(tmp_path, monkeypatch):
    # Not a GPU test at heart: lowering and PTX generation need no device.
    import olaf.backend.cuda.runtime as runtime

    monkeypatch.setenv("OLAF_CACHE_DIR", str(tmp_path))
    graph = lower(olaf.trace(lambda x: olaf.tanh(x) * 2.0, tt(3, 5)).graph)
    task = to_program(graph).launches[0].task
    first = runtime._compiled(task, "sm_86")

    def fail(task):
        raise AssertionError("lowered again")

    monkeypatch.setattr(runtime, "lower_task", fail)
    second = runtime._compiled(task, "sm_86")
    assert second[1] == first[1] and second[0].text == first[0].text
