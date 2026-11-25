"""CUDA lowering of Tile IR: device results must match the eager interpreter (fp32 1e-5, bf16 1e-2)."""

import math

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


def test_linear_layer_gradients_split_and_swap_their_products():
    """A weight gradient with few output tiles and a long batch x sequence reduction runs as
    partial products over the batch (then summed); an input gradient written weight-first
    takes the activation as its rows, so the accumulator's columns are the output's
    contiguous axis and the fused residual add reads along them."""
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    B, S, D = 4, 1024, 96
    dy, x, w, r = (bf16_round(R.normal(size=s)) for s in ((B, S, D), (B, S, D), (D, D), (B, S, D)))

    def fn(dy, x, w, r):
        return olaf.einsum("aby,abz->zy", dy, x), olaf.cast(olaf.einsum("zy,abz->aby", w, dy), F32) + olaf.cast(r, F32)

    graph = lower(olaf.trace(fn, *(tt(*a.shape, dtype=olaf.bf16) for a in (dy, x, w, r))).graph)
    program = to_program(graph, AMPERE_RTX3060)
    contracts = {launch.task.origin[1]: launch.task for launch in program.launches if launch.task.origin[0] == "contract"}
    # The weight gradient: partial products over the batch (and, with 4 tiles still too few for
    # 28 SMs, over parts of the sequence too: 0.150 -> 0.121 ms), summed once.
    (weight,) = [spec for spec in contracts if spec != "zy,abz->aby"]
    partials = weight.split("->")[1][:-2]
    assert "zy,abz->aby" in contracts and "a" in partials, set(contracts)
    assert contracts["zy,abz->aby"].args[0].view.shape == (B, S, D)
    expected, _ = interpret(graph, [dy, x, w, r])
    actual, _ = CudaProgram(program).run([dy, x, w, r])
    for a, e in zip(actual, expected):
        assert rel(a, e) < 1e-2, rel(a, e)


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


@pytest.mark.parametrize("causal", [False, True])
def test_online_softmax_fusion_with_a_bias_on_device(causal):
    from test_tile import _biased_attention_case

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    graph, args = _biased_attention_case(causal)
    program = to_program(graph, AMPERE_RTX3060)
    assert "attention_dkv" in {launch.task.name for launch in program.launches}
    for a, e in zip(CudaProgram(program).run(args)[0], interpret(graph, args)[0]):
        assert rel(a, e) < 1e-2, rel(a, e)


@pytest.mark.parametrize("s", [65, 150])
def test_attention_gradients_stored_merged_on_device(s):
    from test_tile import _projected_attention_case

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    graph, args = _projected_attention_case(s)
    program = to_program(graph, AMPERE_RTX3060)
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


@pytest.mark.gpu
def test_gradient_of_inner_axis_split_stored_in_slices():
    def fn(a, w):
        def loss(a):
            # Products keep the pieces' gradients in tasks of their own (else they fuse into one).
            q, k, v = a[:, :, :4], a[:, :, 4:8], a[:, :, 8:]
            mm = lambda t, spec="bsd,de->bse": olaf.tanh(olaf.einsum(spec, t, w))
            return olaf.sum(mm(olaf.permute(q, (1, 0, 2)))) + olaf.sum(mm(k)) + olaf.sum(mm(v))

        return olaf.grad(loss)(a)

    run_both(fn, [R.normal(size=(2, 6, 12)).astype(np.float32), R.normal(size=(4, 4)).astype(np.float32)], fused=True)


@pytest.mark.gpu
def test_row_task_stores_its_bf16_copy():
    def fn(a, w, c, m):
        y = olaf.layernorm(a, w, c)
        return y, olaf.einsum("bsd,de->bse", olaf.cast(y, olaf.bf16), olaf.cast(m, olaf.bf16))

    arrays = [R.normal(size=s).astype(np.float32) for s in ((3, 6, 40), (40,), (40,), (40, 8))]
    run_both(fn, arrays, fused=True, tol=1e-2)


@pytest.mark.gpu
def test_sibling_column_sums_share_one_loop():
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    shape = (4, 1024, 768)
    arrays = [R.normal(size=shape).astype(np.float32) for _ in range(2)]
    graph = lower(olaf.trace(lambda x, y: (olaf.sum(x * y, axis=(0, 1)), olaf.sum(x, axis=(0, 1))), tt(*shape), tt(*shape)).graph)
    program = to_program(graph, AMPERE_RTX3060)
    assert sum(launch.task.name == "reduce_sum" for launch in program.launches) == 3
    expected, _ = interpret(graph, arrays)
    actual, _ = CudaProgram(program).run(arrays)
    for a, e in zip(actual, expected):
        assert rel(a, e) < 1e-5


def test_offsets_past_i32_are_rejected():
    # Kernels compute element offsets in i32; past 2^31 they would wrap silently. Only the IR is
    # built here (nothing is allocated), so no device is needed.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import check_index_range

    big = to_program(lower(olaf.trace(lambda x: x * 2.0, tt(2**16, 2**15 + 8)).graph), AMPERE_RTX3060)
    with pytest.raises(OverflowError, match="i32"):
        check_index_range(big, {})
    small = to_program(lower(olaf.trace(lambda x: x * 2.0, tt(2**15, 2**16 - 1)).graph), AMPERE_RTX3060)
    check_index_range(small, {})  # 2^31 - 2^15 elements: the last offset still fits


@pytest.mark.gpu
@pytest.mark.parametrize("width", [2048, 4096, 8192])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
def test_long_rows_fuse_and_match(width, norm):
    # Rows past 1024 (Llama's hidden sizes) fuse into one task over more warps.
    from olaf import nn

    def fn(x, w):
        if norm == "layernorm":
            return olaf.layernorm(x, w, w * 0.5)
        return nn.RMSNorm(width)({"weight": w}, x)

    arrays = [R.normal(size=(3, 5, width)).astype(np.float32), R.normal(size=(width,)).astype(np.float32)]
    program = run_both(lambda x, w: olaf.grad(lambda x: olaf.sum(olaf.tanh(fn(x, w))))(x), arrays, fused=True)
    assert any(launch.task.name == "rows" and launch.task.warps > 4 for launch in program.launches)


@pytest.mark.gpu
def test_bf16_scatter_add_rounds_once():
    # Many updates land on few rows: in bf16 each add would round; the sum must round once.
    from olaf.backend.cuda.runtime import CudaProgram
    from olaf.graph.ir import bf16_round

    target = bf16_round(R.normal(size=(4, 8)).astype(np.float32))
    updates = bf16_round(R.normal(size=(300, 8)).astype(np.float32) * 0.01)
    index = (np.arange(300) % 4).astype(np.int32)
    specs = [tt(4, 8, dtype=olaf.bf16), tt(300, dtype=olaf.i32), tt(300, 8, dtype=olaf.bf16)]
    graph = lower(olaf.trace(lambda t, i, u: olaf.scatter_add(t, i, u), *specs).graph)
    (got,), _ = CudaProgram(to_program(graph, fused=True)).run([target, index, updates])
    exact = target.astype(np.float64)
    np.add.at(exact, index, updates.astype(np.float64))
    np.testing.assert_array_equal(got, bf16_round(exact.astype(np.float32)))


def test_gathers_in_packed_row_reductions():
    # GAT's attention gradient: rows of two gathers multiplied and summed over 64, evaluated in
    # the reduction (no gathered tensor written), several rows per block; ragged 37-long rows
    # and a row count that is not a multiple of the rows per block.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram

    h, g = R.normal(size=(30, 4, 64)).astype(np.float32), R.normal(size=(30, 4, 64)).astype(np.float32)
    ragged = R.normal(size=(30, 3, 37)).astype(np.float32)
    src, dst = R.integers(0, 30, (203,)).astype(np.int32), R.integers(0, 30, (203,)).astype(np.int32)

    def fn(h, g, r, src, dst):
        return olaf.sum(olaf.take(g, dst) * olaf.take(h, src), axis=-1), olaf.max(olaf.take(r, src) * 2.0, axis=-1)

    graph = lower(olaf.trace(fn, tt(30, 4, 64), tt(30, 4, 64), tt(30, 3, 37), tt(203, dtype=olaf.i32), tt(203, dtype=olaf.i32)).graph)
    program = to_program(graph, AMPERE_RTX3060)
    assert not any(launch.task.name == "gather" for launch in program.launches)
    reduces = [l for l in program.launches if l.task.name.startswith("reduce")]
    assert len(reduces) == 2 and all(math.prod(l.task.grid) < 203 * 3 for l in reduces), [l.task.grid for l in reduces]
    expected, _ = interpret(graph, [h, g, ragged, src, dst])
    actual, _ = CudaProgram(program).run([h, g, ragged, src, dst])
    for a, e in zip(actual, expected):
        assert rel(a, e) < 1e-6, rel(a, e)


def test_products_of_ragged_operands_on_device():
    """Matrix-instruction products whose instances past a data-dependent count run no reduction
    steps (rows) or stop early (the reduced token axis); the result is unchanged."""
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram
    from test_tile import RAGGED_COUNTS, ragged_products

    E, T, D, F = 4, 256, 64, 96
    arrays = [bf16_round(R.normal(size=s)) for s in ((E, T, D), (E, F, D), (E, T, F))] + [RAGGED_COUNTS]
    specs = [tt(*a.shape, dtype=olaf.bf16) for a in arrays[:3]] + [tt(E, dtype=olaf.i32)]
    graph = lower(olaf.trace(ragged_products, *specs).graph)
    program = to_program(graph, AMPERE_RTX3060)
    assert sum(launch.task.origin[:1] == ("contract",) and len(launch.task.args) > 3 for launch in program.launches) == 2
    expected, _ = interpret(graph, arrays)
    actual, _ = CudaProgram(program).run(arrays)
    for a, e in zip(actual, expected):
        assert rel(a, e) < 1e-2, rel(a, e)
