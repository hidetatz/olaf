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
from olaf.tile.fuse import fuse
from olaf.tile.interp import run_program

R = np.random.default_rng(0)
F32 = olaf.f32


def tt(*shape, dtype=F32):
    return TensorType(tuple(shape), dtype)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(np.ravel(a - b)) / max(np.linalg.norm(np.ravel(b)), 1e-30))


def lower(graph):
    from olaf.pipeline import lower_graph

    return lower_graph(graph)


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


FUSION_CASES = [
    (lambda x, y: olaf.exp(x) * olaf.tanh(y) - x / (olaf.abs(y) + 1.0), (X, Y)),
    (lambda x, y: olaf.softmax(x * 3.0, axis=1) + y, (X, Y)),
    (lambda x: olaf.layernorm(olaf.transpose(x, 0, 1) * 2.0), (X,)),
    (lambda x, y: olaf.einsum("ik,jk->ij", olaf.cast(x > 0.0, F32) * x, y), (X, Y)),
    (lambda x: olaf.sum(olaf.exp(x - 1.0), axis=0) + olaf.max(olaf.exp(x) * -1.0, axis=0), (X,)),
    (lambda x: olaf.concat([olaf.exp(x), x * 2.0], axis=1), (X,)),
]


@pytest.mark.parametrize("index", range(len(FUSION_CASES)))
def test_fusion_preserves_values(index):
    from olaf.tile.fuse import fuse

    fn, arrays = FUSION_CASES[index]
    graph = lower(olaf.trace(fn, *(tt(*a.shape) for a in arrays)).graph)
    unfused = to_tile(graph)
    fused = fuse(unfused)
    expected, _ = interpret(graph, list(arrays))
    actual, _ = run_program(fused, list(arrays))
    for a, e in zip(actual, expected):
        assert rel(a, e) < 1e-5
    assert len(fused.launches) < len(unfused.launches)


def test_fusion_makes_elementwise_chain_one_task():
    from olaf.tile.fuse import fuse

    graph = lower(olaf.trace(FUSION_CASES[0][0], tt(3, 5), tt(3, 5)).graph)
    assert len(fuse(to_tile(graph)).launches) == 1


def test_fusion_masks_inlined_lanes_for_reductions():
    # 300 is not a multiple of the 128-wide reduction tile: the overhanging lanes of the inlined
    # exp(x) must read as -inf for the max, not exp(0) = 1, or every row max would be at least 1.
    from olaf.tile.fuse import fuse

    x = R.normal(size=(4, 300)).astype(np.float32) - 10.0
    graph = lower(olaf.trace(lambda a: olaf.max(olaf.exp(a), axis=1), tt(4, 300)).graph)
    fused = fuse(to_tile(graph))
    assert len(fused.launches) == 1
    assert rel(run_program(fused, [x])[0][0], interpret(graph, [x])[0][0]) < 1e-6


def test_gpt2_small_config_step_fused_matches_eager():
    from test_gpt2 import IDS_SMALL, small_model, split_params, trace_step

    from olaf.tile.fuse import fuse

    model = small_model()
    params = split_params(model)
    graph = lower(trace_step(params, model.config).graph)
    unfused = to_tile(graph)
    fused = fuse(unfused)
    leaves = olaf.tree_flatten((params, IDS_SMALL))[0]
    expected, _ = interpret(graph, leaves)
    actual, _ = run_program(fused, leaves)
    errors = [rel(a, e) for a, e in zip(actual, expected)]
    assert max(errors) < 1e-5, max(errors)
    assert len(fused.launches) < len(unfused.launches) * 0.6


def test_hoisted_offsets_share_templates_across_unrolled_blocks():
    from test_gpt2 import IDS_SMALL, small_model, split_params, trace_step

    from olaf.pipeline import to_program

    model = small_model()
    params = split_params(model)
    graph = lower(trace_step(params, model.config).graph)
    plain = fuse(to_tile(graph))
    hoisted = to_program(graph)
    leaves = olaf.tree_flatten((params, IDS_SMALL))[0]
    for a, e in zip(run_program(hoisted, leaves)[0], interpret(graph, leaves)[0]):
        assert rel(a, e) < 1e-5
    # The two blocks differ only in where their weights start.
    assert len(hoisted.tasks()) < len(plain.tasks())


def test_align_rows_restrides_ragged_bf16_buffers():
    # A bf16 intermediate with 13-element rows, read transposed, sliced and padded: after
    # align_rows its rows start every 16 elements and every reader still sees the same values.
    from olaf.graph.ir import bf16_round
    from olaf.tile.normalize import align_rows

    x = bf16_round(R.normal(size=(6, 13)))
    specs = [tt(6, 13, dtype=olaf.bf16)]

    def fn(a):
        t = a * 2.0
        return olaf.transpose(t, 0, 1)[2:9] + 1.0, olaf.pad(t[1:], (1, 0), (0, 2)), olaf.sum(t, axis=0)

    graph = lower(olaf.trace(fn, *specs).graph)
    program = to_tile(graph)
    aligned = align_rows(program)
    padded = [b for b in aligned.buffers if b.strides is not None]
    assert padded and all(b.strides[-2] == 16 for b in padded)
    for a, e in zip(run_program(aligned, [x])[0], interpret(graph, [x])[0]):
        assert rel(a, e) < 1e-6


@pytest.mark.parametrize("shape,axes", [((1024, 256), (0,)), ((4, 512, 256), (0, 1)), ((2048, 64), (0,))])
def test_split_reductions_preserve_values(shape, axes):
    # With 28 SMs, these column reductions would run as 1-2 blocks; they are split in two passes.
    from olaf.backend.capability import AMPERE_RTX3060

    x = R.normal(size=shape).astype(np.float32)
    graph = lower(olaf.trace(lambda a: olaf.sum(a, axis=axes), tt(*shape)).graph)
    program = to_tile(graph, AMPERE_RTX3060)
    assert len(program.launches) == 2
    assert rel(run_program(program, [x])[0][0], interpret(graph, [x])[0][0]) < 1e-5


def _attention_case(backward: bool, s: int = 100, d: int = 64):
    from olaf.graph.ir import bf16_round

    rng = np.random.default_rng(3)
    q, k, v = (bf16_round(rng.normal(size=(1, 2, s, d))) for _ in range(3))

    def attn(q, k, v):
        mask = olaf.iota((s, s), axis=0) >= olaf.iota((s, s), axis=1)
        return olaf.attention(q, k, v, olaf.broadcast_to(mask, (1, 2, s, s)))

    def fn(q, k, v):
        if not backward:
            return attn(q, k, v)
        return olaf.value_and_grad(lambda q, k, v: olaf.sum(olaf.cast(attn(q, k, v), F32) * olaf.cast(v, F32)), (0, 1, 2))(q, k, v)

    graph = lower(olaf.trace(fn, *[tt(1, 2, s, d, dtype=olaf.bf16)] * 3).graph)
    return graph, [q, k, v]


@pytest.mark.parametrize("backward", [False, True])
def test_online_softmax_fusion(backward):
    """Attention (forward, and its backward) becomes FlashAttention-style tasks with equal results.

    The forward rounds the probabilities to bf16 before normalizing them (FlashAttention's order),
    so it differs from eager at bf16 rounding level; the bf16 tolerance applies.
    """
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.tile.online_softmax import fuse_online_softmax

    graph, args = _attention_case(backward)
    before = fuse(to_tile(graph, AMPERE_RTX3060))
    after = fuse_online_softmax(before)
    names = {launch.task.name for launch in after.launches}
    # dQ comes from its own task, or is accumulated atomically by the dK/dV task after a zeroing task.
    assert "attention" in names and (not backward or ("attention_dkv" in names and names & {"attention_dq", "attention_dq_zero"}))
    assert len(after.launches) < len(before.launches)
    for a, e in zip(run_program(after, args)[0], interpret(graph, args)[0]):
        assert rel(a, e) < 1e-2, rel(a, e)


def test_online_softmax_skips_masked_tiles_of_a_causal_mask():
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.tile.online_softmax import _causal_offset, _match_forward, _Program, fuse_online_softmax

    graph, args = _attention_case(True, s=200)
    before = fuse(to_tile(graph, AMPERE_RTX3060))
    p = _Program(before)
    forward = next(f for f in (_match_forward(p, i) for i in range(len(before.launches))) if f is not None)
    assert _causal_offset(p, forward) == 0
    after = fuse_online_softmax(before)
    for a, e in zip(run_program(after, args)[0], interpret(graph, args)[0]):
        assert rel(a, e) < 1e-2


def test_copies_into_state_and_outputs_are_eliminated():
    from olaf.pipeline import to_program
    from olaf.tile.normalize import eliminate_copies

    w_state = olaf.state("w", tt(3, 5))

    def fn(x):
        w = w_state.read()
        # Expensive enough (over fuse.CHEAP_OPS) that fusion keeps it as one buffer with two readers.
        new = olaf.tanh(olaf.exp(w * 0.5 + x) * x - 1.0)
        w_state.write(new)
        return new * 2.0

    graph = lower(olaf.trace(fn, tt(3, 5)).graph)
    plain = fuse(to_tile(graph))
    program = eliminate_copies(plain)
    assert sum(launch.task.origin == ("copy",) for launch in program.launches) < sum(launch.task.origin == ("copy",) for launch in plain.launches)
    w0 = R.normal(size=(3, 5)).astype(np.float32)
    got, got_state = run_program(to_program(graph), [X], {"w": w0})
    want, want_state = interpret(graph, [X], {"w": w0})
    for a, e in zip(got, want):
        assert rel(a, e) < 1e-6
    assert rel(got_state["w"], want_state["w"]) < 1e-6


def test_epilogue_fusion_absorbs_elementwise_after_a_dot():
    from olaf.tile.fuse import fuse_epilogues

    x, w = R.normal(size=(6, 40)).astype(np.float32), R.normal(size=(40, 24)).astype(np.float32)
    bias = R.normal(size=(24,)).astype(np.float32)

    def fn(a, b, c):
        return olaf.tanh(a @ b + olaf.broadcast_to(c, (6, 24))) * 2.0

    graph = lower(olaf.trace(fn, tt(6, 40), tt(40, 24), tt(24)).graph)
    before = fuse(to_tile(graph))
    after = fuse_epilogues(before)
    assert len(after.launches) == 1 < len(before.launches)
    assert rel(run_program(after, [x, w, bias])[0][0], interpret(graph, [x, w, bias])[0][0]) < 1e-5


def test_reshape_of_strided_views_stays_a_view():
    # The q/k/v slices of a fused projection split into heads without a copy.
    def fn(x):
        q = olaf.reshape(x[:, 4:8], (6, 2, 2))
        return q * 2.0

    x = R.normal(size=(6, 12)).astype(np.float32)
    graph = lower(olaf.trace(fn, tt(6, 12)).graph)
    program = to_tile(graph)
    assert [launch.task.name for launch in program.launches] == ["mul"]
    assert rel(run_program(program, [x])[0][0], interpret(graph, [x])[0][0]) < 1e-7


def test_epilogue_through_a_permuted_read():
    from olaf.tile.fuse import fuse_epilogues

    a, b = R.normal(size=(2, 6, 8)).astype(np.float32), R.normal(size=(2, 8, 4)).astype(np.float32)
    graph = lower(olaf.trace(lambda x, y: olaf.transpose(olaf.einsum("bij,bjk->bik", x, y), 1, 2) * 3.0, tt(2, 6, 8), tt(2, 8, 4)).graph)
    before = fuse(to_tile(graph))
    after = fuse_epilogues(before)
    assert len(after.launches) == 1 < len(before.launches)
    assert rel(run_program(after, [a, b])[0][0], interpret(graph, [a, b])[0][0]) < 1e-5


@pytest.mark.parametrize("backward", [False, True])
def test_row_fusion_merges_layernorm(backward):
    from olaf.tile.rows import fuse_rows

    x = R.normal(size=(3, 6, 40)).astype(np.float32)
    w, bias = R.normal(size=(40,)).astype(np.float32), R.normal(size=(40,)).astype(np.float32)

    def ln(a, w, c):
        return olaf.layernorm(a * 2.0 + 1.0, w, c)

    fn = (lambda a, w, c: olaf.grad(lambda a: olaf.sum(olaf.tanh(ln(a, w, c))))(a)) if backward else ln
    graph = lower(olaf.trace(fn, tt(3, 6, 40), tt(40), tt(40)).graph)
    before = fuse(to_tile(graph))
    after = fuse_rows(before)
    assert any(launch.task.name == "rows" for launch in after.launches)
    assert len(after.launches) < len(before.launches)
    assert rel(run_program(after, [x, w, bias])[0][0], interpret(graph, [x, w, bias])[0][0]) < 1e-5


def test_sibling_fusion_shares_loads_between_outputs():
    from olaf.pipeline import to_program

    m_state = olaf.state("m", tt(4, 300))
    v_state = olaf.state("v", tt(4, 300))
    p_state = olaf.state("p", tt(4, 300))

    def fn(g):
        m = m_state.read() * 0.9 + g * 0.1
        v = v_state.read() * 0.99 + g * g * 0.01
        m_state.write(m)
        v_state.write(v)
        p_state.write(p_state.read() - olaf.exp(m) / (olaf.sqrt(v) + 1e-8))
        return olaf.sum(g)

    graph = lower(olaf.trace(fn, tt(4, 300)).graph)
    program = to_program(graph)
    plain = fuse(to_tile(graph))
    assert len(program.launches) < len(plain.launches)
    g = R.normal(size=(4, 300)).astype(np.float32)
    state = {name: R.normal(size=(4, 300)).astype(np.float32) for name in "mvp"}
    state["v"] = np.abs(state["v"])
    got, got_state = run_program(program, [g], state)
    want, want_state = interpret(graph, [g], state)
    assert rel(got[0], want[0]) < 1e-6
    for name in "mvp":
        assert rel(got_state[name], want_state[name]) < 1e-6


def test_scan_gradient_stacks_are_written_in_place():
    # The per-iteration weight gradients of an unrolled scan are stored straight into their
    # slices of the stacked gradient: no concat task remains.
    from olaf.pipeline import to_program

    w = R.normal(size=(3, 5, 5)).astype(np.float32) * 0.3

    def fn(x, ws):
        return olaf.grad(lambda ws: olaf.sum(olaf.scan(lambda h, w: (olaf.tanh(h @ w), None), x, ws)[0]))(ws)

    graph = lower(olaf.trace(fn, tt(3, 5), tt(3, 5, 5)).graph)
    program = to_program(graph)
    assert not any(launch.task.origin == ("concat",) for launch in program.launches)
    assert rel(run_program(program, [X, w])[0][0], interpret(graph, [X, w])[0][0]) < 1e-5


def test_gradient_of_inner_axis_split_is_stored_in_slices():
    # The gradient of a value split along its last axis (GPT-2's qkv) is a concat of the pieces'
    # gradients; their producers store into strided slices of it, one of them permuted.
    from olaf.pipeline import to_program

    x = R.normal(size=(2, 6, 12)).astype(np.float32)

    def fn(a, w):
        def loss(a):
            # Products keep the pieces' gradients in tasks of their own (else they fuse into one).
            q, k, v = a[:, :, :4], a[:, :, 4:8], a[:, :, 8:]
            mm = lambda t, spec="bsd,de->bse": olaf.tanh(olaf.einsum(spec, t, w))
            return olaf.sum(mm(olaf.permute(q, (1, 0, 2)))) + olaf.sum(mm(k)) + olaf.sum(mm(v))

        return olaf.grad(loss)(a)

    w = R.normal(size=(4, 4)).astype(np.float32)
    graph = lower(olaf.trace(fn, tt(2, 6, 12), tt(4, 4)).graph)
    assert not any(op.kind == "add" and op.outputs[0].type.shape == (2, 6, 12) for op in graph.ops)
    program = to_program(graph)
    assert not any(launch.task.origin == ("concat",) for launch in program.launches)
    assert rel(run_program(program, [x, w])[0][0], interpret(graph, [x, w])[0][0]) < 1e-6


def test_product_is_inlined_into_a_split_column_sum():
    # A split reduction reads its input through a reshape (no source map); a producer laid out
    # like its output (layernorm's weight gradient, sum over rows of dy * xhat) is still inlined.
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import to_program

    shape = (4, 1024, 768)
    a, b = R.normal(size=shape).astype(np.float32), R.normal(size=shape).astype(np.float32)
    graph = lower(olaf.trace(lambda x, y: (olaf.sum(x * y, axis=(0, 1)), olaf.sum(x, axis=(0, 1))), tt(*shape), tt(*shape)).graph)
    program = to_program(graph, AMPERE_RTX3060)  # a GPU's unit count makes the reduction split
    # Two stages each, the first stages merged into one task (fuse_sibling_reductions): x is read once.
    assert sum(launch.task.name == "reduce_sum" for launch in program.launches) == 3
    assert not any(launch.task.name == "mul" for launch in program.launches)
    for got, want in zip(run_program(program, [a, b])[0], interpret(graph, [a, b])[0]):
        assert rel(got, want) < 1e-5


def test_row_task_stores_its_bf16_copy():
    # The f32 normalized rows are an output and are cast to bf16 for a product: the row task
    # stores both, and no separate cast pass remains.
    from olaf.pipeline import to_program

    x = R.normal(size=(3, 6, 40)).astype(np.float32)
    w, bias = R.normal(size=(40,)).astype(np.float32), R.normal(size=(40,)).astype(np.float32)
    wm = R.normal(size=(40, 8)).astype(np.float32)

    def fn(a, w, c, m):
        y = olaf.layernorm(a, w, c)
        return y, olaf.einsum("bsd,de->bse", olaf.cast(y, olaf.bf16), olaf.cast(m, olaf.bf16))

    graph = lower(olaf.trace(fn, tt(3, 6, 40), tt(40), tt(40), tt(40, 8)).graph)
    program = to_program(graph)
    assert not any(launch.task.name == "cast" and launch.task.args[0].view.shape == (3, 6, 40) for launch in program.launches)
    for got, want in zip(run_program(program, [x, w, bias, wm])[0], interpret(graph, [x, w, bias, wm])[0]):
        assert rel(got.astype(np.float64), want.astype(np.float64)) < 1e-5


def test_split_bf16_sum_keeps_partials_in_f32():
    # The unsplit sum accumulates in f32 and rounds once; the split one must not round its
    # partial sums to bf16 in between.
    from olaf.backend.capability import AMPERE_RTX3060

    graph = lower(olaf.trace(lambda x: olaf.sum(x, axis=(0, 1)), tt(4, 1024, 768, dtype=olaf.bf16)).graph)
    program = to_tile(graph, AMPERE_RTX3060)
    buffers = {b.name: b for b in program.buffers}
    stages = [launch for launch in program.launches if launch.task.name == "reduce_sum"]
    assert len(stages) == 2
    partial = next(n for n, a in zip(stages[0].buffers, stages[0].task.args) if a.writes)
    assert buffers[partial].type.dtype is olaf.f32
    x = R.normal(size=(4, 1024, 768)).astype(np.float32)
    from olaf.graph.ir import bf16_round

    x = bf16_round(x)
    got, want = run_program(program, [x])[0][0], interpret(graph, [x])[0][0]
    assert rel(got.astype(np.float64), want.astype(np.float64)) < 1e-2


def test_row_fusion_keeps_statistics_that_kept_reductions_read():
    # top-2 by repeated max: the reduction that finds the max's position is read before the
    # pointwise task row fusion starts from, so it stays, and so must the max it reads.
    from olaf.examples.mixtral import top_k
    from olaf.pipeline import to_program

    x = R.normal(size=(64, 4)).astype(np.float32)

    def fn(a):
        indices, values = top_k(olaf.softmax(a, axis=-1), 2)
        return indices[0], indices[1], values[0] + values[1]

    graph = lower(olaf.trace(fn, tt(64, 4)).graph)
    program = to_program(graph)
    written = {n for launch in program.launches for n, a in zip(launch.buffers, launch.task.args) if a.writes}
    temps = {b.name for b in program.buffers if b.kind.value == "temp"}
    assert {n for launch in program.launches for n in launch.buffers if n in temps} <= written
    for got, want in zip(run_program(program, [x])[0], interpret(graph, [x])[0]):
        np.testing.assert_allclose(got, want, rtol=1e-6)
