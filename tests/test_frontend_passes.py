import numpy as np
import pytest
import torch
import torch.nn.functional as F

import olaf
from olaf.backend.capability import AMPERE_RTX3060, CPU, AsyncCopy
from olaf.graph.builder import GraphBuilder, verify
from olaf.graph.interp import interpret
from olaf.graph.types import StateKind, TensorType
from olaf.passes.canonicalize import canonicalize
from olaf.passes.decompose import decompose

F32 = olaf.f32
R = np.random.default_rng(2)
X = R.normal(size=(3, 4)).astype(np.float32)


def tt(*shape, dtype=F32):
    return TensorType(tuple(shape), dtype)


def rel(a, b):
    return np.linalg.norm(np.ravel(a) - np.ravel(b)) / np.linalg.norm(np.ravel(b))


class TestTrace:
    def test_numpy_style_broadcast_becomes_explicit(self):
        traced = olaf.trace(lambda a, b: a + b, tt(3, 1), tt(4))
        kinds = [op.kind for op in traced.graph.ops]
        assert kinds == ["broadcast", "broadcast", "add"]
        out, _ = traced.eager(np.ones((3, 1), np.float32), np.ones(4, np.float32))
        assert out.shape == (3, 4)

    def test_python_scalar_follows_tensor_dtype(self):
        traced = olaf.trace(lambda a: 2.0 * a + 1, tt(2, dtype=olaf.f64))
        assert traced.eager(np.ones(2))[0].dtype == np.float64

    def test_no_implicit_promotion(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a, b: a + b, tt(2), tt(2, dtype=olaf.f64))

    def test_tensor_truth_value_rejected(self):
        def fn(a):
            if a > 0.0:
                return a
            return -a

        with pytest.raises(TypeError):
            olaf.trace(fn, tt())

    def test_python_control_flow_evaluated_at_trace_time(self):
        def fn(a, flag):
            for _ in range(3):
                a = a * 2.0
            return a if flag else -a

        traced = olaf.trace(lambda a: fn(a, True), tt())
        assert len([op for op in traced.graph.ops if op.kind == "mul"]) == 3

    def test_operations_outside_trace_rejected(self):
        with pytest.raises(RuntimeError):
            olaf.zeros((2,))

    def test_leaked_tensor_rejected(self):
        leaked = []

        def first(a):
            leaked.append(a)
            return a

        olaf.trace(first, tt())
        with pytest.raises(RuntimeError):
            olaf.trace(lambda a: a + leaked[0], tt())

    def test_pytree_inputs_and_outputs(self):
        def fn(tree):
            return {"sum": tree["a"] + tree["b"][0], "same": tree["b"][1]}

        spec = {"a": tt(2), "b": (tt(2), tt(1))}
        traced = olaf.trace(fn, spec)
        out, _ = traced.eager({"a": np.ones(2, np.float32), "b": (np.ones(2, np.float32), np.zeros(1, np.float32))})
        np.testing.assert_array_equal(out["sum"], [2, 2])
        np.testing.assert_array_equal(out["same"], [0])

    def test_getitem(self):
        traced = olaf.trace(lambda a: (a[1], a[:, 1:3], a[..., -1], a[-2:, :2]), tt(3, 4))
        a, b, c, d = traced.eager(X)[0]
        np.testing.assert_array_equal(a, X[1])
        np.testing.assert_array_equal(b, X[:, 1:3])
        np.testing.assert_array_equal(c, X[..., -1])
        np.testing.assert_array_equal(d, X[-2:, :2])

    def test_symbolic_getitem(self):
        S = olaf.sym("S")
        traced = olaf.trace(lambda a: a[:, :-1], tt(2, S))
        assert traced.graph.results[0].type.shape == (2, S - 1)
        assert traced.eager(np.zeros((2, 5), np.float32))[0].shape == (2, 4)

    def test_inexact_python_number_on_integer_tensor_rejected(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: a * 0.5, tt(3, dtype=olaf.i32))
        traced = olaf.trace(lambda a: a * 2.0, tt(3, dtype=olaf.i32))
        np.testing.assert_array_equal(traced.eager(np.array([1, 2, 3], np.int32))[0], [2, 4, 6])

    @pytest.mark.parametrize("key", [slice(None, 100), slice(3, 1), slice(-20, None)])
    def test_out_of_range_slice_rejected_at_trace_time(self, key):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: a[key], tt(10))

    def test_symbolic_slice_checked_at_run_time(self):
        S = olaf.sym("S")
        traced = olaf.trace(lambda a: a[:4], tt(S))
        with pytest.raises(IndexError):
            traced.eager(np.zeros(3, np.float32))

    def test_bool_reduce_rejected(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: olaf.sum(a > 0.0), tt(3))

    def test_sharding_does_not_change_type_equality(self):
        sharded = TensorType((4,), F32, olaf.Sharding(("dp",)))
        decl = olaf.state("w", sharded)

        def fn():
            w = decl.read()
            decl.write(olaf.scan(lambda c, _: (2.0 * c, None), w, None, length=2)[0])

        _, state = olaf.trace(fn).eager(state={"w": np.ones(4, np.float32)})
        np.testing.assert_array_equal(state["w"], [4, 4, 4, 4])

    def test_same_function_gives_same_text(self):
        def fn(a):
            return olaf.sum(olaf.softmax(a) * a)

        assert olaf.trace(fn, tt(3, 4)).graph.text() == olaf.trace(fn, tt(3, 4)).graph.text()

    def test_shape_change_needs_no_retrace(self):
        B, S = olaf.sym("B"), olaf.sym("S", multiple_of=2)
        traced = olaf.trace(lambda a: olaf.sum(a * a, 1), tt(B, S))
        for shape in [(1, 2), (3, 8)]:
            assert traced.eager(np.ones(shape, np.float32))[0].shape == shape[:1]
        with pytest.raises(ValueError):
            traced.eager(np.ones((3, 3), np.float32))

    def test_tensor_equality_is_rejected(self):
        # `if x == 0:` must not silently trace one branch.
        with pytest.raises(TypeError):
            olaf.trace(lambda x: x if x == 0.0 else x * 2.0, tt(3))
        with pytest.raises(TypeError):
            olaf.trace(lambda x: x if x != 0.0 else x * 2.0, tt(3))

    def test_tensor_hashes_by_identity(self):
        def f(x):
            y = x * 2.0
            assert len({x, y, x}) == 2
            return y

        olaf.trace(f, tt(3))

    def test_not_equal_through_where(self):
        traced = olaf.trace(lambda x: olaf.where(olaf.equal(x, 0.0), olaf.constant(False), True), tt(3))
        np.testing.assert_array_equal(traced.eager(np.array([0, 1, 0], np.float32))[0], [False, True, False])

    def test_bf16_training_graph_traces_and_decomposes(self):
        # bf16 eager arrives in P1, but the graph must already be buildable for lowering.
        def loss(w, x):
            return olaf.sum(olaf.gelu(olaf.layernorm(x * w + 1.0), "tanh"))

        traced = olaf.trace(lambda w, x: olaf.value_and_grad(loss)(w, x), tt(4, dtype=olaf.bf16), tt(3, 4, dtype=olaf.bf16))
        verify(decompose(traced.graph))
        with pytest.raises(NotImplementedError):
            traced.eager(np.ones(4, np.float32), np.ones((3, 4), np.float32))


class TestControlFlowCapture:
    def test_scan_captures_outer_value(self):
        def fn(w, xs):
            carry, ys = olaf.scan(lambda c, x: (c + x * w, x * w), olaf.zeros(()), xs)
            return carry, ys

        traced = olaf.trace(fn, tt(), tt(3))
        verify(traced.graph)
        carry, ys = traced.eager(np.float32(2), np.array([1, 2, 3], np.float32))[0]
        assert carry == 12
        np.testing.assert_array_equal(ys, [2, 4, 6])

    def test_nested_capture_lifts_through_each_level(self):
        def fn(w):
            def outer(c, _):
                inner, _ = olaf.scan(lambda d, _: (d + w, None), c, None, length=2)
                return inner, None

            return olaf.scan(outer, olaf.zeros(()), None, length=3)[0]

        traced = olaf.trace(fn, tt())
        verify(traced.graph)
        assert traced.eager(np.float32(1))[0] == 6

    def test_repeat(self):
        traced = olaf.trace(lambda a: olaf.repeat(lambda c: c * 2.0, 3, a), tt())
        assert traced.eager(np.float32(1))[0] == 8
        assert sum(op.kind == "scan" for op in traced.graph.ops) == 1

    def test_repeat_with_per_layer_weights(self):
        traced = olaf.trace(lambda a, w: olaf.repeat(lambda c, x: c * x, 3, a, w), tt(), tt(3))
        assert traced.eager(np.float32(1), np.array([2, 3, 4], np.float32))[0] == 24

    def test_carry_structure_change_rejected(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: olaf.scan(lambda c, _: ((c, c), None), a, None, length=2), tt())

    def test_cond_with_different_captures(self):
        def fn(p, a, b):
            return olaf.cond(p, lambda x: x * a, lambda x: x + b, a)

        traced = olaf.trace(fn, tt(dtype=olaf.DType.bool), tt(), tt())
        verify(traced.graph)
        assert traced.eager(np.True_, np.float32(3), np.float32(5))[0] == 9
        assert traced.eager(np.False_, np.float32(3), np.float32(5))[0] == 8

    def test_while_with_captures(self):
        def fn(limit, step_size):
            return olaf.while_loop(lambda i: i < limit, lambda i: i + step_size, olaf.zeros(()))

        traced = olaf.trace(fn, tt(), tt())
        verify(traced.graph)
        assert traced.eager(np.float32(5), np.float32(2))[0] == 6


class TestStep:
    def test_sgd_on_state(self):
        decl = olaf.state("w", tt(2), StateKind.param)

        @olaf.step
        def train(x, y):
            w = decl.read()
            loss, g = olaf.value_and_grad(lambda p: olaf.mean((p * x - y) ** 2.0))(w)
            decl.write(w - 0.1 * g)
            return loss

        traced = train.trace(tt(2), tt(2))
        assert [d.name for d in traced.graph.states()] == ["w"]
        w = np.array([0.0, 1.0], np.float32)
        x, y = np.array([1.0, 2.0], np.float32), np.array([2.0, 0.0], np.float32)
        state = {"w": w}
        t = torch.tensor(w, requires_grad=True)
        opt = torch.optim.SGD([t], lr=0.1)
        for _ in range(5):
            loss, state = traced.eager(x, y, state=state)
            opt.zero_grad()
            ref = ((t * torch.tensor(x) - torch.tensor(y)) ** 2).mean()
            ref.backward()
            opt.step()
            assert rel(loss, ref.item()) < 1e-5
        assert rel(state["w"], t.detach().numpy()) < 1e-5
        np.testing.assert_array_equal(w, [0, 1])

    def test_has_aux(self):
        def fn(a):
            return olaf.value_and_grad(lambda x: (olaf.sum(x * x), x + 1.0), has_aux=True)(a)

        (loss, aux), g = olaf.trace(fn, tt(2)).eager(np.ones(2, np.float32))[0]
        assert loss == 2
        np.testing.assert_array_equal(aux, [2, 2])
        np.testing.assert_array_equal(g, [2, 2])

    def test_grad_of_pytree(self):
        def fn(tree):
            return olaf.grad(lambda t: olaf.sum(t["a"] * t["b"]))(tree)

        out = olaf.trace(fn, {"a": tt(2), "b": tt(2)}).eager({"a": np.ones(2, np.float32), "b": np.full(2, 3, np.float32)})[0]
        np.testing.assert_array_equal(out["a"], [3, 3])
        np.testing.assert_array_equal(out["b"], [1, 1])

    def test_rng_state_style(self):
        key_state = olaf.state("key", tt(2, dtype=olaf.u32), StateKind.rng)

        @olaf.step
        def draw():
            return olaf.rng_uniform(key_state.read(), olaf.constant(np.zeros(4, np.uint32)), (3,))

        out, _ = draw.trace().eager(state={"key": np.array([1, 2], np.uint32)})
        assert out.shape == (3,) and np.all((out >= 0) & (out < 1))


class TestStateRules:
    decl = olaf.state("s", tt())

    def test_read_after_write_rejected(self):
        def fn():
            a = self.decl.read()
            self.decl.write(a + 1.0)
            return self.decl.read()

        with pytest.raises(ValueError, match="after a write"):
            olaf.trace(fn)

    def test_second_write_rejected(self):
        def fn():
            self.decl.write(olaf.zeros(()))
            self.decl.write(olaf.ones(()))

        with pytest.raises(ValueError, match="after a write"):
            olaf.trace(fn)

    def test_state_in_cond_rejected(self):
        def fn(p):
            return olaf.cond(p > 0.0, lambda: self.decl.read(), lambda: olaf.zeros(()))

        with pytest.raises(ValueError, match="region"):
            olaf.trace(fn, tt())

    def test_reads_see_the_start_of_the_step(self):
        # Two reads, then a write: both reads return the old value whatever order ops run in.
        def fn():
            a = self.decl.read()
            b = self.decl.read() * 2.0
            self.decl.write(a + 1.0)
            return a, b

        (a, b), state = olaf.trace(fn).eager(state={"s": np.float32(3)})
        assert (a, b, state["s"]) == (3, 6, 4)

    def test_write_only_state_needs_no_initial_value(self):
        def init():
            self.decl.write(olaf.full((), 5.0))

        _, state = olaf.trace(init).eager()
        assert state["s"] == 5


class TestCompositesAgainstTorch:
    def _run(self, fn, *arrays):
        specs = [tt(*a.shape, dtype=olaf.i32 if a.dtype == np.int32 else F32) for a in arrays]
        return olaf.trace(fn, *specs).eager(*arrays)[0]

    def test_softmax(self):
        out = self._run(lambda a: olaf.softmax(a, -1), X)
        assert rel(out, F.softmax(torch.from_numpy(X), -1).numpy()) < 1e-5

    def test_softmax_survives_values_that_would_underflow(self):
        x = X - 500.0
        out = self._run(lambda a: olaf.softmax(a, -1), x)
        assert rel(out, F.softmax(torch.from_numpy(x), -1).numpy()) < 1e-5

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_gelu(self, approximate):
        out = self._run(lambda a: olaf.gelu(a, approximate), X)
        assert rel(out, F.gelu(torch.from_numpy(X), approximate=approximate).numpy()) < 1e-5

    def test_gelu_exact_uses_erf(self):
        graph = decompose(olaf.trace(lambda a: olaf.gelu(a), tt(2)).graph)
        assert "erf" in graph.text() and "tanh" not in graph.text()

    def test_layernorm(self):
        w, b = R.normal(size=4).astype(np.float32), R.normal(size=4).astype(np.float32)
        out = self._run(lambda a, u, v: olaf.layernorm(a, u, v), X, w, b)
        ref = F.layer_norm(torch.from_numpy(X), (4,), torch.from_numpy(w), torch.from_numpy(b), 1e-5)
        assert rel(out, ref.numpy()) < 1e-5

    def test_layernorm_symbolic_dims(self):
        B, D = olaf.sym("B"), olaf.sym("D")
        out = olaf.trace(lambda a: olaf.layernorm(a), tt(B, D)).eager(X)[0]
        assert rel(out, F.layer_norm(torch.from_numpy(X), (4,), eps=1e-5).numpy()) < 1e-5

    def test_cross_entropy(self):
        logits = R.normal(size=(2, 5, 7)).astype(np.float32) * 10 - 100
        labels = R.integers(0, 7, (2, 5)).astype(np.int32)
        out = self._run(olaf.cross_entropy, logits, labels)
        ref = F.cross_entropy(torch.from_numpy(logits).reshape(-1, 7), torch.from_numpy(labels).long().reshape(-1))
        assert rel(out, ref.item()) < 1e-5

    def test_attention(self):
        qq, k = R.normal(size=(2, 3, 4)).astype(np.float32), R.normal(size=(2, 5, 4)).astype(np.float32)
        v = R.normal(size=(2, 5, 6)).astype(np.float32)
        out = self._run(olaf.attention, qq, k, v)
        ref = F.scaled_dot_product_attention(*map(torch.from_numpy, (qq, k, v)))
        assert rel(out, ref.numpy()) < 1e-5

    def test_attention_mask(self):
        a = R.normal(size=(2, 4, 8)).astype(np.float32)
        mask = np.tril(np.ones((4, 4), bool))
        out = olaf.trace(lambda x: olaf.attention(x, x, x, olaf.iota((2, 4, 4), axis=1) >= olaf.iota((2, 4, 4), axis=2)), tt(2, 4, 8)).eager(a)[0]
        t = torch.from_numpy(a)
        ref = F.scaled_dot_product_attention(t, t, t, attn_mask=torch.from_numpy(mask))
        assert rel(out, ref.numpy()) < 1e-5


def run_graph(graph, *arrays):
    return interpret(graph, arrays)[0]


class TestCanonicalize:
    def _check(self, fn, *arrays, removes=()):
        specs = [tt(*a.shape) for a in arrays]
        graph = olaf.trace(fn, *specs).graph
        after = canonicalize(graph)
        verify(after)
        for x, y in zip(run_graph(graph, *arrays), run_graph(after, *arrays)):
            np.testing.assert_array_equal(x, y)
        kinds = [op.kind for op in after.ops]
        for kind in removes:
            assert kind not in kinds
        return after

    def test_reshape_chain_collapses(self):
        after = self._check(lambda a: a.reshape(4, 3).reshape(2, 6).reshape(12), X)
        assert [op.kind for op in after.ops] == ["reshape"]

    def test_reshape_identity_removed(self):
        self._check(lambda a: a.reshape(3, 4) + a, X, removes=("reshape",))

    def test_permute_inverse_removed(self):
        self._check(lambda a: olaf.permute(olaf.permute(a, (1, 0)), (1, 0)) * 2.0, X, removes=("permute",))

    def test_permute_composed(self):
        x = R.normal(size=(2, 3, 4)).astype(np.float32)
        after = self._check(lambda a: olaf.permute(olaf.permute(a, (1, 2, 0)), (1, 2, 0)), x)
        assert [op.kind for op in after.ops] == ["permute"]

    def test_slice_and_pad_identity_removed(self):
        self._check(lambda a: olaf.pad(a[:, :], (0, 0), (0, 0)) * 2.0, X, removes=("slice", "pad"))

    def test_real_slice_kept(self):
        after = self._check(lambda a: a[1:, :], X)
        assert [op.kind for op in after.ops] == ["slice"]

    def test_inside_scan_region(self):
        def fn(a):
            return olaf.scan(lambda c, _: (c.reshape(12).reshape(3, 4), None), a, None, length=2)[0]

        graph = olaf.trace(fn, tt(3, 4)).graph
        after = canonicalize(graph)
        verify(after)
        assert [op.kind for op in after.ops[0].regions[0].ops] == []
        np.testing.assert_array_equal(run_graph(after, X)[0], X)

    def test_result_substituted(self):
        b = GraphBuilder()
        x = b.param(tt(2))
        y = b.op("permute", x, perm=(0,))
        after = canonicalize(b.finish([y]))
        verify(after)
        assert after.results[0] is x

    def test_after_autodiff_values_match(self):
        def fn(a):
            return olaf.value_and_grad(lambda x: olaf.sum(olaf.softmax(x.reshape(4, 3).reshape(3, 4)) ** 2.0))(a)

        # Autodiff keeps composites (their VJPs emit softmax too), so decompose before interpreting.
        graph = decompose(olaf.trace(fn, tt(3, 4)).graph)
        after = canonicalize(graph)
        verify(after)
        for x, y in zip(run_graph(graph, X), run_graph(after, X)):
            assert rel(x, y) < 1e-5


class TestDecompose:
    def test_removes_all_composites_including_in_regions(self):
        def fn(a):
            return olaf.scan(lambda c, _: (olaf.softmax(c) + olaf.gelu(c), None), a, None, length=2)[0]

        graph = olaf.trace(fn, tt(3)).graph
        assert "softmax" in graph.text()
        out = decompose(graph)
        verify(out)
        assert "softmax" not in out.text() and "gelu" not in out.text()

    def test_interpreter_rejects_composite(self):
        graph = olaf.trace(olaf.softmax, tt(3)).graph
        with pytest.raises(ValueError):
            interpret(graph, [np.zeros(3, np.float32)])


class TestCapability:
    def test_digest_differs_by_hardware_and_is_stable(self):
        assert AMPERE_RTX3060.digest() != CPU.digest()
        assert AMPERE_RTX3060.digest() == AMPERE_RTX3060.digest()

    def test_ampere_has_no_hopper_features(self):
        assert AMPERE_RTX3060.async_copy is AsyncCopy.software
        assert not AMPERE_RTX3060.cooperation.clusters
        assert not AMPERE_RTX3060.on_chip.dedicated_matrix_accumulator

    def test_matrix_shapes_filter_by_dtype(self):
        assert {i.k for i in AMPERE_RTX3060.matrix_shapes("bf16")} == {8, 16}
        assert AMPERE_RTX3060.matrix_shapes("fp8") == ()

    def test_megakernel_needs_forward_progress(self):
        assert not CPU.supports_megakernel
        assert not AMPERE_RTX3060.supports_megakernel
