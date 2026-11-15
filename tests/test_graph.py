import numpy as np
import pytest
import torch
from scipy.special import erf as scipy_erf

from olaf.graph.builder import GraphBuilder, dead_code_elimination, inline, verify
from olaf.graph.dim import as_symbol, evaluate, product, sym
from olaf.graph.interp import interpret, philox4x32
from olaf.graph.ir import ConstantData, Graph, bf16_bits, bf16_values
from olaf.graph.types import DType, StateDecl, StateKind, TensorType

F32 = DType.f32


def tt(*shape, dtype=F32):
    return TensorType(tuple(shape), dtype)


def run_op(kind, arrays, types=None, **attrs):
    b = GraphBuilder()
    names = {"float32": DType.f32, "float64": DType.f64, "int32": DType.i32, "bool": DType.bool}
    types = types or [tt(*a.shape, dtype=names[a.dtype.name]) for a in arrays]
    params = [b.param(t) for t in types]
    out = b.emit(kind, params, attrs)
    results, _ = interpret(b.finish(out), arrays)
    return results


class TestDim:
    def test_normal_form_compares_structurally(self):
        s, b = sym("S"), sym("B")
        assert s + b == b + s
        assert s * b + 1 == 1 + b * s
        assert (s + 1) * 2 == 2 * s + 2
        assert s - s == 0
        assert isinstance(s - s, int)

    def test_division_simplifies_only_when_exact(self):
        s = sym("S")
        assert (s * 4) // 2 == s * 2
        assert s // 2 != s
        assert evaluate(s // 2, {"S": 7}) == 3
        assert evaluate((s + 1) % 3, {"S": 7}) == 2

    def test_constraint_simplifies_modulo(self):
        s = sym("S", multiple_of=128)
        assert s % 128 == 0
        assert s % 64 == 0
        assert s % 256 != 0

    def test_as_symbol(self):
        s = sym("S")
        assert as_symbol(s).name == "S"
        assert as_symbol(s + 1) is None
        assert as_symbol(s * 2) is None

    def test_product_with_mixed_dims(self):
        assert product([sym("B"), 2, sym("S")]) == 2 * sym("B") * sym("S")


class TestText:
    def _graph(self):
        b = GraphBuilder()
        x = b.param(tt(sym("B"), 3))
        y = b.op("add", x, x)
        z = b.op("reduce", y, fn="sum", axes=(0, 1))
        return b.finish([z])

    def test_text_is_independent_of_object_identity(self):
        assert self._graph().text() == self._graph().text()
        assert self._graph().digest() == self._graph().digest()

    def test_text_format(self):
        text = self._graph().text()
        assert "graph @main(%0: f32[B,3])" in text
        assert '%2 = reduce %1 {axes=[0,1], fn="sum"} : f32[]' in text

    def test_symbol_constraints_are_in_the_text(self):
        def build(s):
            b = GraphBuilder()
            return b.finish([b.param(tt(s, 3))]).text()

        plain, constrained = build(sym("S")), build(sym("S", multiple_of=128, max=4096))
        assert plain != constrained
        assert constrained.startswith("symbols S{multiple_of=128, max=4096}\n")

    def test_conflicting_symbol_constraints_rejected(self):
        b = GraphBuilder()
        graph = b.finish([b.param(tt(sym("S"))), b.param(tt(sym("S", multiple_of=2)))])
        with pytest.raises(ValueError, match="conflicting constraints"):
            graph.text()

    def test_attr_order_does_not_change_text(self):
        def build(**attrs):
            b = GraphBuilder()
            x = b.param(tt(2, 3))
            return b.finish(b.emit("reduce", [x], attrs)).text()

        assert build(fn="sum", axes=(0,)) == build(axes=(0,), fn="sum")

    def test_large_constant_prints_digest_only(self):
        b = GraphBuilder()
        c = b.constant(np.zeros((100, 100), np.float32), F32)
        text = b.finish([c]).text()
        assert "dense<" in text and len(text) < 200

    def test_different_constants_have_different_text(self):
        def build(v):
            b = GraphBuilder()
            return b.finish([b.constant(np.full((2, 2), v, np.float32), F32)]).text()

        assert build(1.0) != build(2.0)


class TestTypeChecks:
    @pytest.mark.parametrize(
        "kind,types,attrs",
        [
            ("add", [tt(2), tt(3)], {}),
            ("add", [tt(2), tt(2, dtype=DType.f64)], {}),
            ("exp", [tt(2, dtype=DType.i32)], {}),
            ("div", [tt(2, dtype=DType.i32), tt(2, dtype=DType.i32)], {}),
            ("reduce", [tt(2)], {"fn": "sum", "axes": (1,)}),
            ("reduce", [tt(2)], {"fn": "prod", "axes": (0,)}),
            ("reshape", [tt(2, 3)], {"shape": (5,)}),
            ("permute", [tt(2, 3)], {"perm": (0, 0)}),
            ("broadcast", [tt(2)], {"shape": (3,), "dims": (0,)}),
            ("contract", [tt(2, 3), tt(4, 5)], {"spec": "ab,bc->ac"}),
            ("contract", [tt(2, 2)], {"spec": "aa->a"}),
            ("gather", [tt(2, 3), tt(2)], {"axis": 0, "batch_dims": 0}),
            ("where", [tt(2), tt(2), tt(2)], {}),
        ],
    )
    def test_rejects(self, kind, types, attrs):
        b = GraphBuilder()
        with pytest.raises(TypeError):
            b.emit(kind, [b.param(t) for t in types], attrs)

    def test_unknown_op_rejected(self):
        with pytest.raises(ValueError):
            GraphBuilder().emit("not_an_op")

    def test_symbolic_reshape_checks_product(self):
        b = GraphBuilder()
        B, S = sym("B"), sym("S")
        x = b.param(tt(B, S, 4))
        y = b.op("reshape", x, shape=(B, S, 2, 2))
        assert y.type.shape == (B, S, 2, 2)
        with pytest.raises(TypeError):
            b.op("reshape", x, shape=(S, 2, 2))

    def test_communication_ops_have_types(self):
        b = GraphBuilder()
        x = b.param(tt(8, 4))
        assert b.op("all_reduce", x, fn="sum", mesh_axis="dp").type.shape == (8, 4)
        assert b.op("all_gather", x, axis=1, group_size=2, mesh_axis="tp").type.shape == (8, 8)
        assert b.op("reduce_scatter", x, axis=0, group_size=2, mesh_axis="tp").type.shape == (4, 4)
        assert b.op("all_to_all", x, split_axis=0, concat_axis=1, group_size=2, mesh_axis="ep").type.shape == (4, 8)
        assert b.op("recv", type=tt(3), peer=1).type == tt(3)
        b.emit("send", [x], {"peer": 1})
        with pytest.raises(NotImplementedError):
            interpret(b.finish([]), [np.zeros((8, 4), np.float32)])


class TestVerify:
    def test_rejects_value_from_outside_region(self):
        outer = GraphBuilder()
        v = outer.param(tt())
        inner = GraphBuilder()
        p = inner.param(tt())
        inner.op("add", p, v)
        body = inner.finish([p])
        with pytest.raises(ValueError):
            verify(body)

    def test_rejects_state_in_scan(self):
        decl = StateDecl("s", tt())
        body = GraphBuilder()
        c = body.param(tt())
        s = body.op("read_state", state=decl)
        region = body.finish([body.op("add", c, s)])
        b = GraphBuilder()
        c0 = b.param(tt())
        out = b.emit("scan", [c0], {"n_carry": 1, "length": 2}, [region])
        with pytest.raises(ValueError):
            verify(b.finish(out))

    def test_rejects_second_definition(self):
        b = GraphBuilder()
        x = b.param(tt())
        y = b.op("exp", x)
        graph = b.finish([y])
        twice = Graph(graph.params, (*graph.ops, graph.ops[0]), graph.results)
        with pytest.raises(ValueError, match="already defined"):
            verify(twice)
        with pytest.raises(ValueError, match="twice among the params"):
            verify(Graph((x, x), graph.ops, graph.results))

    def test_dce_keeps_effects(self):
        decl = StateDecl("s", tt(), StateKind.optim)
        b = GraphBuilder()
        x = b.param(tt())
        b.op("add", x, x)
        kept = b.op("exp", x)
        b.emit("write_state", [kept], {"state": decl})
        graph = dead_code_elimination(b.finish([x]))
        assert [op.kind for op in graph.ops] == ["exp", "write_state"]


RNG = np.random.default_rng(0)
A = RNG.normal(size=(3, 4)).astype(np.float32)
B_ = RNG.normal(size=(3, 4)).astype(np.float32)
POS = np.abs(A) + 0.1


class TestEagerAgainstTorch:
    @pytest.mark.parametrize(
        "kind,fn,x",
        [
            ("neg", torch.neg, A),
            ("abs", torch.abs, A),
            ("exp", torch.exp, A),
            ("log", torch.log, POS),
            ("sqrt", torch.sqrt, POS),
            ("rsqrt", torch.rsqrt, POS),
            ("recip", torch.reciprocal, POS),
            ("tanh", torch.tanh, A),
            ("erf", torch.erf, A),
            ("sin", torch.sin, A),
            ("cos", torch.cos, A),
        ],
    )
    def test_unary(self, kind, fn, x):
        out = run_op(kind, [x])[0]
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, fn(torch.from_numpy(x)).numpy(), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "kind,fn",
        [("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div), ("max", torch.maximum), ("min", torch.minimum)],
    )
    def test_binary(self, kind, fn):
        out = run_op(kind, [A, B_])[0]
        np.testing.assert_allclose(out, fn(torch.from_numpy(A), torch.from_numpy(B_)).numpy(), rtol=1e-5, atol=1e-6)

    def test_pow(self):
        out = run_op("pow", [POS, B_])[0]
        np.testing.assert_allclose(out, torch.pow(torch.from_numpy(POS), torch.from_numpy(B_)).numpy(), rtol=1e-5)

    def test_erf_matches_scipy_in_f64(self):
        x = A.astype(np.float64)
        out = run_op("erf", [x], [tt(3, 4, dtype=DType.f64)])[0]
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, scipy_erf(x))

    def test_contract(self):
        y = RNG.normal(size=(4, 5)).astype(np.float32)
        out = run_op("contract", [A, y], spec="ab,bc->ac")[0]
        np.testing.assert_allclose(out, A @ y, rtol=1e-5, atol=1e-6)

    def test_gather_as_embedding(self):
        table = RNG.normal(size=(5, 2)).astype(np.float32)
        idx = np.array([[0, 4, 1], [2, 2, 3]], dtype=np.int32)
        out = run_op("gather", [table, idx], axis=0, batch_dims=0)[0]
        np.testing.assert_array_equal(out, table[idx])

    def test_gather_with_batch_dims_matches_take_along_axis(self):
        logits = RNG.normal(size=(2, 3, 5)).astype(np.float32)
        labels = RNG.integers(0, 5, (2, 3)).astype(np.int32)
        out = run_op("gather", [logits, labels], axis=2, batch_dims=2)[0]
        expected = torch.gather(torch.from_numpy(logits), 2, torch.from_numpy(labels).long()[..., None])[..., 0]
        np.testing.assert_array_equal(out, expected.numpy())

    def test_gather_middle_axis(self):
        x = RNG.normal(size=(2, 5, 3)).astype(np.float32)
        idx = RNG.integers(0, 5, (4, 6)).astype(np.int32)
        out = run_op("gather", [x, idx], axis=1, batch_dims=0)[0]
        np.testing.assert_array_equal(out, x[:, idx, :])

    def test_gather_rejects_negative_index(self):
        with pytest.raises(IndexError):
            run_op("gather", [A, np.array([-1], dtype=np.int32)], axis=0, batch_dims=0)

    def test_scatter_add_accumulates_duplicates(self):
        target = np.zeros((4, 2), np.float32)
        idx = np.array([1, 1, 3], dtype=np.int32)
        updates = np.ones((3, 2), np.float32)
        out = run_op("scatter_add", [target, idx, updates], axis=0, batch_dims=0)[0]
        np.testing.assert_array_equal(out[:, 0], [0, 2, 0, 1])
        assert target.sum() == 0

    def test_views(self):
        np.testing.assert_array_equal(run_op("reshape", [A], shape=(4, 3))[0], A.reshape(4, 3))
        np.testing.assert_array_equal(run_op("permute", [A], perm=(1, 0))[0], A.T)
        np.testing.assert_array_equal(run_op("slice", [A], starts=(1, 0), stops=(3, 2))[0], A[1:3, 0:2])
        np.testing.assert_array_equal(run_op("concat", [A, B_], axis=1)[0], np.concatenate([A, B_], 1))
        padded = run_op("pad", [A], low=(1, 0), high=(0, 2), value=7.0)[0]
        np.testing.assert_array_equal(padded, np.pad(A, ((1, 0), (0, 2)), constant_values=7.0))

    def test_broadcast_inserts_and_expands(self):
        x = RNG.normal(size=(3, 1)).astype(np.float32)
        out = run_op("broadcast", [x], shape=(2, 3, 4), dims=(1, 2))[0]
        np.testing.assert_array_equal(out, np.broadcast_to(x, (2, 3, 4)))

    def test_reduce_keeps_integer_dtype(self):
        x = np.arange(6, dtype=np.int32).reshape(2, 3)
        out = run_op("reduce", [x], fn="sum", axes=(1,))[0]
        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, [3, 12])

    def test_iota(self):
        b = GraphBuilder()
        out = b.op("iota", shape=(2, 3), axis=1, dtype=DType.i32)
        result = interpret(b.finish([out]), [])[0][0]
        np.testing.assert_array_equal(result, [[0, 1, 2], [0, 1, 2]])

    def test_mismatched_rank_rejected(self):
        b = GraphBuilder()
        x = b.param(tt(2, 3))
        with pytest.raises(ValueError):
            interpret(b.finish([x]), [np.zeros((2,), np.float32)])


class TestSymbolicExecution:
    def test_symbols_bound_from_inputs_and_checked(self):
        S = sym("S", multiple_of=2)
        b = GraphBuilder()
        x = b.param(tt(S))
        y = b.op("reshape", x, shape=(S // 2, 2))
        graph = b.finish([y])
        out = interpret(graph, [np.zeros(6, np.float32)])[0][0]
        assert out.shape == (3, 2)
        with pytest.raises(ValueError):
            interpret(graph, [np.zeros(5, np.float32)])

    def test_inconsistent_symbol_rejected(self):
        S = sym("S")
        b = GraphBuilder()
        x, y = b.param(tt(S)), b.param(tt(S))
        graph = b.finish([b.op("add", x, y)])
        with pytest.raises(ValueError):
            interpret(graph, [np.zeros(2, np.float32), np.zeros(3, np.float32)])

    def test_caller_dims_are_checked_against_constraints(self):
        S = sym("S", multiple_of=8)
        b = GraphBuilder()
        graph = b.finish([b.op("reduce", b.full((S,), 1.0, F32), fn="sum", axes=(0,))])
        assert interpret(graph, [], dims={"S": 16})[0][0] == 16
        with pytest.raises(ValueError):
            interpret(graph, [], dims={"S": 7})

    def test_unknown_dim_name_rejected(self):
        b = GraphBuilder()
        x = b.param(tt(sym("S")))
        with pytest.raises(ValueError):
            interpret(b.finish([x]), [np.zeros(2, np.float32)], dims={"T": 2})


class TestBf16Constants:
    def test_rounding_matches_torch_for_f32_inputs(self):
        # torch converts through f32, so it is a valid reference only for inputs that are already f32.
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(20000) * np.exp2(rng.integers(-140, 126, 20000))).astype(np.float32)
        x = np.concatenate([x, np.array([0.0, -0.0, np.inf, -np.inf, 1e-45, 3.4e38], np.float32)])
        expected = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(bf16_values(bf16_bits(x)), expected)

    def test_rounds_f64_once(self):
        # Just above the midpoint between 1 and 1 + 2^-7. Through f32 it becomes the exact midpoint and
        # then ties to even (1.0); a single rounding goes up.
        assert bf16_values(bf16_bits(1 + 2**-8 + 2**-30)) == 1 + 2**-7

    def test_nan_stays_nan(self):
        assert np.isnan(bf16_values(bf16_bits(np.nan)))

    def test_constant_is_traceable_and_printed_as_its_value(self):
        data = ConstantData.of(0.1, DType.bf16)
        assert data.array.dtype == np.uint16
        assert str(data) == repr(float(bf16_values(data.array)))
        assert ConstantData.of(0.1, DType.bf16) == ConstantData.of(0.1000001, DType.bf16)


class TestState:
    def test_read_write(self):
        decl = StateDecl("w", tt(2))
        b = GraphBuilder()
        x = b.param(tt(2))
        w = b.op("read_state", state=decl)
        updated = b.op("add", w, x)
        b.emit("write_state", [updated], {"state": decl})
        graph = b.finish([w])
        original = np.ones(2, np.float32)
        results, state = interpret(graph, [np.ones(2, np.float32)], {"w": original})
        np.testing.assert_array_equal(results[0], [1, 1])
        np.testing.assert_array_equal(state["w"], [2, 2])
        np.testing.assert_array_equal(original, [1, 1])
        assert graph.states() == (decl,)

    def test_missing_state_rejected(self):
        b = GraphBuilder()
        v = b.op("read_state", state=StateDecl("w", tt()))
        with pytest.raises(ValueError):
            interpret(b.finish([v]), [])


class TestControlFlow:
    def _scan_graph(self, reverse=False):
        body = GraphBuilder()
        c, x = body.param(tt()), body.param(tt())
        s = body.op("add", c, x)
        region = body.finish([s, body.op("mul", s, s)])
        b = GraphBuilder()
        init, xs = b.param(tt()), b.param(tt(3))
        return b.finish(b.emit("scan", [init, xs], {"n_carry": 1, "length": 3, "reverse": reverse}, [region]))

    def test_scan(self):
        carry, ys = interpret(self._scan_graph(), [np.float32(0), np.array([1, 2, 3], np.float32)])[0]
        assert carry == 6
        np.testing.assert_array_equal(ys, [1, 9, 36])

    def test_scan_reverse_keeps_positions(self):
        carry, ys = interpret(self._scan_graph(True), [np.float32(0), np.array([1, 2, 3], np.float32)])[0]
        assert carry == 6
        np.testing.assert_array_equal(ys, [36, 25, 9])

    def test_scan_zero_length_has_declared_shape(self):
        body = GraphBuilder()
        c = body.param(tt())
        region = body.finish([c, c])
        b = GraphBuilder()
        init = b.param(tt())
        graph = b.finish(b.emit("scan", [init], {"n_carry": 1, "length": sym("N")}, [region]))
        carry, ys = interpret(graph, [np.float32(5)], dims={"N": 0})[0]
        assert carry == 5 and ys.shape == (0,)

    def test_cond(self):
        then_b = GraphBuilder()
        a = then_b.param(tt())
        then_g = then_b.finish([then_b.op("neg", a)])
        else_b = GraphBuilder()
        c = else_b.param(tt())
        else_g = else_b.finish([else_b.op("exp", c)])
        b = GraphBuilder()
        p, x = b.param(tt(dtype=DType.bool)), b.param(tt())
        graph = b.finish(b.emit("cond", [p, x, x], {"n_then": 1}, [then_g, else_g]))
        assert interpret(graph, [np.True_, np.float32(2)])[0][0] == -2
        np.testing.assert_allclose(interpret(graph, [np.False_, np.float32(0)])[0][0], 1)

    def test_while(self):
        cond_b = GraphBuilder()
        i = cond_b.param(tt(dtype=DType.i32))
        cond_g = cond_b.finish([cond_b.op("cmp", i, cond_b.constant(5, DType.i32), fn="lt")])
        body_b = GraphBuilder()
        j = body_b.param(tt(dtype=DType.i32))
        body_g = body_b.finish([body_b.op("add", j, body_b.constant(2, DType.i32))])
        b = GraphBuilder()
        start = b.param(tt(dtype=DType.i32))
        graph = b.finish(b.emit("while", [start], {}, [cond_g, body_g]))
        assert interpret(graph, [np.int32(0)])[0][0] == 6


class TestRng:
    # Known answers from Random123 (philox4x32 10 in kat_vectors).
    @pytest.mark.parametrize(
        "counter,key,expected",
        [
            ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
            ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
            ([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0], [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]),
        ],
    )
    def test_philox_known_answers(self, counter, key, expected):
        out = philox4x32(np.array(counter, np.uint32), np.array(key, np.uint32))
        assert out.tolist() == expected

    def _sample(self, shape, counter=(0, 0, 0, 0)):
        b = GraphBuilder()
        k, c = b.param(tt(2, dtype=DType.u32)), b.param(tt(4, dtype=DType.u32))
        graph = b.finish([b.op("rng", k, c, shape=shape, dtype=F32)])
        return interpret(graph, [np.array([1, 2], np.uint32), np.array(counter, np.uint32)])[0][0]

    def test_uniform_range_and_mean(self):
        x = self._sample((10000,))
        assert x.dtype == np.float32 and x.min() >= 0 and x.max() < 1
        assert abs(x.mean() - 0.5) < 0.02

    def test_value_depends_only_on_position(self):
        np.testing.assert_array_equal(self._sample((6,)), self._sample((20,))[:6])

    def test_counter_advances_by_blocks(self):
        np.testing.assert_array_equal(self._sample((8,))[4:], self._sample((4,), (1, 0, 0, 0)))

    def test_counter_carries_across_lanes(self):
        np.testing.assert_array_equal(self._sample((8,), (0xFFFFFFFF, 0, 0, 0))[4:], self._sample((4,), (0, 1, 0, 0)))


class TestObserverAndInline:
    def test_observer_sees_intermediate_values(self):
        b = GraphBuilder()
        x = b.param(tt())
        y = b.op("exp", x)
        z = b.op("neg", y)
        seen = {}
        interpret(b.finish([z]), [np.float32(0)], observer=lambda v, a: seen.__setitem__(v, a))
        assert seen[y] == 1 and seen[z] == -1

    def test_inline_can_be_repeated(self):
        inner = GraphBuilder()
        p = inner.param(tt())
        g = inner.finish([inner.op("add", p, p)])
        b = GraphBuilder()
        x = b.param(tt())
        y = inline(b, g, [x])[0]
        z = inline(b, g, [y])[0]
        graph = b.finish([z])
        verify(graph)
        assert interpret(graph, [np.float32(1)])[0][0] == 4
        assert isinstance(graph, Graph)
