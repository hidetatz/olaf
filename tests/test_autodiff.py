import numpy as np
import pytest
import torch

import olaf
from olaf.frontend.autodiff import grad as graph_grad
from olaf.graph.builder import GraphBuilder, verify
from olaf.graph.interp import interpret
from olaf.graph.types import TensorType
from olaf.passes.decompose import decompose

F64 = olaf.f64


def tt(*shape, dtype=F64):
    return TensorType(tuple(shape), dtype)


def rel(a, b):
    return np.linalg.norm(np.ravel(a) - np.ravel(b)) / max(np.linalg.norm(np.ravel(b)), 1e-300)


def check_numeric(fn, *arrays, tol=1e-6, eps=1e-6):
    """Compare with f64 central differences. fn returns a scalar and is differentiated in all arguments."""
    arrays = [np.asarray(a, np.float64) for a in arrays]
    specs = [tt(*a.shape) for a in arrays]
    wrt = tuple(range(len(arrays)))
    traced = olaf.trace(lambda *xs: olaf.value_and_grad(fn, wrt)(*xs), *specs)
    verify(traced.graph)
    (value, grads), _ = traced.eager(*arrays)

    plain = olaf.trace(fn, *specs)
    for index, array in enumerate(arrays):
        numeric = np.zeros_like(array)
        for pos in np.ndindex(array.shape):
            shifted = [a.copy() for a in arrays]
            shifted[index][pos] += eps
            up = plain.eager(*shifted)[0]
            shifted[index][pos] -= 2 * eps
            down = plain.eager(*shifted)[0]
            numeric[pos] = (up - down) / (2 * eps)
        assert rel(grads[index], numeric) < tol, f"argument {index}: {grads[index]} / {numeric}"
    np.testing.assert_allclose(value, plain.eager(*arrays)[0])


R = np.random.default_rng(1)
X = R.normal(size=(3, 4))
Y = R.normal(size=(3, 4))
P = np.abs(X) + 0.5
Q_ = np.abs(Y) + 0.5


@pytest.mark.parametrize(
    "fn,args",
    [
        (lambda x: olaf.sum(-x * x), [X]),
        (lambda x: olaf.sum(olaf.abs(x) * x), [X]),
        (lambda x: olaf.sum(olaf.exp(x)), [X]),
        (lambda x: olaf.sum(olaf.log(x)), [P]),
        (lambda x: olaf.sum(olaf.sqrt(x)), [P]),
        (lambda x: olaf.sum(olaf.rsqrt(x)), [P]),
        (lambda x: olaf.sum(olaf.recip(x)), [P]),
        (lambda x: olaf.sum(olaf.tanh(x)), [X]),
        (lambda x: olaf.sum(olaf.erf(x)), [X]),
        (lambda x: olaf.sum(olaf.sin(x)), [X]),
        (lambda x: olaf.sum(olaf.cos(x)), [X]),
        (lambda x, y: olaf.sum((x + y) * (x - y)), [X, Y]),
        (lambda x, y: olaf.sum(x / y), [X, Q_]),
        (lambda x, y: olaf.sum(x**y), [P, Y]),
        (lambda x, y: olaf.sum(olaf.maximum(x, y) * x), [X, Y]),
        (lambda x, y: olaf.sum(olaf.minimum(x, y) * y), [X, Y]),
        (lambda x, y: olaf.sum(olaf.where(x < y, x * x, y * x)), [X, Y]),
        (lambda x: olaf.sum(olaf.max(x, 1) ** 2.0), [X]),
        (lambda x: olaf.max(x), [X]),
        (lambda x, y: olaf.sum(olaf.einsum("ab,cb->ac", x, y) ** 2.0), [X, Y]),
        (lambda x, y: olaf.sum(olaf.einsum("ab,ab->a", x, y) ** 2.0), [X, Y]),
        (lambda x: olaf.einsum("ab->", x), [X]),
        (lambda x, y: olaf.einsum("ab,cd->", x, y) ** 2.0, [X, Y]),
        (lambda x, y, z: olaf.sum(olaf.einsum("ab,bc,cd->ad", x, y, z) ** 2.0), [X, Y.T, R.normal(size=(3, 2))]),
        (lambda x: olaf.sum(olaf.reshape(x, (2, 6)) ** 2.0 * olaf.reshape(olaf.iota((12,), dtype=F64), (2, 6))), [X]),
        (lambda x: olaf.sum(olaf.permute(x, (1, 0)) * olaf.iota((4, 3), axis=0, dtype=F64) * x.reshape(4, 3)), [X]),
        (lambda x: olaf.sum(olaf.broadcast_to(x, (2, 3, 4)) ** 2.0), [X]),
        (lambda x: olaf.sum(olaf.broadcast_to(x[:, :1], (3, 4)) ** 3.0), [X]),
        (lambda x: olaf.sum(x[1:3, 1:] ** 2.0), [X]),
        (lambda x: x[1, 2] ** 2.0, [X]),
        (lambda x, y: olaf.sum(olaf.concat([x, y * y], 1) ** 2.0), [X, Y]),
        (lambda x: olaf.sum(olaf.pad(x, (1, 0), (0, 2), 3.0) ** 2.0), [X]),
        (lambda x: olaf.sum(olaf.softmax(x, -1) ** 2.0), [X]),
        (lambda x: olaf.sum(olaf.softmax(x, 0) ** 2.0), [X]),
        (lambda x: olaf.sum(olaf.gelu(x) ** 2.0), [X]),
        (lambda x: olaf.sum(olaf.gelu(x, "tanh") ** 2.0), [X]),
        (lambda x, w, b: olaf.sum(olaf.layernorm(x, w, b) ** 2.0), [X, Y[0], R.normal(size=4)]),
        (lambda x: olaf.sum(olaf.layernorm(x, axes=(0, 1)) ** 3.0), [X]),
    ],
)
def test_matches_finite_difference(fn, args):
    check_numeric(fn, *args, tol=1e-5)


def test_cast_between_floats():
    # Going through f32 breaks finite differences through rounding, so compare with the analytic answer.
    x = np.array([1.5, -2.25])
    traced = olaf.trace(lambda a: olaf.grad(lambda y: olaf.sum(olaf.cast(olaf.cast(y, olaf.f32), F64) ** 2.0))(a), tt(2))
    np.testing.assert_array_equal(traced.eager(x)[0], 2 * x)


def test_gather_and_scatter_add():
    index_values = np.array([1, 1, 0, 2], np.int32)

    def fn(x, u):
        index = olaf.constant(index_values)
        picked = olaf.take(x, index, axis=0)
        spread = olaf.scatter_add(x * x, index, u * u, axis=0)
        return olaf.sum(picked**2.0) + olaf.sum(spread**2.0)

    check_numeric(fn, X, R.normal(size=(4, 4)), tol=1e-5)


def test_gather_with_batch_dims():
    labels = np.array([[0, 3], [2, 1], [1, 1]], np.int32)

    def fn(x):
        return olaf.sum(olaf.take(x.reshape(3, 2, 2) * 1.0, olaf.constant(labels[:, :2] % 2), axis=2, batch_dims=2) ** 2.0)

    check_numeric(fn, X, tol=1e-5)


def test_cross_entropy():
    labels = np.array([0, 3, 1], np.int32)
    check_numeric(lambda x: olaf.cross_entropy(x, olaf.constant(labels)), X, tol=1e-5)


def test_attention_with_mask():
    def fn(qq, k, v):
        mask = olaf.iota((3, 3), axis=0) >= olaf.iota((3, 3), axis=1)
        return olaf.sum(olaf.attention(qq, k, v, mask) ** 2.0)

    check_numeric(fn, R.normal(size=(3, 4)), R.normal(size=(3, 4)), R.normal(size=(3, 2)), tol=1e-5)


def test_attention_with_fully_masked_rows():
    # Query rows 0 and 1 see no key (left padding of two under a causal mask). Their output is
    # defined as 0, so it is constant in q, k and v, and the gradients must be finite and match
    # finite differences instead of being NaN.
    def fn(qq, k, v):
        mask = np.tril(np.ones((4, 4), bool)) & (np.arange(4) >= 2)[None, :]
        return olaf.sum(olaf.attention(qq, k, v, olaf.constant(mask)) ** 2.0)

    check_numeric(fn, R.normal(size=(4, 3)), R.normal(size=(4, 3)), R.normal(size=(4, 2)), tol=1e-5)


class TestAgainstTorchSubgradients:
    def _grad(self, fn, x):
        traced = olaf.trace(lambda a: olaf.grad(fn)(a), tt(*x.shape))
        return traced.eager(np.asarray(x, np.float64))[0]

    def test_abs_at_zero(self):
        x = np.array([-1.0, 0.0, 2.0])
        t = torch.tensor(x, requires_grad=True)
        torch.abs(t).sum().backward()
        np.testing.assert_array_equal(self._grad(lambda a: olaf.sum(olaf.abs(a)), x), t.grad.numpy())

    def test_reduce_max_ties_split_evenly(self):
        x = np.array([1.0, 3.0, 3.0])
        t = torch.tensor(x, requires_grad=True)
        torch.amax(t).backward()
        np.testing.assert_array_equal(self._grad(lambda a: olaf.max(a), x), t.grad.numpy())

    def test_binary_max_ties_split_evenly(self):
        x = np.array([1.0, 2.0, 3.0])
        t = torch.tensor(x, requires_grad=True)
        torch.maximum(t, torch.tensor(2.0, dtype=t.dtype)).sum().backward()
        result = self._grad(lambda a: olaf.sum(olaf.maximum(a, olaf.full((3,), 2.0, F64))), x)
        np.testing.assert_array_equal(result, t.grad.numpy())

    def test_pow_at_zero_base_stays_finite(self):
        def fn(y):
            return olaf.sum(olaf.full((2,), 0.0, F64) ** y)

        result = self._grad(fn, np.array([2.0, 3.0]))
        assert np.all(np.isfinite(result))


class TestActivity:
    def test_unused_input_gets_zero(self):
        traced = olaf.trace(lambda a, b: olaf.value_and_grad(lambda x, y: olaf.sum(x * x), (0, 1))(a, b), tt(2), tt(2))
        (_, (_, gy)), _ = traced.eager(np.ones(2), np.ones(2))
        np.testing.assert_array_equal(gy, [0, 0])

    def test_integer_path_blocks_gradient(self):
        def fn(x):
            return olaf.sum(olaf.cast(olaf.cast(x, olaf.i32), F64) * x)

        traced = olaf.trace(lambda a: olaf.grad(fn)(a), tt(2))
        np.testing.assert_array_equal(traced.eager(np.array([2.5, 3.5]))[0], [2, 3])

    def test_rejects_non_scalar_loss(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: olaf.grad(lambda x: x * x)(a), tt(2))

    def test_rejects_integer_wrt(self):
        with pytest.raises(TypeError):
            olaf.trace(lambda a: olaf.grad(lambda x: olaf.sum(olaf.cast(x, F64)))(a), tt(2, dtype=olaf.i32))


class TestScan:
    def test_carry_only(self):
        def fn(x):
            carry, _ = olaf.scan(lambda c, _: (olaf.sin(c) * x[1], None), olaf.ones((), F64) * x[0], None, length=4)
            return carry

        check_numeric(fn, np.array([0.7, 1.3]))

    def test_with_xs_and_ys(self):
        def fn(w, xs):
            def body(c, x):
                c2 = olaf.tanh(c * x + w)
                return c2, c2 * x

            carry, ys = olaf.scan(body, olaf.zeros((), F64), xs)
            return carry + olaf.sum(ys**2.0)

        check_numeric(fn, np.array(0.4), np.array([0.3, -1.2, 0.8]))

    def test_reverse(self):
        def fn(xs):
            carry, ys = olaf.scan(lambda c, x: (c * x + 1.0, c), olaf.ones((), F64), xs, reverse=True)
            return carry * olaf.sum(ys * olaf.iota((3,), dtype=F64))

        check_numeric(fn, np.array([0.5, 2.0, -1.0]))

    def test_nested(self):
        def fn(w, xs):
            def outer(c, x):
                inner, _ = olaf.scan(lambda d, _: (olaf.sin(d) * x + w, None), c, None, length=2)
                return inner, inner * inner

            carry, ys = olaf.scan(outer, olaf.ones((), F64), xs)
            return carry + olaf.sum(ys)

        check_numeric(fn, np.array(0.3), np.array([0.9, -0.4]))

    def test_matches_unrolled_graph(self):
        xs = np.array([0.5, -0.3, 1.1])

        def scanned(w):
            carry, _ = olaf.scan(lambda c, x: (olaf.exp(c * x) * w, None), olaf.ones((), F64), olaf.constant(xs))
            return carry

        def unrolled(w):
            c = olaf.ones((), F64)
            for value in xs:
                c = olaf.exp(c * value) * w
            return c

        a = olaf.trace(lambda w: olaf.value_and_grad(scanned)(w), tt()).eager(np.array(0.8))[0]
        b = olaf.trace(lambda w: olaf.value_and_grad(unrolled)(w), tt()).eager(np.array(0.8))[0]
        np.testing.assert_allclose(a[0], b[0], rtol=1e-12)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-12)

    def test_inactive_carry_saves_no_residual(self):
        def fn(w):
            def body(c, _):
                counter, value = c
                return (counter + 1.0, value * w), None

            (_, value), _ = olaf.scan(body, (olaf.zeros((), F64), olaf.ones((), F64)), None, length=3)
            return value

        traced = olaf.trace(lambda w: olaf.grad(fn)(w), tt())
        forward = next(op for op in traced.graph.ops if op.kind == "scan")
        # Carries are counter, value and the captured w. counter does not depend on w, so it is no residual.
        # The VJP of mul uses value and w, but w is a pass-through carry and goes to the backward scan
        # as a carry instead, so value is the only stacked residual.
        assert len(forward.outputs) == 3 + 1
        np.testing.assert_allclose(traced.eager(np.array(2.0))[0], 12.0)


    @pytest.mark.parametrize("frozen", ["scan", "while"])
    def test_carry_initialised_by_undifferentiated_loop(self, frozen):
        # The carry becomes active only inside the body (from w), so its initial value h0 is inactive
        # and must get no cotangent; scan and while have no VJP rule, so one would raise KeyError.
        def fn(w, x0):
            if frozen == "scan":
                h0, _ = olaf.scan(lambda c, _: (c * 2.0, None), x0, None, length=3)
            else:
                _, h0 = olaf.while_loop(lambda s: s[0] < 3.0, lambda s: (s[0] + 1.0, s[1] * 2.0), (olaf.zeros((), F64), x0))
            h, _ = olaf.scan(lambda c, wi: (c * wi, None), h0, w)
            return olaf.sum(h)

        traced = olaf.trace(lambda w, x0: olaf.grad(fn)(w, x0), tt(4, 3), tt(3))
        w, x0 = R.normal(size=(4, 3)), R.normal(size=3)
        dw, _ = traced.eager(w, x0)
        expected = 8.0 * x0 * np.prod(w, axis=0) / w
        np.testing.assert_allclose(dw, expected, rtol=1e-12)

    def test_carry_independent_values_are_not_stacked(self):
        # The VJP of the body needs c, c*x, the tanh output, x, w and the broadcast constant. x comes
        # from xs, w is a pass-through carry and the constant depends on neither, so only the first
        # three (which depend on the carry) are stacked. Stacking everything gave 2 + 6 outputs.
        def fn(w, xs):
            carry, _ = olaf.scan(lambda c, x: (olaf.tanh(c * x * w) * olaf.full((2,), 3.0, F64), None), olaf.ones((2,), F64), xs)
            return olaf.sum(carry)

        traced = olaf.trace(lambda w, xs: olaf.value_and_grad(fn, (0, 1))(w, xs), tt(2), tt(4, 2))
        forward = next(op for op in traced.graph.ops if op.kind == "scan")
        assert len(forward.outputs) == 2 + 3
        check_numeric(fn, R.normal(size=2), R.normal(size=(4, 2)))


def test_composites_survive_autodiff():
    # Dedicated kernels match composites before decomposition, so the training graph must keep them.
    # The softmax VJP needs only its output, so no max-shift gradient (cmp eq) appears.
    traced = olaf.trace(lambda x: olaf.grad(lambda y: olaf.sum(olaf.softmax(y) ** 2.0))(x), tt(3, 4))
    text = traced.graph.text()
    assert text.count(" softmax ") == 1 and " cmp " not in text


def test_attention_without_mask():
    check_numeric(lambda qq, k, v: olaf.sum(olaf.attention(qq, k, v) ** 2.0), R.normal(size=(2, 3, 4)), R.normal(size=(2, 5, 4)), R.normal(size=(2, 5, 3)), tol=1e-5)


@pytest.mark.parametrize(
    "fn",
    [
        lambda y: olaf.sum(olaf.softmax(y) ** 3.0),
        lambda y: olaf.sum(olaf.gelu(y, "tanh") ** 2.0),
        lambda y: olaf.sum(olaf.layernorm(y) ** 3.0),
        lambda y: olaf.cross_entropy(y * y, olaf.constant(np.array([0, 3, 1], np.int32))),
    ],
)
def test_second_order_through_composite_vjps(fn):
    # The composite VJPs emit composites and scatter_add, which must be differentiable in turn.
    v = R.normal(size=X.shape)
    check_numeric(lambda x: olaf.sum(olaf.grad(fn)(x) * olaf.constant(v)), X, tol=1e-5)


def test_scan_with_symbolic_length_and_shape():
    N, D = olaf.sym("N"), olaf.sym("D")

    def fn(w, xs):
        carry, _ = olaf.scan(lambda c, x: (c * olaf.exp(x * w), None), olaf.ones((D,), F64), xs)
        return olaf.sum(carry)

    traced = olaf.trace(lambda w, xs: olaf.value_and_grad(fn)(w, xs), tt(D), tt(N, D))
    w, xs = R.normal(size=3), R.normal(size=(5, 3))
    (value, g), _ = traced.eager(w, xs)
    t = torch.tensor(w, requires_grad=True)
    loss = torch.exp(torch.tensor(xs) * t).prod(0).sum()
    loss.backward()
    np.testing.assert_allclose(value, loss.item())
    np.testing.assert_allclose(g, t.grad.numpy(), rtol=1e-10)


def test_second_order():
    def first(x):
        return olaf.grad(lambda y: olaf.sum(y * y * y))(x)

    traced = olaf.trace(lambda x: olaf.grad(lambda y: olaf.sum(first(y)))(x), tt(2))
    np.testing.assert_allclose(traced.eager(np.array([1.0, 2.0]))[0], [6, 12])


def test_second_order_through_scan():
    def inner(y):
        return olaf.grad(lambda z: olaf.scan(lambda c, _: (c * z, None), z, None, length=2)[0])(y)

    traced = olaf.trace(lambda x: olaf.grad(inner)(x), tt())
    # f(z)=z^3, f''(z)=6z
    np.testing.assert_allclose(traced.eager(np.array(1.5))[0], 9.0)


def test_cond_and_while_gradients_are_rejected():
    def fn(x):
        return olaf.cond(x > 0.0, lambda a: a * a, lambda a: a, x)

    with pytest.raises(NotImplementedError):
        olaf.trace(lambda x: olaf.grad(fn)(x), tt())


def test_graph_grad_output_is_valid_ir():
    b = GraphBuilder()
    x = b.param(tt(2))
    y = b.op("reduce", b.op("mul", x, x), fn="sum", axes=(0,))
    out = graph_grad(b.finish([y]), [0])
    verify(out)
    assert len(out.results) == 2
    assert "softmax" not in decompose(out).text()
    results, _ = interpret(out, [np.array([1.0, 2.0])])
    np.testing.assert_array_equal(results[1], [2, 4])
