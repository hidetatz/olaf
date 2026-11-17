"""Optimizers as part of the step graph.

An optimizer update is ordinary graph ops on resident state, so it is captured, fused and
scheduled with forward and backward (principle 2): no host loop over parameters, and the
per-parameter updates become tasks that the step DAG can overlap with the next step later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from olaf.frontend.trace import Tensor, Traced, cast, constant, sqrt, state, trace, tree_flatten, tree_map, tree_unflatten, value_and_grad
from olaf.graph.types import DType, StateKind, TensorType


@dataclass
class AdamW:
    """torch.optim.AdamW, step for step (decoupled weight decay, bias-corrected moments).

    `decay` decides per parameter (by its pytree path) whether weight decay applies; GPT-2 style
    training decays matrices and not biases, layernorm parameters or position embeddings.
    Moments and the step count are f32 resident state, like PyTorch's defaults for f32 parameters.
    """

    param_types: object  # pytree of TensorType
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    decay: Callable[[str], bool] = lambda path: True
    name: str = "adamw"

    def __post_init__(self) -> None:
        leaves, self.structure = tree_flatten(self.param_types)
        self.paths = _paths(self.param_types)
        self.m = [state(f"{self.name}.m.{p}", t, StateKind.optim) for p, t in zip(self.paths, leaves)]
        self.v = [state(f"{self.name}.v.{p}", t, StateKind.optim) for p, t in zip(self.paths, leaves)]
        self.count = state(f"{self.name}.step", TensorType((), DType.f32), StateKind.optim)

    def initial_state(self) -> dict:
        """Zero moments and step count, as NumPy arrays keyed by state name."""
        import numpy as np

        leaves, _ = tree_flatten(self.param_types)
        out = {}
        for p, t in zip(self.paths, leaves):
            out[f"{self.name}.m.{p}"] = np.zeros(t.shape, np.float32)
            out[f"{self.name}.v.{p}"] = np.zeros(t.shape, np.float32)
        out[f"{self.name}.step"] = np.zeros((), np.float32)
        return out

    def update(self, params, grads):
        """Return the updated parameters and write the new moments and step count."""
        b1, b2 = self.betas
        count = self.count.read() + 1.0
        self.count.write(count)
        # The bias corrections 1 - beta ** t cancel badly: in f32, 1 - 0.999 already carries a
        # 1.3e-5 relative error (0.999 rounds to 0.99900001), and the step size inherits it
        # (an MLP's f32 loss curve drifted 300x further from the f64 truth than PyTorch's).
        # PyTorch computes them on the host in double; here they are f64 scalars in the graph,
        # rounded to f32 once, and the per-element update stays f32.
        t = cast(count, DType.f64)
        correction1 = 1.0 - constant(b1, DType.f64) ** t
        correction2 = 1.0 - constant(b2, DType.f64) ** t
        step_size = cast(constant(self.lr, DType.f64) / correction1, DType.f32)
        root2 = cast(sqrt(correction2), DType.f32)
        p_leaves, _ = tree_flatten(params)
        g_leaves, _ = tree_flatten(grads)
        updated = []
        for path, p, g, m_state, v_state in zip(self.paths, p_leaves, g_leaves, self.m, self.v):
            g = cast(g, p.dtype) if g.dtype is not p.dtype else g
            if self.weight_decay and self.decay(path):
                p = p * (1.0 - self.lr * self.weight_decay)
            m = m_state.read() * b1 + g * (1.0 - b1)
            v = v_state.read() * b2 + g * g * (1.0 - b2)
            m_state.write(m)
            v_state.write(v)
            denom = sqrt(v) / root2 + self.eps
            updated.append(p - (m / denom) * step_size)
        return tree_unflatten(self.structure, updated)


@dataclass
class SGD:
    """torch.optim.SGD, step for step: weight decay added to the gradient, then the momentum
    buffer (buf = momentum * buf + g; torch starts it at g, which a zero buffer gives too), and
    p -= lr * buf. No dampening or Nesterov."""

    param_types: object
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0
    name: str = "sgd"

    def __post_init__(self) -> None:
        leaves, self.structure = tree_flatten(self.param_types)
        self.paths = _paths(self.param_types)
        self.buffers = [state(f"{self.name}.momentum.{p}", t, StateKind.optim) for p, t in zip(self.paths, leaves)] if self.momentum else None

    def initial_state(self) -> dict:
        if not self.momentum:
            return {}
        leaves, _ = tree_flatten(self.param_types)
        return {f"{self.name}.momentum.{p}": np.zeros(t.shape, np.float32) for p, t in zip(self.paths, leaves)}

    def update(self, params, grads):
        p_leaves, _ = tree_flatten(params)
        g_leaves, _ = tree_flatten(grads)
        updated = []
        for k, (p, g) in enumerate(zip(p_leaves, g_leaves)):
            g = cast(g, p.dtype) if g.dtype is not p.dtype else g
            if self.weight_decay:
                g = g + p * self.weight_decay
            if self.momentum:
                g = self.buffers[k].read() * self.momentum + g
                self.buffers[k].write(g)
            updated.append(p - g * self.lr)
        return tree_unflatten(self.structure, updated)


def train_step(loss_fn: Callable, params, make_optimizer: Callable, *input_specs, dtype: DType = DType.f32, name: str = "params") -> tuple[Traced, dict]:
    """Trace one training step: read the parameters, differentiate loss_fn(params, *inputs),
    update with the optimizer, write the parameters back, return the loss.

    params: a pytree of arrays (Module.init); they become resident state in `dtype` (the master
    weights; compute dtypes are the modules' business). make_optimizer(param_types) builds the
    optimizer, e.g. lambda t: AdamW(t, lr=3e-4). Returns the traced step and the initial state
    (parameters and optimizer state, keyed by state name).
    """
    np_dtype = {DType.f32: np.float32, DType.f64: np.float64}[dtype]
    types = tree_map(lambda a: TensorType(np.shape(a), dtype), params)
    states, paths = parameter_states(types, name)
    opt = make_optimizer(types)

    def step(*inputs):
        current = read_all(states)
        loss, grads = value_and_grad(lambda p: loss_fn(p, *inputs))(current)
        write_all(states, opt.update(current, grads))
        return loss

    traced = trace(step, *input_specs)
    initial = {f"{name}.{path}": np.asarray(value, np_dtype) for path, value in zip(paths, tree_flatten(params)[0])}
    optim_state = opt.initial_state()
    if dtype is not DType.f32:
        optim_state = {k: v.astype(np_dtype) for k, v in optim_state.items()}
    return traced, {**initial, **optim_state}


def params_from_state(state: dict, like, name: str = "params"):
    """The parameter pytree (shaped like `like`) out of a state dict."""
    _, structure = tree_flatten(like)
    return tree_unflatten(structure, [state[f"{name}.{p}"] for p in _paths(like)])


def _paths(tree, prefix: str = "") -> list[str]:
    if isinstance(tree, dict):
        return [p for k in sorted(tree) for p in _paths(tree[k], f"{prefix}{k}.")]
    if isinstance(tree, (list, tuple)):
        return [p for i, v in enumerate(tree) for p in _paths(v, f"{prefix}{i}.")]
    return [prefix.rstrip(".")]


def parameter_states(param_types, name: str = "params"):
    """One resident param state per leaf, with the same pytree structure."""
    paths = _paths(param_types)
    leaves, structure = tree_flatten(param_types)
    return tree_unflatten(structure, [state(f"{name}.{p}", t, StateKind.param) for p, t in zip(paths, leaves)]), paths


def read_all(states) -> object:
    return tree_map(lambda s: s.read(), states)


def write_all(states, values) -> None:
    s_leaves, _ = tree_flatten(states)
    v_leaves, _ = tree_flatten(values)
    for s, v in zip(s_leaves, v_leaves, strict=True):
        s.write(v)


__all__ = ["AdamW", "SGD", "Tensor", "parameter_states", "params_from_state", "read_all", "train_step", "write_all"]
