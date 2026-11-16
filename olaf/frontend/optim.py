"""Optimizers as part of the step graph.

An optimizer update is ordinary graph ops on resident state, so it is captured, fused and
scheduled with forward and backward (principle 2): no host loop over parameters, and the
per-parameter updates become tasks that the step DAG can overlap with the next step later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from olaf.frontend.trace import Tensor, cast, constant, sqrt, state, tree_flatten, tree_map, tree_unflatten
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
        t = self.count.read() + 1.0
        self.count.write(t)
        # bias corrections from the step count in-graph: (beta ** t) with t an f32 scalar.
        correction1 = 1.0 - constant(b1, DType.f32) ** t
        correction2 = 1.0 - constant(b2, DType.f32) ** t
        step_size = self.lr / correction1
        root2 = sqrt(correction2)
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


__all__ = ["AdamW", "parameter_states", "read_all", "write_all", "Tensor"]
