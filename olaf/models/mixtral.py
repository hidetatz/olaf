"""Mixtral (Jiang et al. 2024): Llama's attention with a sparse mixture-of-experts feed-forward,
as Hugging Face's MixtralForCausalLM (transformers 5 layout, experts stored as 3-D tensors).

Routing is data dependent and the step graph has static shapes, so tokens move with gather and
scatter_add: each expert gets a buffer of `capacity` token slots, a token's slot is the number
of earlier tokens routed to the same expert (a prefix count), and the experts run as one
batched product over their buffers. With capacity = number of tokens no token is ever dropped,
which is what Hugging Face computes; it costs as much as running every expert on every token.
A smaller capacity (Switch Transformer's capacity factor) drops the overflow and changes the
result, so it is not the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType
from olaf.models import llama


@dataclass(frozen=True)
class Config(llama.Config):
    n_experts: int = 8
    top_k: int = 2

    @classmethod
    def from_huggingface(cls, c) -> Config:
        base = llama.Config.from_huggingface(c)
        return cls(**{f: getattr(base, f) for f in base.__dataclass_fields__}, n_experts=c.num_local_experts, top_k=c.num_experts_per_tok)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def top_k(probs, k: int):
    """Indices [T, k] (i32) and values [T, k] of the k largest entries per row, largest first;
    among equal values the lowest index wins, as torch.topk on CPU and CUDA does in practice."""
    T, E = probs.shape
    column = olaf.iota((T, E), axis=1)
    reversed_column = (E - 1) - column
    remaining = probs
    indices, values = [], []
    for _ in range(k):
        top = olaf.max(remaining, axis=1).reshape(T, 1)
        # The first column holding the maximum: the largest E-1-column among them.
        first = (E - 1) - olaf.max(olaf.where(olaf.equal(remaining, top), reversed_column, -1), axis=1)
        chosen = olaf.equal(column, first.reshape(T, 1))
        indices.append(first)
        # The value as a masked sum of probs, so the gradient goes to the chosen entry only.
        values.append(olaf.sum(olaf.where(chosen, probs, 0.0), axis=1))
        remaining = olaf.where(chosen, -np.inf, remaining)
    return indices, values


@dataclass(frozen=True)
class SparseMoE(nn.Module):
    config: Config
    compute: DType | None = None

    def init(self, rng):
        c = self.config
        d, i, e = c.hidden_size, c.intermediate_size, c.n_experts
        bound_d, bound_i = 1.0 / np.sqrt(d), 1.0 / np.sqrt(i)
        return {
            "gate": {"weight": rng.uniform(-bound_d, bound_d, (e, d)).astype(np.float32)},
            "experts": {
                "gate_up_proj": rng.uniform(-bound_d, bound_d, (e, 2 * i, d)).astype(np.float32),
                "down_proj": rng.uniform(-bound_i, bound_i, (e, d, i)).astype(np.float32),
            },
        }

    def __call__(self, p, h):
        c = self.config
        B, S, D = h.shape
        T, E, I = B * S, c.n_experts, c.intermediate_size
        flat = h.reshape(T, D)
        logits = nn.Linear(D, E, bias=False, compute=self.compute)(p["gate"], flat)
        probs = olaf.softmax(_to(logits, DType.f32), axis=-1)
        indices, values = top_k(probs, c.top_k)
        total = values[0]
        for v in values[1:]:
            total = total + v
        weights = [v / total for v in values]

        # Slot of each token in its expert's buffer: how many earlier tokens chose that expert.
        expert = olaf.iota((T, E), axis=1)
        chosen = [olaf.cast(olaf.equal(expert, i.reshape(T, 1)), DType.f32) for i in indices]
        routed = chosen[0]
        for m in chosen[1:]:
            routed = routed + m
        earlier = olaf.cast(olaf.iota((T, T), axis=0) > olaf.iota((T, T), axis=1), DType.f32)
        before = olaf.einsum("ts,se->te", earlier, routed)  # exact: counts below 2^24
        slots = [i * T + olaf.cast(olaf.sum(m * before, axis=1), DType.i32) for i, m in zip(indices, chosen)]

        dtype = self.compute or h.dtype
        buffers = olaf.zeros((E * T, D), dtype)
        for slot in slots:
            buffers = olaf.scatter_add(buffers, slot, _to(flat, dtype))
        x = buffers.reshape(E, T, D)
        w = p["experts"]
        gate_up = olaf.einsum("ecd,efd->ecf", x, _to(w["gate_up_proj"], dtype))
        act = olaf.silu(gate_up[..., :I]) * gate_up[..., I:]
        y = olaf.einsum("eci,edi->ecd", act, _to(w["down_proj"], dtype)).reshape(E * T, D)
        out = None
        for slot, weight in zip(slots, weights):
            # Hugging Face: expert output (bf16 under autocast) times the f32 routing weight.
            term = _to(olaf.take(y, slot), DType.f32) * weight.reshape(T, 1)
            out = term if out is None else out + term
        return out.reshape(B, S, D)


@dataclass(frozen=True)
class Block(llama.Block):
    def _layers(self):
        layers = super()._layers()
        layers["mlp"] = SparseMoE(self.config, self.compute)
        return layers

    def feed_forward(self, pm, h):
        return self._layers()["mlp"](pm, h)


@dataclass(frozen=True)
class Mixtral(llama.Llama):
    block = Block


from_huggingface = llama.from_huggingface
