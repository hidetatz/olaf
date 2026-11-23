"""A character-level language model on torch.nn.LSTM (stacked layers): an embedding,
`num_layers` LSTM layers (gates i, f, g, o in torch's order; c_t = f c_{t-1} + i g,
h_t = o tanh(c_t), from zero states), and a linear head predicting the next character.

As examples/rnn.py: each layer's input projection covers all steps in one product before the
scan, which keeps only W_hh h_{t-1}; the state (h, c) is f32 under bf16 compute and W_hh is
cast inside the step, so the gradients carried back through time and W_hh's own sum up in f32
(passes.accumulate makes that sum one product over the stacked steps). Parameter names follow
tests/test_lstm.py's torch twin (torch.nn.LSTM's).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    hidden_size: int = 1024
    num_layers: int = 2


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _wide(x):
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


def _sigmoid(x):
    return olaf.recip(1.0 + olaf.exp(-x))


def lstm_layer(p: dict, layer: int, xs, compute: DType | None):
    """xs [T, B, I] -> hidden states [T, B, H]."""
    T, B, _ = xs.shape
    w_ih, w_hh = _to(p[f"weight_ih_l{layer}"], compute), p[f"weight_hh_l{layer}"]
    H = w_hh.shape[1]
    pre = _wide(olaf.einsum("tbi,gi->tbg", _to(xs, compute), w_ih))
    bias = _wide(p[f"bias_ih_l{layer}"]) + _wide(p[f"bias_hh_l{layer}"])
    pre = pre + olaf.broadcast_to(_to(bias, pre.dtype).reshape(1, 1, 4 * H), pre.shape)

    def step(state, x):
        h, c = state
        gates = x + _wide(olaf.einsum("bh,gh->bg", _to(h, compute), _to(w_hh, compute)))
        i, f, g, o = (gates[:, k * H : (k + 1) * H] for k in range(4))
        c = _sigmoid(f) * c + _sigmoid(i) * olaf.tanh(g)
        h = _sigmoid(o) * olaf.tanh(c)
        return (h, c), h

    zero = olaf.zeros((B, H), pre.dtype)
    _, hs = olaf.scan(step, (zero, zero), pre)
    return hs


@dataclass(frozen=True)
class CharLSTM(nn.Module):
    config: Config
    compute: DType | None = None

    def init(self, rng):
        c = self.config
        H = c.hidden_size
        bound = 1.0 / np.sqrt(H)
        u = lambda *shape: rng.uniform(-bound, bound, shape).astype(np.float32)  # noqa: E731
        lstm = {}
        for k in range(c.num_layers):
            lstm[f"weight_ih_l{k}"] = u(4 * H, H)
            lstm[f"weight_hh_l{k}"] = u(4 * H, H)
            lstm[f"bias_ih_l{k}"] = u(4 * H)
            lstm[f"bias_hh_l{k}"] = u(4 * H)
        return {
            "embed": nn.Embedding(c.vocab_size, H, 1.0).init(rng),
            "lstm": lstm,
            "head": nn.Linear(H, c.vocab_size).init(rng),
        }

    def __call__(self, params, ids):
        """Logits [B, T, vocab] from ids [B, T]."""
        c, k = self.config, self.compute
        x = olaf.transpose(olaf.take(params["embed"]["weight"], ids), 0, 1)  # [T, B, H], time-major
        for layer in range(c.num_layers):
            x = lstm_layer(params["lstm"], layer, x, k)
        return nn.Linear(c.hidden_size, c.vocab_size, compute=k)(params["head"], olaf.transpose(x, 0, 1))


def loss(model: CharLSTM):
    def fn(params, ids):
        logits = _wide(model(params, ids))
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:])

    return fn
