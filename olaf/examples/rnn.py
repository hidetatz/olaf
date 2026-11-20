"""A character-level language model on torch.nn.RNN (Elman, tanh, stacked layers): an embedding,
`num_layers` recurrent layers h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh) from h_0 = 0,
and a linear head predicting the next character.

The recurrence is a scan over time (unrolled by passes.unroll; the unrolled steps print alike
and share their tasks). Only W_hh h_{t-1} is inside it: each layer's input projection covers
all time steps in one product before the scan, as cuDNN does, a [T * B, I] x [I, H] product
instead of T products of B rows. Layers run one after another over the whole sequence.

Under bf16 compute the products run in bf16 and the pre-activation, tanh and the hidden state
are f32 (cuDNN's bf16 RNN accumulates in f32 too). Parameter names
follow tests/test_rnn.py's torch twin (torch.nn.RNN's).
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


def rnn_layer(p: dict, layer: int, xs, compute: DType | None):
    """xs [T, B, I] -> hidden states [T, B, H]."""
    T, B, _ = xs.shape
    w_ih, w_hh = _to(p[f"weight_ih_l{layer}"], compute), p[f"weight_hh_l{layer}"]
    H = w_hh.shape[0]
    pre = _wide(olaf.einsum("tbi,hi->tbh", _to(xs, compute), w_ih))
    bias = _wide(p[f"bias_ih_l{layer}"]) + _wide(p[f"bias_hh_l{layer}"])
    pre = pre + olaf.broadcast_to(_to(bias, pre.dtype).reshape(1, 1, H), pre.shape)

    def step(h, x):
        # The state stays in the pre-activation's type (f32 under bf16), rounded only as the
        # product's operand, so its gradient back through time is not rounded at every step.
        # W_hh enters in its own type and is cast here: its gradient, one term per step, then
        # sums back through time in f32. Cast before the scan, the bf16 terms summed in bf16
        # (2.8x PyTorch's error over 24 steps; cuDNN takes it as one product over all steps).
        # The unrolled steps' casts are one op after common subexpression elimination.
        h = olaf.tanh(x + _wide(olaf.einsum("bh,gh->bg", _to(h, compute), _to(w_hh, compute))))
        return h, h

    _, hs = olaf.scan(step, olaf.zeros((B, H), pre.dtype), pre)
    return hs


@dataclass(frozen=True)
class CharRNN(nn.Module):
    config: Config
    compute: DType | None = None

    def init(self, rng):
        c = self.config
        bound = 1.0 / np.sqrt(c.hidden_size)
        u = lambda *shape: rng.uniform(-bound, bound, shape).astype(np.float32)  # noqa: E731
        rnn = {}
        for k in range(c.num_layers):
            rnn[f"weight_ih_l{k}"] = u(c.hidden_size, c.hidden_size)
            rnn[f"weight_hh_l{k}"] = u(c.hidden_size, c.hidden_size)
            rnn[f"bias_ih_l{k}"] = u(c.hidden_size)
            rnn[f"bias_hh_l{k}"] = u(c.hidden_size)
        return {
            "embed": nn.Embedding(c.vocab_size, c.hidden_size, 1.0).init(rng),
            "rnn": rnn,
            "head": nn.Linear(c.hidden_size, c.vocab_size).init(rng),
        }

    def __call__(self, params, ids):
        """Logits [B, T, vocab] from ids [B, T]."""
        c, k = self.config, self.compute
        x = olaf.transpose(olaf.take(params["embed"]["weight"], ids), 0, 1)  # [T, B, H], time-major
        for layer in range(c.num_layers):
            x = rnn_layer(params["rnn"], layer, x, k)
        return nn.Linear(c.hidden_size, c.vocab_size, compute=k)(params["head"], olaf.transpose(x, 0, 1))


def loss(model: CharRNN):
    def fn(params, ids):
        logits = _wide(model(params, ids))
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:])

    return fn
