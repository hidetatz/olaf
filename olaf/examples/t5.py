"""T5 (Raffel et al. 2020), as Hugging Face's T5ForConditionalGeneration (the original T5:
ReLU feed-forward, tied embeddings): an encoder and a decoder of pre-norm blocks with RMS layer
norms and no biases, attention without the 1/sqrt(d) scale, and relative position biases.

Each stack learns a [buckets, heads] table; the bucket of a query-key pair depends on its
distance (exact up to buckets / 2, logarithmic up to max_distance; the encoder distinguishes
the sign). The table's rows, gathered for every pair, are added to the scores of every layer of
the stack. Hugging Face keeps the table in the stack's first block; here it sits beside the
stacked blocks (nn.Repeat), and `named` in tests/test_t5.py maps the names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6

    @classmethod
    def from_huggingface(cls, c) -> Config:
        if c.feed_forward_proj != "relu" or not c.tie_word_embeddings:
            raise NotImplementedError("only the original T5 (ReLU feed-forward, tied embeddings)")
        return cls(c.vocab_size, c.d_model, c.d_kv, c.d_ff, c.num_layers, c.num_decoder_layers, c.num_heads, c.relative_attention_num_buckets, c.relative_attention_max_distance, c.layer_norm_epsilon)


def relative_buckets(q_len: int, k_len: int, bidirectional: bool, num_buckets: int, max_distance: int) -> np.ndarray:
    """[q_len, k_len] i32 bucket of each (query, key) pair, as Hugging Face's
    _relative_position_bucket (its f32 logarithm included): static, so a constant."""
    relative = np.arange(k_len)[None, :] - np.arange(q_len)[:, None]
    buckets = np.zeros_like(relative)
    if bidirectional:
        num_buckets //= 2
        buckets += (relative > 0).astype(np.int64) * num_buckets
        relative = np.abs(relative)
    else:
        relative = -np.minimum(relative, 0)
    max_exact = num_buckets // 2
    is_small = relative < max_exact
    with np.errstate(divide="ignore"):
        scaled = np.log(relative.astype(np.float32) / np.float32(max_exact)) / np.float32(math.log(max_distance / max_exact)) * np.float32(num_buckets - max_exact)
    large = np.minimum(max_exact + np.where(is_small, 0, scaled).astype(np.int64), num_buckets - 1)
    return (buckets + np.where(is_small, relative, large)).astype(np.int32)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


@dataclass(frozen=True)
class LayerNorm(nn.Module):
    """Hugging Face's T5LayerNorm: x * rsqrt(mean(x^2) + eps) * weight, no mean subtracted and
    no bias. The variance in f32 for b16 and f32 inputs, the product in x's type (promoted with
    the weight's): unlike LlamaRMSNorm (nn.RMSNorm), x itself is not cast to f32. For f64 inputs
    the variance is f64 too, where Hugging Face's stays f32: an f64 model is the reference, and
    f32 rounding in it (sums in another order than PyTorch's) would be its only error
    (tests/test_t5.py runs Hugging Face's norm in f64 likewise)."""

    dim: int
    eps: float = 1e-6

    def init(self, rng):
        return {"weight": np.ones(self.dim, np.float32)}

    def __call__(self, p, x):
        h = x if x.dtype is DType.f64 else _to(x, DType.f32)
        r = olaf.rsqrt(olaf.mean(h * h, axis=-1).reshape(*x.shape[:-1], 1) + self.eps)
        y = _to(x, DType.f32) * r if x.dtype is DType.bf16 else x * _to(r, x.dtype)
        weight = p["weight"]
        return weight * _to(y, weight.dtype) if weight.dtype is not y.dtype else weight * y


@dataclass(frozen=True)
class Attention(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        inner = c.num_heads * c.d_kv
        return {name: nn.Linear(c.d_model if name != "o" else inner, inner if name != "o" else c.d_model, bias=False, compute=k) for name in ("q", "k", "v", "o")}

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x, kv, bias, causal=False):
        """bias: [heads, Sq, Sk] (the position bias) or None."""
        L, c = self._layers(), self.config
        B, Sq, _ = x.shape
        Sk = kv.shape[1]
        H, D = c.num_heads, c.d_kv

        def heads(name, t, n):
            return olaf.transpose(L[name](p[name], t).reshape(B, n, H, D), 1, 2)

        q, k, v = heads("q", x, Sq), heads("k", kv, Sk), heads("v", kv, Sk)
        mask = None
        if causal:
            mask = olaf.broadcast_to((olaf.iota((Sq, Sk), axis=0) >= olaf.iota((Sq, Sk), axis=1)).reshape(1, 1, Sq, Sk), (B, H, Sq, Sk))
        spread = None if bias is None else bias.reshape(1, H, Sq, Sk)
        # T5 does not scale the scores. Hugging Face adds the f32 bias to bf16 scores in place
        # (rounding the sum to bf16) and takes the softmax in f32; the composite keeps the
        # scores in f32, one rounding fewer, and rounds the probabilities to bf16 as HF does.
        out = olaf.attention(q, k, v, mask, spread, scale=1.0)
        return L["o"](p["o"], olaf.transpose(out, 1, 2).reshape(B, Sq, H * D))


def _ff(config, compute):
    return {"wi": nn.Linear(config.d_model, config.d_ff, bias=False, compute=compute), "wo": nn.Linear(config.d_ff, config.d_model, bias=False, compute=compute)}


def _residual(x, y):
    return x + _to(y, x.dtype)


@dataclass(frozen=True)
class EncoderBlock(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        return {
            "layer": {
                "0": {"SelfAttention": Attention(c, self.compute), "layer_norm": LayerNorm(c.d_model, c.layer_norm_epsilon)},
                "1": {"DenseReluDense": _ff(c, self.compute), "layer_norm": LayerNorm(c.d_model, c.layer_norm_epsilon)},
            }
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x, bias):
        L, P = self._layers()["layer"], p["layer"]
        h = L["0"]["layer_norm"](P["0"]["layer_norm"], x)
        x = _residual(x, L["0"]["SelfAttention"](P["0"]["SelfAttention"], h, h, bias))
        f, pf = L["1"]["DenseReluDense"], P["1"]["DenseReluDense"]
        h = L["1"]["layer_norm"](P["1"]["layer_norm"], x)
        return _residual(x, f["wo"](pf["wo"], olaf.nn.relu(f["wi"](pf["wi"], h))))


@dataclass(frozen=True)
class DecoderBlock(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        norm = lambda: LayerNorm(c.d_model, c.layer_norm_epsilon)  # noqa: E731
        return {
            "layer": {
                "0": {"SelfAttention": Attention(c, self.compute), "layer_norm": norm()},
                "1": {"EncDecAttention": Attention(c, self.compute), "layer_norm": norm()},
                "2": {"DenseReluDense": _ff(c, self.compute), "layer_norm": norm()},
            }
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x, encoded, bias):
        L, P = self._layers()["layer"], p["layer"]
        h = L["0"]["layer_norm"](P["0"]["layer_norm"], x)
        x = _residual(x, L["0"]["SelfAttention"](P["0"]["SelfAttention"], h, h, bias, causal=True))
        h = L["1"]["layer_norm"](P["1"]["layer_norm"], x)
        x = _residual(x, L["1"]["EncDecAttention"](P["1"]["EncDecAttention"], h, encoded, None))
        f, pf = L["2"]["DenseReluDense"], P["2"]["DenseReluDense"]
        h = L["2"]["layer_norm"](P["2"]["layer_norm"], x)
        return _residual(x, f["wo"](pf["wo"], olaf.nn.relu(f["wi"](pf["wi"], h))))


@dataclass(frozen=True)
class T5(nn.Module):
    """Logits [batch, target, vocab] from (input ids [batch, source], decoder input ids
    [batch, target]: the labels shifted right, starting with the pad token)."""

    config: Config
    compute: DType | None = None

    def _stacks(self):
        c = self.config
        return {
            "encoder": {"block": nn.Repeat(EncoderBlock(c, self.compute), c.num_layers), "final_layer_norm": LayerNorm(c.d_model, c.layer_norm_epsilon)},
            "decoder": {"block": nn.Repeat(DecoderBlock(c, self.compute), c.num_decoder_layers), "final_layer_norm": LayerNorm(c.d_model, c.layer_norm_epsilon)},
        }

    def init(self, rng):
        c = self.config
        params = {"shared": nn.Embedding(c.vocab_size, c.d_model, 1.0).init(rng)}
        for name, stack in self._stacks().items():
            params[name] = {k: v.init(rng) for k, v in stack.items()}
            params[name]["relative_attention_bias"] = nn.Embedding(c.relative_attention_num_buckets, c.num_heads, 1.0).init(rng)
        return params

    def _position_bias(self, table, q_len, k_len, bidirectional):
        c = self.config
        buckets = olaf.constant(relative_buckets(q_len, k_len, bidirectional, c.relative_attention_num_buckets, c.relative_attention_max_distance))
        return olaf.permute(olaf.take(table["weight"], buckets), (2, 0, 1))  # [heads, q, k]

    def __call__(self, params, ids, decoder_ids):
        S, T = ids.shape[1], decoder_ids.shape[1]
        stacks, emb = self._stacks(), params["shared"]["weight"]
        enc, dec = params["encoder"], params["decoder"]
        x = olaf.take(emb, ids)
        bias = self._position_bias(enc["relative_attention_bias"], S, S, True)
        x = stacks["encoder"]["block"](enc["block"], x, bias)
        encoded = stacks["encoder"]["final_layer_norm"](enc["final_layer_norm"], x)
        y = olaf.take(emb, decoder_ids)
        bias = self._position_bias(dec["relative_attention_bias"], T, T, False)
        y = stacks["decoder"]["block"](dec["block"], y, encoded, bias)
        y = stacks["decoder"]["final_layer_norm"](dec["final_layer_norm"], y)
        # Tied embeddings: the decoder output is rescaled by d_model^-0.5 before the projection.
        y = y * (self.config.d_model**-0.5)
        return olaf.einsum("btd,vd->btv", _to(y, self.compute), _to(emb, self.compute))


def loss(model: T5):
    def fn(params, ids, decoder_ids, labels):
        logits = model(params, ids, decoder_ids)
        logits = olaf.cast(logits, DType.f32) if logits.dtype is DType.bf16 else logits
        return olaf.cross_entropy(logits, labels)

    return fn
