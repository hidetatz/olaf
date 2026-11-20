"""BERT (Devlin et al. 2019) for masked language modelling, as Hugging Face's BertForMaskedLM:
word, position and token-type embeddings, a post-norm transformer encoder whose attention
ignores padded keys (a per-sequence mask from the data), and the prediction head with its
decoder tied to the word embeddings.

The masked-LM loss is taken at given positions (the original pretraining's
masked_lm_positions: a fixed number per sequence), so the head runs on those tokens only;
Hugging Face computes the head everywhere and ignores the other labels, the same loss.
Parameters use Hugging Face's names, the encoder layers on a leading axis (nn.Repeat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    @classmethod
    def from_huggingface(cls, c) -> Config:
        if c.hidden_act != "gelu":
            raise NotImplementedError(f"hidden_act {c.hidden_act}")
        return cls(c.vocab_size, c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.intermediate_size, c.max_position_embeddings, c.type_vocab_size, c.layer_norm_eps)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _f32(x):
    # torch.autocast runs layer_norm in f32 (its fp32 list): b16 inputs are widened, and the
    # residual stream it writes stays f32.
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


@dataclass(frozen=True)
class Layer(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        d = c.hidden_size
        return {
            "attention": {
                "self": {"query": nn.Linear(d, d, compute=k), "key": nn.Linear(d, d, compute=k), "value": nn.Linear(d, d, compute=k)},
                "output": {"dense": nn.Linear(d, d, compute=k), "LayerNorm": nn.LayerNorm(d, c.layer_norm_eps)},
            },
            "intermediate": {"dense": nn.Linear(d, c.intermediate_size, compute=k)},
            "output": {"dense": nn.Linear(c.intermediate_size, d, compute=k), "LayerNorm": nn.LayerNorm(d, c.layer_norm_eps)},
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x, mask):
        L, H = self._layers(), self.config.num_attention_heads
        B, S, D = x.shape
        a, pa = L["attention"], p["attention"]

        def heads(name):
            t = a["self"][name](pa["self"][name], x)
            return olaf.transpose(t.reshape(B, S, H, D // H), 1, 2)

        context = olaf.transpose(olaf.attention(heads("query"), heads("key"), heads("value"), mask), 1, 2).reshape(B, S, D)
        out = a["output"]["dense"](pa["output"]["dense"], context)
        x = a["output"]["LayerNorm"](pa["output"]["LayerNorm"], _f32(x) + _f32(out))
        h = olaf.gelu(L["intermediate"]["dense"](p["intermediate"]["dense"], x))
        out = L["output"]["dense"](p["output"]["dense"], h)
        return L["output"]["LayerNorm"](p["output"]["LayerNorm"], _f32(x) + _f32(out))


@dataclass(frozen=True)
class Bert(nn.Module):
    """Masked-LM logits [batch, positions, vocab] from (ids [batch, seq], token types [batch,
    seq], attention mask [batch, seq] (1 for tokens, 0 for padding), positions [batch, P])."""

    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        d = c.hidden_size
        return {
            "bert": {
                "embeddings": {
                    "word_embeddings": nn.Embedding(c.vocab_size, d, 0.02),
                    "position_embeddings": nn.Embedding(c.max_position_embeddings, d, 0.02),
                    "token_type_embeddings": nn.Embedding(c.type_vocab_size, d, 0.02),
                    "LayerNorm": nn.LayerNorm(d, c.layer_norm_eps),
                },
                "encoder": {"layer": nn.Repeat(Layer(c, k), c.num_hidden_layers)},
            },
            "cls": {"predictions": {"transform": {"dense": nn.Linear(d, d, compute=k), "LayerNorm": nn.LayerNorm(d, c.layer_norm_eps)}}},
        }

    def init(self, rng):
        params = olaf.tree_map(lambda layer: layer.init(rng), self._layers())
        params["cls"]["predictions"]["bias"] = np.zeros(self.config.vocab_size, np.float32)
        return params

    def __call__(self, params, ids, token_types, attention_mask, positions):
        L, p = self._layers(), params
        E, pe = L["bert"]["embeddings"], p["bert"]["embeddings"]
        B, S = ids.shape
        x = E["word_embeddings"](pe["word_embeddings"], ids) + E["position_embeddings"](pe["position_embeddings"], olaf.iota((S,)))
        x = E["LayerNorm"](pe["LayerNorm"], x + E["token_type_embeddings"](pe["token_type_embeddings"], token_types))
        H = self.config.num_attention_heads
        # Keys at padding are ignored by every query: [B, 1, 1, S] broadcast over heads and
        # queries (padded queries still attend to the real tokens; their outputs are unused).
        keep = olaf.reshape(olaf.cast(attention_mask, DType.bool), (B, 1, 1, S))
        mask = olaf.broadcast_to(keep, (B, H, S, S))
        x = L["bert"]["encoder"]["layer"](p["bert"]["encoder"]["layer"], x, mask)
        picked = olaf.take(x, positions, axis=1, batch_dims=1)
        T, pt = L["cls"]["predictions"]["transform"], p["cls"]["predictions"]["transform"]
        h = T["LayerNorm"](pt["LayerNorm"], _f32(olaf.gelu(T["dense"](pt["dense"], picked))))
        weight = _to(pe["word_embeddings"]["weight"], self.compute)
        logits = olaf.einsum("bpd,vd->bpv", _to(h, self.compute), weight)
        return logits + _to(p["cls"]["predictions"]["bias"], logits.dtype)


def loss(model: Bert):
    """Mean cross entropy of the masked-LM logits at `positions` against `labels` [batch, P]."""

    def fn(params, ids, token_types, attention_mask, positions, labels):
        logits = model(params, ids, token_types, attention_mask, positions)
        logits = olaf.cast(logits, DType.f32) if logits.dtype is DType.bf16 else logits
        return olaf.cross_entropy(logits, labels)

    return fn
