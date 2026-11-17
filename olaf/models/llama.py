"""Llama (Touvron et al. 2023), as Hugging Face's LlamaForCausalLM: RMSNorm, rotary position
embeddings, SwiGLU MLP, grouped-query attention, untied output projection, no biases.

Parameters use Hugging Face's names and layouts ([out, in] projections, nn.Linear), with the
decoder layers stacked on a leading axis (nn.Repeat); `from_huggingface` only stacks.

compute=bf16 mirrors torch.autocast(bfloat16) on Hugging Face's code: projections in bf16;
the residual stream, norms and rotary tables in f32. The rotation multiplies bf16 q and k by
f32 tables, which PyTorch promotes to f32; scaled_dot_product_attention then runs in bf16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 32
    rope_theta: float = 10000.0
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_head

    @classmethod
    def from_huggingface(cls, c) -> Config:
        if getattr(c, "tie_word_embeddings", False):
            raise NotImplementedError("tied Llama embeddings")
        if (getattr(c, "rope_scaling", None) or {}).get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope scaling {c.rope_scaling}")
        theta = getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]
        return cls(c.vocab_size, c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, theta, c.rms_norm_eps)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


@dataclass(frozen=True)
class Block(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, d, hd = self.config, self.config.hidden_size, self.config.head_dim

        def linear(i, o):
            return nn.Linear(i, o, bias=False, compute=self.compute)

        return {
            "input_layernorm": nn.RMSNorm(d, c.eps),
            "self_attn": {"q_proj": linear(d, c.n_head * hd), "k_proj": linear(d, c.n_kv_head * hd), "v_proj": linear(d, c.n_kv_head * hd), "o_proj": linear(c.n_head * hd, d)},
            "post_attention_layernorm": nn.RMSNorm(d, c.eps),
            "mlp": {"gate_proj": linear(d, c.intermediate_size), "up_proj": linear(d, c.intermediate_size), "down_proj": linear(c.intermediate_size, d)},
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x, mask, cos, sin):
        c, layers = self.config, self._layers()
        att, pa = layers["self_attn"], p["self_attn"]
        B, S, _ = x.shape
        hd, rep = c.head_dim, c.n_head // c.n_kv_head
        h = layers["input_layernorm"](p["input_layernorm"], x)

        def heads(t, n):
            return olaf.transpose(t.reshape(B, S, n, hd), 1, 2)

        q = heads(att["q_proj"](pa["q_proj"], h), c.n_head)
        k = heads(att["k_proj"](pa["k_proj"], h), c.n_kv_head)
        v = heads(att["v_proj"](pa["v_proj"], h), c.n_kv_head)
        q, k = (nn.rotate(_to(t, DType.f32), cos, sin) for t in (q, k))
        if rep > 1:
            # Hugging Face's repeat_kv: each key and value head serves `rep` query heads.
            k, v = (olaf.broadcast_to(t.reshape(B, c.n_kv_head, 1, S, hd), (B, c.n_kv_head, rep, S, hd)).reshape(B, c.n_head, S, hd) for t in (k, v))
        dtype = self.compute or x.dtype
        attended = olaf.attention(_to(q, dtype), _to(k, dtype), _to(v, dtype), olaf.broadcast_to(mask, (B, c.n_head, S, S)))
        merged = olaf.transpose(attended, 1, 2).reshape(B, S, c.n_head * hd)
        x = x + _to(att["o_proj"](pa["o_proj"], merged), x.dtype)
        m, pm = layers["mlp"], p["mlp"]
        h = layers["post_attention_layernorm"](p["post_attention_layernorm"], x)
        gated = olaf.silu(m["gate_proj"](pm["gate_proj"], h)) * m["up_proj"](pm["up_proj"], h)
        return x + _to(m["down_proj"](pm["down_proj"], gated), x.dtype)


@dataclass(frozen=True)
class Llama(nn.Module):
    """Logits from token ids [batch, seq]."""

    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        return {
            "model": {"embed_tokens": nn.Embedding(c.vocab_size, c.hidden_size, 0.02), "layers": nn.Repeat(Block(c, self.compute), c.n_layer), "norm": nn.RMSNorm(c.hidden_size, c.eps)},
            "lm_head": nn.Linear(c.hidden_size, c.vocab_size, bias=False, compute=self.compute),
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, params, ids, *, unrolled: bool = False):
        layers, p = self._layers(), params["model"]
        S = ids.shape[1]
        x = layers["model"]["embed_tokens"](p["embed_tokens"], ids)
        mask = olaf.iota((S, S), axis=0) >= olaf.iota((S, S), axis=1)
        cos, sin = nn.rotary_tables(S, self.config.head_dim, self.config.rope_theta)
        x = layers["model"]["layers"](p["layers"], x, mask, cos, sin, unrolled=unrolled)
        x = layers["model"]["norm"](p["norm"], x)
        return layers["lm_head"](params["lm_head"], x)


def from_huggingface(model) -> dict:
    state = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    n = model.config.num_hidden_layers
    tree: dict = {}

    def put(path: str, value) -> None:
        node = tree
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value

    for key, value in state.items():
        if ".layers." not in key:
            put(key, value)
    names = sorted({k.split(".", 3)[3] for k in state if ".layers." in k})
    for name in names:
        put(f"model.layers.{name}", np.stack([state[f"model.layers.{i}.{name}"] for i in range(n)]))
    return tree
