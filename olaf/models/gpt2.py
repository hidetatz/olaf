"""GPT-2 (Radford et al. 2019), as Hugging Face's GPT2LMHeadModel without dropout.

Parameters use Hugging Face's names and layouts, with the blocks' weights stacked on a leading
axis (nn.Repeat): `from_huggingface` only stacks. Hugging Face's GPT-2 projections are Conv1D,
x @ W with W stored [in, out], so this model has the same layer instead of nn.Linear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    activation: str = "tanh"  # gelu approximation: "tanh" (gelu_new) or "none" (exact)
    eps: float = 1e-5

    @classmethod
    def from_huggingface(cls, config) -> Config:
        activation = {"gelu_new": "tanh", "gelu": "none"}[config.activation_function]
        return cls(config.vocab_size, config.n_positions, config.n_embd, config.n_layer, config.n_head, activation, config.layer_norm_epsilon)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


@dataclass(frozen=True)
class Conv1D(nn.Module):
    """x @ W + b with W [in, out] (Hugging Face's Conv1D). compute: as nn.Linear."""

    in_features: int
    out_features: int
    std: float = 0.02
    compute: DType | None = None

    def init(self, rng):
        return {"weight": (rng.standard_normal((self.in_features, self.out_features)) * self.std).astype(np.float32), "bias": np.zeros(self.out_features, np.float32)}

    def __call__(self, params, x):
        return olaf.einsum("bsd,de->bse", _to(x, self.compute), _to(params["weight"], self.compute)) + _to(params["bias"], self.compute)


def _residual(x, y):
    # f32 + bf16 promotes to f32 in PyTorch, so the residual stream stays in the parameters' dtype.
    return x + _to(y, x.dtype)


@dataclass(frozen=True)
class Block(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, d = self.config, self.config.n_embd
        # Hugging Face scales the residual projections' init by 1/sqrt(2 * n_layer).
        scaled = 0.02 / np.sqrt(2 * c.n_layer)
        return {
            "ln_1": nn.LayerNorm(d, c.eps),
            "attn": {"c_attn": Conv1D(d, 3 * d, compute=self.compute), "c_proj": Conv1D(d, d, scaled, self.compute)},
            "ln_2": nn.LayerNorm(d, c.eps),
            "mlp": {"c_fc": Conv1D(d, 4 * d, compute=self.compute), "c_proj": Conv1D(4 * d, d, scaled, self.compute)},
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x, mask):
        layers = self._layers()
        B, S, D = x.shape
        n_head = self.config.n_head
        qkv = layers["attn"]["c_attn"](p["attn"]["c_attn"], layers["ln_1"](p["ln_1"], x))

        def heads(t):
            return olaf.transpose(t.reshape(B, S, n_head, D // n_head), 1, 2)

        q, k, v = (heads(qkv[..., i * D : (i + 1) * D]) for i in range(3))
        attended = olaf.attention(q, k, v, olaf.broadcast_to(mask, (B, n_head, S, S)))
        merged = olaf.transpose(attended, 1, 2).reshape(B, S, D)
        x = _residual(x, layers["attn"]["c_proj"](p["attn"]["c_proj"], merged))
        hidden = olaf.gelu(layers["mlp"]["c_fc"](p["mlp"]["c_fc"], layers["ln_2"](p["ln_2"], x)), self.config.activation)
        return _residual(x, layers["mlp"]["c_proj"](p["mlp"]["c_proj"], hidden))


@dataclass(frozen=True)
class GPT2(nn.Module):
    """Logits from token ids [batch, seq]. compute=bf16 mirrors torch.autocast(bfloat16): bf16
    projections and logits, f32 elsewhere (layer norms, residual stream, softmax inside
    attention). The output projection is tied to the token embedding."""

    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        return {
            "wte": nn.Embedding(c.vocab_size, c.n_embd, 0.02),
            "wpe": nn.Embedding(c.n_positions, c.n_embd, 0.01),
            "h": nn.Repeat(Block(c, self.compute), c.n_layer),
            "ln_f": nn.LayerNorm(c.n_embd, c.eps),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, params, ids, *, unrolled: bool = False):
        layers = self._layers()
        S = ids.shape[1]
        x = layers["wte"](params["wte"], ids) + layers["wpe"](params["wpe"], olaf.iota((S,)))
        mask = olaf.iota((S, S), axis=0) >= olaf.iota((S, S), axis=1)
        x = layers["h"](params["h"], x, mask, unrolled=unrolled)
        x = layers["ln_f"](params["ln_f"], x)
        return olaf.einsum("bsd,vd->bsv", _to(x, self.compute), _to(params["wte"]["weight"], self.compute))


def from_huggingface(model) -> dict:
    """Parameters from a GPT2LMHeadModel (its tied lm_head is the token embedding)."""
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    n = model.config.n_layer
    tree: dict = {}

    def put(path: str, value) -> None:
        node = tree
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value

    for key, value in state.items():
        if key == "lm_head.weight" or ".h." in key:
            continue
        put(key.removeprefix("transformer."), value)
    names = sorted({k.split(".", 3)[3] for k in state if ".h." in k})
    for name in names:
        put(f"h.{name}", np.stack([state[f"transformer.h.{i}.{name}"] for i in range(n)]))
    return tree
