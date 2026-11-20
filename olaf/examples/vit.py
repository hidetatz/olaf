"""ViT (Dosovitskiy et al. 2021), as Hugging Face's ViTForImageClassification: a patch
embedding (a convolution with kernel = stride = patch size), a learned class token prepended and
learned position embeddings added, a pre-norm transformer encoder, a final layer norm, and a
classifier on the class token.

The sequence is 1 + patches long (65 for 32x32 images in 4x4 patches): not a multiple of the
attention tiles. Parameters use Hugging Face's names, the encoder layers on a leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_labels: int = 10
    layer_norm_eps: float = 1e-12

    @classmethod
    def from_huggingface(cls, c, num_labels: int) -> Config:
        if c.hidden_act != "gelu" or not c.qkv_bias:
            raise NotImplementedError("GELU and q/k/v biases only")
        return cls(c.image_size, c.patch_size, c.num_channels, c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.intermediate_size, num_labels, c.layer_norm_eps)

    @property
    def patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _f32(x):
    # torch.autocast runs layer_norm in f32, and PyTorch promotes a bf16 + f32 sum to f32.
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


@dataclass(frozen=True)
class Layer(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        d = c.hidden_size
        return {
            "layernorm_before": nn.LayerNorm(d, c.layer_norm_eps),
            "attention": {name: nn.Linear(d, d, compute=k) for name in ("q_proj", "k_proj", "v_proj", "o_proj")},
            "layernorm_after": nn.LayerNorm(d, c.layer_norm_eps),
            "mlp": {"fc1": nn.Linear(d, c.intermediate_size, compute=k), "fc2": nn.Linear(c.intermediate_size, d, compute=k)},
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def __call__(self, p, x):
        L, H = self._layers(), self.config.num_attention_heads
        B, S, D = x.shape
        a, pa = L["attention"], p["attention"]
        h = L["layernorm_before"](p["layernorm_before"], _f32(x))

        def heads(name):
            return olaf.transpose(a[name](pa[name], h).reshape(B, S, H, D // H), 1, 2)

        context = olaf.transpose(olaf.attention(heads("q_proj"), heads("k_proj"), heads("v_proj")), 1, 2).reshape(B, S, D)
        x = _f32(x) + _f32(a["o_proj"](pa["o_proj"], context))
        h = L["layernorm_after"](p["layernorm_after"], x)
        m, pm = L["mlp"], p["mlp"]
        h = m["fc2"](pm["fc2"], olaf.gelu(m["fc1"](pm["fc1"], h)))
        return x + _f32(h)


@dataclass(frozen=True)
class ViT(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        return {
            "vit": {
                "embeddings": {"patch_embeddings": {"projection": nn.Conv(c.num_channels, c.hidden_size, (c.patch_size, c.patch_size), c.patch_size, compute=k)}},
                "layers": nn.Repeat(Layer(c, k), c.num_hidden_layers),
                "layernorm": nn.LayerNorm(c.hidden_size, c.layer_norm_eps),
            },
            "classifier": nn.Linear(c.hidden_size, c.num_labels, compute=k),
        }

    def init(self, rng):
        c = self.config
        params = olaf.tree_map(lambda layer: layer.init(rng), self._layers())
        params["vit"]["embeddings"]["cls_token"] = (rng.standard_normal((1, 1, c.hidden_size)) * 0.02).astype(np.float32)
        params["vit"]["embeddings"]["position_embeddings"] = (rng.standard_normal((1, c.patches + 1, c.hidden_size)) * 0.02).astype(np.float32)
        return params

    def __call__(self, params, pixels):
        """Logits [N, labels] from images [N, channels, size, size]."""
        L, p, c = self._layers(), params, self.config
        E, pe = L["vit"]["embeddings"], p["vit"]["embeddings"]
        N = pixels.shape[0]
        patches = E["patch_embeddings"]["projection"](pe["patch_embeddings"]["projection"], pixels)  # [N, D, h, w]
        patches = olaf.transpose(patches.reshape(N, c.hidden_size, c.patches), 1, 2)  # [N, patches, D]
        cls = olaf.broadcast_to(pe["cls_token"], (N, 1, c.hidden_size))
        x = olaf.concat([cls, _f32(patches)] if cls.dtype is DType.f32 else [cls, patches], axis=1)
        x = x + olaf.broadcast_to(pe["position_embeddings"], x.shape)
        x = L["vit"]["layers"](p["vit"]["layers"], x)
        x = L["vit"]["layernorm"](p["vit"]["layernorm"], x)
        return L["classifier"](p["classifier"], x[:, 0, :])
