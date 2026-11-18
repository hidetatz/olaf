"""Whisper (Radford et al. 2022), as Hugging Face's WhisperForConditionalGeneration: a
convolutional front end over log-mel features (computed outside, an input), a pre-norm
transformer encoder with fixed sinusoidal positions, and a decoder with causal self-attention,
cross-attention to the encoder, learned positions and an output projection tied to the token
embedding.

Parameters use Hugging Face's names, with each stack's layers on a leading axis (nn.Repeat).
The encoder's sinusoidal positions are not a parameter here (Hugging Face keeps them as one
with requires_grad=False); `encoder_positions` computes them as Hugging Face does.
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
    vocab_size: int = 51865
    num_mel_bins: int = 80
    d_model: int = 384  # tiny
    encoder_layers: int = 4
    decoder_layers: int = 4
    heads: int = 6
    ffn_dim: int = 1536
    max_source_positions: int = 1500
    max_target_positions: int = 448

    @classmethod
    def from_huggingface(cls, c) -> Config:
        if c.encoder_attention_heads != c.decoder_attention_heads or c.encoder_ffn_dim != c.decoder_ffn_dim:
            raise NotImplementedError("different encoder and decoder widths")
        return cls(c.vocab_size, c.num_mel_bins, c.d_model, c.encoder_layers, c.decoder_layers, c.encoder_attention_heads, c.encoder_ffn_dim, c.max_source_positions, c.max_target_positions)


def encoder_positions(length: int, channels: int) -> np.ndarray:
    """Hugging Face's sinusoids(): sin then cos, timescales from 1 to 10000, in f32."""
    increment = math.log(10000) / (channels // 2 - 1)
    inv = np.exp(-increment * np.arange(channels // 2, dtype=np.float32)).astype(np.float32)
    scaled = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


@dataclass(frozen=True)
class Attention(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        d = self.config.d_model
        return {
            "q_proj": nn.Linear(d, d, compute=self.compute),
            "k_proj": nn.Linear(d, d, bias=False, compute=self.compute),
            "v_proj": nn.Linear(d, d, compute=self.compute),
            "out_proj": nn.Linear(d, d, compute=self.compute),
        }

    def init(self, rng):
        return {k: v.init(rng) for k, v in self._layers().items()}

    def __call__(self, p, x, kv, mask=None):
        L, H = self._layers(), self.config.heads
        B, Tq, D = x.shape
        Tk = kv.shape[1]

        def heads(t, n):
            return olaf.transpose(t.reshape(B, n, H, D // H), 1, 2)

        q = heads(L["q_proj"](p["q_proj"], x), Tq)
        k = heads(L["k_proj"](p["k_proj"], kv), Tk)
        v = heads(L["v_proj"](p["v_proj"], kv), Tk)
        if mask is not None:
            mask = olaf.broadcast_to(mask, (B, H, Tq, Tk))
        merged = olaf.transpose(olaf.attention(q, k, v, mask), 1, 2).reshape(B, Tq, D)
        return L["out_proj"](p["out_proj"], merged)


def _ffn(config, compute):
    return {"fc1": nn.Linear(config.d_model, config.ffn_dim, compute=compute), "fc2": nn.Linear(config.ffn_dim, config.d_model, compute=compute)}


def _residual(x, y):
    return x + _to(y, x.dtype)


@dataclass(frozen=True)
class EncoderLayer(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        d = self.config.d_model
        return {"self_attn": Attention(self.config, self.compute), "self_attn_layer_norm": nn.LayerNorm(d), **_ffn(self.config, self.compute), "final_layer_norm": nn.LayerNorm(d)}

    def init(self, rng):
        return {k: v.init(rng) for k, v in self._layers().items()}

    def __call__(self, p, x):
        L = self._layers()
        h = L["self_attn_layer_norm"](p["self_attn_layer_norm"], x)
        x = _residual(x, L["self_attn"](p["self_attn"], h, h))
        h = L["final_layer_norm"](p["final_layer_norm"], x)
        return _residual(x, L["fc2"](p["fc2"], olaf.gelu(L["fc1"](p["fc1"], h))))


@dataclass(frozen=True)
class DecoderLayer(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        d = self.config.d_model
        return {
            "self_attn": Attention(self.config, self.compute),
            "self_attn_layer_norm": nn.LayerNorm(d),
            "encoder_attn": Attention(self.config, self.compute),
            "encoder_attn_layer_norm": nn.LayerNorm(d),
            **_ffn(self.config, self.compute),
            "final_layer_norm": nn.LayerNorm(d),
        }

    def init(self, rng):
        return {k: v.init(rng) for k, v in self._layers().items()}

    def __call__(self, p, x, encoded, mask):
        L = self._layers()
        h = L["self_attn_layer_norm"](p["self_attn_layer_norm"], x)
        x = _residual(x, L["self_attn"](p["self_attn"], h, h, mask))
        h = L["encoder_attn_layer_norm"](p["encoder_attn_layer_norm"], x)
        x = _residual(x, L["encoder_attn"](p["encoder_attn"], h, encoded))
        h = L["final_layer_norm"](p["final_layer_norm"], x)
        return _residual(x, L["fc2"](p["fc2"], olaf.gelu(L["fc1"](p["fc1"], h))))


@dataclass(frozen=True)
class Whisper(nn.Module):
    """Logits from (log-mel features [batch, mels, 2 * max_source_positions], decoder ids)."""

    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        return {
            "model": {
                "encoder": {
                    "conv1": nn.Conv(c.num_mel_bins, c.d_model, (3,), 1, 1, compute=k),
                    "conv2": nn.Conv(c.d_model, c.d_model, (3,), 2, 1, compute=k),
                    "layers": nn.Repeat(EncoderLayer(c, k), c.encoder_layers),
                    "layer_norm": nn.LayerNorm(c.d_model),
                },
                "decoder": {
                    "embed_tokens": nn.Embedding(c.vocab_size, c.d_model, 0.02),
                    "embed_positions": nn.Embedding(c.max_target_positions, c.d_model, 0.02),
                    "layers": nn.Repeat(DecoderLayer(c, k), c.decoder_layers),
                    "layer_norm": nn.LayerNorm(c.d_model),
                },
            }
        }

    def init(self, rng):
        return olaf.tree_map(lambda layer: layer.init(rng), self._layers())

    def encode(self, params, features):
        c, E, p = self.config, self._layers()["model"]["encoder"], params["model"]["encoder"]
        h = olaf.gelu(E["conv1"](p["conv1"], features))
        h = olaf.gelu(E["conv2"](p["conv2"], h))
        h = olaf.permute(h, (0, 2, 1))
        positions = olaf.constant(encoder_positions(c.max_source_positions, c.d_model))
        # A bf16 front end meets the f32 positions in f32, as PyTorch promotes the sum.
        x = _to(h, DType.f32) if h.dtype is DType.bf16 else h
        x = x + olaf.broadcast_to(_to(positions, x.dtype), x.shape)
        return E["layer_norm"](p["layer_norm"], E["layers"](p["layers"], x))

    def __call__(self, params, features, ids):
        encoded = self.encode(params, features)
        D, p = self._layers()["model"]["decoder"], params["model"]["decoder"]
        S = ids.shape[1]
        x = D["embed_tokens"](p["embed_tokens"], ids) + D["embed_positions"](p["embed_positions"], olaf.iota((S,)))
        mask = olaf.iota((S, S), axis=0) >= olaf.iota((S, S), axis=1)
        x = D["layers"](p["layers"], x, encoded, mask)
        x = D["layer_norm"](p["layer_norm"], x)
        return olaf.einsum("bsd,vd->bsv", _to(x, self.compute), _to(p["embed_tokens"]["weight"], self.compute))
