"""Mamba-2 (Dao and Gu 2024), as Hugging Face's Mamba2ForCausalLM: pre-norm blocks whose mixer
projects to a gate, the SSM input with its B and C, and per-head time steps; a causal depthwise
convolution and SiLU over the SSM input; the SSD state space recurrence; a gated RMS norm; and an
output projection. Tied embeddings.

The recurrence h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t, y_t = C_t h_t + D x_t has one scalar
decay per head (Mamba-2's restriction), so it is computed in chunks without a sequential scan,
as Hugging Face's mamba2_chunk_scan does: within a chunk, attention-like products masked by
the decays exp(segment sums of dt A); across chunks, each chunk's final state, decayed into
every later chunk (a product over chunk pairs, not a recurrence). Segment sums are differences
of cumulative sums, and cumulative sums are products with a triangular matrix.

SSD runs in f32 for b16 inputs (Hugging Face casts to float), in the inputs' type otherwise, but
for its products: the einsums run in the compute type, as torch.autocast runs them (the cumulative
sums, written as products here, stay f32: torch.cumsum is not on its bf16 list).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50280
    hidden_size: int = 768
    state_size: int = 128
    num_hidden_layers: int = 24
    num_heads: int = 24
    head_dim: int = 64
    expand: int = 2
    chunk_size: int = 64
    conv_kernel: int = 4
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def from_huggingface(cls, c) -> Config:
        if c.n_groups != 1 or c.use_bias or not c.use_conv_bias or c.hidden_act != "silu" or not c.tie_word_embeddings or tuple(c.time_step_limit) != (0.0, float("inf")):
            raise NotImplementedError("one group, no projection biases, convolution bias, SiLU, tied embeddings, no time step limit")
        return cls(c.vocab_size, c.hidden_size, c.state_size, c.num_hidden_layers, c.num_heads, c.head_dim, c.expand, c.chunk_size, c.conv_kernel, c.layer_norm_epsilon)

    @property
    def intermediate(self) -> int:
        return self.expand * self.hidden_size

    @property
    def conv_dim(self) -> int:
        return self.intermediate + 2 * self.state_size


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _wide(x):
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


def rms_norm(weight, x, eps, gate=None):
    """Hugging Face's Mamba2RMSNorm (gate None) and MambaRMSNormGated: in f32 (in the input's
    type when wider), the gate's SiLU multiplied in first, rounded back to the input's type,
    then the weight."""
    h = _wide(x)
    if gate is not None:
        h = h * olaf.silu(_to(gate, h.dtype))
    h = h * olaf.rsqrt(olaf.mean(h * h, axis=-1).reshape(*h.shape[:-1], 1) + eps)
    return weight * _to(_to(h, x.dtype), weight.dtype)


def softplus(x):
    """log(1 + exp(x)), as torch's softplus above its threshold of 20 (x itself there)."""
    return olaf.where(x > 20.0, x, olaf.log(1.0 + olaf.exp(olaf.minimum(x, 20.0))))


def ssd(x, dt, A, B, C, D, chunk: int, compute: DType | None = None):
    """y [b, s, h, p] from x [b, s, h, p], dt [b, s, h] (after softplus), A [h], B and C
    [b, s, n], D [h]; s a multiple of chunk. Products run in compute (None: the inputs' type)."""

    def einsum(spec, u, v):
        return olaf.einsum(spec, _to(u, compute), _to(v, compute))

    b, s, h, p = x.shape
    n = B.shape[-1]
    c = s // chunk
    dtype = x.dtype
    tri = olaf.constant(np.triu(np.ones((chunk, chunk))).astype(dtype.numpy))  # [l, m]: l <= m
    D_residual = olaf.broadcast_to(D.reshape(1, 1, h, 1), x.shape) * x
    x = x * olaf.broadcast_to(dt.reshape(b, s, h, 1), x.shape)
    a = olaf.broadcast_to(A.reshape(1, 1, h), dt.shape) * dt  # [b, s, h]
    xs = x.reshape(b, c, chunk, h, p)
    Bs = B.reshape(b, c, chunk, n)
    Cs = C.reshape(b, c, chunk, n)
    a = olaf.permute(a.reshape(b, c, chunk, h), (0, 3, 1, 2))  # [b, h, c, l]
    cum = olaf.einsum("bhcl,lm->bhcm", a, tri)  # inclusive cumulative sums within each chunk
    # 1. Within chunks: decays exp(cum[l] - cum[s]) for s <= l, scores C_l . B_s.
    causal = olaf.constant(np.tril(np.ones((chunk, chunk), bool)))
    seg = olaf.broadcast_to(cum.reshape(b, h, c, chunk, 1), (b, h, c, chunk, chunk)) - olaf.broadcast_to(cum.reshape(b, h, c, 1, chunk), (b, h, c, chunk, chunk))
    L = olaf.exp(olaf.where(olaf.broadcast_to(causal, seg.shape), seg, float("-inf")))  # [b, h, c, l, s]
    G = _to(einsum("bcln,bcsn->bcls", Cs, Bs), dtype)
    M = olaf.broadcast_to(G.reshape(b, 1, c, chunk, chunk), L.shape) * L
    y_diag = einsum("bhcls,bcshp->bclhp", M, xs)
    # 2. Each chunk's final state: inputs decayed to the chunk's end.
    last = cum[:, :, :, chunk - 1 :]
    decay = olaf.exp(olaf.broadcast_to(last, cum.shape) - cum)  # [b, h, c, l]
    decayed = xs * olaf.broadcast_to(olaf.permute(decay, (0, 2, 3, 1)).reshape(b, c, chunk, h, 1), xs.shape)
    states = einsum("bcln,bclhp->bchpn", Bs, decayed)
    # 3. The state entering chunk k: chunk j's final state decayed through chunks j+1 .. k-1,
    # summed over j < k (exp of the boundary sums' segment sums, -inf where j >= k).
    ends = cum[:, :, :, chunk - 1]  # [b, h, c]
    total = olaf.einsum("bhc,cd->bhd", ends, olaf.constant(np.triu(np.ones((c, c))).astype(dtype.numpy)))  # inclusive
    # into[k] = sum over chunks < k of ends; from chunk j's end to chunk k's start: into[k] - total[j].
    into = total - ends
    gap = olaf.broadcast_to(into.reshape(b, h, c, 1), (b, h, c, c)) - olaf.broadcast_to(total.reshape(b, h, 1, c), (b, h, c, c))
    earlier = olaf.constant(np.tril(np.ones((c, c), bool), k=-1))  # [k, j]: j < k
    carry = olaf.exp(olaf.where(olaf.broadcast_to(earlier, gap.shape), gap, float("-inf")))  # [b, h, k, j]
    entering = einsum("bhkj,bjhpn->bkhpn", carry, states)
    # 4. Their contribution: decayed from the chunk's start to each position, through C.
    start_decay = olaf.exp(cum)  # [b, h, c, l]
    y_off = _to(einsum("bcln,bchpn->bclhp", Cs, entering), dtype)
    y_off = y_off * olaf.broadcast_to(olaf.permute(start_decay, (0, 2, 3, 1)).reshape(b, c, chunk, h, 1), y_off.shape)
    y = (_to(y_diag, dtype) + y_off).reshape(b, s, h, p)
    return y + D_residual


@dataclass(frozen=True)
class Mixer(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c, k = self.config, self.compute
        return {
            "in_proj": nn.Linear(c.hidden_size, c.intermediate + c.conv_dim + c.num_heads, bias=False, compute=k),
            "conv1d": nn.Conv(c.conv_dim, c.conv_dim, (c.conv_kernel,), 1, c.conv_kernel - 1, groups=c.conv_dim, compute=k),
            "out_proj": nn.Linear(c.intermediate, c.hidden_size, bias=False, compute=k),
        }

    def init(self, rng):
        c = self.config
        params = {name: layer.init(rng) for name, layer in self._layers().items()}
        params["dt_bias"] = rng.uniform(-4.0, -1.0, c.num_heads).astype(np.float32)
        params["A_log"] = np.log(rng.uniform(1.0, 16.0, c.num_heads)).astype(np.float32)
        params["D"] = np.ones(c.num_heads, np.float32)
        params["norm"] = {"weight": np.ones(c.intermediate, np.float32)}
        return params

    def __call__(self, p, x):
        L, c = self._layers(), self.config
        B_, S, _ = x.shape
        I, H, P, N = c.intermediate, c.num_heads, c.head_dim, c.state_size
        proj = L["in_proj"](p["in_proj"], x)
        gate, xbc, dt = proj[:, :, :I], proj[:, :, I : I + c.conv_dim], proj[:, :, I + c.conv_dim :]
        # Causal depthwise convolution: padded by kernel - 1 on both sides, the first S kept.
        conv = L["conv1d"](p["conv1d"], olaf.transpose(xbc, 1, 2))[:, :, :S]
        xbc = olaf.silu(olaf.transpose(conv, 1, 2))
        xs, Bm, Cm = xbc[:, :, :I], xbc[:, :, I : I + N], xbc[:, :, I + N :]
        work = DType.f32 if x.dtype is DType.bf16 else x.dtype
        dt = softplus(_to(dt, work) + olaf.broadcast_to(_to(p["dt_bias"], work).reshape(1, 1, H), dt.shape))
        A = -olaf.exp(_to(p["A_log"], work))
        y = ssd(_to(xs, work).reshape(B_, S, H, P), dt, A, _to(Bm, work), _to(Cm, work), _to(p["D"], work), c.chunk_size, self.compute)
        y = rms_norm(p["norm"]["weight"], y.reshape(B_, S, I), c.layer_norm_epsilon, gate)
        return L["out_proj"](p["out_proj"], _to(y, x.dtype))


@dataclass(frozen=True)
class Block(nn.Module):
    config: Config
    compute: DType | None = None

    def init(self, rng):
        return {"norm": {"weight": np.ones(self.config.hidden_size, np.float32)}, "mixer": Mixer(self.config, self.compute).init(rng)}

    def __call__(self, p, x):
        # Hugging Face keeps the residual in f32 (residual_in_fp32) and normalizes it in the
        # norm weight's type.
        h = rms_norm(p["norm"]["weight"], _to(x, p["norm"]["weight"].dtype), self.config.layer_norm_epsilon)
        out = Mixer(self.config, self.compute)(p["mixer"], h)
        residual = _wide(x)
        return residual + _to(out, residual.dtype)


@dataclass(frozen=True)
class Mamba2(nn.Module):
    """Logits [batch, seq, vocab] from ids [batch, seq] (seq a multiple of chunk_size)."""

    config: Config
    compute: DType | None = None

    def init(self, rng):
        c = self.config
        return {
            "backbone": {
                "embeddings": nn.Embedding(c.vocab_size, c.hidden_size, 0.02).init(rng),
                "layers": nn.Repeat(Block(c, self.compute), c.num_hidden_layers).init(rng),
                "norm_f": {"weight": np.ones(c.hidden_size, np.float32)},
            }
        }

    def __call__(self, params, ids):
        c, p = self.config, params["backbone"]
        x = olaf.take(p["embeddings"]["weight"], ids)
        x = nn.Repeat(Block(c, self.compute), c.num_hidden_layers)(p["layers"], x)
        x = rms_norm(p["norm_f"]["weight"], _to(x, p["norm_f"]["weight"].dtype), c.layer_norm_epsilon)
        return olaf.einsum("bsd,vd->bsv", _to(x, self.compute), _to(p["embeddings"]["weight"], self.compute))


def loss(model: Mamba2):
    def fn(params, ids):
        logits = model(params, ids)
        logits = _wide(logits)
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:])

    return fn
