"""Mamba-2 (SSD: the selective state space recurrence in chunks, a causal depthwise convolution,
gated RMS norm)."""

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F
from transformers import Mamba2Config, Mamba2ForCausalLM

import olaf
from olaf.examples import mamba2

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SEQ, STEPS = 4, 32, 150
HF = dict(vocab_size=64, hidden_size=32, state_size=16, num_hidden_layers=2, num_heads=4, head_dim=16, expand=2, n_groups=1, chunk_size=8, conv_kernel=4, tie_word_embeddings=True, use_bias=False, use_conv_bias=True)
CONFIG = mamba2.Config.from_huggingface(Mamba2Config(**HF))
STACK = "backbone.layers."


def _norm(weight, x, eps, gate=None):
    h = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
    if gate is not None:
        h = h * F.silu(gate.to(h.dtype))
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def _segsum(a):
    """Hugging Face's segment_sum: [..., l] -> [..., l, l] sums over (s, l], -inf above the diagonal."""
    n = a.size(-1)
    a = a[..., None].expand(*a.size(), n)
    a = a.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool, device=a.device), diagonal=-1), 0)
    out = torch.cumsum(a, dim=-2)
    return out.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool, device=a.device), diagonal=0), -torch.inf)


def _ssd(x, dt, A, B, C, D, chunk):
    """Hugging Face's mamba2_chunk_scan (one group, no initial state), in the inputs' type."""
    b, s, h, p = x.shape
    D_residual = D[..., None] * x
    x = x * dt[..., None]
    A = A * dt
    x, A = x.reshape(b, -1, chunk, h, p), A.reshape(b, -1, chunk, h)
    B = B[:, :, None].expand(b, s, h, B.shape[-1]).reshape(b, -1, chunk, h, B.shape[-1])
    C = C[:, :, None].expand(b, s, h, C.shape[-1]).reshape(b, -1, chunk, h, C.shape[-1])
    A = A.permute(0, 3, 1, 2)
    cum = torch.cumsum(A, dim=-1)
    L = torch.exp(_segsum(A))
    M = torch.einsum("bclhn,bcshn->bclsh", C, B) * L.permute(0, 2, 3, 4, 1)
    y_diag = torch.einsum("bclsh,bcshp->bclhp", M, x)
    decay = torch.exp(cum[:, :, :, -1:] - cum)
    states = torch.einsum("bclhn,bclhp->bchpn", B * decay.permute(0, -2, -1, 1)[..., None], x)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(cum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    states = torch.einsum("bzch,bzhpn->bchpn", decay_chunk, states)[:, :-1]
    y_off = torch.einsum("bclhn,bchpn->bclhp", C, states) * torch.exp(cum).permute(0, 2, 3, 1)[..., None]
    return (y_diag + y_off).reshape(b, s, h, p) + D_residual


class TorchMixer(T.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.in_proj = T.Linear(c.hidden_size, c.intermediate + c.conv_dim + c.num_heads, bias=False)
        self.conv1d = T.Conv1d(c.conv_dim, c.conv_dim, c.conv_kernel, groups=c.conv_dim, padding=c.conv_kernel - 1)
        self.dt_bias, self.A_log, self.D = (T.Parameter(torch.zeros(c.num_heads)) for _ in range(3))
        self.norm = T.Module()
        self.norm.weight = T.Parameter(torch.ones(c.intermediate))
        self.out_proj = T.Linear(c.intermediate, c.hidden_size, bias=False)

    def forward(self, x):
        c = self.c
        b, s, _ = x.shape
        I, H, P, N = c.intermediate, c.num_heads, c.head_dim, c.state_size
        gate, xbc, dt = self.in_proj(x).split([I, c.conv_dim, H], dim=-1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2).to(self.conv1d.weight.dtype))[:, :, :s].to(xbc.dtype)).transpose(1, 2)
        xs, B, C = xbc.split([I, N, N], dim=-1)
        work = torch.float64 if x.dtype == torch.float64 else torch.float32
        dt = F.softplus(dt.to(work) + self.dt_bias.to(work))
        A = -torch.exp(self.A_log.to(work))
        y = _ssd(xs.to(work).reshape(b, s, H, P), dt, A, B.to(work), C.to(work), self.D.to(work), c.chunk_size)
        y = _norm(self.norm.weight, y.reshape(b, s, I), c.layer_norm_epsilon, gate)
        return self.out_proj(y.to(x.dtype))


class TorchBlock(T.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.norm = T.Module()
        self.norm.weight = T.Parameter(torch.ones(c.hidden_size))
        self.mixer = TorchMixer(c)

    def forward(self, x):
        h = _norm(self.norm.weight, x.to(self.norm.weight.dtype), self.c.layer_norm_epsilon)
        residual = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
        return residual + self.mixer(h)


class TorchMamba2(T.Module):
    """Hugging Face's Mamba2ForCausalLM (training forward, no cache), its f32 casts kept as the
    inputs' type when wider: in an f64 model Hugging Face's SSD and norms still run in f32, and
    an f64 reference must not. tests below check it against Hugging Face's own in f32."""

    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        self.c = c
        self.backbone = T.Module()
        self.backbone.embeddings = T.Embedding(c.vocab_size, c.hidden_size)
        self.backbone.layers = T.ModuleList([TorchBlock(c) for _ in range(c.num_hidden_layers)])
        self.backbone.norm_f = T.Module()
        self.backbone.norm_f.weight = T.Parameter(torch.ones(c.hidden_size))

    def forward(self, ids):
        x = self.backbone.embeddings(ids)
        for layer in self.backbone.layers:
            x = layer(x)
        x = _norm(self.backbone.norm_f.weight, x.to(self.backbone.norm_f.weight.dtype), self.c.layer_norm_epsilon)
        return x @ self.backbone.embeddings.weight.T.to(x.dtype) if not torch.is_autocast_enabled() else F.linear(x, self.backbone.embeddings.weight)


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        if path.startswith(STACK):
            for i in range(value.shape[0]):
                out[f"{STACK}{i}.{path.removeprefix(STACK)}"] = value[i]
        else:
            out[path] = value
    return out


def torch_model(dtype=torch.float32, device="cpu"):
    return TorchMamba2().to(dtype=dtype, device=device).train()


def torch_loss(model, ids):
    p = next(model.parameters())
    t = torch.from_numpy(ids).long().to(p.device)
    logits = model(t)
    logits = getattr(logits, "logits", logits)  # Hugging Face's model (the benchmark) or the twin
    logits = logits if p.dtype == torch.float64 else logits.float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))


def data(steps, seed=0):
    """Four fixed batches, cycled: each sequence repeats a 4-token pattern."""
    rng = np.random.default_rng(seed)
    batches = [np.tile(rng.integers(0, HF["vocab_size"], (BATCH, 4)), (1, SEQ // 4)).astype(np.int32) for _ in range(4)]
    return [(batches[k % 4],) for k in range(steps)]


SPECS = (olaf.TensorType((BATCH, SEQ), olaf.i32),)


def case(compute=None, seed=1):
    model = mamba2.Mamba2(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), mamba2.loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_twin_matches_huggingface_in_f32():
    c = case()
    (ids,) = data(1)[0]
    hf = Mamba2ForCausalLM(Mamba2Config(**HF)).train()
    hf.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in named(c.params).items()}, strict=False)
    twin = torch_model()
    load(twin, named(c.params))
    t = torch.from_numpy(ids).long()
    ours, theirs = twin(t), hf(t).logits
    assert relative(ours.detach().numpy(), theirs.detach().numpy()) < 1e-5
    ours.square().sum().backward()
    theirs.square().sum().backward()
    hf_grads = dict(hf.named_parameters())
    for name, p in twin.named_parameters():
        assert relative(p.grad.numpy(), hf_grads[name].grad.numpy()) < 1e-4, name


def test_eager_forward_and_backward_match_torch_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    (ids,) = data(1)[0]
    traced = olaf.trace(lambda p, ids: olaf.value_and_grad(lambda p: c.olaf_loss(p, ids))(p), specs, SPECS[0])
    (loss, grads), _ = traced.eager(params64, ids)
    model = torch_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, ids)
    reference.backward()
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=3e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios


def test_chunked_scan_has_no_sequential_steps():
    # SSD in chunks: the launches do not grow with the sequence (no step per token or per chunk),
    # and nothing is quadratic in it (the intra-chunk decays are chunk x chunk).
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.pipeline import lower_graph, to_program

    config = mamba2.Config(vocab_size=256, hidden_size=64, state_size=32, num_hidden_layers=1, num_heads=2, head_dim=64, chunk_size=32)
    model = mamba2.Mamba2(config, olaf.bf16)

    def program(seq):
        spec = olaf.TensorType((2, seq), olaf.i32)
        traced, _ = olaf.train_step(mamba2.loss(model), model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), spec)
        return to_program(lower_graph(traced.graph), AMPERE_RTX3060)

    # 4 chunks against 16: a scan would add at least two launches per chunk (forward and
    # backward), 24 or more. A few differ anyway, as the schedule (split-K, tiles) follows sizes.
    short, long = program(128), program(512)
    assert len(long.launches) - len(short.launches) <= 8, (len(short.launches), len(long.launches))
    assert not any(list(b.type.shape).count(512) >= 2 for b in long.buffers), "a buffer quadratic in the sequence"
