"""Mixed precision training steps (f32 master weights, bf16 compute, AdamW) against PyTorch autocast."""

import numpy as np
import pytest
import torch

import olaf
from olaf.frontend.optim import AdamW, parameter_states, read_all, write_all
from olaf.graph.types import TensorType
from olaf.passes.canonicalize import canonicalize
from olaf.passes.decompose import decompose
from olaf.passes.unroll import unroll_scans

from test_gpt2 import forward, param_specs, small_model, split_params

LR, BETAS, EPS, WD = 3e-3, (0.9, 0.95), 1e-8, 0.1


def decays(path: str) -> bool:
    """Weight decay on matrices only, as nanoGPT's parameter groups (p.dim() >= 2)."""
    return path.endswith("weight") and ".ln" not in path and not path.startswith("ln")


def train_step(params, config, batch, seq, compute=olaf.bf16):
    """The traced step: reads parameters and optimizer state, returns the loss, writes the updates."""
    types = param_specs(params)
    states, _ = parameter_states(types)
    opt = AdamW(types, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, decay=decays)

    def step(ids):
        p = read_all(states)

        def loss_fn(p):
            logits = forward(p, ids, config, compute=compute)
            return olaf.cross_entropy(olaf.cast(logits[:, :-1, :], olaf.f32), ids[:, 1:])

        loss, grads = olaf.value_and_grad(loss_fn)(p)
        write_all(states, opt.update(p, grads))
        return loss

    traced = olaf.trace(step, TensorType((batch, seq), olaf.i32))
    initial = {f"params.{path}": value for path, value in zip(_paths(params), olaf.tree_flatten(params)[0])}
    return traced, {**initial, **opt.initial_state()}


def _paths(params):
    from olaf.frontend.optim import _paths as paths

    return paths(params)


def lower(graph):
    return canonicalize(unroll_scans(decompose(graph)))


def torch_losses(model, batches, device, mixed=True):
    # eval(): GPT-2's config enables dropout (0.1), which olaf's model does not have. Gradients
    # and optimizer steps work the same in eval mode.
    model = model.to(device).eval()
    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": WD},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR, betas=BETAS, eps=EPS)
    losses = []
    for ids in batches:
        t = torch.from_numpy(ids).long().to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=mixed):
            logits = model(t, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model


def batches(n, batch, seq, vocab, seed=0):
    rng = np.random.default_rng(seed)
    # Repeating a short pattern makes the loss fall quickly, so each step's update matters.
    pattern = rng.integers(0, vocab, (batch, 4)).astype(np.int32)
    return [np.tile(pattern, (1, seq // 4)) for _ in range(n)]


def test_f32_adamw_step_matches_torch():
    """No mixed precision: olaf's AdamW and PyTorch's on the same f32 model, step for step (eager)."""
    model = small_model()
    params = split_params(model)
    traced, state = train_step(params, model.config, 2, 8, compute=None)
    data = batches(4, 2, 8, model.config.vocab_size)
    graph = lower(traced.graph)
    from olaf.graph.interp import interpret

    ours = []
    for ids in data:
        (loss,), state = interpret(graph, [ids], state)
        ours.append(float(loss))
    reference, trained = torch_losses(small_model(), data, "cpu", mixed=False)
    np.testing.assert_allclose(ours, reference, rtol=1e-5)
    final = state["params.wte.weight"]
    assert np.linalg.norm(final - trained.transformer.wte.weight.detach().numpy()) / np.linalg.norm(final) < 1e-5


@pytest.mark.gpu
def test_mixed_precision_steps_on_device_match_eager_and_torch():
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram
    from olaf.graph.interp import interpret
    from olaf.tile.from_graph import to_tile
    from olaf.tile.fuse import fuse

    model = small_model()
    params = split_params(model)
    batch, seq = 2, 16
    traced, initial = train_step(params, model.config, batch, seq)
    graph = lower(traced.graph)
    device = CudaProgram(fuse(to_tile(graph, AMPERE_RTX3060)))
    data = batches(6, batch, seq, model.config.vocab_size)
    eager_state, device_state = dict(initial), dict(initial)
    eager, ours = [], []
    for ids in data:
        (loss,), eager_state = interpret(graph, [ids], eager_state)
        eager.append(float(loss))
        (loss,), device_state = device.run([ids], device_state)
        ours.append(float(loss))
    reference, _ = torch_losses(small_model(), data, "cuda")
    # The device and eager differ only in f32 summation order (and the resulting bf16 roundings);
    # measured 7e-8.
    np.testing.assert_allclose(ours, eager, rtol=1e-5)
    # Against PyTorch the bf16 rounding points differ (it rounds the scores before softmax and
    # each gelu step; olaf rounds composites once). Measured 1.8e-5, below PyTorch's own bf16 vs
    # f32 difference (2.1e-5) on these steps; 1e-3 leaves room without hiding a real error.
    np.testing.assert_allclose(ours, reference, rtol=1e-3)
    assert ours[-1] < ours[0] * 0.9, ours
