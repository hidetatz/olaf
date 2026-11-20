"""A character-level RNN language model (torch.nn.RNN: tanh, stacked; the recurrence a scan)."""

import numpy as np
import pytest
import torch
import torch.nn as T
import torch.nn.functional as F

import olaf
from olaf.examples import rnn

from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, SEQ, STEPS = 8, 24, 150
CONFIG = rnn.Config(vocab_size=32, hidden_size=32, num_layers=2)


class TorchCharRNN(T.Module):
    def __init__(self, c=None):
        super().__init__()
        c = c or CONFIG
        self.embed = T.Embedding(c.vocab_size, c.hidden_size)
        self.rnn = T.RNN(c.hidden_size, c.hidden_size, c.num_layers, batch_first=True)
        self.head = T.Linear(c.hidden_size, c.vocab_size)

    def forward(self, ids):
        return self.head(self.rnn(self.embed(ids))[0])


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    return dict(zip(_paths(tree), olaf.tree_flatten(tree)[0]))


def torch_model(dtype=torch.float32, device="cpu"):
    return TorchCharRNN().to(dtype=dtype, device=device).train()


def torch_loss(model, ids):
    p = next(model.parameters())
    t = torch.from_numpy(ids).long().to(p.device)
    logits = model(t)
    logits = logits if p.dtype == torch.float64 else logits.float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))


def data(steps, seed=0):
    """Four fixed batches, cycled: each sequence repeats a 3-character pattern, so predicting
    the next character needs the recurrence (the current character alone does not decide it
    when a pattern repeats a character)."""
    rng = np.random.default_rng(seed)
    batches = [np.tile(rng.integers(0, CONFIG.vocab_size, (BATCH, 3)), (1, SEQ // 3)).astype(np.int32) for _ in range(4)]
    return [(batches[k % 4],) for k in range(steps)]


SPECS = (olaf.TensorType((BATCH, SEQ), olaf.i32),)


def case(compute=None, seed=1):
    model = rnn.CharRNN(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), rnn.loss(model), torch_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


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
