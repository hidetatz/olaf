"""The P1.5 training gates for small models, against PyTorch (docs/design.md, "数値一致の判定").

Both gates measure olaf's error against a PyTorch f64 truth and compare it with PyTorch's own
error at the same precision (on CUDA, like olaf), at MULTIPLIER.

- one step: from a mid-training state, the loss and every gradient;
- curve: the loss curve over a short run, as the relative distance of the curves, over
  several seeds (initial parameters and data), comparing the medians. One seed is not enough:
  Adam's first steps are about lr * sign(g), so a gradient entry that cancels to the size of
  eps (1e-8 against terms of 1e-3) turns an f32 rounding of its sum into a percent of its
  update. Which run hits such an entry is luck: on the MLP, one seed put olaf's curve at 7.7e-6
  from the truth and PyTorch's at 1.6e-7; on five others they were within 1.3x of each other,
  and on one all three runs (olaf, PyTorch on CUDA and on CPU) were at 6.6e-6.

For GPT-2 the curve gate compares against the spread of several PyTorch runs instead; a small
model's PyTorch runs are deterministic, so that spread is zero and only a truth works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import olaf
from olaf.pipeline import lower_graph

MULTIPLIER = 2.0
# PyTorch's error is compared with at least this: a few f32 ulps (2^-23 each), where errors
# reflect only the order of additions. There a tiny model's PyTorch error (6e-8 on the MLP's
# loss curve) is as much luck as olaf's (1.3e-7), and a ratio of the two says nothing. f32 for
# bf16 compute too, since sums accumulate in f32.
FLOOR = 8 * 2.0**-23


def relative(a, b) -> float:
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@dataclass
class Case:
    """A model in both frameworks. params: olaf's pytree; its flattened paths are the torch
    state_dict names. torch_model(dtype, device) builds the twin; torch_loss(model, *inputs)
    computes its loss from NumPy inputs; olaf_loss(params, *inputs) traces olaf's.
    autocast: run PyTorch's precision run under torch.autocast(bfloat16).
    named(tree): olaf parameters (or gradients) as {torch parameter name: array}; by default the
    flattened paths, models with stacked blocks split them per layer."""

    params: dict
    olaf_loss: object
    torch_model: object
    torch_loss: object
    input_specs: tuple
    autocast: bool = False
    named: object = None

    def torch_named(self, tree) -> dict:
        if self.named is not None:
            return self.named(tree)
        from olaf.frontend.optim import _paths

        return dict(zip(_paths(tree), olaf.tree_flatten(tree)[0]))


def load(model: torch.nn.Module, named: dict) -> None:
    target = model.state_dict()
    # strict=False: buffers (rotary frequencies) are not parameters; every parameter must be set.
    missing = {k for k, _ in model.named_parameters()} - set(named)
    assert not missing, missing
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)).to(target[k].dtype) for k, v in named.items()}, strict=False)


def device_program(graph):
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram
    from olaf.pipeline import to_program

    return CudaProgram(to_program(lower_graph(graph), AMPERE_RTX3060))


def torch_grads(case: Case, params, inputs, dtype, device, autocast=False):
    model = case.torch_model(dtype, device)
    load(model, case.torch_named(params))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        loss = case.torch_loss(model, *inputs)
    loss.backward()
    return loss.item(), {k: p.grad.detach().cpu().double().numpy() for k, p in model.named_parameters()}


def one_step(case: Case, params, inputs) -> tuple[float, dict]:
    """Worst per-tensor ratio (loss included) of olaf's error to PyTorch's, against f64."""
    from olaf.frontend.optim import _paths

    specs = olaf.tree_map(lambda a: olaf.TensorType(np.shape(a), olaf.f32), params)
    traced = olaf.trace(lambda p, *xs: olaf.value_and_grad(lambda p: case.olaf_loss(p, *xs))(p), specs, *case.input_specs)
    outputs, _ = device_program(traced.graph).run([*olaf.tree_flatten(params)[0], *inputs])
    _, structure = olaf.tree_flatten(params)
    ours = case.torch_named(olaf.tree_unflatten(structure, list(outputs[1:])))
    truth_loss, truth = torch_grads(case, params, inputs, torch.float64, "cpu")
    theirs_loss, theirs = torch_grads(case, params, inputs, torch.float32, "cuda", case.autocast)
    ratios = {"loss": abs(float(outputs[0]) - truth_loss) / abs(truth_loss) / max(abs(theirs_loss - truth_loss) / abs(truth_loss), FLOOR)}
    for name, g in truth.items():
        ratios[name] = relative(ours[name], g) / max(relative(theirs[name], g), FLOOR)
    return max(ratios.values()), ratios


def olaf_curve(case: Case, make_optimizer, batches, compiled=None) -> list[float]:
    """compiled: a program from an earlier case of the same model (only the values differ)."""
    from olaf.backend.cuda.runtime import Session

    traced, initial = olaf.train_step(case.olaf_loss, case.params, make_optimizer, *case.input_specs)
    session = Session(compiled or device_program(traced.graph), {})
    session.load_state(initial)
    losses = []
    for inputs in batches:
        session.step(list(inputs))
        losses.append(float(session.outputs()[0]))
    return losses


def torch_curve(case: Case, make_optimizer, batches, dtype, device, autocast=False) -> list[float]:
    model = case.torch_model(dtype, device)
    load(model, case.torch_named(case.params))
    opt = make_optimizer(model)
    losses = []
    for inputs in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            loss = case.torch_loss(model, *inputs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def curve(make_case, olaf_optimizer, torch_optimizer, make_batches, seeds=range(5)) -> tuple[float, dict]:
    """make_case(seed) -> Case and make_batches(seed) -> batches. Returns the ratio of the median
    distances to the f64 truth, and per-seed details."""
    d_ours, d_theirs, details = [], [], []
    compiled = None
    for seed in seeds:
        case, batches = make_case(seed), make_batches(seed)
        if compiled is None:
            compiled = device_program(olaf.train_step(case.olaf_loss, case.params, olaf_optimizer, *case.input_specs)[0].graph)
        ours = olaf_curve(case, olaf_optimizer, batches, compiled)
        truth = torch_curve(case, torch_optimizer, batches, torch.float64, "cpu")
        theirs = torch_curve(case, torch_optimizer, batches, torch.float32, "cuda", case.autocast)
        d_ours.append(relative(ours, truth))
        d_theirs.append(relative(theirs, truth))
        details.append({"olaf": d_ours[-1], "torch": d_theirs[-1], "first": truth[0], "last": truth[-1]})
    return float(np.median(d_ours)) / max(float(np.median(d_theirs)), FLOOR), details
