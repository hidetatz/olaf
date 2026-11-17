"""P0 exit criterion: GPT-2 forward and backward on the eager interpreter match PyTorch."""

import copy
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

import olaf
from olaf.graph.builder import verify
from olaf.passes.canonicalize import canonicalize
from olaf.examples import gpt2
from olaf.passes.decompose import decompose

TOLERANCE = 1e-5


def relative_error(a, b):
    return float(np.linalg.norm(np.ravel(a) - np.ravel(b)) / np.linalg.norm(np.ravel(b)))




def split_params(model):
    """Hugging Face's state_dict as olaf's GPT-2 parameters (blocks stacked on the first axis,
    so the step differentiates through one scan instead of n copies)."""
    return gpt2.from_huggingface(model)


def param_specs(params, dtype=olaf.f32):
    return olaf.tree_map(lambda a: olaf.TensorType(a.shape, dtype), params)


def forward(params, ids, config, *, folded=True, compute=None):
    """Logits. compute=bf16 mirrors torch.autocast(bfloat16): bf16 linears and logits, f32 elsewhere."""
    return gpt2.GPT2(gpt2.Config.from_huggingface(config), compute)(params, ids, unrolled=not folded)


def flat_grads(grads, n_layer):
    """{torch parameter name: gradient}, the stacked block gradients split per layer."""
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(grads), olaf.tree_flatten(grads)[0]):
        if path.startswith("h."):
            for layer in range(n_layer):
                out[f"h.{layer}.{path[2:]}"] = value[layer]
        else:
            out[path] = value
    return out


def trace_step(params, config, *, folded=True, dtype=olaf.f32):
    B, S = olaf.sym("B"), olaf.sym("S")

    def loss_fn(p, ids):
        logits = forward(p, ids, config, folded=folded)
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:]), logits

    step = olaf.step(lambda p, ids: olaf.value_and_grad(loss_fn, has_aux=True)(p, ids))
    return step.trace(param_specs(params, dtype), olaf.TensorType((B, S), olaf.i32))


def torch_step(model, ids):
    model.zero_grad()
    t = torch.from_numpy(ids).long()
    out = model(t, labels=t, use_cache=False)
    out.loss.backward()
    return out




def assert_matches(model, params, ids, result):
    (loss, logits), grads = result
    out = torch_step(model, ids)
    errors = {"loss": relative_error(loss, out.loss.item()), "logits": relative_error(logits, out.logits.detach().numpy())}
    named = dict(model.named_parameters())
    for name, value in flat_grads(grads, model.config.n_layer).items():
        errors[name] = relative_error(value, named[f"transformer.{name}"].grad.numpy())
    worst = max(errors, key=errors.get)
    assert errors[worst] < TOLERANCE, f"{worst}: {errors[worst]:.3e}"
    return errors


def small_model(activation="gelu_new"):
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_positions=32,
        vocab_size=50,
        activation_function=activation,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).eval()


IDS_SMALL = np.random.default_rng(0).integers(0, 50, (2, 8)).astype(np.int32)


@pytest.mark.parametrize("activation", ["gelu_new", "gelu"])
def test_small_model_matches_torch(activation):
    model = small_model(activation)
    params = split_params(model)
    traced = trace_step(params, model.config)
    verify(traced.graph)
    result, _ = traced.eager(params, IDS_SMALL)
    assert_matches(model, params, IDS_SMALL, result)


def test_exact_gelu_uses_erf_in_graph():
    model = small_model("gelu")
    text = decompose(trace_step(split_params(model), model.config).graph).text()
    assert " erf " in text


def test_folded_and_unrolled_agree():
    model = small_model()
    params = split_params(model)
    folded = trace_step(params, model.config, folded=True)
    unrolled = trace_step(params, model.config, folded=False)
    assert len(folded.graph.text()) < len(unrolled.graph.text())
    a, _ = folded.eager(params, IDS_SMALL)
    b, _ = unrolled.eager(params, IDS_SMALL)
    assert relative_error(a[0][0], b[0][0]) < TOLERANCE
    for x, y in zip(olaf.tree_flatten(a[1])[0], olaf.tree_flatten(b[1])[0]):
        assert relative_error(x, y) < TOLERANCE


def test_symbolic_shape_runs_at_other_sizes_without_retrace():
    model = small_model()
    params = split_params(model)
    traced = trace_step(params, model.config)
    ids = np.random.default_rng(1).integers(0, 50, (1, 12)).astype(np.int32)
    result, _ = traced.eager(params, ids)
    assert_matches(model, params, ids, result)


def test_canonicalize_preserves_values():
    model = small_model()
    params = split_params(model)
    traced = trace_step(params, model.config)
    after = canonicalize(decompose(traced.graph))
    verify(after)
    from olaf.graph.interp import interpret

    leaves = olaf.tree_flatten((params, IDS_SMALL))[0]
    before_values, _ = interpret(decompose(traced.graph), leaves)
    after_values, _ = interpret(after, leaves)
    for x, y in zip(before_values, after_values):
        assert relative_error(x, y) < TOLERANCE


def assert_gpt2_small(model, activation_label):
    n_params = sum(p.numel() for p in model.parameters())
    assert 123_000_000 < n_params < 125_000_000, n_params
    params = split_params(model)
    traced = trace_step(params, model.config)
    ids = np.random.default_rng(0).integers(0, model.config.vocab_size, (2, 32)).astype(np.int32)
    result, _ = traced.eager(params, ids)
    errors = assert_matches(model, params, ids, result)
    worst = max(errors, key=errors.get)
    print(f"\n{activation_label}: params={n_params:,} tensors={len(errors)} worst={worst} {errors[worst]:.3e} loss={errors['loss']:.3e} logits={errors['logits']:.3e}")


@pytest.mark.slow
@pytest.mark.parametrize("activation", ["gelu_new", "gelu"])
def test_gpt2_small_random_weights(activation):
    """fp32 against fp32 on the real 124M configuration, randomly initialized so it runs offline.

    Pretrained weights are not compared this way: see test_gpt2_small_pretrained_against_f64_truth.
    """
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(activation_function=activation, attn_implementation="eager")).eval()
    assert_gpt2_small(model, activation)


def f64_reference(model, ids):
    """Compute the true value: a double model and a double loss.

    The HF labels= path is unusable here because ForCausalLMLoss casts logits to float32 even
    for a double model, which makes its "f64" gradients carry fp32 noise.
    """
    double = copy.deepcopy(model).double()
    t = torch.from_numpy(ids).long()
    logits = double(t, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))
    loss.backward()
    grads = {k: p.grad.numpy() for k, p in double.named_parameters() if k != "lm_head.weight"}
    return loss.item(), logits.detach().numpy(), grads


def errors_against(model, reference, result):
    (loss, logits), grads = result
    ref_loss, ref_logits, ref_grads = reference
    errors = {"loss": relative_error(loss, ref_loss), "logits": relative_error(logits, ref_logits)}
    for name, value in flat_grads(grads, model.config.n_layer).items():
        errors[name] = relative_error(value, ref_grads[f"transformer.{name}"])
    return errors


ENGLISH = (
    "The quick brown fox jumps over the lazy dog. In a shocking finding, scientists discovered "
    "a herd of unicorns living in a remote valley in the Andes Mountains."
)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("OLAF_GPT2_PRETRAINED") != "1", reason="needs a pretrained weight download; enable with OLAF_GPT2_PRETRAINED=1")
@pytest.mark.parametrize("source", ["random_ids", "english"])
def test_gpt2_small_pretrained_against_f64_truth(source):
    """P0 exit criterion: pretrained weights against a double precision PyTorch reference.

    fp32 PyTorch is itself up to 3.2e-3 away from this reference on these gradients, so two fp32
    implementations cannot agree within 1e-5. Comparing with the true value still detects errors.
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    if source == "english":
        ids = np.array([AutoTokenizer.from_pretrained("gpt2")(ENGLISH)["input_ids"]] * 2, dtype=np.int32)
    else:
        ids = np.random.default_rng(0).integers(0, model.config.vocab_size, (2, 32)).astype(np.int32)
    reference = f64_reference(model, ids)
    params = split_params(model)

    wide = olaf.tree_map(lambda a: a.astype(np.float64), params)
    errors = errors_against(model, reference, trace_step(wide, model.config, dtype=olaf.f64).eager(wide, ids)[0])
    worst = max(errors, key=errors.get)
    print(f"\n{source} f64: tensors={len(errors)} worst={worst} {errors[worst]:.3e}")
    assert errors[worst] < 1e-10, f"{worst}: {errors[worst]:.3e}"

    errors = errors_against(model, reference, trace_step(params, model.config).eager(params, ids)[0])
    worst = max(errors, key=errors.get)
    print(f"{source} f32: tensors={len(errors)} worst={worst} {errors[worst]:.3e}")
    assert errors[worst] < TOLERANCE, f"{worst}: {errors[worst]:.3e}"
