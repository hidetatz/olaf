"""P1.5 model 8: a small Whisper (conv front end, encoder, decoder with cross-attention)."""

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

import olaf
from olaf.examples import whisper

import test_llama
from model_gate import MULTIPLIER, Case, curve, load, one_step_median, relative

BATCH, FRAMES, SEQ, STEPS = 4, 32, 16, 200
HF = dict(vocab_size=64, num_mel_bins=8, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=FRAMES // 2, max_target_positions=SEQ, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
CONFIG = whisper.Config.from_huggingface(WhisperConfig(**HF))


def hf_model(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(WhisperConfig(**HF, attn_implementation="sdpa")).to(dtype=dtype, device=device).train()
    with torch.no_grad():
        # The same fixed positions as olaf's (the f32 sinusoids, cast like the model).
        model.model.encoder.embed_positions.weight.copy_(torch.from_numpy(whisper.encoder_positions(CONFIG.max_source_positions, CONFIG.d_model)))
    return model


def named(tree) -> dict:
    from olaf.frontend.optim import _paths

    out = {}
    for path, value in zip(_paths(tree), olaf.tree_flatten(tree)[0]):
        for stack in ("model.encoder.layers.", "model.decoder.layers."):
            if path.startswith(stack):
                for i in range(value.shape[0]):
                    out[f"{stack}{i}.{path.removeprefix(stack)}"] = value[i]
                break
        else:
            out[path] = value
    return out


def olaf_loss(model):
    def loss(params, features, ids):
        logits = model(params, features, ids)
        logits = olaf.cast(logits, olaf.f32) if logits.dtype is olaf.bf16 else logits
        return olaf.cross_entropy(logits[:, :-1, :], ids[:, 1:])

    return loss


def torch_loss(model, features, ids):
    p = next(model.parameters())
    t = torch.from_numpy(ids).long().to(p.device)
    logits = model(input_features=torch.from_numpy(features).to(device=p.device, dtype=p.dtype), decoder_input_ids=t, use_cache=False).logits
    logits = logits if p.dtype == torch.float64 else logits.float()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))


def data(steps, seed=0):
    """Features drawn per batch; each batch's text a repeated pattern tied to its features."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(4):
        features = rng.normal(size=(BATCH, HF["num_mel_bins"], FRAMES)).astype(np.float32)
        ids = np.tile(rng.integers(3, HF["vocab_size"], (BATCH, 4)), (1, SEQ // 4)).astype(np.int32)
        batches.append((features, ids))
    return [batches[k % 4] for k in range(steps)]


SPECS = (olaf.TensorType((BATCH, HF["num_mel_bins"], FRAMES), olaf.f32), olaf.TensorType((BATCH, SEQ), olaf.i32))


def case(compute=None, seed=1):
    model = whisper.Whisper(CONFIG, compute)
    return Case(model.init(np.random.default_rng(seed)), olaf_loss(model), hf_model, torch_loss, SPECS, autocast=compute is olaf.bf16, named=named)


def test_eager_forward_and_backward_match_huggingface_f64():
    c = case()
    params64 = olaf.tree_map(lambda a: a.astype(np.float64), c.params)
    specs = olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f64), params64)
    features, ids = data(1)[0]
    traced = olaf.trace(lambda p, f, ids: olaf.value_and_grad(lambda p: c.olaf_loss(p, f, ids))(p), specs, olaf.TensorType(features.shape, olaf.f64), SPECS[1])
    (loss, grads), _ = traced.eager(params64, features.astype(np.float64), ids)
    model = hf_model(torch.float64)
    load(model, named(c.params))
    reference = torch_loss(model, features.astype(np.float64), ids)
    reference.backward()
    # The encoder positions are f32 sinusoids in both; the rest runs in f64.
    assert abs(float(loss) - reference.item()) < 1e-10 * abs(reference.item())
    ours = named(grads)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert relative(ours[name], p.grad.numpy()) < 1e-10, name


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("compute", [None, olaf.bf16])
def test_training_gates(compute):
    ours = lambda t: olaf.AdamW(t, lr=3e-3, weight_decay=0.01)  # noqa: E731
    theirs = lambda m: torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.01)  # noqa: E731
    ratio, details = curve(lambda seed: case(compute, seed + 1), ours, theirs, lambda seed: data(STEPS, seed))
    assert all(d["last"] < 0.5 * d["first"] for d in details), details  # it learns
    assert ratio <= MULTIPLIER, (ratio, details)
    worst, ratios = one_step_median(lambda seed: case(compute, seed + 1), lambda seed: data(1, seed=seed + 7)[0])
    assert worst <= MULTIPLIER, ratios
