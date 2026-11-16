"""GPT-2 small: the first N training steps' losses, olaf against PyTorch (the P1 loss gate).

Both sides start from the same random initialization and see the same batches: windows of
GPT-2 tokens from the Python standard library's source (real text, available offline). Training
is the benchmark's configuration (bf16 autocast-style compute, f32 master weights, AdamW lr 6e-4,
betas 0.9/0.95, weight decay 0.1 on matrices, no dropout). Each side runs in its own process
and saves its losses; `compare` reports how far apart they are.

    python bench/gpt2_loss_match.py torch [steps]
    python bench/gpt2_loss_match.py olaf [steps]
    python bench/gpt2_loss_match.py compare
    python bench/gpt2_loss_match.py torch_again [steps]   (PyTorch against itself, for scale)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

BATCH, SEQ = 4, 1024
LR, BETAS, WD = 6e-4, (0.9, 0.95), 0.1
OUT = Path(__file__).resolve().parent / "results"


def corpus(tokens_needed: int) -> np.ndarray:
    """GPT-2 tokens of the standard library's .py files, in a fixed order."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    import sysconfig

    root = Path(sysconfig.get_paths()["stdlib"])
    ids: list[int] = []
    for path in sorted(root.rglob("*.py")):
        ids.extend(tokenizer(path.read_text(errors="replace"))["input_ids"])
        if len(ids) >= tokens_needed:
            break
    if len(ids) < tokens_needed:
        raise RuntimeError(f"only {len(ids)} tokens of text found under {root}")
    return np.array(ids[:tokens_needed], np.int32)


def batches(steps: int) -> list[np.ndarray]:
    tokens = corpus(steps * BATCH * SEQ)
    return [tokens[i * BATCH * SEQ : (i + 1) * BATCH * SEQ].reshape(BATCH, SEQ) for i in range(steps)]


def model_init(attention: str = "eager"):
    """The shared initialization (the attention implementation does not change the weights)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(attn_implementation=attention, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))


def run_torch(steps: int) -> np.ndarray:
    # SDPA, as in bench/gpt2_train.py: eager attention materializes 4 x 12 x 1024 x 1024 scores per layer.
    model = model_init("sdpa").cuda().train()
    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": WD},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR, betas=BETAS)
    losses = []
    for ids in batches(steps):
        t = torch.from_numpy(ids).long().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(t, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.array(losses)


def run_olaf(steps: int) -> np.ndarray:
    import test_training
    from test_gpt2 import split_params

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram, Session
    from olaf.pipeline import to_program

    test_training.LR, test_training.BETAS, test_training.WD = LR, BETAS, WD
    model = model_init().eval()
    traced, initial = test_training.train_step(split_params(model), model.config, BATCH, SEQ)
    session = Session(CudaProgram(to_program(test_training.lower(traced.graph), AMPERE_RTX3060)), {})
    session.load_state(initial)
    losses = []
    for ids in batches(steps):
        session.step([ids])
        losses.append(float(session.outputs()[0]))
    return np.array(losses)


def compare(first: str = "olaf", second: str = "torch") -> None:
    a, b = np.load(OUT / f"loss_{first}.npy"), np.load(OUT / f"loss_{second}.npy")
    print(f"{first} vs {second}:")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rel = np.abs(a - b) / np.abs(b)
    print(f"steps {n}: loss {b[0]:.3f} -> {b[-1]:.3f} ({second}), {a[0]:.3f} -> {a[-1]:.3f} ({first})")
    for lo, hi in [(0, 10), (0, 100), (100, 500), (500, n)]:
        hi = min(hi, n)
        if hi > lo:
            print(f"  steps {lo:4d}-{hi:4d}: mean |{first}/{second} - 1| = {rel[lo:hi].mean():.2e}, max = {rel[lo:hi].max():.2e}")
    window = min(50, n)
    smooth = lambda x: np.convolve(x, np.ones(window) / window, mode="valid")  # noqa: E731
    print(f"  {window}-step moving averages: max relative difference {np.max(np.abs(smooth(a) - smooth(b)) / smooth(b)):.2e}")


if __name__ == "__main__":
    which = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    OUT.mkdir(exist_ok=True)
    if which == "compare":
        compare()
        if (OUT / "loss_torch_again.npy").exists():
            compare("torch_again", "torch")
    else:
        start = time.time()
        losses = run_torch(steps) if which.startswith("torch") else run_olaf(steps)
        np.save(OUT / f"loss_{which}.npy", losses)
        print(f"{which}: {steps} steps in {time.time() - start:.0f} s, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
