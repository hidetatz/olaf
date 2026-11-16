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
    python bench/gpt2_loss_match.py torch_checkpoints     (states, gradients and f64 truth at CHECKPOINTS)
    python bench/gpt2_loss_match.py truth                 (f64 truth of each checkpoint's step)
    python bench/gpt2_loss_match.py onestep               (olaf's one step from each checkpoint)
    python bench/gpt2_loss_match.py curve                 (olaf's curve against all PyTorch runs)

Two checks make up the P1 loss gate (the trajectories themselves cannot match step for step:
training amplifies any rounding difference until runs differ like independent ones):
  one step   from PyTorch's state at a few steps, olaf's loss and gradients against an f64
             truth, as a multiple of PyTorch bf16's own error on the same step;
  the curve  olaf's moving-average loss against several PyTorch runs, as a multiple of how far
             PyTorch runs are from each other.
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


CHECKPOINTS = (0, 100, 500)


def olaf_names(named: dict, n_layer: int) -> dict:
    """HF parameter names -> olaf state paths (per-layer tensors stacked, as split_params does)."""
    from olaf.frontend.optim import _paths

    flat = {k: v for k, v in named.items() if not k.endswith((".attn.bias", ".attn.masked_bias")) and k != "lm_head.weight"}
    tree = {k.removeprefix("transformer."): v for k, v in flat.items() if ".h." not in k}
    names = sorted({k.split(".", 3)[3] for k in flat if ".h." in k})
    tree["h"] = {name: np.stack([flat[f"transformer.h.{i}.{name}"] for i in range(n_layer)]) for name in names}
    from olaf import tree_flatten

    return dict(zip(_paths(tree), tree_flatten(tree)[0]))


def run_torch_checkpoints(steps: int) -> None:
    """Train as run_torch, and at each checkpoint save what the one-step comparison needs."""

    model = model_init("sdpa").cuda().train()
    n_layer = model.config.n_layer
    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": WD},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR, betas=BETAS)
    names = {p: n for n, p in model.named_parameters()}

    def loss_of(m, t, autocast: bool):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = m(t, use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[:, :-1].to(logits.dtype if not autocast else torch.float32).reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))

    for k, ids in enumerate(batches(steps)):
        t = torch.from_numpy(ids).long().cuda()
        if k in CHECKPOINTS:
            state = {f"params.{n}": v for n, v in olaf_names({n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()}, n_layer).items()}
            moments = {"m": {}, "v": {}}
            for p, n in names.items():
                st = opt.state.get(p, {})
                moments["m"][n] = st["exp_avg"].cpu().numpy() if st else np.zeros(p.shape, np.float32)
                moments["v"][n] = st["exp_avg_sq"].cpu().numpy() if st else np.zeros(p.shape, np.float32)
            for which in ("m", "v"):
                state.update({f"adamw.{which}.{n}": v for n, v in olaf_names(moments[which], n_layer).items()})
            state["adamw.step"] = np.float32(k)
            hf_weights = {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()}
        loss = loss_of(model, t, autocast=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if k in CHECKPOINTS:
            grads = olaf_names({n: p.grad.float().cpu().numpy() for n, p in model.named_parameters()}, n_layer)
            np.savez(OUT / f"checkpoint_{k}.npz", ids=ids, torch_loss=loss.item(),
                     **{f"state/{n}": v for n, v in state.items()}, **{f"torch/{n}": v for n, v in grads.items()}, **{f"hf/{n}": v for n, v in hf_weights.items()})
            print(f"checkpoint {k}: torch loss {loss.item():.5f}", flush=True)
        opt.step()
        if k >= max(CHECKPOINTS):
            break


def truth() -> None:
    """The f64 truth of each checkpoint's step (its own process: only the f64 model on the GPU,
    with activations recomputed in the backward to fit; that changes no value)."""
    for k in CHECKPOINTS:
        data = np.load(OUT / f"checkpoint_{k}.npz")
        model = model_init("sdpa")
        model.load_state_dict({n[len("hf/"):]: torch.from_numpy(data[n]) for n in data.files if n.startswith("hf/")}, strict=False)
        model = model.double().cuda().train()
        model.gradient_checkpointing_enable()
        t = torch.from_numpy(data["ids"]).long().cuda()
        logits = model(t, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1))
        del logits
        loss.backward()
        grads = olaf_names({n: p.grad.cpu().numpy() for n, p in model.named_parameters()}, model.config.n_layer)
        np.savez(OUT / f"truth_{k}.npz", loss=loss.item(), **grads)
        print(f"checkpoint {k}: f64 loss {loss.item():.6f}", flush=True)
        del model, loss
        torch.cuda.empty_cache()


def one_step() -> None:
    """olaf's step from each saved checkpoint, against the f64 truth and PyTorch bf16."""
    import test_training
    from test_gpt2 import split_params

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram, Session
    from olaf.pipeline import to_program

    test_training.LR, test_training.BETAS, test_training.WD = LR, BETAS, WD
    model = model_init().eval()
    traced, _ = test_training.train_step(split_params(model), model.config, BATCH, SEQ)
    compiled = CudaProgram(to_program(test_training.lower(traced.graph), AMPERE_RTX3060))
    b1 = BETAS[0]
    rel = lambda a, b: float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))  # noqa: E731
    worst = []
    for k in CHECKPOINTS:
        data = np.load(OUT / f"checkpoint_{k}.npz")
        exact = np.load(OUT / f"truth_{k}.npz")
        state = {name[len("state/"):]: data[name] for name in data.files if name.startswith("state/")}
        session = Session(compiled, {})
        session.load_state(state)
        session.step([data["ids"]])
        loss = float(session.outputs()[0])
        after = session.read_state()
        truth_loss, torch_loss = float(exact["loss"]), float(data["torch_loss"])
        print(f"step {k}: loss error vs f64: olaf {abs(loss / truth_loss - 1):.2e}, torch {abs(torch_loss / truth_loss - 1):.2e}")
        ratios = []
        for path in exact.files:
            if path == "loss":
                continue
            m_old, m_new = state[f"adamw.m.{path}"], after[f"adamw.m.{path}"]
            grad = (m_new - b1 * m_old) / (1 - b1)  # AdamW's first moment holds the gradient exactly
            truth = exact[path]
            ours, theirs = rel(grad, truth), rel(data[f"torch/{path}"], truth)
            ratios.append((ours / theirs, path, ours, theirs))
        ratios.sort(reverse=True)
        print(f"  gradients: olaf / torch error, median {np.median([r[0] for r in ratios]):.2f}, max {ratios[0][0]:.2f} ({ratios[0][1]}: olaf {ratios[0][2]:.2e}, torch {ratios[0][3]:.2e})")
        worst.append(ratios[0][0])
    print(f"worst gradient error ratio over checkpoints: {max(worst):.2f}")


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


def curve(window: int = 50, factor: float = 2.0) -> None:
    """olaf's moving-average loss against every PyTorch run (loss_torch*.npy), compared with how
    far the PyTorch runs are from each other: passes when olaf's mean distance to them is at most
    `factor` times the PyTorch runs' mean distance to each other."""
    import itertools

    runs = {path.stem.removeprefix("loss_"): np.load(path) for path in sorted(OUT.glob("loss_torch*.npy"))}
    ours = np.load(OUT / "loss_olaf.npy")
    n = min(len(ours), *(len(r) for r in runs.values()))
    smooth = lambda x: np.convolve(x[:n], np.ones(window) / window, mode="valid")  # noqa: E731
    ma = {k: smooth(v) for k, v in runs.items()}
    olaf_ma = smooth(ours)
    distance = lambda a, b: float(np.mean(np.abs(a - b) / b))  # noqa: E731
    between = [distance(ma[a], ma[b]) for a, b in itertools.combinations(ma, 2)]
    to_olaf = [distance(olaf_ma, m) for m in ma.values()]
    print(f"{len(runs)} PyTorch runs, {n} steps, {window}-step moving averages")
    print(f"  PyTorch to PyTorch: mean relative distance {np.mean(between):.2e} (pairs: {', '.join(f'{d:.2e}' for d in between)})")
    print(f"  olaf to PyTorch:    mean relative distance {np.mean(to_olaf):.2e} (runs: {', '.join(f'{d:.2e}' for d in to_olaf)})")
    ratio = np.mean(to_olaf) / np.mean(between)
    print(f"  ratio {ratio:.2f} (limit {factor}): {'pass' if ratio <= factor else 'FAIL'}")


if __name__ == "__main__":
    which = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    OUT.mkdir(exist_ok=True)
    if which == "torch_checkpoints":
        run_torch_checkpoints(max(CHECKPOINTS) + 1)
    elif which == "truth":
        truth()
    elif which == "onestep":
        one_step()
    elif which == "curve":
        curve()
    elif which == "compare":
        compare()
        if (OUT / "loss_torch_again.npy").exists():
            compare("torch_again", "torch")
    else:
        start = time.time()
        losses = run_torch(steps) if which.startswith("torch") else run_olaf(steps)
        np.save(OUT / f"loss_{which}.npy", losses)
        print(f"{which}: {steps} steps in {time.time() - start:.0f} s, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
