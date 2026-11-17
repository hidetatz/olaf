"""GPT-2 small training throughput: olaf against torch.compile(mode="max-autotune"), the P1 gate.

Both sides train the same configuration: batch 4, sequence 1024, f32 master weights with bf16
autocast-style compute, AdamW (lr 6e-4, betas 0.9/0.95, weight decay 0.1 on matrices), no dropout.
Each side warms up, then 5 timings of STEPS steps are taken and the medians compared, as
docs/design.md prescribes. The GPU is shared with the display: run the two back to back.

    python bench/gpt2_train.py [torch|olaf|olaf-tuned|both]

olaf-tuned first searches dot schedules (search/tune.py, cached in ~/.cache/olaf).
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

BATCH, SEQ, STEPS = 4, 1024, 10
LR, BETAS, WD = 6e-4, (0.9, 0.95), 0.1


def data(steps: int, vocab: int = 50257):
    rng = np.random.default_rng(0)
    return [rng.integers(0, vocab, (BATCH, SEQ)).astype(np.int32) for _ in range(steps)]


def torch_tokens_per_s() -> float:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(attn_implementation="sdpa", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)).cuda().train()
    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": WD},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR, betas=BETAS, fused=True)

    def loss_fn(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))

    compiled = torch.compile(loss_fn, mode="max-autotune")
    batches = [torch.from_numpy(b).long().cuda() for b in data(STEPS)]

    def step(ids):
        loss = compiled(ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    start = time.time()
    for ids in batches[:3]:
        step(ids)
    torch.cuda.synchronize()
    print(f"torch.compile warm-up (compile included): {time.time() - start:.0f} s", flush=True)
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for ids in batches:
            step(ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - begin) / STEPS)
    ms = statistics.median(times) * 1e3
    print(f"torch.compile max-autotune: {ms:.1f} ms/step, {BATCH * SEQ / ms * 1e3:.0f} tokens/s  {[round(t * 1e3, 1) for t in times]}", flush=True)
    return BATCH * SEQ / ms * 1e3


def olaf_tokens_per_s(tuned: bool = False) -> float:
    from transformers import GPT2Config, GPT2LMHeadModel
    from test_gpt2 import split_params
    from test_training import lower, train_step

    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram, Session
    from olaf.pipeline import to_program

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(attn_implementation="eager")).eval()
    params = split_params(model)
    traced, initial = train_step(params, model.config, BATCH, SEQ)
    graph = lower(traced.graph)
    schedules = None
    if tuned:
        from olaf.search.tune import tune_in_program

        start = time.time()
        schedules = tune_in_program(graph, AMPERE_RTX3060, data(1), initial)
        print(f"olaf schedule search: {time.time() - start:.0f} s", flush=True)
    start = time.time()
    compiled = CudaProgram(to_program(graph, AMPERE_RTX3060, schedules=schedules))
    print(f"olaf compile: {time.time() - start:.0f} s, {len(compiled.program.launches)} launches, {len(compiled.kernels)} unique tasks", flush=True)
    session = Session(compiled, {})
    session.load_state(initial)
    batches = data(STEPS)
    for ids in batches[:3]:
        session.step([ids])
    compiled.stream.synchronize()
    times = []
    for _ in range(5):
        compiled.stream.synchronize()
        begin = time.perf_counter()
        for ids in batches:
            session.step([ids])
        compiled.stream.synchronize()
        times.append((time.perf_counter() - begin) / STEPS)
    ms = statistics.median(times) * 1e3
    print(f"olaf: {ms:.1f} ms/step, {BATCH * SEQ / ms * 1e3:.0f} tokens/s  {[round(t * 1e3, 1) for t in times]}", flush=True)
    return BATCH * SEQ / ms * 1e3


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = {}
    if which in ("torch", "both"):
        results["torch"] = torch_tokens_per_s()
    if which in ("olaf", "both"):
        results["olaf"] = olaf_tokens_per_s()
    if which == "olaf-tuned":
        results["olaf"] = olaf_tokens_per_s(tuned=True)
    if len(results) == 2:
        print(f"olaf / torch.compile = {results['olaf'] / results['torch']:.2f}")
