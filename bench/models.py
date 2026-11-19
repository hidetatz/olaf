"""Training step time of the P1.5 models, olaf against PyTorch eager and torch.compile.

Sizes are realistic for the RTX 3060 (the tests' configurations are so small that launch
overhead is all they show). Every model trains in bf16 autocast style with AdamW (DCGAN: two
Adams); Grad-CAM, an inference-time computation, times one Grad-CAM batch. olaf runs with the
default dot schedules unless --tuned (then the in-program schedule search runs first, its
results cached in ~/.cache/olaf). As docs/design.md prescribes: warm-up, then 5 timings of
STEPS steps, medians compared. Each model runs in its own process (GPU memory).

    python bench/models.py            # all models, one process each
    python bench/models.py resnet     # one model
    python bench/models.py resnet --tuned --olaf-only
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import olaf  # noqa: E402

STEPS = 10
TUNED = "--tuned" in sys.argv
OLAF_ONLY = "--olaf-only" in sys.argv
MODELS = ["mlp", "cnn", "resnet", "dcgan", "yolo", "llama", "mixtral", "whisper", "gradcam", "unet"]


def timed(run, sync) -> float:
    for _ in range(3):
        run()
    sync()
    times = []
    for _ in range(5):
        sync()
        begin = time.perf_counter()
        for _ in range(STEPS):
            run()
        sync()
        times.append((time.perf_counter() - begin) / STEPS * 1e3)
    return statistics.median(times)


# -- olaf ---------------------------------------------------------------------------------------


def olaf_time(traced, initial, inputs) -> tuple[float, float]:
    from olaf.backend.capability import AMPERE_RTX3060
    from olaf.backend.cuda.runtime import CudaProgram, Session
    from olaf.pipeline import lower_graph, to_program

    graph = lower_graph(traced.graph)
    schedules = None
    if TUNED:
        from olaf.search.tune import tune_in_program

        start = time.time()
        schedules = tune_in_program(graph, AMPERE_RTX3060, inputs, initial, log=lambda m: print(m, file=sys.stderr, flush=True))
        print(f"schedule search: {time.time() - start:.0f} s", file=sys.stderr, flush=True)
    start = time.time()
    compiled = CudaProgram(to_program(graph, AMPERE_RTX3060, schedules=schedules))
    compile_s = time.time() - start
    session = Session(compiled, {})
    session.load_state(initial)
    return timed(_replayer(session, inputs), compiled.stream.synchronize), compile_s


def _replayer(session, inputs):
    """Steps without the host upload of the inputs (PyTorch's side keeps them on the device
    too): two steps capture both parities' graphs, then only replays run."""
    session.step(inputs)
    session.step(inputs)

    def replay():
        session.graphs[session.parity].replay()
        session.parity = 1 - session.parity

    return replay


# -- torch --------------------------------------------------------------------------------------


def torch_times(step_fn) -> tuple[float, float, float]:
    """(eager ms, compiled ms, compile seconds) of step_fn() (forward, backward, optimizer).
    step_fn may be (eager step, compiled step) when compiling the whole step does not work."""
    step_fn, compiled = step_fn if isinstance(step_fn, tuple) else (step_fn, torch.compile(step_fn))
    eager = timed(step_fn, torch.cuda.synchronize)
    start = time.time()
    compiled()
    torch.cuda.synchronize()
    compile_s = time.time() - start
    return eager, timed(compiled, torch.cuda.synchronize), compile_s


def torch_train_step(model, loss_fn, params=None):
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=1e-4, fused=True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


# -- models -------------------------------------------------------------------------------------


def classifier(olaf_model, torch_model, x, y, buffers=None):
    """olaf (traced, initial, inputs) and the torch step, for a classifier on (x, y)."""
    params = olaf_model.init(np.random.default_rng(0))
    specs = (olaf.TensorType(x.shape, olaf.f32), olaf.TensorType(y.shape, olaf.i32))
    if buffers is None:

        def loss(p, x, y):
            return olaf.cross_entropy(olaf.cast(olaf_model(p, x), olaf.f32), y)

    else:

        def loss(p, b, x, y):
            logits, new = olaf_model(p, b, x)
            return olaf.cross_entropy(olaf.cast(logits, olaf.f32), y), new

    traced, initial = olaf.train_step(loss, params, lambda t: olaf.AdamW(t, lr=1e-4), *specs, buffers=buffers)
    xt, yt = torch.from_numpy(x).cuda(), torch.from_numpy(y).long().cuda()
    step = torch_train_step(torch_model, lambda m: torch.nn.functional.cross_entropy(m(xt).float(), yt))
    return (traced, initial, [x, y]), step


def build(name):
    rng = np.random.default_rng(0)
    from olaf import nn

    if name == "mlp":
        import test_mlp  # noqa: F401

        B = 256
        model = nn.Sequential(nn.Linear(784, 1024, compute=olaf.bf16), nn.relu, nn.Linear(1024, 1024, compute=olaf.bf16), nn.relu, nn.Linear(1024, 10, compute=olaf.bf16))
        tm = torch.nn.Sequential(torch.nn.Linear(784, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)).cuda()
        return "MLP 784-1024-1024-10, batch 256", *classifier(model, tm, rng.normal(size=(B, 784)).astype(np.float32), rng.integers(0, 10, B).astype(np.int32))
    if name == "cnn":
        import test_cnn

        B = 256
        return "CNN (MNIST, 2 conv + pool), batch 256", *classifier(test_cnn.cnn(olaf.bf16), test_cnn.torch_model(torch.float32, "cuda"), rng.normal(size=(B, 1, 28, 28)).astype(np.float32), rng.integers(0, 10, B).astype(np.int32))
    if name == "resnet":
        import test_resnet
        from olaf.examples import resnet

        test_resnet.CONFIG = resnet.Config(blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), num_classes=10, stem="cifar")
        B = 128
        model = resnet.ResNet(test_resnet.CONFIG, olaf.bf16)
        return "ResNet-18 (CIFAR stem, 32x32), batch 128", *classifier(model, test_resnet.TorchResNet().cuda(), rng.normal(size=(B, 3, 32, 32)).astype(np.float32), rng.integers(0, 10, B).astype(np.int32), buffers=model.init_buffers())
    if name == "dcgan":
        import test_dcgan
        from olaf.examples import dcgan

        config = dcgan.Config(nz=64, width=64, channels=3)
        test_dcgan.CONFIG = config
        B = 128
        g, d, b = dcgan.Generator(config).init(rng), dcgan.Discriminator(config).init(rng), dcgan.Generator(config).init_buffers()
        adam = lambda t, n: olaf.AdamW(t, lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0, name=n)  # noqa: E731
        traced, initial = dcgan.gan_step(config, g, d, b, adam, adam, B, compute=olaf.bf16)
        real = np.tanh(rng.normal(size=(B, 3, 16, 16))).astype(np.float32)
        z = rng.normal(size=(B, config.nz, 1, 1)).astype(np.float32)
        G0, D0 = test_dcgan.TorchG(config).cuda(), test_dcgan.TorchD(config).cuda()
        og, od = torch.optim.Adam(G0.parameters(), 2e-4, (0.5, 0.999), fused=True), torch.optim.Adam(D0.parameters(), 2e-4, (0.5, 0.999), fused=True)
        rt, zt = torch.from_numpy(real).cuda(), torch.from_numpy(z).cuda()
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        # torch.compile of the whole alternating step fails (its second backward reaches freed
        # saved tensors), so the compiled variant compiles the two modules instead.
        def make(G, D):
            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake = G(zt)
                    dr, df = D(rt).float(), D(fake.detach()).float()
                    dl = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
                od.zero_grad(set_to_none=True)
                dl.backward()
                od.step()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    go = D(fake).float()
                    gl = bce(go, torch.ones_like(go))
                og.zero_grad(set_to_none=True)
                gl.backward()
                og.step()

            return step


        return "DCGAN (16x16, width 64), batch 128", (traced, initial, [real, z]), (make(G0, D0), make(torch.compile(G0), torch.compile(D0)))
    if name == "yolo":
        import test_yolo
        from olaf.examples import yolo

        test_yolo.CONFIG = yolo.Config(widths=(32, 64, 128))
        B = 32
        model = yolo.Yolo(test_yolo.CONFIG, olaf.bf16)
        test_yolo.BATCH = B
        x, obj, box, cls = test_yolo.data(1)[0]
        params, buffers = model.init(rng), model.init_buffers()
        specs = (olaf.TensorType(x.shape, olaf.f32), olaf.TensorType(obj.shape, olaf.f32), olaf.TensorType(box.shape, olaf.f32), olaf.TensorType(cls.shape, olaf.i32))
        traced, initial = olaf.train_step(test_yolo.olaf_loss(model), params, lambda t: olaf.AdamW(t, lr=1e-4), *specs, buffers=buffers)
        tm = test_yolo.TorchYolo().cuda()
        args = [torch.from_numpy(a).cuda() for a in (x, obj, box)] + [torch.from_numpy(cls).long().cuda()]
        step = torch_train_step(tm, lambda m: test_yolo.torch_yolo_loss(m(args[0]), *args[1:]))
        return "YOLO (64x64, widths 32-64-128), batch 32", (traced, initial, [x, obj, box, cls]), step
    if name in ("llama", "mixtral"):
        import test_llama
        from transformers import LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM

        B, S = 4, 1024
        common = dict(vocab_size=32000, hidden_size=1024, num_attention_heads=16, num_key_value_heads=4, max_position_embeddings=S, tie_word_embeddings=False, attn_implementation="sdpa")
        if name == "llama":
            from olaf.examples import llama

            hf = LlamaConfig(**common, intermediate_size=2816, num_hidden_layers=8)
            model, tm, label = llama.Llama(llama.Config.from_huggingface(hf), olaf.bf16), LlamaForCausalLM(hf), "Llama (hidden 1024, 8 layers, GQA 16/4), batch 4 x 1024"
        else:
            from olaf.examples import mixtral

            hf = MixtralConfig(**common, intermediate_size=1408, num_hidden_layers=4, num_local_experts=8, num_experts_per_tok=2, router_jitter_noise=0.0)
            model, tm, label = mixtral.Mixtral(mixtral.Config.from_huggingface(hf), olaf.bf16), MixtralForCausalLM(hf), "Mixtral (hidden 1024, 4 layers, 8 experts top-2), batch 4 x 1024"
        ids = rng.integers(0, 32000, (B, S)).astype(np.int32)
        traced, initial = olaf.train_step(test_llama.olaf_loss(model), model.init(rng), lambda t: olaf.AdamW(t, lr=1e-4), olaf.TensorType((B, S), olaf.i32))
        tm = tm.cuda().train()
        step = torch_train_step(tm, lambda m: test_llama.torch_loss(m, ids))
        return label, (traced, initial, [ids]), step
    if name == "whisper":
        import test_whisper
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from olaf.examples import whisper

        hf = WhisperConfig(vocab_size=51865, num_mel_bins=80, d_model=384, encoder_layers=4, decoder_layers=4, encoder_attention_heads=6, decoder_attention_heads=6, encoder_ffn_dim=1536, decoder_ffn_dim=1536, max_source_positions=1500, max_target_positions=448, attn_implementation="sdpa")
        B, T = 8, 128
        model = whisper.Whisper(whisper.Config.from_huggingface(hf), olaf.bf16)
        features = rng.normal(size=(B, 80, 3000)).astype(np.float32)
        ids = rng.integers(3, 51865, (B, T)).astype(np.int32)
        specs = (olaf.TensorType(features.shape, olaf.f32), olaf.TensorType(ids.shape, olaf.i32))
        traced, initial = olaf.train_step(test_whisper.olaf_loss(model), model.init(rng), lambda t: olaf.AdamW(t, lr=1e-4), *specs)
        tm = WhisperForConditionalGeneration(hf).cuda().train()
        step = torch_train_step(tm, lambda m: test_whisper.torch_loss(m, features, ids), [p for p in tm.parameters() if p.requires_grad])
        return "Whisper tiny (30 s audio, 128 tokens), batch 8", (traced, initial, [features, ids]), step
    if name == "unet":
        import test_unet
        from olaf.examples import unet

        # A latent-diffusion-sized step: 4 x 32 x 32 latents, widths 128 / 256, 77 context
        # tokens of 256 (a text encoder's output), batch 16.
        config = unet.Config(channels=4, widths=(128, 256), groups=32, heads=4, context_dim=256, time_dim=128)
        test_unet.CONFIG = config
        B, S, L = 16, 32, 77
        model = unet.UNet(config, olaf.bf16)
        x = rng.normal(size=(B, 4, S, S)).astype(np.float32)
        t = rng.integers(0, 1000, B).astype(np.int32)
        context = rng.normal(size=(B, L, config.context_dim)).astype(np.float32)
        noise = rng.normal(size=(B, 4, S, S)).astype(np.float32)
        specs = tuple(olaf.TensorType(a.shape, olaf.i32 if a.dtype == np.int32 else olaf.f32) for a in (x, t, context, noise))
        traced, initial = olaf.train_step(unet.loss(model), model.init(rng), lambda p: olaf.AdamW(p, lr=1e-4), *specs)
        tm = test_unet.TorchUNet(config).cuda().train()
        step = torch_train_step(tm, lambda m: test_unet.torch_loss(m, x, t, context, noise))
        return f"U-Net (latents 4x{S}x{S}, widths 128-256, context {L}x256), batch {B}", (traced, initial, [x, t, context, noise]), step
    if name == "gradcam":
        import test_resnet
        from olaf.examples import resnet

        test_resnet.CONFIG = resnet.Config(blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), num_classes=1000, stem="imagenet")
        B = 32
        model = resnet.ResNet(test_resnet.CONFIG, olaf.bf16)
        params, buffers = model.init(rng), model.init_buffers()
        x = rng.normal(size=(B, 3, 224, 224)).astype(np.float32)
        classes = rng.integers(0, 1000, B).astype(np.int32)
        spec = lambda t: olaf.tree_map(lambda a: olaf.TensorType(a.shape, olaf.f32), t)  # noqa: E731
        traced = olaf.trace(lambda p, b, x, c: resnet.grad_cam(model, p, b, x, c), spec(params), spec(buffers), olaf.TensorType(x.shape, olaf.f32), olaf.TensorType(classes.shape, olaf.i32))
        inputs = [*olaf.tree_flatten(params)[0], *olaf.tree_flatten(buffers)[0], x, classes]
        tm = test_resnet.TorchResNet().cuda().eval()
        xt, ct = torch.from_numpy(x).cuda(), torch.from_numpy(classes).long().cuda()

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = torch.relu(tm.bn1(tm.conv1(xt)))
                h = torch.nn.functional.max_pool2d(h, 3, 2, 1)
                for i in range(4):
                    h = getattr(tm, f"layer{i + 1}")(h)
                h = h.detach().requires_grad_()
                logits = tm.fc(h.mean(dim=(2, 3)))
                (g,) = torch.autograd.grad(logits[torch.arange(B), ct].sum(), h)
                cam = torch.relu((h * g.mean(dim=(2, 3), keepdim=True)).sum(1, keepdim=True))
                torch.nn.functional.interpolate(cam.float(), size=(224, 224), mode="bilinear", align_corners=False)

        return "Grad-CAM on ResNet-18 (224x224), batch 32", (traced, {}, inputs), step
    raise KeyError(name)


def run_one(name):
    label, (traced, initial, inputs), step = build(name)
    ms, compile_s = olaf_time(traced, initial, inputs)
    eager, compiled_ms, torch_compile_s = (float("nan"),) * 3 if OLAF_ONLY else torch_times(step)
    print(f"RESULT|{name}|{label}|{ms:.2f}|{eager:.2f}|{compiled_ms:.2f}|{compile_s:.0f}|{torch_compile_s:.0f}", flush=True)


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if names and "--child" in flags:
        run_one(names[0])
        sys.exit()
    for name in names or MODELS:
        out = subprocess.run([sys.executable, __file__, name, "--child", *flags], capture_output=True, text=True, timeout=7200)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT|")), None)
        if line is None:
            print(f"{name}: failed\n{out.stderr[-3000:]}", flush=True)
            continue
        _, n, label, ms, eager, comp, cs, tcs = line.split("|")
        print(f"{label}: olaf {ms} ms (compile {cs} s), torch eager {eager} ms, torch.compile {comp} ms (compile {tcs} s); olaf / eager {float(eager) / float(ms):.2f}x, olaf / compile {float(comp) / float(ms):.2f}x", flush=True)
