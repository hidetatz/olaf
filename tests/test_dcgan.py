"""P1.5 model 6: DCGAN (transposed convolutions; two optimizers updated in turn in one step)."""

import numpy as np
import pytest
import torch
import torch.nn as T

import olaf
from olaf.examples import dcgan

from model_gate import FLOOR, MULTIPLIER, device_program, load, relative

BATCH, STEPS = 16, 200
CONFIG = dcgan.Config()
BETAS = (0.5, 0.999)


class TorchG(T.Module):
    def __init__(self, c):
        super().__init__()
        self.convt1, self.bn1 = T.ConvTranspose2d(c.nz, 2 * c.width, 4, 1, 0, bias=False), T.BatchNorm2d(2 * c.width)
        self.convt2, self.bn2 = T.ConvTranspose2d(2 * c.width, c.width, 4, 2, 1, bias=False), T.BatchNorm2d(c.width)
        self.convt3 = T.ConvTranspose2d(c.width, c.channels, 4, 2, 1, bias=False)

    def forward(self, z):
        h = torch.relu(self.bn1(self.convt1(z)))
        h = torch.relu(self.bn2(self.convt2(h)))
        return torch.tanh(self.convt3(h))


class TorchD(T.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = T.Conv2d(c.channels, c.width, 4, 2, 1), T.Conv2d(c.width, 2 * c.width, 4, 2, 1), T.Conv2d(2 * c.width, 1, 4, 1, 0)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        h = torch.nn.functional.leaky_relu(self.conv2(h), 0.2)
        return self.conv3(h).reshape(x.shape[0])


def named(tree):
    from olaf.frontend.optim import _paths

    return dict(zip(_paths(tree), olaf.tree_flatten(tree)[0]))


def torch_losses(g_params, d_params, batches, dtype, device):
    G, D = TorchG(CONFIG).to(dtype=dtype, device=device), TorchD(CONFIG).to(dtype=dtype, device=device)
    load(G, named(g_params))
    load(D, named(d_params))
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=BETAS)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    out = []
    for real, z in batches:
        real, z = (torch.from_numpy(a).to(dtype=dtype, device=device) for a in (real, z))
        fake = G(z)
        d_real, d_fake = D(real), D(fake.detach())
        d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        g_out = D(fake)
        g_loss = bce(g_out, torch.ones_like(g_out))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        out.extend([d_loss.item(), g_loss.item()])
    return out


def params(seed):
    rng = np.random.default_rng(seed)
    return dcgan.Generator(CONFIG).init(rng), dcgan.Discriminator(CONFIG).init(rng), dcgan.Generator(CONFIG).init_buffers()


def data(steps, seed=0):
    """Real images: four fixed 16x16 patterns with noise; z from a normal."""
    rng = np.random.default_rng(seed)
    patterns = np.tanh(rng.normal(size=(4, 1, 16, 16)) * 2.0)
    out = []
    for _ in range(steps):
        real = patterns[rng.integers(0, 4, BATCH)] + 0.05 * rng.normal(size=(BATCH, 1, 16, 16))
        out.append((real.astype(np.float32), rng.normal(size=(BATCH, CONFIG.nz, 1, 1)).astype(np.float32)))
    return out


def olaf_losses(compiled, initial, batches):
    from olaf.backend.cuda.runtime import Session

    session = Session(compiled, {})
    session.load_state(initial)
    out = []
    for real, z in batches:
        session.step([real, z])
        out.extend(float(v) for v in session.outputs())
    return out


ADAM = lambda types, name: olaf.AdamW(types, lr=2e-4, betas=BETAS, weight_decay=0.0, name=name)  # noqa: E731


def test_step_matches_torch_eager_f64():
    g, d, b = params(1)
    traced, state = dcgan.gan_step(CONFIG, g, d, b, ADAM, ADAM, BATCH, dtype=olaf.f64)
    from olaf.pipeline import lower_graph
    from olaf.graph.interp import interpret

    graph = lower_graph(traced.graph)
    batches = data(3)
    ours = []
    for real, z in batches:
        losses, state = interpret(graph, [real.astype(np.float64), z.astype(np.float64)], state)
        ours.extend(float(v) for v in losses)
    want = torch_losses(g, d, batches, torch.float64, "cpu")
    np.testing.assert_allclose(ours, want, rtol=1e-10)


@pytest.mark.model
@pytest.mark.gpu
def test_training_gate():
    """Both losses over the run, against f64 PyTorch, over five seeds (see tests/model_gate.py)."""
    compiled, d_ours, d_theirs = None, [], []
    for seed in range(5):
        g, d, b = params(seed + 1)
        traced, initial = dcgan.gan_step(CONFIG, g, d, b, ADAM, ADAM, BATCH)
        compiled = compiled or device_program(traced.graph)
        batches = data(STEPS, seed)
        truth = torch_losses(g, d, batches, torch.float64, "cpu")
        d_ours.append(relative(olaf_losses(compiled, initial, batches), truth))
        d_theirs.append(relative(torch_losses(g, d, batches, torch.float32, "cuda"), truth))
    ratio = float(np.median(d_ours)) / max(float(np.median(d_theirs)), FLOOR)
    assert ratio <= MULTIPLIER, (ratio, d_ours, d_theirs)
