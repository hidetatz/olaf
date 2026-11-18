"""DCGAN (Radford et al. 2016) for small images: a generator of transposed convolutions with
batch norm and ReLU, a discriminator of strided convolutions with leaky ReLU, and the training
step that alternates them (`gan_step`).

Sizes: z [batch, nz, 1, 1] -> 4x4 -> 8x8 -> 16x16 images with `channels` channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.frontend.optim import parameter_states, read_all, write_all
from olaf.graph.types import DType, StateKind, TensorType


@dataclass(frozen=True)
class Config:
    nz: int = 16
    width: int = 32
    channels: int = 1


@dataclass(frozen=True)
class Generator(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        return {
            "convt1": nn.ConvTranspose(c.nz, 2 * c.width, (4, 4), 1, 0, bias=False, compute=self.compute),
            "bn1": nn.BatchNorm(2 * c.width),
            "convt2": nn.ConvTranspose(2 * c.width, c.width, (4, 4), 2, 1, bias=False, compute=self.compute),
            "bn2": nn.BatchNorm(c.width),
            "convt3": nn.ConvTranspose(c.width, c.channels, (4, 4), 2, 1, bias=False, compute=self.compute),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def init_buffers(self):
        return {name: layer.init_buffers() for name, layer in self._layers().items() if isinstance(layer, nn.BatchNorm)}

    def __call__(self, p, buffers, z):
        L, new = self._layers(), {}
        h = z
        for i in (1, 2):
            h, new[f"bn{i}"] = L[f"bn{i}"](p[f"bn{i}"], L[f"convt{i}"](p[f"convt{i}"], h), buffers[f"bn{i}"])
            h = nn.relu(h)
        return olaf.tanh(L["convt3"](p["convt3"], h)), new


@dataclass(frozen=True)
class Discriminator(nn.Module):
    config: Config
    compute: DType | None = None

    def _layers(self):
        c = self.config
        return {
            "conv1": nn.Conv(c.channels, c.width, (4, 4), 2, 1, compute=self.compute),
            "conv2": nn.Conv(c.width, 2 * c.width, (4, 4), 2, 1, compute=self.compute),
            "conv3": nn.Conv(2 * c.width, 1, (4, 4), 1, 0, compute=self.compute),
        }

    def init(self, rng):
        return {name: layer.init(rng) for name, layer in self._layers().items()}

    def __call__(self, p, x):
        L = self._layers()
        h = nn.leaky_relu(L["conv1"](p["conv1"], x), 0.2)
        h = nn.leaky_relu(L["conv2"](p["conv2"], h), 0.2)
        out = L["conv3"](p["conv3"], h).reshape(x.shape[0])
        return olaf.cast(out, DType.f32) if out.dtype is DType.bf16 else out


def gan_step(config: Config, g_params, d_params, g_buffers, make_g_opt, make_d_opt, batch: int, size: int = 16, dtype: DType = DType.f32, compute: DType | None = None):
    """The traced alternating step, as PyTorch's DCGAN loop: generate once; update the
    discriminator on real and (detached) generated images; then update the generator through
    the updated discriminator. Inputs (real images, z); returns (d_loss, g_loss).
    Returns the traced step and its initial state."""
    G, D = Generator(config, compute), Discriminator(config, compute)
    types = lambda tree: olaf.tree_map(lambda a: TensorType(np.shape(a), dtype), tree)  # noqa: E731
    g_states, g_paths = parameter_states(types(g_params), "g")
    d_states, d_paths = parameter_states(types(d_params), "d")
    from olaf.frontend.optim import _paths

    b_types = types(g_buffers)
    b_leaves, b_structure = olaf.tree_flatten(b_types)
    b_states = olaf.tree_unflatten(b_structure, [olaf.state(f"g_buffers.{p}", t, StateKind.buffer) for p, t in zip(_paths(b_types), b_leaves)])
    g_opt, d_opt = make_g_opt(types(g_params), "g_adam"), make_d_opt(types(d_params), "d_adam")
    bce = nn.binary_cross_entropy_with_logits

    def step(real, z):
        pg, pd, held = read_all(g_states), read_all(d_states), read_all(b_states)
        fake, new_buffers = G(pg, held, z)
        d_loss, d_grads = olaf.value_and_grad(lambda pd: bce(D(pd, real), 1.0) + bce(D(pd, fake), 0.0))(pd)
        pd = d_opt.update(pd, d_grads)
        write_all(d_states, pd)
        g_loss, g_grads = olaf.value_and_grad(lambda pg: bce(D(pd, G(pg, held, z)[0]), 1.0))(pg)
        write_all(g_states, g_opt.update(pg, g_grads))
        write_all(b_states, new_buffers)
        return d_loss, g_loss

    traced = olaf.trace(step, TensorType((batch, config.channels, size, size), dtype), TensorType((batch, config.nz, 1, 1), dtype))
    initial = {}
    for prefix, tree in (("g", g_params), ("d", d_params), ("g_buffers", g_buffers)):
        initial.update({f"{prefix}.{p}": np.asarray(v, np.float64 if dtype is DType.f64 else np.float32) for p, v in zip(_paths(tree), olaf.tree_flatten(tree)[0])})
    for opt in (g_opt, d_opt):
        # Moments in the parameters' dtype; the step count stays an f32 scalar.
        initial.update({k: v if k.endswith(".step") else v.astype(np.float64 if dtype is DType.f64 else np.float32) for k, v in opt.initial_state().items()})
    return traced, initial
