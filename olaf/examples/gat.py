"""GAT (Velickovic et al. 2018), as PyTorch Geometric's GATConv stack for node classification:
each layer projects the nodes, scores every edge j -> i with LeakyReLU(a_src . W x_j + a_dst .
W x_i), normalizes the scores over each node's incoming edges (a softmax per destination), and
sums the projected sources weighted by them. Self-loops are added; hidden layers concatenate
their heads and pass through ELU, the last averages them.

Message passing is a gather of edge endpoints (take) and a scatter_add into destinations: the
graph is data (edge lists of static length), not structure in the program.

The per-destination softmax subtracts the per-head maximum over all edges, not each
destination's own maximum as PyG does (olaf has no scatter-max): a softmax is unchanged by any
constant shift of a segment, so the two agree exactly in arithmetic. It only differs when one
destination's scores all lie ~80 (f32) below the global maximum and underflow; GAT's scores,
LeakyReLU of sums of two projections, stay far from that.

Under bf16 compute only the projections run in bf16: as torch.autocast does (exp is an f32 op
and a bf16 * f32 product promotes), everything after them is f32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import olaf
from olaf import nn
from olaf.graph.types import DType


@dataclass(frozen=True)
class Config:
    in_features: int = 128
    hidden: int = 64
    heads: int = 4
    num_layers: int = 3
    num_classes: int = 40


def _to(x, dtype):
    return x if dtype is None or x.dtype is dtype else olaf.cast(x, dtype)


def _wide(x):
    return _to(x, DType.f32) if x.dtype is DType.bf16 else x


def elu(x):
    return olaf.where(x > 0.0, x, olaf.exp(olaf.minimum(x, 0.0)) - 1.0)


@dataclass(frozen=True)
class GATConv(nn.Module):
    in_features: int
    out_features: int
    heads: int
    concat: bool
    compute: DType | None = None

    def init(self, rng):
        H, C = self.heads, self.out_features
        glorot = lambda shape, fan: (rng.uniform(-1, 1, shape) * np.sqrt(6.0 / fan)).astype(np.float32)  # noqa: E731
        return {
            "lin": {"weight": glorot((H * C, self.in_features), self.in_features + H * C)},
            "att_src": glorot((1, H, C), H + C),
            "att_dst": glorot((1, H, C), H + C),
            "bias": np.zeros(H * C if self.concat else C, np.float32),
        }

    def __call__(self, p, x, src, dst):
        """x [N, in] -> [N, heads * out] (concat) or [N, out]; src, dst [E] include self-loops."""
        N, H, C, E = x.shape[0], self.heads, self.out_features, src.shape[0]
        w = _to(p["lin"]["weight"], self.compute)
        h = _wide(olaf.einsum("ni,oi->no", _to(x, self.compute), w)).reshape(N, H, C)

        def score(att):
            return olaf.sum(h * olaf.broadcast_to(_to(att, h.dtype), h.shape), axis=-1)  # [N, H]

        e = nn.leaky_relu(olaf.take(score(p["att_src"]), src) + olaf.take(score(p["att_dst"]), dst), 0.2)  # [E, H]
        top = olaf.max(e, axis=0).reshape(1, H)
        weights = olaf.exp(e - olaf.broadcast_to(top, e.shape))
        totals = olaf.scatter_add(olaf.zeros((N, H), weights.dtype), dst, weights)
        alpha = weights / olaf.take(totals, dst)
        messages = olaf.take(h, src) * olaf.broadcast_to(alpha.reshape(E, H, 1), (E, H, C))
        out = olaf.scatter_add(olaf.zeros((N, H, C), messages.dtype), dst, messages)
        out = out.reshape(N, H * C) if self.concat else olaf.mean(out, axis=1)
        return out + olaf.broadcast_to(_to(p["bias"], out.dtype).reshape(1, out.shape[1]), out.shape)


@dataclass(frozen=True)
class GAT(nn.Module):
    config: Config
    compute: DType | None = None

    def _convs(self):
        c, k = self.config, self.compute
        dims = [c.in_features] + [c.hidden * c.heads] * (c.num_layers - 1)
        hidden = [GATConv(d, c.hidden, c.heads, True, k) for d in dims[:-1]]
        return hidden + [GATConv(dims[-1], c.num_classes, 1, False, k)]

    def init(self, rng):
        return {"convs": [conv.init(rng) for conv in self._convs()]}

    def __call__(self, params, x, src, dst):
        """Logits [N, classes] from node features [N, in] and edges src -> dst [E] (self-loops
        not included: added here, as GATConv does)."""
        N = x.shape[0]
        loops = olaf.iota((N,), 0, src.dtype)
        src, dst = olaf.concat([src, loops], axis=0), olaf.concat([dst, loops], axis=0)
        convs = self._convs()
        for i, (conv, p) in enumerate(zip(convs, params["convs"])):
            x = conv(p, x, src, dst)
            if i < len(convs) - 1:
                x = elu(x)
        return x


def loss(model: GAT):
    def fn(params, x, src, dst, labels):
        return olaf.cross_entropy(_wide(model(params, x, src, dst)), labels)

    return fn
