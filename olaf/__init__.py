"""olaf: a deep learning framework that handles training and inference with one compiler."""

from olaf.frontend.trace import (
    Step,
    Tensor,
    Traced,
    abs,
    as_tensor,
    attention,
    broadcast_to,
    cast,
    concat,
    cond,
    constant,
    cos,
    cross_entropy,
    einsum,
    equal,
    erf,
    exp,
    full,
    gelu,
    grad,
    iota,
    layernorm,
    log,
    matmul,
    max,
    maximum,
    mean,
    minimum,
    neg,
    ones,
    pad,
    permute,
    recip,
    repeat,
    reshape,
    rng_uniform,
    rsqrt,
    scan,
    scatter_add,
    silu,
    sin,
    softmax,
    sqrt,
    state,
    step,
    sum,
    take,
    tanh,
    trace,
    transpose,
    tree_flatten,
    tree_map,
    tree_unflatten,
    value_and_grad,
    where,
    while_loop,
    zeros,
)
from olaf.frontend import nn
from olaf.frontend.optim import SGD, AdamW, params_from_state, train_step
from olaf.graph.dim import Dim, sym
from olaf.graph.types import DType, Sharding, StateDecl, StateKind, TensorType

f32 = DType.f32
f64 = DType.f64
bf16 = DType.bf16
i32 = DType.i32
i64 = DType.i64
u32 = DType.u32
