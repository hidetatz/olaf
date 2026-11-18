"""Programs build and compile (to PTX) for every described NVIDIA target, without the device.

Only the development machine's GPU can run anything; the other descriptors (A100, RTX 4090,
H100) come from documentation. Building the training steps of two models with each descriptor
exercises the decisions taken from it (split reductions by unit count, dot schedules,
shared memory), and compiling every task for its target catches code generation that would
only work for sm_86.
"""

import numpy as np
import pytest

import olaf
from olaf.backend.capability import NVIDIA
from olaf.examples import mixtral
from olaf.pipeline import lower_graph, to_program

import test_cnn
import test_mixtral


def mixtral_step(capability):
    model = mixtral.Mixtral(test_mixtral.CONFIG, olaf.bf16)
    traced, _ = olaf.train_step(test_mixtral.case(olaf.bf16).olaf_loss, model.init(np.random.default_rng(0)), lambda t: olaf.AdamW(t), *test_mixtral.SPECS)
    return to_program(lower_graph(traced.graph), capability)


def cnn_step(capability):
    c = test_cnn.case(olaf.bf16)
    traced, _ = olaf.train_step(c.olaf_loss, c.params, lambda t: olaf.SGD(t, lr=0.1, momentum=0.9), *c.input_specs)
    return to_program(lower_graph(traced.graph), capability)


@pytest.mark.slow
@pytest.mark.parametrize("capability", NVIDIA, ids=lambda c: c.target)
@pytest.mark.parametrize("build", [mixtral_step, cnn_step], ids=["mixtral", "cnn"])
def test_programs_compile_for_target(capability, build):
    from olaf.backend.cuda.runtime import compile_for

    program = build(capability)
    ptx = compile_for(program, capability)
    assert len(ptx) == len(program.tasks())
    for kernels in ptx.values():
        for text in kernels.values():
            assert f".target {capability.target}" in text


def test_descriptors_change_decisions():
    # The unit count decides how column reductions split: 28 SMs and 132 do not split alike.
    from olaf.backend.capability import AMPERE_RTX3060, HOPPER_H100

    graph = lower_graph(olaf.trace(lambda x: olaf.sum(x, axis=(0, 1)), olaf.TensorType((2, 4096, 768), olaf.bf16)).graph)
    grids = {cap.target: [l.task.grid for l in to_program(graph, cap).launches if l.task.name == "reduce_sum"] for cap in (AMPERE_RTX3060, HOPPER_H100)}
    assert grids["sm_86"] != grids["sm_90"], grids
