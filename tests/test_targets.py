"""Programs build and compile (to PTX) for every described NVIDIA target, without the device.

Only the development machine's GPU can run anything; the other descriptors (A100, RTX 4090,
H100) come from documentation. Building every model's training step with each descriptor
exercises the decisions taken from it (split reductions by unit count, dot schedules,
shared memory), and compiling every task for its target catches code generation that would
only work for sm_86.
"""

import importlib

import numpy as np
import pytest

import olaf
from olaf.backend.capability import NVIDIA
from olaf.pipeline import lower_graph, to_program

# Every model with a training gate: the step each gate trains, in bf16 where the gate has a
# bf16 variant. Task kinds come with models (depthwise stencils, attention with a bias, the
# scan of Mamba-2), so a target is only covered when every model's tasks compile for it.
CASES = ["mlp", "cnn", "resnet", "llama", "mixtral", "whisper", "unet", "bert", "t5", "mobilenet", "efficientnet", "convnext", "vit", "dit", "mamba2", "gat", "rnn", "lstm"]


def case_step(name):
    def build(capability):
        c = importlib.import_module(f"test_{name}").case(olaf.bf16)
        traced, _ = olaf.train_step(c.olaf_loss, c.params, lambda t: olaf.AdamW(t), *c.input_specs, buffers=c.buffers)
        return to_program(lower_graph(traced.graph), capability)

    return build


def dcgan_step(capability):
    import test_dcgan
    from olaf.examples import dcgan

    g, d, b = test_dcgan.params(1)
    traced, _ = dcgan.gan_step(test_dcgan.CONFIG, g, d, b, test_dcgan.ADAM, test_dcgan.ADAM, test_dcgan.BATCH)
    return to_program(lower_graph(traced.graph), capability)


def gpt2_step(capability):
    from test_gpt2 import small_model, split_params
    from test_training import train_step

    model = small_model()
    traced, _ = train_step(split_params(model), model.config, 2, 16)
    return to_program(lower_graph(traced.graph), capability)


BUILDS = {**{name: case_step(name) for name in CASES}, "dcgan": dcgan_step, "gpt2": gpt2_step}


@pytest.mark.slow
@pytest.mark.parametrize("capability", NVIDIA, ids=lambda c: c.target)
@pytest.mark.parametrize("model", list(BUILDS))
def test_programs_compile_for_target(capability, model):
    from olaf.backend.cuda.runtime import compile_for

    program = BUILDS[model](capability)
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
