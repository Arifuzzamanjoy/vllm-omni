# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LoRA on a layer quantized by the native online FP8 path (real CUDA kernel)."""

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.lora.layers import DiffusionReplicatedLinearWithLoRA
from vllm_omni.diffusion.lora.utils import from_layer_diffusion
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_lora_adds_delta_over_online_fp8_linear():
    config = build_quant_config("fp8")
    od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16, quantization_config=config)
    vllm_config = create_diffusion_vllm_config(torch.device("cuda:0"), od_config)

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        with (
            set_current_vllm_config(vllm_config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda:0"),
            torch.no_grad(),
        ):
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            layer = ReplicatedLinear(
                128, 128, bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="linear", disable_tp=True
            )
            assert isinstance(layer.quant_method, Fp8PerTensorOnlineLinearMethod)

            generator = torch.Generator(device="cuda").manual_seed(42)
            weights = torch.randn(128, 128, generator=generator, dtype=torch.bfloat16) / 128**0.5
            layer.weight.weight_loader(layer.weight, weights)
            assert layer.weight.dtype == torch.float8_e4m3fn

            # The wrapper is created after quantization, as when an adapter first arrives.
            lora_config = LoRAConfig(max_lora_rank=16, max_loras=1, max_cpu_loras=1, lora_dtype=torch.bfloat16)
            wrapped = from_layer_diffusion(layer, 1, lora_config, [], None)
            assert isinstance(wrapped, DiffusionReplicatedLinearWithLoRA)
            assert all(t.device.type == "cuda" and t.dtype == torch.bfloat16 for t in wrapped.lora_a_stacked)

            lora_a = torch.randn(8, 128, generator=generator, dtype=torch.bfloat16) * 0.1
            lora_b = torch.randn(128, 8, generator=generator, dtype=torch.bfloat16) * 0.1
            wrapped.set_lora(0, lora_a, lora_b)

            inputs = torch.randn(32, 128, generator=generator, dtype=torch.bfloat16)
            base = layer.quant_method.apply(layer, inputs, None).float()
            delta = (inputs.float() @ lora_a.float().t()) @ lora_b.float().t()
            expected = base + delta

            actual = wrapped.apply(inputs)

            assert actual.dtype == torch.bfloat16
            relative_mse = (actual.float() - expected).square().mean() / expected.square().mean()
            assert relative_mse.item() < 0.002
            # The adapter must visibly change the FP8 output.
            assert delta.square().mean() / base.square().mean() > 0.01

            wrapped.suspend_lora()
            torch.testing.assert_close(wrapped.apply(inputs).float(), base, rtol=0, atol=0)
            wrapped.resume_lora()
            torch.testing.assert_close(wrapped.apply(inputs), actual, rtol=0, atol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
