# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LoRA on real vLLM linear layers combined with FP8 weights and CPU offload hooks.

The tests build real ``ReplicatedLinear`` / ``QKVParallelLinear`` layers and wrap
them through the diffusion LoRA manager. CPU cannot run the CUDA FP8 kernels, so
the FP8 layers keep ``float8_e4m3fn`` weights and dequantize inside a stand-in
``quant_method.apply``; the LoRA wrapper only sees ``quant_method.apply``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear

from tests.diffusion.offloader.helpers import DummyStream, patch_offload_runtime
from vllm_omni.diffusion.lora.layers import (
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.lora.utils import from_layer_diffusion
from vllm_omni.diffusion.offloader import layerwise_backend as layerwise_backend_module
from vllm_omni.diffusion.offloader.sequential_backend import (
    SequentialOffloadHook,
    apply_sequential_offload,
)
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 16
HEAD_SIZE = 4
NUM_HEADS = 2
QKV_SLICES = (NUM_HEADS * HEAD_SIZE,) * 3
CPU = torch.device("cpu")


def _cpu_vllm_config() -> VllmConfig:
    # An explicit device keeps config construction independent of accelerator discovery.
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


@pytest.fixture(autouse=True)
def _cpu_linear_env(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    with torch.no_grad():
        yield


class _Fp8DequantMethod:
    """Stand-in for an FP8 linear method: fp8 weight, per-tensor scale, dequantize in apply()."""

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
        return F.linear(x, weight, bias)


def _fill(layer: torch.nn.Module, generator: torch.Generator, *, fp8: bool) -> None:
    weight = torch.randn(layer.weight.shape, generator=generator) / HIDDEN**0.5
    if fp8:
        scale = weight.abs().max() / 448.0
        layer.weight = torch.nn.Parameter((weight / scale).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.quant_method = _Fp8DequantMethod()
    else:
        layer.weight.data.copy_(weight)


def _make_replicated(out_features: int, dtype: torch.dtype, seed: int, *, fp8: bool) -> ReplicatedLinear:
    with set_current_vllm_config(_cpu_vllm_config()):
        layer = ReplicatedLinear(
            HIDDEN, out_features, bias=False, params_dtype=dtype, disable_tp=True, prefix=f"proj{seed}"
        )
    _fill(layer, torch.Generator().manual_seed(seed), fp8=fp8)
    return layer


def _make_qkv(dtype: torch.dtype, seed: int, *, fp8: bool) -> QKVParallelLinear:
    with set_current_vllm_config(_cpu_vllm_config()):
        layer = QKVParallelLinear(
            HIDDEN, HEAD_SIZE, NUM_HEADS, bias=False, params_dtype=dtype, disable_tp=True, prefix=f"qkv{seed}"
        )
    _fill(layer, torch.Generator().manual_seed(seed), fp8=fp8)
    return layer


def _lora_config(rank: int, dtype: torch.dtype) -> LoRAConfig:
    return LoRAConfig(max_lora_rank=rank, max_loras=1, max_cpu_loras=1, lora_dtype=dtype)


def _lora_pair(rank: int, out_features: int, dtype: torch.dtype, seed: int):
    generator = torch.Generator().manual_seed(seed)
    lora_a = (torch.randn(rank, HIDDEN, generator=generator) * 0.3).to(dtype)
    lora_b = (torch.randn(out_features, rank, generator=generator) * 0.3).to(dtype)
    return lora_a, lora_b


def _delta(x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return ((x.float() @ lora_a.float().t()) @ lora_b.float().t()) * scale


@pytest.mark.parametrize("kind", ["replicated", "qkv"])
@pytest.mark.parametrize("fp8", [True, False])
def test_lora_over_quantized_linear_adds_delta_to_base_output(kind, fp8):
    dtype = torch.bfloat16
    rank = 4
    if kind == "replicated":
        layer = _make_replicated(24, dtype, seed=1, fp8=fp8)
        slices: tuple[int, ...] = (24,)
        packed: list[str] = []
        expected_cls = DiffusionReplicatedLinearWithLoRA
    else:
        layer = _make_qkv(dtype, seed=1, fp8=fp8)
        slices = QKV_SLICES
        packed = ["q", "k", "v"]
        expected_cls = DiffusionMergedQKVParallelLinearWithLoRA
    if fp8:
        assert layer.weight.dtype == torch.float8_e4m3fn

    wrapped = from_layer_diffusion(layer, 1, _lora_config(8, dtype), packed, None)

    assert isinstance(wrapped, expected_cls)
    assert all(t.device == CPU and t.dtype == dtype for t in (*wrapped.lora_a_stacked, *wrapped.lora_b_stacked))

    pairs = [_lora_pair(rank, size, dtype, seed=10 + i) for i, size in enumerate(slices)]
    if kind == "replicated":
        wrapped.set_lora(0, pairs[0][0], pairs[0][1])
    else:
        wrapped.set_lora(0, [a for a, _ in pairs], [b for _, b in pairs])

    x = torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(3)).to(dtype)
    base = layer.quant_method.apply(layer, x, None).float()
    expected = base + torch.cat([_delta(x, a, b) for a, b in pairs], dim=-1)

    actual = wrapped.apply(x)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
    assert not torch.allclose(actual.float(), base, atol=3e-2)


class _Pipeline(torch.nn.Module):
    """Pipeline with one plain and one packed projection under ``transformer``."""

    def __init__(self, *, fp8: bool, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.proj = _make_replicated(24, dtype, seed=21, fp8=fp8)
        self.transformer.qkv = _make_qkv(dtype, seed=22, fp8=fp8)


class _Adapter:
    def __init__(self, adapter_id: int, rank: int, dtype: torch.dtype, block_prefix: str = "transformer"):
        self.id = adapter_id
        self.rank = rank
        self.loras = {}
        for name, out_features, seed in (("proj", 24, 1), ("qkv", sum(QKV_SLICES), 2)):
            lora_a, lora_b = _lora_pair(rank, out_features, dtype, seed=adapter_id * 100 + seed)
            full_name = f"{block_prefix}.{name}"
            self.loras[full_name] = LoRALayerWeights(
                module_name=full_name, rank=rank, lora_alpha=rank, lora_a=lora_a, lora_b=lora_b
            )

    def get_lora(self, name: str):
        return self.loras.get(name)


def _manager(
    pipeline: torch.nn.Module, adapters: dict[int, Any], monkeypatch, device: torch.device = CPU
) -> DiffusionLoRAManager:
    manager = DiffusionLoRAManager(pipeline=pipeline, device=device, dtype=torch.float32, max_cached_adapters=4)

    def load_adapter(request: LoRARequest):
        adapter = adapters[request.lora_int_id]
        return adapter, SimpleNamespace(r=adapter.rank, target_modules=None, lora_alpha=adapter.rank)

    monkeypatch.setattr(manager, "_load_adapter", load_adapter)
    return manager


def _request(adapter_id: int) -> LoRARequest:
    return LoRARequest(lora_name=f"adapter{adapter_id}", lora_int_id=adapter_id, lora_path=f"/unused/{adapter_id}")


def _apply(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, BaseLayerWithLoRA):
        return layer.apply(x)
    return layer.quant_method.apply(layer, x, None)


def _run(pipeline: _Pipeline, x: torch.Tensor) -> torch.Tensor:
    transformer = pipeline.transformer
    return torch.cat([_apply(transformer.proj, x), _apply(transformer.qkv, x)], dim=-1)


def _expected(pipeline_base: _Pipeline, adapter: _Adapter | None, x: torch.Tensor, scale: float) -> torch.Tensor:
    base = _run(pipeline_base, x)
    if adapter is None:
        return base
    lora_a, lora_b = adapter.loras["transformer.proj"].lora_a, adapter.loras["transformer.proj"].lora_b
    proj_delta = _delta(x, lora_a, lora_b, scale)
    lora_a, lora_b = adapter.loras["transformer.qkv"].lora_a, adapter.loras["transformer.qkv"].lora_b
    qkv_delta = _delta(x, lora_a, lora_b, scale)
    return base + torch.cat([proj_delta, qkv_delta], dim=-1)


@pytest.mark.parametrize("fp8", [False, True])
def test_adapter_switch_and_scale_change_match_fresh_activation(fp8, monkeypatch):
    adapters = {
        1: _Adapter(1, rank=4, dtype=torch.float32),
        2: _Adapter(2, rank=8, dtype=torch.float32),
        # Larger than the first allocation, so activating it re-allocates every buffer.
        3: _Adapter(3, rank=32, dtype=torch.float32),
    }
    x = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(9))
    reference = _Pipeline(fp8=fp8)
    pipeline = _Pipeline(fp8=fp8)
    manager = _manager(pipeline, adapters, monkeypatch)

    def fresh(adapter_id: int, scale: float) -> torch.Tensor:
        other = _Pipeline(fp8=fp8)
        other_manager = _manager(other, adapters, monkeypatch)
        other_manager.set_active_adapter(_request(adapter_id), scale)
        return _run(other, x)

    sequence = [(1, 1.0), (2, 1.0), (1, 1.0), (1, 0.5), (1, 1.0), (3, 1.0), (1, 1.0), (2, 0.25), (1, 0.5)]
    results = {}
    for adapter_id, scale in sequence:
        manager.set_active_adapter(_request(adapter_id), scale)
        actual = _run(pipeline, x)
        torch.testing.assert_close(actual, fresh(adapter_id, scale), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual, _expected(reference, adapters[adapter_id], x, scale), rtol=1e-4, atol=1e-4)
        results[(adapter_id, scale)] = actual

    assert not torch.allclose(results[(1, 1.0)], results[(2, 1.0)], atol=1e-4)
    assert not torch.allclose(results[(1, 1.0)], results[(1, 0.5)], atol=1e-4)

    base = _expected(reference, None, x, 1.0)
    manager.set_active_adapter(_request(1), 0.0)
    torch.testing.assert_close(_run(pipeline, x), base, rtol=1e-5, atol=1e-5)
    manager.set_active_adapter(None)
    torch.testing.assert_close(_run(pipeline, x), base, rtol=1e-5, atol=1e-5)

    # Resuming a suspended adapter must match a fresh activation as well.
    manager.set_active_adapter(_request(1), 0.5)
    torch.testing.assert_close(_run(pipeline, x), fresh(1, 0.5), rtol=1e-5, atol=1e-5)


def _lora_tensors(wrapper) -> list[torch.Tensor]:
    return [*wrapper.lora_a_stacked, *wrapper.lora_b_stacked]


def _set_random_lora(wrapper, rank: int, seed: int) -> None:
    pairs = [_lora_pair(rank, size, torch.float32, seed + i) for i, size in enumerate(wrapper.output_slices)]
    if wrapper.n_slices == 1:
        wrapper.set_lora(0, *pairs[0])
    else:
        wrapper.set_lora(0, [a for a, _ in pairs], [b for _, b in pairs])


@pytest.fixture
def accelerator_device() -> torch.device:
    if current_omni_platform.get_device_count() == 0:
        pytest.skip("Accelerator required for this test")
    return current_omni_platform.get_torch_device(0)


def test_rank_growth_reallocates_buffers_where_they_were_before(monkeypatch):
    pipeline = _Pipeline(fp8=True)
    manager = _manager(pipeline, {}, monkeypatch)
    manager._replace_layers_with_lora(SimpleNamespace(r=4))
    wrappers = list(manager._lora_modules.values())
    assert len(wrappers) == 2
    assert all(w.lora_a_stacked[0].shape[2] == 8 for w in wrappers)

    # The base weights leave the device afterwards (stand-in for CPU offload)...
    for wrapper in wrappers:
        weight = wrapper.base_layer.weight
        wrapper.base_layer.weight = torch.nn.Parameter(
            torch.empty(weight.shape, dtype=weight.dtype, device="meta"), requires_grad=False
        )
    # ...while a longer adapter forces every buffer to be allocated again.
    manager._ensure_max_lora_rank(32)

    for wrapper in wrappers:
        assert wrapper.lora_a_stacked[0].shape[2] == 32
        assert all(t.device == CPU for t in _lora_tensors(wrapper))


def test_lora_buffers_are_not_parameters_or_registered_buffers(monkeypatch):
    pipeline = _Pipeline(fp8=True)
    manager = _manager(pipeline, {}, monkeypatch)
    manager._replace_layers_with_lora(SimpleNamespace(r=4))

    registered = {id(t) for t in (*pipeline.parameters(), *pipeline.buffers())}
    for wrapper in manager._lora_modules.values():
        assert not any(id(t) in registered for t in _lora_tensors(wrapper))


def test_model_level_offload_moves_base_weights_but_not_lora_buffers(accelerator_device, monkeypatch):
    pipeline = _Pipeline(fp8=True).to(accelerator_device)
    manager = _manager(pipeline, {}, monkeypatch, accelerator_device)
    manager._replace_layers_with_lora(SimpleNamespace(r=4))
    wrappers = list(manager._lora_modules.values())
    for index, wrapper in enumerate(wrappers):
        _set_random_lora(wrapper, rank=4, seed=7 + index)
    tensors = {id(w): _lora_tensors(w) for w in wrappers}
    before = {id(w): [t.clone() for t in _lora_tensors(w)] for w in wrappers}

    assert SequentialOffloadHook._move_params(pipeline.transformer, CPU)

    assert all(p.device == CPU for p in pipeline.transformer.parameters())
    for wrapper in wrappers:
        assert all(a is b for a, b in zip(_lora_tensors(wrapper), tensors[id(wrapper)]))
        for current, original in zip(_lora_tensors(wrapper), before[id(wrapper)]):
            assert current.device.type == accelerator_device.type
            torch.testing.assert_close(current, original)


def test_sequential_offload_cycle_keeps_lora_output_and_buffers(accelerator_device, monkeypatch):
    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))

        def forward(self, x):
            return x

    class _Dit(torch.nn.Module):
        def __init__(self, transformer: torch.nn.Module):
            super().__init__()
            self.transformer = transformer

        def forward(self, x):
            return _run(self, x)

    pipeline = _Pipeline(fp8=True).to(accelerator_device)
    manager = _manager(pipeline, {1: _Adapter(1, rank=4, dtype=torch.float32)}, monkeypatch, accelerator_device)
    manager.set_active_adapter(_request(1), 1.0)
    wrappers = list(manager._lora_modules.values())
    tensors = [t for w in wrappers for t in _lora_tensors(w)]
    dit, encoder = _Dit(pipeline.transformer), _Encoder().to(accelerator_device)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1)).to(accelerator_device)
    expected = dit(x)

    apply_sequential_offload([dit], [encoder], device=accelerator_device, pin_memory=False)
    for _ in range(2):
        encoder(torch.zeros(1, device=accelerator_device))
        assert all(p.device == CPU for p in dit.parameters())
        torch.testing.assert_close(dit(x), expected)
        assert all(p.device == CPU for p in encoder.parameters())

    after = [t for w in wrappers for t in _lora_tensors(w)]
    assert all(a is b for a, b in zip(after, tensors))
    assert all(t.device.type == accelerator_device.type for t in after)


class _Block(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.proj = _make_replicated(24, torch.float32, seed=seed, fp8=False)

    def forward(self, x):
        return _apply(self.proj, x)


class _LayerwisePipeline(torch.nn.Module):
    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.blocks = torch.nn.ModuleList(_Block(40 + i) for i in range(num_blocks))


def test_layerwise_offload_hooks_keep_lora_correct_when_wrapped_after_hook_install(monkeypatch):
    patch_offload_runtime(monkeypatch, layerwise_backend_module.current_omni_platform)
    pipeline = _LayerwisePipeline()
    blocks = pipeline.transformer.blocks
    base_weights = [block.proj.weight.detach().clone() for block in blocks]
    x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(2))

    # Same order as production: layerwise hooks first, LoRA layers when an adapter first arrives.
    layerwise_backend_module._install_layerwise_hook_group(blocks, CPU, DummyStream(), False)
    assert all(block.proj.weight.numel() == 0 for block in blocks)

    adapter = SimpleNamespace(id=5, rank=4, loras={})
    for index in range(len(blocks)):
        name = f"transformer.blocks.{index}.proj"
        lora_a, lora_b = _lora_pair(4, 24, torch.float32, seed=60 + index)
        adapter.loras[name] = LoRALayerWeights(module_name=name, rank=4, lora_alpha=4, lora_a=lora_a, lora_b=lora_b)
    adapter.get_lora = adapter.loras.get
    manager = _manager(pipeline, {5: adapter}, monkeypatch)

    seen_tensors: list[torch.Tensor] = []
    for scale in (1.0, 0.5, 1.0):
        manager.set_active_adapter(_request(5), scale)
        wrappers = [manager._lora_modules[f"transformer.blocks.{i}.proj"] for i in range(len(blocks))]
        if not seen_tensors:
            seen_tensors = [t for w in wrappers for t in _lora_tensors(w)]
        for _ in range(2):  # two passes through the ring
            for index, block in enumerate(blocks):
                lora = adapter.loras[f"transformer.blocks.{index}.proj"]
                expected = x @ base_weights[index].t() + _delta(x, lora.lora_a, lora.lora_b, scale)
                torch.testing.assert_close(block(x), expected, rtol=1e-5, atol=1e-5)

    assert all(a is b for a, b in zip([t for w in wrappers for t in _lora_tensors(w)], seen_tensors))
    assert all(t.device == CPU for t in seen_tensors)
