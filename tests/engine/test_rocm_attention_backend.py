# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for _apply_rocm_attention_backend."""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def _rocm_env(monkeypatch):
    """Stub the platform as ROCm and provide a controllable aiter module."""
    monkeypatch.setattr(
        "vllm_omni.engine.stage_init_utils.current_omni_platform",
        types.SimpleNamespace(is_rocm=lambda: True),
    )


def _patch_aiter(monkeypatch, *, enabled: bool):
    aiter_module = types.ModuleType("vllm._aiter_ops")
    aiter_module.rocm_aiter_ops = types.SimpleNamespace(is_enabled=lambda: enabled)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_module)


@pytest.mark.usefixtures("_rocm_env")
class TestApplyRocmAttentionBackend:
    def test_aiter_enabled_selects_rocm_aiter_fa(self, monkeypatch):
        _patch_aiter(monkeypatch, enabled=True)
        from vllm_omni.engine.stage_init_utils import _apply_rocm_attention_backend

        engine_args: dict = {}
        _apply_rocm_attention_backend(engine_args, "llm")
        assert engine_args["attention_backend"] == "ROCM_AITER_FA"

    def test_aiter_disabled_selects_triton(self, monkeypatch):
        _patch_aiter(monkeypatch, enabled=False)
        from vllm_omni.engine.stage_init_utils import _apply_rocm_attention_backend

        engine_args: dict = {}
        _apply_rocm_attention_backend(engine_args, "llm")
        assert engine_args["attention_backend"] == "TRITON_ATTN"

    def test_diffusion_stage_skipped(self, monkeypatch):
        _patch_aiter(monkeypatch, enabled=True)
        from vllm_omni.engine.stage_init_utils import _apply_rocm_attention_backend

        engine_args: dict = {}
        _apply_rocm_attention_backend(engine_args, "diffusion")
        assert "attention_backend" not in engine_args

    def test_explicit_backend_not_overwritten(self, monkeypatch):
        _patch_aiter(monkeypatch, enabled=True)
        from vllm_omni.engine.stage_init_utils import _apply_rocm_attention_backend

        engine_args: dict = {"attention_backend": "FLASH_ATTN"}
        _apply_rocm_attention_backend(engine_args, "llm")
        assert engine_args["attention_backend"] == "FLASH_ATTN"


def test_non_rocm_platform_skipped(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.engine.stage_init_utils.current_omni_platform",
        types.SimpleNamespace(is_rocm=lambda: False),
    )
    from vllm_omni.engine.stage_init_utils import _apply_rocm_attention_backend

    engine_args: dict = {}
    _apply_rocm_attention_backend(engine_args, "llm")
    assert "attention_backend" not in engine_args
