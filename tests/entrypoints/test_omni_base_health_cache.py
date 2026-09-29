# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for OmniBase.check_health TTL caching."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pool(stage_id: int, clients: list) -> SimpleNamespace:
    return SimpleNamespace(stage_id=stage_id, clients=clients)


def _make_omni_base(engine: MagicMock):
    with (
        patch("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", return_value=engine),
        patch("vllm_omni.entrypoints.omni_base.omni_snapshot_download", side_effect=lambda x: x),
        patch("vllm_omni.entrypoints.omni_base.weakref.finalize"),
    ):
        from vllm_omni.entrypoints.omni_base import OmniBase

        return OmniBase(model="test-model")


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.num_stages = 1
    engine.is_alive.return_value = True
    engine.default_sampling_params_list = [MagicMock()]
    engine.get_stage_metadata.return_value = SimpleNamespace(final_output_type="text", final_output=True)
    engine.stage_pools = None
    engine.stage_configs = []
    return engine


class TestHealthCheckCache:
    def test_second_call_within_ttl_skips_probe(self):
        engine = _make_engine()
        client = MagicMock()
        client.check_health.return_value = None
        engine.stage_pools = [_make_pool(0, [client])]
        base = _make_omni_base(engine)

        base.check_health()
        base.check_health()

        assert client.check_health.call_count == 1

    def test_call_after_ttl_re_probes(self, monkeypatch):
        engine = _make_engine()
        client = MagicMock()
        client.check_health.return_value = None
        engine.stage_pools = [_make_pool(0, [client])]
        base = _make_omni_base(engine)

        base.check_health()
        assert client.check_health.call_count == 1

        base._health_cache = (time.monotonic() - 2.0, None)
        base.check_health()
        assert client.check_health.call_count == 2

    def test_cached_error_re_raised_within_ttl(self):
        engine = _make_engine()
        engine.is_alive.return_value = False
        base = _make_omni_base(engine)
        engine.is_alive.return_value = False

        with pytest.raises(EngineDeadError):
            base.check_health()

        engine.is_alive.return_value = True
        with pytest.raises(EngineDeadError):
            base.check_health()

    def test_recovery_after_ttl_expires(self):
        engine = _make_engine()
        engine.is_alive.return_value = False
        base = _make_omni_base(engine)

        with pytest.raises(EngineDeadError):
            base.check_health()

        engine.is_alive.return_value = True
        base._health_cache = (time.monotonic() - 2.0, None)
        base.check_health()

    def test_no_pools_is_healthy(self):
        engine = _make_engine()
        engine.stage_pools = None
        base = _make_omni_base(engine)

        base.check_health()

    def test_dead_stage_raises(self):
        engine = _make_engine()
        client = MagicMock()
        client.check_health.side_effect = EngineDeadError("dead")
        engine.stage_pools = [_make_pool(0, [client])]
        base = _make_omni_base(engine)

        with pytest.raises(EngineDeadError, match="Stage-0"):
            base.check_health()
