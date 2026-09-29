# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for per-device init file locks in stage_init_utils."""

import fcntl
import os

import pytest

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.stage_init_utils import acquire_device_locks, release_device_locks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    """Redirect device lock files into tmp_path."""
    monkeypatch.setattr(
        stage_init_utils, "device_init_lock_path", lambda device_id: str(tmp_path / f"{device_id}.lock")
    )
    return tmp_path


def _is_locked_by_other(path) -> bool:
    with open(path, "a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe, fcntl.LOCK_UN)
    return False


def test_acquire_and_release_locks_all_devices(lock_dir):
    fds = acquire_device_locks(0, {"tensor_parallel_size": 2}, stage_init_timeout=1, visible_devices="0,1")
    try:
        assert len(fds) == 2
        for device_id in (0, 1):
            assert _is_locked_by_other(lock_dir / f"{device_id}.lock")
    finally:
        release_device_locks(fds)

    for device_id in (0, 1):
        assert not _is_locked_by_other(lock_dir / f"{device_id}.lock")


def test_open_failure_keeps_previously_acquired_lock(monkeypatch, lock_dir):
    """An open failure on device N must not close device N-1's held fd."""
    real_open = stage_init_utils.open_device_lock_file
    failing = str(lock_dir / "1.lock")

    def open_or_fail(lock_file):
        if lock_file == failing:
            raise PermissionError(13, "Permission denied", lock_file)
        return real_open(lock_file)

    monkeypatch.setattr(stage_init_utils, "open_device_lock_file", open_or_fail)

    fds = acquire_device_locks(0, {"tensor_parallel_size": 2}, stage_init_timeout=1, visible_devices="0,1")
    try:
        assert len(fds) == 1
        # The returned fd must still be open and still hold device 0's lock.
        os.fstat(fds[0])
        assert _is_locked_by_other(lock_dir / "0.lock")
    finally:
        release_device_locks(fds)

    assert not _is_locked_by_other(lock_dir / "0.lock")
