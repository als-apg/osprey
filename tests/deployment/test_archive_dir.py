"""The agent-record archive root is provisioned operator-private, and only when deployed."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from osprey.deployment.compose_generator import _ensure_agent_data_structure, ensure_archive_dir


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_created_at_exactly_0700(tmp_path):
    old = os.umask(0)  # a permissive umask must not widen it
    try:
        target = ensure_archive_dir(tmp_path)
    finally:
        os.umask(old)

    assert target == tmp_path / "var" / "archive"
    assert _mode(target) == 0o700


@pytest.mark.parametrize("existing", [0o750, 0o755, 0o700])
def test_an_existing_mode_is_left_alone(tmp_path, existing):
    target = tmp_path / "var" / "archive"
    target.mkdir(parents=True)
    os.chmod(target, existing)

    ensure_archive_dir(tmp_path)

    assert _mode(target) == existing


@pytest.mark.parametrize(("deployed", "expected"), [(["archive"], True), ([], False)])
def test_provisioned_only_when_the_archive_is_deployed(tmp_path, deployed, expected):
    config = {"project_root": str(tmp_path), "deployed_services": deployed}

    _ensure_agent_data_structure(config)

    assert (tmp_path / "var" / "archive").is_dir() is expected


def test_a_failure_warns_and_does_not_raise(tmp_path, monkeypatch, caplog):
    def _refuse(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", _refuse)

    with caplog.at_level(logging.WARNING, logger="deployment.compose"):
        assert ensure_archive_dir(tmp_path) is None

    assert "Could not provision archive directory" in caplog.text
