"""Host-side provisioning of the simulator's model-log directories.

``var/simulator/`` and ``var/simulator/standin/`` are bound read-write into
every container that runs the composite, each container appending under its
own uid. They are created before compose runs, setgid and group-writable, so
the containers share one group and none of them finds a root-owned source.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.deployment.compose_generator import (
    _ensure_agent_data_structure,
    ensure_simulator_log_dirs,
    simulated_target_configured,
)
from osprey_connectors.simulation.composite import Composite


def _assert_shared(path: Path) -> None:
    mode = path.stat().st_mode
    assert path.is_dir()
    assert mode & stat.S_ISGID
    assert mode & stat.S_IWGRP


def test_both_log_directories_are_provisioned_setgid_and_group_writable(
    tmp_path: Path,
) -> None:
    gid = ensure_simulator_log_dirs(tmp_path)

    _assert_shared(tmp_path / "var" / "simulator")
    _assert_shared(tmp_path / "var" / "simulator" / "standin")
    assert gid == (tmp_path / "var" / "simulator").stat().st_gid


def _config(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {"project_root": str(tmp_path), **extra}


@pytest.mark.parametrize(
    "extra",
    [
        {"deployed_services": ["virtual_accelerator"]},
        {"control_system": {"type": "mock"}},
        {
            "control_system": {
                "type": "epics",
                "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"port": 5064}},
            }
        },
    ],
    ids=["va-deployed", "mock", "va-target-beside-live"],
)
def test_a_deployment_with_a_simulated_target_provisions_them(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    _ensure_agent_data_structure(_config(tmp_path, **extra))

    _assert_shared(tmp_path / "var" / "simulator")
    _assert_shared(tmp_path / "var" / "simulator" / "standin")


def test_a_live_only_deployment_provisions_none(tmp_path: Path) -> None:
    config = _config(
        tmp_path, control_system={"type": "epics", "connector": {"epics": {"gateways": {}}}}
    )

    assert not simulated_target_configured(config)
    _ensure_agent_data_structure(config)

    assert not (tmp_path / "var" / "simulator").exists()


def _append(log_dir: Path, instance: str) -> None:
    """Append one record through the composite's own log writer."""
    writer = SimpleNamespace(_instance=instance, _log_dir=log_dir, _log_line=Composite._log_line)
    Composite._log(writer, "M", {"event": "noticed"})  # type: ignore[arg-type]


def test_two_writers_append_to_one_log_under_the_container_umask(tmp_path: Path) -> None:
    ensure_simulator_log_dirs(tmp_path)
    log_dir = tmp_path / "var" / "simulator"
    previous = os.umask(0o022)
    try:
        _append(log_dir, "virtual_accelerator")
        _append(log_dir, "inprocess")
    finally:
        os.umask(previous)

    log = log_dir / "M.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o664
    assert log.stat().st_gid == log_dir.stat().st_gid
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="needs root")
def test_two_uids_in_one_group_append_to_one_log(tmp_path: Path) -> None:
    """Two uids sharing the directory's group both append, and both lines survive."""
    ensure_simulator_log_dirs(tmp_path)
    log_dir = tmp_path / "var" / "simulator"
    os.chmod(tmp_path, 0o755)
    os.chmod(tmp_path / "var", 0o755)
    gid = log_dir.stat().st_gid
    for uid, instance in ((60001, "virtual_accelerator"), (60002, "inprocess")):
        pid = os.fork()
        if pid == 0:
            try:
                os.setgroups([gid])
                os.setgid(gid)
                os.setuid(uid)
                os.umask(0o022)
                _append(log_dir, instance)
            finally:
                os._exit(0)
        os.waitpid(pid, 0)

    lines = (log_dir / "M.log").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if "virtual_accelerator" in line]
    assert [line for line in lines if "inprocess" in line]
