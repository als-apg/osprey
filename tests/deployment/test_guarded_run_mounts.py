"""Every container that runs the agent shares one host ``var/guarded_run``.

A guarded run's lock and journal have to be one per control target for the
whole deployment, so the deploy provisions ``var/guarded_run/<target>/`` on the
host before compose runs and binds ``var/guarded_run`` read-write under the
container repo root of every container that runs the agent.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from osprey.deployment.compose_generator import (
    _ensure_agent_data_structure,
    ensure_guarded_run_dirs,
)

#: A deployment on a real control system with the Virtual Accelerator beside it.
LIVE_AND_VA = {
    "type": "epics",
    "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"port": 5064}},
}


def _assert_shared(path: Path) -> None:
    mode = path.stat().st_mode
    assert path.is_dir()
    assert mode & stat.S_ISGID
    assert mode & stat.S_IWGRP


def test_each_configured_target_is_provisioned(tmp_path: Path) -> None:
    gid = ensure_guarded_run_dirs(tmp_path, {"control_system": LIVE_AND_VA})

    root = tmp_path / "var" / "guarded_run"
    _assert_shared(root)
    assert sorted(path.name for path in root.iterdir()) == ["live", "va"]
    for target in root.iterdir():
        _assert_shared(target)
    assert gid == root.stat().st_gid


def test_a_deployment_without_a_control_system_gets_its_baseline(tmp_path: Path) -> None:
    ensure_guarded_run_dirs(tmp_path, {})

    assert [path.name for path in (tmp_path / "var" / "guarded_run").iterdir()] == ["live"]


def test_the_build_path_provisions_it(tmp_path: Path) -> None:
    config: dict[str, Any] = {"project_root": str(tmp_path), "control_system": {"type": "mock"}}

    _ensure_agent_data_structure(config)

    _assert_shared(tmp_path / "var" / "guarded_run" / "live")
