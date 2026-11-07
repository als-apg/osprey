"""The deploy path writes a scenario's session history once, before ``compose up``.

``_start_stack`` activates the profile's default set ahead of the store stages
and the bring-up; that activation is the only writer of
``scenario-sessions.json``. A second ``up``, an archive reseed and an upgraded
deployment whose recorded set predates ``demo-history`` write nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests.deployment.test_collect_all_preflight import stubbed_start  # noqa: F401 - fixture
from tests.simulation.test_apply_scenario_sessions import (  # noqa: F401 - fixture
    HOST,
    SCENARIO_SESSIONS_FILENAME,
    VIEW_CONFIG,
    built_view,
)


class _Reached(Exception):
    """Raised by the first store stage, which runs before any ``compose up``."""


@pytest.fixture
def deploy_project(built_view: Path):  # noqa: F811 - the imported fixture
    """Stage a deployment repo holding the example facility's real view and *config*."""

    def _make(root: Path, config: dict[str, Any]) -> Path:
        shutil.copytree(built_view, root / "data" / "simulator")
        (root / "config.yml").write_text(yaml.safe_dump(config))
        return root

    return _make


def _config(defaults: list[str]) -> dict[str, Any]:
    return {
        **VIEW_CONFIG,
        "project_name": "hello",
        "deployed_services": [],
        "deployment": {"host_identity": HOST},
        "simulation": {"default_scenarios": defaults},
    }


def _seeds(project: Path) -> list[Path]:
    return sorted((project / "var/agent_data").rglob(SCENARIO_SESSIONS_FILENAME))


def _start_until_the_archiver_stage(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], project: Path
) -> dict[str, Any]:
    """Run ``_start_stack`` up to its first store stage; return what that stage saw."""
    seen: dict[str, Any] = {}

    def _stage(*_args: Any, scenarios_activated: bool = False, **_kwargs: Any) -> None:
        seen["scenarios_activated"] = scenarios_activated
        seen["seeds"] = _seeds(project)
        raise _Reached

    def _compose_up(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("compose up ran before the scenario stages")

    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_archiver_store_deployed", lambda config: True)
    monkeypatch.setattr(container_lifecycle, "_stage_archiver_store", _stage)
    monkeypatch.setattr(container_lifecycle, "watched_compose_up", _compose_up)
    with pytest.raises(_Reached):
        container_lifecycle._start_stack(config, [], project, detached=True)
    return seen


@pytest.mark.usefixtures("stubbed_start")
def test_first_deploy_writes_the_scenario_seed_before_compose_up(
    monkeypatch: pytest.MonkeyPatch, deploy_project, tmp_path: Path
) -> None:
    config = _config(["rf-thermal", "demo-history"])
    project = deploy_project(tmp_path, config)

    seen = _start_until_the_archiver_stage(monkeypatch, config, project)

    assert seen["scenarios_activated"] is True
    assert seen["seeds"] == [
        project / "var/agent_data/control_target" / HOST / SCENARIO_SESSIONS_FILENAME
    ]


@pytest.mark.usefixtures("stubbed_start")
def test_a_failed_scenario_seed_still_reports_scenarios_activated(
    monkeypatch: pytest.MonkeyPatch, deploy_project, tmp_path: Path
) -> None:
    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("seed writer exploded")

    monkeypatch.setattr("osprey.simulation.apply.write_scenario_sessions", _boom)
    config = _config(["rf-thermal", "demo-history"])
    project = deploy_project(tmp_path, config)

    seen = _start_until_the_archiver_stage(monkeypatch, config, project)

    assert seen["scenarios_activated"] is True
    assert seen["seeds"] == []


def test_a_second_up_writes_no_scenario_seed(deploy_project, tmp_path: Path) -> None:
    config = _config(["rf-thermal", "demo-history"])
    project = deploy_project(tmp_path, config)
    assert container_lifecycle._activate_default_scenarios(config, project)
    for seed in _seeds(project):
        seed.unlink()

    assert container_lifecycle._activate_default_scenarios(config, project) == ()
    assert _seeds(project) == []


def test_an_archive_reseed_writes_no_scenario_seed(
    monkeypatch: pytest.MonkeyPatch, deploy_project, tmp_path: Path
) -> None:
    """``_reapply_active_scenarios`` goes through ``apply_scenarios``, never activation."""
    config = _config(["rf-thermal", "demo-history"])
    project = deploy_project(tmp_path, config)
    assert container_lifecycle._activate_default_scenarios(config, project)
    for seed in _seeds(project):
        seed.unlink()
    monkeypatch.setattr("osprey.simulation.apply.seed_archiver", lambda *a, **k: None)

    container_lifecycle._reapply_active_scenarios(config, project)

    assert _seeds(project) == []


def test_an_upgraded_deployment_without_demo_history_gets_no_scenario_seed(
    deploy_project, tmp_path: Path
) -> None:
    """A deployment that already chose a set keeps it: no activation, so no history."""
    project = deploy_project(tmp_path, _config(["rf-thermal"]))
    assert container_lifecycle._activate_default_scenarios(_config(["rf-thermal"]), project)
    upgraded = _config(["rf-thermal", "demo-history"])
    (project / "config.yml").write_text(yaml.safe_dump(upgraded))
    assert (resolve_simulation_state_dir(upgraded, project) / ACTIVE_SCENARIOS_FILENAME).is_file()

    assert container_lifecycle._activate_default_scenarios(upgraded, project) == ()
    assert _seeds(project) == []
