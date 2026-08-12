"""One agent-data root: ``agent_data.base_dir`` and nothing else.

``file_paths.agent_data_dir`` used to name the same directory for a disjoint set
of readers — ``get_agent_dir``, the ``file_system`` health row, and the compose
generator's mkdir — while the artifact/workspace/executor stack resolved
``agent_data.base_dir``.  Neither key won, and roughly eight sites hardcode the
literal ``_agent_data`` regardless, so a project that overrode the ``file_paths``
copy silently desynced its container mounts from its data.

The ``file_paths`` copy is retired.  Every reader now resolves the root through
:func:`osprey.utils.workspace.agent_data_base_dir`, and ``file_paths`` is left
holding only the *subdirectory* names below that root.  These tests pin that
split from both sides: the root follows ``agent_data.base_dir`` and ignores the
retired key, and a subdirectory name declared in ``file_paths`` reaches both the
runtime resolver and the compose-time mkdir that has to pre-create the mount
point — the interaction that makes ``registry_exports_dir`` land in
``registry_exports/`` instead of a directory named after its own config key.

Anchoring is now unified on ``project_root`` — the deployment repo root, not the
directory holding ``config.yml``, which since the repo/render split is one level
down in ``build/``.  Callers holding a config anchor on it directly; the ones
with none in hand go through :func:`resolve_agent_data_root`, which loads the
config, anchors on the same key, and adds ``OSPREY_SESSION_ID`` isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.compose_generator import _ensure_agent_data_structure
from osprey.health.core.file_system import file_system
from osprey.health.models import Status
from osprey.utils.config import get_agent_dir
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR, agent_data_base_dir

# The shipped templates' declarations, as rendered into every project.
TEMPLATE_BASE_DIR = "./_agent_data"
TEMPLATE_EXPORTS_DIR = "registry_exports"


def _write_config(project: Path, body: str) -> Path:
    cfg = project / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def _exports_config(project: Path, monkeypatch) -> Path:
    """Write the shipped declarations and make them the active ``CONFIG_FILE``.

    Kept a helper rather than a fixture: the other tests here write deliberately
    different bodies, and one caller needs the path back to pass as ``config_path``.
    """
    cfg = _write_config(
        project,
        f"project_root: {project}\n"
        f'agent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
        f"file_paths:\n  registry_exports_dir: {TEMPLATE_EXPORTS_DIR}\n",
    )
    monkeypatch.setenv("CONFIG_FILE", str(cfg))
    return cfg


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "demo-project"
    root.mkdir()
    return root


class TestAgentDataBaseDir:
    """The single key read, shared by every config-holding reader."""

    def test_reads_the_declared_base_dir(self) -> None:
        assert agent_data_base_dir({"agent_data": {"base_dir": "./data"}}) == "./data"

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"agent_data": {}},
            {"agent_data": None},
            {"agent_data": {"base_dir": None}},
            # The retired key names the same directory but no longer supplies it.
            {"file_paths": {"agent_data_dir": "_legacy_data"}},
        ],
        ids=["none", "empty", "empty-section", "null-section", "null-value", "retired-key-only"],
    )
    def test_falls_back_to_the_default_root(self, config: dict[str, Any] | None) -> None:
        assert agent_data_base_dir(config) == DEFAULT_AGENT_DATA_BASE_DIR


class TestGetAgentDir:
    """``get_agent_dir`` resolves subdirectories under the workspace root."""

    def test_resolves_under_project_root_and_agent_data_base_dir(
        self, project: Path, monkeypatch
    ) -> None:
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            f'agent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
            "file_paths:\n  api_calls_dir: api_calls\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        assert Path(get_agent_dir("api_calls_dir")) == project / "_agent_data" / "api_calls"

    def test_custom_base_dir_relocates_every_subdirectory(self, project: Path, monkeypatch) -> None:
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            'agent_data:\n  base_dir: "./scratch-data"\n'
            "file_paths:\n  api_calls_dir: api_calls\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        assert Path(get_agent_dir("api_calls_dir")) == project / "scratch-data" / "api_calls"

    def test_retired_file_paths_key_no_longer_moves_the_root(
        self, project: Path, monkeypatch
    ) -> None:
        """A legacy config keeps loading; its retired key just stops steering."""
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            "file_paths:\n"
            "  agent_data_dir: _legacy_data\n"
            "  api_calls_dir: api_calls\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        resolved = Path(get_agent_dir("api_calls_dir"))
        assert resolved == project / DEFAULT_AGENT_DATA_BASE_DIR / "api_calls"
        assert "_legacy_data" not in resolved.parts

    def test_declared_subdirectory_beats_the_key_name_fallback(
        self, project: Path, monkeypatch
    ) -> None:
        """The bug #73 fixed: an undeclared key lands in a dir named after itself."""
        _exports_config(project, monkeypatch)
        assert Path(get_agent_dir("registry_exports_dir")) == (
            project / "_agent_data" / "registry_exports"
        )


class TestRegistryExport:
    """The registry's auto-export writes under the resolved root."""

    def test_export_lands_in_registry_exports(self, project: Path, monkeypatch) -> None:
        cfg = _exports_config(project, monkeypatch)

        from osprey.registry import manager

        exported_to: dict[str, str] = {}

        class _StubRegistry:
            def initialize(self, silent: bool = False) -> None:
                pass

            def export_registry_to_json(self, out_dir: str) -> None:
                exported_to["dir"] = out_dir
                Path(out_dir, "registry_export.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(manager, "get_registry", lambda config_path=None: _StubRegistry())
        manager.initialize_registry(auto_export=True, config_path=str(cfg))

        expected = project / "_agent_data" / "registry_exports"
        assert Path(exported_to["dir"]) == expected
        assert (expected / "registry_export.json").is_file()
        # The literal-name directory the missing declaration used to produce.
        assert not (project / "_agent_data" / "registry_exports_dir").exists()


class TestComposeStructure:
    """The compose-time mkdir pre-creates exactly what the runtime resolves."""

    def test_creates_root_and_declared_subdirectory(self, project: Path, monkeypatch) -> None:
        config = {
            "project_root": str(project),
            "agent_data": {"base_dir": TEMPLATE_BASE_DIR},
            "file_paths": {"registry_exports_dir": TEMPLATE_EXPORTS_DIR},
        }
        _ensure_agent_data_structure(config)

        assert (project / "_agent_data").is_dir()
        assert (project / "_agent_data" / "registry_exports").is_dir()

        # Compose-created mount point and runtime resolution are the same path.
        _exports_config(project, monkeypatch)
        assert Path(get_agent_dir("registry_exports_dir")) == (
            project / "_agent_data" / "registry_exports"
        )

    def test_undeclared_subdirectory_is_not_created(self, project: Path) -> None:
        """The mkdir loop stays keyed on ``file_paths`` — it invents no directories."""
        _ensure_agent_data_structure(
            {"project_root": str(project), "agent_data": {"base_dir": TEMPLATE_BASE_DIR}}
        )
        assert (project / "_agent_data").is_dir()
        assert list((project / "_agent_data").iterdir()) == []

    def test_root_follows_the_configured_base_dir(self, project: Path) -> None:
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": "./scratch-data"},
                "file_paths": {
                    "agent_data_dir": "_legacy_data",
                    "registry_exports_dir": TEMPLATE_EXPORTS_DIR,
                },
            }
        )
        assert (project / "scratch-data" / "registry_exports").is_dir()
        assert not (project / "_legacy_data").exists()


class TestScenarioStateMountPoint:
    """The VA compose service mounts the scenario state dir; pre-create its source.

    Left to the container runtime, a rootful daemon creates a missing bind-mount
    source root-owned — and ``osprey sim apply``, which owns that file, runs as the
    host user.
    """

    def test_created_for_a_virtual_accelerator_deployment(self, project: Path) -> None:
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": TEMPLATE_BASE_DIR},
                "deployed_services": ["virtual_accelerator"],
            }
        )

        assert (project / "_agent_data" / "simulation").is_dir()

    def test_not_created_without_the_service(self, project: Path) -> None:
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": TEMPLATE_BASE_DIR},
                "deployed_services": ["channel_finder"],
            }
        )

        assert list((project / "_agent_data").iterdir()) == []

    def test_follows_a_relocated_agent_data_root(self, project: Path) -> None:
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": "./scratch-data"},
                "deployed_services": ["virtual_accelerator"],
            }
        )

        assert (project / "scratch-data" / "simulation").is_dir()

    def test_matches_what_the_engine_resolves(self, project: Path) -> None:
        """The pre-created mount point and the runtime write target are one path."""
        from osprey.simulation.engine import resolve_state_dir

        config: dict[str, Any] = {
            "project_root": str(project),
            "agent_data": {"base_dir": TEMPLATE_BASE_DIR},
            "deployed_services": ["virtual_accelerator"],
        }
        _ensure_agent_data_structure(config)

        assert resolve_state_dir(config, project).is_dir()


class TestHealthRow:
    """The ``agent_data_dir`` health row checks the same directory."""

    def test_row_reports_the_configured_base_dir(self, project: Path) -> None:
        (project / "scratch-data").mkdir()
        config = {"project_root": str(project), "agent_data": {"base_dir": "./scratch-data"}}
        rows = {r.name: r for r in file_system(config, None, cwd=project)()}
        assert rows["agent_data_dir"].status is Status.OK
        assert str(project / "scratch-data") in rows["agent_data_dir"].message

    def test_row_ignores_the_retired_key(self, project: Path) -> None:
        config = {"project_root": str(project), "file_paths": {"agent_data_dir": "_legacy_data"}}
        rows = {r.name: r for r in file_system(config, None, cwd=project)()}
        assert str(project / DEFAULT_AGENT_DATA_BASE_DIR) in rows["agent_data_dir"].message
        assert "_legacy_data" not in rows["agent_data_dir"].message
