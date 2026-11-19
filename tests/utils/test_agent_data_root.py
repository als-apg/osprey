"""One agent-data root: ``agent_data.base_dir`` and nothing else.

``file_paths.agent_data_dir`` used to name the same directory for a disjoint set
of readers — the old subdirectory resolver, the ``file_system`` health row, and
the compose generator's mkdir — while the artifact/workspace/executor stack
resolved ``agent_data.base_dir``.  Neither key won, and roughly eight sites
hardcode the literal ``_agent_data`` regardless, so a project that overrode the
``file_paths`` copy silently desynced its container mounts from its data.

The ``file_paths`` copy is retired.  Every reader now resolves the root through
:func:`osprey_connectors.workspace.agent_data_base_dir` (a loaded config's
:attr:`~osprey_connectors.config.LoadedConfig.agent_data_root`), and
``file_paths`` is left holding only the *subdirectory* names below that root.
These tests pin that split from both sides: the root follows
``agent_data.base_dir`` and ignores the retired key, and a subdirectory name
declared in ``file_paths`` reaches both the registry's export directory and the
compose-time mkdir that has to pre-create the mount point — the interaction that
makes ``registry_exports_dir`` land in ``registry_exports/`` instead of a
directory named after its own config key.

Anchoring is unified on ``project_root`` — the deployment repo root, not the
directory holding ``config.yml``, which since the repo/render split is one level
down in ``build/``.  A loaded config anchors on it
(:attr:`~osprey_connectors.config.LoadedConfig.repo_root`), and so do the
callers holding only a config mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.compose_generator import _ensure_agent_data_structure
from osprey.health.core.file_system import file_system
from osprey.health.models import Status
from osprey_connectors.config import current_config
from osprey_connectors.testing import config_of
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR, agent_data_base_dir

# The shipped templates' declarations, as rendered into every project.
TEMPLATE_BASE_DIR = "./_agent_data"
TEMPLATE_EXPORTS_DIR = "registry_exports"


def _mkdir_loop_subdirs(root: Path) -> list[str]:
    """Names under the agent-data root that some config key asked for.

    The control-context tree is not one of them and is not evidence that the
    ``file_paths`` loop invented a directory: it is provisioned on every build
    because every container that reads it binds it read-only and so can create
    none of it, and no config key declares it.
    """
    from osprey_connectors.write_switch import CONTROL_STATE_DIR_NAME

    return sorted(entry.name for entry in root.iterdir() if entry.name != CONTROL_STATE_DIR_NAME)


def _write_config(project: Path, body: str) -> Path:
    cfg = project / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def _exports_config(project: Path, monkeypatch) -> Path:
    """Write the shipped declarations and make them the active ``OSPREY_CONFIG``.

    Kept a helper rather than a fixture: the other tests here write deliberately
    different bodies, and one caller needs the path back to pass as ``config_path``.
    """
    cfg = _write_config(
        project,
        f"project_root: {project}\n"
        f'agent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
        f"file_paths:\n  registry_exports_dir: {TEMPLATE_EXPORTS_DIR}\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
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
            # The retired key names the same directory but does not supply it.
            {"file_paths": {"agent_data_dir": "_legacy_data"}},
        ],
        ids=["none", "empty", "empty-section", "null-section", "null-value", "retired-key-only"],
    )
    def test_falls_back_to_the_default_root(self, config: dict[str, Any] | None) -> None:
        assert agent_data_base_dir(config) == DEFAULT_AGENT_DATA_BASE_DIR


def _exported_to(monkeypatch, config_path: Path | None = None) -> Path:
    """Where the registry's auto-export writes under the process config.

    The registry itself is stubbed: what is pinned is the directory
    :func:`osprey.registry.manager.initialize_registry` computes and creates.
    """
    from osprey.registry import manager

    exported_to: dict[str, str] = {}

    class _StubRegistry:
        def initialize(self, silent: bool = False) -> None:
            pass

        def export_registry_to_json(self, out_dir: str) -> None:
            exported_to["dir"] = out_dir
            Path(out_dir, "registry_export.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(manager, "get_registry", lambda config_path=None: _StubRegistry())
    manager.initialize_registry(
        auto_export=True, config_path=None if config_path is None else str(config_path)
    )
    return Path(exported_to["dir"])


class TestAgentDataRoot:
    """A loaded config's agent-data root, anchored on its ``project_root``."""

    def test_resolves_under_project_root_and_agent_data_base_dir(
        self, project: Path, monkeypatch
    ) -> None:
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            f'agent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
            "file_paths:\n  api_calls_dir: api_calls\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert current_config().agent_data_root == (project / "_agent_data").resolve()

    def test_custom_base_dir_relocates_every_subdirectory(self, project: Path, monkeypatch) -> None:
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            'agent_data:\n  base_dir: "./scratch-data"\n'
            f"file_paths:\n  registry_exports_dir: {TEMPLATE_EXPORTS_DIR}\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert current_config().agent_data_root == (project / "scratch-data").resolve()
        assert (
            _exported_to(monkeypatch) == (project / "scratch-data" / "registry_exports").resolve()
        )

    def test_retired_file_paths_key_does_not_move_the_root(
        self, project: Path, monkeypatch
    ) -> None:
        """A legacy config keeps loading; its retired key does not steer."""
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            "file_paths:\n"
            "  agent_data_dir: _legacy_data\n"
            "  api_calls_dir: api_calls\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        resolved = current_config().agent_data_root
        assert resolved == (project / DEFAULT_AGENT_DATA_BASE_DIR).resolve()
        assert "_legacy_data" not in resolved.parts


class TestRegistryExport:
    """The registry's auto-export writes under the resolved root."""

    def test_export_lands_in_registry_exports(self, project: Path, monkeypatch) -> None:
        """The bug #73 fixed: an undeclared key lands in a dir named after itself."""
        cfg = _exports_config(project, monkeypatch)

        exported = _exported_to(monkeypatch, cfg)

        expected = (project / "_agent_data" / "registry_exports").resolve()
        assert exported == expected
        assert (expected / "registry_export.json").is_file()
        # The literal-name directory the missing declaration used to produce.
        assert not (project / "_agent_data" / "registry_exports_dir").exists()

    def test_an_undeclared_exports_dir_is_named_after_its_key(
        self, project: Path, monkeypatch
    ) -> None:
        cfg = _write_config(
            project, f'project_root: {project}\nagent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        assert (
            _exported_to(monkeypatch)
            == (project / "_agent_data" / "registry_exports_dir").resolve()
        )

    def test_stale_applications_block_does_not_override_file_paths(
        self, project: Path, monkeypatch
    ) -> None:
        """``applications:`` is retired — only the top-level ``file_paths`` wins.

        Projects that still carry a legacy ``applications:`` block keep loading,
        but its nested ``file_paths`` does not scope directory resolution.
        """
        cfg = _write_config(
            project,
            f"project_root: {project}\n"
            f'agent_data:\n  base_dir: "{TEMPLATE_BASE_DIR}"\n'
            f"file_paths:\n  registry_exports_dir: {TEMPLATE_EXPORTS_DIR}\n"
            "applications:\n"
            "  app1:\n"
            "    file_paths:\n"
            "      registry_exports_dir: app_exports\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        assert (
            _exported_to(monkeypatch) == (project / "_agent_data" / TEMPLATE_EXPORTS_DIR).resolve()
        )


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

        # Compose-created mount point and the registry's export are the same path.
        _exports_config(project, monkeypatch)
        assert _exported_to(monkeypatch) == (project / "_agent_data" / "registry_exports").resolve()

    def test_undeclared_subdirectory_is_not_created(self, project: Path) -> None:
        """The mkdir loop stays keyed on ``file_paths`` — it invents no directories."""
        _ensure_agent_data_structure(
            {"project_root": str(project), "agent_data": {"base_dir": TEMPLATE_BASE_DIR}}
        )
        assert (project / "_agent_data").is_dir()
        assert _mkdir_loop_subdirs(project / "_agent_data") == []

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

    def test_home_relative_root_is_expanded_not_taken_literally(
        self, project: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """``~/state`` is the home directory, here as everywhere else.

        Joining the configured value under the project root instead would leave a
        directory literally named ``~`` inside the repo, next to the real root
        every runtime resolver goes on to use — so the mount point created at
        build time would not be the one the containers read.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": "~/state"},
                "file_paths": {"registry_exports_dir": TEMPLATE_EXPORTS_DIR},
            }
        )

        assert (tmp_path / "state" / "registry_exports").is_dir()
        assert not (project / "~").exists()


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

        assert _mkdir_loop_subdirs(project / "_agent_data") == []

    def test_follows_a_relocated_agent_data_root(self, project: Path) -> None:
        _ensure_agent_data_structure(
            {
                "project_root": str(project),
                "agent_data": {"base_dir": "./scratch-data"},
                "deployed_services": ["virtual_accelerator"],
            }
        )

        assert (project / "scratch-data" / "simulation").is_dir()

    def test_matches_the_resolved_simulation_state_dir(self, project: Path) -> None:
        """The pre-created mount point and the runtime write target are one path."""
        config: dict[str, Any] = {
            "project_root": str(project),
            "agent_data": {"base_dir": TEMPLATE_BASE_DIR},
            "deployed_services": ["virtual_accelerator"],
        }
        _ensure_agent_data_structure(config)

        assert config_of(config, path=project / "config.yml").simulation_state_dir.is_dir()


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
