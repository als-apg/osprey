"""Reading a value out of the process config, and the agent-data root it names.

This file deliberately does NOT re-test the loader, ``resolve_env_vars``, or the
timezone/ISO helpers — those are covered by ``tests/config/test_load_config.py``
and ``tests/utils/test_resolve_env_vars.py``. It targets
``current_config().get`` over the file ``OSPREY_CONFIG`` names, and
:attr:`~osprey_connectors.config.LoadedConfig.agent_data_root` anchoring.

The autouse reset clears the config state around every test, and
``restore_environ`` clears ``OSPREY_CONFIG``, so these serial tests leak no
global state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_connectors.config import current_config
from osprey_connectors.errors import ConfigurationError


def _write_config(dir_path: Path, body: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def _serve_file(monkeypatch, dir_path: Path, body: str) -> Path:
    cfg = _write_config(dir_path, body)
    monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
    return cfg


class TestProcessConfigGet:
    def test_reads_the_file_osprey_config_names(self, tmp_path, monkeypatch):
        _serve_file(
            monkeypatch,
            tmp_path,
            "project_root: /via-env\ncontrol_system:\n  limits:\n    max_channels: 100\n",
        )

        assert current_config().get("project_root") == "/via-env"
        assert current_config().get("control_system.limits.max_channels", 0) == 100

    def test_missing_key_returns_default(self, tmp_path, monkeypatch):
        _serve_file(monkeypatch, tmp_path, "project_root: /x\n")
        assert current_config().get("nope.missing", "fallback") == "fallback"

    def test_resolves_env_placeholders(self, tmp_path, monkeypatch):
        _serve_file(monkeypatch, tmp_path, "system:\n  timezone: ${CA_TZ:-UTC}\n")
        monkeypatch.setenv("CA_TZ", "Europe/Berlin")
        assert current_config().get("system.timezone") == "Europe/Berlin"

    def test_empty_key_raises_value_error(self, tmp_path, monkeypatch):
        _serve_file(monkeypatch, tmp_path, "project_root: /x\n")

        with pytest.raises(ConfigurationError, match="cannot be empty") as raised:
            current_config().get("")
        assert isinstance(raised.value, ValueError)

    def test_the_value_handed_out_is_a_copy(self, tmp_path, monkeypatch):
        _serve_file(monkeypatch, tmp_path, "control_system:\n  writes: allowed\n")

        section = current_config().get("control_system")
        assert isinstance(section, dict)
        section["writes"] = "never"

        assert current_config().get("control_system.writes") == "allowed"


class TestAgentDataRoot:
    def test_resolves_under_project_root_and_base_dir(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        _serve_file(
            monkeypatch,
            tmp_path / "elsewhere",
            f"project_root: {repo}\nagent_data:\n  base_dir: _agent_data\n",
        )

        assert current_config().agent_data_root == (repo / "_agent_data").resolve()

    def test_a_configured_root_that_is_absent_anchors_on_the_loaded_configs_repo(
        self, tmp_path, monkeypatch
    ):
        """A host path a container does not have: the loaded config's repo is the root here."""
        _serve_file(
            monkeypatch,
            tmp_path,
            "project_root: /nonexistent/project/root\nagent_data:\n  base_dir: _agent_data\n",
        )

        assert current_config().agent_data_root == (tmp_path / "_agent_data").resolve()

    def test_no_project_root_anchors_on_the_configs_repo_not_the_cwd(self, tmp_path, monkeypatch):
        """A render at ``<repo>/build/config.yml`` belongs to ``<repo>``, wherever the process stands."""
        repo = tmp_path / "repo"
        _serve_file(monkeypatch, repo / "build", "agent_data:\n  base_dir: _agent_data\n")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        assert current_config().agent_data_root == (repo / "_agent_data").resolve()

    def test_a_session_has_its_own_directory_under_the_root(self, tmp_path, monkeypatch):
        _serve_file(
            monkeypatch,
            tmp_path,
            f"project_root: {tmp_path}\nagent_data:\n  base_dir: _agent_data\n",
        )
        cfg = current_config()

        assert cfg.session_data_root("s-1") == cfg.agent_data_root / "sessions" / "s-1"
        with pytest.raises(ValueError, match="non-blank session id"):
            cfg.session_data_root("  ")
