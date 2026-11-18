"""Tests for config.py's value accessor and its path helper.

This file deliberately does NOT re-test the loader, ``resolve_env_vars``, or the
timezone/ISO helpers — those are covered by ``tests/config/test_load_config.py``
and ``tests/utils/test_resolve_env_vars.py``. It targets ``get_config_value``
over the process config and an explicit path, and ``get_agent_dir`` path
resolution.

The autouse ``reset_state_between_tests`` fixture clears the config state
around every test, and ``restore_environ`` clears ``OSPREY_CONFIG``, so these
serial tests leak no global state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey_connectors.config import get_agent_dir, get_config_value


def _write_config(dir_path: Path, body: str) -> Path:
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


class TestGetConfigValue:
    def test_no_path_reads_the_file_osprey_config_names(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path,
            "project_root: /via-env\ncontrol_system:\n  limits:\n    max_channels: 100\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        assert get_config_value("project_root") == "/via-env"
        assert get_config_value("control_system.limits.max_channels", 0) == 100

    def test_explicit_path_dot_access(self, tmp_path):
        cfg = _write_config(tmp_path, "control_system:\n  limits:\n    max: 100\n")
        assert get_config_value("control_system.limits.max", 0, str(cfg)) == 100

    def test_missing_path_returns_default(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        assert get_config_value("nope.missing", "fallback", str(cfg)) == "fallback"

    def test_resolves_env_placeholders(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, "system:\n  timezone: ${CA_TZ:-UTC}\n")
        monkeypatch.setenv("CA_TZ", "Europe/Berlin")
        assert get_config_value("system.timezone", None, str(cfg)) == "Europe/Berlin"

    def test_empty_path_raises_value_error(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")

        with pytest.raises(ValueError, match="cannot be empty"):
            get_config_value("", None, str(cfg))

    def test_the_value_handed_out_is_a_copy(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, "control_system:\n  writes: allowed\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        section = get_config_value("control_system")
        assert isinstance(section, dict)
        section["writes"] = "never"

        assert get_config_value("control_system.writes") == "allowed"


class TestGetAgentDir:
    def test_resolves_under_project_root_and_agent_data_dir(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path,
            f"project_root: {tmp_path}\n"
            "agent_data:\n"
            "  base_dir: _agent_data\n"
            "file_paths:\n"
            "  api_calls_dir: logs/api\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        result = Path(get_agent_dir("api_calls_dir"))
        assert result == tmp_path / "_agent_data" / "logs" / "api"

    def test_unknown_sub_dir_falls_back_to_its_name(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path,
            f"project_root: {tmp_path}\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        result = Path(get_agent_dir("registry_exports_dir"))
        assert result == tmp_path / "_agent_data" / "registry_exports_dir"

    def test_stale_applications_block_does_not_override_file_paths(self, tmp_path, monkeypatch):
        """``applications:`` is retired — only the top-level ``file_paths`` wins.

        Projects that still carry a legacy ``applications:`` block keep loading,
        but its nested ``file_paths`` does not scope directory resolution.
        """
        cfg = _write_config(
            tmp_path,
            f"project_root: {tmp_path}\n"
            "agent_data:\n"
            "  base_dir: _agent_data\n"
            "file_paths:\n"
            "  plans_dir: default_plans\n"
            "applications:\n"
            "  app1:\n"
            "    file_paths:\n"
            "      plans_dir: app_plans\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        result = Path(get_agent_dir("plans_dir"))
        assert result == tmp_path / "_agent_data" / "default_plans"

    def test_a_configured_root_that_is_absent_anchors_on_the_loaded_configs_repo(
        self, tmp_path, monkeypatch
    ):
        """A host path a container does not have: the loaded config's repo is the root here."""
        cfg = _write_config(
            tmp_path,
            "project_root: /nonexistent/project/root\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        result = Path(get_agent_dir("memory_dir"))

        assert result == tmp_path / "_agent_data" / "memory_dir"

    def test_host_path_resolves_under_the_configured_root_even_when_it_is_absent(
        self, tmp_path, monkeypatch
    ):
        cfg = _write_config(
            tmp_path,
            "project_root: /nonexistent/project/root\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))

        result = Path(get_agent_dir("memory_dir", host_path=True))

        assert result == Path("/nonexistent/project/root/_agent_data/memory_dir")

    def test_no_project_root_resolves_relative_to_the_cwd_with_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        cfg = _write_config(tmp_path, "agent_data:\n  base_dir: _agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            result = Path(get_agent_dir("memory_dir"))

        assert result == (tmp_path / "_agent_data" / "memory_dir").resolve()
        assert "No project root configured" in caplog.text
