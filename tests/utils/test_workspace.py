"""Tests for cross-layer workspace / config-path resolution helpers.

These functions resolve where a project's config and agent-data live. They read
``OSPREY_CONFIG`` (config location), ``OSPREY_SESSION_ID`` (per-session data
isolation), and the config's ``agent_data.base_dir`` / ``project_root``. Paths
are resolved relative to the config file's parent directory, with a
current-working-directory fallback when no config is present.

The autouse ``reset_state_between_tests`` fixture clears ``OSPREY_CONFIG`` /
``CONFIG_FILE`` and the config-cache singleton around every test.
"""

from __future__ import annotations

from pathlib import Path

from osprey.utils.workspace import (
    load_osprey_config,
    reset_config_cache,
    resolve_agent_data_root,
    resolve_config_path,
    resolve_path,
    resolve_shared_data_root,
    resolve_workspace_root,
)


def _write_config(dir_path: Path, body: str) -> Path:
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


class TestResolveConfigPath:
    def test_uses_osprey_config_env_when_set(self, tmp_path, monkeypatch):
        cfg = tmp_path / "custom.yml"
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_config_path() == cfg

    def test_defaults_to_cwd_config_yml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path() == tmp_path / "config.yml"

    def test_expands_shell_variables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("OSPREY_CONFIG", "$MY_PROJECT_DIR/config.yml")
        assert resolve_config_path() == tmp_path / "config.yml"


class TestLoadOspreyConfig:
    def test_loads_parsed_config(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /somewhere\nfoo:\n  bar: 7\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        config = load_osprey_config()
        assert config["project_root"] == "/somewhere"
        assert config["foo"]["bar"] == 7

    def test_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does_not_exist.yml"))
        assert load_osprey_config() == {}

    def test_resolves_env_var_placeholders(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "system:\n  timezone: ${WS_TZ:-UTC}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.setenv("WS_TZ", "America/Los_Angeles")
        config = load_osprey_config()
        assert config["system"]["timezone"] == "America/Los_Angeles"


class TestResetConfigCache:
    def test_clears_singleton_and_cache(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /x\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        load_osprey_config()  # populates default singleton

        from osprey.utils import config as config_module

        assert config_module._default_config is not None
        reset_config_cache()
        assert config_module._default_config is None
        assert config_module._default_configurable is None
        assert config_module._config_cache == {}


class TestResolveAgentDataRoot:
    def test_uses_base_dir_relative_to_config_parent(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: ./_agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        assert resolve_agent_data_root() == (tmp_path / "_agent_data").resolve()

    def test_custom_base_dir(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: custom/data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        assert resolve_agent_data_root() == (tmp_path / "custom" / "data").resolve()

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_data_root() == (tmp_path / "_agent_data").resolve()

    def test_session_id_appends_isolation_path(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: ./_agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        expected = (tmp_path / "_agent_data").resolve() / "sessions" / "sess-123"
        assert resolve_agent_data_root() == expected

    def test_no_session_id_has_no_sessions_segment(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: ./_agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
        assert "sessions" not in resolve_agent_data_root().parts

    def test_workspace_root_alias(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: ./_agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        assert resolve_workspace_root() == resolve_agent_data_root()


class TestResolveSharedDataRoot:
    def test_ignores_session_isolation(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "agent_data:\n  base_dir: ./_agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        # Shared root must NOT include the per-session segment.
        assert resolve_shared_data_root() == (tmp_path / "_agent_data").resolve()
        assert "sessions" not in resolve_shared_data_root().parts

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_shared_data_root() == (tmp_path / "_agent_data").resolve()


class TestResolvePath:
    def test_absolute_path_returned_unchanged(self, tmp_path, monkeypatch):
        _write_config(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        abs_path = tmp_path / "already" / "absolute"
        assert resolve_path(str(abs_path)) == abs_path

    def test_relative_path_resolved_against_project_root(self, tmp_path, monkeypatch):
        _write_config(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        assert resolve_path("data/limits.json") == tmp_path / "data" / "limits.json"
