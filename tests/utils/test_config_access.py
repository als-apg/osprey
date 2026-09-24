"""Tests for config.py's singleton/caching layer and path/accessor helpers.

This file deliberately does NOT re-test ``ConfigBuilder`` construction,
``resolve_env_vars``, or the timezone/ISO helpers — those are covered by
``tests/config/test_config_builder.py`` and ``tests/utils/test_resolve_env_vars.py``.
It targets the public functions those files leave untested: the
``get_config_builder`` / ``load_config`` / ``get_full_configuration`` caching and
set-as-default semantics, ``get_agent_dir`` path resolution, the service
accessor, and the built-in ``execution`` / ``python_executor`` defaults.

The autouse ``reset_state_between_tests`` fixture clears the config singleton and
``CONFIG_FILE`` around every test, so these serial tests leak no global state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.utils.config import (
    ConfigBuilder,
    get_agent_dir,
    get_config_builder,
    get_config_value,
    get_framework_service_config,
    get_full_configuration,
    load_config,
    load_project_config,
)


def _write_config(dir_path: Path, body: str) -> Path:
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


class TestGetConfigBuilderCaching:
    def test_same_explicit_path_returns_cached_instance(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        first = get_config_builder(str(cfg))
        second = get_config_builder(str(cfg))
        assert first is second

    def test_different_paths_return_distinct_instances(self, tmp_path):
        cfg_a = _write_config(tmp_path, "project_root: /a\n")
        other = tmp_path / "b"
        other.mkdir()
        cfg_b = _write_config(other, "project_root: /b\n")
        assert get_config_builder(str(cfg_a)) is not get_config_builder(str(cfg_b))

    def test_set_as_default_makes_no_arg_calls_return_it(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        explicit = get_config_builder(str(cfg), set_as_default=True)
        assert get_config_builder() is explicit

    def test_no_arg_uses_config_file_env(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, "project_root: /via-env\n")
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        builder = get_config_builder()
        assert builder.get("project_root") == "/via-env"

    def test_returns_config_builder_instance(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        assert isinstance(get_config_builder(str(cfg)), ConfigBuilder)


class TestLoadConfig:
    def test_returns_raw_config_dict(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\napi:\n  key: abc\n")
        config = load_config(str(cfg))
        assert config["project_root"] == "/x"
        assert config["api"]["key"] == "abc"

    def test_resolves_env_placeholders(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, "system:\n  timezone: ${CA_TZ:-UTC}\n")
        monkeypatch.setenv("CA_TZ", "Europe/Berlin")
        config = load_config(str(cfg))
        assert config["system"]["timezone"] == "Europe/Berlin"


class TestGetConfigValue:
    def test_explicit_path_dot_access(self, tmp_path):
        cfg = _write_config(tmp_path, "control_system:\n  limits:\n    max: 100\n")
        assert get_config_value("control_system.limits.max", 0, str(cfg)) == 100

    def test_missing_path_returns_default(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        assert get_config_value("nope.missing", "fallback", str(cfg)) == "fallback"

    def test_empty_path_raises_value_error(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /x\n")
        get_config_builder(str(cfg), set_as_default=True)
        import pytest

        with pytest.raises(ValueError, match="cannot be empty"):
            get_config_value("")


class TestGetFullConfiguration:
    def test_returns_configurable_and_sets_default(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /root\nsystem:\n  timezone: UTC\n")
        configurable = get_full_configuration(str(cfg))
        assert configurable["project_root"] == "/root"
        # Passing an explicit path also promotes it to the default singleton.
        assert get_config_builder().get("project_root") == "/root"

    def test_execution_defaults_when_section_absent(self, tmp_path):
        """A config without ``execution:`` still exposes the subprocess defaults."""
        cfg = _write_config(tmp_path, "project_root: /root\n")
        configurable = get_full_configuration(str(cfg))
        assert configurable["execution"] == {"execution_method": "subprocess"}

    def test_python_executor_defaults_when_section_absent(self, tmp_path):
        cfg = _write_config(tmp_path, "project_root: /root\n")
        configurable = get_full_configuration(str(cfg))
        assert configurable["python_executor"] == {"execution_timeout_seconds": 600}


class TestGetFrameworkServiceConfig:
    def test_returns_named_service_config(self, tmp_path):
        cfg = _write_config(
            tmp_path,
            "services:\n  channel_finder:\n    port: 9100\n",
        )
        get_config_builder(str(cfg), set_as_default=True)
        assert get_framework_service_config("channel_finder") == {"port": 9100}

    def test_unknown_service_returns_empty_dict(self, tmp_path):
        cfg = _write_config(tmp_path, "services:\n  other: {}\n")
        get_config_builder(str(cfg), set_as_default=True)
        assert get_framework_service_config("missing") == {}


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
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        result = Path(get_agent_dir("api_calls_dir"))
        assert result == tmp_path / "_agent_data" / "logs" / "api"

    def test_unknown_sub_dir_falls_back_to_its_name(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path,
            f"project_root: {tmp_path}\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
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
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        result = Path(get_agent_dir("plans_dir"))
        assert result == tmp_path / "_agent_data" / "default_plans"

    def test_missing_project_root_falls_back_to_relative(self, tmp_path, monkeypatch):
        cfg = _write_config(
            tmp_path,
            "project_root: /nonexistent/project/root\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        result = Path(get_agent_dir("memory_dir"))
        # Degrades to an absolute cwd-relative path, not the dead project root.
        assert result.is_absolute()
        assert result.name == "memory_dir"
        assert "_agent_data" in result.parts

    def test_host_path_resolves_under_the_configured_root_even_when_it_is_absent(
        self, tmp_path, monkeypatch
    ):
        cfg = _write_config(
            tmp_path,
            "project_root: /nonexistent/project/root\nagent_data:\n  base_dir: _agent_data\n",
        )
        monkeypatch.setenv("CONFIG_FILE", str(cfg))

        result = Path(get_agent_dir("memory_dir", host_path=True))

        assert result == Path("/nonexistent/project/root/_agent_data/memory_dir")

    def test_no_project_root_resolves_relative_to_the_cwd_with_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        cfg = _write_config(tmp_path, "agent_data:\n  base_dir: _agent_data\n")
        monkeypatch.setenv("CONFIG_FILE", str(cfg))
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            result = Path(get_agent_dir("memory_dir"))

        assert result == (tmp_path / "_agent_data" / "memory_dir").resolve()
        assert "No project root configured" in caplog.text


class TestLoadProjectConfig:
    def test_returns_the_expanded_mapping(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_TEST_ROOT", "/expanded")
        cfg = _write_config(tmp_path, "project_root: ${OSPREY_TEST_ROOT}\n")

        assert load_project_config(cfg)["project_root"] == "/expanded"

    def test_without_wrap_errors_the_original_failure_is_raised(self, tmp_path):
        cfg = _write_config(tmp_path, "- not\n- a mapping\n")

        with pytest.raises(ValueError, match="must contain a dictionary/mapping"):
            load_project_config(cfg)

    def test_with_wrap_errors_the_failure_names_the_config_file(self, tmp_path):
        cfg = _write_config(tmp_path, "- not\n- a mapping\n")

        with pytest.raises(RuntimeError, match=f"Could not load config file {cfg}: ") as caught:
            load_project_config(cfg, wrap_errors=True)

        assert isinstance(caught.value.__cause__, ValueError)
