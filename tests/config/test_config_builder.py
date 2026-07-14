"""Tests for configuration system.

Tests the ConfigBuilder class and configuration loading mechanism,
including YAML loading, environment variable resolution, and nested access.
"""

import pytest
import yaml

from osprey.utils.config import ConfigBuilder, get_config_value


class TestConfigBuilder:
    """Test ConfigBuilder class."""

    def test_config_builder_loads_yaml(self, tmp_path):
        """Test that ConfigBuilder loads valid YAML configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
project_root: /test/project
registry_path: ./my_app/registry.py
models:
  orchestrator:
    provider: openai
    model_id: gpt-4
"""
        )

        builder = ConfigBuilder(str(config_file))

        assert builder.raw_config is not None
        assert builder.raw_config["project_root"] == "/test/project"
        assert builder.raw_config["registry_path"] == "./my_app/registry.py"
        assert builder.raw_config["models"]["orchestrator"]["provider"] == "openai"

    def test_config_path_is_a_directory_raises_clear_error(self, tmp_path):
        """A directory at the config path (e.g. a missing bind-mount source the
        container runtime materialized as an empty dir) fails fast with an
        actionable IsADirectoryError, not a raw loader error."""
        config_dir = tmp_path / "config.yml"
        config_dir.mkdir()

        with pytest.raises(IsADirectoryError, match="is a directory, not a file"):
            ConfigBuilder(str(config_dir))

    def test_missing_explicit_config_path_raises_file_not_found(self, tmp_path):
        """An explicitly provided path that does not exist fails fast with a
        clear FileNotFoundError rather than a later open() error."""
        missing = tmp_path / "does_not_exist" / "config.yml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ConfigBuilder(str(missing))

    def test_unreadable_env_file_does_not_crash(self, tmp_path, monkeypatch):
        """A ``.env`` that exists but can't be read (e.g. a 0600 file owned by
        another uid, as happens when the non-root dispatch worker reads a
        host-owned secrets file) must degrade gracefully, not crash-loop the
        process. Regression test for the worker startup PermissionError."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project_root: /test/project\n")
        (tmp_path / ".env").write_text("SOME_KEY=value\n")
        monkeypatch.chdir(tmp_path)

        def _raise_permission_error(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr("dotenv.load_dotenv", _raise_permission_error)

        builder = ConfigBuilder(str(config_file))

        assert builder.raw_config["project_root"] == "/test/project"

    def test_environment_variable_resolution(self, tmp_path, monkeypatch):
        """Test that environment variables are resolved in config."""
        # Set environment variables
        monkeypatch.setenv("TEST_API_KEY", "secret-key-123")
        monkeypatch.setenv("TEST_PROJECT_ROOT", "/home/user/project")

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
project_root: ${TEST_PROJECT_ROOT}
api:
  providers:
    openai:
      api_key: ${TEST_API_KEY}
      base_url: https://api.openai.com
"""
        )

        builder = ConfigBuilder(str(config_file))

        assert builder.raw_config["project_root"] == "/home/user/project"
        assert builder.raw_config["api"]["providers"]["openai"]["api_key"] == "secret-key-123"
        # Non-env-var values should remain unchanged
        assert (
            builder.raw_config["api"]["providers"]["openai"]["base_url"] == "https://api.openai.com"
        )

    def test_nested_path_access_with_get(self, tmp_path):
        """Test accessing nested configuration via dot notation."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
control_system:
  limits:
    max_channels: 100
    timeout: 30
models:
  orchestrator:
    provider: openai
"""
        )

        builder = ConfigBuilder(str(config_file))

        # Test nested path access
        assert builder.get("control_system.limits.max_channels") == 100
        assert builder.get("control_system.limits.timeout") == 30
        assert builder.get("models.orchestrator.provider") == "openai"

        # Test missing paths return default
        assert builder.get("nonexistent.path", "default") == "default"
        assert builder.get("models.nonexistent", None) is None

    def test_missing_config_file_raises_error(self, tmp_path, monkeypatch):
        """Test that missing config file raises clear error."""
        # Change to empty directory where no config.yml exists
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="No config.yml found"):
            ConfigBuilder(None)

    def test_empty_config_file_handled(self, tmp_path):
        """Test that empty config file is handled gracefully."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("")  # Empty file

        builder = ConfigBuilder(str(config_file))

        # Should return empty dict, not crash
        assert builder.raw_config == {}

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises clear error."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
invalid: yaml: syntax:
  - this is broken
    - nested incorrectly
"""
        )

        with pytest.raises(yaml.YAMLError):
            ConfigBuilder(str(config_file))

    def test_configurable_dict_building(self, tmp_path):
        """Test that configurable dict is built correctly."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
project_root: /test/project
registry_path: ./app/registry.py
models:
  orchestrator:
    provider: openai
control_system:
  limits:
    max_channels: 100
"""
        )

        builder = ConfigBuilder(str(config_file))

        assert builder.configurable is not None
        assert isinstance(builder.configurable, dict)
        assert "model_configs" in builder.configurable
        assert "project_root" in builder.configurable

    def test_model_configs_loaded(self, tmp_path):
        """Test that model configurations are loaded correctly."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
models:
  orchestrator:
    provider: cborg
    model_id: anthropic/claude-sonnet
  response:
    provider: openai
    model_id: gpt-4
    max_tokens: 4096
"""
        )

        builder = ConfigBuilder(str(config_file))

        model_configs = builder.configurable["model_configs"]
        assert "orchestrator" in model_configs
        assert model_configs["orchestrator"]["provider"] == "cborg"
        assert model_configs["orchestrator"]["model_id"] == "anthropic/claude-sonnet"

        assert "response" in model_configs
        assert model_configs["response"]["provider"] == "openai"
        assert model_configs["response"]["max_tokens"] == 4096

    def test_registry_path_from_config(self, tmp_path):
        """Test that registry_path is accessible from config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
registry_path: ./src/my_app/registry.py
application:
  registry_path: ./src/other_app/registry.py
"""
        )

        builder = ConfigBuilder(str(config_file))

        # Should support both formats
        assert builder.get("registry_path") == "./src/my_app/registry.py"
        assert builder.get("application.registry_path") == "./src/other_app/registry.py"

    def test_get_unexpanded_config_preserves_env_var_placeholders(self, tmp_path, monkeypatch):
        """Test that get_unexpanded_config() preserves ${VAR} placeholders.

        This is critical for deployment security - we must NOT write actual
        API keys to config files in the build directory. Instead, ${VAR}
        placeholders should be preserved and resolved at container runtime.

        Fixes: https://github.com/als-apg/osprey/issues/118
        """
        # Set environment variable that would normally be expanded
        monkeypatch.setenv("MY_SECRET_API_KEY", "actual-secret-value-12345")

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
api:
  providers:
    cborg:
      api_key: ${MY_SECRET_API_KEY}
      base_url: https://api.cborg.lbl.gov/v1
project_root: /test/project
"""
        )

        builder = ConfigBuilder(str(config_file))

        # raw_config should have the expanded (resolved) value
        assert (
            builder.raw_config["api"]["providers"]["cborg"]["api_key"]
            == "actual-secret-value-12345"
        )

        # get_unexpanded_config() should preserve the ${VAR} placeholder
        unexpanded = builder.get_unexpanded_config()
        assert unexpanded["api"]["providers"]["cborg"]["api_key"] == "${MY_SECRET_API_KEY}"

        # Non-env-var values should be the same in both
        assert unexpanded["api"]["providers"]["cborg"]["base_url"] == "https://api.cborg.lbl.gov/v1"
        assert unexpanded["project_root"] == "/test/project"

    def test_get_unexpanded_config_returns_deep_copy(self, tmp_path):
        """Test that get_unexpanded_config() returns a deep copy (mutations don't affect original)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
api:
  key: ${SECRET}
"""
        )

        builder = ConfigBuilder(str(config_file))

        # Get unexpanded config and mutate it
        unexpanded1 = builder.get_unexpanded_config()
        unexpanded1["api"]["key"] = "MUTATED"

        # Second call should return original value, not mutated
        unexpanded2 = builder.get_unexpanded_config()
        assert unexpanded2["api"]["key"] == "${SECRET}"


class TestConfigGlobalAccess:
    """Test global configuration access functions."""

    def test_get_config_value_with_path(self, tmp_path, monkeypatch):
        """Test get_config_value function with dot-separated path."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
project_root: /test/project
control_system:
  limits:
    max_channels: 100
"""
        )

        # Set up global config
        monkeypatch.setenv("CONFIG_FILE", str(config_file))

        # Reset global config
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None

        # Test access
        value = get_config_value("control_system.limits.max_channels", 0)
        assert value == 100  # Should retrieve the value from config


class TestGetFacilityTimezone:
    """``get_facility_timezone`` must never raise — it is the safe default for
    every synthesis primitive and timestamp render site, so an unloaded config
    or a misconfigured zone name must degrade to UTC, not blow up the caller."""

    @staticmethod
    def _reset_config_singleton():
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None

    def test_returns_utc_when_no_config_loaded(self, tmp_path, monkeypatch):
        """No config.yml in cwd and no CONFIG_FILE → UTC, not FileNotFoundError."""
        from zoneinfo import ZoneInfo

        from osprey.utils.config import get_facility_timezone

        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.chdir(tmp_path)  # empty dir, no config.yml
        self._reset_config_singleton()

        assert get_facility_timezone() == ZoneInfo("UTC")

    def test_returns_utc_when_configured_zone_is_invalid(self, tmp_path, monkeypatch):
        """A typo'd system.timezone → UTC fallback, not ZoneInfoNotFoundError."""
        from zoneinfo import ZoneInfo

        from osprey.utils.config import get_facility_timezone

        config_file = tmp_path / "config.yml"
        config_file.write_text("system:\n  timezone: Not/ARealZone\n")
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        self._reset_config_singleton()

        assert get_facility_timezone() == ZoneInfo("UTC")

    def test_returns_configured_zone_when_valid(self, tmp_path, monkeypatch):
        """A valid system.timezone is honored — the fallback must not swallow it."""
        from zoneinfo import ZoneInfo

        from osprey.utils.config import get_facility_timezone

        config_file = tmp_path / "config.yml"
        config_file.write_text("system:\n  timezone: America/Los_Angeles\n")
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        self._reset_config_singleton()

        assert get_facility_timezone() == ZoneInfo("America/Los_Angeles")


class TestToFacilityIso:
    """``to_facility_iso`` is the single shared timestamp-egress transform for the
    ARIEL MCP and web output paths — both must render the same facility-local ISO
    string so the two paths can't drift again."""

    @staticmethod
    def _reset_config_singleton():
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None

    def test_none_passes_through(self):
        from osprey.utils.config import to_facility_iso

        assert to_facility_iso(None) is None

    def test_non_datetime_degrades_to_str(self):
        """A value already serialized upstream (or any non-datetime) must not crash."""
        from osprey.utils.config import to_facility_iso

        assert to_facility_iso("2026-06-01T00:00:00+00:00") == "2026-06-01T00:00:00+00:00"

    def test_naive_datetime_is_stamped_facility_local_not_box_local(self, tmp_path, monkeypatch):
        """A naive datetime is assumed facility-local wall-clock (stamped, not
        re-interpreted as box-local). Guards against ``astimezone`` reintroducing
        $TZ dependence on the egress side — mirrors the parse-side contract."""
        from datetime import datetime

        from osprey.utils.config import to_facility_iso

        config_file = tmp_path / "config.yml"
        config_file.write_text("system:\n  timezone: Asia/Tokyo\n")  # +09:00
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        # A divergent box $TZ must NOT influence the result.
        monkeypatch.setenv("TZ", "America/Los_Angeles")
        self._reset_config_singleton()

        out = to_facility_iso(datetime(2026, 6, 1, 9, 0, 0))  # naive 09:00
        assert out == "2026-06-01T09:00:00+09:00"  # 09:00 stamped Tokyo, not shifted

    def test_aware_datetime_rendered_in_facility_zone(self, tmp_path, monkeypatch):
        """A UTC instant is rendered in the configured facility zone, with offset."""
        from datetime import UTC, datetime

        from osprey.utils.config import to_facility_iso

        config_file = tmp_path / "config.yml"
        config_file.write_text("system:\n  timezone: Asia/Tokyo\n")  # +09:00, no DST
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        self._reset_config_singleton()

        out = to_facility_iso(datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC))  # midnight UTC
        assert out == "2026-06-01T09:00:00+09:00"  # 09:00 Tokyo, explicit offset


class TestTimezoneDriftWarning:
    """``get_facility_timezone`` warns once if the effective container ``$TZ``
    disagrees with an explicitly-configured ``system.timezone`` — a guardrail for
    a misconfigured deploy where the OS clock and the agent zone diverge. The
    resolver reads ``$TZ`` at call time, so tests set it after loading config."""

    @staticmethod
    def _reset(monkeypatch):
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        monkeypatch.setattr(config_module, "_tz_drift_warned", False, raising=False)

    def _load_explicit_zone(self, tmp_path, monkeypatch, zone: str):
        config_file = tmp_path / "config.yml"
        config_file.write_text(f"system:\n  timezone: {zone}\n")
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        self._reset(monkeypatch)
        # Trigger the one-time config load (runs load_dotenv) before we pin $TZ,
        # so the value we set below is what the resolver reads at call time.
        from osprey.utils.config import get_config_value

        get_config_value("system.timezone", None)

    def test_warns_once_on_drift(self, tmp_path, monkeypatch, caplog):
        import logging
        from zoneinfo import ZoneInfo

        from osprey.utils.config import get_facility_timezone

        self._load_explicit_zone(tmp_path, monkeypatch, "America/Los_Angeles")
        monkeypatch.setenv("TZ", "UTC")  # diverges from system.timezone

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            get_facility_timezone()
            get_facility_timezone()  # second call must not re-warn

        drift = [r for r in caplog.records if "differs from the facility" in r.message]
        assert len(drift) == 1
        # The value returned is still authoritative system.timezone, not $TZ.
        assert get_facility_timezone() == ZoneInfo("America/Los_Angeles")

    def test_no_warning_when_tz_matches(self, tmp_path, monkeypatch, caplog):
        import logging

        from osprey.utils.config import get_facility_timezone

        self._load_explicit_zone(tmp_path, monkeypatch, "America/Los_Angeles")
        monkeypatch.setenv("TZ", "America/Los_Angeles")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            get_facility_timezone()

        assert not [r for r in caplog.records if "differs from the facility" in r.message]

    def test_no_warning_when_timezone_not_explicitly_configured(
        self, tmp_path, monkeypatch, caplog
    ):
        """Config absent → implicit-UTC default; a set $TZ must stay quiet (else
        every CI run with $TZ set but no system.timezone would warn spuriously)."""
        import logging

        from osprey.utils.config import get_config_value, get_facility_timezone

        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")  # no system.timezone
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        self._reset(monkeypatch)
        get_config_value("system.timezone", None)
        monkeypatch.setenv("TZ", "America/New_York")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            get_facility_timezone()

        assert not [r for r in caplog.records if "differs from the facility" in r.message]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
