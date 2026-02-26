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
execution_control:
  limits:
    max_retries: 3
    graph_recursion_limit: 100
models:
  orchestrator:
    provider: openai
"""
        )

        builder = ConfigBuilder(str(config_file))

        # Test nested path access
        assert builder.get("execution_control.limits.max_retries") == 3
        assert builder.get("execution_control.limits.graph_recursion_limit") == 100
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
execution_control:
  limits:
    graph_recursion_limit: 100
"""
        )

        builder = ConfigBuilder(str(config_file))

        assert builder.configurable is not None
        assert isinstance(builder.configurable, dict)
        assert "model_configs" in builder.configurable
        assert "execution_limits" in builder.configurable
        assert "project_root" in builder.configurable

    def test_execution_limits_loaded(self, tmp_path):
        """Test that execution limits are loaded and accessible."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
execution_control:
  limits:
    graph_recursion_limit: 150
    max_reclassifications: 2
    max_planning_attempts: 3
    max_step_retries: 5
    max_execution_time_seconds: 600
"""
        )

        builder = ConfigBuilder(str(config_file))

        limits = builder.configurable["execution_limits"]
        assert limits["graph_recursion_limit"] == 150
        assert limits["max_reclassifications"] == 2
        assert limits["max_planning_attempts"] == 3
        assert limits["max_step_retries"] == 5
        assert limits["max_execution_time_seconds"] == 600

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
execution_control:
  limits:
    max_retries: 3
"""
        )

        # Set up global config
        monkeypatch.setenv("CONFIG_FILE", str(config_file))

        # Reset global config
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None

        # Test access
        value = get_config_value("execution_control.limits.max_retries", 0)
        assert value == 3  # Should retrieve the value from config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
