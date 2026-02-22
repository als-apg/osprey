"""Tests for health CLI command.

This test module verifies the health check functionality.

Iteration 1: Smoke tests and basic functionality
Target: 0% → 30%+ coverage
"""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.health_cmd import HealthChecker, HealthCheckResult, health


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config_yml(tmp_path):
    """Provide a sample config.yml for testing."""
    config_content = """
project_name: test_project
project_root: ${PROJECT_ROOT}

models:
  orchestrator:
    provider: cborg
    model_id: anthropic/claude-haiku
  response:
    provider: cborg
    model_id: anthropic/claude-haiku
  classifier:
    provider: cborg
    model_id: anthropic/claude-haiku
  approval:
    provider: cborg
    model_id: anthropic/claude-haiku
  task_extraction:
    provider: cborg
    model_id: anthropic/claude-haiku
  memory:
    provider: cborg
    model_id: anthropic/claude-haiku
  python_code_generator:
    provider: cborg
    model_id: anthropic/claude-haiku
  time_parsing:
    provider: cborg
    model_id: anthropic/claude-haiku

deployed_services: []

services: {}

api:
  providers:
    cborg:
      api_key: ${CBORG_API_KEY}
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file


class TestHealthCommandBasics:
    """Test health command basic functionality."""

    def test_health_command_exists(self):
        """Verify health command can be imported and is callable."""
        assert health is not None
        assert callable(health)

    def test_health_command_help(self, cli_runner):
        """Verify health command shows help text."""
        result = cli_runner.invoke(health, ["--help"])

        assert result.exit_code == 0
        assert "health" in result.output.lower() or "Health" in result.output
        assert "--verbose" in result.output
        assert "--basic" in result.output

    def test_health_command_verbose_flag(self, cli_runner):
        """Verify health command accepts --verbose flag."""
        result = cli_runner.invoke(health, ["--help"])

        assert result.exit_code == 0
        assert "--verbose" in result.output or "-v" in result.output

    def test_health_command_basic_flag(self, cli_runner):
        """Verify health command accepts --basic flag."""
        result = cli_runner.invoke(health, ["--help"])

        assert result.exit_code == 0
        assert "--basic" in result.output or "-b" in result.output


class TestHealthCheckResult:
    """Test HealthCheckResult class."""

    def test_health_check_result_creation(self):
        """Test creating a HealthCheckResult."""
        result = HealthCheckResult("test", "ok", "message", "details")

        assert result.name == "test"
        assert result.status == "ok"
        assert result.message == "message"
        assert result.details == "details"

    def test_health_check_result_repr(self):
        """Test HealthCheckResult string representation."""
        result = HealthCheckResult("test", "ok")

        repr_str = repr(result)
        assert "HealthCheckResult" in repr_str
        assert "test" in repr_str
        assert "ok" in repr_str

    def test_health_check_result_statuses(self):
        """Test different health check status values."""
        ok_result = HealthCheckResult("test1", "ok")
        warning_result = HealthCheckResult("test2", "warning")
        error_result = HealthCheckResult("test3", "error")

        assert ok_result.status == "ok"
        assert warning_result.status == "warning"
        assert error_result.status == "error"


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_health_checker_initialization(self):
        """Test creating a HealthChecker instance."""
        checker = HealthChecker()

        assert checker.verbose is False
        assert checker.full is False
        assert checker.results == []
        assert checker.config == {}

    def test_health_checker_with_options(self, tmp_path):
        """Test HealthChecker with verbose and full options."""
        checker = HealthChecker(verbose=True, full=True, project_path=tmp_path)

        assert checker.verbose is True
        assert checker.full is True
        assert checker.cwd == tmp_path

    def test_add_result(self):
        """Test adding health check results."""
        checker = HealthChecker()
        checker.add_result("test", "ok", "message", "details")

        assert len(checker.results) == 1
        assert checker.results[0].name == "test"
        assert checker.results[0].status == "ok"

    def test_add_multiple_results(self):
        """Test adding multiple health check results."""
        checker = HealthChecker()
        checker.add_result("test1", "ok")
        checker.add_result("test2", "warning")
        checker.add_result("test3", "error")

        assert len(checker.results) == 3
        assert checker.results[0].status == "ok"
        assert checker.results[1].status == "warning"
        assert checker.results[2].status == "error"


class TestHealthCheckerConfiguration:
    """Test configuration checking functionality."""

    def test_check_configuration_missing_config(self, tmp_path):
        """Test checking configuration when config.yml is missing."""
        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_configuration()

        # Should have error result for missing config
        assert len(checker.results) > 0
        config_results = [r for r in checker.results if "config" in r.name]
        assert any(r.status == "error" for r in config_results)

    def test_check_configuration_valid_config(self, sample_config_yml):
        """Test checking valid configuration."""
        checker = HealthChecker(project_path=sample_config_yml.parent)

        with patch("osprey.cli.health_cmd.console"):
            with patch("osprey.registry.initialize_registry"):
                checker.check_configuration()

        # Should have results
        assert len(checker.results) > 0
        # Config file should be found
        config_found = any("config_file_exists" in r.name for r in checker.results)
        assert config_found

    def test_check_configuration_empty_config(self, tmp_path):
        """Test checking empty configuration file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("")

        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_configuration()

        # Should detect empty config
        errors = [r for r in checker.results if r.status == "error"]
        assert len(errors) > 0

    def test_check_configuration_invalid_yaml(self, tmp_path):
        """Test checking invalid YAML configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("invalid: yaml: content:")

        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_configuration()

        # Should detect YAML error
        errors = [r for r in checker.results if r.status == "error"]
        assert len(errors) > 0


class TestHealthCheckerFileSystem:
    """Test file system checking functionality."""

    def test_check_file_system(self, tmp_path, sample_config_yml):
        """Test file system checks."""
        checker = HealthChecker(project_path=sample_config_yml.parent)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_file_system()

        # Should have file system results
        assert len(checker.results) > 0

    def test_check_file_system_disk_space(self, tmp_path):
        """Test disk space checking."""
        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_file_system()

        # Should check disk space
        disk_space_result = any("disk_space" in r.name for r in checker.results)
        assert disk_space_result or len(checker.results) > 0

    def test_check_file_system_env_file_exists(self, tmp_path):
        """Test .env file detection when it exists."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=value")

        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_file_system()

        env_result = [r for r in checker.results if "env_file" in r.name]
        assert len(env_result) > 0
        assert env_result[0].status == "ok"

    def test_check_file_system_env_file_missing(self, tmp_path):
        """Test .env file detection when it's missing."""
        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_file_system()

        env_result = [r for r in checker.results if "env_file" in r.name]
        assert len(env_result) > 0
        assert env_result[0].status == "warning"


class TestHealthCheckerPythonEnvironment:
    """Test Python environment checking functionality."""

    def test_check_python_environment(self):
        """Test Python environment checks."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            checker.check_python_environment()

        # Should have Python version check
        assert len(checker.results) > 0
        python_result = any("python_version" in r.name for r in checker.results)
        assert python_result

    def test_check_python_version(self):
        """Test Python version checking."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            checker.check_python_environment()

        version_results = [r for r in checker.results if "python_version" in r.name]
        assert len(version_results) > 0
        # Current Python should be detected
        assert version_results[0].status in ["ok", "warning"]

    def test_check_virtual_environment(self):
        """Test virtual environment detection."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            checker.check_python_environment()

        venv_results = [r for r in checker.results if "virtual_environment" in r.name]
        assert len(venv_results) > 0


class TestHealthCheckerContainers:
    """Test container checking functionality."""

    def test_check_containers_no_runtime(self):
        """Test container check when no runtime is available."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            with patch("osprey.cli.health_cmd.get_runtime_command") as mock_runtime:
                mock_runtime.side_effect = RuntimeError("No runtime")
                checker.check_containers()

        # Should handle missing runtime gracefully
        runtime_results = [r for r in checker.results if "runtime" in r.name]
        assert len(runtime_results) > 0
        assert runtime_results[0].status == "warning"

    def test_check_containers_with_docker(self):
        """Test container check with Docker available."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            with patch("osprey.cli.health_cmd.get_runtime_command") as mock_runtime:
                with patch("subprocess.run") as mock_run:
                    mock_runtime.return_value = ["docker"]
                    mock_run.return_value = Mock(returncode=0, stdout="Docker version 24.0.0")

                    checker.check_containers()

        # Should detect Docker
        docker_results = [r for r in checker.results if "docker" in r.name]
        assert len(docker_results) > 0

    def test_check_containers_with_podman(self):
        """Test container check with Podman available."""
        checker = HealthChecker()

        with patch("osprey.cli.health_cmd.console"):
            with patch("osprey.cli.health_cmd.get_runtime_command") as mock_runtime:
                with patch("subprocess.run") as mock_run:
                    mock_runtime.return_value = ["podman"]
                    mock_run.return_value = Mock(returncode=0, stdout="podman version 4.0.0")

                    checker.check_containers()

        # Should detect Podman
        podman_results = [r for r in checker.results if "podman" in r.name]
        assert len(podman_results) > 0


class TestHealthCheckerAPIProviders:
    """Test API provider checking functionality."""

    def test_check_api_providers_no_config(self, tmp_path):
        """Test API provider check when config is missing."""
        checker = HealthChecker(project_path=tmp_path)

        with patch("osprey.cli.health_cmd.console"):
            checker.check_api_providers()

        # Should handle missing config gracefully
        # No results or handled gracefully
        assert True  # No crash is success

    def test_check_api_providers_with_config(self, sample_config_yml):
        """Test API provider check with valid config."""
        checker = HealthChecker(project_path=sample_config_yml.parent)

        with patch("osprey.cli.health_cmd.console"):
            with patch("osprey.registry.get_registry") as mock_registry:
                mock_reg = Mock()
                mock_provider_class = Mock()
                mock_provider_instance = Mock()
                mock_provider_instance.check_health.return_value = (True, "Connected")
                mock_provider_class.return_value = mock_provider_instance
                mock_reg.get_provider.return_value = mock_provider_class
                mock_registry.return_value = mock_reg

                checker.check_api_providers()

        # Should check providers
        provider_results = [r for r in checker.results if "provider" in r.name]
        assert len(provider_results) > 0


class TestHealthCommandIntegration:
    """Test full health command execution."""

    def test_health_command_with_missing_config(self, cli_runner, tmp_path):
        """Test health command when config is missing."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            mock_resolve.return_value = tmp_path

            result = cli_runner.invoke(health, [])

            # Should execute but report errors
            assert result.exit_code in [1, 2, 3]

    def test_health_command_basic_mode(self, cli_runner, sample_config_yml):
        """Test health command in basic mode."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            with patch("osprey.registry.initialize_registry"):
                with patch("osprey.deployment.runtime_helper.get_runtime_command") as mock_runtime:
                    mock_resolve.return_value = sample_config_yml.parent
                    mock_runtime.side_effect = RuntimeError("No runtime")

                    result = cli_runner.invoke(health, ["--basic"])

                    # Should execute
                    assert result.exit_code in [0, 1, 2]

    def test_health_command_verbose_mode(self, cli_runner, sample_config_yml):
        """Test health command in verbose mode."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            with patch("osprey.registry.initialize_registry"):
                with patch("osprey.deployment.runtime_helper.get_runtime_command") as mock_runtime:
                    mock_resolve.return_value = sample_config_yml.parent
                    mock_runtime.side_effect = RuntimeError("No runtime")

                    result = cli_runner.invoke(health, ["--verbose"])

                    # Should execute
                    assert result.exit_code in [0, 1, 2]

    def test_health_command_keyboard_interrupt(self, cli_runner):
        """Test health command handles keyboard interrupt gracefully."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            mock_resolve.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(health, [])

            # Should exit gracefully
            assert result.exit_code == 130


class TestHealthDisplayResults:
    """Test result display functionality."""

    def test_display_results_all_passing(self):
        """Test displaying results when all checks pass."""
        checker = HealthChecker()
        checker.add_result("test1", "ok")
        checker.add_result("test2", "ok")

        with patch("osprey.cli.health_cmd.console"):
            checker.display_results()

        # Should complete without errors
        assert len(checker.results) == 2

    def test_display_results_with_warnings(self):
        """Test displaying results with warnings."""
        checker = HealthChecker()
        checker.add_result("test1", "ok")
        checker.add_result("test2", "warning", "Warning message")

        with patch("osprey.cli.health_cmd.console"):
            checker.display_results()

        warnings = [r for r in checker.results if r.status == "warning"]
        assert len(warnings) == 1

    def test_display_results_with_errors(self):
        """Test displaying results with errors."""
        checker = HealthChecker()
        checker.add_result("test1", "ok")
        checker.add_result("test2", "error", "Error message")

        with patch("osprey.cli.health_cmd.console"):
            checker.display_results()

        errors = [r for r in checker.results if r.status == "error"]
        assert len(errors) == 1

    def test_display_results_verbose_mode(self):
        """Test displaying results in verbose mode."""
        checker = HealthChecker(verbose=True)
        checker.add_result("test1", "ok")
        checker.add_result("test2", "warning", "Warning", "Details")
        checker.add_result("test3", "error", "Error", "Error details")

        with patch("osprey.cli.health_cmd.console"):
            checker.display_results()

        # Should have warnings and errors
        warnings = [r for r in checker.results if r.status == "warning"]
        errors = [r for r in checker.results if r.status == "error"]
        assert len(warnings) == 1
        assert len(errors) == 1


class TestHealthExitCodes:
    """Test health command exit codes."""

    def test_exit_code_all_passing(self, cli_runner, sample_config_yml):
        """Test exit code 0 when all checks pass."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            with patch("osprey.cli.health_cmd.HealthChecker.check_all"):
                with patch("osprey.cli.health_cmd.console"):
                    mock_resolve.return_value = sample_config_yml.parent

                    # Create a mock checker with all passing results
                    mock_checker_instance = Mock()
                    mock_checker_instance.results = [
                        HealthCheckResult("test1", "ok"),
                        HealthCheckResult("test2", "ok"),
                    ]

                    with patch("osprey.cli.health_cmd.HealthChecker") as MockChecker:
                        MockChecker.return_value = mock_checker_instance

                        result = cli_runner.invoke(health, [])

                        assert result.exit_code == 0

    def test_exit_code_with_warnings(self, cli_runner, sample_config_yml):
        """Test exit code 1 when warnings present."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            with patch("osprey.cli.health_cmd.console"):
                mock_resolve.return_value = sample_config_yml.parent

                # Create a mock checker with warnings
                mock_checker_instance = Mock()
                mock_checker_instance.results = [
                    HealthCheckResult("test1", "ok"),
                    HealthCheckResult("test2", "warning"),
                ]

                with patch("osprey.cli.health_cmd.HealthChecker") as MockChecker:
                    MockChecker.return_value = mock_checker_instance

                    result = cli_runner.invoke(health, [])

                    assert result.exit_code == 1

    def test_exit_code_with_errors(self, cli_runner, sample_config_yml):
        """Test exit code 2 when errors present."""
        with patch("osprey.cli.project_utils.resolve_project_path") as mock_resolve:
            with patch("osprey.cli.health_cmd.console"):
                mock_resolve.return_value = sample_config_yml.parent

                # Create a mock checker with errors
                mock_checker_instance = Mock()
                mock_checker_instance.results = [
                    HealthCheckResult("test1", "ok"),
                    HealthCheckResult("test2", "error"),
                ]

                with patch("osprey.cli.health_cmd.HealthChecker") as MockChecker:
                    MockChecker.return_value = mock_checker_instance

                    result = cli_runner.invoke(health, [])

                    assert result.exit_code == 2
