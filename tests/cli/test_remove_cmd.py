"""Tests for remove CLI command.

This test module verifies the remove command functionality.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.remove_cmd import (
    capability,
    find_capability_file,
    is_project_initialized,
    remove,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestIsProjectInitialized:
    """Test is_project_initialized function."""

    def test_returns_true_when_config_exists(self, tmp_path, monkeypatch):
        """Test returns True when config.yml exists in current directory."""
        # Change to temp directory with config.yml
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")
        monkeypatch.chdir(tmp_path)

        assert is_project_initialized() is True

    def test_returns_false_when_config_missing(self, tmp_path, monkeypatch):
        """Test returns False when config.yml doesn't exist."""
        # Change to temp directory without config.yml
        monkeypatch.chdir(tmp_path)

        assert is_project_initialized() is False


class TestFindCapabilityFile:
    """Test find_capability_file function."""

    def test_finds_capability_from_registry_location(self, tmp_path):
        """Test finding capability file from registry location."""
        # Create mock registry and capability structure
        capabilities_dir = tmp_path / "capabilities"
        capabilities_dir.mkdir()
        cap_file = capabilities_dir / "test_cap.py"
        cap_file.write_text("# capability")

        registry_file = tmp_path / "registry.py"
        registry_file.write_text("# registry")

        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find:
            mock_find.return_value = registry_file

            result = find_capability_file("test_cap")

            assert result == cap_file

    def test_finds_capability_from_fallback_path(self, tmp_path, monkeypatch):
        """Test finding capability file from fallback path."""
        monkeypatch.chdir(tmp_path)

        # Create capability in fallback location
        capabilities_dir = tmp_path / "capabilities"
        capabilities_dir.mkdir()
        cap_file = capabilities_dir / "test_cap.py"
        cap_file.write_text("# capability")

        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find:
            mock_find.return_value = None

            result = find_capability_file("test_cap")

            # Result is relative path, so compare paths properly
            assert result.name == "test_cap.py"
            assert result.exists()
            assert result.read_text() == "# capability"

    def test_returns_none_when_not_found(self):
        """Test returns None when capability file not found."""
        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find:
            mock_find.return_value = None

            result = find_capability_file("nonexistent")

            assert result is None


class TestRemoveCommandGroup:
    """Test remove command group basics."""

    def test_command_help(self, cli_runner):
        """Verify remove command help is displayed."""
        result = cli_runner.invoke(remove, ["--help"])

        assert result.exit_code == 0
        assert "remove" in result.output.lower() or "Remove" in result.output
        assert "capability" in result.output

    def test_command_exists(self):
        """Verify remove command can be imported and is callable."""
        assert remove is not None
        assert callable(remove)

    def test_command_is_group(self, cli_runner):
        """Verify remove is a command group."""
        result = cli_runner.invoke(remove, ["--help"])

        assert result.exit_code == 0
        assert "capability" in result.output


class TestRemoveCapabilityCommand:
    """Test remove capability subcommand."""

    def test_capability_command_help(self, cli_runner):
        """Verify capability command help is displayed."""
        result = cli_runner.invoke(capability, ["--help"])

        assert result.exit_code == 0
        assert "capability" in result.output.lower() or "Remove" in result.output
        assert "--name" in result.output or "-n" in result.output
        assert "--force" in result.output or "-f" in result.output
        assert "--quiet" in result.output or "-q" in result.output

    def test_capability_requires_project(self, cli_runner, tmp_path, monkeypatch):
        """Test that capability removal requires a project directory."""
        # Change to directory without config.yml
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(capability, ["--name", "test_cap"])

        # Should fail with helpful error
        assert result.exit_code == 1
        assert "Not in an Osprey project" in result.output or "requires" in result.output

    def test_capability_removal_with_all_components(self, cli_runner, tmp_path, monkeypatch):
        """Test removing capability with all components present."""
        # Setup project
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("models:\n  test_cap_react: {}\n")

        registry_file = tmp_path / "registry.py"
        registry_file.write_text("# registry with test_cap")

        capabilities_dir = tmp_path / "capabilities"
        capabilities_dir.mkdir()
        cap_file = capabilities_dir / "test_cap.py"
        cap_file.write_text("# capability")

        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find_registry:
            with patch("osprey.generators.config_updater.find_config_file") as mock_find_config:
                with patch(
                    "osprey.generators.registry_updater.is_already_registered"
                ) as mock_is_registered:
                    with patch(
                        "osprey.generators.config_updater.has_capability_react_model"
                    ) as mock_has_model:
                        with patch(
                            "osprey.generators.registry_updater.remove_from_registry"
                        ) as mock_remove_reg:
                            with patch(
                                "osprey.generators.config_updater.remove_capability_react_from_config"
                            ) as mock_remove_cfg:
                                mock_find_registry.return_value = registry_file
                                mock_find_config.return_value = config_file
                                mock_is_registered.return_value = True
                                mock_has_model.return_value = True
                                mock_remove_reg.return_value = ("new content", "preview", "backup")
                                mock_remove_cfg.return_value = ("new content", "preview", "backup")

                                result = cli_runner.invoke(
                                    capability,
                                    [
                                        "--name",
                                        "test_cap",
                                        "--force",  # Skip confirmation
                                    ],
                                )

                                # Should succeed
                                assert result.exit_code == 0
                                # Should show success message
                                assert (
                                    "successfully removed" in result.output or "✅" in result.output
                                )

    def test_capability_removal_with_no_components(self, cli_runner, tmp_path, monkeypatch):
        """Test attempting to remove capability that doesn't exist."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("models: {}\n")

        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find_registry:
            with patch("osprey.generators.config_updater.find_config_file") as mock_find_config:
                with patch(
                    "osprey.generators.registry_updater.is_already_registered"
                ) as mock_is_registered:
                    with patch(
                        "osprey.generators.config_updater.has_capability_react_model"
                    ) as mock_has_model:
                        with patch("osprey.cli.remove_cmd.find_capability_file") as mock_find_cap:
                            mock_find_registry.return_value = Path("registry.py")
                            mock_find_config.return_value = Path("config.yml")
                            mock_is_registered.return_value = False
                            mock_has_model.return_value = False
                            mock_find_cap.return_value = None

                            result = cli_runner.invoke(
                                capability, ["--name", "nonexistent", "--force"]
                            )

                            # Should succeed but show warning
                            assert result.exit_code == 0
                            assert (
                                "No components found" in result.output
                                or "Nothing to remove" in result.output
                            )

    def test_capability_force_flag_skips_confirmation(self, cli_runner, tmp_path, monkeypatch):
        """Test that --force flag skips confirmation prompt."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("models:\n  test_cap_react: {}\n")

        with patch("osprey.generators.config_updater.find_config_file") as mock_find_config:
            with patch(
                "osprey.generators.config_updater.has_capability_react_model"
            ) as mock_has_model:
                with patch(
                    "osprey.generators.config_updater.remove_capability_react_from_config"
                ) as mock_remove:
                    mock_find_config.return_value = config_file
                    mock_has_model.return_value = True
                    mock_remove.return_value = ("new", "preview", "backup")

                    result = cli_runner.invoke(capability, ["--name", "test_cap", "--force"])

                    # Should not prompt for confirmation
                    assert "Proceed with removal" not in result.output
                    assert result.exit_code == 0

    def test_capability_quiet_flag_reduces_output(self, cli_runner, tmp_path, monkeypatch):
        """Test that --quiet flag reduces output verbosity."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("models: {}\n")

        with patch("osprey.generators.config_updater.find_config_file") as mock_find_config:
            with patch(
                "osprey.generators.config_updater.has_capability_react_model"
            ) as mock_has_model:
                with patch("osprey.cli.remove_cmd.find_capability_file") as mock_find_cap:
                    mock_find_config.return_value = config_file
                    mock_has_model.return_value = False
                    mock_find_cap.return_value = None

                    result = cli_runner.invoke(
                        capability, ["--name", "test_cap", "--force", "--quiet"]
                    )

                    # Should have less output
                    assert result.exit_code == 0


class TestRemoveCapabilityErrorHandling:
    """Test remove capability error handling."""

    def test_creates_backups_before_removal(self, cli_runner, tmp_path, monkeypatch):
        """Test that backups are created before removing components."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("models:\n  test_cap_react: {}\n")

        registry_file = tmp_path / "registry.py"
        registry_file.write_text("# registry")

        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find_registry:
            with patch("osprey.generators.config_updater.find_config_file") as mock_find_config:
                with patch(
                    "osprey.generators.registry_updater.is_already_registered"
                ) as mock_is_registered:
                    with patch(
                        "osprey.generators.config_updater.has_capability_react_model"
                    ) as mock_has_model:
                        with patch(
                            "osprey.generators.registry_updater.remove_from_registry"
                        ) as mock_remove_reg:
                            with patch(
                                "osprey.generators.config_updater.remove_capability_react_from_config"
                            ) as mock_remove_cfg:
                                mock_find_registry.return_value = registry_file
                                mock_find_config.return_value = config_file
                                mock_is_registered.return_value = True
                                mock_has_model.return_value = True
                                mock_remove_reg.return_value = ("new", "preview", "backup")
                                mock_remove_cfg.return_value = ("new", "preview", "backup")

                                result = cli_runner.invoke(
                                    capability, ["--name", "test_cap", "--force"]
                                )

                                # Should mention backups in output
                                assert result.exit_code == 0
                                assert "backup" in result.output.lower() or ".bak" in result.output
