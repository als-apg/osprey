"""Tests for registry CLI command display functionality.

This test module verifies the registry display functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey.cli.registry_cmd import (
    _display_providers_table,
    _display_services_table,
    display_registry_contents,
    handle_registry_action,
)


@pytest.fixture
def mock_registry():
    """Create a mock registry with test data."""
    registry = MagicMock()

    # Mock stats
    registry.get_stats.return_value = {
        "services": 1,
        "service_names": ["test_service"],
    }

    # Mock services
    registry.get_service.return_value = MagicMock(__class__=MagicMock(__name__="TestService"))

    # Mock providers
    registry.list_providers.return_value = ["test_provider"]
    registry.get_provider.return_value = MagicMock(description="Test AI provider")

    registry._initialized = True

    return registry


class TestDisplayRegistryContents:
    """Test display_registry_contents function."""

    def test_displays_registry_with_initialized_registry(self, mock_registry):
        """Test displaying registry contents when registry is already initialized."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=False)

                # Should succeed
                assert result is True
                # Should get stats
                assert mock_registry.get_stats.called

    def test_initializes_registry_if_not_initialized(self, mock_registry):
        """Test that uninitialized registry gets initialized."""
        mock_registry._initialized = False

        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=False)

                # Should initialize registry
                assert mock_registry.initialize.called
                assert result is True

    def test_handles_exceptions_gracefully(self):
        """Test that exceptions are handled gracefully."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.side_effect = Exception("Test error")

                result = display_registry_contents(verbose=False)

                # Should return False on error
                assert result is False

    def test_verbose_mode_shows_additional_info(self, mock_registry):
        """Test that verbose mode displays additional information."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=True)

                # Should succeed
                assert result is True


class TestDisplayServicesTable:
    """Test _display_services_table function."""

    def test_displays_services(self, mock_registry):
        """Test displaying services table."""
        # Should not raise exception
        _display_services_table(mock_registry, verbose=False)

        # Should get stats for service names
        assert mock_registry.get_stats.called


class TestDisplayProvidersTable:
    """Test _display_providers_table function."""

    def test_displays_providers(self, mock_registry):
        """Test displaying providers table."""
        providers = ["test_provider", "another_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=False)

        # Should get provider classes
        assert mock_registry.get_provider.called

    def test_displays_providers_verbose(self, mock_registry):
        """Test displaying providers table in verbose mode."""
        providers = ["test_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=True)

    def test_handles_missing_provider(self, mock_registry):
        """Test handling of provider that doesn't exist."""
        mock_registry.get_provider.return_value = None
        providers = ["nonexistent_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=False)


class TestHandleRegistryAction:
    """Test handle_registry_action function."""

    def test_displays_registry_in_current_directory(self, mock_registry):
        """Test displaying registry in current directory."""
        with patch("osprey.cli.registry_cmd.display_registry_contents") as mock_display:
            with patch("builtins.input"):  # Mock the "Press ENTER" input
                mock_display.return_value = True

                handle_registry_action(project_path=None, verbose=False)

                # Should call display_registry_contents
                assert mock_display.called

    def test_changes_to_project_directory(self, tmp_path, mock_registry):
        """Test changing to project directory before displaying."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        with patch("osprey.cli.registry_cmd.display_registry_contents") as mock_display:
            with patch("builtins.input"):
                mock_display.return_value = True

                handle_registry_action(project_path=project_dir, verbose=False)

                # Should call display
                assert mock_display.called

    def test_handles_directory_change_error(self, mock_registry):
        """Test handling error when changing directory."""
        bad_path = Path("/nonexistent/directory")

        with patch("builtins.input"):
            # Should not raise exception
            handle_registry_action(project_path=bad_path, verbose=False)

    def test_restores_original_directory(self, tmp_path, mock_registry):
        """Test that original directory is restored after display."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        original_cwd = Path.cwd()

        with patch("osprey.cli.registry_cmd.display_registry_contents") as mock_display:
            with patch("builtins.input"):
                mock_display.return_value = True

                handle_registry_action(project_path=project_dir, verbose=False)

                # Should be back in original directory
                assert Path.cwd() == original_cwd

    def test_handles_display_exception(self, mock_registry):
        """Test handling exception during display."""
        with patch("osprey.cli.registry_cmd.display_registry_contents") as mock_display:
            with patch("builtins.input"):
                mock_display.side_effect = Exception("Test error")

                # Should not raise exception
                handle_registry_action(project_path=None, verbose=False)

    def test_verbose_mode_passed_to_display(self, mock_registry):
        """Test that verbose flag is passed to display function."""
        with patch("osprey.cli.registry_cmd.display_registry_contents") as mock_display:
            with patch("builtins.input"):
                mock_display.return_value = True

                handle_registry_action(project_path=None, verbose=True)

                # Should call with verbose=True
                call_kwargs = mock_display.call_args[1]
                assert call_kwargs["verbose"] is True
