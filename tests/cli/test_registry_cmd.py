"""Tests for registry CLI command display functionality.

This test module verifies the registry display functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.cli.registry_cmd import (
    _display_providers_table,
    _display_services_table,
    display_registry_contents,
)


def _printed_text(mock_console) -> str:
    """Join every positional argument passed to a patched ``console.print``."""
    return " ".join(str(call.args[0]) for call in mock_console.print.call_args_list if call.args)


@pytest.fixture
def mock_registry():
    """Create a mock registry with test data."""
    registry = MagicMock()

    # Mock stats
    registry.get_stats.return_value = {
        "initialized": True,
        "services": 1,
        "service_names": ["test_service"],
    }

    # Mock services
    registry.get_service.return_value = MagicMock(__class__=MagicMock(__name__="TestService"))

    # Mock providers
    registry.list_providers.return_value = ["test_provider"]
    registry.get_provider.return_value = MagicMock(description="Test AI provider")

    return registry


class TestDisplayRegistryContents:
    """Test display_registry_contents function."""

    def test_displays_registry_with_initialized_registry(self, mock_registry):
        """Test displaying registry contents when registry is already initialized."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                with patch("osprey.cli.registry_cmd.console") as mock_console:
                    mock_get_registry.return_value = mock_registry

                    result = display_registry_contents(verbose=False)

                    # Should succeed
                    assert result is True
                    # Should get stats
                    assert mock_registry.get_stats.called
                    # Already initialized -- no progress notice
                    assert "Initializing registry" not in _printed_text(mock_console)

    def test_initializes_registry_if_not_initialized(self, mock_registry):
        """Test that uninitialized registry gets initialized."""
        mock_registry.get_stats.return_value["initialized"] = False

        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                with patch("osprey.cli.registry_cmd.console") as mock_console:
                    mock_get_registry.return_value = mock_registry

                    result = display_registry_contents(verbose=False)

                    # Should initialize registry
                    assert mock_registry.initialize.called
                    assert result is True
                    # Cold run announces the load, which is otherwise silent
                    assert "Initializing registry" in _printed_text(mock_console)

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
