"""Tests for registry CLI command display functionality.

This test module verifies the registry display functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey.cli.registry_cmd import (
    _display_capabilities_table,
    _display_context_classes_table,
    _display_data_sources_table,
    _display_nodes_table,
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
        "capabilities": 2,
        "nodes": 3,
        "context_classes": 1,
        "data_sources": 2,
        "services": 1,
        "capability_names": ["test_capability", "another_capability"],
        "node_names": ["test_node", "infrastructure_node"],
        "context_types": ["test_context"],
        "data_source_names": ["test_ds"],
        "service_names": ["test_service"],
    }

    # Mock capabilities
    cap1 = MagicMock()
    cap1.name = "test_capability"
    cap1.provides = ["test_context"]
    cap1.requires = []
    cap1.description = "Test capability"

    cap2 = MagicMock()
    cap2.name = "another_capability"
    cap2.provides = ["another_context"]
    cap2.requires = ["test_context"]
    cap2.description = "Another capability"

    registry.get_all_capabilities.return_value = [cap1, cap2]

    # Mock nodes
    registry.get_all_nodes.return_value = {
        "test_node": MagicMock(__class__=MagicMock(__name__="TestNode")),
        "infrastructure_node": MagicMock(__class__=MagicMock(__name__="InfraNode")),
    }

    # Mock context classes
    registry.get_all_context_classes.return_value = {"test_context": type("TestContext", (), {})}

    # Mock data sources
    registry.get_data_source.return_value = MagicMock(
        __class__=MagicMock(__name__="TestDataSource")
    )

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


class TestDisplayCapabilitiesTable:
    """Test _display_capabilities_table function."""

    def test_displays_capabilities(self, mock_registry):
        """Test displaying capabilities table."""
        # Should not raise exception
        _display_capabilities_table(mock_registry, verbose=False)

        # Should call get_all_capabilities
        assert mock_registry.get_all_capabilities.called

    def test_displays_capabilities_verbose(self, mock_registry):
        """Test displaying capabilities table in verbose mode."""
        # Should not raise exception
        _display_capabilities_table(mock_registry, verbose=True)

        assert mock_registry.get_all_capabilities.called

    def test_handles_tuple_provides_and_requires(self, mock_registry):
        """Test handling of tuple values in provides/requires."""
        # Create capability with tuple values
        cap = MagicMock()
        cap.name = "test"
        cap.provides = [("context", "type")]
        cap.requires = [("required", "context")]
        cap.description = "Test"

        mock_registry.get_all_capabilities.return_value = [cap]

        # Should not raise exception
        _display_capabilities_table(mock_registry, verbose=True)

    def test_handles_empty_provides_and_requires(self, mock_registry):
        """Test handling of empty provides/requires."""
        cap = MagicMock()
        cap.name = "test"
        cap.provides = []
        cap.requires = []
        cap.description = "Test"

        mock_registry.get_all_capabilities.return_value = [cap]

        # Should not raise exception
        _display_capabilities_table(mock_registry, verbose=False)


class TestDisplayNodesTable:
    """Test _display_nodes_table function."""

    def test_displays_infrastructure_nodes(self, mock_registry):
        """Test displaying infrastructure nodes table."""
        # Should not raise exception
        _display_nodes_table(mock_registry, verbose=False)

        # Should get capabilities to filter them out
        assert mock_registry.get_all_capabilities.called
        # Should get nodes
        assert mock_registry.get_all_nodes.called

    def test_filters_out_capability_nodes(self, mock_registry):
        """Test that capability nodes are filtered from infrastructure nodes."""
        # Add a node with same name as capability
        nodes = mock_registry.get_all_nodes.return_value
        nodes["test_capability"] = MagicMock()

        # Should not raise exception
        _display_nodes_table(mock_registry, verbose=False)


class TestDisplayContextClassesTable:
    """Test _display_context_classes_table function."""

    def test_displays_context_classes(self, mock_registry):
        """Test displaying context classes table."""
        # Should not raise exception
        _display_context_classes_table(mock_registry, verbose=False)

        # Should get context classes
        assert mock_registry.get_all_context_classes.called


class TestDisplayDataSourcesTable:
    """Test _display_data_sources_table function."""

    def test_displays_data_sources(self, mock_registry):
        """Test displaying data sources table."""
        # Should not raise exception
        _display_data_sources_table(mock_registry, verbose=False)

        # Should get stats for data source names
        assert mock_registry.get_stats.called


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
