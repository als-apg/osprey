"""Tests for registry CLI command display functionality.

The registry verb prints through the CLI renderer: prose goes out as report /
note / section lines, and its panel and tables are built by the shared
factories in ``osprey.cli.styles``. The tests below pin both halves -- the words
an operator reads, and the fact that the surfaces come from the factories rather
than from borders spelled at the call site.
"""

from unittest.mock import MagicMock, patch

import pytest
from rich import box
from rich.panel import Panel
from rich.table import Table

from osprey.cli.registry_cmd import (
    _display_providers_table,
    _display_services_table,
    display_registry_contents,
)
from osprey.cli.styles import Styles


@pytest.fixture
def printed():
    """Every renderable the module handed to ``output.table``, in order.

    The renderable is the honest surface for a look assertion: the border style
    is a theme token resolved at render time, so reading it off the object is
    what proves which factory built it.
    """
    recorded: list[object] = []
    with patch("osprey.cli.output.table", side_effect=recorded.append):
        yield recorded


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

    def test_displays_registry_with_initialized_registry(self, mock_registry, capsys, printed):
        """Test displaying registry contents when registry is already initialized."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=False)

                # Should succeed
                assert result is True
                # Should get stats
                assert mock_registry.get_stats.called
                # Already initialized -- no progress notice
                assert "Initializing registry" not in capsys.readouterr().out

    def test_initializes_registry_if_not_initialized(self, mock_registry, capsys, printed):
        """Test that uninitialized registry gets initialized."""
        mock_registry.get_stats.return_value["initialized"] = False

        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=False)

                # Should initialize registry
                assert mock_registry.initialize.called
                assert result is True
                # Cold run announces the load, which is otherwise silent
                assert "Initializing registry" in capsys.readouterr().out

    def test_summary_prints_the_service_count_as_a_section(self, mock_registry, capsys, printed):
        """The counts are facts about the registry, so they print as a section."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                display_registry_contents(verbose=False)

        out = capsys.readouterr().out
        assert "Registry Summary" in out
        assert "Services" in out
        assert "1" in out

    def test_handles_exceptions_gracefully(self, capsys):
        """Test that exceptions are handled gracefully."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.side_effect = Exception("Test error")

                result = display_registry_contents(verbose=False)

                # Should return False on error
                assert result is False

    def test_failure_reaches_stderr_with_its_cause(self, capsys):
        """Trouble is stderr's, whatever the caller does with stdout."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.side_effect = Exception("Test error")

                display_registry_contents(verbose=False)

        captured = capsys.readouterr()
        assert "✗ Could not display the registry" in captured.err
        assert "Test error" in captured.err
        assert "Could not display the registry" not in captured.out

    def test_header_panel_comes_from_the_shared_factory(self, mock_registry, printed):
        """The panel's frame is the factory's, not the call site's.

        Re-pinned: this panel used to ask ``ThemeConfig.get_border_style()`` for
        a ``border`` frame. The factory gives every panel and every data table
        the one dim border, which is the convergence the shared look is for.
        """
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                display_registry_contents(verbose=False)

        panels = [r for r in printed if isinstance(r, Panel)]
        assert len(panels) == 1
        assert panels[0].border_style == Styles.BORDER_DIM
        assert panels[0].box is box.ROUNDED
        assert panels[0].expand is False

    def test_verbose_mode_shows_additional_info(self, mock_registry, printed):
        """Test that verbose mode displays additional information."""
        with patch("osprey.cli.registry_cmd.get_registry") as mock_get_registry:
            with patch("osprey.utils.log_filter.quiet_logger"):
                mock_get_registry.return_value = mock_registry

                result = display_registry_contents(verbose=True)

                # Should succeed
                assert result is True


class TestDisplayServicesTable:
    """Test _display_services_table function."""

    def test_displays_services(self, mock_registry, printed):
        """Test displaying services table."""
        # Should not raise exception
        _display_services_table(mock_registry, verbose=False)

        # Should get stats for service names
        assert mock_registry.get_stats.called

    def test_table_comes_from_the_shared_factory(self, mock_registry, printed):
        """Re-pinned: the table used to spell ``dim`` for its own border."""
        _display_services_table(mock_registry, verbose=False)

        tables = [r for r in printed if isinstance(r, Table)]
        assert len(tables) == 1
        assert tables[0].border_style == Styles.BORDER_DIM
        assert tables[0].header_style == Styles.HEADER
        assert tables[0].box is box.HEAVY_HEAD

    def test_heading_and_rows_reach_stdout(self, mock_registry, capsys):
        """The heading is the verb's own output, so it survives any reporter."""
        _display_services_table(mock_registry, verbose=False)

        out = capsys.readouterr().out
        assert "Services" in out
        assert "test_service" in out


class TestDisplayProvidersTable:
    """Test _display_providers_table function."""

    def test_displays_providers(self, mock_registry, printed):
        """Test displaying providers table."""
        providers = ["test_provider", "another_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=False)

        # Should get provider classes
        assert mock_registry.get_provider.called

    def test_displays_providers_verbose(self, mock_registry, printed):
        """Test displaying providers table in verbose mode."""
        providers = ["test_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=True)

        tables = [r for r in printed if isinstance(r, Table)]
        assert len(tables) == 1
        assert [column.header for column in tables[0].columns] == [
            "Name",
            "Available",
            "Description",
        ]

    def test_handles_missing_provider(self, mock_registry, printed):
        """Test handling of provider that doesn't exist."""
        mock_registry.get_provider.return_value = None
        providers = ["nonexistent_provider"]

        # Should not raise exception
        _display_providers_table(mock_registry, providers, verbose=False)

    def test_table_comes_from_the_shared_factory(self, mock_registry, printed):
        """Re-pinned, for the providers table's own border."""
        _display_providers_table(mock_registry, ["test_provider"], verbose=False)

        tables = [r for r in printed if isinstance(r, Table)]
        assert tables[0].border_style == Styles.BORDER_DIM
        assert tables[0].header_style == Styles.HEADER
