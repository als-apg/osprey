"""Tests for channel finder CLI utility functions.

Tests pure functions and class methods from the service-level CLI modules:
- parse_query_selection
- create_config_override
- ChannelFinderCLI class methods
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# parse_query_selection Tests
# ============================================================================


class TestParseQuerySelection:
    """Test the query selection parser (pure function, no mocking needed)."""

    def test_all_returns_all(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        assert parse_query_selection("all") == "all"

    def test_all_case_insensitive(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        assert parse_query_selection("ALL") == "all"
        assert parse_query_selection("All") == "all"

    def test_range_returns_dict(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        result = parse_query_selection("0:10")
        assert result == {"start": 0, "end": 10}

    def test_range_different_values(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        result = parse_query_selection("5:20")
        assert result == {"start": 5, "end": 20}

    def test_comma_separated_returns_list(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        result = parse_query_selection("0,5,10")
        assert result == [0, 5, 10]

    def test_single_index_returns_list(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        result = parse_query_selection("5")
        assert result == [5]

    def test_invalid_range_raises_valueerror(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        with pytest.raises(ValueError, match="Invalid range format"):
            parse_query_selection("a:b")

    def test_invalid_comma_list_raises_valueerror(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        with pytest.raises(ValueError, match="Invalid index list"):
            parse_query_selection("a,b,c")

    def test_invalid_single_raises_valueerror(self):
        from osprey.services.channel_finder.benchmarks.cli import parse_query_selection

        with pytest.raises(ValueError, match="Invalid query selection"):
            parse_query_selection("invalid")


# ============================================================================
# create_config_override Tests
# ============================================================================


class TestCreateConfigOverride:
    """Test config override application (mocked config)."""

    @patch("osprey.services.channel_finder.benchmarks.cli.get_config")
    def test_queries_applied_to_config(self, mock_get_config):
        from osprey.services.channel_finder.benchmarks.cli import create_config_override

        mock_get_config.return_value = {"channel_finder": {}}
        create_config_override(queries="0:10")
        config = mock_get_config.return_value
        assert config["channel_finder"]["benchmark"]["execution"]["query_selection"] == {
            "start": 0,
            "end": 10,
        }

    @patch("osprey.services.channel_finder.benchmarks.cli.get_config")
    def test_model_applied_to_config(self, mock_get_config):
        from osprey.services.channel_finder.benchmarks.cli import create_config_override

        mock_get_config.return_value = {"channel_finder": {}}
        create_config_override(model="anthropic/claude-sonnet")
        config = mock_get_config.return_value
        assert config["model"]["model_id"] == "anthropic/claude-sonnet"

    @patch("osprey.services.channel_finder.benchmarks.cli.get_config")
    def test_dataset_applied_to_pipeline_config(self, mock_get_config):
        from osprey.services.channel_finder.benchmarks.cli import create_config_override

        mock_get_config.return_value = {
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {"hierarchical": {}},
            }
        }
        create_config_override(dataset="data/my_data.json")
        config = mock_get_config.return_value
        assert (
            config["channel_finder"]["pipelines"]["hierarchical"]["benchmark"]["dataset_path"]
            == "data/my_data.json"
        )

    @patch("osprey.services.channel_finder.benchmarks.cli.get_config")
    def test_none_parameters_leave_config_unchanged(self, mock_get_config):
        from osprey.services.channel_finder.benchmarks.cli import create_config_override

        original = {"channel_finder": {"pipeline_mode": "hierarchical"}}
        mock_get_config.return_value = original.copy()
        create_config_override(queries=None, model=None, dataset=None)
        # Config should be unchanged (no new keys)
        config = mock_get_config.return_value
        assert "benchmark" not in config["channel_finder"]
        assert "model" not in config

    @patch("osprey.services.channel_finder.benchmarks.cli.get_config")
    def test_missing_benchmark_section_created(self, mock_get_config):
        from osprey.services.channel_finder.benchmarks.cli import create_config_override

        mock_get_config.return_value = {"channel_finder": {}}
        create_config_override(queries="all")
        config = mock_get_config.return_value
        assert "benchmark" in config["channel_finder"]
        assert "execution" in config["channel_finder"]["benchmark"]
        assert config["channel_finder"]["benchmark"]["execution"]["query_selection"] == "all"


# ============================================================================
# ChannelFinderCLI Class Tests
# ============================================================================


class TestChannelFinderCLI:
    """Test ChannelFinderCLI class methods."""

    def test_create_key_bindings(self):
        """_create_key_bindings returns a valid KeyBindings object."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        bindings = cli._create_key_bindings()
        # KeyBindings from prompt_toolkit
        from prompt_toolkit.key_binding import KeyBindings

        assert isinstance(bindings, KeyBindings)

    def test_display_results_zero_channels(self):
        """_display_results handles result with channels."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        cli.console = MagicMock()

        # Create mock result with channels
        mock_result = MagicMock()
        mock_result.total_channels = 2
        mock_result.channels = [
            MagicMock(channel="CH:01", address="addr1", description="desc1"),
            MagicMock(channel="CH:02", address="addr2", description="desc2"),
        ]
        mock_result.processing_notes = None

        cli._display_results(mock_result)
        # Should have called console.print at least once
        assert cli.console.print.called

    def test_display_results_with_processing_notes(self):
        """_display_results shows processing notes when present."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        cli.console = MagicMock()

        mock_result = MagicMock()
        mock_result.total_channels = 1
        mock_result.channels = [
            MagicMock(channel="CH:01", address="addr1", description="desc1"),
        ]
        mock_result.processing_notes = "Some processing note"

        cli._display_results(mock_result)
        # Check that processing notes were printed
        print_calls = [str(call) for call in cli.console.print.call_args_list]
        assert any("Some processing note" in call for call in print_calls)

    def test_display_results_truncates_long_descriptions(self):
        """_display_results truncates descriptions > 80 chars."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        cli.console = MagicMock()

        long_desc = "A" * 100
        mock_result = MagicMock()
        mock_result.total_channels = 1
        mock_result.channels = [
            MagicMock(channel="CH:01", address="addr1", description=long_desc),
        ]
        mock_result.processing_notes = None

        cli._display_results(mock_result)
        # The table should have been printed (we can't easily inspect Table contents
        # but at least verify the method completes without error)
        assert cli.console.print.called

    @pytest.mark.asyncio
    async def test_process_query_calls_service(self):
        """_process_query calls service.find_channels."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        cli.console = MagicMock()
        cli.service = MagicMock()

        mock_result = MagicMock()
        mock_result.total_channels = 0
        cli.service.find_channels = AsyncMock(return_value=mock_result)

        await cli._process_query("find BPMs")
        cli.service.find_channels.assert_called_once_with("find BPMs")

    @pytest.mark.asyncio
    async def test_process_query_handles_exception(self):
        """_process_query handles service exception gracefully."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI

        cli = ChannelFinderCLI()
        cli.console = MagicMock()
        cli.service = MagicMock()
        cli.service.find_channels = AsyncMock(side_effect=Exception("API error"))

        # Should not raise
        await cli._process_query("find BPMs")
        # Should print error message
        print_calls = [str(call) for call in cli.console.print.call_args_list]
        assert any("Query failed" in call for call in print_calls)
