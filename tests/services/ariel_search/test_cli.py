"""Tests for ARIEL CLI commands.

Tests for CLI command registration and basic validation.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import ariel_group


class TestARIELCLIGroup:
    """Tests for ARIEL CLI command group."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_ariel_group_exists(self):
        """ariel command group exists."""
        assert ariel_group is not None
        assert ariel_group.name == "ariel"

    def test_ariel_help(self, runner):
        """ariel --help shows available commands."""
        result = runner.invoke(ariel_group, ["--help"])
        assert result.exit_code == 0
        assert "ARIEL search service commands" in result.output

    def test_status_command_exists(self, runner):
        """status subcommand exists."""
        result = runner.invoke(ariel_group, ["status", "--help"])
        assert result.exit_code == 0
        assert "ARIEL service status" in result.output

    def test_migrate_command_exists(self, runner):
        """migrate subcommand exists."""
        result = runner.invoke(ariel_group, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "database migrations" in result.output

    def test_ingest_command_exists(self, runner):
        """ingest subcommand exists."""
        result = runner.invoke(ariel_group, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "source file" in result.output.lower()

    def test_enhance_command_exists(self, runner):
        """enhance subcommand exists."""
        result = runner.invoke(ariel_group, ["enhance", "--help"])
        assert result.exit_code == 0
        assert "enhancement modules" in result.output.lower()

    def test_models_command_exists(self, runner):
        """models subcommand exists."""
        result = runner.invoke(ariel_group, ["models", "--help"])
        assert result.exit_code == 0
        assert "embedding" in result.output.lower()

    def test_search_command_exists(self, runner):
        """search subcommand exists."""
        result = runner.invoke(ariel_group, ["search", "--help"])
        assert result.exit_code == 0
        assert "Search the logbook" in result.output

    def test_ingest_requires_source(self, runner):
        """ingest command requires --source option."""
        result = runner.invoke(ariel_group, ["ingest"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_ingest_adapter_choices(self, runner):
        """ingest command validates adapter choices."""
        result = runner.invoke(ariel_group, ["ingest", "--help"])
        assert "als_logbook" in result.output
        assert "jlab_logbook" in result.output
        assert "ornl_logbook" in result.output
        assert "generic_json" in result.output

    def test_enhance_module_choices(self, runner):
        """enhance command validates module choices."""
        result = runner.invoke(ariel_group, ["enhance", "--help"])
        assert "text_embedding" in result.output
        assert "semantic_processor" in result.output

    def test_search_mode_choices(self, runner):
        """search command validates mode choices."""
        result = runner.invoke(ariel_group, ["search", "--help"])
        assert "keyword" in result.output
        assert "semantic" in result.output
        assert "rag" in result.output
        assert "auto" in result.output

    def test_reembed_command_exists(self, runner):
        """reembed subcommand exists."""
        result = runner.invoke(ariel_group, ["reembed", "--help"])
        assert result.exit_code == 0
        assert "Re-embed entries" in result.output

    def test_reembed_requires_model(self, runner):
        """reembed command requires --model option."""
        result = runner.invoke(ariel_group, ["reembed", "--dimension", "768"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--model" in result.output

    def test_reembed_requires_dimension(self, runner):
        """reembed command requires --dimension option."""
        result = runner.invoke(ariel_group, ["reembed", "--model", "nomic-embed-text"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--dimension" in result.output

    def test_reembed_has_dry_run_option(self, runner):
        """reembed command has --dry-run option."""
        result = runner.invoke(ariel_group, ["reembed", "--help"])
        assert "--dry-run" in result.output

    def test_reembed_has_force_option(self, runner):
        """reembed command has --force option."""
        result = runner.invoke(ariel_group, ["reembed", "--help"])
        assert "--force" in result.output

    def test_reembed_has_batch_size_option(self, runner):
        """reembed command has --batch-size option."""
        result = runner.invoke(ariel_group, ["reembed", "--help"])
        assert "--batch-size" in result.output

    def test_ingest_missing_tables_shows_user_friendly_error(self, runner, tmp_path, monkeypatch):
        """ingest shows helpful error when database tables don't exist."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from osprey.services.ariel_search.exceptions import DatabaseQueryError

        # Create a dummy source file
        source_file = tmp_path / "test.jsonl"
        source_file.write_text('{"entry_id": "1", "raw_text": "test"}\n')

        # Mock config to return valid ARIEL config
        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        # Mock create_ariel_service to raise DatabaseQueryError with missing table message
        error = DatabaseQueryError(
            'Failed to upsert entry: relation "enhanced_entries" does not exist'
        )

        async def mock_create_service(*args, **kwargs):
            mock_service = MagicMock()
            mock_service.__aenter__ = AsyncMock(return_value=mock_service)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            mock_service.repository = MagicMock()
            mock_service.repository.upsert_entry = AsyncMock(side_effect=error)
            mock_service.pool = MagicMock()
            mock_service.pool.connection = MagicMock(return_value=AsyncMock())
            return mock_service

        with patch(
            "osprey.services.ariel_search.create_ariel_service",
            side_effect=mock_create_service,
        ):
            with patch("osprey.services.ariel_search.ingestion.get_adapter") as mock_adapter:
                # Mock adapter to return one entry
                async def mock_fetch(*args, **kwargs):
                    yield {"entry_id": "1", "raw_text": "test"}

                adapter_instance = MagicMock()
                adapter_instance.source_system_name = "test"
                adapter_instance.fetch_entries = mock_fetch
                mock_adapter.return_value = adapter_instance

                result = runner.invoke(
                    ariel_group,
                    ["ingest", "-s", str(source_file), "-a", "generic_json"],
                )

        assert result.exit_code == 1
        assert "ARIEL database is not initialized" in result.output
        assert "osprey ariel migrate" in result.output
