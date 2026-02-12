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

    def test_ingest_help_shows_url_support(self, runner):
        """ingest --help mentions file path or URL."""
        result = runner.invoke(ariel_group, ["ingest", "--help"])
        assert "file path or URL" in result.output

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

    def test_ingest_tracks_runs(self, runner, tmp_path, monkeypatch):
        """ingest command calls start_ingestion_run and complete_ingestion_run."""
        from unittest.mock import AsyncMock, MagicMock, patch

        source_file = tmp_path / "entries.jsonl"
        source_file.write_text('{"entry_id": "1", "raw_text": "hello"}\n')

        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        mock_repo = MagicMock()
        mock_repo.start_ingestion_run = AsyncMock(return_value=42)
        mock_repo.complete_ingestion_run = AsyncMock()
        mock_repo.fail_ingestion_run = AsyncMock()
        mock_repo.upsert_entry = AsyncMock()
        mock_repo.mark_enhancement_complete = AsyncMock()
        mock_repo.mark_enhancement_failed = AsyncMock()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        conn_cm = AsyncMock()
        conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        conn_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pool.connection = MagicMock(return_value=conn_cm)

        mock_service = MagicMock()
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=None)
        mock_service.repository = mock_repo
        mock_service.pool = mock_pool

        async def _fetch(*args, **kwargs):
            yield {"entry_id": "1", "raw_text": "hello"}

        mock_adapter = MagicMock()
        mock_adapter.source_system_name = "test"
        mock_adapter.fetch_entries = _fetch

        with (
            patch(
                "osprey.services.ariel_search.create_ariel_service",
                new_callable=AsyncMock,
                return_value=mock_service,
            ),
            patch(
                "osprey.services.ariel_search.ingestion.get_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
                return_value=[],
            ),
        ):
            result = runner.invoke(
                ariel_group,
                ["ingest", "-s", str(source_file), "-a", "generic_json"],
            )

        assert result.exit_code == 0, result.output
        mock_repo.start_ingestion_run.assert_called_once_with("test")
        mock_repo.complete_ingestion_run.assert_called_once_with(
            42, entries_added=1, entries_updated=0, entries_failed=0
        )

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
            mock_service.repository.start_ingestion_run = AsyncMock(return_value=1)
            mock_service.repository.fail_ingestion_run = AsyncMock()
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


class TestWatchCommand:
    """Tests for the ariel watch command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_watch_command_exists(self, runner):
        """watch subcommand is registered."""
        result = runner.invoke(ariel_group, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch a source" in result.output

    def test_watch_help_shows_options(self, runner):
        """watch --help lists all options."""
        result = runner.invoke(ariel_group, ["watch", "--help"])
        assert "--once" in result.output
        assert "--interval" in result.output
        assert "--dry-run" in result.output
        assert "--source" in result.output
        assert "--adapter" in result.output

    def test_watch_once_runs_poll(self, runner, monkeypatch):
        """watch --once invokes poll_once and shows result."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {"adapter": "generic_json", "source_url": "https://api.example.com/log"},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        mock_service = MagicMock()
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=None)
        mock_service.repository = MagicMock()

        from osprey.services.ariel_search.ingestion.scheduler import IngestionPollResult

        poll_result = IngestionPollResult(
            entries_added=3,
            entries_updated=0,
            entries_failed=0,
            duration_seconds=1.2,
            since=None,
        )

        with (
            patch(
                "osprey.services.ariel_search.create_ariel_service",
                new_callable=AsyncMock,
                return_value=mock_service,
            ),
            patch(
                "osprey.services.ariel_search.ingestion.scheduler.IngestionScheduler.poll_once",
                new_callable=AsyncMock,
                return_value=poll_result,
            ) as mock_poll,
        ):
            result = runner.invoke(ariel_group, ["watch", "--once"])

        assert result.exit_code == 0
        mock_poll.assert_called_once_with(dry_run=False)
        assert "Poll complete" in result.output
        assert "3 added" in result.output

    def test_watch_once_dry_run(self, runner, monkeypatch):
        """watch --once --dry-run shows dry-run prefix in output."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {"adapter": "generic_json", "source_url": "https://api.example.com/log"},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        mock_service = MagicMock()
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=None)
        mock_service.repository = MagicMock()

        from osprey.services.ariel_search.ingestion.scheduler import IngestionPollResult

        poll_result = IngestionPollResult(
            entries_added=5,
            entries_updated=0,
            entries_failed=0,
            duration_seconds=0.8,
            since=None,
        )

        with (
            patch(
                "osprey.services.ariel_search.create_ariel_service",
                new_callable=AsyncMock,
                return_value=mock_service,
            ),
            patch(
                "osprey.services.ariel_search.ingestion.scheduler.IngestionScheduler.poll_once",
                new_callable=AsyncMock,
                return_value=poll_result,
            ) as mock_poll,
        ):
            result = runner.invoke(ariel_group, ["watch", "--once", "--dry-run"])

        assert result.exit_code == 0
        mock_poll.assert_called_once_with(dry_run=True)
        assert "[dry-run]" in result.output
        assert "Poll complete" in result.output

    def test_watch_no_source_shows_error(self, runner, monkeypatch):
        """watch shows error when ingestion has no source_url."""
        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {"adapter": "generic_json"},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        result = runner.invoke(ariel_group, ["watch", "--once"])

        assert result.exit_code == 1
        assert "source" in result.output.lower()

    def test_watch_no_config_shows_error(self, runner, monkeypatch):
        """watch shows error when ARIEL not configured."""
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: default,
        )
        result = runner.invoke(ariel_group, ["watch", "--once"])
        assert result.exit_code == 1
        assert "not configured" in result.output.lower()

    def test_watch_adapter_choices(self, runner):
        """watch command validates adapter choices."""
        result = runner.invoke(ariel_group, ["watch", "--help"])
        assert "als_logbook" in result.output
        assert "generic_json" in result.output


class TestQuickstartCommand:
    """Tests for the ariel quickstart command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_quickstart_command_exists(self, runner):
        """quickstart subcommand is registered."""
        result = runner.invoke(ariel_group, ["quickstart", "--help"])
        assert result.exit_code == 0
        assert "Quick setup" in result.output

    def test_quickstart_has_source_option(self, runner):
        """quickstart has --source option."""
        result = runner.invoke(ariel_group, ["quickstart", "--help"])
        assert "--source" in result.output

    def test_quickstart_no_config_shows_error(self, runner, monkeypatch):
        """quickstart shows error when ARIEL not configured."""
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: default,
        )
        result = runner.invoke(ariel_group, ["quickstart"])
        assert result.exit_code == 1
        assert "not configured" in result.output.lower()

    def test_quickstart_connection_failure_shows_guidance(self, runner, monkeypatch):
        """quickstart shows 'osprey deploy up' guidance on connection failure."""
        from unittest.mock import AsyncMock, patch

        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/demo.json"},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        with patch(
            "osprey.services.ariel_search.database.connection.create_connection_pool",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ):
            result = runner.invoke(ariel_group, ["quickstart"])

        assert result.exit_code == 1
        assert "osprey deploy up" in result.output

    def test_quickstart_success_flow(self, runner, monkeypatch, tmp_path):
        """quickstart completes successfully with mocked database."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create demo data file
        demo_file = tmp_path / "demo_logbook.json"
        demo_file.write_text(
            '{"entries": [{"id": "1", "timestamp": "2024-01-01T00:00:00Z", "text": "test"}]}'
        )

        mock_config = {
            "database": {"uri": "postgresql://localhost/test"},
            "ingestion": {"adapter": "generic_json", "source_url": str(demo_file)},
            "search_modules": {"keyword": {"enabled": True}},
        }
        monkeypatch.setattr(
            "osprey.cli.ariel.get_config_value",
            lambda key, default=None: mock_config if key == "ariel" else default,
        )

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        mock_service = MagicMock()
        mock_service.__aenter__ = AsyncMock(return_value=mock_service)
        mock_service.__aexit__ = AsyncMock(return_value=None)
        mock_service.repository = MagicMock()
        mock_service.repository.upsert_entry = AsyncMock()

        with (
            patch(
                "osprey.services.ariel_search.database.connection.create_connection_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "osprey.services.ariel_search.database.migrate.run_migrations",
                new_callable=AsyncMock,
                return_value=["core_schema"],
            ),
            patch(
                "osprey.services.ariel_search.create_ariel_service",
                new_callable=AsyncMock,
                return_value=mock_service,
            ),
        ):
            result = runner.invoke(ariel_group, ["quickstart"])

        assert result.exit_code == 0
        assert "complete" in result.output.lower()
