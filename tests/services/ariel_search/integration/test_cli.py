"""Integration tests for ARIEL CLI commands.

Tests CLI command execution with real services (TEST-M006 / INT-004).

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.integration]


class TestCLIStatusCommand:
    """Test 'osprey ariel status' command."""

    def test_status_command_with_test_database(self, database_url):
        """Status command returns formatted status when database is available."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        # Patch get_config to return test database URL
        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
            },
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["status"])

        # Should succeed
        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
        assert "ARIEL Status:" in result.output

    def test_status_command_json_output(self, database_url):
        """Status command with --json outputs valid JSON."""
        import json

        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["status", "--json"])

        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "status" in data

    def test_status_command_unconfigured(self):
        """Status command handles missing configuration gracefully."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        with patch("osprey.cli.ariel.get_config_value", return_value={}):
            result = runner.invoke(ariel_group, ["status"])

        assert "not configured" in result.output or "error" in result.output.lower()


class TestCLIMigrateCommand:
    """Test 'osprey ariel migrate' command."""

    def test_migrate_command_runs_migrations(self, database_url):
        """Migrate command applies database migrations."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": {"enabled": True},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["migrate"])

        assert result.exit_code == 0
        assert "Migrations complete" in result.output or "Running migrations" in result.output


class TestCLIIngestCommand:
    """Test 'osprey ariel ingest' command."""

    @pytest.fixture
    def sample_entries_path(self) -> Path:
        """Path to sample ALS entries fixture file."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "fixtures"
            / "ariel"
            / "sample_als_entries.jsonl"
        )

    def test_ingest_command_dry_run(self, database_url, sample_entries_path):
        """Ingest command with --dry-run parses without storing."""
        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {"database": {"uri": database_url}}

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(
                ariel_group,
                [
                    "ingest",
                    "--source",
                    str(sample_entries_path),
                    "--adapter",
                    "als_logbook",
                    "--dry-run",
                    "--limit",
                    "5",
                ],
            )

        assert result.exit_code == 0, f"Exit: {result.exit_code}, Output: {result.output}"
        assert "Dry run complete" in result.output
        assert "entries would be ingested" in result.output

    def test_ingest_command_stores_entries(self, database_url, sample_entries_path, migrated_pool):
        """Ingest command stores entries in database."""
        if not sample_entries_path.exists():
            pytest.skip(f"Fixture file not found: {sample_entries_path}")

        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {"keyword": {"enabled": True}},
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(
                ariel_group,
                [
                    "ingest",
                    "--source",
                    str(sample_entries_path),
                    "--adapter",
                    "als_logbook",
                    "--limit",
                    "3",
                ],
            )

        assert result.exit_code == 0, f"Exit: {result.exit_code}, Output: {result.output}"
        assert "entries stored" in result.output

    def test_ingest_command_invalid_source(self, database_url):
        """Ingest command fails gracefully with invalid source path."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {"database": {"uri": database_url}}

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(
                ariel_group,
                [
                    "ingest",
                    "--source",
                    "/nonexistent/path/file.jsonl",
                    "--adapter",
                    "als_logbook",
                ],
            )

        # Should fail because file doesn't exist
        assert result.exit_code != 0


class TestCLIModelsCommand:
    """Test 'osprey ariel models' command."""

    def test_models_command_lists_tables(self, database_url, migrated_pool):
        """Models command lists embedding tables."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["models"])

        # Should succeed
        assert result.exit_code == 0


class TestCLISearchCommand:
    """Test 'osprey ariel search' command."""

    def test_search_command_with_query(self, database_url):
        """Search command executes query."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": {"enabled": True},
            },
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["search", "beam loss"])

        # Should succeed (even if no results)
        assert result.exit_code == 0
        assert "Query:" in result.output

    def test_search_command_json_output(self, database_url):
        """Search command with --json outputs valid JSON."""
        import json

        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {"keyword": {"enabled": True}},
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["search", "test query", "--json"])

        assert result.exit_code == 0, f"Exit: {result.exit_code}, Output: {result.output}"
        # Extract JSON from output (may have log messages before it)
        output = result.output
        # Find the JSON object in output (skip any log lines)
        json_start = output.find("{")
        if json_start >= 0:
            output = output[json_start:]
        data = json.loads(output)
        assert "query" in data

    def test_search_command_with_mode(self, database_url):
        """Search command accepts --mode parameter."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {"keyword": {"enabled": True}},
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["search", "beam", "--mode", "keyword"])

        assert result.exit_code == 0


class TestCLIEnhanceCommand:
    """Test 'osprey ariel enhance' command."""

    def test_enhance_command_no_modules(self, database_url):
        """Enhance command handles no modules enabled."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {
            "database": {"uri": database_url},
            "enhancement_modules": {
                "text_embedding": {"enabled": False},
            },
        }

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(ariel_group, ["enhance"])

        assert result.exit_code == 0
        assert "No enhancement modules" in result.output


class TestCLIReembedCommand:
    """Test 'osprey ariel reembed' command."""

    def test_reembed_command_dry_run(self, database_url):
        """Reembed command with --dry-run shows plan without executing."""
        from osprey.cli.ariel import ariel_group

        runner = CliRunner()

        mock_config = {"database": {"uri": database_url}}

        with patch("osprey.cli.ariel.get_config_value", return_value=mock_config):
            result = runner.invoke(
                ariel_group,
                [
                    "reembed",
                    "--model",
                    "nomic-embed-text",
                    "--dimension",
                    "768",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output


class TestCLIIngestCleanup:
    """Clean up CLI test data."""

    async def test_cleanup(self, migrated_pool):
        """Clean up entries created during CLI tests."""
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE source_system = 'ALS eLog'
                AND entry_id ~ '^[0-9]+$'
            """)
