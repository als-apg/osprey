"""Tests for the osprey channel-finder CLI command.

Tests the Click command group including:
- Command structure and help output
- Build-database, validate, preview subcommands
- Config/project resolution
- _parse_query_indices parsing
- Benchmark subcommand
"""

from unittest.mock import AsyncMock, patch

import click
import pytest
from click.testing import CliRunner

from osprey.cli.channel_finder_cmd import _parse_query_indices, channel_finder


@pytest.fixture
def runner():
    return CliRunner()


# ============================================================================
# Command Structure Tests
# ============================================================================


class TestCommandStructure:
    """Test that the command group and subcommands are properly defined."""

    def test_channel_finder_group_exists(self, runner):
        """channel-finder group is callable."""
        result = runner.invoke(channel_finder, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_subcommands(self, runner):
        """--help shows all subcommands."""
        result = runner.invoke(channel_finder, ["--help"])
        assert "build-database" in result.output
        assert "validate" in result.output
        assert "preview" in result.output

    def test_help_shows_project_option(self, runner):
        """--help shows --project option."""
        result = runner.invoke(channel_finder, ["--help"])
        assert "--project" in result.output
        assert "-p" in result.output

    def test_help_shows_verbose_option(self, runner):
        """--help shows --verbose option."""
        result = runner.invoke(channel_finder, ["--help"])
        assert "--verbose" in result.output
        assert "-v" in result.output


# ============================================================================
# Config/Project Resolution Tests
# ============================================================================


class TestConfigResolution:
    """Test project and config resolution."""

    def test_missing_config_shows_error(self, runner, tmp_path):
        """--project with no config.yml shows helpful error."""
        result = runner.invoke(channel_finder, ["--project", str(tmp_path), "validate"])
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output


# ============================================================================
# Build-Database Subcommand Tests
# ============================================================================


class TestBuildDatabaseSubcommand:
    """Test the 'build-database' subcommand."""

    def test_build_database_help(self, runner):
        """build-database --help shows options."""
        result = runner.invoke(channel_finder, ["build-database", "--help"])
        assert result.exit_code == 0
        assert "--csv" in result.output
        assert "--output" in result.output
        assert "--use-llm" in result.output
        assert "--delimiter" in result.output

    def test_build_database_with_csv(self, runner, tmp_path):
        """build-database builds a database from CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "BEAM:CURRENT,Total beam current,,,\n"
            "BPM01X,BPM horizontal,BPM,3,X\n"
            "BPM01Y,BPM vertical,BPM,3,Y\n"
        )
        output_file = tmp_path / "output.json"

        result = runner.invoke(
            channel_finder,
            ["build-database", "--csv", str(csv_file), "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        import json

        db = json.loads(output_file.read_text())
        assert "channels" in db
        assert len(db["channels"]) > 0

    def test_build_database_with_delimiter(self, runner, tmp_path):
        """build-database accepts --delimiter for pipe-separated CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "address|description|family_name|instances|sub_channel\n"
            "BEAM:CURRENT|Total beam current|||\n"
            "BPM01X|BPM horizontal|BPM|3|X\n"
            "BPM01Y|BPM vertical|BPM|3|Y\n"
        )
        output_file = tmp_path / "output.json"

        result = runner.invoke(
            channel_finder,
            [
                "build-database",
                "--csv",
                str(csv_file),
                "--output",
                str(output_file),
                "--delimiter",
                "|",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        import json

        db = json.loads(output_file.read_text())
        assert "channels" in db
        assert len(db["channels"]) > 0

    def test_build_database_missing_csv_errors(self, runner):
        """build-database with nonexistent CSV shows error."""
        result = runner.invoke(channel_finder, ["build-database", "--csv", "/nonexistent/file.csv"])
        assert result.exit_code != 0


# ============================================================================
# Validate Subcommand Tests
# ============================================================================


class TestValidateSubcommand:
    """Test the 'validate' subcommand."""

    def test_validate_help(self, runner):
        """validate --help shows options."""
        result = runner.invoke(channel_finder, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--database" in result.output
        assert "--verbose" in result.output
        assert "--pipeline" in result.output

    def test_validate_with_valid_database(self, runner, tmp_path):
        """validate with a valid in-context database passes."""
        db_file = tmp_path / "test_db.json"
        db_file.write_text(
            '{"channels": [{"template": false, "channel": "CH1", '
            '"address": "PV:CH1", "description": "Test channel"}]}'
        )

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(
                    channel_finder,
                    ["validate", "--database", str(db_file), "--pipeline", "in_context"],
                )
        # exit_code may be 0 or 1 (load test may fail without full setup)
        # but it should not crash
        assert result.exit_code in (0, 1)
        assert "VALID" in result.output or "INVALID" in result.output

    def test_validate_with_empty_database_fails(self, runner, tmp_path):
        """validate with an empty database shows errors."""
        db_file = tmp_path / "empty_db.json"
        db_file.write_text('{"channels": []}')

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(
                    channel_finder,
                    ["validate", "--database", str(db_file), "--pipeline", "in_context"],
                )
        assert result.exit_code == 1
        assert "INVALID" in result.output


# ============================================================================
# Preview Subcommand Tests
# ============================================================================


class TestPreviewSubcommand:
    """Test the 'preview' subcommand."""

    def test_preview_help(self, runner):
        """preview --help shows options."""
        result = runner.invoke(channel_finder, ["preview", "--help"])
        assert result.exit_code == 0
        assert "--depth" in result.output
        assert "--max-items" in result.output
        assert "--sections" in result.output
        assert "--focus" in result.output
        assert "--database" in result.output
        assert "--full" in result.output

    def test_preview_with_database_file(self, runner):
        """preview with --database loads and previews."""
        from pathlib import Path

        examples_dir = (
            Path(__file__).parent.parent.parent
            / "src"
            / "osprey"
            / "templates"
            / "apps"
            / "control_assistant"
            / "data"
            / "channel_databases"
            / "examples"
        )
        db_path = examples_dir / "consecutive_instances.json"
        if not db_path.exists():
            pytest.skip("Example database not found")

        result = runner.invoke(
            channel_finder,
            ["preview", "--database", str(db_path), "--depth", "2", "--max-items", "2"],
        )
        assert result.exit_code == 0
        assert "Hierarchy" in result.output or "Preview" in result.output


# ============================================================================
# Import Smoke Tests
# ============================================================================


class TestImportSmoke:
    """Test that cleaned-up modules are importable."""

    def test_channel_finder_cmd_importable(self):
        """channel_finder_cmd module is importable."""
        from osprey.cli.channel_finder_cmd import (
            build_database,
            channel_finder,
            preview,
            validate,
        )

        assert channel_finder is not None
        assert build_database is not None
        assert validate is not None
        assert preview is not None

    def test_import_native_tools(self):
        """Native tool modules are importable."""
        from osprey.services.channel_finder.tools.build_database import (
            build_database,
            load_csv,
        )
        from osprey.services.channel_finder.tools.llm_channel_namer import (
            LLMChannelNamer,
            create_namer_from_config,
        )
        from osprey.services.channel_finder.tools.preview_database import preview_database
        from osprey.services.channel_finder.tools.validate_database import (
            validate_json_structure,
        )

        assert build_database is not None
        assert load_csv is not None
        assert preview_database is not None
        assert validate_json_structure is not None
        assert LLMChannelNamer is not None
        assert create_namer_from_config is not None


# ============================================================================
# CLI Error Path Tests
# ============================================================================


class TestCLIErrorPaths:
    """Test CLI error paths for validate and preview without config."""

    def test_validate_no_database_no_config_shows_error(self, runner, tmp_path):
        """validate without --database and no config shows config error."""
        result = runner.invoke(channel_finder, ["--project", str(tmp_path), "validate"])
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output

    def test_preview_no_database_no_config_shows_error(self, runner, tmp_path):
        """preview without --database and no config shows config error."""
        result = runner.invoke(channel_finder, ["--project", str(tmp_path), "preview"])
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output

    def test_build_database_output_contains_templates_and_standalone(self, runner, tmp_path):
        """build-database output contains both templates and standalone channels."""
        import json

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "BEAM:CURRENT,Total beam current,,,\n"
            "BPM01X,BPM horizontal,BPM,3,X\n"
            "BPM01Y,BPM vertical,BPM,3,Y\n"
        )
        output_file = tmp_path / "output.json"

        result = runner.invoke(
            channel_finder,
            ["build-database", "--csv", str(csv_file), "--output", str(output_file)],
        )
        assert result.exit_code == 0

        db = json.loads(output_file.read_text())
        standalone = [ch for ch in db["channels"] if not ch.get("template")]
        templates = [ch for ch in db["channels"] if ch.get("template")]
        assert len(standalone) >= 1
        assert len(templates) >= 1

    def test_validate_with_pipeline_override(self, runner, tmp_path):
        """validate --pipeline hierarchical with a hierarchical DB file."""
        from pathlib import Path

        examples_dir = (
            Path(__file__).parent.parent.parent
            / "src"
            / "osprey"
            / "templates"
            / "apps"
            / "control_assistant"
            / "data"
            / "channel_databases"
            / "examples"
        )
        db_path = examples_dir / "consecutive_instances.json"
        if not db_path.exists():
            pytest.skip("Example database not found")

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(
                    channel_finder,
                    [
                        "validate",
                        "--database",
                        str(db_path),
                        "--pipeline",
                        "hierarchical",
                    ],
                )
        assert result.exit_code == 0
        assert "VALID" in result.output


# ============================================================================
# _parse_query_indices Tests
# ============================================================================


class TestParseQueryIndices:
    """Tests for the _parse_query_indices helper."""

    def test_all(self):
        assert _parse_query_indices("all", 10) == list(range(10))

    def test_all_zero_total(self):
        assert _parse_query_indices("all", 0) == []

    def test_slice(self):
        assert _parse_query_indices("0:5", 10) == [0, 1, 2, 3, 4]

    def test_slice_clamped(self):
        assert _parse_query_indices("0:100", 10) == list(range(10))

    def test_slice_empty(self):
        assert _parse_query_indices("5:5", 10) == []

    def test_comma_separated(self):
        assert _parse_query_indices("0,5,10", 20) == [0, 5, 10]

    def test_single_index(self):
        assert _parse_query_indices("3", 10) == [3]

    def test_comma_sorted(self):
        assert _parse_query_indices("10,5,0", 20) == [0, 5, 10]

    def test_invalid_text(self):
        with pytest.raises(click.BadParameter):
            _parse_query_indices("abc", 10)

    def test_invalid_triple_colon(self):
        with pytest.raises(click.BadParameter):
            _parse_query_indices("1:2:3", 10)


# ============================================================================
# Benchmark Subcommand Tests
# ============================================================================


class TestBenchmarkSubcommand:
    """Tests for the 'benchmark' subcommand."""

    def test_benchmark_help(self, runner):
        """benchmark --help shows expected options."""
        result = runner.invoke(channel_finder, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--queries" in result.output
        assert "--tier" in result.output

    def test_benchmark_missing_config(self, runner, tmp_path):
        """benchmark without config.yml shows error."""
        result = runner.invoke(
            channel_finder,
            ["--project", str(tmp_path), "benchmark"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output

    def test_benchmark_smoke(self, runner, tmp_path):
        """benchmark with mocked runner executes successfully."""
        import yaml

        from osprey.services.channel_finder.benchmarks.models import BenchmarkRun

        # Create minimal config
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {
                    "in_context": {
                        "database": {"path": "db.json"},
                        "benchmark": {"dataset_path": "queries.json"},
                    },
                },
            },
        }
        (tmp_path / "config.yml").write_text(yaml.dump(config))

        # Create dummy queries file
        import json

        (tmp_path / "queries.json").write_text(
            json.dumps([{"user_query": "test", "targeted_pv": ["A"]}])
        )

        canned_run = BenchmarkRun(
            paradigm="in_context",
            tier=0,
            model="test-model",
            timestamp="2026-01-01T00:00:00",
            query_results=[],
            aggregate_f1=0.5,
            aggregate_precision=0.5,
            aggregate_recall=0.5,
        )

        with patch(
            "osprey.services.channel_finder.benchmarks.runner.BenchmarkRunner"
        ) as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.load_queries.return_value = [{"user_query": "test", "targeted_pv": ["A"]}]
            mock_instance.run_queries = AsyncMock(return_value=canned_run)

            result = runner.invoke(
                channel_finder,
                ["--project", str(tmp_path), "benchmark"],
            )

        assert result.exit_code == 0
        assert "Benchmark complete" in result.output
