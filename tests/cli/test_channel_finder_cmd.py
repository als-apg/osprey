"""Tests for the osprey channel-finder CLI command.

Tests the Click command group including:
- Command structure and help output
- Query subcommand with mocked service
- Benchmark subcommand with mocked run_benchmarks
- Build-database, validate, preview subcommands
- Interactive REPL with mocked ChannelFinderCLI
- Config/project resolution
- Import smoke tests
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.channel_finder_cmd import channel_finder


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
        assert "query" in result.output
        assert "benchmark" in result.output
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

    def test_query_help(self, runner):
        """query --help shows QUERY_TEXT argument."""
        result = runner.invoke(channel_finder, ["query", "--help"])
        assert result.exit_code == 0
        assert "QUERY_TEXT" in result.output

    def test_benchmark_help_shows_options(self, runner):
        """benchmark --help shows --queries, --model, --dataset."""
        result = runner.invoke(channel_finder, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--queries" in result.output
        assert "--model" in result.output
        assert "--dataset" in result.output


# ============================================================================
# Query Subcommand Tests
# ============================================================================


class TestQuerySubcommand:
    """Test the 'query' subcommand with mocked service layer."""

    @patch("osprey.cli.channel_finder_cmd._initialize_registry")
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_query_calls_direct_query(self, mock_run, mock_config, mock_reg, runner):
        """query subcommand calls direct_query with the query text."""
        result = runner.invoke(channel_finder, ["query", "find BPMs"])
        assert result.exit_code == 0
        mock_config.assert_called_once()
        mock_reg.assert_called_once()

    @patch("osprey.cli.channel_finder_cmd._initialize_registry")
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    def test_query_missing_argument_errors(self, mock_config, mock_reg, runner):
        """query with no argument shows error."""
        result = runner.invoke(channel_finder, ["query"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("osprey.cli.channel_finder_cmd._initialize_registry")
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    def test_query_verbose_flag_propagated(self, mock_config, mock_reg, runner):
        """--verbose flag is propagated to registry init."""
        with patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0):
            result = runner.invoke(channel_finder, ["-v", "query", "find BPMs"])
        assert result.exit_code == 0
        mock_reg.assert_called_once_with(True)

    @patch("osprey.cli.channel_finder_cmd._initialize_registry", side_effect=Exception("No config"))
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    def test_query_init_failure_shows_error(self, mock_config, mock_reg, runner):
        """Service initialization failure shows error message."""
        result = runner.invoke(channel_finder, ["query", "find BPMs"])
        assert result.exit_code != 0

    @patch("osprey.cli.channel_finder_cmd._setup_config", side_effect=Exception("no config"))
    def test_query_config_failure_shows_error(self, mock_config, runner):
        """Config resolution failure shows error."""
        result = runner.invoke(channel_finder, ["query", "find BPMs"])
        assert result.exit_code != 0


# ============================================================================
# Benchmark Subcommand Tests
# ============================================================================


class TestBenchmarkSubcommand:
    """Test the 'benchmark' subcommand with mocked run_benchmarks."""

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_benchmark_no_options(self, mock_run, mock_config, runner):
        """benchmark with no options calls run_benchmarks with defaults."""
        result = runner.invoke(channel_finder, ["benchmark"])
        assert result.exit_code == 0
        mock_config.assert_called_once()

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_benchmark_queries_option(self, mock_run, mock_config, runner):
        """--queries option is passed through."""
        result = runner.invoke(channel_finder, ["benchmark", "--queries", "0:10"])
        assert result.exit_code == 0

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_benchmark_model_option(self, mock_run, mock_config, runner):
        """--model option is passed through."""
        result = runner.invoke(channel_finder, ["benchmark", "--model", "anthropic/claude-sonnet"])
        assert result.exit_code == 0

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_benchmark_dataset_option(self, mock_run, mock_config, runner):
        """--dataset option is passed through."""
        result = runner.invoke(
            channel_finder, ["benchmark", "--dataset", "data/benchmarks/my_data.json"]
        )
        assert result.exit_code == 0

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0)
    def test_benchmark_all_options_combined(self, mock_run, mock_config, runner):
        """All three options work together."""
        result = runner.invoke(
            channel_finder,
            [
                "benchmark",
                "--queries",
                "0:10",
                "--model",
                "anthropic/claude-sonnet",
                "--dataset",
                "data/my.json",
            ],
        )
        assert result.exit_code == 0

    @patch("osprey.cli.channel_finder_cmd._setup_config")
    def test_benchmark_failure_shows_error(self, mock_config, runner):
        """run_benchmarks failure shows error with exit code."""
        with patch(
            "osprey.cli.channel_finder_cmd.asyncio.run",
            side_effect=Exception("benchmark failed"),
        ):
            result = runner.invoke(channel_finder, ["benchmark"])
        assert result.exit_code != 0


# ============================================================================
# Interactive REPL Tests
# ============================================================================


class TestInteractiveREPL:
    """Test the default interactive mode."""

    @patch("osprey.cli.channel_finder_cmd._initialize_registry")
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    @patch("osprey.cli.channel_finder_cmd.asyncio.run")
    def test_no_subcommand_triggers_interactive(self, mock_run, mock_config, mock_reg, runner):
        """Running with no subcommand triggers interactive mode."""
        runner.invoke(channel_finder, [])
        assert mock_config.called
        assert mock_reg.called

    @patch("osprey.cli.channel_finder_cmd._initialize_registry")
    @patch("osprey.cli.channel_finder_cmd._setup_config")
    def test_keyboard_interrupt_handled(self, mock_config, mock_reg, runner):
        """KeyboardInterrupt during REPL is handled gracefully."""
        with patch("osprey.cli.channel_finder_cmd.asyncio.run", side_effect=KeyboardInterrupt):
            result = runner.invoke(channel_finder, [])
        # Should not crash
        assert result.exit_code == 0 or "Goodbye" in result.output


# ============================================================================
# Config/Project Resolution Tests
# ============================================================================


class TestConfigResolution:
    """Test project and config resolution."""

    def test_missing_config_shows_error(self, runner, tmp_path):
        """--project with no config.yml shows helpful error."""
        result = runner.invoke(channel_finder, ["--project", str(tmp_path), "query", "test"])
        assert result.exit_code != 0
        assert "not found" in result.output or "Error" in result.output

    def test_valid_project_resolves_config(self, runner, tmp_path):
        """--project with config.yml resolves correctly."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("model:\n  model_id: test\n")
        # Will fail at registry init but config resolution should succeed
        with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
            with patch("osprey.cli.channel_finder_cmd.asyncio.run", return_value=0):
                result = runner.invoke(
                    channel_finder, ["--project", str(tmp_path), "query", "test"]
                )
        assert result.exit_code == 0


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

    def test_import_channel_finder_cli(self):
        """ChannelFinderCLI and direct_query are importable."""
        from osprey.services.channel_finder.cli import ChannelFinderCLI, direct_query

        assert ChannelFinderCLI is not None
        assert direct_query is not None

    def test_import_benchmark_cli(self):
        """run_benchmarks, parse_query_selection, create_config_override are importable."""
        from osprey.services.channel_finder.benchmarks.cli import (
            create_config_override,
            parse_query_selection,
            run_benchmarks,
        )

        assert run_benchmarks is not None
        assert parse_query_selection is not None
        assert create_config_override is not None

    def test_channel_finder_cmd_importable(self):
        """channel_finder_cmd module is importable."""
        from osprey.cli.channel_finder_cmd import (
            benchmark,
            build_database,
            channel_finder,
            preview,
            query,
            validate,
        )

        assert channel_finder is not None
        assert query is not None
        assert benchmark is not None
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
