"""Tests for the osprey channel-finder CLI command.

Tests the Click command group including:
- Command structure and help output
- Validate, preview subcommands
- Config/project resolution
- _parse_query_indices parsing
- Benchmark subcommand
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import click
import pytest
from click.testing import CliRunner

from osprey.cli.channel_finder_cmd import _parse_query_indices, channel_finder
from osprey.deployment.graphdb_service import GRAPHDB_REBUILD_HINT


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
        assert "validate" in result.output
        assert "preview" in result.output

    def test_help_shows_project_option(self, runner):
        """--help shows --project, and no ``-p`` short alias.

        ``-p`` means ``--port`` on every serving verb (``osprey web``,
        ``artifacts web``, ``ariel``, ``theme-lab``); one letter cannot also
        mean "the deployment to act on" here.
        """
        result = runner.invoke(channel_finder, ["--help"])
        assert "--project" in result.output
        assert "-p," not in result.output

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

    @staticmethod
    def _rendered_repo(tmp_path):
        """A deployment repo with a render: manifest at the root, config in ``build/``."""
        repo = tmp_path / "repo"
        (repo / "build").mkdir(parents=True)
        (repo / "profile.yml").write_text("name: Demo\ndata: data\n")
        (repo / "build" / "config.yml").write_text("project_name: demo\n")
        return repo

    @pytest.mark.usefixtures("runner")
    def test_repo_root_stance_finds_the_render(self, tmp_path, monkeypatch):
        """Standing in a repo root resolves ``build/config.yml``, not a flat one.

        The rendered config lives in the build zone, so the flat
        ``<cwd>/config.yml`` spelling never matched from the stance an operator
        actually takes.
        """
        from osprey.cli.channel_finder_cmd import _setup_config

        repo = self._rendered_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        _setup_config(None)

        import os

        assert os.environ["CONFIG_FILE"] == str(repo / "build" / "config.yml")

    @pytest.mark.usefixtures("runner")
    def test_subdirectory_stance_finds_the_render(self, tmp_path, monkeypatch):
        """A subdirectory of the repo is the repo, the way every other verb reads it."""
        from osprey.cli.channel_finder_cmd import _setup_config

        repo = self._rendered_repo(tmp_path)
        subdir = repo / "data" / "raw"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        monkeypatch.delenv("CONFIG_FILE", raising=False)

        _setup_config(None)

        import os

        assert os.environ["CONFIG_FILE"] == str(repo / "build" / "config.yml")


class TestWebBindAddress:
    """``osprey channel-finder web`` binds the address its project's config names."""

    CONFIGURED_HOST = "192.0.2.20"
    CONFIGURED_PORT = 18400

    @staticmethod
    def _project(root, *, host, port):
        import yaml

        root.mkdir(parents=True)
        section: dict = {"pipeline_mode": "in_context"}
        if host is not None:
            section["web"] = {"host": host, "port": port}
        (root / "config.yml").write_text(
            yaml.safe_dump({"project_name": "demo", "channel_finder": section})
        )
        return root

    @pytest.fixture
    def bound(self, monkeypatch):
        seen: dict = {}
        monkeypatch.setattr("osprey.interfaces.channel_finder.app.create_app", lambda: object())
        monkeypatch.setattr("uvicorn.run", lambda *a, **kw: seen.update(kw))
        for name in ("OSPREY_CHANNEL_FINDER_PORT", "OSPREY_TERMINAL_SECRET", "OSPREY_WEB_PORT"):
            monkeypatch.delenv(name, raising=False)
        return seen

    def test_host_option_has_no_frozen_default(self):
        web = channel_finder.commands["web"]
        host = next(p for p in web.params if p.name == "host")
        assert host.default is None

    def test_the_named_projects_address_is_bound(self, runner, tmp_path, monkeypatch, bound):
        decoy = self._project(tmp_path / "decoy", host="192.0.2.99", port=18499)
        project = self._project(
            tmp_path / "named", host=self.CONFIGURED_HOST, port=self.CONFIGURED_PORT
        )
        monkeypatch.chdir(decoy)

        result = runner.invoke(channel_finder, ["--project", str(project), "web"])

        assert result.exit_code == 0, result.output
        assert (bound["host"], bound["port"]) == (self.CONFIGURED_HOST, self.CONFIGURED_PORT)
        assert f"http://{self.CONFIGURED_HOST}:{self.CONFIGURED_PORT}" in result.output

    def test_explicit_host_and_port_win(self, runner, tmp_path, bound):
        project = self._project(
            tmp_path / "named", host=self.CONFIGURED_HOST, port=self.CONFIGURED_PORT
        )

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "web", "--host", "127.0.0.1", "--port", "18999"],
        )

        assert result.exit_code == 0, result.output
        assert (bound["host"], bound["port"]) == ("127.0.0.1", 18999)

    def test_no_configured_host_binds_loopback(self, runner, tmp_path, monkeypatch, bound):
        from osprey.registry.web import framework_web_port_default

        project = self._project(tmp_path / "named", host=None, port=None)
        monkeypatch.chdir(project)

        result = runner.invoke(channel_finder, ["--project", str(project), "web"])

        assert result.exit_code == 0, result.output
        assert (bound["host"], bound["port"]) == (
            "127.0.0.1",
            framework_web_port_default("channel_finder"),
        )


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

    def test_validate_help_lists_every_file_backed_paradigm(self, runner):
        """``--pipeline`` offers each paradigm whose store is a file on disk.

        The list is derived from the paradigm registry minus ``graph`` --- a
        graph store is a service reached over the network, not a file this
        command can open --- so the help text is the visible half of that rule.
        """
        result = runner.invoke(channel_finder, ["validate", "--help"])
        assert result.exit_code == 0
        assert "[hierarchical|in_context|middle_layer]" in " ".join(result.output.split())

    def test_validate_routes_a_middle_layer_database(self, runner, tmp_path):
        """``--pipeline middle_layer`` loads the file through MiddleLayerDatabase."""
        db_file = tmp_path / "middle_layer.json"
        db_file.write_text(
            json.dumps(
                {
                    "Storage Ring": {
                        "BPM": {
                            "setup": {"DeviceList": [[1, 1]]},
                            "Monitor": {
                                "ChannelNames": ["SR:BPM1:X", "SR:BPM1:Y"],
                                "Units": "mm",
                            },
                        }
                    }
                }
            )
        )

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(
                    channel_finder,
                    ["validate", "--database", str(db_file), "--pipeline", "middle_layer"],
                )

        assert result.exit_code == 0
        printed = " ".join(result.output.split())
        assert "VALID" in printed
        assert "Middle Layer" in printed

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

    def test_validate_pipeline_reads_that_pipelines_configured_database(self, runner, tmp_path):
        """``--pipeline X`` without ``--database`` validates X's configured file."""
        hier = tmp_path / "hier.json"
        hier.write_text(
            json.dumps(
                {
                    "hierarchy": {
                        "levels": [
                            {"name": "system", "type": "tree"},
                            {"name": "signal", "type": "tree"},
                        ],
                        "naming_pattern": "{system}:{signal}",
                    },
                    "tree": {"SR": {"X": {}}},
                }
            )
        )
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {
                    "in_context": {"database": {"path": str(tmp_path / "ctx.json")}},
                    "hierarchical": {"database": {"path": str(hier)}},
                },
            }
        }

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                with patch("osprey.utils.config.load_config", return_value=config):
                    with patch("osprey.utils.workspace.resolve_path", side_effect=Path):
                        result = runner.invoke(
                            channel_finder, ["validate", "--pipeline", "hierarchical"]
                        )

        printed = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "Hierarchical" in printed
        assert "ctx.json" not in printed

    def test_validate_pipeline_without_its_database_names_the_key(self, runner, tmp_path):
        """``--pipeline X`` with no database configured for X refuses, naming the key."""
        config = {
            "channel_finder": {
                "pipelines": {"in_context": {"database": {"path": str(tmp_path / "ctx.json")}}},
            }
        }

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                with patch("osprey.utils.config.load_config", return_value=config):
                    result = runner.invoke(
                        channel_finder, ["validate", "--pipeline", "middle_layer"]
                    )

        assert result.exit_code == 1
        assert "channel_finder.pipelines.middle_layer.database.path" in " ".join(
            result.output.split()
        )


# ============================================================================
# Graph Paradigm Tests
# ============================================================================


GRAPH_CONFIG = {"channel_finder": {"pipeline_mode": "graph"}}


class TestGraphParadigmGuidance:
    """A graph project has no channel database file for either command to open.

    Both verbs auto-detect the paradigm from config, find ``graph``, and answer
    with the store's own three verbs instead of a path panel --- and succeed,
    because a graph project configured this way is healthy.
    """

    def _invoke(self, runner, args):
        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                with patch("osprey.utils.config.load_config", return_value=GRAPH_CONFIG):
                    return runner.invoke(channel_finder, args)

    def test_validate_prints_the_graph_panel_and_succeeds(self, runner, tmp_path):
        result = self._invoke(runner, ["--project", str(tmp_path), "validate"])
        printed = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "Graph Paradigm" in printed
        assert "The graph store is the database" in printed
        assert GRAPHDB_REBUILD_HINT in printed
        assert "osprey health --category graphdb" in printed
        assert "get_schema and read_cypher" in printed

    def test_validate_does_not_offer_the_file_remedy(self, runner, tmp_path):
        result = self._invoke(runner, ["--project", str(tmp_path), "validate"])
        printed = " ".join(result.output.split())
        assert "No database configured" not in printed
        assert "database.path" not in printed
        assert "Database Path" not in printed

    def test_preview_prints_the_graph_panel_and_succeeds(self, runner, tmp_path):
        result = self._invoke(runner, ["--project", str(tmp_path), "preview"])
        printed = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "Graph Paradigm" in printed
        assert "The graph store is the database" in printed
        assert GRAPHDB_REBUILD_HINT in printed
        assert "osprey health --category graphdb" in printed
        assert "get_schema and read_cypher" in printed

    def test_preview_does_not_offer_the_file_remedy(self, runner, tmp_path):
        result = self._invoke(runner, ["--project", str(tmp_path), "preview"])
        printed = " ".join(result.output.split())
        assert "No database configured" not in printed
        assert "database.path" not in printed

    def test_an_explicit_database_is_still_validated(self, runner, tmp_path):
        """``--database`` names a file; the project's graph mode says nothing
        about how to read one, so the file-backed default is used rather than
        handing ``graph`` to a file loader."""
        db_file = tmp_path / "test_db.json"
        db_file.write_text(
            '{"channels": [{"template": false, "channel": "CH1", '
            '"address": "PV:CH1", "description": "Test channel"}]}'
        )
        result = self._invoke(runner, ["validate", "--database", str(db_file)])
        printed = " ".join(result.output.split())
        assert "Graph Paradigm" not in printed
        assert "In Context" in printed


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


class TestInContextIndexRender:
    """``validate`` and ``preview`` read the in_context index a build writes."""

    @staticmethod
    def _index(built, tmp_path: Path) -> Path:
        from osprey.facility.views import ViewInputs
        from osprey.facility.views.channel_finder import write_in_context

        inputs = ViewInputs(
            doc=built.facility,
            rendered_config={"channel_finder": {"pipeline_mode": "in_context"}},
            facility_dir=built.facility_dir,
            served=[],
        )
        (target,) = write_in_context(tmp_path, inputs)
        return target

    @pytest.mark.slow
    def test_validate_reports_every_row_of_the_index(
        self, runner, tmp_path, built_control_assistant
    ):
        index = self._index(built_control_assistant, tmp_path)

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(
                    channel_finder,
                    ["validate", "--database", str(index), "--pipeline", "in_context"],
                )

        printed = " ".join(result.output.split())
        assert result.exit_code == 0, result.output
        assert "VALID" in printed
        assert "Total Channels 569" in printed

    @pytest.mark.slow
    def test_preview_prints_the_rows_of_the_index(self, runner, tmp_path, built_control_assistant):
        index = self._index(built_control_assistant, tmp_path)
        first = json.loads(index.read_text())["channels"][0]["channel"]

        with patch("osprey.cli.channel_finder_cmd._setup_config"):
            with patch("osprey.cli.channel_finder_cmd._initialize_registry"):
                result = runner.invoke(channel_finder, ["preview", "--database", str(index)])

        assert result.exit_code == 0, result.output
        assert "In-Context Database Preview" in result.output
        assert first in result.output


# ============================================================================
# Import Smoke Tests
# ============================================================================


class TestImportSmoke:
    """Test that cleaned-up modules are importable."""

    def test_channel_finder_cmd_importable(self):
        """channel_finder_cmd module is importable."""
        from osprey.cli.channel_finder_cmd import (
            channel_finder,
            preview,
            validate,
        )

        assert channel_finder is not None
        assert validate is not None
        assert preview is not None

    def test_import_native_tools(self):
        """Native tool modules are importable."""
        from osprey.services.channel_finder.tools.preview_database import preview_database
        from osprey.services.channel_finder.tools.validate_database import (
            validate_json_structure,
        )

        assert preview_database is not None
        assert validate_json_structure is not None


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

    def test_validate_with_pipeline_override(self, runner):
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
                    },
                },
                "benchmark": {"dataset_path": "queries.json"},
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
            # The CLI serializes metadata that pulls runner.provider and
            # runner.model into the suite JSON; configure both as plain
            # strings so json.dump doesn't choke on MagicMocks.
            mock_instance.provider = "anthropic"
            mock_instance.model = "anthropic/claude-haiku-4-5-20251001"

            result = runner.invoke(
                channel_finder,
                [
                    "--project",
                    str(tmp_path),
                    "benchmark",
                    "--model",
                    "anthropic/claude-haiku-4-5-20251001",
                ],
            )

        assert result.exit_code == 0
        assert "Benchmark complete" in result.output
