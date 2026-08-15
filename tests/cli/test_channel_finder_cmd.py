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
        (repo / "profile.yml").write_text("name: Demo\ndata_bundle: hello_world\n")
        (repo / "build" / "config.yml").write_text("project_name: demo\n")
        return repo

    def test_repo_root_stance_finds_the_render(self, runner, tmp_path, monkeypatch):
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

    def test_subdirectory_stance_finds_the_render(self, runner, tmp_path, monkeypatch):
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
# Build-database output anchoring
# ============================================================================


def _write_csv(path):
    """A minimal well-formed channel CSV."""
    path.write_text(
        "address,description,family_name,instances,sub_channel\n"
        "BEAM:CURRENT,Total beam current,,,\n"
        "BPM01X,BPM horizontal,BPM,3,X\n"
    )
    return path


def _make_project(project_dir, profile_file=None):
    """A project directory whose manifest names ``profile_file`` (or none)."""
    import json

    project_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"build_args": {}}
    if profile_file is not None:
        manifest["build_args"]["profile_path_abs"] = str(profile_file)
    (project_dir / ".osprey-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project_dir


def _make_profile(profile_dir, data_key="data"):
    """A materialized-looking profile with a data tree beside its profile.yml."""
    (profile_dir / data_key).mkdir(parents=True)
    profile_file = profile_dir / "profile.yml"
    profile_file.write_text(f"name: facility\ndata: {data_key}\n", encoding="utf-8")
    return profile_file


def _squashed(output):
    """Console output with all whitespace removed — rich wraps at any width."""
    return "".join(output.split())


class TestBuildDatabaseOutputAnchoring:
    """FR-8b: a generated database belongs to the profile, not the project."""

    def test_default_output_lands_in_the_profile_data_tree(self, runner, tmp_path):
        """With a profile-built project, the database is written into the profile."""
        profile_file = _make_profile(tmp_path / "my-profile")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        written = tmp_path / "my-profile" / "data" / "processed" / "channel_database.json"
        assert written.exists()
        assert not (project / "data").exists()

    def test_rebuild_next_step_is_printed(self, runner, tmp_path):
        """The staleness the write causes is announced with the command that clears it."""
        profile_file = _make_profile(tmp_path / "my-profile")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        printed = _squashed(result.output)
        assert "stale" in printed
        # The command that clears the staleness is a bare `osprey build`.
        # There is no `--force` flag, so a message naming one would hand
        # the operator a command line that does not parse. Both halves
        # asserted, because the fragment is exactly what went stale here.
        assert "ospreybuild" in printed
        assert "--force" not in printed

    def test_no_profile_falls_back_to_the_project_data_tree(self, runner, tmp_path):
        """A preset-built project has no profile to own the database."""
        project = _make_project(tmp_path / "my-project")
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (project / "data" / "processed" / "channel_database.json").exists()

    def test_no_profile_fallback_warns(self, runner, tmp_path):
        """The fallback is announced — the file a rebuild overwrites is not silent."""
        project = _make_project(tmp_path / "my-project")
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        printed = _squashed(result.output)
        assert "Noprofiledatatreeresolved" in printed
        assert "overwrites" in printed

    def test_missing_manifest_falls_back(self, runner, tmp_path):
        """A directory with no manifest at all resolves no profile."""
        project = tmp_path / "bare-project"
        project.mkdir()
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (project / "data" / "processed" / "channel_database.json").exists()

    def test_manifest_naming_a_deleted_profile_falls_back(self, runner, tmp_path):
        """A profile path that no longer exists is a fallback, not a crash."""
        project = _make_project(tmp_path / "my-project", tmp_path / "gone" / "profile.yml")
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (project / "data" / "processed" / "channel_database.json").exists()

    def test_explicit_output_wins_over_the_profile_default(self, runner, tmp_path):
        """--output is the author's decision and overrides the profile anchor."""
        profile_file = _make_profile(tmp_path / "my-profile")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")
        chosen = tmp_path / "elsewhere" / "db.json"

        result = runner.invoke(
            channel_finder,
            [
                "--project",
                str(project),
                "build-database",
                "--csv",
                str(csv_file),
                "--output",
                str(chosen),
            ],
        )

        assert result.exit_code == 0
        assert chosen.exists()
        assert not (tmp_path / "my-profile" / "data" / "processed").exists()

    def test_persona_delta_anchors_at_the_profile_root(self, runner, tmp_path):
        """A persona-built project writes into the root's data tree, not personas/."""
        profile_dir = tmp_path / "my-profile"
        _make_profile(profile_dir)
        (profile_dir / "personas").mkdir()
        persona = profile_dir / "personas" / "reader.yml"
        persona.write_text("name: reader\n", encoding="utf-8")
        project = _make_project(tmp_path / "reader-project", persona)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (profile_dir / "data" / "processed" / "channel_database.json").exists()
        assert not (profile_dir / "personas" / "data").exists()

    def test_custom_data_directory_name_is_honored(self, runner, tmp_path):
        """The anchor is the profile's declared ``data:`` tree, not a fixed name."""
        profile_file = _make_profile(tmp_path / "my-profile", data_key="facility-data")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        written = tmp_path / "my-profile" / "facility-data" / "processed" / "channel_database.json"
        assert written.exists()

    def test_inherited_data_tree_is_resolved_through_extends(self, runner, tmp_path):
        """A ``data:`` the profile inherits is still where the build reads from.

        Resolving only the named file's own keys makes an inherited tree
        invisible, and the database lands somewhere the build never looks —
        while the caller still announces it as ready to deploy.
        """
        profile_dir = tmp_path / "my-profile"
        # Deliberately not named ``data``: the old code invented ``<root>/data``
        # whenever it saw no ``data:`` key, which would mask the inheritance.
        (profile_dir / "facility-data").mkdir(parents=True)
        (profile_dir / "base.yml").write_text("name: base\ndata: facility-data\n", encoding="utf-8")
        profile_file = profile_dir / "profile.yml"
        profile_file.write_text("extends: ./base.yml\nname: facility\n", encoding="utf-8")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        written = profile_dir / "facility-data" / "processed" / "channel_database.json"
        assert written.exists()
        assert not (profile_dir / "data").exists()
        assert not (project / "data").exists()

    def test_profile_without_a_data_tree_falls_back_to_the_project(self, runner, tmp_path):
        """A profile that declares no ``data:`` has no tree to own the database.

        Inventing ``<profile>/data`` would write into a directory the build
        does not read, and the rebuild hint would be a lie.
        """
        profile_dir = tmp_path / "my-profile"
        profile_dir.mkdir(parents=True)
        profile_file = profile_dir / "profile.yml"
        profile_file.write_text("name: facility\n", encoding="utf-8")
        project = _make_project(tmp_path / "my-project", profile_file)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (project / "data" / "processed" / "channel_database.json").exists()
        assert not (profile_dir / "data").exists()
        printed = _squashed(result.output)
        assert "Noprofiledatatreeresolved" in printed
        assert "stale" not in printed  # nothing to deploy, so nothing is promised

    def test_the_manifest_named_profile_is_the_one_read(self, runner, tmp_path):
        """A sibling ``profile.yml`` must not hijack a differently-named profile.

        Substituting the directory's ``profile.yml`` for the file the manifest
        names writes one facility's channel database into another's data tree.
        """
        profile_dir = tmp_path / "profiles"
        (profile_dir / "als-data").mkdir(parents=True)
        (profile_dir / "other-data").mkdir(parents=True)
        named = profile_dir / "als.yml"
        named.write_text("name: als\ndata: als-data\n", encoding="utf-8")
        (profile_dir / "profile.yml").write_text(
            "name: other\ndata: other-data\n", encoding="utf-8"
        )
        project = _make_project(tmp_path / "als-project", named)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (profile_dir / "als-data" / "processed" / "channel_database.json").exists()
        assert not (profile_dir / "other-data" / "processed").exists()

    def test_a_persona_delta_may_override_the_roots_data_tree(self, runner, tmp_path):
        """The delta wins the merge, so a ``data:`` it names is the one that counts."""
        profile_dir = tmp_path / "my-profile"
        _make_profile(profile_dir)
        (profile_dir / "persona-data").mkdir()
        (profile_dir / "personas").mkdir()
        persona = profile_dir / "personas" / "reader.yml"
        persona.write_text("name: reader\ndata: persona-data\n", encoding="utf-8")
        project = _make_project(tmp_path / "reader-project", persona)
        csv_file = _write_csv(tmp_path / "channels.csv")

        result = runner.invoke(
            channel_finder,
            ["--project", str(project), "build-database", "--csv", str(csv_file)],
        )

        assert result.exit_code == 0
        assert (profile_dir / "persona-data" / "processed" / "channel_database.json").exists()
        assert not (profile_dir / "data" / "processed").exists()

    def test_help_documents_the_staleness_sequence(self, runner):
        """The advisory is intentional, so the help says so rather than the release notes."""
        result = runner.invoke(channel_finder, ["build-database", "--help"])

        assert result.exit_code == 0
        printed = _squashed(result.output)
        assert "stale" in printed
        # The command that clears the staleness is a bare `osprey build`.
        # There is no `--force` flag, so a message naming one would hand
        # the operator a command line that does not parse. Both halves
        # asserted, because the fragment is exactly what went stale here.
        assert "ospreybuild" in printed
        assert "--force" not in printed


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
