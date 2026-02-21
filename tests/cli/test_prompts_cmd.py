"""Tests for ``osprey prompts`` CLI subcommands."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.prompts_cmd import prompts
from osprey.cli.templates import MANIFEST_FILENAME


@pytest.fixture()
def project_dir(tmp_path):
    """Create a minimal OSPREY project for prompts tests."""
    runner = CliRunner()
    result = runner.invoke(init, ["prompts-test", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path / "prompts-test"


class TestPromptsList:
    """Tests for ``osprey prompts list``."""

    def test_list_shows_all_artifacts(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts, ["list", "--project", str(project_dir)]
        )
        assert result.exit_code == 0
        assert "Prompt Artifacts" in result.output
        assert "claude-md" in result.output
        assert "agents/channel-finder" in result.output
        assert "hooks/error-guidance" in result.output

    def test_list_shows_framework_managed(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts, ["list", "--project", str(project_dir)]
        )
        assert "Framework-managed" in result.output

    def test_list_shows_facility(self, project_dir):
        """rules/facility appears either as framework-managed or overridden."""
        runner = CliRunner()
        result = runner.invoke(
            prompts, ["list", "--project", str(project_dir)]
        )
        assert "rules/facility" in result.output


class TestPromptsScaffold:
    """Tests for ``osprey prompts scaffold``."""

    def test_scaffold_creates_override_file(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0

        override_path = project_dir / "overrides" / ".claude" / "rules" / "safety.md"
        assert override_path.exists()
        content = override_path.read_text()
        assert len(content.strip()) > 0

    def test_scaffold_updates_config(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        assert "prompts" in config
        assert "overrides" in config["prompts"]
        assert "rules/safety" in config["prompts"]["overrides"]

    def test_scaffold_updates_manifest(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )

        manifest = json.loads(
            (project_dir / MANIFEST_FILENAME).read_text()
        )
        assert "overrides" in manifest
        assert "rules/safety" in manifest["overrides"]
        assert "framework_hash" in manifest["overrides"]["rules/safety"]
        assert "scaffolded_at" in manifest["overrides"]["rules/safety"]

    def test_scaffold_unknown_name_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts,
            ["scaffold", "nonexistent/thing", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "Unknown artifact" in result.output

    def test_scaffold_already_exists_errors(self, project_dir):
        runner = CliRunner()
        # Scaffold once
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        # Scaffold again should error
        result = runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "already exists" in result.output


class TestPromptsDiff:
    """Tests for ``osprey prompts diff``."""

    def test_diff_no_override_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "not overridden" in result.output

    def test_diff_no_changes(self, project_dir):
        runner = CliRunner()
        # Scaffold (content matches framework)
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(
            prompts,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "no differences" in result.output

    def test_diff_shows_changes(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )

        # Read config to find override path
        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        override_rel = config["prompts"]["overrides"]["rules/safety"]
        override_path = project_dir / override_rel
        override_path.write_text("# Modified safety rules\nCustom content.\n")

        result = runner.invoke(
            prompts,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "---" in result.output  # unified diff header
        assert "+++" in result.output


class TestPromptsUnoverride:
    """Tests for ``osprey prompts unoverride``."""

    def test_unoverride_removes_config_entry(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(
            prompts,
            ["unoverride", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        overrides = config.get("prompts", {}).get("overrides", {})
        assert "rules/safety" not in overrides

    def test_unoverride_removes_manifest_entry(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        runner.invoke(
            prompts,
            ["unoverride", "rules/safety", "--project", str(project_dir)],
        )

        manifest = json.loads(
            (project_dir / MANIFEST_FILENAME).read_text()
        )
        assert "rules/safety" not in manifest.get("overrides", {})

    def test_unoverride_keeps_file_by_default(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        override_rel = config["prompts"]["overrides"]["rules/safety"]

        runner.invoke(
            prompts,
            ["unoverride", "rules/safety", "--project", str(project_dir)],
        )

        assert (project_dir / override_rel).exists()

    def test_unoverride_deletes_file_when_requested(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        override_rel = config["prompts"]["overrides"]["rules/safety"]

        runner.invoke(
            prompts,
            [
                "unoverride",
                "rules/safety",
                "--delete-file",
                "--project",
                str(project_dir),
            ],
        )

        assert not (project_dir / override_rel).exists()

    def test_unoverride_not_overridden_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            prompts,
            ["unoverride", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "not overridden" in result.output


class TestPromptsListWithOverrides:
    """Test that list correctly shows overridden artifacts."""

    def test_list_shows_overridden_section(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            prompts,
            ["scaffold", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(
            prompts, ["list", "--project", str(project_dir)]
        )
        assert "Overridden" in result.output
        assert "rules/safety" in result.output
