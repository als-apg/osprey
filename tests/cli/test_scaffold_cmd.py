"""Tests for ``osprey scaffold`` CLI subcommands."""

import json

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.scaffold_cmd import scaffold
from osprey.cli.templates.manifest import MANIFEST_FILENAME


@pytest.fixture()
def project_dir(tmp_path):
    """Create a minimal OSPREY project for scaffold tests."""
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "scaffold-test",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "scaffold-test"


class TestPromptsList:
    """Tests for ``osprey scaffold list``."""

    def test_list_shows_all_artifacts(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--project", str(project_dir)])
        assert result.exit_code == 0
        assert "Build Artifacts" in result.output
        assert "claude-md" in result.output
        assert "agents/channel-finder" in result.output
        assert "hooks/error-guidance" in result.output

    def test_list_shows_framework_managed(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--project", str(project_dir)])
        assert "Framework-managed" in result.output

    def test_list_shows_facility(self, project_dir):
        """rules/facility appears either as framework-managed or user-owned."""
        runner = CliRunner()
        result = runner.invoke(scaffold, ["list", "--project", str(project_dir)])
        assert "rules/facility" in result.output


class TestPromptsClaim:
    """Tests for ``osprey scaffold claim``."""

    def test_claim_marks_as_user_owned(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        assert "scaffold" in config
        assert "user_owned" in config["scaffold"]
        assert "rules/safety" in config["scaffold"]["user_owned"]

    def test_claim_file_stays_in_place(self, project_dir):
        """Claiming an existing file doesn't create overrides/ directory."""
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )

        # File should be in canonical location, not in overrides/
        assert (project_dir / ".claude" / "rules" / "safety.md").exists()
        assert not (project_dir / "overrides").exists()

    def test_claim_renders_missing_file(self, project_dir):
        """Claiming a file that doesn't exist renders it in-place."""
        # Delete the safety file first
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.unlink()
        assert not safety_file.exists()

        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0

        # File should be rendered in canonical location
        assert safety_file.exists()
        assert len(safety_file.read_text().strip()) > 0

    def test_claim_updates_manifest(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )

        manifest = json.loads((project_dir / MANIFEST_FILENAME).read_text())
        assert "user_owned" in manifest
        assert "rules/safety" in manifest["user_owned"]
        assert "framework_hash" in manifest["user_owned"]["rules/safety"]
        assert "claimed_at" in manifest["user_owned"]["rules/safety"]

    def test_claim_unknown_name_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["claim", "nonexistent/thing", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "Unknown artifact" in result.output

    def test_claim_already_owned_errors(self, project_dir):
        runner = CliRunner()
        # Claim once
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        # Claim again should error
        result = runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "already user-owned" in result.output


class TestPromptsDiff:
    """Tests for ``osprey scaffold diff``."""

    def test_diff_not_owned_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "not user-owned" in result.output

    def test_diff_no_changes(self, project_dir):
        runner = CliRunner()
        # Claim (content matches framework)
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(
            scaffold,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "no differences" in result.output

    def test_diff_shows_changes(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )

        # Modify the file in-place
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.write_text("# Modified safety rules\nCustom content.\n")

        result = runner.invoke(
            scaffold,
            ["diff", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "---" in result.output  # unified diff header
        assert "+++" in result.output


class TestPromptsUnclaim:
    """Tests for ``osprey scaffold unclaim``."""

    def test_unclaim_removes_config_entry(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(
            scaffold,
            ["unclaim", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code == 0

        with open(project_dir / "config.yml") as f:
            config = yaml.safe_load(f)
        user_owned = config.get("scaffold", {}).get("user_owned", [])
        assert "rules/safety" not in user_owned

    def test_unclaim_removes_manifest_entry(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        runner.invoke(
            scaffold,
            ["unclaim", "rules/safety", "--project", str(project_dir)],
        )

        manifest = json.loads((project_dir / MANIFEST_FILENAME).read_text())
        assert "rules/safety" not in manifest.get("user_owned", {})

    def test_unclaim_keeps_file(self, project_dir):
        """Unclaim keeps the file in place (user decides what to do)."""
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        runner.invoke(
            scaffold,
            ["unclaim", "rules/safety", "--project", str(project_dir)],
        )

        # File should still exist
        assert (project_dir / ".claude" / "rules" / "safety.md").exists()

    def test_unclaim_not_owned_errors(self, project_dir):
        runner = CliRunner()
        result = runner.invoke(
            scaffold,
            ["unclaim", "rules/safety", "--project", str(project_dir)],
        )
        assert result.exit_code != 0
        assert "not user-owned" in result.output


class TestPromptsListWithUserOwned:
    """Test that list correctly shows user-owned artifacts."""

    def test_list_shows_user_owned_section(self, project_dir):
        runner = CliRunner()
        runner.invoke(
            scaffold,
            ["claim", "rules/safety", "--project", str(project_dir)],
        )
        result = runner.invoke(scaffold, ["list", "--project", str(project_dir)])
        assert "User-owned" in result.output
        assert "rules/safety" in result.output


def test_main_cli_lists_scaffold_not_prompts():
    """`osprey --help` exposes the renamed `scaffold` group, not the legacy `prompts`."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scaffold" in result.stdout
    # The old name must not appear as a command — match only at line start with
    # surrounding whitespace so we don't false-positive on documentation prose.
    for line in result.stdout.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("prompts "), f"legacy 'prompts' command leaked: {line!r}"
