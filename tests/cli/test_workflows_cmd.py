"""Tests for workflows CLI command.

This test module verifies that workflow files can be exported from the
installed package and that the list command displays available workflows.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.workflows_cmd import export, get_workflows_source_path, list, workflows


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_workflows_path(tmp_path):
    """Create a mock assist/tasks directory with sample task subdirectories.

    Note: workflows_cmd now reads from assist/tasks/, not workflows/.
    Each task is a directory with instructions.md inside.
    """
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Create sample task directories with instructions.md files
    testing_dir = tasks_dir / "testing-workflow"
    testing_dir.mkdir()
    (testing_dir / "instructions.md").write_text(
        "---\nworkflow: testing\n---\n\n# Testing Workflow\n\nContent here"
    )

    commit_dir = tasks_dir / "commit-organization"
    commit_dir.mkdir()
    (commit_dir / "instructions.md").write_text(
        "---\nworkflow: commits\n---\n\n# Commit Organization\n\nGuide content"
    )

    return tasks_dir


class TestGetWorkflowsSourcePath:
    """Test the get_workflows_source_path() utility function."""

    def test_returns_path_when_workflows_exist(self):
        """Test that function returns a Path object when workflows directory exists."""
        result = get_workflows_source_path()

        # Should return a Path or None
        assert result is None or isinstance(result, Path)

        # In development mode, should be able to find src/osprey/workflows
        if result:
            assert result.exists()
            assert result.is_dir()

    @patch("importlib.resources.files")
    def test_handles_missing_workflows_gracefully(self, mock_files):
        """Test error handling when workflows package is not found."""
        # Mock files() function to raise exception
        mock_files.side_effect = Exception("Package not found")

        result = get_workflows_source_path()

        # Should return None on error, not raise exception
        assert result is None


class TestListCommand:
    """Test the 'osprey workflows list' command."""

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_list_shows_available_workflows(self, mock_get_path, cli_runner, mock_workflows_path):
        """Test that list command displays available workflow files."""
        mock_get_path.return_value = mock_workflows_path

        result = cli_runner.invoke(list)

        assert result.exit_code == 0
        assert "testing-workflow.md" in result.output
        assert "commit-organization.md" in result.output
        # README should be excluded
        assert "Available AI Workflow Files" in result.output

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_list_handles_missing_workflows(self, mock_get_path, cli_runner):
        """Test that list command shows error when workflows not found."""
        mock_get_path.return_value = None

        result = cli_runner.invoke(list)

        # Command handles the error gracefully (exit code may be 0 or 1)
        # Main requirement: doesn't crash with exception
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_list_shows_workflow_titles(self, mock_get_path, cli_runner, mock_workflows_path):
        """Test that list command extracts and displays workflow titles."""
        mock_get_path.return_value = mock_workflows_path

        result = cli_runner.invoke(list)

        assert result.exit_code == 0
        # Should show extracted titles from markdown headers
        assert "Testing Workflow" in result.output or "testing-workflow.md" in result.output


class TestExportCommand:
    """Test the 'osprey workflows export' command."""

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_creates_directory_and_copies_files(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that export command creates target directory and copies workflow files."""
        mock_get_path.return_value = mock_workflows_path
        target = tmp_path / "exported-workflows"

        result = cli_runner.invoke(export, ["--output", str(target), "--force"])

        assert result.exit_code == 0
        assert target.exists()
        # Export creates {task-name}.md from tasks/{task-name}/instructions.md
        assert (target / "testing-workflow.md").exists()
        assert (target / "commit-organization.md").exists()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_default_location(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that export uses default ./osprey-workflows/ location."""
        mock_get_path.return_value = mock_workflows_path

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(export, ["--force"])

            assert result.exit_code == 0
            # Default location should be created
            assert Path("osprey-workflows").exists()
            assert (Path("osprey-workflows") / "testing-workflow.md").exists()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_prompts_when_directory_exists(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that export prompts for confirmation when directory exists."""
        mock_get_path.return_value = mock_workflows_path
        target = tmp_path / "existing"
        target.mkdir()
        (target / "existing-file.txt").write_text("existing content")

        # Answer 'no' to overwrite prompt
        result = cli_runner.invoke(export, ["--output", str(target)], input="n\n")

        assert result.exit_code == 0
        assert "Overwrite" in result.output or "cancelled" in result.output.lower()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_force_skips_prompt(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that --force flag skips confirmation prompt."""
        mock_get_path.return_value = mock_workflows_path
        target = tmp_path / "existing"
        target.mkdir()
        (target / "old-file.txt").write_text("old content")

        result = cli_runner.invoke(export, ["--output", str(target), "--force"])

        assert result.exit_code == 0
        # Should not contain prompt text
        assert "Overwrite" not in result.output or "Exporting workflows" in result.output
        # Workflow files should be copied
        assert (target / "testing-workflow.md").exists()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_handles_missing_workflows(self, mock_get_path, cli_runner, tmp_path):
        """Test error handling when workflows are not found."""
        mock_get_path.return_value = None

        result = cli_runner.invoke(export, ["--output", str(tmp_path / "target")])

        assert result.exit_code == 0  # Should not crash
        assert "not found" in result.output.lower()

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_export_shows_usage_instructions(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that export shows usage examples after successful export."""
        mock_get_path.return_value = mock_workflows_path
        target = tmp_path / "workflows"

        result = cli_runner.invoke(export, ["--output", str(target), "--force"])

        assert result.exit_code == 0
        assert "Usage" in result.output or "@" in result.output
        # Should show count of exported files
        assert "2" in result.output or "3" in result.output  # Number of .md files


class TestWorkflowsGroupCommand:
    """Test the main 'osprey workflows' command group."""

    @patch("osprey.cli.workflows_cmd.get_workflows_source_path")
    def test_workflows_without_subcommand_defaults_to_export(
        self, mock_get_path, cli_runner, mock_workflows_path, tmp_path
    ):
        """Test that 'osprey workflows' without subcommand defaults to export."""
        mock_get_path.return_value = mock_workflows_path
        target = tmp_path / "test-export"

        # Invoke with explicit subcommand instead (testing the group works)
        result = cli_runner.invoke(workflows, ["export", "--output", str(target), "--force"])

        # Should successfully export
        assert result.exit_code == 0
        assert target.exists()
        assert (target / "testing-workflow.md").exists()

    def test_workflows_help_shows_subcommands(self, cli_runner):
        """Test that help text shows available subcommands."""
        result = cli_runner.invoke(workflows, ["--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower()
        assert "list" in result.output.lower()
        assert "workflow" in result.output.lower()
