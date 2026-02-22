"""Tests for tasks CLI command.

This test module verifies that the task browsing commands work correctly,
including listing tasks, utility functions, and editor/clipboard detection.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.tasks_cmd import (
    copy_to_clipboard,
    detect_editor,
    get_atmention_path,
    get_available_integrations,
    get_available_tasks,
    get_instructions_path,
    get_task_description,
    get_tasks_root,
    has_claude_integration,
    list_tasks,
    tasks,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_tasks_path(tmp_path):
    """Create a mock tasks directory with sample task files."""
    # Create tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Create migrate task
    migrate_dir = tasks_dir / "migrate"
    migrate_dir.mkdir()
    (migrate_dir / "instructions.md").write_text(
        "# Migration Assistant\n\nUpgrade downstream projects to newer OSPREY versions.\n\n## Steps\n..."
    )

    # Create pre-commit task
    precommit_dir = tasks_dir / "pre-commit"
    precommit_dir.mkdir()
    (precommit_dir / "instructions.md").write_text(
        "# Pre-Commit Validation\n\nValidate code before committing.\n"
    )

    # Create testing-workflow task with frontmatter
    testing_dir = tasks_dir / "testing-workflow"
    testing_dir.mkdir()
    (testing_dir / "instructions.md").write_text(
        "---\nworkflow: testing\n---\n\n# Testing Workflow\n\nComprehensive testing guide.\n"
    )

    # Create a task without instructions.md (should be ignored)
    incomplete_dir = tasks_dir / "incomplete-task"
    incomplete_dir.mkdir()
    (incomplete_dir / "notes.txt").write_text("This task has no instructions.md")

    # Create comments task (no skill_description, for testing no-integration case)
    comments_dir = tasks_dir / "comments"
    comments_dir.mkdir()
    (comments_dir / "instructions.md").write_text(
        "# Comments Guidelines\n\nWrite purposeful inline comments.\n"
    )

    # Create integrations directory
    integrations_dir = tmp_path / "integrations"
    integrations_dir.mkdir()

    # Create claude_code integration for migrate and pre-commit (with SKILL.md files)
    claude_code_dir = integrations_dir / "claude_code"
    claude_code_dir.mkdir()
    migrate_int_dir = claude_code_dir / "migrate"
    migrate_int_dir.mkdir()
    (migrate_int_dir / "SKILL.md").write_text("# Migrate Skill\n")
    precommit_int_dir = claude_code_dir / "pre-commit"
    precommit_int_dir.mkdir()
    (precommit_int_dir / "SKILL.md").write_text("# Pre-commit Skill\n")

    return tmp_path


class TestGetTasksRoot:
    """Test the get_tasks_root() utility function."""

    def test_returns_path_object(self):
        """Test that function returns a Path object."""
        result = get_tasks_root()
        assert isinstance(result, Path)

    def test_path_ends_with_tasks(self):
        """Test that the path ends with 'tasks'."""
        result = get_tasks_root()
        assert result.name == "tasks"


class TestGetAvailableTasks:
    """Test the get_available_tasks() function."""

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_list_of_tasks(self, mock_root, mock_tasks_path):
        """Test that function returns list of task directory names."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_available_tasks()

        assert isinstance(result, list)
        assert "migrate" in result
        assert "pre-commit" in result
        assert "testing-workflow" in result
        # incomplete-task should not be included (no instructions.md)
        assert "incomplete-task" not in result

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_empty_list_when_no_tasks_dir(self, mock_root, tmp_path):
        """Test that function returns empty list when tasks directory doesn't exist."""
        mock_root.return_value = tmp_path / "nonexistent"

        result = get_available_tasks()

        assert result == []

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_sorted_list(self, mock_root, mock_tasks_path):
        """Test that tasks are returned in sorted order."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_available_tasks()

        assert result == sorted(result)


class TestGetAvailableIntegrations:
    """Test the get_available_integrations() function."""

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    def test_returns_list_of_integrations(self, mock_root, mock_tasks_path):
        """Test that function returns list of integration directory names."""
        mock_root.return_value = mock_tasks_path / "integrations"

        result = get_available_integrations()

        assert isinstance(result, list)
        assert "claude_code" in result

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    def test_returns_empty_list_when_no_integrations_dir(self, mock_root, tmp_path):
        """Test that function returns empty list when integrations directory doesn't exist."""
        mock_root.return_value = tmp_path / "nonexistent"

        result = get_available_integrations()

        assert result == []


class TestGetTaskDescription:
    """Test the get_task_description() function."""

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_first_content_line(self, mock_root, mock_tasks_path):
        """Test that function returns the first non-header line."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_task_description("migrate")

        assert "Upgrade downstream" in result

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_skips_frontmatter(self, mock_root, mock_tasks_path):
        """Test that function skips YAML frontmatter."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_task_description("testing-workflow")

        # Should skip frontmatter and return first content line after headers
        assert "Comprehensive testing guide" in result

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_empty_for_nonexistent(self, mock_root, tmp_path):
        """Test that function returns empty string for nonexistent task."""
        mock_root.return_value = tmp_path / "tasks"

        result = get_task_description("nonexistent")

        assert result == ""


class TestHasClaudeIntegration:
    """Test the has_claude_integration() function."""

    @patch("osprey.cli.claude_cmd.can_generate_skill")
    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    def test_returns_true_when_custom_wrapper_exists(
        self, mock_root, mock_can_gen, mock_tasks_path
    ):
        """Test that function returns True when custom Claude integration exists."""
        mock_root.return_value = mock_tasks_path / "integrations"
        mock_can_gen.return_value = False  # No auto-generation

        assert has_claude_integration("migrate") is True
        assert has_claude_integration("pre-commit") is True

    @patch("osprey.cli.claude_cmd.can_generate_skill")
    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    def test_returns_true_when_auto_generatable(self, mock_root, mock_can_gen, mock_tasks_path):
        """Test that function returns True when task has skill_description."""
        mock_root.return_value = mock_tasks_path / "integrations"
        mock_can_gen.return_value = True  # Has skill_description

        # testing-workflow has no custom wrapper but can be auto-generated
        assert has_claude_integration("testing-workflow") is True

    @patch("osprey.cli.claude_cmd.can_generate_skill")
    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    def test_returns_false_when_no_integration(self, mock_root, mock_can_gen, mock_tasks_path):
        """Test that function returns False when no Claude integration."""
        mock_root.return_value = mock_tasks_path / "integrations"
        mock_can_gen.return_value = False  # No auto-generation

        # comments has no custom wrapper and no skill_description
        assert has_claude_integration("comments") is False


class TestGetInstructionsPath:
    """Test the get_instructions_path() function."""

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_correct_path(self, mock_root, mock_tasks_path):
        """Test that function returns correct instructions path."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_instructions_path("migrate")

        assert result == mock_tasks_path / "tasks" / "migrate" / "instructions.md"


class TestGetAtmentionPath:
    """Test the get_atmention_path() function."""

    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_returns_atmention_format(self, mock_root, mock_tasks_path):
        """Test that function returns @-prefixed path."""
        mock_root.return_value = mock_tasks_path / "tasks"

        result = get_atmention_path("migrate")

        assert result.startswith("@")
        assert "instructions.md" in result


class TestDetectEditor:
    """Test the detect_editor() function."""

    @patch("shutil.which")
    def test_detects_cursor(self, mock_which):
        """Test that function detects Cursor editor."""
        mock_which.side_effect = lambda cmd: cmd == "cursor"

        result = detect_editor()

        assert result == ("cursor", "Cursor")

    @patch("shutil.which")
    def test_detects_vscode(self, mock_which):
        """Test that function detects VS Code editor."""
        mock_which.side_effect = lambda cmd: cmd == "code"

        result = detect_editor()

        assert result == ("code", "VS Code")

    @patch("shutil.which")
    @patch.dict("os.environ", {"EDITOR": "vim"})
    def test_falls_back_to_editor_env(self, mock_which):
        """Test that function falls back to $EDITOR."""
        mock_which.side_effect = lambda cmd: cmd == "vim"

        result = detect_editor()

        assert result == ("vim", "vim")

    @patch("shutil.which")
    def test_returns_none_when_no_editor(self, mock_which):
        """Test that function returns None when no editor found."""
        mock_which.return_value = None

        result = detect_editor()

        assert result is None


class TestCopyToClipboard:
    """Test the copy_to_clipboard() function."""

    @patch("platform.system")
    @patch("subprocess.run")
    def test_uses_pbcopy_on_macos(self, mock_run, mock_system):
        """Test that function uses pbcopy on macOS."""
        mock_system.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=0)

        result = copy_to_clipboard("test")

        assert result is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["pbcopy"]

    @patch("platform.system")
    @patch("subprocess.run")
    def test_returns_false_on_failure(self, mock_run, mock_system):
        """Test that function returns False on failure."""
        mock_system.return_value = "Darwin"
        mock_run.side_effect = Exception("Failed")

        result = copy_to_clipboard("test")

        assert result is False


class TestTasksListCommand:
    """Test the 'osprey tasks list' command."""

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_list_shows_available_tasks(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_tasks_path
    ):
        """Test that list command displays available tasks."""
        mock_tasks_root.return_value = mock_tasks_path / "tasks"
        mock_int_root.return_value = mock_tasks_path / "integrations"

        result = cli_runner.invoke(list_tasks)

        assert result.exit_code == 0
        assert "migrate" in result.output
        assert "pre-commit" in result.output
        assert "testing-workflow" in result.output
        assert "Available Tasks" in result.output

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_list_shows_task_descriptions(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_tasks_path
    ):
        """Test that list command shows task descriptions."""
        mock_tasks_root.return_value = mock_tasks_path / "tasks"
        mock_int_root.return_value = mock_tasks_path / "integrations"

        result = cli_runner.invoke(list_tasks)

        assert result.exit_code == 0
        # Should show first non-header line from instructions.md
        assert (
            "Upgrade downstream projects" in result.output or "downstream" in result.output.lower()
        )

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_list_shows_integrations(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_tasks_path
    ):
        """Test that list command shows available integrations."""
        mock_tasks_root.return_value = mock_tasks_path / "tasks"
        mock_int_root.return_value = mock_tasks_path / "integrations"

        result = cli_runner.invoke(list_tasks)

        assert result.exit_code == 0
        assert "Claude Code" in result.output

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_list_handles_no_tasks(self, mock_tasks_root, mock_int_root, cli_runner, tmp_path):
        """Test that list command handles case when no tasks exist."""
        empty_tasks = tmp_path / "tasks"
        empty_tasks.mkdir()
        mock_tasks_root.return_value = empty_tasks
        mock_int_root.return_value = tmp_path / "integrations"

        result = cli_runner.invoke(list_tasks)

        assert result.exit_code == 0
        assert "No tasks available" in result.output

    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_list_shows_paths(self, mock_tasks_root, mock_int_root, cli_runner, mock_tasks_path):
        """Test that list command shows file paths for @-mentioning."""
        mock_tasks_root.return_value = mock_tasks_path / "tasks"
        mock_int_root.return_value = mock_tasks_path / "integrations"

        result = cli_runner.invoke(list_tasks)

        assert result.exit_code == 0
        assert "instructions.md" in result.output


class TestTasksGroupCommand:
    """Test the main 'osprey tasks' command group."""

    @patch("osprey.cli.tasks_cmd.QUESTIONARY_AVAILABLE", False)
    @patch("osprey.cli.tasks_cmd.get_integrations_root")
    @patch("osprey.cli.tasks_cmd.get_tasks_root")
    def test_tasks_without_questionary_falls_back_to_list(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_tasks_path
    ):
        """Test that 'osprey tasks' falls back to list when questionary unavailable."""
        mock_tasks_root.return_value = mock_tasks_path / "tasks"
        mock_int_root.return_value = mock_tasks_path / "integrations"

        result = cli_runner.invoke(tasks)

        assert result.exit_code == 0
        # Should show fallback warning and list output
        assert "Available Tasks" in result.output

    def test_tasks_help_shows_subcommands(self, cli_runner):
        """Test that help text shows available subcommands."""
        result = cli_runner.invoke(tasks, ["--help"])

        assert result.exit_code == 0
        assert "list" in result.output.lower()

    def test_tasks_help_mentions_claude_install(self, cli_runner):
        """Test that help text mentions how to install for Claude."""
        result = cli_runner.invoke(tasks, ["--help"])

        assert result.exit_code == 0
        assert "claude" in result.output.lower()
