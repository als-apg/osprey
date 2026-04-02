"""Tests for claude CLI command.

This test module verifies that the Claude Code integration commands
work correctly, including regen, status, and chat.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.claude_cmd import (
    chat_claude,
    claude,
    get_claude_skills_dir,
    get_installed_skills,
    regen,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestGetClaudeSkillsDir:
    """Test the get_claude_skills_dir() utility function."""

    def test_returns_path_in_cwd(self):
        """Test that function returns path in current working directory."""
        result = get_claude_skills_dir()
        assert result == Path.cwd() / ".claude" / "skills"


class TestGetInstalledSkills:
    """Test the get_installed_skills() function."""

    def test_returns_empty_list_when_no_skills_dir(self, tmp_path):
        """Test that function returns empty list when .claude/skills doesn't exist."""
        with patch("osprey.cli.claude_cmd.get_claude_skills_dir") as mock_dir:
            mock_dir.return_value = tmp_path / ".claude" / "skills"
            result = get_installed_skills()
            assert result == []

    def test_returns_list_of_installed_skills(self, tmp_path):
        """Test that function returns installed skill names."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "migrate").mkdir()
        (skills_dir / "pre-commit").mkdir()

        with patch("osprey.cli.claude_cmd.get_claude_skills_dir") as mock_dir:
            mock_dir.return_value = skills_dir
            result = get_installed_skills()

            assert "migrate" in result
            assert "pre-commit" in result


class TestClaudeGroupCommand:
    """Test the main 'osprey claude' command group."""

    def test_claude_without_subcommand_shows_help(self, cli_runner):
        """Test that 'osprey claude' without subcommand shows help."""
        result = cli_runner.invoke(claude)

        assert result.exit_code == 0
        assert "regen" in result.output.lower()
        assert "status" in result.output.lower()

    def test_claude_help_shows_subcommands(self, cli_runner):
        """Test that help text shows available subcommands."""
        result = cli_runner.invoke(claude, ["--help"])

        assert result.exit_code == 0
        assert "regen" in result.output.lower()
        assert "chat" in result.output.lower()

    def test_claude_help_shows_regen(self, cli_runner):
        """Test that help text shows regen subcommand."""
        result = cli_runner.invoke(claude, ["--help"])

        assert result.exit_code == 0
        assert "regen" in result.output.lower()

    def test_claude_help_shows_chat(self, cli_runner):
        """Test that help text shows chat subcommand."""
        result = cli_runner.invoke(claude, ["--help"])

        assert result.exit_code == 0
        assert "chat" in result.output.lower()


class TestClaudeRegenCommand:
    """Test the 'osprey claude regen' command."""

    def test_regen_command_exists(self, cli_runner):
        """'osprey claude regen --help' returns exit code 0."""
        result = cli_runner.invoke(regen, ["--help"])
        assert result.exit_code == 0
        assert "config.yml" in result.output

    def test_regen_in_project(self, cli_runner, tmp_path):
        """Regen succeeds in a valid project directory."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-cli-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        result = cli_runner.invoke(regen, ["--project", str(project_dir)])
        assert result.exit_code == 0
        assert "regenerated" in result.output.lower() or "up to date" in result.output.lower()

    def test_regen_dry_run_flag(self, cli_runner, tmp_path):
        """--dry-run shows what would change without modifying files."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-run-cli",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        result = cli_runner.invoke(regen, ["--project", str(project_dir), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    def test_regen_outside_project_error(self, cli_runner, tmp_path):
        """Running in non-project directory shows clear error."""
        result = cli_runner.invoke(regen, ["--project", str(tmp_path)])
        assert result.exit_code == 1


class TestClaudeChatCommand:
    """Test the 'osprey claude chat' command."""

    def test_chat_command_exists(self, cli_runner):
        """'osprey claude chat --help' returns exit code 0."""
        result = cli_runner.invoke(chat_claude, ["--help"])
        assert result.exit_code == 0
        assert "claude code" in result.output.lower()

    @patch("osprey.cli.claude_cmd._launch_companion_servers", return_value=[])
    @patch("subprocess.run", return_value=Mock(returncode=0))
    def test_chat_calls_regen_then_exec(self, mock_run, mock_servers, cli_runner, tmp_path):
        """Chat command regenerates then launches claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-cli-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        result = cli_runner.invoke(chat_claude, ["--project", str(project_dir)])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "claude"
        # claude CLI uses cwd as project root (no --project-dir flag)
        assert "--project-dir" not in call_args

    @patch("osprey.cli.claude_cmd._launch_companion_servers", return_value=[])
    @patch("subprocess.run", return_value=Mock(returncode=0))
    def test_chat_passes_resume_flag(self, mock_run, mock_servers, cli_runner, tmp_path):
        """Chat command passes --resume flag to claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-resume-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        cli_runner.invoke(
            chat_claude,
            ["--project", str(project_dir), "--resume", "abc123"],
        )

        call_args = mock_run.call_args[0][0]
        assert "--resume" in call_args
        assert "abc123" in call_args

    @patch("osprey.cli.claude_cmd._launch_companion_servers", return_value=[])
    @patch("subprocess.run", return_value=Mock(returncode=0))
    def test_chat_passes_print_flag(self, mock_run, mock_servers, cli_runner, tmp_path):
        """Chat command passes --print flag to claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-print-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        cli_runner.invoke(
            chat_claude,
            ["--project", str(project_dir), "--print"],
        )

        call_args = mock_run.call_args[0][0]
        assert "--print" in call_args

    @patch(
        "osprey.cli.claude_cmd._launch_companion_servers",
        return_value=[
            ("Artifact gallery", "http://127.0.0.1:8086"),
        ],
    )
    @patch("subprocess.run", return_value=Mock(returncode=0))
    def test_chat_launches_companion_servers(self, mock_run, mock_servers, cli_runner, tmp_path):
        """Chat command launches companion servers before claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-servers-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        result = cli_runner.invoke(chat_claude, ["--project", str(project_dir)])

        assert result.exit_code == 0
        mock_servers.assert_called_once()
        assert "Artifact gallery" in result.output
        assert "http://127.0.0.1:8086" in result.output

    @patch("osprey.cli.claude_cmd._launch_companion_servers", return_value=[])
    @patch("subprocess.run", return_value=Mock(returncode=0))
    def test_chat_always_uses_subprocess_run(self, mock_run, mock_servers, cli_runner, tmp_path):
        """Chat always uses subprocess.run (never os.execvp) for daemon thread safety."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-subprocess-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        with patch("osprey.cli.claude_cmd.os.execvp") as mock_execvp:
            result = cli_runner.invoke(chat_claude, ["--project", str(project_dir)])

        assert result.exit_code == 0
        mock_execvp.assert_not_called()
        mock_run.assert_called_once()
