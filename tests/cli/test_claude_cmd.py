"""Tests for claude CLI command.

This test module verifies that the Claude Code skill installation commands
work correctly, including installing skills and listing installed skills.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.claude_cmd import (
    chat_claude,
    claude,
    get_claude_skills_dir,
    get_installed_skills,
    install_skill,
    list_skills,
    regen,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_assist_path(tmp_path):
    """Create a mock assist directory with tasks and integrations."""
    # Create tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Create migrate task
    migrate_dir = tasks_dir / "migrate"
    migrate_dir.mkdir()
    (migrate_dir / "instructions.md").write_text(
        "# Migration Assistant\n\nUpgrade downstream projects.\n"
    )

    # Create pre-commit task
    precommit_dir = tasks_dir / "pre-commit"
    precommit_dir.mkdir()
    (precommit_dir / "instructions.md").write_text(
        "# Pre-Commit Validation\n\nValidate code before commits.\n"
    )

    # Create testing-workflow task (no Claude integration)
    testing_dir = tasks_dir / "testing-workflow"
    testing_dir.mkdir()
    (testing_dir / "instructions.md").write_text("# Testing Workflow\n\nTesting guide.\n")

    # Create integrations directory
    integrations_dir = tmp_path / "integrations"
    integrations_dir.mkdir()

    # Create claude_code integration for migrate and pre-commit
    claude_code_dir = integrations_dir / "claude_code"
    claude_code_dir.mkdir()

    migrate_skill_dir = claude_code_dir / "migrate"
    migrate_skill_dir.mkdir()
    (migrate_skill_dir / "SKILL.md").write_text(
        "---\nname: osprey-migrate\n---\n\n# Migration\n\nFollow instructions.md\n"
    )

    precommit_skill_dir = claude_code_dir / "pre-commit"
    precommit_skill_dir.mkdir()
    (precommit_skill_dir / "SKILL.md").write_text(
        "---\nname: osprey-pre-commit\n---\n\n# Pre-Commit\n\nRun checks.\n"
    )

    return tmp_path


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


class TestClaudeInstallCommand:
    """Test the 'osprey claude install' command."""

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_creates_skill_directory(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install command creates the skill directory."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(install_skill, ["migrate"])

            assert result.exit_code == 0
            skill_dir = Path(".claude") / "skills" / "migrate"
            assert skill_dir.exists()

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_copies_skill_file(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install command copies SKILL.md file."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(install_skill, ["migrate"])

            assert result.exit_code == 0
            skill_file = Path(".claude") / "skills" / "migrate" / "SKILL.md"
            assert skill_file.exists()
            content = skill_file.read_text()
            assert "osprey-migrate" in content

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_copies_instructions(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install command copies instructions.md."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(install_skill, ["migrate"])

            assert result.exit_code == 0
            instructions_file = Path(".claude") / "skills" / "migrate" / "instructions.md"
            assert instructions_file.exists()

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_warns_when_exists(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install warns when skill already exists."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing installation
            skill_dir = Path(".claude") / "skills" / "migrate"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("existing content")

            result = cli_runner.invoke(install_skill, ["migrate"])

            assert result.exit_code == 0
            assert "already installed" in result.output
            assert "--force" in result.output

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_force_overwrites(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install --force overwrites existing installation."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing installation
            skill_dir = Path(".claude") / "skills" / "migrate"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("old content")

            result = cli_runner.invoke(install_skill, ["migrate", "--force"])

            assert result.exit_code == 0
            assert "Installed" in result.output
            # Content should be updated
            new_content = (skill_dir / "SKILL.md").read_text()
            assert "osprey-migrate" in new_content

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_handles_unknown_task(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path
    ):
        """Test that install command handles unknown task gracefully."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        result = cli_runner.invoke(install_skill, ["nonexistent-task"])

        assert result.exit_code == 0  # Doesn't crash
        assert "not found" in result.output.lower()

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_handles_task_without_integration(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install warns when task has no Claude integration."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # testing-workflow has no claude_code integration
            result = cli_runner.invoke(install_skill, ["testing-workflow"])

            assert result.exit_code == 0
            assert "No Claude Code skill available" in result.output

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    def test_install_shows_usage_hint(
        self, mock_tasks_root, mock_int_root, cli_runner, mock_assist_path, tmp_path
    ):
        """Test that install shows usage hint after success."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(install_skill, ["pre-commit"])

            assert result.exit_code == 0
            assert "Usage" in result.output
            assert "Ask Claude" in result.output


class TestClaudeListCommand:
    """Test the 'osprey claude list' command."""

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    @patch("osprey.cli.claude_cmd.get_claude_skills_dir")
    def test_list_shows_installed_skills(
        self,
        mock_skills_dir,
        mock_tasks_root,
        mock_int_root,
        cli_runner,
        mock_assist_path,
        tmp_path,
    ):
        """Test that list shows installed skills."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        # Create installed skill
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "migrate").mkdir()
        mock_skills_dir.return_value = skills_dir

        result = cli_runner.invoke(list_skills)

        assert result.exit_code == 0
        assert "migrate" in result.output
        assert "✓" in result.output

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    @patch("osprey.cli.claude_cmd.get_claude_skills_dir")
    def test_list_shows_available_to_install(
        self,
        mock_skills_dir,
        mock_tasks_root,
        mock_int_root,
        cli_runner,
        mock_assist_path,
        tmp_path,
    ):
        """Test that list shows skills available to install."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        # No skills installed
        skills_dir = tmp_path / ".claude" / "skills"
        mock_skills_dir.return_value = skills_dir

        result = cli_runner.invoke(list_skills)

        assert result.exit_code == 0
        assert "Available to install" in result.output
        assert "migrate" in result.output
        assert "pre-commit" in result.output

    @patch("osprey.cli.claude_cmd.get_integrations_root")
    @patch("osprey.cli.claude_cmd.get_tasks_root")
    @patch("osprey.cli.claude_cmd.get_claude_skills_dir")
    def test_list_shows_tasks_without_integration(
        self,
        mock_skills_dir,
        mock_tasks_root,
        mock_int_root,
        cli_runner,
        mock_assist_path,
        tmp_path,
    ):
        """Test that list shows tasks without Claude integration."""
        mock_tasks_root.return_value = mock_assist_path / "tasks"
        mock_int_root.return_value = mock_assist_path / "integrations"

        skills_dir = tmp_path / ".claude" / "skills"
        mock_skills_dir.return_value = skills_dir

        result = cli_runner.invoke(list_skills)

        assert result.exit_code == 0
        # testing-workflow has no integration
        assert "testing-workflow" in result.output


class TestClaudeGroupCommand:
    """Test the main 'osprey claude' command group."""

    def test_claude_without_subcommand_shows_help(self, cli_runner):
        """Test that 'osprey claude' without subcommand shows help."""
        result = cli_runner.invoke(claude)

        assert result.exit_code == 0
        assert "install" in result.output.lower()
        assert "list" in result.output.lower()

    def test_claude_help_shows_subcommands(self, cli_runner):
        """Test that help text shows available subcommands."""
        result = cli_runner.invoke(claude, ["--help"])

        assert result.exit_code == 0
        assert "install" in result.output.lower()
        assert "list" in result.output.lower()

    def test_claude_help_mentions_tasks(self, cli_runner):
        """Test that help text mentions how to browse tasks."""
        result = cli_runner.invoke(claude, ["--help"])

        assert result.exit_code == 0
        assert "tasks" in result.output.lower()

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
            template_name="control_assistant",
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
            template_name="control_assistant",
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

    @patch("osprey.cli.claude_cmd.os.execvp")
    def test_chat_calls_regen_then_exec(self, mock_execvp, cli_runner, tmp_path):
        """Chat command regenerates then launches claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-cli-test",
            output_dir=tmp_path,
            template_name="control_assistant",
        )

        result = cli_runner.invoke(chat_claude, ["--project", str(project_dir)])

        assert result.exit_code == 0
        mock_execvp.assert_called_once()
        call_args = mock_execvp.call_args
        assert call_args[0][0] == "claude"
        assert "--project-dir" in call_args[0][1]

    @patch("osprey.cli.claude_cmd.os.execvp")
    def test_chat_passes_resume_flag(self, mock_execvp, cli_runner, tmp_path):
        """Chat command passes --resume flag to claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-resume-test",
            output_dir=tmp_path,
            template_name="control_assistant",
        )

        cli_runner.invoke(
            chat_claude,
            ["--project", str(project_dir), "--resume", "abc123"],
        )

        call_args = mock_execvp.call_args[0][1]
        assert "--resume" in call_args
        assert "abc123" in call_args

    @patch("osprey.cli.claude_cmd.os.execvp")
    def test_chat_passes_print_flag(self, mock_execvp, cli_runner, tmp_path):
        """Chat command passes --print flag to claude CLI."""
        from osprey.cli.templates.manager import TemplateManager

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="chat-print-test",
            output_dir=tmp_path,
            template_name="control_assistant",
        )

        cli_runner.invoke(
            chat_claude,
            ["--project", str(project_dir), "--print"],
        )

        call_args = mock_execvp.call_args[0][1]
        assert "--print" in call_args
