"""Tests for template generation system.

Tests the TemplateManager class and template rendering,
including validation that generated projects use the new
registry helper pattern correctly.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.templates import TemplateManager


class TestTemplateManager:
    """Test TemplateManager class."""

    def test_template_manager_initialization(self):
        """Test that TemplateManager initializes correctly."""
        manager = TemplateManager()

        assert manager.template_root is not None
        assert manager.template_root.exists()
        assert manager.jinja_env is not None

    def test_list_app_templates(self):
        """Test listing available application templates."""
        manager = TemplateManager()
        templates = manager.list_app_templates()

        # Should have at least the main templates
        assert "minimal" in templates
        assert "hello_world_weather" in templates
        assert "control_assistant" in templates
        assert len(templates) >= 3

    def test_create_project_minimal(self, tmp_path):
        """Test creating project with minimal template."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="test-project",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Verify structure (Claude Code mode — no src/ or pyproject.toml)
        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "_agent_data").exists()

        # Claude Code integration
        assert (project_dir / "CLAUDE.md").exists()
        assert (project_dir / ".mcp.json").exists()

    def test_create_project_hello_world(self, tmp_path):
        """Test creating project with hello_world_weather template."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="weather-app", output_dir=tmp_path, template_name="hello_world_weather"
        )

        # Verify basic structure (Claude Code mode — no src/ package)
        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()
        assert (project_dir / "CLAUDE.md").exists()

    def test_duplicate_project_raises_error(self, tmp_path):
        """Test that creating duplicate project raises error."""
        manager = TemplateManager()

        # Create first project
        manager.create_project("test-project", tmp_path, "minimal")

        # Try to create again
        with pytest.raises(ValueError, match="already exists"):
            manager.create_project("test-project", tmp_path, "minimal")

    def test_invalid_template_raises_error(self, tmp_path):
        """Test that invalid template name raises error."""
        manager = TemplateManager()

        with pytest.raises(ValueError, match="not found"):
            manager.create_project("test-project", tmp_path, "nonexistent_template")


class TestCLIIntegration:
    """Test CLI command integration with templates."""

    def test_init_command_basic(self, tmp_path):
        """Test basic init command."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Project created successfully" in result.output
        # New init shows Mode: Claude Code
        assert "Mode: Claude Code" in result.output

    def test_init_command_with_template(self, tmp_path):
        """Test init command with specific template."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            ["weather-app", "--template", "hello_world_weather", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        # Match template name (may have ANSI color codes)
        assert "Using template:" in result.output
        assert "hello_world_weather" in result.output

    def test_init_command_shows_next_steps(self, tmp_path):
        """Test that init command shows helpful next steps."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        # Should show next steps
        assert "Next steps:" in result.output
        assert "cd test-project" in result.output
        assert "claude" in result.output

    def test_init_command_with_channel_finder_mode(self, tmp_path):
        """Test init command with --channel-finder-mode option."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "cf-app",
                "--template",
                "control_assistant",
                "--channel-finder-mode",
                "in_context",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        # Check manifest has the option
        import json

        manifest_path = tmp_path / "cf-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["init_args"]["channel_finder_mode"] == "in_context"
        assert "--channel-finder-mode in_context" in manifest["reproducible_command"]

    def test_init_command_with_code_generator(self, tmp_path):
        """Test init command with --code-generator option."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "gen-app",
                "--template",
                "control_assistant",
                "--code-generator",
                "basic",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        # Check manifest has the option
        import json

        manifest_path = tmp_path / "gen-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["init_args"]["code_generator"] == "basic"
        assert "--code-generator basic" in manifest["reproducible_command"]

    def test_init_command_reproducible_command_complete(self, tmp_path):
        """Test that reproducible_command includes all options."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "full-app",
                "--template",
                "control_assistant",
                "--provider",
                "cborg",
                "--model",
                "anthropic/claude-haiku",
                "--channel-finder-mode",
                "hierarchical",
                "--code-generator",
                "claude_code",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        import json

        manifest_path = tmp_path / "full-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())

        cmd = manifest["reproducible_command"]
        assert "--template control_assistant" in cmd
        assert "--provider cborg" in cmd
        assert "--model anthropic/claude-haiku" in cmd
        assert "--channel-finder-mode hierarchical" in cmd
        assert "--code-generator claude_code" in cmd


class TestGeneratorConfigRendering:
    """Test code generator config file rendering."""

    def test_claude_code_generator_config_rendered(self, tmp_path):
        """Test that claude_code generator creates config file."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="claude-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "claude_code"},
        )

        config_file = project_dir / "claude_generator_config.yml"

        # Should exist
        assert config_file.exists(), "claude_generator_config.yml should be created"

        # Should have expected content
        content = config_file.read_text()
        assert "profile:" in content or "phases:" in content
        # Should have system prompt extensions section
        assert "system_prompt_extensions" in content

    def test_basic_generator_config_rendered(self, tmp_path):
        """Test that basic generator creates config file."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="basic-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "basic"},
        )

        config_file = project_dir / "basic_generator_config.yml"

        # Should exist
        assert config_file.exists(), "basic_generator_config.yml should be created"

        # Should have expected content
        content = config_file.read_text()
        assert "system_role:" in content
        assert "core_requirements:" in content
        assert "system_prompt_extensions:" in content

    def test_no_generator_config_for_default(self, tmp_path):
        """Test that no generator config is created when generator not specified."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="default-app",
            output_dir=tmp_path,
            template_name="minimal",
            # No code_generator in context
        )

        # Neither config should exist
        assert not (project_dir / "claude_generator_config.yml").exists()
        assert not (project_dir / "basic_generator_config.yml").exists()

    def test_generator_configs_mutually_exclusive(self, tmp_path):
        """Test that only one generator config is created at a time."""
        manager = TemplateManager()

        # Create with claude_code
        project_dir = manager.create_project(
            project_name="exclusive-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "claude_code"},
        )

        # Only claude config should exist
        assert (project_dir / "claude_generator_config.yml").exists()
        assert not (project_dir / "basic_generator_config.yml").exists()

    def test_generator_config_at_project_root(self, tmp_path):
        """Test that generator config is at project root."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="placement-test",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "basic"},
        )

        # Should be at project root
        assert (project_dir / "basic_generator_config.yml").exists()


class TestGitIsolation:
    """Test that osprey init creates a self-contained git repo."""

    def test_init_creates_git_repo(self, tmp_path):
        """Test that osprey init creates a .git directory."""
        runner = CliRunner()

        result = runner.invoke(init, ["git-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "git-test"
        assert (project_dir / ".git").exists(), ".git directory should be created"

    def test_init_initial_commit(self, tmp_path):
        """Test that osprey init creates an initial commit with expected files."""
        import subprocess

        runner = CliRunner()

        result = runner.invoke(init, ["commit-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "commit-test"

        # Verify there is exactly one commit
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert log.returncode == 0
        commits = log.stdout.strip().splitlines()
        assert len(commits) == 1
        assert "Initial project from osprey init" in commits[0]

        # Verify key files are tracked
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        tracked_files = tracked.stdout.strip().splitlines()
        assert "config.yml" in tracked_files
        assert "CLAUDE.md" in tracked_files
        assert ".mcp.json" in tracked_files

    def test_gitignore_tracks_claude_dir(self, tmp_path):
        """Test that .claude/ is NOT in .gitignore (should be tracked)."""
        runner = CliRunner()

        result = runner.invoke(init, ["track-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "track-test"

        gitignore_content = (project_dir / ".gitignore").read_text()
        # .claude/ should NOT appear as a top-level ignore pattern
        lines = [line.strip() for line in gitignore_content.splitlines()]
        assert ".claude/" not in lines, ".claude/ should not be ignored"
        assert "CLAUDE.md" not in lines, "CLAUDE.md should not be ignored"
        assert ".mcp.json" not in lines, ".mcp.json should not be ignored"

    def test_gitignore_ignores_local_settings(self, tmp_path):
        """Test that personal local settings are ignored."""
        runner = CliRunner()

        result = runner.invoke(init, ["local-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "local-test"

        gitignore_content = (project_dir / ".gitignore").read_text()
        assert ".claude/settings.local.json" in gitignore_content
        assert "CLAUDE.local.md" in gitignore_content

    def test_init_output_mentions_git(self, tmp_path):
        """Test that init output confirms git initialization."""
        runner = CliRunner()

        result = runner.invoke(init, ["msg-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Initialized git repository" in result.output
        assert "standalone git repo" in result.output

    def test_claude_dir_in_initial_commit(self, tmp_path):
        """Test that .claude/ files are included in the initial commit."""
        import subprocess

        runner = CliRunner()

        result = runner.invoke(init, ["claude-track-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "claude-track-test"

        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        tracked_files = tracked.stdout.strip().splitlines()
        # At least some .claude/ files should be tracked
        claude_files = [f for f in tracked_files if f.startswith(".claude/")]
        assert len(claude_files) > 0, ".claude/ files should be tracked in git"

    def test_init_inside_existing_repo_warns(self, tmp_path):
        """Test that creating a project inside an existing git repo shows a warning."""
        import subprocess

        # Create a parent git repo
        subprocess.run(
            ["git", "init"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "parent init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={
                **__import__("os").environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@test",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@test",
            },
        )

        runner = CliRunner()
        result = runner.invoke(init, ["nested-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        # Should still create the git repo (for isolation)
        project_dir = tmp_path / "nested-test"
        assert (project_dir / ".git").exists()
        # Should warn about nested repo
        assert "nested git repo" in result.output

    def test_init_inside_existing_repo_still_isolates(self, tmp_path):
        """Test that nested repo has its own independent git root."""
        import subprocess
        from pathlib import Path

        # Create a parent git repo
        subprocess.run(
            ["git", "init"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "parent init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={
                **__import__("os").environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@test",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@test",
            },
        )

        runner = CliRunner()
        result = runner.invoke(init, ["isolated-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "isolated-test"

        # The nested project's git root should be itself, not the parent
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert git_root.returncode == 0
        assert Path(git_root.stdout.strip()).resolve() == project_dir.resolve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
