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

    def test_create_project_minimal_compact(self, tmp_path):
        """Test creating project with minimal template (compact style)."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="test-project",
            output_dir=tmp_path,
            template_name="minimal",
            registry_style="compact",
        )

        # Verify structure
        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / "pyproject.toml").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "src").exists()
        assert (project_dir / "src" / "test_project").exists()
        assert (project_dir / "src" / "test_project" / "registry.py").exists()

        # Verify services copied
        assert (project_dir / "services").exists()
        assert (project_dir / "services" / "docker-compose.yml.j2").exists()
        assert (project_dir / "services" / "jupyter").exists()

    def test_create_project_hello_world(self, tmp_path):
        """Test creating project with hello_world_weather template."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="weather-app", output_dir=tmp_path, template_name="hello_world_weather"
        )

        # Verify weather-specific structure
        assert (
            project_dir / "src" / "weather_app" / "capabilities" / "current_weather.py"
        ).exists()
        assert (project_dir / "src" / "weather_app" / "context_classes.py").exists()
        assert (project_dir / "src" / "weather_app" / "mock_weather_api.py").exists()

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


class TestGeneratedRegistries:
    """Test that generated registries use helper functions correctly."""

    def test_minimal_uses_helper(self, tmp_path):
        """Test that minimal template generates registry with helper."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        registry_file = project_dir / "src" / "test_app" / "registry.py"
        content = registry_file.read_text()

        # Should import helper
        assert "from osprey.registry import" in content
        assert "extend_framework_registry" in content

        # Should use helper
        assert "return extend_framework_registry(" in content

        # Should have capabilities and context_classes parameters
        assert "capabilities=[" in content
        assert "context_classes=[" in content

    def test_hello_world_uses_helper(self, tmp_path):
        """Test that hello_world_weather uses helper."""
        manager = TemplateManager()
        project_dir = manager.create_project("weather", tmp_path, "hello_world_weather")

        registry_file = project_dir / "src" / "weather" / "registry.py"
        content = registry_file.read_text()

        # Should use helper
        assert "extend_framework_registry" in content
        assert "return extend_framework_registry(" in content

        # Should have weather capability
        assert '"current_weather"' in content
        assert "CurrentWeatherCapability" in content


class TestCLIIntegration:
    """Test CLI command integration with templates."""

    def test_init_command_basic(self, tmp_path):
        """Test basic init command."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Project created successfully" in result.output
        # Match registry style (may have ANSI color codes)
        assert "Registry style:" in result.output
        assert "extend" in result.output

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

    def test_init_command_with_registry_style(self, tmp_path):
        """Test init command with registry style option."""
        runner = CliRunner()

        # Test extend style
        result = runner.invoke(
            init, ["extend-app", "--registry-style", "extend", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        # Match registry style (may have ANSI color codes)
        assert "Registry style:" in result.output
        assert "extend" in result.output

        # Test standalone style
        result = runner.invoke(
            init,
            ["standalone-app", "--registry-style", "standalone", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        assert "Registry style:" in result.output
        assert "standalone" in result.output

    def test_init_command_shows_next_steps(self, tmp_path):
        """Test that init command shows helpful next steps."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        # Should show next steps
        assert "Next steps:" in result.output
        assert "cd test-project" in result.output
        assert "osprey deploy up" in result.output
        assert "osprey chat" in result.output

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

    def test_generator_config_excluded_from_app_package(self, tmp_path):
        """Test that generator config is at project root, not in src/."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="placement-test",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "basic"},
        )

        # Should be at project root
        assert (project_dir / "basic_generator_config.yml").exists()

        # Should NOT be in src/ directory
        src_dir = project_dir / "src" / "placement_test"
        assert not (src_dir / "basic_generator_config.yml").exists()
        assert not (src_dir / "basic_generator_config.yml.j2").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
