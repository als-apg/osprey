"""Tests for template generation system.

Tests the TemplateManager class and template rendering,
including validation that generated projects use the new
registry helper pattern correctly.
"""

from pathlib import Path

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
