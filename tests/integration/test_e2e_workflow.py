"""End-to-end integration test for the pip-installable framework architecture.

This test validates the complete workflow from project creation to runtime,
ensuring all components work together correctly.
"""

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.registry.manager import RegistryManager
from osprey.utils.config import ConfigBuilder


class TestE2EWorkflow:
    """Test complete project lifecycle."""

    def test_complete_project_lifecycle(self, tmp_path):
        """
        Test the complete workflow: framework init → config loads → registry discovers.

        This is the most critical test - validates entire pip-installable architecture.

        Workflow:
        1. Create project with `framework init`
        2. Verify project structure
        3. Verify config loads correctly
        4. Verify registry discovers application
        5. Verify generated code is valid Python
        6. Verify imports work
        """
        runner = CliRunner()

        # ==========================================
        # STEP 1: Create project with framework init
        # ==========================================
        result = runner.invoke(
            init, ["test-project", "--template", "minimal", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0, f"Init command failed: {result.output}"
        assert "Project created successfully" in result.output

        project_dir = tmp_path / "test-project"
        assert project_dir.exists(), "Project directory should exist"

        # ==========================================
        # STEP 2: Verify project structure
        # ==========================================
        # Root files
        assert (project_dir / "config.yml").exists(), "config.yml should exist"
        assert (project_dir / "pyproject.toml").exists(), "pyproject.toml should exist"
        assert (project_dir / "README.md").exists(), "README.md should exist"
        assert (project_dir / ".env.example").exists(), ".env.example should exist"
        assert (project_dir / ".gitignore").exists(), ".gitignore should exist"

        # Source directory structure
        src_dir = project_dir / "src"
        assert src_dir.exists(), "src/ directory should exist"

        app_dir = src_dir / "test_project"
        assert app_dir.exists(), "Application directory should exist"
        assert (app_dir / "__init__.py").exists(), "Application __init__.py should exist"
        assert (app_dir / "registry.py").exists(), "registry.py should exist"
        assert (app_dir / "context_classes.py").exists(), "context_classes.py should exist"
        assert (app_dir / "capabilities").is_dir(), "capabilities/ directory should exist"

        # Services directory
        services_dir = project_dir / "services"
        assert services_dir.exists(), "services/ directory should exist"
        assert (services_dir / "jupyter").is_dir(), "jupyter service should exist"
        assert (services_dir / "open-webui").is_dir(), "open-webui service should exist"
        assert (services_dir / "pipelines").is_dir(), "pipelines service should exist"

        # ==========================================
        # STEP 3: Verify config loads correctly
        # ==========================================
        config_path = project_dir / "config.yml"

        # Config file should be valid YAML and load without errors
        config_builder = ConfigBuilder(str(config_path))

        assert config_builder.raw_config is not None
        assert isinstance(config_builder.raw_config, dict)

        # Verify key configuration elements exist
        assert "project_root" in config_builder.raw_config
        assert "registry_path" in config_builder.raw_config
        assert "models" in config_builder.raw_config
        assert "services" in config_builder.raw_config

        # Verify registry path is correctly set
        registry_path = config_builder.get("registry_path")
        assert registry_path is not None
        assert "test_project" in registry_path
        assert registry_path.endswith("registry.py")

        # ==========================================
        # STEP 4: Verify registry discovers application
        # ==========================================
        # Add src directory to Python path for imports
        sys.path.insert(0, str(src_dir))

        try:
            # Resolve registry path relative to project directory
            if registry_path.startswith("./"):
                abs_registry_path = project_dir / registry_path[2:]
            else:
                abs_registry_path = project_dir / registry_path

            # Load registry from generated project
            registry_manager = RegistryManager(registry_path=str(abs_registry_path))

            # Registry should load without errors
            assert registry_manager.config is not None

            # Verify registry has the expected structure
            assert hasattr(registry_manager.config, "capabilities")
            assert hasattr(registry_manager.config, "context_classes")
            assert hasattr(registry_manager.config, "core_nodes")

            # For minimal template, capabilities list may be empty but should be a list
            assert isinstance(registry_manager.config.capabilities, list)
            assert isinstance(registry_manager.config.context_classes, list)

        finally:
            # Clean up sys.path
            sys.path.remove(str(src_dir))

            # Clean up imported modules
            modules_to_remove = [key for key in sys.modules.keys() if "test_project" in key]
            for module in modules_to_remove:
                del sys.modules[module]

        # ==========================================
        # STEP 5: Verify generated code is valid Python
        # ==========================================
        # Test that registry.py is valid Python
        registry_file = app_dir / "registry.py"
        registry_code = registry_file.read_text()

        # Should contain key elements
        assert "RegistryConfigProvider" in registry_code
        assert "RegistryConfig" in registry_code
        assert "get_registry_config" in registry_code

        # Should use helper function (compact style)
        assert "extend_framework_registry" in registry_code

        # Verify it's valid Python by compiling
        try:
            compile(registry_code, str(registry_file), "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated registry.py has syntax error: {e}")

        # ==========================================
        # STEP 6: Verify imports work
        # ==========================================
        # Test that context_classes.py is valid Python
        context_file = app_dir / "context_classes.py"
        context_code = context_file.read_text()

        # Verify it's valid Python
        try:
            compile(context_code, str(context_file), "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated context_classes.py has syntax error: {e}")

        # ==========================================
        # SUCCESS: All checks passed!
        # ==========================================
        # If we got here, the entire pip-installable architecture is working:
        # ✓ Project creation works
        # ✓ Generated files are valid
        # ✓ Configuration loads correctly
        # ✓ Registry discovery works
        # ✓ Module imports work

    def test_generated_config_is_complete(self, tmp_path):
        """Test that generated config file contains all necessary sections."""
        runner = CliRunner()

        result = runner.invoke(
            init, ["config-test-project", "--template", "minimal", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0

        config_file = tmp_path / "config-test-project" / "config.yml"
        config_content = config_file.read_text()

        # Verify all major sections are present
        required_sections = [
            "project_root",
            "registry_path",
            "models",
            "services",
            "approval",
            "execution_control",
            "file_paths",
            "execution",
            "pipeline",
            "development",
            "logging",
            "api",
        ]

        for section in required_sections:
            assert section in config_content, f"Config should contain '{section}' section"

    def test_hello_world_template_generates_correctly(self, tmp_path):
        """Test that hello_world_weather template generates with all its components."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            ["weather-app", "--template", "hello_world_weather", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0

        project_dir = tmp_path / "weather-app"
        app_dir = project_dir / "src" / "weather_app"

        # Verify hello_world_weather specific files
        assert (app_dir / "registry.py").exists()
        assert (app_dir / "context_classes.py").exists()
        assert (app_dir / "capabilities" / "current_weather.py").exists()
        assert (app_dir / "mock_weather_api.py").exists()

        # Verify registry uses helper
        registry_code = (app_dir / "registry.py").read_text()
        assert "extend_framework_registry" in registry_code

        # Verify hello_world_weather does NOT include services (no containers needed)
        assert not (project_dir / "services").exists()

        # Verify config does NOT include production-only sections
        config_content = (project_dir / "config.yml").read_text()

        # Check for deployed services and services configuration
        assert "deployed_services:" not in config_content  # No deployed services list
        assert "jupyter:" not in config_content  # No Jupyter service
        assert "open_webui:" not in config_content  # No OpenWebUI service
        assert "containers:" not in config_content  # No container configurations

        # Check for production-only sections - using unique strings from those sections
        assert "container_runtime:" not in config_content  # No container runtime config
        assert (
            "global_mode:" not in config_content
        )  # No approval configuration (unique to approval section)
        assert "execution_control:" not in config_content  # No execution control
        assert (
            "execution_method:" not in config_content
        )  # No execution infrastructure (unique to execution section)
        assert "max_generation_retries:" not in config_content  # No python executor section
        assert "gateways:" not in config_content  # No EPICS gateway configuration
        assert "writes_enabled:" not in config_content  # No EPICS writes configuration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
