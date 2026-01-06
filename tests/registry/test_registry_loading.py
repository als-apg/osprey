"""Tests for path-based registry loading and configuration formats.

This test module validates the path-based registry loading mechanism,
covering:
- Loading registries from file paths (absolute/relative)
- Different configuration formats (framework-only, single application)
- sys.path configuration for application module imports
"""

import sys
from pathlib import Path

import pytest

from osprey.registry.manager import RegistryManager


class TestPathBasedLoading:
    """Test loading registries from explicit file paths."""

    def test_load_from_relative_path(self, tmp_path):
        """Test loading registry from relative path."""
        # Create test registry file
        registry_dir = tmp_path / "test_app"
        registry_dir.mkdir(parents=True)
        registry_file = registry_dir / "registry.py"

        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration
)

class TestRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="test_capability",
                    module_path="test_app.capabilities.test",
                    class_name="TestCapability",
                    description="Test capability",
                    provides=[],
                    requires=[]
                )
            ]
        )
"""
        )

        # Change to temp directory to test relative paths
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)

            # Load registry using relative path
            manager = RegistryManager(registry_path="./test_app/registry.py")

            # Should not raise - initialization happens in _build_merged_configuration
            assert manager.registry_path == "./test_app/registry.py"

        finally:
            os.chdir(original_cwd)

    def test_load_from_absolute_path(self, tmp_path):
        """Test loading registry from absolute path."""
        # Create test registry file
        registry_dir = tmp_path / "test_app"
        registry_dir.mkdir(parents=True)
        registry_file = registry_dir / "registry.py"

        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry
)

class TestRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
        )
"""
        )

        # Load registry using absolute path
        absolute_path = str(registry_file.absolute())
        manager = RegistryManager(registry_path=absolute_path)

        # Should not raise
        assert manager.registry_path == absolute_path

    def test_load_registry_with_path_containing_spaces(self, tmp_path):
        """Test loading registry from path with spaces."""
        # Create directory with spaces in name
        registry_dir = tmp_path / "my app" / "test registry"
        registry_dir.mkdir(parents=True)
        registry_file = registry_dir / "registry.py"

        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        # Should handle paths with spaces correctly
        manager = RegistryManager(registry_path=str(registry_file))
        assert manager.registry_path == str(registry_file)


class TestConfigFormats:
    """Test different configuration format support."""

    def test_framework_only_config(self):
        """Test registry with no applications (framework only)."""
        manager = RegistryManager(registry_path=None)

        # Should create framework-only registry
        assert manager.registry_path is None

        # Config should have framework components
        assert len(manager.config.capabilities) > 0
        assert len(manager.config.core_nodes) > 0

        # Should have standard framework capabilities
        cap_names = [c.name for c in manager.config.capabilities]
        assert "memory" in cap_names
        assert "python" in cap_names
        assert "respond" in cap_names

    def test_single_application_path(self, tmp_path):
        """Test with single application registry path."""
        # Create simple registry
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        assert manager.registry_path == str(registry_file)

        # Should include framework capabilities (extend mode)
        cap_names = [c.name for c in manager.config.capabilities]
        assert "memory" in cap_names

    def test_explicit_registry_config_format(self, tmp_path):
        """Test registry using explicit RegistryConfig (not helper)."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration,
    ContextClassRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        # Using helper (recommended approach)
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="app_capability",
                    module_path="app.capabilities.test",
                    class_name="TestCapability",
                    description="Test",
                    provides=["TEST"],
                    requires=[]
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="TEST",
                    module_path="app.context_classes",
                    class_name="TestContext"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should merge with framework
        cap_names = [c.name for c in manager.config.capabilities]
        assert "app_capability" in cap_names
        assert "memory" in cap_names  # Framework capability

        ctx_types = [c.context_type for c in manager.config.context_classes]
        assert "TEST" in ctx_types


class TestSysPathManagement:
    """Test sys.path configuration for application module imports.

    These tests verify that the registry manager correctly configures sys.path
    to enable imports of application modules (context_classes, capabilities, etc.)
    following the industry-standard pattern used by pytest, sphinx, and airflow.
    """

    def test_syspath_configured_for_src_structure(self, tmp_path):
        """Test that src/ directory is added to sys.path for src/app structure."""
        # Create typical generated project structure: ./src/app_name/
        src_dir = tmp_path / "src"
        app_dir = src_dir / "my_app"
        app_dir.mkdir(parents=True)

        # Create registry that references app modules
        registry_file = app_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ContextClassRegistration

class TestProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[
                ContextClassRegistration(
                    context_type="TEST_CONTEXT",
                    module_path="my_app.context_classes",  # Requires src/ on sys.path
                    class_name="TestContext"
                )
            ]
        )
"""
        )

        # Create the referenced module
        context_file = app_dir / "context_classes.py"
        context_file.write_text(
            """
from osprey.context import BaseContext

class TestContext(BaseContext):
    def __init__(self):
        super().__init__("TEST_CONTEXT")
"""
        )

        # Store original sys.path
        original_syspath = sys.path.copy()

        try:
            # Load registry - should automatically add src/ to sys.path
            _ = RegistryManager(registry_path=str(registry_file))

            # Verify src/ was added to sys.path
            src_dir_str = str(src_dir.resolve())
            assert src_dir_str in sys.path, f"Expected {src_dir_str} in sys.path"

            # Verify it was added at the beginning (higher priority)
            assert sys.path.index(src_dir_str) < 10, "src/ should be near beginning of sys.path"

        finally:
            # Clean up sys.path
            sys.path[:] = original_syspath

    def test_syspath_configured_for_flat_structure(self, tmp_path):
        """Test that app directory is added to sys.path for flat structure."""
        # Create flat structure: ./app_name/ (no src/)
        app_dir = tmp_path / "my_app"
        app_dir.mkdir(parents=True)

        registry_file = app_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        original_syspath = sys.path.copy()

        try:
            # Load registry - should add app directory to sys.path
            _ = RegistryManager(registry_path=str(registry_file))

            # Verify app directory was added (since no src/ exists)
            app_dir_str = str(app_dir.resolve())
            assert app_dir_str in sys.path, f"Expected {app_dir_str} in sys.path"

        finally:
            sys.path[:] = original_syspath

    def test_syspath_not_duplicated(self, tmp_path):
        """Test that sys.path entries aren't duplicated on repeated loads."""
        src_dir = tmp_path / "src"
        app_dir = src_dir / "my_app"
        app_dir.mkdir(parents=True)

        registry_file = app_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        original_syspath = sys.path.copy()

        try:
            # Load registry multiple times
            _ = RegistryManager(registry_path=str(registry_file))
            initial_syspath_len = len(sys.path)

            _ = RegistryManager(registry_path=str(registry_file))

            # sys.path should not grow (deduplication working)
            assert len(sys.path) == initial_syspath_len, "sys.path should not have duplicates"

            # Verify only one occurrence of src_dir
            src_dir_str = str(src_dir.resolve())
            count = sys.path.count(src_dir_str)
            assert count == 1, f"Expected 1 occurrence of {src_dir_str}, found {count}"

        finally:
            sys.path[:] = original_syspath

    def test_application_modules_can_import_after_syspath_setup(self, tmp_path, monkeypatch):
        """Test that application modules can actually be imported after sys.path setup."""
        # Create realistic project structure
        src_dir = tmp_path / "src"
        app_dir = src_dir / "weather_app"
        app_dir.mkdir(parents=True)

        # Create a minimal config.yml to avoid config loading errors
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            """
project_root: .
models:
  orchestrator:
    provider: openai
    model_id: gpt-4
"""
        )

        # Set CONFIG_FILE environment variable
        monkeypatch.setenv("CONFIG_FILE", str(config_file))

        # Create context_classes module
        context_file = app_dir / "context_classes.py"
        context_file.write_text(
            """
from osprey.context.base import CapabilityContext

class WeatherContext(CapabilityContext):
    def __init__(self):
        super().__init__("WEATHER_DATA")
        self.temperature = None
"""
        )

        # Create registry that references it
        registry_file = app_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ContextClassRegistration

class WeatherProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[
                ContextClassRegistration(
                    context_type="WEATHER_DATA",
                    module_path="weather_app.context_classes",
                    class_name="WeatherContext"
                )
            ]
        )
"""
        )

        original_syspath = sys.path.copy()

        try:
            # Load registry - sys.path should be configured
            manager = RegistryManager(registry_path=str(registry_file))
            manager.initialize()  # Initialize to load context classes

            # Verify we can now import the application module
            import weather_app.context_classes

            assert hasattr(weather_app.context_classes, "WeatherContext")

            # Verify the context class was loaded correctly
            assert "WEATHER_DATA" in manager._registries["contexts"]

        finally:
            # Clean up
            sys.path[:] = original_syspath
            if "weather_app" in sys.modules:
                del sys.modules["weather_app"]
            if "weather_app.context_classes" in sys.modules:
                del sys.modules["weather_app.context_classes"]

    def test_syspath_detection_with_explicit_src_dir(self, tmp_path):
        """Test Pattern 2: Registry not in src/ but src/ exists."""
        # Create structure: ./config/registry.py and ./src/ exists
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True)

        registry_file = config_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        original_syspath = sys.path.copy()

        try:
            _ = RegistryManager(registry_path=str(registry_file))

            # Should detect src/ directory and add it
            src_dir_str = str(src_dir.resolve())
            assert src_dir_str in sys.path, f"Expected {src_dir_str} in sys.path (Pattern 2)"

        finally:
            sys.path[:] = original_syspath


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
