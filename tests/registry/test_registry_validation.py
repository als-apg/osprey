"""Tests for registry validation and error handling.

This test module validates error handling and validation in the registry system:
- Missing/invalid registry files
- Invalid Python syntax
- Missing/multiple providers
- Configuration validation
- Helpful error messages
"""

import pytest

from osprey.registry.manager import RegistryError, RegistryManager


class TestFileLoadingErrors:
    """Test error handling for file loading issues."""

    def test_load_missing_file_raises_error(self, tmp_path):
        """Test that missing registry file raises clear error."""
        # Error is raised during __init__ when config is built
        with pytest.raises(RegistryError, match="Registry file not found"):
            _ = RegistryManager(registry_path="./nonexistent/registry.py")

    def test_load_invalid_python_raises_error(self, tmp_path):
        """Test that invalid Python file raises clear error."""
        # Create invalid Python file
        registry_file = tmp_path / "bad_registry.py"
        registry_file.write_text("this is not valid python syntax{{{")

        # Error is raised during __init__
        with pytest.raises(RegistryError, match="Failed to load Python module"):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_load_directory_instead_of_file_raises_error(self, tmp_path):
        """Test that passing a directory instead of file raises error."""
        # Create a directory
        registry_dir = tmp_path / "app"
        registry_dir.mkdir(parents=True)

        # Error is raised during __init__
        with pytest.raises(RegistryError, match="not a file"):
            _ = RegistryManager(registry_path=str(registry_dir))

    def test_missing_file_error_message_includes_path(self, tmp_path):
        """Test error message includes the problematic path."""
        missing_path = "./does_not_exist.py"

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=missing_path)

        error_msg = str(exc_info.value)
        assert "Registry file not found" in error_msg
        assert "does_not_exist.py" in error_msg

    def test_load_file_with_import_errors(self, tmp_path):
        """Test that files with import errors provide clear messages."""
        registry_file = tmp_path / "bad_imports.py"
        registry_file.write_text(
            """
from nonexistent_module import something  # This will fail

from osprey.registry import RegistryConfigProvider, RegistryConfig

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(capabilities=[], context_classes=[])
"""
        )

        # Should raise during module loading
        with pytest.raises(RegistryError, match="Failed to load"):
            _ = RegistryManager(registry_path=str(registry_file))


class TestProviderValidationErrors:
    """Test validation of RegistryConfigProvider implementations."""

    def test_no_provider_raises_error(self, tmp_path):
        """Test that registry without RegistryConfigProvider raises error."""
        # Create valid Python file but no provider
        registry_file = tmp_path / "no_provider.py"
        registry_file.write_text(
            """
# Valid Python but no RegistryConfigProvider
def some_function():
    pass

class SomeClass:
    pass
"""
        )

        # Error is raised during __init__
        with pytest.raises(RegistryError, match="No RegistryConfigProvider implementation found"):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_multiple_providers_raises_error(self, tmp_path):
        """Test that registry with multiple providers raises error."""
        # Create file with two providers
        registry_file = tmp_path / "multi_provider.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class Provider1(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(capabilities=[], context_classes=[])

class Provider2(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(capabilities=[], context_classes=[])
"""
        )

        # Error is raised during __init__
        with pytest.raises(RegistryError, match="Multiple RegistryConfigProvider"):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_no_provider_error_includes_helpful_example(self, tmp_path):
        """Test error message when no provider found includes example."""
        registry_file = tmp_path / "registry.py"
        registry_file.write_text("# No provider here\n")

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=str(registry_file))

        error_msg = str(exc_info.value)
        assert "No RegistryConfigProvider implementation found" in error_msg
        assert "Example:" in error_msg  # Should include helpful example

    def test_provider_with_syntax_error_in_method(self, tmp_path):
        """Test that syntax errors in provider methods are caught."""
        registry_file = tmp_path / "bad_method.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        # Syntax error in method
        return RegistryConfig(
            capabilities=[],
            context_classes=[]
        ]  # Missing closing paren
"""
        )

        with pytest.raises(RegistryError):
            _ = RegistryManager(registry_path=str(registry_file))


class TestConfigurationValidation:
    """Test validation of registry configurations."""

    def test_standalone_mode_validates_required_components(self, tmp_path, caplog):
        """Test validation warns about missing required components in standalone mode."""
        import logging

        caplog.set_level(logging.WARNING)

        # Test 1: Missing required infrastructure nodes
        registry_file = tmp_path / "app1" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    CapabilityRegistration
)

class IncompleteProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(
            core_nodes=[],  # Missing required nodes!
            capabilities=[
                CapabilityRegistration(
                    name="test",
                    module_path="app.cap",
                    class_name="TestCap",
                    description="Test",
                    provides=[],
                    requires=[]
                )
            ],
            context_classes=[]
        )
"""
        )

        caplog.clear()
        _ = RegistryManager(registry_path=str(registry_file))  # Trigger validation

        # Check for node warnings
        assert any(
            "missing framework infrastructure nodes" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

        # Test 2: Missing required capabilities
        registry_file2 = tmp_path / "app2" / "registry.py"
        registry_file2.parent.mkdir(parents=True)
        registry_file2.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    NodeRegistration
)

class IncompleteProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(
            core_nodes=[
                NodeRegistration(
                    name="router",
                    module_path="osprey.infrastructure.router_node",
                    function_name="RouterNode",
                    description="Router"
                )
            ],
            capabilities=[],  # Missing respond and clarify!
            context_classes=[]
        )
"""
        )

        caplog.clear()
        _ = RegistryManager(registry_path=str(registry_file2))  # Trigger validation

        # Check for capability warnings
        assert any(
            "missing critical communication capabilities" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_invalid_capability_registration_caught(self, tmp_path, caplog):
        """Test that invalid capability registrations are logged as warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration
)

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="",  # Empty name - invalid!
                    module_path="app.cap",
                    class_name="Cap",
                    description="Test",
                    provides=[],
                    requires=[]
                )
            ]
        )
"""
        )

        # Registry loads but logs warnings for invalid capabilities
        manager = RegistryManager(registry_path=str(registry_file))
        manager.initialize()

        # Check that warnings were logged
        assert any(
            "failed to initialize" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )


class TestHelpfulErrorMessages:
    """Test that error messages are helpful and actionable."""

    def test_missing_file_error_is_actionable(self):
        """Test missing file error provides actionable guidance."""
        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path="./missing/registry.py")

        error_msg = str(exc_info.value).lower()
        # Should mention the file wasn't found
        assert "not found" in error_msg or "does not exist" in error_msg

    def test_no_provider_error_shows_example(self, tmp_path):
        """Test no provider error includes example code."""
        registry_file = tmp_path / "empty.py"
        registry_file.write_text("")

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=str(registry_file))

        error_msg = str(exc_info.value)
        # Should include example implementation
        assert "class" in error_msg
        assert "RegistryConfigProvider" in error_msg

    def test_multiple_providers_error_lists_classes(self, tmp_path):
        """Test multiple providers error lists the conflicting classes."""
        registry_file = tmp_path / "multi.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class FirstProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(capabilities=[], context_classes=[])

class SecondProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(capabilities=[], context_classes=[])
"""
        )

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=str(registry_file))

        error_msg = str(exc_info.value)
        # Should mention there are multiple providers
        assert "Multiple" in error_msg or "multiple" in error_msg


class TestEdgeCases:
    """Test edge cases in validation and error handling."""

    def test_provider_returns_none(self, tmp_path):
        """Test handling when provider returns None."""
        registry_file = tmp_path / "none_config.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider

class BadProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return None  # Oops!
"""
        )

        # Should raise appropriate error
        with pytest.raises((RegistryError, TypeError, AttributeError)):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_provider_raises_exception(self, tmp_path):
        """Test handling when provider method raises exception."""
        registry_file = tmp_path / "raises.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class BadProvider(RegistryConfigProvider):
    def get_registry_config(self):
        raise ValueError("Something went wrong!")
"""
        )

        # Should wrap and re-raise as RegistryError
        with pytest.raises((RegistryError, ValueError)):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_empty_file_handled_gracefully(self, tmp_path):
        """Test that empty registry file gives clear error."""
        registry_file = tmp_path / "empty.py"
        registry_file.write_text("")

        with pytest.raises(RegistryError, match="No RegistryConfigProvider"):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_file_with_only_comments(self, tmp_path):
        """Test file with only comments gives clear error."""
        registry_file = tmp_path / "comments.py"
        registry_file.write_text(
            """
# This is just a comment
# No actual code here
"""
        )

        with pytest.raises(RegistryError, match="No RegistryConfigProvider"):
            _ = RegistryManager(registry_path=str(registry_file))


class TestConfigurationErrorMessages:
    """Test configuration-related error messages."""

    def test_invalid_module_path_error(self, tmp_path, caplog):
        """Test that invalid module paths are logged as warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration
)

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="test",
                    module_path="nonexistent.module.path",
                    class_name="TestCap",
                    description="Test",
                    provides=[],
                    requires=[]
                )
            ]
        )
"""
        )

        # Create manager and initialize
        manager = RegistryManager(registry_path=str(registry_file))
        manager.initialize()

        # Should log warning about failed initialization
        assert any(
            "failed to initialize capability" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_invalid_class_name_error(self, tmp_path, caplog):
        """Test that missing class names are logged as warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        # Create the module file
        app_dir = tmp_path / "app"
        app_dir.mkdir(parents=True)

        cap_file = app_dir / "cap.py"
        cap_file.write_text(
            """
class ExistingClass:
    pass
"""
        )

        registry_file = app_dir / "registry.py"
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration
)

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="test",
                    module_path="app.cap",
                    class_name="NonExistentClass",  # Doesn't exist!
                    description="Test",
                    provides=[],
                    requires=[]
                )
            ]
        )
"""
        )

        # Create manager and initialize
        manager = RegistryManager(registry_path=str(registry_file))
        manager.initialize()

        # Should log warning about failed initialization
        assert any(
            "failed to initialize capability" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
