"""Tests for registry validation and error handling.

This test module validates error handling and validation in the registry system:
- Missing/invalid registry files
- Invalid Python syntax
- Missing/multiple providers
- Configuration validation
- Helpful error messages
"""

import pytest

from osprey.registry import RegistryConfig
from osprey.registry.manager import RegistryError, RegistryManager, resolve_registry_path


class TestFileLoadingErrors:
    """Test error handling for file loading issues."""

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            pytest.param(
                "this is not valid python syntax{{{",
                "Failed to load Python module",
                id="invalid-python",
            ),
            pytest.param(
                """
from nonexistent_module import something  # This will fail

from osprey.registry import RegistryConfigProvider, RegistryConfig

class Provider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
""",
                "Failed to load",
                id="import-errors",
            ),
        ],
    )
    def test_a_bad_registry_file_is_refused(self, tmp_path, source, match):
        """A registry file that cannot be loaded raises a clear error during __init__."""
        registry_file = tmp_path / "bad_registry.py"
        registry_file.write_text(source)

        with pytest.raises(RegistryError, match=match):
            _ = RegistryManager(registry_path=str(registry_file))

    def test_load_directory_instead_of_file_raises_error(self, tmp_path):
        """Test that passing a directory instead of file raises error."""
        # Create a directory
        registry_dir = tmp_path / "app"
        registry_dir.mkdir(parents=True)

        # Error is raised during __init__
        with pytest.raises(RegistryError, match="not a file"):
            _ = RegistryManager(registry_path=str(registry_dir))

    def test_missing_file_error_message_includes_path(self):
        """Test error message includes the problematic path."""
        missing_path = "./does_not_exist.py"

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=missing_path)

        error_msg = str(exc_info.value)
        assert "Registry file not found" in error_msg
        assert "does_not_exist.py" in error_msg


class TestProviderValidationErrors:
    """Test validation of RegistryConfigProvider implementations."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(
                """
# Valid Python but no RegistryConfigProvider
def some_function():
    pass

class SomeClass:
    pass
""",
                id="code-without-provider",
            ),
            pytest.param("# No provider here\n", id="one-comment"),
            pytest.param("", id="empty-file"),
            pytest.param(
                """
# This is just a comment
# No actual code here
""",
                id="only-comments",
            ),
        ],
    )
    def test_no_provider_raises_error_with_an_example(self, tmp_path, source):
        """A registry without a RegistryConfigProvider is refused, and the error shows one."""
        registry_file = tmp_path / "registry.py"
        registry_file.write_text(source)

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=str(registry_file))

        error_msg = str(exc_info.value)
        assert "No RegistryConfigProvider implementation found" in error_msg
        # Should include an example implementation
        assert "Example:" in error_msg
        assert "class MyRegistryProvider(RegistryConfigProvider)" in error_msg

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

    def test_standalone_mode_validates_configuration(self, tmp_path):
        """Test that standalone registry config can be loaded.

        Note: With the unified TypedEvent pipeline, warnings are emitted as TypedEvents
        not Python logs. The underlying validation still works.
        """
        registry_file2 = tmp_path / "app2" / "registry.py"
        registry_file2.parent.mkdir(parents=True)
        registry_file2.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
)

class MinimalProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
"""
        )

        # Validation still runs during RegistryManager init
        # Warnings are emitted as TypedEvents (not Python logs)
        manager = RegistryManager(registry_path=str(registry_file2))

        # Standalone: the app's own config is used as-is, the framework's is not merged in.
        assert manager.config == RegistryConfig()


class TestHelpfulErrorMessages:
    """Test that error messages are helpful and actionable."""

    def test_multiple_providers_error_lists_classes(self, tmp_path):
        """Test multiple providers error lists the conflicting classes."""
        registry_file = tmp_path / "multi.py"
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class FirstProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()

class SecondProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
"""
        )

        with pytest.raises(RegistryError) as exc_info:
            _ = RegistryManager(registry_path=str(registry_file))

        error_msg = str(exc_info.value)
        assert "Multiple RegistryConfigProvider" in error_msg
        assert "FirstProvider" in error_msg
        assert "SecondProvider" in error_msg


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


class TestResolveRegistryPath:
    """The one resolver every reader of the registry path goes through.

    Three spellings reach the same file, and a reader that knows only one of
    them disagrees with the loader about whether a registry is configured at
    all — which is how the health report came to omit a row for a registry
    that had loaded.
    """

    def test_top_level_spelling(self):
        assert resolve_registry_path({"registry_path": "app/registry.py"}) == "app/registry.py"

    def test_nested_spelling_is_an_accepted_alias(self):
        config = {"application": {"registry_path": "app/registry.py"}}
        assert resolve_registry_path(config) == "app/registry.py"

    def test_top_level_wins_over_nested(self):
        config = {
            "registry_path": "canonical.py",
            "application": {"registry_path": "alias.py"},
        }
        assert resolve_registry_path(config) == "canonical.py"

    def test_env_var_outranks_both(self, monkeypatch):
        monkeypatch.setenv("REGISTRY_PATH", "/app/from_env.py")
        config = {
            "registry_path": "canonical.py",
            "application": {"registry_path": "alias.py"},
        }
        assert resolve_registry_path(config) == "/app/from_env.py"

    def test_no_spelling_resolves_to_none(self):
        assert resolve_registry_path({}) is None

    def test_relative_path_resolves_against_base_path(self, tmp_path):
        resolved = resolve_registry_path({"registry_path": "app/registry.py"}, base_path=tmp_path)
        assert resolved == str(tmp_path / "app" / "registry.py")

    def test_absolute_path_is_left_alone(self, tmp_path):
        absolute = str(tmp_path / "registry.py")
        assert resolve_registry_path({"registry_path": absolute}, base_path=tmp_path) == absolute

    def test_env_vars_in_the_value_are_expanded(self, monkeypatch):
        monkeypatch.setenv("APP_DIR", "myapp")
        resolved = resolve_registry_path({"registry_path": "${APP_DIR}/registry.py"})
        assert resolved == "myapp/registry.py"

    def test_project_root_anchors_a_relative_path_with_no_base_path(self, tmp_path):
        """The anchor a relative spelling means, for a caller that has none.

        Every runtime reader reaches the registry through ``get_registry()``
        with no config path, so it offers no ``base_path``; anchoring on the
        working directory instead found the file only from the repo root.
        """
        (tmp_path / "project").mkdir()
        config = {"project_root": str(tmp_path), "registry_path": "project/registry.py"}

        resolved = resolve_registry_path(config)

        assert resolved == str(tmp_path / "project" / "registry.py")

    def test_an_explicit_base_path_still_wins_over_project_root(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        config = {"project_root": str(tmp_path), "registry_path": "registry.py"}

        resolved = resolve_registry_path(config, base_path=elsewhere)

        assert resolved == str(elsewhere / "registry.py")

    def test_a_project_root_that_is_not_here_leaves_the_path_relative(self, tmp_path):
        """A rendered config read on another machine must not anchor.

        A service's config bind-mounted into a container carries the
        ``project_root`` of the machine that rendered it. Anchoring on a path
        that does not exist here would be confidently wrong.
        """
        config = {
            "project_root": str(tmp_path / "not-on-this-machine"),
            "registry_path": "project/registry.py",
        }

        assert resolve_registry_path(config) == "project/registry.py"

    def test_project_root_does_not_touch_an_absolute_path(self, tmp_path):
        absolute = str(tmp_path / "registry.py")
        config = {"project_root": str(tmp_path), "registry_path": absolute}

        assert resolve_registry_path(config) == absolute

    def test_env_vars_in_project_root_are_expanded(self, tmp_path, monkeypatch):
        (tmp_path / "project").mkdir()
        monkeypatch.setenv("DEPLOY_ROOT", str(tmp_path))
        config = {"project_root": "${DEPLOY_ROOT}", "registry_path": "project/registry.py"}

        resolved = resolve_registry_path(config)

        assert resolved == str(tmp_path / "project" / "registry.py")


class TestConfigurationErrorMessages:
    """Test configuration-related error messages."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
