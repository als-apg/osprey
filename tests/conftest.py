"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

import pytest

# ===================================================================
# Auto-reset Registry and Config Between Tests
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_state_between_tests():
    """Auto-reset registry and config before each test to ensure isolation.

    This prevents state leakage between tests by:
    - Clearing OSPREY_CONFIG and CONFIG_FILE env vars
    - Resetting the registry
    - Clearing config caches (utils.config and utils.workspace)
    - Resetting approval manager singleton
    """
    # Reset before test
    import os as _os

    from osprey.registry import reset_registry
    from osprey.utils.workspace import reset_config_cache

    # Clear config-related env vars that may leak between tests.
    # The web terminal lifespan sets OSPREY_CONFIG directly via os.environ,
    # and monkeypatch undo ordering can leave it set between tests.
    _os.environ.pop("OSPREY_CONFIG", None)
    _os.environ.pop("CONFIG_FILE", None)

    reset_registry()
    reset_config_cache()

    yield

    # Reset after test
    _os.environ.pop("OSPREY_CONFIG", None)
    _os.environ.pop("CONFIG_FILE", None)
    reset_registry()
    reset_config_cache()


# ===================================================================
# Marker-driven resource skip-gating
# ===================================================================
#
# Tests use @pytest.mark.requires_<resource> to declare what they need.
# This hook turns those markers into real skips when the resource is
# unavailable. Markers are registered in pyproject.toml under
# [tool.pytest.ini_options].markers; predicates live here.
#
# To add a new resource: append to _RESOURCE_CHECKS. Predicate must be
# zero-arg, side-effect-free, and fast (gets called once per item per
# resource at collection time).


def _has_als_apg_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ALS_APG_API_KEY"))


def _has_anthropic_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ANTHROPIC_API_KEY"))


def _has_any_provider_api_key() -> bool:
    """True if any supported LLM provider key is set.

    Mirrors the inline detection in tests/e2e/test_llm_channel_namer.py:
    als-apg, cborg, amsc, anthropic.
    """
    import os as _os

    return any(
        _os.environ.get(k)
        for k in ("ALS_APG_API_KEY", "CBORG_API_KEY", "AMSC_I2_API_KEY", "ANTHROPIC_API_KEY")
    )


def _is_ollama_available() -> bool:
    """True if a local Ollama server responds at localhost:11434."""
    try:
        import requests

        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


_RESOURCE_CHECKS: dict[str, tuple[callable, str]] = {
    "requires_als_apg": (_has_als_apg_api_key, "ALS_APG_API_KEY not set"),
    "requires_anthropic": (_has_anthropic_api_key, "ANTHROPIC_API_KEY not set"),
    "requires_api": (
        _has_any_provider_api_key,
        "No LLM provider API key set (ALS_APG_API_KEY / CBORG_API_KEY / "
        "AMSC_I2_API_KEY / ANTHROPIC_API_KEY)",
    ),
    "requires_ollama": (_is_ollama_available, "Ollama not reachable at localhost:11434"),
}


def pytest_collection_modifyitems(config, items):
    """Auto-skip items whose `requires_<resource>` marker's resource is missing.

    Each predicate is evaluated at most once per pytest run via cache. A
    missing resource adds a real `pytest.mark.skip(reason=...)`; satisfied
    markers are no-ops.
    """
    cache: dict[str, bool] = {}
    for item in items:
        for marker_name, (predicate, reason) in _RESOURCE_CHECKS.items():
            if marker_name not in item.keywords:
                continue
            available = cache.get(marker_name)
            if available is None:
                available = predicate()
                cache[marker_name] = available
            if not available:
                item.add_marker(pytest.mark.skip(reason=reason))


# ===================================================================
# Prompt Testing Helpers
# ===================================================================


class PromptTestHelpers:
    """Helper methods for testing prompt structure and content."""

    @staticmethod
    def extract_section(prompt: str, section_header: str) -> str:
        """Extract a specific section from the prompt by its header.

        Args:
            prompt: The full prompt text
            section_header: The header marking the start of the section

        Returns:
            The extracted section content (without the header)
        """
        lines = prompt.split("\n")
        section_lines = []
        in_section = False

        for line in lines:
            if section_header in line:
                in_section = True
                continue
            if in_section:
                # Stop at next all-caps header with colon
                if (
                    line.strip()
                    and line.strip().replace(" ", "").replace("'", "").isupper()
                    and ":" in line
                ):
                    break
                section_lines.append(line)

        return "\n".join(section_lines).strip()

    @staticmethod
    def get_section_positions(prompt: str, *section_headers: str) -> dict[str, int]:
        """Get the positions of multiple section headers in the prompt.

        Args:
            prompt: The full prompt text
            *section_headers: Variable number of section headers to find

        Returns:
            Dictionary mapping section headers to their positions (-1 if not found)
        """
        positions = {}
        for header in section_headers:
            try:
                positions[header] = prompt.index(header)
            except ValueError:
                positions[header] = -1
        return positions


# ===================================================================
# Pytest Fixtures
# ===================================================================


@pytest.fixture
def prompt_helpers():
    """Fixture providing prompt testing helper methods."""
    return PromptTestHelpers


# ===================================================================
# Test Configuration Helpers
# ===================================================================


@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a minimal test configuration file.

    Creates a temporary config.yml with sensible defaults for testing.
    Tests can override or extend this as needed.

    Returns:
        Path to the created config file

    Examples:
        Basic usage::

            def test_something(test_config):
                # Config file exists at test_config
                assert test_config.exists()
                # Set environment to use it
                os.environ['CONFIG_FILE'] = str(test_config)

        With custom modifications::

            def test_custom(test_config):
                # Read existing config
                import yaml
                with open(test_config) as f:
                    config = yaml.safe_load(f)
                # Modify
                config['execution']['execution_method'] = 'container'
                # Write back
                with open(test_config, 'w') as f:
                    yaml.dump(config, f)
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry()
"""
    )

    # Minimal working configuration for tests
    config = {
        "project_root": str(tmp_path),
        "registry_path": str(registry_file),
        "approval": {
            "enabled": True,
            "default_policy": "selective",
        },
        "control_system": {
            "type": "mock",  # Use mock for tests - default patterns include write_channel/read_channel
        },
        "execution": {
            "execution_method": "local",  # Fast for tests
            "modes": {
                "read_only": {
                    "kernel_name": "python3",
                    "allows_writes": False,
                    "requires_approval": False,
                },
                "write_access": {
                    "kernel_name": "python3",
                    "allows_writes": True,
                    "requires_approval": True,
                },
            },
            "limits": {"max_retries": 3, "max_execution_time_seconds": 30},
        },
        "models": {
            "orchestrator": {"provider": "openai", "model_id": "gpt-4"},
            "python_code_generator": {"provider": "openai", "model_id": "gpt-4"},
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let each test handle registry initialization
    # to avoid state pollution between tests
    return config_file


@pytest.fixture
def test_config_with_approval(tmp_path):
    """Fixture providing a test configuration WITH approval enabled.

    This is specifically for testing approval workflows.
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry()
"""
    )

    # Configuration with approval ENABLED
    config = {
        "project_root": str(tmp_path),
        "registry_path": str(registry_file),
        "approval": {
            "enabled": True,
            "default_policy": "selective",
        },
        "control_system": {
            "type": "mock",  # Use mock for tests - default patterns include write_channel/read_channel
        },
        "execution": {
            "execution_method": "local",
            "modes": {
                "read_only": {
                    "kernel_name": "python3",
                    "allows_writes": False,
                    "requires_approval": False,
                },
                "write_access": {
                    "kernel_name": "python3",
                    "allows_writes": True,
                    "requires_approval": True,
                },
            },
            "limits": {"max_retries": 3, "max_execution_time_seconds": 30},
        },
        "models": {
            "orchestrator": {"provider": "openai", "model_id": "gpt-4"},
            "python_code_generator": {"provider": "openai", "model_id": "gpt-4"},
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let the test do it
    # to avoid state pollution between tests

    return config_file
