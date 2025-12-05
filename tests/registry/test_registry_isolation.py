"""Unit tests for registry isolation and reset behavior.

These tests verify that registry reset/initialization works correctly
in test scenarios, catching the kinds of state pollution issues that
were causing e2e test failures.
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from osprey.registry import get_registry, initialize_registry, reset_registry


@pytest.fixture
def temp_config():
    """Create a temporary minimal config file for testing."""
    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"

        # Create minimal config with framework-only registry
        config_content = """
application:
  name: "test_app"

models:
  classifier:
    provider: cborg
    model: anthropic/claude-haiku
  orchestrator:
    provider: cborg
    model: anthropic/claude-haiku
"""
        config_path.write_text(config_content)

        yield str(config_path)


class TestRegistryReset:
    """Test registry reset and reinitialization behavior."""

    def test_reset_clears_initialized_flag(self, temp_config):
        """Verify reset_registry() clears the _initialized flag."""
        # Initialize registry
        reset_registry()
        initialize_registry(config_path=temp_config)
        registry = get_registry()

        assert registry._initialized, "Registry should be initialized"
        assert len(registry.get_all_nodes()) > 0, "Should have nodes after init"

        # Reset should clear initialized flag
        reset_registry()
        registry = get_registry(config_path=temp_config)

        assert not registry._initialized, "Registry should not be initialized after reset"
        assert len(registry.get_all_nodes()) == 0, "Should have no nodes after reset"

    def test_reset_allows_reinitialization(self, temp_config):
        """Verify we can reinitialize after reset."""
        # First initialization
        reset_registry()
        initialize_registry(config_path=temp_config)
        registry1 = get_registry()
        nodes1 = len(registry1.get_all_nodes())

        assert registry1._initialized
        assert nodes1 > 0

        # Reset and reinitialize
        reset_registry()
        initialize_registry(config_path=temp_config)
        registry2 = get_registry()
        nodes2 = len(registry2.get_all_nodes())

        assert registry2._initialized
        assert nodes2 > 0
        assert nodes2 == nodes1, "Should have same number of nodes after reinit"

    def test_multiple_resets_in_sequence(self, temp_config):
        """Verify multiple reset/init cycles work correctly."""
        for i in range(3):
            reset_registry()

            # Before init
            registry = get_registry(config_path=temp_config)
            assert not registry._initialized, f"Cycle {i}: Should not be initialized before init"

            # After init
            initialize_registry(config_path=temp_config)
            registry = get_registry()
            assert registry._initialized, f"Cycle {i}: Should be initialized after init"
            assert len(registry.get_all_nodes()) > 0, f"Cycle {i}: Should have nodes"

    def test_initialize_is_idempotent(self, temp_config):
        """Verify calling initialize_registry() multiple times is safe."""
        reset_registry()
        initialize_registry(config_path=temp_config)
        registry = get_registry()
        nodes_first = len(registry.get_all_nodes())

        # Call initialize again - should be idempotent
        initialize_registry(config_path=temp_config)
        registry = get_registry()
        nodes_second = len(registry.get_all_nodes())

        assert nodes_first == nodes_second
        assert registry._initialized


class TestRegistryIsolation:
    """Test that registry provides proper isolation between test scenarios."""

    def test_config_cache_cleared_on_reset(self, temp_config):
        """Verify config cache is properly isolated."""
        from osprey.utils import config as config_module

        # Initialize with first config
        reset_registry()
        initialize_registry(config_path=temp_config)

        # Manually clear config cache (simulating what test fixtures should do)
        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Reset registry
        reset_registry()

        # Should be able to initialize with new config
        initialize_registry(config_path=temp_config)
        registry = get_registry()
        assert registry._initialized

    def test_approval_manager_reset(self, temp_config):
        """Verify approval manager singleton is properly reset."""
        try:
            import osprey.approval.approval_manager as approval_module
            from osprey.approval.approval_manager import (
                _approval_manager,
                get_approval_manager,
            )

            # Initialize registry (which initializes approval manager)
            reset_registry()
            initialize_registry(config_path=temp_config)

            manager1 = get_approval_manager()
            assert manager1 is not None

            # Reset approval manager
            approval_module._approval_manager = None

            # Reset and reinitialize registry
            reset_registry()
            initialize_registry(config_path=temp_config)

            manager2 = get_approval_manager()
            assert manager2 is not None

        except ImportError:
            pytest.skip("Approval manager not available")

    def test_env_var_isolation(self, temp_config):
        """Verify CONFIG_FILE env var doesn't pollute between tests."""
        # Set CONFIG_FILE
        os.environ["CONFIG_FILE"] = temp_config

        reset_registry()
        initialize_registry()
        registry1 = get_registry()
        assert registry1._initialized

        # Clear env var (simulating test cleanup)
        del os.environ["CONFIG_FILE"]

        # Reset and reinitialize with explicit path
        reset_registry()
        initialize_registry(config_path=temp_config)
        registry2 = get_registry()
        assert registry2._initialized

    def test_sys_modules_cleanup(self, temp_config):
        """Verify imported application modules can be cleaned up."""
        # Simulate importing an application module
        test_module_name = "test_fake_app_12345"

        # Create a fake module
        import types

        fake_module = types.ModuleType(test_module_name)
        sys.modules[test_module_name] = fake_module

        assert test_module_name in sys.modules

        # Clean up (simulating what test cleanup should do)
        modules_to_remove = [key for key in sys.modules.keys() if test_module_name in key]
        for module in modules_to_remove:
            del sys.modules[module]

        assert test_module_name not in sys.modules


class TestRegistryStateAfterBenchmark:
    """Simulate what benchmark tests do and verify cleanup."""

    def test_simulated_benchmark_cleanup(self, temp_config):
        """Simulate benchmark test pattern and verify proper cleanup."""
        # Simulate what benchmark test does
        original_cwd = os.getcwd()
        original_config = os.environ.get("CONFIG_FILE")

        try:
            # Set CONFIG_FILE like benchmark does
            os.environ["CONFIG_FILE"] = temp_config

            # Initialize registry (benchmark runner might do this internally)
            reset_registry()
            initialize_registry()
            registry1 = get_registry()
            assert registry1._initialized

        finally:
            # Cleanup like benchmark test finally block
            os.chdir(original_cwd)
            if original_config is not None:
                os.environ["CONFIG_FILE"] = original_config
            elif "CONFIG_FILE" in os.environ:
                del os.environ["CONFIG_FILE"]

        # Now simulate the next test (like the failing e2e tests)
        # This should work with proper cleanup
        reset_registry()

        # Clear config cache like test fixture should
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Initialize with new config
        initialize_registry(config_path=temp_config)
        registry2 = get_registry()

        assert registry2._initialized, "Next test should be able to initialize"
        assert len(registry2.get_all_nodes()) > 0, "Next test should have nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
