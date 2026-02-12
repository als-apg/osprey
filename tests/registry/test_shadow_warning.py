"""Tests for shadow warning system.

Validates that the registry emits warnings when old-style applications
shadow native framework capabilities without explicitly overriding them.
"""

import logging

from osprey.registry.base import (
    CapabilityRegistration,
    RegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry


class TestShadowWarningSystem:
    """Test shadow warnings for non-explicit overrides of native capabilities."""

    def _make_cap_reg(self, name: str, explicit: bool = False) -> CapabilityRegistration:
        """Create a capability registration for testing."""
        cap = CapabilityRegistration(
            name=name,
            module_path=f"my_app.capabilities.{name}",
            class_name=f"{name.title().replace('_', '')}Capability",
            description=f"Test {name} capability",
            provides=[],
            requires=[],
        )
        if explicit:
            cap._is_explicit_override = True
        return cap

    def test_explicit_override_flag_set_by_extend_framework_registry(self):
        """Test that override_capabilities sets _is_explicit_override = True."""
        config = extend_framework_registry(
            override_capabilities=[
                self._make_cap_reg("channel_finding"),
            ]
        )

        # Find the override capability
        cap = next(c for c in config.capabilities if c.name == "channel_finding")
        assert cap._is_explicit_override is True

    def test_regular_capability_not_flagged(self):
        """Test that regular capabilities don't have _is_explicit_override set."""
        config = extend_framework_registry(
            capabilities=[
                self._make_cap_reg("my_custom_capability"),
            ]
        )

        cap = next(c for c in config.capabilities if c.name == "my_custom_capability")
        assert getattr(cap, "_is_explicit_override", False) is False

    def test_explicit_override_no_shadow_warning(self, caplog):
        """Test that explicit overrides don't produce shadow warnings."""
        from osprey.registry.manager import RegistryManager

        manager = RegistryManager()

        # Create framework config with native capability
        framework_config = RegistryConfig(
            capabilities=[self._make_cap_reg("channel_finding")],
            context_classes=[],
        )

        # Create app config with explicit override
        app_cap = self._make_cap_reg("channel_finding", explicit=True)
        app_config = RegistryConfig(capabilities=[app_cap], context_classes=[])

        with caplog.at_level(logging.WARNING):
            manager._merge_application_with_override(framework_config, app_config, "test_app")

        # Should NOT have shadow warning
        shadow_warnings = [r for r in caplog.records if "shadows native framework" in r.message]
        assert len(shadow_warnings) == 0

    def test_non_explicit_native_override_emits_shadow_warning(self, caplog):
        """Test that non-explicit overrides of native capabilities produce shadow warnings."""
        from osprey.registry.manager import RegistryManager

        manager = RegistryManager()

        # Create framework config with native capability
        framework_config = RegistryConfig(
            capabilities=[self._make_cap_reg("channel_finding")],
            context_classes=[],
        )

        # Create app config with NON-explicit override (simulating old-style app)
        app_cap = self._make_cap_reg("channel_finding")
        # Explicitly NOT setting _is_explicit_override
        app_config = RegistryConfig(capabilities=[app_cap], context_classes=[])

        with caplog.at_level(logging.WARNING):
            manager._merge_application_with_override(framework_config, app_config, "test_app")

        # Should have shadow warning
        shadow_warnings = [r for r in caplog.records if "shadows native framework" in r.message]
        assert len(shadow_warnings) == 1
        assert "channel_finding" in shadow_warnings[0].message
        assert "osprey migrate check" in shadow_warnings[0].message

    def test_non_native_override_no_shadow_warning(self, caplog):
        """Test that overriding non-native capabilities doesn't produce shadow warning."""
        from osprey.registry.manager import RegistryManager

        manager = RegistryManager()

        # Create framework config with a non-native capability
        framework_config = RegistryConfig(
            capabilities=[self._make_cap_reg("memory")],
            context_classes=[],
        )

        # Override it without explicit flag
        app_cap = self._make_cap_reg("memory")
        app_config = RegistryConfig(capabilities=[app_cap], context_classes=[])

        with caplog.at_level(logging.WARNING):
            manager._merge_application_with_override(framework_config, app_config, "test_app")

        # Should NOT have shadow warning (memory is not a native control capability)
        shadow_warnings = [r for r in caplog.records if "shadows native framework" in r.message]
        assert len(shadow_warnings) == 0
