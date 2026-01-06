"""Tests for registry mode detection (Standalone vs Extend).

This test module validates the registry mode detection mechanism,
covering:
- Standalone mode: Complete registry with all components
- Extend mode: Registry that extends framework defaults
- Type-based mode detection
- Registry validation for standalone mode
"""

import pytest

from osprey.registry.base import (
    ExtendedRegistryConfig,
    RegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry
from osprey.registry.manager import RegistryManager


class TestExtendMode:
    """Test extend mode detection and behavior."""

    def test_extend_mode_with_helper_function(self, tmp_path):
        """Test that extend_framework_registry() triggers extend mode."""
        # Create registry using helper (returns ExtendedRegistryConfig)
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration
)

class ExtendProvider(RegistryConfigProvider):
    def get_registry_config(self):
        # extend_framework_registry returns ExtendedRegistryConfig
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="app_capability",
                    module_path="app.capabilities.test",
                    class_name="TestCapability",
                    description="Application capability",
                    provides=["APP_DATA"],
                    requires=[]
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should merge with framework (extend mode)
        cap_names = [c.name for c in manager.config.capabilities]

        # Application capability present
        assert "app_capability" in cap_names

        # Framework capabilities also present (merged)
        assert "memory" in cap_names
        assert "python" in cap_names
        assert "respond" in cap_names

    def test_extend_mode_preserves_initialization_order(self, tmp_path):
        """Test that extend mode uses framework initialization order."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class ExtendProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should use framework initialization order
        # Note: actual order may vary based on framework config
        # Key assertion: order should match framework's order
        from osprey.registry.helpers import get_framework_defaults

        framework = get_framework_defaults()

        assert manager.config.initialization_order == framework.initialization_order

    def test_extend_mode_includes_framework_nodes(self, tmp_path):
        """Test that extend mode includes framework core nodes."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class ExtendProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should have framework nodes
        node_names = [n.name for n in manager.config.core_nodes]
        assert "router" in node_names
        assert "orchestrator" in node_names
        assert "classifier" in node_names

    def test_extend_mode_with_application_additions(self, tmp_path):
        """Test extend mode properly adds application components."""
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

class ExtendProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="custom_cap1",
                    module_path="app.cap1",
                    class_name="Cap1",
                    description="Custom 1",
                    provides=["DATA1"],
                    requires=[]
                ),
                CapabilityRegistration(
                    name="custom_cap2",
                    module_path="app.cap2",
                    class_name="Cap2",
                    description="Custom 2",
                    provides=["DATA2"],
                    requires=["DATA1"]
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="DATA1",
                    module_path="app.context",
                    class_name="Data1Context"
                ),
                ContextClassRegistration(
                    context_type="DATA2",
                    module_path="app.context",
                    class_name="Data2Context"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Application components added
        cap_names = [c.name for c in manager.config.capabilities]
        assert "custom_cap1" in cap_names
        assert "custom_cap2" in cap_names

        ctx_types = [c.context_type for c in manager.config.context_classes]
        assert "DATA1" in ctx_types
        assert "DATA2" in ctx_types

        # Framework components still present
        assert "memory" in cap_names
        assert "MEMORY_CONTEXT" in ctx_types


class TestStandaloneMode:
    """Test standalone mode detection and behavior."""

    def test_standalone_mode_with_direct_registry_config(self, tmp_path):
        """Test that direct RegistryConfig use triggers standalone mode."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    NodeRegistration,
    CapabilityRegistration,
    ContextClassRegistration
)

class StandaloneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        # Direct RegistryConfig = standalone mode
        return RegistryConfig(
            core_nodes=[
                NodeRegistration(
                    name="router",
                    module_path="osprey.infrastructure.router_node",
                    function_name="RouterNode",
                    description="Router"
                ),
                NodeRegistration(
                    name="orchestrator",
                    module_path="osprey.infrastructure.orchestrator_node",
                    function_name="OrchestratorNode",
                    description="Orchestrator"
                ),
                NodeRegistration(
                    name="classifier",
                    module_path="osprey.infrastructure.classifier_node",
                    function_name="ClassifierNode",
                    description="Classifier"
                ),
                NodeRegistration(
                    name="error",
                    module_path="osprey.infrastructure.error_node",
                    function_name="ErrorNode",
                    description="Error handler"
                ),
                NodeRegistration(
                    name="task_extraction",
                    module_path="osprey.infrastructure.task_extraction_node",
                    function_name="TaskExtractionNode",
                    description="Task extraction"
                )
            ],
            capabilities=[
                CapabilityRegistration(
                    name="respond",
                    module_path="osprey.capabilities.respond",
                    class_name="RespondCapability",
                    description="Respond to user",
                    provides=["FINAL_RESPONSE"],
                    requires=[],
                    always_active=True
                ),
                CapabilityRegistration(
                    name="clarify",
                    module_path="osprey.capabilities.clarify",
                    class_name="ClarifyCapability",
                    description="Ask for clarification",
                    provides=["CLARIFICATION_REQUEST"],
                    requires=[],
                    always_active=True
                ),
                CapabilityRegistration(
                    name="custom_capability",
                    module_path="app.capabilities.custom",
                    class_name="CustomCapability",
                    description="Custom functionality",
                    provides=["CUSTOM_DATA"],
                    requires=[]
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="FINAL_RESPONSE",
                    module_path="osprey.context.final_response",
                    class_name="FinalResponseContext"
                ),
                ContextClassRegistration(
                    context_type="CLARIFICATION_REQUEST",
                    module_path="osprey.context.clarification",
                    class_name="ClarificationContext"
                ),
                ContextClassRegistration(
                    context_type="CUSTOM_DATA",
                    module_path="app.context",
                    class_name="CustomContext"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should be standalone - only components from config
        cap_names = [c.name for c in manager.config.capabilities]

        # Application capability present
        assert "custom_capability" in cap_names

        # Required framework capabilities present
        assert "respond" in cap_names
        assert "clarify" in cap_names

        # Other framework capabilities NOT present (standalone mode)
        assert "memory" not in cap_names
        assert "python" not in cap_names
        assert "time_range_parsing" not in cap_names

    def test_standalone_mode_validation_warnings(self, tmp_path, caplog):
        """Test that standalone mode validates required components."""
        import logging

        caplog.set_level(logging.WARNING)

        # Create minimal standalone registry missing required nodes
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    CapabilityRegistration,
    ContextClassRegistration
)

class IncompleteStandaloneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        # Standalone but missing required infrastructure
        return RegistryConfig(
            core_nodes=[],  # Missing required nodes!
            capabilities=[
                CapabilityRegistration(
                    name="test_cap",
                    module_path="app.cap",
                    class_name="TestCap",
                    description="Test",
                    provides=["TEST"],
                    requires=[]
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="TEST",
                    module_path="app.context",
                    class_name="TestContext"
                )
            ]
        )
"""
        )

        # Load registry - should warn about missing components
        _ = RegistryManager(registry_path=str(registry_file))  # noqa: F841

        # Check that validation warnings were logged
        assert any(
            "missing framework infrastructure nodes" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_standalone_mode_complete_registry(self, tmp_path):
        """Test standalone mode with complete, valid registry."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    NodeRegistration,
    CapabilityRegistration,
    ContextClassRegistration
)

class CompleteStandaloneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(
            core_nodes=[
                NodeRegistration(
                    name="router",
                    module_path="osprey.infrastructure.router_node",
                    function_name="RouterNode",
                    description="Router"
                ),
                NodeRegistration(
                    name="orchestrator",
                    module_path="osprey.infrastructure.orchestrator_node",
                    function_name="OrchestratorNode",
                    description="Orchestrator"
                ),
                NodeRegistration(
                    name="classifier",
                    module_path="osprey.infrastructure.classifier_node",
                    function_name="ClassifierNode",
                    description="Classifier"
                ),
                NodeRegistration(
                    name="error",
                    module_path="osprey.infrastructure.error_node",
                    function_name="ErrorNode",
                    description="Error"
                ),
                NodeRegistration(
                    name="task_extraction",
                    module_path="osprey.infrastructure.task_extraction_node",
                    function_name="TaskExtractionNode",
                    description="Task extraction"
                )
            ],
            capabilities=[
                CapabilityRegistration(
                    name="respond",
                    module_path="osprey.capabilities.respond",
                    class_name="RespondCapability",
                    description="Respond",
                    provides=["FINAL_RESPONSE"],
                    requires=[],
                    always_active=True
                ),
                CapabilityRegistration(
                    name="clarify",
                    module_path="osprey.capabilities.clarify",
                    class_name="ClarifyCapability",
                    description="Clarify",
                    provides=["CLARIFICATION_REQUEST"],
                    requires=[],
                    always_active=True
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="FINAL_RESPONSE",
                    module_path="osprey.context.final_response",
                    class_name="FinalResponseContext"
                ),
                ContextClassRegistration(
                    context_type="CLARIFICATION_REQUEST",
                    module_path="osprey.context.clarification",
                    class_name="ClarificationContext"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Should have exactly the specified components
        node_names = [n.name for n in manager.config.core_nodes]
        assert len(node_names) == 5
        assert "router" in node_names
        assert "orchestrator" in node_names

        cap_names = [c.name for c in manager.config.capabilities]
        assert len(cap_names) == 2
        assert "respond" in cap_names
        assert "clarify" in cap_names


class TestModeDetection:
    """Test type-based mode detection mechanism."""

    def test_helper_returns_extended_config(self):
        """Test that extend_framework_registry returns ExtendedRegistryConfig."""
        config = extend_framework_registry(capabilities=[], context_classes=[])

        assert isinstance(config, ExtendedRegistryConfig)
        assert isinstance(config, RegistryConfig)

    def test_mode_detection_in_actual_loading(self, tmp_path):
        """Test mode detection happens during registry loading."""
        # Create extend mode registry
        extend_file = tmp_path / "extend_app" / "registry.py"
        extend_file.parent.mkdir(parents=True)
        extend_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class ExtendProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(capabilities=[], context_classes=[])
"""
        )

        # Create standalone mode registry
        standalone_file = tmp_path / "standalone_app" / "registry.py"
        standalone_file.parent.mkdir(parents=True)
        standalone_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class StandaloneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(
            capabilities=[],
            context_classes=[],
            core_nodes=[]
        )
"""
        )

        # Load extend mode
        extend_manager = RegistryManager(registry_path=str(extend_file))

        # Load standalone mode
        standalone_manager = RegistryManager(registry_path=str(standalone_file))

        # Extend mode should have framework capabilities
        extend_caps = [c.name for c in extend_manager.config.capabilities]
        assert len(extend_caps) > 0
        assert "memory" in extend_caps

        # Standalone mode should not have framework capabilities
        standalone_caps = [c.name for c in standalone_manager.config.capabilities]
        assert len(standalone_caps) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
