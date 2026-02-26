"""Tests for dynamic connector import via dotted module paths."""

import logging

import pytest

from osprey.connectors.factory import ConnectorFactory


@pytest.fixture(autouse=True)
def clean_factory():
    """Clear factory registries between tests."""
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._archiver_connectors.clear()
    yield
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._archiver_connectors.clear()


class TestDynamicConnectorImport:
    """Test dynamic import fallback for dotted module paths."""

    @pytest.mark.asyncio
    async def test_dynamic_import_connector(self):
        """Dotted module path resolves and instantiates the connector."""
        config = {
            "type": "tests.connectors._mock_dynamic_connector.MockDynamicConnector",
            "connector": {},
        }
        connector = await ConnectorFactory.create_control_system_connector(config)
        result = await connector.read_channel("TEST:CH")
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_dynamic_import_caches(self):
        """Second call with same dotted path uses cached class (no re-import)."""
        dotted = "tests.connectors._mock_dynamic_connector.MockDynamicConnector"
        config = {"type": dotted, "connector": {}}

        await ConnectorFactory.create_control_system_connector(config)
        assert dotted in ConnectorFactory._control_system_connectors

        # Second call should hit the cache, not importlib
        connector2 = await ConnectorFactory.create_control_system_connector(config)
        result = await connector2.read_channel("TEST:CH")
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_invalid_module_path(self):
        """Non-existent module raises ValueError with clear message."""
        config = {"type": "nonexistent.module.Foo", "connector": {}}
        with pytest.raises(ValueError, match="Could not import connector module"):
            await ConnectorFactory.create_control_system_connector(config)

    @pytest.mark.asyncio
    async def test_invalid_class_name(self):
        """Valid module but missing class raises ValueError."""
        config = {
            "type": "tests.connectors._mock_dynamic_connector.NoSuchClass",
            "connector": {},
        }
        with pytest.raises(ValueError, match="has no class 'NoSuchClass'"):
            await ConnectorFactory.create_control_system_connector(config)

    @pytest.mark.asyncio
    async def test_simple_unknown_type_error_message(self):
        """Non-dotted unknown type still raises with helpful message."""
        config = {"type": "tango", "connector": {}}
        with pytest.raises(ValueError, match="Use a dotted module path"):
            await ConnectorFactory.create_control_system_connector(config)


class TestMCPRegistryValidation:
    """Test MCPRegistry._validate() accepts dotted paths."""

    def test_validate_accepts_dotted_path(self, caplog):
        """Dotted path type does not produce a warning."""
        from osprey.mcp_server.control_system.registry import MCPRegistry, MCPServerConfig

        registry = MCPRegistry()
        registry._config = MCPServerConfig(
            raw={
                "control_system": {"type": "my.custom.Connector"},
                "archiver": {"type": "my.custom.Archiver"},
            }
        )
        with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.control_system.registry"):
            registry._validate()
        assert "Unknown control_system.type" not in caplog.text
        assert "Unknown archiver.type" not in caplog.text

    def test_validate_warns_unknown_simple_type(self, caplog):
        """Non-dotted unknown type still produces a warning."""
        from osprey.mcp_server.control_system.registry import MCPRegistry, MCPServerConfig

        registry = MCPRegistry()
        registry._config = MCPServerConfig(
            raw={
                "control_system": {"type": "tango"},
                "archiver": {"type": "custom_archiver"},
            }
        )
        with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.control_system.registry"):
            registry._validate()
        assert "Unknown control_system.type: tango" in caplog.text
        assert "Unknown archiver.type: custom_archiver" in caplog.text
