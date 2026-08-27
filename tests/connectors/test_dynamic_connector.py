"""Tests for dynamic connector import via dotted module paths."""

import logging

import pytest

from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries


@pytest.fixture(autouse=True)
def clean_factory():
    """Run each test against empty factory registries.

    Snapshot/restore brackets the clear so registrations made elsewhere in the
    process survive this module's teardown.
    """
    with isolated_connector_registries(clear=True):
        yield


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


class TestControlSystemContextValidation:
    """Test ControlSystemContext._validate() accepts dotted paths."""

    def test_validate_accepts_dotted_path(self, caplog):
        """Dotted path type does not produce a warning."""
        from osprey.mcp_server.control_system.server_context import (
            ControlSystemContext,
            MCPServerConfig,
        )

        registry = ControlSystemContext()
        registry._config = MCPServerConfig(
            raw={
                "control_system": {"type": "my.custom.Connector"},
                "archiver": {"type": "my.custom.Archiver"},
            }
        )
        with caplog.at_level(
            logging.WARNING, logger="osprey.mcp_server.control_system.server_context"
        ):
            registry._validate()
        assert "Unknown control_system.type" not in caplog.text
        assert "Unknown archiver.type" not in caplog.text

    def test_validate_warns_unknown_simple_type(self, caplog):
        """Non-dotted unknown type still produces a warning."""
        from osprey.mcp_server.control_system.server_context import (
            ControlSystemContext,
            MCPServerConfig,
        )

        registry = ControlSystemContext()
        registry._config = MCPServerConfig(
            raw={
                "control_system": {"type": "tango"},
                "archiver": {"type": "custom_archiver"},
            }
        )
        with caplog.at_level(
            logging.WARNING, logger="osprey.mcp_server.control_system.server_context"
        ):
            registry._validate()
        assert "Unknown control_system.type: tango" in caplog.text
        assert "Unknown archiver.type: custom_archiver" in caplog.text


class TestCustomConnectorVerificationDefault:
    """A custom connector keeps its own ``verification_level`` default.

    ``MockDynamicConnector`` predates the omission sentinel and declares
    ``verification_level="callback"`` outright. The batch path must leave the
    keyword off rather than forward ``None``, or that default would be lost.
    """

    @pytest.fixture
    def writes_enabled(self, monkeypatch):
        # A custom connector is stamped with its dotted type and resolves its
        # posture from the ``control_system`` section, which carries no block for
        # that type and so inherits the deployment-wide key.
        section = {"writes_enabled": True}
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda key, default=None: section if key == "control_system" else default,
        )

    async def _connector(self):
        config = {
            "type": "tests.connectors._mock_dynamic_connector.MockDynamicConnector",
            "connector": {},
        }
        return await ConnectorFactory.create_control_system_connector(config)

    @pytest.mark.asyncio
    async def test_omitted_level_leaves_connector_default_in_place(self, writes_enabled):
        """Single write with the keyword omitted: the connector's default applies."""
        connector = await self._connector()

        result = await connector.write_channel("TEST:CH", 1.0)

        assert result.success is True
        assert result.verification.level == "callback"

    @pytest.mark.asyncio
    async def test_batch_omits_keyword_for_custom_connector(self, writes_enabled):
        """Batch write with the keyword omitted: no ``None`` reaches write_channel."""
        connector = await self._connector()

        seen_kwargs = []
        original = connector.write_channel

        async def spy(channel_address, value, **kwargs):
            seen_kwargs.append(kwargs)
            return await original(channel_address, value, **kwargs)

        connector.write_channel = spy

        results = await connector.write_multiple_channels([("TEST:CH1", 1.0), ("TEST:CH2", 2.0)])

        assert [r.channel_address for r in results] == ["TEST:CH1", "TEST:CH2"]
        for result in results:
            assert result.verification.level == "callback"
        # The keyword is absent, not None — forwarding None would override the default.
        assert seen_kwargs and all("verification_level" not in kw for kw in seen_kwargs)

    @pytest.mark.asyncio
    async def test_batch_forwards_explicit_level(self, writes_enabled):
        """An explicit level still reaches a custom connector unchanged."""
        connector = await self._connector()

        results = await connector.write_multiple_channels(
            [("TEST:CH1", 1.0), ("TEST:CH2", 2.0)], verification_level="readback"
        )

        for result in results:
            assert result.verification.level == "readback"
