"""Tests for ControlSystemConnector._writes_enabled property."""

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)


class _StubConnector(ControlSystemConnector):
    """Minimal concrete subclass for testing base-class properties."""

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        raise NotImplementedError

    async def write_multiple_channels(self, operations, **kwargs):
        raise NotImplementedError

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        raise NotImplementedError

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None:
        pass

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        raise NotImplementedError

    async def validate_channel(self, channel_address: str) -> bool:
        raise NotImplementedError


class _WritableStub(ControlSystemConnector):
    """Concrete subclass with working write methods for pass-through tests."""

    async def connect(self, config):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, addr, timeout=None):
        raise NotImplementedError

    async def write_channel(self, channel_address, value, **kwargs):
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
        )

    async def write_multiple_channels(self, operations, **kwargs):
        return [
            ChannelWriteResult(
                channel_address=addr,
                value_written=val,
                outcome=WriteOutcome.CONFIRMED,
            )
            for addr, val in operations
        ]

    async def read_multiple_channels(self, addrs, timeout=None):
        return {}

    async def subscribe(self, addr, cb):
        return "sub"

    async def unsubscribe(self, sub_id):
        pass

    async def get_metadata(self, addr):
        raise NotImplementedError

    async def validate_channel(self, addr):
        return True


class TestInitSubclassWrapping:
    """Tests for __init_subclass__ write_channel wrapping."""

    def test_subclass_write_channel_is_wrapped(self):
        """write_channel on a subclass should NOT be the original method."""
        # _StubConnector defines write_channel, so it should be wrapped
        connector = _StubConnector()
        # The method should have been replaced by _guarded
        assert hasattr(connector.write_channel, "__wrapped__")

    @pytest.mark.asyncio
    async def test_write_blocked_when_disabled(self):
        """With _writes_enabled=False, the write is refused and never attempted."""
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            result = await connector.write_channel("TEST:PV", 1.0)
        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "WRITES_DISABLED"
        assert "TEST:PV" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

    @pytest.mark.asyncio
    async def test_write_passes_through_when_enabled(self):
        """With _writes_enabled=True, the original write_channel is called."""
        connector = _WritableStub()
        with patch("osprey.utils.config.get_config_value", return_value=True):
            result = await connector.write_channel("TEST:PV", 42.0)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.value_written == 42.0

    @pytest.mark.asyncio
    async def test_error_message_contains_channel_and_config_path(self):
        """Error message must contain the channel name and config path."""
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            result = await connector.write_channel("MY:SPECIAL:PV", 99.9)
        assert "MY:SPECIAL:PV" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

    @pytest.mark.asyncio
    async def test_write_multiple_blocked_when_disabled(self):
        """With _writes_enabled=False, write_multiple returns failures."""
        connector = _StubConnector()
        ops = [("PV:A", 1.0), ("PV:B", 2.0)]
        with patch("osprey.utils.config.get_config_value", return_value=False):
            results = await connector.write_multiple_channels(ops)
        assert len(results) == 2
        assert all(r.outcome is WriteOutcome.REFUSED for r in results)
        assert "PV:A" in results[0].error_message
        assert "PV:B" in results[1].error_message
        assert "writes are disabled" in results[0].error_message

    @pytest.mark.asyncio
    async def test_write_multiple_passes_through_when_enabled(self):
        """With _writes_enabled=True, the original write_multiple_channels is called."""
        connector = _WritableStub()
        ops = [("PV:A", 1.0), ("PV:B", 2.0)]
        with patch("osprey.utils.config.get_config_value", return_value=True):
            results = await connector.write_multiple_channels(ops)
        assert len(results) == 2
        assert all(r.outcome is WriteOutcome.CONFIRMED for r in results)


class TestWritesEnabledProperty:
    """Tests for the _writes_enabled base-class property."""

    def test_returns_false_when_config_says_false(self):
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            assert connector._writes_enabled is False

    def test_returns_true_when_config_says_true(self):
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=True):
            assert connector._writes_enabled is True

    def test_returns_false_on_file_not_found(self):
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=FileNotFoundError("no config"),
        ):
            assert connector._writes_enabled is False

    def test_returns_false_on_runtime_error(self):
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=RuntimeError("config broken"),
        ):
            assert connector._writes_enabled is False


class TestMockWritesDisabledViaBaseClass:
    """Tests that MockConnector write blocking now comes from base class."""

    @pytest.mark.asyncio
    async def test_mock_blocks_writes_when_disabled(self):
        """MockConnector blocks writes via base class when writes_enabled=false."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await connector.connect({"response_delay_ms": 0})
            result = await connector.write_channel("TEST:PV", 1.0)
        assert result.outcome is WriteOutcome.REFUSED
        assert "writes are disabled" in result.error_message  # base class message

    @pytest.mark.asyncio
    async def test_mock_allows_writes_when_enabled(self):
        """MockConnector allows writes when writes_enabled=true."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        def _writes_enabled_config(key, default=None):
            if key == "control_system.writes_enabled":
                return True
            return default

        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            await connector.connect({"response_delay_ms": 0})
            result = await connector.write_channel("TEST:PV", 1.0)
        assert result.outcome is not WriteOutcome.REFUSED

    def test_mock_has_no_enable_writes_attr(self):
        """MockConnector must not carry an _enable_writes attribute."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        assert not hasattr(connector, "_enable_writes")


@pytest.mark.integration
class TestWriteBlockedIntegration:
    """Integration test: full write path with real MockConnector."""

    @pytest.mark.asyncio
    async def test_write_blocked_full_path(self):
        """Full path: MockConnector with writes_enabled=false blocks writes."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await connector.connect({"response_delay_ms": 0})
            result = await connector.write_channel("BEAM:CURRENT", 500.0)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert "BEAM:CURRENT" in result.error_message
        assert "writes are disabled" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

    @pytest.mark.asyncio
    async def test_write_allowed_full_path(self):
        """Full path: MockConnector with writes_enabled=true allows writes."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        def _config_for_write_test(key, default=None):
            if key == "control_system.writes_enabled":
                return True
            return default

        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_for_write_test,
        ):
            await connector.connect({"response_delay_ms": 0})
            result = await connector.write_channel("BEAM:CURRENT", 500.0)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is not WriteOutcome.REFUSED
        assert result.channel_address == "BEAM:CURRENT"
        assert result.value_written == 500.0


#: A deployment that arms its simulator only: writes off deployment-wide, on for
#: the virtual accelerator, and an ``epics`` block that says nothing about writes.
_SIMULATOR_ARMED_SECTION = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "virtual_accelerator": {"writes_enabled": True},
        "epics": {"port": 5064},
    },
}

#: The mirror image: armed deployment-wide, disarmed for ``epics`` in particular.
_LIVE_DISARMED_SECTION = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {"epics": {"writes_enabled": False}},
}


def _config_reader(section: dict[str, Any]) -> Callable[..., Any]:
    """A ``get_config_value`` stand-in serving one ``control_system:`` section.

    Answers the two paths the posture is read through — the section itself and
    the deployment-wide key inside it — the way dot-path lookup would.
    """

    def _get(key: str, default: Any = None) -> Any:
        if key == "control_system":
            return section
        if key == "control_system.writes_enabled":
            return section.get("writes_enabled", default)
        return default

    return _get


class _RecordingConnector(_WritableStub):
    """Records the type stamp visible to ``connect()``."""

    def __init__(self):
        self.type_seen_in_connect: Any = "not connected"

    async def connect(self, config):
        self.type_seen_in_connect = self._connector_type


class TestFactoryTypeStamp:
    """The factory stamps the connector type it built, before connect() runs."""

    def test_unstamped_by_default(self):
        """An instance nobody built through the factory carries no type."""
        assert _StubConnector()._connector_type is None

    @pytest.mark.asyncio
    async def test_stamp_is_visible_inside_connect(self):
        from osprey.connectors import types
        from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries

        with isolated_connector_registries(clear=True):
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _RecordingConnector)
            connector = await ConnectorFactory.create_control_system_connector(
                {"type": types.VIRTUAL_ACCELERATOR}
            )

        assert connector.type_seen_in_connect == types.VIRTUAL_ACCELERATOR
        assert connector._connector_type == types.VIRTUAL_ACCELERATOR


class TestPerTypeWritePosture:
    """``control_system.connector.<type>.writes_enabled`` decides, per type."""

    def test_armed_type_writes(self):
        connector = _StubConnector()
        connector._connector_type = "virtual_accelerator"
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_SIMULATOR_ARMED_SECTION),
        ):
            assert connector._writes_enabled is True

    def test_other_type_stays_disarmed(self):
        """The ``epics`` block names no posture, so it inherits the global false."""
        connector = _StubConnector()
        connector._connector_type = "epics"
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_SIMULATOR_ARMED_SECTION),
        ):
            assert connector._writes_enabled is False

    def test_type_block_overrides_armed_global(self):
        connector = _StubConnector()
        connector._connector_type = "epics"
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_LIVE_DISARMED_SECTION),
        ):
            assert connector._writes_enabled is False

    def test_unstamped_connector_reads_the_global_key(self):
        """No type means no block to key a posture on: the global key is the posture."""
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_LIVE_DISARMED_SECTION),
        ):
            assert connector._writes_enabled is True

    @pytest.mark.parametrize("stand_in", ["true", 1, "yes"])
    def test_unstamped_connector_arms_only_on_a_literal_true(self, stand_in):
        """The global key is read under the same literal-``true`` rule as a block."""
        connector = _StubConnector()
        section = {"type": "epics", "writes_enabled": stand_in}
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(section),
        ):
            assert connector._writes_enabled is False

    @pytest.mark.asyncio
    async def test_refusal_names_the_per_type_key(self):
        connector = _StubConnector()
        connector._connector_type = "epics"
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_SIMULATOR_ARMED_SECTION),
        ):
            result = await connector.write_channel("TEST:PV", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert "control_system.connector.epics.writes_enabled" in result.error_message
        assert "Set control_system.writes_enabled" not in result.error_message

    @pytest.mark.asyncio
    async def test_refusal_on_an_unstamped_connector_names_the_global_key(self):
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_reader(_SIMULATOR_ARMED_SECTION),
        ):
            result = await connector.write_channel("TEST:PV", 1.0)

        assert "Set control_system.writes_enabled: true" in result.error_message
