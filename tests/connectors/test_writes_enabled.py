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
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
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
        """With _writes_enabled=False, write returns ChannelWriteResult(success=False)."""
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            result = await connector.write_channel("TEST:PV", 1.0)
        assert isinstance(result, ChannelWriteResult)
        assert result.success is False
        assert "TEST:PV" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

    @pytest.mark.asyncio
    async def test_write_passes_through_when_enabled(self):
        """With _writes_enabled=True, the original write_channel is called."""

        # Need a subclass with a working write_channel
        class _WritableStub(ControlSystemConnector):
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
                    success=True,
                )

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

        connector = _WritableStub()
        with patch("osprey.utils.config.get_config_value", return_value=True):
            result = await connector.write_channel("TEST:PV", 42.0)
        assert result.success is True
        assert result.value_written == 42.0

    @pytest.mark.asyncio
    async def test_error_message_contains_channel_and_config_path(self):
        """Error message must contain the channel name and config path."""
        connector = _StubConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            result = await connector.write_channel("MY:SPECIAL:PV", 99.9)
        assert "MY:SPECIAL:PV" in result.error_message
        assert "control_system.writes_enabled" in result.error_message


class TestWritesEnabledProperty:
    """Tests for the _writes_enabled base-class property."""

    def test_returns_false_when_config_says_false(self):
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value", return_value=False
        ):
            assert connector._writes_enabled is False

    def test_returns_true_when_config_says_true(self):
        connector = _StubConnector()
        with patch(
            "osprey.utils.config.get_config_value", return_value=True
        ):
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
        assert result.success is False
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
        assert result.success is True

    def test_mock_no_longer_has_enable_writes_attr(self):
        """MockConnector should no longer have _enable_writes after cleanup."""
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
        assert result.success is False
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
        assert result.success is True
        assert result.channel_address == "BEAM:CURRENT"
        assert result.value_written == 500.0
