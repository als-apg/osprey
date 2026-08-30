"""Unit tests for osprey.runtime module.

Tests the runtime utilities for control system operations in generated Python code.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey.connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey.runtime import (
    _write_channel_async,
    cleanup_runtime,
    read_channel,
    write_channel,
    write_channels,
)


class MockConnector(ControlSystemConnector):
    """Mock control system connector for testing.

    A real ``ControlSystemConnector`` subclass so the runtime exercises the same
    denial contract (``write_channel_checked``) it uses against live connectors.
    Writes are forced enabled; ``write_channel`` returns ``self.canned_result``
    when one is set, otherwise a confirmed success.
    """

    def __init__(self, canned_result: ChannelWriteResult | None = None):
        self.write_calls: list[tuple[str, Any, dict]] = []
        self.read_calls: list[tuple[str, dict]] = []
        self.disconnect_called = False
        self.canned_result = canned_result

    @property
    def _writes_enabled(self) -> bool:
        return True

    async def write_channel(self, channel_address: str, value, **kwargs):
        """Mock write operation."""
        self.write_calls.append((channel_address, value, kwargs))
        if self.canned_result is not None:
            return self.canned_result
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
            observed_value=value,
        )

    async def write_multiple_channels(self, operations, **kwargs):
        """Mock batch write — delegates to write_channel."""
        results = []
        for address, value in operations:
            results.append(await self.write_channel(address, value, **kwargs))
        return results

    async def read_channel(self, channel_address: str, **kwargs):
        """Mock read operation."""
        self.read_calls.append((channel_address, kwargs))
        channel_value = MagicMock()
        channel_value.value = 42.0
        return channel_value

    async def disconnect(self):
        """Mock disconnect."""
        self.disconnect_called = True

    # --- Unused abstract-method stubs -------------------------------------
    async def connect(self, config: dict[str, Any]) -> None: ...
    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        raise NotImplementedError

    async def subscribe(self, channel_address, callback) -> str:
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None: ...
    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        raise NotImplementedError

    async def validate_channel(self, channel_address: str) -> bool:
        raise NotImplementedError


@pytest.fixture
def clear_runtime_state():
    """Clear runtime module state before each test."""
    import osprey.runtime as runtime

    runtime._runtime_connector = None
    runtime._limits_validator = None
    yield
    # Cleanup after test
    runtime._runtime_connector = None
    runtime._limits_validator = None


def test_write_channel_success(clear_runtime_state):
    """Test write_channel with successful write."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV", 42.0)

        assert len(mock_connector.write_calls) == 1
        assert mock_connector.write_calls[0][0] == "TEST:PV"
        assert mock_connector.write_calls[0][1] == 42.0


def test_write_channel_failure(clear_runtime_state):
    """A write the control system could not deliver raises ChannelWriteFailedError."""
    mock_connector = MockConnector(
        canned_result=ChannelWriteResult(
            channel_address="TEST:PV",
            value_written=42.0,
            outcome=WriteOutcome.FAILED,
            error_message="Write failed",
        )
    )

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        with pytest.raises(ChannelWriteFailedError, match="Write failed") as excinfo:
            write_channel("TEST:PV", 42.0)

        assert excinfo.value.reason == "FAILED"


class TestRuntimeWriteConfirmation:
    """The runtime must not report success for a write that did not come back confirmed.

    ``write_channel_checked`` raises for every outcome except ``confirmed`` and
    ``unrequested``; agent-authored Python calling the runtime has to see a
    ``mismatch``, a ``failed`` write, or an ``unconfirmed`` re-read as a
    failure, never as a silent return.
    """

    def test_mismatch_raises(self, clear_runtime_state):
        """A MISMATCH outcome raises with both the sent and observed values."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV",
                value_written=42.0,
                outcome=WriteOutcome.MISMATCH,
                observed_value=0.0,
                error_message=None,
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            with pytest.raises(ChannelWriteFailedError) as excinfo:
                write_channel("TEST:PV", 42.0)

            assert excinfo.value.reason == "MISMATCH"
            assert excinfo.value.channel_address == "TEST:PV"
            assert excinfo.value.observed_value == 0.0
            # Both numbers must be nameable from the exception alone.
            assert "42.0" in str(excinfo.value)
            assert "0.0" in str(excinfo.value)

    def test_multi_channel_mismatch_raises(self, clear_runtime_state):
        """The multi-channel path enforces the same contract as the single path."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV1",
                value_written=1.0,
                outcome=WriteOutcome.MISMATCH,
                observed_value=9.0,
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            with pytest.raises(ChannelWriteFailedError) as excinfo:
                write_channels({"TEST:PV1": 1.0, "TEST:PV2": 2.0})

            assert excinfo.value.reason == "MISMATCH"

    def test_unconfirmed_raises(self, clear_runtime_state):
        """A confirming re-read that itself failed (UNCONFIRMED) still raises."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV",
                value_written=42.0,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message="readback timed out",
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            with pytest.raises(ChannelWriteFailedError) as excinfo:
                write_channel("TEST:PV", 42.0)

            assert excinfo.value.reason == "UNCONFIRMED"

    def test_refused_write_raises_blocked(self, clear_runtime_state):
        """A refusal (never attempted) surfaces as ChannelWriteBlockedError."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV",
                value_written=42.0,
                outcome=WriteOutcome.REFUSED,
                error_message="writes are disabled",
                refusal_reason="WRITES_DISABLED",
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            with pytest.raises(ChannelWriteBlockedError) as excinfo:
                write_channel("TEST:PV", 42.0)

            assert excinfo.value.reason == "WRITES_DISABLED"

    def test_confirmed_with_alarm_returns(self, clear_runtime_state):
        """CONFIRMED returns even in an alarm state -- alarm severity is reported,
        never raised on."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV",
                value_written=42.0,
                outcome=WriteOutcome.CONFIRMED,
                observed_value=42.0,
                alarm_severity=2,
                alarm_status="HIHI",
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            write_channel("TEST:PV", 42.0)  # must not raise

    def test_unrequested_returns(self, clear_runtime_state):
        """confirm=False means nothing was checked (UNREQUESTED): the write returns."""
        mock_connector = MockConnector(
            canned_result=ChannelWriteResult(
                channel_address="TEST:PV",
                value_written=42.0,
                outcome=WriteOutcome.UNREQUESTED,
            )
        )

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            write_channel("TEST:PV", 42.0)  # must not raise


def test_read_channel_success(clear_runtime_state):
    """Test read_channel with successful read."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        value = read_channel("TEST:PV")

        assert len(mock_connector.read_calls) == 1
        assert mock_connector.read_calls[0][0] == "TEST:PV"
        assert value == 42.0


def test_write_channels_bulk(clear_runtime_state):
    """Test write_channels bulk operation."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channels({"PV1": 1.0, "PV2": 2.0, "PV3": 3.0})

        assert len(mock_connector.write_calls) == 3
        assert mock_connector.write_calls[0][0] == "PV1"
        assert mock_connector.write_calls[1][0] == "PV2"
        assert mock_connector.write_calls[2][0] == "PV3"


@pytest.mark.asyncio
async def test_cleanup_runtime(clear_runtime_state):
    """Test cleanup_runtime properly releases resources."""
    import osprey.runtime as runtime

    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV", 42.0)

        assert runtime._runtime_connector is not None

        await cleanup_runtime()

        assert mock_connector.disconnect_called
        assert runtime._runtime_connector is None


def test_connector_reuse(clear_runtime_state):
    """Test that connector is created once and reused."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV1", 1.0)
        write_channel("TEST:PV2", 2.0)
        read_channel("TEST:PV3")

        mock_factory.assert_called_once()

        assert len(mock_connector.write_calls) == 2
        assert len(mock_connector.read_calls) == 1


@pytest.mark.asyncio
async def test_connector_recreated_after_cleanup(clear_runtime_state):
    """Test that connector is recreated after cleanup."""
    mock_connector1 = MockConnector()
    mock_connector2 = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.side_effect = [mock_connector1, mock_connector2]

        write_channel("TEST:PV", 1.0)
        assert len(mock_connector1.write_calls) == 1

        await cleanup_runtime()

        write_channel("TEST:PV", 2.0)
        assert len(mock_connector2.write_calls) == 1

        assert mock_factory.call_count == 2


class TestRuntimeLimitsValidation:
    """Tests that _limits_validator fires before the connector is called (I-2)."""

    @pytest.mark.asyncio
    async def test_limits_violation_raises_before_connector(self, clear_runtime_state):
        """When _limits_validator rejects a value, ChannelLimitsViolationError is raised
        and _get_connector() is never called."""
        import osprey.runtime as runtime

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"allow_unlisted_pvs": False})
        runtime._limits_validator = validator

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as mock_get_connector:
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await _write_channel_async("TEST:PV", 150.0)

            mock_get_connector.assert_not_called()
            assert exc_info.value.channel_address == "TEST:PV"
            assert exc_info.value.attempted_value == 150.0

    @pytest.mark.asyncio
    async def test_no_validator_calls_connector_normally(self, clear_runtime_state):
        """When _limits_validator is None, the connector is called normally."""
        import osprey.runtime as runtime

        assert runtime._limits_validator is None

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            await _write_channel_async("TEST:PV", 42.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][0] == "TEST:PV"
            assert mock_connector.write_calls[0][1] == 42.0

    @pytest.mark.asyncio
    async def test_valid_value_passes_through_to_connector(self, clear_runtime_state):
        """When _limits_validator approves the value, the connector write proceeds."""
        import osprey.runtime as runtime

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"allow_unlisted_pvs": False})
        runtime._limits_validator = validator

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            await _write_channel_async("TEST:PV", 50.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][1] == 50.0
