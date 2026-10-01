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
    saved_observers = list(runtime._write_observers)
    runtime._write_observers.clear()
    yield
    # Cleanup after test
    runtime._runtime_connector = None
    runtime._limits_validator = None
    runtime._write_observers[:] = saved_observers


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_channel_success():
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


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_channel_failure():
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_mismatch_raises(self):
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_multi_channel_mismatch_raises(self):
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_unconfirmed_raises(self):
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_refused_write_raises_blocked(self):
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_confirmed_with_alarm_returns(self):
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

    @pytest.mark.usefixtures("clear_runtime_state")
    def test_unrequested_returns(self):
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


@pytest.mark.usefixtures("clear_runtime_state")
def test_read_channel_success():
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


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_channels_bulk():
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
@pytest.mark.usefixtures("clear_runtime_state")
async def test_cleanup_runtime():
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


@pytest.mark.usefixtures("clear_runtime_state")
def test_connector_reuse():
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
@pytest.mark.usefixtures("clear_runtime_state")
async def test_connector_recreated_after_cleanup():
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
    """Tests that _limits_validator fires before anything is written (I-2)."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_limits_violation_raises_before_the_write(self):
        """A rejected value raises and nothing is sent to the control system.

        The connector itself is acquired first — the safety net asks it for the
        fresh-read primitive the ``max_step`` check needs — but acquiring a
        connector writes nothing, and no write call is made.
        """
        import osprey.runtime as runtime

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"mode": "exclusive"})
        runtime._limits_validator = validator

        mock_connector = MockConnector()

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as mock_get_connector:
            mock_get_connector.return_value = mock_connector
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await _write_channel_async("TEST:PV", 150.0)

            assert mock_connector.write_calls == []
            assert exc_info.value.channel_address == "TEST:PV"
            assert exc_info.value.attempted_value == 150.0


class TestRuntimeStepCheckReader:
    """The safety net measures max_step with the connector's own client.

    The validator injected into ``osprey.runtime`` owns no control-system
    client. Without a reader every ``max_step`` channel would be refused
    ``STEP_CHECK_FAILED`` here, before the connector that can measure the step
    is ever asked to write.
    """

    @staticmethod
    def _validator():
        return LimitsValidator(
            {
                "TEST:PV": ChannelLimitsConfig(
                    channel_address="TEST:PV",
                    min_value=0.0,
                    max_value=100.0,
                    max_step=2.0,
                    writable=True,
                ),
                "OTHER:PV": ChannelLimitsConfig(
                    channel_address="OTHER:PV", min_value=0.0, max_value=100.0, writable=True
                ),
            },
            {"mode": "exclusive"},
        )

    @staticmethod
    def _connector(reads):
        class ReadingConnector(MockConnector):
            def _current_value_reader(self):
                def read_current(channel_address):
                    reads.append(channel_address)
                    return 50.0

                return read_current

        return ReadingConnector()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_step_within_limit_is_measured_and_written(self):
        import osprey.runtime as runtime

        runtime._limits_validator = self._validator()
        reads: list[str] = []
        connector = self._connector(reads)

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get_connector:
            get_connector.return_value = connector
            await _write_channel_async("TEST:PV", 51.0)

        assert reads == ["TEST:PV"]
        assert connector.write_calls[0][:2] == ("TEST:PV", 51.0)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_step_beyond_limit_is_refused_as_a_step_violation(self):
        import osprey.runtime as runtime

        runtime._limits_validator = self._validator()
        reads: list[str] = []
        connector = self._connector(reads)

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get_connector:
            get_connector.return_value = connector
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await _write_channel_async("TEST:PV", 90.0)

        assert exc_info.value.violation_type == "MAX_STEP_EXCEEDED"
        assert connector.write_calls == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_bulk_write_gets_the_same_reader(self):
        """write_channels' multi-channel branch validates with the connector's reader too."""
        import osprey.runtime as runtime
        from osprey.runtime import _write_channels_async

        runtime._limits_validator = self._validator()
        reads: list[str] = []
        connector = self._connector(reads)

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get_connector:
            get_connector.return_value = connector
            await _write_channels_async({"TEST:PV": 51.0, "OTHER:PV": 10.0})

        # Only the max_step channel needed a read.
        assert reads == ["TEST:PV"]
        assert len(connector.write_calls) == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_no_validator_calls_connector_normally(self):
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
    @pytest.mark.usefixtures("clear_runtime_state")
    async def test_valid_value_passes_through_to_connector(self):
        """When _limits_validator approves the value, the connector write proceeds."""
        import osprey.runtime as runtime

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"mode": "exclusive"})
        runtime._limits_validator = validator

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            await _write_channel_async("TEST:PV", 50.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][1] == 50.0


# ========================================================
# Write observer (private three-phase hook)
# ========================================================


class PerChannelConnector(MockConnector):
    """MockConnector whose outcome is chosen per channel address."""

    def __init__(self, outcomes: dict[str, WriteOutcome]):
        super().__init__()
        self.outcomes = outcomes

    async def write_channel(self, channel_address: str, value, **kwargs):
        self.write_calls.append((channel_address, value, kwargs))
        outcome = self.outcomes.get(channel_address, WriteOutcome.CONFIRMED)
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=outcome,
            observed_value=value if outcome is WriteOutcome.CONFIRMED else None,
            refusal_reason="WRITES_DISABLED" if outcome is WriteOutcome.REFUSED else None,
        )


def _record_observer():
    """Register a recording observer; return the list it appends (address, phase) to."""
    import osprey.runtime as runtime

    events: list[tuple[str, str]] = []

    def observer(address: str, phase: str) -> None:
        events.append((address, phase))

    runtime._register_write_observer(observer)
    return events


def _patched_factory(connector):
    return patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        return_value=connector,
    )


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_attempt_then_landed_on_confirmed():
    events = _record_observer()
    with _patched_factory(MockConnector()):
        write_channel("TEST:PV", 1.0)
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "landed")]


@pytest.mark.parametrize(
    "outcome",
    [WriteOutcome.MISMATCH, WriteOutcome.UNCONFIRMED, WriteOutcome.FAILED],
)
@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_sent_on_unverified_outcome_and_still_raises(outcome):
    """Every outcome where the value was sent but not verified is 'sent'.

    FAILED is classified 'sent' too: the value went out and the control system
    did not take it, so what the channel holds is not known to be unchanged.
    """
    events = _record_observer()
    with _patched_factory(PerChannelConnector({"TEST:PV": outcome})):
        with pytest.raises(ChannelWriteFailedError):
            write_channel("TEST:PV", 1.0)
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "sent")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_sent_on_unrequested():
    """confirm=False returns normally but nothing was verified: 'sent', not 'landed'."""
    events = _record_observer()
    with _patched_factory(PerChannelConnector({"TEST:PV": WriteOutcome.UNREQUESTED})):
        write_channel("TEST:PV", 1.0)
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "sent")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_refused_gets_attempt_only():
    events = _record_observer()
    with _patched_factory(PerChannelConnector({"TEST:PV": WriteOutcome.REFUSED})):
        with pytest.raises(ChannelWriteBlockedError):
            write_channel("TEST:PV", 1.0)
    assert events == [("TEST:PV", "attempt")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_not_called_when_limits_net_refuses():
    """'attempt' fires after the local limits net, so a net refusal notifies nothing."""
    import osprey.runtime as runtime

    runtime._limits_validator = LimitsValidator(
        {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=10.0, writable=True
            )
        },
        {"mode": "exclusive"},
    )
    events = _record_observer()
    connector = MockConnector()
    with _patched_factory(connector):
        with pytest.raises(ChannelLimitsViolationError):
            write_channel("TEST:PV", 99.0)
        with pytest.raises(ChannelLimitsViolationError):
            write_channels({"TEST:PV": 1.0, "OTHER:PV": 2.0})
    assert events == []
    assert connector.write_calls == []


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_not_called_when_target_pin_refuses():
    import osprey.runtime as runtime

    events = _record_observer()
    with patch.object(
        runtime, "_assert_target_pin", side_effect=runtime.ControlTargetChangedError("moved")
    ):
        with pytest.raises(runtime.ControlTargetChangedError):
            write_channel("TEST:PV", 1.0)
        with pytest.raises(runtime.ControlTargetChangedError):
            write_channels({"A:PV": 1.0, "B:PV": 2.0})
    assert events == []


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_attempt_fires_before_the_connector_write():
    seen_writes_at_attempt: list[int] = []
    connector = MockConnector()
    import osprey.runtime as runtime

    def observer(_address: str, phase: str) -> None:
        if phase == "attempt":
            seen_writes_at_attempt.append(len(connector.write_calls))

    runtime._register_write_observer(observer)
    with _patched_factory(connector):
        write_channel("TEST:PV", 1.0)
    assert seen_writes_at_attempt == [0]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_multi_channel_notifies_every_result_before_raising():
    """Three channels, the second mismatches: all three are notified, then it raises."""
    events = _record_observer()
    connector = PerChannelConnector(
        {
            "A:PV": WriteOutcome.CONFIRMED,
            "B:PV": WriteOutcome.MISMATCH,
            "C:PV": WriteOutcome.CONFIRMED,
        }
    )
    with _patched_factory(connector):
        with pytest.raises(ChannelWriteFailedError) as excinfo:
            write_channels({"A:PV": 1.0, "B:PV": 2.0, "C:PV": 3.0})
    assert excinfo.value.channel_address == "B:PV"
    assert events == [
        ("A:PV", "attempt"),
        ("B:PV", "attempt"),
        ("C:PV", "attempt"),
        ("A:PV", "landed"),
        ("B:PV", "sent"),
        ("C:PV", "landed"),
    ]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_multi_channel_raises_first_failure():
    events = _record_observer()
    connector = PerChannelConnector(
        {"A:PV": WriteOutcome.UNCONFIRMED, "B:PV": WriteOutcome.REFUSED}
    )
    with _patched_factory(connector):
        with pytest.raises(ChannelWriteFailedError) as excinfo:
            write_channels({"A:PV": 1.0, "B:PV": 2.0})
    assert excinfo.value.channel_address == "A:PV"
    # A refused channel reached no wire: it gets no outcome notification.
    assert events == [("A:PV", "attempt"), ("B:PV", "attempt"), ("A:PV", "sent")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_single_item_write_channels_notifies_once():
    """The one-item path delegates to the single-channel path: no double notify."""
    events = _record_observer()
    with _patched_factory(MockConnector()):
        write_channels({"TEST:PV": 1.0})
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "landed")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_registration_is_idempotent():
    import osprey.runtime as runtime

    events: list[tuple[str, str]] = []

    def observer(address: str, phase: str) -> None:
        events.append((address, phase))

    runtime._register_write_observer(observer)
    runtime._register_write_observer(observer)
    with _patched_factory(MockConnector()):
        write_channel("TEST:PV", 1.0)
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "landed")]


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_raising_observer_logs_warning_and_never_blocks():
    import osprey.runtime as runtime

    def broken(_address: str, _phase: str) -> None:
        raise RuntimeError("observer exploded")

    runtime._register_write_observer(broken)
    events = _record_observer()
    connector = MockConnector()
    with patch.object(runtime.logger, "warning") as warn:
        with _patched_factory(connector):
            write_channel("TEST:PV", 1.0)
    assert len(connector.write_calls) == 1
    # Later observers still run.
    assert events == [("TEST:PV", "attempt"), ("TEST:PV", "landed")]
    # One WARNING per failed notification (attempt + landed).
    assert warn.call_count == 2


@pytest.mark.usefixtures("clear_runtime_state")
def test_write_observer_raising_observer_does_not_mask_write_failure():
    import osprey.runtime as runtime

    def broken(_address: str, _phase: str) -> None:
        raise ValueError("boom")

    runtime._register_write_observer(broken)
    with _patched_factory(PerChannelConnector({"TEST:PV": WriteOutcome.MISMATCH})):
        with pytest.raises(ChannelWriteFailedError):
            write_channel("TEST:PV", 1.0)


def test_write_observer_api_is_private():
    import osprey.runtime as runtime

    assert "_register_write_observer" not in runtime.__all__
    assert not any("observer" in name for name in runtime.__all__)
