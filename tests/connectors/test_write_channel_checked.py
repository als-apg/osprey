"""Denial-contract tests for ControlSystemConnector.write_channel_checked.

write_channel_checked is the correctness primitive a plan device setter wraps.
It awaits the connector-agnostic write_channel and collapses the six
``WriteOutcome`` words into a single raise-or-return contract:

- ``refused`` -> raises ChannelWriteBlockedError, carrying the refusal_reason;
- ``failed`` / ``mismatch`` / ``unconfirmed`` -> raises ChannelWriteFailedError
  under that same word, uppercased;
- a native CA-layer ConnectionError/TimeoutError -> propagates unchanged;
- ``confirmed`` (in any alarm state) and ``unrequested`` -> return the
  ChannelWriteResult untouched.

The verdict is read off ``result.outcome`` alone — nothing here re-derives one.
The helper lives on the base class and speaks only the generic interface, so a
minimal in-file fake connector exercises it without any EPICS/DOOCS machinery.
"""

from typing import Any

import pytest

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)


class _FakeConnector(ControlSystemConnector):
    """Minimal concrete connector whose write_channel returns a canned result.

    write_channel returns ``self._canned_result`` or, if ``self._canned_exc`` is
    set, raises it. Writes are forced enabled so the base __init_subclass__ guard
    passes straight through to our stub (this test targets write_channel_checked,
    not the writes-disabled guard). Every other abstract method is an unused stub.
    """

    def __init__(
        self,
        result: ChannelWriteResult | None = None,
        exc: Exception | None = None,
    ):
        self._canned_result = result
        self._canned_exc = exc

    @property
    def _writes_enabled(self) -> bool:
        return True

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        if self._canned_exc is not None:
            raise self._canned_exc
        return self._canned_result

    # --- Unused abstract-method stubs -------------------------------------
    async def connect(self, config: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...
    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        raise NotImplementedError

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


def _result(**overrides) -> ChannelWriteResult:
    """Build a ChannelWriteResult with sensible defaults overridable per test."""
    base: dict[str, Any] = {
        "channel_address": "TEST:PV",
        "value_written": 42.0,
        "outcome": WriteOutcome.CONFIRMED,
    }
    base.update(overrides)
    return ChannelWriteResult(**base)


class TestRefusals:
    """``refused`` — nothing was written -> ChannelWriteBlockedError."""

    @pytest.mark.asyncio
    async def test_writes_disabled_result_raises_blocked(self):
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.REFUSED,
                refusal_reason="WRITES_DISABLED",
                error_message="writes are disabled",
            )
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "WRITES_DISABLED"
        assert excinfo.value.channel_address == "TEST:PV"

    @pytest.mark.asyncio
    async def test_validation_error_result_raises_blocked(self):
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.REFUSED,
                refusal_reason="VALIDATION_ERROR",
                error_message="validate() raised",
            )
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "VALIDATION_ERROR"
        assert excinfo.value.channel_address == "TEST:PV"

    @pytest.mark.asyncio
    async def test_refusal_without_a_reason_falls_back_to_writes_disabled(self):
        """A refusal that names no policy still raises under a reason code."""
        connector = _FakeConnector(
            result=_result(outcome=WriteOutcome.REFUSED, error_message="refused")
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "WRITES_DISABLED"

    @pytest.mark.asyncio
    async def test_limits_violation_normalized_to_blocked(self):
        """write_channel RAISES ChannelLimitsViolationError -> helper normalizes
        it to ChannelWriteBlockedError(reason="LIMITS") with the original chained.
        """
        violation = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=999.0,
            violation_type="max_value",
            violation_reason="above max",
        )
        connector = _FakeConnector(exc=violation)

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 999.0)

        assert excinfo.value.reason == "LIMITS"
        assert excinfo.value.channel_address == "TEST:PV"
        assert excinfo.value.__cause__ is violation


class TestFailures:
    """Sent but not confirmed -> ChannelWriteFailedError under the outcome word."""

    @pytest.mark.asyncio
    async def test_failed_result_raises_failed(self):
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.FAILED,
                error_message="Failed to write TEST:PV",
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "FAILED"
        assert excinfo.value.outcome is WriteOutcome.FAILED
        assert excinfo.value.channel_address == "TEST:PV"
        assert "Failed to write TEST:PV" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_mismatch_result_raises_failed_naming_both_values(self):
        """A mismatch carries no error_message: both numbers are the report."""
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.MISMATCH,
                value_written=42.0,
                observed_value=41.0,
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "MISMATCH"
        assert excinfo.value.outcome is WriteOutcome.MISMATCH
        assert excinfo.value.value_written == 42.0
        assert excinfo.value.observed_value == 41.0
        assert "42.0" in str(excinfo.value)
        assert "41.0" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unconfirmed_result_raises_failed(self):
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.UNCONFIRMED,
                error_message="confirming read failed to connect",
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "UNCONFIRMED"
        assert excinfo.value.outcome is WriteOutcome.UNCONFIRMED
        assert excinfo.value.observed_value is None
        assert "confirming read failed to connect" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_notes_are_never_read_by_the_raise_path(self):
        """``error_message`` is the only free text the raise path reads.

        ``notes`` is display-only; a failure that carries notes and no message
        gets the exception's own wording, not the note.
        """
        connector = _FakeConnector(
            result=_result(
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=None,
                notes="cannot be compared",
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "UNCONFIRMED"
        assert "cannot be compared" not in str(excinfo.value)


class TestPropagation:
    """Native CA-layer errors propagate unchanged — never reclassified."""

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self):
        connector = _FakeConnector(exc=ConnectionError("gateway down"))

        with pytest.raises(ConnectionError):
            await connector.write_channel_checked("TEST:PV", 42.0)

    @pytest.mark.asyncio
    async def test_timeout_error_propagates(self):
        connector = _FakeConnector(exc=TimeoutError("caput timed out"))

        with pytest.raises(TimeoutError):
            await connector.write_channel_checked("TEST:PV", 42.0)


class TestReturns:
    """``confirmed`` and ``unrequested`` return the result unchanged."""

    @pytest.mark.asyncio
    async def test_confirmed_returns_result(self):
        result = _result(outcome=WriteOutcome.CONFIRMED, observed_value=42.0)
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_confirmed_in_alarm_returns_result(self):
        """Alarm state is reported with the write, never a reason to raise."""
        result = _result(
            outcome=WriteOutcome.CONFIRMED,
            observed_value=42.0,
            alarm_status="HIHI",
            alarm_severity=2,
        )
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result
        assert returned.alarm_severity == 2

    @pytest.mark.asyncio
    async def test_unrequested_returns_result(self):
        """confirm=False: nothing was checked, and nothing is claimed."""
        result = _result(outcome=WriteOutcome.UNREQUESTED)
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_kwargs_pass_through_to_write_channel(self):
        """confirm/timeout reach write_channel verbatim."""
        seen: dict[str, Any] = {}

        result = _result(outcome=WriteOutcome.UNREQUESTED)
        connector = _FakeConnector(result=result)

        async def _capture(channel_address, value, **kwargs):
            seen["channel_address"] = channel_address
            seen["value"] = value
            seen.update(kwargs)
            return result

        connector.write_channel = _capture  # type: ignore[method-assign]

        returned = await connector.write_channel_checked(
            "TEST:PV", 42.0, confirm=False, timeout=3.0
        )

        assert returned is result
        assert seen == {
            "channel_address": "TEST:PV",
            "value": 42.0,
            "confirm": False,
            "timeout": 3.0,
        }
