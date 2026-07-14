"""Denial-contract tests for ControlSystemConnector.write_channel_checked.

Task 3.1: write_channel_checked is the correctness primitive a scan device
setter wraps. It awaits the connector-agnostic write_channel and collapses its
four documented outcomes into a single raise-or-return contract:

- a REFUSAL (writes disabled, limits, or non-limits validation) -> raises
  ChannelWriteBlockedError;
- an ATTEMPTED-but-failed / unverified write (caput False, readback mismatch,
  readback failed) -> raises ChannelWriteFailedError;
- a native CA-layer ConnectionError/TimeoutError -> propagates unchanged;
- a verified successful write (or an accepted level="none" success) -> returns
  the ChannelWriteResult untouched.

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
    WriteVerification,
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
        verification_level: str = "callback",
        tolerance: float | None = None,
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
    base = {
        "channel_address": "TEST:PV",
        "value_written": 42.0,
        "success": True,
    }
    base.update(overrides)
    return ChannelWriteResult(**base)


class TestRefusals:
    """Refused writes (never attempted) -> ChannelWriteBlockedError."""

    @pytest.mark.asyncio
    async def test_writes_disabled_result_raises_blocked(self):
        connector = _FakeConnector(
            result=_result(
                success=False,
                blocked=True,
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
                success=False,
                blocked=True,
                refusal_reason="VALIDATION_ERROR",
                error_message="validate() raised",
            )
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "VALIDATION_ERROR"
        assert excinfo.value.channel_address == "TEST:PV"

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
    """Attempted-but-failed / unverified writes -> ChannelWriteFailedError."""

    @pytest.mark.asyncio
    async def test_caput_failed_result_raises_failed(self):
        connector = _FakeConnector(
            result=_result(
                success=False,
                blocked=False,
                error_message="Failed to write TEST:PV",
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "CAPUT_FAILED"
        assert excinfo.value.channel_address == "TEST:PV"

    @pytest.mark.asyncio
    async def test_readback_unverified_result_raises_failed(self):
        connector = _FakeConnector(
            result=_result(
                success=True,
                verification=WriteVerification(
                    level="readback",
                    verified=False,
                    readback_value=41.0,
                    tolerance_used=0.1,
                    notes="readback 41.0 != 42.0",
                ),
                error_message="readback mismatch",
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "READBACK_UNVERIFIED"
        assert excinfo.value.channel_address == "TEST:PV"

    @pytest.mark.asyncio
    async def test_readback_unverified_uses_notes_when_no_error_message(self):
        """When error_message is absent, the verification notes carry the detail."""
        connector = _FakeConnector(
            result=_result(
                success=True,
                verification=WriteVerification(
                    level="readback",
                    verified=False,
                    notes="readback failed to connect",
                ),
                error_message=None,
            )
        )

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await connector.write_channel_checked("TEST:PV", 42.0)

        assert excinfo.value.reason == "READBACK_UNVERIFIED"
        assert "readback failed to connect" in str(excinfo.value)


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


class TestSuccess:
    """Verified successes (and accepted level="none") return unchanged."""

    @pytest.mark.asyncio
    async def test_verified_readback_success_returns_result(self):
        result = _result(
            success=True,
            verification=WriteVerification(
                level="readback",
                verified=True,
                readback_value=42.0,
                tolerance_used=0.1,
            ),
        )
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_verified_callback_success_returns_result(self):
        result = _result(
            success=True,
            verification=WriteVerification(level="callback", verified=True),
        )
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_level_none_success_returns_despite_unverified(self):
        """level="none" means no verification was requested: an unverified
        success is ACCEPTED and returned (the v.level != "none" guard).
        """
        result = _result(
            success=True,
            verification=WriteVerification(level="none", verified=False),
        )
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_success_with_no_verification_returns(self):
        """A bare success with verification=None returns unchanged."""
        result = _result(success=True, verification=None)
        connector = _FakeConnector(result=result)

        returned = await connector.write_channel_checked("TEST:PV", 42.0)

        assert returned is result

    @pytest.mark.asyncio
    async def test_kwargs_pass_through_to_write_channel(self):
        """verification_level/tolerance/timeout reach write_channel verbatim."""
        seen: dict[str, Any] = {}

        result = _result(success=True)
        connector = _FakeConnector(result=result)

        async def _capture(channel_address, value, **kwargs):
            seen["channel_address"] = channel_address
            seen["value"] = value
            seen.update(kwargs)
            return result

        connector.write_channel = _capture  # type: ignore[method-assign]

        returned = await connector.write_channel_checked(
            "TEST:PV", 42.0, verification_level="readback", tolerance=0.5, timeout=3.0
        )

        assert returned is result
        assert seen == {
            "channel_address": "TEST:PV",
            "value": 42.0,
            "verification_level": "readback",
            "tolerance": 0.5,
            "timeout": 3.0,
        }
