"""Standalone proof that write_channel_checked is RunEngine-consumable.

Task 3.2: this is a throwaway fixture that demonstrates the reference monitor's
denial contract composes with an ophyd_async device setter — WITHOUT importing
any of the dependent feature's device layer. A ~15-line stub setter calls
``write_channel_checked`` inside ``AsyncStatus.wrap``; awaiting the resulting
``AsyncStatus`` is exactly what a RunEngine does when it drives ``set()``.

``AsyncStatus.wrap`` decorates an ``async def`` and returns a callable producing
an ``AsyncStatus``. Awaiting that status runs the wrapped coroutine and re-raises
any exception it throws — so a refused write surfaces as a failed status, which
aborts the plan. A verified write returns and the status succeeds.

The connector helper speaks only the generic interface, so a minimal in-file
fake connector exercises it without any EPICS/DOOCS machinery.
"""

from typing import Any

import pytest

ophyd_async = pytest.importorskip("ophyd_async")
from ophyd_async.core import AsyncStatus  # noqa: E402

from osprey.connectors.control_system.base import (  # noqa: E402
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteVerification,
)
from osprey.errors import ChannelWriteBlockedError  # noqa: E402


class _FakeConnector(ControlSystemConnector):
    """Minimal concrete connector whose write_channel returns a canned result.

    Writes are forced enabled so the base __init_subclass__ guard passes straight
    through to our stub. Every non-write abstract method is an unused stub.
    """

    def __init__(self, result: ChannelWriteResult):
        self._canned_result = result

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


def _refusing_connector() -> _FakeConnector:
    """Connector whose write is REFUSED (writes disabled) — checked-write raises."""
    return _FakeConnector(
        ChannelWriteResult(
            channel_address="TEST:PV",
            value_written=1.0,
            success=False,
            blocked=True,
            refusal_reason="WRITES_DISABLED",
            error_message="writes are disabled",
        )
    )


def _verified_connector() -> _FakeConnector:
    """Connector whose write is VERIFIED — checked-write returns the result."""
    return _FakeConnector(
        ChannelWriteResult(
            channel_address="TEST:PV",
            value_written=1.0,
            success=True,
            verification=WriteVerification(
                level="readback", verified=True, readback_value=1.0, tolerance_used=0.1
            ),
        )
    )


class _StubConnectorSetter:
    """Minimal stand-in for a RunEngine device setter backed by the connector."""

    def __init__(self, connector, channel):
        self._connector = connector
        self._channel = channel

    @AsyncStatus.wrap
    async def set(self, value):
        await self._connector.write_channel_checked(self._channel, value)


@pytest.mark.asyncio
async def test_refused_write_aborts_the_status():
    """A refusal makes the AsyncStatus raise — a RunEngine plan would abort."""
    setter = _StubConnectorSetter(_refusing_connector(), "TEST:PV")

    # set() returns immediately; a RunEngine then AWAITS the status, which is
    # where the wrapped coroutine runs and re-raises the connector's refusal.
    status = setter.set(1.0)
    with pytest.raises(ChannelWriteBlockedError):
        await status


@pytest.mark.asyncio
async def test_verified_write_completes():
    """A verified write lets the status succeed — the setter is usable for a pass."""
    setter = _StubConnectorSetter(_verified_connector(), "TEST:PV")

    # No raise: awaiting the status completes, proving the happy path is consumable.
    await setter.set(1.0)
