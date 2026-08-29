"""A connector whose only job is to raise a chosen refusal, on cue.

The connector-host child builds whatever connector its init frame names, and
:func:`osprey_connectors.types.resolve_target` returns a dotted class path
verbatim for ``live``, so this module is reachable from a real child by asking
for ``tests.mcp_server._raising_connector.RaisingConnector``. It exists because
the two refusal classes that matter to the error envelope cannot both be
provoked out of the mock connector: a limits violation can (with a limits
database), but only ever with ``max_step`` and ``current_value`` unset, and a
``ChannelWriteBlockedError`` is raised by ``write_channel_checked``, one layer
above the method the child actually serves.

So the fields the envelope reads are set here explicitly, all of them at once,
and pinned as module constants the test asserts against on the far side of the
pipe. Nothing is randomised and nothing is derived: what the child raises is
what the parent must see, field for field.

Writes still pass through the base class's ``writes_enabled`` guard, so a child
using this connector needs a config with ``control_system.writes_enabled: true``
— otherwise the guard returns a blocked *result* and this connector's
``write_channel`` is never entered.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

#: Writing this channel raises a fully populated limits violation.
LIMITS_CHANNEL = "TEST:RAISE:LIMITS:SP"

#: Writing this channel raises a reference-monitor refusal.
BLOCKED_CHANNEL = "TEST:RAISE:BLOCKED:SP"

#: Every field ``ChannelLimitsViolationError`` carries, set to a distinct value
#: so a field that crossed as another field's value is visible as a failure.
LIMITS_FIELDS: dict[str, Any] = {
    "channel_address": LIMITS_CHANNEL,
    "attempted_value": 150.5,
    "violation_type": "MAX_EXCEEDED",
    "violation_reason": "Value 150.5 above maximum 100.0",
    "min_value": -25.0,
    "max_value": 100.0,
    "max_step": 5.0,
    "current_value": 12.25,
}

#: The refusal reason and its operator-facing text. The text is deliberately
#: not the class's default rendering: a custom message has to survive the
#: boundary verbatim, and one that matches the default would not prove it.
BLOCKED_REASON = "LIMITS"
BLOCKED_MESSAGE = "the reference monitor refused this write while the interlock was engaged"

#: What a channel with no instructions reads back as.
READ_VALUE = 3.25


def make_limits_violation() -> ChannelLimitsViolationError:
    """The exception this connector raises, built in the caller's process.

    The test builds the in-process comparison exception from the same
    constants, so "the same envelope" is a claim about the boundary rather
    than about two hand-copied literals.
    """
    fields = dict(LIMITS_FIELDS)
    return ChannelLimitsViolationError(value=fields.pop("attempted_value"), **fields)


def make_write_blocked() -> ChannelWriteBlockedError:
    """The refusal this connector raises, built in the caller's process."""
    return ChannelWriteBlockedError(BLOCKED_CHANNEL, BLOCKED_REASON, message=BLOCKED_MESSAGE)


class RaisingConnector(ControlSystemConnector):
    """A connector that reads normally and refuses writes on named channels."""

    async def connect(self, config: dict[str, Any] | None = None) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        return ChannelValue(
            value=READ_VALUE,
            timestamp=datetime.now(UTC),
            metadata=ChannelMetadata(units="A"),
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        return {
            address: await self.read_channel(address, timeout=timeout)
            for address in channel_addresses
        }

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        if channel_address == LIMITS_CHANNEL:
            raise make_limits_violation()
        if channel_address == BLOCKED_CHANNEL:
            raise make_write_blocked()
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            # This connector sends nothing anywhere and reads nothing back, so
            # the only honest word is the one for a write that was not checked.
            outcome=WriteOutcome.UNREQUESTED,
        )

    async def subscribe(self, channel_address: str, callback: Any) -> str:
        raise NotImplementedError("the raising connector serves reads and writes only")

    async def unsubscribe(self, subscription_id: str) -> None:
        raise NotImplementedError("the raising connector serves reads and writes only")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        return ChannelMetadata(units="A")

    async def validate_channel(self, channel_address: str) -> bool:
        return True
