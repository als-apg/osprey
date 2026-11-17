"""A connector whose only job is to refuse a chosen write, on cue.

The connector-host child builds whatever connector its init frame names, and
:func:`osprey_connectors.targets.resolve_target` returns a dotted class path
verbatim for ``live``, so this module is reachable from a real child by asking
for ``tests.mcp_server._raising_connector.RaisingConnector``. It exists because
the two refusals that matter to the error envelope cannot both be provoked out
of the mock connector: a limits refusal can (with a limits database), but only
ever with ``max_step`` and ``current_value`` unset, and a
``ChannelWriteBlockedError`` is raised by ``write_channel_checked``, one layer
above the method the child actually serves.

So the limits this connector checks against refuse :data:`LIMITS_CHANNEL` with
every field the envelope reads set, all at once, and pinned as module
constants the test asserts against on the far side of the pipe; the refusal
comes back as a ``refused`` result carrying that violation. A send to
:data:`BLOCKED_CHANNEL` raises the refusal error itself. Nothing is randomised
and nothing is derived: what the child answers is what the parent must see,
field for field.

Writes still pass through the base class's write verdict, so a child using
this connector needs a config whose tier is ``control_system.writes: allowed``
and its target's switch on — otherwise the guard returns a blocked *result*
and neither the limits nor the send are reached.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ControlSystemConnector,
    SendReport,
)
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    LimitsViolation,
)

#: Writing this channel raises a fully populated limits violation.
LIMITS_CHANNEL = "TEST:RAISE:LIMITS:SP"

#: Writing this channel raises a reference-monitor refusal.
BLOCKED_CHANNEL = "TEST:RAISE:BLOCKED:SP"

#: Every field ``ChannelLimitsViolationError`` carries, set to a distinct value
#: so a field that crossed as another field's value is visible as a failure.
LIMITS_FIELDS: dict[str, Any] = {
    "address": LIMITS_CHANNEL,
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
    """The violation this connector's limits raise, built in the caller's process.

    The test builds the in-process comparison exception from the same
    constants, so "the same envelope" is a claim about the boundary rather
    than about two hand-copied literals.
    """
    fields = dict(LIMITS_FIELDS)
    return ChannelLimitsViolationError(value=fields.pop("attempted_value"), **fields)


def expected_violation() -> LimitsViolation:
    """The violation a refused :data:`LIMITS_CHANNEL` result carries."""
    return LimitsViolation.from_error(make_limits_violation())


class _Limits:
    """The limits this connector checks a write against: :data:`LIMITS_CHANNEL` is refused.

    Every write is left unconfirmed by policy: this connector sends nothing
    anywhere and reads nothing back, so the only honest word is the one for a
    write that was not checked.
    """

    def validate(self, address: str, value: Any, *, read_current: Any = None) -> None:  # noqa: ARG002 - the validator's signature
        if address == LIMITS_CHANNEL:
            raise make_limits_violation()

    def resolve_confirm(self, address: str) -> bool:  # noqa: ARG002 - the validator's signature
        return False

    def measures_step(self, address: str) -> bool:  # noqa: ARG002 - the validator's signature
        return False


def make_write_blocked() -> ChannelWriteBlockedError:
    """The refusal this connector raises, built in the caller's process."""
    return ChannelWriteBlockedError(BLOCKED_CHANNEL, BLOCKED_REASON, message=BLOCKED_MESSAGE)


class RaisingConnector(ControlSystemConnector):
    """A connector that reads normally and refuses writes on named channels."""

    async def _open(self, config: dict[str, Any] | None = None) -> None:  # noqa: ARG002 - the connector interface fixes the signature
        return None

    async def disconnect(self) -> None:
        return None

    async def read_channel(
        self,
        address: str,
        *,
        timeout_s: float | None = None,  # noqa: ARG002 - the connector interface fixes the signature
    ) -> ChannelValue:
        return ChannelValue(
            address=address,
            value=READ_VALUE,
            sampled_at=datetime.now(UTC),
            metadata=ChannelMetadata(units="A"),
        )

    def _validator(self) -> Any:
        return _Limits()

    async def _send(
        self,
        address: str,
        value: Any,  # noqa: ARG002 - the send hook fixes the signature
        *,
        timeout_s: float,  # noqa: ARG002 - the send hook fixes the signature
        wait: bool,  # noqa: ARG002 - the send hook fixes the signature
    ) -> SendReport:
        if address == BLOCKED_CHANNEL:
            raise make_write_blocked()
        return SendReport()

    async def subscribe(self, address: str, callback: Any, *, on_connection: Any = None) -> str:
        raise NotImplementedError("the raising connector serves reads and writes only")

    async def unsubscribe(self, subscription_id: str) -> None:
        raise NotImplementedError("the raising connector serves reads and writes only")

    async def get_metadata(self, address: str) -> ChannelMetadata:  # noqa: ARG002 - the connector interface fixes the signature
        return ChannelMetadata(units="A")

    async def validate_channel(self, address: str) -> bool:  # noqa: ARG002 - the connector interface fixes the signature
        return True
