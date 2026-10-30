"""Error classification and handling for the Osprey Framework.

Stability: the exception class names and ``reason`` codes defined here are
public API. Removing or renaming any of them is a major version bump; adding
new ones is a minor version bump.

This module is the connector failure vocabulary.

Three contract roots partition what a connector raises when a call fails, and
no class lies under two of them: :class:`ChannelError`, a failure about one
named channel that leaves the connector in place; every single-channel failure
(a limits violation, a refused write, a failed write, an address that is not a
channel, an unreachable channel) is one, and its ``channel_address`` names the
channel. :class:`ConnectorUnavailableError`, the connector itself is gone and
its supervisor replaces it. :class:`ConnectorInternalError`, a connector bug
that nothing recovers from. :class:`ChannelReadFailedError` names several
channels and lies under no root.

:class:`RefusalReason` names why a write was refused, who refused it and what
to do about it. ``ChannelWriteFailedError``'s reason codes are the uppercased
``WriteOutcome`` words for the outcomes that raise, so the reason a caller
branches on and the outcome the result reports are one vocabulary.

The connector-host IPC registry is derived from this module: every class in
``__all__`` under a contract root crosses the process boundary as itself,
with its channel, so a public failure needs no registry entry of its own.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from osprey_connectors.control_system import WriteOutcome


__all__ = [
    "NOT_A_CHANNEL",
    "RAW_CLIENT_WRITE_MARKER",
    "BuildProfileError",
    "ChannelError",
    "ChannelLimitsViolationError",
    "ChannelNotFoundError",
    "ChannelReadFailedError",
    "ChannelUnreachableError",
    "ChannelWriteBlockedError",
    "ChannelWriteFailedError",
    "ClientLibraryRefusedError",
    "ConfigurationError",
    "ConnectorInternalError",
    "ConnectorUnavailableError",
    "RefusalReason",
    "RegistryError",
    "raw_client_write_message",
]


class ChannelError(Exception):
    """A failure about one named channel; the connector that raised it stays in place.

    ``channel_address`` names the channel, in process and across the
    connector-host boundary alike. No supervisor recovers from a channel
    error: it is reported against its channel and nothing is restarted. It has
    no builtin base that another root shares, so no ``except ConnectionError``
    or ``except TimeoutError`` reads it as a lost connector.
    """

    def __init__(self, channel_address: str, message: str):
        self.channel_address = channel_address
        super().__init__(message)


class ChannelLimitsViolationError(ChannelError):
    """Raised when a channel write violates configured safety limits.

    Covers min/max range violations, read-only channel writes,
    excessive step sizes, and writes to unlisted channels.
    """

    def __init__(
        self,
        channel_address: str,
        value: Any,
        violation_type: str,
        violation_reason: str,
        min_value: float | None = None,
        max_value: float | None = None,
        max_step: float | None = None,
        current_value: Any | None = None,
    ):
        self.channel_address = channel_address
        self.attempted_value = value
        self.violation_type = violation_type
        self.violation_reason = violation_reason
        self.min_value = min_value
        self.max_value = max_value
        self.max_step = max_step
        self.current_value = current_value

        super().__init__(channel_address, self._format_violation_message())

    def _format_violation_message(self) -> str:
        """Format a user-friendly violation message with all relevant details."""
        msg = [
            "\n" + "=" * 70,
            "CHANNEL LIMITS VIOLATION DETECTED",
            "=" * 70,
            f"Channel Address: {self.channel_address}",
            f"Attempted Value: {self.attempted_value}",
        ]

        if self.current_value is not None:
            msg.append(f"Current Value: {self.current_value}")

        msg.append(f"Violation: {self.violation_reason}")

        if self.min_value is not None or self.max_value is not None:
            msg.append(f"Allowed Range: [{self.min_value}, {self.max_value}]")

        if self.max_step is not None:
            msg.append(f"Maximum Step Size: {self.max_step}")

        msg.extend(
            [
                "=" * 70,
                "⚠️  Write operation BLOCKED for safety",
                "=" * 70,
            ]
        )

        return "\n".join(msg)


_REFERENCE_MONITOR = "the reference monitor"
_CONTROL_SYSTEM = "the control system"
_NEVER_SENT = (
    "This write was refused on policy grounds; it was never sent to the control system.",
)


class RefusalReason(StrEnum):
    """Why a write was refused, with who refused it and what to do about it.

    Each member's value is the reason word a result and an exception carry.
    :attr:`refuser` names who refused a write for the reason, and is ``None``
    for ``CHANNEL_NOT_FOUND``: nothing was denied, the address is not a
    channel. :attr:`remedy` holds the reason's own suggestion sentences, in
    the order a message lists them; it is empty for ``CHANNEL_NOT_FOUND``,
    whose answer is rendered as "not a channel".

    ``RAW_CLIENT_WRITE`` means user code called a control-system client
    library's put directly instead of going through
    ``osprey.runtime.write_channel``; nothing was sent.
    ``CONTROL_SYSTEM_REFUSED`` is the one reason under which the write was
    sent: the control system was asked and denied it.
    """

    refuser: str | None
    remedy: tuple[str, ...]

    def __new__(
        cls, value: str, refuser: str | None = None, remedy: tuple[str, ...] = ()
    ) -> "RefusalReason":
        member = str.__new__(cls, value)
        member._value_ = value
        member.refuser = refuser
        member.remedy = remedy
        return member

    WRITES_DISABLED = ("WRITES_DISABLED", _REFERENCE_MONITOR, _NEVER_SENT)
    LIMITS = ("LIMITS", _REFERENCE_MONITOR, _NEVER_SENT)
    VALIDATION_ERROR = ("VALIDATION_ERROR", _REFERENCE_MONITOR, _NEVER_SENT)
    CONTROL_SYSTEM_REFUSED = (
        "CONTROL_SYSTEM_REFUSED",
        _CONTROL_SYSTEM,
        (
            "The control system itself denied this write (access security); no value was written.",
            "Report it to the operator: write access for this channel is granted "
            "by the control system, not by OSPREY.",
        ),
    )
    RAW_CLIENT_WRITE = ("RAW_CLIENT_WRITE", _REFERENCE_MONITOR, _NEVER_SENT)
    CHANNEL_NOT_FOUND = ("CHANNEL_NOT_FOUND", None, ())


def _known_reason(reason: str | None) -> RefusalReason | None:
    """Return the :class:`RefusalReason` member ``reason`` names, or None."""
    if reason is None:
        return None
    try:
        return RefusalReason(reason)
    except ValueError:
        return None


class ChannelWriteBlockedError(ChannelError):
    """Raised when a channel write was refused and no value was written.

    Distinct from ChannelWriteFailedError, which means the write was attempted
    and its outcome is a failure rather than a denial.

    ``reason`` is a :class:`RefusalReason` member when the reason is a known
    one, and then names its refuser and remedy. An unknown reason, or no reason
    at all, is kept as given and names neither: no refuser is guessed.
    """

    @classmethod
    def refuser(cls, reason: str | None) -> str | None:
        """Return who refused a write for ``reason``, or None when no one is known."""
        known = _known_reason(reason)
        return known.refuser if known is not None else None

    @classmethod
    def remedy(cls, reason: str | None) -> tuple[str, ...]:
        """Return the suggestion sentences for ``reason``, or ``()`` when none is known."""
        known = _known_reason(reason)
        return known.remedy if known is not None else ()

    def __init__(self, channel_address: str, reason: str | None, message: str | None = None):
        self.channel_address = channel_address
        self.reason: str | None = _known_reason(reason) or reason
        refuser = self.refuser(reason)
        if refuser is not None:
            fallback = f"Write to '{channel_address}' refused by {refuser} ({reason})"
        elif reason == ChannelNotFoundError.reason:
            fallback = str(ChannelNotFoundError(channel_address))
        elif reason is None:
            fallback = f"Write to '{channel_address}' refused"
        else:
            fallback = f"Write to '{channel_address}' refused ({reason})"
        super().__init__(channel_address, message or fallback)


#: Shared substring of every raw-client-write refusal message. Refusals raised
#: inside an executor subprocess reach the parent only as stderr text, so the
#: parent recognises them by this marker; a test pins the message to it.
RAW_CLIENT_WRITE_MARKER = "raw client write refused"


def raw_client_write_message(address: str) -> str:
    """Return the refusal text for a direct client-library write to ``address``.

    ``address`` may be ``"<unknown>"`` when the refused call does not expose a
    channel name (a low-level handle put, for instance).
    """
    return (
        f"{RAW_CLIENT_WRITE_MARKER}: write to '{address}' bypassed the reference "
        "monitor. Use osprey.runtime.write_channel(address, value) or "
        "osprey.runtime.write_channels({address: value, ...}) so limits and "
        "approval apply."
    )


class ClientLibraryRefusedError(RuntimeError):
    """Raised when a process that may hold no control-system client loads one.

    reason is "RAW_CLIENT_IMPORT": user code imported a control-system client
    package, or loaded a client shared library, in a process whose only route
    to the machine is ``osprey.runtime``. Nothing was loaded and no channel was
    touched. ``package`` names the refused top-level package and ``library``
    the refused shared library; one of the two is set.

    A ``RuntimeError`` rather than an ``ImportError``: code that treats a
    missing optional client as absent (``except ImportError``) must not read a
    refusal as "not installed" and carry on down another route.
    """

    #: Shared substring of every client refusal message.
    MARKER = "control-system client refused"

    def __init__(self, *, package: str | None = None, library: str | None = None):
        self.package = package
        self.library = library
        self.reason = "RAW_CLIENT_IMPORT"
        if library is not None:
            what = f"loading '{library}' refused: it is a control-system client library"
        else:
            what = f"'{package}' is not loaded in a notebook kernel"
        super().__init__(
            f"{self.MARKER}: {what}. A kernel reaches the machine only through "
            "osprey.runtime, which follows the control target: read_channel(address), "
            "write_channel(address, value), write_channels({address: value, ...})."
        )


class ChannelWriteFailedError(ChannelError):
    """Raised when a channel write was attempted but was not confirmed.

    The control system was asked to write but the write failed, the channel now
    holds a different value, or the write was sent but not acknowledged in time
    or the confirming re-read itself failed.
    Distinct from ChannelWriteBlockedError (a refusal — no value was written).
    A scan consumer must abort on this.

    reason is one of: "FAILED", "MISMATCH", "UNCONFIRMED" — the ``WriteOutcome``
    words that raise. All three are protocol-neutral on purpose: the same codes
    describe an EPICS, DOOCS, or simulated write.

    ``outcome``, ``value_written`` and ``observed_value`` carry the detail the
    result held, so a caller that only sees the exception can still say what was
    sent and what the channel holds. ``outcome`` is a ``WriteOutcome`` member on
    either side of the connector-host boundary: the plain string the wire
    carries is coerced on construction, as ``ChannelWriteResult`` coerces its own.
    """

    _VALID_REASONS = ("FAILED", "MISMATCH", "UNCONFIRMED")

    def __init__(
        self,
        channel_address: str,
        reason: str,
        message: str | None = None,
        *,
        outcome: "WriteOutcome | None" = None,
        value_written: Any = None,
        observed_value: Any = None,
    ):
        if outcome is not None:
            from osprey_connectors.control_system import WriteOutcome

            if not isinstance(outcome, WriteOutcome):
                outcome = WriteOutcome(outcome)
        self.channel_address = channel_address
        self.reason = reason
        self.outcome = outcome
        self.value_written = value_written
        self.observed_value = observed_value
        super().__init__(channel_address, message or self._default_message())

    def _default_message(self) -> str:
        """Name both values for a mismatch — the difference is the whole report.

        Only when there is a value to name: ``value_written`` is optional, and
        the producer (``raise_for_write_result``) always supplies it, so a
        mismatch raised without it came from somewhere that has no detail to
        report. Saying "sent None, channel holds None" would dress that up as a
        reading, so the generic wording is used instead.
        """
        if self.reason == "MISMATCH" and self.value_written is not None:
            return (
                f"Write to '{self.channel_address}' not confirmed (MISMATCH): "
                f"sent {self.value_written}, channel holds {self.observed_value}"
            )
        return f"Write to '{self.channel_address}' failed ({self.reason})"


class ChannelReadFailedError(Exception):
    """Raised when a batch read or a settle's readback read could not produce a value.

    A channel counts as failed when the connector's result leaves it out (a
    connector drops a channel whose read raised) or reports it with the value
    ``None`` (the EPICS-family connectors' answer to a read timeout).

    ``addresses`` names every failed channel, each once, in the order the
    caller asked for them, so a caller can report or retry all of them rather
    than only the first.

    ``causes`` maps a failed channel to the exception its own read raised, so a
    timeout, a refused connection and a missing channel stay distinguishable.
    A failed channel whose read reported no value without raising has no
    entry.
    """

    def __init__(
        self,
        addresses: "Sequence[str]",
        message: str | None = None,
        *,
        causes: "Mapping[str, BaseException] | None" = None,
    ):
        self.addresses = list(addresses)
        self.causes: dict[str, BaseException] = dict(causes or {})
        if not message:
            named = [
                f"{address} ({type(self.causes[address]).__name__}: {self.causes[address]})"
                if address in self.causes
                else address
                for address in self.addresses
            ]
            message = f"Read failed for {len(self.addresses)} channel(s): {', '.join(named)}"
        super().__init__(message)


#: What an address outside the facility is, in every message that names one.
NOT_A_CHANNEL = "not a channel of this facility"


class ChannelNotFoundError(ChannelError, LookupError):
    """Raised when an address is not a channel of the facility a connector serves.

    Only a connector that holds the facility's whole address set raises it. A
    connector that cannot tell an unknown name from an unreachable one (a
    Channel Access search that finds nothing times out) raises
    :class:`ChannelUnreachableError` instead. A write to such an address is not
    raised: it comes back ``refused`` with ``refusal_reason`` :attr:`reason`.
    """

    #: The refusal reason a write to an address outside the facility carries.
    reason = RefusalReason.CHANNEL_NOT_FOUND

    def __init__(self, channel_address: str):
        super().__init__(channel_address, f"{channel_address} is {NOT_A_CHANNEL}")


class ChannelUnreachableError(ChannelError):
    """One channel could not be reached: it was lost, or it did not answer in time.

    The connector that raised it is healthy and keeps serving every other
    channel, so nothing is replaced for it; the failure is that channel's
    device, not the control system as a whole. Its text names the address.
    """

    def __init__(self, channel_address: str, message: str | None = None):
        super().__init__(channel_address, message or f"{channel_address} did not answer")


class ConnectorUnavailableError(ConnectionError):
    """The connector itself is gone: its transport, context or process is lost.

    Every channel it serves fails alike until it is replaced, and the
    supervisor that owns the connector's lifetime replaces it at most once per
    instance. ``recovery`` is what that supervisor did, one
    ``{"action", "text"}`` record per step, set by the supervisor and by
    nothing else; it is empty until a supervisor has acted.
    """

    def __init__(self, message: str, recovery: "Sequence[Mapping[str, str]]" = ()):
        self.recovery: tuple[Mapping[str, str], ...] = tuple(recovery)
        super().__init__(message)


class ConnectorInternalError(RuntimeError):
    """Raised in the parent for something the connector-host child sent outside the contract.

    It has two sources, and both fail one call only. A connector bug: the
    child raised an exception the IPC registry does not carry typed, and the
    message carries the child's ``repr``. An unrebuildable reply: the child
    answered with a value the parent cannot rebuild into its declared type.
    Neither is a reachability answer, and nothing about the connector or the
    child is implied, so no supervisor recovers from it.

    It is a ``RuntimeError`` and never a ``ConnectionError`` or ``OSError``, so
    that no ``except ConnectionError`` or ``except OSError`` reads a connector
    bug as an unreachable control system.

    ``class_tag`` names the class the child sent: the exception's class for a
    connector bug, the value's class for an unrebuildable reply.
    """

    def __init__(self, message: str, class_tag: str | None = None):
        self.class_tag = class_tag
        super().__init__(message)


class RegistryError(Exception):
    """Exception for registry-related errors.

    Raised when issues occur with component registration, lookup, or
    management within the framework's registry system.
    """


class ConfigurationError(Exception):
    """Exception for configuration-related errors.

    Raised when configuration files are invalid, missing required settings,
    or contain incompatible values that prevent proper system operation.
    """


class BuildProfileError(ConfigurationError):
    """Raised when a build profile is invalid or missing required files."""
