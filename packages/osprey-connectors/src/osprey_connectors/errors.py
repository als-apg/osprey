"""Error classification and handling for the Osprey Framework.

Stability: the exception class names and ``reason`` codes defined here are
public API. Removing or renaming any of them is a major version bump; adding
new ones is a minor version bump.

This module is the connector failure vocabulary.

:class:`ConnectorsError` is the one base: every public exception the package
raises inherits it, so ``except ConnectorsError`` catches anything the package
raised on purpose. A class keeps a builtin second base where callers read the
builtin, and the base is part of what the class means: a lost connector is a
``ConnectionError``, a missed deadline a ``TimeoutError``, a bad config value
a ``ValueError``. The hierarchy, with each builtin second base::

    ConnectorsError
    ├── ChannelError                          a failure about one channel
    │   ├── ChannelLimitsViolationError
    │   ├── ChannelWriteBlockedError
    │   ├── ChannelWriteFailedError
    │   ├── ChannelNotFoundError              (LookupError)
    │   └── ChannelUnreachableError
    ├── ConnectorUnavailableError             (ConnectionError) the connector is gone
    │   └── ipc.ConnectorHostError, its Start/Lost/Unresponsive/WriteTimeout kin
    ├── ConnectorInternalError                (RuntimeError) a connector bug
    ├── ChannelReadFailedError                several channels; settle.SettleReadError
    ├── ConnectRefusedError                   (ConnectionError) connect refused by policy
    ├── ClientLibraryRefusedError             (RuntimeError)
    ├── ConfigurationError                    (ValueError) a config value, by its key
    ├── StateUnavailableError                 (RuntimeError) no state directory
    ├── StateLockedError                      (RuntimeError) a state lock stayed held
    ├── AlreadyApprovedError                  (FileExistsError)
    ├── write_authorization.AuthorizationRefusedError
    ├── control_context.StoppedSinceError, control_context.RecordUnavailableError
    ├── simulation.view.SimulatorViewError
    │   ├── NoSimulatorViewError              (LookupError)
    │   └── ViewSchemaError                   (ValueError)
    ├── simulation.composite.WriteRefusedError (ValueError)
    ├── settle.SettleTimeoutError             (TimeoutError)
    ├── ipc.ChildUnresponsiveError            (TimeoutError)
    ├── ipc.ConnectorHostSlowCallError        (TimeoutError)
    ├── ipc.FrameDecodeError                  (ValueError)
    └── ipc.FrameEncodeError                  (TypeError)

Three contract roots classify what a connector call raises, and no class lies
under two of them: :class:`ChannelError`, a failure about one named channel
that leaves the connector in place; every single-channel failure (a limits
violation, a refused write, a failed write, an address that is not a channel,
an unreachable channel) is one, and its ``address`` names the channel.
:class:`ConnectorUnavailableError`, the connector itself is gone and its
supervisor replaces it. :class:`ConnectorInternalError`, a connector bug that
nothing recovers from. They do not cover every failure a call can raise.
:class:`ChannelReadFailedError` names several channels. Three timeouts lie
under no root on purpose, because each is read by its ``TimeoutError`` base
alone and a root would change what a supervisor does about it:
``settle.SettleTimeoutError`` (a readback that did not settle),
``ipc.ChildUnresponsiveError`` (the connector-host child missed the proxy's
deadline) and ``ipc.ConnectorHostSlowCallError`` (it missed a call's deadline
and still answers a ping, so it was kept; under
:class:`ConnectorUnavailableError` it would read as a lost connection).
:class:`ConnectRefusedError` is a policy refusal of ``connect()``, not a lost
connector, and is read by its ``ConnectionError`` base as a bare one was.

:class:`RefusalReason` names why a write was refused, who refused it and what
to do about it. ``ChannelWriteFailedError``'s reason codes are the uppercased
``WriteOutcome`` words for the outcomes that raise, so the reason a caller
branches on and the outcome the result reports are one vocabulary.

The connector-host IPC registry is derived from this module: every class in
``__all__`` under a contract root crosses the process boundary as itself,
with its channel, so a public failure needs no registry entry of its own.
"""

import errno
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from osprey_connectors.model import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from osprey_connectors.control_system import WriteOutcome


__all__ = [
    "NOT_A_CHANNEL",
    "RAW_CLIENT_WRITE_MARKER",
    "AlreadyApprovedError",
    "ChannelError",
    "ChannelLimitsViolationError",
    "ChannelNotFoundError",
    "ChannelReadFailedError",
    "ChannelUnreachableError",
    "ChannelWriteBlockedError",
    "ChannelWriteFailedError",
    "ClientLibraryRefusedError",
    "ConfigurationError",
    "ConnectRefusedError",
    "ConnectorInternalError",
    "ConnectorUnavailableError",
    "ConnectorsError",
    "LimitsViolationKind",
    "RecoveryRecord",
    "RefusalReason",
    "StateLockedError",
    "StateUnavailableError",
    "raw_client_write_message",
]


class ConnectorsError(Exception):
    """The base of every exception the package raises on purpose.

    It carries nothing of its own; each subclass carries its context as
    attributes. ``except ConnectorsError`` catches every package failure and
    no builtin one, so a caller that means "anything the connectors reported"
    names it rather than ``Exception``.
    """


class ChannelError(ConnectorsError):
    """A failure about one named channel; the connector that raised it stays in place.

    ``address`` names the channel, in process and across the
    connector-host boundary alike. No supervisor recovers from a channel
    error: it is reported against its channel and nothing is restarted. It has
    no builtin base that another root shares, so no ``except ConnectionError``
    or ``except TimeoutError`` reads it as a lost connector.

    ``commands_fired`` is set in process by a batch command writer when the
    batch raised: the addresses, in order, that may have gone out. It is
    ``None`` when no batch writer set it, and it is not a wire field.
    """

    commands_fired: list[str] | None = None

    def __init__(self, address: str, message: str):
        self.address = address
        super().__init__(message)


#: Why the limits refused a write: the ``violation_type`` of
#: :class:`ChannelLimitsViolationError`.
LimitsViolationKind = Literal[
    "MIN_EXCEEDED",
    "MAX_EXCEEDED",
    "MAX_STEP_EXCEEDED",
    "STEP_CHECK_FAILED",
    "READ_ONLY_CHANNEL",
    "UNLISTED_CHANNEL",
    "LIMITS_DATABASE_UNAVAILABLE",
    "INVALID_NUMERIC_VALUE",
]


class ChannelLimitsViolationError(ChannelError):
    """Raised when a channel write violates configured safety limits.

    Covers min/max range violations, read-only channel writes,
    excessive step sizes, and writes to unlisted channels.
    """

    def __init__(
        self,
        address: str,
        value: object,
        violation_type: LimitsViolationKind,
        violation_reason: str,
        min_value: float | None = None,
        max_value: float | None = None,
        max_step: float | None = None,
        current_value: object = None,
    ):
        self.address = address
        self.attempted_value = value
        self.violation_type = violation_type
        self.violation_reason = violation_reason
        self.min_value = min_value
        self.max_value = max_value
        self.max_step = max_step
        self.current_value = current_value

        super().__init__(address, self._format_violation_message())

    def _format_violation_message(self) -> str:
        """Format a user-friendly violation message with all relevant details."""
        msg = [
            "\n" + "=" * 70,
            "CHANNEL LIMITS VIOLATION DETECTED",
            "=" * 70,
            f"Channel Address: {self.address}",
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
    ``BATCH_REFUSED`` is a write of a batch that passed every check of its own
    and was not sent because another write of the batch was refused.
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
    BATCH_REFUSED = (
        "BATCH_REFUSED",
        _REFERENCE_MONITOR,
        (
            "No write of this batch was sent: another write in it was refused.",
            "Resolve that write's refusal and submit the batch again.",
        ),
    )
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

    ``verdict_reason`` narrows ``WRITES_DISABLED`` to the reason the write
    verdict gave, a member of :data:`osprey_connectors.refusals.REASONS`, when
    it gave one: ``switch_off``, ``auth_exhausted``, ``auth_range`` and the
    rest, so a program branches on the word rather than the message.
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

    def __init__(
        self,
        address: str,
        reason: str | None,
        message: str | None = None,
        verdict_reason: str | None = None,
    ):
        self.address = address
        self.reason: str | None = _known_reason(reason) or reason
        self.verdict_reason = verdict_reason
        refuser = self.refuser(reason)
        if refuser is not None:
            fallback = f"Write to '{address}' refused by {refuser} ({reason})"
        elif reason == ChannelNotFoundError.reason:
            fallback = str(ChannelNotFoundError(address))
        elif reason is None:
            fallback = f"Write to '{address}' refused"
        else:
            fallback = f"Write to '{address}' refused ({reason})"
        super().__init__(address, message or fallback)


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


class ClientLibraryRefusedError(ConnectorsError, RuntimeError):
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
        address: str,
        reason: str,
        message: str | None = None,
        *,
        outcome: "WriteOutcome | str | None" = None,
        value_written: object = None,
        observed_value: object = None,
    ):
        if outcome is not None:
            from osprey_connectors.control_system import WriteOutcome

            if not isinstance(outcome, WriteOutcome):
                outcome = WriteOutcome(outcome)
        self.address = address
        self.reason = reason
        self.outcome = outcome
        self.value_written = value_written
        self.observed_value = observed_value
        super().__init__(address, message or self._default_message())

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
                f"Write to '{self.address}' not confirmed (MISMATCH): "
                f"sent {self.value_written}, channel holds {self.observed_value}"
            )
        return f"Write to '{self.address}' failed ({self.reason})"


class ChannelReadFailedError(ConnectorsError):
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

    def __init__(self, address: str):
        super().__init__(address, f"{address} is {NOT_A_CHANNEL}")


class ChannelUnreachableError(ChannelError):
    """One channel could not be reached: it was lost, or it did not answer in time.

    The connector that raised it is healthy and keeps serving every other
    channel, so nothing is replaced for it; the failure is that channel's
    device, not the control system as a whole. Its text names the address.
    """

    def __init__(self, address: str, message: str | None = None):
        super().__init__(address, message or f"{address} did not answer")


class RecoveryRecord(FrozenModel):
    """One thing a supervisor did about a failure, as a failure's ``recovery`` carries it.

    ``action`` is the word naming the step
    (:class:`~osprey_connectors.recovery.RecoveryAction`), ``text`` the sentence
    an operator reads.
    """

    action: str
    text: str


class ConnectorUnavailableError(ConnectorsError, ConnectionError):
    """The connector itself is gone: its transport, context or process is lost.

    Every channel it serves fails alike until it is replaced, and the
    supervisor that owns the connector's lifetime replaces it at most once per
    instance. ``recovery`` is what that supervisor did, one
    :class:`RecoveryRecord` per step, set by the supervisor and by
    nothing else; it is empty until a supervisor has acted.
    """

    def __init__(self, message: str, recovery: "Sequence[RecoveryRecord]" = ()):
        self.recovery: tuple[RecoveryRecord, ...] = tuple(recovery)
        super().__init__(message)


class ConnectorInternalError(ConnectorsError, RuntimeError):
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


class ConnectRefusedError(ConnectorsError, ConnectionError):
    """``connect()`` refused to bring a connector up, on policy grounds.

    A readonly run whose only configured Channel Access gateway is the write
    gateway, for instance: connecting with no gateway would search the ambient
    network, so the connector refuses instead. Nothing was lost, so it is not a
    :class:`ConnectorUnavailableError`. It keeps the ``ConnectionError`` base
    the bare refusal had, so every reader that answers a connection failure
    (the recovery table, the framework's tool envelopes) answers it as before.
    """


class ConfigurationError(ConnectorsError, ValueError):
    """A configuration value is missing, of the wrong shape, or refused.

    ``key`` is the dotted config key the value sits under (for example
    ``control_system.connector.epics.timeout_s``), or ``None`` when the
    refusal names no single key. A ``ValueError``, as the bare refusals it
    replaces were, so a caller that catches ``ValueError`` still catches it.
    """

    def __init__(self, message: str, *, key: str | None = None) -> None:
        self.key = key
        super().__init__(message)


class StateUnavailableError(ConnectorsError, RuntimeError):
    """There is no state directory for a write-path writer to write in.

    No agent-data root resolves, or no authorizations directory is reachable,
    so nothing was written: a file stored where no reader would look is worse
    than none. ``path`` is the directory that was looked for, or ``None`` when
    no path resolves at all. A ``RuntimeError``, as the bare refusal it
    replaces was, so every ``except RuntimeError`` that answers "the store is
    unavailable" still answers it.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class StateLockedError(ConnectorsError, RuntimeError):
    """A write-path lock stayed held past its bounded wait; nothing was written.

    ``path`` is the file whose lock was waited on and ``waited_s`` the wait in
    seconds. A ``RuntimeError``, as the bare refusal it replaces was.
    """

    def __init__(self, message: str, *, path: Path, waited_s: float) -> None:
        self.path = path
        self.waited_s = waited_s
        super().__init__(message)


class AlreadyApprovedError(ConnectorsError, FileExistsError):
    """An authorization was approved already: its grant file exists.

    ``authorization`` is the authorization's id and ``path`` the grant file
    that exists. A ``FileExistsError`` (``errno.EEXIST``), as the bare refusal it
    replaces was, so its text and every ``except FileExistsError`` stay as
    they were.
    """

    def __init__(self, path: Path, *, authorization: str) -> None:
        self.path = path
        self.authorization = authorization
        super().__init__(errno.EEXIST, f"{path} exists; nothing was written")
