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
    │   ├── ipc.ConnectorHostError, its Start/Lost/Unresponsive/WriteTimeout kin
    │   └── ipc.ChildLostError                (also a ChildLinkError) the pipe ended
    ├── ConnectorInternalError                (RuntimeError) a connector bug
    ├── ChannelReadFailedError                several channels; settle.SettleReadError
    ├── ConnectRefusedError                   (ConnectionError) connect refused by policy
    ├── ClientLibraryRefusedError             (RuntimeError)
    ├── ConfigurationError                    (ValueError) a config value, by its key
    ├── ConfigLoadError                       a config file that did not load, by its path
    │   ├── ConfigNotFoundError               (also a ConfigurationError; FileNotFoundError)
    │   └── ConfigParseError                  (also a ConfigurationError) not YAML, or no mapping
    ├── NaiveDatetimeError                    (ValueError) an instant with no zone
    ├── ArchiveQueryError                     (ValueError) an archive query argument
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
    ├── ChildLinkError                        made by the parent's link to a child
    │   ├── ipc.ChildUnresponsiveError        (TimeoutError)
    │   ├── ipc.ConnectorHostSlowCallError    (TimeoutError)
    │   └── ipc.ChildLostError                (ConnectorUnavailableError)
    ├── ipc._frames.FrameDecodeError          (ValueError) internal to the wire codec
    └── ipc._frames.FrameEncodeError          (TypeError) internal to the wire codec

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
The last two share :class:`ChildLinkError`, the base of what the parent's
link to a connector-host child raises on its own, which is no root either:
it names where a failure was made, never what a supervisor does about it.
Its third class, ``ipc.ChildLostError``, is a lost pipe, and so a
:class:`ConnectorUnavailableError` as well.
:class:`ConnectRefusedError` is a policy refusal of ``connect()``, not a lost
connector, and is read by its ``ConnectionError`` base as a bare one was.

:class:`~osprey_connectors.refusals.RefusalReason` names why a write was
refused, who refused it and what to do about it. ``ChannelWriteFailedError`` carries the ``WriteOutcome`` of the
write that raised, so the outcome a caller branches on and the outcome the
result reports are one vocabulary.

The connector-host IPC registry is derived from this module: every class in
``__all__`` under a contract root crosses the process boundary as itself,
with its channel, so a public failure needs no registry entry of its own.
"""

import errno
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import field_validator

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.model import FrozenModel
from osprey_connectors.refusals import RefusalReason, VerdictReason

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from osprey_connectors.control_system import WriteOutcome


__all__ = [
    "NOT_A_CHANNEL",
    "RAW_CLIENT_WRITE_MARKER",
    "AlreadyApprovedError",
    "ArchiveQueryError",
    "ChannelError",
    "ChannelLimitsViolationError",
    "ChannelNotFoundError",
    "ChannelReadFailedError",
    "ChannelUnreachableError",
    "ChannelWriteBlockedError",
    "ChannelWriteFailedError",
    "ChildLinkError",
    "ClientLibraryRefusedError",
    "ConfigLoadError",
    "ConfigNotFoundError",
    "ConfigParseError",
    "ConfigurationError",
    "ConnectRefusedError",
    "ConnectorInternalError",
    "ConnectorUnavailableError",
    "ConnectorsError",
    "LimitsViolation",
    "LimitsViolationKind",
    "NaiveDatetimeError",
    "RecoveryAction",
    "RecoveryRecord",
    "StateLockedError",
    "StateUnavailableError",
    "raw_client_write_message",
    "require_aware",
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


class LimitsViolation(FrozenModel):
    """Why the limits refused one write, as a write result and its refusal carry it.

    The detail of a :class:`ChannelLimitsViolationError`, without the channel
    and the value: the result or the refusal that carries this record names
    both already. ``kind`` is the violation's type, ``reason`` its one-line
    sentence, and the bounds are the ones the refused value was judged by,
    each ``None`` where the check that refused it has none. ``current_value``
    is what the channel held when a ``max_step`` check measured the step.
    """

    kind: LimitsViolationKind
    reason: str
    min_value: float | None = None
    max_value: float | None = None
    max_step: float | None = None
    current_value: ContractAny = None

    @classmethod
    def from_error(cls, exc: ChannelLimitsViolationError) -> "LimitsViolation":
        """The violation ``exc`` reports."""
        return cls(
            kind=exc.violation_type,
            reason=exc.violation_reason,
            min_value=exc.min_value,
            max_value=exc.max_value,
            max_step=exc.max_step,
            current_value=exc.current_value,
        )


class ChannelWriteBlockedError(ChannelError):
    """Raised when a channel write was refused and no value was written.

    Distinct from ChannelWriteFailedError, which means the write was attempted
    and its outcome is a failure rather than a denial.

    ``reason`` is the :class:`~osprey_connectors.refusals.RefusalReason` member
    that says why, and its :attr:`~osprey_connectors.refusals.RefusalReason.refuser`
    and :attr:`~osprey_connectors.refusals.RefusalReason.remedy` say who
    refused and what to do. The word of a member is taken for the member, as
    the connector-host IPC carries it; any other word raises ``ValueError``.

    ``verdict_reason`` narrows ``WRITES_DISABLED`` to the reason the write
    verdict gave, a :class:`~osprey_connectors.refusals.VerdictReason`, when
    it gave one: ``switch_off``, ``auth_exhausted``, ``auth_range`` and the
    rest, so a program branches on the word rather than the message.

    ``violation`` is the :class:`LimitsViolation` of a ``LIMITS`` refusal, and
    ``None`` on every other.
    """

    def __init__(
        self,
        address: str,
        reason: RefusalReason | str,
        *,
        message: str | None = None,
        verdict_reason: VerdictReason | str | None = None,
        violation: LimitsViolation | None = None,
    ):
        self.address = address
        self.reason = RefusalReason(reason)
        self.verdict_reason = None if verdict_reason is None else VerdictReason(verdict_reason)
        self.violation = violation
        refuser = self.reason.refuser
        if refuser is not None:
            fallback = f"Write to '{address}' refused by {refuser} ({self.reason})"
        else:
            fallback = str(ChannelNotFoundError(address))
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

    User code imported a control-system client package, or loaded a client
    shared library, in a process whose only route to the machine is
    ``osprey.runtime``. Nothing was loaded and no channel was
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

    ``outcome`` is the :class:`~osprey_connectors.control_system.WriteOutcome`
    of the write that raised: ``failed``, ``mismatch`` or ``unconfirmed``, the
    words a write raises under. All three are protocol-neutral on purpose: the
    same outcome describes an EPICS, DOOCS, or simulated write. The word of a
    member is taken for the member, as the connector-host IPC carries it; any
    other outcome raises ``ValueError``.

    ``value_written`` and ``observed_value`` carry the detail the result held,
    so a caller that only sees the exception can still say what was sent and
    what the channel holds.
    """

    def __init__(
        self,
        address: str,
        outcome: "WriteOutcome | str",
        *,
        message: str | None = None,
        value_written: object = None,
        observed_value: object = None,
    ):
        from osprey_connectors.control_system import WriteOutcome

        member = WriteOutcome(outcome)
        if member not in _FAILED_OUTCOMES:
            raise ValueError(
                f"Write to '{address}': {member} is not an outcome a write raises under"
            )
        self.address = address
        self.outcome = member
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
        word = self.outcome.upper()
        if word == "MISMATCH" and self.value_written is not None:
            return (
                f"Write to '{self.address}' not confirmed (MISMATCH): "
                f"sent {self.value_written}, channel holds {self.observed_value}"
            )
        return f"Write to '{self.address}' failed ({word})"


#: The outcome words a write raises :class:`ChannelWriteFailedError` under.
_FAILED_OUTCOMES = frozenset({"failed", "mismatch", "unconfirmed"})


class ChannelReadFailedError(ConnectorsError):
    """Raised when a batch read or a settle's readback read could not produce a value.

    A channel counts as failed when its own read raised a :class:`ChannelError`
    (:meth:`ChannelReadBatch.raise_for_failures
    <osprey_connectors.control_system.ChannelReadBatch.raise_for_failures>`).

    ``addresses`` names every failed channel, each once, in the order the
    caller asked for them, so a caller can report or retry all of them rather
    than only the first.

    ``causes`` maps a failed channel to the exception its own read raised, so a
    timeout, a refused connection and a missing channel stay distinguishable.
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
    raised: it comes back ``refused`` with ``refusal_reason``
    :attr:`~osprey_connectors.refusals.RefusalReason.CHANNEL_NOT_FOUND`.
    """

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


class RecoveryAction(StrEnum):
    """What a supervisor did about a failure."""

    CONNECTOR_DROPPED = "connector_dropped"
    CHILD_REPLACED = "child_replaced"
    CHILD_KEPT = "child_kept"
    RESPAWN_REFUSED = "respawn_refused"


class RecoveryRecord(FrozenModel):
    """One thing a supervisor did about a failure, in one sentence.

    What :meth:`~osprey_connectors.recovery.RecoveryPolicy.recover` answers, and
    what a failure's ``recovery`` carries. ``action`` names the step and takes
    the :class:`RecoveryAction` member or its word, holding the member; a word
    that names no step is refused. ``text`` is the sentence an operator reads,
    and may not be empty.
    """

    action: RecoveryAction
    text: str

    @field_validator("action", mode="before")
    @classmethod
    def _member(cls, action: object) -> RecoveryAction:
        """The member *action* is, or whose word it is, as ``RecoveryAction(action)`` finds it."""
        member = next((member for member in RecoveryAction if member == action), None)
        if member is None:
            raise ValueError(f"{action!r} is not a valid RecoveryAction")
        return member

    @field_validator("text")
    @classmethod
    def _stated(cls, text: str) -> str:
        if not text:
            raise ValueError("a recovery record states what was done; its text is empty")
        return text


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


class ChildLinkError(ConnectorsError):
    """Raised in the parent by the link to a connector-host child; never sent by the child.

    The base of every failure the parent's end of the pipe makes on its own: a
    child that missed the proxy's deadline, a pipe that ended, a call the pool
    kept waiting on a child that still answers a ping. Its class is how a
    supervisor tells the link's failure from the same builtin the child
    reported: an exception from a child call that is a ``ChildLinkError`` came
    from the link, and any other came from the child. No such class is in the
    IPC registry, so none can be rebuilt from a frame.

    It is no contract root: each subclass keeps the builtin or root its
    meaning names (a ``TimeoutError`` for a missed deadline, a
    :class:`ConnectorUnavailableError` for a lost pipe), and is classified by
    that. ``recovery`` is what the supervisor did, as on
    :class:`ConnectorUnavailableError`; it is empty until a supervisor has
    acted.
    """

    def __init__(self, message: str, recovery: "Sequence[RecoveryRecord]" = ()):
        super().__init__(message)
        # Set after the other bases ran, so a ConnectorUnavailableError base
        # cannot reset it.
        self.recovery: tuple[RecoveryRecord, ...] = tuple(recovery)


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


class ConfigLoadError(ConnectorsError):
    """A config file did not load; ``path`` is the file that was asked for.

    Raised as itself when the file could not be read at all: the path names a
    directory (a container bind-mount whose source was missing is created as
    one), or reading it failed. A directory is deliberately not a
    :class:`ConfigNotFoundError`: the readers that degrade on a
    ``FileNotFoundError`` (the limits validator, the connector host's write
    report) would read it as "no config here" and turn checks off, where today
    the load fails. Its two subclasses name the other ways a load fails.
    """

    def __init__(self, message: str, *, path: Path) -> None:
        self.path = path
        super().__init__(message)


class ConfigNotFoundError(ConfigLoadError, ConfigurationError, FileNotFoundError):
    """No config file is there; ``searched`` is every path that was looked at, in order.

    A ``FileNotFoundError``, as the bare refusal it replaces was, and that base
    is load-bearing: the readers that run without a config (the limits
    validator answers "no validator", the connector host reports writes off)
    catch it by that name. ``path`` is the path the load would have read: the
    one named, or the last place a lookup that named none looked.
    """

    def __init__(self, message: str, *, path: Path, searched: "Sequence[Path]") -> None:
        self.searched = tuple(searched)
        super().__init__(message, path=path)


class ConfigParseError(ConfigLoadError, ConfigurationError):
    """A config file was read but holds no usable config.

    It is not YAML (``line`` is the 1-based line the parser stopped at, when it
    says), or its document is not a mapping, or one of the sections the loader
    reads whole is not one (``key`` names it). A ``ValueError``, as the
    refusals it replaces were.
    """

    def __init__(
        self, message: str, *, path: Path, line: int | None = None, key: str | None = None
    ) -> None:
        self.line = line
        super().__init__(message, path=path)
        # After the bases ran: ConfigurationError's own __init__ sets it to None.
        self.key = key


class NaiveDatetimeError(ConnectorsError, ValueError):
    """An instant was handed over without a timezone, so it names no moment.

    Every instant the package takes is a timezone-aware ``datetime``; a naive
    one is refused rather than read in a zone the package would have to guess.
    ``parameter`` names the argument or field that carried it, and ``value``
    is the naive ``datetime`` itself. A ``ValueError``, as a bad argument is.
    """

    def __init__(self, parameter: str, value: datetime) -> None:
        self.parameter = parameter
        self.value = value
        super().__init__(
            f"{parameter} must be a timezone-aware datetime, got the naive {value.isoformat()}; "
            "attach the zone it was read in (tzinfo=...)"
        )


class ArchiveQueryError(ConnectorsError, ValueError):
    """An archive query the archive cannot answer as asked: the caller's argument.

    ``parameter`` names the argument (``processing``, ``precision_ms``,
    ``end_at``, ``addresses``) and ``address`` the channel it was refused for,
    or ``None`` when the refusal is about the whole query. The one failure an
    archive raises besides a refused instant: a service that gives no answer
    answers :class:`~osprey_connectors.service_answers.Unavailable` instead. A
    ``ValueError``, as a bad argument is.
    """

    def __init__(self, message: str, *, parameter: str, address: str | None = None) -> None:
        self.parameter = parameter
        self.address = address
        super().__init__(message)


def require_aware(value: object, *, name: str) -> datetime:
    """``value`` if it is a timezone-aware ``datetime``; otherwise refuse it.

    The one boundary check for an instant a caller hands the package. It
    neither converts nor guesses: an aware value comes back as given.

    Args:
        value: The caller's instant, unvalidated.
        name: The parameter or field it was given as, for the refusal.

    Returns:
        ``value``, unchanged.

    Raises:
        TypeError: If ``value`` is not a ``datetime``, naming ``name``.
        NaiveDatetimeError: If ``value`` is a naive ``datetime``.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime object, got {type(value)}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NaiveDatetimeError(name, value)
    return value


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
