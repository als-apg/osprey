"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

"""

import asyncio
import functools
import logging
import math
import numbers
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from osprey_connectors import posture_store
from osprey_connectors.control_system.write_door import open_door
from osprey_connectors.errors import RefusalReason
from osprey_connectors.posture_env import is_readonly_run
from osprey_connectors.posture_store import WriteVerdict

logger = logging.getLogger("osprey_connectors.control_system")


@dataclass
class ChannelMetadata:
    """Metadata a control system reports about one of its channels.

    ``display_low`` / ``display_high`` are the range the control system suggests
    for DISPLAYING this channel (EPICS LOPR/HOPR and their equivalents). They are
    deliberately NOT named ``min_value`` / ``max_value``: those belong to
    :class:`~osprey_connectors.control_system.ChannelLimitsConfig`
    and are the bounds OSPREY refuses to write past. Only the latter is enforced —
    a value inside the display range can still be refused, and a connector that
    reports no display range constrains nothing.

    ``enum_labels`` / ``enum_label`` are present only for enum-typed records
    (EPICS ``mbbi``/``mbbo``/``bi``/``bo``, PVAccess ``NTEnum``, and their
    equivalents elsewhere) and are ``None`` on every other channel. They are the
    *readable* half of an enum reading: ``ChannelValue.value`` stays the integer
    index — one machine-readable type across every protocol — and the label for
    that index rides here, resolved at read time. Either may be ``None`` even on
    an enum channel: a control system reports the label list only when it has
    one, and an index outside it resolves to nothing.

    ``enum_labels`` is a ``list`` rather than a tuple because that is what
    survives the connector-host IPC seam: :mod:`osprey_connectors.ipc._frames`
    encodes sequences as JSON arrays and decodes them as lists, so a tuple set
    here would compare unequal to the same field read through a proxy.
    """

    units: str = ""
    precision: int | None = None
    #: Alarm name reported for the reading, e.g. "NO_ALARM"/"HIHI"; None if not reported.
    alarm_status: str | None = None
    #: Alarm severity: 0 healthy, higher is worse; ``None`` means "not reported",
    #: which stays distinct from a reported healthy value (``alarm_severity=0``) —
    #: the same convention as :attr:`ChannelWriteResult.alarm_severity`.
    alarm_severity: int | None = None
    timestamp: datetime | None = None
    description: str | None = None
    display_low: float | None = None
    display_high: float | None = None
    #: Every state this enum-typed channel can report, in index order; None off enums.
    enum_labels: list[str] | None = None
    #: The label for the value of THIS reading; None off enums and when unresolvable.
    enum_label: str | None = None
    raw_metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self):
        """Ensure raw_metadata is a dict."""
        if self.raw_metadata is None:
            self.raw_metadata = {}


@dataclass
class ChannelValue:
    """Value of a control system channel with metadata."""

    value: Any
    timestamp: datetime
    metadata: ChannelMetadata = field(default_factory=ChannelMetadata)


class WriteOutcome(StrEnum):
    """What became of one channel write — the single owned verdict.

    The connector that performed the write is the only producer of this value;
    every consumer (MCP tool, ``osprey.runtime``, the Bluesky bridge, the
    connector-host IPC) reads it and none re-derives a verdict of its own.

    * ``REFUSED`` — nothing was written; ``refusal_reason`` says why.
    * ``FAILED`` — the value was sent and the control system did not take it.
    * ``CONFIRMED`` — a re-read of the channel holds the value sent, in any
      alarm state. Alarm state is reported, never raised on.
    * ``MISMATCH`` — the re-read holds a different value (a clamped or rounded
      setpoint is reported here, not tolerated).
    * ``UNCONFIRMED`` — the value was sent, but not acknowledged in time or
      the re-read itself failed, so what the channel holds is unknown.
    * ``UNREQUESTED`` — ``confirm=False``: nothing was checked.
    """

    REFUSED = "refused"
    FAILED = "failed"
    CONFIRMED = "confirmed"
    MISMATCH = "mismatch"
    UNCONFIRMED = "unconfirmed"
    UNREQUESTED = "unrequested"


def _is_sequence(value: Any) -> bool:
    """True when ``value`` is a vector rather than a scalar, for comparison.

    Admits list, tuple and ndarray. Excluded, in order:

    * str/bytes, which have a length but read back as one value;
    * mappings and anything else without ``__getitem__``, because a vector is
      compared by zipping the two sides positionally and a mapping iterates its
      *keys* — two dicts with the same keys and different values would zip to a
      match. Left as scalars, they compare by ``==``, which is what a mapping's
      equality already means;
    * 0-d numpy arrays (``ndim == 0``), which hold a single value and must
      compare like the scalar they wrap.
    """
    return (
        not isinstance(value, str | bytes | Mapping)
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and getattr(value, "ndim", 1) != 0
    )


def _unwrap_length_one(value: Any) -> Any:
    """A length-1 sequence reduced to its element; anything else unchanged.

    ``[5.0]`` and ``5.0`` name the same reading: control systems differ on
    whether a single-element channel reads back boxed, and that difference must
    not decide an outcome.
    """
    if _is_sequence(value) and len(value) == 1:
        return value[0]
    return value


def _scalars_match(sent: Any, observed: Any) -> bool:
    """Compare two scalars: floats by relative closeness, everything else by ``==``."""
    if (
        isinstance(sent, numbers.Real)
        and isinstance(observed, numbers.Real)
        and not isinstance(sent, bool)
        and not isinstance(observed, bool)
    ):
        return math.isclose(float(sent), float(observed), rel_tol=1e-6, abs_tol=0.0)

    result = sent == observed
    if isinstance(result, bool):
        return result
    # A numpy (or other array-like) elementwise comparison reduces to one verdict.
    return bool(result.all()) if hasattr(result, "all") else bool(result)


def values_match(sent: Any, observed: Any, *, enum_label: str | None = None) -> bool:
    """True when ``observed`` is the value ``sent`` — the one comparison rule.

    Applied in order, and symmetric: swapping the arguments cannot change the
    answer.

    1. A string against a reported ``enum_label`` compares as text: an enum
       channel written as ``"Open"`` reads back as its integer index, and the
       label the connector resolved for that reading is what the text is
       compared with. A connector that reports no ``enum_label`` (DOOCS) falls
       through to the ordinary comparison.
    2. A length-1 sequence is unwrapped to its element, on both sides.
    3. After unwrapping, exactly one side a sequence is ``False``: a scalar is
       not a vector. This is what stops a numpy broadcast
       (``np.array([5.0, 5.0]) == 5.0`` reduced by ``.all()``) from
       manufacturing a match.
    4. Two sequences of unequal length are ``False``; equal lengths compare
       elementwise.
    5. Two real numbers (bools excluded) compare by ``math.isclose`` with
       ``rel_tol=1e-6`` — roughly eight float32 epsilons, so a legitimate
       float32 store still matches — and ``abs_tol=0.0``, so nothing is close
       to zero but zero. There is no configurable tolerance anywhere.
    6. Anything else compares by ``==``.
    7. Any exception means the two cannot be compared, which is not a match.
    """
    try:
        if enum_label is not None:
            if isinstance(sent, str):
                return sent == enum_label
            if isinstance(observed, str):
                return observed == enum_label

        sent = _unwrap_length_one(sent)
        observed = _unwrap_length_one(observed)

        sent_is_seq = _is_sequence(sent)
        observed_is_seq = _is_sequence(observed)
        if sent_is_seq != observed_is_seq:
            return False

        if sent_is_seq:
            if len(sent) != len(observed):
                return False
            return all(_scalars_match(a, b) for a, b in zip(sent, observed, strict=True))

        return _scalars_match(sent, observed)
    except Exception:
        return False


def as_number(value: Any) -> float | None:
    """``value`` as a float, or ``None`` when it is not a number.

    A reading that is not a number — a string, an enum label, a waveform — has
    no numeric form and is reported as ``None`` ("no numeric value"), never
    coerced into one. A string that happens to parse as a number is still a
    string here: what the channel holds is text, and inventing a number for it
    is the fabrication this helper exists to prevent.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


#: The outcomes whose result carries an ``error_message``: each has something
#: the operator has to be told.
_MESSAGE_OUTCOMES = frozenset({WriteOutcome.REFUSED, WriteOutcome.FAILED, WriteOutcome.UNCONFIRMED})


@dataclass
class ChannelWriteResult:
    """What became of one channel write, and what the channel was seen to hold.

    The control-system-agnostic result type returned by every connector.
    :attr:`outcome` is the single owned verdict (see :class:`WriteOutcome`),
    set by the connector that performed the write; no consumer re-derives one.

    ``error_message`` is set **if and only if** ``outcome`` is one of
    ``refused``, ``failed`` or ``unconfirmed`` — the three words that carry
    something the operator has to be told. A ``mismatch`` needs no message:
    both numbers are on the result already, in ``value_written`` and
    ``observed_value``. Construction enforces it: a result that breaks it
    raises ``ValueError``.

    ``notes`` is display-only text ("observed 2.4, sent 2.5", "cannot be
    compared"). No consumer reads it, and nothing classifies a write by
    parsing it.

    ``refusal_reason`` is the :class:`~osprey_connectors.errors.RefusalReason`
    that says why, and who, refused; it is set **if and only if** the outcome
    is ``refused``. Construction enforces it too: a refused result with no
    reason or an unknown one, or a reason on any other outcome, raises
    ``ValueError``. ``observed_value`` is whatever the confirming
    re-read returned, in the type the channel holds — a number, a string, an
    enum label, a waveform; :attr:`observed_number` narrows it to a float
    where that makes sense.

    Alarm state (``alarm_status``, ``alarm_severity``) is carried
    protocol-agnostically — the alarm *name* as a string, the severity as an
    int — and is information reported with the write, never a reason to raise.
    ``None`` means "not reported", which stays distinct from a reported
    healthy value (``alarm_severity=0``). Mapping a protocol's raw codes to
    those names is the connector's job, so this shared module imports no
    control-system constants.
    """

    channel_address: str  # Channel that was written
    value_written: Any  # Value that was sent
    outcome: WriteOutcome  # The one verdict, set by the connector
    # Set iff refused, enforced at construction
    refusal_reason: RefusalReason | None = None
    # Set iff refused, failed or unconfirmed, enforced at construction
    error_message: str | None = None
    observed_value: Any = None  # What the confirming re-read returned
    # Alarm name reported for the reading, e.g. "NO_ALARM"/"HIHI"; None if not reported
    alarm_status: str | None = None
    # Alarm severity: 0 healthy, higher is worse; None if not reported
    alarm_severity: int | None = None
    notes: str | None = None  # Display-only detail

    def __post_init__(self) -> None:
        """Coerce ``outcome`` and ``refusal_reason`` to members, then check the shape.

        A ``StrEnum`` crosses the connector-host IPC as the plain string JSON
        encodes it to, and the codec rebuilds the dataclass by calling it with
        the decoded fields — so what arrives here on the parent side is a
        ``str``. Coercing once, on construction, is what lets every consumer
        compare against the members whichever side of the process boundary it
        runs on.

        Raises:
            ValueError: ``refusal_reason`` is not a known reason, is missing on
                a ``refused`` result or set on any other, or ``error_message``
                is missing on a ``refused``, ``failed`` or ``unconfirmed``
                result or set on any other.
        """
        if not isinstance(self.outcome, WriteOutcome):
            self.outcome = WriteOutcome(self.outcome)
        if self.refusal_reason is not None and not isinstance(self.refusal_reason, RefusalReason):
            try:
                self.refusal_reason = RefusalReason(self.refusal_reason)
            except ValueError:
                raise ValueError(
                    f"Write result for '{self.channel_address}' (outcome {self.outcome}): "
                    f"refusal_reason {self.refusal_reason!r} is not a known refusal reason"
                ) from None
        refused = self.outcome is WriteOutcome.REFUSED
        if refused != (self.refusal_reason is not None):
            expected = "a known refusal reason" if refused else "None"
            raise ValueError(
                f"Write result for '{self.channel_address}' (outcome {self.outcome}): "
                f"refusal_reason must be {expected}, got {self.refusal_reason!r}"
            )
        if bool(self.error_message) != (self.outcome in _MESSAGE_OUTCOMES):
            expected = "a message" if self.outcome in _MESSAGE_OUTCOMES else "None"
            raise ValueError(
                f"Write result for '{self.channel_address}' (outcome {self.outcome}): "
                f"error_message must be {expected}, got {self.error_message!r}"
            )

    @property
    def observed_number(self) -> float | None:
        """:attr:`observed_value` as a float, or ``None`` when it is not a number."""
        return as_number(self.observed_value)


#: Package path of the OSPREY MCP servers, as consecutive parts of the file
#: they are launched with (``python -m osprey.mcp_server.<server>`` sets
#: ``sys.argv[0]`` to that package's ``__main__.py``).
_MCP_SERVER_PACKAGE_PARTS = ("osprey", "mcp_server")


def _in_mcp_server_process() -> bool:
    """True when this process is an OSPREY MCP server, not a script it ran.

    This is the discriminator between the two remaining readonly stories, and
    it deliberately does not look at the environment: it *cannot*. Both stories
    speak through the same variable at the same value — the deployment-wide
    read-only run stamps ``OSPREY_EXECUTION_MODE=readonly`` on the whole
    deployment, and the python executor stamps the very same string on the
    sandbox subprocess it builds for one readonly submission. So the value says
    nothing about which one this is. What does differ is the process: an MCP
    server runs the server package's ``__main__``, while the executor's sandbox
    runs a script file it wrote for one submission. Answering True therefore
    means "the deployment is read-only", and answering False means "this one
    run was submitted readonly".

    Answering **False** is the safe way to be wrong. It yields the older,
    script-shaped message, which is what every readonly refusal said before
    the deployment-wide run had its own wording; a launch shape this check does
    not recognise therefore degrades to that text rather than to a message
    telling an operator their whole deployment is read-only when it is not. The
    refusal itself is identical either way — only the sentence telling the
    operator where to go changes.

    Neither story is the narrowing an operator makes from the
    control-target chip in the header: that never lived in the environment at
    all, and is read from the posture store further down this function's
    caller.
    """
    argv = getattr(sys, "argv", None)
    if not argv:
        return False
    try:
        parts = Path(argv[0]).parts
    except (TypeError, ValueError):
        return False
    span = len(_MCP_SERVER_PACKAGE_PARTS)
    return any(parts[i : i + span] == _MCP_SERVER_PACKAGE_PARTS for i in range(len(parts)))


#: What a refusal says in place of a phrase when the verdict itself allowed the
#: write and something else refused it: a subclass that overrides
#: :attr:`ControlSystemConnector._writes_enabled` and answers no. No reason of
#: the refusals contract applies, so none is claimed.
_OVERRIDE_REFUSAL = "Writes are refused by this connector's own write check."

#: The words a read-only run's refusal carries beside its phrase:
#: ``wrapper.READONLY_REFUSAL_MARKER``, which the executor's stderr matcher
#: looks for to name the runtime guard as the layer that refused. Kept out of
#: the phrase, which is a closed contract shared with the hooks.
_READONLY_MARKER = "readonly execution mode"

#: What a read-only refusal adds when the process is a script the executor ran
#: for one readonly submission rather than an MCP server of a read-only
#: deployment (:func:`_in_mcp_server_process`): that run alone can be resubmitted.
_READONLY_SCRIPT_REMEDY = (
    "This script was submitted readonly; resubmit it with execution_mode='readwrite' "
    "(human approval required) if the write is intended."
)


def _monitor_verdict(control_target: str | None) -> "WriteVerdict":
    """The verdict the reference monitor decides one write (or batch) by.

    :func:`osprey_connectors.posture_store.write_verdict` over this deployment's
    ``control_system`` section for *control_target*, the owner taken from the
    ladder. An unstamped connector (*control_target* ``None``) resolves no
    machine and is refused ``target_unknown`` unless an earlier term refuses
    first. A read-only run is answered before any config is read: it refuses
    whatever the config says. A config that cannot be read arms nothing — the
    section is then empty, which the tier answers ``never``.

    Never raises; reads the record at most once.
    """
    if is_readonly_run():
        return posture_store.write_verdict({}, control_target, owner=None)
    try:
        from osprey_connectors.config import get_config_value

        section: Any = get_config_value("control_system", {})
    except (FileNotFoundError, RuntimeError):
        section = {}
    return posture_store.write_verdict(section, control_target, owner=None)


def _verdict_refusal(
    channel_address: str, value: Any, verdict: "WriteVerdict"
) -> ChannelWriteResult:
    """The REFUSED result the monitor returns for a write *verdict* refused.

    The operator's words are :func:`~osprey_connectors.refusals.refusal_text`
    for the verdict's reason and machine kind — the same phrase the hooks
    speak, ending with the ``[osprey-refusal: <reason>]`` trailer — after the
    channel it concerns. A read-only run's refusal also
    carries :data:`_READONLY_MARKER`, and in a script the executor ran for one
    readonly submission :data:`_READONLY_SCRIPT_REMEDY` as well. A verdict that
    allowed (an overriding subclass refused on its own) names no contract reason.
    """
    from osprey_connectors.refusals import refusal_text

    marker = ""
    remedy = None
    if verdict.reason == "readonly_run":
        marker = f" ({_READONLY_MARKER})"
        if not _in_mcp_server_process():
            # One script submitted readonly, not the deployment: the same
            # refusal, and the one remedy that applies to it alone.
            remedy = _READONLY_SCRIPT_REMEDY
    if verdict.allowed or verdict.reason is None:
        spoken = _OVERRIDE_REFUSAL
    else:
        try:
            spoken = refusal_text(verdict.reason, verdict.kind, detail=remedy)
        except ValueError:
            spoken = _OVERRIDE_REFUSAL
    return ChannelWriteResult(
        channel_address=channel_address,
        value_written=value,
        outcome=WriteOutcome.REFUSED,
        refusal_reason=RefusalReason.WRITES_DISABLED,
        error_message=f"Write to '{channel_address}' blocked{marker}: {spoken}",
    )


#: The outcomes a caller may proceed on: the channel was confirmed to hold the
#: value sent, or no confirmation was asked for. Deliberately an allowlist and
#: not a denylist — a :class:`WriteOutcome` member added after this line was
#: written raises rather than silently becoming proceed-able.
_PROCEEDABLE_OUTCOMES: frozenset[WriteOutcome] = frozenset(
    {WriteOutcome.CONFIRMED, WriteOutcome.UNREQUESTED}
)


def raise_for_write_result(result: ChannelWriteResult) -> ChannelWriteResult:
    """Enforce the reference monitor's denial contract on one write result.

    The single place that decides whether a ``ChannelWriteResult`` counts as a
    write the caller may proceed on. Every caller that must not proceed on an
    unconfirmed write routes through here, so the refusal/failure distinction
    cannot drift between the single-channel and multi-channel paths.

    The verdict is read off :attr:`ChannelWriteResult.outcome` and nothing else.
    The connector that performed the write already decided; re-deriving a verdict
    here from a second field is what let a write to an unreadable channel report
    itself as a success.

    Args:
        result: The result returned by a connector's ``write_channel``.

    Returns:
        The result unchanged, when the outcome is ``confirmed`` — whatever the
        alarm severity, since alarm state is reported and never raised on — or
        ``unrequested``, where no confirmation was asked for.

    Raises:
        ChannelWriteBlockedError: The outcome is ``refused``: no value was
            written. ``refusal_reason`` names why;
            :meth:`ChannelWriteBlockedError.refuser` says who refused.
        ChannelWriteFailedError: The value was sent and the write did not come
            back confirmed — ``failed``, ``mismatch`` or ``unconfirmed``, each
            raising under its own word. The exception carries the outcome, the
            value sent and the value observed, so a ``mismatch`` message names
            both numbers.
    """
    from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

    if result.outcome is WriteOutcome.REFUSED:
        raise ChannelWriteBlockedError(
            result.channel_address,
            result.refusal_reason,
            message=result.error_message,
        )
    if result.outcome in _PROCEEDABLE_OUTCOMES:
        return result
    # Everything else is a write the caller must not proceed on. The reason code
    # is the outcome word uppercased, so the reason a caller branches on and the
    # outcome a consumer reads stay one vocabulary by construction rather than by
    # a table that a new outcome could be left out of.
    raise ChannelWriteFailedError(
        result.channel_address,
        result.outcome.upper(),
        message=result.error_message,
        outcome=result.outcome,
        value_written=result.value_written,
        observed_value=result.observed_value,
    )


class ControlSystemConnector(ABC):
    """
    Abstract base class for control system connectors.

    Implementations provide interfaces to different control systems
    (EPICS, LabVIEW, Tango, the simulator, etc.) using a unified API.

    Example:
        >>> connector = await ConnectorFactory.create_control_system_connector()
        >>> try:
        >>>     channel_value = await connector.read_channel('BEAM:CURRENT')
        >>>     print(f"Beam current: {channel_value.value} {channel_value.metadata.units}")
        >>> finally:
        >>>     await connector.disconnect()
    """

    _limits_validator: Any = None  # Initialized by subclasses in connect()
    # The connector type this instance was built as. Stamped by
    # ConnectorFactory.build_control_system_connector() between construction and
    # connect(), so connect() can already read the posture
    # (create_control_system_connector() is that build plus connect()). Stays
    # None on an instance nobody built through the factory: no type, so no
    # per-type block.
    _connector_type: str | None = None
    # The control target this instance was built for. Stamped by the same
    # factory seam as _connector_type, from the target the *caller* named — the
    # init payload's target, the sandbox's stamp, the deployment baseline, the
    # bridge lane's own target. Stays None on an instance nobody built through
    # the factory, and on a caller that has no target to name.
    _control_target: str | None = None
    # The wire this instance speaks, stamped by the same factory seam from the
    # type and the serving leaf. Stays None on an instance nobody built through
    # the factory.
    _transport: str | None = None

    @property
    def transport(self) -> str | None:
        """The factory-stamped wire this instance speaks (``None`` off the factory)."""
        return self._transport

    # The verdict the last :attr:`_writes_enabled` evaluation reached, so the
    # refusal it leads to is worded from the same answer without reading the
    # record a second time — a switch flipped between two reads would otherwise
    # produce a refusal for a cause the decision never saw. The write guard
    # clears it before each evaluation, so a subclass that OVERRIDES
    # ``_writes_enabled`` (and therefore never sets it) leaves it None and the
    # refusal asks for a verdict itself.
    _last_verdict: "WriteVerdict | None" = None

    @property
    def _writes_enabled(self) -> bool:
        """Whether the write verdict allows a write to this connector's machine.

        Decided by :func:`osprey_connectors.posture_store.write_verdict` for
        :attr:`_control_target` (see :func:`_monitor_verdict`), re-read on every
        call: a read-only run, the login's write tier, the target, the owner,
        the login's switch for that machine and the launch pin, first refusal
        wins. The switch is the live half: turning a machine on or off from the
        chip lands on the very next write of a session already running, with no
        respawn and no config edit.

        An unstamped connector — one nobody built through the factory — names
        no machine and is refused ``target_unknown``. Returns False rather than
        raising on any failure.

        The verdict is memoised on :attr:`_last_verdict` for the refusal the
        write guard is about to build, so one write reads the record once; a
        subclass overriding this property never sets it, and the refusal asks
        for a verdict itself.
        """
        verdict = _monitor_verdict(self._control_target)
        self._last_verdict = verdict
        return verdict.allowed

    def current_value_reader(self) -> Callable[[str], Any] | None:
        """This connector's own way to read a channel's present value.

        Handed to :meth:`LimitsValidator.validate` so the ``max_step`` check
        measures the step over the client and the target the write itself is
        bound for. The validator is shared by every connector and owns no
        client; a connector that cannot read synchronously answers ``None``
        here, and a ``max_step`` channel on it fails closed rather than being
        measured over somebody else's protocol.
        """
        return None

    def _validation_refusal(
        self, channel_address: str, value: Any, error: BaseException
    ) -> ChannelWriteResult:
        """The REFUSED result a non-limits validation error owes the caller.

        Limits validation answers a write in one of two ways: a
        :class:`ChannelLimitsViolationError`, which carries its own semantics
        and propagates to the caller unchanged, or any other error — which
        says the check could not be MADE. A check that could not be made is
        not permission to write: a validator that never answered has not said
        yes. So every connector turns that second case into this one REFUSED
        result and sends nothing.

        Spelled once, here, because the six-outcome contract is a promise
        every connector makes with the same words
        (``tests/connectors/test_cross_connector_parity.py``); four private
        copies of it would be four chances to drift.
        """
        logger.warning(
            f"Validation error - refusing write (fail-closed): {channel_address}: {error}"
        )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.REFUSED,
            refusal_reason=RefusalReason.VALIDATION_ERROR,
            error_message=f"Write to '{channel_address}' refused: validation error: {error}",
        )

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap write methods with the write-verdict pre-check.

        Any subclass that defines ``write_channel()`` or
        ``write_multiple_channels()`` gets them transparently wrapped.
        The wrapper checks ``_writes_enabled`` before calling the original
        method.  When the verdict refuses, returns ``REFUSED`` results worded
        by :func:`_verdict_refusal` — no exception is raised.

        This fires before limits validation (intentional: fast-reject when
        writes are disabled, avoiding unnecessary validation work).

        Past that check the original method runs inside
        :func:`~osprey_connectors.control_system.write_door.open_door`, so the
        raw-client guard attributes its put to a connector. A connector whose
        writes are disabled never opens the door.
        """
        super().__init_subclass__(**kwargs)

        original_write = cls.__dict__.get("write_channel")
        if original_write is not None:

            @functools.wraps(original_write)
            async def _guarded_write(self, channel_address, value, *args, **kwargs):
                self._last_verdict = None
                if not self._writes_enabled:
                    verdict = self._last_verdict or _monitor_verdict(self._control_target)
                    return _verdict_refusal(channel_address, value, verdict)
                with open_door():
                    return await original_write(self, channel_address, value, *args, **kwargs)

            cls.write_channel = _guarded_write

        original_multi = cls.__dict__.get("write_multiple_channels")
        if original_multi is not None:

            @functools.wraps(original_multi)
            async def _guarded_multi(self, operations, *args, **kwargs):
                self._last_verdict = None
                if not self._writes_enabled:
                    # One verdict for the batch, not one per operation: the
                    # whole batch was refused by one answer, and every result
                    # must tell the same story about why.
                    verdict = self._last_verdict or _monitor_verdict(self._control_target)
                    return [_verdict_refusal(addr, val, verdict) for addr, val in operations]
                with open_door():
                    return await original_multi(self, operations, *args, **kwargs)

            cls.write_multiple_channels = _guarded_multi

    def _resolve_confirm(self, channel_address: str) -> bool:
        """Whether a write to this channel must be confirmed by re-reading it.

        The limits database is the single home of write policy: the channel's
        own ``confirm``, else ``True``. A connector with no validator has limits
        checking disabled and no policy to read, so it takes the fleet default
        and confirms.
        """
        if self._limits_validator is None:
            return True
        return bool(self._limits_validator.resolve_confirm(channel_address))

    @abstractmethod
    async def connect(self, config: dict[str, Any]) -> None:
        """
        Establish connection to control system.

        Args:
            config: Control system-specific configuration

        Raises:
            ConnectorUnavailableError: If connection cannot be established
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to control system and cleanup resources."""

    @abstractmethod
    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value of a channel.

        Args:
            channel_address: Address/name of the channel
            timeout: Optional timeout in seconds

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ChannelUnreachableError: This channel cannot be reached or did not
                answer in time; the connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone
            TimeoutError: If operation times out
            ValueError: If channel address is invalid
            ChannelNotFoundError: The address is not a channel of the facility
                (a connector that holds the facility's address set)
        """

    @abstractmethod
    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to a channel, confirming it unless asked not to.

        Args:
            channel_address: Address/name of the channel
            value: Value to write
            timeout: Optional timeout in seconds
            confirm: Whether to re-read the channel and compare, or ``None`` to
                let the connector resolve the policy for this channel

        Returns:
            ChannelWriteResult carrying the outcome and what the channel was
            seen to hold

        Raises:
            ChannelUnreachableError: This channel cannot be reached or did not
                answer in time; the connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone
            TimeoutError: If operation times out
            ValueError: If value is invalid for this channel
            PermissionError: If write access is not allowed

        Outcomes:
            The connector sets exactly one :class:`WriteOutcome` on the result,
            and no consumer re-derives a verdict of its own:

            - ``refused`` — nothing was written; ``refusal_reason`` says why.
            - ``failed`` — the value was sent and the control system did not
              take it.
            - ``confirmed`` — a fresh re-read holds the value sent, in any alarm
              state. Alarm state is reported with the write, never raised on.
            - ``mismatch`` — the re-read holds a different value. A clamped or
              rounded setpoint is reported here, not tolerated.
            - ``unconfirmed`` — the value was sent, but not acknowledged in
              time or the re-read itself failed, so what the channel holds is
              unknown.
            - ``unrequested`` — ``confirm=False``: nothing was checked.

            The comparison is :func:`values_match`, one rule shared by every
            connector; there is no configurable tolerance anywhere.

        Omission sentinel:
            ``confirm=None`` means "no opinion" — it is NOT ``False``. A caller
            with no opinion leaves the keyword off entirely; the connector then
            resolves the policy for this specific channel through
            :meth:`_resolve_confirm`: the channel's own ``confirm`` entry in the
            limits database, else ``True``. The limits database is the single
            home of write policy.

            An explicit ``confirm=False`` is an answer and must travel as one, so
            every guard on the omission is ``if confirm is not None`` and never
            ``if confirm:`` — the truth test would silently turn a declined
            confirmation back into "resolve it" and confirm anyway.

        Two-layer safety model:
            **Per-write mechanical safety** lives INSIDE the connector and is applied
            per individual channel write: the ``writes_enabled`` gate, limits
            validation (min/max/step/writable), and the fail-closed validation path.
            This is complete mediation at the write primitive — every write passes
            through it.

            **Per-intent human authorization** is a SEPARATE layer at the tool
            boundary: the PreToolUse approval hook (and, for scans, the promote token)
            gate the *intent* to write, once per intent — not once per channel write.

            The two are orthogonal and complementary: the connector cannot be talked
            out of a mechanical refusal, and the approval layer cannot substitute for
            that refusal.
        """

    async def write_channel_checked(
        self, channel_address: str, value: Any, **kwargs: Any
    ) -> ChannelWriteResult:
        """Await write_channel and enforce the reference monitor's denial contract.

        ``refused`` (policy/limits/validation) -> raises ChannelWriteBlockedError.
        ``failed`` / ``mismatch`` / ``unconfirmed`` -> raises ChannelWriteFailedError.
        Connection failures and timeouts propagate unchanged.
        ``confirmed`` or ``unrequested`` -> returns the ChannelWriteResult.

        A scan device setter wraps this so any raise aborts the RunEngine, while a
        confirmed write returns and the scan proceeds; ``osprey.runtime.write_channel``
        routes through it for the same reason. Extra keyword arguments (``confirm``,
        ``timeout``) pass straight through to write_channel. The result inspection
        itself lives in :func:`raise_for_write_result`, which the multi-channel paths
        share.
        """
        from osprey_connectors.errors import (
            ChannelLimitsViolationError,
            ChannelWriteBlockedError,
        )

        try:
            result = await self.write_channel(channel_address, value, **kwargs)
        except ChannelLimitsViolationError as exc:
            # A limits refusal is a REFUSAL — normalize it into the unified denial type
            # so consumers key on one refusal signal. Connection failures and
            # timeouts are not caught here, so they propagate unchanged.
            raise ChannelWriteBlockedError(channel_address, "LIMITS", message=str(exc)) from exc

        return raise_for_write_result(result)

    async def _read_concurrently(
        self, channel_addresses: list[str], timeout: float | None
    ) -> dict[str, ChannelValue]:
        """Read channels concurrently, omitting failed reads and re-raising cancellations."""
        results = await asyncio.gather(
            *(self.read_channel(address, timeout) for address in channel_addresses),
            return_exceptions=True,
        )
        values: dict[str, ChannelValue] = {}
        for address, result in zip(channel_addresses, results, strict=True):
            if isinstance(result, Exception):
                continue
            if isinstance(result, BaseException):
                raise result
            values[address] = result
        return values

    @abstractmethod
    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """
        Read multiple channels efficiently (can be optimized per control system).

        Args:
            channel_addresses: List of channel addresses to read
            timeout: Optional timeout in seconds

        Returns:
            Dictionary mapping channel address to ChannelValue
            (May exclude channels that failed to read)
        """

    async def write_multiple_channels(
        self,
        operations: list[tuple[str, Any]],
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> list[ChannelWriteResult]:
        """
        Write multiple channels. Override for atomic/batched behavior.

        Default implementation writes sequentially via write_channel().
        Subclasses can override to provide transactional semantics (e.g.,
        disabling lattice recalculation between writes in a simulator).

        Args:
            operations: List of (channel_address, value) tuples
            timeout: Optional timeout in seconds
            confirm: Whether every channel in the batch is confirmed, or ``None``
                to let each channel resolve its own policy

        Returns:
            List of ChannelWriteResult in the same order as operations

        Note:
            A batch carries one ``confirm`` for every channel in it, so an omitted
            ``confirm`` is *not* resolved here — the keyword is simply left off
            each per-channel ``write_channel()`` call, and each channel then
            resolves its own policy from the limits database as documented on
            :meth:`write_channel`. That is what makes a batch write behave like a
            sequence of single writes.

            An explicit ``confirm`` — ``False`` every bit as much as ``True`` — is
            forwarded unchanged to every channel, which is why the guard below
            tests ``is not None`` rather than the value's truth.
        """
        results = []
        for address, value in operations:
            kwargs: dict[str, Any] = {"timeout": timeout}
            if confirm is not None:
                kwargs["confirm"] = confirm
            result = await self.write_channel(address, value, **kwargs)
            results.append(result)
        return results

    @abstractmethod
    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to channel changes.

        Args:
            channel_address: Address/name of the channel
            callback: Function called when value changes (receives ChannelValue)

        Returns:
            Subscription ID for later unsubscribe
        """

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Cancel subscription to channel changes.

        Args:
            subscription_id: Subscription ID returned by subscribe()
        """

    @abstractmethod
    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """
        Get metadata about a channel.

        Args:
            channel_address: Address/name of the channel

        Returns:
            ChannelMetadata with units, alarm status, description, and (where the
            control system reports one) a display range

        Raises:
            ChannelUnreachableError: This channel cannot be reached; the
                connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone
        """

    @abstractmethod
    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if channel exists and is accessible.

        Args:
            channel_address: Address/name of the channel

        Returns:
            True if channel is valid and accessible
        """
