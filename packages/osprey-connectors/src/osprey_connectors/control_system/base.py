"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

"""

import functools
import logging
import math
import numbers
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_connectors.types import (
    WRITES_ENABLED_KEY,
    type_writes_enabled,
    writes_enabled_answering_key,
    writes_enabled_key,
    writes_enabled_remedy,
)

if TYPE_CHECKING:
    # Type-checking only, and every annotation naming these is a string:
    # ``posture_store`` imports THIS module for :func:`is_readonly_run`, so a
    # load-time import here is the cycle both modules document. The functions
    # that need the names at runtime import them inside their own bodies.
    #
    # What that pairing costs: ``typing.get_type_hints()`` raises ``NameError``
    # on ``ControlSystemConnector`` and on ``_writes_disabled_result``, whose
    # string annotations name these types and nothing else in this module's
    # runtime namespace does. A reader of annotations — a schema generator, a
    # doc builder, a test — has to supply the two names itself.
    from osprey_connectors.posture_store import StoreVerdict, StoreVerdictDetail

logger = logging.getLogger("osprey_connectors.control_system")


@dataclass
class ChannelMetadata:
    """Metadata a control system reports about one of its channels.

    ``display_low`` / ``display_high`` are the range the control system suggests
    for DISPLAYING this channel (EPICS LOPR/HOPR and their equivalents). They are
    deliberately NOT named ``min_value`` / ``max_value``: those belong to
    :class:`~osprey_connectors.control_system.limits_validator.ChannelLimitsConfig`
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
    survives the connector-host IPC seam: :mod:`osprey_connectors.ipc.frames`
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
       compared with. A connector that reports no ``enum_label`` (Mock, DOOCS)
       falls through to the ordinary comparison.
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
    ``observed_value``.

    ``notes`` is display-only text ("observed 2.4, sent 2.5", "cannot be
    compared"). No consumer reads it, and nothing classifies a write by
    parsing it.

    ``refusal_reason`` names the policy that refused, and is set only when the
    outcome is ``refused``. ``observed_value`` is whatever the confirming
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
    # "WRITES_DISABLED" | "LIMITS" | "VALIDATION_ERROR" | "CONTROL_SYSTEM_REFUSED"
    # when refused, else None
    refusal_reason: str | None = None
    error_message: str | None = None  # Set iff refused, failed or unconfirmed
    observed_value: Any = None  # What the confirming re-read returned
    # Alarm name reported for the reading, e.g. "NO_ALARM"/"HIHI"; None if not reported
    alarm_status: str | None = None
    # Alarm severity: 0 healthy, higher is worse; None if not reported
    alarm_severity: int | None = None
    notes: str | None = None  # Display-only detail

    def __post_init__(self) -> None:
        """Coerce a string ``outcome`` to the :class:`WriteOutcome` member.

        A ``StrEnum`` crosses the connector-host IPC as the plain string JSON
        encodes it to, and the codec rebuilds the dataclass by calling it with
        the decoded fields — so what arrives here on the parent side is a
        ``str``. Coercing once, on construction, is what lets every consumer
        compare against :class:`WriteOutcome` members whichever side of the
        process boundary it runs on.
        """
        if not isinstance(self.outcome, WriteOutcome):
            self.outcome = WriteOutcome(self.outcome)

    @property
    def observed_number(self) -> float | None:
        """:attr:`observed_value` as a float, or ``None`` when it is not a number."""
        return as_number(self.observed_value)


def is_readonly_run() -> bool:
    """True when this process must refuse control-system writes, whatever it is.

    Two things set ``OSPREY_EXECUTION_MODE=readonly``, and this answers yes to
    both:

    * the **deployment**, when the whole run is started read-only — the
      variable is on every process the deployment owns, the MCP servers
      included, so ``mcp__controls__channel_write`` is refused right here
      rather than only by the (best-effort) hook chain;
    * the **python executor**, onto the sandbox subprocess of a run submitted
      with ``execution_mode="readonly"`` — so the per-run claim is enforceable
      at runtime, independent of the pre-execution pattern scan.

    Both are refusals; only the operator-facing wording differs, which
    :func:`_writes_disabled_result` decides through
    :func:`_in_mcp_server_process` — the discriminator, because the two are
    the same variable at the same value and only the process tells them apart.

    An operator's narrowing of ONE control target is deliberately NOT here: it
    never lived in the environment, and is read from the deployment's
    control-context record instead. So outside these two the variable is unset
    and the answer is still not "writes" by itself — the deployment posture
    (``control_system.writes_enabled``) and that record both apply, and
    :func:`_writes_disabled_result` asks them in turn.

    The comparison is by VALUE, never presence: ``readwrite`` is the writes
    posture, and a stale or misspelled value must not silently sandbox
    anything.
    """
    return os.environ.get("OSPREY_EXECUTION_MODE") == "readonly"


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


def _posture_store_verdict(control_target: str | None) -> "StoreVerdictDetail":
    """What the operator's record says about writing to *control_target*.

    The answer is a verdict and not a bool, because three things are findable
    and only one of them is a grant: nothing narrows this target, somebody
    narrowed it, or the narrowing could not be read at all. Collapsed into a
    bool, the last two become each other, and the refusal that follows tells an
    operator to go and flip a chip nobody ever consulted. The pair's second
    half is the sentence for that third answer alone — the path that could not
    be read, the bind that named none — which exists in the store's reader and
    nowhere this connector could reconstruct it from.

    This function is the ENV read and nothing else: the clause itself is
    :func:`osprey_connectors.posture_store.store_verdict_detail`, and it is
    delegated to rather than restated. The store contract's rule 3 has exactly
    two implementations — that module and the stdlib-only hook — and a
    connector that spelled the combining terms a third time would be a third
    thing to keep in step. What stays here is the ceiling: this connector's
    deployment posture is keyed on the connector TYPE, which is not a ceiling
    ``effective_writes`` can derive from a target, so the AND happens in
    :attr:`ControlSystemConnector._writes_enabled` rather than there.

    Evaluated on EVERY write, never cached: unlike the deployment posture, a
    narrowing set from the control-target chip has to land on a session that is
    already mid-conversation, which is the whole point of storing it in a file
    instead of delivering it by respawn. It is also evaluated BEFORE any thread
    or executor offload: the owner whose record it reads rides a context
    variable, and a hop that does not copy the context would read "nobody owns
    this" and hand a narrowed user the deployment ceiling.
    """
    # Imported here, not at module scope: ``posture_store`` imports this module
    # for :func:`is_readonly_run`, and the two must not import each other while
    # loading.
    from osprey_connectors import posture_store

    return posture_store.store_verdict_detail(control_target)


def _posture_store_permits(control_target: str | None) -> bool:
    """Whether the operator has left *control_target* writable on this deployment.

    :func:`_posture_store_verdict` compared against ``PERMITTED`` — the bool
    spelling, for a caller that has to decide whether to proceed and not what
    to say about it. The comparison is by NAME: every verdict is a non-empty
    string, so a safety answer that reaches a ``bool`` call site reads as yes
    for the two answers that refuse.

    It exists as that bool spelling and as nothing else: the monitor and the
    refusal both go through :func:`_posture_store_verdict`, and what holds this
    function is the parity table, which pins this module's two spellings
    against the store's own two. A reader finding no other caller is looking at
    a contract rather than at dead code.
    """
    from osprey_connectors.posture_store import StoreVerdict

    return _posture_store_verdict(control_target).verdict is StoreVerdict.PERMITTED


def _posture_record_verdict(control_target: str | None) -> "StoreVerdictDetail":
    """The record clause without the launch pin, for *control_target*.

    :func:`_posture_store_verdict` is the pin followed by this, and is what
    decides a write. This exists for the one caller that must tell a pin
    refusal from a record refusal: a refusal naming the pin alone would send an
    operator to re-run a script into a narrowing nobody mentioned. Delegated to
    :func:`osprey_connectors.posture_store.record_verdict_detail` rather than
    restated, for the reason :func:`_posture_store_verdict` gives.

    Reads the record, which the pin's own refusal does not, so it is asked only
    where the pin's early return left that read unspent.
    """
    # Imported here for the reason :func:`_posture_store_verdict` gives: the
    # two modules may not import each other while loading.
    from osprey_connectors import posture_store

    return posture_store.record_verdict_detail(control_target)


def _deployment_posture(connector_type: str | None) -> tuple[bool, str]:
    """The deployment half of the write posture, and the key that answered it.

    Config alone, fail-safe, reading no environment: see
    :func:`~osprey_connectors.types.type_writes_enabled`. The key comes back
    beside the verdict because a refusal names the line that refused, and on a
    deployment that states its posture once for everything that line is the
    deployment-wide key rather than the per-type one an operator would edit.

    A config nothing can read arms nothing, and names the key the remedy names:
    no line answered, so there is no second line to point at.
    """
    try:
        from osprey_connectors.config import get_config_value

        if connector_type is None:
            return get_config_value(WRITES_ENABLED_KEY, False) is True, WRITES_ENABLED_KEY
        section = get_config_value("control_system", {})
        return (
            type_writes_enabled(section, connector_type),
            writes_enabled_answering_key(section, connector_type),
        )
    except (FileNotFoundError, RuntimeError):
        return False, writes_enabled_key(connector_type)


def _deployment_writes_enabled(connector_type: str | None) -> bool:
    """The deployment half of the write posture — config alone, fail-safe.

    Split out of :attr:`ControlSystemConnector._writes_enabled` so the refusal
    message can ask the same question the monitor asked, rather than carry a
    second spelling of it.
    """
    return _deployment_posture(connector_type)[0]


#: How a refusal introduces its second and further reasons. Each reason is a
#: sentence of its own, so an operator reads what to do about one without having
#: to take the others apart.
_FURTHER_REASON = "Also, "


def _deployment_reason(
    connector_type: str | None, answering_key: str | None = None, *, one_line: bool = False
) -> str:
    """The sentence for a deployment whose config does not arm this type.

    *answering_key* is the key that refused. It is spoken only when the remedy
    neither arms it nor already quotes it — a deployment stating its posture
    once for everything is refused on a different line from the one that arms
    this single type, and that is the case worth a clause. The two suppressions
    are different needs: a key the remedy arms would be stated twice, and a
    dotted type's own key is one an operator must never be sent to write, since
    a profile splits it on its dots and nothing would read the result.
    ``None`` names no key at all — the last-resort wording, for a refusal no
    posture this function can read accounts for.

    *one_line* keeps the sentence on one line, for a message that carries more
    than this reason.
    """
    remedy = writes_enabled_remedy(connector_type, one_line=one_line)
    speak_key = (
        answering_key is not None
        and answering_key != writes_enabled_key(connector_type)
        and answering_key not in remedy
    )
    named = f" by {answering_key}" if speak_key else ""
    return f"writes are disabled{named}. {remedy} Then rebuild and redeploy."


def _readonly_reason(*, deployment_wide: bool) -> str:
    """The sentence for a run started read-only, in the shape it was started.

    *deployment_wide* separates the two: the deployment-wide run and one
    readonly script submission are the same variable at the same value, and only
    the process tells them apart (:func:`_in_mcp_server_process`). Both carry the
    same "readonly execution mode" substring
    (``wrapper.READONLY_REFUSAL_MARKER``), which keeps the refusal recognisable
    to the stderr matcher whichever shape produced it.
    """
    if deployment_wide:
        return (
            "this deployment is running in readonly execution mode "
            "(OSPREY_EXECUTION_MODE=readonly), which refuses control-system writes "
            "for every session. The control-target chip in the header cannot lift it."
        )
    return (
        "this script is running in readonly execution mode. Resubmit it with "
        "execution_mode='readwrite' (human approval required) if the write is intended."
    )


def _launch_reason(control_target: str | None) -> str | None:
    """The sentence for a launch pin that refuses this run, or ``None``.

    Read from the environment and nothing else, so it costs no record read and
    can be asked before anything opens a config or a file. Its two wordings
    separate "writes were off for this target when the run started" from
    "nothing could be resolved at launch, so the run was pinned everywhere" —
    the second is nobody's decision and must not be reported as one.
    """
    # Imported here for the reason :func:`_posture_store_verdict` gives: the
    # two modules may not import each other while loading.
    from osprey_connectors.posture_store import (
        LAUNCH_POSTURE_ALL_TARGETS,
        launch_narrowed_target,
        launch_permits,
    )

    if launch_permits(control_target):
        return None
    launched = launch_narrowed_target()
    if launched == LAUNCH_POSTURE_ALL_TARGETS:
        # The executor could not name a target, or could not read the store at
        # all, and pinned the run everywhere. Nobody decided this, so the message
        # must not send anyone to the chip to undo a decision they never made.
        return (
            "this run launched under the most restrictive write state — at launch "
            "neither its control target nor the recorded write state for it could "
            "be resolved, so the run was pinned with writes off for every target. "
            "A write state set since applies to the next run, not to one already "
            "in flight. Re-run the script to pick up the current write state."
        )
    return (
        f"this run launched while writes were off for '{launched}'; a write state "
        "set since applies to the next run, not to one already in flight. "
        "Re-run the script to pick it up."
    )


def _unreadable_reason(control_target: str | None, reason: str | None) -> str:
    """The sentence for a record that could not be read at all.

    Nobody decided this, so the chip is not offered, and *reason* — the
    reader's own sentence naming the path that failed — is quoted because no
    other code can compose it.
    """
    unreadable_for = f" for '{control_target}'" if control_target else ""
    cause = reason or "the reader that refused did not say which path failed"
    return (
        f"the recorded write state{unreadable_for} could not be read, so the write "
        f"is refused rather than run at the deployment ceiling: {cause}. This is "
        "not a narrowing anybody set, and the control-target chip does not lift "
        "it: the store answered control_context_unavailable."
    )


def _chip_reason(control_target: str | None, *, deployment_arms: bool) -> str:
    """The sentence for a write the operator has turned off from the chip."""
    if control_target:
        where = f"the '{control_target}' control target"
        remedy = f"Turn writes back on for '{control_target}' from the chip"
    else:
        # No stamp, so the most restrictive entry decided and this connector
        # genuinely cannot say which target that was. Naming one would be a
        # guess an operator then acts on.
        where = (
            "at least one control target (this connector was built "
            "without one, so the most restrictive of them decides)"
        )
        remedy = "Turn writes back on from the chip"
    narrowing = (
        f"writes are off for {where} — turned off from the control-target chip in the header"
    )
    if not deployment_arms:
        # A reason, not a remedy: the deployment refuses this write whichever way
        # the chip is set, so the flip is not offered and config.yml is not
        # denied as the gate.
        return f"{narrowing}. The store answered narrowing."
    return (
        f"{narrowing}; applies deployment-wide. {remedy} if the write is intended; "
        "config.yml is not the gate here. The store answered narrowing."
    )


def _writes_disabled_result(
    channel_address: str,
    value: Any,
    connector_type: str | None = None,
    control_target: str | None = None,
    *,
    store_verdict: "StoreVerdict | None" = None,
    store_reason: str | None = None,
    record_verdict: "StoreVerdict | None" = None,
    record_reason: str | None = None,
) -> ChannelWriteResult:
    """Build the refusal result for a write the monitor never attempted.

    *store_verdict* is the store answer the monitor already got, handed down so
    that one write makes exactly ONE store read. Without it this function would
    ask again, and a narrowing lifted between the two reads would produce a
    refusal that named the wrong cause. ``None`` means nobody answered — a
    direct caller rather than the write guard — and the store is read here. It
    travels as the verdict rather than as a bool: the store can refuse for a
    reason that is nobody's decision, and a message that reports that as a chip
    somebody set sends an operator to undo something they never did.

    *store_reason* is the sentence that belongs with that verdict — which
    record could not be read, which bind names no path — and it is set for
    ``CONTROL_CONTEXT_UNAVAILABLE`` alone. It is composed where the read
    failed and nowhere this function could reconstruct it, so without it that
    refusal names a cause and no remedy. It rides down beside the verdict on
    the guarded path and comes off the same single read on the direct one.

    *record_verdict* and *record_reason* are the record clause without the
    pin, handed down so a pinned batch reads the record once. They matter only
    when the pin refused: the pin answers the store without opening the record,
    so without them a pinned run would name the pin and hide a narrowing or an
    unreadable record behind it. ``None`` means nobody answered, and the record
    is read here — once, and only where the pin left that read unspent.

    Several policies can refuse one write at once, and each sends the operator
    somewhere different, so the message names EVERY reason that holds. Lifting
    one leaves the others in force, and a refusal naming a single reason has an
    operator arm a deployment, rebuild it, redeploy it and be refused again.

    The reasons, in the order they are spoken:

    1. the **deployment** does not arm writes for this connector type, which is
       the only one ``writes_enabled`` governs. First because it outlives every
       run and every session: whatever else is lifted, this one still refuses.
       Posture is per type, so the remedy names the block an operator actually
       edits — ``control_system.connector.<type>.writes_enabled`` when the
       connector knows its type, the deployment-wide key when it does not —
       while the reason names the key that answered, which on a deployment with
       one posture for everything is the deployment-wide key
       (:func:`~osprey_connectors.types.writes_enabled_answering_key`).
    2. the run is **readonly** — the whole deployment started that way
       (``OSPREY_EXECUTION_MODE=readonly``, which refuses control-system writes
       for every session and cannot be lifted from the chip), or this one
       script was submitted readonly and can be resubmitted as readwrite. A
       readonly run stops at its own reason rather than reading the record: a
       run that refuses whatever the record says has no business opening it.
    3. the run's **launch pin**, which no widen reaches and only a re-run picks
       up, and then the **record**: the operator's own writes-off for one
       control target, which the chip lifts, or a record that could not be read
       at all, which nobody set and no click lifts. Both are spoken when both
       hold — a pinned run that is also narrowed on the chip would otherwise
       send an operator to re-run a script into a narrowing nobody named. It
       costs no extra record read: the pin refuses ahead of the record, leaving
       the one read this write is allowed unspent.

    Only the wording forks. The ``refused`` outcome and ``refusal_reason`` stay
    the same however many reasons are spoken: the same thing happened — the
    monitor refused, and the control system was never asked — and every caller
    of :func:`raise_for_write_result` already handles it under that one word.
    The verdict that produced a store clause therefore rides in
    ``error_message``, which is the only carrier that reaches an operator, and
    each store clause closes on the verdict of the clause it reports. After a
    pin that is the record's verdict, and the pin's own wording names the run,
    not the store.
    """
    # Imported here for the reason :func:`_posture_store_verdict` gives: the
    # two modules may not import each other while loading.
    from osprey_connectors.posture_store import StoreVerdict

    # The run's own terms and the store are read BEFORE any config: the first
    # config read of a process loads the deployment's .env chain into the
    # environment, so an explanation assembled after it could describe a run
    # the monitor's own verdict never saw.
    readonly = is_readonly_run()
    deployment_wide = readonly and _in_mcp_server_process()
    launched: str | None = None
    if not readonly:
        if store_verdict is None:
            # Both halves off the one read: asking separately for the sentence
            # would be a second read of a file that can change between them.
            detail = _posture_store_verdict(control_target)
            store_verdict = detail.verdict
            store_reason = detail.reason
        launched = _launch_reason(control_target)
        if launched is not None:
            # The pin refused without the record being opened, so the read this
            # spends is the one this write has not spent.
            if record_verdict is None:
                record = _posture_record_verdict(control_target)
                record_verdict = record.verdict
                record_reason = record.reason
        else:
            record_verdict, record_reason = store_verdict, store_reason

    deployment_arms, answering_key = _deployment_posture(connector_type)

    run_reasons: list[str] = []
    if readonly:
        run_reasons.append(_readonly_reason(deployment_wide=deployment_wide))
    else:
        if launched is not None:
            run_reasons.append(launched)
        # By NAME, never by truthiness: every verdict is a non-empty string.
        # Anything that is not the unavailable verdict and not a grant words as
        # a narrowing, so a verdict added later refuses rather than reads as
        # permitted.
        if record_verdict is StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE:
            run_reasons.append(_unreadable_reason(control_target, record_reason))
        elif record_verdict is not StoreVerdict.PERMITTED:
            run_reasons.append(_chip_reason(control_target, deployment_arms=deployment_arms))

    reasons: list[str] = []
    if not deployment_arms:
        reasons.append(
            _deployment_reason(connector_type, answering_key, one_line=bool(run_reasons))
        )
    reasons.extend(run_reasons)
    if not reasons:
        # Nothing the monitor can see refuses this write: a direct caller, or a
        # narrowing lifted between the monitor's verdict and this message. The
        # deployment's own wording is the last resort, and names no key, because
        # no key answered.
        reasons.append(_deployment_reason(connector_type))
    spoken = " ".join(
        reason if position == 0 else f"{_FURTHER_REASON}{reason}"
        for position, reason in enumerate(reasons)
    )
    return ChannelWriteResult(
        channel_address=channel_address,
        value_written=value,
        outcome=WriteOutcome.REFUSED,
        refusal_reason="WRITES_DISABLED",
        error_message=f"Write to '{channel_address}' blocked: {spoken}",
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
            written, and ``refusal_reason`` names the policy that refused —
            the monitor on posture, limits or validation grounds, or the
            control system itself (``CONTROL_SYSTEM_REFUSED``).
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
            result.refusal_reason or "WRITES_DISABLED",
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
    (EPICS, LabVIEW, Tango, Mock, etc.) using a unified API.

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
    # What the last :attr:`_writes_enabled` evaluation saw in the posture store,
    # so the refusal it leads to can name the right cause without reading the
    # store a second time — a narrowing lifted between the two reads would
    # otherwise produce a refusal blaming the chip for a write the deployment
    # turned down. The write guard clears it before each evaluation, so a
    # subclass that OVERRIDES ``_writes_enabled`` (and therefore never sets it)
    # leaves it None and the refusal reads the store for itself, exactly as it
    # did before this memo existed. It holds the VERDICT, not whether the write
    # may proceed: one of the three answers is nobody's decision, and a refusal
    # that cannot tell it from a chip flip words it as one.
    _last_store_verdict: "StoreVerdict | None" = None
    # The operator sentence that belongs with the verdict above, and only ever
    # set for the one verdict that has one: which record could not be read,
    # which bind names no path. The reader composes it and no other process can,
    # so it rides here rather than being derived where the refusal is worded.
    _last_store_reason: str | None = None
    # The record clause of that same evaluation, without the launch pin, and
    # its sentence. They equal the store pair above whenever the pin permits,
    # and are read separately only when the pin refused ahead of the record —
    # which is what lets that refusal name a narrowing or an unreadable record
    # the pin's own answer hid, off one record read per write or batch.
    _last_record_verdict: "StoreVerdict | None" = None
    _last_record_reason: str | None = None

    @property
    def _writes_enabled(self) -> bool:
        """Check whether writes are enabled for this connector's type.

        Returns False (fail-safe) when config is unavailable.

        Posture is per connector type:
        ``control_system.connector.<type>.writes_enabled`` arms one type, and a
        deployment that says nothing about a type keeps the deployment-wide
        ``control_system.writes_enabled`` for it — see
        :func:`~osprey_connectors.types.type_writes_enabled`. An unstamped
        connector has no type to key that on, so the deployment-wide key is the
        whole posture it can read — under the same rule as every other reader:
        only a literal ``true`` arms it, never a truthy stand-in.

        ``control_system.writes_enabled`` is a **launch-time deployment posture,
        not a live kill-switch.** It is read from config and process-cached, so
        flipping it in ``config.yml`` does NOT take effect in a running process.
        The enforced kill-switch lives at the harness layer: a renderer
        ``permissions.deny`` on the write tool, followed by regenerating and
        relaunching the agent. In-flight control of an active scan is the
        RunEngine's own ``abort`` / ``pause`` — never a config flag.

        That paragraph describes the DEPLOYMENT half alone. Two live terms are
        ANDed with it, and both are re-read on every call:

        * a readonly sandbox run (see :func:`is_readonly_run`) is refused
          regardless of the deployment posture;
        * the operator's own narrowing for :attr:`_control_target`, read from
          the deployment's control-context record on every write (see
          :func:`_posture_store_verdict`, whose bool spelling for a caller that
          needs no reason is :func:`_posture_store_permits`). That is the live
          half the deployment posture deliberately is not: a target flipped to
          read-only from the control-target chip refuses the very next write on
          a session that is already running, with no respawn and no config edit.

        The record can only narrow. Nothing in it widens the deployment's
        ceiling, and an unreadable or absent record leaves that ceiling exactly
        as it was.

        The store answer is memoised on :attr:`_last_store_verdict`, with its
        sentence on :attr:`_last_store_reason`, for the refusal the write guard
        is about to build, so one write reads the store once; a subclass
        overriding this property simply never sets them, and the refusal falls
        back to reading for itself. :attr:`_last_record_verdict` and
        :attr:`_last_record_reason` carry the record clause of the same
        evaluation without the launch pin, so a pinned refusal can name what
        the record says without opening it a second time.
        """
        if is_readonly_run():
            return False
        # Imported here for the reason :func:`_posture_store_verdict` gives:
        # the two modules may not import each other while loading.
        from osprey_connectors.posture_store import StoreVerdict, launch_permits

        store = _posture_store_verdict(self._control_target)
        self._last_store_verdict = store.verdict
        self._last_store_reason = store.reason
        # The record clause, for at most one record read per evaluation. While
        # the pin permits, the store's answer IS the record's and the read is
        # spent. Only a pin refusal leaves the record unread, and only there is
        # it read.
        if launch_permits(self._control_target):
            self._last_record_verdict = store.verdict
            self._last_record_reason = store.reason
        else:
            record = _posture_record_verdict(self._control_target)
            self._last_record_verdict = record.verdict
            self._last_record_reason = record.reason
        if store.verdict is not StoreVerdict.PERMITTED:
            return False
        return _deployment_writes_enabled(self._connector_type)

    def _current_value_reader(self) -> Callable[[str], Any] | None:
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
            refusal_reason="VALIDATION_ERROR",
            error_message=f"Write to '{channel_address}' refused: validation error: {error}",
        )

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap write methods with writes_enabled pre-check.

        Any subclass that defines ``write_channel()`` or
        ``write_multiple_channels()`` gets them transparently wrapped.
        The wrapper checks ``_writes_enabled`` before calling the original
        method.  When writes are disabled, returns failure results with an
        operator-facing error message — no exception is raised.

        This fires before limits validation (intentional: fast-reject when
        writes are disabled, avoiding unnecessary validation work).
        """
        super().__init_subclass__(**kwargs)

        original_write = cls.__dict__.get("write_channel")
        if original_write is not None:

            @functools.wraps(original_write)
            async def _guarded_write(self, channel_address, value, *args, **kwargs):
                self._last_store_verdict = None
                self._last_store_reason = None
                self._last_record_verdict = None
                self._last_record_reason = None
                if not self._writes_enabled:
                    return _writes_disabled_result(
                        channel_address,
                        value,
                        self._connector_type,
                        self._control_target,
                        store_verdict=self._last_store_verdict,
                        store_reason=self._last_store_reason,
                        record_verdict=self._last_record_verdict,
                        record_reason=self._last_record_reason,
                    )
                return await original_write(self, channel_address, value, *args, **kwargs)

            cls.write_channel = _guarded_write

        original_multi = cls.__dict__.get("write_multiple_channels")
        if original_multi is not None:

            @functools.wraps(original_multi)
            async def _guarded_multi(self, operations, *args, **kwargs):
                self._last_store_verdict = None
                self._last_store_reason = None
                self._last_record_verdict = None
                self._last_record_reason = None
                if not self._writes_enabled:
                    # One store read for the batch, not one per operation: the
                    # whole batch was refused by one verdict, and every result
                    # must tell the same story about why.
                    verdict = self._last_store_verdict
                    reason = self._last_store_reason
                    recorded = self._last_record_verdict
                    recorded_reason = self._last_record_reason
                    return [
                        _writes_disabled_result(
                            addr,
                            val,
                            self._connector_type,
                            self._control_target,
                            store_verdict=verdict,
                            store_reason=reason,
                            record_verdict=recorded,
                            record_reason=recorded_reason,
                        )
                        for addr, val in operations
                    ]
                return await original_multi(self, operations, *args, **kwargs)

            cls.write_multiple_channels = _guarded_multi

    def _resolve_confirm(self, channel_address: str) -> bool:
        """Whether a write to this channel must be confirmed by re-reading it.

        The limits database is the single home of write policy: the channel's
        own ``confirm`` → the ``defaults`` block's ``confirm`` → ``True``. A
        connector with no validator has limits checking disabled and no policy
        to read, so it takes the fleet default and confirms.
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
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to control system and cleanup resources."""
        pass

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
            ConnectionError: If channel cannot be reached
            TimeoutError: If operation times out
            ValueError: If channel address is invalid
        """
        pass

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
            ConnectionError: If channel cannot be reached
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
            limits database, then the ``defaults`` block, then ``True``. The
            limits database is the single home of write policy.

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
        pass

    async def write_channel_checked(
        self, channel_address: str, value: Any, **kwargs: Any
    ) -> ChannelWriteResult:
        """Await write_channel and enforce the reference monitor's denial contract.

        ``refused`` (policy/limits/validation) -> raises ChannelWriteBlockedError.
        ``failed`` / ``mismatch`` / ``unconfirmed`` -> raises ChannelWriteFailedError.
        Native ConnectionError/TimeoutError from the transport -> propagate unchanged.
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
            # so consumers key on one refusal signal. (ConnectionError/TimeoutError are
            # NOT caught here, so they propagate unchanged.)
            raise ChannelWriteBlockedError(channel_address, "LIMITS", message=str(exc)) from exc

        return raise_for_write_result(result)

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
        pass

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
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Cancel subscription to channel changes.

        Args:
            subscription_id: Subscription ID returned by subscribe()
        """
        pass

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
            ConnectionError: If channel cannot be reached
        """
        pass

    @abstractmethod
    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if channel exists and is accessible.

        Args:
            channel_address: Address/name of the channel

        Returns:
            True if channel is valid and accessible
        """
        pass
