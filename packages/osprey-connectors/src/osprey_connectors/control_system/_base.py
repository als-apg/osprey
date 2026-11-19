"""
Abstract base class for control system connectors.

Provides protocol-agnostic interfaces for reading/writing process variables,
subscribing to changes, and retrieving metadata from various control systems.

"""

import asyncio
import logging
import math
import numbers
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    final,
)

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import override

from osprey_connectors import write_authorization, write_switch
from osprey_connectors._contract_types import ContractAny
from osprey_connectors.call_timeout import DEFAULT_TIMEOUT_S, TIMEOUT_KEY, call_timeout_s
from osprey_connectors.config_sections import (
    ConnectorSettings,
    ControlSystemSection,
    settings_key,
)
from osprey_connectors.connector_types import Transport
from osprey_connectors.control_system.write_door import open_door
from osprey_connectors.errors import (
    ChannelError,
    ChannelReadFailedError,
    LimitsViolation,
)
from osprey_connectors.execution_mode import is_readonly_run
from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping, evolve
from osprey_connectors.refusals import RefusalReason, VerdictReason, VerdictReasonField
from osprey_connectors.write_switch import Allowed, Refused, WriteVerdict

if TYPE_CHECKING:
    from osprey_connectors.control_system._limits_validator import (
        CurrentValueReader,
        LimitsValidator,
    )
    from osprey_connectors.errors import ChannelLimitsViolationError

logger = logging.getLogger("osprey_connectors.control_system")

#: The limit pairs of :class:`ChannelMetadata`, each ``(low, high)``.
_LIMIT_PAIRS = (
    ("display_low", "display_high"),
    ("control_low", "control_high"),
    ("warning_low", "warning_high"),
    ("alarm_low", "alarm_high"),
)


class ChannelMetadata(FrozenModel):
    """Metadata a control system reports about one of its channels.

    ``display_low`` / ``display_high`` are the range the control system suggests
    for DISPLAYING this channel (EPICS LOPR/HOPR and their equivalents). They are
    deliberately NOT named ``min_value`` / ``max_value``: those belong to
    :class:`~osprey_connectors.control_system.ChannelLimitsConfig`
    and are the bounds OSPREY refuses to write past. Only the latter is enforced —
    a value inside the display range can still be refused, and a connector that
    reports no display range constrains nothing.

    The record holds what describes the channel, not one reading of it: the
    alarm state and the label of the value read ride on the
    :class:`ChannelValue` itself (:attr:`ChannelValue.alarm`,
    :attr:`ChannelValue.enum_label`).

    ``enum_labels`` is present only for enum-typed records (EPICS
    ``mbbi``/``mbbo``/``bi``/``bo``, PVAccess ``NTEnum``, and their equivalents
    elsewhere) and is ``None`` on every other channel: every state the channel
    can report, in index order. It may be ``None`` even on an enum channel: a
    control system reports the label list only when it has one.

    ``enum_labels`` is a tuple, and takes a list on input: the connector-host
    IPC (:mod:`osprey_connectors.ipc._frames`) carries it as a JSON array,
    which decodes as a list, and the field turns it back into the tuple the
    sender held, so the field compares equal on both sides of the seam.

    ``control_low`` / ``control_high`` (EPICS DRVL/DRVH), ``warning_low`` /
    ``warning_high`` (LOW/HIGH) and ``alarm_low`` / ``alarm_high`` (LOLO/HIHI)
    are the remaining limit pairs a control system reports. Like the display
    range they describe the channel; none of them is a write bound OSPREY
    enforces. ``writable`` is this client's access right on the channel (EPICS
    CA write access) — not the user's tier — and ``None`` where the control
    system does not report one.

    Every limit pair, the display range included, passes through
    :func:`normalise_limit_pair` on construction: a missing, non-numeric or non-finite bound becomes ``None``, and a pair whose
    low is not below its high (EPICS reports an unset range as ``0``/``0``)
    becomes ``None`` on both sides. A connector may therefore hand a limit over
    as whatever its transport reported. The normalisation is idempotent, so a
    value decoded on the far side of the connector-host IPC seam — which
    validates it again — is unchanged. ``writable`` takes a ``bool`` or an
    integer ``0``/``1`` and reads anything else as "not reported"; a
    ``raw_metadata`` of ``None`` is an empty one, and the record holds it
    read-only (:data:`~osprey_connectors.model.ReadOnlyMapping`). Every other
    field holds a value of its own type, or the construction raises.
    """

    units: str = ""
    precision: int | None = None
    description: str | None = None
    display_low: float | None = None
    display_high: float | None = None
    #: Every state this enum-typed channel can report, in index order; None off enums.
    enum_labels: FrozenTuple[str] | None = None
    raw_metadata: ReadOnlyMapping[str, ContractAny] | None = Field(default_factory=empty_mapping)
    #: Drive range (EPICS DRVL/DRVH); None when unreported or not a valid range.
    control_low: float | None = None
    control_high: float | None = None
    #: Minor-alarm range (EPICS LOW/HIGH); None when unreported or not a valid range.
    warning_low: float | None = None
    warning_high: float | None = None
    #: Major-alarm range (EPICS LOLO/HIHI); None when unreported or not a valid range.
    alarm_low: float | None = None
    alarm_high: float | None = None
    #: This client's write access to the channel; None when the control system does not say.
    writable: bool | None = None

    @field_validator("raw_metadata", mode="before")
    @classmethod
    def _raw_metadata_dict(cls, value: object) -> object:
        """``None`` is an empty ``raw_metadata``."""
        return {} if value is None else value

    @field_validator(*(bound for pair in _LIMIT_PAIRS for bound in pair), mode="before")
    @classmethod
    def _limit(cls, value: object) -> float | None:
        """One bound as a finite ``float``, or ``None`` for anything else."""
        return _limit_bound(value)

    @field_validator("writable", mode="before")
    @classmethod
    def _writable(cls, value: object) -> bool | None:
        """A reported write-access right as ``bool``, or ``None`` when not reported."""
        return _writable_flag(value)

    @model_validator(mode="before")
    @classmethod
    def _limit_pairs(cls, data: object) -> object:
        """Each limit pair as one pair: a low not below its high leaves neither.

        Runs on the fields as given, before each bound is validated, so the
        record is built once with the pair it keeps.
        """
        if not isinstance(data, Mapping):
            return data
        fields = dict(data)
        for low_name, high_name in _LIMIT_PAIRS:
            if low_name in fields or high_name in fields:
                fields[low_name], fields[high_name] = normalise_limit_pair(
                    fields.get(low_name), fields.get(high_name)
                )
        return fields


def _limit_bound(value: object) -> float | None:
    """One limit as a finite ``float``, or ``None`` for anything else.

    Bools are excluded (``True`` is not a limit), as is every non-numeric value —
    including a ``MagicMock`` standing in for a PV in tests — so a connector may
    hand over whatever its transport reported without checking it first.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        bound = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return bound if math.isfinite(bound) else None


def normalise_limit_pair(low: object, high: object) -> tuple[float | None, float | None]:
    """Normalise one ``(low, high)`` limit pair the way :class:`ChannelMetadata` stores it.

    Each bound that is missing, non-numeric or non-finite becomes ``None``. When
    both survive and ``low >= high`` the pair describes no range — EPICS reports
    an unset range as ``0``/``0`` — so both become ``None``. A pair with only one
    valid bound keeps it. Idempotent.
    """
    low_bound, high_bound = _limit_bound(low), _limit_bound(high)
    if low_bound is not None and high_bound is not None and low_bound >= high_bound:
        return None, None
    return low_bound, high_bound


def _writable_flag(value: object) -> bool | None:
    """A reported write-access right as ``bool``, or ``None`` when not reported.

    A client may report access rights as ``bool`` or ``0``/``1``; anything that
    is not an integer (a ``MagicMock`` attribute, say) is "not reported".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return bool(value)
    return None


class Alarm(FrozenModel):
    """The alarm state a control system reported with one reading.

    Carried protocol-agnostically: the alarm *name* as a string ("NO_ALARM",
    "HIHI"), the severity as an int, 0 healthy and higher worse. Mapping a
    protocol's raw codes to those words is the connector's job, so this shared
    module imports no control-system constants. Either field is ``None`` when
    the control system did not report it, which stays distinct from a reported
    healthy reading (``severity=0``).
    """

    #: The alarm name, e.g. "NO_ALARM"/"HIHI"; None if not reported.
    status: str | None = None
    #: 0 healthy, higher is worse; None if not reported.
    severity: int | None = None


class ChannelValue(FrozenModel):
    """One reading of a control system channel.

    ``address`` names the channel read. ``sampled_at`` is the instant the
    control system stamped the reading with (the IOC's or device's own time
    stamp), or the instant the connector read it when the source reported
    none. Timezone-aware; every built-in connector stamps it in UTC.

    ``alarm`` is the alarm state reported with this reading, filled by the
    connector at read time, and ``None`` when the control system reports no
    alarm state at all (DOOCS). Alarm state is information, never a reason to
    raise.

    ``enum_label`` is the *readable* half of an enum reading: ``value`` stays
    the integer index — one machine-readable type across every protocol — and
    the label for that index rides here, resolved at read time. It is ``None``
    off enum channels and on an index outside the reported labels
    (:attr:`ChannelMetadata.enum_labels`).

    ``metadata`` holds what describes the channel rather than this reading.
    """

    address: str
    value: ContractAny
    sampled_at: AwareDatetime
    alarm: Alarm | None = None
    #: The label for the value of THIS reading; None off enums and when unresolvable.
    enum_label: str | None = None
    metadata: ChannelMetadata = Field(default_factory=ChannelMetadata)


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


class _Vector(Protocol):
    """What :func:`values_match` compares positionally: a list, a tuple or an ndarray."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> object: ...

    def __iter__(self) -> Iterator[object]: ...


def _is_sequence(value: object) -> TypeGuard[_Vector]:
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


def _unwrap_length_one(value: object) -> object:
    """A length-1 sequence reduced to its element; anything else unchanged.

    ``[5.0]`` and ``5.0`` name the same reading: control systems differ on
    whether a single-element channel reads back boxed, and that difference must
    not decide an outcome.
    """
    if _is_sequence(value) and len(value) == 1:
        return value[0]
    return value


def _scalars_match(sent: object, observed: object) -> bool:
    """Compare two scalars: floats by relative closeness, everything else by ``==``."""
    # The bools are excluded first: ``bool`` is a ``numbers.Real`` at runtime,
    # though the type stubs do not say so.
    if (
        not isinstance(sent, bool)
        and not isinstance(observed, bool)
        and isinstance(sent, numbers.Real)
        and isinstance(observed, numbers.Real)
    ):
        return math.isclose(float(sent), float(observed), rel_tol=1e-6, abs_tol=0.0)

    # Not always a bool: a numpy comparison answers an array.
    result: object = sent == observed
    if isinstance(result, bool):
        return result
    # A numpy (or other array-like) elementwise comparison reduces to one verdict.
    return bool(result.all()) if hasattr(result, "all") else bool(result)


def values_match(sent: object, observed: object, *, enum_label: str | None = None) -> bool:
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

        if _is_sequence(sent) and _is_sequence(observed):
            if len(sent) != len(observed):
                return False
            return all(_scalars_match(a, b) for a, b in zip(sent, observed, strict=True))
        if _is_sequence(sent) or _is_sequence(observed):
            return False

        return _scalars_match(sent, observed)
    except Exception:
        return False


def as_number(value: object) -> float | None:
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

#: The outcomes a confirming re-read decided, the only ones that carry it.
_OBSERVED_OUTCOMES = frozenset({WriteOutcome.CONFIRMED, WriteOutcome.MISMATCH})


class ChannelWriteResult(FrozenModel):
    """What became of one channel write, and what the channel was seen to hold.

    The control-system-agnostic result type returned by every connector.
    :attr:`outcome` is the single owned verdict (see :class:`WriteOutcome`),
    set by the connector that performed the write; no consumer re-derives one.

    ``error_message`` is set **if and only if** ``outcome`` is one of
    ``refused``, ``failed`` or ``unconfirmed`` — the three words that carry
    something the operator has to be told. A ``mismatch`` needs no message:
    both numbers are on the result already, in ``value_written`` and
    ``observed``. Construction enforces it: a result that breaks it raises
    ``ValueError``.

    ``notes`` is display-only text ("observed 2.4, sent 2.5", "cannot be
    compared"). No consumer reads it, and nothing classifies a write by
    parsing it.

    ``refusal_reason`` is the :class:`~osprey_connectors.refusals.RefusalReason`
    that says why, and who, refused; it is set **if and only if** the outcome
    is ``refused``. Construction enforces it too: a refused result with no
    reason or an unknown one, or a reason on any other outcome, raises
    ``ValueError``. ``verdict_reason`` narrows a ``WRITES_DISABLED`` refusal
    to the reason the write verdict gave, a
    :class:`~osprey_connectors.refusals.VerdictReason` — ``switch_off``,
    ``auth_exhausted``, ``auth_range`` and the rest — so a program can tell
    them apart without parsing ``error_message``; it is ``None`` on every
    other result, and construction refuses it on any other.

    ``observed`` is the confirming re-read itself, the whole
    :class:`ChannelValue`: its value in the type the channel holds (a number,
    a string, a waveform), its stamp, its :class:`Alarm` and its enum label.
    It is set only on a ``confirmed`` or ``mismatch`` result, the two outcomes
    a re-read decided; construction enforces it. :attr:`observed_number`
    narrows its value to a float where that makes sense. The alarm state
    (``observed.alarm``) is information reported with the write, never a
    reason to raise.

    ``limits_violation`` is the
    :class:`~osprey_connectors.errors.LimitsViolation` of a refusal whose
    reason is ``LIMITS``, and is set **if and only if** that is the reason;
    construction enforces it.

    The base class's write pipeline builds every result through the named
    constructors (:meth:`refused`, :meth:`failed`, :meth:`unconfirmed`,
    :meth:`confirmed`, :meth:`mismatch`, :meth:`unrequested`), one per
    outcome, each taking only what its outcome carries.
    """

    address: str  # Channel that was written
    value_written: ContractAny  # Value that was sent
    outcome: WriteOutcome  # The one verdict, set by the connector
    # Set iff refused, enforced at construction
    refusal_reason: RefusalReason | None = None
    # Set iff refused, failed or unconfirmed, enforced at construction
    error_message: str | None = None
    # The confirming re-read, set only on a confirmed or mismatch result
    observed: ChannelValue | None = None
    notes: str | None = None  # Display-only detail
    # The write verdict's reason, for a WRITES_DISABLED refusal that has one
    verdict_reason: VerdictReasonField | None = None
    # What the limits refused, set iff the refusal reason is LIMITS
    limits_violation: LimitsViolation | None = None

    @field_validator("outcome", mode="before")
    @classmethod
    def _outcome_member(cls, value: object) -> WriteOutcome:
        """``outcome`` as a :class:`WriteOutcome` member, from the member or its word.

        A ``StrEnum`` crosses the connector-host IPC as the plain string JSON
        encodes it to, so what the parent side rebuilds a result from is a
        ``str``. Coercing once, on construction, is what lets every consumer
        compare against the members whichever side of the process boundary it
        runs on.

        Raises:
            ValueError: ``value`` is neither a member nor the word of one.
        """
        if isinstance(value, WriteOutcome):
            return value
        if not isinstance(value, str):
            # The words the enum's own lookup raises for a value it does not hold.
            raise ValueError(f"{value!r} is not a valid {WriteOutcome.__qualname__}")
        return WriteOutcome(value)

    @field_validator("refusal_reason", mode="before")
    @classmethod
    def _refusal_reason_member(
        cls, value: object, info: ValidationInfo[None]
    ) -> RefusalReason | None:
        """``refusal_reason`` as a :class:`RefusalReason` member, from the member or its word.

        Raises:
            ValueError: ``value`` is not ``None`` and not a known reason.
        """
        if value is None or isinstance(value, RefusalReason):
            return value
        try:
            if not isinstance(value, str):
                raise ValueError(value)
            return RefusalReason(value)
        except ValueError:
            raise ValueError(
                f"Write result for '{info.data.get('address')}' "
                f"(outcome {info.data.get('outcome')}): "
                f"refusal_reason {value!r} is not a known refusal reason"
            ) from None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Check that the reason and the message are set exactly where the outcome says.

        Raises:
            ValueError: ``refusal_reason`` is missing on a ``refused`` result or
                set on any other, ``error_message`` is missing on a
                ``refused``, ``failed`` or ``unconfirmed`` result or set on any
                other, ``verdict_reason`` is set on a result whose refusal
                reason is not ``WRITES_DISABLED``, or ``observed`` is set on a
                result that is neither ``confirmed`` nor ``mismatch``.
        """
        refused = self.outcome is WriteOutcome.REFUSED
        if refused != (self.refusal_reason is not None):
            expected = "a known refusal reason" if refused else "None"
            raise ValueError(
                f"Write result for '{self.address}' (outcome {self.outcome}): "
                f"refusal_reason must be {expected}, got {self.refusal_reason!r}"
            )
        if bool(self.error_message) != (self.outcome in _MESSAGE_OUTCOMES):
            expected = "a message" if self.outcome in _MESSAGE_OUTCOMES else "None"
            raise ValueError(
                f"Write result for '{self.address}' (outcome {self.outcome}): "
                f"error_message must be {expected}, got {self.error_message!r}"
            )
        if (
            self.verdict_reason is not None
            and self.refusal_reason is not RefusalReason.WRITES_DISABLED
        ):
            raise ValueError(
                f"Write result for '{self.address}' (refusal_reason "
                f"{self.refusal_reason}): verdict_reason must be None, "
                f"got {self.verdict_reason!r}"
            )
        limits = self.refusal_reason is RefusalReason.LIMITS
        if limits != (self.limits_violation is not None):
            expected = "a LimitsViolation" if limits else "None"
            raise ValueError(
                f"Write result for '{self.address}' (refusal_reason "
                f"{self.refusal_reason}): limits_violation must be {expected}, "
                f"got {self.limits_violation!r}"
            )
        if self.observed is not None and self.outcome not in _OBSERVED_OUTCOMES:
            raise ValueError(
                f"Write result for '{self.address}' (outcome {self.outcome}): "
                f"observed must be None, got {self.observed!r}"
            )
        return self

    @property
    def observed_number(self) -> float | None:
        """The value :attr:`observed` holds as a float; ``None`` when not a number or not read."""
        return None if self.observed is None else as_number(self.observed.value)

    @classmethod
    def refused(
        cls,
        address: str,
        value: ContractAny,
        reason: RefusalReason,
        message: str,
        *,
        verdict_reason: VerdictReason | None = None,
        limits: LimitsViolation | None = None,
    ) -> Self:
        """Nothing was written: *reason* says why, *message* tells the operator."""
        return cls(
            address=address,
            value_written=value,
            outcome=WriteOutcome.REFUSED,
            refusal_reason=reason,
            error_message=message,
            verdict_reason=verdict_reason,
            limits_violation=limits,
        )

    @classmethod
    def failed(
        cls, address: str, value: ContractAny, message: str, *, notes: str | None = None
    ) -> Self:
        """The value was sent, or its send attempted, and the control system did not take it."""
        return cls(
            address=address,
            value_written=value,
            outcome=WriteOutcome.FAILED,
            error_message=message,
            notes=notes,
        )

    @classmethod
    def unconfirmed(
        cls, address: str, value: ContractAny, message: str, *, notes: str | None = None
    ) -> Self:
        """The value was sent and what the channel holds is unknown."""
        return cls(
            address=address,
            value_written=value,
            outcome=WriteOutcome.UNCONFIRMED,
            error_message=message,
            notes=notes,
        )

    @classmethod
    def confirmed(cls, address: str, value: ContractAny, observed: ChannelValue) -> Self:
        """The confirming read *observed* holds the value sent.

        The reading, with its alarm state, is reported on the result.
        """
        return cls(
            address=address, value_written=value, outcome=WriteOutcome.CONFIRMED, observed=observed
        )

    @classmethod
    def mismatch(cls, address: str, value: ContractAny, observed: ChannelValue) -> Self:
        """The confirming read *observed* holds a value other than the one sent.

        No message: both values are on the result, and the raising path
        composes "sent X, channel holds Y" from them.
        """
        return cls(
            address=address,
            value_written=value,
            outcome=WriteOutcome.MISMATCH,
            observed=observed,
            notes=f"Channel holds {observed.value}, sent {value}",
        )

    @classmethod
    def unrequested(cls, address: str, value: ContractAny, *, notes: str | None = None) -> Self:
        """The value was sent and no confirmation was asked for: nothing was checked."""
        return cls(
            address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED, notes=notes
        )


#: Package path of the OSPREY MCP servers, as consecutive parts of the file
#: they are launched with (``python -m osprey.mcp_server.<server>`` sets
#: ``sys.argv[0]`` to that package's ``__main__.py``).
_MCP_SERVER_PACKAGE_PARTS = ("osprey", "mcp_server")

#: Which story a readonly refusal on a connector tells (see
#: :func:`_readonly_is_deployment_wide`). ``script``: one run the executor was
#: handed readonly, which can be resubmitted. ``deployment``: a long-lived
#: control session the deployment started, which runs no script to resubmit.
ReadonlyWording: TypeAlias = Literal["script", "deployment"]


#: The dotted key of every control-system connector block, ahead of its type.
_CONNECTOR_BLOCK_PREFIX = "control_system.connector."


def connector_block_key(connector_type: str | None) -> str:
    """The dotted key of *connector_type*'s block; ``<type>`` stands in for no type."""
    return f"{_CONNECTOR_BLOCK_PREFIX}{connector_type or '<type>'}"


class ControlSystemSettings(ConnectorSettings):
    """A control-system connector's own block, ``control_system.connector.<type>``, read.

    The settings every connector type's
    :attr:`ControlSystemConnector.settings_model` extends: it reads
    ``timeout_s``, the call bound every connector has, and keeps every other key
    as written, so an out-of-tree connector that declares no model of its own
    reads its keys from :attr:`~pydantic.BaseModel.model_extra`. Keys the
    package reads elsewhere (``limits_checking``, ``serving``) stay there too.

    Read with :meth:`from_block` under :func:`connector_block_key`, which names
    the type in a refusal.
    """

    #: The seconds one call is given when it names no bound of its own
    #: (:func:`~osprey_connectors.call_timeout.call_timeout_s`'s rule; the
    #: retired ``timeout`` key is refused).
    timeout_s: float = DEFAULT_TIMEOUT_S

    @model_validator(mode="before")
    @classmethod
    def _bounded(cls, data: object, info: ValidationInfo[object]) -> object:
        if not isinstance(data, Mapping):
            return data
        return {**data, TIMEOUT_KEY: call_timeout_s(data, _settings_type(info))}


def _settings_type(info: ValidationInfo[object]) -> str | None:
    """The connector type a settings validation's block key names, for a refusal's key."""
    key = settings_key(info)
    if key is None or not key.startswith(_CONNECTOR_BLOCK_PREFIX):
        return None
    connector_type = key.removeprefix(_CONNECTOR_BLOCK_PREFIX)
    return None if connector_type == "<type>" else connector_type


_Settings = TypeVar("_Settings", bound=ControlSystemSettings)


def _settings_as(settings: ControlSystemSettings, model: type[_Settings]) -> _Settings:
    """*settings* as the model its connector declares, which ``connect()`` made it.

    Raises:
        TypeError: *settings* is another model: ``_open`` was called by hand.
    """
    if not isinstance(settings, model):
        raise TypeError(f"{model.__name__} expected, got {type(settings).__name__}")
    return settings


class ConnectorContext(FrozenModel):
    """What a control-system connector is built for: the one argument of ``connect()``.

    :meth:`~osprey_connectors.factory.ConnectorFactory.build_control_system_connector`
    builds it from a ``control_system`` section; a caller that builds a
    connector by hand writes one. It holds no limits validator and none of the
    section the reference monitor reads: every write is judged by the
    deployment's loaded config, and :meth:`ControlSystemConnector.connect`
    builds the validator from that config too, whatever a caller hands it.
    """

    #: The type the connector was built as: a registry key or a dotted path.
    #: It selects the connector block (``control_system.connector.<type>``) the
    #: write and limits postures are read from; ``None`` reads the
    #: deployment-wide ones.
    connector_type: str | None = None
    #: The control target the connector serves, as the caller named it. Wide on
    #: purpose: a target that does not resolve is carried as named, so the
    #: write verdict refuses it ``target_unknown``. ``None``: no target.
    control_target: str | None = None
    #: The wire the connector speaks, from the type and the serving leaf.
    transport: Transport | None = None
    #: The connector's own block, ``control_system.connector.<type>``, read: the
    #: factory reads it into the type's
    #: :attr:`ControlSystemConnector.settings_model`. A mapping given here is read
    #: into the base model, and ``connect()`` reads it into the connector's own
    #: before :meth:`ControlSystemConnector._open` is handed it.
    settings: ControlSystemSettings = Field(default_factory=ControlSystemSettings)
    #: How a readonly refusal on this connector is worded. Only the remedy
    #: sentence changes; the refusal is the same either way.
    readonly_wording: ReadonlyWording = "script"

    @model_validator(mode="before")
    @classmethod
    def _settings_of_its_type(cls, data: object) -> object:
        """Settings given as a mapping are read for the context's own type, named in a refusal."""
        if not isinstance(data, Mapping) or not isinstance(data.get("settings"), Mapping):
            return data
        connector_type = data.get("connector_type")
        settings = ControlSystemSettings.from_block(
            data["settings"],
            key=connector_block_key(connector_type if isinstance(connector_type, str) else None),
        )
        return {**data, "settings": settings}


def _readonly_is_deployment_wide(wording: ReadonlyWording) -> bool:
    """Whether a readonly refusal here is the deployment's rather than one script's.

    True in an OSPREY MCP server (:func:`_in_mcp_server_process`) and on a
    connector whose context says ``deployment`` (a long-lived control session
    builds its connector so); everything else keeps the script wording.
    """
    return wording == "deployment" or _in_mcp_server_process()


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
    all, and is read by the write switch further down this function's
    caller.
    """
    argv: object = getattr(sys, "argv", None)
    if not isinstance(argv, Sequence) or not argv:
        return False
    launched = argv[0]
    if not isinstance(launched, str | PathLike):
        return False
    try:
        parts = Path(launched).parts
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
#: deployment or a connector of a long-lived control session
#: (:func:`_readonly_is_deployment_wide`): that run alone can be resubmitted.
_READONLY_SCRIPT_REMEDY = (
    "This script was submitted readonly; resubmit it with execution_mode='readwrite' "
    "(human approval required) if the write is intended."
)


def _monitor_section() -> ControlSystemSection:
    """The ``control_system`` section the reference monitor judges by.

    Empty for a read-only run, which is answered before any config is read: it
    refuses whatever the config says. A config that cannot be read arms
    nothing — the section is then empty, which the tier answers ``never``.
    """
    if is_readonly_run():
        return ControlSystemSection()
    try:
        from osprey_connectors.config import current_config

        return current_config().control_system
    except (FileNotFoundError, RuntimeError):
        return ControlSystemSection()


def _named_authorization() -> str | None:
    """The program's write authorization this process names, or ``None`` for none.

    :data:`~osprey_connectors.write_authorization.AUTHORIZATION_ENV_VAR`, the
    one place it is read: the write guard below and the connector child's
    report ask here, and every verdict is handed the answer. Blank is unset.
    A value that is not a well-formed id is returned as it is, so the verdict
    refuses it rather than treating the process as naming none.
    """
    value = (os.environ.get(write_authorization.AUTHORIZATION_ENV_VAR) or "").strip()
    return value or None


def _monitor_verdict(control_target: str | None) -> "WriteVerdict":
    """The verdict the reference monitor decides one write (or batch) by.

    :func:`osprey_connectors.write_switch.write_verdict` over this deployment's
    ``control_system`` section for *control_target*, the owner taken from the
    ladder. An unstamped connector (*control_target* ``None``) resolves no
    machine and is refused ``target_unknown`` unless an earlier term refuses
    first. A read-only run is answered before any config is read: it refuses
    whatever the config says. A config that cannot be read arms nothing — the
    section is then empty, which the tier answers ``never``.

    A process that names a program's write authorization
    (:func:`_named_authorization`) is answered by
    :func:`~osprey_connectors.write_authorization.authorized_write_verdict`
    instead: whether the authorization would let it write at all, channels
    aside. That is the answer the gateway is chosen on; each write is judged
    on its own channels and values by the write guard.

    Never raises; reads the record at most once.
    """
    section = _monitor_section()
    authorization = _named_authorization()
    if authorization is None:
        return write_switch.write_verdict(section, control_target, owner=None)
    return write_authorization.authorized_write_verdict(
        section, control_target, authorization=authorization, owner=None
    )


def _refusing_verdict(connector: "ControlSystemConnector") -> "WriteVerdict":
    """The verdict behind a ``_writes_enabled`` that has just answered no.

    The one that evaluation memoised on ``_last_verdict``, or a fresh
    :func:`_monitor_verdict` for a subclass that overrides the property and so
    never sets it. Read here, not where the guard cleared the memo: the property
    sets it as a side effect the caller's own narrowing cannot see.
    """
    return connector._last_verdict or _monitor_verdict(connector._control_target)


def _verdict_refusal(
    address: str, value: object, verdict: "WriteVerdict", *, wording: ReadonlyWording
) -> ChannelWriteResult:
    """The REFUSED result the monitor returns for a write *verdict* refused.

    The operator's words are :func:`~osprey_connectors.refusals.refusal_text`
    for the verdict's reason and machine kind — the same phrase the hooks
    speak, ending with the ``[osprey-refusal: <reason>]`` trailer — after the
    channel it concerns, with the verdict's own detail after the phrase. A
    read-only run's refusal also carries :data:`_READONLY_MARKER`, and in a script
    the executor ran for one readonly submission :data:`_READONLY_SCRIPT_REMEDY`
    as well; *wording* is the connector's :attr:`ConnectorContext.readonly_wording`.
    A verdict that allowed (an overriding subclass refused on its own) names no
    contract reason.
    """
    from osprey_connectors.refusals import refusal_text

    marker = ""
    remedy = None
    if verdict.reason is VerdictReason.READONLY_RUN:
        marker = f" ({_READONLY_MARKER})"
        if not _readonly_is_deployment_wide(wording):
            # One script submitted readonly, not the deployment: the same
            # refusal, and the one remedy that applies to it alone.
            remedy = _READONLY_SCRIPT_REMEDY
    if isinstance(verdict, Allowed):
        # The verdict allowed: the connector's own write check refused, and it
        # claims no reason of the contract.
        spoken = _OVERRIDE_REFUSAL
        verdict_reason = None
    else:
        spoken = refusal_text(verdict.reason, verdict.kind, detail=remedy or verdict.detail)
        verdict_reason = verdict.reason
    return ChannelWriteResult.refused(
        address,
        value,
        RefusalReason.WRITES_DISABLED,
        f"Write to '{address}' blocked{marker}: {spoken}",
        verdict_reason=verdict_reason,
    )


def _authorized_refusal(
    connector: "ControlSystemConnector",
    operations: Sequence[tuple[str, object]],
    authorization: str,
) -> "WriteVerdict | None":
    """The verdict refusing one write call of a process that names an authorization.

    ``None`` means the call may go ahead and its step is spent. Otherwise the
    refusing verdict, which :func:`_verdict_refusal` words; an allowed verdict
    returned here is a subclass's own ``_writes_enabled`` answering no.

    The order: a subclass that overrides ``_writes_enabled`` and answers no
    refuses as it does for every process. A connector that cannot carry a
    write over the gateway it selected (:meth:`ControlSystemConnector.
    _write_route_open`) refuses ``switching`` when the authorization would
    allow the call, its own reason when not, and spends nothing either way.
    Otherwise :func:`~osprey_connectors.write_authorization.spend_authorized_write`
    judges the whole call and, when it allows, spends its one step before
    anything is put: a write the limits then refuse has still spent it.
    """
    connector._last_verdict = None
    if _overrides_writes_enabled(connector) and not connector._writes_enabled:
        return _refusing_verdict(connector)
    section = _monitor_section()
    target = connector._control_target
    if not connector._write_route_open():
        judged = write_authorization.authorized_execution_verdict(
            section, target, operations, authorization=authorization
        )
        connector._last_verdict = judged
        if not judged.allowed:
            return judged
        return Refused(
            reason=VerdictReason.SWITCHING,
            target=judged.target,
            kind=judged.kind,
            authorization=authorization,
        )
    verdict = write_authorization.spend_authorized_write(
        section, target, operations, authorization=authorization
    )
    connector._last_verdict = verdict
    return None if verdict.allowed else verdict


def _overrides_writes_enabled(connector: object) -> bool:
    """Whether *connector*'s class answers ``_writes_enabled`` with its own rule."""
    return getattr(type(connector), "_writes_enabled", None) is not _BASE_WRITES_ENABLED


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
            written. Its ``reason`` is the result's ``refusal_reason``, whose
            ``refuser`` says who refused, and its ``violation`` the result's
            ``limits_violation``.
        ChannelWriteFailedError: The value was sent and the write did not come
            back confirmed — ``failed``, ``mismatch`` or ``unconfirmed``, each
            raising under its own word. The exception carries the outcome, the
            value sent and the value observed, so a ``mismatch`` message names
            both numbers.
    """
    from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

    if result.outcome is WriteOutcome.REFUSED and result.refusal_reason is not None:
        raise ChannelWriteBlockedError(
            result.address,
            result.refusal_reason,
            message=result.error_message,
            verdict_reason=result.verdict_reason,
            violation=result.limits_violation,
        )
    if result.outcome in _PROCEEDABLE_OUTCOMES:
        return result
    # Everything else is a write the caller must not proceed on, raised under
    # its own outcome, so the outcome a caller branches on and the one a
    # consumer reads stay one vocabulary by construction rather than by a table
    # that a new outcome could be left out of.
    raise ChannelWriteFailedError(
        result.address,
        result.outcome,
        message=result.error_message,
        value_written=result.value_written,
        observed_value=None if result.observed is None else result.observed.value,
    )


def raise_for_write_results(results: Sequence[ChannelWriteResult]) -> Sequence[ChannelWriteResult]:
    """Enforce the denial contract on a batch's results.

    :func:`raise_for_write_result` for each result in batch order, the writes
    their batch only held back (``BATCH_REFUSED``) last: a batch stopped at its
    checks raises for the write that was refused, wherever it stands.

    Returns:
        The results unchanged, when every one may be proceeded on.
    """
    held_back = [r for r in results if r.refusal_reason == RefusalReason.BATCH_REFUSED]
    for result in (*(r for r in results if r not in held_back), *held_back):
        raise_for_write_result(result)
    return results


def _limits_refusal(
    address: str, value: object, error: "ChannelLimitsViolationError"
) -> ChannelWriteResult:
    """The REFUSED result for a write the limits refused, single or batched.

    One line, the violation's type and reason; the violation itself rides on
    the result as its ``limits_violation``.
    """
    kind = error.violation_type
    reason = error.violation_reason
    said = f"({kind}): {reason}" if reason else f": {error}"
    return ChannelWriteResult.refused(
        address,
        value,
        RefusalReason.LIMITS,
        f"Write to '{address}' refused by the limits {said}",
        limits=LimitsViolation.from_error(error),
    )


def _refused_batch(
    named: Sequence[tuple[str, object]], refusals: Sequence[ChannelWriteResult | None]
) -> list[ChannelWriteResult]:
    """One result per write of a batch that was not sent because a write was refused."""
    refused = [refusal for refusal in refusals if refusal is not None]
    first = refused[0]
    others = f" and {len(refused) - 1} more" if len(refused) > 1 else ""
    why = (
        f"the write to '{first.address}' was refused "
        f"({first.refusal_reason}){others}. No write of the batch was sent."
    )
    return [
        refusal
        if refusal is not None
        else ChannelWriteResult.refused(
            name, value, RefusalReason.BATCH_REFUSED, f"Write to '{name}' not sent: {why}"
        )
        for (name, value), refusal in zip(named, refusals, strict=True)
    ]


def _repeated_refusal(address: str, value: object, times: int) -> ChannelWriteResult:
    """The refusal of one write to a channel its batch names more than once."""
    return ChannelWriteResult.refused(
        address,
        value,
        RefusalReason.VALIDATION_ERROR,
        f"Write to '{address}' refused: the batch names this channel more than "
        f"once ({times} writes). A batch writes a channel once; send the later value "
        "in a batch of its own.",
    )


def _batch_send_verdict(address: str, value: object, error: Exception) -> ChannelWriteResult:
    """The result of one write of a batch whose send raised, or the raise itself.

    A limits refusal raised at send time is a ``LIMITS`` refusal. A channel
    that could not be reached or did not answer in time is ``unconfirmed``: the
    others of the batch are not recalled, and what this one holds is unknown.
    Any other error is the connector's own and is raised.
    """
    from osprey_connectors.errors import ChannelLimitsViolationError, ChannelUnreachableError

    if isinstance(error, ChannelLimitsViolationError):
        return _limits_refusal(address, value, error)
    if isinstance(error, ChannelUnreachableError | TimeoutError):
        return ChannelWriteResult.unconfirmed(
            address,
            value,
            f"Write to '{address}' could not be confirmed, its outcome is unknown: {error}",
        )
    raise error


#: Why a send did not deliver its value, as :class:`SendReport` says it. Each
#: word is one outcome of the write:
#:
#: * ``not_sent`` — nothing left this process: ``failed``.
#: * ``not_taken`` — the control system answered the send and did not take the
#:   value: ``failed``.
#: * ``unacknowledged`` — the value went out and its acknowledgement did not
#:   come in time, so it may still arrive: ``unconfirmed``.
#: * ``cs_refused`` — the control system denied the write (access security):
#:   ``refused``, ``CONTROL_SYSTEM_REFUSED``.
#: * ``unwritable`` — the channel cannot hold the value as given, found before
#:   anything was sent: ``refused``, ``VALIDATION_ERROR``.
SendFault: TypeAlias = Literal[
    "not_sent", "not_taken", "unacknowledged", "cs_refused", "unwritable"
]


class SendReport(FrozenModel):
    """What a connector's :meth:`ControlSystemConnector._send` says about one send.

    The connector reports and the base class's write pipeline decides: every
    :class:`ChannelWriteResult` of a write is built there, from this report and
    the confirming read, so every connector words the same situation the same
    way.

    ``fault`` is ``None`` when the value went out (and, when the send was asked
    to wait, was acknowledged), else the :data:`SendFault` that says how it did
    not. ``detail`` is the transport's own words for the fault, quoted in the
    result's message. ``sent`` is the value as the wire took it, when the
    connector changed its form on the way out (text read into a number for a
    numeric field, say); the confirming read is compared with it, and with the
    value written when it is ``None``.
    """

    fault: SendFault | None = None
    detail: str = ""
    sent: ContractAny = None


def _fault_result(address: str, value: object, report: SendReport) -> ChannelWriteResult:
    """The result of a send that reported a fault, worded the same for every connector."""
    detail = report.detail
    if report.fault == "cs_refused":
        logger.warning(f"Control system refused write to {address}: {detail}")
        return ChannelWriteResult.refused(
            address,
            value,
            RefusalReason.CONTROL_SYSTEM_REFUSED,
            f"Write to '{address}' refused by the control system; no value was written: {detail}",
        )
    if report.fault == "unwritable":
        return ChannelWriteResult.refused(
            address,
            value,
            RefusalReason.VALIDATION_ERROR,
            f"Write to '{address}' refused: {detail}. No value was sent.",
        )
    if report.fault == "unacknowledged":
        logger.warning(f"Write to {address} not acknowledged: {detail}")
        return ChannelWriteResult.unconfirmed(
            address,
            value,
            f"Write to '{address}' could not be confirmed — {detail}",
            notes="The value may still arrive; what the channel holds is unknown",
        )
    if report.fault == "not_sent":
        return ChannelWriteResult.failed(
            address, value, f"Failed to write to '{address}': no value was sent ({detail})"
        )
    logger.warning(f"Control system did not take the write to {address}: {detail}")
    return ChannelWriteResult.failed(address, value, f"Write to '{address}' failed: {detail}")


class ControlSystemConnector(ABC):
    """
    Abstract base class for control system connectors.

    Implementations provide interfaces to different control systems
    (EPICS, LabVIEW, Tango, the simulator, etc.) using a unified API.

    :meth:`connect` and the write methods — :meth:`write_channel`,
    :meth:`write_multiple_channels` and :meth:`write_channel_checked` — are
    final: they are the reference monitor, and a subclass that defines one is
    refused when its class is created. A connector opens its connection by
    implementing :meth:`_open` and writes by implementing :meth:`_send`, and
    adjusts the pipeline through the protected hooks, each optional:

    - :meth:`_read_many` — how a batch read reads its channels (default: each
      with :meth:`read_channel`, concurrently);
    - :meth:`_precheck` — its own refusals that need no I/O, and the one name
      a write uses for its channel;
    - :meth:`_confirming_read` — the read a confirmed write is compared with
      (default: :meth:`read_channel`);
    - :meth:`_run_check` — where the limits check runs (default: a worker
      thread, bounded by the write's timeout);
    - :meth:`_bound` — the timeout a write's phases get when the caller gives
      none;
    - :meth:`_current_values` — what a batch's ``max_step`` checks measure
      from, read together;
    - :meth:`_send_batch` — how a checked batch is sent (default: one write
      after the other);
    - :meth:`current_value_reader` — the synchronous read a ``max_step`` check
      measures through;
    - :meth:`_write_route_open` — whether the gateway selected at connect can
      carry a write;
    - ``_writes_enabled`` — a connector's own write rule, on top of the
      verdict.

    Example:
        >>> connector = await ConnectorFactory.create_control_system_connector()
        >>> try:
        >>>     channel_value = await connector.read_channel('BEAM:CURRENT')
        >>>     print(f"Beam current: {channel_value.value} {channel_value.metadata.units}")
        >>> finally:
        >>>     await connector.disconnect()
    """

    #: The model this connector's block is read into: the factory reads
    #: ``control_system.connector.<type>`` into it, and :meth:`connect` reads a
    #: context's settings into it when they are another model. A connector type
    #: that reads keys of its own declares a subclass of
    #: :class:`ControlSystemSettings`; one that declares none reads them from
    #: :attr:`~pydantic.BaseModel.model_extra`.
    settings_model: ClassVar[type[ControlSystemSettings]] = ControlSystemSettings

    #: What this connector was built for, set by :meth:`connect`. A connector
    #: nobody connected holds the empty context: no type, no target, no wire.
    _context: ConnectorContext = ConnectorContext()
    #: The limits validator :meth:`connect` built from the deployment's loaded
    #: config for this connector's type; ``None`` while limits checking is off
    #: for that type, or before ``connect()``.
    _limits_validator: "LimitsValidator | None" = None

    @property
    def context(self) -> ConnectorContext:
        """What this connector was built for: the context :meth:`connect` was handed."""
        return self._context

    @property
    def transport(self) -> Transport | None:
        """The wire this connector speaks (``None`` when its context names none)."""
        return self._context.transport

    @property
    def _connector_type(self) -> str | None:
        """The type this connector was built as; it keys the per-type write and limits postures."""
        return self._context.connector_type

    @property
    def _control_target(self) -> str | None:
        """The control target this connector serves; it indexes the session posture."""
        return self._context.control_target

    @property
    def _call_timeout_s(self) -> float:
        """The call bound this connector's block declares.

        Read by :func:`~osprey_connectors.call_timeout.call_timeout_s` from the
        context's settings, naming the context's type in a refusal.

        Raises:
            ConfigurationError: The block still carries ``timeout``, or its
                ``timeout_s`` is not a positive, finite number.
        """
        return self._context.settings.timeout_s

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

        Decided by :func:`osprey_connectors.write_switch.write_verdict` for
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

    def current_value_reader(self) -> "CurrentValueReader | None":
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
        self,
        address: str,
        # The value write_channel was given: the extension contract leaves it open.
        value: ContractAny,
        error: BaseException,
    ) -> ChannelWriteResult:
        """The REFUSED result a write owes the caller when its check could not be made.

        Limits validation answers a write in one of two ways: a
        :class:`ChannelLimitsViolationError`, which is a ``LIMITS`` refusal, or
        any other error — which says the check could not be MADE. A check that
        could not be made is not permission to write: a validator that never
        answered has not said yes. So that second case is this one REFUSED
        result, and nothing is sent. A connector's :meth:`_precheck` answers
        an address it cannot parse with it too.
        """
        logger.warning(f"Validation error - refusing write (fail-closed): {address}: {error}")
        return ChannelWriteResult.refused(
            address,
            value,
            RefusalReason.VALIDATION_ERROR,
            f"Write to '{address}' refused: validation error: {error}",
        )

    @override
    def __init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass that defines one of the final methods.

        :meth:`connect`, :meth:`write_channel`, :meth:`write_multiple_channels`
        and :meth:`write_channel_checked` are the reference monitor: the limits
        validator built from the loaded config, the write verdict, the session
        hold, the door, the limits and the six outcomes live in them, once, for
        every connector. A connector opens its connection in :meth:`_open`, and
        changes how a write is sent, read back or checked through the protected
        hooks the class docstring lists, never by replacing one of them.

        Raises:
            TypeError: The class body defines ``connect``, ``write_channel``,
                ``write_multiple_channels`` or ``write_channel_checked``.
        """
        super().__init_subclass__(**kwargs)
        defined = sorted(_FINAL_METHODS & set(cls.__dict__))
        if defined:
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__} defines {', '.join(defined)}: "
                "these methods of ControlSystemConnector are final. Implement _open in "
                "place of connect, and _send (and, where needed, _precheck, "
                "_confirming_read, _run_check, _current_values or _send_batch) in place "
                "of a write method."
            )

    def _write_route_open(self) -> bool:
        """Whether the route this connector selected at ``connect()`` can carry a write.

        ``True`` here; a connector that selects a gateway by the verdict at
        connect time and refuses a write on the read gateway ``switching``
        (EPICS) answers for itself. The write guard asks it before an
        authorized write spends its step, so a write that could not go out
        costs nothing.
        """
        return True

    def _resolve_confirm(self, address: str) -> bool:
        """Whether a write to this channel must be confirmed by re-reading it.

        The limits database is the single home of write policy: the channel's
        own ``confirm``, else ``True``. With limits checking off there is no
        policy to read, so the write takes the fleet default and confirms.
        """
        return _confirm_policy(self._validator(), address)

    def _validator(self) -> "LimitsValidator | None":
        """The limits validator this connector's writes are checked against.

        The one :meth:`connect` built from the deployment's loaded config for
        this connector's type, whatever the connector's own :meth:`_open` loads
        or not. ``None`` when limits checking is off for that type, and before
        ``connect()``, when the connector names no machine and the write
        verdict refuses every write anyway.
        """
        return self._limits_validator

    def _bound(self, timeout_s: float | None) -> float:
        """The seconds a write's phases are given: *timeout_s*, else this connector's own.

        The default is the ``timeout_s`` this connector's block declares
        (:attr:`_call_timeout_s`). A connector that keeps its bound elsewhere
        answers it.
        """
        return self._call_timeout_s if timeout_s is None else timeout_s

    @final
    async def connect(self, context: ConnectorContext) -> None:
        """Connect this connector for *context*.

        Builds the limits validator from the deployment's loaded config for
        ``context.connector_type``
        (:meth:`~osprey_connectors.control_system.LimitsValidator.from_config`),
        stores it with the context, then opens the connection with
        :meth:`_open`. Final: a subclass implements :meth:`_open`, and defining
        ``connect`` raises ``TypeError`` when the class is created, so no
        connector can skip the validator.

        A validator that cannot be loaded raises here, before anything is
        stored: the connector keeps the context it had, so one never connected
        still names no machine and every write it is asked for is refused.

        Args:
            context: What the connector is built for, usually from
                :meth:`~osprey_connectors.factory.ConnectorFactory.build_control_system_connector`.

        Raises:
            ConnectorUnavailableError: If connection cannot be established
            ConnectRefusedError: If policy refuses the connection (a readonly
                run with no gateway it may use, for instance)
        """
        from osprey_connectors.control_system._limits_validator import LimitsValidator

        validator = LimitsValidator.from_config(connector_type=context.connector_type)
        settings = context.settings
        if not isinstance(settings, self.settings_model):
            settings = self.settings_model.read(
                settings, key=connector_block_key(context.connector_type)
            )
            context = evolve(context, settings=settings)
        self._context = context
        self._limits_validator = validator
        await self._open(settings)

    @abstractmethod
    async def _open(self, settings: ControlSystemSettings) -> None:
        """Establish the connection to the control system.

        Called by :meth:`connect` once the context is stored, so
        :attr:`context` and the write verdict already answer for this
        connector.

        Args:
            settings: The connector's own block, ``control_system.connector.<type>``,
                read into :attr:`settings_model`.

        Raises:
            ConnectorUnavailableError: If connection cannot be established
            ConnectRefusedError: If policy refuses the connection (a readonly
                run with no gateway it may use, for instance)
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to control system and cleanup resources."""

    @abstractmethod
    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> ChannelValue:
        """
        Read current value of a channel.

        Args:
            address: Address/name of the channel
            timeout_s: Optional timeout in seconds

        Returns:
            ChannelValue with current value, ``sampled_at``, and metadata

        Raises:
            ChannelUnreachableError: This channel cannot be reached or did not
                answer in time; the connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone
            ValueError: If channel address is invalid
            ChannelNotFoundError: The address is not a channel of the facility
                (a connector that holds the facility's address set)
        """

    @final
    async def write_channel(
        self,
        address: str,
        value: ContractAny,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to a channel, confirming it unless asked not to.

        Final: every connector writes through this one pipeline, in order —

        1. the write guard: the write verdict (or, in a process that names a
           program's write authorization, that authorization, which spends
           the call's one step here) and the session hold;
        2. the door (:func:`~osprey_connectors.control_system.write_door.open_door`);
        3. :meth:`_precheck`, the connector's own refusals that need no I/O;
        4. the confirmation policy, when ``confirm`` is omitted;
        5. the limits (:meth:`_run_check`, bounded by ``timeout_s``): a
           violation is a ``LIMITS`` refusal, and a check that raised anything
           else or outlived its bound a ``VALIDATION_ERROR`` refusal — either
           way nothing is sent;
        6. :meth:`_send`, whose :class:`SendReport` fault decides a refused,
           failed or unconfirmed write;
        7. ``unrequested`` when no confirmation was asked for;
        8. :meth:`_confirming_read`; a read that raises leaves the write
           ``unconfirmed``;
        9. :func:`values_match` of the value sent and the value read back:
           ``confirmed`` or ``mismatch``.

        Args:
            address: Address/name of the channel
            value: Value to write
            timeout_s: Seconds each phase is given — the limits check, the
                send and the confirming read — or ``None`` for the
                connector's own (:meth:`_bound`)
            confirm: Whether to re-read the channel and compare, or ``None`` to
                let the connector resolve the policy for this channel

        Returns:
            ChannelWriteResult carrying the outcome and what the channel was
            seen to hold

        Raises:
            ChannelUnreachableError: The send could not reach the channel and
                its outcome is unknown; the connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone

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
        authorization = _named_authorization()
        if authorization is not None:
            refused = _authorized_refusal(self, [(address, value)], authorization)
            if refused is not None:
                return _verdict_refusal(
                    address, value, refused, wording=self.context.readonly_wording
                )
        else:
            self._last_verdict = None
            if not self._writes_enabled:
                return _verdict_refusal(
                    address, value, _refusing_verdict(self), wording=self.context.readonly_wording
                )
            held = write_switch.session_hold_refusal()
            if held is not None:
                return _verdict_refusal(address, value, held, wording=self.context.readonly_wording)
        with open_door():
            return await self._write_one(address, value, timeout_s=timeout_s, confirm=confirm)

    @final
    async def write_channel_checked(
        self,
        address: str,
        value: ContractAny,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Await write_channel and enforce the reference monitor's denial contract.

        ``refused`` (policy/limits/validation) -> raises ChannelWriteBlockedError.
        ``failed`` / ``mismatch`` / ``unconfirmed`` -> raises ChannelWriteFailedError.
        Connection failures and timeouts propagate unchanged.
        ``confirmed`` or ``unrequested`` -> returns the ChannelWriteResult.

        A scan device setter wraps this so any raise aborts the RunEngine, while a
        confirmed write returns and the scan proceeds; ``osprey.runtime.write_channel``
        routes through it for the same reason. ``timeout_s`` and ``confirm`` pass
        straight through to write_channel, ``None`` as the omission it is there. The
        result inspection itself is :func:`raise_for_write_result`, which the
        multi-channel paths share; a limits refusal raises there like any other,
        with its :class:`~osprey_connectors.errors.LimitsViolation`.
        """
        return raise_for_write_result(
            await self.write_channel(address, value, timeout_s=timeout_s, confirm=confirm)
        )

    async def read_multiple_channels(
        self, addresses: Sequence[str], /, *, timeout_s: float | None = None
    ) -> "ChannelReadBatch":
        """
        Read several channels at once, each answering for itself.

        Every address asked for is in the batch: in ``values`` when it was read,
        in ``failures`` with the :class:`~osprey_connectors.errors.ChannelError`
        its own read raised when it was not (unreachable, not a channel, no
        answer in time). :meth:`ChannelReadBatch.raise_for_failures` turns any
        failure into one :class:`~osprey_connectors.errors.ChannelReadFailedError`.

        An error that is not about one channel is raised, not reported: a
        connector whose transport is gone (``ConnectorUnavailableError``) or
        that failed itself fails the whole call, so the caller replaces it
        rather than reading an empty batch. A repeated address is read once.

        This method is the batch read on every connector. A connector that can
        read many channels in one request implements :meth:`_read_many`; the
        default reads each channel with :meth:`read_channel`, concurrently.

        Args:
            addresses: The channels to read
            timeout_s: Optional timeout in seconds, given to each read

        Returns:
            The batch: each channel's value or its failure
        """
        return await self._read_many(list(dict.fromkeys(addresses)), timeout_s=timeout_s)

    async def _read_many(
        self, addresses: Sequence[str], *, timeout_s: float | None
    ) -> "ChannelReadBatch":
        """Read every one of *addresses*, distinct, for :meth:`read_multiple_channels`.

        An optional hook. Each address goes into exactly one of the batch's
        ``values`` and ``failures``, and only a
        :class:`~osprey_connectors.errors.ChannelError` is a failure: any
        other exception is raised. This default gathers :meth:`read_channel`
        over every address and raises the first non-channel error in address
        order once every read has finished; a cancellation is re-raised as it is.
        """
        results = await asyncio.gather(
            *(self.read_channel(address, timeout_s=timeout_s) for address in addresses),
            return_exceptions=True,
        )
        values: dict[str, ChannelValue] = {}
        failures: dict[str, ChannelError] = {}
        for address, result in zip(addresses, results, strict=True):
            if isinstance(result, ChannelError):
                failures[address] = result
            elif isinstance(result, BaseException):
                raise result
            else:
                values[address] = result
        return ChannelReadBatch(values=values, failures=failures)

    @final
    async def write_multiple_channels(
        self,
        # Each value as write_channel takes it: the extension contract leaves it open.
        operations: Sequence[tuple[str, ContractAny]],
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> list[ChannelWriteResult]:
        """
        Write several channels as one batch: every write checked, then every write sent.

        What a batch means, on every connector:

        1. **The checks are all-or-nothing.** Every write passes every check a
           single :meth:`write_channel` makes before it sends: the write
           verdict, the connector's own refusals and the limits, ``max_step``
           included. The channels a step
           is measured on are read together, once, before anything is sent. If
           one write is refused there, **no write of the batch is sent**, and every
           channel gets a ``refused`` result: a refused write its own reason,
           each other write ``BATCH_REFUSED`` naming a refused one.
        2. **Past the checks every write is sent and verified on its own.** One
           result per channel, with the outcome a single write would report. A
           write that fails, mismatches or stays unconfirmed does not recall
           the others, which may already hold their values. The batch has
           failed unless every result may be proceeded on
           (:func:`raise_for_write_result`).

        The promise covers what is known before the send. A refusal the control
        system gives only at the send (a connector with no
        :meth:`_precheck`, a value the target coerces or a read-only
        channel it rejects) can come after earlier writes of the batch went
        out; each result still says what happened to its own channel.

        A batch names a channel once. One that names a channel more than once
        is refused whole: a later write to it could only be checked against an
        earlier one that may never land.

        The write verdict is asked again right before the send, so a switch
        turned off, a hold or a stop that arrives while the batch is checked
        refuses every write not yet sent.

        A limits refusal is a ``refused`` result with reason ``LIMITS``, as on a
        single write. A channel whose send raised ``ChannelUnreachableError`` or
        timed out is ``unconfirmed``. An error of the connector itself
        (``ConnectorUnavailableError``) is raised, and the batch's outcome is
        then unknown.

        Final, like :meth:`write_channel`. A connector changes how a batch is
        checked or sent through :meth:`_precheck`, :meth:`_current_values`
        and :meth:`_send_batch`.

        Args:
            operations: List of (address, value) tuples
            timeout_s: Optional timeout in seconds, given to each write
            confirm: Whether every channel in the batch is confirmed, or ``None``
                to let each channel resolve its own policy

        Returns:
            List of ChannelWriteResult in the same order as operations

        Note:
            An omitted ``confirm`` is not resolved here: each channel resolves
            its own policy from the limits database, as :meth:`write_channel`
            documents. An explicit ``confirm``, ``False`` as much as ``True``,
            reaches every channel unchanged.

            In a process that names a program's write authorization the whole
            call is judged and its one step spent before anything is checked
            or sent; a batch the limits then refuse has still spent it.
        """
        batch: list[tuple[str, object]] = list(operations)
        authorization = _named_authorization()
        refused: WriteVerdict | None
        if authorization is not None:
            refused = _authorized_refusal(self, batch, authorization)
        else:
            # Every write is judged, as a single write is, before any is sent.
            # The first refusal answers for the batch: every result of a
            # refused batch tells the same story about why.
            refused = None
            for _ in batch:
                self._last_verdict = None
                if not self._writes_enabled:
                    refused = _refusing_verdict(self)
                else:
                    refused = write_switch.session_hold_refusal()
                if refused is not None:
                    break
        if refused is not None:
            return [
                _verdict_refusal(address, value, refused, wording=self.context.readonly_wording)
                for address, value in batch
            ]
        if not batch:
            return []
        with open_door():
            named, refusals = await self._check_batch(batch)
            if any(refusal is not None for refusal in refusals):
                return _refused_batch(named, refusals)
            return await self._send_batch(named, timeout_s, confirm)

    def _precheck(
        self,
        address: str,
        # The value write_channel was given: the extension contract leaves it open.
        value: ContractAny,  # noqa: ARG002 - a connector's refusal carries the value
    ) -> tuple[str, ChannelWriteResult | None]:
        """One write's name, and the refusal this connector answers before the limits.

        The checks a write makes without touching the control system, ahead of
        limits validation, on a single write and on every write of a batch. The
        name is the one the limits, the send and the result use for the
        channel. A refusal is built with :meth:`ChannelWriteResult.refused`
        (or :meth:`_validation_refusal` for an address that does not parse).
        """
        return address, None

    async def _current_values(self, addresses: list[str]) -> Mapping[str, object] | None:
        """What the named channels hold now, for a batch's ``max_step`` checks.

        ``None`` when the connector has no reader, and the step checks then
        fail closed. A channel that could not be read maps to the exception
        its read raised, or to ``None``. The default reads through
        :meth:`current_value_reader` in one worker thread for the whole batch.
        """
        reader = self.current_value_reader()
        if reader is None:
            return None

        def read_all() -> dict[str, object]:
            held: dict[str, object] = {}
            for address in addresses:
                try:
                    held[address] = reader(address)
                except Exception as exc:
                    held[address] = exc
            return held

        return await asyncio.to_thread(read_all)

    async def _check_batch(
        self, batch: Sequence[tuple[str, object]]
    ) -> tuple[list[tuple[str, object]], list[ChannelWriteResult | None]]:
        """Check every write of a batch; nothing is sent.

        Returns the writes under the names :meth:`_precheck` gave them,
        and one entry per write: its refusal, or ``None`` when it passed.
        """
        from osprey_connectors.errors import ChannelLimitsViolationError

        named: list[tuple[str, object]] = []
        refusals: list[ChannelWriteResult | None] = []
        for address, value in batch:
            name, refusal = self._precheck(address, value)
            named.append((name, value))
            refusals.append(refusal)

        times = Counter(name for name, _ in named)
        for index, (name, value) in enumerate(named):
            if times[name] > 1 and refusals[index] is None:
                refusals[index] = _repeated_refusal(name, value, times[name])

        try:
            validator = self._validator()
        except Exception as exc:
            # A validator hook that raised has not said yes.
            return named, [
                refusal if refusal is not None else self._validation_refusal(name, value, exc)
                for (name, value), refusal in zip(named, refusals, strict=True)
            ]
        if validator is None:
            return named, refusals

        def measured(name: str) -> bool:
            try:
                return bool(validator.measures_step(name))
            except Exception:
                return True  # cannot tell: read it

        stepped = [
            name
            for (name, _), refusal in zip(named, refusals, strict=True)
            if refusal is None and measured(name)
        ]
        held: Mapping[str, object] | None = await self._current_values(stepped) if stepped else {}

        def read_current(name: str) -> object:
            current = held.get(name) if held is not None else None
            if isinstance(current, BaseException):
                raise current
            return current

        reader = None if held is None else read_current
        for index, (name, value) in enumerate(named):
            if refusals[index] is not None:
                continue
            try:
                validator.validate(name, value, read_current=reader)
            except ChannelLimitsViolationError as exc:
                refusals[index] = _limits_refusal(name, value, exc)
            except Exception as exc:
                # The check could not be made, which is not permission to write.
                refusals[index] = self._validation_refusal(name, value, exc)
        return named, refusals

    def _send_refusal(self, batch: Sequence[tuple[str, object]]) -> "WriteVerdict | None":
        """The verdict refusing what a checked batch has not sent yet, or ``None``.

        The batch was judged before its checks, and the checks take time: a
        switch turned off, a hold or a stop since then must not be outrun by
        the send. This asks again what the write guard asks of a single write.
        In a process that names an authorization the call is judged again
        without spending; ``auth_exhausted`` alone is no refusal here, since
        the step this batch spent may have been the last.
        """
        authorization = _named_authorization()
        self._last_verdict = None
        if authorization is None:
            if not self._writes_enabled:
                return _refusing_verdict(self)
            return write_switch.session_hold_refusal()
        if _overrides_writes_enabled(self) and not self._writes_enabled:
            return _refusing_verdict(self)
        verdict = write_authorization.authorized_execution_verdict(
            _monitor_section(), self._control_target, batch, authorization=authorization
        )
        return (
            None if verdict.allowed or verdict.reason is VerdictReason.AUTH_EXHAUSTED else verdict
        )

    async def _send_batch(
        self,
        # Each value as write_channel takes it: the extension contract leaves it open.
        batch: list[tuple[str, ContractAny]],
        timeout_s: float | None,
        confirm: bool | None,
    ) -> list[ChannelWriteResult]:
        """Send a batch that passed every check: one result per write, in order.

        The default writes one after the other as a single write does past its
        guard, which already judged the batch — the precheck, the limits, the
        send and the confirming read — and asks :meth:`_send_refusal` before
        each: once it refuses, every write not yet sent is refused by that
        verdict. A connector whose control system takes concurrent puts sends
        them together through :meth:`_send_and_confirm`, and asks once,
        immediately before the puts start.
        """
        results: list[ChannelWriteResult] = []
        for index, (address, value) in enumerate(batch):
            refused = self._send_refusal(batch)
            if refused is not None:
                results.extend(
                    _verdict_refusal(*held, refused, wording=self.context.readonly_wording)
                    for held in batch[index:]
                )
                break
            try:
                results.append(
                    await self._write_one(address, value, timeout_s=timeout_s, confirm=confirm)
                )
            except Exception as exc:
                results.append(_batch_send_verdict(address, value, exc))
        return results

    async def _write_one(
        self,
        address: str,
        # The value write_channel was given: the extension contract leaves it open.
        value: ContractAny,
        *,
        timeout_s: float | None,
        confirm: bool | None,
    ) -> ChannelWriteResult:
        """One write past its guard: steps 3 to 9 of :meth:`write_channel`.

        The single write runs it inside the door its guard opened; the default
        batch runs it for each write of a batch that was judged whole.
        """
        address, refusal = self._precheck(address, value)
        if refusal is not None:
            return refusal
        bound = self._bound(timeout_s)
        try:
            validator = self._validator()
        except Exception as exc:
            # A validator hook that raised has not said yes.
            return self._validation_refusal(address, value, exc)
        if confirm is None:
            confirm = _confirm_policy(validator, address)
        if validator is not None:
            refusal = await self._limits_check(address, value, validator, bound)
            if refusal is not None:
                return refusal
        return await self._send_and_confirm(address, value, timeout_s=bound, confirm=confirm)

    async def _limits_check(
        self, address: str, value: object, validator: "LimitsValidator", timeout_s: float
    ) -> ChannelWriteResult | None:
        """The refusal the limits answer one write with, or ``None`` when it passed.

        The check runs through :meth:`_run_check` within *timeout_s*. A
        violation is a ``LIMITS`` refusal. Any other error, a check that
        outlived its bound among them, means the check could not be made, and a
        check that could not be made is not permission to write: a
        ``VALIDATION_ERROR`` refusal. Either way nothing is sent.
        """
        from osprey_connectors.errors import ChannelLimitsViolationError

        reader = self.current_value_reader()

        def check() -> None:
            validator.validate(address, value, read_current=reader)

        try:
            await self._run_check(check, timeout_s=timeout_s)
        except ChannelLimitsViolationError as exc:
            return _limits_refusal(address, value, exc)
        except TimeoutError:
            return self._validation_refusal(
                address,
                value,
                TimeoutError(f"the limits check did not finish within {timeout_s}s"),
            )
        except Exception as exc:
            return self._validation_refusal(address, value, exc)
        logger.debug(f"Limits validation passed: {address}={value}")
        return None

    async def _send_and_confirm(
        self,
        address: str,
        # The value write_channel was given: the extension contract leaves it open.
        value: ContractAny,
        *,
        timeout_s: float,
        confirm: bool,
    ) -> ChannelWriteResult:
        """Send one checked write and confirm it when asked: steps 6 to 9 of :meth:`write_channel`.

        What a batch that sends its writes together calls for each, once every
        write of it was checked.
        """
        report = await self._send(address, value, timeout_s=timeout_s, wait=confirm)
        if report.fault is not None:
            return _fault_result(address, value, report)
        if not confirm:
            logger.debug(f"Write (unconfirmed by request): {address} = {value}")
            return ChannelWriteResult.unrequested(address, value)
        try:
            observed = await self._confirming_read(address, report, timeout_s=timeout_s)
        except Exception as exc:
            logger.warning(f"Confirming read failed for {address}: {exc}")
            return ChannelWriteResult.unconfirmed(
                address,
                value,
                f"Write to '{address}' could not be confirmed — "
                f"the read that followed it failed: {exc}",
                notes="The value was sent; what the channel holds is unknown",
            )
        sent = value if report.sent is None else report.sent
        if values_match(sent, observed.value, enum_label=observed.enum_label):
            logger.debug(f"Write confirmed: {address} = {observed.value!r}")
            return ChannelWriteResult.confirmed(address, value, observed)
        logger.warning(
            f"Write mismatch on {address}: sent {value!r}, channel holds {observed.value!r}"
        )
        return ChannelWriteResult.mismatch(address, value, observed)

    @abstractmethod
    async def _send(
        self,
        address: str,
        # The value write_channel was given: the extension contract leaves it open.
        value: ContractAny,
        *,
        timeout_s: float,
        wait: bool,
    ) -> SendReport:
        """Put one value that passed every check, and say what became of it.

        The one transport step a connector must write: the write guard, the
        limits and every result are the base class's. *wait* is ``True`` when
        the write will be confirmed: a transport with an acknowledgement of its
        own (the Channel Access put-callback) waits for it then, so the
        confirming read does not race the control system's processing.

        Return a :class:`SendReport` — no fault when the value went out, else
        the :data:`SendFault` and the transport's words — within *timeout_s*:
        a send that outlives it is ``unacknowledged``, since the value may
        still arrive. Raise only when the outcome is genuinely unknown and no
        fault describes it: :class:`~osprey_connectors.errors.ChannelUnreachableError`
        propagates from a single write and is ``unconfirmed`` in a batch.
        """

    async def _confirming_read(
        self,
        address: str,
        sent: SendReport,  # noqa: ARG002 - what a connector's own read-back may need
        *,
        timeout_s: float,
    ) -> ChannelValue:
        """Read back the channel a write just sent *sent* to.

        The default is :meth:`read_channel` within *timeout_s*. A connector
        whose ordinary read is not a fresh one (a cached or moving value)
        answers with what the control system now holds. A read that raises
        leaves the write ``unconfirmed``.
        """
        return await self.read_channel(address, timeout_s=timeout_s)

    async def _run_check(self, check: Callable[[], None], *, timeout_s: float) -> None:
        """Run one write's limits check, *check*, which may block on a read.

        The default runs it in a worker thread within *timeout_s*, so a
        ``max_step`` read never stalls the event loop; a check that outlives its
        bound raises ``TimeoutError`` and the write is refused with nothing
        sent. A connector with an executor of its own for write work, or one
        whose reads never block, runs it there.
        """
        await asyncio.wait_for(asyncio.to_thread(check), timeout_s)

    @abstractmethod
    async def subscribe(
        self,
        address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        """
        Subscribe to channel changes.

        Args:
            address: Address/name of the channel
            callback: Function called when value changes (receives ChannelValue)
            on_connection: Optional function told the channel's connection state:
                ``True`` when it connects (before the first value of that
                connection), ``False`` when it drops. A channel that never
                connects reports nothing. Called on the event loop, like
                ``callback``. Connectors whose control system reports no
                connection state (DOOCS, Tango) accept and ignore it; every
                ``subscribe`` takes it.

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

    async def wait_stale(self) -> None:
        """
        Raise :class:`~osprey_connectors.errors.ConnectorUnavailableError` once this connector's transport has gone stale.

        Stale means the control system serves a channel again while this
        process's client library still holds it down and will not notice on its
        own; only a new connector in a new process cures it. That is a lost
        connector, so it is raised as the lost-connector root, and the
        supervisor that owns the connector's lifetime replaces it as it would
        after any :class:`~osprey_connectors.errors.ConnectorUnavailableError`
        (:mod:`osprey_connectors.recovery`). It never returns normally. A
        connector that can tell (EPICS with a name server) overrides this
        method.

        This default serves every connector where staleness cannot occur or
        cannot be told (DOOCS, Tango, ...): it waits for ever. A caller that is
        shutting down cancels it.

        An overriding connector counts only channels that connected in this
        process, so in the successor of a stale process a channel that never
        comes back cannot drive a respawn loop.
        """
        await asyncio.get_running_loop().create_future()

    def channel_key(self, address: str) -> str:
        """The one name every spelling of a channel's address shares on this control system.

        A caller that must treat two addresses as one channel (a session
        admitting one write per channel at a time) compares these keys; the
        read or write itself still goes to the address as given. Pure and
        synchronous. This default strips surrounding whitespace; a connector
        whose protocol spells one channel several ways (EPICS: ``ca://X``,
        ``X.VAL`` and ``X``) overrides it.
        """
        return address.strip()

    async def get_metadata(self, address: str) -> ChannelMetadata:
        """
        Get metadata about a channel.

        The default is the metadata of one :meth:`read_channel`; a connector
        whose control system describes a channel beyond a reading (units, a
        description, the drive range) overrides it.

        Args:
            address: Address/name of the channel

        Returns:
            ChannelMetadata with units, precision, description, and (where the
            control system reports one) a display range

        Raises:
            ChannelUnreachableError: This channel cannot be reached; the
                connector stays in place
            ConnectorUnavailableError: The connector's own transport is gone
        """
        return (await self.read_channel(address)).metadata

    async def validate_channel(self, address: str) -> bool:
        """
        Check if channel exists and is accessible.

        The default is a :meth:`read_channel` that does not raise; a connector
        that can tell without reading overrides it.

        Args:
            address: Address/name of the channel

        Returns:
            True if channel is valid and accessible
        """
        try:
            await self.read_channel(address)
        except Exception as exc:
            logger.debug(f"Channel validation failed for {address}: {exc}")
            return False
        return True


#: The base class's own ``_writes_enabled``, which a subclass may override.
_BASE_WRITES_ENABLED: object = ControlSystemConnector.__dict__["_writes_enabled"]

#: The methods a subclass may not define (:meth:`ControlSystemConnector.__init_subclass__`).
_FINAL_METHODS = frozenset(
    {"connect", "write_channel", "write_multiple_channels", "write_channel_checked"}
)


def _confirm_policy(validator: "LimitsValidator | None", address: str) -> bool:
    """Whether a write to *address* is confirmed: the limits database's ``confirm``, else ``True``."""
    if validator is None:
        return True
    return bool(validator.resolve_confirm(address))


class ChannelReadBatch(FrozenModel):
    """What a batch read found: each channel's value, or the error its own read raised.

    Every address the batch was asked for is in exactly one of the two
    mappings. ``failures`` holds each failed channel's own
    :class:`~osprey_connectors.errors.ChannelError`, so a timeout, a refused
    connection and a missing channel stay distinguishable without reading
    any channel again.
    """

    # The failures are live exceptions, which pydantic can only check by class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: ReadOnlyMapping[str, ChannelValue] = Field(default_factory=empty_mapping)
    failures: ReadOnlyMapping[str, ChannelError] = Field(default_factory=empty_mapping)

    @model_validator(mode="after")
    def _one_answer_per_channel(self) -> Self:
        both = sorted(set(self.values) & set(self.failures))
        if both:
            raise ValueError(f"a batch read answers each channel once; read and failed: {both}")
        return self

    def raise_for_failures(self) -> Self:
        """Return the batch unchanged when every channel was read.

        Raises:
            ChannelReadFailedError: Naming every failed channel, in the order the
                batch holds them, with each one's own error as its cause; the
                first failure is also the raised error's ``__cause__``.
        """
        if not self.failures:
            return self
        first = next(iter(self.failures.values()))
        raise ChannelReadFailedError(list(self.failures), causes=self.failures) from first


class ControlSystem(Protocol):
    """A control system as a consumer holds it: read, write, and let go.

    The surface every connector handle offers, whether it is a
    :class:`ControlSystemConnector` in this process, a
    :class:`~osprey_connectors.ipc.ConnectorHostProxy` on a connector-host
    child, or a :class:`~osprey_connectors.ipc.PooledConnector` a pool serves.
    A caller that only reads, writes and disconnects types its connector as
    this, so it takes any of the three.

    ``subscribe``, ``get_metadata``, ``channel_key``, ``wait_stale`` and
    ``transport`` are not part of it: the proxy has no wire method for them,
    so a caller that needs one holds a :class:`ControlSystemConnector`.

    :meth:`disconnect` means "this handle is finished". What a later call on a
    finished handle does is the handle's own: a connector or a proxy refuses
    it, while a pooled handle starts a fresh child for its key.
    """

    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> ChannelValue:
        """Read one channel."""
        ...

    async def read_multiple_channels(
        self, addresses: Sequence[str], /, *, timeout_s: float | None = None
    ) -> ChannelReadBatch:
        """Read several channels, each answering for itself."""
        ...

    async def write_channel(
        self,
        address: str,
        value: ContractAny,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Write one channel through the reference monitor."""
        ...

    async def write_multiple_channels(
        self,
        operations: Sequence[tuple[str, ContractAny]],
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> list[ChannelWriteResult]:
        """Write several channels as one batch."""
        ...

    async def write_channel_checked(
        self,
        address: str,
        value: ContractAny,
        *,
        timeout_s: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Write one channel, raising unless it may be proceeded on."""
        ...

    async def validate_channel(self, address: str) -> bool:
        """Whether the channel exists and is reachable."""
        ...

    async def disconnect(self) -> None:
        """Finish with this handle."""
        ...
