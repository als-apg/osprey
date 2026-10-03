"""Record and result types for the authoritative channel roster.

The roster answers one question -- *which channels does this facility have,
and which way do they point* -- from one source per build: the facility file
the build writes at the root of every render. Never the write-limits
projection (``channel_limits.json``), which gates a subset and was never a
roster.

This module is purely declarative data, in the style of
:mod:`osprey.simulation.channel_schema`: stdlib-only, no I/O, no source
selection, no parsing. The reader (:mod:`osprey.channel_roster.sources`)
builds these types; the consumers --
plan-device derivation, the channel snapshot, the build's fact lines, the
channel-finder web routes -- read them.

**Absence is data, not a missing return.** A roster that cannot be built is
reported as a :class:`RosterAbsence` carrying its reason and the subjects it
has to name (a path, the failure underneath), and every
consumer renders it through :meth:`RosterAbsence.message`. The phrasing of each
reason therefore lives once, in :data:`ABSENCE_TEMPLATES`, rather than in a
per-consumer ``if`` chain -- build facts and HTTP 503 bodies say the same true
thing because they read the same sentence. Adding a reason without phrasing it
raises there instead of rendering a blank.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Literal

#: Which way a channel points. ``None`` means the source could not say -- an
#: honest unknown, never a stand-in for "read".
ChannelDirection = Literal["read", "write"]

_DIRECTIONS: frozenset[str] = frozenset({"read", "write"})


class RosterSourceKind(Enum):
    """Which kind of source a roster was enumerated from."""

    #: The facility file a build writes at the root of every render, where each
    #: channel record states its role and a setpoint states its readback.
    FACILITY = "facility"


#: How each source kind is named to a human. A table rather than a branch in
#: every consumer, so the build fact line and the web body call the same file
#: the same thing, and a kind nobody has phrased raises here instead of being
#: described as the wrong one.
SOURCE_LABELS: Mapping[RosterSourceKind, str] = {
    RosterSourceKind.FACILITY: "this project's facility file",
}


class RosterAbsenceReason(Enum):
    """Why a roster -- or the direction half of one -- could not be built.

    Named rather than left to each caller's prose because these are different
    situations with different remedies, and every surface that reports one has
    to tell them apart: the facility file is not built, it is built and
    declares no channels, or it is there and unreadable.
    """

    #: No build has written the facility file into this render. Fail-soft,
    #: and its own reason because its remedy is one command rather than a
    #: config edit: the file is an output of ``osprey build``, declared by no
    #: key.
    FACILITY_NOT_BUILT = "facility-not-built"

    #: The facility file is built and holds no channel record. Fail-soft, and
    #: its own reason because its remedy is in the project rather than in the
    #: build: the file says what the project's ``data/facility`` tree declares,
    #: so building again writes the same file.
    FACILITY_EMPTY = "facility-empty"

    #: A configured source is present and could not be read or parsed.
    #: Fail-closed: this is a broken deployment, not an absent one.
    CORRUPT_SOURCE = "corrupt-source"


#: The one place each absence reason is phrased for a human. Fields in braces
#: are filled from the :class:`RosterAbsence` carrying the reason, and a reason
#: whose template names a subject the absence did not supply is rejected at
#: construction -- an absence can never render "at None".
ABSENCE_TEMPLATES: Mapping[RosterAbsenceReason, str] = {
    RosterAbsenceReason.FACILITY_NOT_BUILT: (
        "The facility file {path} is not built, so the set of channels this "
        "facility has is unknown. Run `osprey build`."
    ),
    RosterAbsenceReason.FACILITY_EMPTY: (
        "The facility file {path} declares no channels: the project's "
        "data/facility tree holds no channel records."
    ),
    RosterAbsenceReason.CORRUPT_SOURCE: (
        "The channel roster source at {path} could not be read: {detail}."
    ),
}


def _template_fields(template: str) -> tuple[str, ...]:
    """Return the brace-named fields ``template`` interpolates, in order."""
    return tuple(name for _, name, _, _ in Formatter().parse(template) if name)


@dataclass(frozen=True, slots=True)
class RosterSource:
    """The source a roster was enumerated from, for provenance.

    Attributes:
        kind: Which kind of source this is.
        path: The resolved on-disk path that was read -- what every reader
            opens and what the memo key fingerprints. Resolution (the
            render-relative rule for the facility file) happens in
            :mod:`osprey.channel_roster.sources`; by the time it is here it is
            settled.
        spelled: The name the file is shown under, or None when the resolved
            path is the one to show. The facility file is resolved into the
            render root the build wrote it to, a path nobody retypes or edits,
            while its name, ``facility.json``, is what the build fact and the
            web body call it. Display uses this; I/O never does.
    """

    kind: RosterSourceKind
    path: Path
    spelled: str | None = None

    def describe(self) -> str:
        """Render this source the way a build fact names it.

        Names :attr:`spelled` when there is one, so an operator is handed the
        file's name rather than the render path it was resolved to.

        Returns:
            e.g. ``"this project's facility file (facility.json)"``.
        """
        return f"{SOURCE_LABELS[self.kind]} ({self.spelled or self.path})"

    def for_display(self) -> str:
        """The source path as an operator would recognise it (see :attr:`spelled`)."""
        return self.spelled or str(self.path)


@dataclass(frozen=True, slots=True)
class RosterAbsence:
    """A reason a roster is missing, with the subjects it has to name.

    Attributes:
        reason: Which absence this is.
        path: The source path the reason is about, when it is about one --
            resolved, so a caller can act on it.
        spelled: How that path is named to a human, when the configured
            spelling differs from the resolved one (see
            :attr:`RosterSource.spelled`). The message renders this; ``path``
            stays the resolved file either way.
        detail: The underlying failure, for
            :attr:`RosterAbsenceReason.CORRUPT_SOURCE`.

    Raises:
        ValueError: If the reason's phrasing names a subject this absence did
            not supply -- a corrupt source with no path, say. Caught here
            rather than rendered as "at None" three subsystems later.
    """

    reason: RosterAbsenceReason
    path: Path | None = None
    detail: str | None = None
    spelled: str | None = None

    def __post_init__(self) -> None:
        template = ABSENCE_TEMPLATES[self.reason]
        values = self._values()
        missing = [name for name in _template_fields(template) if not values[name]]
        if missing:
            raise ValueError(
                f"Absence reason {self.reason.value!r} names {', '.join(missing)} in "
                "its message and cannot be built without it."
            )

    def _values(self) -> dict[str, str]:
        """Return the fill values for this absence's :data:`ABSENCE_TEMPLATES` entry."""
        return {
            "path": self.spelled or (str(self.path) if self.path is not None else ""),
            "detail": self.detail or "",
        }

    def message(self) -> str:
        """Render this absence as one honest sentence, for any consumer."""
        return ABSENCE_TEMPLATES[self.reason].format(**self._values())


@dataclass(frozen=True, slots=True)
class ChannelRecord:
    """One channel the facility has, as the roster knows it.

    Attributes:
        address: The channel address, which is also its device name downstream
            (device name == channel address).
        source: Where this record came from, carried per record so a consumer
            holding a bare record can still say what it derives from.
        direction: ``"write"`` for a settable channel, ``"read"`` for a
            readable one, ``None`` when the source could not say.
        readback: The paired readback address of a settable channel: the
            facility file's ``pair`` when it names another channel. ``None``
            when the setpoint is its own pair -- the worker then reads the
            setpoint back.
        role: The facility file's ``role`` for the channel (``"setpoint"``,
            ``"readback"`` or ``"none"``), or ``None`` from a source that
            states only a direction.
        value_type: The facility file's ``value_type``, or ``None`` when the
            source states none.
        description: The channel's description, or ``None`` when it has none.
        on: The device or place the channel belongs to, as the facility file's
            ``on`` states it (``("device", <id>)`` or ``("place", <id>)``), or
            ``None`` when the channel belongs to nothing.

    Raises:
        ValueError: On an empty address, a direction outside ``"read"``/
            ``"write"`` (a typo there would silently unsettle every channel a
            consumer compares), or a readback on a record that is not a write.
    """

    address: str
    source: RosterSource
    direction: ChannelDirection | None = None
    readback: str | None = None
    role: str | None = None
    value_type: str | None = None
    description: str | None = None
    on: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.address:
            raise ValueError("A channel record needs an address.")
        if self.direction is not None and self.direction not in _DIRECTIONS:
            raise ValueError(
                f"Unknown channel direction {self.direction!r} for {self.address}: "
                f"expected one of {sorted(_DIRECTIONS)}, or None when unknown."
            )
        if self.readback is not None and self.direction != "write":
            raise ValueError(
                f"{self.address} carries a readback but is not a write channel; a "
                "readback pairs a setpoint."
            )


@dataclass(frozen=True, slots=True)
class RosterResult:
    """The roster for one build, or the honest reason there is none.

    Attributes:
        records: Every channel the source enumerated, in source order.
        source: What was read, or ``None`` when nothing was.
        absence: Why the roster -- or its direction half -- is missing;
            ``None`` when a complete roster was built.

    Raises:
        ValueError: If the result says nothing (neither a source nor an
            absence), or carries records with no source to attribute them to.
            Either would let a consumer report an empty facility as a fact.
    """

    records: tuple[ChannelRecord, ...] = ()
    source: RosterSource | None = None
    absence: RosterAbsence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if self.source is None:
            if self.records:
                raise ValueError("A roster with records must name the source they came from.")
            if self.absence is None:
                raise ValueError("A roster with no source must say why.")

    @property
    def write_records(self) -> tuple[ChannelRecord, ...]:
        """The settable channels -- the devices a plan may drive."""
        return tuple(record for record in self.records if record.direction == "write")

    @property
    def read_records(self) -> tuple[ChannelRecord, ...]:
        """The readable channels -- one device each, no grammar filter."""
        return tuple(record for record in self.records if record.direction == "read")

    @property
    def addresses(self) -> tuple[str, ...]:
        """Every enumerated address, in source order."""
        return tuple(record.address for record in self.records)
