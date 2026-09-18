"""The shape of ``mapping.yaml``, parsed into frozen dataclasses.

``mapping.yaml`` holds every semantic decision an MML install makes: the
facility token, what each system is called, how each family is typed, and the
direction of every signal group. This module turns an already-loaded document
(a plain ``dict``; YAML parsing stays in the CLI layer) into :class:`Mapping`
and checks *structure only*: required keys are present, every value has the
right type, and no unknown key slips through. A misspelt key is refused rather
than ignored, because an ignored key is a decision that silently never lands.

Nulls are structurally valid wherever the document may carry an undecided
slot (a description, a facility token, a family's ``branch``/``class``, a
direction, a judgment answer, a virtual-accelerator system or slot answer).
Rejecting them is the semantic checker's job, which also owns PN_LOCAL,
permutation and cross-reference rules; keeping those out of here lets a freshly
written skeleton parse before anyone has filled it in.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

__all__ = [
    "ATTYPE_KIND",
    "DIRECTION_VALUES",
    "ESCAPE_HATCH_KIND",
    "ROWS_BEYOND_KIND",
    "SHARED_FIELD_KIND",
    "SHARED_KIND",
    "UNBOUND_KIND",
    "UNIT_CLASSES",
    "VA_KINDS",
    "VA_SLOT_KINDS",
    "VA_VERDICTS",
    "AttypeAnswer",
    "Branch",
    "Direction",
    "EscapeHatchAnswer",
    "Facility",
    "Family",
    "FamilyJudgments",
    "Field",
    "FieldAnswer",
    "KickAnswer",
    "Mapping",
    "MappingError",
    "MonitorAnswer",
    "OwnerAnswer",
    "OwnerMap",
    "RowAnswer",
    "SharedAnswer",
    "SharedFieldAnswer",
    "StrengthAnswer",
    "System",
    "UnboundAnswer",
    "VAAnswer",
    "VAFamily",
    "VASlot",
    "VirtualAccelerator",
    "judgment_key",
    "parse_mapping",
]

#: The values a ``directions.*.direction`` slot may hold; ``None`` is undecided.
DIRECTION_VALUES: frozenset[str | None] = frozenset({"read", "write", None})

#: The document spelling of each judgment kind: the key a family's ``judgments``
#: entry carries it under, and the word :func:`judgment_key` renders it by.
ROWS_BEYOND_KIND = "rows_beyond_devices"
UNBOUND_KIND = "unbound_devices"
SHARED_KIND = "shared_pvs"

#: The document spelling of each virtual-accelerator slot kind: the question a
#: rule leaves to a reviewer when it cannot decide a family by itself.
ATTYPE_KIND = "attype"
SHARED_FIELD_KIND = "shared_field"
ESCAPE_HATCH_KIND = "escape_hatch"

#: What a rule reaches for one family: the model drives it, or it stands still.
VA_VERDICTS: frozenset[str] = frozenset({"couple", "latch"})

#: The element kinds a coupled family binds, the vocabulary the bindings file
#: and the VA MAP card share.
VA_KINDS: frozenset[str] = frozenset({"strength", "kick", "monitor", "energy", "rf"})

#: The kinds of slot a reviewer is ever asked to answer.
VA_SLOT_KINDS: frozenset[str] = frozenset({ATTYPE_KIND, SHARED_FIELD_KIND, ESCAPE_HATCH_KIND})

#: The physics-unit words that spell each element kind, folded to lower case:
#: an angle for a kick, a length for a monitor. ``strength`` carries no entry
#: because its class is any other non-empty word, so a unit word is a
#: strength's exactly when it is in neither set.
UNIT_CLASSES: dict[str, frozenset[str]] = {
    "kick": frozenset({"rad", "radian", "radians", "mrad", "mradian", "urad"}),
    "monitor": frozenset({"m", "meter", "meters", "metre", "mm"}),
}


class MappingError(ValueError):
    """Raised when a mapping document has the wrong structure.

    Args:
        key: Dotted path to the offending key, e.g.
            ``families.BPMx.fields.Monitor.provenance``; list items are
            written ``section_order[2]``.
        message: What is wrong, phrased for the person editing the file.
    """

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message

    def __str__(self) -> str:
        """Return ``"<key>: <message>"``, the form ``map --check`` prints."""
        return f"{self.key}: {self.message}"


@dataclass(frozen=True)
class Facility:
    """The facility block: the token every IRI and output filename carries."""

    token: str | None
    title: str | None
    description: str | None
    provenance: str


@dataclass(frozen=True)
class System:
    """One system, keyed in the document by its raw ``ao.json`` token."""

    raw: str
    name: str
    description: str | None
    provenance: str


@dataclass(frozen=True)
class Branch:
    """A facility-declared ontology class that new family classes may extend."""

    name: str
    parent: str
    description: str | None


@dataclass(frozen=True)
class Field:
    """The description of one family field."""

    description: str | None
    provenance: str


@dataclass(frozen=True)
class Family:
    """One family, keyed in the document by its raw ``ao.json`` token.

    ``class_`` (``class`` in the document) is either a packaged ontology class
    or a new token. ``branch`` is ``None`` for a packaged class, and both are
    ``None`` for a zero-channel family, which carries neither key. ``provenance``
    covers ``description`` only.
    """

    raw: str
    rename: str | None
    branch: str | None
    class_: str | None
    aliases: tuple[str, ...]
    description: str | None
    provenance: str
    channels: int
    fields: dict[str, Field]


@dataclass(frozen=True)
class Direction:
    """The direction of one signal group, keyed ``<raw family>.<field>``."""

    direction: str | None
    provenance: str
    override: bool


@dataclass(frozen=True)
class FieldAnswer:
    """The ``{field: <name>}`` answer: move a row into a field of its own."""

    name: str


#: What one row beyond a family's devices may be answered with; ``None`` is
#: undecided.
RowAnswer = Literal["drop", "device"] | FieldAnswer

#: What one device bound by no channel may be answered with; ``None`` is
#: undecided.
UnboundAnswer = Literal["drop", "keep"]


@dataclass(frozen=True)
class OwnerMap:
    """One owning device per supply group, keyed by the group's lowest ordinal.

    Ordinals are 1-based, as the document and ``PROFILE.md`` write them. A
    group answered ``keep_all`` keeps its PV on every member.
    """

    owners: dict[int, int | Literal["keep_all"]]


#: What a family's shared PVs may be answered with; ``None`` is undecided.
SharedAnswer = Literal["keep_all"] | OwnerMap


@dataclass(frozen=True)
class FamilyJudgments:
    """The reviewer's answers for one family, keyed by its raw token.

    A family carries only the kinds that apply to it, so an absent kind is an
    empty ``dict``. ``shared_pvs`` is one slot rather than a dict, so
    ``shared_pvs_present`` tells an absent slot (no shared PVs in the export)
    from a ``null`` one (a pending answer). Answers carry no ``provenance``:
    the machine never pre-fills one, so a non-null answer is stated by
    construction.
    """

    rows_beyond: dict[str, dict[str, RowAnswer | None]] = field(default_factory=dict)
    unbound_devices: dict[int, UnboundAnswer | None] = field(default_factory=dict)
    shared_pvs: SharedAnswer | None = None
    shared_pvs_present: bool = False


def judgment_key(
    family: str,
    kind: str | None = None,
    field: str | None = None,
    signal: str | None = None,
    ordinal: object | None = None,
) -> str:
    """Render the document path of one judgment slot.

    A signal goes in square brackets, because a PV name carries dots of its
    own and the dotted rendering ``<key>: <message>`` would otherwise be
    ambiguous.

    Args:
        family: The raw family token the slot lives under.
        kind: ``rows_beyond_devices``, ``unbound_devices`` or ``shared_pvs``.
        field: The field name, for a row beyond devices.
        signal: The signal of that row, written in square brackets.
        ordinal: A device ordinal or a supply group's lowest ordinal.

    Returns:
        The path, e.g.
        ``judgments.DCCT.rows_beyond_devices.Monitor[SR:C03-BI{DCCT:1}Lifetime-I]``.
    """
    key = f"judgments.{family}"
    if kind is not None:
        key += f".{kind}"
    if field is not None:
        key += f".{field}"
    if signal is not None:
        key += f"[{signal}]"
    if ordinal is not None:
        key += f".{ordinal}"
    return key


@dataclass(frozen=True)
class StrengthAnswer:
    """The ``strength:<PolynomB|PolynomA>[<index>]`` answer: a magnet strength.

    ``index`` is the position in the polynomial, so ``PolynomB[1]`` is a
    quadrupole gradient and ``PolynomB[2]`` a sextupole one.
    """

    attribute: Literal["PolynomB", "PolynomA"]
    index: int


@dataclass(frozen=True)
class KickAnswer:
    """The ``kick:<0|1>`` answer: a corrector, in the plane of ``KickAngle``."""

    plane: Literal[0, 1]


@dataclass(frozen=True)
class MonitorAnswer:
    """The ``monitor:<x|y>`` answer: a beam-position reading, in one plane."""

    plane: Literal["x", "y"]


@dataclass(frozen=True)
class OwnerAnswer:
    """The ``owner:<family>`` answer: which family of a collision binds the field.

    The named family is checked against the collision by the semantic checker;
    here it is a token.
    """

    family: str


#: What an ``attype`` slot may be answered with; ``None`` is undecided.
AttypeAnswer = Literal["latch", "energy", "rf"] | StrengthAnswer | KickAnswer | MonitorAnswer

#: What a ``shared_field`` slot may be answered with; ``None`` is undecided.
SharedFieldAnswer = Literal["latch"] | OwnerAnswer

#: What an ``escape_hatch`` slot may be answered with; ``None`` is undecided.
EscapeHatchAnswer = Literal["latch", "ignore_hook"]

#: Any slot answer, whatever the slot's kind.
VAAnswer = AttypeAnswer | SharedFieldAnswer | EscapeHatchAnswer


@dataclass(frozen=True)
class VASlot:
    """The one question a rule left open for a family, and its answer.

    ``question`` is written out because the card asks it as it stands.
    ``answer`` is the reviewer's word, parsed into the decision it states;
    ``None`` is a pending answer, which ``map --check`` refuses.
    """

    kind: str
    question: str
    answer: VAAnswer | None


@dataclass(frozen=True)
class VAFamily:
    """What the virtual accelerator does with one family.

    A ``couple`` verdict carries what the model needs to drive the family:
    its ``kind``, the ``element_field`` it writes, the ``calibration`` kind
    that converts hardware to physics, and where its nominal came from. A
    ``latch`` verdict carries a ``reason`` instead and binds nothing. Either
    may carry a ``slot``, which is the question that decides the family once
    it is answered.
    """

    verdict: str
    kind: str | None = None
    element_field: str | None = None
    calibration: str | None = None
    nominal_source: str | None = None
    reason: str | None = None
    slot: VASlot | None = None


@dataclass(frozen=True)
class VirtualAccelerator:
    """The virtual-accelerator block: one verdict per exported family.

    ``system`` names the system the block describes, ``None`` while the choice
    is the reviewer's. ``families`` keeps document order, so the card lists
    them as the file spells them.
    """

    system: str | None
    families: dict[str, VAFamily] = field(default_factory=dict)


@dataclass(frozen=True)
class Mapping:
    """A structurally valid ``mapping.yaml``.

    Every dict preserves document order, so emitters that iterate it are
    deterministic for a given file.
    """

    facility: Facility
    systems: dict[str, System]
    section_order: tuple[str, ...]
    branches: dict[str, Branch] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    directions: dict[str, Direction] = field(default_factory=dict)
    judgments: dict[str, FamilyJudgments] = field(default_factory=dict)
    virtual_accelerator: VirtualAccelerator | None = None

    def mapped(self, raw_family: str) -> str:
        """Return the token a raw family is known by downstream.

        Args:
            raw_family: A family key exactly as it appears under ``families:``.

        Returns:
            The family's ``rename`` when set, otherwise ``raw_family``.

        Raises:
            KeyError: ``raw_family`` is not a key of ``families:``.
        """
        family = self.families[raw_family]
        return family.rename if family.rename is not None else raw_family

    def ordered_systems(
        self, present: Iterable[str], order: Sequence[str] | None = None
    ) -> list[tuple[str, list[str]]]:
        """Group the raw system tokens of an export under their mapped names.

        Args:
            present: The raw system tokens the export holds.
            order: The mapped names to list, default :attr:`section_order`.

        Returns:
            ``(mapped name, raw tokens)`` pairs in ``order``, one per name any
            present token maps to; the tokens keep ``systems:`` document order.
            Two raw systems sharing a name land in one pair.

        Raises:
            ValueError: An entry of ``order`` names no present system, or a
                present system is under no entry of ``order``.
        """
        if order is None:
            order = self.section_order
        present = list(present)
        by_name: dict[str, list[str]] = {}
        for raw, system in self.systems.items():
            by_name.setdefault(system.name, []).append(raw)

        ordered: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        for name in order:
            raws = [raw for raw in by_name.get(name, []) if raw in present and raw not in seen]
            if not raws and not any(raw in seen for raw in by_name.get(name, [])):
                raise ValueError(f"section_order names {name!r}, which is no system in ao.json")
            if raws:
                seen.update(raws)
                ordered.append((name, raws))
        missing = [raw for raw in present if raw not in seen]
        if missing:
            raise ValueError(f"section_order leaves out the ao.json systems {missing!r}")
        return ordered

    def provenance_slots(self) -> Iterator[tuple[str, str]]:
        """Yield ``(key, provenance)`` for every provenance-bearing entry.

        Keys are ``facility``, ``systems.<raw>``, ``families.<raw>``,
        ``families.<raw>.fields.<field>`` and ``directions.<raw>.<field>``, in
        that order and in document order within each block.
        """
        yield "facility", self.facility.provenance
        for raw, system in self.systems.items():
            yield f"systems.{raw}", system.provenance
        for raw, family in self.families.items():
            yield f"families.{raw}", family.provenance
            for name, fld in family.fields.items():
                yield f"families.{raw}.fields.{name}", fld.provenance
        for key, direction in self.directions.items():
            yield f"directions.{key}", direction.provenance


# -- structural helpers -------------------------------------------------------

_MISSING = object()


def _dict(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise MappingError(key, f"must be a mapping, got {_type_name(value)}")
    return value


def _entries(value: Any, key: str) -> Iterator[tuple[str, dict, str]]:
    """Yield ``(name, body, path)`` for a dict of named dict entries."""
    for name, body in _dict(value, key).items():
        path = f"{key}.{name}"
        if not isinstance(name, str):
            raise MappingError(path, f"key must be a string, got {_type_name(name)}")
        yield name, _dict(body, path), path


def _keys(body: dict, path: str, required: frozenset[str], optional: frozenset[str]) -> None:
    for name in body:
        if name not in required and name not in optional:
            raise MappingError(_join(path, name), "unknown key")
    for name in sorted(required):
        if name not in body:
            raise MappingError(_join(path, name), "required key is missing")


def _join(path: str, name: object) -> str:
    return f"{path}.{name}" if path else str(name)


@overload
def _str(body: dict, name: str, path: str, *, nullable: Literal[True]) -> str | None: ...


@overload
def _str(body: dict, name: str, path: str, *, nullable: Literal[False]) -> str: ...


def _str(body: dict, name: str, path: str, *, nullable: bool) -> str | None:
    value = body.get(name)
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        expected = "a string or null" if nullable else "a string"
        raise MappingError(f"{path}.{name}", f"must be {expected}, got {_type_name(value)}")
    return value


def _str_list(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MappingError(key, f"must be a list, got {_type_name(value)}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise MappingError(f"{key}[{index}]", f"must be a string, got {_type_name(item)}")
    return tuple(value)


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _shown(value: Any) -> str:
    """Name a refused answer: the word itself when it is one, else its type."""
    return repr(value) if isinstance(value, str) else _type_name(value)


# -- blocks -------------------------------------------------------------------

_TOP_REQUIRED = frozenset({"facility", "systems", "section_order", "families", "directions"})
_TOP_OPTIONAL = frozenset({"branches", "judgments", "virtual_accelerator"})
_FACILITY_KEYS = frozenset({"token", "title", "description", "provenance"})
_SYSTEM_KEYS = frozenset({"name", "description", "provenance"})
_BRANCH_KEYS = frozenset({"parent", "description"})
_FAMILY_REQUIRED = frozenset({"aliases", "description", "provenance", "channels", "fields"})
_FAMILY_OPTIONAL = frozenset({"rename", "branch", "class"})
_FIELD_KEYS = frozenset({"description", "provenance"})
_DIRECTION_REQUIRED = frozenset({"direction", "provenance"})
_DIRECTION_OPTIONAL = frozenset({"override"})
_NONE: frozenset[str] = frozenset()
_JUDGMENT_KINDS = frozenset({ROWS_BEYOND_KIND, UNBOUND_KIND, SHARED_KIND})
_FIELD_ANSWER_KEYS = frozenset({"field"})
_ROW_ANSWERS = "drop, device, a field: entry or null"
_VA_KEYS = frozenset({"system", "families"})
_VA_FAMILY_REQUIRED = frozenset({"verdict"})
_VA_FAMILY_OPTIONAL = frozenset(
    {"kind", "element_field", "calibration", "nominal_source", "reason", "slot"}
)
_VA_SLOT_KEYS = frozenset({"kind", "question", "answer"})
_VA_VERDICTS_SHOWN = "couple or latch"
_VA_KINDS_SHOWN = "strength, kick, monitor, energy, rf or null"
_VA_SLOT_KINDS_SHOWN = "attype, shared_field or escape_hatch"
_ATTYPE_ANSWERS = (
    "latch, strength:<PolynomB|PolynomA>[<i>], kick:<0|1>, energy, rf, monitor:<x|y> or null"
)
_SHARED_FIELD_ANSWERS = "owner:<family>, latch or null"
_ESCAPE_HATCH_ANSWERS = "latch, ignore_hook or null"
_STRENGTH_ANSWER = re.compile(r"strength:(PolynomB|PolynomA)\[(\d+)\]")


def _facility(value: Any) -> Facility:
    body = _dict(value, "facility")
    _keys(body, "facility", _FACILITY_KEYS, _NONE)
    return Facility(
        token=_str(body, "token", "facility", nullable=True),
        title=_str(body, "title", "facility", nullable=True),
        description=_str(body, "description", "facility", nullable=True),
        provenance=_str(body, "provenance", "facility", nullable=False),
    )


def _systems(value: Any) -> dict[str, System]:
    systems: dict[str, System] = {}
    for raw, body, path in _entries(value, "systems"):
        _keys(body, path, _SYSTEM_KEYS, _NONE)
        systems[raw] = System(
            raw=raw,
            name=_str(body, "name", path, nullable=False),
            description=_str(body, "description", path, nullable=True),
            provenance=_str(body, "provenance", path, nullable=False),
        )
    return systems


def _branches(value: Any) -> dict[str, Branch]:
    branches: dict[str, Branch] = {}
    for name, body, path in _entries(value, "branches"):
        _keys(body, path, _BRANCH_KEYS, _NONE)
        branches[name] = Branch(
            name=name,
            parent=_str(body, "parent", path, nullable=False),
            description=_str(body, "description", path, nullable=True),
        )
    return branches


def _fields(value: Any, key: str) -> dict[str, Field]:
    fields: dict[str, Field] = {}
    for name, body, path in _entries(value, key):
        _keys(body, path, _FIELD_KEYS, _NONE)
        fields[name] = Field(
            description=_str(body, "description", path, nullable=True),
            provenance=_str(body, "provenance", path, nullable=False),
        )
    return fields


def _channels(body: dict, path: str) -> int:
    value = body["channels"]
    # bool is an int subclass; a YAML ``true`` is not a count.
    if not isinstance(value, int) or isinstance(value, bool):
        raise MappingError(f"{path}.channels", f"must be an integer, got {_type_name(value)}")
    if value < 0:
        raise MappingError(f"{path}.channels", f"must not be negative, got {value}")
    return value


def _families(value: Any) -> dict[str, Family]:
    families: dict[str, Family] = {}
    for raw, body, path in _entries(value, "families"):
        _keys(body, path, _FAMILY_REQUIRED, _FAMILY_OPTIONAL)
        families[raw] = Family(
            raw=raw,
            rename=_str(body, "rename", path, nullable=True),
            branch=_str(body, "branch", path, nullable=True),
            class_=_str(body, "class", path, nullable=True),
            aliases=_str_list(body["aliases"], f"{path}.aliases"),
            description=_str(body, "description", path, nullable=True),
            provenance=_str(body, "provenance", path, nullable=False),
            channels=_channels(body, path),
            fields=_fields(body["fields"], f"{path}.fields"),
        )
    return families


def _directions(value: Any) -> dict[str, Direction]:
    directions: dict[str, Direction] = {}
    for key, body, path in _entries(value, "directions"):
        family, _, fld = key.partition(".")
        if not family or not fld or "." in fld:
            raise MappingError(path, "key must be '<family>.<field>'")
        _keys(body, path, _DIRECTION_REQUIRED, _DIRECTION_OPTIONAL)
        direction = body["direction"]
        if not (
            direction is None or (isinstance(direction, str) and direction in DIRECTION_VALUES)
        ):
            raise MappingError(
                f"{path}.direction", f"must be read, write or null, got {direction!r}"
            )
        override = body.get("override", False)
        if not isinstance(override, bool):
            raise MappingError(
                f"{path}.override", f"must be true or false, got {_type_name(override)}"
            )
        directions[key] = Direction(
            direction=direction,
            provenance=_str(body, "provenance", path, nullable=False),
            override=override,
        )
    return directions


def _row_answer(value: Any, key: str) -> RowAnswer | None:
    if value is None:
        return None
    if isinstance(value, dict):
        _keys(value, key, _FIELD_ANSWER_KEYS, _NONE)
        return FieldAnswer(name=_str(value, "field", key, nullable=False))
    if value == "drop":
        return "drop"
    if value == "device":
        return "device"
    raise MappingError(key, f"must be {_ROW_ANSWERS}, got {_shown(value)}")


def _rows_beyond(value: Any, family: str) -> dict[str, dict[str, RowAnswer | None]]:
    kind = ROWS_BEYOND_KIND
    rows: dict[str, dict[str, RowAnswer | None]] = {}
    for name, body, _ in _entries(value, judgment_key(family, kind)):
        answers: dict[str, RowAnswer | None] = {}
        for signal, answer in body.items():
            key = judgment_key(family, kind, name, signal)
            if not isinstance(signal, str):
                raise MappingError(key, f"key must be a signal, got {_type_name(signal)}")
            answers[signal] = _row_answer(answer, key)
        rows[name] = answers
    return rows


def _unbound_answer(value: Any, key: str) -> UnboundAnswer | None:
    if value is None:
        return None
    if value == "drop":
        return "drop"
    if value == "keep":
        return "keep"
    raise MappingError(key, f"must be drop, keep or null, got {_shown(value)}")


def _unbound_devices(value: Any, family: str) -> dict[int, UnboundAnswer | None]:
    kind = UNBOUND_KIND
    unbound: dict[int, UnboundAnswer | None] = {}
    for ordinal, answer in _dict(value, judgment_key(family, kind)).items():
        key = judgment_key(family, kind, ordinal=ordinal)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise MappingError(key, f"key must be a device ordinal, got {_type_name(ordinal)}")
        unbound[ordinal] = _unbound_answer(answer, key)
    return unbound


def _shared_pvs(value: Any, family: str) -> SharedAnswer | None:
    kind = SHARED_KIND
    key = judgment_key(family, kind)
    if value is None:
        return None
    if isinstance(value, dict):
        owners: dict[int, int | Literal["keep_all"]] = {}
        for group, owner in value.items():
            slot = judgment_key(family, kind, ordinal=group)
            if isinstance(group, bool) or not isinstance(group, int):
                raise MappingError(slot, f"key must be a device ordinal, got {_type_name(group)}")
            if owner == "keep_all":
                owners[group] = "keep_all"
                continue
            if isinstance(owner, bool) or not isinstance(owner, int):
                raise MappingError(
                    slot, f"must be a device ordinal or keep_all, got {_shown(owner)}"
                )
            owners[group] = owner
        return OwnerMap(owners=owners)
    if value == "keep_all":
        return "keep_all"
    raise MappingError(key, f"must be keep_all, an owner map or null, got {_shown(value)}")


def _judgments(value: Any) -> dict[str, FamilyJudgments]:
    judgments: dict[str, FamilyJudgments] = {}
    for family, body, path in _entries(value, "judgments"):
        _keys(body, path, _NONE, _JUDGMENT_KINDS)
        rows = body.get(ROWS_BEYOND_KIND, _MISSING)
        unbound = body.get(UNBOUND_KIND, _MISSING)
        shared = body.get(SHARED_KIND, _MISSING)
        judgments[family] = FamilyJudgments(
            rows_beyond={} if rows is _MISSING else _rows_beyond(rows, family),
            unbound_devices={} if unbound is _MISSING else _unbound_devices(unbound, family),
            shared_pvs=None if shared is _MISSING else _shared_pvs(shared, family),
            shared_pvs_present=shared is not _MISSING,
        )
    return judgments


def _attype_answer(value: Any, key: str) -> AttypeAnswer | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in ("latch", "energy", "rf"):
            return value
        match = _STRENGTH_ANSWER.fullmatch(value)
        if match is not None:
            attribute: Literal["PolynomB", "PolynomA"] = (
                "PolynomB" if match[1] == "PolynomB" else "PolynomA"
            )
            return StrengthAnswer(attribute=attribute, index=int(match[2]))
        if value in ("kick:0", "kick:1"):
            return KickAnswer(plane=0 if value == "kick:0" else 1)
        if value in ("monitor:x", "monitor:y"):
            return MonitorAnswer(plane="x" if value == "monitor:x" else "y")
    raise MappingError(key, f"must be {_ATTYPE_ANSWERS}, got {_shown(value)}")


def _shared_field_answer(value: Any, key: str) -> SharedFieldAnswer | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "latch":
            return value
        owner, sep, family = value.partition(":")
        if owner == "owner" and sep and family:
            return OwnerAnswer(family=family)
    raise MappingError(key, f"must be {_SHARED_FIELD_ANSWERS}, got {_shown(value)}")


def _escape_hatch_answer(value: Any, key: str) -> EscapeHatchAnswer | None:
    if value is None:
        return None
    if isinstance(value, str) and value in ("latch", "ignore_hook"):
        return value
    raise MappingError(key, f"must be {_ESCAPE_HATCH_ANSWERS}, got {_shown(value)}")


#: The typed parser of each slot kind's answer vocabulary. Every vocabulary is
#: closed: a word outside it is refused by name, never carried through.
_VA_ANSWERS = {
    ATTYPE_KIND: _attype_answer,
    SHARED_FIELD_KIND: _shared_field_answer,
    ESCAPE_HATCH_KIND: _escape_hatch_answer,
}


def _va_slot(value: Any, path: str) -> VASlot | None:
    if value is None:
        return None
    body = _dict(value, path)
    _keys(body, path, _VA_SLOT_KEYS, _NONE)
    kind = body["kind"]
    if not (isinstance(kind, str) and kind in VA_SLOT_KINDS):
        raise MappingError(f"{path}.kind", f"must be {_VA_SLOT_KINDS_SHOWN}, got {_shown(kind)}")
    return VASlot(
        kind=kind,
        question=_str(body, "question", path, nullable=False),
        answer=_VA_ANSWERS[kind](body["answer"], f"{path}.answer"),
    )


def _va_family(body: dict, path: str) -> VAFamily:
    _keys(body, path, _VA_FAMILY_REQUIRED, _VA_FAMILY_OPTIONAL)
    verdict = body["verdict"]
    if not (isinstance(verdict, str) and verdict in VA_VERDICTS):
        raise MappingError(
            f"{path}.verdict", f"must be {_VA_VERDICTS_SHOWN}, got {_shown(verdict)}"
        )
    kind = body.get("kind")
    if not (kind is None or (isinstance(kind, str) and kind in VA_KINDS)):
        raise MappingError(f"{path}.kind", f"must be {_VA_KINDS_SHOWN}, got {_shown(kind)}")
    return VAFamily(
        verdict=verdict,
        kind=kind,
        element_field=_str(body, "element_field", path, nullable=True),
        calibration=_str(body, "calibration", path, nullable=True),
        nominal_source=_str(body, "nominal_source", path, nullable=True),
        reason=_str(body, "reason", path, nullable=True),
        slot=_va_slot(body.get("slot"), f"{path}.slot"),
    )


def _virtual_accelerator(value: Any) -> VirtualAccelerator:
    key = "virtual_accelerator"
    body = _dict(value, key)
    _keys(body, key, _VA_KEYS, _NONE)
    families: dict[str, VAFamily] = {}
    for raw, family, path in _entries(body["families"], f"{key}.families"):
        families[raw] = _va_family(family, path)
    return VirtualAccelerator(system=_str(body, "system", key, nullable=True), families=families)


def parse_mapping(data: dict) -> Mapping:
    """Parse a loaded ``mapping.yaml`` document, checking structure only.

    Args:
        data: The document as a plain dict (e.g. from ``yaml.safe_load``). It
            is not modified.

    Returns:
        The document as a :class:`Mapping`.

    Raises:
        MappingError: A required key is missing, a value has the wrong type,
            or a key is not part of the schema. ``key`` names the path.
    """
    if not isinstance(data, dict):
        raise MappingError("<document>", f"must be a mapping, got {_type_name(data)}")
    _keys(data, "", _TOP_REQUIRED, _TOP_OPTIONAL)
    branches = data.get("branches", _MISSING)
    judgments = data.get("judgments", _MISSING)
    va = data.get("virtual_accelerator", _MISSING)
    return Mapping(
        facility=_facility(data["facility"]),
        systems=_systems(data["systems"]),
        section_order=_str_list(data["section_order"], "section_order"),
        branches={} if branches is _MISSING else _branches(branches),
        families=_families(data["families"]),
        directions=_directions(data["directions"]),
        judgments={} if judgments is _MISSING else _judgments(judgments),
        virtual_accelerator=None if va is _MISSING else _virtual_accelerator(va),
    )
