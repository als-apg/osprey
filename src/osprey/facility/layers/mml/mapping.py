"""The mml layer's mapping file, ``imported/mml/mapping.yaml``.

The mapping holds every decision an MML import needs that the export does not
state: what each exported system is called as a model, how each family is
typed, which way each signal group points, how a reviewer answered each
judgment the export pends, and what each model wires. The document shape is::

    facility:                 # optional; seeds identity.yaml once
      code: <PN_LOCAL>
      name: <text>            # optional
      description: <text>     # optional
    models:                   # keyed by the raw system token of the export
      <raw>:
        name: <PN_LOCAL model name>
        description: <text | null>
        provenance: <stated | derived | imported>
        wiring:               # optional; the families the model drives or reads
          <raw family>:
            element_field: <the family field the model wires | null>
            engine: {attribute?, index?, axis?} | null   # pyAT's words
            calibration: linear | table | null
    section_order: [<model name>, ...]
    branches: {<class>: {parent, description}}            # optional
    families: {<raw family>: {rename?, branch?, class?, aliases, description,
                              provenance, channels, fields}}
    directions: {<raw family>.<field>: {direction: read | write | null,
                                        provenance, override?}}
    judgments: {<raw family>: {rows_beyond_devices?, unbound_devices?,
                               shared_pvs?}}             # optional

:func:`parse_mapping` checks structure only: required keys are present, every
value has the right type and no unknown key slips through, because an ignored
key is a decision that silently never lands. ``null`` is structurally valid in
every slot a reviewer decides, so a freshly written draft parses;
:func:`require_decided` refuses the import while any such slot is still
``null`` (``import mml: mapping-undecided``).

Every channel's role follows from its field's direction (:func:`field_roles`):
``write`` is a setpoint, ``read`` a readback, and a family's ``Setpoint`` field
reads back through its ``Monitor`` field when the family has both; every other
setpoint is its own pair. A pair is always a readback, so a ``Monitor`` whose
direction is ``write`` is a setpoint of its own and pairs with nothing.

The importer's stops are :class:`ImportStop`: one line each, prefixed
``import mml: <problem>:``, exit status 1.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, TypeGuard, overload

import click

__all__ = [
    "CALIBRATION_KINDS",
    "DIRECTION_VALUES",
    "ENGINE_AXES",
    "ROWS_BEYOND_KIND",
    "SHARED_KIND",
    "UNBOUND_KIND",
    "Branch",
    "Direction",
    "EngineBlock",
    "Family",
    "FamilyJudgments",
    "Field",
    "FieldAnswer",
    "FieldRole",
    "Identity",
    "ImportStop",
    "Mapping",
    "MappingError",
    "Model",
    "OwnerMap",
    "RowAnswer",
    "SharedAnswer",
    "UnboundAnswer",
    "WiringFamily",
    "field_roles",
    "judgment_key",
    "parse_mapping",
    "read_mapping",
    "require_decided",
    "undecided_slots",
]

#: The values a ``directions.*.direction`` slot may hold; ``None`` is undecided.
DIRECTION_VALUES: frozenset[str | None] = frozenset({"read", "write", None})

#: The calibration kinds a wiring family may name: the export states the
#: numbers per device, the mapping only which shape they take.
CALIBRATION_KINDS: frozenset[str] = frozenset({"linear", "table"})

#: The transverse axes a monitor's engine block may read.
ENGINE_AXES: frozenset[str] = frozenset({"x", "y"})

#: The document spelling of each judgment kind.
ROWS_BEYOND_KIND = "rows_beyond_devices"
UNBOUND_KIND = "unbound_devices"
SHARED_KIND = "shared_pvs"


#: The field a family's setpoints are written through and the one it reads
#: back through: a family carrying both pairs the first with the second.
SETPOINT_FIELD = "Setpoint"
MONITOR_FIELD = "Monitor"


class ImportStop(click.ClickException):
    """An importer stop: ``import mml: <problem>: <what>``, one line per finding.

    Args:
        problem: The stop's word, ``mapping-undecided`` or ``mapping-draft``.
        lines: What stopped the import, one entry per printed line.
    """

    exit_code = 1

    def __init__(self, problem: str, lines: list[str]) -> None:
        self.problem = problem
        self.lines = tuple(lines)
        super().__init__("\n".join(f"import mml: {problem}: {line}" for line in lines))

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the stop's lines alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        if file is None:
            click.echo(self.format_message(), err=True, color=self.show_color)
        else:
            click.echo(self.format_message(), file=file, color=self.show_color)


class MappingError(ValueError):
    """Raised when a mapping document has the wrong structure.

    Args:
        key: Dotted path to the offending key, e.g.
            ``models.SR.wiring.QF.engine.index``; list items are written
            ``section_order[2]``.
        message: What is wrong, phrased for the person editing the file.
    """

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message

    def __str__(self) -> str:
        """Return ``"<key>: <message>"``."""
        return f"{self.key}: {self.message}"


@dataclass(frozen=True)
class Identity:
    """The ``facility:`` block, read only to seed ``identity.yaml`` once."""

    code: str
    name: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class EngineBlock:
    """What a wired family is to the engine, in pyAT's words.

    ``attribute`` and ``index`` name the element attribute a setpoint drives
    (``PolynomB`` index 1 is a quadrupole gradient, ``KickAngle`` index 0 a
    horizontal kick); ``axis`` names the orbit plane a monitor reads.
    """

    attribute: str | None = None
    index: int | None = None
    axis: str | None = None


@dataclass(frozen=True)
class WiringFamily:
    """How one model wires one family.

    The mapping states only family words; the element of each device and the
    numbers of its calibration come from the export. ``None`` in any slot is a
    decision nobody has made yet.
    """

    element_field: str | None
    engine: EngineBlock | None
    calibration: str | None


@dataclass(frozen=True)
class Model:
    """One model, keyed in the document by the raw system token it is read from."""

    raw: str
    name: str
    description: str | None
    provenance: str
    wiring: dict[str, WiringFamily] = field(default_factory=dict)


@dataclass(frozen=True)
class Branch:
    """A facility-declared device class that family classes may extend."""

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
    """One family, keyed in the document by its raw export token.

    ``class_`` (``class`` in the document) is a vocabulary class or a new one;
    ``branch`` is the class a new one extends. A family with no channels
    carries neither key.
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


#: What one row beyond a family's devices may be answered with.
RowAnswer = Literal["drop", "device"] | FieldAnswer

#: What one device bound by no channel may be answered with.
UnboundAnswer = Literal["drop", "keep"]


@dataclass(frozen=True)
class OwnerMap:
    """One owning device per supply group, keyed by the group's lowest ordinal.

    Ordinals are 1-based. A group answered ``keep_all`` keeps its channel on
    every member.
    """

    owners: dict[int, int | Literal["keep_all"]]


#: What a family's shared channels may be answered with.
SharedAnswer = Literal["keep_all"] | OwnerMap


@dataclass(frozen=True)
class FamilyJudgments:
    """The reviewer's answers for one family, keyed by its raw token.

    ``shared_pvs_present`` tells an absent slot (nothing shared in the export)
    from a ``null`` one (an answer still pending).
    """

    rows_beyond: dict[str, dict[str, RowAnswer | None]] = field(default_factory=dict)
    unbound_devices: dict[int, UnboundAnswer | None] = field(default_factory=dict)
    shared_pvs: SharedAnswer | None = None
    shared_pvs_present: bool = False


@dataclass(frozen=True)
class Mapping:
    """A structurally valid ``mapping.yaml``; every dict keeps document order."""

    identity: Identity | None
    models: dict[str, Model]
    section_order: tuple[str, ...]
    branches: dict[str, Branch] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    directions: dict[str, Direction] = field(default_factory=dict)
    judgments: dict[str, FamilyJudgments] = field(default_factory=dict)

    def mapped(self, raw_family: str) -> str:
        """Return the token a raw family is known by downstream.

        Raises:
            KeyError: ``raw_family`` is not a key of ``families:``.
        """
        family = self.families[raw_family]
        return family.rename if family.rename is not None else raw_family


def judgment_key(
    family: str,
    kind: str | None = None,
    field: str | None = None,
    signal: str | None = None,
    ordinal: object | None = None,
) -> str:
    """Render the document path of one judgment slot.

    A signal goes in square brackets, because a channel name carries dots of
    its own.

    Returns:
        The path, e.g. ``judgments.DCCT.rows_beyond_devices.Monitor[SR:DCCT:Life]``.
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


# -- structural helpers -------------------------------------------------------

_MISSING = object()
_NONE: frozenset[str] = frozenset()


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _shown(value: Any) -> str:
    """Name a refused answer: the word itself when it is one, else its type."""
    return repr(value) if isinstance(value, str) else _type_name(value)


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


def _join(path: str, name: object) -> str:
    return f"{path}.{name}" if path else str(name)


def _keys(body: dict, path: str, required: frozenset[str], optional: frozenset[str]) -> None:
    for name in body:
        if name not in required and name not in optional:
            raise MappingError(_join(path, name), "unknown key")
    for name in sorted(required):
        if name not in body:
            raise MappingError(_join(path, name), "required key is missing")


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


def _optional_str(body: dict, name: str, path: str) -> str | None:
    """A key that may be left out, and is a string when it is written."""
    if name not in body:
        return None
    return _str(body, name, path, nullable=False)


def _str_list(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MappingError(key, f"must be a list, got {_type_name(value)}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise MappingError(f"{key}[{index}]", f"must be a string, got {_type_name(item)}")
    return tuple(value)


def _ordinal(value: Any) -> TypeGuard[int]:
    """Whether ``value`` is an integer and not a bool (a YAML ``true`` is no count)."""
    return isinstance(value, int) and not isinstance(value, bool)


# -- blocks -------------------------------------------------------------------

_TOP_REQUIRED = frozenset({"models", "section_order", "families", "directions"})
_TOP_OPTIONAL = frozenset({"facility", "branches", "judgments"})
_FACILITY_REQUIRED = frozenset({"code"})
_FACILITY_OPTIONAL = frozenset({"name", "description"})
_MODEL_REQUIRED = frozenset({"name", "description", "provenance"})
_MODEL_OPTIONAL = frozenset({"wiring"})
_WIRING_KEYS = frozenset({"element_field", "engine", "calibration"})
_ENGINE_KEYS = frozenset({"attribute", "index", "axis"})
_BRANCH_KEYS = frozenset({"parent", "description"})
_FAMILY_REQUIRED = frozenset({"aliases", "description", "provenance", "channels", "fields"})
_FAMILY_OPTIONAL = frozenset({"rename", "branch", "class"})
_FIELD_KEYS = frozenset({"description", "provenance"})
_DIRECTION_REQUIRED = frozenset({"direction", "provenance"})
_DIRECTION_OPTIONAL = frozenset({"override"})
_JUDGMENT_KINDS = frozenset({ROWS_BEYOND_KIND, UNBOUND_KIND, SHARED_KIND})
_FIELD_ANSWER_KEYS = frozenset({"field"})


def _identity(value: Any) -> Identity:
    body = _dict(value, "facility")
    _keys(body, "facility", _FACILITY_REQUIRED, _FACILITY_OPTIONAL)
    return Identity(
        code=_str(body, "code", "facility", nullable=False),
        name=_optional_str(body, "name", "facility"),
        description=_optional_str(body, "description", "facility"),
    )


def _engine(value: Any, path: str) -> EngineBlock | None:
    if value is None:
        return None
    body = _dict(value, path)
    _keys(body, path, _NONE, _ENGINE_KEYS)
    if not body:
        raise MappingError(path, "must name an attribute, an index or an axis")
    attribute = _optional_str(body, "attribute", path)
    if attribute is not None and not attribute.strip():
        raise MappingError(f"{path}.attribute", "must name an attribute, got an empty string")
    index = body.get("index")
    if "index" in body and not (_ordinal(index) and index >= 0):
        shown = index if _ordinal(index) else _type_name(index)
        raise MappingError(f"{path}.index", f"must be a non-negative integer, got {shown}")
    axis = body.get("axis")
    if "axis" in body and not (isinstance(axis, str) and axis in ENGINE_AXES):
        raise MappingError(f"{path}.axis", f"must be x or y, got {_shown(axis)}")
    return EngineBlock(attribute=attribute, index=index, axis=axis)


def _wiring(value: Any, path: str) -> dict[str, WiringFamily]:
    wiring: dict[str, WiringFamily] = {}
    for raw, body, entry in _entries(value, path):
        _keys(body, entry, _WIRING_KEYS, _NONE)
        calibration = body["calibration"]
        if not (calibration is None or calibration in CALIBRATION_KINDS):
            raise MappingError(
                f"{entry}.calibration", f"must be linear, table or null, got {_shown(calibration)}"
            )
        wiring[raw] = WiringFamily(
            element_field=_str(body, "element_field", entry, nullable=True),
            engine=_engine(body["engine"], f"{entry}.engine"),
            calibration=calibration,
        )
    return wiring


def _models(value: Any) -> dict[str, Model]:
    models: dict[str, Model] = {}
    for raw, body, path in _entries(value, "models"):
        _keys(body, path, _MODEL_REQUIRED, _MODEL_OPTIONAL)
        wiring = body.get("wiring", _MISSING)
        models[raw] = Model(
            raw=raw,
            name=_str(body, "name", path, nullable=False),
            description=_str(body, "description", path, nullable=True),
            provenance=_str(body, "provenance", path, nullable=False),
            wiring={} if wiring is _MISSING else _wiring(wiring, f"{path}.wiring"),
        )
    return models


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
    if not _ordinal(value):
        raise MappingError(f"{path}.channels", f"must be an integer, got {_type_name(value)}")
    if value < 0:
        raise MappingError(f"{path}.channels", f"must not be negative, got {value}")
    return int(value)


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
                f"{path}.direction", f"must be read, write or null, got {_shown(direction)}"
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
    raise MappingError(key, f"must be drop, device, a field: entry or null, got {_shown(value)}")


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


def _unbound_devices(value: Any, family: str) -> dict[int, UnboundAnswer | None]:
    kind = UNBOUND_KIND
    unbound: dict[int, UnboundAnswer | None] = {}
    for ordinal, answer in _dict(value, judgment_key(family, kind)).items():
        key = judgment_key(family, kind, ordinal=ordinal)
        if not _ordinal(ordinal):
            raise MappingError(key, f"key must be a device ordinal, got {_type_name(ordinal)}")
        if answer is None:
            unbound[ordinal] = None
        elif answer == "drop":
            unbound[ordinal] = "drop"
        elif answer == "keep":
            unbound[ordinal] = "keep"
        else:
            raise MappingError(key, f"must be drop, keep or null, got {_shown(answer)}")
    return unbound


def _shared_pvs(value: Any, family: str) -> SharedAnswer | None:
    kind = SHARED_KIND
    if value is None:
        return None
    if isinstance(value, dict):
        owners: dict[int, int | Literal["keep_all"]] = {}
        for group, owner in value.items():
            slot = judgment_key(family, kind, ordinal=group)
            if not _ordinal(group):
                raise MappingError(slot, f"key must be a device ordinal, got {_type_name(group)}")
            if owner == "keep_all":
                owners[group] = "keep_all"
            elif _ordinal(owner):
                owners[group] = owner
            else:
                raise MappingError(
                    slot, f"must be a device ordinal or keep_all, got {_shown(owner)}"
                )
        return OwnerMap(owners=owners)
    if value == "keep_all":
        return "keep_all"
    raise MappingError(
        judgment_key(family, kind),
        f"must be keep_all, an owner map or null, got {_shown(value)}",
    )


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


def parse_mapping(data: dict) -> Mapping:
    """Parse a loaded ``mapping.yaml`` document, checking structure only.

    Args:
        data: The document as a plain dict (e.g. from ``yaml.safe_load``). It
            is not modified.

    Returns:
        The document as a :class:`Mapping`.

    Raises:
        MappingError: A required key is missing, a value has the wrong type,
            or a key is not part of the shape. ``key`` names the path.
    """
    if not isinstance(data, dict):
        raise MappingError("<document>", f"must be a mapping, got {_type_name(data)}")
    _keys(data, "", _TOP_REQUIRED, _TOP_OPTIONAL)
    facility = data.get("facility", _MISSING)
    branches = data.get("branches", _MISSING)
    judgments = data.get("judgments", _MISSING)
    return Mapping(
        identity=None if facility is _MISSING else _identity(facility),
        models=_models(data["models"]),
        section_order=_str_list(data["section_order"], "section_order"),
        branches={} if branches is _MISSING else _branches(branches),
        families=_families(data["families"]),
        directions=_directions(data["directions"]),
        judgments={} if judgments is _MISSING else _judgments(judgments),
    )


def read_mapping(path: Path) -> Mapping:
    """Read and parse one mapping file.

    Args:
        path: The file, normally ``data/facility/imported/mml/mapping.yaml``.

    Returns:
        The parsed mapping.

    Raises:
        MappingError: The file is not valid YAML or has the wrong structure.
        OSError: The file cannot be read.
    """
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MappingError("<document>", f"is not valid YAML ({exc})") from exc
    return parse_mapping(data)


# -- roles --------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRole:
    """The role every channel of one family field takes, and what it pairs with.

    ``pair`` names the family field a setpoint reads back through; ``None``
    means the channel is its own pair (and always ``None`` for a readback).
    """

    role: Literal["setpoint", "readback"]
    pair: str | None = None


def _undecided_direction(key: str) -> ImportStop:
    return ImportStop("mapping-undecided", [f"directions.{key}.direction: write read or write"])


def field_roles(mapping: Mapping) -> dict[str, FieldRole]:
    """Derive the role of every family field from its direction.

    ``write`` is a setpoint and ``read`` a readback. A family's ``Setpoint``
    field pairs with its ``Monitor`` field when the family has both; every
    other setpoint is its own pair. A setpoint reads back only through a
    readback, so the pair holds only while the ``Monitor`` is ``read``: a
    ``Monitor`` the mapping marks ``write`` is itself a setpoint, and the
    ``Setpoint`` beside it is its own pair.

    Args:
        mapping: The parsed mapping.

    Returns:
        ``{"<raw family>.<field>": FieldRole}`` in ``directions`` order.

    Raises:
        ImportStop: ``mapping-undecided``, naming the first direction still
            ``null``.
    """
    roles: dict[str, FieldRole] = {}
    for key, direction in mapping.directions.items():
        if direction.direction is None:
            raise _undecided_direction(key)
        roles[key] = FieldRole(role="setpoint" if direction.direction == "write" else "readback")
    for key, found in roles.items():
        family, _, name = key.partition(".")
        monitor = roles.get(f"{family}.{MONITOR_FIELD}")
        if name == SETPOINT_FIELD and found.role == "setpoint" and monitor is not None:
            if monitor.role == "readback":
                roles[key] = FieldRole(role="setpoint", pair=MONITOR_FIELD)
    return roles


# -- undecided slots ----------------------------------------------------------

_DESCRIBE = "describe it"
_ANSWER_ROW = "answer drop, device or {field: <name>}"
_ANSWER_UNBOUND = "answer drop or keep"
_ANSWER_SHARED = "answer keep_all or name each group's owner"


def _vocabulary_classes() -> frozenset[str]:
    from osprey.facility.validate import known_classes

    return known_classes()


def undecided_slots(mapping: Mapping) -> list[tuple[str, str]]:
    """List every slot a reviewer still has to decide, in document order.

    A slot is undecided while it is ``null``: a model, family or field
    description; a wiring family's ``element_field``, ``engine`` or
    ``calibration``; the ``class`` of a family with channels, and its
    ``branch`` unless the class is a vocabulary class or a declared branch; a
    direction; and every judgment answer the document carries.

    Args:
        mapping: The parsed mapping.

    Returns:
        ``(key, what to write)`` pairs; empty when everything is decided.
    """
    found: list[tuple[str, str]] = []
    for raw, model in mapping.models.items():
        path = f"models.{raw}"
        if model.description is None:
            found.append((f"{path}.description", _DESCRIBE))
        for wired, wiring in model.wiring.items():
            entry = f"{path}.wiring.{wired}"
            if wiring.element_field is None:
                found.append((f"{entry}.element_field", "name the family field the model wires"))
            if wiring.engine is None:
                found.append(
                    (
                        f"{entry}.engine",
                        "name the attribute and index, or the axis, the model wires",
                    )
                )
            if wiring.calibration is None:
                found.append((f"{entry}.calibration", "write linear or table"))
    known = _vocabulary_classes() | set(mapping.branches)
    for raw, family in mapping.families.items():
        path = f"families.{raw}"
        if family.channels > 0:
            if family.class_ is None:
                found.append((f"{path}.class", "name the device class"))
            elif family.branch is None and family.class_ not in known:
                found.append((f"{path}.branch", "name the class it extends"))
        if family.description is None:
            found.append((f"{path}.description", _DESCRIBE))
        for name, fld in family.fields.items():
            if fld.description is None:
                found.append((f"{path}.fields.{name}.description", _DESCRIBE))
    for key, direction in mapping.directions.items():
        if direction.direction is None:
            found.append((f"directions.{key}.direction", "write read or write"))
    for raw, judgments in mapping.judgments.items():
        for name, answers in judgments.rows_beyond.items():
            for signal, row in answers.items():
                if row is None:
                    found.append((judgment_key(raw, ROWS_BEYOND_KIND, name, signal), _ANSWER_ROW))
        for ordinal, unbound in judgments.unbound_devices.items():
            if unbound is None:
                found.append((judgment_key(raw, UNBOUND_KIND, ordinal=ordinal), _ANSWER_UNBOUND))
        if judgments.shared_pvs_present and judgments.shared_pvs is None:
            found.append((judgment_key(raw, SHARED_KIND), _ANSWER_SHARED))
    return found


def require_decided(mapping: Mapping) -> None:
    """Stop the import while any slot of the mapping is undecided.

    Args:
        mapping: The parsed mapping.

    Raises:
        ImportStop: ``mapping-undecided``, one line per undecided slot.
    """
    found = undecided_slots(mapping)
    if found:
        raise ImportStop("mapping-undecided", [f"{key}: {what}" for key, what in found])
