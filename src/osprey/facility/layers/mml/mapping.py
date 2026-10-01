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
            voltage: <volts>  # optional; a Frequency family's cavity voltage
    section_order: [<model name>, ...]
    branches: {<class>: {parent, description}}            # optional
    families: {<raw family>: {rename?, branch?, class?, devices?, aliases,
                              description, provenance, channels, fields}}
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

A family's ``devices`` says which device each of its slots is, the one thing
an export may leave unstated: ``names`` (the export's ``CommonNames`` by
position), a list of local names (one per device), ``address`` (each slot
named by the device segment of its address) or ``{same_as: <raw family>}``
(slot ``i`` is the device slot ``i`` of the named family is). A family
without the key is identified by ``names``; the import stops
(``mapping-undecided``) where the export names no device for one of its
slots, and (``mapping-invalid``) where an answer does not fit the export.

Every channel's role follows from its field's direction (:func:`field_roles`):
``write`` is a setpoint, ``read`` a readback, and a family's ``Setpoint`` field
reads back through its ``Monitor`` field when the family has both; every other
setpoint is its own pair. A pair is always a readback, so a ``Monitor`` whose
direction is ``write`` is a setpoint of its own and pairs with nothing.

:func:`check_mapping` checks meaning: identity and model names are PN_LOCAL,
``section_order`` lists every model once, classes and branches resolve against
the vocabulary, a ``devices`` answer names another family or one-word device
names, every direction and wiring family names a field the mapping
describes, and -- given the export -- the mapping names exactly the export's
systems and families, gives every channel-bearing field a direction, a
``stated`` direction agrees with the export's own vote unless ``override``, and
the judgment answers match what the export pends: every pending row, unbound
device and shared supply has a slot, every answer names a judgment the export
pends, an owner is a member of its supply group, and a ``field:`` answer is a
name the family can take.

When the file is absent, :func:`load_or_draft` writes a draft built from the
export by :func:`draft_mapping` and stops the import
(``import mml: mapping-draft``), so the reviewer edits one stable file rather
than writing it from scratch. The draft proposes every family's ``devices``:
``names`` where the export names every device, ``same_as`` where exactly one
such family binds the other axis of the same addresses, else ``address`` with
the ids it yields in a comment beneath.

The importer's stops are :class:`ImportStop`: one line each, prefixed
``import mml: <problem>:``, exit status 1.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Callable, Iterator, Sequence
from collections.abc import Mapping as Map
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, TypeGuard, overload

import click

__all__ = [
    "CALIBRATION_KINDS",
    "MAPPING_FILE",
    "DIRECTION_VALUES",
    "DeviceIdentity",
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
    "SameAs",
    "SharedAnswer",
    "Problem",
    "UnboundAnswer",
    "WiringFamily",
    "check_mapping",
    "draft_mapping",
    "draft_text",
    "dump_mapping",
    "field_roles",
    "judgment_key",
    "load_or_draft",
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

#: Where the mapping lives, relative to ``data/facility/``.
MAPPING_FILE = "imported/mml/mapping.yaml"

#: A device's local name, the part of its id after ``<model>/``.
_LOCAL_NAME = re.compile(r"[0-9A-Za-z_]+")

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
    numbers of its calibration come from the export. ``None`` in any slot but
    ``voltage`` is a decision nobody has made yet. ``voltage`` is the volts of
    the cavity a ``Frequency`` family drives, answered only where the deck
    holds no cavity and one is built for it; ``None`` everywhere else.
    """

    element_field: str | None
    engine: EngineBlock | None
    calibration: str | None
    voltage: float | None = None


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
class SameAs:
    """The ``{same_as: <raw family>}`` answer: each slot is the named family's device."""

    family: str


#: How a family's devices are identified: by the export's names, by address,
#: by the mapping's own list of local names, or as another family's devices.
DeviceIdentity = Literal["names", "address"] | tuple[str, ...] | SameAs


@dataclass(frozen=True)
class Family:
    """One family, keyed in the document by its raw export token.

    ``class_`` (``class`` in the document) is a vocabulary class or a new one;
    ``branch`` is the class a new one extends. A family with no channels
    carries neither key. ``devices`` is ``None`` both when the key is left
    out and when it is ``null``; ``devices_present`` tells the two apart.
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
    devices: DeviceIdentity | None = None
    devices_present: bool = False


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
_WIRING_OPTIONAL = frozenset({"voltage"})
_ENGINE_KEYS = frozenset({"attribute", "index", "axis"})
_BRANCH_KEYS = frozenset({"parent", "description"})
_FAMILY_REQUIRED = frozenset({"aliases", "description", "provenance", "channels", "fields"})
_FAMILY_OPTIONAL = frozenset({"rename", "branch", "class", "devices"})
_SAME_AS_KEYS = frozenset({"same_as"})
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
        _keys(body, entry, _WIRING_KEYS, _WIRING_OPTIONAL)
        calibration = body["calibration"]
        if not (calibration is None or calibration in CALIBRATION_KINDS):
            raise MappingError(
                f"{entry}.calibration", f"must be linear, table or null, got {_shown(calibration)}"
            )
        engine = _engine(body["engine"], f"{entry}.engine")
        wiring[raw] = WiringFamily(
            element_field=_str(body, "element_field", entry, nullable=True),
            engine=engine,
            calibration=calibration,
            voltage=_voltage(body, engine, entry),
        )
    return wiring


def _voltage(body: dict, engine: EngineBlock | None, entry: str) -> float | None:
    """The cavity voltage a ``Frequency`` family answers, or ``None`` when it states none."""
    if "voltage" not in body:
        return None
    path = f"{entry}.voltage"
    if engine is None or engine.attribute != "Frequency":
        raise MappingError(
            path, "only a family whose engine attribute is Frequency takes a voltage"
        )
    value = body["voltage"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MappingError(path, f"must be a positive number of volts, got {_shown(value)}")
    if not (math.isfinite(value) and value > 0):
        raise MappingError(path, f"must be a positive number of volts, got {value}")
    return float(value)


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


def _devices(body: dict, path: str) -> DeviceIdentity | None:
    key = f"{path}.devices"
    value = body.get("devices")
    if value is None:
        return None
    if isinstance(value, list):
        names = _str_list(value, key)
        if not names:
            raise MappingError(key, "must name at least one device")
        for index, name in enumerate(names):
            if not name.strip():
                raise MappingError(f"{key}[{index}]", "must name a device, got an empty string")
        return names
    if isinstance(value, dict):
        _keys(value, key, _SAME_AS_KEYS, _NONE)
        return SameAs(family=_str(value, "same_as", key, nullable=False))
    if value == "names":
        return "names"
    if value == "address":
        return "address"
    raise MappingError(
        key,
        f"must be names, address, a list of names, a same_as: entry or null, got {_shown(value)}",
    )


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
            devices=_devices(body, path),
            devices_present="devices" in body,
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
_ANSWER_DEVICES = "write names, address, a list of names or {same_as: <family>}"


def _vocabulary() -> dict[str, str | None]:
    """The vocabulary's device classes, each mapped to its parent (the root to None)."""
    import json
    from importlib import resources

    table = resources.files("osprey.facility.schema._generated") / "vocabulary.json"
    classes = json.loads(table.read_text(encoding="utf-8"))["classes"]
    return {row["name"]: row["parent"] for row in classes}


def _vocabulary_classes() -> frozenset[str]:
    return frozenset(_vocabulary())


def undecided_slots(mapping: Mapping) -> list[tuple[str, str]]:
    """List every slot a reviewer still has to decide, in document order.

    A slot is undecided while it is ``null``: a model, family or field
    description; a wiring family's ``element_field``, ``engine`` or
    ``calibration``; the ``class`` of a family with channels, and its
    ``branch`` unless the class is a vocabulary class or a declared branch; a
    family's ``devices`` when the key is written; a direction; and every
    judgment answer the document carries.

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
        if family.devices_present and family.devices is None:
            found.append((f"{path}.devices", _ANSWER_DEVICES))
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


def require_decided(mapping: Mapping, ao: dict | None = None) -> None:
    """Stop the import while any slot of the mapping is undecided.

    Given the export, a judgment it pends that the document leaves no slot
    for is undecided too: it is listed after the document's own ``null``
    slots, in export order.

    Args:
        mapping: The parsed mapping.
        ao: The merged export the mapping answers, or ``None`` to read the
            document alone.

    Raises:
        ImportStop: ``mapping-undecided``, one line per undecided slot.
    """
    found = undecided_slots(mapping)
    if ao is not None:
        missing = _all_missing_slots(mapping, _pending_by_family(ao))
        found.extend((key, what) for key, _, what in missing)
    if found:
        raise ImportStop("mapping-undecided", [f"{key}: {what}" for key, what in found])


# -- semantic check -----------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One refusal of :func:`check_mapping`: the offending key path and what is wrong."""

    key: str
    message: str

    def __str__(self) -> str:
        """Return ``"<key>: <message>"``."""
        return f"{self.key}: {self.message}"


def _pn_local(token: str) -> bool:
    from osprey.facility import PN_LOCAL

    return PN_LOCAL.fullmatch(token) is not None


def _not_pn_local(key: str, token: str) -> Problem:
    return Problem(key, f"{token!r} is not PN_LOCAL")


def _identity_code(mapping: Mapping) -> Iterator[Problem]:
    if mapping.identity is not None and not _pn_local(mapping.identity.code):
        yield _not_pn_local("facility.code", mapping.identity.code)


def _model_names(mapping: Mapping) -> Iterator[Problem]:
    folded: dict[str, str] = {}
    for raw, model in mapping.models.items():
        key = f"models.{raw}.name"
        if not _pn_local(model.name):
            yield _not_pn_local(key, model.name)
            continue
        first = folded.setdefault(model.name.lower(), raw)
        if first != raw:
            yield Problem(key, f"{model.name!r} is models.{first}'s name up to case")


def _section_order(mapping: Mapping) -> Iterator[Problem]:
    names = {model.name for model in mapping.models.values()}
    seen: set[str] = set()
    for index, name in enumerate(mapping.section_order):
        key = f"section_order[{index}]"
        if name not in names:
            yield Problem(key, f"{name!r} is no model name")
        elif name in seen:
            yield Problem(key, f"{name!r} is listed twice")
        seen.add(name)
    for model in mapping.models.values():
        if model.name not in seen:
            yield Problem("section_order", f"leaves out the model {model.name}")


def _family_tokens(mapping: Mapping) -> Iterator[Problem]:
    folded: dict[str, str] = {}
    for raw, family in mapping.families.items():
        if family.rename is not None and not _pn_local(family.rename):
            yield _not_pn_local(f"families.{raw}.rename", family.rename)
            continue
        token = mapping.mapped(raw)
        first = folded.setdefault(token.lower().replace("-", "_"), raw)
        if first != raw:
            yield Problem(
                f"families.{raw}", f"maps to {token!r}, which families.{first} also maps to"
            )


def _unknown_class(name: str) -> str:
    return f"{name!r} is no vocabulary class and no declared branch"


def _family_classes(mapping: Mapping) -> Iterator[Problem]:
    vocabulary = _vocabulary()
    for raw, family in mapping.families.items():
        path = f"families.{raw}"
        if family.class_ is None:
            continue
        if not _pn_local(family.class_):
            yield _not_pn_local(f"{path}.class", family.class_)
            continue
        if family.class_ in vocabulary and vocabulary[family.class_] is None:
            yield Problem(
                f"{path}.class",
                f"{family.class_!r} is the vocabulary root; name a class under it",
            )
            continue
        branch = family.branch
        if branch is None:
            continue
        parent = vocabulary.get(family.class_)
        if parent is not None and branch != parent:
            yield Problem(
                f"{path}.branch",
                f"{family.class_} is a vocabulary class under {parent}, not {branch}",
            )
        elif branch not in vocabulary and branch not in mapping.branches:
            yield Problem(f"{path}.branch", _unknown_class(branch))


def _family_devices(mapping: Mapping) -> Iterator[Problem]:
    for raw, family in mapping.families.items():
        key = f"families.{raw}.devices"
        answer = family.devices
        if isinstance(answer, SameAs):
            if answer.family == raw:
                yield Problem(key, f"{raw} cannot take its devices from itself")
            elif answer.family not in mapping.families:
                yield Problem(key, f"{answer.family} is no family")
        elif isinstance(answer, tuple):
            for index, name in enumerate(answer):
                if _LOCAL_NAME.fullmatch(name) is None:
                    yield Problem(
                        f"{key}[{index}]", f"{name!r} is not one word of letters, digits and _"
                    )


def _declared_branches(mapping: Mapping) -> Iterator[Problem]:
    vocabulary = _vocabulary()
    for name, branch in mapping.branches.items():
        path = f"branches.{name}"
        if name in vocabulary:
            yield Problem(path, "is a vocabulary class already")
            continue
        if branch.parent not in vocabulary and branch.parent not in mapping.branches:
            yield Problem(f"{path}.parent", _unknown_class(branch.parent))
            continue
        chain: list[str] = []
        parent = branch.parent
        while parent in mapping.branches and parent not in chain and parent != name:
            chain.append(parent)
            parent = mapping.branches[parent].parent
        if parent == name:
            yield Problem(f"{path}.parent", f"{name} extends itself through {' -> '.join(chain)}")


def _family_field(mapping: Mapping, family: str, name: str) -> str | None:
    """Say why ``family.name`` is no field the mapping describes, else ``None``."""
    if family not in mapping.families:
        return f"{family} is no family"
    if name not in mapping.families[family].fields:
        return f"{family} has no field {name}"
    return None


def _direction_fields(mapping: Mapping) -> Iterator[Problem]:
    for key in mapping.directions:
        family, _, name = key.partition(".")
        why = _family_field(mapping, family, name)
        if why is not None:
            yield Problem(f"directions.{key}", why)


def _wiring_fields(mapping: Mapping) -> Iterator[Problem]:
    for raw, model in mapping.models.items():
        for family, wiring in model.wiring.items():
            entry = f"models.{raw}.wiring.{family}"
            if family not in mapping.families:
                yield Problem(entry, f"{family} is no family")
            elif wiring.element_field is not None:
                why = _family_field(mapping, family, wiring.element_field)
                if why is not None:
                    yield Problem(f"{entry}.element_field", why)


def _judgment_families(mapping: Mapping) -> Iterator[Problem]:
    for family in mapping.judgments:
        if family not in mapping.families:
            yield Problem(f"judgments.{family}", f"{family} is no family")


_RULES: tuple[Callable[[Mapping], Iterator[Problem]], ...] = (
    _identity_code,
    _model_names,
    _section_order,
    _family_tokens,
    _family_classes,
    _family_devices,
    _declared_branches,
    _direction_fields,
    _wiring_fields,
    _judgment_families,
)


def _created_fields(mapping: Mapping) -> set[str]:
    """The ``<family>.<field>`` keys a ``field:`` judgment answer creates."""
    return {
        f"{family}.{answer.name}"
        for family, judgments in mapping.judgments.items()
        for answers in judgments.rows_beyond.values()
        for answer in answers.values()
        if isinstance(answer, FieldAnswer)
    }


def _against_export(mapping: Mapping, ao: dict) -> Iterator[Problem]:
    from osprey.services.mml.directions import vote_directions
    from osprey.services.mml.family import family_views, system_bodies

    systems = dict(system_bodies(ao))
    views = {
        raw: {view.raw_name: view for view in family_views(raw, body)}
        for raw, body in systems.items()
    }
    for raw in mapping.models:
        if raw not in systems:
            yield Problem(f"models.{raw}", f"{raw} is no exported system")
    for raw in systems:
        if raw not in mapping.models:
            yield Problem("models", f"leaves out the exported system {raw}")

    exported: dict[str, set[str]] = {}
    for carried_by in views.values():
        for family, found in carried_by.items():
            exported.setdefault(family, set()).update(found.fields)
    for family in mapping.families:
        if family not in exported:
            yield Problem(f"families.{family}", f"{family} is no exported family")
    for family in exported:
        if family not in mapping.families:
            yield Problem("families", f"leaves out the exported family {family}")

    created = _created_fields(mapping)
    carried = {f"{family}.{name}" for family, names in exported.items() for name in names}
    for key in mapping.directions:
        if key not in carried and key not in created:
            yield Problem(f"directions.{key}", f"the export carries no channels under {key}")
    for key in sorted(carried - set(mapping.directions)):
        yield Problem("directions", f"{key} carries channels and has no direction")

    for raw, model in mapping.models.items():
        system = views.get(raw)
        if system is None:
            continue
        for family, wiring in model.wiring.items():
            entry = f"models.{raw}.wiring.{family}"
            view = system.get(family)
            if view is None:
                yield Problem(entry, f"{raw} carries no family {family}")
            elif wiring.element_field is not None and wiring.element_field not in view.fields:
                yield Problem(
                    f"{entry}.element_field",
                    f"{raw} carries no channels under {family}.{wiring.element_field}",
                )

    votes = vote_directions(ao)
    for key, direction in mapping.directions.items():
        family, _, name = key.partition(".")
        vote = votes.get((family, name))
        if (
            vote is None
            or vote.direction is None
            or direction.direction is None
            or direction.override
            or direction.provenance != "stated"
            or vote.direction == direction.direction
        ):
            continue
        yield Problem(
            f"directions.{key}",
            f"stated {direction.direction}, the export votes {vote.direction}; "
            "set override: true to keep it",
        )

    yield from _judgment_problems(mapping, _pending_by_family(ao))


# -- judgment answers against the export ---------------------------------------


def _pending_by_family(ao: dict) -> dict[str, list[Any]]:
    """What each raw family asks its reviewer, one entry per system carrying it.

    The judgments are read off the raw export, before any answer is applied,
    in export order; a family pending nothing is listed too.
    """
    from osprey.services.mml.judgments import all_pending_judgments

    found: dict[str, list[Any]] = {}
    for judged in all_pending_judgments(ao).values():
        found.setdefault(judged.family, []).append(judged)
    return found


def _answer_kind(answer: RowAnswer) -> str:
    """The word a problem calls this kind of row answer by."""
    return "field:" if isinstance(answer, FieldAnswer) else answer


def _unpended_answers(raw: str, judgments: FamilyJudgments, found: list[Any]) -> Iterator[Problem]:
    """Refuse every row and device answer no system of the export pends."""
    rows = {(row.field, row.signal) for judged in found for row in judged.rows_beyond}
    for name, answers in judgments.rows_beyond.items():
        for signal, answer in answers.items():
            if answer is not None and (name, signal) not in rows:
                yield Problem(
                    judgment_key(raw, ROWS_BEYOND_KIND, name, signal),
                    f"names no row beyond the devices of {raw} in any system",
                )
    ordinals = {ordinal for judged in found for ordinal in judged.unbound_devices}
    for ordinal, unbound in judgments.unbound_devices.items():
        if unbound is not None and ordinal not in ordinals:
            yield Problem(
                judgment_key(raw, UNBOUND_KIND, ordinal=ordinal),
                f"names no unbound device of {raw} in any system",
            )


def _group_collisions(raw: str, found: list[Any]) -> Iterator[Problem]:
    """Refuse an owner map whose group keys cannot say which devices they own.

    A group is keyed by its lowest ordinal, so two groups sharing a device in
    one system, or one ordinal keying different members in two systems, leave
    a key naming no single group. ``keep_all`` needs no key.
    """
    from itertools import combinations

    key = judgment_key(raw, SHARED_KIND)
    for judged in found:
        for first, second in combinations(judged.groups, 2):
            if shared := sorted(set(first.ordinals) & set(second.ordinals)):
                yield Problem(
                    key,
                    f"supply groups {first.lowest} and {second.lowest} of {raw} in "
                    f"{judged.system} share device {shared[0]}; answer `keep_all`",
                )
    members: dict[int, tuple[str, tuple[int, ...]]] = {}
    for judged in found:
        for group in judged.groups:
            system, ordinals = members.setdefault(group.lowest, (judged.system, group.ordinals))
            if ordinals != group.ordinals:
                yield Problem(
                    key,
                    f"supply group {group.lowest} of {raw} has the members "
                    f"{list(ordinals)} in {system} and {list(group.ordinals)} in "
                    f"{judged.system}; answer `keep_all`",
                )


def _owner_problems(raw: str, judgments: FamilyJudgments, found: list[Any]) -> Iterator[Problem]:
    """Refuse an owner map naming a group or an owner the export does not have.

    ``keep_all`` keeps every channel where the export put it, so no export
    refuses it. An owner map names one member of every supply group; a
    collision refuses the whole map at the family's key.
    """
    answer = judgments.shared_pvs
    if not isinstance(answer, OwnerMap):
        return
    groups = [(judged, group) for judged in found for group in judged.groups]
    if not groups:
        yield Problem(
            judgment_key(raw, SHARED_KIND), f"names no shared supply of {raw} in any system"
        )
        return
    collision = next(_group_collisions(raw, found), None)
    if collision is not None:
        yield collision
        return
    keyed = {group.lowest for _, group in groups}
    for ordinal in answer.owners:
        if ordinal not in keyed:
            yield Problem(
                judgment_key(raw, SHARED_KIND, ordinal=ordinal),
                f"names no supply group of {raw} in any system",
            )
    for judged, group in groups:
        slot = judgment_key(raw, SHARED_KIND, ordinal=group.lowest)
        owner = answer.owners.get(group.lowest)
        where = f"{raw} in {judged.system}"
        if owner is None:
            yield Problem(slot, f"supply group {group.lowest} of {where} has no owner")
        elif owner != "keep_all" and owner not in group.ordinals:
            yield Problem(
                slot, f"device {owner} is not a member of supply group {group.lowest} of {where}"
            )


def _answered_rows(
    raw: str, judgments: FamilyJudgments, judged: Any
) -> list[tuple[str, Any, RowAnswer]]:
    """The rows one system pends that the document answers, in document order."""
    rows = {(row.field, row.signal): row for row in judged.rows_beyond}
    answered: list[tuple[str, Any, RowAnswer]] = []
    for name, answers in judgments.rows_beyond.items():
        for signal, answer in answers.items():
            row = rows.get((name, signal))
            if row is not None and answer is not None:
                answered.append((judgment_key(raw, ROWS_BEYOND_KIND, name, signal), row, answer))
    return answered


def _row_problems(raw: str, judgments: FamilyJudgments, judged: Any) -> Iterator[Problem]:
    """Refuse the row answers one system's export cannot carry.

    A row answered ``device`` must not already be bound below the family's
    devices, or it would mint a supply group nobody answered; a ``field:``
    name must be PN_LOCAL, must not collide with a key of the family body or
    another row's field, and every channel key of its row is answered alike.
    """
    answered = _answered_rows(raw, judgments, judged)
    kinds: dict[tuple[str, int], set[str]] = {}
    for _, row, answer in answered:
        kinds.setdefault((row.field, row.index), set()).add(_answer_kind(answer))
    named: dict[str, tuple[str, int]] = {}
    where = f"{raw} in {judged.system}"
    for key, row, answer in answered:
        if answer == "device":
            if row.signal in judged.bound_below:
                yield Problem(
                    key,
                    f"{row.signal!r} is also bound below device {judged.n_devices + 1} of "
                    f"{where}; answer `drop` or `field:`",
                )
            continue
        if not isinstance(answer, FieldAnswer):
            continue
        name = answer.name
        if not _pn_local(name):
            yield Problem(key, f"the field name {name!r} for {where} is not PN_LOCAL")
        elif name in judged.body_keys:
            yield Problem(
                key, f"the field name {name!r} is a key {raw} already carries in {judged.system}"
            )
        elif named.setdefault(name, (row.field, row.index)) != (row.field, row.index):
            yield Problem(key, f"the field name {name!r} is answered on another row of {where}")
        elif others := sorted(kinds[(row.field, row.index)] - {"field:"}):
            yield Problem(
                key, f"the other channel key of this row is answered {others[0]} in {judged.system}"
            )


def _created_field_problems(
    raw: str, judgments: FamilyJudgments, found: list[Any], mapping: Mapping, refused: set[str]
) -> Iterator[Problem]:
    """Ask for the family field and direction every accepted ``field:`` answer creates."""
    family = mapping.families.get(raw)
    for judged in found:
        for key, _, answer in _answered_rows(raw, judgments, judged):
            if key in refused or not isinstance(answer, FieldAnswer):
                continue
            described = family is not None and answer.name in family.fields
            if not described or f"{raw}.{answer.name}" not in mapping.directions:
                yield Problem(
                    key,
                    f"creates the field {answer.name!r} of {raw} in {judged.system}; add "
                    f"families.{raw}.fields.{answer.name} and directions.{raw}.{answer.name}",
                )


def _missing_slots(
    raw: str, judgments: FamilyJudgments | None, found: list[Any]
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(key, system, what to write)`` per pending judgment the document leaves out."""
    rows = {} if judgments is None else judgments.rows_beyond
    ordinals = {} if judgments is None else judgments.unbound_devices
    shared = judgments is not None and judgments.shared_pvs_present
    for judged in found:
        for row in judged.rows_beyond:
            if row.signal not in rows.get(row.field, {}):
                key = judgment_key(raw, ROWS_BEYOND_KIND, row.field, row.signal)
                yield key, judged.system, _ANSWER_ROW
        for ordinal in judged.unbound_devices:
            if ordinal not in ordinals:
                key = judgment_key(raw, UNBOUND_KIND, ordinal=ordinal)
                yield key, judged.system, _ANSWER_UNBOUND
        if judged.groups and not shared:
            yield judgment_key(raw, SHARED_KIND), judged.system, _ANSWER_SHARED


def _all_missing_slots(
    mapping: Mapping, pending: dict[str, list[Any]]
) -> list[tuple[str, str, str]]:
    """Every pending judgment the document leaves out, once per key, in export order."""
    found: dict[str, tuple[str, str, str]] = {}
    for raw, carried in pending.items():
        for entry in _missing_slots(raw, mapping.judgments.get(raw), carried):
            found.setdefault(entry[0], entry)
    return list(found.values())


def _judgment_problems(mapping: Mapping, pending: dict[str, list[Any]]) -> list[Problem]:
    """Hold every judgment answer to what the export pends.

    Per family, in document order: answers the export cannot carry, then the
    mapping entries an accepted ``field:`` answer needs; after them, every
    pending judgment the document leaves no slot for. A key is reported once,
    at the first system that refuses it. A family the export does not carry is
    left to the family rules.
    """
    found: dict[str, Problem] = {}
    for raw, judgments in mapping.judgments.items():
        carried = pending.get(raw)
        if carried is None:
            continue
        refused = [
            *_unpended_answers(raw, judgments, carried),
            *_owner_problems(raw, judgments, carried),
            *(problem for judged in carried for problem in _row_problems(raw, judgments, judged)),
        ]
        created = _created_field_problems(
            raw, judgments, carried, mapping, {problem.key for problem in refused}
        )
        for problem in (*refused, *created):
            found.setdefault(problem.key, problem)
    for key, system, _ in _all_missing_slots(mapping, pending):
        found.setdefault(key, Problem(key, f"is pending in {system} and has no answer"))
    return list(found.values())


def check_mapping(mapping: Mapping, ao: dict | None = None) -> list[Problem]:
    """Check what a parsed mapping means, alone and against its export.

    Undecided slots are not problems here; :func:`require_decided` stops on
    them.

    Args:
        mapping: The parsed mapping.
        ao: The merged export, ``{system: {family: body}}`` plus ``_``-prefixed
            bookkeeping keys; ``None`` checks the mapping alone.

    Returns:
        Every problem, in rule order and document order within a rule.
    """
    problems = [problem for rule in _RULES for problem in rule(mapping)]
    if ao is not None:
        problems.extend(_against_export(mapping, ao))
    return problems


# -- draft --------------------------------------------------------------------

#: Provenance of prose and directions the draft derives from export facts.
DERIVED = "derived"

#: Provenance of a description carried over from the export as written.
IMPORTED = "imported"

#: What each lattice type token drives, in pyAT's words, keyed by the token
#: folded to lower case. A token absent here -- the dipole string among them,
#: whose energy knob is the reviewer's call -- is proposed with a null engine.
ENGINE_BY_TYPE: dict[str, dict[str, Any]] = {
    **dict.fromkeys(("quad", "k", "quadrupole"), {"attribute": "PolynomB", "index": 1}),
    **dict.fromkeys(("sext", "k2", "sextupole"), {"attribute": "PolynomB", "index": 2}),
    **dict.fromkeys(("octu", "k3"), {"attribute": "PolynomB", "index": 3}),
    **dict.fromkeys(("skewquad", "ks", "ks1", "skewq"), {"attribute": "PolynomA", "index": 1}),
    **dict.fromkeys(("hcm", "hcor"), {"attribute": "KickAngle", "index": 0}),
    **dict.fromkeys(("vcm", "vcor"), {"attribute": "KickAngle", "index": 1}),
    **dict.fromkeys(("rf", "rf cavity"), {"attribute": "Frequency"}),
    **dict.fromkeys(("x", "bpmx", "xturns"), {"axis": "x"}),
    **dict.fromkeys(("y", "bpmy", "yturns"), {"axis": "y"}),
}

#: The fields a family is wired through, in preference order: a family that
#: sets something is wired through what it sets, one that only reads through
#: what it reads.
_WIRED_FIELDS = (SETPOINT_FIELD, MONITOR_FIELD)


def _text(value: Any) -> str | None:
    """Return stripped text when ``value`` is non-blank text, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _one_unit(value: Any) -> str | None:
    """The one unit ``HWUnits`` states, scalar or agreeing per device, else ``None``."""
    if isinstance(value, (list, tuple)):
        distinct = {unit for item in value if (unit := _text(item)) is not None}
        return distinct.pop() if len(distinct) == 1 else None
    return _text(value)


def _system_order(ao: dict) -> list[str]:
    """The export's systems: ``_import_order`` first, then the rest sorted."""
    from osprey.services.mml.family import system_bodies

    present = sorted(raw for raw, _ in system_bodies(ao))
    order = ao.get("_import_order")
    listed = [raw for raw in order if raw in present] if isinstance(order, list) else []
    return list(dict.fromkeys([*listed, *present]))


def _block(blocks: dict | None, system: str) -> dict:
    block = blocks.get(system) if isinstance(blocks, dict) else None
    return block if isinstance(block, dict) else {}


def _draft_identity(ad: dict | None, systems: list[str]) -> dict | None:
    from osprey.facility import fold_code

    machine = next(
        (found for raw in systems if (found := _text(_block(ad, raw).get("Machine")))), None
    )
    if machine is None:
        return None
    return {
        "code": fold_code(machine),
        "name": machine,
        "description": f"The {machine} accelerator facility, as exported by its MATLAB Middle Layer.",
    }


def _model_prose(raw: str, body: dict, ad_block: dict) -> tuple[str | None, str]:
    native = _text(body.get("_description"))
    if native is not None:
        return native, IMPORTED
    machine = _text(ad_block.get("Machine"))
    sub = _text(ad_block.get("SubMachine"))
    mode = _text(ad_block.get("OperationalMode"))
    if machine is None and sub is None and mode is None:
        return None, DERIVED
    prose = f"The {sub or raw} system"
    if machine is not None:
        prose += f" of {machine}"
    prose += "."
    if mode is not None:
        prose += f" Operational mode: {mode}."
    return prose, DERIVED


def _family_prose(raw: str, views: list[Any]) -> str:
    systems = ", ".join(view.system for view in views)
    devices = sum(view.n_devices for view in views)
    channels = sum(view.channel_count for view in views)
    names = list(dict.fromkeys(name for view in views for name in view.fields))
    prose = (
        f"MML family {raw} in {systems}: {_plural(devices, 'device')}, "
        f"{_plural(channels, 'channel')}"
    )
    if names:
        prose += f" across fields {', '.join(names)}"
    return prose + "."


def _field_prose(name: str, views: list[Any]) -> str:
    found = [view.fields[name] for view in views if name in view.fields]
    keys = list(dict.fromkeys(key for fld in found for key in fld.keys))
    channels = sum(fld.channel_count for fld in found)
    prose = f"Field {name}: {_plural(channels, 'channel')} via {', '.join(keys)}"
    if any(fld.broadcast for fld in found):
        prose += ", one channel broadcast to every device"
    units = list(
        dict.fromkeys(unit for fld in found if (unit := _one_unit(fld.body.get("HWUnits"))))
    )
    if len(units) == 1:
        prose += f", hardware units {units[0]}"
    return prose + "."


def _draft_family(raw: str, views: list[Any], devices: Any = None) -> dict[str, Any]:
    channels = sum(view.channel_count for view in views)
    native = next((view.description for view in views if view.description is not None), None)
    entry: dict[str, Any] = {}
    if channels > 0:
        entry["branch"] = None
        entry["class"] = raw
        entry["devices"] = devices
    entry["aliases"] = [raw]
    if native is not None:
        entry["description"], entry["provenance"] = native[0].strip(), IMPORTED
    else:
        entry["description"], entry["provenance"] = _family_prose(raw, views), DERIVED
    entry["channels"] = channels
    fields: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(name for view in views for name in view.fields):
        text = next(
            (
                found
                for view in views
                if name in view.fields and (found := _text(view.fields[name].description))
            ),
            None,
        )
        if text is not None:
            fields[name] = {"description": text, "provenance": IMPORTED}
        else:
            fields[name] = {"description": _field_prose(name, views), "provenance": DERIVED}
    entry["fields"] = fields
    return entry


def _draft_devices(
    views: dict[str, list[Any]], models: dict[str, str], imported: list[Any]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Propose how the devices of every family that carries a channel are identified.

    ``names`` where the export names every device of the family in every
    system carrying it; ``{same_as: <family>}`` where exactly one such family
    binds, in each of those systems, the other axis of the same addresses;
    else ``address``.

    Args:
        views: Every family's views, one per system carrying it, in import
            order.
        models: The model each system is drafted as, keyed by raw token, in
            import order.
        imported: Every view, in the order an import reads them.

    Returns:
        Each family's ``devices`` value as it is written, and the ids each
        ``address`` answer yields, both keyed by raw family.
    """
    from osprey.facility.layers.mml.identity import axis_twins, device_ids, stated_ids

    systems = list(models)
    carried = {
        raw: [view for view in found if view.channel_count > 0] for raw, found in views.items()
    }
    carried = {raw: found for raw, found in carried.items() if found}
    named = [
        raw
        for raw, found in carried.items()
        if all(None not in stated_ids(view, systems, models) for view in found)
    ]

    def twin(raw: str, other: str) -> bool:
        theirs = {view.system: view for view in carried[other]}
        return all(
            view.system in theirs and axis_twins(view, theirs[view.system]) for view in carried[raw]
        )

    answers: dict[str, DeviceIdentity] = {}
    for raw in carried:
        if raw in named:
            answers[raw] = "names"
            continue
        twins = [other for other in named if twin(raw, other)]
        answers[raw] = SameAs(twins[0]) if len(twins) == 1 else "address"

    ordered = [view for view in imported if view.channel_count > 0]
    yielded: dict[str, list[str]] = {}
    for view, ids in zip(ordered, device_ids(ordered, models, answers), strict=True):
        if answers[view.raw_name] == "address":
            yielded.setdefault(view.raw_name, []).extend(ids)
    written = {
        raw: {"same_as": answer.family} if isinstance(answer, SameAs) else answer
        for raw, answer in answers.items()
    }
    return written, yielded


def _draft_judgments(views: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Every family's pending judgment slots, unioned over the systems carrying it."""
    from osprey.services.mml.judgments import pending_judgments

    block: dict[str, dict[str, Any]] = {}
    for raw, carried in views.items():
        pending = [pending_judgments(view) for view in carried]
        rows: dict[str, dict[str, None]] = {}
        for item in pending:
            for row in item.rows_beyond:
                rows.setdefault(row.field, {})[row.signal] = None
        ordinals = sorted({ordinal for item in pending for ordinal in item.unbound_devices})
        entry: dict[str, Any] = {}
        if rows:
            entry[ROWS_BEYOND_KIND] = rows
        if ordinals:
            entry[UNBOUND_KIND] = dict.fromkeys(ordinals)
        if any(item.groups for item in pending):
            entry[SHARED_KIND] = None
        if entry:
            block[raw] = entry
    return block


def _bound_type(view: Any) -> str | None:
    """The lattice type a family binds, folded to lower case; ``None`` when it binds none."""
    lattice = view.body.get("AT")
    if not isinstance(lattice, dict):
        return None
    indices = lattice.get("ATIndex")
    if indices is None or (isinstance(indices, (list, tuple)) and not indices):
        return None
    token = _text(lattice.get("ATType"))
    return token.lower() if token is not None else ""


def _wired_field(view: Any, engine: dict[str, Any] | None, facts: dict) -> str | None:
    if engine is not None and "axis" in engine:
        return MONITOR_FIELD if MONITOR_FIELD in view.fields else None
    nominals = facts.get("nominals")
    stated = nominals[0] if isinstance(nominals, list) and nominals else None
    if isinstance(stated, str) and stated in view.fields:
        return stated
    return next((name for name in _WIRED_FIELDS if name in view.fields), None)


def _calibration_kind(facts: dict, name: str) -> str | None:
    calibration = _block(facts, name).get("calibration")
    kind = calibration.get("kind") if isinstance(calibration, dict) else None
    return kind if kind in CALIBRATION_KINDS else None


def _draft_wiring(views: list[Any], va_block: dict) -> dict[str, dict[str, Any]]:
    """Propose one system's wiring from the lattice types its families bind.

    A family the model's sampled facts do not cover, or that binds no element,
    is not proposed; one whose type the table does not know is proposed with a
    null engine for the reviewer to answer or delete.
    """
    sampled = va_block.get("families")
    if not isinstance(sampled, dict):
        return {}
    wiring: dict[str, dict[str, Any]] = {}
    for view in views:
        facts = sampled.get(view.raw_name)
        token = _bound_type(view)
        if not isinstance(facts, dict) or token is None:
            continue
        known = ENGINE_BY_TYPE.get(token)
        engine = dict(known) if known is not None else None
        name = _wired_field(view, engine, facts)
        if name is None:
            continue
        wiring[view.raw_name] = {
            "element_field": name,
            "engine": engine,
            "calibration": _calibration_kind(facts, name),
        }
    return wiring


def draft_mapping(ao: dict, ad: dict | None = None, va: dict | None = None) -> dict[str, Any]:
    """Build the draft mapping for a merged export.

    Every slot the export has a fact for is filled and the rest left ``null``:
    identity from ``AD.Machine`` (no block without it); one model per system,
    keyed by its raw token and named by it folded to PN_LOCAL; families merged
    across systems, each class pre-filled with the raw token and each
    ``devices`` proposed from the names and addresses the export states;
    directions from
    the export's own vote (``derived``); the judgment slots the export pends;
    and, for a system whose sampled model facts are given, the wiring its
    families' lattice types imply.

    Args:
        ao: ``{system: {family: body}}`` with optional ``_import_order``; not
            modified.
        ad: Accelerator data keyed by system, or ``None``.
        va: Each system's sampled model facts (the export's ``va.json``
            document), keyed by system, or ``None``.

    Returns:
        The document as plain dicts in the order it is written; it parses with
        :func:`parse_mapping`.
    """
    return _draft(ao, ad, va)[0]


def _draft(
    ao: dict, ad: dict | None, va: dict | None
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """The draft document and the ids each of its ``address`` answers yields."""
    from osprey.facility import fold_code
    from osprey.services.mml.directions import vote_directions
    from osprey.services.mml.family import family_views

    order = _system_order(ao)
    views: dict[str, list[Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    imported: list[Any] = []
    for raw in order:
        carried = list(family_views(raw, ao[raw]))
        imported.extend(carried)
        for view in carried:
            views.setdefault(view.raw_name, []).append(view)
        description, provenance = _model_prose(raw, ao[raw], _block(ad, raw))
        model: dict[str, Any] = {
            "name": raw if _pn_local(raw) else fold_code(raw),
            "description": description,
            "provenance": provenance,
        }
        wiring = _draft_wiring(carried, _block(va, raw))
        if wiring:
            model["wiring"] = wiring
        models[raw] = model

    names = {raw: model["name"] for raw, model in models.items()}
    devices, yielded = _draft_devices(views, names, imported)
    families = {
        raw: _draft_family(raw, carried, devices.get(raw)) for raw, carried in views.items()
    }
    votes = vote_directions(ao)
    directions: dict[str, dict[str, Any]] = {}
    for raw, entry in families.items():
        for name in entry["fields"]:
            vote = votes.get((raw, name))
            directions[f"{raw}.{name}"] = {
                "direction": None if vote is None else vote.direction,
                "provenance": DERIVED,
                "override": False,
            }

    document: dict[str, Any] = {}
    identity = _draft_identity(ad, order)
    if identity is not None:
        document["facility"] = identity
    document["models"] = models
    document["section_order"] = [model["name"] for model in models.values()]
    document["families"] = families
    document["directions"] = directions
    judgments = _draft_judgments(views)
    if judgments:
        document["judgments"] = judgments
    return document, yielded


#: The line of a family block that answers its devices by address.
_ADDRESS_LINE = "    devices: address"

#: The page width the mapping is written to.
_WIDTH = 100


def _annotate(text: str, yielded: Map[str, Sequence[str]]) -> str:
    """Write, under each family's ``devices: address`` line, the ids it yields."""
    import yaml

    lines: list[str] = []
    inside = False
    family: str | None = None
    for line in text.splitlines():
        lines.append(line)
        if not line.startswith((" ", "-")):
            inside = line == "families:"
        elif not inside:
            continue
        elif line.startswith("  ") and not line.startswith("   "):
            key = yaml.safe_load(line)
            family = str(next(iter(key))) if isinstance(key, dict) and len(key) == 1 else None
        elif line == _ADDRESS_LINE and family is not None and yielded.get(family):
            lines.extend(
                textwrap.wrap(
                    ", ".join(yielded[family]),
                    width=_WIDTH,
                    initial_indent="    # ",
                    subsequent_indent="    # ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(lines) + "\n"


def draft_text(ao: dict, ad: dict | None = None, va: dict | None = None) -> str:
    """The draft mapping as it is written to an absent mapping file.

    Args:
        ao: The merged export, as handed to :func:`draft_mapping`.
        ad: Accelerator data keyed by system, or ``None``.
        va: Each system's sampled model facts, keyed by system, or ``None``.

    Returns:
        :func:`draft_mapping`'s document as YAML, with a comment under every
        ``devices: address`` listing the device ids the answer yields.
    """
    document, yielded = _draft(ao, ad, va)
    return _annotate(dump_mapping(document), yielded)


def dump_mapping(document: dict[str, Any]) -> str:
    """Serialise a mapping document as block-style YAML in insertion order.

    Args:
        document: The document, e.g. from :func:`draft_mapping`.

    Returns:
        The YAML text; ``None`` is written ``null``.
    """
    import yaml

    return yaml.safe_dump(
        document, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )


def load_or_draft(
    facility_dir: Path, ao: dict, ad: dict | None = None, va: dict | None = None
) -> Mapping:
    """Load the layer's mapping, or write a draft and stop when there is none.

    Args:
        facility_dir: ``data/facility/``; the mapping is :data:`MAPPING_FILE`
            under it.
        ao: The merged export the draft is built from.
        ad: Accelerator data keyed by system, or ``None``.
        va: Each system's sampled model facts, keyed by system, or ``None``.

    Returns:
        The parsed mapping, every slot decided.

    Raises:
        ImportStop: ``mapping-draft`` after writing a draft to an absent file;
            ``mapping-undecided`` while any slot of a present file is ``null``.
        MappingError: The present file has the wrong structure.
    """
    path = facility_dir / MAPPING_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(draft_text(ao, ad, va), encoding="utf-8")
        raise ImportStop("mapping-draft", [f"{MAPPING_FILE} written; review it, then re-run"])
    mapping = read_mapping(path)
    require_decided(mapping, ao)
    return mapping
