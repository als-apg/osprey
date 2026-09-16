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
direction). Rejecting them is the semantic checker's job, which also owns
PN_LOCAL, permutation and cross-reference rules; keeping those out of here
lets a freshly written skeleton parse before anyone has filled it in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DIRECTION_VALUES",
    "Branch",
    "Direction",
    "Facility",
    "Family",
    "Field",
    "Mapping",
    "MappingError",
    "System",
    "parse_mapping",
]

#: The values a ``directions.*.direction`` slot may hold; ``None`` is undecided.
DIRECTION_VALUES: frozenset[str | None] = frozenset({"read", "write", None})


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


# -- blocks -------------------------------------------------------------------

_TOP_REQUIRED = frozenset({"facility", "systems", "section_order", "families", "directions"})
_TOP_OPTIONAL = frozenset({"branches"})
_FACILITY_KEYS = frozenset({"token", "title", "description", "provenance"})
_SYSTEM_KEYS = frozenset({"name", "description", "provenance"})
_BRANCH_KEYS = frozenset({"parent", "description"})
_FAMILY_REQUIRED = frozenset({"aliases", "description", "provenance", "channels", "fields"})
_FAMILY_OPTIONAL = frozenset({"rename", "branch", "class"})
_FIELD_KEYS = frozenset({"description", "provenance"})
_DIRECTION_REQUIRED = frozenset({"direction", "provenance"})
_DIRECTION_OPTIONAL = frozenset({"override"})
_NONE = frozenset()


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
    if isinstance(value, bool) or not isinstance(value, int):
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
    return Mapping(
        facility=_facility(data["facility"]),
        systems=_systems(data["systems"]),
        section_order=_str_list(data["section_order"], "section_order"),
        branches={} if branches is _MISSING else _branches(branches),
        families=_families(data["families"]),
        directions=_directions(data["directions"]),
    )
