"""Where on an imported deck each wired family writes, and what the element is called.

The Middle Layer addresses a deck element by its one-based position in the
saved deck. The model addresses it by name, and a saved deck names a hundred
elements ``DR``: the join between the two is a renaming, not a lookup. This
module performs it. It takes one model of the layer's mapping, the deck the
export saved for it and the export's sampled model facts (``<stem>.va.json``),
and returns the deck renamed plus the table saying, per family and per device,
which elements that device drives and what they are now called.

A family's elements are the positions its sampled facts state beside the
nominal of the field the mapping wires it through (``element_field``), one row
per device of its ``device_list``. The family whose engine attribute is
``Frequency`` states the deck's own cavities instead: the Middle Layer's index
into them is not the model's, so the class decides. A wired family whose facts
state no position binds nothing.

The renaming has one rule, and the rule is ownership. Several families reach
the same element -- a horizontal and a vertical corrector are one magnet, a
sextupole and the skew quadrupole wound on it are one body, both planes of a
beam monitor are one pickup -- and the element can carry only one name, so one
family owns it and the others address it by the owner's name while still
writing their own field of it. The owner is the family whose engine block
ranks lowest in :data:`OWNER_RANK`, ties broken by sorted family name: a
monitor reading an orbit axis first, then ``PolynomB``, ``PolynomA``,
``KickAngle`` and the cavity's ``Frequency``. The rank is not a preference
between families; it is the order in which a name tells a reader what the
element *is*.

The name is ``<owner>_<sector>_<num>``, the owner's family token and the device
row the export lists it under, so an operator reading the imported deck finds
the same device they would name at the console. A device split over several
elements adds ``_<n>``, the slot the piece was stated in -- counted over the
stated row, so a device missing its middle piece yields ``_1`` and ``_3``
rather than renumbering its remaining pieces.

Two things are refused rather than carried. A stated position outside the deck
is refused naming the family, the position and the deck's length: the export
and the deck disagree, and every later step would compound it. Two elements
that would end up with one name are refused the same way -- the model's wiring
binds by name, so a collision is a write landing on the wrong magnet. An
element no wired family reaches keeps whatever the deck called it, duplicates
included; nothing addresses it.

The element changes beyond a name are one rule read both ways. A beam monitor
that the deck saved as a plain marker becomes an ``at.Monitor``, because a
marker reads nothing; only a marker is converted -- an element with a length is
renamed and left in its class, since replacing it with a zero-length monitor
would shorten the deck. The other way round, a monitor no family reads whose
name the deck gives to another monitor becomes a plain marker of that same
name, because a reading is addressed by its element's name and a repeated one
addresses nothing; one carrying a length is refused instead. Both conversions
leave the deck as long as it was and carry over everything the element held
that neither class implies -- its apertures and the transformations it sits in.

pyAT is imported inside the functions that need it.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from osprey.facility.layers.mml.mapping import EngineBlock, Model

__all__ = [
    "CARRIED_FIELDS",
    "FREQUENCY",
    "MONITOR",
    "OWNER_RANK",
    "Addressing",
    "ElementBinding",
    "ElementSlice",
    "ServedMarker",
    "address_elements",
]

#: The rank token of a family whose engine block reads an orbit axis.
MONITOR = "monitor"

#: The engine attribute of the family that drives the deck's cavity.
FREQUENCY = "Frequency"

#: What a family claims on an element, the strongest claim first: a monitor by
#: the axis it reads, every other family by the engine attribute it writes, so
#: a normal multipole outranks the skew one wound on the same body.
OWNER_RANK: tuple[str, ...] = (MONITOR, "PolynomB", "PolynomA", "KickAngle", FREQUENCY)

#: What an element carries that neither the marker nor the monitor class
#: implies: the apertures the beam is lost against and the transformations the
#: element sits in. A conversion between the two classes takes them along; a
#: class carries them only where the deck stated them.
CARRIED_FIELDS: tuple[str, ...] = ("EApertures", "RApertures", "T1", "T2", "R1", "R2")


@dataclass(frozen=True)
class ElementSlice:
    """One element a device drives, and what the imported deck calls it.

    Attributes:
        element: The name the element carries in the renamed deck.
        position: Its zero-based position in that deck.
        slot: The one-based slot of the stated row this piece came from, which
            is the suffix of a split device's name and 1 for a whole one.
        owner: The family the element is named after, this one or another.
    """

    element: str
    position: int
    slot: int
    owner: str


@dataclass(frozen=True)
class ElementBinding:
    """What one device of one wired family drives, addressed in the imported deck.

    Attributes:
        family: The raw family token the mapping wires.
        device: The sector and number the export lists the device under.
        engine: The family's engine block, as the mapping states it.
        slices: Every element the device drives, in stated order.
    """

    family: str
    device: tuple[int, ...]
    engine: EngineBlock
    slices: tuple[ElementSlice, ...]

    @property
    def element(self) -> str:
        """The element a readback is read from: the device's first slice."""
        return self.slices[0].element

    @property
    def owner(self) -> str:
        """The family that element is named after."""
        return self.slices[0].owner


@dataclass(frozen=True)
class ServedMarker:
    """One name the deck carries as a marker rather than as a monitor.

    Attributes:
        name: The name the deck gives those elements, which they keep.
        elements: How many elements of that name became markers.
    """

    name: str
    elements: int


@dataclass(frozen=True)
class Addressing:
    """The renamed deck, and the table that addresses it.

    Attributes:
        deck: The deck, renamed, with its markers converted. The deck the
            caller passed is left as it was.
        bindings: Every wired family that drives an element, in mapping order,
            each with one entry per device that drives one.
        owners: The family each bound element is named after, keyed by its
            zero-based position in the deck.
        markers: The repeated monitor names nothing reads, carried as plain
            markers, one entry per name in name order.
        monitors: How many monitor-type elements the deck is left with.
    """

    deck: Any
    bindings: Mapping[str, tuple[ElementBinding, ...]]
    owners: Mapping[int, str]
    markers: tuple[ServedMarker, ...] = ()
    monitors: int = 0


@dataclass(frozen=True)
class _Row:
    """One device's stated elements: the slots it has, of the width it states.

    ``width`` counts the slots the export wrote, the empty ones included, so a
    device missing a piece keeps the numbering of the pieces it has.
    """

    device: tuple[int, ...]
    width: int
    slots: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Claim:
    """One wired family, read against the deck and waiting for its names."""

    family: str
    engine: EngineBlock
    token: str
    rows: tuple[_Row, ...]


def address_elements(model: Model, deck: Sequence[Any], va_block: Mapping[str, Any]) -> Addressing:
    """Address every wired family's elements and name them for their owner.

    Args:
        model: One model of the layer's mapping; its ``wiring`` names the
            families addressed, every slot decided.
        deck: The deck the export saved for the model, every element kept and
            in saved order, as the Middle Layer's positions index it.
        va_block: The model's sampled facts (``<stem>.va.json``), carrying
            each family's ``device_list`` and the ``at_index`` of its nominals.

    Returns:
        The renamed deck, the per-device element table, the owner of each
        bound element, the monitor names carried as plain markers and how
        many monitor-type elements the deck is left with.

    Raises:
        ValueError: A wired family has no engine block or one no element is
            ranked by, a stated position lies outside the deck, a family states
            elements and no devices to name them after or more element rows
            than it has devices, the renaming would give two elements one
            name, or a duplicated monitor no family reads does more than mark a
            position.
    """
    claims = _claims(model, va_block, deck)
    owners = _owners(claims)
    names = _names(claims, owners)

    renamed = copy.deepcopy(deck)
    tokens = {claim.family: claim.token for claim in claims}
    for position, name in names.items():
        _rename(renamed, position, name, tokens[owners[position]])
    _refuse_collisions(renamed, names)

    markers = _serve_as_markers(renamed, names)
    return Addressing(
        deck=renamed,
        bindings={claim.family: _bindings(claim, names, owners) for claim in claims},
        owners=dict(owners),
        markers=markers,
        monitors=_monitors(renamed),
    )


def _is_cavity(element: Any) -> bool:
    """Whether a deck element is a cavity, under every spelling of its class."""
    return "RFCavity" in (
        type(element).__name__,
        getattr(element, "Class", None),
        getattr(element, "tag", None),
    )


def _monitors(deck: Sequence[Any]) -> int:
    """How many monitor-type elements a deck carries."""
    import at

    return sum(1 for element in deck if isinstance(element, at.Monitor))


def _rank_token(family: str, engine: EngineBlock) -> str:
    """What a family claims on an element, as :data:`OWNER_RANK` spells it."""
    if engine.axis is not None:
        return MONITOR
    if engine.attribute in OWNER_RANK:
        return engine.attribute
    ranked = ", ".join(OWNER_RANK[1:-1])
    raise ValueError(
        f"family {family} drives {engine.attribute}; wire it to an axis or to "
        f"{ranked} or {FREQUENCY}"
    )


def _claims(model: Model, va_block: Mapping[str, Any], deck: Sequence[Any]) -> list[_Claim]:
    """Read every wired family that drives an element, in mapping order."""
    sampled = va_block.get("families")
    families = sampled if isinstance(sampled, dict) else {}
    claims: list[_Claim] = []
    for family, wiring in model.wiring.items():
        engine = wiring.engine
        if engine is None:
            raise ValueError(f"family {family} is wired with no engine block")
        token = _rank_token(family, engine)
        block = families.get(family)
        if not isinstance(block, dict):
            continue
        stated = _stated_rows(family, block, wiring.element_field, token, deck)
        if not stated:
            continue
        rows = _with_devices(family, block, stated)
        if rows:
            claims.append(_Claim(family=family, engine=engine, token=token, rows=rows))
    return claims


def _stated_rows(
    family: str,
    block: Mapping[str, Any],
    element_field: str | None,
    token: str,
    deck: Sequence[Any],
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    """One row of slots per device, each slot a zero-based deck position."""
    if token == FREQUENCY:
        cavities = tuple(
            enumerate((index for index, element in enumerate(deck) if _is_cavity(element)), 1)
        )
        return [(len(cavities), cavities)] if cavities else []

    rows: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for stated in _index_rows(_at_index(block, element_field)):
        slots = tuple(
            (slot, _position(family, value, deck))
            for slot, value in enumerate(stated, start=1)
            if _index(value) is not None
        )
        if slots:
            rows.append((len(stated), slots))
    return rows


def _with_devices(
    family: str,
    block: Mapping[str, Any],
    stated: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> tuple[_Row, ...]:
    """Pair each stated row with the device the export lists it under."""
    from osprey.services.mml.family import device_rows

    devices = device_rows(block.get("device_list"))
    if devices is None:
        raise ValueError(
            f"family {family} binds {len(stated)} element rows and lists no device "
            "to name them after"
        )
    if len(stated) > len(devices):
        raise ValueError(
            f"family {family} binds {len(stated)} element rows over {len(devices)} devices"
        )
    return tuple(
        _Row(device=tuple(_whole(number) for number in device), width=width, slots=slots)
        for device, (width, slots) in zip(devices, stated, strict=False)
    )


def _owners(claims: list[_Claim]) -> dict[int, str]:
    """The family each bound element is named after, by rank then by name."""
    claimed: dict[int, tuple[int, str]] = {}
    for claim in claims:
        key = (OWNER_RANK.index(claim.token), claim.family)
        for row in claim.rows:
            for _, position in row.slots:
                if claimed.get(position, key) >= key:
                    claimed[position] = key
    return {position: family for position, (_, family) in claimed.items()}


def _names(claims: list[_Claim], owners: Mapping[int, str]) -> dict[int, str]:
    """The name each bound element carries, written by the family that owns it."""
    names: dict[int, str] = {}
    for claim in claims:
        for row in claim.rows:
            for slot, position in row.slots:
                if owners[position] != claim.family or position in names:
                    continue
                device = "_".join(str(number) for number in row.device)
                suffix = f"_{slot}" if row.width > 1 else ""
                names[position] = f"{claim.family}_{device}{suffix}"
    return names


def _bindings(
    claim: _Claim,
    names: Mapping[int, str],
    owners: Mapping[int, str],
) -> tuple[ElementBinding, ...]:
    """One entry per device of one family, in the order the export lists them."""
    return tuple(
        ElementBinding(
            family=claim.family,
            device=row.device,
            engine=claim.engine,
            slices=tuple(
                ElementSlice(
                    element=names[position],
                    position=position,
                    slot=slot,
                    owner=owners[position],
                )
                for slot, position in row.slots
            ),
        )
        for row in claim.rows
    )


def _rename(deck: Any, position: int, name: str, token: str) -> None:
    """Name one bound element, converting a marker a monitor family reads."""
    import at

    element = deck[position]
    if token == MONITOR and not isinstance(element, at.Monitor) and _is_plain(element):
        deck[position] = _carried(element, at.Monitor(name))
        return
    element.FamName = name


def _is_plain(element: Any) -> bool:
    """Whether an element occupies no space and leaves the beam untouched.

    Such an element is a marker in everything but its class, so each
    conversion between the marker and the monitor class moves only the class.
    """
    return (
        getattr(element, "Length", 0) == 0 and getattr(element, "PassMethod", "") == "IdentityPass"
    )


def _carried(element: Any, built: Any) -> Any:
    """Return ``built`` holding everything ``element`` carried beyond its class."""
    for name in CARRIED_FIELDS:
        value = getattr(element, name, None)
        if value is not None:
            setattr(built, name, copy.deepcopy(value))
    return built


def _not_plain(element: Any) -> str:
    """Say what an element does beyond marking a position, in one clause."""
    length = getattr(element, "Length", 0)
    if length:
        return f"is {length:g} m long"
    return f"passes the beam as {getattr(element, 'PassMethod', '')!r}"


def _serve_as_markers(deck: Any, names: Mapping[int, str]) -> tuple[ServedMarker, ...]:
    """Carry a repeated monitor no family reads as a plain marker of its name.

    A reading is addressed by the name of the element it is read from, so a
    deck cannot repeat a monitor's name. A facility marks structure it never
    reads -- the ends of a girder, a straight -- with the monitor type and one
    name for all of them; each becomes a marker, in place, under the deck's own
    name. Only an element no family reaches is converted, only where the name
    is shared, and only one that marks a position and does nothing else.

    Args:
        deck: The renamed deck, altered in place.
        names: The name each bound element carries, keyed by position.

    Returns:
        One entry per converted name, in name order.

    Raises:
        ValueError: A duplicated monitor no family reads does more than mark
            a position.
    """
    import at

    monitors = Counter(element.FamName for element in deck if isinstance(element, at.Monitor))
    converted: Counter[str] = Counter()
    for position, element in enumerate(deck):
        name = element.FamName
        if position in names or not isinstance(element, at.Monitor) or monitors[name] < 2:
            continue
        if not _is_plain(element):
            raise ValueError(
                f"the deck carries {monitors[name]} monitor-type elements named {name!r} "
                f"that no family reads, and the one at position {position + 1} "
                f"{_not_plain(element)}; wire it to a family or give it a unique name"
            )
        deck[position] = _carried(element, at.Marker(name))
        converted[name] += 1
    return tuple(
        ServedMarker(name=name, elements=count) for name, count in sorted(converted.items())
    )


def _refuse_collisions(deck: Sequence[Any], names: Mapping[int, str]) -> None:
    """Refuse a renaming that leaves two elements answering to one name."""
    seen: dict[str, list[int]] = {}
    for position, element in enumerate(deck):
        seen.setdefault(element.FamName, []).append(position)
    for name in sorted(set(names.values())):
        sharing = seen[name]
        if len(sharing) > 1:
            raise ValueError(
                f"the renamed deck carries {len(sharing)} elements named {name!r}, "
                f"at positions {', '.join(str(index + 1) for index in sharing)}"
            )


def _at_index(block: Mapping[str, Any], element_field: str | None) -> Any:
    """The positions a family states beside the nominal of its wired field."""
    nominals = block.get("nominals")
    if not isinstance(nominals, dict) or element_field is None:
        return None
    nominal = nominals.get(element_field)
    return nominal.get("at_index") if isinstance(nominal, dict) else None


def _position(family: str, value: Any, deck: Sequence[Any]) -> int:
    """One stated position, read against the deck and made zero-based."""
    position = _index(value)
    if position is None or not 1 <= position <= len(deck):
        raise ValueError(
            f"family {family} binds ATIndex {position} of a deck of {len(deck)} elements"
        )
    return position - 1


def _index_rows(value: Any) -> list[list[Any]]:
    """Read stated positions as one row per device, every slot kept.

    A bare number is the one row it is, and the slots an export writes as a
    not-a-number stay in the row so the ones beside them keep their number.
    """
    rows = value if isinstance(value, (list, tuple)) else [value]
    read: list[list[Any]] = []
    for row in rows:
        slots = list(row) if isinstance(row, (list, tuple)) else [row]
        if any(_index(slot) is not None for slot in slots):
            read.append(slots)
    return read


def _index(value: Any) -> int | None:
    """One stated position, or ``None`` for a slot the device does not have."""
    number = _number(value)
    if number is None or not math.isfinite(number) or number != int(number):
        return None
    return int(number)


def _whole(value: Any) -> int:
    """One device number, as the name spells it."""
    number = _number(value)
    return int(number) if number is not None and math.isfinite(number) else 0


def _number(value: Any) -> float | None:
    """One exported number, which an export may spell as a word."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
