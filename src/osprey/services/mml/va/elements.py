"""Where on the deck each coupled family writes, and what to call the element.

The Middle Layer addresses a lattice element by its one-based position in the
saved ring. The model addresses it by name, and a saved ring names a hundred
elements ``DR``: the join between the two is a renaming, not a lookup. This
module performs it. It takes the export block, the deck the export was
sampled over and the verdicts :mod:`~osprey.services.mml.va.verdicts` reached,
and returns the deck renamed plus the table saying, per family and per device,
which elements that device drives and what they are now called.

The renaming has one rule, and the rule is ownership. Several families reach
the same element -- a horizontal and a vertical corrector are one magnet, a
sextupole and the skew quadrupole wound on it are one body, both planes of a
beam monitor are one pickup -- and the element can carry only one name, so one
family owns it and the others address it by the owner's name while still
writing their own field of it. The owner is the family with the lowest rank in
:data:`OWNER_RANK`, ties broken by sorted family name: a monitor first, then
the normal multipole, the skew one, the corrector, the energy knob and the
cavity. The rank is not a preference between families; it is the order in
which a name tells a reader what the element *is*.

The name is ``<owner>_<sector>_<num>``, the owner's family token and the
device row the export lists it under, so an operator reading the emitted deck
finds the same device they would name at the console. A device split over
several elements adds ``_<n>``, the slot the piece was stated in -- counted
over the stated row, so a device missing its middle piece yields ``_1`` and
``_3`` rather than renumbering its remaining pieces into a lie. The energy
knob and the cavity take part in the ranking like any other family, so an
element only they reach is named for them too, whatever the bindings document
later makes of that binding.

Two things are refused rather than carried. A stated position outside the deck
is refused naming the family, the position and the ring length: the export and
the deck disagree, and every later step would compound it. Two elements that
would end up with one name are refused the same way -- names are what the
bindings document binds by, so a collision is a write landing on the wrong
magnet. An element no coupled family reaches keeps whatever the deck called
it, duplicates included; nothing addresses it.

The one element change beyond a name: a beam monitor that the deck saved as a
plain marker becomes an :class:`at.Monitor`, because a marker reads nothing.
Only a marker is converted -- an element with a length is renamed and left in
its class, since silently replacing it with a zero-length monitor would
shorten the ring.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.services.mml.family import device_rows
from osprey.services.mml.va.verdicts import is_cavity
from osprey.services.virtual_accelerator.bindings import ATTRIBUTES_BY_KIND

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from osprey.services.mml.mapping.schema import VAFamily

__all__ = [
    "OWNER_RANK",
    "Addressing",
    "ElementBinding",
    "ElementSlice",
    "address_elements",
]

#: What a family claims on an element, the strongest claim first. The claim a
#: strength makes is the polynomial it writes, so a normal multipole outranks
#: the skew one wound on the same body; every other kind claims by its kind.
OWNER_RANK: tuple[str, ...] = ("monitor", "PolynomB", "PolynomA", "KickAngle", "energy", "rf")

#: The attribute the cavity kind writes, as the bindings document spells it.
_RF_ATTRIBUTE = next(iter(ATTRIBUTES_BY_KIND["rf"]))


@dataclass(frozen=True)
class ElementSlice:
    """One element a device drives, and what the emitted deck calls it.

    Attributes:
        element: The name the element carries in the renamed ring.
        position: Its zero-based position in that ring.
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
    """What one device of one family drives, addressed in the emitted deck.

    Attributes:
        family: The family token the export keys the device under.
        kind: What the verdict decided the family drives.
        device: The sector and number the export lists the device under.
        attribute: The element attribute a write lands in, the axis a monitor
            reads, or ``None`` for the energy knob.
        index: The component of that attribute, or ``None`` when it has none.
        slices: Every element the device drives, in stated order.
    """

    family: str
    kind: str
    device: tuple[int, ...]
    attribute: str | None
    index: int | None
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
class Addressing:
    """The deck as the emit lane writes it, and the table that addresses it.

    Attributes:
        ring: The deck, renamed, with its markers converted. The deck the
            caller passed is left as it was.
        bindings: Every coupled family that drives an element, in export
            order, each with one entry per device that drives one.
        owners: The family each bound element is named after, keyed by its
            zero-based position in the ring.
    """

    ring: Any
    bindings: Mapping[str, tuple[ElementBinding, ...]]
    owners: Mapping[int, str]


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
    """One coupled family, read against the deck and waiting for its names."""

    family: str
    kind: str
    token: str
    attribute: str | None
    index: int | None
    rows: tuple[_Row, ...]


def address_elements(
    system_block: Mapping[str, Any],
    ring: Sequence[Any],
    verdicts: Mapping[str, VAFamily],
) -> Addressing:
    """Address every coupled family's elements and name them for their owner.

    Args:
        system_block: One system's virtual-accelerator export block, carrying
            its ``families``. Positions are read from the nominal block the
            family's verdict names, and device rows from its ``device_list``.
        ring: The deck the export was sampled over, every element kept and in
            saved order, as the Middle Layer's positions index it.
        verdicts: One verdict per family, as
            :func:`~osprey.services.mml.va.verdicts.propose` reached them. A
            family that does not couple binds nothing and is not addressed.

    Returns:
        The renamed deck, the per-device element table and the owner of each
        bound element.

    Raises:
        ValueError: A stated position lies outside the deck, a family states
            elements and no devices to name them after, it states more element
            rows than it has devices, a coupled family names no element field,
            or the renaming would give two elements one name.
    """
    claims = _claims(system_block, ring, verdicts)
    owners = _owners(claims)
    names = _names(claims, owners)

    renamed = copy.deepcopy(ring)
    kinds = {claim.family: claim.kind for claim in claims}
    for position, name in names.items():
        _rename(renamed, position, name, kinds[owners[position]])
    _refuse_collisions(renamed, names)

    return Addressing(
        ring=renamed,
        bindings={claim.family: _bindings(claim, names, owners) for claim in claims},
        owners=dict(owners),
    )


def _claims(
    system_block: Mapping[str, Any],
    ring: Sequence[Any],
    verdicts: Mapping[str, VAFamily],
) -> list[_Claim]:
    """Read every coupled family that drives an element, in export order."""
    families = system_block.get("families")
    if not isinstance(families, dict):
        return []

    claims: list[_Claim] = []
    for family, block in families.items():
        verdict = verdicts.get(family)
        if not isinstance(block, dict) or verdict is None or verdict.verdict != "couple":
            continue
        kind = verdict.kind
        if kind is None:
            continue
        attribute, index = _attribute_of(family, kind, verdict.element_field)
        stated = _stated_rows(family, block, verdict, ring)
        if not stated:
            continue
        rows = _with_devices(family, block, stated)
        if rows:
            claims.append(
                _Claim(
                    family=family,
                    kind=kind,
                    token=_rank_token(kind, attribute),
                    attribute=attribute,
                    index=index,
                    rows=rows,
                )
            )
    return claims


def _rank_token(kind: str, attribute: str | None) -> str:
    """What this family claims on an element, as :data:`OWNER_RANK` spells it.

    A strength claims the polynomial it writes, because the normal and the
    skew multipole of one body are ranked apart; every other kind claims by
    what it is, the axis a monitor reads and the cavity's frequency being one
    claim each rather than two.
    """
    return kind if kind in ("monitor", "energy", "rf") else str(attribute)


def _attribute_of(
    family: str, kind: str, element_field: str | None
) -> tuple[str | None, int | None]:
    """The attribute and component a family writes, from the field it binds."""
    if kind == "energy":
        return None, None
    if kind == "rf":
        return _RF_ATTRIBUTE, None
    if not element_field:
        raise ValueError(f"family {family} couples as {kind} and names no element field")

    attribute, _, tail = element_field.partition("[")
    index = int(tail[:-1]) if tail.endswith("]") else None
    if attribute not in ATTRIBUTES_BY_KIND.get(kind, frozenset()):
        raise ValueError(f"family {family} couples as {kind}, which writes no {attribute}")
    return attribute, index


def _stated_rows(
    family: str,
    block: Mapping[str, Any],
    verdict: VAFamily,
    ring: Sequence[Any],
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    """One row of slots per device, each slot a zero-based deck position.

    The cavity states its row as the deck's own cavities: the Middle Layer's
    index into them is not the model's, so the class decides, as the verdict
    did.
    """
    if verdict.kind == "rf":
        cavities = tuple(
            (slot, position)
            for slot, position in enumerate(
                (index for index, element in enumerate(ring) if is_cavity(element)), start=1
            )
        )
        return [(len(cavities), cavities)] if cavities else []

    rows: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for stated in _index_rows(_at_index(block, verdict)):
        slots = tuple(
            (slot, _position(family, value, ring))
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
    devices = device_rows(block.get("device_list"))
    if devices is None:
        raise ValueError(
            f"family {family} binds {len(stated)} element rows and lists no device to name them after"
        )
    if len(stated) > len(devices):
        raise ValueError(
            f"family {family} binds {len(stated)} element rows over {len(devices)} devices"
        )
    return tuple(
        _Row(device=tuple(device), width=width, slots=slots)
        for device, (width, slots) in zip(devices, stated, strict=False)
    )


def _owners(claims: list[_Claim]) -> dict[int, str]:
    """The family each bound element is named after, by rank then by name."""
    claimed: dict[int, tuple[int, str]] = {}
    for claim in claims:
        rank = OWNER_RANK.index(claim.token)
        for row in claim.rows:
            for _, position in row.slots:
                key = (rank, claim.family)
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
                device = "_".join(str(_whole(number)) for number in row.device)
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
            kind=claim.kind,
            device=row.device,
            attribute=claim.attribute,
            index=claim.index,
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


def _rename(ring: Any, position: int, name: str, kind: str) -> None:
    """Name one bound element, converting a marker a monitor family reads."""
    import at

    element = ring[position]
    if kind == "monitor" and _is_marker(element, at):
        ring[position] = at.Monitor(name)
        return
    element.FamName = name


def _is_marker(element: Any, at: Any) -> bool:
    """Whether an element reads nothing where a beam monitor is expected.

    A monitor already saved as one is left alone, and so is anything with a
    length: replacing a real element with a zero-length monitor would shorten
    the ring to make a name fit.
    """
    return (
        not isinstance(element, at.Monitor)
        and getattr(element, "Length", 0) == 0
        and getattr(element, "PassMethod", "") == "IdentityPass"
    )


def _refuse_collisions(ring: Sequence[Any], names: Mapping[int, str]) -> None:
    """Refuse a renaming that leaves two elements answering to one name."""
    seen: dict[str, list[int]] = {}
    for position, element in enumerate(ring):
        seen.setdefault(element.FamName, []).append(position)
    for name in sorted(set(names.values())):
        sharing = seen[name]
        if len(sharing) > 1:
            raise ValueError(
                f"the renamed ring carries {len(sharing)} elements named {name!r}, "
                f"at positions {', '.join(str(index + 1) for index in sharing)}"
            )


def _at_index(block: Mapping[str, Any], verdict: VAFamily) -> Any:
    """The positions a family states beside the nominal its verdict read."""
    nominals = block.get("nominals")
    if not isinstance(nominals, dict) or verdict.nominal_source is None:
        return None
    nominal = nominals.get(verdict.nominal_source)
    return nominal.get("at_index") if isinstance(nominal, dict) else None


def _position(family: str, value: Any, ring: Sequence[Any]) -> int:
    """One stated position, read against the deck and made zero-based."""
    position = _index(value)
    if position is None or not 1 <= position <= len(ring):
        raise ValueError(
            f"family {family} binds ATIndex {position} of a ring of {len(ring)} elements"
        )
    return position - 1


def _index_rows(value: Any) -> list[list]:
    """Read stated positions as one row per device, every slot kept.

    A bare number is the one row it is, and the slots an export writes as a
    not-a-number stay in the row so the ones beside them keep their number.
    """
    rows = value if isinstance(value, (list, tuple)) else [value]
    read: list[list] = []
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
    return int(number) if number is not None else 0


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
