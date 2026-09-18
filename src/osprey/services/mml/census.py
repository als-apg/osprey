"""The import census: everything ``PROFILE.md`` reports about a merged export.

``take_census(ao, ad, mapping)`` walks a merged, normalised ``ao``
(``{system: {family: body}}`` plus ``_``-prefixed bookkeeping keys such as
``_exports`` and ``_import_order``) and the merged ``ad``, and returns one
frozen :class:`Census`. The profile renderer prints it; the import-chain tests
pin its totals.

Rules:

* Every grain number (device count, fields, channel keys, broadcast rows, raw
  and binding counts, alignment of per-device arrays) is read from
  :class:`~osprey.services.mml.family.FamilyView`; nothing here re-derives it.
* ``_``-prefixed keys are never systems and never families.
* Systems are listed in ``_import_order``, then any remaining systems in ``ao``
  order; families and fields keep export order. Every hazard list is sorted.
* A shared PV is an exact stripped channel string owned by two or more distinct
  ``(system, family, field, index)`` slots as exported, so a 1-row broadcast
  list is one owner, and the same string under both channel keys at one index
  is one owner.
* Pending judgments are read from the **raw** views, because the profile is
  rendered at import, before any mapping exists. Every other number follows the
  reviewer's answers when a mapping is given, and the export as imported when
  it is not.
* A 2.0 export's ``va.json`` and ``response.json`` blocks are read per system
  into a :class:`VACensus`; a system the import carried no block for states
  ``None``. Those blocks are stored as the exporter wrote them, so they are
  read as written, while the AT blocks and units they do not restate are read
  from the AO beside them and the deck's cavity count from the facts the
  caller loaded it into.
* A Position slot is real when it is a finite number; ``None``, a string
  (including ``"Inf"``/``"-Inf"``/``"NaN"``) or an unaligned array is a stand-in.
  A DeviceType slot is real when it is a non-blank string. The coverage totals
  count every family, as the per-system coverage table does, so the four
  coverage rows sum to ``devices``.

The module is pure and depends on the standard library, ``FamilyView``, the
pending judgments and the Turtle local-name pattern.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from osprey.services.facility_knowledge.ttl_generator.model import PN_LOCAL
from osprey.services.mml.family import FamilyView, family_views, system_bodies
from osprey.services.mml.judgments import (
    PendingJudgments,
    judged_family_views,
    pending_judgments,
)
from osprey.services.mml.mapping.schema import Mapping

__all__ = [
    "KNOWN_HANDLE_KEYS",
    "KNOWN_MML_KEYS",
    "Census",
    "DualKeyField",
    "FamilyCensus",
    "FieldCensus",
    "FunctionHandle",
    "Hazards",
    "KeyLocation",
    "ListLocation",
    "NonFiniteRange",
    "Owner",
    "PartialList",
    "SharedPV",
    "SystemCensus",
    "TagCensus",
    "Totals",
    "UnitShape",
    "VA_HOOK_KEYS",
    "VA_SAMPLED_FIELDS",
    "VACensus",
    "VAFamilyCensus",
    "VAFieldCensus",
    "VAHook",
    "VAResponseCensus",
    "take_census",
    "va_census",
]

#: Keys that legitimately hold a function handle; a handle under any other key
#: is also a typo key.
KNOWN_HANDLE_KEYS: frozenset[str] = frozenset(
    {"HW2PhysicsFcn", "Physics2HWFcn", "SpecialFunctionGet", "SpecialFunctionSet"}
)

#: MML family and field keys; a key equal to one of these but for case is a typo key.
KNOWN_MML_KEYS: frozenset[str] = frozenset(
    {
        *KNOWN_HANDLE_KEYS,
        "ChannelNames",
        "TangoNames",
        "CommonNames",
        "DataType",
        "Description",
        "DeviceList",
        "DeviceType",
        "ElementList",
        "FamilyName",
        "HW2PhysicsParams",
        "HWUnits",
        "MemberOf",
        "Mode",
        "Physics2HWParams",
        "PhysicsUnits",
        "Position",
        "Range",
        "Status",
        "Tolerance",
        "Units",
    }
)

#: The fields a 2.0 export samples per family; a family listing any other
#: field carries it as an extra field.
VA_SAMPLED_FIELDS: tuple[str, ...] = ("Setpoint", "Monitor")

#: ``AT`` block keys that hand a family to facility MATLAB code rather than to
#: a lattice element.
VA_HOOK_KEYS: tuple[str, ...] = (
    "SpecialFunctionGet",
    "SpecialFunctionSet",
    "ATParameterGroup",
)

#: Spellings the normaliser gives non-finite numbers.
_NON_FINITE: frozenset[str] = frozenset({"Inf", "-Inf", "NaN"})

#: Per-field keys whose shape decides whether a binding property is emitted.
_UNIT_KEYS: tuple[str, ...] = ("DataType", "HWUnits")

_LOWER_KNOWN: dict[str, str] = {key.lower(): key for key in KNOWN_MML_KEYS}

UnitKind = Literal["empty", "per-device", "non-scalar", "non-string"]


@dataclass(frozen=True, order=True)
class Owner:
    """One exported channel slot."""

    system: str
    family: str
    field: str
    index: int


@dataclass(frozen=True)
class SharedPV:
    """A PV bound by two or more slots, with every owner in sorted order."""

    pv: str
    owners: tuple[Owner, ...]


@dataclass(frozen=True)
class FunctionHandle:
    """A ``{"$fn", "file"}`` handle at ``path`` inside a family body."""

    system: str
    family: str
    path: tuple[str, ...]
    function: str | None
    file: str | None


@dataclass(frozen=True)
class KeyLocation:
    """A key at ``path`` inside a family body."""

    system: str
    family: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class NonFiniteRange:
    """A field ``Range`` holding a non-finite value."""

    system: str
    family: str
    field: str
    value: Any


@dataclass(frozen=True)
class UnitShape:
    """A field ``HWUnits`` or ``DataType`` that is not a non-empty string."""

    system: str
    family: str
    field: str
    key: str
    kind: UnitKind


@dataclass(frozen=True)
class DualKeyField:
    """A field carrying both channel keys."""

    system: str
    family: str
    field: str


@dataclass(frozen=True)
class ListLocation:
    """One channel key of one field."""

    system: str
    family: str
    field: str
    key: str


@dataclass(frozen=True)
class PartialList:
    """A channel list too short for its family, whose gap the family still covers.

    A list shorter than ``n_devices`` is a hazard only when every device it
    misses is reached by another list of the family; when nothing reaches them
    those devices are a pending judgment. A list longer than ``n_devices`` is a
    pending judgment too, never a hazard.
    """

    system: str
    family: str
    field: str
    key: str
    length: int
    n_devices: int


@dataclass(frozen=True)
class Hazards:
    """The hazards found in one system, each list sorted."""

    function_handles: tuple[FunctionHandle, ...]
    typo_keys: tuple[KeyLocation, ...]
    non_finite_ranges: tuple[NonFiniteRange, ...]
    unit_shapes: tuple[UnitShape, ...]
    case_duplicate_families: tuple[tuple[str, ...], ...]
    dual_key_fields: tuple[DualKeyField, ...]
    empty_channel_lists: tuple[ListLocation, ...]
    partial_channel_lists: tuple[PartialList, ...]
    broadcast_rows: tuple[ListLocation, ...]
    zero_channel_families: tuple[str, ...]


@dataclass(frozen=True)
class FieldCensus:
    """One channel-bearing field."""

    name: str
    keys: tuple[str, ...]
    raw_slots: int
    bindings: int
    broadcast: bool
    description: str | None


@dataclass(frozen=True)
class FamilyCensus:
    """One family in one system."""

    system: str
    name: str
    n_devices: int
    n_devices_from_fallback: bool
    arrays_source: Literal["family", "setup"]
    fields: tuple[FieldCensus, ...]
    raw_slots: int
    bindings: int
    disabled_devices: tuple[int, ...]
    description: str | None
    position_real: int
    position_stand_in: int
    device_type_real: int
    device_type_stand_in: int


@dataclass(frozen=True)
class VAHook:
    """A hook inside an ``AT`` block that hands a family to MATLAB code.

    ``field`` is ``None`` for the family-level block and the field's name for a
    field-level one. ``value`` is the function a handle names, or the parameter
    group's name.
    """

    field: str | None
    key: str
    value: str | None


@dataclass(frozen=True)
class VAFieldCensus:
    """One field of one family as the 2.0 export sampled it."""

    name: str
    calibration_kind: str | None
    grid_source: str | None
    nominal_units: str | None
    nominal_synthetic: bool | None
    physics_units: str | None


@dataclass(frozen=True)
class VAFamilyCensus:
    """One family of one system's virtual-accelerator block.

    ``at_devices`` counts the device rows the deck holds at least one element
    for and ``at_elements`` every element across them, so a family whose rows
    each span several slices states more elements than rows.
    """

    name: str
    devices: int
    at_type: str | None
    at_devices: int
    at_elements: int
    fields: tuple[VAFieldCensus, ...]
    extra_fields: tuple[str, ...]
    disagreeing_units: tuple[tuple[str, str], ...]
    hooks: tuple[VAHook, ...]
    energy_candidate: bool
    refused: str | None


@dataclass(frozen=True)
class VAResponseCensus:
    """One block of the response document: two families and the matrix between them."""

    monitor: str | None
    actuator: str | None
    origin: str | None
    timestamp: str | None
    rows: int
    columns: int


@dataclass(frozen=True)
class VACensus:
    """One system's virtual-accelerator export, as it was sampled.

    ``cavities`` is ``None`` when no deck was loaded beside the block, which is
    not the same fact as a deck holding no cavity.
    """

    system: str
    exporter: str | None
    deck: str | None
    elements: int | None
    energy_gev: float | None
    cavities: int | None
    calibrations: tuple[tuple[str, int], ...]
    nominals: int
    synthetic_nominals: int
    families: tuple[VAFamilyCensus, ...]
    refused: tuple[tuple[str, str], ...]
    response: tuple[VAResponseCensus, ...]


@dataclass(frozen=True)
class TagCensus:
    """A verbatim ``MemberOf`` tag and its ``(family, field)`` owners.

    ``field`` is ``None`` for a family-level tag.
    """

    tag: str
    owners: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class SystemCensus:
    """Everything reported for one system (sub-machine)."""

    name: str
    families: tuple[FamilyCensus, ...]
    member_of: tuple[TagCensus, ...]
    families_with_descriptions: tuple[tuple[str, str], ...]
    families_without_descriptions: tuple[str, ...]
    setup_families: tuple[str, ...]
    fallback_families: tuple[str, ...]
    hazards: Hazards
    pending: tuple[PendingJudgments, ...]
    ad_scalars: tuple[tuple[str, str | int | float], ...]
    virtual_accelerator: VACensus | None = None


@dataclass(frozen=True)
class Totals:
    """The import-walk totals across every system."""

    fields: int
    raw_slots: int
    raw_non_blank: int
    blank: int
    bindings: int
    broadcast_fields: int
    distinct_pvs: int
    system_families: int
    families: int
    devices: int
    setup_families: int
    fallback_families: int
    position_real: int
    position_stand_in: int
    device_type_real: int
    device_type_stand_in: int


@dataclass(frozen=True)
class Census:
    """The whole census of one merged export."""

    systems: tuple[SystemCensus, ...]
    shared_pvs: tuple[SharedPV, ...]
    pending: tuple[PendingJudgments, ...]
    illegal_system_tokens: tuple[str, ...]
    totals: Totals


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _systems(ao: dict) -> list[str]:
    present = [raw for raw, _ in system_bodies(ao)]
    order = ao.get("_import_order")
    listed = [s for s in order if s in present] if isinstance(order, list) else []
    return list(dict.fromkeys([*listed, *present]))


def _handles(value: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], dict]]:
    if isinstance(value, dict):
        if "$fn" in value:
            yield path, value
            return
        for key, item in value.items():
            yield from _handles(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _handles(item, (*path, str(index)))


def _keys(value: Any, path: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    if isinstance(value, dict) and "$fn" not in value:
        for key, item in value.items():
            if isinstance(key, str):
                yield (*path, key)
                yield from _keys(item, (*path, key))


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    return isinstance(value, str) and value in _NON_FINITE


def _frozen(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _unit_kind(value: Any, n_devices: int) -> UnitKind | None:
    if _is_text(value):
        return None
    if value is None or value == [] or isinstance(value, str):
        return "empty"
    if isinstance(value, list):
        return "per-device" if len(value) == n_devices else "non-scalar"
    if isinstance(value, dict):
        return "non-scalar"
    return "non-string"


def _reach(view: FamilyView) -> int:
    """Return how many devices the family's longest expanded channel list reaches."""
    return max(
        (len(field.slots(key)) for field in view.fields.values() for key in field.keys),
        default=0,
    )


def _coverage(view: FamilyView, name: str, real: Any) -> tuple[int, int]:
    slots = view.aligned(name)
    if slots is None:
        return 0, view.n_devices
    count = sum(1 for slot in slots if real(slot))
    return count, view.n_devices - count


def _ad_scalars(value: dict, prefix: str = "") -> Iterator[tuple[str, str | int | float]]:
    for key, item in value.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        dotted = f"{prefix}{key}"
        if isinstance(item, dict):
            yield from _ad_scalars(item, f"{dotted}.")
        elif isinstance(item, (str, int, float)):
            yield dotted, item


def _ad_by_system(ad: dict | None, systems: list[str]) -> dict[str, dict]:
    if not isinstance(ad, dict):
        return {}
    keyed = {s: ad[s] for s in systems if isinstance(ad.get(s), dict)}
    if keyed:
        return keyed
    if len(systems) == 1:
        return {systems[0]: ad}
    return {}


def _typo_keys(
    system: str, family: str, body: dict, handle_paths: set[tuple[str, ...]]
) -> list[KeyLocation]:
    """A handle under an unknown key, an unknown ``*Fcn`` key, or a mis-cased MML key."""
    found = {path for path in handle_paths if path[-1] not in KNOWN_HANDLE_KEYS}
    for path in _keys(body, ()):
        key = path[-1]
        if key.startswith("_"):
            continue
        unknown_fcn = key.endswith("Fcn") and key not in KNOWN_HANDLE_KEYS
        miscased = _LOWER_KNOWN.get(key.lower()) not in (None, key)
        if unknown_fcn or miscased:
            found.add(path)
    return [KeyLocation(system, family, path) for path in found]


class _SystemWalk:
    """Accumulates one system's census while the totals and PV owners are shared."""

    def __init__(self, name: str, owners: dict[str, set[Owner]]) -> None:
        self.name = name
        self.owners = owners
        self.families: list[FamilyCensus] = []
        self.views: list[FamilyView] = []
        self.pending: list[PendingJudgments] = []
        self.tags: dict[str, set[tuple[str, str | None]]] = {}
        self.handles: list[FunctionHandle] = []
        self.typos: list[KeyLocation] = []
        self.ranges: list[NonFiniteRange] = []
        self.units: list[UnitShape] = []
        self.dual: list[DualKeyField] = []
        self.empty: list[ListLocation] = []
        self.partial: list[PartialList] = []
        self.broadcast: list[ListLocation] = []

    def add(self, view: FamilyView) -> None:
        self.views.append(view)
        system, family, body = self.name, view.raw_name, view.body

        self._tag(view.arrays.get("MemberOf"), family, None)
        handle_paths = set()
        for path, handle in _handles(body, ()):
            handle_paths.add(path)
            self.handles.append(
                FunctionHandle(system, family, path, handle.get("$fn"), handle.get("file"))
            )
        self.typos.extend(_typo_keys(system, family, body, handle_paths))

        fields = []
        reach = _reach(view)
        for field in view.fields.values():
            fields.append(
                FieldCensus(
                    field.name,
                    field.keys,
                    field.raw_slot_count,
                    field.channel_count,
                    field.broadcast,
                    field.description,
                )
            )
            self._field(view, field, reach)

        position = _coverage(view, "Position", _is_real_number)
        device_type = _coverage(view, "DeviceType", _is_text)
        self.families.append(
            FamilyCensus(
                system=system,
                name=family,
                n_devices=view.n_devices,
                n_devices_from_fallback=view.n_devices_from_fallback,
                arrays_source=view.arrays_source,
                fields=tuple(fields),
                raw_slots=view.raw_slot_count,
                bindings=view.channel_count,
                disabled_devices=view.disabled_devices,
                description=None if view.description is None else view.description[0],
                position_real=position[0],
                position_stand_in=position[1],
                device_type_real=device_type[0],
                device_type_stand_in=device_type[1],
            )
        )

    def pend(self, raw_views: Iterable[FamilyView]) -> None:
        """Take the pending judgments from the raw views of the system.

        What the reviewer was asked is a property of the export alone, so a walk
        counting judged views still reports the judgments the raw export pends.
        """
        for view in raw_views:
            pending = pending_judgments(view)
            if not pending.is_empty:
                self.pending.append(pending)

    def _tag(self, tags: Any, family: str, field: str | None) -> None:
        if not isinstance(tags, list):
            tags = [tags]
        for tag in tags:
            if isinstance(tag, str):
                self.tags.setdefault(tag, set()).add((family, field))

    def _field(self, view: FamilyView, field: Any, reach: int) -> None:
        system, family, name = self.name, view.raw_name, field.name
        body = field.body
        if "MemberOf" in body:
            self._tag(body["MemberOf"], family, name)
        if "Range" in body and _has_non_finite(body["Range"]):
            self.ranges.append(NonFiniteRange(system, family, name, _frozen(body["Range"])))
        for key in _UNIT_KEYS:
            if key in body:
                kind = _unit_kind(body[key], view.n_devices)
                if kind is not None:
                    self.units.append(UnitShape(system, family, name, key, kind))
        if len(field.keys) > 1:
            self.dual.append(DualKeyField(system, family, name))
        for key in field.keys:
            raw = field.raw_slots(key)
            expanded = field.slots(key)
            location = ListLocation(system, family, name, key)
            if not raw:
                self.empty.append(location)
            elif len(expanded) != len(raw):
                self.broadcast.append(location)
            elif len(raw) < view.n_devices and reach >= view.n_devices:
                self.partial.append(
                    PartialList(system, family, name, key, len(raw), view.n_devices)
                )
            for index, slot in enumerate(raw):
                if _is_text(slot):
                    self.owners.setdefault(slot.strip(), set()).add(
                        Owner(system, family, name, index)
                    )

    def finish(self, ad_scalars: tuple, virtual_accelerator: VACensus | None) -> SystemCensus:
        by_lower: dict[str, list[str]] = {}
        for family in self.families:
            by_lower.setdefault(family.name.lower(), []).append(family.name)
        hazards = Hazards(
            function_handles=tuple(sorted(self.handles, key=_location_key)),
            typo_keys=tuple(sorted(self.typos, key=_location_key)),
            non_finite_ranges=tuple(sorted(self.ranges, key=lambda r: (r.family, r.field))),
            unit_shapes=tuple(sorted(self.units, key=lambda u: (u.family, u.field, u.key))),
            case_duplicate_families=tuple(
                sorted(tuple(sorted(names)) for names in by_lower.values() if len(names) > 1)
            ),
            dual_key_fields=tuple(sorted(self.dual, key=lambda d: (d.family, d.field))),
            empty_channel_lists=tuple(sorted(self.empty, key=_list_key)),
            partial_channel_lists=tuple(sorted(self.partial, key=_list_key)),
            broadcast_rows=tuple(sorted(self.broadcast, key=_list_key)),
            zero_channel_families=tuple(sorted(f.name for f in self.families if f.bindings == 0)),
        )
        return SystemCensus(
            name=self.name,
            families=tuple(self.families),
            member_of=tuple(
                TagCensus(tag, tuple(sorted(owners, key=lambda o: (o[0], o[1] or ""))))
                for tag, owners in sorted(self.tags.items())
            ),
            families_with_descriptions=tuple(
                (f.name, f.description) for f in self.families if f.description is not None
            ),
            families_without_descriptions=tuple(
                f.name for f in self.families if f.description is None
            ),
            setup_families=tuple(f.name for f in self.families if f.arrays_source == "setup"),
            fallback_families=tuple(f.name for f in self.families if f.n_devices_from_fallback),
            hazards=hazards,
            pending=tuple(self.pending),
            ad_scalars=ad_scalars,
            virtual_accelerator=virtual_accelerator,
        )


def _location_key(item: FunctionHandle | KeyLocation) -> tuple:
    return (item.family, item.path)


def _list_key(item: ListLocation | PartialList) -> tuple:
    return (item.family, item.field, item.key)


def _va_device_rows(device_list: Any) -> int:
    """Return how many device rows a virtual-accelerator block states.

    MATLAB writes a one-device family's ``DeviceList`` flat, so a list of two
    numbers is one row and a list of lists is one row each.
    """
    if not isinstance(device_list, list) or not device_list:
        return 0
    if all(isinstance(row, list) for row in device_list):
        return len(device_list)
    return 1


def _at_slices(index: Any) -> list[int]:
    """Return how many lattice elements each device row of ``ATIndex`` holds.

    A scalar is one row, a flat list one row per entry and a list of lists one
    row per list. ``"NaN"`` is the spelling a row uses where it has fewer
    slices than its siblings, so it counts as no element.
    """
    if _is_real_number(index):
        return [1]
    if not isinstance(index, list) or not index:
        return []
    if all(isinstance(row, list) for row in index):
        return [sum(1 for item in row if _is_real_number(item)) for row in index]
    return [1 if _is_real_number(item) else 0 for item in index]


def _text_or_none(value: Any) -> str | None:
    return value.strip() if _is_text(value) else None


def _deck_name(ad_body: Any) -> str | None:
    """Return the deck a system runs on, named as the knowledge pages name it."""
    if not isinstance(ad_body, dict):
        return None
    ops = ad_body.get("OpsData")
    if isinstance(ops, dict) and _is_text(ops.get("LatticeFile")):
        return _text_or_none(ops["LatticeFile"])
    return _text_or_none(ad_body.get("ATModel"))


def _hook_value(value: Any) -> str | None:
    if isinstance(value, dict):
        return _text_or_none(value.get("$fn"))
    return _text_or_none(value)


def _at_block(body: Any) -> dict:
    at = body.get("AT") if isinstance(body, dict) else None
    return at if isinstance(at, dict) else {}


def _va_hooks(ao_family: Any) -> list[VAHook]:
    """Return every hook of one family's AT blocks, the family-level block first."""
    if not isinstance(ao_family, dict):
        return []
    blocks: list[tuple[str | None, dict]] = [(None, _at_block(ao_family))]
    for name, body in ao_family.items():
        if isinstance(name, str) and not name.startswith("_") and name != "AT":
            at = _at_block(body)
            if at:
                blocks.append((name, at))
    return [
        VAHook(field, key, _hook_value(at[key]))
        for field, at in blocks
        for key in VA_HOOK_KEYS
        if key in at
    ]


def _at_facts(family: dict, ao_family: Any) -> tuple[str | None, Any]:
    """Return the AT type and index the export states for one family.

    The exporter copies both beside every nominal it sampled; a family whose
    nominal it refused still states them in the export the block sits beside.
    """
    nominals = family.get("nominals")
    if isinstance(nominals, dict):
        for block in nominals.values():
            if isinstance(block, dict) and _is_text(block.get("at_type")):
                return _text_or_none(block["at_type"]), block.get("at_index")
    at = _at_block(ao_family)
    return _text_or_none(at.get("ATType")), at.get("ATIndex")


def _va_field(name: str, family: dict, ao_family: Any) -> VAFieldCensus:
    """Return one sampled field of one family."""
    body = family.get(name)
    calibration = body.get("calibration") if isinstance(body, dict) else None
    calibration = calibration if isinstance(calibration, dict) else {}
    nominals = family.get("nominals")
    nominal = nominals.get(name) if isinstance(nominals, dict) else None
    nominal = nominal if isinstance(nominal, dict) else {}
    ao_field = ao_family.get(name) if isinstance(ao_family, dict) else None
    return VAFieldCensus(
        name=name,
        calibration_kind=_text_or_none(calibration.get("kind")),
        grid_source=_text_or_none(calibration.get("grid_source")),
        nominal_units=_text_or_none(nominal.get("units")),
        nominal_synthetic=bool(nominal["synthetic"]) if "synthetic" in nominal else None,
        physics_units=(
            _text_or_none(ao_field.get("PhysicsUnits")) if isinstance(ao_field, dict) else None
        ),
    )


def _disagreeing_units(fields: Iterable[VAFieldCensus]) -> tuple[tuple[str, str], ...]:
    """Return every field's units when the siblings do not spell one unit.

    Which spelling the verdict follows is the reviewer's to settle; the census
    only reports that the family states more than one.
    """
    stated = [(f.name, f.physics_units) for f in fields if f.physics_units is not None]
    if len({units.casefold() for _, units in stated}) < 2:
        return ()
    return tuple(stated)


def _va_family(name: str, family: dict, ao_family: Any) -> VAFamilyCensus:
    """Return one family of one virtual-accelerator block."""
    listed = family.get("fields")
    listed = [f for f in listed if isinstance(f, str)] if isinstance(listed, list) else []
    fields = tuple(
        _va_field(f, family, ao_family)
        for f in VA_SAMPLED_FIELDS
        if isinstance(family.get(f), dict)
    )
    at_type, at_index = _at_facts(family, ao_family)
    slices = _at_slices(at_index)
    return VAFamilyCensus(
        name=name,
        devices=_va_device_rows(family.get("device_list")),
        at_type=at_type,
        at_devices=sum(1 for count in slices if count),
        at_elements=sum(slices),
        fields=fields,
        extra_fields=tuple(f for f in listed if f not in VA_SAMPLED_FIELDS),
        disagreeing_units=_disagreeing_units(fields),
        hooks=tuple(_va_hooks(ao_family)),
        energy_candidate=bool(family.get("energy_candidate")),
        refused=_text_or_none(family.get("refused")),
    )


def _matrix_size(data: Any) -> tuple[int, int]:
    """Return the rows and columns of a response matrix, a flat list being one row."""
    if not isinstance(data, list) or not data:
        return 0, 0
    if all(isinstance(row, list) for row in data):
        return len(data), max(len(row) for row in data)
    return 1, len(data)


def _response_blocks(response: Any) -> tuple[VAResponseCensus, ...]:
    """Return one record per block of a response document, in export order."""
    blocks = response.get("blocks") if isinstance(response, dict) else None
    if not isinstance(blocks, list):
        return ()
    found = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rows, columns = _matrix_size(block.get("data"))
        found.append(
            VAResponseCensus(
                monitor=_side_family(block.get("monitor")),
                actuator=_side_family(block.get("actuator")),
                origin=_text_or_none(block.get("origin")),
                timestamp=_text_or_none(block.get("timestamp")),
                rows=rows,
                columns=columns,
            )
        )
    return tuple(found)


def _side_family(side: Any) -> str | None:
    return _text_or_none(side.get("family")) if isinstance(side, dict) else None


def _nominal_counts(families: Iterable[dict]) -> tuple[int, int]:
    """Return how many nominals the export sampled and how many stood in."""
    blocks = [
        block
        for family in families
        if isinstance(family.get("nominals"), dict)
        for block in family["nominals"].values()
        if isinstance(block, dict)
    ]
    return len(blocks), sum(1 for block in blocks if block.get("synthetic"))


def va_census(
    system: str,
    va: dict | None,
    response: dict | None = None,
    *,
    ao_body: dict | None = None,
    ad_body: dict | None = None,
    ring_facts: dict | None = None,
) -> VACensus | None:
    """Take the census of one system's virtual-accelerator export.

    Args:
        system: The system token the block was imported under.
        va: The system's ``va.json`` block as the exporter wrote it, or
            ``None`` when the import carried none. It is not modified.
        response: The system's ``response.json`` block, or ``None``.
        ao_body: The system's AO body, which carries the AT blocks and the
            units the block itself does not restate.
        ad_body: The system's AD, which names the deck.
        ring_facts: Facts read from the loaded deck, of which ``cavities`` is
            the one reported. ``None`` leaves the cavity count unstated.

    Returns:
        The frozen census, or ``None`` when the system has no block.
    """
    if not isinstance(va, dict):
        return None
    lattice = va.get("lattice") if isinstance(va.get("lattice"), dict) else {}
    export = va.get("_export") if isinstance(va.get("_export"), dict) else {}
    bodies = va.get("families") if isinstance(va.get("families"), dict) else {}
    families = {
        name: body
        for name, body in bodies.items()
        if isinstance(name, str) and isinstance(body, dict)
    }
    ao_families = ao_body if isinstance(ao_body, dict) else {}
    records = tuple(
        _va_family(name, body, ao_families.get(name)) for name, body in families.items()
    )
    kinds: dict[str, int] = {}
    for family in records:
        for field in family.fields:
            if field.calibration_kind is not None:
                kinds[field.calibration_kind] = kinds.get(field.calibration_kind, 0) + 1
    nominals, synthetic = _nominal_counts(families.values())
    facts = ring_facts if isinstance(ring_facts, dict) else {}
    cavities = facts.get("cavities")
    return VACensus(
        system=system,
        exporter=_text_or_none(export.get("exporter")),
        deck=_deck_name(ad_body),
        elements=lattice.get("elements") if _is_real_number(lattice.get("elements")) else None,
        energy_gev=(
            lattice.get("energy_gev") if _is_real_number(lattice.get("energy_gev")) else None
        ),
        cavities=cavities if _is_real_number(cavities) else None,
        calibrations=tuple(sorted(kinds.items())),
        nominals=nominals,
        synthetic_nominals=synthetic,
        families=records,
        refused=tuple((f.name, f.refused) for f in records if f.refused is not None),
        response=_response_blocks(response),
    )


def take_census(
    ao: dict,
    ad: dict | None,
    mapping: Mapping | None = None,
    *,
    va: dict | None = None,
    response: dict | None = None,
    ring_facts: dict | None = None,
) -> Census:
    """Take the census of a merged, normalised export.

    Args:
        ao: The merged AO, ``{system: {family: normalised body}}`` plus
            ``_``-prefixed bookkeeping keys. It is not modified.
        ad: The merged AD keyed by system, a flat AD when one system was
            imported, or ``None``. It is not modified.
        mapping: The mapping whose judgment answers every count, hazard and
            owner follows, or ``None`` to read the export as it was imported.
            The pending judgments are read from the raw export either way.
        va: The canonical ``va.json``, ``{system: block}``, or ``None``. It is
            not modified.
        response: The canonical ``response.json``, ``{system: block}``, or
            ``None``. It is not modified.
        ring_facts: Facts read from each system's loaded deck,
            ``{system: {"cavities": n}}``, or ``None`` to leave the cavity
            count unstated.

    Returns:
        The frozen census.
    """
    systems = _systems(ao)
    ad_by_system = _ad_by_system(ad, systems)
    owners: dict[str, set[Owner]] = {}
    walks: list[_SystemWalk] = []
    for system in systems:
        walk = _SystemWalk(system, owners)
        body = ao[system]
        raw = list(family_views(system, body))
        for view in raw if mapping is None else judged_family_views(system, body, mapping):
            walk.add(view)
        walk.pend(raw)
        walks.append(walk)

    by_system = va if isinstance(va, dict) else {}
    responses = response if isinstance(response, dict) else {}
    facts = ring_facts if isinstance(ring_facts, dict) else {}
    system_censuses = tuple(
        walk.finish(
            tuple(sorted(_ad_scalars(ad_by_system[walk.name])))
            if walk.name in ad_by_system
            else (),
            va_census(
                walk.name,
                by_system.get(walk.name),
                responses.get(walk.name),
                ao_body=ao[walk.name],
                ad_body=ad_by_system.get(walk.name),
                ring_facts=facts.get(walk.name),
            ),
        )
        for walk in walks
    )

    views = [view for walk in walks for view in walk.views]
    fields = [field for view in views for field in view.fields.values()]
    raw_slots = sum(view.raw_slot_count for view in views)
    raw_non_blank = sum(
        1
        for field in fields
        for key in field.keys
        for slot in field.raw_slots(key)
        if _is_text(slot)
    )
    every_family = [family for census in system_censuses for family in census.families]
    totals = Totals(
        fields=len(fields),
        raw_slots=raw_slots,
        raw_non_blank=raw_non_blank,
        blank=raw_slots - raw_non_blank,
        bindings=sum(view.channel_count for view in views),
        broadcast_fields=sum(1 for field in fields if field.broadcast),
        distinct_pvs=len(owners),
        system_families=len(views),
        families=len({view.raw_name for view in views}),
        devices=sum(view.n_devices for view in views),
        setup_families=sum(1 for view in views if view.arrays_source == "setup"),
        fallback_families=sum(1 for view in views if view.n_devices_from_fallback),
        position_real=sum(f.position_real for f in every_family),
        position_stand_in=sum(f.position_stand_in for f in every_family),
        device_type_real=sum(f.device_type_real for f in every_family),
        device_type_stand_in=sum(f.device_type_stand_in for f in every_family),
    )
    return Census(
        systems=system_censuses,
        shared_pvs=tuple(
            SharedPV(pv, tuple(sorted(slots)))
            for pv, slots in sorted(owners.items())
            if len(slots) > 1
        ),
        pending=tuple(pending for census in system_censuses for pending in census.pending),
        illegal_system_tokens=tuple(s for s in systems if PN_LOCAL.fullmatch(s) is None),
        totals=totals,
    )
