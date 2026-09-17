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
    "take_census",
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

    def finish(self, ad_scalars: tuple) -> SystemCensus:
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
        )


def _location_key(item: FunctionHandle | KeyLocation) -> tuple:
    return (item.family, item.path)


def _list_key(item: ListLocation | PartialList) -> tuple:
    return (item.family, item.field, item.key)


def take_census(ao: dict, ad: dict | None, mapping: Mapping | None = None) -> Census:
    """Take the census of a merged, normalised export.

    Args:
        ao: The merged AO, ``{system: {family: normalised body}}`` plus
            ``_``-prefixed bookkeeping keys. It is not modified.
        ad: The merged AD keyed by system, a flat AD when one system was
            imported, or ``None``. It is not modified.
        mapping: The mapping whose judgment answers every count, hazard and
            owner follows, or ``None`` to read the export as it was imported.
            The pending judgments are read from the raw export either way.

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

    system_censuses = tuple(
        walk.finish(
            tuple(sorted(_ad_scalars(ad_by_system[walk.name]))) if walk.name in ad_by_system else ()
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
