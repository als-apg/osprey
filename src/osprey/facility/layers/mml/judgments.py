"""Pending reviewer judgments, detected from one raw family view.

Three extraction shapes are not decidable by rule, so the mapping asks a
reviewer instead of guessing. This module finds them, and only them, from a
raw :class:`~osprey.facility.layers.mml.family.FamilyView`, per (system, family):

* **rows beyond devices** -- a non-blank slot at an index at or past
  ``n_devices`` of a channel list longer than ``n_devices``. Per field, the
  same string is one question however many slots carry it; a dual-key row
  whose two keys carry different strings is two questions.
* **unbound devices** -- a device ordinal no channel list reaches, measured
  against the *expanded* list lengths, so a family with a broadcast field has
  none.
* **shared PVs** -- one string bound at two or more device indices of a single
  non-broadcast list. Strings sharing an index set across the family's fields
  form one supply group, keyed by its lowest 1-based ordinal, so a reviewer
  answers a family's supply once rather than PV by PV.

Not judgments, by rule: blank slots inside a full-length list, broadcast rows,
zero-length lists, zero-channel families, and a PV repeated across different
fields or families. A family whose device count comes from the longest-list
fallback (no ``DeviceList``) has no row and no device grain to judge, so it can
pend shared PVs alone.

Ordinals leaving this module are 1-based, the numbering the mapping
speaks; ``PendingRow.index`` is the 0-based export position.
Signals are carried exactly as exported, so a mapping key matches the export
string by construction.

:func:`apply_judgments` reads the reviewer's answers back
onto one family body, before any view a consumer uses is built. A dropped
device leaves every list that holds one slot per device, a kept one gains an
empty slot in the lists that stopped short of it, an owned supply group keeps
its PVs on the owning device alone, and the family's device count follows its
``DeviceList``. It holds no rule text of its own. An answer that is not
pending in the system it is handed is ignored, because the same family may be
judged differently in another system; an answer the mapping check would
refuse is an internal ``AssertionError`` rather than a second refusal path.
:func:`judged_family_views` is that half over a whole system, standing in for
:func:`~osprey.facility.layers.mml.family.family_views` wherever a consumer reads the
grain the reviewer settled on, and :func:`judged_va_block` is the same answers
over the virtual-accelerator block the export wrote beside the family, so both
documents are read in one device order.

The module is pure: the standard library, the family view and the layer's
mapping, no I/O.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from osprey.facility.layers.mml.family import (
    FAMILY_ARRAYS,
    FamilyView,
    family_views,
    system_bodies,
)
from osprey.facility.layers.mml.mapping import (
    FamilyJudgments,
    FieldAnswer,
    Mapping,
    OwnerMap,
    RowAnswer,
    UnboundAnswer,
)
from osprey.services.channel_finder.databases.middle_layer import CHANNEL_KEYS

__all__ = [
    "FIELD_METADATA_KEYS",
    "PendingJudgments",
    "PendingRow",
    "SupplyGroup",
    "all_pending_judgments",
    "apply_judgments",
    "judged_family_views",
    "judged_va_block",
    "pending_judgments",
]

#: Field metadata the middle-layer loader copies onto each channel, in order.
FIELD_METADATA_KEYS: tuple[str, ...] = (
    "DataType",
    "Mode",
    "Units",
    "HWUnits",
    "PhysicsUnits",
    "MemberOf",
    "Range",
    "Tolerance",
)

#: Sub-dicts that carry copies of the family arrays, in the order consulted.
_SETUP_KEYS: tuple[str, ...] = ("setup", "_setup")

#: Family arrays holding one slot per device; ``DeviceList`` grows by rule and
#: ``MemberOf`` describes the family rather than its devices.
_PER_DEVICE_ARRAYS: tuple[str, ...] = tuple(
    name for name in FAMILY_ARRAYS if name not in ("DeviceList", "MemberOf")
)

#: Field keys holding one slot per device; every other field key is left as
#: exported, because no consumer reads it per device.
_PER_DEVICE_FIELD_KEYS: tuple[str, ...] = ("HWUnits", "PhysicsUnits", "DataType")

#: Field keys holding one number per device beside the channel lists: the
#: limits a device is driven between, the gain and offset a reading is
#: corrected by, the angle and the shear that carry a monitor's planes into the
#: model's, and the value a reading is compared to.
_PER_DEVICE_VALUE_KEYS: tuple[str, ...] = (
    "Range",
    "Gain",
    "Offset",
    "Roll",
    "Crunch",
    "Golden",
)

#: Field keys holding the parameters of a conversion. A cell states one row
#: per parameter and one column per device, the transpose of every other
#: per-device list, so both axes are offered and the device count settles
#: which one the family is realigned along.
_PER_DEVICE_PARAM_KEYS: tuple[str, ...] = ("HW2PhysicsParams", "Physics2HWParams")

#: The sub-dict carrying a family's element indices, and the key under it that
#: holds one index per device -- or one row of slice indices per device.
_AT_KEY = "AT"
_AT_ROW_KEYS: tuple[str, ...] = ("ATIndex",)

#: Keys whose flat pair of numbers is one low/high span rather than two
#: devices' slots. Every other key reads a flat list as one slot per device.
_SPAN_KEYS = frozenset({"Range", "device_list", "finite_span"})

#: The rows of one ``va.json`` family block that carry a slot per device: the
#: nominal a device sits at, and the parameters of one conversion.
_VA_NOMINAL_ROW_KEYS: tuple[str, ...] = ("values", "at_index", "synthetic")
_VA_CONVERSION_KEYS: tuple[str, ...] = ("calibration", "monitor_inverse")
_VA_CONVERSION_ROW_KEYS: tuple[str, ...] = ("gain", "offset", "grid", "values", "finite_span")

#: The sub-dict carrying what a field's readings are corrected by, and the keys
#: under it that hold one value per device.
_VA_READOUT_KEY = "readout"
_VA_READOUT_ROW_KEYS: tuple[str, ...] = ("gain", "offset", "roll", "crunch")

#: Field metadata a moved row carries over whole, list or not.
_VERBATIM_FIELD_KEYS: tuple[str, ...] = ("MemberOf", "Range", "Tolerance")

#: The blank of a channel list; the DB writer renders it ``""``.
_CHANNEL_BLANK = None

#: The blank of a per-device list: the export's own spelling for "no value".
_DEVICE_BLANK = ""

#: The blank of a per-device number: what the export writes for an absent one.
_NUMBER_BLANK = "NaN"


def _signal(slot: Any) -> str | None:
    """Return the slot's string when it binds, else ``None``.

    A slot binds when it is a non-blank string; the string is returned exactly
    as exported, so a judgment key matches the export by construction.
    """
    if isinstance(slot, str) and slot.strip():
        return slot
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class PendingRow:
    """One row beyond the family's devices, awaiting an answer.

    Attributes:
        field: The field whose list carries the row.
        keys: The channel keys carrying this signal, in export order.
        index: The lowest 0-based export position carrying it, at or past
            ``n_devices``.
        signal: The signal string, exactly as exported.
    """

    field: str
    keys: tuple[str, ...]
    index: int
    signal: str


@dataclass(frozen=True)
class SupplyGroup:
    """One set of devices bound by the same PVs.

    Attributes:
        lowest: The group's lowest 1-based ordinal; its key in the mapping.
        ordinals: Every member ordinal, 1-based and ascending.
        pvs: The PVs the members share, sorted.
        device_rows: The ``DeviceList`` row of each member, in ``ordinals``
            order, or ``None`` where the family states none.
        group_only_members: How many members carry no channel outside this
            group. An owner answer leaves the others bound by nothing.
    """

    lowest: int
    ordinals: tuple[int, ...]
    pvs: tuple[str, ...]
    device_rows: tuple[tuple[int, int] | None, ...]
    group_only_members: int


@dataclass(frozen=True)
class PendingJudgments:
    """Everything one family of one system asks its reviewer.

    Attributes:
        system: The system the family was found in.
        family: The family's raw name, as exported.
        n_devices: The family's device count, as exported.
        rows_beyond: Rows beyond the devices, in export order.
        unbound_devices: Ordinals no channel list reaches, 1-based ascending.
        groups: The family's supply groups, by lowest ordinal.
        body_keys: Every key the family body carries, so a ``field:`` answer
            can be refused before it collides with one.
        bound_below: Every signal bound below ``n_devices``, so a row answered
            ``device`` can be refused before it mints a supply nobody asked
            about.
        unbound_rows: The ``DeviceList`` row of each unbound ordinal, in
            ``unbound_devices`` order, or ``None`` where the family states none.
    """

    system: str
    family: str
    n_devices: int
    rows_beyond: tuple[PendingRow, ...]
    unbound_devices: tuple[int, ...]
    groups: tuple[SupplyGroup, ...]
    body_keys: tuple[str, ...] = ()
    bound_below: frozenset[str] = frozenset()
    unbound_rows: tuple[tuple[int, int] | None, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the family asks nothing."""
        return not (self.rows_beyond or self.unbound_devices or self.groups)

    @property
    def count(self) -> int:
        """The number of answer slots: one per row, one per ordinal, one supply."""
        return len(self.rows_beyond) + len(self.unbound_devices) + (1 if self.groups else 0)


def _rows_beyond(view: FamilyView) -> tuple[PendingRow, ...]:
    """Return the rows past ``n_devices``, one entry per signal per field."""
    n_devices = view.n_devices
    found: dict[tuple[str, str], tuple[list[str], int]] = {}
    for field in view.fields.values():
        for key in field.keys:
            raw = field.raw_slots(key)
            if len(raw) <= n_devices:
                continue
            for index in range(n_devices, len(raw)):
                signal = _signal(raw[index])
                if signal is None:
                    continue
                keys, first = found.setdefault((field.name, signal), ([], index))
                if key not in keys:
                    keys.append(key)
                found[(field.name, signal)] = (keys, min(first, index))
    return tuple(
        PendingRow(field, tuple(keys), index, signal)
        for (field, signal), (keys, index) in found.items()
    )


def _reach(view: FamilyView) -> int:
    """Return how many devices the family's longest expanded list reaches."""
    return max(
        (len(field.slots(key)) for field in view.fields.values() for key in field.keys),
        default=0,
    )


def _unbound_devices(view: FamilyView) -> tuple[int, ...]:
    """Return the ordinals past the longest expanded list, 1-based."""
    if view.channel_count == 0:
        return ()
    return tuple(range(_reach(view) + 1, view.n_devices + 1))


def _device_row(view: FamilyView, index: int) -> tuple[int, int] | None:
    """Return the ``DeviceList`` sector/device pair at ``index``, when stated."""
    rows = view.device_rows
    if rows is None or index >= len(rows):
        return None
    row = rows[index]
    if not all(_is_number(item) for item in row):
        return None
    return (int(row[0]), int(row[1]))


def _group_only_members(view: FamilyView, indices: list[int], pvs: frozenset[str]) -> int:
    """Count members whose every non-blank expanded slot is a PV of the group."""
    total = 0
    for index in indices:
        outside = False
        for field in view.fields.values():
            for key in field.keys:
                slots = field.slots(key)
                if index >= len(slots):
                    continue
                signal = _signal(slots[index])
                if signal is not None and signal not in pvs:
                    outside = True
        if not outside:
            total += 1
    return total


def _groups(view: FamilyView) -> tuple[SupplyGroup, ...]:
    """Return the family's supply groups, PVs collected by shared index set."""
    n_devices = view.n_devices
    shared: dict[frozenset[int], dict[str, None]] = {}
    for field in view.fields.values():
        for key in field.keys:
            raw = field.raw_slots(key)
            if len(field.slots(key)) != len(raw):
                continue
            where: dict[str, list[int]] = {}
            for index, slot in enumerate(raw[:n_devices]):
                signal = _signal(slot)
                if signal is not None:
                    where.setdefault(signal, []).append(index)
            for pv, indices in where.items():
                if len(indices) > 1:
                    shared.setdefault(frozenset(indices), {})[pv] = None

    groups = []
    for index_set, pvs in shared.items():
        indices = sorted(index_set)
        ordinals = tuple(index + 1 for index in indices)
        groups.append(
            SupplyGroup(
                lowest=ordinals[0],
                ordinals=ordinals,
                pvs=tuple(sorted(pvs)),
                device_rows=tuple(_device_row(view, index) for index in indices),
                group_only_members=_group_only_members(view, indices, frozenset(pvs)),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.ordinals))


def _bound_below(view: FamilyView) -> frozenset[str]:
    """Return every signal the family binds below its device count."""
    return frozenset(
        signal
        for field in view.fields.values()
        for key in field.keys
        for slot in field.raw_slots(key)[: view.n_devices]
        if (signal := _signal(slot)) is not None
    )


def pending_judgments(view: FamilyView) -> PendingJudgments:
    """Return everything ``view`` asks its reviewer.

    Args:
        view: A raw family view; judgments are detected before any answer is
            applied, and the view is only read.

    Returns:
        The family's pending judgments, empty tuples where nothing pends.
    """
    fallback = view.n_devices_from_fallback
    unbound = () if fallback else _unbound_devices(view)
    return PendingJudgments(
        system=view.system,
        family=view.raw_name,
        n_devices=view.n_devices,
        rows_beyond=() if fallback else _rows_beyond(view),
        unbound_devices=unbound,
        groups=_groups(view),
        body_keys=tuple(view.body),
        bound_below=_bound_below(view),
        unbound_rows=tuple(_device_row(view, ordinal - 1) for ordinal in unbound),
    )


def all_pending_judgments(ao: dict) -> dict[tuple[str, str], PendingJudgments]:
    """Return what every family of every system of ``ao`` asks its reviewer.

    Keyed by ``(system, raw family)`` and taken off the raw export before any
    answer is applied, so every check of a mapping judges the same questions.

    Args:
        ao: The merged export, keyed by raw system token plus bookkeeping keys;
            only read.

    Returns:
        One :class:`PendingJudgments` per family of every system, in export
        order, the families asking nothing included.
    """
    return {
        (system, view.raw_name): pending_judgments(view)
        for system, families in system_bodies(ao)
        for view in family_views(system, families)
    }


@dataclass
class _MovedRow:
    """One row on its way out of its field and into a field of its own.

    Attributes:
        field: The source field the row was exported under.
        index: The row's 0-based export position.
        signals: The signal carried under each channel key of the row.
    """

    field: str
    index: int
    signals: dict[str, str]


def _as_list(value: Any) -> list:
    """Return a channel-key value as a list of slots, a bare string as one."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _is_flat_pair(value: Any) -> bool:
    """Whether ``value`` is a bare ``[sector, device]`` pair of numbers."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_number(item) for item in value)
    )


def _array_containers(body: dict) -> list[dict]:
    """Return every dict of ``body`` that may carry the family arrays."""
    containers = [body]
    for key in _SETUP_KEYS:
        setup = body.get(key)
        if isinstance(setup, dict):
            containers.append(setup)
    return containers


def _row_answers(
    pending: PendingJudgments, judgments: FamilyJudgments
) -> dict[tuple[str, str], tuple[PendingRow, RowAnswer]]:
    """Return the answered rows this system actually pends, by field and signal.

    An answer naming a row the system does not pend is left out: the same
    family is judged per system, and a row of another system's export is not
    this one's to apply.
    """
    answered: dict[tuple[str, str], tuple[PendingRow, RowAnswer]] = {}
    for row in pending.rows_beyond:
        answer = judgments.rows_beyond.get(row.field, {}).get(row.signal)
        if answer is not None:
            answered[(row.field, row.signal)] = (row, answer)
    return answered


def _guard_row_answers(
    pending: PendingJudgments, answered: dict[tuple[str, str], tuple[PendingRow, RowAnswer]]
) -> None:
    """Assert the answers are ones the mapping check would have passed."""
    named: dict[str, tuple[str, int]] = {}
    for (field_name, signal), (row, answer) in answered.items():
        if answer == "device":
            assert signal not in pending.bound_below, (
                f"{field_name}[{signal}] is answered 'device' but the signal is also "
                f"bound below device {pending.n_devices + 1}"
            )
        if isinstance(answer, FieldAnswer):
            assert answer.name not in pending.body_keys, (
                f"{field_name}[{signal}] is answered 'field: {answer.name}' but the "
                "family already carries that key"
            )
            source = named.setdefault(answer.name, (field_name, row.index))
            assert source == (field_name, row.index), (
                f"field name {answer.name!r} is answered on two different rows"
            )


def _normalise_single_device(body: dict) -> None:
    """Write a one-device family's scalars as the one-slot lists they stand for.

    A one-device family may state a flat ``DeviceList`` pair and a scalar where
    every other family writes a list. Only such a family is normalised: a
    scalar on a family of several devices says something else -- one value for
    the whole family -- and is left alone.
    """
    for container in _array_containers(body):
        if _is_flat_pair(container.get("DeviceList")):
            container["DeviceList"] = [list(container["DeviceList"])]
        for name in _PER_DEVICE_ARRAYS:
            if name not in container:
                continue
            value = container[name]
            if value is not None and not isinstance(value, (list, tuple)):
                container[name] = [value]


def _remove_answered_rows(
    body: dict,
    view: FamilyView,
    answered: dict[tuple[str, str], tuple[PendingRow, RowAnswer]],
    n_devices: int,
) -> dict[str, _MovedRow]:
    """Cut every dropped and moved row out of its lists.

    Returns:
        The rows on their way to a field of their own, by new field name.
    """
    moved: dict[str, _MovedRow] = {}
    doomed: dict[tuple[str, str], set[int]] = {}
    for (field_name, signal), (row, answer) in answered.items():
        if answer == "device":
            continue
        source = view.fields[field_name]
        for key in source.keys:
            slots = source.raw_slots(key)
            for index in range(n_devices, len(slots)):
                if _signal(slots[index]) == signal:
                    doomed.setdefault((field_name, key), set()).add(index)
        if isinstance(answer, FieldAnswer):
            entry = moved.setdefault(answer.name, _MovedRow(field_name, row.index, {}))
            for key in row.keys:
                entry.signals[key] = signal
    for (field_name, key), indices in doomed.items():
        field_body = body[field_name]
        slots = _as_list(field_body[key])
        field_body[key] = [slot for index, slot in enumerate(slots) if index not in indices]
    return moved


def _moved_field_body(source: dict, signals: dict[str, str], n_devices: int) -> dict:
    """Return the body of a field created for one moved row.

    The row's PV sits on device 1 and every other device is blank. The source
    field's metadata comes along, because the row was one of its channels: a
    scalar as it stands, the tag and limit keys whole, and a per-device unit or
    data-type list only when its devices agree on one value.
    """

    entry: dict[str, Any] = {
        key: [signals[key]] + [_CHANNEL_BLANK] * (n_devices - 1)
        for key in CHANNEL_KEYS
        if key in signals
    }
    for key in FIELD_METADATA_KEYS:
        if key not in source:
            continue
        value = source[key]
        if key in _VERBATIM_FIELD_KEYS or not isinstance(value, (list, tuple)):
            entry[key] = copy.deepcopy(value)
        elif (
            key in _PER_DEVICE_FIELD_KEYS
            and len(value) == n_devices
            and all(item == value[0] for item in value)
        ):
            entry[key] = copy.deepcopy(value[0])
    return entry


def _rows_past(body: dict, view: FamilyView, n_devices: int) -> int:
    """Return how many rows still sit past ``n_devices`` in the longest list."""
    past = 0
    for field in FamilyView(view.system, view.raw_name, body).fields.values():
        for key in field.keys:
            slots = _as_list(field.body[key])
            for index in range(len(slots) - 1, n_devices - 1, -1):
                if _signal(slots[index]) is not None:
                    past = max(past, index + 1 - n_devices)
                    break
    return past


def _grow_device_list(body: dict, n_devices: int, new_count: int) -> None:
    """Add one ``DeviceList`` row per promoted row, counting on from the last."""
    for container in _array_containers(body):
        rows = container.get("DeviceList")
        if not isinstance(rows, list) or not rows:
            continue
        last = rows[-1]
        if not (isinstance(last, (list, tuple)) and len(last) == 2):
            continue
        sector, device = last[0], last[1]
        for step in range(1, new_count - n_devices + 1):
            number = device + step if _is_number(device) else n_devices + step
            rows.append([copy.deepcopy(sector), number])


class _CellRow:
    """One row of a parameter cell, presented as the container that holds it.

    A cell states one row per parameter and one column per device, so the
    slot a device owns sits inside a row rather than being one. Handing each
    row over under the cell's own key lets it be realigned by the same
    registry as every list whose own slots are the devices'.
    """

    def __init__(self, cell: list, index: int, key: str) -> None:
        self._cell = cell
        self._index = index
        self._key = key

    def get(self, key: str, default: Any = None) -> Any:
        """Return the row when ``key`` names the cell, else ``default``."""
        return self._cell[self._index] if key == self._key else default

    def __contains__(self, key: str) -> bool:
        return key == self._key

    def __getitem__(self, key: str) -> Any:
        if key != self._key:
            raise KeyError(key)
        return self._cell[self._index]

    def __setitem__(self, key: str, value: Any) -> None:
        if key != self._key:
            raise KeyError(key)
        self._cell[self._index] = value


def _row_blank(value: list) -> Any:
    """Return the blank a new device joins ``value`` with.

    A list of rows blanks with a row of its own width, so the shape the export
    wrote survives a device being added; a flat list blanks with one number.
    """
    first = value[0] if value else None
    if isinstance(first, (list, tuple)):
        return [_NUMBER_BLANK] * len(first)
    return _NUMBER_BLANK


def _is_cell(value: list) -> bool:
    """Whether ``value``'s two axes can be told apart.

    A cell of rows of one width has a device axis wherever the device count
    is; a square one says nothing about which of its axes that is, so it is
    left to be read along its rows like every other per-device list.
    """
    if not value or not all(isinstance(row, (list, tuple)) for row in value):
        return False
    widths = {len(row) for row in value}
    return len(widths) == 1 and widths != {len(value)}


def _numeric_rows(
    container: Any, keys: Iterable[str], *, cells: bool = False
) -> Iterator[tuple[Any, str, Any]]:
    """Yield the per-device value lists of one container, each with its blank.

    A key is yielded only where it holds a list, because a bare number is one
    value for the whole family and no list to realign, and a flat pair under a
    span key is one low/high pair rather than two devices' slots. A cell is
    offered along both of its axes where they differ in length, so the device
    count settles which one the family is realigned along; a square cell says
    nothing about which axis is which and is read along its rows, the way
    every other per-device list is read.
    """
    if not isinstance(container, dict):
        return
    for key in keys:
        value = container.get(key)
        if not isinstance(value, (list, tuple)):
            continue
        if key in _SPAN_KEYS and _is_flat_pair(value):
            continue
        rows = list(value)
        yield container, key, _row_blank(rows)
        if cells and isinstance(value, list) and _is_cell(rows):
            for index in range(len(value)):
                yield _CellRow(value, index, key), key, _NUMBER_BLANK


def _per_device_lists(body: dict, fields: Iterable[dict]) -> Iterator[tuple[Any, str, Any]]:
    """Yield every ``(container, key, blank)`` of ``body`` that holds a slot per device.

    The channel keys of each field blank as a channel list does; the field's
    unit and data-type keys and the family arrays blank as the export spells a
    missing value; a field's per-device numbers, the parameters of its
    conversions and the element indices under ``AT`` blank as the export
    spells a missing number. One registry answers for all of them, so a device
    leaves -- or joins -- every list that holds a slot for it at once.

    A key of the first two groups is yielded whether or not its container
    carries it, so each consumer decides what an absent or scalar value means
    to it. A key of the third is yielded only where the container holds a
    list, because there only the value's own shape tells a list of devices
    from one value stated for the whole family.
    """
    for field_body in fields:
        for key in CHANNEL_KEYS:
            yield field_body, key, _CHANNEL_BLANK
        for key in _PER_DEVICE_FIELD_KEYS:
            yield field_body, key, _DEVICE_BLANK
        yield from _numeric_rows(field_body, _PER_DEVICE_VALUE_KEYS)
        yield from _numeric_rows(field_body, _PER_DEVICE_PARAM_KEYS, cells=True)
        yield from _numeric_rows(field_body.get(_AT_KEY), _AT_ROW_KEYS)
    for container in _array_containers(body):
        for name in _PER_DEVICE_ARRAYS:
            yield container, name, _DEVICE_BLANK
        yield from _numeric_rows(container.get(_AT_KEY), _AT_ROW_KEYS)


def _pad(
    container: dict, key: str, n_devices: int, new_count: int, blank: Any, *, lists_only: bool
) -> None:
    """Pad one list out to the family's new device count, when it was full."""
    if key not in container:
        return
    value = container[key]
    if isinstance(value, (list, tuple)):
        slots = list(value)
    elif not lists_only and isinstance(value, str):
        slots = [value]
    else:
        return
    if n_devices <= len(slots) < new_count:
        container[key] = slots + [blank] * (new_count - len(slots))


def _pad_to_devices(body: dict, view: FamilyView, n_devices: int, new_count: int) -> None:
    """Give every list that spanned the family's devices a slot for the new ones.

    A list that already fell short of the export's device count is a gap the
    export itself states, so it is left as it is rather than quietly filled. A
    bare string under a channel key is one channel slot; a scalar under any
    other key is one value for the whole family and no list to pad.
    """
    fields = [field.body for field in FamilyView(view.system, view.raw_name, body).fields.values()]
    for container, key, blank in _per_device_lists(body, fields):
        _pad(container, key, n_devices, new_count, blank, lists_only=key not in CHANNEL_KEYS)


def _apply_rows_beyond(
    body: dict, view: FamilyView, pending: PendingJudgments, judgments: FamilyJudgments
) -> None:
    """Apply the family's ``rows_beyond_devices`` answers to ``body`` in place."""
    answered = _row_answers(pending, judgments)
    if not answered:
        return
    n_devices = view.n_devices
    _guard_row_answers(pending, answered)
    if n_devices == 1:
        _normalise_single_device(body)
    for name, row in _remove_answered_rows(body, view, answered, n_devices).items():
        body[name] = _moved_field_body(view.fields[row.field].body, row.signals, n_devices)
    new_count = n_devices + _rows_past(body, view, n_devices)
    if new_count > n_devices:
        _grow_device_list(body, n_devices, new_count)
        _pad_to_devices(body, view, n_devices, new_count)


def _unbound_answers(
    pending: PendingJudgments, judgments: FamilyJudgments
) -> dict[int, UnboundAnswer]:
    """Return the answered ordinals this system actually pends, by ordinal.

    An answer naming an ordinal the system does not pend is left out, and so is
    an undecided one: the same family is judged per system, and a device one
    export leaves unbound may be bound in another.
    """
    answered: dict[int, UnboundAnswer] = {}
    for ordinal in pending.unbound_devices:
        answer = judgments.unbound_devices.get(ordinal)
        if answer is not None:
            answered[ordinal] = answer
    return answered


def _drop_slots(container: dict, key: str, indices: frozenset[int], n_devices: int) -> None:
    """Cut the dropped devices out of one list holding a slot per device."""
    value = container.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != n_devices:
        return
    container[key] = [slot for index, slot in enumerate(value) if index not in indices]


def _drop_devices(body: dict, view: FamilyView, indices: frozenset[int], n_devices: int) -> None:
    """Remove every dropped device from the lists that span the family's devices.

    The positions are the export's own, so the slots go in one pass and the
    devices that survive keep the order and the numbering they were exported
    with. The lists come from the one registry the family is realigned by, so
    a dropped device leaves its channel name, its element index and its
    conversion parameters together; the shortened ``DeviceList`` is the
    family's new device count.
    """
    for container in _array_containers(body):
        _drop_slots(container, "DeviceList", indices, n_devices)
    fields = [field for name in view.fields if isinstance(field := body.get(name), dict)]
    for container, key, _blank in _per_device_lists(body, fields):
        _drop_slots(container, key, indices, n_devices)


def _pad_exact(container: dict, key: str, length: int, count: int, blank: Any) -> None:
    """Add ``count`` blanks to a list of exactly ``length`` entries."""
    value = container.get(key)
    if isinstance(value, (list, tuple)) and len(value) == length:
        container[key] = list(value) + [blank] * count


def _pad_kept_devices(body: dict, view: FamilyView, reach: int, count: int) -> None:
    """Give the lists that stop at the last bound device a slot per kept device.

    A list of exactly ``reach`` entries is one that stops there, and so one of
    the lists that left the kept devices unbound; padding it keeps a device
    query's channel names as long as its device rows. A list the export left
    shorter states a gap of its own, and a list already spanning the devices
    carries the kept device's slot as exported.
    """
    fields = [field for name in view.fields if isinstance(field := body.get(name), dict)]
    for container, key, blank in _per_device_lists(body, fields):
        _pad_exact(container, key, reach, count, blank)


def _apply_unbound_devices(
    body: dict, view: FamilyView, pending: PendingJudgments, judgments: FamilyJudgments
) -> None:
    """Apply the family's ``unbound_devices`` answers to ``body`` in place."""
    answered = _unbound_answers(pending, judgments)
    if not answered:
        return
    dropped = frozenset(ordinal - 1 for ordinal, answer in answered.items() if answer == "drop")
    kept = sum(1 for answer in answered.values() if answer == "keep")
    if dropped:
        _drop_devices(body, view, dropped, view.n_devices)
    if kept:
        _pad_kept_devices(body, view, _reach(view), kept)


def _owner_answers(
    pending: PendingJudgments, judgments: FamilyJudgments
) -> dict[int, tuple[SupplyGroup, int]]:
    """Return the owned supply groups this system actually pends, by lowest ordinal.

    A family answered ``keep_all``, a group answered ``keep_all``, an undecided
    answer and a group this system does not pend all leave nothing to apply:
    the same family is judged per system, and a supply one export shares may be
    one device's own in another.
    """
    answer = judgments.shared_pvs
    if not isinstance(answer, OwnerMap):
        return {}
    owned: dict[int, tuple[SupplyGroup, int]] = {}
    for group in pending.groups:
        owner = answer.owners.get(group.lowest)
        if owner is None or isinstance(owner, str):
            continue
        assert owner in group.ordinals, (
            f"supply group {group.lowest} is owned by device {owner}, which is not "
            f"one of its members {list(group.ordinals)}"
        )
        owned[group.lowest] = (group, owner)
    return owned


def _blank_shared_slots(body: dict, view: FamilyView, group: SupplyGroup, owner: int) -> None:
    """Blank the group's PVs everywhere but on the device that owns the supply.

    Only a slot carrying a PV of the group goes, so a member holding a reading
    of its own beside the shared supply keeps the reading. A one-row list is
    left alone: it broadcasts to every device and carries no member's slot.
    """
    indices = frozenset(ordinal - 1 for ordinal in group.ordinals if ordinal != owner)
    pvs = frozenset(group.pvs)
    for field in FamilyView(view.system, view.raw_name, body).fields.values():
        for key in field.keys:
            slots = _as_list(field.body[key])
            if len(slots) < 2:
                continue
            kept = [
                _CHANNEL_BLANK if index in indices and _signal(slot) in pvs else slot
                for index, slot in enumerate(slots)
            ]
            if kept != slots:
                field.body[key] = kept


def _apply_shared_pvs(
    body: dict, view: FamilyView, pending: PendingJudgments, judgments: FamilyJudgments
) -> None:
    """Apply the family's ``shared_pvs`` answer to ``body`` in place."""
    for group, owner in _owner_answers(pending, judgments).values():
        _blank_shared_slots(body, view, group, owner)


def apply_judgments(system: str, raw: str, body: dict, mapping: Mapping) -> dict:
    """Return ``body`` with the reviewer's answers for ``raw`` applied.

    The answers are read against what this system's export actually pends, so
    an answer written for the same family in another system is ignored here.

    Args:
        system: The system the body sits under.
        raw: The family's raw name, the key its answers are written under.
        body: The normalised family body, as exported; never modified.
        mapping: The mapping carrying the answers.

    Returns:
        A new body. It is a copy of ``body`` when the family has no answers
        this system pends.
    """
    judged = copy.deepcopy(body)
    judgments = mapping.judgments.get(raw)
    if judgments is None:
        return judged
    raw_view = FamilyView(system, raw, body)
    pending = pending_judgments(raw_view)
    _apply_rows_beyond(judged, raw_view, pending, judgments)
    _apply_unbound_devices(judged, raw_view, pending, judgments)
    _apply_shared_pvs(judged, raw_view, pending, judgments)
    return judged


def judged_family_views(system: str, body: dict, mapping: Mapping) -> Iterator[FamilyView]:
    """Yield one family view per family of one system, the answers applied.

    Args:
        system: The raw system token the body sits under.
        body: The system body, keyed by raw family token plus bookkeeping keys;
            never modified.
        mapping: The mapping carrying the answers.

    Yields:
        The families in ``body`` key order, each seen through its judged body,
        so a consumer reads the grain the reviewer settled on rather than the
        export's.
    """
    for view in family_views(system, body):
        yield FamilyView(
            system, view.raw_name, apply_judgments(system, view.raw_name, view.body, mapping)
        )


def _va_families(system: str, va_json: dict) -> dict:
    """Return the family blocks of one system's virtual-accelerator document."""
    block = va_json.get(system, va_json)
    if not isinstance(block, dict):
        return {}
    families = block.get("families")
    return families if isinstance(families, dict) else {}


def _va_device_count(block: dict) -> int:
    """Return how many devices the block's rows were written for."""
    rows = block.get("device_list")
    if not isinstance(rows, (list, tuple)):
        return 0
    return 1 if _is_flat_pair(rows) else len(rows)


def _va_per_device_lists(block: dict) -> Iterator[tuple[Any, str, Any]]:
    """Yield every ``(container, key, blank)`` of one VA block holding a slot per device.

    The block's device list, the nominal of each field, the per-device
    parameters of its calibration and of its monitor inverse and what its
    readings are corrected by all carry one entry per device, a sampled
    conversion one row of points per device. The energy table describes the
    ramp of the one device it names rather than a slot per device, so it is
    not one of them.
    """
    yield from _numeric_rows(block, ("device_list",))
    nominals = block.get("nominals")
    if isinstance(nominals, dict):
        for nominal in nominals.values():
            yield from _numeric_rows(nominal, _VA_NOMINAL_ROW_KEYS)
    for field_block in block.values():
        if not isinstance(field_block, dict):
            continue
        for key in _VA_CONVERSION_KEYS:
            yield from _numeric_rows(field_block.get(key), _VA_CONVERSION_ROW_KEYS)
        yield from _numeric_rows(field_block.get(_VA_READOUT_KEY), _VA_READOUT_ROW_KEYS)


def judged_va_block(
    system: str,
    family: str,
    va_json: dict,
    mapping: Mapping,
    *,
    devices: int | None = None,
) -> dict:
    """Return one family's ``va.json`` block in the judged device order.

    The block holds one row per device the family was exported with -- its
    device list, its nominals, its element indices and the per-device
    parameters of every conversion -- so an answer that changed the family's
    devices changes its rows too, and a consumer reads one device order across
    the export's two documents.

    Args:
        system: The system the block sits under, where the document holds one
            block per system; a document that is already one system's is read
            as it stands.
        family: The family's raw name, the key its answers are written under.
        va_json: The imported virtual-accelerator document; never modified.
        mapping: The mapping carrying the answers.
        devices: How many devices the judged family has, where the caller has
            judged it already. The rows are padded out to it, because a row
            promoted to a device was no device when the export sampled them.
            Left out, the block keeps the devices its own list states, less
            the dropped ones.

    Returns:
        A new block.

    Raises:
        KeyError: The document holds no block for that system and family.
    """
    families = _va_families(system, va_json)
    if family not in families:
        raise KeyError(f"{system!r} has no virtual-accelerator block for family {family!r}")
    judged: dict = copy.deepcopy(families[family])
    judgments = mapping.judgments.get(family)
    if judgments is None:
        return judged
    n_devices = _va_device_count(judged)
    dropped = frozenset(
        ordinal - 1
        for ordinal, answer in judgments.unbound_devices.items()
        if answer == "drop" and 1 <= ordinal <= n_devices
    )
    for container, key, _blank in _va_per_device_lists(judged):
        _drop_slots(container, key, dropped, n_devices)
    kept = n_devices - len(dropped)
    if devices is not None and devices > kept:
        for container, key, blank in _va_per_device_lists(judged):
            _pad_exact(container, key, kept, devices - kept, blank)
    return judged
