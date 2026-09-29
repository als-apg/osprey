"""Stage S2: combine every layer's records, apply fixes.yaml, fill the defaults.

The merge works per (kind, id, field). Each layer states only the fields its
source states; combine collects every layer's present value and stops on
nothing while collecting. A field is a top-level slot of the record, and a
map-valued slot (``simulation``, ``settings``, ``engine``, ``calibration``,
``span``, ``measurement``, ``attributes``, ``signals``) is one value, compared
whole like a list.

Then the fixes apply as a set — ``set`` and ``add`` first, then ``drop`` — and
the conflict check runs over what is left: two present unequal values on a
field no ``set`` names stop the build (``layer-conflict``). Last, the schema
defaults are filled and recorded in each record's provenance.

``fixes.yaml``::

    schema: osprey.facility.fixes/1
    fixes:
      - op: set                      # set | add | drop
        kind: channel                # place | device | channel | wiring | group | model
        id: SR:QF:SP
        fields: {role: setpoint}     # set: the fields it replaces, each whole
        was: {role: {mml: readback}} # set: per field, every layer's present value
        why: The export lists the setpoint as a monitor.
      - op: add
        kind: device
        id: SR/QF9
        record: {class: Quadrupole}  # add: the record's slots
        why: Missing from the export.

Every list compares and emits in source order except the set-valued slots,
which compare and emit sorted by their string form. Records emit sorted by id,
models by name with ``texture`` last, wiring inside its model by id. The result
does not depend on the order of fixes.yaml.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import yaml

from osprey.facility import TEXTURE
from osprey.facility.errors import FacilityBuildError, quoted_slots
from osprey.facility.provenance import build_provenance
from osprey.facility.sources import AUTHORED, COMPUTED_SLOTS, Sources

__all__ = [
    "FIXES_FILE",
    "FIXES_HEADER",
    "SET_VALUED",
    "CombineResult",
    "combine",
]

FIXES_FILE = "fixes.yaml"

#: The header fixes.yaml carries.
FIXES_HEADER = "osprey.facility.fixes/1"

#: The slots whose lists are sets: compared and emitted sorted by string form.
SET_VALUED: dict[str, frozenset[str]] = {
    "channel": frozenset({"tags", "former_addresses", "endpoint_of"}),
    "group": frozenset({"members"}),
}

_OPS = ("set", "add", "drop")
_KINDS = ("place", "device", "channel", "wiring", "group", "model")
_FIX_KEYS = frozenset({"op", "kind", "id", "fields", "record", "why", "was"})

_ROLE = "readback"
_VALUE_TYPE = "float"
_BOOL_OPTIONS = ("FALSE", "TRUE")
_SLICE_WEIGHT = 1

#: The file whose name the list of each kind emits under.
_PLURAL = {
    "place": "places",
    "device": "devices",
    "channel": "channels",
    "group": "groups",
    "model": "models",
}


@dataclass(frozen=True)
class _Fix:
    position: int
    op: str
    kind: str
    id: str
    why: str
    fields: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)
    was: dict[str, Any] | None = None

    def entry(self) -> dict[str, Any]:
        return {"op": self.op, "kind": self.kind, "id": self.id, "why": self.why}


@dataclass
class _Value:
    layer: str | None
    file: str | None
    value: Any
    filled: frozenset[int] = frozenset()


@dataclass
class _Record:
    kind: str
    id: str
    values: dict[str, list[_Value]] = field(default_factory=dict)
    sources: list[tuple[str, str, list[str]]] = field(default_factory=list)
    set_fields: dict[str, _Value] = field(default_factory=dict)
    fix: _Fix | None = None

    def layers(self) -> set[str]:
        return {layer for layer, _file, _fields in self.sources}

    def files(self) -> list[str]:
        return sorted({file for _layer, file, _fields in self.sources})

    def current(self, name: str) -> list[Any]:
        """Every value the field holds once the sets have applied."""
        if name in self.set_fields:
            return [self.set_fields[name].value]
        return [v.value for v in self.values.get(name, ())]


@dataclass
class CombineResult:
    """What stage S2 returns.

    Attributes:
        document: The combined records in facility-file shape: ``identity``
            (when a source states one), ``classes``, ``places``, ``devices``,
            ``channels``, ``groups``, ``models`` (each with its ``wiring``),
            ``limits`` (when a source states one) and ``scenarios``. Complete
            only when ``errors`` is empty.
        errors: Every stop found.
        dropped: ``(kind, id)`` of each record a ``drop`` removed, to the fix
            that removed it (a channel's or model's wiring records name the
            channel's or model's fix).
    """

    document: dict[str, Any]
    errors: list[FacilityBuildError]
    dropped: dict[tuple[str, str], dict[str, Any]]


def combine(sources: Sources) -> CombineResult:
    """Combine the layers and apply the fixes (stage S2).

    Args:
        sources: What stage S1 read, with no S1 stop.

    Returns:
        The combined document and every stop found: ``fix-missing``,
        ``fix-stale``, ``fix-duplicate``, ``fix-computed``, ``fix-authored``,
        ``fix-referenced``, ``layer-conflict``, and ``source-invalid`` for a
        fixes.yaml entry of the wrong shape.
    """
    return _Combiner(sources).run()


def _same(a: Any, b: Any) -> bool:
    """Value equality in which a boolean never equals a number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _show(value: Any) -> str:
    """One-line YAML for a value in an error line."""
    text = yaml.safe_dump(value, default_flow_style=True, sort_keys=True, width=1 << 30)
    return text.strip().removesuffix("...").strip()


def _normalize(kind: str, name: str, value: Any) -> tuple[Any, frozenset[int]]:
    """A value in the form it is compared and emitted in.

    Set-valued lists are sorted by string form, a measurement's ``kinds`` too,
    and each slice without a ``weight`` gets the constant default.

    Returns:
        The value and the positions of the slices whose weight was filled.
    """
    value = copy.deepcopy(value)
    if name in SET_VALUED.get(kind, ()) and isinstance(value, list):
        return sorted(value, key=str), frozenset()
    if kind == "model" and name == "measurement" and isinstance(value, dict):
        kinds = value.get("kinds")
        if isinstance(kinds, list):
            value["kinds"] = sorted(kinds, key=str)
        return value, frozenset()
    if kind == "wiring" and name == "slices" and isinstance(value, list):
        filled = set()
        for position, piece in enumerate(value):
            if isinstance(piece, dict) and "weight" not in piece:
                piece["weight"] = _SLICE_WEIGHT
                filled.add(position)
        return value, frozenset(filled)
    return value, frozenset()


def _id_key(kind: str) -> str:
    return "name" if kind == "model" else "id"


class _Combiner:
    def __init__(self, sources: Sources) -> None:
        self.sources = sources
        self.errors: list[FacilityBuildError] = []
        self.records: dict[tuple[str, str], _Record] = {}
        self.dropped: dict[tuple[str, str], dict[str, Any]] = {}

    def run(self) -> CombineResult:
        fixes = self._parse_fixes()
        self._collect()
        fixes = self._drop_duplicate_fixes(fixes)
        for fix in sorted(fixes, key=lambda f: f.kind == "wiring"):
            if fix.op == "add":
                self._apply_add(fix)
        self._attach_seeds_and_measurement()
        drops: list[_Fix] = []
        for fix in fixes:
            if fix.op == "set":
                self._apply_set(fix)
            elif fix.op == "drop" and self._check_target(fix):
                drops.append(fix)
        self._apply_drops(drops)
        self._check_conflicts()
        document = self._emit()
        return CombineResult(document, self.errors, self.dropped)

    # --- fixes.yaml shape -----------------------------------------------------

    def _parse_fixes(self) -> list[_Fix]:
        data = self.sources.fixes
        if data is None:
            return []
        if not isinstance(data, dict):
            self._shape_error(FIXES_FILE, "is not a mapping", "write it as a mapping")
            return []
        for key in sorted(set(data) - {"schema", "fixes"}, key=str):
            self._shape_error(f"{FIXES_FILE}.{key}", "is an unknown key", f"remove `{key}`")
        if data.get("schema") != FIXES_HEADER:
            self._shape_error(
                f"{FIXES_FILE}.schema",
                f"header is {data.get('schema')!r}",
                f"write `schema: {FIXES_HEADER}`",
            )
        entries = data.get("fixes") or []
        if not isinstance(entries, list):
            self._shape_error(f"{FIXES_FILE}.fixes", "is not a list", "write it as a list")
            return []
        fixes = []
        for position, entry in enumerate(entries):
            fix = self._parse_fix(position, entry)
            if fix is not None:
                fixes.append(fix)
        return fixes

    def _parse_fix(self, position: int, entry: Any) -> _Fix | None:
        where = f"{FIXES_FILE}.fixes.{position}"
        if not isinstance(entry, dict):
            self._shape_error(where, "is not a mapping", "write the fix as a mapping")
            return None
        problems = [f"unknown key `{k}`" for k in sorted(set(entry) - _FIX_KEYS, key=str)]
        op: Any = entry.get("op")
        kind: Any = entry.get("kind")
        rid: Any = entry.get("id")
        why: Any = entry.get("why")
        if op not in _OPS:
            problems.append(f"`op` {op!r} is not one of {', '.join(_OPS)}")
        if kind not in _KINDS:
            problems.append(f"`kind` {kind!r} is not one of {', '.join(_KINDS)}")
        if not isinstance(rid, str) or not rid:
            problems.append("`id` is not a string")
        if not isinstance(why, str) or not why.strip():
            problems.append("`why` is missing")
        present = {k for k in ("fields", "record", "was") if k in entry}
        allowed = {"set": {"fields", "was"}, "add": {"record"}, "drop": set()}.get(op, present)
        problems += [f"`{k}` does not belong on `{op}`" for k in sorted(present - allowed)]
        fields: Any = entry.get("fields")
        if op == "set" and not (isinstance(fields, dict) and fields):
            problems.append("`fields` is not a non-empty mapping")
        elif op == "set" and kind in _KINDS:
            for name in (_id_key(kind), "wiring" if kind == "model" else None):
                if name is not None and name in fields:
                    problems.append(f"`fields` names `{name}`")
        record = entry.get("record")
        if op == "add" and not isinstance(record, dict):
            problems.append("`record` is not a mapping")
        was = entry.get("was")
        if was is not None and not (
            isinstance(was, dict) and all(isinstance(v, dict) for v in was.values())
        ):
            problems.append("`was` is not a mapping of field to {layer: value}")
        if problems:
            self._shape_error(where, "; ".join(problems), "fix the entry")
            return None
        return _Fix(
            position,
            op,
            kind,
            rid,
            why,
            fields=dict(fields or {}),
            record=dict(record or {}),
            was=None if "was" not in entry else dict(was or {}),
        )

    # --- collecting -----------------------------------------------------------

    def _record(self, kind: str, rid: str) -> _Record:
        key = (kind, rid)
        if key not in self.records:
            self.records[key] = _Record(kind, rid)
        return self.records[key]

    def _state(
        self, record: _Record, layer: str | None, file: str | None, fields: dict[str, Any]
    ) -> None:
        for name, raw in fields.items():
            value, filled = _normalize(record.kind, name, raw)
            record.values.setdefault(name, []).append(_Value(layer, file, value, filled))
        if layer is not None and file is not None:
            record.sources.append((layer, file, list(fields)))

    def _collect(self) -> None:
        for source in self.sources.records:
            self._state(
                self._record(source.kind, source.id), source.layer, source.file, source.fields
            )

    def _attach_seeds_and_measurement(self) -> None:
        for address, seed in sorted(self.sources.seeds.items()):
            target = self.records.get(("channel", address))
            if target is not None:
                self._state(target, AUTHORED, "seeds.yaml", {"simulation": seed})
        for model, measurement in sorted(self.sources.measurement.items()):
            target = self.records.get(("model", model))
            if target is not None:
                rel = f"measurement/{model}.yaml"
                self._state(target, AUTHORED, rel, {"measurement": measurement})

    # --- fixes ------------------------------------------------------------------

    def _drop_duplicate_fixes(self, fixes: list[_Fix]) -> list[_Fix]:
        by_target: dict[tuple[str, str], list[_Fix]] = {}
        for fix in fixes:
            by_target.setdefault((fix.kind, fix.id), []).append(fix)
        kept = []
        for fix in fixes:
            group = by_target[(fix.kind, fix.id)]
            if len(group) == 1:
                kept.append(fix)
            elif fix is group[0]:
                ops = " and ".join(f"`{f.op}`" for f in group)
                self._error(
                    "fix-duplicate",
                    fix,
                    [FIXES_FILE],
                    f"fixes.yaml has {ops} for it",
                    "keep one fix per record in fixes.yaml",
                )
        return kept

    def _apply_add(self, fix: _Fix) -> None:
        key = (fix.kind, fix.id)
        existing = self.records.get(key)
        if existing is not None:
            layers = ", ".join(sorted(existing.layers()))
            self._error(
                "fix-duplicate",
                fix,
                [FIXES_FILE, *existing.files()],
                f"`add` of a record layer {layers} already produces",
                "change the fix to `set`",
            )
            return
        record = dict(fix.record)
        id_key = _id_key(fix.kind)
        if id_key in record and record.pop(id_key) != fix.id:
            self._shape_error(
                f"{FIXES_FILE}.fixes.{fix.position}.record.{id_key}",
                f"`{id_key}` is not the fix's id {fix.id}",
                f"remove `{id_key}` from `record`",
            )
            return
        wiring = record.pop("wiring", None) if fix.kind == "model" else None
        adds = [(fix.kind, fix.id, record)]
        if fix.kind == "wiring":
            model, _, address = fix.id.partition("/")
            if record.setdefault("address", address) != address or not model:
                self._shape_error(
                    f"{FIXES_FILE}.fixes.{fix.position}.id",
                    f"wiring id {fix.id} is not <model>/<address of the record>",
                    "write the id as <model>/<address>",
                )
                return
            if ("model", model) not in self.records:
                self._error(
                    "fix-missing",
                    fix,
                    [FIXES_FILE],
                    f"model {model} does not exist",
                    "add the model or remove the fix from fixes.yaml",
                )
                return
        for entry in wiring if isinstance(wiring, list) else []:
            wired = entry.get("address") if isinstance(entry, dict) else None
            if not isinstance(wired, str):
                self._shape_error(
                    f"{FIXES_FILE}.fixes.{fix.position}.record.wiring",
                    "has an entry without a string `address`",
                    "add `address`",
                )
                return
            adds.append(("wiring", f"{fix.id}/{wired}", dict(entry)))
        for kind, rid, fields in adds:
            computed = sorted(set(fields) & COMPUTED_SLOTS[kind])
            if computed:
                self._error(
                    "fix-computed",
                    fix,
                    [FIXES_FILE],
                    f"`add` writes {quoted_slots(computed)}, which the build computes",
                    f"remove {quoted_slots(computed)} from the fix",
                )
                return
            if (kind, rid) in self.records:
                self._error(
                    "fix-duplicate",
                    fix,
                    [FIXES_FILE, *self.records[(kind, rid)].files()],
                    f"`add` of {kind} {rid}, which a layer already produces",
                    "change the fix to `set`",
                )
                return
        for kind, rid, fields in adds:
            target = self._record(kind, rid)
            fields.pop("id", None)
            self._state(target, None, None, fields)
            target.fix = fix

    def _check_target(self, fix: _Fix) -> _Record | None:
        target = self.records.get((fix.kind, fix.id))
        if target is None:
            self._missing(fix)
            return None
        if target.layers() == {AUTHORED}:
            files = target.files()
            self._error(
                "fix-authored",
                fix,
                [FIXES_FILE, *files],
                f"`{fix.op}` targets a record only {AUTHORED} sources state",
                f"edit data/facility/{files[0]}",
            )
            return None
        return target

    def _missing(self, fix: _Fix) -> None:
        address = fix.id
        prefix = ""
        if fix.kind == "wiring":
            model, _, address = fix.id.partition("/")
            prefix = f"{model}/"
        moved = None
        if fix.kind in ("channel", "wiring"):
            for (kind, rid), record in sorted(self.records.items()):
                if kind != "channel":
                    continue
                if any(address in (v or []) for v in record.current("former_addresses")):
                    moved = rid
                    break
        if moved is None:
            self._error(
                "fix-missing",
                fix,
                [FIXES_FILE],
                f"no {fix.kind} {fix.id} exists",
                "remove the fix from fixes.yaml",
            )
            return
        self._error(
            "fix-missing",
            fix,
            [FIXES_FILE],
            f"no {fix.kind} {fix.id} exists; channel {moved} lists {address} in former_addresses",
            f"point the fix at {fix.kind} {prefix}{moved}",
        )

    def _apply_set(self, fix: _Fix) -> None:
        target = self._check_target(fix)
        if target is None:
            return
        computed = sorted(set(fix.fields) & COMPUTED_SLOTS[fix.kind])
        if computed:
            self._error(
                "fix-computed",
                fix,
                [FIXES_FILE],
                f"`set` names {quoted_slots(computed)}, which the build computes",
                f"remove {quoted_slots(computed)} from the fix",
            )
            return
        current: dict[str, dict[str, Any]] = {}
        for name in sorted(fix.fields):
            values = target.values.get(name, [])
            stated = {v.layer: v.value for v in values if v.layer is not None}
            if stated:
                current[name] = stated
        if not self._was_matches(fix, current):
            corrected = {
                "op": "set",
                "kind": fix.kind,
                "id": fix.id,
                "fields": fix.fields,
                "was": current,
                "why": fix.why,
            }
            if not current:
                del corrected["was"]
            state = "is missing" if fix.was is None else "does not match the layers"
            self._error(
                "fix-stale",
                fix,
                [FIXES_FILE, *target.files()],
                f"`was` {state}",
                f"replace the fix with {_show(corrected)}",
            )
            return
        for name, raw in fix.fields.items():
            value, filled = _normalize(fix.kind, name, raw)
            target.set_fields[name] = _Value(None, None, value, filled)
        target.fix = fix

    def _was_matches(self, fix: _Fix, current: dict[str, dict[str, Any]]) -> bool:
        if fix.was is None:
            return not current
        was = {
            name: {layer: _normalize(fix.kind, name, value)[0] for layer, value in layers.items()}
            for name, layers in fix.was.items()
        }
        return _same(was, current)

    def _apply_drops(self, drops: list[_Fix]) -> None:
        for fix in sorted(drops, key=lambda f: (f.kind, f.id)):
            if (fix.kind, fix.id) not in self.records:
                continue
            self._remove((fix.kind, fix.id), fix)
            if fix.kind in ("channel", "model"):
                for kind, rid in sorted(self.records):
                    if kind != "wiring":
                        continue
                    model, _, address = rid.partition("/")
                    if (fix.kind == "channel" and address == fix.id) or (
                        fix.kind == "model" and model == fix.id
                    ):
                        self._remove((kind, rid), fix)
        for fix in drops:
            referrers = sorted(set(self._referrers(fix.kind, fix.id)))
            if referrers:
                self._error(
                    "fix-referenced",
                    fix,
                    [FIXES_FILE],
                    f"`drop` leaves {len(referrers)} referrer(s): {'; '.join(referrers)}",
                    "drop or re-point each referrer by a fix",
                )

    def _remove(self, key: tuple[str, str], fix: _Fix) -> None:
        # A record's own drop names it; otherwise the first drop in (kind, id) order does.
        self.records.pop(key, None)
        if key == (fix.kind, fix.id) or key not in self.dropped:
            self.dropped[key] = fix.entry()

    def _live(self, kind: str) -> Iterator[_Record]:
        return (record for (k, _rid), record in sorted(self.records.items()) if k == kind)

    def _referrers(self, kind: str, rid: str) -> Iterator[str]:
        if kind == "device":
            yield from self._on_referrers("device", rid)
            for channel in self._live("channel"):
                if any(rid in (v or []) for v in channel.current("endpoint_of")):
                    yield f"channel {channel.id} (endpoint_of)"
            for group in self._live("group"):
                if any(rid in (v or []) for v in group.current("members")):
                    yield f"group {group.id} (members)"
            for wiring in self._live("wiring"):
                for slices in wiring.current("slices"):
                    if any(isinstance(s, dict) and s.get("device") == rid for s in slices or []):
                        yield f"wiring {wiring.id} (slices)"
        elif kind == "place":
            yield from self._on_referrers("place", rid)
            for device in self._live("device"):
                if rid in device.current("place"):
                    yield f"device {device.id} (place)"
            for place in self._live("place"):
                if "/" in place.id and place.id.rsplit("/", 1)[0] == rid:
                    yield f"place {place.id} (parent)"
        elif kind == "model":
            for place in self._live("place"):
                if any(
                    isinstance(s, dict) and s.get("model") == rid for s in place.current("span")
                ):
                    yield f"place {place.id} (span)"
            for scenario in self.sources.scenarios:
                faults = scenario.get("faults")
                if isinstance(faults, dict) and rid in faults:
                    yield f"scenarios/{scenario['name']}.yaml (faults)"
            if rid in self.sources.measurement:
                yield f"measurement/{rid}.yaml"
        elif kind == "group":
            for model, measurement in sorted(self.sources.measurement.items()):
                groups = measurement.get("groups")
                if ("model", model) in self.records and isinstance(groups, dict):
                    if rid in groups.values():
                        yield f"measurement/{model}.yaml (groups)"

    def _on_referrers(self, kind: str, rid: str) -> Iterator[str]:
        for channel in self._live("channel"):
            if any(isinstance(on, dict) and on.get(kind) == rid for on in channel.current("on")):
                yield f"channel {channel.id} (on)"

    # --- conflicts --------------------------------------------------------------

    def _check_conflicts(self) -> None:
        for (_kind, _rid), record in sorted(self.records.items()):
            for name in sorted(record.values):
                if name in record.set_fields:
                    continue
                values = record.values[name]
                if all(_same(values[0].value, v.value) for v in values[1:]):
                    continue
                shown = "; ".join(f"{v.layer}={_show(v.value)}" for v in values)
                self.errors.append(
                    FacilityBuildError(
                        "layer-conflict",
                        record.id,
                        record.files(),
                        f"add a `set` fix for `{name}` to fixes.yaml",
                        record_kind=record.kind,
                        detail=f"`{name}` differs: {shown}",
                    )
                )

    # --- emission ---------------------------------------------------------------

    def _merged(self, record: _Record) -> tuple[dict[str, Any], set[str]]:
        fields: dict[str, Any] = {}
        defaults: set[str] = set()
        for name in sorted(set(record.values) | set(record.set_fields)):
            chosen = record.set_fields.get(name)
            candidates = [chosen] if chosen is not None else record.values[name]
            fields[name] = copy.deepcopy(candidates[0].value)
            if name == "slices" and frozenset.intersection(*(c.filled for c in candidates)):
                defaults.add("slices.weight")
        return fields, defaults

    def _emit_record(
        self, record: _Record, fields: dict[str, Any], defaults: Iterable[str]
    ) -> dict[str, Any]:
        fixes = [(record.fix.op, record.fix.why)] if record.fix is not None else []
        provenance = build_provenance(sources=record.sources, fixes=fixes, defaults=defaults)
        return {_id_key(record.kind): record.id, **fields, "provenance": provenance}

    def _emit(self) -> dict[str, Any]:
        merged = {key: self._merged(record) for key, record in sorted(self.records.items())}
        for (kind, rid), (fields, defaults) in merged.items():
            if kind == "channel":
                _channel_defaults(rid, fields, defaults)
        for (kind, rid), (fields, defaults) in merged.items():
            if kind == "wiring":
                channel = merged.get(("channel", rid.partition("/")[2]))
                on = channel[0].get("on") if channel is not None else None
                device = on.get("device") if isinstance(on, dict) else None
                _slice_devices(device, fields, defaults)
        emitted = {
            key: self._emit_record(self.records[key], fields, defaults)
            for key, (fields, defaults) in merged.items()
        }
        document: dict[str, Any] = {}
        if self.sources.identity is not None:
            document["identity"] = copy.deepcopy(self.sources.identity)
        document["classes"] = sorted(
            copy.deepcopy(self.sources.classes), key=lambda row: str(row.get("class"))
        )
        for kind in ("place", "device", "channel", "group"):
            document[_PLURAL[kind]] = [rec for (k, _), rec in emitted.items() if k == kind]
        models = []
        for (kind, name), model in emitted.items():
            if kind != "model":
                continue
            wiring = [
                rec
                for (k, rid), rec in emitted.items()
                if k == "wiring" and rid.partition("/")[0] == name
            ]
            provenance = model.pop("provenance")
            models.append({**model, "wiring": wiring, "provenance": provenance})
        document["models"] = sorted(models, key=lambda m: (m["name"] == TEXTURE, m["name"]))
        if self.sources.limits is not None:
            document["limits"] = copy.deepcopy(self.sources.limits)
        document["scenarios"] = copy.deepcopy(self.sources.scenarios)
        return document

    # --- errors -----------------------------------------------------------------

    def _shape_error(self, path: str, detail: str, remedy: str) -> None:
        self.errors.append(
            FacilityBuildError(
                "source-invalid", path, [FIXES_FILE], remedy, record_kind="path", detail=detail
            )
        )

    def _error(self, kind: str, fix: _Fix, files: list[str], detail: str, remedy: str) -> None:
        self.errors.append(
            FacilityBuildError(kind, fix.id, files, remedy, record_kind=fix.kind, detail=detail)
        )


def _channel_defaults(address: str, fields: dict[str, Any], defaults: set[str]) -> None:
    """Fill a channel's schema defaults, each recorded as a default."""
    if "role" not in fields:
        fields["role"] = _ROLE
        defaults.add("role")
    if "value_type" not in fields:
        fields["value_type"] = _VALUE_TYPE
        defaults.add("value_type")
    if "on" not in fields:
        defaults.add("on")
    if fields["role"] == "setpoint" and "pair" not in fields:
        fields["pair"] = address
        defaults.add("pair")
    if fields["value_type"] == "bool" and "options" not in fields:
        fields["options"] = list(_BOOL_OPTIONS)
        defaults.add("options")


def _slice_devices(device: Any, fields: dict[str, Any], defaults: set[str]) -> None:
    """Give each slice without a ``device`` the channel's device."""
    slices = fields.get("slices")
    if device is None or not isinstance(slices, list):
        return
    for piece in slices:
        if isinstance(piece, dict) and "device" not in piece:
            piece["device"] = device
            defaults.add("slices.device")
