"""The mml layer's seed-once files: authored files an import creates when absent.

The layer's records under ``imported/mml/`` are rewritten by every import. The
files here are authored: a person owns them, and the import only writes one
that does not exist yet, opening it with :data:`HEADER`. A file that exists is
never rewritten, whoever wrote it.

What :func:`seed_once` writes, relative to ``data/facility/``:

* ``limits.yaml``: one record per setpoint whose field states a ``Range``,
  holding the band as stated (:func:`band`). A setpoint a model wires carries
  ``writable: true`` beside a band with both edges; every other setpoint
  carries its band only. The band is never widened to reach a nominal: a
  nominal outside it is the build's ``seed-invalid`` stop. A wired setpoint
  whose ``Range`` states no finite band on both edges is not made writable,
  and the import names it.
* ``seeds.yaml``: the nominal the export states for each channel no model
  wires. A wired channel starts from its deck, so its nominal is not written
  and the import says how many it skipped.
* ``measurement/<model>.yaml``: for each model that carries wiring, the groups
  and instruments its wiring names, the measurements they allow and pyAML's
  step and settle keys (:data:`TUNING`).
* ``classes.yaml``: one row per class of the mapping the vocabulary lacks,
  each declared branch ahead of the classes that extend it.
* ``identity.yaml``: the mapping's ``facility:`` block, which is then removed
  from ``mapping.yaml``; every other line of the mapping stays as written.

When ``limits.yaml`` exists, each record whose band differs from the export's
is reported and left as it is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    Mapping,
    Model,
    dump_mapping,
    field_roles,
)

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.facility.layers.mml.importer import Exports
    from osprey.services.mml.family import FamilyView, FieldView

__all__ = [
    "CLASSES_FILE",
    "HEADER",
    "IDENTITY_FILE",
    "LIMITS_FILE",
    "MEASUREMENT_DIR",
    "SEEDS_FILE",
    "TUNING",
    "Seeded",
    "band",
    "seed_once",
]

#: The first line of every file the layer seeds.
HEADER = "# Seeded once by the mml import; edit this file by hand."

LIMITS_FILE = "limits.yaml"
SEEDS_FILE = "seeds.yaml"
CLASSES_FILE = "classes.yaml"
IDENTITY_FILE = "identity.yaml"

#: The directory of the per-model measurement files.
MEASUREMENT_DIR = "measurement"

#: pyAML's step and settle keys: counts, the fit order, sleeps in s, and the
#: corrector (rad), quadrupole (1/m), sextupole (1/m**2) and RF (Hz) steps.
TUNING: dict[str, int | float] = {
    "n_step": 5,
    "n_avg_meas": 1,
    "fit_order": 2,
    "singular_values": 16,
    "sleep_between_step": 0.0,
    "sleep_between_meas": 0.0,
    "corrector_delta": 1.0e-5,
    "quad_delta": 1.0e-3,
    "sextu_delta": 1.0e-2,
    "frequency_delta": 100.0,
}

#: The field key a family states its operating band under.
_RANGE_KEY = "Range"

#: What a nominal's ``units`` reads when it is not a hardware value.
_PHYSICS_UNITS = "physics"

#: The mapping key that holds the identity block.
_FACILITY_KEY = "facility"

_SETPOINT = "setpoint"
_SINGLE_PASS = "single_pass"

#: Measurement group role -> the engine block of the family that fills it.
_GROUP_ENGINES: tuple[tuple[str, tuple[str, int]], ...] = (
    ("hcor", ("KickAngle", 0)),
    ("vcor", ("KickAngle", 1)),
    ("quad", ("PolynomB", 1)),
    ("sext", ("PolynomB", 2)),
)

#: The engine attribute of the family whose setpoint is the ``rf`` instrument.
_RF_ATTRIBUTE = "Frequency"


@dataclass
class Seeded:
    """What one seeding pass did.

    Attributes:
        written: Every file written, in write order.
        lines: What the import prints, one line each.
    """

    written: list[Path] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Claim:
    """The family field an address is a channel of, and its place in it."""

    view: FamilyView
    fld: FieldView
    indices: tuple[int, ...]
    role: str | None


def seed_once(
    exports: Exports,
    mapping: Mapping,
    facility_dir: Path,
    views: Sequence[FamilyView],
) -> Seeded:
    """Write each seed-once file that does not exist yet.

    Runs after the layer's records are written: the wired addresses are read
    from the layer's ``models.yaml``.

    Args:
        exports: What the import read.
        mapping: The layer's mapping, every slot decided.
        facility_dir: The ``data/facility`` directory.
        views: The judged families that carry channel records, in the order
            the importer wrote their channels.

    Returns:
        The files written and the lines to print.
    """
    from osprey.facility.layers.mml.importer import LAYER_DIR

    models = _load(facility_dir / LAYER_DIR / "models.yaml") or []
    wired = {str(record["address"]) for model in models for record in model.get("wiring") or ()}
    claims = _claims(views, mapping)
    seeded = Seeded()

    records, unbanded = _limit_records(claims, wired)
    limits = facility_dir / LIMITS_FILE
    if limits.exists():
        seeded.lines.extend(_differences(_load(limits), records))
    else:
        if records:
            seeded.written.append(_write(limits, {"records": records}))
        seeded.lines.extend(unbanded)

    seeds = facility_dir / SEEDS_FILE
    if not seeds.exists():
        golden = _golden(claims, exports)
        skipped = sorted(set(golden) & wired)
        kept = {address: {"nominal": golden[address]} for address in sorted(set(golden) - wired)}
        if kept:
            seeded.written.append(_write(seeds, kept))
        seeded.lines.append(f"golden skipped: {len(skipped)} wired channels")

    by_name = {model.name: model for model in mapping.models.values()}
    carried: dict[str, set[str]] = {}
    for view in views:
        carried.setdefault(view.system, set()).add(view.raw_name)
    for entry in models:
        model = by_name.get(str(entry.get("name")))
        path = facility_dir / MEASUREMENT_DIR / f"{entry.get('name')}.yaml"
        if model is None or not entry.get("wiring") or path.exists():
            continue
        document = _measurement(model, entry, mapping, carried.get(model.raw, set()), claims)
        seeded.written.append(_write(path, document))

    classes = facility_dir / CLASSES_FILE
    if not classes.exists():
        rows = _class_rows(mapping)
        if rows:
            seeded.written.append(_write(classes, rows))

    seeded.written.extend(_identity(mapping, facility_dir, seeded))
    return seeded


# -- channels -----------------------------------------------------------------


def _claims(views: Iterable[FamilyView], mapping: Mapping) -> dict[str, _Claim]:
    """Each address under the first family field that names it, as its channel record is."""
    roles = field_roles(mapping)
    claims: dict[str, _Claim] = {}
    for view in views:
        for fld in view.fields.values():
            role = roles.get(f"{view.raw_name}.{fld.name}")
            found: dict[str, list[int]] = {}
            for key in fld.keys:
                for index, slot in enumerate(fld.slots(key)[: view.n_devices]):
                    address = _text(slot)
                    if address is None or address in claims:
                        continue
                    positions = found.setdefault(address, [])
                    if index not in positions:
                        positions.append(index)
            for address, positions in found.items():
                claims[address] = _Claim(
                    view, fld, tuple(positions), None if role is None else role.role
                )
    return claims


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    """A finite number as a float, or ``None``; an export spells a non-finite one as text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# -- limits -------------------------------------------------------------------


def band(declared: Any, indices: Sequence[int], devices: int) -> tuple[float | None, float | None]:
    """The band a ``Range`` states for one address, as two finite edges.

    A flat pair is the whole family's band. A table states one row per device,
    and the address takes the rows of the devices it sits at; a supply feeding
    several devices answers the band every one of them takes, the intersection
    of their rows. An edge that is not a finite number is no edge and comes
    back ``None``, as does every edge of a table whose rows do not line up
    with the family's devices.

    Args:
        declared: The field's ``Range`` value.
        indices: The 0-based device positions the address sits at.
        devices: How many devices the family has.

    Returns:
        ``(low, high)``, each ``None`` where the export states no finite edge.
    """
    if not isinstance(declared, (list, tuple)) or not declared:
        return None, None
    pairs: list[Any] = [declared]
    if any(isinstance(row, (list, tuple)) for row in declared):
        if len(declared) != devices:
            return None, None
        pairs = [declared[index] for index in indices]
    lows: list[float] = []
    highs: list[float] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        low, high = _number(pair[0]), _number(pair[1])
        if low is not None and high is not None and low > high:
            low, high = high, low
        if low is not None:
            lows.append(low)
        if high is not None:
            highs.append(high)
    return (max(lows) if lows else None), (min(highs) if highs else None)


def _limit_records(
    claims: dict[str, _Claim], wired: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """One record per setpoint its field bands, and one line per wired one it does not.

    Returns:
        The records, sorted by address, and a line for each wired setpoint
        whose band lacks an edge, which is therefore not made writable.
    """
    records: list[dict[str, Any]] = []
    unbanded: list[str] = []
    for address in sorted(claims):
        claim = claims[address]
        if claim.role != _SETPOINT:
            continue
        low, high = band(claim.fld.body.get(_RANGE_KEY), claim.indices, claim.view.n_devices)
        if address in wired and (low is None or high is None):
            unbanded.append(f"limits unbanded: {address} export [{_edge(low)},{_edge(high)}]")
        if low is None and high is None:
            continue
        record: dict[str, Any] = {"address": address}
        if low is not None:
            record["min_value"] = low
        if high is not None:
            record["max_value"] = high
        if address in wired and low is not None and high is not None:
            record["writable"] = True
        records.append(record)
    return records, unbanded


def _edge(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:g}"


def _differences(document: Any, records: list[dict[str, Any]]) -> list[str]:
    """One line per record of an existing file whose band is not the export's."""
    rows = document.get("records") if isinstance(document, dict) else None
    stated = {
        str(row.get("address")): row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict)
    }
    lines: list[str] = []
    for record in records:
        row = stated.get(record["address"])
        if row is None:
            continue
        ours = (_number(record.get("min_value")), _number(record.get("max_value")))
        theirs = (_number(row.get("min_value")), _number(row.get("max_value")))
        if ours != theirs:
            lines.append(
                f"limits differ: {record['address']} "
                f"file [{_edge(theirs[0])},{_edge(theirs[1])}] "
                f"export [{_edge(ours[0])},{_edge(ours[1])}]"
            )
    return lines


# -- seeds --------------------------------------------------------------------


def _golden(claims: dict[str, _Claim], exports: Exports) -> dict[str, float]:
    """The hardware nominal the export states for each address that has one.

    A nominal in physics units is no hardware value, and rows that do not line
    up with the family's devices name no device; neither is a golden value.
    """
    golden: dict[str, float] = {}
    for address, claim in claims.items():
        block = exports.va.get(claim.view.system)
        families = block.get("families") if isinstance(block, dict) else None
        family = families.get(claim.view.raw_name) if isinstance(families, dict) else None
        nominals = family.get("nominals") if isinstance(family, dict) else None
        nominal = nominals.get(claim.fld.name) if isinstance(nominals, dict) else None
        if not isinstance(nominal, dict):
            continue
        units = nominal.get("units")
        if isinstance(units, str) and units.strip().lower() == _PHYSICS_UNITS:
            continue
        values = nominal.get("values")
        if isinstance(values, (list, tuple)):
            if len(values) != claim.view.n_devices:
                continue
            values = values[claim.indices[0]]
        value = _number(values)
        if value is not None:
            golden[address] = value
    return golden


# -- measurement --------------------------------------------------------------


def _measurement(
    model: Model,
    entry: dict[str, Any],
    mapping: Mapping,
    carried: set[str],
    claims: dict[str, _Claim],
) -> dict[str, Any]:
    """One wired model's measurement file.

    Each group role is the first family of the model's wiring whose engine
    block fills it; ``rf`` is the first setpoint wired to the engine's
    frequency. ``kinds`` lists the measurements those resolve: the orbit
    response needs the monitors and both corrector planes, and dispersion,
    on a model that is not solved in a single pass, the ``rf`` instrument too.
    """
    groups: dict[str, str] = {}
    monitors: dict[str, str] = {}
    for raw, wired in model.wiring.items():
        if raw not in carried or wired.engine is None:
            continue
        engine = wired.engine
        token = mapping.mapped(raw)
        if engine.axis is not None:
            monitors.setdefault(engine.axis, token)
        for role, block in _GROUP_ENGINES:
            if (engine.attribute, engine.index) == block:
                groups.setdefault(role, token)
    if monitors:
        groups["bpm"] = monitors.get("x", next(iter(monitors.values())))
    ordered = {
        role: groups[role] for role in ("bpm", "hcor", "vcor", "quad", "sext") if role in groups
    }

    instruments: dict[str, str] = {}
    frequency = sorted(
        str(record["address"])
        for record in entry.get("wiring") or ()
        if (record.get("engine") or {}).get("attribute") == _RF_ATTRIBUTE
        and claims.get(str(record["address"])) is not None
        and claims[str(record["address"])].role == _SETPOINT
    )
    if frequency:
        instruments["rf"] = frequency[0]

    settings = (entry.get("settings") or {}).get(entry.get("engine")) or {}
    kinds: list[str] = []
    if {"bpm", "hcor", "vcor"} <= set(ordered):
        kinds.append("orm")
        if "rf" in instruments and settings.get("solve") != _SINGLE_PASS:
            kinds.append("dispersion")

    document: dict[str, Any] = {"kinds": kinds, "groups": ordered}
    if instruments:
        document["instruments"] = instruments
    document.update(TUNING)
    return document


# -- classes ------------------------------------------------------------------


def _class_rows(mapping: Mapping) -> list[dict[str, str]]:
    """A row per class the vocabulary lacks, every parent ahead of its children."""
    from osprey.facility.validate import known_classes

    known = set(known_classes())
    rows: list[dict[str, str]] = []
    pending = {
        name: branch.parent for name, branch in mapping.branches.items() if name not in known
    }
    while pending:
        ready = sorted(name for name, parent in pending.items() if parent not in pending)
        if not ready:  # a cycle the mapping check refuses; state it and let the build stop
            ready = sorted(pending)
        for name in ready:
            rows.append({"class": name, "parent": pending.pop(name)})
            known.add(name)
    classes: dict[str, str] = {}
    for family in mapping.families.values():
        if family.class_ is None or family.branch is None or family.class_ in known:
            continue
        classes.setdefault(family.class_, family.branch)
    rows.extend({"class": name, "parent": classes[name]} for name in sorted(classes))
    return rows


# -- identity -----------------------------------------------------------------


def _identity(mapping: Mapping, facility_dir: Path, seeded: Seeded) -> list[Path]:
    """Seed ``identity.yaml`` from the mapping's ``facility:`` block and remove the block."""
    identity = mapping.identity
    if identity is None:
        return []
    path = facility_dir / IDENTITY_FILE
    if path.exists():
        seeded.lines.append(f"facility: block ignored; {IDENTITY_FILE} exists")
        return []
    document: dict[str, Any] = {"code": identity.code}
    if identity.name is not None:
        document["name"] = identity.name
    if identity.description is not None:
        document["description"] = identity.description
    written = [_write(path, document)]
    source = facility_dir / MAPPING_FILE
    text = source.read_text(encoding="utf-8")
    stated = _parse(text)
    if isinstance(stated, dict) and _FACILITY_KEY in stated:
        del stated[_FACILITY_KEY]
        kept = _without_block(text, _FACILITY_KEY)
        if kept is None or _parse(kept) != stated:
            kept = dump_mapping(stated)
        source.write_text(kept, encoding="utf-8")
        written.append(source)
    return written


def _without_block(text: str, key: str) -> str | None:
    """A block-style document without one top-level key, every other line as written.

    The key's block is its own line and the indented lines under it, through
    the last one that is not blank.

    Returns:
        The remaining text, or ``None`` when no line opens with the key.
    """
    lines = text.splitlines(keepends=True)
    start = next((n for n, line in enumerate(lines) if line.startswith(f"{key}:")), None)
    if start is None:
        return None
    end = start + 1
    for n in range(start + 1, len(lines)):
        if not lines[n].strip():
            continue
        if not lines[n][0].isspace():
            break
        end = n + 1
    return "".join(lines[:start] + lines[end:])


# -- reading and writing ------------------------------------------------------


def _load(path: Path) -> Any:
    return _parse(path.read_text(encoding="utf-8"))


def _parse(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def _write(path: Path, document: Any) -> Path:
    import yaml

    text = yaml.safe_dump(
        document, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{HEADER}\n{text}", encoding="utf-8")
    return path
