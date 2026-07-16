"""Grow the shipped tier-3 channel databases to the full ALS-U AR inventory.

Emits all three shipped tier-3 channel-database formats --
``hierarchical.json``, ``in_context.json``, ``middle_layer.json`` (under
``src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/
tier3/``) -- address-identically, from two declarative sources:

* :data:`osprey.simulation.facility_spec.ALS_U_AR` -- the 11 SR-ring
  families and their grown device counts.
* :data:`osprey.simulation.channel_schema.CHANNEL_SCHEMA` -- the FIELD/
  SUBFIELD structure and display metadata for each family.

Address format: ``SR:{system}:{family}:{DD}:{FIELD}:{SUBFIELD}``, where
``DD`` is the zero-padded 2-digit device id. Ring is always ``SR``.

Merge-preserve semantics
-------------------------
The generator only ever GROWS the 11 FacilitySpec SR ring families
(``SR:MAG:{QF,QD,QFA,DIPOLE,SF,SD,SHF,SHD,HCM,VCM}`` and ``SR:DIAG:BPM``).
Every other address -- BR/BTS rings, ``SR:RF``, ``SR:VAC``,
``SR:DIAG:DCCT``/``GAMMA``/``NEUTRON``, and any other system -- is left
byte-for-byte untouched.

For an address in the target grown inventory that already exists in the
loaded database, the existing entry is kept verbatim (hand-authored prose is
preserved). Only genuinely new addresses -- a new family (QFA/SHF/SHD) or a
device index beyond the old count -- are templated from the schema. Once a
family has been grown to its target count, every target address already
exists, so re-running is idempotent (identical bytes).

The three genuinely-new families (QFA, SHF, SHD) have no shipped precedent;
their prose (``in_context.json`` channel/description tokens,
``hierarchical.json`` family-level blurb) is synthesized from the schema
rather than copied from a shipped file, extending the naming convention
established by their structural analog (QF, SF, SD respectively).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.channel_finder.databases.template import ChannelDatabase
from osprey.simulation.channel_schema import CHANNEL_SCHEMA, FamilyChannelSchema
from osprey.simulation.facility_spec import ALS_U_AR

RING = "SR"

TIER3_FILENAMES: dict[str, str] = {
    "hierarchical": "hierarchical.json",
    "in_context": "in_context.json",
    "middle_layer": "middle_layer.json",
}

# Devices per DeviceList "sector" bucket in middle_layer.json's per-family
# _setup block, mirroring the shipped SR arrays (SF/SD pack 1 device per
# sector, every other family packs 2). New families inherit their closest
# structural analog's packing.
_DEVICES_PER_SECTOR: dict[str, int] = {"SF": 1, "SD": 1, "SHF": 1, "SHD": 1}

# in_context.json PascalCase channel-name token and lowercase noun phrase
# per family, sourced verbatim from the shipped SR entries where a shipped
# precedent exists (e.g. "QuadFocus" / "focusing quadrupole" for QF).
# QFA/SHF/SHD have no shipped precedent; their tokens extend the convention
# established by their structural analog (QF, SF, SD).
_IN_CONTEXT_TOKEN: dict[str, str] = {
    "DIPOLE": "Dipole",
    "QD": "QuadDefocus",
    "QF": "QuadFocus",
    "QFA": "QuadFocusAchromat",
    "SF": "SextFocus",
    "SD": "SextDefocus",
    "SHF": "SextHarmFocus",
    "SHD": "SextHarmDefocus",
    "HCM": "HorizCorr",
    "VCM": "VertCorr",
    "BPM": "BPM",
}
_IN_CONTEXT_NOUN: dict[str, str] = {
    "DIPOLE": "dipole bending magnet",
    "QD": "defocusing quadrupole",
    "QF": "focusing quadrupole",
    "QFA": "achromat focusing quadrupole",
    "SF": "sextupole (focusing)",
    "SD": "sextupole (defocusing)",
    "SHF": "harmonic sextupole (focusing)",
    "SHD": "harmonic sextupole (defocusing)",
    "HCM": "horizontal corrector",
    "VCM": "vertical corrector",
    "BPM": "beam position monitor",
}
# FIELD-level channel-name token / description phrase. "GOLDEN" here is the
# BPM top-level field (golden orbit), distinct from magnets' CURRENT:GOLDEN
# *subfield* (see _SUBFIELD_TOKEN below).
_FIELD_TOKEN: dict[str, str] = {
    "CURRENT": "Current",
    "STATUS": "Status",
    "POSITION": "Position",
    "OFFSET": "Offset",
    "GOLDEN": "GoldenOrbit",
}
_FIELD_PHRASE: dict[str, str] = {
    "CURRENT": "current",
    "STATUS": "status",
    "POSITION": "position",
    "OFFSET": "offset",
    "GOLDEN": "golden orbit",
}
_SUBFIELD_TOKEN: dict[str, str] = {
    "SP": "Setpoint",
    "RB": "Readback",
    "GOLDEN": "Golden",
    "READY": "Ready",
    "ON": "On",
    "FAULT": "Fault",
    "VALID": "Valid",
    "CONNECTED": "Connected",
    "X": "X",
    "Y": "Y",
}


def _address(fam: str, system: str, device_id: int, field: str, subfield: str) -> str:
    """Build the tier-3 address for one channel."""
    return f"{RING}:{system}:{fam}:{device_id:02d}:{field}:{subfield}"


def _target_channels(
    fam: str, schema: FamilyChannelSchema, count: int
) -> list[tuple[int, str, str, str]]:
    """Every ``(device_id, field, subfield, address)`` the grown family should
    have, in canonical (device, field, subfield) order."""
    out = []
    for device_id in range(1, count + 1):
        for field, subs in schema.fields.items():
            for subfield in subs:
                addr = _address(fam, schema.system, device_id, field, subfield)
                out.append((device_id, field, subfield, addr))
    return out


# ---------------------------------------------------------------------------
# middle_layer.json
# ---------------------------------------------------------------------------


def _grow_middle_layer(data: dict[str, Any]) -> dict[str, Any]:
    """Grow ``data`` (a loaded ``middle_layer.json``) in place and return it."""
    sr = data.setdefault(RING, {})

    for family in ALS_U_AR.families:
        fam, count = family.name, family.count
        schema = CHANNEL_SCHEMA[fam]
        fam_node = sr.setdefault(fam, {})
        fam_node.setdefault("_description", schema.display_name.lower())

        for field, subs in schema.fields.items():
            field_node = fam_node.setdefault(field, {})
            field_node.setdefault("_description", field.lower())

            for subfield, subschema in subs.items():
                leaf = field_node.setdefault(subfield, {})
                leaf.setdefault("_description", subschema.description)

                existing = leaf.get("ChannelNames", [])
                if isinstance(existing, str):
                    existing = [existing]
                have = set(existing)
                grown = list(existing)
                for device_id in range(1, count + 1):
                    addr = _address(fam, schema.system, device_id, field, subfield)
                    if addr not in have:
                        grown.append(addr)
                leaf["ChannelNames"] = grown

                leaf.setdefault("DataType", subschema.data_type)
                leaf.setdefault("HWUnits", subschema.hw_units)

        # CommonNames/DeviceList/ElementList are positional metadata derived
        # purely from the device count, not per-address hand-authored prose:
        # regenerating them from scratch reproduces the shipped values
        # exactly for already-grown devices, so this stays idempotent.
        dps = _DEVICES_PER_SECTOR.get(fam, 2)
        fam_node["_setup"] = {
            "CommonNames": [schema.common_name(i) for i in range(1, count + 1)],
            "DeviceList": [[(i - 1) // dps + 1, (i - 1) % dps + 1] for i in range(1, count + 1)],
            "ElementList": list(range(1, count + 1)),
        }

    return data


# ---------------------------------------------------------------------------
# hierarchical.json
# ---------------------------------------------------------------------------


def _new_hierarchical_family(fam: str, schema: FamilyChannelSchema, count: int) -> dict[str, Any]:
    """Build a ``hierarchical.json`` family subtree for a family with no
    shipped precedent (QFA/SHF/SHD): structurally cloned from the schema,
    with synthesized (not hand-authored) prose."""
    fields: dict[str, Any] = {}
    for field, subs in schema.fields.items():
        field_desc = (
            f"{schema.display_name} Current (Amperes): current through {fam} coil windings."
            if field == "CURRENT"
            else f"Operational status for {schema.display_name.lower()}."
        )
        node: dict[str, Any] = {"_description": field_desc}
        for subfield, subschema in subs.items():
            node[subfield] = {"_description": subschema.description}
        fields[field] = node

    return {
        "_description": (
            f"{schema.display_name}s ({fam}): ALS-U Accumulator Ring "
            f"{schema.display_name.lower()} magnets."
        ),
        "DEVICE": {
            "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, count]},
            **fields,
        },
    }


def _grow_hierarchical(data: dict[str, Any]) -> dict[str, Any]:
    """Grow ``data`` (a loaded ``hierarchical.json``) in place and return it."""
    tree = data["tree"]
    sr = tree.setdefault(RING, {})

    for family in ALS_U_AR.families:
        fam, count = family.name, family.count
        schema = CHANNEL_SCHEMA[fam]
        system_node = sr.setdefault(schema.system, {})

        if fam in system_node:
            # Existing family: only widen the DEVICE range. The per-field/
            # subfield "_description" prose is hand-authored and untouched.
            system_node[fam]["DEVICE"]["_expansion"]["_range"] = [1, count]
        else:
            system_node[fam] = _new_hierarchical_family(fam, schema, count)

    return data


# ---------------------------------------------------------------------------
# in_context.json
# ---------------------------------------------------------------------------


def _in_context_channel(fam: str, device_id: int, field: str, subfield: str) -> str:
    return (
        f"StorageRing_{_IN_CONTEXT_TOKEN[fam]}_{device_id:02d}_"
        f"{_FIELD_TOKEN[field]}_{_SUBFIELD_TOKEN[subfield]}"
    )


def _in_context_description(
    fam: str, device_id: int, field: str, subfield_desc: str
) -> str:
    return (
        f"Storage ring {_IN_CONTEXT_NOUN[fam]} {device_id:02d} "
        f"{_FIELD_PHRASE[field]} {subfield_desc}"
    )


def _grow_in_context(data: dict[str, Any]) -> dict[str, Any]:
    """Grow ``data`` (a loaded ``in_context.json``) in place and return it."""
    channels: list[dict[str, str]] = data.setdefault("channels", [])
    have = {c["address"] for c in channels}

    new_entries = []
    for family in ALS_U_AR.families:
        fam, count = family.name, family.count
        schema = CHANNEL_SCHEMA[fam]
        for device_id, field, subfield, addr in _target_channels(fam, schema, count):
            if addr in have:
                continue
            subschema = schema.fields[field][subfield]
            new_entries.append(
                {
                    "channel": _in_context_channel(fam, device_id, field, subfield),
                    "address": addr,
                    "description": _in_context_description(
                        fam, device_id, field, subschema.description
                    ),
                }
            )

    channels.extend(new_entries)
    meta = data.setdefault("_metadata", {})
    meta["total_channels"] = len(channels)
    return data


# ---------------------------------------------------------------------------
# Cross-paradigm identity gate
# ---------------------------------------------------------------------------


def address_set(path: Path, loader: type) -> set[str]:
    """Load ``path`` through a channel-finder database ``loader`` and return
    its address set (reuses the shipped parsers -- never hand-rolls address
    extraction)."""
    db = loader(str(path))
    return {c["address"] for c in db.get_all_channels()}


def verify_cross_paradigm_identity(hier_path: Path, ctx_path: Path, ml_path: Path) -> None:
    """Assert the three tier-3 formats expose an identical address set.

    Raises:
        AssertionError: If any two formats disagree, naming a sample of the
            symmetric difference to make debugging tractable.
    """
    hier_addrs = address_set(hier_path, HierarchicalChannelDatabase)
    ctx_addrs = address_set(ctx_path, ChannelDatabase)
    ml_addrs = address_set(ml_path, MiddleLayerDatabase)

    if hier_addrs == ctx_addrs == ml_addrs:
        return

    raise AssertionError(
        "Cross-paradigm address identity violated:\n"
        f"  hierarchical - in_context (sample): {sorted(hier_addrs - ctx_addrs)[:10]}\n"
        f"  in_context - hierarchical (sample): {sorted(ctx_addrs - hier_addrs)[:10]}\n"
        f"  hierarchical - middle_layer (sample): {sorted(hier_addrs - ml_addrs)[:10]}\n"
        f"  middle_layer - hierarchical (sample): {sorted(ml_addrs - hier_addrs)[:10]}\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _dump(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def generate(source_dir: Path | str, dest_dir: Path | str | None = None) -> dict[str, Path]:
    """Regenerate the three tier-3 channel databases from ``source_dir``.

    Grows the ALS-U AR SR-ring families to their full :data:`ALS_U_AR`
    inventory while preserving every other address byte-for-byte, writes the
    result to ``dest_dir`` (defaults to ``source_dir`` -- an in-place
    regeneration), and verifies cross-paradigm address identity before
    returning.

    Args:
        source_dir: Directory containing the three shipped tier-3 files.
        dest_dir: Directory to write the regenerated files to. Defaults to
            ``source_dir``.

    Returns:
        ``{format_name: written_path}`` for the three tier-3 files.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir) if dest_dir is not None else source_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    hier_data = _grow_hierarchical(_load(source_dir / TIER3_FILENAMES["hierarchical"]))
    ctx_data = _grow_in_context(_load(source_dir / TIER3_FILENAMES["in_context"]))
    ml_data = _grow_middle_layer(_load(source_dir / TIER3_FILENAMES["middle_layer"]))

    written = {name: dest_dir / filename for name, filename in TIER3_FILENAMES.items()}
    _dump(written["hierarchical"], hier_data)
    _dump(written["in_context"], ctx_data)
    _dump(written["middle_layer"], ml_data)

    verify_cross_paradigm_identity(
        written["hierarchical"], written["in_context"], written["middle_layer"]
    )
    return written


if __name__ == "__main__":
    import sys

    default_source = Path(
        "src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/tier3"
    )
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default_source
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    for name, path in generate(src, dest).items():
        print(f"{name}: {path}")
