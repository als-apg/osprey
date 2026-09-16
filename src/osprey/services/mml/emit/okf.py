"""The OKF emitter: the facility page and one page per device family.

``write_okf_bundle`` turns an imported MML install and its mapping into the
pages of an OKF knowledge bundle, then regenerates the bundle's indexes.

Rules:

* ``facility.md`` is typed ``Facility``; its title and description come from
  ``mapping.facility`` (the title falls back to the facility token). Its body
  lists the machine and the sub-machines in ``section_order``, then one section
  per system in that order: the mapping's system description, then the mode,
  energy, injection energy, circumference, harmonic number, MCF and lattice
  file from the accelerator data. A fact the data lacks is omitted; the section
  is written whether or not any accelerator data reaches it.
* One ``families/<system>-<family>.md`` is written per ``(system, family)``
  whose family view carries at least one channel, where ``<system>`` is the
  mapped system name and ``<family>`` the mapped family token. It is typed
  ``DeviceFamily``, titled ``<system> <family>``, and carries the mapping's
  family description and provenance. Its body is a table of the family's
  fields: channel keys, mapping direction, channel count and field description.
* Every page carries the ``exporter``, ``ao_sha256`` and ``mapping_sha256``
  provenance keys and is serialised through ``OKFDocument``.
* Pages are written only when their bytes change, atomically, so a second
  write of the same inputs changes no byte.
* Systems are visited in ``section_order`` (mapped names), families in the
  ``ao.json`` key order; ``_``-prefixed and non-dict ``ao`` entries are
  bookkeeping, never systems or families.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any

from osprey.services.facility_knowledge.okf.document import OKFDocument
from osprey.services.facility_knowledge.okf.index import regenerate_indexes
from osprey.services.mml.canonical import write_if_changed
from osprey.services.mml.emit.context import EmitContext
from osprey.services.mml.family import FamilyView, family_views, system_bodies
from osprey.services.mml.mapping.schema import Mapping

__all__ = ["FACILITY_PAGE", "FAMILIES_DIR", "write_okf_bundle"]

#: The facility page, relative to the bundle root.
FACILITY_PAGE = "facility.md"

#: The directory holding one page per device family.
FAMILIES_DIR = "families"

#: Per-system accelerator-data facts, in body order: (label, key paths). The
#: first path that resolves wins, so a fallback spelling follows its preferred
#: one.
_SYSTEM_FACTS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("Mode", (("OperationalMode",),)),
    ("Energy (GeV)", (("Energy",),)),
    ("Injection energy (GeV)", (("InjectionEnergy",),)),
    ("Circumference (m)", (("Circumference",),)),
    ("Harmonic number", (("HarmonicNumber",),)),
    ("MCF", (("MCF",),)),
    ("Lattice file", (("OpsData", "LatticeFile"), ("ATModel",))),
)

_UNSET_DIRECTION = "unset"


def write_okf_bundle(
    ao: dict,
    ad: dict | None,
    mapping: Mapping,
    ctx: EmitContext,
    bundle_root: Path,
) -> list[Path]:
    """Write the facility and family pages, then regenerate the indexes.

    Args:
        ao: The canonical ``ao.json``, keyed by raw system token.
        ad: The canonical ``ad.json``, ``{raw system: accelerator data}``, or
            ``None`` when no accelerator data was imported.
        mapping: The parsed mapping.
        ctx: The shared provenance context.
        bundle_root: The bundle directory; created when absent.

    Returns:
        The pages written, then the ``index.md`` files regenerated.

    Raises:
        ValueError: The mapping names no facility description or title, the
            order and ``ao`` disagree on the systems, or the mapping does not
            name a system or family, or a family has no description.
    """
    bundle_root = Path(bundle_root)
    present = [raw for raw, _ in system_bodies(ao)]
    systems = [raw for _, raws in mapping.ordered_systems(present) for raw in raws]

    pages: list[Path] = []
    facility_path = bundle_root / FACILITY_PAGE
    write_if_changed(facility_path, _facility_page(ad, mapping, ctx, systems).serialize())
    pages.append(facility_path)

    for raw_system in systems:
        for view in family_views(raw_system, ao[raw_system]):
            if view.channel_count == 0:
                continue
            doc = _family_page(view, mapping, ctx)
            path = bundle_root / FAMILIES_DIR / _family_file(view, mapping)
            write_if_changed(path, doc.serialize())
            pages.append(path)

    return pages + regenerate_indexes(bundle_root)


# -- facility page -------------------------------------------------------------


def _facility_page(
    ad: dict | None, mapping: Mapping, ctx: EmitContext, systems: list[str]
) -> OKFDocument:
    facility = mapping.facility
    title = facility.title or facility.token
    if not title:
        raise ValueError("mapping names no facility.title or facility.token")
    if not facility.description:
        raise ValueError("mapping has no facility.description")

    blocks = [_ad_block(ad, raw) for raw in systems]
    head: list[str] = []
    machine = next((m for b in blocks if (m := _fact(b, ("Machine",))) is not None), None)
    if machine is not None:
        head.append(f"- Machine: {machine}")
    head.append(f"- Sub-machines: {', '.join(mapping.section_order)}")

    sections = [f"# {title}", facility.description, "\n".join(head)]
    for raw, block in zip(systems, blocks, strict=True):
        entry = mapping.systems[raw]
        body = [f"## {entry.name}", ""]
        if entry.description:
            body.extend([entry.description, ""])
        body.extend(
            f"- {label}: {value}"
            for label, paths in _SYSTEM_FACTS
            if (value := _first_fact(block, paths)) is not None
        )
        sections.append("\n".join(body).rstrip("\n"))

    frontmatter = {"type": "Facility", "title": title, "description": facility.description}
    frontmatter.update(ctx.front_matter)
    return OKFDocument(frontmatter=frontmatter, body="\n\n".join(sections) + "\n")


def _ad_block(ad: dict | None, system: str) -> dict:
    if not isinstance(ad, dict):
        return {}
    block = ad.get(system)
    return block if isinstance(block, dict) else {}


def _first_fact(block: MappingABC[str, Any], paths: tuple[tuple[str, ...], ...]) -> str | None:
    """Return the first of ``paths`` that resolves to a scalar, or ``None``."""
    for path in paths:
        value = _fact(block, path)
        if value is not None:
            return value
    return None


def _fact(block: MappingABC[str, Any], path: tuple[str, ...]) -> str | None:
    """Return the scalar at ``path`` in ``block`` as text, or ``None`` when absent."""
    value: Any = block
    for key in path:
        if not isinstance(value, MappingABC) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return str(int(value)) if value.is_integer() else repr(value)
    return None


# -- family pages --------------------------------------------------------------


def _family_file(view: FamilyView, mapping: Mapping) -> str:
    return f"{_system_name(view.system, mapping)}-{_family_token(view, mapping)}.md"


def _family_page(view: FamilyView, mapping: Mapping, ctx: EmitContext) -> OKFDocument:
    title = f"{_system_name(view.system, mapping)} {_family_token(view, mapping)}"
    family = mapping.families[view.raw_name]
    if not family.description:
        raise ValueError(f"mapping has no families.{view.raw_name}.description")

    rows = [
        "| Field | Channel keys | Direction | Channels | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, field_view in view.fields.items():
        direction = mapping.directions.get(f"{view.raw_name}.{name}")
        mapped_field = family.fields.get(name)
        cells = (
            name,
            ", ".join(field_view.keys),
            (direction.direction if direction and direction.direction else _UNSET_DIRECTION),
            str(field_view.channel_count),
            (mapped_field.description if mapped_field and mapped_field.description else ""),
        )
        rows.append("| " + " | ".join(_cell(c) for c in cells) + " |")

    body = "\n\n".join(
        [
            f"# {title}",
            family.description,
            f"- Devices: {view.n_devices}\n- Channels: {view.channel_count}",
            "\n".join(rows),
        ]
    )
    frontmatter = {
        "type": "DeviceFamily",
        "title": title,
        "description": family.description,
        "provenance": family.provenance,
    }
    frontmatter.update(ctx.front_matter)
    return OKFDocument(frontmatter=frontmatter, body=body + "\n")


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", r"\|")


def _system_name(raw_system: str, mapping: Mapping) -> str:
    system = mapping.systems.get(raw_system)
    if system is None:
        raise ValueError(f"mapping names no system {raw_system!r}")
    return system.name


def _family_token(view: FamilyView, mapping: Mapping) -> str:
    if view.raw_name not in mapping.families:
        raise ValueError(
            f"mapping names no family {view.raw_name!r} (system {view.system!r} in ao.json)"
        )
    return mapping.mapped(view.raw_name)
