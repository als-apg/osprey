"""Render the import census as ``PROFILE.md``.

``render_profile(census, votes)`` turns the census of a merged export and the
direction votes into one Markdown document: a facility section (totals, system
tokens that fail PN_LOCAL, every shared PV with all its owners), then one
section per sub-machine in census order.

Rules:

* Every system section carries every heading of :data:`SYSTEM_HEADINGS`, and
  its hazards every heading of :data:`HAZARD_HEADINGS`; an empty list is
  written as ``None.`` so a reader can tell "checked, nothing found" from
  "not reported".
* Censuses are tables, hazards are bullet lists.
* Ordering is deterministic: systems, families and fields keep the census
  (export) order, votes are sorted by ``(family, field)``, per-system verdicts
  by system token. No timestamps.
* Votes are facility-wide; a system section lists the votes its system took
  part in. A disagreement is an undecided vote whose systems reached two
  different decided verdicts; every undecided vote is listed as well.
* Free text is folded to one line and ``|`` is escaped, so a table row stays
  one row.

The module is pure and depends on the standard library and the census and vote
records only.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from typing import Any

from osprey.services.mml.census import Census, SharedPV, SystemCensus
from osprey.services.mml.directions import Vote

__all__ = ["HAZARD_HEADINGS", "SYSTEM_HEADINGS", "TITLE", "render_profile"]

#: The document title.
TITLE = "MML import profile"

#: The level-3 headings of every system section, in order.
SYSTEM_HEADINGS: tuple[str, ...] = (
    "Families",
    "Fields and channel keys",
    "Device counts",
    "Disabled devices",
    "MemberOf census",
    "Direction votes",
    "Descriptions",
    "Hazards",
    "Position and DeviceType coverage",
    "Families with arrays from setup",
    "AD scalars",
)

#: The level-4 headings under each system's ``Hazards``, in order.
HAZARD_HEADINGS: tuple[str, ...] = (
    "Function handles",
    "Typo keys",
    "Non-finite ranges",
    "HWUnits and DataType shapes",
    "Case-duplicate families",
    "System token not PN_LOCAL",
    "Dual-key fields",
    "Empty channel lists",
    "Partial channel lists",
    "Broadcast rows",
    "Families with zero channels",
    "Shared PVs",
)

_NONE = "None."


def _inline(value: Any) -> str:
    """Fold ``value`` to one line of Markdown-safe text."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _code(value: Any) -> str:
    return f"`{_inline(value)}`"


def _verdict(direction: str | None) -> str:
    return direction if direction is not None else "undecided"


def _table(header: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> list[str]:
    rows = list(rows)
    if not rows:
        return [_NONE]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(_inline(cell) for cell in row) + " |" for row in rows)
    return lines


def _bullets(items: Iterable[str]) -> list[str]:
    lines = [f"- {item}" for item in items]
    return lines or [_NONE]


def _block(heading: str, level: int, body: list[str]) -> list[str]:
    return ["#" * level + " " + heading, "", *body, ""]


def _shared_pv_lines(shared: Iterable[SharedPV]) -> list[str]:
    lines: list[str] = []
    for item in shared:
        lines.append(f"- {_code(item.pv)}")
        lines.extend(
            f"  - `({_inline(o.system)}, {_inline(o.family)}, {_inline(o.field)}, {o.index})`"
            for o in item.owners
        )
    if not lines:
        return [_NONE]
    return ["Owners are `(system, family, field, index)`; the index is 0-based.", "", *lines]


def _dotted(*parts: Any) -> str:
    return _code(".".join(str(part) for part in parts))


def _json(value: Any) -> str:
    return _inline(json.dumps(value, ensure_ascii=False, default=str))


def _per_system(vote: Vote) -> str:
    return ", ".join(
        f"{_inline(system)} {_verdict(vote.per_system[system])}"
        for system in sorted(vote.per_system)
    )


def _votes_lines(system: str, votes: list[tuple[tuple[str, str], Vote]]) -> list[str]:
    mine = [(key, vote) for key, vote in votes if system in vote.per_system]
    table = _table(
        ("Family", "Field", "This system", "Facility vote", "Source"),
        (
            (
                family,
                field,
                _verdict(vote.per_system[system]),
                _verdict(vote.direction),
                vote.source,
            )
            for (family, field), vote in mine
        ),
    )
    undecided = [(key, vote) for key, vote in mine if vote.direction is None]
    disagreements = [
        (key, vote)
        for key, vote in undecided
        if len({v for v in vote.per_system.values() if v is not None}) > 1
    ]
    return [
        *table,
        "",
        *_block(
            "Disagreements",
            4,
            _bullets(f"{_dotted(*key)}: {_per_system(vote)}" for key, vote in disagreements),
        ),
        *_block(
            "Undecided",
            4,
            _bullets(f"{_dotted(*key)}: {_per_system(vote)}" for key, vote in undecided),
        )[:-1],
    ]


def _hazard_lines(census: Census, system: SystemCensus) -> list[str]:
    hazards = system.hazards
    name = system.name
    items: dict[str, list[str]] = {
        "Function handles": _bullets(
            f"{_dotted(h.family, *h.path)}: function {_code(h.function)}, file {_code(h.file)}"
            for h in hazards.function_handles
        ),
        "Typo keys": _bullets(_dotted(k.family, *k.path) for k in hazards.typo_keys),
        "Non-finite ranges": _bullets(
            f"{_dotted(r.family, r.field, 'Range')}: {_json(r.value)}"
            for r in hazards.non_finite_ranges
        ),
        "HWUnits and DataType shapes": _bullets(
            f"{_dotted(u.family, u.field, u.key)}: {u.kind}" for u in hazards.unit_shapes
        ),
        "Case-duplicate families": _bullets(
            ", ".join(_inline(n) for n in names) for names in hazards.case_duplicate_families
        ),
        "System token not PN_LOCAL": _bullets(
            [_code(name)] if name in census.illegal_system_tokens else []
        ),
        "Dual-key fields": _bullets(_dotted(d.family, d.field) for d in hazards.dual_key_fields),
        "Empty channel lists": _bullets(
            _dotted(e.family, e.field, e.key) for e in hazards.empty_channel_lists
        ),
        "Partial channel lists": _bullets(
            f"{_dotted(p.family, p.field, p.key)}: {p.length} of {p.n_devices} slots"
            for p in hazards.partial_channel_lists
        ),
        "Broadcast rows": _bullets(
            _dotted(b.family, b.field, b.key) for b in hazards.broadcast_rows
        ),
        "Families with zero channels": _bullets(_inline(f) for f in hazards.zero_channel_families),
        "Shared PVs": _shared_pv_lines(
            s for s in census.shared_pvs if any(o.system == name for o in s.owners)
        ),
    }
    lines: list[str] = []
    for heading in HAZARD_HEADINGS:
        lines.extend(_block(heading, 4, items[heading]))
    return lines[:-1]


def _system_lines(
    census: Census, system: SystemCensus, votes: list[tuple[tuple[str, str], Vote]]
) -> list[str]:
    families = system.families
    disabled = [f for f in families if f.disabled_devices]
    items: dict[str, list[str]] = {
        "Families": _table(
            ("Family", "Devices", "Fields", "Raw slots", "Bindings"),
            ((f.name, f.n_devices, len(f.fields), f.raw_slots, f.bindings) for f in families),
        ),
        "Fields and channel keys": _table(
            ("Family", "Field", "Channel keys", "Raw slots", "Bindings", "Broadcast"),
            (
                (
                    f.name,
                    field.name,
                    ", ".join(field.keys),
                    field.raw_slots,
                    field.bindings,
                    "yes" if field.broadcast else "no",
                )
                for f in families
                for field in f.fields
            ),
        ),
        "Device counts": _table(
            ("Family", "Devices", "Source"),
            (
                (f.name, f.n_devices, "fallback" if f.n_devices_from_fallback else "DeviceList")
                for f in families
            ),
        ),
        "Disabled devices": _table(
            ("Family", "Disabled indices (0-based)"),
            ((f.name, ", ".join(str(i) for i in f.disabled_devices)) for f in disabled),
        ),
        "MemberOf census": _table(
            ("Tag", "Owners"),
            (
                (
                    tag.tag,
                    ", ".join(
                        family if field is None else f"{family}.{field}"
                        for family, field in tag.owners
                    ),
                )
                for tag in system.member_of
            ),
        ),
        "Direction votes": _votes_lines(system.name, votes),
        "Descriptions": [
            *_block(
                "Families with native descriptions",
                4,
                _bullets(
                    f"{_inline(name)}: {_inline(text)}"
                    for name, text in system.families_with_descriptions
                ),
            ),
            *_block(
                "Families without descriptions",
                4,
                _bullets(_inline(n) for n in system.families_without_descriptions),
            )[:-1],
        ],
        "Hazards": _hazard_lines(census, system),
        "Position and DeviceType coverage": _table(
            (
                "Family",
                "Position real",
                "Position stand-in",
                "DeviceType real",
                "DeviceType stand-in",
            ),
            (
                (
                    f.name,
                    f.position_real,
                    f.position_stand_in,
                    f.device_type_real,
                    f.device_type_stand_in,
                )
                for f in families
            ),
        ),
        "Families with arrays from setup": _bullets(_inline(n) for n in system.setup_families),
        "AD scalars": _table(
            ("Key", "Value"), ((f"`{key}`", value) for key, value in system.ad_scalars)
        ),
    }
    lines = [f"## System {_code(system.name)}", ""]
    for heading in SYSTEM_HEADINGS:
        lines.extend(_block(heading, 3, items[heading]))
    return lines


def render_profile(census: Census, votes: Mapping[tuple[str, str], Vote]) -> str:
    """Render ``PROFILE.md`` for one merged export.

    Args:
        census: The census from :func:`~osprey.services.mml.census.take_census`.
        votes: The votes from
            :func:`~osprey.services.mml.directions.vote_directions`, keyed by
            ``(raw_family, field)``; the order of the mapping is ignored.

    Returns:
        The Markdown document, ending in exactly one newline. Rendering the
        same inputs twice gives identical text.
    """
    ordered = sorted(votes.items(), key=lambda item: item[0])
    totals = census.totals
    lines = [
        f"# {TITLE}",
        "",
        "## Facility",
        "",
        *_block(
            "Totals",
            3,
            _table(
                ("Metric", "Value"),
                ((field.name, getattr(totals, field.name)) for field in dataclasses.fields(totals)),
            ),
        ),
        *_block(
            "System tokens not PN_LOCAL",
            3,
            _bullets(_code(token) for token in census.illegal_system_tokens),
        ),
        *_block("Shared PVs", 3, _shared_pv_lines(census.shared_pvs)),
    ]
    for system in census.systems:
        lines.extend(_system_lines(census, system, ordered))
    return "\n".join(lines).rstrip("\n") + "\n"
