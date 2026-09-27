"""Facility identity resolved from a project config.

One resolution order, shared by every reader that needs the facility display
name: the build path that renders the agent prompts and the interface apps that
label their UI. Keeping it in one place is the point — the two spellings drifted
precisely because each reader picked its own.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any
from zoneinfo import available_timezones

__all__ = [
    "DEFAULT_FACILITY_ZONE",
    "closest_zone_name",
    "is_zone_name",
    "resolve_facility_name",
]

#: The zone every reader falls back to and every preset pins.
DEFAULT_FACILITY_ZONE = "UTC"


def resolve_facility_name(config: dict[str, Any], default: str) -> str:
    """Resolve the facility display name from a parsed project config.

    ``facility.name`` is the canonical spelling — the same ``facility:`` block
    that carries ``prefix``. Top-level ``facility_name`` is the older spelling;
    it is honored as a fallback so a config written before the consolidation
    keeps working unchanged. An empty value at either level falls through, since
    a blank facility name reaches prompts and UI labels as a hole in the
    sentence.

    Args:
        config: Parsed ``config.yml`` dictionary.
        default: Value to use when neither key carries a name. Callers differ:
            the build path passes the project name, an interface app passes the
            empty string it would otherwise have shown.

    Returns:
        str: Facility display name.
    """
    facility = config.get("facility")
    if isinstance(facility, dict) and facility.get("name"):
        return str(facility["name"])
    return str(config.get("facility_name") or default)


def is_zone_name(name: str) -> bool:
    """Whether *name* is a zone in the IANA time zone database, spelled exactly.

    Membership, not ``ZoneInfo(name)``: the lookup opens a file, so on a
    case-insensitive filesystem it accepts ``america/los_angeles``, while the
    case-sensitive filesystem of every container refuses it. A host with no
    zone database at all (no system tzdata and no ``tzdata`` wheel) cannot
    judge, so it accepts every name rather than refusing all of them.

    Args:
        name: The zone name as written.

    Returns:
        bool: ``True`` when the name is a zone, or when there is no database.
    """
    if name == DEFAULT_FACILITY_ZONE:
        return True
    zones = available_timezones()
    if not zones:
        return True
    return name in zones


def closest_zone_name(name: str) -> str | None:
    """The zone a mistyped *name* most likely meant.

    A case-insensitive exact match first, then the closest spelling.

    Args:
        name: The zone name as written.

    Returns:
        str | None: A zone name, or ``None`` when nothing is close.
    """
    zones = available_timezones()
    folded = name.casefold()
    for zone in zones:
        if zone.casefold() == folded:
            return zone
    matches = get_close_matches(name, sorted(zones), n=1, cutoff=0.8)
    return matches[0] if matches else None
