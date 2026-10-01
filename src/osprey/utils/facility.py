"""Facility identity and facility zone, resolved in one place.

Every reader that needs the facility's name asks here, so the build path that
renders the agent prompts and the servers and apps that label their answers
cannot each pick their own spelling.
"""

from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Any, TypedDict
from zoneinfo import available_timezones

__all__ = [
    "DEFAULT_FACILITY_ZONE",
    "SET_FACILITY_ZONE",
    "FacilityIdentity",
    "closest_zone_name",
    "facility_identity",
    "is_zone_name",
    "resolve_facility_name",
]

logger = logging.getLogger(__name__)

#: The zone every reader falls back to and every preset pins.
DEFAULT_FACILITY_ZONE = "UTC"

#: The one remedy sentence the build reminder and the health row both print.
SET_FACILITY_ZONE = (
    "Set system.timezone under `config:` in profile.yml to your facility's zone, "
    "for example America/New_York."
)


class FacilityIdentity(TypedDict):
    """Who a render's facility is.

    Attributes:
        code: The ``PN_LOCAL`` token that names the facility in identifiers.
        name: The display name.
        description: One free-form sentence, or ``None`` when none is authored.
    """

    code: str
    name: str
    description: str | None


def facility_identity(
    render_root: Path, project_name: str | None = None
) -> FacilityIdentity | None:
    """Read the facility identity of a render.

    The facility file at the root of the render is the source. A render without
    one (or with one that names no identity code) answers with the identity the
    build would write for a project that authors none: the project name as the
    display name and its fold as the code.

    Args:
        render_root: The directory that holds the rendered ``config.yml``.
        project_name: The project's name, used where the facility file names no
            display name and as the whole identity where there is no file.

    Returns:
        FacilityIdentity | None: The identity, or ``None`` when there is neither
        a facility file nor a project name.
    """
    recorded = _recorded_identity(render_root)
    if recorded is not None:
        description = recorded.get("description")
        return {
            "code": str(recorded["code"]),
            "name": str(recorded.get("name") or project_name or recorded["code"]),
            "description": str(description) if description else None,
        }
    if not project_name:
        return None

    from osprey.facility import fold_code

    return {"code": fold_code(project_name), "name": project_name, "description": None}


def _recorded_identity(render_root: Path) -> dict[str, Any] | None:
    """Return the ``identity`` record of the render's facility file, if it has one.

    Args:
        render_root: The directory that holds the rendered ``config.yml``.

    Returns:
        dict[str, Any] | None: The record, or ``None`` when the file is absent,
        cannot be parsed, or carries no identity code.
    """
    from osprey.facility.render import FACILITY_FILE

    path = Path(render_root) / FACILITY_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("The facility file %s could not be read", path, exc_info=True)
        return None
    identity = document.get("identity") if isinstance(document, dict) else None
    if not isinstance(identity, dict) or not identity.get("code"):
        logger.warning("The facility file %s names no identity code", path)
        return None
    return identity


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
