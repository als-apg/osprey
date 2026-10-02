"""Facility identity and facility zone, resolved in one place.

Every reader that needs the facility's name asks here, so the build path that
renders the agent prompts and the servers and apps that label their answers
cannot each pick their own spelling.
"""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any, TypedDict
from zoneinfo import available_timezones

from osprey.facility import FACILITY_FILE, fold_code

__all__ = [
    "DEFAULT_FACILITY_ZONE",
    "SET_FACILITY_ZONE",
    "FacilityFileError",
    "FacilityIdentity",
    "closest_zone_name",
    "facility_identity",
    "is_zone_name",
]

#: The zone every reader falls back to and every preset pins.
DEFAULT_FACILITY_ZONE = "UTC"

#: The one remedy sentence the build reminder and the health row both print.
SET_FACILITY_ZONE = (
    "Set system.timezone under `config:` in profile.yml to your facility's zone, "
    "for example America/New_York."
)


class FacilityFileError(ValueError):
    """A render's facility file is present but names no usable identity.

    Only an absent file falls back to the project name: a file that is there
    and cannot be read, is not JSON, or names no identity code is a broken
    render, and a reader that labelled it with the project name would hide
    that.
    """


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
    one answers with the identity the build would write for a project that
    authors none: the project name as the display name and its fold as the
    code.

    Args:
        render_root: The directory that holds the rendered ``config.yml``.
        project_name: The project's name, used where the facility file names no
            display name and as the whole identity where there is no file.

    Returns:
        FacilityIdentity | None: The identity, or ``None`` when there is neither
        a facility file nor a project name.

    Raises:
        FacilityFileError: The facility file is present but cannot be read, is
            not JSON, or names no identity code.
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

    return {"code": fold_code(project_name), "name": project_name, "description": None}


def _recorded_identity(render_root: Path) -> dict[str, Any] | None:
    """Return the ``identity`` record of the render's facility file, if it has one.

    Args:
        render_root: The directory that holds the rendered ``config.yml``.

    Returns:
        dict[str, Any] | None: The record, or ``None`` when the file is absent.

    Raises:
        FacilityFileError: The file is present but cannot be read, is not
            JSON, or names no identity code.
    """
    path = Path(render_root) / FACILITY_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise FacilityFileError(f"The facility file {path} cannot be read: {exc}") from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise FacilityFileError(f"The facility file {path} is not JSON: {exc}") from exc
    identity = document.get("identity") if isinstance(document, dict) else None
    if not isinstance(identity, dict) or not identity.get("code"):
        raise FacilityFileError(f"The facility file {path} names no identity code")
    return identity


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
