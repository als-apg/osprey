"""Facility identity resolved from a project config.

One resolution order, shared by every reader that needs the facility display
name: the build path that renders the agent prompts and the interface apps that
label their UI. Keeping it in one place is the point — the two spellings drifted
precisely because each reader picked its own.
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_facility_name"]


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
