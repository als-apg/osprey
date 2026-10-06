"""``system.timezone`` judged on the profile's merged ``config:`` block.

The two commands that gate on the profile, ``osprey build`` and
``osprey validate``, call these producers beside the other profile-side checks.
The render bakes the zone into the agent's rules and every container's ``TZ``,
so a name no reader can open is refused before anything is written.

A value that references the environment (it contains ``$``) is judged neither
way here. The profile-side pass resolves nothing from the environment: the
shell that runs the build is not the deployment's ``.env``. ``osprey health``
loads that ``.env`` and judges the resolved value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.utils.facility import (
    DEFAULT_FACILITY_ZONE,
    SET_FACILITY_ZONE,
    closest_zone_name,
    is_zone_name,
)

TIMEZONE_KEY = "system.timezone"


def system_timezone_errors(config: Mapping[str, Any]) -> list[str]:
    """One message per spelling of ``system.timezone`` that names no zone.

    Args:
        config: The profile's merged ``config:`` block.

    Returns:
        list[str]: Messages naming the key as written, the value and one fix.
    """
    if not isinstance(config, Mapping):
        return []

    # Imported in-function: the reach registry behind `spelled_values` pulls
    # the whole service-resolution package in with it.
    from .build_profile_reach import spelled_values

    errors: list[str] = []
    for spelling, value in spelled_values(config, TIMEZONE_KEY):
        if isinstance(value, str) and "$" in value:
            continue
        if isinstance(value, str) and value and is_zone_name(value):
            continue
        message = (
            f"The profile's config: block sets `{spelling}` to {value!r}, which names no time zone."
        )
        hint = closest_zone_name(value) if isinstance(value, str) else None
        if hint is not None:
            message += f" Did you mean {hint!r}?"
        message += " Zone names are spelled as in the IANA time zone database, case included."
        errors.append(message)
    return errors


def system_timezone_reminders(config: Mapping[str, Any]) -> list[str]:
    """At most one reminder, when ``system.timezone`` is still UTC.

    Unset counts as UTC, since every reader then falls back to it. A value
    that references the environment is left to ``osprey health``, and a value
    that names no zone is already refused by :func:`system_timezone_errors`.

    Args:
        config: The profile's merged ``config:`` block.

    Returns:
        list[str]: One reminder, or none.
    """
    if not isinstance(config, Mapping):
        return []

    from .build_profile_reach import spelled_values

    values = [value for _, value in spelled_values(config, TIMEZONE_KEY)]
    if not values or any(value == DEFAULT_FACILITY_ZONE for value in values):
        return [
            f"system.timezone is UTC, so operator times are read and shown in UTC. {SET_FACILITY_ZONE}"
        ]
    return []
