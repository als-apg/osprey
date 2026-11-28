"""The names a Bluesky run can record: a device's name and each data key.

A run refuses an object name or data key that holds ``.`` or ``/``, so a plan
device is named only by a name :func:`is_plan_name` accepts. The pattern is
event-model's own grammar for every key of an event descriptor; a framework test
holds this copy equal to the one event-model ships.
"""

from __future__ import annotations

import re

__all__ = ["PLAN_NAME_PATTERN", "is_plan_name"]

PLAN_NAME_PATTERN = "^([^./]+)$"
"""The pattern every name a run records must match."""


def is_plan_name(name: str) -> bool:
    """Whether a run can record ``name`` as a device name or a data key.

    Matches the way a JSON Schema validator matches the pattern, so the verdict
    and the run's agree on every name.

    Args:
        name: A channel address or device id.

    Returns:
        True when ``name`` matches :data:`PLAN_NAME_PATTERN`.
    """
    return re.search(PLAN_NAME_PATTERN, name) is not None
