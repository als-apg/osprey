"""Times the event dispatcher hands the agent, rendered in the facility zone.

The dispatcher keeps every instant in UTC. What the agent reads is rendered in the
facility zone that ``system.timezone`` names, the zone of the agent's own clock line.
Only an instant the dispatcher produced is rendered: JSON can carry no ``datetime``,
so the type tells a stamped instant from a caller's text, which is shown as it came.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from osprey.utils.config import to_facility_iso


def _render(value: Any) -> Any:
    """Render a value JSON cannot encode: a ``datetime`` in the facility zone, else ``str``."""
    if isinstance(value, datetime):
        return to_facility_iso(value)
    return str(value)


def agent_json(value: Any) -> str:
    """Serialize a value for the agent, rendering every ``datetime`` in the facility zone."""
    return json.dumps(value, indent=2, default=_render)


def registry_instant(stamp: str | None) -> str | None:
    """Render a registry's UTC ISO stamp in the facility zone; ``None`` stays ``None``."""
    if stamp is None:
        return None
    return to_facility_iso(datetime.fromisoformat(stamp))
