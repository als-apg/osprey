"""Built-in web panel registry — single source of truth across the build chain.

The web terminal (``interfaces/web_terminal/app.py``), the template manifest
validator (``cli/templates/manifest.py``), and the preset profile validator
(``cli/build_profile.py``) all gate on this set. Drift between them is what
let unknown panel IDs slip past for two registries simultaneously.
"""

from __future__ import annotations

UNIVERSAL_PANELS: set[str] = {"artifacts"}

BUILTIN_PANELS: set[str] = {
    "artifacts",
    "ariel",
    "channel-finder",
    "lattice",
    "okf",
}

# The event-dispatcher dashboard (``events``) is intentionally NOT a builtin: it
# is a URL-backed custom panel (the control-assistant preset sets
# ``web.panels.events.url`` to the dispatcher's ``/dashboard``). Listing it here
# would make ``_load_panel_config`` discard that url and the frontend has no
# builtin ``events`` tab, so the tab would never render.

# Canonical display labels for built-in panels.  This is the single source of
# truth consumed by the /api/panels endpoint and by MCP panel tools — do not
# duplicate these strings elsewhere in the framework.  ``events`` is omitted on
# purpose: it is a URL-backed custom panel (see above) and carries its own label.
BUILTIN_PANEL_LABELS: dict[str, str] = {
    "artifacts": "WORKSPACE",
    "ariel": "ARIEL",
    "channel-finder": "CHANNELS",
    "lattice": "LATTICE",
    "okf": "KNOWLEDGE",
}

# Frontend fallback when a profile/config doesn't pin a default tab.
# The web terminal opens this tab first on cold load.
DEFAULT_PANEL_FALLBACK: str = "artifacts"
