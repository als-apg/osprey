"""Built-in web panel registry — single source of truth across the build chain.

The web terminal (``interfaces/web_terminal/app.py``), the template manifest
validator (``cli/templates/manifest.py``), and the preset profile validator
(``cli/build_profile_model.py``) all gate on this set. Drift between them is what
let unknown panel IDs slip past for two registries simultaneously.

The set itself is derived from ``registry.web.FRAMEWORK_WEB_SERVERS`` — the panel
ids and the companion servers behind them are one fact, and every place that
kept its own copy of the panel-id ↔ registry-key relation drifted from the
others. Cross between the two namespaces via
``registry.web.PANEL_ID_TO_REGISTRY_KEY`` or ``WebServerDefinition.panel_id``,
never a local table.
"""

from __future__ import annotations

from osprey.registry.web import FRAMEWORK_WEB_SERVERS

UNIVERSAL_PANELS: set[str] = {"artifacts"}

#: Every panel id the framework serves itself, derived from the companion
#: web-server registry rather than re-listed here.
#:
#: Registering a companion server IS registering its panel: the registry entry
#: already declares the panel id it is reached by
#: (``WebServerDefinition.panel_id``), and a second hand-written copy of that
#: namespace could only ever agree with the first by luck. Keeping it derived
#: also means a new panel needs no edit in this file at all.
BUILTIN_PANELS: set[str] = {definition.panel_id for definition in FRAMEWORK_WEB_SERVERS.values()}

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
    "system-health": "SYSTEM",
}

# Frontend fallback when a profile/config doesn't pin a default tab.
# The web terminal opens this tab first on cold load.
DEFAULT_PANEL_FALLBACK: str = "artifacts"
