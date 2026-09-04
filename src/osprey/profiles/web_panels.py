"""Built-in web panel registry — single source of truth across the build chain.

The web terminal (``interfaces/web_terminal/app.py``), the template manifest
validator (``cli/templates/manifest.py``), and the preset profile validator
(``cli/build_profile_model.py``) all gate on this set. Drift between them is what
let unknown panel IDs slip past for two registries simultaneously.

The set itself is derived from ``registry.web.FRAMEWORK_WEB_SERVERS`` and
:data:`SIDECAR_PANELS` — the panel ids and the servers behind them are one fact,
and every place that kept its own copy of the panel-id ↔ registry-key relation
drifted from the others. Cross between the two namespaces via
``registry.web.PANEL_ID_TO_REGISTRY_KEY`` or ``WebServerDefinition.panel_id``,
never a local table; a sidecar panel id is in neither, so cross with ``.get``.
"""

from __future__ import annotations

from dataclasses import dataclass

from osprey.registry.web import FRAMEWORK_WEB_SERVERS

UNIVERSAL_PANELS: set[str] = {"artifacts"}


@dataclass(frozen=True)
class SidecarPanel:
    """A built-in panel served by a process the web terminal starts itself.

    The companion-server registry (``registry.web``) is scoped to servers
    ``ServerLauncher`` runs from a config section with a host, a port and an
    ``auto_launch`` key. A sidecar has none of those: it is started by the
    terminal's lifespan, listens on an ephemeral loopback port, and is reached
    only through the terminal's own panel proxy. Registering one there would
    hand it a port band, a port env var and a health probe that mean nothing.

    Attributes:
        name: One-line description, used as the build interview's menu entry
            for the panel id.
        factory_path: ``module:attribute`` of the sidecar class the terminal
            imports to start it. A dotted path rather than the object so this
            module stays importable without pulling the sidecar's dependencies
            into every process that reads the panel registry.
    """

    name: str
    factory_path: str


#: Built-in panels the web terminal serves from a sidecar process rather than a
#: registered companion server. Keyed by panel id, exactly like the ids derived
#: from the companion registry, so every consumer of :data:`BUILTIN_PANELS`
#: covers them without a second lookup.
SIDECAR_PANELS: dict[str, SidecarPanel] = {
    "jupyter": SidecarPanel(
        "JupyterLab notebooks",
        "osprey.interfaces.web_terminal.jupyter_sidecar:JupyterSidecar",
    ),
}

#: Every panel id the framework serves itself: the companion web servers, plus
#: the sidecars the terminal starts. Derived from both registries rather than
#: re-listed here.
#:
#: Registering a companion server IS registering its panel: the registry entry
#: already declares the panel id it is reached by
#: (``WebServerDefinition.panel_id``), and a second hand-written copy of that
#: namespace could only ever agree with the first by luck. Keeping it derived
#: also means a new panel needs no edit in this file at all.
BUILTIN_PANELS: set[str] = {
    definition.panel_id for definition in FRAMEWORK_WEB_SERVERS.values()
} | set(SIDECAR_PANELS)

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
    "jupyter": "NOTEBOOKS",
    "okf": "KNOWLEDGE",
    "system-health": "SYSTEM",
}

#: The bar items whose availability is a panel's presence, and the panel each
#: needs. The build-time half of ``BAR_ITEM_AVAILABILITY`` in
#: ``interfaces/web_terminal/app.py``: the server drops such an item from the
#: default it serves when the panel is absent, and this is what lets the build
#: say so first. ``identity`` is gated too, but on a runtime fact (a terminal
#: user or a deployment name) a build cannot judge, so it is deliberately not
#: here; ``tests/interfaces/web_terminal/test_bar_items_ssr.py`` pins the two
#: tables together.
BAR_ITEM_PANEL_GATES: dict[str, str] = {
    "bluesky-queue": "bluesky",
    "system-health": "system-health",
}

# Frontend fallback when a profile/config doesn't pin a default tab.
# The web terminal opens this tab first on cold load.
DEFAULT_PANEL_FALLBACK: str = "artifacts"


def panel_spec_enabled(spec: object) -> bool:
    """Whether a ``web.panels.<id>`` value is a panel this render shows.

    The one predicate behind every reader of a panel block — the web terminal's
    tab strip, the health probe, the credential grants keyed on a panel
    declaration — so they cannot answer differently for the same block. Builtin
    and custom blocks are read alike: a bare ``true`` or a mapping without
    ``enabled: false`` is on; ``enabled: false`` is off; anything that is not a
    block is off (fail closed). The build writes ``enabled`` onto every block a
    render carries from the profile's ``web_panels`` selection
    (:func:`osprey.cli.build_profile_panels.panel_selection_overrides`), so a
    block inherited for a tab the profile does not select reads as off here.

    Args:
        spec: The value under ``web.panels.<id>`` — a mapping, ``true``, or
            whatever a hand-edited config holds there.

    Returns:
        ``True`` if the panel is on.
    """
    if spec is True:
        return True
    if isinstance(spec, dict):
        return spec.get("enabled", True) is not False
    return False
