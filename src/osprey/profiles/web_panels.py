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

from collections.abc import Iterable, Mapping
from typing import Any

from osprey.interfaces.common_middleware import MOUNT_SEGMENT_RE

# The sidecar registry lives in a module that imports nothing, so a reader that
# cannot afford this one's imports (the MCP server registry, the web credential
# module) reads the same objects. Re-exported: this module stays the place
# panel code imports panel names from.
from osprey.profiles.sidecar_panels import JUPYTER_PANEL_ID as JUPYTER_PANEL_ID
from osprey.profiles.sidecar_panels import SIDECAR_PANELS as SIDECAR_PANELS
from osprey.profiles.sidecar_panels import SIDECAR_REGISTRY as SIDECAR_REGISTRY
from osprey.profiles.sidecar_panels import ApplicationPanel as ApplicationPanel
from osprey.profiles.sidecar_panels import PanelRegistry as PanelRegistry
from osprey.profiles.sidecar_panels import SidecarPanel as SidecarPanel
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

UNIVERSAL_PANELS: set[str] = {"artifacts"}

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

#: :data:`BUILTIN_PANELS` in catalog order: the companion servers in registry
#: order, then the sidecars in registry order. A list of panels that leaves the
#: server (the rail the browser seeds from ``/api/panels``) is put in this order,
#: never in a set's: a set of strings iterates in an order drawn from the
#: process's hash seed, so every restart would reshuffle the rail.
BUILTIN_PANEL_ORDER: tuple[str, ...] = (
    *dict.fromkeys(definition.panel_id for definition in FRAMEWORK_WEB_SERVERS.values()),
    *SIDECAR_PANELS,
)


def in_catalog_order(panel_ids: Iterable[str]) -> list[str]:
    """*panel_ids* as a list in :data:`BUILTIN_PANEL_ORDER`, without duplicates.

    An id the catalog does not name sorts after every built-in, alphabetically,
    so the result never depends on how *panel_ids* happened to iterate.
    """
    rank = {panel_id: index for index, panel_id in enumerate(BUILTIN_PANEL_ORDER)}
    return sorted(set(panel_ids), key=lambda pid: (rank.get(pid, len(rank)), pid))


# The event-dispatcher dashboard (``events``) is intentionally NOT a builtin: it
# is a URL-backed custom panel (the demo preset sets
# ``web.panels.events.url`` to the dispatcher's ``/dashboard``). Listing it here
# would make ``_load_panel_config`` discard that url and the frontend has no
# builtin ``events`` tab, so the tab would never render.

# Canonical display labels for built-in panels.  This is the single source of
# truth consumed by the /api/panels endpoint and by MCP panel tools — do not
# duplicate these strings elsewhere in the framework.  ``events`` is omitted on
# purpose: it is a URL-backed custom panel (see above) and carries its own label.
# A sidecar's label is its registry entry's, so the entry is the one place a
# sidecar panel is described.
BUILTIN_PANEL_LABELS: dict[str, str] = {
    "artifacts": "Artifacts",
    "ariel": "LOGBOOK",
    "channel-finder": "CHANNELS",
    "okf": "KNOWLEDGE",
    "system-health": "Health",
} | {panel_id: panel.label or panel_id.upper() for panel_id, panel in SIDECAR_PANELS.items()}

#: Sidecar panels whose config route predates the generic
#: ``/api/panels/<id>/server``. The older route still answers, and the page
#: keeps naming it for these ids, so a browser bundle or a client that polls it
#: reads the same body as before.
SIDECAR_CONFIG_ENDPOINTS: dict[str, str] = {JUPYTER_PANEL_ID: "/api/jupyter-server"}


def _authored_text(panels_config: object, panel_id: str, key: str) -> str | None:
    """The non-empty string a ``web.panels.<panel_id>`` block writes under *key*.

    Anything else there (an absent key, an empty string, a value that is not a
    string, a block that is not a mapping) is ``None``: the build refuses those
    spellings (:func:`osprey.cli.build_profile_panels.panel_presentation_errors`),
    and a hand-edited config that still carries one reads as having said nothing.
    """
    if not isinstance(panels_config, Mapping):
        return None
    spec = panels_config.get(panel_id)
    if not isinstance(spec, Mapping):
        return None
    value = spec.get(key)
    return value if isinstance(value, str) and value.strip() else None


def panel_labels(panels_config: object = None) -> dict[str, str]:
    """The built-in panels' display labels as this deployment names them.

    :data:`BUILTIN_PANEL_LABELS` is the framework's wording. A deployment that
    calls a panel something else says so once, as ``web.panels.<id>.label``,
    the key a custom panel has always been named by, and every reader of a
    built-in label (the page's roster stamp, ``/api/panels``, the health rows,
    the sidecar status sentences) reads it through here.

    Args:
        panels_config: The config's ``web.panels`` mapping, or ``None`` for the
            registry's own labels.

    Returns:
        One label per built-in panel id, registry order.
    """
    return {
        panel_id: _authored_text(panels_config, panel_id, "label") or label
        for panel_id, label in BUILTIN_PANEL_LABELS.items()
    }


def authored_panel_labels(panels_config: object) -> list[str]:
    """Every label this deployment wrote on a ``web.panels`` block.

    The sidebar reads an all-capitals label in title case, which is the right
    reading of a name the framework derives (a built-in's canonical label, an
    id standing in for a missing one) and the wrong one for a name the
    deployment wrote: an acronym has no lower-case letter to mark it as the
    deployment's own spelling. The page stamps these so the sidebar shows each
    as written.

    Args:
        panels_config: The config's ``web.panels`` mapping, or ``None``.

    Returns:
        The distinct labels, sorted; empty when the deployment writes none.
    """
    if not isinstance(panels_config, Mapping):
        return []
    labels = {_authored_text(panels_config, panel_id, "label") for panel_id in panels_config}
    return sorted(label for label in labels if label)


def panel_label(panel_id: str, panels_config: object = None) -> str:
    """The display label of one panel, built-in or config-declared.

    Args:
        panel_id: The panel's id.
        panels_config: The config's ``web.panels`` mapping, or ``None``.

    Returns:
        ``web.panels.<panel_id>.label`` when the block writes one, else the
        registry's label for a built-in, else the id in capitals.
    """
    return (
        _authored_text(panels_config, panel_id, "label")
        or BUILTIN_PANEL_LABELS.get(panel_id)
        or panel_id.upper()
    )


def panel_icons(panels_config: object) -> dict[str, str]:
    """The sidebar glyph each ``web.panels.<id>.icon`` names, by panel id.

    One key for built-in and config-declared panels alike. A panel the mapping
    does not give an icon keeps the one it has: the catalog's for a companion
    panel, the registry entry's for a sidecar, the generic glyph for a custom
    panel. The names are the design system's (``icons.js``); this function does
    not judge them, because the build does
    (:func:`osprey.cli.build_profile_panels.panel_presentation_errors`) and the
    browser draws only a name it knows.

    Args:
        panels_config: The config's ``web.panels`` mapping.

    Returns:
        Panel id to icon name, for every block that writes one, config order.
    """
    if not isinstance(panels_config, Mapping):
        return {}
    icons: dict[str, str] = {}
    for panel_id in panels_config:
        icon = _authored_text(panels_config, panel_id, "icon")
        if icon:
            icons[str(panel_id)] = icon
    return icons


def sidecar_panel_descriptors(panels_config: object = None) -> list[dict[str, Any]]:
    """The browser's descriptor of every sidecar panel, in registry order.

    The page stamps this list on ``<html data-sidecar-panels>`` and
    ``panel-catalog.js`` builds one sidebar row and tab per entry from it, so a
    sidecar added to :data:`SIDECAR_PANELS` needs no edit in the browser
    bundle. The keys are the catalog's own field names. Read at call time, so
    it answers for the registry as it stands.

    Args:
        panels_config: The config's ``web.panels`` mapping. A sidecar's block
            may rename the panel (``label``) and pick its glyph (``icon``);
            ``None`` answers with the registry's own.

    Returns:
        One mapping per sidecar: ``id``, ``label``, ``icon`` (``None`` for the
        generic glyph), ``description`` (``None`` for a bare label),
        ``configEndpoint``, ``healthEndpoint`` (``None`` for a panel that is
        not polled) and ``startEndpoint``.
    """
    return [
        {
            "id": panel_id,
            "label": (
                _authored_text(panels_config, panel_id, "label") or panel.label or panel_id.upper()
            ),
            "icon": _authored_text(panels_config, panel_id, "icon") or panel.icon or None,
            "description": panel.description or None,
            "configEndpoint": SIDECAR_CONFIG_ENDPOINTS.get(
                panel_id, f"/api/panels/{panel_id}/server"
            ),
            "healthEndpoint": panel.health_endpoint,
            "startEndpoint": f"/api/panels/{panel_id}/start",
        }
        for panel_id, panel in SIDECAR_PANELS.items()
    ]


# Frontend fallback when a profile/config doesn't pin a default tab.
# The web terminal opens this tab first on cold load.
DEFAULT_PANEL_FALLBACK: str = "artifacts"


def panel_id_refusal(panel_id: str) -> str | None:
    """Say why *panel_id* cannot name a panel, or ``None`` when it can.

    A panel id is one URL path segment — ``/panel/<id>`` through the reverse
    proxy, ``/panel-static/<id>/`` for a local bundle — and the proxy also
    splices it into the ``x-forwarded-prefix`` value of every request it
    forwards for that panel, escaping it for neither. The id therefore has to
    stand for itself in a path and in a header alike: ASCII, and nothing a path
    would have to percent-encode. That is the question a container's URL mount
    asks, so the answer is that mount's class
    (:data:`~osprey.interfaces.common_middleware.MOUNT_SEGMENT_RE`) rather than
    a second one — and its leading-alphanumeric rule is what keeps ``.`` and
    ``..`` from naming a panel, since ``/panel/..`` climbs out of the panel it
    claims to name wherever something resolves it.

    One implementation for two readers: the terminal refuses to start on a
    declared id it cannot serve, and the build lint
    (:func:`osprey.cli.build_profile_panels.panel_id_errors`) refuses the same
    id in a profile, long before a container is built from it.

    Args:
        panel_id: The id declared under ``web.panels``.

    Returns:
        One message naming the config key, the id and the class it has to
        match, or ``None`` for an id the terminal can serve.
    """
    if MOUNT_SEGMENT_RE.fullmatch(panel_id):
        return None
    return (
        f"web.panels.{panel_id}: a panel id must match {MOUNT_SEGMENT_RE.pattern!r}, "
        f"and {panel_id!r} does not. The id is spliced unescaped into the panel's own "
        "URL path and into the forwarded-prefix header of every request the proxy "
        "carries for it, so a character outside that class reaches no route and fails "
        "the hop that would serve the panel. Rename the panel."
    )


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
