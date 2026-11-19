"""Discovery-by-convention for local, static panel bundles.

A *panel* is a directory holding a ``manifest.json``
(``id``/``label``/``entry``/``version``) plus the ``entry`` HTML and its assets
— a static file bundle, not a running server.  This module finds such bundles
under a conventioned location and wires them into the same hub surfaces that
runtime-registered (URL-backed) panels use, so they appear in the tab strip,
``GET /api/panels``, focus, and visibility with no parallel mechanism.

Two design rules govern discovery:

* **Fail closed (OSPREY Principle 1).**  A malformed manifest, a missing entry
  file, a page that never marks itself ready, or any other non-compliant UI check
  makes that one panel *not served* — it is logged and skipped, and never affects
  the other panels.  :func:`discover_panels` never raises.
* **Human gate = the existing opt-in.**  Discovery runs only when
  ``web.allow_runtime_panels`` is true (it defaults to ``False``).  The operator
  editing ``config.yml`` to enable it is the human approval; there is no separate
  auth subsystem.  Off by default means agent- or developer-authored panels are
  inert until a human opts in.

Serving is same-origin from disk via ``GET /panel-static/{id}/{path}`` (see
``routes/panels.py``); the URL reverse-proxy (``routes/proxy.py``) is for
URL-backed panels only and is left untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from osprey.interfaces.design_system.panels.manifest import load_manifest_file
from osprey.interfaces.design_system.panels.validator import (
    MANIFEST_FILENAME,
    validate_panel,
)
from osprey.profiles.web_panels import BUILD_PANELS, BUILD_PANELS_DIRNAME, BUILTIN_PANELS

logger = logging.getLogger(__name__)

#: Conventioned panel-bundle location, relative to the project working dir.
PANELS_DIRNAME = "panels"

__all__ = [
    "PANELS_DIRNAME",
    "DiscoveredPanel",
    "discover_panels",
    "apply_build_panels",
    "apply_discovered_panels",
]


@dataclass(frozen=True)
class DiscoveredPanel:
    """A validated, ready-to-serve local panel bundle.

    Attributes:
        id: The manifest ``id`` (stable kebab slug); also the ``/panel-static/{id}``
            serving prefix and the hub tab id.
        label: Human-readable display name from the manifest.
        entry: The manifest ``entry`` HTML filename, served for the bare
            ``/panel-static/{id}/`` request.
        directory: Absolute path to the panel bundle directory on disk.
    """

    id: str
    label: str
    entry: str
    directory: Path


def discover_panels(
    panels_root: str | Path, *, reserved: Iterable[str] = BUILTIN_PANELS
) -> list[DiscoveredPanel]:
    """Scan ``panels_root`` for compliant panel bundles, fail-closed.

    Each immediate subdirectory containing a ``manifest.json`` is a candidate.
    A candidate is validated with :func:`validate_panel`; if it has *any* error
    (malformed manifest, missing entry, raw hex colors, missing design-system
    link, a page that never marks itself ready, …) it is **logged and skipped** — never served, never fatal to the
    others.  A candidate whose id collides with a built-in panel or with an
    already-discovered panel is likewise skipped.

    Args:
        panels_root: The conventioned panels directory (``<project>/panels``).
            A non-existent or non-directory path yields an empty list.
        reserved: Ids a bundle here may not take. The framework's own panel
            ids by default.

    Returns:
        The compliant panels, ordered by directory name.  Never raises.
    """
    root = Path(panels_root)
    if not root.is_dir():
        return []

    discovered: list[DiscoveredPanel] = []
    seen_ids: set[str] = set()
    reserved_ids = set(reserved)

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        # Every per-candidate step is wrapped so *any* failure inspecting one
        # bundle (a non-UTF-8 or unreadable .css/.js that makes validate_panel's
        # read_text raise, a filesystem error, a manifest race) skips only that
        # panel — discovery never raises and one bad bundle never sinks the rest.
        try:
            if not child.is_dir():
                continue
            if not (child / MANIFEST_FILENAME).is_file():
                # Not a panel bundle — silent skip (e.g. a README or stray dir).
                continue

            errors = validate_panel(child)
            if errors:
                logger.warning(
                    "Panel %r is not served: %d validation error(s): %s",
                    child.name,
                    len(errors),
                    "; ".join(str(e) for e in errors),
                )
                continue

            # validate_panel already parsed the manifest cleanly; re-load for
            # the typed fields.
            manifest = load_manifest_file(child / MANIFEST_FILENAME)

            # Entry must be a plain bundle-relative path.  Serving is already
            # traversal-guarded, but rejecting an absolute/parent entry here
            # keeps such a bundle out entirely rather than relying on downstream
            # guards, and avoids a validation-time read outside the bundle.
            entry_path = Path(manifest.entry)
            if entry_path.is_absolute() or ".." in entry_path.parts:
                logger.warning(
                    "Panel %r is not served: entry %r is not a bundle-relative path.",
                    child.name,
                    manifest.entry,
                )
                continue

            if manifest.id in reserved_ids:
                logger.warning(
                    "Panel %r is not served: id %r collides with a built-in panel.",
                    child.name,
                    manifest.id,
                )
                continue
            if manifest.id in seen_ids:
                logger.warning(
                    "Panel %r is not served: id %r already discovered in another directory.",
                    child.name,
                    manifest.id,
                )
                continue

            seen_ids.add(manifest.id)
            discovered.append(
                DiscoveredPanel(
                    id=manifest.id,
                    label=manifest.label,
                    entry=manifest.entry,
                    directory=child.resolve(),
                )
            )
        except Exception:
            logger.warning(
                "Panel %r is not served: discovery raised while inspecting it.",
                child.name,
                exc_info=True,
            )
            continue

    return discovered


def apply_discovered_panels(app) -> list[DiscoveredPanel]:
    """Discover local panels and wire them into ``app.state`` (gated, fail-closed).

    No-op unless ``app.state.allow_runtime_panels`` is true — that config opt-in
    is the human gate.  When enabled, discovered panels are appended to the same
    ``custom_panels`` / ``visible_panels`` surfaces registered panels use (so the
    hub renders them with no extra code), given a same-origin
    ``/panel-static/{id}/`` URL, and recorded in ``app.state.discovered_panel_dirs``
    (``{id: DiscoveredPanel}``) for the local-file serving route.

    A discovered id that already exists in ``custom_panels`` (e.g. a
    runtime-registered panel) is skipped so discovery never clobbers a live
    registration.

    Args:
        app: The FastAPI app whose ``state`` has ``project_cwd``,
            ``allow_runtime_panels``, ``custom_panels``, and ``visible_panels``
            already populated (called from the lifespan after panel config load).

    Returns:
        The panels actually applied (may be shorter than what was discovered).
    """
    if not getattr(app.state, "allow_runtime_panels", False):
        return []

    discovered = discover_panels(Path(app.state.project_cwd) / PANELS_DIRNAME)
    if not discovered:
        return []

    custom_panels: list[dict] = list(getattr(app.state, "custom_panels", []))
    visible_panels: list[str] = list(getattr(app.state, "visible_panels", []))
    panel_dirs: dict[str, DiscoveredPanel] = dict(getattr(app.state, "discovered_panel_dirs", {}))
    existing_ids = {cp.get("id") for cp in custom_panels}

    applied: list[DiscoveredPanel] = []
    for panel in discovered:
        if panel.id in existing_ids:
            logger.warning(
                "Discovered panel %r is not served: id already registered as a custom panel.",
                panel.id,
            )
            continue
        custom_panels.append(
            {
                "id": panel.id,
                "label": panel.label,
                # Same-origin static URL; GET /api/panels leaves leading-slash
                # URLs as-is (only raw upstream URLs get the /panel/{id} rewrite).
                "url": f"/panel-static/{panel.id}/",
                "healthEndpoint": None,
                "path": "/",
                "discovered": True,
            }
        )
        if panel.id not in visible_panels:
            visible_panels.append(panel.id)
        panel_dirs[panel.id] = panel
        existing_ids.add(panel.id)
        applied.append(panel)

    app.state.custom_panels = custom_panels
    app.state.visible_panels = visible_panels
    app.state.discovered_panel_dirs = panel_dirs

    if applied:
        logger.info(
            "Discovered %d local panel(s) under %s/: %s",
            len(applied),
            PANELS_DIRNAME,
            ", ".join(p.id for p in applied),
        )
    return applied


def apply_build_panels(app) -> list[DiscoveredPanel]:
    """Wire the panels ``osprey build`` wrote into this render into ``app.state``.

    A panel in :data:`~osprey.profiles.web_panels.BUILD_PANELS` is declared by
    the render's config like any custom panel and has no url: its page is a
    bundle the build wrote under
    :data:`~osprey.profiles.web_panels.BUILD_PANELS_DIRNAME`. Each one the config
    enables is given the same-origin ``/panel-static/{id}/`` URL a discovered
    bundle has and recorded in ``app.state.discovered_panel_dirs``, so the one
    route serves both. The config is the opt-in here, so this runs whatever
    ``web.allow_runtime_panels`` says.

    A block that names its own ``url`` is the deployment's own panel under that
    id and is left alone. A bundle that is missing or fails the panel contract
    takes its panel off the rail, with a warning: an entry that could only
    answer 404 is not shown.

    Args:
        app: The FastAPI app whose ``state`` has ``project_cwd``,
            ``custom_panels`` and ``visible_panels`` populated.

    Returns:
        The panels wired.
    """
    custom_panels: list[dict] = list(getattr(app.state, "custom_panels", []))
    visible_panels: list[str] = list(getattr(app.state, "visible_panels", []))
    panel_dirs: dict[str, DiscoveredPanel] = dict(getattr(app.state, "discovered_panel_dirs", {}))
    root = Path(app.state.project_cwd) / BUILD_PANELS_DIRNAME

    applied: list[DiscoveredPanel] = []
    kept: list[dict] = []
    for entry in custom_panels:
        panel_id = entry.get("id")
        if panel_id not in BUILD_PANELS or entry.get("url"):
            kept.append(entry)
            continue
        found = [panel for panel in discover_panels(root, reserved=()) if panel.id == panel_id]
        if not found:
            logger.warning(
                "Panel %r is not served: no valid bundle under %s; rebuild the project.",
                panel_id,
                root / str(panel_id),
            )
            visible_panels = [pid for pid in visible_panels if pid != panel_id]
            continue
        kept.append({**entry, "url": f"/panel-static/{panel_id}/", "discovered": True})
        panel_dirs[panel_id] = found[0]
        applied.append(found[0])

    app.state.custom_panels = kept
    app.state.visible_panels = visible_panels
    app.state.discovered_panel_dirs = panel_dirs
    return applied
