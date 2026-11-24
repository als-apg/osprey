"""Discovery-by-convention for local, static panel bundles.

A *panel* is a directory holding a ``manifest.json``
(``id``/``label``/``entry``/``version``) plus the ``entry`` HTML and its assets
— a static file bundle, not a running server.  This module finds such bundles
under a conventioned location and wires them into the same hub surfaces that
runtime-registered (URL-backed) panels use, so they appear in the tab strip,
``GET /api/panels``, focus, and visibility with no parallel mechanism.

Two design rules govern discovery:

* **Fail closed (OSPREY Principle 1).**  A finding of a panel rule that refuses
  (a malformed manifest, a missing entry file, a page that never marks itself
  ready, …) makes that one panel *not served*, and never affects the other
  panels.  A refused panel is returned with every refusing reason, so the
  server's log, its panels list and ``osprey build`` name it from one verdict.
  A finding of a rule that only reports leaves the panel served, and the
  server's log names it.  :func:`discover_panels` never raises.
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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from osprey.interfaces.design_system.errors import SourcedFinding
from osprey.interfaces.design_system.panels.manifest import load_manifest_file
from osprey.interfaces.design_system.panels.validator import (
    MANIFEST_FILENAME,
    PanelFinding,
    refusing,
    validate_panel,
)
from osprey.profiles.web_panels import (
    BUILD_PANELS,
    BUILD_PANELS_DIRNAME,
    BUILTIN_PANELS,
    panel_spec_enabled,
)

logger = logging.getLogger(__name__)

#: Conventioned panel-bundle location, relative to the project working dir.
PANELS_DIRNAME = "panels"

__all__ = [
    "PANELS_DIRNAME",
    "DiscoveredPanel",
    "DiscoveryFinding",
    "DiscoveryRule",
    "PanelDiscovery",
    "RefusedPanel",
    "configured_panel_ids",
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
        reported: The panel's findings of rules that only report, in check order.
    """

    id: str
    label: str
    entry: str
    directory: Path
    reported: tuple[PanelFinding, ...] = ()


class DiscoveryRule(StrEnum):
    """The refusals discovery makes itself, about where a bundle sits among the others.

    Every rule about a bundle's own content is the panel validator's
    (:class:`~osprey.interfaces.design_system.panels.validator.PanelRule`).
    """

    #: The manifest's ``entry`` is absolute or climbs out of the bundle.
    ENTRY_OUTSIDE_BUNDLE = "entry_outside_bundle"
    #: The manifest's id is one a built-in panel holds.
    ID_RESERVED = "id_reserved"
    #: The manifest's id is one a configured panel or an earlier bundle holds.
    ID_TAKEN = "id_taken"
    #: Inspecting the bundle raised, so nothing about it could be judged.
    UNINSPECTABLE = "uninspectable"


@dataclass(frozen=True)
class DiscoveryFinding(SourcedFinding):
    """One reason discovery refused a bundle that is not about its content.

    Attributes:
        rule: Which discovery check refused the bundle.
        message: Human-readable description of the refusal.
        source: The manifest path, or the bundle directory when it could not
            be inspected.
    """

    rule: DiscoveryRule


@dataclass(frozen=True)
class RefusedPanel:
    """A bundle under ``panels/`` that is not served, with every reason.

    Attributes:
        id: The bundle's directory name; it exists even when the manifest is
            the thing refused.
        findings: Every reason, in check order; never empty.
        claims: The manifest id when the manifest parsed, else ``None``. Read
            by the panels list to withhold a refusal naming a withheld panel;
            never serialized.
    """

    id: str
    findings: tuple[PanelFinding | DiscoveryFinding, ...]
    claims: str | None


@dataclass(frozen=True)
class PanelDiscovery:
    """The verdict on every bundle under one panels directory.

    Attributes:
        served: The compliant panels, in directory order.
        refused: The bundles not served, in directory order.
    """

    served: tuple[DiscoveredPanel, ...]
    refused: tuple[RefusedPanel, ...]


def configured_panel_ids(web_config: Mapping[str, Any]) -> set[str]:
    """The ids of the custom panels a ``web`` config block declares.

    The rule the web terminal's panel-config loader turns into custom panels:
    a ``web.panels`` block that is on (:func:`panel_spec_enabled`), is a
    mapping, and does not name a built-in panel. A bundle under ``panels/``
    may not take one of these ids.

    Args:
        web_config: The config's ``web`` mapping.

    Returns:
        The configured custom panel ids.
    """
    panels = web_config.get("panels")
    if not isinstance(panels, Mapping):
        return set()
    return {
        str(panel_id)
        for panel_id, spec in panels.items()
        if panel_spec_enabled(spec) and isinstance(spec, Mapping) and panel_id not in BUILTIN_PANELS
    }


def discover_panels(
    panels_root: str | Path,
    *,
    reserved: Iterable[str] = BUILTIN_PANELS,
    taken: Iterable[str] = (),
) -> PanelDiscovery:
    """Scan ``panels_root`` for compliant panel bundles, fail-closed.

    Each immediate subdirectory containing a ``manifest.json`` is a candidate.
    A candidate is validated with :func:`validate_panel`; if a finding refuses
    it (malformed manifest, missing entry, raw hex colors, missing design-system
    link, a page that never marks itself ready, …) it is refused with the
    validator's refusing findings — never served, never fatal to the others.
    A finding of a rule that only reports leaves it served, carried in
    :attr:`DiscoveredPanel.reported`.  A
    candidate whose entry leaves the bundle, whose id collides with a built-in
    panel, a configured panel or an already-discovered panel, or whose
    inspection raised is refused with a :class:`DiscoveryFinding`.

    Args:
        panels_root: The conventioned panels directory (``<project>/panels``).
            A non-existent or non-directory path yields an empty verdict.
        reserved: Ids a bundle here may not take. The framework's own panel
            ids by default.
        taken: Ids the deployment already configures as panels.

    Returns:
        The served panels and the refused ones, ordered by directory name.
        Never raises, never logs a refusal.
    """
    root = Path(panels_root)
    if not root.is_dir():
        return PanelDiscovery(served=(), refused=())

    discovered: list[DiscoveredPanel] = []
    refused: list[RefusedPanel] = []
    seen_ids: set[str] = set()
    reserved_ids = set(reserved)
    taken_ids = set(taken)

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

            findings = validate_panel(child)
            refused_findings = refusing(findings)
            if refused_findings:
                refused.append(RefusedPanel(child.name, tuple(refused_findings), None))
                continue

            # validate_panel already parsed the manifest cleanly; re-load for
            # the typed fields.
            manifest_path = child / MANIFEST_FILENAME
            manifest = load_manifest_file(manifest_path)

            # Entry must be a plain bundle-relative path.  Serving is already
            # traversal-guarded, but rejecting an absolute/parent entry here
            # keeps such a bundle out entirely rather than relying on downstream
            # guards, and avoids a validation-time read outside the bundle.
            entry_path = Path(manifest.entry)
            refusal: tuple[DiscoveryRule, str] | None = None
            if entry_path.is_absolute() or ".." in entry_path.parts:
                refusal = (
                    DiscoveryRule.ENTRY_OUTSIDE_BUNDLE,
                    f"entry {manifest.entry!r} is not a bundle-relative path",
                )
            elif manifest.id in reserved_ids:
                refusal = (
                    DiscoveryRule.ID_RESERVED,
                    f"id {manifest.id!r} collides with a built-in panel",
                )
            elif manifest.id in taken_ids:
                refusal = (
                    DiscoveryRule.ID_TAKEN,
                    f"id {manifest.id!r} is already a panel this deployment configures "
                    "under web.panels",
                )
            elif manifest.id in seen_ids:
                refusal = (
                    DiscoveryRule.ID_TAKEN,
                    f"id {manifest.id!r} already discovered in another directory",
                )
            if refusal is not None:
                rule, message = refusal
                finding = DiscoveryFinding(rule, message, str(manifest_path))
                refused.append(RefusedPanel(child.name, (finding,), manifest.id))
                continue

            seen_ids.add(manifest.id)
            discovered.append(
                DiscoveredPanel(
                    id=manifest.id,
                    label=manifest.label,
                    entry=manifest.entry,
                    directory=child.resolve(),
                    reported=tuple(findings),
                )
            )
        except Exception as exc:
            logger.debug("Inspecting panel %r raised.", child.name, exc_info=True)
            refused.append(
                RefusedPanel(
                    child.name,
                    (
                        DiscoveryFinding(
                            DiscoveryRule.UNINSPECTABLE,
                            f"could not be inspected: {type(exc).__name__}: {exc}",
                            str(child),
                        ),
                    ),
                    None,
                )
            )
            continue

    return PanelDiscovery(served=tuple(discovered), refused=tuple(refused))


def _log_refused(refused: RefusedPanel) -> None:
    """Log one refused panel on the server's log.

    A refusal made of the validator's findings keeps the validator's line;
    any other refusal names its reasons.
    """
    if all(isinstance(f, PanelFinding) for f in refused.findings):
        logger.warning(
            "Panel %r is not served: %d validation error(s): %s",
            refused.id,
            len(refused.findings),
            "; ".join(str(f) for f in refused.findings),
        )
        return
    logger.warning(
        "Panel %r is not served: %s.",
        refused.id,
        "; ".join(f.message for f in refused.findings),
    )


def _log_reported(panel: DiscoveredPanel) -> None:
    """Log the report-only findings of one served panel on the server's log."""
    if panel.reported:
        logger.info(
            "Panel %r is served with %d reported finding(s): %s",
            panel.id,
            len(panel.reported),
            "; ".join(str(f) for f in panel.reported),
        )


def apply_discovered_panels(app) -> list[DiscoveredPanel]:
    """Discover local panels and wire them into ``app.state`` (gated, fail-closed).

    No-op unless ``app.state.allow_runtime_panels`` is true — that config opt-in
    is the human gate.  When enabled, discovered panels are appended to the same
    ``custom_panels`` / ``visible_panels`` surfaces registered panels use (so the
    hub renders them with no extra code), given a same-origin
    ``/panel-static/{id}/`` URL, and recorded in ``app.state.discovered_panel_dirs``
    (``{id: DiscoveredPanel}``) for the local-file serving route.

    Every refused bundle is logged and recorded in ``app.state.refused_panels``
    (empty when the gate is off: nothing was judged); a served panel with
    report-only findings is logged with them. An id ``custom_panels``
    already holds is ``taken`` in the verdict, so discovery never clobbers a
    configured panel.

    Args:
        app: The FastAPI app whose ``state`` has ``project_cwd``,
            ``allow_runtime_panels``, ``custom_panels``, and ``visible_panels``
            already populated (called from the lifespan after panel config load).

    Returns:
        The panels applied.
    """
    app.state.refused_panels = []
    if not getattr(app.state, "allow_runtime_panels", False):
        return []

    custom_panels: list[dict] = list(getattr(app.state, "custom_panels", []))
    verdict = discover_panels(
        Path(app.state.project_cwd) / PANELS_DIRNAME,
        taken={str(cp["id"]) for cp in custom_panels if cp.get("id")},
    )
    for refused in verdict.refused:
        _log_refused(refused)
    app.state.refused_panels = list(verdict.refused)
    if not verdict.served:
        return []

    visible_panels: list[str] = list(getattr(app.state, "visible_panels", []))
    panel_dirs: dict[str, DiscoveredPanel] = dict(getattr(app.state, "discovered_panel_dirs", {}))

    applied: list[DiscoveredPanel] = []
    for panel in verdict.served:
        _log_reported(panel)
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
    takes its panel off the rail, with a warning naming every reason: an entry
    that could only answer 404 is not shown.

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
        verdict = discover_panels(root, reserved=())
        found = [panel for panel in verdict.served if panel.id == panel_id]
        if not found:
            for refused in verdict.refused:
                if refused.id == panel_id:
                    _log_refused(refused)
            logger.warning(
                "Panel %r is not served: no valid bundle under %s; rebuild the project.",
                panel_id,
                root / str(panel_id),
            )
            visible_panels = [pid for pid in visible_panels if pid != panel_id]
            continue
        _log_reported(found[0])
        kept.append({**entry, "url": f"/panel-static/{panel_id}/", "discovered": True})
        panel_dirs[panel_id] = found[0]
        applied.append(found[0])

    app.state.custom_panels = kept
    app.state.visible_panels = visible_panels
    app.state.discovered_panel_dirs = panel_dirs
    return applied
