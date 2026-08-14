"""Core ``web_panels`` health category.

Reports whether each web panel the operator has enabled is actually reachable —
one row per panel, all probed concurrently.

This is the single home for panel liveness. The web terminal's rail renders only
the coarse enabled/disabled state — a browser-side poll would put a status board
in a navigation surface and report nothing at all while the terminal is closed.
The detailed readout lives here, where it sits beside the other core categories,
is reachable from ``osprey health`` and the MCP surface, and works with no
browser involved.

Panels come from two places and are probed accordingly:

* **Built-in panels** (``web.panels.<id>`` enabled, id in ``BUILTIN_PANELS``) —
  host/port resolved from :data:`FRAMEWORK_WEB_SERVERS` the same way
  ``server_launcher`` resolves them, then ``GET /health``.
* **Custom panels** (any other ``web.panels.<id>`` with a ``url``) — ``GET
  url + health_endpoint`` when one is configured. When it isn't (the scan
  ``plan``/``results`` panels declare none), the panel's own entry ``path`` is
  fetched instead and any non-5xx answer counts as reachable; the row says so
  rather than implying a real health contract exists.

Rows are advisory (``ok``/``warning``), matching the ``ariel`` and ``containers``
categories: a panel that is configured but down is a warning, never a suite
error, and an unreachable probe never propagates an exception. With no ``web``
block configured the category contributes no rows at all, so a minimal build
shows no panel tile.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from osprey.health.models import CheckResult, Status
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS, UNIVERSAL_PANELS
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "web_panels"

_PROBE_TIMEOUT_S = 5.0
_DEFAULT_BIND = "127.0.0.1"

#: Panel id (as it appears under ``web.panels``) → registry key in
#: :data:`FRAMEWORK_WEB_SERVERS`. The two namespaces differ for some panels
#: (``artifacts``/``artifact``, ``lattice``/``lattice_dashboard``), so the
#: mapping is explicit rather than derived by string munging.
_PANEL_TO_REGISTRY_KEY: dict[str, str] = {
    "artifacts": "artifact",
    "ariel": "ariel",
    "channel-finder": "channel_finder",
    "lattice": "lattice_dashboard",
    "okf": "okf",
    "system-health": "system_health",
}


class _Target(NamedTuple):
    """One panel to probe.

    Attributes:
        panel_id: The id under ``web.panels``, used as the row's name suffix.
        label: Display label for the row message.
        url: Absolute URL to fetch.
        contract: ``True`` when ``url`` is a declared health endpoint, ``False``
            when it is merely the panel's entry path being used as a
            reachability stand-in. Controls both the message wording and which
            status codes count as healthy.
    """

    panel_id: str
    label: str
    url: str
    contract: bool


def web_panels(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CategoryCallable:
    """Build the ``web_panels`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Read for ``web.panels`` and ``deployment.bind_address``.
        context: Health runtime. Unused — panels are probed over HTTP, no
            control-system connector is needed.
        transport: Optional httpx transport for dependency injection in tests
            (e.g. :class:`httpx.MockTransport`); ``None`` uses httpx's default.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        targets = _resolve_targets(cfg)
        if not targets:
            return []
        # Probe concurrently: a facility can enable half a dozen panels, and
        # serially waiting out the timeout on each unreachable one would push
        # the category past the suite deadline on its own.
        return list(await asyncio.gather(*(_probe(t, transport) for t in targets)))

    return _run


def _resolve_targets(cfg: Mapping[str, Any]) -> list[_Target]:
    """Enumerate the enabled panels and the URL each should be probed at.

    Mirrors ``web_terminal.app._load_panel_config``'s read of ``web.panels``:
    a built-in id is enabled by ``true`` or by a mapping without
    ``enabled: false``; anything else is a custom panel carrying its own url.
    Universal panels (``artifacts``) are always on, exactly as the terminal
    treats them.
    """
    web_cfg = cfg.get("web")
    if not isinstance(web_cfg, dict):
        return []
    panels_cfg = web_cfg.get("panels")
    if not isinstance(panels_cfg, dict):
        panels_cfg = {}

    bind = (cfg.get("deployment") or {}).get("bind_address", _DEFAULT_BIND)

    enabled_builtin: set[str] = set(UNIVERSAL_PANELS)
    targets: list[_Target] = []

    for panel_id, spec in panels_cfg.items():
        if panel_id in BUILTIN_PANELS:
            if spec is True or (isinstance(spec, dict) and spec.get("enabled", True)):
                enabled_builtin.add(panel_id)
            continue
        if not isinstance(spec, dict):
            continue
        target = _custom_target(panel_id, spec)
        if target is not None:
            targets.append(target)

    for panel_id in sorted(enabled_builtin):
        target = _builtin_target(panel_id, cfg, bind)
        if target is not None:
            targets.append(target)

    targets.sort(key=lambda t: t.panel_id)
    return targets


def _builtin_target(panel_id: str, cfg: Mapping[str, Any], bind: str) -> _Target | None:
    """Resolve a built-in panel's ``/health`` URL from the web-server registry.

    Host/port are read the way ``server_launcher`` reads them — the registry
    entry's ``config_key`` (optionally nested under ``config_web_subkey``) with
    the definition's defaults as fallback — so a facility that moves a panel's
    port in config is probed at the port it actually runs on.
    """
    registry_key = _PANEL_TO_REGISTRY_KEY.get(panel_id)
    if registry_key is None:
        return None
    defn = FRAMEWORK_WEB_SERVERS.get(registry_key)
    if defn is None:
        return None

    section = cfg.get(defn.config_key) or {}
    if not isinstance(section, dict):
        section = {}
    if defn.config_web_subkey:
        nested = section.get(defn.config_web_subkey) or {}
        section = nested if isinstance(nested, dict) else {}

    host = section.get("host", defn.host_default) or bind
    port = section.get("port", defn.port_default)
    label = BUILTIN_PANEL_LABELS.get(panel_id, panel_id.upper())
    return _Target(panel_id, label, f"http://{host}:{port}/health", contract=True)


def _custom_target(panel_id: str, spec: Mapping[str, Any]) -> _Target | None:
    """Resolve a custom panel's probe URL, or ``None`` when it declares no url.

    Prefers the configured ``health_endpoint``. Falling back to the panel's
    ``path`` is deliberate: ``plan``/``results`` declare no health endpoint, and
    reporting them as unprobeable would leave the two panels most likely to be
    served by a lagging container with no row at all.
    """
    url = str(spec.get("url") or "").rstrip("/")
    if not url:
        return None
    label = str(spec.get("label") or panel_id.upper())

    endpoint = spec.get("health_endpoint")
    if endpoint:
        return _Target(panel_id, label, f"{url}{endpoint}", contract=True)

    path = str(spec.get("path") or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    return _Target(panel_id, label, f"{url}{path}", contract=False)


async def _probe(target: _Target, transport: httpx.AsyncBaseTransport | None) -> CheckResult:
    """Fetch one panel and turn the outcome into a row.

    A declared health endpoint must answer 2xx. A reachability stand-in only has
    to answer at all below 500 — a panel entry path may legitimately redirect or
    require auth without the panel being down.
    """
    # `<category>.<panel id>`: the dashboard's fmtName() strips the leading
    # category segment and title-cases the rest, so the row reads "Channel
    # Finder" / "Events" rather than a redundant "Panel Channel Finder". The
    # PANEL ID is the identifier here, not the display label — a facility
    # renaming a custom panel's label must not rename its check.
    name = f"{CATEGORY}.{target.panel_id.replace('-', '_')}"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=_PROBE_TIMEOUT_S, transport=transport, follow_redirects=True
        ) as client:
            resp = await client.get(target.url)
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{target.label}: unreachable",
            value="offline",
            details=(
                f"{target.url} — {exc}. The panel is enabled in `web.panels` but is not "
                "answering; check that its server (an `osprey web` sidecar or the "
                "container backing a custom panel) is running."
            ),
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    healthy = resp.is_success if target.contract else resp.status_code < 500

    if healthy:
        return CheckResult(
            name,
            CATEGORY,
            Status.OK,
            f"{target.label}: reachable"
            + ("" if target.contract else " (no health endpoint configured)"),
            value="up",
            latency_ms=latency_ms,
        )

    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        f"{target.label}: HTTP {resp.status_code}",
        value="degraded",
        latency_ms=latency_ms,
        details=f"{target.url} answered {resp.status_code}.",
    )
