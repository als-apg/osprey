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
  host/port resolved by ``registry.web.resolve_web_server_address``, the one
  resolver ``server_launcher`` itself is bound to, then ``GET /health``.
* **Custom panels** (any other ``web.panels.<id>`` with a ``url``) — ``GET
  url + health_endpoint`` when one is configured. When it isn't (the scan
  ``plan``/``results`` panels declare none), the panel's own entry ``path`` is
  fetched instead and any non-5xx answer counts as reachable; the row says so
  rather than implying a real health contract exists.

A built-in panel whose own config section writes ``host``/``port`` at a depth
nothing reads has no address to probe at all; it gets a row saying so, naming the
key and the depth it belongs at, instead of being probed at a port the operator
never chose.

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
from osprey.registry.web import (
    PANEL_ID_TO_REGISTRY_KEY,
    WebServerConfigDepthError,
    resolve_web_server_address,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "web_panels"

_PROBE_TIMEOUT_S = 5.0


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
            Read for ``web.panels`` and each panel's own config section.
        context: Health runtime. Unused — panels are probed over HTTP, no
            control-system connector is needed.
        transport: Optional httpx transport for dependency injection in tests
            (e.g. :class:`httpx.MockTransport`); ``None`` uses httpx's default.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        targets, config_rows = _resolve_targets(cfg)
        if not targets:
            return config_rows
        # Probe concurrently: a facility can enable half a dozen panels, and
        # serially waiting out the timeout on each unreachable one would push
        # the category past the suite deadline on its own.
        probed = list(await asyncio.gather(*(_probe(t, transport) for t in targets)))
        return sorted(probed + config_rows, key=lambda row: row.name)

    return _run


def _resolve_targets(cfg: Mapping[str, Any]) -> tuple[list[_Target], list[CheckResult]]:
    """Enumerate the enabled panels and the URL each should be probed at.

    Mirrors ``web_terminal.app._load_panel_config``'s read of ``web.panels``:
    a built-in id is enabled by ``true`` or by a mapping without
    ``enabled: false``; anything else is a custom panel carrying its own url.
    Universal panels (``artifacts``) are always on, exactly as the terminal
    treats them.

    Returns:
        ``(targets, config_rows)`` — the panels to probe, plus ready-made rows
        for panels whose own config section is malformed and therefore has no
        probeable address at all.
    """
    web_cfg = cfg.get("web")
    if not isinstance(web_cfg, dict):
        return [], []
    panels_cfg = web_cfg.get("panels")
    if not isinstance(panels_cfg, dict):
        panels_cfg = {}

    enabled_builtin: set[str] = set(UNIVERSAL_PANELS)
    targets: list[_Target] = []
    config_rows: list[CheckResult] = []

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
        try:
            target = _builtin_target(panel_id, cfg)
        except WebServerConfigDepthError as exc:
            config_rows.append(_misconfigured_row(panel_id, exc))
            continue
        if target is not None:
            targets.append(target)

    targets.sort(key=lambda t: t.panel_id)
    return targets, config_rows


def _builtin_target(panel_id: str, cfg: Mapping[str, Any]) -> _Target | None:
    """Resolve a built-in panel's ``/health`` URL from the web-server registry.

    Host and port come from ``registry.web.resolve_web_server_address`` — the one
    resolver ``ServerLauncher`` is partial-bound to. Re-deriving them here read
    the config section only, which meant a multi-user deployment (where compose
    exports ``OSPREY_<CONFIG_KEY>_PORT`` per user because the per-user containers
    share the host network namespace) was probed at the config port while the
    panel listened on the env port — every panel reported offline.

    Raises:
        WebServerConfigDepthError: The panel's config section writes host/port at
            a depth nothing reads, so no address can be resolved.
    """
    registry_key = PANEL_ID_TO_REGISTRY_KEY.get(panel_id)
    if registry_key is None:
        return None

    host, port = resolve_web_server_address(registry_key, cfg)
    label = BUILTIN_PANEL_LABELS.get(panel_id, panel_id.upper())
    return _Target(panel_id, label, f"http://{host}:{port}/health", contract=True)


def _misconfigured_row(panel_id: str, exc: WebServerConfigDepthError) -> CheckResult:
    """Report a panel whose config section cannot yield an address.

    A warning, not an error, to keep the category advisory: the loud refusal
    belongs to the launch path (``osprey web``'s pre-flight fails, the launcher
    declines to start the panel). What this row owes the operator is the exact
    key to fix.
    """
    label = BUILTIN_PANEL_LABELS.get(panel_id, panel_id.upper())
    return CheckResult(
        f"{CATEGORY}.{panel_id.replace('-', '_')}",
        CATEGORY,
        Status.WARNING,
        f"{label}: misconfigured",
        value="misconfigured",
        details=str(exc),
    )


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
