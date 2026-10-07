"""Core ``openobserve`` health category.

Probes the OpenObserve telemetry store, but **only when it is deployed**. The
category stays a valid ``--category`` name at all times; when ``openobserve``
is not in the project's ``deployed_services`` it simply contributes no rows (a
silent skip). When deployed it emits two rows:

* ``openobserve_healthz`` — ``GET /healthz`` against
  ``deployment.bind_address`` + ``services.openobserve.port`` (with no such key,
  the layout's ``openobserve`` slot at this deployment's ``deployment.port_base``),
  the port being
  :func:`~osprey.build.claude_code_telemetry.openobserve_published_port`'s;
  ``ok`` on HTTP 200, ``warning`` on any other status or when the store is
  unreachable (``running`` is not ``ready``);
* ``openobserve_retention`` — ``warning`` when ``services.openobserve.retention_days``
  is below OpenObserve's floor of 3 (a capless named volume is size-bounded only
  by retention), else ``ok``.

Both rows are advisory (``ok``/``warning``); the store's absence is never an error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from osprey.build.claude_code_telemetry import OPENOBSERVE_PORT_REMEDY, openobserve_published_port
from osprey.deployment.qmd_service import DEFAULT_BIND_ADDRESS, dial_address
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "openobserve"

_HEALTHZ_TIMEOUT_S = 5.0
_RETENTION_FLOOR_DAYS = 3
_DEFAULT_RETENTION_DAYS = 14  # mirror the compose default


def openobserve(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CategoryCallable:
    """Build the ``openobserve`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``deployed_services``, ``services.openobserve``,
            ``deployment.bind_address`` and ``deployment.port_base``.
        context: Health runtime. Unused — no control-system connector is needed.
        transport: Optional httpx transport for dependency injection in tests
            (e.g. :class:`httpx.MockTransport`); ``None`` uses httpx's default.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        deployed = cfg.get("deployed_services", []) or []
        if "openobserve" not in deployed:
            return []

        oo = (cfg.get("services", {}) or {}).get("openobserve", {}) or {}
        bind = (cfg.get("deployment", {}) or {}).get("bind_address", DEFAULT_BIND_ADDRESS)
        retention = oo.get("retention_days", _DEFAULT_RETENTION_DAYS)

        try:
            port = openobserve_published_port(cfg)
        except ValueError as exc:
            healthz = CheckResult(
                "openobserve_healthz",
                CATEGORY,
                Status.WARNING,
                str(exc),
                details=OPENOBSERVE_PORT_REMEDY,
            )
        else:
            healthz = await _check_healthz(bind, port, transport)

        return [healthz, _check_retention(retention)]

    return _run


async def _check_healthz(
    bind: str, port: Any, transport: httpx.AsyncBaseTransport | None
) -> CheckResult:
    """Probe the ``/healthz`` readiness endpoint; ``running`` is not ``ready``.

    The bind address is resolved to a dialable host first
    (:func:`~osprey.deployment.qmd_service.dial_address`): a wildcard publish is
    reached on loopback, a pinned interface on that interface.
    """
    url = f"http://{dial_address(bind)}:{port}/healthz"
    try:
        async with httpx.AsyncClient(timeout=_HEALTHZ_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(
            "openobserve_healthz",
            CATEGORY,
            Status.WARNING,
            f"Store unreachable at {url}: {exc}",
            details="Deploy it with `osprey up`, or check the bind address / port.",
        )

    if resp.status_code == 200:
        return CheckResult("openobserve_healthz", CATEGORY, Status.OK, f"Store ready ({url})")
    return CheckResult(
        "openobserve_healthz",
        CATEGORY,
        Status.WARNING,
        f"/healthz returned HTTP {resp.status_code}",
    )


def _check_retention(retention: Any) -> CheckResult:
    """Warn when the configured retention is below OpenObserve's 3-day floor."""
    if isinstance(retention, int) and retention < _RETENTION_FLOOR_DAYS:
        return CheckResult(
            "openobserve_retention",
            CATEGORY,
            Status.WARNING,
            f"retention_days={retention} is below OpenObserve's floor of {_RETENTION_FLOOR_DAYS}",
            details="OpenObserve will not honor a retention under 3 days.",
        )
    return CheckResult(
        "openobserve_retention", CATEGORY, Status.OK, f"Retention: {retention} day(s)"
    )
