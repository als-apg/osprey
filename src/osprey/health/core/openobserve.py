"""Core ``openobserve`` health category.

Probes the OpenObserve telemetry store, but **only when it is deployed**. The
category stays a valid ``--category`` name at all times; when ``openobserve``
is not in the project's ``deployed_services`` it simply contributes no rows (a
silent skip). When deployed it emits two rows:

* ``openobserve_healthz`` — ``GET /healthz`` against
  ``deployment.bind_address`` + ``services.openobserve.port``; ``ok`` on HTTP 200,
  ``warning`` on any other status or when the store is unreachable (``running`` is
  not ``ready``);
* ``openobserve_retention`` — ``warning`` when ``services.openobserve.retention_days``
  is below OpenObserve's floor of 3 (a capless named volume is size-bounded only
  by retention), else ``ok``.

Both rows are advisory (``ok``/``warning``); the store's absence is never an error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "openobserve"

_HEALTHZ_TIMEOUT_S = 5.0
_RETENTION_FLOOR_DAYS = 3
_DEFAULT_PORT = 5080
_DEFAULT_BIND = "127.0.0.1"
_DEFAULT_RETENTION_DAYS = 14  # mirror the compose default


def openobserve(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CategoryCallable:
    """Build the ``openobserve`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``deployed_services``, ``services.openobserve``, and
            ``deployment.bind_address``.
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
        port = oo.get("port", _DEFAULT_PORT)
        bind = (cfg.get("deployment", {}) or {}).get("bind_address", _DEFAULT_BIND)
        retention = oo.get("retention_days", _DEFAULT_RETENTION_DAYS)

        return [
            await _check_healthz(bind, port, transport),
            _check_retention(retention),
        ]

    return _run


async def _check_healthz(
    bind: str, port: Any, transport: httpx.AsyncBaseTransport | None
) -> CheckResult:
    """Probe the ``/healthz`` readiness endpoint; ``running`` is not ``ready``."""
    url = f"http://{bind}:{port}/healthz"
    try:
        async with httpx.AsyncClient(timeout=_HEALTHZ_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(
            "openobserve_healthz",
            CATEGORY,
            Status.WARNING,
            f"Store unreachable at {url}: {exc}",
            details="Deploy it with `osprey deploy up`, or check the bind address / port.",
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
