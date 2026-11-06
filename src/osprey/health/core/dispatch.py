"""Core ``dispatch`` health category.

Reads the event dispatcher's own surfaces, but **only when the dispatcher is
configured**: presence is keyed on the ``services.event_dispatcher`` block, the
one the compose fragment and the port layout read. Without it the category
contributes no rows. The dispatcher is dialed on loopback at that block's
``port`` (the layout's ``dispatcher`` slot at this config's base when unset),
where it is published for the host's processes and every host-network render.

* ``dispatch.pool`` — ``GET /health``: running dispatches out of the pool size,
  plus the queue when anything waits (``ok``); ``warning`` when the dispatcher
  does not answer or answers with an error status;
* ``dispatch.failed_24h`` — ``GET /dashboard/state?timeline_hours=24``: worker
  runs created in the last 24 hours whose status is ``error`` or ``failed``.
  ``ok`` at zero, ``warning`` otherwise (naming the newest failed run); a
  ``warning`` when the dispatcher cannot read its worker's runs or refuses the
  credential. The route is bearer-gated on ``EVENT_DISPATCHER_TOKEN``; a process
  that does not hold that variable gets a ``skip`` row rather than a guess.
  Absent when the pool row already says the dispatcher is unreachable.

Every row is advisory: a stopped dispatcher is a service that is not running,
not a broken build.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "dispatch"

#: Per-request bound in seconds.
_TIMEOUT_S = 5.0

#: The worker run statuses that count as a failed run.
_FAILED = frozenset({"error", "failed"})

_WINDOW_S = 24 * 3600.0
_TOKEN_ENV = "EVENT_DISPATCHER_TOKEN"  # noqa: S105 -- the variable name, not a secret


def dispatch(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    now: Callable[[], float] | None = None,
) -> CategoryCallable:
    """Build the ``dispatch`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``services.event_dispatcher`` and ``deployment.port_base``.
        context: Health runtime. Unused.
        transport: Optional httpx transport, for tests.
        now: Epoch-seconds clock, for tests; defaults to :func:`time.time`.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}
    clock = now or time.time

    async def _run() -> list[CheckResult]:
        base_url = _dispatcher_url(cfg)
        if base_url is None:
            return []
        async with httpx.AsyncClient(
            base_url=base_url, timeout=_TIMEOUT_S, transport=transport
        ) as client:
            pool = await _pool_row(client)
            if pool.status is not Status.OK:
                return [pool]
            return [pool, await _failed_row(client, clock())]

    return _run


def _dispatcher_url(cfg: Mapping[str, Any]) -> str | None:
    """Loopback URL of the configured dispatcher, or ``None`` when none is configured."""
    services = cfg.get("services")
    block = services.get("event_dispatcher") if isinstance(services, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    port = block.get("port")
    if not isinstance(port, int) or isinstance(port, bool):
        from osprey.port_layout import default_port, resolve_port_base

        port = default_port("dispatcher", base=resolve_port_base(cfg))
    return f"http://127.0.0.1:{port}"


def _unreachable(name: str, message: str, value: str, latency_ms: float) -> CheckResult:
    return CheckResult(
        name, CATEGORY, Status.WARNING, message, value=value, latency_ms=latency_ms, probed=True
    )


async def _pool_row(client: httpx.AsyncClient) -> CheckResult:
    name = "dispatch.pool"
    started = time.perf_counter()
    try:
        response = await client.get("/health")
    except httpx.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        row = _unreachable(
            name, f"Event dispatcher unreachable at {client.base_url}", "unreachable", elapsed
        )
        row.details = str(exc) or type(exc).__name__
        return row
    elapsed = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        return _unreachable(
            name,
            f"Event dispatcher answered HTTP {response.status_code}",
            f"HTTP {response.status_code}",
            elapsed,
        )
    try:
        pool = response.json().get("pool") or {}
        running, size = int(pool.get("running", 0)), int(pool.get("max", 0))
        queued = int(pool.get("queued", 0))
    except (ValueError, AttributeError, TypeError):
        return _unreachable(
            name, "Event dispatcher /health is not a pool report", "unreadable", elapsed
        )
    value = f"{running} / {size} busy" + (f" · {queued} queued" if queued else "")
    return CheckResult(
        name,
        CATEGORY,
        Status.OK,
        f"Dispatch pool: {running} of {size} slots running",
        value=value,
        latency_ms=elapsed,
        probed=True,
    )


async def _failed_row(client: httpx.AsyncClient, current: float) -> CheckResult:
    name = "dispatch.failed_24h"
    token = os.environ.get(_TOKEN_ENV, "")
    if not token:
        return CheckResult(
            name,
            CATEGORY,
            Status.SKIP,
            f"Run history needs {_TOKEN_ENV}, which this process does not hold",
            value="no dispatcher token here",
        )
    started = time.perf_counter()
    try:
        response = await client.get(
            "/dashboard/state",
            params={"timeline_hours": "24"},
            headers={"Authorization": f"Bearer {token}"},
        )
    except httpx.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        row = _unreachable(name, "Dispatcher run history unreachable", "unreachable", elapsed)
        row.details = str(exc) or type(exc).__name__
        return row
    elapsed = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        return _unreachable(
            name,
            f"Dispatcher run history answered HTTP {response.status_code}",
            f"HTTP {response.status_code}",
            elapsed,
        )
    try:
        state = response.json()
        runs = state.get("runs") or []
        worker_error = state.get("worker_error")
    except (ValueError, AttributeError):
        return _unreachable(name, "Dispatcher run history is unreadable", "unreadable", elapsed)
    if worker_error:
        row = _unreachable(
            name, "The dispatcher cannot read its worker's runs", "worker unreachable", elapsed
        )
        row.details = str(worker_error)
        return row
    failed = sorted(
        (
            run
            for run in runs
            if isinstance(run, Mapping)
            and run.get("status") in _FAILED
            and _created(run) >= current - _WINDOW_S
        ),
        key=_created,
        reverse=True,
    )
    if not failed:
        return CheckResult(
            name,
            CATEGORY,
            Status.OK,
            "No failed dispatch runs in the last 24 h",
            value="0",
            latency_ms=elapsed,
            probed=True,
        )
    newest = str(failed[0].get("run_id") or "?")[:8]
    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        f"{len(failed)} failed dispatch run(s) in the last 24 h",
        value=f"{len(failed)} · run {newest}",
        details="\n".join(str(run.get("run_id")) for run in failed),
        latency_ms=elapsed,
        probed=True,
    )


def _created(run: Mapping[str, Any]) -> float:
    try:
        return float(run.get("created_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0
