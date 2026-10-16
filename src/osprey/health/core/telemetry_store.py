"""Core ``telemetry_store`` health category.

One row, **only when ``telemetry_store`` is in ``deployed_services``** — the
category stays a valid ``--category`` name at all times and contributes no rows
to a deployment that runs no store.

* ``telemetry_store_ping`` — ``GET /ping`` against the store's HTTP interface at
  the address the topology rule gives for where the check runs
  (:func:`~osprey.deployment.telemetry_config.store_address`): the service name
  and the image's port from a container on the compose network, loopback and
  the published port from the host and from a container in the host's network
  namespace. ``ok`` on HTTP 200, ``warning`` on any other answer or none.

Where the check runs is declared, never sniffed: a container on the compose
network carries ``OSPREY_OTEL_GATEWAY_HOST=telemetry_gateway`` (the
``telemetry_gateway`` macro in ``templates/services/_store_dial.j2`` writes it);
anything else is in the host's network namespace.

The row is advisory: a store that is down loses no record — the gateway queues
on disk and retries — so it degrades the deployment, it does not break it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from osprey.deployment import telemetry_config as tc
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "telemetry_store"

_PING_TIMEOUT_S = 5.0


def on_compose_network(environ: Mapping[str, str]) -> bool:
    """Whether the process whose environment is *environ* is on the compose network.

    A rendered container on the compose network names the gateway by its
    service name; a host-namespace container names an address, and a host
    launch names nothing.
    """
    return (environ.get(tc.GATEWAY_HOST_ENV) or "").strip() == tc.GATEWAY_SERVICE


def telemetry_store(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> CategoryCallable:
    """Build the ``telemetry_store`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Health runtime. Unused — the store is dialled over HTTP.
        transport: Optional httpx transport for tests; ``None`` uses httpx's default.
        environ: The process environment the network shape is read from;
            defaults to :data:`os.environ`, read when the callable runs.

    Returns:
        A no-argument async callable returning the category's check results.
    """

    async def _run() -> list[CheckResult]:
        if not tc.store_deployed(config):
            return []
        env = os.environ if environ is None else environ
        try:
            url = tc.store_url(config, bridge=on_compose_network(env))
        except ValueError as exc:
            return [CheckResult("telemetry_store_ping", CATEGORY, Status.WARNING, str(exc))]
        return [await _ping(url, transport)]

    return _run


async def _ping(base: str, transport: httpx.AsyncBaseTransport | None) -> CheckResult:
    url = f"{base}/ping"
    try:
        async with httpx.AsyncClient(timeout=_PING_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(
            "telemetry_store_ping",
            CATEGORY,
            Status.WARNING,
            f"Store unreachable at {url}: {exc}",
            details="Start it with `osprey up`; `osprey logs telemetry_store` shows why it stopped.",
        )
    if resp.status_code == 200:
        return CheckResult("telemetry_store_ping", CATEGORY, Status.OK, f"Store answers ({url})")
    return CheckResult(
        "telemetry_store_ping",
        CATEGORY,
        Status.WARNING,
        f"/ping returned HTTP {resp.status_code} ({url})",
    )
