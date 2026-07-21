"""HTTP health probe — async ``GET`` with status and latency assertions.

Issues a single ``GET`` against a configured URL and grades the response:

* the status code must equal ``expect_status`` (default ``200``), otherwise the
  check is an ``error``;
* on a matching status, elapsed latency is banded against optional
  ``warn_latency_ms`` / ``error_latency_ms`` ceilings;
* any transport failure (connection refused, DNS, protocol) is an ``error`` whose
  ``details`` carries ``str(exc)``;
* a request timeout (``timeout_s`` → the httpx client timeout) maps to the
  configured ``timeout_status`` (``error`` by default, ``warning`` opt-in).

``latency_ms`` is measured from a single ``perf_counter()`` taken before the
request and read again in *every* branch — success, mismatch, timeout, and
transport failure alike — so the wire result always reports how long the attempt
took.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

import httpx

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext


async def run(
    spec: Mapping[str, Any],
    ctx: ProbeContext,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CheckResult:
    """Probe an HTTP endpoint with a single ``GET``.

    Args:
        spec: Parsed check parameters. Recognized keys:

            * ``url`` (str, required): the endpoint to GET.
            * ``name`` (str): result name (default ``"http"``).
            * ``category`` (str): result category (default ``"http"``).
            * ``expect_status`` (int): required status code (default ``200``).
            * ``warn_latency_ms`` (float | None): warn if latency exceeds this.
            * ``error_latency_ms`` (float | None): error if latency exceeds this.
            * ``timeout_s`` (float): request timeout in seconds (default ``5.0``).
            * ``timeout_status`` (``"error"`` | ``"warning"``): status for a
              timed-out request (default ``"error"``).
        ctx: Shared per-run context. Unused by this probe (HTTP needs no
            control-system connector) but part of the uniform probe interface.
        transport: Optional httpx transport for dependency injection in tests
            (e.g. :class:`httpx.MockTransport`); ``None`` uses httpx's default.

    Returns:
        A :class:`~osprey.health.models.CheckResult` with ``latency_ms`` set on
        every outcome and ``value`` carrying the observed status code when a
        response was received.
    """
    name = str(spec.get("name", "http"))
    category = str(spec.get("category", "http"))
    url = str(spec.get("url", ""))
    expect_status = int(spec.get("expect_status", 200))
    timeout_s = float(spec.get("timeout_s", 5.0))
    timeout_status = Status(spec.get("timeout_status", "error"))
    warn_ms = spec.get("warn_latency_ms")
    error_ms = spec.get("error_latency_ms")

    t0 = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            resp = await client.get(url)
    except httpx.TimeoutException as exc:
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            timeout_status,
            f"GET {url} timed out after {timeout_s:g}s",
            latency_ms=latency_ms,
            details=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure is an error result
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"GET {url} failed",
            latency_ms=latency_ms,
            details=str(exc),
        )

    latency_ms = (perf_counter() - t0) * 1000.0
    status_code = resp.status_code

    if status_code != expect_status:
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"GET {url} returned {status_code}, expected {expect_status}",
            value=str(status_code),
            latency_ms=latency_ms,
        )

    if error_ms is not None and latency_ms > float(error_ms):
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"GET {url} → {status_code} in {latency_ms:.0f}ms "
            f"(over {float(error_ms):.0f}ms error ceiling)",
            value=str(status_code),
            latency_ms=latency_ms,
        )

    if warn_ms is not None and latency_ms > float(warn_ms):
        return CheckResult(
            name,
            category,
            Status.WARNING,
            f"GET {url} → {status_code} in {latency_ms:.0f}ms "
            f"(over {float(warn_ms):.0f}ms warn ceiling)",
            value=str(status_code),
            latency_ms=latency_ms,
        )

    return CheckResult(
        name,
        category,
        Status.OK,
        f"GET {url} → {status_code} in {latency_ms:.0f}ms",
        value=str(status_code),
        latency_ms=latency_ms,
    )
