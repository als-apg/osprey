"""Core ``ariel`` health category.

Probes the ARIEL logbook-search web interface, but **only when it is
configured**. The ARIEL panel is served by an ``osprey web`` sidecar — there is
no compose service, so it never appears in ``deployed_services``. Its presence is
therefore keyed on a top-level ``ariel`` config block. The category stays a valid
``--category`` name at all times; when no ``ariel`` block is configured it simply
contributes no rows (a silent skip), so a minimal build shows no ARIEL tile at
all.

When configured the category issues a single ``GET /api/status`` request against
``deployment.bind_address`` + ``ariel.web.port`` and derives every row from that
one response:

* ``ariel_status`` — the interface is reachable (``ok`` with ``latency_ms``);
  ``warning`` when the store reports itself unhealthy, and — as the sole row —
  ``warning`` when the interface is unreachable (the request is not repeated);
* ``ariel_entries`` — the logbook entry count as ``value`` (e.g. ``"48,291
  entries"``); ``warning`` when it is zero or absent;
* ``ariel_last_ingestion`` — the age of the last successful ingestion as a
  human-readable ``value`` (e.g. ``"2 h ago"``); ``warning`` only when the field
  is absent (never ran). Cadences differ per facility, so a stale-but-present
  timestamp is never flagged;
* ``ariel_search_modules`` — the enabled search modules (count in the message,
  names in ``value``); ``warning`` when none are enabled, since search is core;
* ``ariel_enhancement_modules`` — the enabled enhancement/enrichment modules
  (count in the message, names in ``value``); always ``ok`` — enrichment is
  optional, so an empty set is not a fault.

Every row is advisory (``ok``/``warning``); the interface's absence is never an
error, and a single unreachable probe never crashes the suite.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "ariel"

_STATUS_TIMEOUT_S = 5.0
_DEFAULT_PORT = 8085  # mirrors registry.web ariel port_default
_DEFAULT_BIND = "127.0.0.1"


def ariel(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CategoryCallable:
    """Build the ``ariel`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for the top-level ``ariel`` block (presence gate + ``ariel.web.port``)
            and ``deployment.bind_address``.
        context: Health runtime. Unused — ARIEL is probed over HTTP, no
            control-system connector is needed.
        transport: Optional httpx transport for dependency injection in tests
            (e.g. :class:`httpx.MockTransport`); ``None`` uses httpx's default.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        ariel_block = cfg.get("ariel")
        if not isinstance(ariel_block, dict) or not ariel_block:
            return []

        ariel_cfg = ariel_block.get("web") or {}
        port = ariel_cfg.get("port", _DEFAULT_PORT)
        bind = (cfg.get("deployment", {}) or {}).get("bind_address", _DEFAULT_BIND)
        url = f"http://{bind}:{port}/api/status"

        status_row, payload = await _fetch_status(url, transport)
        if payload is None:
            # Unreachable or unusable response: the status row carries the
            # diagnostic and no further rows can be derived.
            return [status_row]

        return [
            status_row,
            _entries_row(payload),
            _last_ingestion_row(payload),
            _modules_row(
                "ariel_search_modules",
                payload.get("enabled_search_modules"),
                noun="search",
                warn_when_empty=True,
            ),
            _modules_row(
                "ariel_enhancement_modules",
                payload.get("enabled_enhancement_modules"),
                noun="enhancement",
                warn_when_empty=False,
            ),
        ]

    return _run


async def _fetch_status(
    url: str, transport: httpx.AsyncBaseTransport | None
) -> tuple[CheckResult, dict[str, Any] | None]:
    """Issue the single ``GET /api/status`` call and build the ``ariel_status`` row.

    Returns ``(row, payload)``. ``payload`` is the parsed status body on a usable
    HTTP 200, or ``None`` when the interface is unreachable, answered non-200, or
    returned a body that is not a JSON object — in which case the row is a
    ``warning`` and the caller emits it alone.
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_STATUS_TIMEOUT_S, transport=transport) as client:
            resp = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        return (
            CheckResult(
                "ariel_status",
                CATEGORY,
                Status.WARNING,
                f"ARIEL unreachable at {url}: {exc}",
                details=(
                    "The ARIEL panel sidecar runs with `osprey web` — check that it is "
                    "up, or verify the bind address / port."
                ),
            ),
            None,
        )

    latency_ms = (time.perf_counter() - start) * 1000.0

    if resp.status_code != 200:
        # FastAPI error bodies carry a human-readable `detail` — surface it.
        try:
            detail = resp.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""
        return (
            CheckResult(
                "ariel_status",
                CATEGORY,
                Status.WARNING,
                f"/api/status returned HTTP {resp.status_code}",
                latency_ms=latency_ms,
                details=str(detail) if detail else "",
            ),
            None,
        )

    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return (
            CheckResult(
                "ariel_status",
                CATEGORY,
                Status.WARNING,
                f"/api/status returned a non-JSON body ({url})",
                latency_ms=latency_ms,
            ),
            None,
        )

    if payload.get("healthy", True):
        row = CheckResult(
            "ariel_status", CATEGORY, Status.OK, f"ARIEL reachable ({url})", latency_ms=latency_ms
        )
    else:
        errors = payload.get("errors") or []
        detail = "; ".join(str(e) for e in errors) if errors else ""
        row = CheckResult(
            "ariel_status",
            CATEGORY,
            Status.WARNING,
            "ARIEL is reachable but reports unhealthy",
            latency_ms=latency_ms,
            details=detail,
        )
    return row, payload


def _entries_row(payload: dict[str, Any]) -> CheckResult:
    """Report the logbook entry count; ``warning`` when zero or absent."""
    count = payload.get("entry_count")
    if not isinstance(count, int) or count <= 0:
        return CheckResult(
            "ariel_entries",
            CATEGORY,
            Status.WARNING,
            "Logbook is empty or entry count is unavailable",
        )
    return CheckResult(
        "ariel_entries",
        CATEGORY,
        Status.OK,
        "Logbook entries indexed",
        value=f"{count:,} entries",
    )


def _last_ingestion_row(payload: dict[str, Any]) -> CheckResult:
    """Report the age of the last ingestion; ``warning`` only when never run.

    Facilities ingest on very different cadences, so a stale-but-present
    timestamp is reported ``ok`` — only a missing/unparseable value warns.
    """
    raw = payload.get("last_ingestion")
    if not raw:
        return CheckResult(
            "ariel_last_ingestion",
            CATEGORY,
            Status.WARNING,
            "No successful ingestion has been recorded yet",
        )

    parsed = _parse_timestamp(raw)
    if parsed is None:
        return CheckResult(
            "ariel_last_ingestion",
            CATEGORY,
            Status.OK,
            "Last ingestion recorded",
            value=str(raw),
        )
    return CheckResult(
        "ariel_last_ingestion",
        CATEGORY,
        Status.OK,
        "Last successful ingestion",
        value=_humanize_age(parsed),
    )


def _modules_row(name: str, modules: Any, *, noun: str, warn_when_empty: bool) -> CheckResult:
    """Report the enabled ``noun`` modules; count in message, names in ``value``.

    ``warn_when_empty`` gates whether an empty set is a ``warning`` (search, which
    is core) or benign ``ok`` (enhancement, which is optional).
    """
    names = [str(m) for m in modules] if isinstance(modules, list) else []
    if not names:
        status = Status.WARNING if warn_when_empty else Status.OK
        return CheckResult(name, CATEGORY, status, f"No {noun} modules enabled")
    return CheckResult(
        name,
        CATEGORY,
        Status.OK,
        f"{len(names)} {noun} module(s) enabled",
        value=", ".join(names),
    )


def _parse_timestamp(raw: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a ``datetime`` (``None`` on fail)."""
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _humanize_age(then: datetime) -> str:
    """Render the elapsed time since ``then`` as a compact human-readable age.

    Matches ``then``'s awareness (aware vs naive) when sampling "now" so a
    timezone-qualified timestamp and a naive one both subtract cleanly.
    """
    now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
    seconds = (now - then).total_seconds()
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} m ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} h ago"
    days = hours / 24
    return f"{int(days)} d ago"
