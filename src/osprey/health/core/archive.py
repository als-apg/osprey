"""Core ``archive`` health category.

Reads the agent-record archive's own manifests, but **only when the archive
service is deployed**. When ``archive`` is not in the project's
``deployed_services`` the category contributes no rows. When deployed:

* ``archive_last_pass`` — ``warning`` when ``var/archive`` is missing or
  unreadable or holds no pass, when the newest pass is older than twice
  ``services.archive.interval_seconds``, or when it recorded errors (the message
  names the first source); else ``ok`` with the pass's age;
* ``archive_telemetry_day`` — only when ``openobserve`` is deployed too:
  ``warning`` when no telemetry day was exported or the newest is older than the
  day before yesterday (UTC); else ``ok`` naming the day.

Both rows are advisory (``ok``/``warning``). The container carries no probe: a
running archive that no longer completes a pass is exactly what these rows see
and a container probe would not.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.deployment.compose_generator import ARCHIVE_DEFAULT_INTERVAL_SECONDS
from osprey.health.models import CheckResult, Status
from osprey.services.archive.manifest import load_state
from osprey.utils.workspace import ARCHIVE_DIR_RELPATH

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "archive"


def archive(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    cwd: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> CategoryCallable:
    """Build the ``archive`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``deployed_services`` and ``services.archive.interval_seconds``.
        context: Health runtime. Unused.
        cwd: The deployment repo root the archive directory hangs off. Defaults
            to :func:`Path.cwd`, resolved when the callable runs.
        now: Clock, for tests; defaults to the current UTC time.

    Returns:
        A no-argument callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}
    clock = now or (lambda: datetime.now(UTC))

    def _run() -> list[CheckResult]:
        deployed = {str(name) for name in cfg.get("deployed_services") or []}
        if "archive" not in deployed:
            return []
        root = (cwd or Path.cwd()) / ARCHIVE_DIR_RELPATH
        block = (cfg.get("services") or {}).get("archive") or {}
        interval = block.get("interval_seconds") if isinstance(block, dict) else None
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            interval = ARCHIVE_DEFAULT_INTERVAL_SECONDS
        current = clock()

        if not root.is_dir():
            return _rows(
                _warn_last_pass(f"No archive at {ARCHIVE_DIR_RELPATH}"),
                deployed,
                None,
                current,
            )
        try:
            state = load_state(root)
        except OSError as exc:
            return _rows(_warn_last_pass(f"Archive unreadable: {exc}"), deployed, None, current)
        return _rows(
            _last_pass_row(state.last_pass, interval, current),
            deployed,
            state.telemetry_days,
            current,
        )

    return _run


def _rows(
    last_pass: CheckResult,
    deployed: set[str],
    telemetry_days: set[str] | None,
    current: datetime,
) -> list[CheckResult]:
    rows = [last_pass]
    if "openobserve" in deployed:
        rows.append(_telemetry_row(telemetry_days or set(), current))
    return rows


def _warn_last_pass(message: str, details: str = "") -> CheckResult:
    return CheckResult(
        "archive_last_pass",
        CATEGORY,
        Status.WARNING,
        message,
        details=details or "Check the archive container with `osprey logs archive`.",
    )


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _age(delta: timedelta) -> str:
    seconds = max(int(delta.total_seconds()), 0)
    if seconds >= 3600:
        return f"{seconds // 3600} h"
    return f"{seconds // 60} min"


def _last_pass_row(last: dict[str, Any] | None, interval: int, current: datetime) -> CheckResult:
    if not last:
        return _warn_last_pass("The archive has completed no pass")
    finished = _parse_time(last.get("completed_at")) or _parse_time(last.get("started_at"))
    if finished is None:
        return _warn_last_pass("The archive's last pass has no time")
    age = current - finished
    if age > timedelta(seconds=2 * interval):
        return _warn_last_pass(f"Last archive pass {_age(age)} ago")
    errors = last.get("errors") or []
    if errors:
        first = errors[0] if isinstance(errors[0], dict) else {}
        return _warn_last_pass(
            f"Last archive pass recorded {len(errors)} error(s), first at "
            f"{first.get('source', '?')}",
            details=str(first.get("error", "")),
        )
    return CheckResult(
        "archive_last_pass",
        CATEGORY,
        Status.OK,
        f"Last archive pass {_age(age)} ago",
        value=_age(age),
    )


def _telemetry_row(days: set[str], current: datetime) -> CheckResult:
    parsed: list[date] = []
    for day in days:
        try:
            parsed.append(date.fromisoformat(day))
        except ValueError:
            continue
    if not parsed:
        return CheckResult(
            "archive_telemetry_day",
            CATEGORY,
            Status.WARNING,
            "No telemetry day archived",
            details="Check the archive container with `osprey logs archive`.",
        )
    newest = max(parsed)
    oldest_fresh = current.astimezone(UTC).date() - timedelta(days=2)
    if newest < oldest_fresh:
        return CheckResult(
            "archive_telemetry_day",
            CATEGORY,
            Status.WARNING,
            f"Newest archived telemetry day is {newest.isoformat()}",
            details="Check the archive container with `osprey logs archive`.",
        )
    return CheckResult(
        "archive_telemetry_day",
        CATEGORY,
        Status.OK,
        f"Telemetry archived through {newest.isoformat()}",
        value=newest.isoformat(),
    )
