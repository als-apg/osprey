"""Core ``channel_finder`` health category.

Probes the Channel Finder's channel database, but **only when it is
configured**. The Channel Finder panel is served by an ``osprey web`` sidecar —
there is no compose service, so it never appears in ``deployed_services``. Its
presence is therefore keyed on a top-level ``channel_finder`` config block. The
category stays a valid ``--category`` name at all times; when no
``channel_finder`` block is configured it simply contributes no rows (a silent
skip), so a minimal build shows no tile at all.

The category reads the pipeline's on-disk channel database directly, so it gives
a useful reading whether or not any web surface is running. The active pipeline
mode (``in_context``/``hierarchical``/``middle_layer``) selects which database
file is consulted under ``channel_finder.pipelines.<mode>.database``.

When configured the category emits:

* ``channel_finder_pipeline`` — the active ``pipeline_mode`` as ``value``
  (``ok``); ``warning`` when no mode is set, or when the named
  ``pipelines.<mode>`` block is absent;
* ``channel_finder_database`` — the pipeline's ``database.path`` file exists and
  is non-empty (``ok``, file size as ``value``); ``error`` when a path is
  configured but the file is missing or empty (a broken build — the data bundle
  materializes these files); ``warning`` when no ``database.path`` is configured;
* ``channel_finder_freshness`` — that database file's modification age as a
  human-readable ``value`` (e.g. ``"built 3 d ago"``); always ``ok`` and emitted
  only when the file exists — build cadence is informational, not a fault;
* ``channel_finder_channels`` — emitted **only** for the ``middle_layer``
  pipeline with a configured ``duckdb_path``: the channel row count read from the
  materialized DuckDB (``value``); ``warning`` when zero, when the database
  cannot be opened, or when the ``duckdb`` package is unavailable (the check
  degrades, it never crashes the suite). In every other mode this row is simply
  absent.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "channel_finder"

_CHANNELS_TABLE = "channels"


def channel_finder(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    cwd: Path | None = None,
) -> CategoryCallable:
    """Build the ``channel_finder`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for the top-level ``channel_finder`` block (presence gate,
            ``pipeline_mode``, and ``pipelines.<mode>.database.path`` /
            ``duckdb_path``).
        context: Health runtime. Unused — the check reads database files on disk,
            so no control-system connector is needed.
        cwd: Project root used to resolve relative database paths. Defaults to
            :func:`Path.cwd` (resolved when the callable runs); ``build_records``
            threads the project path here just as it does for ``file_system``.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}
    base_dir = cwd

    async def _run() -> list[CheckResult]:
        cf = cfg.get("channel_finder")
        if not isinstance(cf, dict) or not cf:
            return []

        mode = cf.get("pipeline_mode")
        pipelines = cf.get("pipelines", {}) or {}
        mode_block = pipelines.get(mode) if isinstance(mode, str) and mode else None
        if not isinstance(mode_block, dict):
            mode_block = None

        rows = [_pipeline_row(mode, mode_block)]

        db_conf = (mode_block or {}).get("database", {}) or {}
        db_path = _resolve(db_conf.get("path"), base_dir)

        rows.append(_database_row(db_path))
        if db_path is not None and db_path.exists():
            rows.append(_freshness_row(db_path))

        # The channel count is a middle-layer-only reading: only that pipeline
        # materializes the DuckDB the count is read from.
        if mode == "middle_layer":
            duckdb_path = _resolve(db_conf.get("duckdb_path"), base_dir)
            if duckdb_path is not None:
                rows.append(await asyncio.to_thread(_count_row, duckdb_path))

        return rows

    return _run


def _resolve(raw_path: Any, base_dir: Path | None) -> Path | None:
    """Resolve a configured path relative to the project root (``None`` if unset)."""
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path


def _pipeline_row(mode: Any, mode_block: dict[str, Any] | None) -> CheckResult:
    """Report the active pipeline mode; ``warning`` when unset or unbacked."""
    if not mode:
        return CheckResult(
            "channel_finder_pipeline",
            CATEGORY,
            Status.WARNING,
            "No pipeline_mode configured",
        )
    if mode_block is None:
        return CheckResult(
            "channel_finder_pipeline",
            CATEGORY,
            Status.WARNING,
            f"pipeline_mode '{mode}' has no pipelines.{mode} configuration",
            value=str(mode),
        )
    return CheckResult(
        "channel_finder_pipeline",
        CATEGORY,
        Status.OK,
        "Active channel-finder pipeline",
        value=str(mode),
    )


def _database_row(db_path: Path | None) -> CheckResult:
    """Check that the pipeline's database file exists and is non-empty.

    ``error`` when a path is configured but the file is missing or empty (the
    data bundle should have materialized it); ``warning`` when no path is
    configured at all; ``ok`` otherwise, with the file size as ``value``.
    """
    if db_path is None:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.WARNING,
            "No channel database path configured",
            details="Set the pipeline's database.path in config.yml.",
        )

    try:
        size = db_path.stat().st_size
    except OSError:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.ERROR,
            f"Channel database missing at {db_path}",
            details="The data bundle materializes this file — rebuild it, or fix database.path.",
        )

    if size == 0:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.ERROR,
            f"Channel database is empty ({db_path})",
            details="The file exists but has zero bytes — rebuild the channel database.",
        )

    return CheckResult(
        "channel_finder_database",
        CATEGORY,
        Status.OK,
        "Channel database present",
        value=_humanize_size(size),
    )


def _freshness_row(db_path: Path) -> CheckResult:
    """Report the database file's modification age (always informational)."""
    try:
        mtime = db_path.stat().st_mtime
    except OSError:
        return CheckResult(
            "channel_finder_freshness",
            CATEGORY,
            Status.OK,
            "Database build age is unavailable",
        )
    return CheckResult(
        "channel_finder_freshness",
        CATEGORY,
        Status.OK,
        "Channel database build age",
        value=f"built {_humanize_age(datetime.fromtimestamp(mtime))}",
    )


def _count_row(duckdb_path: Path) -> CheckResult:
    """Open the DuckDB read-only and count channels; ``warning`` on 0 or failure.

    Runs on a worker thread (blocking I/O). ``duckdb`` is imported lazily so a
    missing package degrades to a ``warning`` row rather than crashing the suite.
    """
    try:
        import duckdb
    except ImportError:
        return CheckResult(
            "channel_finder_channels",
            CATEGORY,
            Status.WARNING,
            "Cannot count channels: the 'duckdb' package is not installed",
            details="Install the 'duckdb' dependency to enable channel counting.",
        )

    try:
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            row = con.execute(f"SELECT COUNT(*) FROM {_CHANNELS_TABLE}").fetchone()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - any duckdb error degrades to a warning
        return CheckResult(
            "channel_finder_channels",
            CATEGORY,
            Status.WARNING,
            f"Could not read the channel database: {exc}",
        )

    count = int(row[0]) if row and row[0] is not None else 0
    if count <= 0:
        return CheckResult(
            "channel_finder_channels",
            CATEGORY,
            Status.WARNING,
            "Channel database contains no channels",
        )
    return CheckResult(
        "channel_finder_channels",
        CATEGORY,
        Status.OK,
        "Channels indexed",
        value=f"{count:,} channels",
    )


def _humanize_age(then: datetime) -> str:
    """Render the elapsed time since ``then`` as a compact human-readable age."""
    seconds = (datetime.now() - then).total_seconds()
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


def _humanize_size(size: int) -> str:
    """Render a byte count as a compact human-readable size."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
