"""Core ``channel_finder`` health category.

Probes the Channel Finder's channel database, but **only when it is
configured**. The Channel Finder panel is served by an ``osprey web`` sidecar —
there is no compose service, so it never appears in ``deployed_services``. Its
presence is therefore keyed on a top-level ``channel_finder`` config block. The
category stays a valid ``--category`` name at all times; when no
``channel_finder`` block is configured it simply contributes no rows (a silent
skip), so a minimal build shows no tile at all.

The category reads the pipeline's channel store directly, so it gives a useful
reading whether or not any web surface is running. The active ``pipeline_mode``
selects what is consulted: the file-backed paradigms name a database file under
``channel_finder.pipelines.<mode>.database``, while ``graph`` answers from the
deployment's graph store and so reads ``services.graphdb`` instead. The mode
must name a real paradigm
(:data:`~osprey.build.build_tiers.VALID_CHANNEL_FINDER_MODES`); a mode nothing
answers to raises :class:`~osprey.services.channel_finder.core.exceptions.PipelineModeError`
rather than degrading to a row, because that is a defect in the configuration
and not a store that happens to be down. The health runner isolates the failure
to this category, so it surfaces as one error row naming the bad mode.

When configured the category emits:

* ``channel_finder_pipeline`` — the active ``pipeline_mode`` as ``value``
  (``ok``); ``warning`` when no mode is set, or when the named
  ``pipelines.<mode>`` block is absent. The ``graph`` paradigm is the exception
  to that second rule: it stores nothing under ``pipelines.graph``, so the row
  is derived from the mode alone and reads ``graph (store-backed)``;
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

The ``graph`` paradigm emits neither the database rows nor the channel count —
there is no file to stat and no DuckDB to open. In their place it restates the
three readings the ``graphdb`` category takes, taken by that category's own
builders so both tiles report one store the same way and a fix named in one
place is named in both:

* ``channel_finder_store`` — the store answered over bolt (``ok``, with the
  round-trip latency); ``warning`` when it is unreachable, rejected the
  credential, or is not described by a ``services.graphdb`` block at all;
* ``channel_finder_resources`` — the number of ``(:Resource)`` nodes the corpus
  imported (``value``); ``warning`` when that is zero or could not be read;
* ``channel_finder_seed`` — the corpus the store was imported from, as the
  digest prefix its seed marker carries (``ok``); ``warning`` when the store is
  unseeded, or holds a corpus no marker identifies.

and adds one reading of its own, because the graph paradigm answers a click
from a file rather than from the store:

* ``channel_finder_search_index`` — the DuckDB search index ``osprey build``
  derives from the same corpus: ``ok`` with its binding and device counts and
  its corpus digest when that digest matches the seed row's; ``warning`` when
  the index is absent, unreadable, or built from a different corpus than the
  store holds. When the store's corpus is unknown — it is down, or unseeded —
  the row still reports the index it found and says the seed is unknown, since
  the index is a build artifact and readable whether or not the store is up.

Every one is advisory: a stopped store is a service that is not running, not a
broken build, so the graph paradigm produces no ``error`` row. The rows carry
this category's names because they answer this category's question — whether
channel search has anything to answer from — next to a ``graphdb`` tile that
reports the same store as a service.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.deployment.graphdb_service import (
    GRAPHDB_BUILD_INDEX_COMMAND,
    GRAPHDB_SEED_COMMAND,
    UNRESOLVED_INDEX_PATH_REMEDY,
    resolve_graph_index_path,
    unresolved_index_path_detail,
)
from osprey.health.core.graphdb import _CONNECTION_ROW as _GRAPHDB_CONNECTION_ROW
from osprey.health.core.graphdb import _DIGEST_PREFIX_LEN as _GRAPHDB_DIGEST_PREFIX_LEN
from osprey.health.core.graphdb import _RESOURCES_ROW as _GRAPHDB_RESOURCES_ROW
from osprey.health.core.graphdb import _SEED_ROW as _GRAPHDB_SEED_ROW
from osprey.health.core.graphdb import _graphdb_block, seed_digest
from osprey.health.core.graphdb import graphdb as _graph_store_category
from osprey.health.models import CheckResult, Status
from osprey.services.channel_finder.core.exceptions import PipelineModeError
from osprey.services.channel_finder.graph_index import META_KEYS, SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "channel_finder"

_CHANNELS_TABLE = "channels"

#: The paradigm whose channels live in the graph store rather than in a file.
GRAPH_MODE = "graph"

_STORE_ROW = "channel_finder_store"
_RESOURCES_ROW = "channel_finder_resources"
_SEED_ROW = "channel_finder_seed"
_SEARCH_INDEX_ROW = "channel_finder_search_index"

#: How the ``graphdb`` category's rows are renamed on their way into this one.
#: The reading, its severity and its remedy are the graphdb builders' own — only
#: the row id changes, so the two tiles never disagree about one store.
_GRAPH_ROW_NAMES = {
    _GRAPHDB_CONNECTION_ROW: _STORE_ROW,
    _GRAPHDB_RESOURCES_ROW: _RESOURCES_ROW,
    _GRAPHDB_SEED_ROW: _SEED_ROW,
}

#: How much of a corpus digest this category's rows show. Taken from the graphdb
#: tile rather than re-chosen, so an operator comparing the seed row against the
#: index row is comparing two prefixes of the same length.
_DIGEST_PREFIX_LEN = _GRAPHDB_DIGEST_PREFIX_LEN


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
            ``duckdb_path``), and — in ``graph`` mode — for ``services.graphdb``,
            which describes the store that paradigm answers from.
        context: Health runtime. Unused — the check reads the pipeline's store
            directly, so no control-system connector is needed.
        cwd: Project root used to resolve relative database paths. Defaults to
            :func:`Path.cwd` (resolved when the callable runs); ``build_records``
            threads the project path here just as it does for ``file_system``.

    Returns:
        A no-argument async callable returning the category's check results.

    Raises:
        PipelineModeError: When the callable runs and ``pipeline_mode`` names
            something that is not a channel-finder paradigm.
    """
    cfg: Mapping[str, Any] = config or {}
    base_dir = cwd

    async def _run() -> list[CheckResult]:
        cf = cfg.get("channel_finder")
        if not isinstance(cf, dict) or not cf:
            return []

        mode = cf.get("pipeline_mode")
        if mode and mode not in VALID_CHANNEL_FINDER_MODES:
            raise PipelineModeError(
                f"Unknown channel_finder.pipeline_mode {mode!r} "
                f"(valid modes: {', '.join(VALID_CHANNEL_FINDER_MODES)})"
            )
        pipelines = cf.get("pipelines", {}) or {}
        mode_block = pipelines.get(mode) if isinstance(mode, str) and mode else None
        if not isinstance(mode_block, dict):
            mode_block = None

        rows = [_pipeline_row(mode, mode_block)]

        # The graph paradigm has no database file to stat: its channels live in
        # the store, so the store's own readings are what this category reports.
        if mode == GRAPH_MODE:
            rows.extend(await _graph_store_rows(cfg, base_dir))
            return rows

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
    """Report the active pipeline mode; ``warning`` when unset or unbacked.

    The ``graph`` paradigm is read from the mode alone. A ``pipelines.graph``
    block would have nothing to hold — the store is described by
    ``services.graphdb`` — so a graph build correctly ships without one, and
    treating its absence as a gap would make every such build warn.
    """
    if not mode:
        return CheckResult(
            "channel_finder_pipeline",
            CATEGORY,
            Status.WARNING,
            "No pipeline_mode configured",
        )
    if mode == GRAPH_MODE:
        return CheckResult(
            "channel_finder_pipeline",
            CATEGORY,
            Status.OK,
            "Active channel-finder pipeline",
            value=f"{GRAPH_MODE} (store-backed)",
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


async def _graph_store_rows(cfg: Mapping[str, Any], base_dir: Path | None) -> list[CheckResult]:
    """Restate the graph store's readings, then check the index built from it.

    The store's readings come from the ``graphdb`` category rather than from a
    second probe written here: one store, one way of dialing it, one set of
    remedies. Only the row ids change, so an operator comparing the two tiles
    sees the same verdict twice rather than two checks that can drift apart.

    The search index is this category's own reading — the ``graphdb`` tile
    reports a service, and a file the build wrote is not one — but it is taken
    here because it is only meaningful against the store's seed marker, which
    the restated rows have just read.

    Args:
        cfg: The parsed config mapping — read for ``services.graphdb``, which is
            where a graph-mode project describes the store it answers from and
            the index derived from the same corpus.
        base_dir: Directory holding ``config.yml``, which is what this category
            is handed as ``cwd`` and what a relative ``index_path`` resolves
            against.

    Returns:
        The store's reachability row, the resource-count and seed rows when the
        store answered, and the search-index row; or a single ``warning`` row
        when no store is configured at all — a project with no store has no
        corpus to derive an index from either. Never an ``error``: a store that
        is down is a service that is not running.
    """
    if _graphdb_block(cfg) is None:
        return [
            CheckResult(
                _STORE_ROW,
                CATEGORY,
                Status.WARNING,
                "The graph pipeline has no graph store to answer from",
                details=(
                    "Add a services.graphdb block to config.yml — the graph paradigm "
                    "reads its channels from that store."
                ),
            )
        ]

    probe = cast("Callable[[], Awaitable[list[CheckResult]]]", _graph_store_category(cfg))
    rows = [
        replace(row, name=_GRAPH_ROW_NAMES.get(row.name, row.name), category=CATEGORY)
        for row in await probe()
    ]

    # The digest is read off the seed row rather than from a second driver: that
    # row is where the store's corpus is read, and asking twice could compare
    # the index against a reading the tile above it never showed.
    seed = next((row for row in rows if row.name == _SEED_ROW), None)
    rows.append(await _search_index_row(cfg, base_dir, seed_digest(seed) if seed else ""))
    return rows


async def _search_index_row(
    cfg: Mapping[str, Any], base_dir: Path | None, seed: str
) -> CheckResult:
    """Resolve the index's path and read it, or say why neither could happen.

    Args:
        cfg: The parsed config mapping, read for ``services.graphdb.index_path``.
        base_dir: Directory holding ``config.yml``.
        seed: The store's corpus digest, or ``""`` when the store could not be
            read or carries no marker.

    Returns:
        The ``channel_finder_search_index`` row. A malformed ``index_path`` is
        reported as that row rather than raised, for the reason the graphdb
        category gives for a malformed block: the same typo already refuses
        loudly at deploy time, and a suite that crashes names no key.
    """
    try:
        index_path = resolve_graph_index_path(cfg, base_dir)
    except ValueError as exc:
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            unresolved_index_path_detail(exc),
            details=UNRESOLVED_INDEX_PATH_REMEDY,
        )
    return await asyncio.to_thread(_index_row, index_path, seed)


def _index_row(index_path: Path, seed: str) -> CheckResult:
    """Read the search index's ``meta`` row and compare it against the store.

    Runs on a worker thread (blocking I/O). ``duckdb`` is imported lazily so a
    missing package degrades to a ``warning`` row rather than crashing the
    suite, exactly as :func:`_count_row` does.

    Args:
        index_path: Where the index should be, as
            :func:`~osprey.deployment.graphdb_service.resolve_graph_index_path`
            resolved it. It need not exist.
        seed: The store's corpus digest, or ``""`` when it is unknown.

    Returns:
        The ``channel_finder_search_index`` row. ``ok`` when the index reads and
        either carries the store's corpus or the store's corpus is unknown;
        ``warning`` when the index is absent, unreadable, or was built from a
        different corpus than the store holds. Never an ``error``: an index the
        build has not written yet is a step not taken.
    """
    if not index_path.exists():
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            f"No search index at {index_path}",
            details=(
                "'osprey build' writes it from the TTL corpus; rebuild it in place "
                f"with `{GRAPHDB_BUILD_INDEX_COMMAND}`."
            ),
        )

    try:
        import duckdb
    except ImportError:
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            "Cannot read the search index: the 'duckdb' package is not installed",
            details="Install the 'duckdb' dependency to enable search-index checks.",
        )

    try:
        con = duckdb.connect(str(index_path), read_only=True)
        try:
            row = con.execute(f"SELECT {', '.join(META_KEYS)} FROM meta").fetchone()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - any duckdb error degrades to a warning
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            f"Could not read the search index: {exc}",
            details=f"Rebuild it with `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
        )

    if row is None:
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            f"Search index carries no meta row ({index_path})",
            details=f"Rebuild it with `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
        )

    meta = dict(zip(META_KEYS, row, strict=True))
    version = int(meta["schema_version"])
    if version != SCHEMA_VERSION:
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            f"Search index is schema v{version}, this osprey reads v{SCHEMA_VERSION}",
            details=f"Rebuild it with `{GRAPHDB_BUILD_INDEX_COMMAND}`.",
        )

    digest = str(meta["corpus_sha256"])
    if seed and digest != seed:
        return CheckResult(
            _SEARCH_INDEX_ROW,
            CATEGORY,
            Status.WARNING,
            "Search index and graph store were built from different corpora",
            value=f"index {digest[:_DIGEST_PREFIX_LEN]} · store {seed[:_DIGEST_PREFIX_LEN]}",
            details=(
                f"Rebuild the index with `{GRAPHDB_BUILD_INDEX_COMMAND}`, or reseed the "
                f"store with `{GRAPHDB_SEED_COMMAND}`."
            ),
        )

    value = f"{_index_counts(meta)} · {digest[:_DIGEST_PREFIX_LEN]}"
    if not seed:
        value = f"{value} (store's seed unknown)"
    return CheckResult(
        _SEARCH_INDEX_ROW,
        CATEGORY,
        Status.OK,
        "Search index built from the TTL corpus",
        value=value,
        details=f"Built from {meta['corpus_filename']}: {_index_detail_counts(meta)}.",
    )


def _index_counts(meta: dict[str, Any]) -> str:
    """The two counts the row's ``value`` carries, as one compact phrase."""
    return f"{int(meta['binding_count']):,} bindings · {int(meta['device_count']):,} devices"


def _index_detail_counts(meta: dict[str, Any]) -> str:
    """Every count the ``meta`` row holds, for the ok row's ``details``."""
    return ", ".join(
        f"{int(meta[key]):,} {label}"
        for key, label in (
            ("binding_count", "bindings"),
            ("device_count", "devices"),
            ("class_count", "classes"),
            ("signal_count", "signals"),
            ("section_count", "sections"),
        )
    )


def _database_row(db_path: Path | None) -> CheckResult:
    """Check that the pipeline's database file exists and is non-empty.

    ``error`` when a path is configured but the file is missing or empty (the
    data bundle should have materialized it); ``warning`` when no path is
    configured at all; ``ok`` otherwise, with the file size as ``value``.

    Every message names *the active pipeline's* ``database.path`` file, because
    that one file is all this row stats. A deployment can hold more than one
    channel database — the middle-layer DuckDB that ``channel_finder_channels``
    reports on separately is a different file, under a different key — so a row
    saying "the channel database" would be vouching for artifacts it never
    looked at.
    """
    if db_path is None:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.WARNING,
            "No database.path configured for the active pipeline",
            details="Set the pipeline's database.path under config: in profile.yml and run 'osprey build'.",
        )

    try:
        size = db_path.stat().st_size
    except OSError:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.ERROR,
            f"Active pipeline's channel database file missing at {db_path}",
            details="The data bundle materializes this file — rebuild it, or fix database.path.",
        )

    if size == 0:
        return CheckResult(
            "channel_finder_database",
            CATEGORY,
            Status.ERROR,
            f"Active pipeline's channel database file is empty ({db_path})",
            details="The file exists but has zero bytes — rebuild it, or fix database.path.",
        )

    return CheckResult(
        "channel_finder_database",
        CATEGORY,
        Status.OK,
        "Active pipeline's channel database file present",
        value=_humanize_size(size),
    )


def _freshness_row(db_path: Path) -> CheckResult:
    """Report the ``database.path`` file's modification age (informational).

    Named as narrowly as :func:`_database_row`, and for the same reason: this
    is the age of one file, not of every channel database the deployment holds.
    """
    try:
        mtime = db_path.stat().st_mtime
    except OSError:
        return CheckResult(
            "channel_finder_freshness",
            CATEGORY,
            Status.OK,
            "Build age of the active pipeline's channel database file is unavailable",
        )
    return CheckResult(
        "channel_finder_freshness",
        CATEGORY,
        Status.OK,
        "Active pipeline's channel database file build age",
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
