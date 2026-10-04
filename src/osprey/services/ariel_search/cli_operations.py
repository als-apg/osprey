"""Business logic for ARIEL CLI commands.

Extracted from ``osprey.cli.ariel`` so that the CLI handlers are thin
wrappers around these service-layer functions.  Each function accepts a
raw config dict (from ``get_config_value("ariel", {})``) and returns a
structured result; the CLI layer handles Click decorators, output
formatting, and ``SystemExit`` translation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime

    from osprey.models.providers.base import BaseProvider
    from osprey.services.ariel_search import ARIELConfig
    from osprey.services.ariel_search.attachments.copy import CopyRun
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter
    from osprey.services.ariel_search.ingestion.ingest import EntryIngestOutcome
    from osprey.services.ariel_search.ingestion.scheduler import StopReason
    from osprey.services.ariel_search.models import EnhancedLogbookEntry


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    count: int
    enhanced_count: int
    failed_count: int
    unreadable_count: int
    dry_run: bool
    enhancer_names: list[str] = field(default_factory=list)


@dataclass
class WatchOnceResult:
    entries_added: int
    entries_updated: int
    entries_failed: int
    duration_seconds: float
    since: datetime | None
    dry_run: bool


@dataclass
class EnhanceResult:
    """Outcome of one enhancement pass.

    ``succeeded``, ``failed`` and ``set_aside`` count enhancements, meaning
    (entry, module) pairs, in this pass; ``set_aside`` counts the failures that
    reached the attempt cap.
    """

    entries_processed: int
    module_names: list[str]
    succeeded: int = 0
    failed: int = 0
    set_aside: int = 0


@dataclass
class ReembedResult:
    processed: int
    skipped: int
    errors: int
    dry_run: bool


@dataclass
class QuickstartResult:
    count: int
    enhanced_count: int
    failed_count: int
    migrations_applied: int
    enabled_search: list[str]


@dataclass
class PurgeInfo:
    """What a purge would delete, for its confirmation prompt.

    Attributes:
        entry_count: Logbook entries in the store.
        embedding_tables: The ``text_embeddings_*`` tables.
        image_embedding_tables: The ``image_embeddings_*`` tables.
    """

    entry_count: int
    embedding_tables: list[str]
    image_embedding_tables: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    migrations_applied: int
    entries_ingested: int
    entries_enhanced: int
    entries_failed: int
    was_initial_ingest: bool
    busy_skipped: list[str] = field(default_factory=list)


@dataclass
class QmdResyncResult:
    """Outcome of one qmd markdown-mirror resync pass.

    Attributes:
        scanned: Rows the changed-entry scan examined.
        written: Files created or replaced because their content differed.
        unchanged: Rows whose file already held exactly the rendered bytes, so
            nothing was written and no mtime moved.
        failed: Rows that could not be mirrored (unmirrorable identifier or a
            filesystem error); each is logged and the pass continues.
        removed: Files deleted by the ``rebuild`` wipe; always zero otherwise.
        rebuild: Whether this pass wiped the mirror and re-exported everything.
        mirror_path: Absolute mirror root the pass wrote into.
        watermark: Highest ``updated_at`` observed, which becomes the starting
            point of the next pass. ``None`` when the scan matched no rows.
    """

    scanned: int
    written: int
    unchanged: int
    failed: int
    removed: int
    rebuild: bool
    mirror_path: str
    watermark: datetime | None


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

_ProgressCb = Callable[[str], None] | None


def _postgresql_services() -> dict:
    """Return the ``services.postgresql`` mapping from the loaded config.

    The CLI hands these functions the ``ariel`` section alone, but a config
    that leaves ``ariel.database.uri`` unset derives its DSN from the Postgres
    the project actually runs — so the block is read here rather than threaded
    through every command signature.

    A caller that supplies its own ``ariel`` section without a project on disk
    (tests, an embedding host) gets an empty block and the shipped Postgres
    defaults, rather than a crash about a missing config.yml.
    """
    from osprey.utils.config import get_config_value

    try:
        return get_config_value("services.postgresql", {}) or {}
    except FileNotFoundError:
        return {}


def _port_base() -> int:
    """Return the port base of the deployment this CLI is running against.

    The Postgres block read by :func:`_postgresql_services` names a port only
    when the project pinned one; otherwise the DSN is derived from the layout,
    and the layout needs this deployment's base or it lands in the default
    block — somebody else's Postgres.

    The ``deployment`` subtree is re-wrapped as ``{"deployment": ...}`` because
    :func:`~osprey.port_layout.resolve_port_base` takes one input shape, and
    the re-wrap is what keeps a base that arrives here range-checked by the
    same code that checks one read from a rendered config.

    A caller with no project on disk gets the layout default, matching
    :func:`_postgresql_services`: an operation run outside a project resolves
    the shipped defaults rather than crashing about a missing config.yml.
    """
    from osprey.port_layout import resolve_port_base
    from osprey.utils.config import get_config_value

    try:
        deployment = get_config_value("deployment", {})
    except FileNotFoundError:
        deployment = {}
    return resolve_port_base({"deployment": deployment})


def _ariel_config(config_dict: dict) -> ARIELConfig:
    """Build an :class:`ARIELConfig` from a CLI-supplied ``ariel`` section.

    Every operation below needs the Postgres block alongside the section it was
    handed, so the pairing lives here rather than at each call site.
    """
    from osprey.services.ariel_search import ARIELConfig

    return ARIELConfig.from_dict(config_dict, _postgresql_services(), base=_port_base())


def _require_ingestion_block(config_dict: dict) -> None:
    """Refuse a logbook-reading command whose config declares no ingestion.

    ``ariel watch`` and ``ariel sync`` both read a logbook, so an ``ariel``
    section with no ``ingestion`` block at all has not configured the thing they
    do. Left to the parser, that shape produces a message about the block's
    ``adapter`` field -- naming a key inside a block the operator never wrote,
    which reads as a typo in something they have rather than as something
    missing. Checked here, before the parse, so the refusal names the block.

    A block that IS present keeps the parser's message: ``adapter`` really is
    the missing answer then.

    Args:
        config_dict: The raw ``ariel`` section, after any CLI override that
            mints the block has been applied.

    Raises:
        ConfigurationError: If the section carries no ``ingestion`` block.
    """
    if config_dict.get("ingestion"):
        return

    from osprey.services.ariel_search.exceptions import ConfigurationError

    raise ConfigurationError(
        "ariel.ingestion is not configured: `ariel watch` and `ariel sync` read "
        "a logbook, so name the adapter and source_url under ariel.ingestion",
        config_key="ingestion",
    )


def check_vocabulary(
    config_dict: dict,
    path: str | None = None,
    *,
    config_dir: Path | None = None,
) -> dict:
    """Validate a facility vocabulary file without touching the database.

    The whole point of the check is that it runs before anything is deployed:
    no Postgres, no embedding provider, and no ``ariel`` section required when
    the file is named on the command line.

    Args:
        config_dict: The raw ``ariel`` section. Only its ``vocabulary`` block is
            read — for the default path and for the two direction gates, which
            decide whether a form ever reaches ``plainto_tsquery`` and so
            whether a stopword-valued form is worth warning about.
        path: An explicit file to check. Wins over the configured path; a
            relative value is resolved against the process working directory,
            because it came from a shell rather than from a config file.
        config_dir: Directory holding the config.yml *config_dict* came from, so
            a relative ``ariel.vocabulary.path`` resolves to the same file the
            web panel and the MCP server will read.

    Returns:
        ``{"status": "ok" | "invalid", "path": str, "concepts": int,
        "errors": [...], "warnings": [...]}``. With no file to check at all the
        status is ``"error"`` and ``message`` says so. Content problems are
        reported, never raised.
    """
    from osprey.services.ariel_search.config import VocabularyConfig
    from osprey.services.ariel_search.vocabulary import load_vocabulary
    from osprey.utils.config_paths import resolve_config_relative_path

    block = config_dict.get("vocabulary")
    if not isinstance(block, dict):
        block = {}

    try:
        vocabulary_config = VocabularyConfig.from_dict(block)
    except ValueError as exc:
        # A malformed knob is refused rather than defaulted, exactly as the
        # config parser refuses it; the check is where an operator should find
        # that out.
        return {
            "status": "invalid",
            "path": str(path) if path else None,
            "concepts": 0,
            "errors": [str(exc)],
            "warnings": [],
        }

    if path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
    elif vocabulary_config.path:
        resolved = resolve_config_relative_path(vocabulary_config.path, config_dir)
    else:
        return {
            "status": "error",
            "message": "no vocabulary path: pass PATH or set ariel.vocabulary.path",
            "path": None,
            "concepts": 0,
            "errors": [],
            "warnings": [],
        }

    vocabulary, errors, warnings = load_vocabulary(
        resolved,
        canonical_to_acronym=vocabulary_config.canonical_to_acronym,
        canonical_to_shorthand=vocabulary_config.canonical_to_shorthand,
    )
    return {
        "status": "invalid" if errors else "ok",
        "path": str(resolved),
        "concepts": vocabulary.concept_count if vocabulary is not None else 0,
        "errors": errors,
        "warnings": warnings,
    }


def vocabulary_status(config_dict: dict, config_dir: Path | None = None) -> dict:
    """Summarize the configured vocabulary for ``osprey ariel status``.

    Deliberately database-free: the operator most in need of this line is the
    one whose panel will not start, and a status command that needed Postgres
    before it could say the vocabulary is broken would never reach them.

    Args:
        config_dict: The raw ``ariel`` section.
        config_dir: Directory holding the config.yml it came from.

    Returns:
        ``{"status": "ok" | "invalid" | "disabled", "concepts": int,
        "errors": [...]}``. A config that will not even parse — a legacy
        ``ariel.pipelines`` section, a malformed block — is ``invalid`` carrying
        the parse message, because nothing downstream can expand anything
        either.
    """
    from osprey.services.ariel_search import ARIELConfig

    if not config_dict:
        return {"status": "disabled", "concepts": 0, "errors": []}

    try:
        # Not ``_ariel_config``: this is the one caller that must resolve a
        # relative vocabulary path against the config file's own directory.
        config = ARIELConfig.from_dict(
            config_dict, _postgresql_services(), config_dir=config_dir, base=_port_base()
        )
        if not config.vocabulary.enabled:
            return {"status": "disabled", "concepts": 0, "errors": []}
        # validate() is the single source of truth for what is wrong with the
        # block: it carries the loader's errors AND "enabled without a path".
        # The prefix filter keeps unrelated config errors out of this one line.
        errors = [error for error in config.validate() if error.startswith("ariel.vocabulary")]
    except Exception as exc:  # any parse failure is reportable here
        return {"status": "invalid", "concepts": 0, "errors": [str(exc)]}

    if errors:
        return {"status": "invalid", "concepts": 0, "errors": errors}

    vocabulary = config.loaded_vocabulary
    return {
        "status": "ok",
        "concepts": vocabulary.concept_count if vocabulary is not None else 0,
        "errors": [],
    }


_EMPTY_MODULE_COUNTS = {"complete": 0, "failed": 0, "pending": 0, "gave_up": 0}


def _module_counts(entry: object) -> dict[str, int]:
    """Return the ``{complete, failed, pending, gave_up}`` counts carried by *entry*.

    A module the store has never seen has no entry at all, and reads as zeros —
    the count of rows it has produced, which is what "never seen" means. Only
    marker modules carry ``gave_up``; every other module reads it as 0.
    """
    if not isinstance(entry, dict):
        return dict(_EMPTY_MODULE_COUNTS)
    return {key: int(entry.get(key, 0)) for key in _EMPTY_MODULE_COUNTS}


async def _attachments_status(config: ARIELConfig, repository: ARIELRepository) -> dict[str, Any]:
    """Return the ``attachments`` object of ``osprey ariel status``.

    The capability block of :func:`attachments_capability` plus what only a
    status call can know: ``bytes`` (the on-disk size of ``attachment_files``,
    ``None`` when the store cannot size it), ``pending`` and ``skipped`` (the
    copy-state counts, ``{code: count}`` for skips; both ``None`` while the
    schema predates the copy state) and ``render`` (``ok`` when a render worker
    can be spawned, else ``unavailable``).
    """
    from osprey.imaging.render import probe_render_worker
    from osprey.services.ariel_search.capabilities import attachments_capability
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    try:
        size: int | None = await repository.get_attachment_bytes()
    except DatabaseQueryError:
        size = None

    pending: int | None = None
    skipped: dict[str, int] | None = None
    if (await repository.schema_facts()).has_copy_state:
        pending, skipped = await repository.get_attachment_copy_counts()

    return {
        **attachments_capability(config),
        "bytes": size,
        "pending": pending,
        "skipped": skipped,
        "render": "ok" if await probe_render_worker() else "unavailable",
    }


#: Seconds ``status`` gives each module's ``health_check()``.
_STATUS_HEALTH_TIMEOUT_S = 5.0

#: Where a module's ``health`` verdict was taken: the process answering ``status``.
_PROBED_FROM = "this process"


def _health_entry(reachable: bool | None, reason: str | None) -> dict[str, Any]:
    """One module's ``health`` object of ``osprey ariel status``."""
    return {"reachable": reachable, "reason": reason, "probed_from": _PROBED_FROM}


async def _module_health(
    config: ARIELConfig, name: str, repository: ARIELRepository
) -> dict[str, Any]:
    """The ``health`` object of one enabled module.

    The module is built alone (``stage='all'``), so a ``configure()`` error
    reports ``config`` for it and leaves the rest of the status intact. A
    picture module (``runs_inline=False``) answers through
    :func:`~osprey.services.ariel_search.enhancement.availability.preflight`,
    the same check a catch-up pass runs; any other module's ``health_check()``
    is awaited under a 5 s timeout. ``health_reason()``, when the module
    states one, overrides the reason; ``reachable`` stays the check's answer.

    Args:
        config: The ARIEL configuration.
        name: An enabled registered module.
        repository: Repository of the store the status describes.

    Returns:
        ``{"reachable": bool | None, "reason": str | None, "probed_from": "this process"}``.
    """
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.enhancement.availability import (
        preflight,
        unavailable_reason,
    )
    from osprey.services.ariel_search.enhancement.base import HealthResult, as_health_result

    try:
        built = [
            m
            for m in create_enhancers_from_config(config, stage="all", names=[name])
            if m.name == name
        ]
    except Exception as exc:  # one misconfigured module never hides the others
        return _health_entry(False, unavailable_reason(exc) or "config")
    if not built:
        return _health_entry(None, None)
    module = built[0]

    try:
        if getattr(module, "runs_inline", True):
            result = as_health_result(
                await asyncio.wait_for(module.health_check(), _STATUS_HEALTH_TIMEOUT_S)
            )
        else:
            result = await preflight(module, repository)
    except TimeoutError:
        result = HealthResult(False, "no answer in time", "unreachable")
    except Exception as exc:
        result = HealthResult(False, str(exc), unavailable_reason(exc) or "unreachable")

    reason = result.reason
    if result.reachable is False and reason is None:
        reason = "unreachable"
    try:
        override = module.health_reason()
    except Exception:
        override = None
    if override:
        reason = override
    return _health_entry(result.reachable, reason)


async def _modules_health(
    config: ARIELConfig, names: Sequence[str], repository: ARIELRepository
) -> dict[str, dict[str, Any]]:
    """The ``health`` object of every enabled module in *names*, checked concurrently."""
    enabled = [name for name in names if config.is_enhancement_module_enabled(name)]
    results = await asyncio.gather(*(_module_health(config, name, repository) for name in enabled))
    return dict(zip(enabled, results, strict=True))


#: ``health`` reasons of ``image_embedding`` that ``picture_search_unavailable`` names.
_PICTURE_SEARCH_REASONS = frozenset({"unreachable", "model", "auth", "config"})


def _image_embedding_marker(config: ARIELConfig) -> tuple[str | None, bool]:
    """The completion marker of an enabled ``image_embedding`` block.

    Returns:
        ``(table, False)`` when the block resolves, ``(None, True)`` when it is
        enabled but misconfigured, and ``(None, False)`` when it is off.
    """
    from osprey.services.ariel_search.database.migrations import image_embedding_target
    from osprey.services.ariel_search.exceptions import ModuleConfigError

    if not config.is_enhancement_module_enabled("image_embedding"):
        return None, False
    try:
        target = image_embedding_target(
            config.get_enhancement_module_config("image_embedding") or {}
        )
    except ModuleConfigError:
        return None, True
    return target.table, False


def _picture_search_unavailable(
    config: ARIELConfig, health: Mapping[str, Mapping[str, Any]]
) -> str | None:
    """Why picture search cannot answer, from the ``image_embedding`` health verdict.

    Returns:
        ``unreachable``, ``model``, ``auth`` or ``config`` while the enabled
        module is not reachable; None when it is reachable, unchecked, off, or
        has no reader (``no_reader``: ``picture_search`` is already false then).
    """
    if not config.is_enhancement_module_enabled("image_embedding"):
        return None
    entry = health.get("image_embedding") or {}
    if entry.get("reachable") is not False:
        return None
    reason = entry.get("reason")
    if reason == "no_reader":
        return None
    return reason if reason in _PICTURE_SEARCH_REASONS else "unreachable"


async def get_status(config_dict: dict, *, config_dir: Path | None = None) -> dict:
    """Return ARIEL service status as a plain dict.

    Args:
        config_dict: The raw ``ariel`` section.
        config_dir: Directory holding the config.yml it came from, used to
            resolve a relative ``ariel.vocabulary.path``.

    Returns:
        The status document. Every path through this function — unconfigured,
        healthy, and both failure branches — carries a ``vocabulary`` key, so a
        caller never has to reach the database to learn what the vocabulary is
        doing. The healthy paths also carry ``last_ingestion``: the ISO-8601
        timestamp of the newest successful ingestion run, or ``None`` when the
        store has never been ingested.

        On the healthy paths ``enhancement_modules`` is one table over the
        registered modules, each carrying ``enabled`` alongside its
        ``complete``/``failed``/``pending``/``gave_up`` counts, and
        ``orphaned_enhancement_modules`` carries the same counts for store keys
        no registered module claims — rows present with nothing left to write
        them. Every enabled module's entry also carries ``health``
        (:func:`_module_health`): ``reachable`` (None when the module has no
        health check), ``reason`` (``auth``, ``model``, ``unreachable``,
        ``no_reader``, ``config`` or None) and ``probed_from``. ``attachments``
        is the object :func:`_attachments_status` builds, its
        ``picture_search_unavailable`` taken from the ``image_embedding``
        health verdict (:func:`_picture_search_unavailable`).
        ``image_embedding_tables`` lists the picture tables
        (``table``, ``pictures``, ``dimension``, ``active``) apart from the
        text tables of ``embedding_tables``.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.attachments.compose import caption_model_id
    from osprey.services.ariel_search.config import registered_ariel_names

    # Computed first and unconditionally: the vocabulary line must survive a
    # database that is down and a config that will not parse.
    vocabulary = vocabulary_status(config_dict, config_dir)

    if not config_dict:
        return {"status": "error", "message": "ARIEL not configured", "vocabulary": vocabulary}

    try:
        config = _ariel_config(config_dict)
        service = await create_ariel_service(config)
        async with service:
            healthy, message = await service.health_check()
            markers: dict[str, str] = {}
            caption_marker = caption_model_id(config)
            if caption_marker is not None:
                markers["image_caption"] = caption_marker
            image_marker, image_config_bad = _image_embedding_marker(config)
            if image_marker is not None:
                markers["image_embedding"] = image_marker
            if markers:
                stats = await service.repository.get_enhancement_stats(markers=markers)
            else:
                stats = await service.repository.get_enhancement_stats()
            registered = registered_ariel_names("ariel_enhancement_modules")
            tables = await service.repository.get_embedding_tables()
            image_tables = await service.repository.get_image_embedding_tables()
            last_ingestion = await service.repository.get_last_ingestion()
            attachments = await _attachments_status(config, service.repository)
            health = await _modules_health(config, registered, service.repository)
            if image_config_bad:
                health["image_embedding"] = _health_entry(False, "config")
            # After the shared capability block: this process's lane state is
            # empty, so the module's own health verdict is the answer here.
            attachments["picture_search_unavailable"] = _picture_search_unavailable(config, health)

            return {
                "status": "healthy" if healthy else "unhealthy",
                "message": message,
                "database": {
                    "uri": (
                        config.database.uri.split("@")[-1]
                        if "@" in config.database.uri
                        else config.database.uri
                    ),
                    "connected": healthy,
                },
                "entries": stats.get("total_entries", 0),
                "last_ingestion": last_ingestion.isoformat() if last_ingestion else None,
                "embedding_tables": [
                    {
                        "table": t.table_name,
                        "entries": t.entry_count,
                        "dimension": t.dimension,
                        "active": t.is_active,
                    }
                    for t in tables
                ],
                "image_embedding_tables": [
                    {
                        "table": t.table_name,
                        "pictures": t.entry_count,
                        "dimension": t.dimension,
                        "active": t.is_active,
                    }
                    for t in image_tables
                ],
                "enhancement_modules": {
                    name: {
                        "enabled": config.is_enhancement_module_enabled(name),
                        **_module_counts(stats.get(name)),
                        **({"health": health[name]} if name in health else {}),
                    }
                    for name in registered
                },
                "orphaned_enhancement_modules": {
                    key: _module_counts(counts)
                    for key, counts in stats.items()
                    if key != "total_entries" and key not in registered
                },
                "search_modules": {
                    name: config.is_search_module_enabled(name)
                    for name in registered_ariel_names("ariel_search_modules")
                },
                "vocabulary": vocabulary,
                "attachments": attachments,
            }

    except Exception as e:
        msg = str(e)
        if "connection" in msg.lower() or "connect" in msg.lower():
            return {
                "status": "error",
                "message": "Cannot connect to the ARIEL database. "
                "Make sure the database is running: osprey up",
                "vocabulary": vocabulary,
            }
        return {"status": "error", "message": msg, "vocabulary": vocabulary}


async def run_migrate(
    config_dict: dict,
    progress: _ProgressCb = None,
) -> list[str]:
    """Run database migrations, waiting for any other session migrating.

    Returns:
        Migrations skipped because their tables were busy; empty when every
        pending migration ran (or skipped for a missing prerequisite).
    """
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations_detailed

    config = _ariel_config(config_dict)

    if progress:
        progress(f"Connecting to database: {config.database.uri.split('@')[-1]}")

    pool = await create_connection_pool(config.database)

    try:
        if progress:
            progress("Running migrations...")
        result = await run_migrations_detailed(pool, config)
        if progress:
            progress("Migrations complete.")
        return result.busy_skipped
    finally:
        await pool.close()


async def run_sync(
    config_dict: dict,
    limit: int | None = None,
    progress: _ProgressCb = None,
) -> SyncResult:
    """Sync ARIEL database: migrate, incremental ingest, enhance.

    Composes existing operations into a single idempotent command:

    1. Run database migrations (skips already-applied)
    2. Incremental ingest via ``IngestionScheduler.poll_once`` — fetches
       only entries added since the last successful run
    3. Catch-up (:func:`run_catchup`) — processes entries with incomplete
       enhancements from prior runs (new entries are enhanced inline during
       step 2), then the picture modules within :func:`catchup_budget`
    """
    import copy

    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations_detailed
    from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

    _require_ingestion_block(config_dict)
    config = _ariel_config(config_dict)

    # Step 1: Migrate
    if progress:
        progress("Running migrations...")

    pool = await create_connection_pool(config.database)
    try:
        migrated = await run_migrations_detailed(pool, config)
        migrations_applied = len(migrated.applied)
        busy_skipped = list(migrated.busy_skipped)
        if migrations_applied and progress:
            progress(f"  {migrations_applied} migrations applied")
        elif progress:
            progress("  Already up to date")
    finally:
        await pool.close()

    # Step 2: Incremental ingest via scheduler
    # Override require_initial_ingest so sync does a full ingest on fresh databases
    # (the scheduler default skips when no prior run exists)
    sync_dict = copy.deepcopy(config_dict)
    sync_dict.setdefault("ingestion", {}).setdefault("watch", {})["require_initial_ingest"] = False
    sync_config = _ariel_config(sync_dict)

    service = await create_ariel_service(sync_config)
    async with service:
        scheduler = IngestionScheduler(config=sync_config, repository=service.repository)
        if progress:
            ingestion = sync_config.ingestion
            source = (ingestion.source_url if ingestion else None) or "unknown"
            progress(f"Polling for new entries (source: {source})...")

        poll_start = time.monotonic()
        poll_result = await scheduler.poll_once(limit=limit)
        was_initial = poll_result.since is None

    if progress:
        progress(f"  {poll_result.entries_added} entries ingested")
        if was_initial:
            progress("  (initial full ingest)")

    # Step 3: Catch-up — enhancements earlier runs left incomplete, then the
    # picture modules, bounded by one poll interval like a watch pass.
    enhance_result = await run_catchup(
        config_dict,
        budget_s=catchup_budget(config_dict, time.monotonic() - poll_start),
        stop_event=asyncio.Event(),
        progress=progress,
    )

    return SyncResult(
        migrations_applied=migrations_applied,
        entries_ingested=poll_result.entries_added,
        entries_enhanced=enhance_result.entries_processed,
        entries_failed=poll_result.entries_failed,
        was_initial_ingest=was_initial,
        busy_skipped=busy_skipped,
    )


@asynccontextmanager
async def _ingest_copy_run(
    repository: ARIELRepository, adapter: FacilityAdapter, config: ARIELConfig
) -> AsyncIterator[CopyRun]:
    """Yield the one :class:`CopyRun` an ingest or quickstart shares across its entries.

    On a store with the attachment copy state the run is entered, so every
    fetch of the run goes through one session and one host breaker. On a
    store whose schema predates it nothing is fetched; the run is yielded
    unentered and only scopes the once-per-run schema warning.
    """
    from osprey.services.ariel_search.attachments.copy import CopyRun
    from osprey.services.ariel_search.attachments.fetch import origins_for

    if not (await repository.schema_facts()).has_copy_state:
        yield CopyRun(adapter, frozenset())
        return
    async with CopyRun(adapter, origins_for(adapter, config)) as copy_run:
        yield copy_run


def _entry_failures(outcome: EntryIngestOutcome) -> int:
    """Return what one stored entry adds to a run's failure count.

    Each failed enhancer counts once, and the entry counts once more when its
    attachment rows were not recorded.
    """
    return outcome.enhancer_failed + (0 if outcome.attachments_recorded else 1)


async def run_ingest(
    config_dict: dict,
    source: str,
    adapter: str | None,
    since: datetime | None,
    limit: int | None,
    dry_run: bool,
    progress: _ProgressCb = None,
) -> IngestResult:
    """Ingest logbook entries from a source.

    Args:
        config_dict: Raw ARIEL configuration mapping.
        source: Source file path or URL; always an override.
        adapter: Override for ``ingestion.adapter``. ``None`` leaves the
            configured adapter in place — the same rule ``run_watch`` follows,
            so a project that names its adapter in config.yml does not have to
            repeat it on every ingest.
        since: Only ingest entries after this date; a value without an offset
            is facility-local.
        limit: Maximum entries to ingest.
        dry_run: Parse entries without storing them.
        progress: Optional callback for human-readable progress lines.

    Returns:
        The run's counts.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.ingestion import get_adapter
    from osprey.services.ariel_search.ingestion.ingest import ingest_one
    from osprey.utils.config import localize_facility

    if "ingestion" not in config_dict:
        config_dict["ingestion"] = {}
    config_dict["ingestion"]["source_url"] = source
    if adapter:
        config_dict["ingestion"]["adapter"] = adapter

    config = _ariel_config(config_dict)
    since = localize_facility(since)
    adapter_instance = get_adapter(config)

    if progress:
        progress(f"Using adapter: {adapter_instance.source_system_name}")
        progress(f"Source: {source}")

    # The default stage is inline: a catch-up module never runs during ingest.
    enhancers = create_enhancers_from_config(config)
    enhancer_names = [e.name for e in enhancers]
    if enhancers and progress:
        progress(f"Enhancement modules: {enhancer_names}")

    if dry_run:
        count = 0
        async for _entry in adapter_instance.fetch_entries(since=since, limit=limit):
            count += 1
            if count % 100 == 0 and progress:
                progress(f"  Parsed {count} entries...")
        return IngestResult(
            count=count,
            enhanced_count=0,
            failed_count=0,
            unreadable_count=adapter_instance.unreadable_entries,
            dry_run=True,
            enhancer_names=enhancer_names,
        )

    service = await create_ariel_service(config)
    async with service:
        source_system = adapter_instance.source_system_name
        run_id = await service.repository.start_ingestion_run(source_system)

        count = 0
        enhanced_count = 0
        failed_count = 0

        try:
            async with _ingest_copy_run(service.repository, adapter_instance, config) as copy_run:
                async for entry in adapter_instance.fetch_entries(since=since, limit=limit):
                    outcome = await ingest_one(
                        entry, adapter_instance, service.repository, enhancers, config, copy_run
                    )
                    count += 1
                    enhanced_count += outcome.enhanced
                    failed_count += _entry_failures(outcome)

                    if count % 100 == 0 and progress:
                        if enhancers:
                            progress(f"  Ingested and enhanced {count} entries...")
                        else:
                            progress(f"  Ingested {count} entries...")

            await service.repository.complete_ingestion_run(
                run_id,
                entries_added=count,
                entries_updated=0,
                entries_failed=failed_count + adapter_instance.unreadable_entries,
            )
        except Exception as e:
            await service.repository.fail_ingestion_run(run_id, str(e))
            raise

    return IngestResult(
        count=count,
        enhanced_count=enhanced_count,
        failed_count=failed_count,
        unreadable_count=adapter_instance.unreadable_entries,
        dry_run=False,
        enhancer_names=enhancer_names,
    )


async def run_watch(
    config_dict: dict,
    source: str | None,
    adapter: str | None,
    once: bool,
    interval: int | None,
    dry_run: bool,
    progress: _ProgressCb = None,
    *,
    require_initial_ingest: bool | None = None,
    install_signal_handlers: bool = True,
    stop_reason_out: list[str] | None = None,
    initial_busy_skipped: Sequence[str] | None = (),
) -> WatchOnceResult | None:
    """Watch a source for new logbook entries.

    Args:
        config_dict: Raw ARIEL configuration mapping.
        source: Override for ``ingestion.source_url``.
        adapter: Override for ``ingestion.adapter``.
        once: Run a single poll cycle instead of the daemon loop.
        interval: Override for ``ingestion.poll_interval_seconds``.
        dry_run: Poll without writing entries.
        progress: Optional callback for human-readable progress lines.
        require_initial_ingest: When not ``None``, overrides
            ``ingestion.watch.require_initial_ingest`` in *config_dict*.
        install_signal_handlers: Install SIGINT/SIGTERM handlers that stop the
            daemon loop. A caller that already owns the process signals, such as
            one running the watch alongside other work, passes ``False``.
        stop_reason_out: When given, the reason the daemon loop ended is
            appended to this list. A loop that reports no reason appends
            nothing.
        initial_busy_skipped: Migrations an earlier sync skipped because their
            tables were busy. While any remain, each poll first retries the
            migrations (never waiting on the migrate lock) and replaces the set
            with what is still busy. The empty default retries nothing, as a
            standalone watch does; ``None`` means the earlier sync's outcome is
            unknown, so one attempt is made before the first poll and its busy
            set adopted.

    Returns:
        A ``WatchOnceResult`` when *once* is ``True``.
        In daemon mode runs until stopped and returns ``None``.
    """
    import asyncio
    import signal

    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler
    from osprey.utils.logger import get_logger

    # Only an actual override may mint the block: a config with no `ingestion`
    # at all must reach `_ariel_config` still missing it, or the "no ingestion
    # source configured" refusal below turns into an empty-source crash.
    if source or adapter or interval is not None or require_initial_ingest is not None:
        ingestion = config_dict.setdefault("ingestion", {})
        if source:
            ingestion["source_url"] = source
        if adapter:
            ingestion["adapter"] = adapter
        if interval is not None:
            ingestion["poll_interval_seconds"] = interval
        if require_initial_ingest is not None:
            ingestion.setdefault("watch", {})["require_initial_ingest"] = require_initial_ingest

    _require_ingestion_block(config_dict)
    config = _ariel_config(config_dict)

    if not config.ingestion or not config.ingestion.source_url:
        raise ValueError(
            "No ingestion source configured. "
            "Set ingestion.source_url in config.yml or use --source."
        )

    service = await create_ariel_service(config)
    async with service:
        scheduler = IngestionScheduler(
            config=config,
            repository=service.repository,
        )

        if once:
            if progress:
                progress(f"Running single poll cycle (source: {config.ingestion.source_url})")

            result = await scheduler.poll_once(dry_run=dry_run)

            return WatchOnceResult(
                entries_added=result.entries_added,
                entries_updated=result.entries_updated,
                entries_failed=result.entries_failed,
                duration_seconds=result.duration_seconds,
                since=result.since,
                dry_run=dry_run,
            )

        # Daemon mode
        poll_secs = config.ingestion.poll_interval_seconds
        if progress:
            progress(f"Watching: {config.ingestion.source_url}")
            progress(f"Poll interval: {poll_secs:g}s")
            progress("Press Ctrl+C to stop\n")

        if install_signal_handlers:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(scheduler.stop()))

        # Every loop iteration re-exports the entries that changed outside the
        # enhancement pipeline before it polls for new ones, and retries the
        # enhancements an earlier pass left incomplete after it. The scheduler
        # owns the loop, so both steps are attached to the call it makes each
        # pass rather than to a loop this function can see. The cleanup runs in
        # its own guard: the wrapper stands in for ``poll_once``, so an
        # exception escaping it would be counted as an ingestion failure and
        # could stop the daemon on the consecutive-failure cap.
        inner_poll_once = scheduler.poll_once
        busy: list[str] | None = (
            None if initial_busy_skipped is None else list(initial_busy_skipped)
        )

        async def _retry_busy_migrations() -> None:
            nonlocal busy
            from osprey.services.ariel_search.database.migrations import (
                run_migrations_detailed,
            )

            log = get_logger("ariel")
            try:
                migrated = await run_migrations_detailed(service.pool, config, lock="try")
            except Exception as e:  # a failed retry must not fail the poll.
                log.warning(f"Migration retry failed, continuing: {e}")
                return
            if migrated.lock_held:
                log.info("migrate running elsewhere; retrying busy migrations next poll")
                return
            if migrated.applied:
                # The repository's pool outlives this migrate, so its cached
                # schema facts would hide the new objects until the TTL ran out.
                service.repository.invalidate_schema_facts()
            busy = list(migrated.busy_skipped)

        async def _poll_once_with_resync(*args: Any, **kwargs: Any):
            poll_start = time.monotonic()
            if busy is None or busy:
                await _retry_busy_migrations()
            await resync_qmd_mirror_best_effort(config_dict, progress)
            result = await inner_poll_once(*args, **kwargs)
            try:
                await run_catchup(
                    config_dict,
                    budget_s=catchup_budget(config_dict, time.monotonic() - poll_start),
                    stop_event=scheduler._stop_event,
                    progress=progress,
                )
            except Exception as e:  # cleanup is not an ingestion failure.
                get_logger("ariel").warning(f"Enhance cleanup failed, continuing: {e}")
            return result

        scheduler.poll_once = _poll_once_with_resync  # type: ignore[method-assign]

        stop_reason = await scheduler.run_forever()
        if stop_reason_out is not None and stop_reason is not None:
            stop_reason_out.append(stop_reason)
        return None


async def run_sync_watch(
    config_dict: dict,
    progress: _ProgressCb = None,
) -> StopReason | None:
    """Sync once, then watch the source until a signal or the failure cap.

    The long-running form of :func:`run_sync`, and the entry point a container
    that owns an ARIEL database runs. Both halves run inside one task, so the
    single SIGINT/SIGTERM handler installed here -- before the sync starts --
    ends whichever half is in flight; ``run_watch`` installs none of its own.
    Cancellation is how a signalled run stops, so it leaves an in-flight ingest
    recorded as a run that did not succeed and the next start ignores it.

    A sync that fails is logged and does not stop the daemon: the watch that
    follows is asked for a full first ingest, so a container that started while
    its source was unreachable still ingests everything on the first poll that
    succeeds, and the scheduler's backoff owns the retries in between.

    Args:
        config_dict: Raw ARIEL configuration mapping. Mutated in place with the
            watch override, as the other overrides on ``run_watch`` are.
        progress: Optional callback for human-readable progress lines.

    Returns:
        The reason the daemon loop ended, or ``None`` when it reported none.

    Raises:
        asyncio.CancelledError: When a signal cancelled the run. ``CancelledError``
            is a ``BaseException``, so it passes through the fail-soft sync guard.
    """
    import asyncio
    import signal

    from osprey.services.ariel_search.ingestion.scheduler import StopReason
    from osprey.utils.logger import get_logger

    async def _sync_then_watch() -> StopReason | None:
        busy_skipped: list[str] | None
        try:
            synced = await run_sync(config_dict, progress=progress)
            busy_skipped = list(synced.busy_skipped)
        except Exception as e:  # the loop's backoff owns the retries.
            failure = f"{type(e).__name__}: {e}"
            get_logger("ariel").warning(f"Initial sync failed, watching anyway: {failure}")
            if progress:
                progress(f"  Initial sync failed ({failure}); watching anyway")
            busy_skipped = None

        reasons: list[str] = []
        await run_watch(
            config_dict,
            source=None,
            adapter=None,
            once=False,
            interval=None,
            dry_run=False,
            progress=progress,
            require_initial_ingest=False,
            install_signal_handlers=False,
            stop_reason_out=reasons,
            initial_busy_skipped=busy_skipped,
        )
        return StopReason(reasons[0]) if reasons else None

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(_sync_then_watch())
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await task
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


# ---------------------------------------------------------------------------
# qmd markdown-mirror resync
# ---------------------------------------------------------------------------

#: File at the mirror root holding the last resync watermark. Dot-prefixed so
#: the sidecar skips it as a corpus document.
QMD_WATERMARK_NAME = ".qmd-resync-watermark"

#: Serialization schema for the watermark marker itself.
QMD_WATERMARK_FORMAT_VERSION = 1

#: Markdown renderer format represented by a current watermark. Bump this when
#: existing rows must be re-rendered even though their database values did not
#: change; old markers will then force one full backfill scan.
QMD_RENDERER_VERSION = 1

#: Rows fetched per page while scanning for changed entries. Bounds the memory
#: a rebuild of a large logbook needs without making the scan chatty.
QMD_RESYNC_PAGE_SIZE = 500


def _qmd_mirror_root(config: ARIELConfig) -> Path:
    """Resolve the configured qmd mirror root to an absolute directory.

    Resolution goes through the exporter's own rule, so the resync and the
    enhancement module that normally writes the mirror cannot land on different
    directories from the same config value.

    Args:
        config: Resolved ARIEL configuration with ``qmd_export`` enabled.

    Returns:
        Absolute path to the mirror root. The directory need not exist.

    Raises:
        ValueError: If ``enhancement_modules.qmd_export.mirror_path`` is unset.
            An enabled exporter with nowhere to write is a broken config, not a
            reason to silently skip the mirror.
    """
    from osprey.services.ariel_search.enhancement.qmd_export import resolve_mirror_path

    module_config = config.get_enhancement_module_config("qmd_export") or {}
    raw = module_config.get("mirror_path")
    if not raw:
        raise ValueError(
            "ariel.enhancement_modules.qmd_export.mirror_path is required when "
            "the qmd_export module is enabled"
        )
    return resolve_mirror_path(raw)


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` atomically.

    The content lands in a temp file beside the target, so the rename stays on
    one filesystem and a reader sees either the old file or the whole new one.

    Args:
        path: Destination file. Its parent directory must already exist.
        text: Full file content, UTF-8 encoded.

    Raises:
        OSError: If the temp file cannot be written or renamed. The temp file
            is removed first, leaving ``path`` untouched.
    """
    import os
    import tempfile

    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _read_qmd_watermark(mirror_root: Path) -> datetime | None:
    """Read the resync watermark stored at the mirror root.

    Args:
        mirror_root: Mirror root directory, which may not exist yet.

    Returns:
        The stored timestamp as an aware UTC ``datetime``, or ``None`` when no
        current, valid watermark exists. Missing, legacy, corrupt, and
        version-mismatched markers all fall back to a full scan, which is
        correct but slower -- never wrong.
    """
    import json
    from datetime import UTC, datetime

    marker = mirror_root / QMD_WATERMARK_NAME
    try:
        raw = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    format_version = payload.get("format_version")
    renderer_version = payload.get("renderer_version")
    if (
        type(format_version) is not int
        or format_version != QMD_WATERMARK_FORMAT_VERSION
        or type(renderer_version) is not int
        or renderer_version != QMD_RENDERER_VERSION
    ):
        return None

    raw_moment = payload.get("updated_at")
    if not isinstance(raw_moment, str):
        return None
    try:
        moment = datetime.fromisoformat(raw_moment)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _write_qmd_watermark(mirror_root: Path, moment: datetime) -> None:
    """Store the resync watermark at the mirror root.

    Args:
        mirror_root: Mirror root directory. It is created if missing.
        moment: Highest ``updated_at`` the pass observed.

    Raises:
        OSError: If the watermark cannot be written. A pass that exported rows
            but could not record where it stopped must fail loudly, because the
            silent alternative is re-exporting from the old watermark forever.
    """
    import json
    from datetime import UTC

    normalized = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    payload = {
        "format_version": QMD_WATERMARK_FORMAT_VERSION,
        "renderer_version": QMD_RENDERER_VERSION,
        "updated_at": normalized.isoformat(),
    }
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    mirror_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(mirror_root / QMD_WATERMARK_NAME, f"{serialized}\n")


def _touch_qmd_marker(mirror_root: Path) -> bool:
    """Advance the sidecar's freshness marker at the mirror root.

    Uses the exporter's marker name, because a resync and an enhancer write
    that are seen as different files would leave the sidecar polling one of
    them.

    Args:
        mirror_root: Mirror root directory. It is created if missing.

    Returns:
        ``True`` if the marker was rewritten. A failure returns ``False``
        rather than raising: the exported documents are already on disk, so the
        sidecar's interval sweep still finds them, just later.
    """
    from datetime import UTC, datetime

    from osprey.services.ariel_search.enhancement.qmd_export import TOUCH_MARKER_NAME
    from osprey.utils.logger import get_logger

    try:
        mirror_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            mirror_root / TOUCH_MARKER_NAME,
            f"{datetime.now(UTC).isoformat()}\n",
        )
    except OSError as e:
        get_logger("ariel").warning(
            f"Could not update the qmd freshness marker in {mirror_root}: {e} "
            "-- the sidecar will pick the changes up on its next sweep"
        )
        return False
    return True


def _wipe_qmd_mirror(mirror_root: Path) -> int:
    """Delete every mirrored document under the mirror root.

    Only the shard tree goes: dot-prefixed entries at the root are the
    watermark and the freshness marker, which are bookkeeping rather than
    corpus. This is what makes ``--rebuild`` able to drop files for entries
    that no longer exist in Postgres.

    Args:
        mirror_root: Mirror root directory, which may not exist.

    Returns:
        Number of markdown files removed.
    """
    import shutil

    if not mirror_root.is_dir():
        return 0

    removed = 0
    for child in mirror_root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            removed += sum(1 for _ in child.rglob("*.md"))
            shutil.rmtree(child)
        else:
            if child.suffix == ".md":
                removed += 1
            child.unlink()
    return removed


async def _fetch_changed_page(
    cur: Any,
    watermark: datetime | None,
    cursor_key: tuple[datetime, str] | None,
    page_size: int,
) -> list[Any]:
    """Fetch one page of entries at or after the watermark.

    Pages are keyset-paginated on ``(updated_at, entry_id)`` so a rebuild of a
    large logbook never holds more than one page in memory and never skips a
    row because an earlier page shifted under it.

    Args:
        cur: Open ``dict_row`` cursor.
        watermark: Lower bound for the first page, inclusive. Ties are re-read
            deliberately -- the byte-compare writer makes a re-read of an
            unchanged row free, and the inclusive bound means a row sharing the
            watermark's timestamp can never be skipped.
        cursor_key: Last ``(updated_at, entry_id)`` of the previous page, or
            ``None`` for the first page.
        page_size: Maximum rows to return.

    Returns:
        The page's rows, ordered by ``(updated_at, entry_id)``.
    """
    if cursor_key is not None:
        await cur.execute(
            """
            SELECT * FROM enhanced_entries
            WHERE (updated_at, entry_id) > (%s, %s)
            ORDER BY updated_at, entry_id
            LIMIT %s
            """,
            [cursor_key[0], cursor_key[1], page_size],
        )
    elif watermark is not None:
        await cur.execute(
            """
            SELECT * FROM enhanced_entries
            WHERE updated_at >= %s
            ORDER BY updated_at, entry_id
            LIMIT %s
            """,
            [watermark, page_size],
        )
    else:
        await cur.execute(
            """
            SELECT * FROM enhanced_entries
            ORDER BY updated_at, entry_id
            LIMIT %s
            """,
            [page_size],
        )
    return list(await cur.fetchall())


async def run_qmd_resync(
    config_dict: dict,
    rebuild: bool = False,
    page_size: int = QMD_RESYNC_PAGE_SIZE,
    progress: _ProgressCb = None,
) -> QmdResyncResult | None:
    """Re-export entries whose rows changed outside the enhancement loop.

    The markdown mirror is normally written by the ``qmd_export`` enhancement
    module as entries flow through ingestion. Several mutation paths write
    straight to ``enhanced_entries`` and never reach an enhancer, so this pass
    is what keeps the mirror honest: it scans for rows changed since the stored
    watermark, re-renders each one through the byte-compare writer, and moves
    the watermark forward.

    Bookkeeping-only churn is therefore free. A row whose ``updated_at`` moved
    without its content changing renders to the same bytes, the writer leaves
    the file alone, and the sidecar's next scan finds nothing to re-embed.

    Args:
        config_dict: Raw ``ariel`` config section.
        rebuild: Wipe the mirror and re-export every entry. This is the only
            pass that removes files for entries that no longer exist, so it is
            what recovers a mirror after ``osprey ariel purge``.
        page_size: Rows per scan page.
        progress: Optional progress sink.

    Returns:
        A :class:`QmdResyncResult`, or ``None`` when the ``qmd_export`` module
        is not enabled and there is no mirror to keep.

    Raises:
        ValueError: If ``qmd_export`` is enabled without a ``mirror_path``.
    """
    from psycopg.rows import dict_row

    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.enhancement.qmd_export.writer import write_entry
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")

    config = _ariel_config(config_dict)
    if not config.is_enhancement_module_enabled("qmd_export"):
        return None

    mirror_root = _qmd_mirror_root(config)
    removed = 0
    watermark = None
    if rebuild:
        removed = _wipe_qmd_mirror(mirror_root)
        # Drop the old bound too. A rebuild that finds nothing -- the shape of a
        # rebuild right after a purge -- must not leave a watermark behind that
        # would make the next incremental pass skip everything older than it.
        (mirror_root / QMD_WATERMARK_NAME).unlink(missing_ok=True)
    else:
        watermark = _read_qmd_watermark(mirror_root)

    if progress:
        scope = "full rebuild" if rebuild else f"changed since {watermark or 'the beginning'}"
        progress(f"Resyncing qmd mirror at {mirror_root} ({scope})...")

    scanned = written = unchanged = failed = 0
    highest: datetime | None = None
    cursor_key: tuple[datetime, str] | None = None

    pool = await create_connection_pool(config.database)
    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                while True:
                    rows = await _fetch_changed_page(cur, watermark, cursor_key, page_size)
                    page_key = cursor_key

                    for row in rows:
                        entry = dict(row)
                        scanned += 1
                        moment = entry.get("updated_at")
                        if moment is not None:
                            page_key = (moment, str(entry.get("entry_id", "")))
                            if highest is None or moment > highest:
                                highest = moment
                        try:
                            if write_entry(mirror_root, entry):
                                written += 1
                            else:
                                unchanged += 1
                        except (ValueError, OSError) as e:
                            failed += 1
                            logger.warning(f"Could not mirror entry {entry.get('entry_id')!r}: {e}")

                    if len(rows) < page_size:
                        break
                    if page_key == cursor_key:
                        # A full page advanced nothing, which only happens when
                        # every row in it carried a NULL updated_at. Those sort
                        # last, so there is no key to page past them with.
                        logger.warning(
                            "Stopping the qmd resync scan: a full page of entries "
                            "carried no updated_at timestamp"
                        )
                        break
                    cursor_key = page_key
    finally:
        await pool.close()

    if written:
        _touch_qmd_marker(mirror_root)
    if highest is not None:
        _write_qmd_watermark(mirror_root, highest)

    if progress:
        progress(
            f"  qmd mirror: {written} written, {unchanged} unchanged, "
            f"{failed} failed of {scanned} scanned"
        )

    return QmdResyncResult(
        scanned=scanned,
        written=written,
        unchanged=unchanged,
        failed=failed,
        removed=removed,
        rebuild=rebuild,
        mirror_path=str(mirror_root),
        watermark=highest,
    )


async def resync_qmd_mirror_best_effort(
    config_dict: dict,
    progress: _ProgressCb = None,
) -> QmdResyncResult | None:
    """Run :func:`run_qmd_resync` as a pre-step that cannot fail its caller.

    Ingest and watch run this before doing their own work. Their job is getting
    entries into Postgres; a mirror that cannot be written is worth a warning,
    not an aborted ingestion.

    Args:
        config_dict: Raw ``ariel`` config section.
        progress: Optional progress sink, used only when something was written.

    Returns:
        The pass result, or ``None`` when the module is disabled or the pass
        failed.
    """
    from osprey.utils.logger import get_logger

    try:
        result = await run_qmd_resync(config_dict)
    except Exception as e:  # a mirror problem must not stop ingestion.
        get_logger("ariel").warning(f"qmd mirror resync failed, continuing: {e}")
        return None

    if result is not None and result.written and progress:
        progress(f"qmd mirror: re-exported {result.written} entries changed outside ingestion")
    return result


def _runs_in_catchup(module: str) -> bool:
    """Return whether the registered module ``module`` runs only in the catch-up.

    Reads ``runs_inline`` from the registered class without instantiating or
    configuring it, so a misconfigured catch-up module cannot raise here.
    """
    from osprey.registry import get_registry

    registry = get_registry()
    registry.initialize(silent=True)
    found = registry.get_ariel_enhancement_module(module)
    if found is None:
        return False
    cls, _registration = found
    return not getattr(cls, "runs_inline", True)


#: Why ``enhance --force`` refuses a picture module.
FORCE_REFUSAL = (
    "--force does not re-run image_caption/image_embedding: their results are kept per "
    "picture and model. Change model.model_id (captions) or model/dimensions (embeddings) "
    "to re-run, or use --retry-failed for per-picture failures."
)


async def run_enhance(
    config_dict: dict,
    module: str | None,
    force: bool,
    limit: int,
    progress: _ProgressCb = None,
    *,
    stop_event: asyncio.Event | None = None,
    retry_failed: bool = False,
) -> EnhanceResult:
    """Run enhancement modules on entries.

    Text modules (``runs_inline=True``) run entry-major through their
    ``enhance()``. Picture modules (``runs_inline=False``) run through
    :func:`~osprey.services.ariel_search.enhancement.image_driver.drive_image_module`
    with no budget, each under its advisory lock ``ariel_enhance:<module>``;
    their ``enhance()`` is never called and no picture is fetched.

    Args:
        config_dict: Raw ``ariel`` config block.
        module: Only this module when given; every enabled module otherwise,
            the text modules first.
        force: Re-run the text modules on the newest entries instead of the
            incomplete ones. Picture modules keep their results per picture
            and model, so ``force`` skips them.
        limit: Most entries read per text module, and most entries handed to
            ``run_entry`` per picture module.
        progress: Optional progress callback.
        stop_event: Checked before each entry; once set, the remaining entries
            are left for a later pass.
        retry_failed: With ``module``, give its failed entries a new set of
            attempts first (see :func:`retry_failed_entries`).

    Raises:
        ValueError: With :data:`FORCE_REFUSAL` when ``force`` names a picture module.
    """
    config = _ariel_config(config_dict)
    if module and _runs_in_catchup(module):
        if force:
            raise ValueError(FORCE_REFUSAL)
        modules = _build_image_modules(config, [module], progress)
        if not modules:
            return EnhanceResult(entries_processed=0, module_names=[])
        if retry_failed:
            await retry_failed_entries(config, module, progress)
        walked = await _drive_image_modules(
            config, modules, budget=None, stop_event=stop_event, progress=progress, limit=limit
        )
        return EnhanceResult(entries_processed=walked, module_names=[m.name for m in modules])

    if module and retry_failed:
        await retry_failed_entries(config, module, progress)
    image_names = [] if module else _catchup_module_names(config)
    text = await _run_text_enhance(
        config,
        module,
        force,
        limit,
        progress,
        stop_event=stop_event,
        report_empty=not image_names,
    )
    if not image_names:
        return text
    if force and progress:
        progress(
            f"--force re-runs the text modules only; {', '.join(image_names)}"
            " run their normal (unforced) pass"
        )
    modules = _build_image_modules(config, image_names, progress)
    walked = 0
    if modules:
        walked = await _drive_image_modules(
            config, modules, budget=None, stop_event=stop_event, progress=progress, limit=limit
        )
    return EnhanceResult(
        entries_processed=text.entries_processed + walked,
        module_names=[*text.module_names, *(m.name for m in modules)],
        succeeded=text.succeeded,
        failed=text.failed,
        set_aside=text.set_aside,
    )


async def _run_text_enhance(
    config: ARIELConfig,
    module: str | None,
    force: bool,
    limit: int,
    progress: _ProgressCb,
    *,
    stop_event: asyncio.Event | None,
    report_empty: bool = True,
) -> EnhanceResult:
    """Run the text modules (``runs_inline=True``) entry-major over their owed entries."""
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.database.repository import (
        MAX_ENHANCEMENT_ATTEMPTS,
        text_mark_kwargs,
    )
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")
    enhancers = create_enhancers_from_config(
        config, stage="inline", names=[module] if module else None
    )
    if module:
        enhancers = [e for e in enhancers if e.name == module]

    if not enhancers:
        if progress and report_empty:
            progress("No enhancement modules enabled or selected")
        return EnhanceResult(entries_processed=0, module_names=[])

    module_names = [e.name for e in enhancers]
    if progress:
        progress(f"Enhancement modules: {module_names}")

    service = await create_ariel_service(config)
    async with service:
        # Which enhancers each entry is owed, so a module that finished an entry
        # or set it aside never runs on it again.
        owed: dict[str, set[str]] = {}
        entries: list[EnhancedLogbookEntry] = []

        def _collect(found: list[EnhancedLogbookEntry], names: list[str]) -> None:
            for entry in found:
                entry_id = entry["entry_id"]
                if entry_id not in owed:
                    owed[entry_id] = set()
                    entries.append(entry)
                owed[entry_id].update(names)

        if force:
            _collect(await service.repository.search_by_time_range(limit=limit), module_names)
        else:
            for enhancer in enhancers:
                incomplete = await service.repository.get_incomplete_entries(
                    module_name=enhancer.name,
                    limit=limit,
                )
                _collect(incomplete, [enhancer.name])

        if progress:
            progress(f"Processing {len(entries)} entries...")

        has_copy_state = (await service.repository.schema_facts()).has_copy_state
        succeeded = failed = set_aside = 0
        async with service.pool.connection() as conn:
            for i, entry in enumerate(entries):
                if stop_event is not None and stop_event.is_set():
                    break
                for enhancer in enhancers:
                    if enhancer.name not in owed[entry["entry_id"]]:
                        continue
                    mark_kwargs = text_mark_kwargs(enhancer.name, entry, has_copy_state)
                    try:
                        await enhancer.enhance(entry, conn)
                        await service.repository.mark_enhancement_complete(
                            entry["entry_id"],
                            enhancer.name,
                            **mark_kwargs,
                        )
                        succeeded += 1
                    except Exception as e:
                        failed += 1
                        attempts = await service.repository.mark_enhancement_failed(
                            entry["entry_id"],
                            enhancer.name,
                            str(e),
                        )
                        if attempts >= MAX_ENHANCEMENT_ATTEMPTS:
                            set_aside += 1
                            logger.warning(
                                f"Entry {entry['entry_id']}: {enhancer.name} failed {attempts} "
                                f"times; it is left out of later passes ({str(e)[:200]})"
                            )

                if (i + 1) % 10 == 0 and progress:
                    progress(f"  Processed {i + 1} entries...")

    if progress:
        progress(
            f"Enhancement complete: {len(entries)} entries, {succeeded} succeeded, "
            f"{failed} failed, {set_aside} set aside after {MAX_ENHANCEMENT_ATTEMPTS} "
            "failed attempts"
        )
    return EnhanceResult(
        entries_processed=len(entries),
        module_names=module_names,
        succeeded=succeeded,
        failed=failed,
        set_aside=set_aside,
    )


def _build_image_modules(config: ARIELConfig, names: list[str], progress: _ProgressCb) -> list[Any]:
    """Build each named picture module alone; a ``configure()`` error skips only that one."""
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.enhancement.availability import (
        fix_for,
        report_unavailable,
    )

    modules: list[Any] = []
    for name in names:
        try:
            built = [
                m
                for m in create_enhancers_from_config(config, stage="catchup", names=[name])
                if m.name == name and not getattr(m, "runs_inline", True)
            ]
        except Exception as exc:  # one misconfigured module never stops the others
            report_unavailable(name, "config", str(exc), fix_for(name, "config", exc))
            if progress:
                progress(f"{name}: skipped, unavailable (config: {exc})")
            continue
        if not built and progress:
            progress(f"{name}: not enabled")
        modules.extend(built)
    return modules


async def _drive_image_modules(
    config: ARIELConfig,
    modules: list[Any],
    *,
    budget: float | None,
    stop_event: asyncio.Event | None,
    progress: _ProgressCb,
    limit: int | None = None,
) -> int:
    """Drive each picture module once, module-major, under its advisory lock.

    With a ``budget``, each module gets an equal share of what is left, so one
    that finishes early leaves its rest to the next. A connection is held only
    around the driver's own statements, never across a model call.

    Returns:
        The entries handed to ``run_entry`` over every module.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.enhancement.image_driver import drive_image_module
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")
    walked = 0
    started = time.monotonic()
    service = await create_ariel_service(config)
    async with service:
        repository = service.repository
        for index, module in enumerate(modules):
            if stop_event is not None and stop_event.is_set():
                break
            share = None
            if budget is not None:
                left = max(0.0, budget - (time.monotonic() - started))
                share = left / (len(modules) - index)
            async with _module_lock(repository.pool, module.name) as held:
                if not held:
                    _say(progress, logger, f"{module.name}: running in another process")
                    continue
                try:
                    outcome = await drive_image_module(
                        module,
                        repository,
                        budget=share,
                        stop_event=stop_event,
                        progress=progress,
                        limit=limit,
                    )
                except Exception as exc:  # one module never stops the others
                    logger.warning(f"{module.name}: picture pass failed, continuing: {exc}")
                    continue
            walked += outcome.entries_walked
            if outcome.skipped == "unavailable":
                _say(progress, logger, f"{module.name}: skipped, unavailable ({outcome.ended})")
            elif outcome.skipped == "busy":
                _say(progress, logger, f"{module.name}: skipped, a cancelled call is still running")
            elif progress:
                progress(f"{module.name}: {outcome.entries_walked} entries walked")
    return walked


def _module_lock(pool: Any, module: str) -> AbstractAsyncContextManager[bool]:
    """The advisory lock ``ariel_enhance:<module>`` every pass of ``module`` takes, not waiting."""
    from osprey.services.ariel_search.database import connection as connection_mod

    # The pool is always built from a DSN string.
    return connection_mod.try_advisory_lock(cast(str, pool.conninfo), f"ariel_enhance:{module}")


def _say(progress: _ProgressCb, logger: Any, message: str) -> None:
    """Report ``message`` through ``progress`` when there is one, else log it at INFO."""
    if progress:
        progress(message)
    else:
        logger.info(message)


#: Resets the attempt count of a module's failed entries, ``gave_up`` included.
_RESET_ATTEMPTS_SQL = """
UPDATE enhanced_entries
SET enhancement_status = jsonb_set(
    enhancement_status,
    %(path)s::text[],
    (enhancement_status->%(module)s) - 'gave_up' - 'attempts'
)
WHERE enhancement_status->%(module)s->>'status' = 'failed'
AND (enhancement_status->%(module)s ? 'gave_up' OR enhancement_status->%(module)s ? 'attempts')
RETURNING entry_id
"""

#: Drops a module's key from one entry's status, only when it is there.
_CLEAR_MODULE_KEY_SQL = (
    "UPDATE enhanced_entries SET enhancement_status = enhancement_status - %(module)s"
    " WHERE entry_id = %(entry_id)s AND enhancement_status ? %(module)s"
)

#: Entries holding a failed caption under ``%(model)s`` other than ``over_image_cap``.
_CAPTION_ERROR_ENTRIES_SQL = """
SELECT e.entry_id FROM enhanced_entries e
WHERE jsonb_typeof(e.attachment_captions) = 'object'
AND EXISTS (
    SELECT 1 FROM jsonb_each(e.attachment_captions) AS c(attachment_id, per_model)
    WHERE jsonb_typeof(c.per_model) = 'object'
    AND jsonb_typeof(c.per_model->%(model)s) = 'object'
    AND c.per_model->%(model)s ? 'error'
    AND c.per_model->%(model)s->>'error' IS DISTINCT FROM 'over_image_cap'
)
ORDER BY e.entry_id
"""

#: The caption error that is a decision, not a failure, and is never retried.
_OVER_IMAGE_CAP = "over_image_cap"


async def retry_failed_entries(config: ARIELConfig, module: str, progress: _ProgressCb) -> int:
    """Give ``module``'s failed entries a new set of attempts.

    Every module: a failed status loses its ``gave_up`` and ``attempts``. The
    picture modules also forget their per-picture failures under the current
    model, one transaction per entry, the entry row locked first:
    ``image_caption`` deletes the current model's ``{error}`` captions (keeping
    ``over_image_cap``), recomposes ``attachment_text`` and clears the text keys
    when it changed; ``image_embedding`` deletes the current table's rows with a
    ``skip_reason``. An entry that lost a failure then has its module key
    cleared, so the next pass walks it again.

    The whole step runs under the module's advisory lock
    ``ariel_enhance:<module>``, the one every pass of that module takes, so a
    concurrent pass cannot undo it from a stale read. When another process
    holds the lock, nothing is changed. A misconfigured ``image_embedding``
    (no current table) is reported as unavailable and changes nothing.

    Returns:
        The distinct entries given a new set of attempts.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.enhancement.availability import (
        fix_for,
        report_unavailable,
    )
    from osprey.services.ariel_search.exceptions import ModuleConfigError
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")
    table: str | None = None
    if module == "image_embedding":
        try:
            table = image_embedding_current_table(config)
        except ModuleConfigError as exc:
            report_unavailable(module, "config", str(exc), fix_for(module, "config", exc))
            _say(progress, logger, f"{module}: skipped, unavailable (config: {exc})")
            return 0

    retried: set[str] = set()
    service = await create_ariel_service(config)
    async with service:
        pool = service.repository.pool
        async with _module_lock(pool, module) as held:
            if not held:
                _say(progress, logger, f"{module}: running in another process")
                return 0
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    _RESET_ATTEMPTS_SQL, {"path": [module], "module": module}
                )
                retried.update(row[0] for row in await cursor.fetchall())
            if module == "image_caption":
                retried |= await _forget_caption_failures(pool, config)
            elif table is not None:
                retried |= await _forget_embedding_failures(pool, table)
    if progress:
        progress(f"{module}: {len(retried)} failed entries will be retried")
    return len(retried)


async def _forget_caption_failures(pool: Any, config: ARIELConfig) -> set[str]:
    """Delete the current model's failed captions, one entry per transaction."""
    from psycopg.types.json import Jsonb

    from osprey.services.ariel_search.attachments.compose import (
        caption_model_id,
        compose_attachment_text,
    )
    from osprey.services.ariel_search.database.repository import ARIELRepository

    model_id = caption_model_id(config)
    if not model_id:
        return set()
    async with pool.connection() as conn:
        cursor = await conn.execute(_CAPTION_ERROR_ENTRIES_SQL, {"model": model_id})
        entry_ids = [row[0] for row in await cursor.fetchall()]
    forgotten: set[str] = set()
    for entry_id in entry_ids:
        async with pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                "SELECT attachments, attachment_text, attachment_captions"
                " FROM enhanced_entries WHERE entry_id = %(entry_id)s FOR UPDATE",
                {"entry_id": entry_id},
            )
            row = await cursor.fetchone()
            if row is None:
                continue
            attachments, old_text, stored = row
            captions: dict[str, Any] = dict(stored) if isinstance(stored, Mapping) else {}
            removed = False
            for attachment_id, per_model in list(captions.items()):
                if not isinstance(per_model, Mapping):
                    continue
                value = per_model.get(model_id)
                if not isinstance(value, Mapping) or "error" not in value:
                    continue
                if value.get("error") == _OVER_IMAGE_CAP:
                    continue
                kept = {k: v for k, v in per_model.items() if k != model_id}
                if kept:
                    captions[attachment_id] = kept
                else:
                    del captions[attachment_id]
                removed = True
            if not removed:
                continue
            text = compose_attachment_text(entry_id, attachments, captions, model_id)
            await conn.execute(
                "UPDATE enhanced_entries"
                " SET attachment_text = %(text)s, attachment_captions = %(captions)s::jsonb"
                " WHERE entry_id = %(entry_id)s",
                {"entry_id": entry_id, "text": text, "captions": Jsonb(captions)},
            )
            if text != old_text:
                await ARIELRepository.clear_text_status_keys(conn, entry_id)
            await conn.execute(
                _CLEAR_MODULE_KEY_SQL, {"entry_id": entry_id, "module": "image_caption"}
            )
            forgotten.add(entry_id)
    return forgotten


def image_embedding_current_table(config: ARIELConfig) -> str:
    """The image table ``image_embedding`` writes under ``config``.

    Raises:
        ModuleConfigError: If the module's model or dimensions are not usable.
    """
    from osprey.services.ariel_search.database.migrations import image_embedding_target

    module_cfg = config.get_enhancement_module_config("image_embedding") or {}
    return image_embedding_target(module_cfg).table


async def _forget_embedding_failures(pool: Any, table: str) -> set[str]:
    """Delete ``table``'s skip rows, one entry per transaction."""
    from psycopg import sql

    from osprey.services.ariel_search.database.repository import ARIELRepository

    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT to_regclass(%(table)s)", {"table": table})
        found = await cursor.fetchone()
        if found is None or found[0] is None:
            return set()
        cursor = await conn.execute(
            sql.SQL(
                "SELECT DISTINCT f.entry_id FROM {table} t"
                " JOIN attachment_files f ON f.attachment_id = t.attachment_id"
                " WHERE t.skip_reason IS NOT NULL ORDER BY f.entry_id"
            ).format(table=sql.Identifier(table))
        )
        entry_ids = [row[0] for row in await cursor.fetchall()]
    forgotten: set[str] = set()
    delete = sql.SQL(
        "DELETE FROM {table} t USING attachment_files f"
        " WHERE f.attachment_id = t.attachment_id AND f.entry_id = %(entry_id)s"
        " AND t.skip_reason IS NOT NULL"
    ).format(table=sql.Identifier(table))
    for entry_id in entry_ids:
        async with pool.connection() as conn, conn.transaction():
            if not await ARIELRepository.lock_entry(conn, entry_id):
                continue
            cursor = await conn.execute(delete, {"entry_id": entry_id})
            if cursor.rowcount == 0:
                continue
            await conn.execute(
                _CLEAR_MODULE_KEY_SQL, {"entry_id": entry_id, "module": "image_embedding"}
            )
            forgotten.add(entry_id)
    return forgotten


#: Seconds a picture call may take when a module states no ``timeout_seconds``.
_DEFAULT_MODULE_TIMEOUT_S = 300.0

#: Slack added to the image stage's backstop on top of budget and module timeout.
_CATCHUP_BACKSTOP_SLACK_S = 60.0

_BUDGET_KEY = "ariel.enhancement.catchup_budget_seconds"


def catchup_budget(config_dict: Mapping[str, Any], poll_elapsed: float) -> float:
    """Return the seconds one catch-up pass may spend after a poll.

    The authored ``ariel.enhancement.catchup_budget_seconds`` when set, else the
    rest of the poll interval: ``max(0, poll_interval_seconds - poll_elapsed)``,
    so the next poll stays on schedule.

    Args:
        config_dict: Raw ``ariel`` config block.
        poll_elapsed: Seconds the poll before this catch-up took.

    Returns:
        The budget in seconds, never negative.

    Raises:
        ValueError: If the key is present but not null or a number >= 0.
    """
    raw = (config_dict.get("enhancement", {}) or {}).get("catchup_budget_seconds")
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not raw >= 0:
            raise ValueError(f"{_BUDGET_KEY} must be null or a number >= 0, got {raw!r}")
        return float(raw)
    ingestion = _ariel_config(dict(config_dict)).ingestion
    interval = float(ingestion.poll_interval_seconds) if ingestion else 3600.0
    return max(0.0, interval - poll_elapsed)


def _catchup_module_names(config: ARIELConfig) -> list[str]:
    """Enabled registered ``runs_inline=False`` modules, in execution order."""
    from osprey.registry import get_registry

    registry = get_registry()
    registry.initialize(silent=True)
    return [
        name
        for name in registry.list_ariel_enhancement_modules()
        if config.is_enhancement_module_enabled(name) and _runs_in_catchup(name)
    ]


def _module_timeout(module: Any) -> float:
    """The per-call timeout a module states, or the default."""
    value = getattr(module, "timeout_seconds", None)
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return _DEFAULT_MODULE_TIMEOUT_S


async def run_catchup(
    config_dict: dict,
    *,
    budget_s: float | None,
    stop_event: asyncio.Event | None,
    progress: _ProgressCb = None,
) -> EnhanceResult:
    """Run one catch-up pass: the text modules, then the picture modules.

    Never fetches or renders a picture; copying is the poll's job.

    1. Text modules (``runs_inline=True``) catch up as :func:`run_enhance` does,
       up to 1000 entries and with no outer timeout.
    2. Picture modules (``runs_inline=False``) share what is left of
       ``budget_s`` module-major: each gets an equal share of the remaining
       time, so one that finishes early leaves its rest to the next. Each is
       built alone (a ``configure()`` error skips only it), runs under its own
       advisory lock ``ariel_enhance:<module>`` and through
       :func:`~osprey.services.ariel_search.enhancement.image_driver.drive_image_module`.
       The whole stage runs under ``budget + largest module timeout + 60 s``.

    Args:
        config_dict: Raw ``ariel`` config block.
        budget_s: Seconds the pass may spend; None for no limit.
        stop_event: Checked before each entry and each picture.
        progress: Optional progress callback.

    Returns:
        The text stage's result, with the picture entries walked added to
        ``entries_processed`` and the picture modules to ``module_names``.
    """
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")
    started = time.monotonic()
    config = _ariel_config(config_dict)
    text = await _run_text_enhance(config, None, False, 1000, progress, stop_event=stop_event)
    names = _catchup_module_names(config)
    if not names:
        return text
    image_budget = None if budget_s is None else max(0.0, budget_s - (time.monotonic() - started))

    modules = _build_image_modules(config, names, None)
    if not modules:
        return text

    walked = 0

    async def _image_stage() -> None:
        nonlocal walked
        walked = await _drive_image_modules(
            config, modules, budget=image_budget, stop_event=stop_event, progress=progress
        )

    if image_budget is None:
        await _image_stage()
    else:
        backstop = image_budget + max(_module_timeout(m) for m in modules)
        backstop += _CATCHUP_BACKSTOP_SLACK_S
        try:
            await asyncio.wait_for(_image_stage(), backstop)
        except TimeoutError:
            logger.warning(
                f"Picture catch-up stopped after its {backstop:.0f} s backstop; "
                "the rest waits for the next pass"
            )

    return EnhanceResult(
        entries_processed=text.entries_processed + walked,
        module_names=[*text.module_names, *(m.name for m in modules)],
        succeeded=text.succeeded,
        failed=text.failed,
        set_aside=text.set_aside,
    )


async def list_models(config_dict: dict) -> list[dict]:
    """Return embedding model info as a list of dicts."""
    from osprey.services.ariel_search import create_ariel_service

    config = _ariel_config(config_dict)
    service = await create_ariel_service(config)
    async with service:
        tables = await service.repository.get_embedding_tables()
        return [
            {
                "table_name": t.table_name,
                "entry_count": t.entry_count,
                "dimension": t.dimension,
                "is_active": t.is_active,
            }
            for t in tables
        ]


def _entry_summary(entry: dict) -> dict:
    """Compact, JSON-safe summary of a search-result entry for CLI display."""
    from datetime import datetime

    timestamp = entry.get("timestamp")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    raw_text = (entry.get("raw_text") or "").strip()
    title = raw_text.splitlines()[0][:100] if raw_text else ""
    return {
        "entry_id": entry.get("entry_id", ""),
        "timestamp": str(timestamp or ""),
        "author": entry.get("author", ""),
        "title": title,
        "score": entry.get("_score"),
    }


async def run_search(config_dict: dict, query: str, mode: str | None, limit: int) -> dict:
    """Execute a search query and return the result as a dict.

    Args:
        config_dict: Raw ``ariel`` config section.
        query: Search query text.
        mode: Search module name, e.g. ``"keyword"``. Case and surrounding
            whitespace are normalized; whether the name is registered and
            enabled is decided by the search service. ``None`` leaves the
            choice to the deployment's ``ariel.default_search_mode``.
        limit: Maximum number of entries to return.

    Returns:
        Result dict, or ``{"error": ...}`` when the search could not run.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.models import normalize_search_mode

    if not config_dict:
        return {"error": "ARIEL not configured"}

    config = _ariel_config(config_dict)

    search_mode: str | None = None
    if mode is not None:
        try:
            search_mode = normalize_search_mode(mode)
        except ValueError as e:
            return {"error": str(e)}

    try:
        service = await create_ariel_service(config)
        async with service:
            result = await service.search(
                query=query,
                max_results=limit,
                mode=search_mode,
            )

            # Hybrid may add picture-only entries beyond ``limit``; the CLI shows
            # ``limit`` entries and only the sources of the entries it shows.
            entries = list(result.entries)[:limit]
            sources = list(result.sources)
            if len(result.entries) > limit:
                shown = {e.get("entry_id") for e in entries}
                sources = [s for s in sources if s in shown]
            return {
                "query": query,
                "answer": result.answer,
                "sources": sources,
                "search_modes": list(result.search_modes_used),
                "reasoning": result.reasoning,
                "entries": [_entry_summary(e) for e in entries],
            }
    except Exception as e:
        msg = str(e)
        if "connection" in msg.lower() or "connect" in msg.lower():
            return {
                "error": "Cannot connect to the ARIEL database. "
                "Make sure the database is running: osprey up"
            }
        if "relation" in msg and "does not exist" in msg:
            return {
                "error": "Logbook database tables not found. "
                "Run 'osprey ariel migrate' to create tables, then "
                "'osprey ariel ingest' to populate data."
            }
        return {"error": msg}


def _embedding_input_limit(config: ARIELConfig, model: str) -> int:
    """Return the input limit, in tokens, the text embedding module states for ``model``.

    A model not listed under ``text_embedding.models`` gets the module's default limit.
    """
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
        DEFAULT_MAX_INPUT_TOKENS,
        max_input_tokens,
    )

    module_config = config.enhancement_modules.get("text_embedding")
    for m in (module_config.models if module_config else None) or []:
        if m.name == model:
            return max_input_tokens({"name": m.name, "max_input_tokens": m.max_input_tokens})
    return DEFAULT_MAX_INPUT_TOKENS


def _reembed_provider(config: ARIELConfig) -> tuple[type[BaseProvider], dict[str, Any]]:
    """Resolve the provider ``run_reembed`` embeds with, as the text_embedding module does.

    The provider comes from ``enhancement_modules.text_embedding`` (its own
    ``provider``, else ``ariel.embedding.provider``); a deployment with no
    ``text_embedding`` block uses ``ariel.embedding.provider`` (default
    ``ollama``).

    Args:
        config: The loaded ``ARIELConfig``.

    Returns:
        ``(provider class, its api.providers entry)``.

    Raises:
        ValueError: If the provider is unknown or serves no embeddings; the
            message names the config key it came from.
    """
    from osprey.models.provider_registry import get_provider_registry

    module_config = config.get_enhancement_module_config("text_embedding")
    if module_config is not None:
        name = module_config.get("provider") or "ollama"
        key = module_config["provider_key"]
    else:
        name = config.embedding.provider or "ollama"
        key = "ariel.embedding.provider"

    provider_cls = get_provider_registry().get_provider(name)
    if provider_cls is None or not provider_cls.supports_embeddings():
        raise ValueError(f"{key}: {name!r} is not a provider that serves embeddings")

    try:
        from osprey.models.config import get_provider_config

        provider_config = get_provider_config(name)
    except FileNotFoundError:
        provider_config = {}
    return provider_cls, provider_config


async def run_reembed(
    config_dict: dict,
    model: str,
    dimension: int,
    batch_size: int,
    dry_run: bool,
    force: bool,
    progress: _ProgressCb = None,
) -> ReembedResult:
    """Re-embed entries with a new or existing model."""
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.database.migrations import model_to_table_name
    from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingMigration
    from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
        embedding_input,
        fit_to_input_limit,
    )

    config = _ariel_config(config_dict)
    table_name = model_to_table_name(model)
    limit = _embedding_input_limit(config, model)

    if dry_run:
        if progress:
            progress(f"DRY RUN - Would re-embed entries using model: {model}")
            progress(f"  Table: {table_name}")
            progress(f"  Dimension: {dimension}")
            progress(f"  Batch size: {batch_size}")
            progress(f"  Input limit: {limit} tokens")
            progress(f"  Force overwrite: {force}")
        return ReembedResult(processed=0, skipped=0, errors=0, dry_run=True)

    service = await create_ariel_service(config)
    async with service:
        tables = await service.repository.get_embedding_tables()
        table_exists = any(t.table_name == table_name for t in tables)

        if not table_exists:
            if progress:
                progress(f"Creating embedding table: {table_name}")
            migration = TextEmbeddingMigration([(model, dimension)])
            async with service.pool.connection() as conn:
                await migration.up(conn)
            if progress:
                progress(f"  Table created: {table_name}")

        entry_count = await service.repository.count_entries()
        if progress:
            progress(f"Found {entry_count} entries to embed")

        if entry_count == 0:
            if progress:
                progress("No entries to embed.")
            return ReembedResult(processed=0, skipped=0, errors=0, dry_run=False)

        provider_cls, provider_config = _reembed_provider(config)
        embedder = provider_cls()
        base_url = provider_cls.effective_base_url(provider_config.get("base_url"))
        api_key = provider_config.get("api_key")
        # Only a truncating provider is told the length; every other one is
        # called exactly as before, since models such as ada-002 refuse it.
        dimensions = dimension if provider_cls.truncates_to_dimensions else None

        processed = 0
        skipped = 0
        errors = 0

        has_copy_state = (await service.repository.schema_facts()).has_copy_state

        async with service.pool.connection() as conn:
            async with conn.cursor() as cur:
                rows: list[Any]
                if has_copy_state:
                    await cur.execute(
                        "SELECT entry_id, raw_text, attachment_text FROM enhanced_entries "
                        "ORDER BY entry_id"
                    )
                    rows = list(await cur.fetchall())
                else:
                    await cur.execute(
                        "SELECT entry_id, raw_text FROM enhanced_entries ORDER BY entry_id"
                    )
                    rows = [(*row, None) for row in await cur.fetchall()]

                batch_texts: list[str] = []
                batch_ids: list[str] = []

                for entry_id, raw_text, attachment_text in rows:
                    if not force:
                        await cur.execute(
                            f"SELECT 1 FROM {table_name} WHERE entry_id = %s",
                            (entry_id,),
                        )
                        if await cur.fetchone():
                            skipped += 1
                            continue

                    if attachment_text and attachment_text.strip():
                        batch_texts.append(embedding_input(raw_text or "", attachment_text, limit))
                    else:
                        batch_texts.append(fit_to_input_limit(raw_text or "", limit))
                    batch_ids.append(entry_id)

                    if len(batch_texts) >= batch_size:
                        p, e = await _embed_batch(
                            cur,
                            embedder,
                            batch_texts,
                            batch_ids,
                            model,
                            base_url,
                            table_name,
                            force,
                            progress,
                            api_key=api_key,
                            dimensions=dimensions,
                        )
                        processed += p
                        errors += e
                        batch_texts = []
                        batch_ids = []

                if batch_texts:
                    p, e = await _embed_batch(
                        cur,
                        embedder,
                        batch_texts,
                        batch_ids,
                        model,
                        base_url,
                        table_name,
                        force,
                        progress,
                        api_key=api_key,
                        dimensions=dimensions,
                    )
                    processed += p
                    errors += e

    return ReembedResult(processed=processed, skipped=skipped, errors=errors, dry_run=False)


async def _embed_batch(
    cur,
    embedder,
    batch_texts: list[str],
    batch_ids: list[str],
    model: str,
    base_url: str | None,
    table_name: str,
    force: bool,
    progress: _ProgressCb,
    *,
    api_key: str | None = None,
    dimensions: int | None = None,
) -> tuple[int, int]:
    """Embed a batch of texts and upsert into the table. Returns (processed, errors).

    ``dimensions`` is passed to the provider only when given, so a provider that
    does not cut vectors to the table's length is called without the key.
    """
    try:
        embed_kwargs: dict[str, Any] = {}
        if api_key is not None:
            embed_kwargs["api_key"] = api_key
        if dimensions is not None:
            embed_kwargs["dimensions"] = dimensions
        embeddings = embedder.execute_embedding(
            texts=batch_texts,
            model_id=model,
            base_url=base_url,
            **embed_kwargs,
        )

        conflict_clause = (
            "ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            if force
            else "ON CONFLICT (entry_id) DO NOTHING"
        )
        for eid, emb in zip(batch_ids, embeddings, strict=True):
            await cur.execute(
                f"""
                INSERT INTO {table_name} (entry_id, embedding)
                VALUES (%s, %s)
                {conflict_clause}
                """,
                (eid, emb),
            )
        if progress:
            # Use cumulative count -- caller tracks total
            progress(f"  Processed {len(batch_ids)} entries in batch...")
        return len(batch_ids), 0
    except Exception as e:
        if progress:
            progress(f"  Error in batch: {e}")
        return 0, len(batch_ids)


async def run_quickstart(
    config_dict: dict,
    source: str | None,
    progress: _ProgressCb = None,
) -> QuickstartResult:
    """Run the complete ARIEL quickstart sequence."""
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.ingestion import get_adapter
    from osprey.services.ariel_search.ingestion.ingest import ingest_one
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")

    if source:
        if "ingestion" not in config_dict:
            config_dict["ingestion"] = {}
        config_dict["ingestion"]["source_url"] = source
        config_dict["ingestion"]["adapter"] = "generic_json"

    config = _ariel_config(config_dict)

    if progress:
        progress("Checking database connection...")

    pool = await create_connection_pool(config.database)

    if progress:
        progress("  Database: connected")

    count = 0
    enhanced_count = 0
    failed_count = 0
    migrations_applied = 0

    try:
        if progress:
            progress("Running migrations...")

        applied = await run_migrations(pool, config)
        migrations_applied = len(applied) if applied else 0

        if applied and progress:
            progress(f"  Tables: created ({migrations_applied} migrations applied)")
        elif progress:
            progress("  Tables: already up to date")

        if (not config.ingestion or not config.ingestion.source_url) and config_dict.get(
            "demo_narrative"
        ):
            count, enhanced_count = await _quickstart_narrative(config_dict, progress)
        elif not config.ingestion or not config.ingestion.source_url:
            if progress:
                progress("\nNo ingestion source configured. Skipping data ingestion.")
        else:
            if progress:
                progress(f"Ingesting data from: {config.ingestion.source_url}")
            adapter_instance = get_adapter(config)

            # The default stage is inline: a catch-up module never runs here.
            enhancers = create_enhancers_from_config(config)
            if enhancers and progress:
                progress(f"  Enhancement modules: {[e.name for e in enhancers]}")

            service = await create_ariel_service(config)
            async with service:
                async with _ingest_copy_run(
                    service.repository, adapter_instance, config
                ) as copy_run:
                    async for entry in adapter_instance.fetch_entries():
                        outcome = await ingest_one(
                            entry, adapter_instance, service.repository, enhancers, config, copy_run
                        )
                        count += 1
                        enhanced_count += outcome.enhanced
                        failed_count += _entry_failures(outcome)
                        if outcome.enhancer_failed:
                            logger.debug(
                                f"Enhancement failed for {entry['entry_id']}: "
                                f"{outcome.enhancer_failed} module(s)"
                            )

                if progress:
                    progress(f"  Entries: {count} ingested")
                    if adapter_instance.unreadable_entries:
                        progress(
                            f"  Skipped: {adapter_instance.unreadable_entries} entries"
                            " that could not be read"
                        )
                    if enhancers:
                        msg = f"  Enhancements: {enhanced_count} applied"
                        if failed_count:
                            msg += f", {failed_count} failed"
                        progress(msg)

        enabled_search = config.get_enabled_search_modules()

        if progress:
            progress(
                f"\nARIEL quickstart complete!"
                f"\n  Search modules: {', '.join(enabled_search) or 'none'}"
            )
            progress('\nTry it: osprey ariel search "What happened with the RF cavity?"')

    finally:
        await pool.close()

    return QuickstartResult(
        count=count,
        enhanced_count=enhanced_count,
        failed_count=failed_count,
        migrations_applied=migrations_applied,
        enabled_search=enabled_search,
    )


async def _quickstart_narrative(config_dict: dict, progress: _ProgressCb) -> tuple[int, int]:
    """Seed ``ariel.demo_narrative`` into an empty logbook, then enhance what it holds.

    The narrative is written the way a deploy writes it -- rows and pictures,
    no enhancement (see :func:`~osprey.simulation.apply.seed_narrative_if_empty`)
    -- so a quickstart after an ``osprey up`` that already seeded it finds the
    entries in place and adds none. The enhancement pass that follows is what
    the quickstart adds over the deploy: embeddings for semantic and hybrid
    search, and captions where a vision model answers.

    Returns:
        ``(entries seeded, entries the enhancement pass processed)``.
    """
    from datetime import datetime

    from osprey.simulation.apply import demo_narrative_logbook, seed_narrative_if_empty
    from osprey.utils.config import get_facility_timezone

    logbook = demo_narrative_logbook(config_dict)
    if progress:
        progress(f"Seeding the demo narrative from: {config_dict['demo_narrative']}")
    seeded = await seed_narrative_if_empty(
        config_dict, logbook, datetime.now(get_facility_timezone())
    )
    if progress:
        if seeded:
            progress(f"  Entries: {seeded} seeded")
        else:
            progress("  Entries: the logbook already holds entries; none added")
    enhanced = await run_enhance(config_dict, None, False, max(len(logbook), 1), progress)
    return seeded, enhanced.entries_processed


async def get_purge_info(config_dict: dict) -> PurgeInfo:
    """Get current counts for purge confirmation display."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.repository import image_embedding_table_names

    config = _ariel_config(config_dict)
    pool = await create_connection_pool(config.database)

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM enhanced_entries")
                row = await cur.fetchone()
                entry_count = row[0] if row else 0

                await cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                """)
                embedding_tables = [r[0] for r in await cur.fetchall()]
                image_tables = await image_embedding_table_names(cur)
    finally:
        await pool.close()

    return PurgeInfo(
        entry_count=entry_count,
        embedding_tables=embedding_tables,
        image_embedding_tables=image_tables,
    )


async def _drop_image_embedding_tables(cur, progress: _ProgressCb = None) -> list[str]:
    """Drop every image-embedding table and forget which entries were embedded.

    The image migration counts as applied while its table exists, so the next
    ``osprey ariel migrate`` recreates the dropped table; removing the
    ``image_embedding`` key from every ``enhancement_status`` lets the next
    catch-up pass embed the pictures again.

    Returns:
        The dropped table names.
    """
    from osprey.services.ariel_search.database.repository import image_embedding_table_names

    tables = await image_embedding_table_names(cur)
    for table in tables:
        await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        if progress:
            progress(f"  Dropped {table}")
    if tables:
        await cur.execute(
            """
            UPDATE enhanced_entries
            SET enhancement_status = enhancement_status - 'image_embedding'
            WHERE enhancement_status ? 'image_embedding'
            """
        )
    return tables


async def _unrecord_embedding_migration(cur) -> None:
    """Remove the text_embedding row from the migration bookkeeping table.

    Purging drops the migration-owned ``text_embeddings_*`` tables; leaving the
    migration recorded as applied would make a subsequent ``osprey ariel
    migrate`` a silent no-op, so the tables would never be recreated.
    """
    await cur.execute(
        """
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables
                       WHERE table_schema = 'public' AND table_name = 'ariel_migrations') THEN
                DELETE FROM ariel_migrations WHERE name = 'text_embedding';
            END IF;
        END $$
        """
    )


async def execute_purge(config_dict: dict, embeddings_only: bool, progress: _ProgressCb = None):
    """Execute the actual purge operation."""
    from osprey.services.ariel_search.database.connection import create_connection_pool

    config = _ariel_config(config_dict)
    pool = await create_connection_pool(config.database)

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if embeddings_only:
                    await cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                    """)
                    embedding_tables = [r[0] for r in await cur.fetchall()]
                    for table in embedding_tables:
                        await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                        if progress:
                            progress(f"  Dropped {table}")
                    await _unrecord_embedding_migration(cur)
                    await _drop_image_embedding_tables(cur, progress)
                    if progress:
                        progress("\n✓ Embedding tables purged. Entries preserved.")
                else:
                    await cur.execute("TRUNCATE enhanced_entries CASCADE")
                    await cur.execute("TRUNCATE ingestion_runs CASCADE")
                    await cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                    """)
                    embedding_tables = [r[0] for r in await cur.fetchall()]
                    for table in embedding_tables:
                        await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    await _unrecord_embedding_migration(cur)
                    await _drop_image_embedding_tables(cur)
                    if progress:
                        progress("\n✓ All ARIEL data purged.")
    finally:
        await pool.close()


async def logbook_entry_count(config_dict: dict) -> int:
    """How many entries the logbook currently holds.

    The question a caller has to answer before writing anything it did not
    author: an empty logbook is one nothing is lost by seeding, and a non-empty
    one is history — an operator's own entries, or a narrative already seeded and
    since edited — that no automated step may overwrite. Migrations must already
    have run (call :func:`run_migrate` first).

    Args:
        config_dict: ARIEL config dict (``ARIELConfig.from_dict`` shape).

    Returns:
        The total number of entries.
    """
    from osprey.services.ariel_search import create_ariel_service

    config = _ariel_config(config_dict)
    service = await create_ariel_service(config)
    async with service:
        return await service.repository.count_entries()


async def seed_logbook_entries(
    config_dict: dict,
    entries: list[EnhancedLogbookEntry],
    progress: _ProgressCb = None,
    *,
    pictures: Mapping[str, Sequence[Path]] | None = None,
) -> int:
    """Bulk-upsert pre-built logbook entries into the ARIEL database.

    A thin seeder for deterministic, locally-authored entries (e.g. simulation
    scenario bundles): unlike :func:`run_ingest` it skips the adapter fetch and
    the enhancement passes and just upserts the given entries inside one
    ingestion run. Keyword search (Postgres trigram/FTS) needs no embeddings, so
    semantic enrichment is left to an optional follow-up. Migrations must
    already have run (call :func:`run_migrate` first).

    Pictures are stored the way a natively written entry stores them
    (:func:`~osprey.services.ariel_search.attachments.store_native_attachment`),
    so each is copied with its viewable rendition and linked on the entry the
    moment seeding returns. No caption or picture-embedding module runs here.

    Args:
        config_dict: ARIEL config dict (``ARIELConfig.from_dict`` shape).
        entries: Fully-built :class:`EnhancedLogbookEntry` records to upsert.
        progress: Optional progress callback.
        pictures: Picture files to attach, keyed by entry id.

    Returns:
        The number of entries seeded.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.attachments import (
        guess_mime_type,
        store_native_attachment,
    )

    config = _ariel_config(config_dict)
    service = await create_ariel_service(config)
    pictures = pictures or {}
    count = 0
    async with service:
        run_id = await service.repository.start_ingestion_run("Simulation")
        try:
            for entry in entries:
                await service.repository.upsert_entry(entry)
                files = pictures.get(entry["entry_id"], ())
                if files:
                    infos = [
                        await store_native_attachment(
                            service.repository,
                            entry["entry_id"],
                            filename=path.name,
                            declared_mime=guess_mime_type(path.name),
                            data=path.read_bytes(),
                        )
                        for path in files
                    ]
                    await service.repository.upsert_entry({**entry, "attachments": infos})
                count += 1
                if count % 100 == 0 and progress:
                    progress(f"  Seeded {count} entries...")
            await service.repository.complete_ingestion_run(
                run_id, entries_added=count, entries_updated=0, entries_failed=0
            )
        except Exception as exc:
            await service.repository.fail_ingestion_run(run_id, str(exc))
            raise
    if progress:
        progress(f"✓ Seeded {count} logbook entries.")
    return count


# ---------------------------------------------------------------------------
# osprey ariel attachments backfill
# ---------------------------------------------------------------------------

#: Entries one backfill page locks, records and then copies.
BACKFILL_PAGE_SIZE = 500

#: Host label for a relative path read from a file source.
_FILE_HOST = "(file)"

#: Host label for a url that names no host.
_NO_HOST = "(none)"

#: Type label for an item or row that declares no usable type.
_NO_TYPE = "(none)"


@dataclass
class BackfillDryRun:
    """What a backfill would do, counted with no network and no write.

    Every counter maps ``(declared type, host)`` to a count. The census
    counters say what the store holds; the plan counters say what a backfill
    would do with it.

    Attributes:
        no_row: JSONB items that are fetchable but have no ``attachment_files`` row.
        pending: Rows still ``pending``.
        skipped: Config- and source-skipped rows, by skip code.
        would_fetch: Items and rows a backfill would fetch.
        per_entry_limit: Items and rows a backfill would leave ``per_entry_limit``
            because their entry's copy budget is spent; never in ``would_fetch``.
        still_skipped: Items and rows the current configuration still skips, by code.
        not_fetchable: JSONB items whose url can never be fetched from this source
            (a relative path on an http source, a ``..`` segment); never in
            ``would_fetch``.
        would_render: Copied rows holding their bytes but no rendition yet.
        entries: Entries examined.
        candidates: The urls behind ``would_fetch``, in visiting order.
    """

    no_row: dict[tuple[str, str], int] = field(default_factory=dict)
    pending: dict[tuple[str, str], int] = field(default_factory=dict)
    skipped: dict[str, dict[tuple[str, str], int]] = field(default_factory=dict)
    would_fetch: dict[tuple[str, str], int] = field(default_factory=dict)
    per_entry_limit: dict[tuple[str, str], int] = field(default_factory=dict)
    still_skipped: dict[str, dict[tuple[str, str], int]] = field(default_factory=dict)
    not_fetchable: dict[tuple[str, str], int] = field(default_factory=dict)
    would_render: dict[tuple[str, str], int] = field(default_factory=dict)
    entries: int = 0
    candidates: list[str] = field(default_factory=list)


@dataclass
class BackfillProbe:
    """An extrapolated estimate from ``HEAD`` requests to a random sample.

    Attributes:
        sampled: ``HEAD`` requests sent.
        reachable: Sampled urls the source answered as fetchable.
        outcomes: Sampled outcomes by kind (``ok``, ``transient`` or a skip code).
        estimate: ``reachable / sampled`` of the would-fetch total, rounded; an
            estimate, never a count.
    """

    sampled: int
    reachable: int
    outcomes: dict[str, int]
    estimate: int


@dataclass
class BackfillResult:
    """Outcome of ``osprey ariel attachments backfill``.

    Attributes:
        status: ``done`` after a backfill or dry run; ``locked`` when another
            process holds the copy lock (nothing was done); ``no_copy_state``
            when the schema predates the copy state (run ``osprey ariel migrate``
            first).
        dry_run: Whether nothing was written.
        entries: Entries whose rows were recorded and whose pictures were copied.
        record_failed: Entries whose record step failed; their pictures were not copied.
        copy_failed: Entries whose copy step raised.
        fetches: Fetch calls made.
        copied: Rows written ``copied``.
        rendered: Render-only rows given a rendition or a content skip.
        pending: Fetch candidates left ``pending``.
        skipped: Rows written ``skipped``, by code.
        decoder_reset: ``decoder_failed`` rows cleared for a re-render
            (``--retry-decoder-failed``).
        plan: The dry-run counts; set for a dry run.
        probe: The probe estimate; set when ``--probe`` ran.
        proxy: The adapter's proxy, redacted, or ``None`` for a direct connection.
        ca_bundle: The CA bundle fetches verify against, or ``None`` for the
            image trust store.
    """

    status: str
    dry_run: bool = False
    entries: int = 0
    record_failed: int = 0
    copy_failed: int = 0
    fetches: int = 0
    copied: int = 0
    rendered: int = 0
    pending: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    decoder_reset: int = 0
    plan: BackfillDryRun | None = None
    probe: BackfillProbe | None = None
    proxy: str | None = None
    ca_bundle: str | None = None


def backfill_runtime_name(config: Any) -> str:
    """Return the container runtime named in the backfill hint, without probing.

    ``CONTAINER_RUNTIME`` when it is ``docker`` or ``podman``, else the config's
    ``container_runtime`` when it is one of those, else ``docker``. Nothing is
    executed or looked up on ``PATH``: the hint is printed inside the ariel-sync
    container, where no runtime binary exists.

    Args:
        config: The full project config mapping (``container_runtime`` is a
            top-level key), or ``None``.
    """
    import os

    for candidate in (
        os.environ.get("CONTAINER_RUNTIME"),
        config.get("container_runtime") if isinstance(config, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip().lower() in ("docker", "podman"):
            return candidate.strip().lower()
    return "docker"


def backfill_exec_line(config: Any, args: Sequence[str] = ()) -> str:
    """Return the canonical invocation that runs backfill in the ariel-sync container.

    ``<runtime> exec <project_name>-ariel-sync osprey ariel attachments backfill
    <args>``, with no ``-it`` so it pastes into cron and scripts.

    Args:
        config: The full project config mapping (``container_runtime``,
            ``project_name``, ``project_root``), or ``None``.
        args: The backfill options to repeat after the command.
    """
    import shlex

    from osprey.deployment.compose_generator import resolve_project_name

    project = resolve_project_name(config if isinstance(config, dict) else {})
    words = [
        backfill_runtime_name(config),
        "exec",
        f"{project}-ariel-sync",
        "osprey",
        "ariel",
        "attachments",
        "backfill",
        *args,
    ]
    return " ".join(shlex.quote(w) for w in words)


def _bump(counter: dict, key: Any, by: int = 1) -> None:
    counter[key] = counter.get(key, 0) + by


def _type_label(declared: object) -> str:
    from osprey.services.ariel_search.attachments.copy import validated_declared_type

    return validated_declared_type(declared) or _NO_TYPE


def _host_label(url: object) -> str:
    from osprey.services.ariel_search.attachments.fetch import origin_of

    origin = origin_of(url if isinstance(url, str) else None)
    if origin is not None:
        return origin[1]
    if isinstance(url, str) and url and "://" not in url and not url.startswith(("/", "\\")):
        return _FILE_HOST
    return _NO_HOST


async def _backfill_page(
    repository: ARIELRepository,
    cursor: tuple[Any, str] | None,
    limit: int,
    *,
    conn: Any = None,
    lock: bool,
) -> list[dict[str, Any]]:
    """Return one page of entries holding attachments, newest first below ``cursor``.

    Args:
        repository: The repository whose pool serves the read when ``conn`` is None.
        cursor: ``(timestamp, entry_id)`` of the last entry already visited.
        limit: Most entries returned.
        conn: Connection inside the page transaction; required with ``lock``.
        lock: Take ``FOR UPDATE`` on every returned entry row.

    Returns:
        Dict rows with ``entry_id``, ``timestamp``, ``attachments``,
        ``attachment_text`` and ``attachment_captions``.
    """
    from psycopg.rows import dict_row

    params: dict[str, Any] = {"limit": limit}
    cursor_sql = ""
    if cursor is not None:
        cursor_sql = "AND (timestamp, entry_id) < (%(after_ts)s, %(after_id)s)"
        params["after_ts"], params["after_id"] = cursor
    sql = f"""
        SELECT entry_id, timestamp, attachments, attachment_text, attachment_captions
        FROM enhanced_entries
        WHERE attachments <> '[]'::jsonb
        {cursor_sql}
        ORDER BY timestamp DESC, entry_id DESC
        LIMIT %(limit)s
        {"FOR UPDATE" if lock else ""}
    """

    async def _read(c: Any) -> list[dict[str, Any]]:
        async with c.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    if conn is not None:
        return await _read(conn)
    async with repository.pool.connection() as c:
        return await _read(c)


async def _plan_entry(
    repository: ARIELRepository,
    row: dict[str, Any],
    config: ARIELConfig,
    adapter: FacilityAdapter,
    origins: frozenset[Any],
    plan: BackfillDryRun,
) -> None:
    """Count what a backfill would do with one entry, with no network and no write."""
    from osprey.services.ariel_search.attachments import (
        attachment_id_for,
        fetchable_url,
        is_native_item,
    )
    from osprey.services.ariel_search.attachments.copy import (
        BYTE_BUDGET_FILES,
        COPY_MAX_PER_ENTRY,
        _attachment_list,
        still_skipped,
    )
    from osprey.services.ariel_search.attachments.fetch import is_file_source
    from osprey.services.ariel_search.attachments.formats import (
        CONFIG_SKIP_REASONS,
        SOURCE_SKIP_REASONS,
    )

    entry_id = row["entry_id"]
    file_source = is_file_source(adapter)
    stored = {r["attachment_id"]: r for r in await repository.get_copy_rows(entry_id)}

    # Fetch candidates in JSONB list order, as copy_entry takes them; rows the
    # list no longer names follow.
    ordered: list[tuple[str, dict[str, Any] | None, Mapping[str, Any] | None]] = []
    seen: set[str] = set()
    for item in _attachment_list(row.get("attachments")):
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url or is_native_item(item):
            continue
        attachment_id = attachment_id_for(entry_id, item)
        if not fetchable_url(url, file_source=file_source) or attachment_id is None:
            _bump(plan.not_fetchable, (_type_label(item.get("type")), _host_label(url)))
            continue
        if attachment_id in seen:
            continue
        seen.add(attachment_id)
        ordered.append((attachment_id, stored.get(attachment_id), item))
    ordered.extend((aid, r, None) for aid, r in stored.items() if aid not in seen)

    to_fetch: list[tuple[str, tuple[str, str]]] = []
    for _aid, stored_row, item in ordered:
        if stored_row is None:
            assert item is not None
            url = item["url"]
            key = (_type_label(item.get("type")), _host_label(url))
            _bump(plan.no_row, key)
            decided: dict[str, Any] = {
                "source_url": url,
                "mime_type": item.get("type"),
                "size_bytes": None,
            }
        else:
            url = stored_row.get("source_url")
            key = (_type_label(stored_row.get("mime_type")), _host_label(url))
            status, reason = stored_row.get("copy_status"), stored_row.get("skip_reason")
            if (
                status == "copied"
                and stored_row.get("has_data")
                and stored_row.get("rendition_sha256") is None
                and reason is None
            ):
                _bump(plan.would_render, key)
                continue
            if status == "pending":
                _bump(plan.pending, key)
            elif status == "skipped" and reason in (CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS):
                _bump(plan.skipped.setdefault(reason, {}), key)
            else:
                continue
            decided = dict(stored_row)
            if reason in SOURCE_SKIP_REASONS:
                decided["mime_type"] = None
        code = still_skipped(decided, config, origins, file_source=file_source)
        if code is not None:
            _bump(plan.still_skipped.setdefault(code, {}), key)
        else:
            to_fetch.append((str(url), key))

    if not to_fetch:
        return
    cap = config.attachments.max_file_mb * 1024 * 1024
    count, used = await repository.count_copied_attachments(entry_id)
    slots = max(0, COPY_MAX_PER_ENTRY - count) if BYTE_BUDGET_FILES * cap - used > 0 else 0
    for url, key in to_fetch[:slots]:
        _bump(plan.would_fetch, key)
        plan.candidates.append(url)
    for _url, key in to_fetch[slots:]:
        _bump(plan.per_entry_limit, key)


async def _probe_sample(
    plan: BackfillDryRun,
    n: int,
    config: ARIELConfig,
    adapter: FacilityAdapter,
    origins: frozenset[Any],
) -> BackfillProbe:
    """``HEAD`` a random sample of the would-fetch urls inside the origin set."""
    import random

    from osprey.services.ariel_search.attachments import copy as copy_mod
    from osprey.services.ariel_search.attachments.fetch import origin_of

    inside = [u for u in dict.fromkeys(plan.candidates) if origin_of(u) in origins]
    sample = random.sample(inside, min(n, len(inside)))
    cap = config.attachments.max_file_mb * 1024 * 1024
    outcomes: dict[str, int] = {}
    reachable = 0
    async with copy_mod.CopyRun(adapter, origins) as run:
        for url in sample:
            outcome = await copy_mod.fetch_attachment_bytes(
                url, cap, origins, adapter, "HEAD", semaphore=run.semaphore, session=run.session
            )
            if outcome.ok:
                reachable += 1
                kind = "ok"
            else:
                kind = outcome.code or "transient"
            _bump(outcomes, kind)
    total = sum(plan.would_fetch.values())
    estimate = round(total * reachable / len(sample)) if sample else 0
    return BackfillProbe(
        sampled=len(sample), reachable=reachable, outcomes=outcomes, estimate=estimate
    )


async def _backfill_dry_run(
    repository: ARIELRepository,
    config: ARIELConfig,
    adapter: FacilityAdapter,
    limit: int | None,
) -> BackfillDryRun:
    """Walk the store newest first and count what a backfill would do."""
    from osprey.services.ariel_search.attachments.fetch import origins_for

    origins = origins_for(adapter, config)
    plan = BackfillDryRun()
    cursor: tuple[Any, str] | None = None
    while True:
        page_limit = (
            BACKFILL_PAGE_SIZE if limit is None else min(BACKFILL_PAGE_SIZE, limit - plan.entries)
        )
        if page_limit <= 0:
            break
        rows = await _backfill_page(repository, cursor, page_limit, lock=False)
        for row in rows:
            await _plan_entry(repository, row, config, adapter, origins, plan)
        plan.entries += len(rows)
        if len(rows) < page_limit:
            break
        cursor = (rows[-1]["timestamp"], rows[-1]["entry_id"])
    return plan


async def _reset_decoder_failed(repository: ARIELRepository, entry_id: str) -> int:
    """Clear the copy-side reason and rendition of an entry's ``decoder_failed`` rows.

    One transaction, entry row locked first. Such a row keeps its stored
    original and has never had a rendition, so it has no image-table row and
    no caption to delete; the render-only pass of ``copy_entry`` then renders
    it again, and a written rendition clears the image-module status keys.

    Args:
        repository: The repository.
        entry_id: The entry whose rows are reset.

    Returns:
        The number of rows reset.
    """
    from osprey.services.ariel_search.database.repository import ARIELRepository

    async with repository.pool.connection() as conn, conn.transaction():
        if not await ARIELRepository.lock_entry(conn, entry_id):
            return 0
        cur = await conn.execute(
            """
            UPDATE attachment_files
            SET skip_reason = NULL,
                rendition_bytes = NULL,
                rendition_mime = NULL,
                rendition_w = NULL,
                rendition_h = NULL,
                rendition_sha256 = NULL
            WHERE entry_id = %(entry_id)s
              AND copy_status = 'copied'
              AND skip_reason = 'decoder_failed'
            """,
            {"entry_id": entry_id},
        )
        return max(cur.rowcount, 0)


async def backfill_store(
    repository: ARIELRepository,
    adapter: FacilityAdapter,
    config: ARIELConfig,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    probe: int | None = None,
    wait: bool = False,
    retry_decoder_failed: bool = False,
    progress: _ProgressCb = None,
    lock_factory: Any = None,
) -> BackfillResult:
    """Record and copy the pictures of every stored entry, newest first.

    Each page of :data:`BACKFILL_PAGE_SIZE` entries is one transaction: the
    page SELECT locks the entry rows (``FOR UPDATE``) and returns their
    ``attachments``, ``attachment_text`` and ``attachment_captions``;
    ``record_and_compose`` runs on each locked row in its own savepoint; the
    page COMMITs. Only then does ``copy_entry(…, retry_skipped=True)`` run on
    each entry, outside any transaction, because its per-row writes take the
    entry lock the page transaction would still hold. The whole run holds the
    ``ariel_copy`` advisory lock; held elsewhere, nothing is done (``--wait``
    waits for it instead).

    With ``retry_decoder_failed``, each entry's ``decoder_failed`` rows are
    reset in their own transaction just before its ``copy_entry``, which then
    renders the stored originals again.

    A dry run (and a ``probe``, which implies one) writes nothing and takes no
    lock; only the probe touches the network, with ``HEAD`` requests.

    Args:
        repository: The repository.
        adapter: The ingestion adapter whose source the attachments come from.
        config: The ARIEL config.
        limit: Most entries visited.
        dry_run: Count what would be done instead of doing it.
        probe: ``HEAD`` this many randomly chosen would-fetch urls inside the
            origin set and extrapolate an estimate.
        wait: Wait for the copy lock instead of returning ``locked``.
        retry_decoder_failed: Reset ``decoder_failed`` rows so they render again.
        progress: Optional callback for human-readable progress lines.
        lock_factory: The advisory-lock context manager factory; defaults to
            ``try_advisory_lock``.

    Returns:
        The :class:`BackfillResult`.
    """
    from osprey.services.ariel_search.attachments import copy as copy_mod
    from osprey.services.ariel_search.attachments.fetch import origins_for, redact_url
    from osprey.services.ariel_search.ingestion.scheduler import COPY_LOCK_KEY
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")
    result = BackfillResult(
        status="done",
        dry_run=dry_run or probe is not None,
        proxy=redact_url(adapter.proxy_url) if adapter.proxy_url else None,
        ca_bundle=adapter.ca_bundle,
    )
    if not (await repository.schema_facts()).has_copy_state:
        result.status = "no_copy_state"
        return result

    if result.dry_run:
        result.plan = await _backfill_dry_run(repository, config, adapter, limit)
        if probe is not None and probe > 0:
            result.probe = await _probe_sample(
                result.plan, probe, config, adapter, origins_for(adapter, config)
            )
        return result

    if lock_factory is None:
        from osprey.services.ariel_search.database.connection import try_advisory_lock

        lock_factory = try_advisory_lock

    async with lock_factory(repository.pool.conninfo, COPY_LOCK_KEY, wait=wait) as held:
        if not held:
            result.status = "locked"
            return result
        async with copy_mod.CopyRun(adapter, origins_for(adapter, config)) as copy_run:
            cursor: tuple[Any, str] | None = None
            visited = 0
            while True:
                page_limit = (
                    BACKFILL_PAGE_SIZE
                    if limit is None
                    else min(BACKFILL_PAGE_SIZE, limit - visited)
                )
                if page_limit <= 0:
                    break
                recorded: list[str] = []
                async with repository.pool.connection() as conn, conn.transaction():
                    rows = await _backfill_page(
                        repository, cursor, page_limit, conn=conn, lock=True
                    )
                    for row in rows:
                        try:
                            async with conn.transaction():
                                await copy_mod.record_and_compose(
                                    conn, row["entry_id"], row, config, adapter
                                )
                            recorded.append(row["entry_id"])
                        except Exception as exc:
                            result.record_failed += 1
                            logger.warning(
                                "%s: attachments not recorded by backfill (%s)",
                                row["entry_id"],
                                exc,
                            )
                for entry_id in recorded:
                    try:
                        if retry_decoder_failed:
                            result.decoder_reset += await _reset_decoder_failed(
                                repository, entry_id
                            )
                        report = await copy_mod.copy_entry(
                            repository, entry_id, config, copy_run, retry_skipped=True
                        )
                    except Exception as exc:
                        result.copy_failed += 1
                        logger.warning("%s: backfill copy failed (%s)", entry_id, exc)
                        continue
                    result.fetches += report.fetches
                    result.copied += report.copied
                    result.rendered += report.rendered
                    result.pending += report.pending
                    for code, n in report.skipped.items():
                        _bump(result.skipped, code, n)
                result.entries += len(recorded)
                visited += len(rows)
                if progress and rows:
                    progress(f"  Backfilled {visited} entries...")
                if len(rows) < page_limit:
                    break
                cursor = (rows[-1]["timestamp"], rows[-1]["entry_id"])
    return result


async def run_backfill(
    config_dict: dict,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    probe: int | None = None,
    wait: bool = False,
    retry_decoder_failed: bool = False,
    progress: _ProgressCb = None,
) -> BackfillResult:
    """Run ``osprey ariel attachments backfill`` against the configured store.

    Args:
        config_dict: The raw ``ariel`` section.
        limit: Most entries visited.
        dry_run: Count what would be done instead of doing it.
        probe: ``HEAD`` this many sampled urls and print an estimate (implies
            a dry run).
        wait: Wait for the copy lock held by another process.
        retry_decoder_failed: Reset ``decoder_failed`` rows so they render again.
        progress: Optional callback for human-readable progress lines.

    Returns:
        The :class:`BackfillResult`.
    """
    from osprey.services.ariel_search import create_ariel_service
    from osprey.services.ariel_search.ingestion import get_adapter

    config = _ariel_config(config_dict)
    adapter = get_adapter(config)
    service = await create_ariel_service(config)
    async with service:
        return await backfill_store(
            service.repository,
            adapter,
            config,
            limit=limit,
            dry_run=dry_run,
            probe=probe,
            wait=wait,
            retry_decoder_failed=retry_decoder_failed,
            progress=progress,
        )
