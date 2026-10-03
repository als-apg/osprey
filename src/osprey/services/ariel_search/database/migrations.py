"""ARIEL database migrations.

This module provides the migration base class, utilities, runner, and
convenience function for managing ARIEL database schema changes.

Components:
    - BaseMigration: Abstract base class for individual migrations
    - MigrationSkippedError: Raised when prerequisites are missing
    - MigrationBusyError: A skip because the tables are in use; retry later
    - acquire_nonqueueing_lock(): Table lock that never queues behind readers
    - model_to_table_name(): Converts model names to table names
    - image_table_name(): Image-embedding table name for a model and width
    - image_index_name(): HNSW index name of an image-embedding table
    - image_embedding_target(): Model, width and table of the image block
    - KNOWN_MIGRATIONS: Registry of all known migration classes
    - MIGRATION_ARGS: Constructor arguments of the migrations not built bare
    - MigrationRunner: Discovers, orders, and executes migrations
    - run_migrations(): Convenience function for the runner
    - run_migrations_detailed(): Same, reporting busy skips and the lock state
"""

import asyncio
import hashlib
import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from osprey.services.ariel_search.exceptions import ConfigurationError, ModuleConfigError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")


# ---------------------------------------------------------------------------
# Base class & utilities
# ---------------------------------------------------------------------------


class MigrationSkippedError(Exception):
    """Raised when a migration cannot run due to missing prerequisites.

    The migration runner catches this and logs a warning instead of failing.
    The migration is NOT marked as applied so it retries when prerequisites
    are later installed (e.g., pgvector extension).
    """


class MigrationBusyError(MigrationSkippedError):
    """Raised when a migration cannot take the table lock it needs right now.

    A skip like any other -- the migration stays unmarked and its dependents
    wait -- but one that is expected to clear on its own, so the runner reports
    it separately for a caller to retry.
    """


#: Name of the advisory lock that serializes migration runs. The runner is the
#: only place it is taken; a caller that held it around a runner call would
#: make a second session's runner wait on the first forever.
MIGRATION_LOCK_KEY = "ariel_migrate"

#: The table-lock modes PostgreSQL accepts in ``LOCK TABLE ... IN <mode> MODE``.
TABLE_LOCK_MODES = frozenset(
    {
        "ACCESS SHARE",
        "ROW SHARE",
        "ROW EXCLUSIVE",
        "SHARE UPDATE EXCLUSIVE",
        "SHARE",
        "SHARE ROW EXCLUSIVE",
        "EXCLUSIVE",
        "ACCESS EXCLUSIVE",
    }
)

#: Attempts and pause of :func:`acquire_nonqueueing_lock` before it gives up:
#: about five seconds of retrying, long enough for short reads to drain.
LOCK_ATTEMPTS = 10
LOCK_RETRY_DELAY_S = 0.5


async def acquire_nonqueueing_lock(conn: "AsyncConnection", table: str, mode: str) -> None:
    """Take a table lock without ever queueing for it.

    A plain ``LOCK TABLE`` that cannot be granted waits in the lock queue, and
    every later request on the table -- including plain reads -- queues behind
    it, so one long-running reader would stall the whole service behind a
    migration. ``NOWAIT`` fails instead of queueing; this retries that a few
    times and then gives up with :class:`MigrationBusyError`, which the runner
    treats as a skip to retry later.

    Must be called inside the migration's transaction (the runner opens one
    around ``up()``): each attempt runs in a savepoint, a refused attempt rolls
    only the savepoint back, and a granted lock is kept until the migration's
    transaction ends.

    Args:
        conn: Connection inside the migration's transaction.
        table: Table to lock (a plain, unqualified name).
        mode: Lock mode, one of :data:`TABLE_LOCK_MODES`. ``ACCESS EXCLUSIVE``
            for ALTER TABLE; ``SHARE`` for an index build, which conflicts with
            no reader, so reads proceed during the build.

    Raises:
        ValueError: If *mode* is not a PostgreSQL table-lock mode.
        MigrationBusyError: If the lock was refused on every attempt.
    """
    if mode not in TABLE_LOCK_MODES:
        raise ValueError(f"unknown table lock mode: {mode!r}")

    from psycopg import errors, sql

    statement = sql.SQL("LOCK TABLE {} IN {} MODE NOWAIT").format(
        sql.Identifier(table), sql.SQL(mode)
    )
    for attempt in range(LOCK_ATTEMPTS):
        try:
            async with conn.transaction():
                await conn.execute(statement)
            return
        except errors.LockNotAvailable:
            if attempt + 1 < LOCK_ATTEMPTS:
                await asyncio.sleep(LOCK_RETRY_DELAY_S)
    raise MigrationBusyError("busy, retry")


class BaseMigration(ABC):
    """Base class for ARIEL database migrations.

    Each enhancement module that needs database schema changes extends this
    class. Migrations are discovered and executed by the MigrationRunner.

    Attributes:
        name: Migration identifier (matches module name)
        depends_on: List of migrations that must run first
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return migration identifier.

        This should match the module name (e.g., 'core_schema', 'text_embedding').
        """

    @property
    def depends_on(self) -> list[str]:
        """Return list of migration names this migration depends on.

        Override to declare dependencies. Default is empty list.
        """
        return []

    @abstractmethod
    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the migration.

        Args:
            conn: Database connection to use for the migration
        """

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the migration.

        Override to provide rollback support. Default raises NotImplementedError.

        Args:
            conn: Database connection to use for the rollback
        """
        raise NotImplementedError(f"Rollback not implemented for migration: {self.name}")

    async def is_applied(self, conn: "AsyncConnection") -> bool:
        """Check if migration has already been applied.

        Args:
            conn: Database connection to use for the check

        Returns:
            True if migration has been applied
        """
        result = await conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'ariel_migrations'
            )
            """
        )
        row = await result.fetchone()
        if not row or not row[0]:
            return False

        result = await conn.execute(
            "SELECT EXISTS (SELECT 1 FROM ariel_migrations WHERE name = %s)",
            [self.name],
        )
        row = await result.fetchone()
        return bool(row and row[0])

    async def mark_applied(self, conn: "AsyncConnection") -> None:
        """Mark this migration as applied in the tracking table.

        Args:
            conn: Database connection to use
        """
        await conn.execute(
            """
            INSERT INTO ariel_migrations (name, applied_at)
            VALUES (%s, NOW())
            ON CONFLICT (name) DO NOTHING
            """,
            [self.name],
        )

    async def mark_unapplied(self, conn: "AsyncConnection") -> None:
        """Remove this migration from the tracking table.

        Args:
            conn: Database connection to use
        """
        await conn.execute(
            "DELETE FROM ariel_migrations WHERE name = %s",
            [self.name],
        )


def model_to_table_name(model_name: str) -> str:
    """Convert model name to database table name.

    Converts model names like 'nomic-embed-text' to valid PostgreSQL
    table names like 'text_embeddings_nomic_embed_text'.

    Args:
        model_name: Model name (e.g., 'nomic-embed-text')

    Returns:
        Table name (e.g., 'text_embeddings_nomic_embed_text')
    """
    safe_name = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    safe_name = safe_name.lower()
    return f"text_embeddings_{safe_name}"


_IMAGE_TABLE_PREFIX = "image_embeddings_"
_IMAGE_MODEL_SLUG_MAX = 31
_IMAGE_MODULE_KEY = "ariel.enhancement_modules.image_embedding"
_IMAGE_DIMENSIONS_DEFAULT = 1024
_IMAGE_DIMENSIONS_MAX = 2000


def image_table_name(model: str, dims: int) -> str:
    """Return the image-embedding table name for a model and vector width.

    ``image_embeddings_<slug[:31]>_<sha1(model)[:8]>_d<dims>``: the slug keeps
    the name readable, the hash of the exact model id keeps ids that differ
    only in punctuation or case apart, and the width keeps two widths of one
    model apart. The longest name is 63 bytes, PostgreSQL's identifier limit.

    Args:
        model: Exact model id as configured.
        dims: Vector width, 1..2000.

    Returns:
        Table name.
    """
    slug = re.sub(r"[^a-z0-9_]+", "_", model.lower()).strip("_")[:_IMAGE_MODEL_SLUG_MAX]
    digest = hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]
    return f"{_IMAGE_TABLE_PREFIX}{slug}_{digest}_d{dims}"


def image_index_name(table: str) -> str:
    """Return the HNSW index name of an image-embedding table.

    Args:
        table: Table name from :func:`image_table_name`.

    Returns:
        ``idx_img_<sha1(table)[:16]>``, always within the identifier limit.
    """
    return f"idx_img_{hashlib.sha1(table.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class ImageEmbeddingTarget:
    """The model, vector width and table the image-embedding block names.

    Attributes:
        model: Exact model id.
        dims: Vector width.
        table: Table name from :func:`image_table_name`.
    """

    model: str
    dims: int
    table: str


def image_embedding_target(module_cfg: Mapping[str, Any]) -> ImageEmbeddingTarget:
    """Read the image-embedding block into its model, width and table.

    The one reader of the block: every caller resolves the table through it,
    so they all name the same table for the same configuration.

    Args:
        module_cfg: The flat dict ``get_enhancement_module_config(
            "image_embedding")`` returns, the shape ``configure()`` receives.

    Returns:
        The resolved target.

    Raises:
        ModuleConfigError: The model is missing, or the dimensions are not an
            integer in 1..2000.
    """
    model = module_cfg.get("model")
    model_key = f"{_IMAGE_MODULE_KEY}.model"
    if model is None or (isinstance(model, str) and not model.strip()):
        raise ModuleConfigError(f"{model_key} is required", key=model_key)
    if not isinstance(model, str):
        raise ModuleConfigError(f"{model_key} must be a string", key=model_key)

    dims = module_cfg.get("dimensions", _IMAGE_DIMENSIONS_DEFAULT)
    dims_key = f"{_IMAGE_MODULE_KEY}.dimensions"
    if (
        isinstance(dims, bool)
        or not isinstance(dims, int)
        or not 1 <= dims <= _IMAGE_DIMENSIONS_MAX
    ):
        raise ModuleConfigError(
            f"{dims_key} must be an integer from 1 to {_IMAGE_DIMENSIONS_MAX}, got {dims!r}",
            key=dims_key,
        )

    return ImageEmbeddingTarget(model=model, dims=dims, table=image_table_name(model, dims))


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


def _embedding_models_args(runner: "MigrationRunner") -> tuple:
    """Constructor arguments of the per-model text-embedding migrations."""
    return (runner._configured_embedding_models(),)


def _caption_model_args(runner: "MigrationRunner") -> tuple:
    """Constructor arguments of the upstream caption fold: the caption model id."""
    from osprey.services.ariel_search.attachments.compose import caption_model_id

    return (caption_model_id(runner.config),)


def _image_embedding_args(runner: "MigrationRunner") -> tuple:
    """Constructor arguments of the image-embedding migration: the configured target."""
    module_cfg = runner.config.get_enhancement_module_config("image_embedding") or {}
    return (image_embedding_target(module_cfg),)


# Constructor arguments of the migrations that are not built bare, resolved
# from the runner's config. The text-embedding migrations write per-model
# objects, so a name missing here would work the hardcoded default table
# instead of the ones the deployment configured; the fold composes text under
# the configured caption model, so it never erases that model's captions. The
# image-embedding migration creates the table its configured model and width
# name. A resolver that raises ``ValueError`` (a configuration the migration
# cannot be built from) skips that migration with one warning, as
# ``MigrationSkippedError`` does; every other migration still runs.
MIGRATION_ARGS: dict[str, Callable[["MigrationRunner"], tuple]] = {
    "text_embedding": _embedding_models_args,
    "text_embedding_hnsw_index": _embedding_models_args,
    "attachment_text_upstream_fold": _caption_model_args,
    "image_embedding": _image_embedding_args,
}

# Format: (name, module_path, class_name, requires_module)
# requires_module is None for core_schema (always runs), otherwise module name
KNOWN_MIGRATIONS: list[tuple[str, str, str, str | None]] = [
    (
        "core_schema",
        "osprey.services.ariel_search.database.core_migration",
        "CoreMigration",
        None,
    ),
    (
        "keyword_search_fts_index",
        "osprey.services.ariel_search.database.keyword_search_migration",
        "KeywordSearchFtsMigration",
        None,
    ),
    (
        "semantic_processor",
        "osprey.services.ariel_search.enhancement.semantic_processor.migration",
        "SemanticProcessorMigration",
        "semantic_processor",
    ),
    (
        "semantic_processor_search_index",
        "osprey.services.ariel_search.enhancement.semantic_processor.search_migration",
        "SemanticProcessorSearchMigration",
        "semantic_processor",
    ),
    (
        "text_embedding",
        "osprey.services.ariel_search.enhancement.text_embedding.migration",
        "TextEmbeddingMigration",
        "text_embedding",
    ),
    (
        "text_embedding_hnsw_index",
        "osprey.services.ariel_search.enhancement.text_embedding.hnsw_migration",
        "TextEmbeddingHnswIndexMigration",
        "text_embedding",
    ),
    (
        "attachment_files",
        "osprey.services.ariel_search.database.attachment_migration",
        "AttachmentMigration",
        None,  # Always runs
    ),
    (
        "attachment_text_columns",
        "osprey.services.ariel_search.database.attachment_text_migration",
        "AttachmentTextColumnsMigration",
        None,  # Always runs
    ),
    (
        "attachment_text_upstream_fold",
        "osprey.services.ariel_search.database.attachment_text_migration",
        "AttachmentTextUpstreamFoldMigration",
        None,  # Always runs
    ),
    (
        "raw_text_fts_index_v2",
        "osprey.services.ariel_search.database.attachment_text_migration",
        "RawTextFtsIndexV2Migration",
        None,  # Always runs
    ),
    (
        "semantic_processor_search_index_v2",
        "osprey.services.ariel_search.enhancement.semantic_processor.search_migration",
        "SemanticProcessorSearchIndexV2Migration",
        "semantic_processor",
    ),
    (
        "attachment_files_copy_state",
        "osprey.services.ariel_search.database.attachment_migration",
        "AttachmentFilesCopyStateMigration",
        None,  # Always runs
    ),
    (
        "image_embedding",
        "osprey.services.ariel_search.enhancement.image_embedding.migration",
        "ImageEmbeddingMigration",
        "image_embedding",
    ),
    (
        "qmd_resync_index",
        "osprey.services.ariel_search.database.qmd_resync_migration",
        "QmdResyncMigration",
        "qmd_export",
    ),
    (
        "als_logbook_plain_text",
        "osprey.services.ariel_search.ingestion.adapters.als_text_migration",
        "ALSPlainTextMigration",
        None,  # Always runs
    ),
]


LockFactory = Callable[..., AbstractAsyncContextManager[bool]]
"""Signature of :func:`~osprey.services.ariel_search.database.connection.try_advisory_lock`:
``(conninfo, key, *, wait) -> async context manager yielding whether the lock is held``."""

LockMode = Literal["wait", "try"]


@dataclass
class MigrationResult:
    """Outcome of :func:`run_migrations_detailed`.

    Attributes:
        applied: Migrations applied in this run.
        busy_skipped: Migrations skipped because their tables were busy; a
            retry later is expected to apply them (and their dependents).
        lock_held: True when ``lock='try'`` found another session already
            migrating, so this run did nothing.
    """

    applied: list[str] = field(default_factory=list)
    busy_skipped: list[str] = field(default_factory=list)
    lock_held: bool = False


class MigrationRunner:
    """Discovers, orders, and executes ARIEL database migrations.

    Migrations are discovered from the KNOWN_MIGRATIONS registry and
    filtered based on enabled modules in the config.

    ``run`` and ``rollback`` hold the :data:`MIGRATION_LOCK_KEY` advisory lock
    for their whole body, so concurrent runs against one database serialize.
    """

    def __init__(
        self,
        pool: "AsyncConnectionPool",
        config: "ARIELConfig",
        *,
        lock_factory: LockFactory | None = None,
    ) -> None:
        """Initialize the migration runner.

        Args:
            pool: Database connection pool
            config: ARIEL configuration
            lock_factory: Context-manager factory that holds the migration
                lock; defaults to ``try_advisory_lock``, which opens its own
                connection from ``pool.conninfo``.
        """
        if lock_factory is None:
            from osprey.services.ariel_search.database.connection import try_advisory_lock

            lock_factory = try_advisory_lock
        self.pool = pool
        self.config = config
        self.lock_factory = lock_factory
        #: Whether the last ``run(lock='try')`` found the lock held elsewhere.
        self.lock_held = False
        #: Migrations the last ``_get_enabled_migrations`` left out because
        #: their ``MIGRATION_ARGS`` resolver raised, mapped to the reason.
        self.config_skipped: dict[str, str] = {}
        self._warned_config_skips: set[str] = set()

    def _migration_lock(self, *, wait: bool) -> AbstractAsyncContextManager[bool]:
        """Open the migration lock on the database this runner's pool migrates."""
        return self.lock_factory(getattr(self.pool, "conninfo", ""), MIGRATION_LOCK_KEY, wait=wait)

    def _get_enabled_migrations(self) -> list[BaseMigration]:
        """Get list of migrations to run based on enabled modules.

        A migration whose ``MIGRATION_ARGS`` resolver raises ``ValueError`` is
        left out with one warning (per runner) naming it and the configuration
        key, so the rest of the chain still runs. It is recorded in
        :attr:`config_skipped`, so :meth:`run` holds its dependents back the
        way it does for :class:`MigrationSkippedError`.

        Returns:
            List of migration instances in no particular order
        """
        migrations: list[BaseMigration] = []
        self.config_skipped = {}

        for name, module_path, class_name, requires_module in KNOWN_MIGRATIONS:
            if requires_module is None:
                should_run = True
            else:
                should_run = self.config.is_enhancement_module_enabled(requires_module)

            if should_run:
                resolve = MIGRATION_ARGS.get(name)
                try:
                    args = resolve(self) if resolve is not None else ()
                except ValueError as e:
                    key = getattr(e, "key", None)
                    where = f" ({key})" if key and key not in str(e) else ""
                    reason = f"{e}{where}"
                    self.config_skipped[name] = reason
                    if name not in self._warned_config_skips:
                        self._warned_config_skips.add(name)
                        logger.warning(f"Migration {name} skipped: {reason}")
                    continue
                try:
                    module = importlib.import_module(module_path)
                    migration_class = getattr(module, class_name)
                    migration = migration_class(*args)
                    migrations.append(migration)
                    logger.debug(f"Loaded migration: {name}")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Failed to load migration {name}: {e}")

        return migrations

    def _configured_embedding_models(self) -> list[tuple[str, int]] | None:
        """Read the configured text_embedding (model, dimension) pairs.

        Returns None when no models are configured, so the migration falls back
        to its own default.
        """
        module_config = self.config.enhancement_modules.get("text_embedding")
        if module_config and module_config.models:
            return [(m.name, m.dimension) for m in module_config.models]
        return None

    def _topological_sort(self, migrations: list[BaseMigration]) -> list[BaseMigration]:
        """Sort migrations by dependencies using topological sort.

        Args:
            migrations: Unsorted list of migrations

        Returns:
            Migrations sorted by dependency order

        Raises:
            ConfigurationError: If circular dependency detected
        """
        migration_map = {m.name: m for m in migrations}

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {m.name: 0 for m in migrations}
        graph: dict[str, list[str]] = {m.name: [] for m in migrations}

        for migration in migrations:
            for dep in migration.depends_on:
                if dep in migration_map:
                    graph[dep].append(migration.name)
                    in_degree[migration.name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_names: list[str] = []

        while queue:
            name = queue.pop(0)
            sorted_names.append(name)

            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_names) != len(migrations):
            raise ConfigurationError(
                "Circular dependency detected in migrations",
                config_key="ariel.migrations",
            )

        return [migration_map[name] for name in sorted_names]

    async def run(
        self, dry_run: bool = False, *, lock: LockMode = "wait"
    ) -> tuple[list[str], list[str]]:
        """Run all pending migrations.

        The migration lock is taken before the first ``is_applied`` and held
        to the end. A migration that skips (:class:`MigrationSkippedError`)
        stays unmarked, and every migration depending on a skipped one, or on
        one whose configuration could not build it, is skipped too, with the
        warning ``<name> waits for <dep>``.

        Args:
            dry_run: If True, only report what would be done
            lock: ``'wait'`` blocks until the migration lock is free;
                ``'try'`` returns ``([], [])`` at once, with
                :attr:`lock_held` set, when another session holds it.

        Returns:
            ``(applied, busy_skipped)``: migrations applied (or that would be,
            on a dry run), and those skipped with :class:`MigrationBusyError`.
        """
        if lock not in ("wait", "try"):
            raise ValueError(f"lock must be 'wait' or 'try', not {lock!r}")

        migrations = self._get_enabled_migrations()
        sorted_migrations = self._topological_sort(migrations)

        applied: list[str] = []
        busy_skipped: list[str] = []
        skipped: set[str] = set(self.config_skipped)
        self.lock_held = False

        async with self._migration_lock(wait=lock == "wait") as acquired:
            if not acquired:
                logger.info("Another session is running ARIEL migrations; not waiting")
                self.lock_held = True
                return applied, busy_skipped

            async with self.pool.connection() as conn:
                for migration in sorted_migrations:
                    is_applied = await migration.is_applied(conn)

                    if is_applied:
                        logger.debug(f"Migration {migration.name} already applied")
                        continue

                    blocker = next((d for d in migration.depends_on if d in skipped), None)
                    if blocker is not None:
                        logger.warning(f"{migration.name} waits for {blocker}")
                        skipped.add(migration.name)
                        continue

                    if dry_run:
                        logger.info(f"Would apply migration: {migration.name}")
                        applied.append(migration.name)
                        continue

                    logger.info(f"Applying migration: {migration.name}")
                    try:
                        # One transaction per migration, covering `up()` AND the
                        # bookkeeping row together. The pool is opened with
                        # autocommit=True (`connection.py`), so without this every
                        # statement inside `up()` commits on its own -- and a
                        # migration that drops an index before creating its
                        # replacement destroys the old one for good the moment the
                        # create fails. Two in this registry do exactly that
                        # (`text_embedding_hnsw_index`, and
                        # `semantic_processor_search_index` on the FTS index), and
                        # the create is an index build: it is the statement most
                        # likely to fail on resources, which is precisely when the
                        # drop must not have happened. Committing the two together
                        # also closes the narrower hole where `up()` succeeded and
                        # `mark_applied()` then failed, leaving a drop-then-create
                        # migration to run a second time against the state it
                        # already made.
                        #
                        # Every migration in KNOWN_MIGRATIONS is transactional DDL
                        # (CREATE/DROP INDEX, CREATE TABLE, ALTER TABLE ... ADD
                        # COLUMN, CREATE EXTENSION, CREATE FUNCTION), except one
                        # that rewrites rows with UPDATE, which is transactional
                        # DML. Nothing here uses CREATE INDEX CONCURRENTLY, VACUUM
                        # or CREATE DATABASE, which are the statements PostgreSQL
                        # forbids inside a transaction block -- so a migration that
                        # needs one of those cannot simply be added here; it needs
                        # its own escape from this block, deliberately.
                        async with conn.transaction():
                            await migration.up(conn)
                            await migration.mark_applied(conn)
                        applied.append(migration.name)
                        logger.info(f"Applied migration: {migration.name}")
                    except MigrationSkippedError as e:
                        skipped.add(migration.name)
                        if isinstance(e, MigrationBusyError):
                            busy_skipped.append(migration.name)
                            logger.warning(f"Migration {migration.name} busy: {e}")
                        else:
                            logger.warning(f"Migration {migration.name} skipped: {e}")
                    except Exception as e:
                        logger.error(f"Failed to apply migration {migration.name}: {e}")
                        raise

        return applied, busy_skipped

    async def rollback(self, migration_name: str) -> bool:
        """Rollback a specific migration.

        Args:
            migration_name: Name of the migration to rollback

        Returns:
            True if rollback was successful
        """
        migrations = self._get_enabled_migrations()
        migration_map = {m.name: m for m in migrations}

        if migration_name in self.config_skipped:
            reason = self.config_skipped[migration_name]
            logger.error(f"Migration {migration_name} cannot be built: {reason}")
            return False

        if migration_name not in migration_map:
            logger.error(f"Migration not found: {migration_name}")
            return False

        migration = migration_map[migration_name]

        async with self._migration_lock(wait=True), self.pool.connection() as conn:
            is_applied = await migration.is_applied(conn)

            if not is_applied:
                logger.info(f"Migration {migration_name} is not applied")
                return True

            logger.info(f"Rolling back migration: {migration_name}")
            try:
                await migration.down(conn)
                await migration.mark_unapplied(conn)
                logger.info(f"Rolled back migration: {migration_name}")
                return True
            except NotImplementedError:
                logger.error(f"Rollback not implemented for migration: {migration_name}")
                return False
            except Exception as e:
                logger.error(f"Failed to rollback migration {migration_name}: {e}")
                raise

    async def status(self) -> dict[str, dict[str, bool | str]]:
        """Get status of all migrations.

        Returns:
            Dict mapping migration name to status info
        """
        migrations = self._get_enabled_migrations()
        status: dict[str, dict[str, bool | str]] = {}

        async with self.pool.connection() as conn:
            for migration in migrations:
                is_applied = await migration.is_applied(conn)
                status[migration.name] = {
                    "applied": is_applied,
                    "depends_on": ", ".join(migration.depends_on)
                    if migration.depends_on
                    else "(none)",
                }

        return status


async def run_migrations(
    pool: "AsyncConnectionPool",
    config: "ARIELConfig",
    dry_run: bool = False,
    *,
    lock_factory: LockFactory | None = None,
) -> list[str]:
    """Convenience function to run migrations, waiting for the migration lock.

    Args:
        pool: Database connection pool
        config: ARIEL configuration
        dry_run: If True, only report what would be done
        lock_factory: Forwarded to :class:`MigrationRunner`.

    Returns:
        List of migration names that were applied
    """
    runner = MigrationRunner(pool, config, lock_factory=lock_factory)
    applied, _busy = await runner.run(dry_run=dry_run)
    return applied


async def run_migrations_detailed(
    pool: "AsyncConnectionPool",
    config: "ARIELConfig",
    *,
    lock: LockMode = "wait",
    lock_factory: LockFactory | None = None,
) -> MigrationResult:
    """Run migrations and report busy skips and whether the lock was free.

    Args:
        pool: Database connection pool
        config: ARIEL configuration
        lock: ``'wait'`` or ``'try'``, as for :meth:`MigrationRunner.run`.
        lock_factory: Forwarded to :class:`MigrationRunner`.

    Returns:
        The applied and busy-skipped names, and ``lock_held`` when
        ``lock='try'`` found another session migrating.
    """
    runner = MigrationRunner(pool, config, lock_factory=lock_factory)
    applied, busy_skipped = await runner.run(lock=lock)
    return MigrationResult(applied, busy_skipped, runner.lock_held)
