"""ARIEL database migrations.

This module provides the migration base class, utilities, runner, and
convenience function for managing ARIEL database schema changes.

Components:
    - BaseMigration: Abstract base class for individual migrations
    - MigrationSkippedError: Raised when prerequisites are missing
    - model_to_table_name(): Converts model names to table names
    - KNOWN_MIGRATIONS: Registry of all known migration classes
    - MigrationRunner: Discovers, orders, and executes migrations
    - run_migrations(): Convenience function for the runner
"""

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from osprey.services.ariel_search.exceptions import ConfigurationError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")


# ---------------------------------------------------------------------------
# Base class & utilities (formerly migration.py)
# ---------------------------------------------------------------------------


class MigrationSkippedError(Exception):
    """Raised when a migration cannot run due to missing prerequisites.

    The migration runner catches this and logs a warning instead of failing.
    The migration is NOT marked as applied so it retries when prerequisites
    are later installed (e.g., pgvector extension).
    """


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


# ---------------------------------------------------------------------------
# Migration runner (formerly migrate.py)
# ---------------------------------------------------------------------------

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
        "semantic_processor",
        "osprey.services.ariel_search.enhancement.semantic_processor.migration",
        "SemanticProcessorMigration",
        "semantic_processor",
    ),
    (
        "text_embedding",
        "osprey.services.ariel_search.enhancement.text_embedding.migration",
        "TextEmbeddingMigration",
        "text_embedding",
    ),
    (
        "attachment_files",
        "osprey.services.ariel_search.database.attachment_migration",
        "AttachmentMigration",
        None,  # Always runs
    ),
]


class MigrationRunner:
    """Discovers, orders, and executes ARIEL database migrations.

    Migrations are discovered from the KNOWN_MIGRATIONS registry and
    filtered based on enabled modules in the config.
    """

    def __init__(self, pool: "AsyncConnectionPool", config: "ARIELConfig") -> None:
        """Initialize the migration runner.

        Args:
            pool: Database connection pool
            config: ARIEL configuration
        """
        self.pool = pool
        self.config = config

    def _get_enabled_migrations(self) -> list[BaseMigration]:
        """Get list of migrations to run based on enabled modules.

        Returns:
            List of migration instances in no particular order
        """
        migrations: list[BaseMigration] = []

        for name, module_path, class_name, requires_module in KNOWN_MIGRATIONS:
            if requires_module is None:
                should_run = True
            else:
                should_run = self.config.is_enhancement_module_enabled(requires_module)

            if should_run:
                try:
                    module = importlib.import_module(module_path)
                    migration_class = getattr(module, class_name)
                    migrations.append(migration_class())
                    logger.debug(f"Loaded migration: {name}")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Failed to load migration {name}: {e}")

        return migrations

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

    async def run(self, dry_run: bool = False) -> list[str]:
        """Run all pending migrations.

        Args:
            dry_run: If True, only report what would be done

        Returns:
            List of migration names that were applied (or would be applied)
        """
        migrations = self._get_enabled_migrations()
        sorted_migrations = self._topological_sort(migrations)

        applied: list[str] = []

        async with self.pool.connection() as conn:
            for migration in sorted_migrations:
                is_applied = await migration.is_applied(conn)

                if is_applied:
                    logger.debug(f"Migration {migration.name} already applied")
                    continue

                if dry_run:
                    logger.info(f"Would apply migration: {migration.name}")
                    applied.append(migration.name)
                    continue

                logger.info(f"Applying migration: {migration.name}")
                try:
                    await migration.up(conn)
                    await migration.mark_applied(conn)
                    applied.append(migration.name)
                    logger.info(f"Applied migration: {migration.name}")
                except MigrationSkippedError as e:
                    logger.warning(f"Migration {migration.name} skipped: {e}")
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration.name}: {e}")
                    raise

        return applied

    async def rollback(self, migration_name: str) -> bool:
        """Rollback a specific migration.

        Args:
            migration_name: Name of the migration to rollback

        Returns:
            True if rollback was successful
        """
        migrations = self._get_enabled_migrations()
        migration_map = {m.name: m for m in migrations}

        if migration_name not in migration_map:
            logger.error(f"Migration not found: {migration_name}")
            return False

        migration = migration_map[migration_name]

        async with self.pool.connection() as conn:
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
) -> list[str]:
    """Convenience function to run migrations.

    Args:
        pool: Database connection pool
        config: ARIEL configuration
        dry_run: If True, only report what would be done

    Returns:
        List of migration names that were applied
    """
    runner = MigrationRunner(pool, config)
    return await runner.run(dry_run=dry_run)
