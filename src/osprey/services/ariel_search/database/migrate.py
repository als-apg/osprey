"""ARIEL migration runner.

This module provides the MigrationRunner class that discovers, orders,
and executes database migrations.

See 04_OSPREY_INTEGRATION.md Sections 11.1-11.8 for specification.
"""

import importlib
from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migration import BaseMigration
from osprey.services.ariel_search.exceptions import ConfigurationError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")


# Known migrations registry (MVP approach - hardcoded list)
# Format: (name, module_path, class_name, requires_module)
# requires_module is None for core_schema (always runs), otherwise module name
KNOWN_MIGRATIONS: list[tuple[str, str, str, str | None]] = [
    (
        "core_schema",
        "osprey.services.ariel_search.database.core_migration",
        "CoreMigration",
        None,  # Always runs
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
            # Check if module should be enabled
            if requires_module is None:
                # Core schema always runs
                should_run = True
            else:
                # Check if enhancement module is enabled
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
        # Build name -> migration map
        migration_map = {m.name: m for m in migrations}

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {m.name: 0 for m in migrations}
        graph: dict[str, list[str]] = {m.name: [] for m in migrations}

        # Build dependency graph
        for migration in migrations:
            for dep in migration.depends_on:
                if dep in migration_map:
                    graph[dep].append(migration.name)
                    in_degree[migration.name] += 1

        # Start with nodes that have no dependencies
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
