"""ARIEL migration base class and utilities.

This module provides the base class for ARIEL database migrations
and utility functions for migration management.

See 01_DATA_LAYER.md Sections 3.1-3.5 for specification.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import AsyncConnection


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
        # Check ariel_migrations table
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
            # ariel_migrations table doesn't exist yet
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
    # Replace hyphens and other invalid characters with underscores
    safe_name = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    # Remove any double underscores
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    # Lowercase
    safe_name = safe_name.lower()
    return f"text_embeddings_{safe_name}"
