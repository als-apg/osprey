"""ARIEL text embedding migration.

This module provides the database migration for the text embedding
enhancement module.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    MigrationSkippedError,
    model_to_table_name,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


def vector_index_name(table_name: str) -> str:
    """Return the name of the HNSW embedding index on ``table_name``."""
    return f"idx_{table_name}_hnsw"


def legacy_vector_index_name(table_name: str) -> str:
    """Return the name the superseded IVFFlat index was created under."""
    return f"idx_{table_name}_vector"


def create_vector_index_sql(table_name: str) -> str:
    """Return the DDL creating the embedding index on ``table_name``.

    The single producer of that statement: the creating migration and the
    reconcile migration must write the same index, so neither spells it out.
    """
    return (
        f"CREATE INDEX IF NOT EXISTS {vector_index_name(table_name)} "
        f"ON {table_name} USING hnsw (embedding vector_cosine_ops)"
    )


async def pgvector_available(conn: "AsyncConnection") -> bool:
    """Report whether the pgvector extension can be installed on this server.

    Returns:
        True if pgvector is available for installation
    """
    result = await conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
    )
    row = await result.fetchone()
    return bool(row and row[0])


class TextEmbeddingMigration(BaseMigration):
    """Text embedding enhancement migration.

    Creates:
    - pgvector extension
    - text_embeddings_<model_name> table for each configured model
    - HNSW vector indexes
    """

    def __init__(self, models: list[tuple[str, int]] | None = None) -> None:
        """Initialize the migration.

        Args:
            models: List of (model_name, dimension) tuples to create tables for.
                   If None, uses a default for testing.
        """
        super().__init__()
        self._models = models

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "text_embedding"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema."""
        return ["core_schema"]

    def _get_models(self) -> list[tuple[str, int]]:
        """Get the list of models to create tables for.

        Returns:
            List of (model_name, dimension) tuples
        """
        if self._models:
            return self._models
        # Default: nomic-embed-text (most common)
        return [("nomic-embed-text", 768)]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the text embedding migration."""
        if not await pgvector_available(conn):
            raise MigrationSkippedError(
                "pgvector extension is not available in this PostgreSQL installation. "
                "Install pgvector to enable semantic search. "
                "ARIEL will fall back to keyword-only search."
            )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        models = self._get_models()
        for model_name, dimension in models:
            table_name = model_to_table_name(model_name)

            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id              SERIAL PRIMARY KEY,
                    entry_id        TEXT NOT NULL REFERENCES enhanced_entries(entry_id) ON DELETE CASCADE,
                    embedding       vector({dimension}),
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(entry_id)
                )
                """  # noqa: S608
            )

            await conn.execute(create_vector_index_sql(table_name))

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the text embedding migration."""
        models = self._get_models()
        for model_name, _dimension in models:
            table_name = model_to_table_name(model_name)
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")  # noqa: S608

        # Note: We don't drop the vector extension as other things may use it
