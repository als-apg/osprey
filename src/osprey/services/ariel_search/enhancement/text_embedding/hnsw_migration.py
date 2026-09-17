"""Reconcile an existing embedding index onto HNSW.

A deployment that already ran ``text_embedding`` carries its record in
``ariel_migrations`` and never re-runs it, so the index it created stays as it
was made. This additive migration moves that index across without mutating the
historical migration record.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    MigrationSkippedError,
    model_to_table_name,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    create_vector_index_sql,
    legacy_vector_index_name,
    pgvector_available,
    vector_index_name,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class TextEmbeddingHnswIndexMigration(BaseMigration):
    """Rebuilds each per-model embedding index as an HNSW index.

    The index is rebuilt once per model table. On a large corpus that build is
    the cost of the move.
    """

    def __init__(self, models: list[tuple[str, int]] | None = None) -> None:
        """Initialize the migration.

        Args:
            models: The (model_name, dimension) pairs the creating migration
                was given. If None, uses the same default it does.
        """
        super().__init__()
        self._models = models

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "text_embedding_hnsw_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the per-model tables the creating migration makes."""
        return ["text_embedding"]

    def _get_models(self) -> list[tuple[str, int]]:
        """Get the list of models whose index is reconciled.

        Returns:
            List of (model_name, dimension) tuples
        """
        if self._models:
            return self._models
        return [("nomic-embed-text", 768)]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the HNSW index reconciliation."""
        if not await pgvector_available(conn):
            raise MigrationSkippedError(
                "pgvector extension is not available in this PostgreSQL installation. "
                "Install pgvector to enable semantic search. "
                "ARIEL will fall back to keyword-only search."
            )

        for model_name, _dimension in self._get_models():
            table_name = model_to_table_name(model_name)
            result = await conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
            row = await result.fetchone()
            if not (row and row[0]):
                continue

            await conn.execute(f"DROP INDEX IF EXISTS {legacy_vector_index_name(table_name)}")
            await conn.execute(create_vector_index_sql(table_name))

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the HNSW index reconciliation."""
        for model_name, _dimension in self._get_models():
            table_name = model_to_table_name(model_name)
            await conn.execute(f"DROP INDEX IF EXISTS {vector_index_name(table_name)}")
