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


#: Used when ``ariel.enhancement_modules.text_embedding.index_lists`` is unset.
#: The IVFFlat rule of thumb is rows/1000 for corpora up to a million entries;
#: this suits a few hundred thousand entries and is a workable index for far
#: fewer. It cannot be derived at migration time — ``osprey ariel migrate``
#: runs against an empty table.
DEFAULT_INDEX_LISTS = 224


class TextEmbeddingMigration(BaseMigration):
    """Text embedding enhancement migration.

    Creates:
    - pgvector extension
    - text_embeddings_<model_name> table for each configured model
    - IVFFlat vector indexes
    """

    def __init__(
        self,
        models: list[tuple[str, int]] | None = None,
        index_lists: int | None = None,
    ) -> None:
        """Initialize the migration.

        Args:
            models: List of (model_name, dimension) tuples to create tables for.
                   If None, uses a default for testing.
            index_lists: IVFFlat ``lists`` for the vector index, from
                ``ariel.enhancement_modules.text_embedding.index_lists``. None
                uses :data:`DEFAULT_INDEX_LISTS`.

        Raises:
            ValueError: If ``index_lists`` is not a positive integer.
        """
        super().__init__()
        self._models = models
        if index_lists is not None and (
            not isinstance(index_lists, int) or isinstance(index_lists, bool) or index_lists < 1
        ):
            raise ValueError(
                "ariel.enhancement_modules.text_embedding.index_lists must be an "
                f"integer >= 1 (got {index_lists!r})"
            )
        self._index_lists = DEFAULT_INDEX_LISTS if index_lists is None else index_lists

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

    async def _is_pgvector_available(self, conn: "AsyncConnection") -> bool:
        """Check if the pgvector extension is available in PostgreSQL.

        Returns:
            True if pgvector is available for installation
        """
        result = await conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
        )
        row = await result.fetchone()
        return bool(row and row[0])

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the text embedding migration."""
        if not await self._is_pgvector_available(conn):
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

            index_name = f"idx_{table_name}_vector"
            # `lists` is baked into the index at creation, so changing the key
            # later needs the index dropped and recreated — it is not re-read.
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self._index_lists})
                """  # noqa: S608
            )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the text embedding migration."""
        models = self._get_models()
        for model_name, _dimension in models:
            table_name = model_to_table_name(model_name)
            await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")  # noqa: S608

        # Note: We don't drop the vector extension as other things may use it
