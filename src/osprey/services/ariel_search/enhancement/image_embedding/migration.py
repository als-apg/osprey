"""ARIEL image embedding migration.

Creates the one table the configured image-embedding model and width name,
keyed by attachment.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    ImageEmbeddingTarget,
    MigrationSkippedError,
    image_index_name,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    create_vector_index_sql,
    pgvector_available,
    vector_index_name,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


def create_image_index_sql(table: str) -> str:
    """Return the DDL creating the HNSW cosine index on an image table.

    The statement comes from the text module's single producer of the index
    DDL, under the image index name, so the two lanes index the same way.
    """
    return create_vector_index_sql(table).replace(
        vector_index_name(table), image_index_name(table), 1
    )


class ImageEmbeddingMigration(BaseMigration):
    """Image embedding enhancement migration.

    Creates:
    - pgvector extension, when absent (``IF NOT EXISTS``; an installed one is
      never altered, so the store may own it or a DBA may)
    - the ``image_embeddings_<model>_d<dims>`` table, one row per attachment;
      a NULL ``embedding`` with a ``skip_reason`` records an attachment that
      was looked at and could not be embedded
    - an HNSW cosine index on the embedding column

    Applied means the table exists, so a store whose configured model or width
    changes gets the new table on the next run.
    """

    def __init__(self, target: ImageEmbeddingTarget) -> None:
        """Initialize the migration.

        Args:
            target: The configured model, width and table, from
                :func:`~osprey.services.ariel_search.database.migrations.image_embedding_target`.
        """
        super().__init__()
        self.target = target

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "image_embedding"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the attachment table with its copy state."""
        return ["attachment_files_copy_state"]

    async def is_applied(self, conn: "AsyncConnection") -> bool:
        """Report whether the configured table exists."""
        result = await conn.execute("SELECT to_regclass(%s) IS NOT NULL", (self.target.table,))
        row = await result.fetchone()
        return bool(row and row[0])

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the image embedding migration."""
        if not await pgvector_available(conn):
            raise MigrationSkippedError(
                "pgvector extension is not available in this PostgreSQL installation; "
                "the image_embedding module needs it. Install pgvector to enable "
                "image search."
            )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        table = self.target.table
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                attachment_id   TEXT PRIMARY KEY
                                REFERENCES attachment_files(attachment_id) ON DELETE CASCADE,
                embedding       vector({self.target.dims}) NULL,
                skip_reason     TEXT,
                model_ref       TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        await conn.execute(create_image_index_sql(table))

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the configured table; the vector extension stays."""
        await conn.execute(f"DROP TABLE IF EXISTS {self.target.table} CASCADE")
