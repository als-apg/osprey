"""ARIEL ``input_key`` column on every text-embedding table.

``input_key`` records which input a stored vector was computed from, so a
re-embed can tell a vector built from the current entry text apart from one
built from an older rendering of it. Text-embedding tables are created per
model by ``text_embedding``; this migration adds the column to every one of
them that predates it.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection

#: Probe for one table's ``input_key`` column in the current schema; its one
#: parameter is the table name. A row comes back when the column exists.
INPUT_KEY_COLUMN_SQL = """
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = current_schema()
    AND table_name = %s
    AND column_name = 'input_key'
"""

#: Text-embedding tables in the current schema, filtered on whether they
#: already carry ``input_key``. Run without parameters, so ``%`` is literal;
#: the backslashes escape LIKE's ``_`` wildcard.
_TABLES_SQL = r"""
    SELECT t.table_name
    FROM information_schema.tables t
    WHERE t.table_schema = current_schema()
      AND t.table_name LIKE 'text\_embeddings\_%'
      AND {exists} (
          SELECT 1 FROM information_schema.columns c
          WHERE c.table_schema = t.table_schema
            AND c.table_name = t.table_name
            AND c.column_name = 'input_key'
      )
    ORDER BY t.table_name
"""


async def _embedding_tables(conn: "AsyncConnection", *, with_column: bool) -> list[str]:
    """Return the text-embedding tables that have (or lack) ``input_key``."""
    query = _TABLES_SQL.format(exists="EXISTS" if with_column else "NOT EXISTS")
    result = await conn.execute(query)
    rows = await result.fetchall()
    return [row[0] for row in rows]


def _quoted(table: str) -> str:
    """Quote *table* as an SQL identifier."""
    return '"' + table.replace('"', '""') + '"'


class EmbeddingInputKeyMigration(BaseMigration):
    """``input_key TEXT`` on every ``text_embeddings_*`` table -- always runs.

    Applied means the tracking row exists *and* no text-embedding table lacks
    the column. A table created after the row was written by an older
    ``text_embedding`` therefore makes the migration unapplied again, and the
    next run's ``up()`` heals a partially-columned database by adding the
    column to just the tables that miss it. Each ``ALTER TABLE`` is taken
    under :func:`acquire_nonqueueing_lock`, which never queues behind readers.
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "embedding_input_key"

    @property
    def depends_on(self) -> list[str]:
        """Needs the tracking table and ``enhanced_entries``."""
        return ["core_schema"]

    async def is_applied(self, conn: "AsyncConnection") -> bool:
        """True when the tracking row exists and every table has the column."""
        if not await super().is_applied(conn):
            return False
        return not await _embedding_tables(conn, with_column=False)

    async def up(self, conn: "AsyncConnection") -> None:
        """Add ``input_key`` to every text-embedding table that lacks it.

        No table lacking it (including no table at all) is a no-op.

        Raises:
            MigrationBusyError: If a table stayed in use for every lock
                attempt; the runner leaves the migration unapplied.
        """
        for table in await _embedding_tables(conn, with_column=False):
            await acquire_nonqueueing_lock(conn, table, "ACCESS EXCLUSIVE")
            await conn.execute(
                f"ALTER TABLE {_quoted(table)} ADD COLUMN IF NOT EXISTS input_key TEXT"
            )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop ``input_key`` from every text-embedding table carrying it.

        The runner calls ``down()`` outside a transaction, so this opens its
        own: the non-queueing locks must be held until each drop commits.

        Raises:
            MigrationBusyError: If a table stayed in use for every lock attempt.
        """
        async with conn.transaction():
            for table in await _embedding_tables(conn, with_column=True):
                await acquire_nonqueueing_lock(conn, table, "ACCESS EXCLUSIVE")
                await conn.execute(f"ALTER TABLE {_quoted(table)} DROP COLUMN IF EXISTS input_key")
