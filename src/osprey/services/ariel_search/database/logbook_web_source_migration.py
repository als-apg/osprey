"""Rename the ``source_system`` of the Logbook's own local saves.

The Logbook panel's compose sheet used to save entries with ``source_system``
``ARIEL Web``; it now writes
:data:`~osprey.services.ariel_search.entry_url.LOGBOOK_WEB_SOURCE_SYSTEM`.
``logbook_web_source_system`` renames the stored rows so the Source System
facet shows one name. Only that exact value changes: every other
``source_system`` (``ARIEL MCP``, a facility's own name, a value that merely
resembles the old one) and every other column is left alone.

The trigger moves ``updated_at`` on each renamed row on purpose: the qmd
mirror writes ``Source: <source_system>.`` into each entry's document and
re-exports rows by ``updated_at`` (``osprey ariel qmd-resync``), so a kept
``updated_at`` would leave the old name in the hybrid index. Facets and the
``lower(source_system)`` index read ``enhanced_entries`` directly, and
``ingestion_runs.source_system`` names a facility adapter's ingest runs, which
a Logbook save never creates, so neither needs a change.

The ``UPDATE`` matches only rows still holding the old value, so a second run
renames nothing.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.entry_url import LOGBOOK_WEB_SOURCE_SYSTEM
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

logger = get_logger("ariel")

#: The value the compose sheet stored before it wrote ``NEW_SOURCE_SYSTEM``.
OLD_SOURCE_SYSTEM = "ARIEL Web"

#: The value it writes now.
NEW_SOURCE_SYSTEM = LOGBOOK_WEB_SOURCE_SYSTEM

_RENAME_SQL = """
    UPDATE enhanced_entries
    SET source_system = %(new)s
    WHERE source_system = %(old)s
"""


class LogbookWebSourceMigration(BaseMigration):
    """Renames ``source_system`` ``ARIEL Web`` to ``OSPREY Logbook Web`` -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "logbook_web_source_system"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for ``enhanced_entries``."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Rename every row still holding the old value."""
        result = await conn.execute(
            _RENAME_SQL,
            {"new": NEW_SOURCE_SYSTEM, "old": OLD_SOURCE_SYSTEM},
        )
        logger.info(f"{self.name}: {max(result.rowcount, 0)} entries renamed")

    async def down(self, conn: "AsyncConnection") -> None:
        """Nothing to undo: the old name is not restored.

        Entries the compose sheet saved after the rename carry the new name too,
        and the old and new saves cannot be told apart.
        """
