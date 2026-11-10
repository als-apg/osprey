"""Record ``sync_status`` ``local_only`` on ARIEL's own entries stored without it.

Every write path now records ``metadata.sync_status``, and
:func:`~osprey.services.ariel_search.entry_url.is_ariel_only` reads it alone.
Rows written before that did not all carry it: an ARIEL MCP native entry never
did, a Logbook compose-sheet local save before 2026-10-04 did not, and a
simulation seed never did. ``ariel_only_sync_status`` stamps those rows once,
so no read path has to guess from the source name.

Only rows whose ``source_system`` is one ARIEL itself writes, and whose metadata
has no ``sync_status`` at all, change. A facility entry and any row that already
records a status are left alone. ``ARIEL Web`` is listed beside its successor so
the result does not depend on the order this and ``logbook_web_source_system``
run in.

The ``UPDATE`` matches only rows still missing the key, so a second run changes
nothing.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.entry_url import (
    ARIEL_NATIVE_SOURCE_SYSTEM,
    LOGBOOK_WEB_SOURCE_SYSTEM,
)
from osprey.services.ariel_search.models import SyncStatus
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

logger = get_logger("ariel")

#: The source systems of entries that only ever live in ARIEL.
ARIEL_ONLY_SOURCE_SYSTEMS = (
    ARIEL_NATIVE_SOURCE_SYSTEM,
    LOGBOOK_WEB_SOURCE_SYSTEM,
    "ARIEL Web",
    "Simulation",
)

_STAMP_SQL = """
    UPDATE enhanced_entries
    SET metadata = CASE
            WHEN jsonb_typeof(metadata) = 'object' THEN metadata
            ELSE '{}'::jsonb
        END || jsonb_build_object('sync_status', %(status)s::text)
    WHERE source_system = ANY(%(sources)s)
      AND NOT (jsonb_typeof(metadata) = 'object' AND metadata ? 'sync_status')
"""


class ArielOnlySyncStatusMigration(BaseMigration):
    """Stamps ``sync_status`` on ARIEL-only rows stored without it -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "ariel_only_sync_status"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for ``enhanced_entries``."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Stamp every ARIEL-only row still missing the key."""
        result = await conn.execute(
            _STAMP_SQL,
            {"status": SyncStatus.LOCAL_ONLY.value, "sources": list(ARIEL_ONLY_SOURCE_SYSTEMS)},
        )
        logger.info(f"{self.name}: {max(result.rowcount, 0)} entries stamped local_only")

    async def down(self, conn: "AsyncConnection") -> None:
        """Nothing to undo: stamped rows cannot be told from rows written with the key."""
