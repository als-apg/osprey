"""E2E custom enhancement module: watermark enhancer.

Stamps ``DEMO-001`` with ``{"e2e_watermark": "CANARY_12345"}`` in the
``metadata`` JSONB column.  Used by ``TestCustomModuleIntegration`` to
prove that custom enhancement modules run and persist data that custom
search modules can then retrieve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.models import EnhancedLogbookEntry

WATERMARK_ENTRY_ID = "DEMO-001"
WATERMARK_VALUE = "CANARY_12345"


class WatermarkEnhancer(BaseEnhancementModule):
    """Stamp a single entry with a distinctive watermark in metadata."""

    @property
    def name(self) -> str:
        return "watermark"

    async def enhance(
        self,
        entry: EnhancedLogbookEntry,
        conn: AsyncConnection,
    ) -> None:
        """Stamp DEMO-001 with the watermark; skip all other entries."""
        if entry["entry_id"] != WATERMARK_ENTRY_ID:
            return

        await conn.execute(
            """
            UPDATE enhanced_entries
            SET metadata = metadata || %s::jsonb
            WHERE entry_id = %s
            """,
            [f'{{"e2e_watermark": "{WATERMARK_VALUE}"}}', WATERMARK_ENTRY_ID],
        )
