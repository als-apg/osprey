"""ARIEL text embedding module.

This module generates text embeddings for logbook entries to enable semantic search.

See 01_DATA_LAYER.md Section 6.3 for specification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.migration import model_to_table_name
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.database.migration import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = logging.getLogger(__name__)

# Default characters per token estimate (conservative)
CHARS_PER_TOKEN = 4


class TextEmbeddingModule(BaseEnhancementModule):
    """Generate text embeddings for logbook entries.

    Supports multiple embedding models, each with its own dedicated table.
    Follows Osprey's zero-argument constructor pattern with lazy loading
    of expensive resources.
    """

    def __init__(self) -> None:
        """Initialize the module.

        Zero-argument constructor (Osprey pattern).
        Expensive resources (embedding provider) are lazy-loaded.
        """
        self._provider: BaseEmbeddingProvider | None = None
        self._models: list[dict[str, Any]] = []
        self._provider_config: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return module identifier."""
        return "text_embedding"

    @property
    def migration(self) -> type[BaseMigration]:
        """Return migration class for this module."""
        return TextEmbeddingMigration

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the module with settings from config.yml.

        Args:
            config: The enhancement_modules.text_embedding config dict
                   containing 'models' list, etc.
        """
        self._models = config.get("models", [])
        self._provider_config = config.get("provider", {})

    def _get_provider(self) -> BaseEmbeddingProvider:
        """Lazy-load and return the embedding provider.

        Returns:
            Configured embedding provider instance
        """
        if self._provider is None:
            from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

            self._provider = OllamaEmbeddingProvider()

        return self._provider

    async def enhance(
        self,
        entry: EnhancedLogbookEntry,
        conn: AsyncConnection,
    ) -> None:
        """Generate embeddings for entry and store in database.

        Lazy-loads the embedding provider on first call.
        Truncates text to model's max input tokens to prevent API failures.

        Args:
            entry: The entry to enhance
            conn: Database connection from pool
        """
        if not self._models:
            logger.warning("No embedding models configured, skipping text embedding")
            return

        provider = self._get_provider()
        raw_text = entry.get("raw_text", "")

        if not raw_text.strip():
            logger.debug(f"Skipping empty entry {entry.get('entry_id')}")
            return

        for model_config in self._models:
            try:
                model_name = model_config["name"]

                # Truncate to model's max input (conservative estimate: 4 chars/token)
                max_tokens = model_config.get("max_input_tokens", 8192)
                max_chars = max_tokens * CHARS_PER_TOKEN
                text = raw_text[:max_chars]

                # Get base_url from provider config
                base_url = self._provider_config.get(
                    "base_url",
                    provider.default_base_url,
                )

                # Generate embedding
                embeddings = provider.execute_embedding(
                    texts=[text],
                    model_id=model_name,
                    base_url=base_url,
                )

                if embeddings and len(embeddings) > 0:
                    await self._store_embedding(
                        entry_id=entry["entry_id"],
                        model_name=model_name,
                        embedding=embeddings[0],
                        conn=conn,
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for entry {entry.get('entry_id')} "
                    f"with model {model_config.get('name')}: {e}"
                )
                continue

    async def _store_embedding(
        self,
        entry_id: str,
        model_name: str,
        embedding: list[float],
        conn: AsyncConnection,
    ) -> None:
        """Store embedding in model-specific table.

        Args:
            entry_id: Entry ID
            model_name: Model name for table lookup
            embedding: Embedding vector
            conn: Database connection
        """
        table_name = model_to_table_name(model_name)

        await conn.execute(
            f"""
            INSERT INTO {table_name} (entry_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (entry_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            """,  # noqa: S608
            [entry_id, embedding],
        )

    async def health_check(self) -> tuple[bool, str]:
        """Check if module is ready.

        Verifies that the embedding provider is accessible.

        Returns:
            Tuple of (healthy, message)
        """
        try:
            provider = self._get_provider()
            base_url = self._provider_config.get(
                "base_url",
                provider.default_base_url,
            )
            return provider.check_health(
                api_key=None,
                base_url=base_url,
            )
        except Exception as e:
            return (False, str(e))
