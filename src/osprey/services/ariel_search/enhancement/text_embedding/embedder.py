"""ARIEL text embedding module.

This module generates text embeddings for logbook entries to enable semantic search.

See 01_DATA_LAYER.md Section 6.3 for specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.migration import model_to_table_name
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.database.migration import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

# Default characters per token estimate (conservative)
CHARS_PER_TOKEN = 4


class TextEmbeddingModule(BaseEnhancementModule):
    """Generate text embeddings for logbook entries.

    Supports multiple embedding models, each with its own dedicated table.
    Follows Osprey's zero-argument constructor pattern with lazy loading
    of expensive resources.

    Uses Osprey's provider configuration system for credentials.
    The provider name references api.providers for api_key and base_url.
    """

    def __init__(self) -> None:
        """Initialize the module.

        Zero-argument constructor (Osprey pattern).
        Expensive resources (embedding provider) are lazy-loaded.
        """
        self._provider: BaseEmbeddingProvider | None = None
        self._models: list[dict[str, Any]] = []
        self._provider_name: str = "ollama"
        self._resolved_provider_config: dict[str, Any] = {}
        self._tables_exist: bool | None = None  # Cached result of table existence check

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
                   containing 'provider' (provider name string or inline config dict)
                   and 'models' list.
        """
        self._models = config.get("models", [])
        provider_config = config.get("provider", "ollama")

        # Handle both provider name (string) and inline config (dict)
        if isinstance(provider_config, dict):
            # Inline provider config - use directly
            self._provider_name = provider_config.get("name", "ollama")
            self._resolved_provider_config = provider_config
            return

        # Provider name - resolve via Osprey's config system
        self._provider_name = provider_config

        # Resolve provider credentials via Osprey's config system
        # This may fail in test environments without config.yml
        try:
            from osprey.utils.config import get_provider_config

            self._resolved_provider_config = get_provider_config(self._provider_name)
        except FileNotFoundError:
            # Test environment without config.yml - use empty config
            logger.debug(
                f"No config.yml found, using empty provider config for '{self._provider_name}'"
            )
            self._resolved_provider_config = {}

    def _get_provider(self) -> BaseEmbeddingProvider:
        """Lazy-load and return the embedding provider.

        Returns:
            Configured embedding provider instance based on provider_name
        """
        if self._provider is None:
            # Dynamic provider selection based on config
            if self._provider_name == "ollama":
                from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

                self._provider = OllamaEmbeddingProvider()
            else:
                # For other providers, default to Ollama for now
                # Additional providers can be added here as they are implemented
                logger.warning(
                    f"Embedding provider '{self._provider_name}' not yet supported, "
                    f"falling back to 'ollama'"
                )
                from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

                self._provider = OllamaEmbeddingProvider()

        return self._provider

    async def _check_tables_exist(self, conn: AsyncConnection) -> bool:
        """Check if embedding tables exist (cached after first call).

        Returns:
            True if at least one configured model's table exists
        """
        if self._tables_exist is not None:
            return self._tables_exist

        for model_config in self._models:
            table_name = model_to_table_name(model_config["name"])
            result = await conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                [table_name],
            )
            row = await result.fetchone()
            if row and row[0]:
                self._tables_exist = True
                return True

        logger.warning(
            "Embedding tables do not exist (pgvector migration was skipped). "
            "Skipping text embedding enhancement."
        )
        self._tables_exist = False
        return False

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

        if not await self._check_tables_exist(conn):
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
                max_tokens = model_config.get("max_input_tokens") or 8192
                max_chars = max_tokens * CHARS_PER_TOKEN
                text = raw_text[:max_chars]

                # Get credentials from resolved provider config
                base_url = self._resolved_provider_config.get(
                    "base_url",
                    provider.default_base_url,
                )
                api_key = self._resolved_provider_config.get("api_key")

                # Generate embedding
                embeddings = provider.execute_embedding(
                    texts=[text],
                    model_id=model_name,
                    base_url=base_url,
                    api_key=api_key,
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
            base_url = self._resolved_provider_config.get(
                "base_url",
                provider.default_base_url,
            )
            api_key = self._resolved_provider_config.get("api_key")
            return provider.check_health(
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            return (False, str(e))
