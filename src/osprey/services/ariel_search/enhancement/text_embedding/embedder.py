"""ARIEL text embedding module.

This module generates text embeddings for logbook entries to enable semantic search.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement._offload import run_blocking
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule, HealthResult
from osprey.services.ariel_search.enhancement.provider_resolver import (
    ResolvedProvider,
    resolve_provider,
    resolve_reachable_base_url,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.models.providers.base import BaseProvider
    from osprey.services.ariel_search.database.migrations import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

#: Input limit, in tokens, used for a model whose entry names no ``max_input_tokens``. It is
#: the smallest window among the models the Ollama provider lists (``all-minilm`` and
#: ``mxbai-embed-large`` serve 512), so an unstated limit is never larger than the model's.
DEFAULT_MAX_INPUT_TOKENS = 512

#: Config key the module's provider comes from when ``configure`` is given none.
_PROVIDER_KEY = "ariel.enhancement_modules.text_embedding.provider"

#: Tokens held back from a model's input limit for the special tokens a tokenizer wraps every
#: input in (``nomic-embed-text`` adds two).
RESERVED_TOKENS = 8


def max_input_tokens(model_config: Mapping[str, Any]) -> int:
    """Return the input limit, in tokens, of one configured embedding model.

    Args:
        model_config: One entry of ``ariel.enhancement_modules.text_embedding.models``.

    Returns:
        The entry's ``max_input_tokens``, or ``DEFAULT_MAX_INPUT_TOKENS`` when it is absent or
        null.

    Raises:
        ValueError: If the limit is not an integer greater than ``RESERVED_TOKENS``.
    """
    value = model_config.get("max_input_tokens")
    if value is None:
        return DEFAULT_MAX_INPUT_TOKENS
    if isinstance(value, int) and not isinstance(value, bool) and value > RESERVED_TOKENS:
        return value
    raise ValueError(
        "ariel.enhancement_modules.text_embedding.models: max_input_tokens for "
        f"{model_config.get('name')!r} must be an integer greater than {RESERVED_TOKENS} "
        f"(got {value!r})"
    )


def fit_to_input_limit(text: str, max_input_tokens: int) -> str:
    """Cut a text so it fits an embedding model's input limit, keeping its start.

    The returned text is at most ``max_input_tokens - RESERVED_TOKENS`` UTF-8 bytes long. A
    tokenizer emits at most one token per character (WordPiece, measured on
    ``nomic-embed-text`` with punctuation, digits, CJK, accented and compatibility characters)
    or one per byte (byte-level BPE), so a cut in UTF-8 bytes fits every text, whereas a
    characters-per-token ratio fits only the text it was measured on.

    Args:
        text: The text to embed.
        max_input_tokens: The model's input limit, in tokens.

    Returns:
        ``text`` itself when it fits, otherwise its longest prefix that fits. A character the
        cut would split is dropped whole.
    """
    budget = max_input_tokens - RESERVED_TOKENS
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[:budget].decode("utf-8", errors="ignore")


def embedding_input(raw_text: str, attachment_text: str | None, max_input_tokens: int) -> str:
    """Build the text an entry is embedded from: its own text, then its picture text.

    With no picture text this is exactly ``fit_to_input_limit(raw_text, max_input_tokens)``.
    Otherwise the picture text, cut to a quarter of the byte budget, is appended after a
    newline and the entry's text is cut to the room left, so the whole input stays within
    ``max_input_tokens - RESERVED_TOKENS`` UTF-8 bytes and a long entry fills it exactly.

    Args:
        raw_text: The entry's text.
        attachment_text: The entry's composed picture text, or ``None``.
        max_input_tokens: The model's input limit, in tokens.

    Returns:
        The embedding input. A character a cut would split is dropped whole.
    """
    att = attachment_text or ""
    if not att.strip():
        return fit_to_input_limit(raw_text, max_input_tokens)
    budget = max_input_tokens - RESERVED_TOKENS
    att = _utf8_prefix(att, budget // 4)
    room = budget - len(att.encode("utf-8")) - 1
    return _utf8_prefix(raw_text, room) + "\n" + att


def _utf8_prefix(text: str, budget: int) -> str:
    """Return the longest prefix of ``text`` that is at most ``budget`` UTF-8 bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[: max(budget, 0)].decode("utf-8", errors="ignore")


class TextEmbeddingModule(BaseEnhancementModule):
    """Generate text embeddings for logbook entries.

    Supports multiple embedding models, each with its own dedicated table.
    The provider name references api.providers for api_key and base_url.
    """

    def __init__(self) -> None:
        """Initialize the module."""
        self._provider: BaseProvider | None = None
        self._resolved: ResolvedProvider | None = None
        self._reachable_base_url: str | None = None
        self._models: list[dict[str, Any]] = []
        self._provider_key: str = _PROVIDER_KEY
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

        Resolves the provider through :func:`resolve_provider` (default
        ``ollama``) without touching the network: a server that is down still
        configures, and :meth:`health_check` reports it.

        Args:
            config: The enhancement_modules.text_embedding config dict
                   containing 'provider' (provider name string or inline config dict),
                   'provider_key' (the config key the provider came from) and
                   'models' list.

        Raises:
            ValueError: If a model's ``max_input_tokens`` is malformed.
            ModuleConfigError: If the provider is unknown, serves no embeddings,
                or its adapter refuses the base URL (the message names
                ``provider_key``).
        """
        self._models = config.get("models", [])
        for model_config in self._models:
            max_input_tokens(model_config)
        self._provider_key = config.get("provider_key") or _PROVIDER_KEY
        self._provider = None
        self._reachable_base_url = None
        self._resolved = None
        self._resolved = self._resolve(config.get("provider"))

    def _resolve(self, provider: str | dict[str, Any] | None) -> ResolvedProvider:
        """Resolve the module's provider (default ``ollama``) under its config key."""
        return resolve_provider(provider, provider_key=self._provider_key, default="ollama")

    def _resolution(self) -> ResolvedProvider:
        """The provider resolved in ``configure`` (the default resolved now if it was not)."""
        if self._resolved is None:
            self._resolved = self._resolve(None)
        return self._resolved

    def _get_provider(self) -> BaseProvider:
        """Return the embedding provider adapter, instantiated once.

        Returns:
            The adapter instance of the class resolved in ``configure``.
        """
        if self._provider is None:
            self._provider = self._resolution().instance

        return self._provider

    def _resolved_class(self) -> type[BaseProvider]:
        """The provider class resolved in ``configure``."""
        return self._resolution().cls

    def _base_url(self, *, refresh: bool = False) -> str | None:
        """The base URL embedding calls use.

        The configured URL, swapped for the reachable one for a provider that
        resolves its fallback outside calls (blocking on the first call, then
        cached; ``refresh`` re-checks it).
        """
        resolved = self._resolution()
        if resolved.base_url is None:
            return None
        if self._reachable_base_url is None or refresh:
            self._reachable_base_url = resolve_reachable_base_url(
                resolved.cls, resolved.base_url, refresh=refresh
            )
        return self._reachable_base_url

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
        Embeds the entry's text followed by its picture text (``attachment_text``), cut to
        the model's input limit so the start of each is embedded.

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
        attachment_text = entry.get("attachment_text")

        if not raw_text.strip() and not (attachment_text or "").strip():
            logger.debug(f"Skipping empty entry {entry.get('entry_id')}")
            return

        errors: list[str] = []
        for model_config in self._models:
            try:
                model_name = model_config["name"]

                limit = max_input_tokens(model_config)
                text = embedding_input(raw_text, attachment_text, limit)
                if not attachment_text and text is not raw_text:
                    logger.info(
                        "Entry %s is %d characters; only the first %d were embedded with %s "
                        "(its input limit is %d tokens)",
                        entry.get("entry_id"),
                        len(raw_text),
                        len(text),
                        model_name,
                        limit,
                    )

                api_key = self._resolution().api_key
                embed_kwargs: dict[str, Any] = {}
                if self._resolved_class().truncates_to_dimensions:
                    embed_kwargs["dimensions"] = model_config["dimension"]

                embeddings = provider.execute_embedding(
                    texts=[text],
                    model_id=model_name,
                    base_url=self._base_url(),
                    api_key=api_key,
                    **embed_kwargs,
                )

                if embeddings and len(embeddings) > 0:
                    await self._store_embedding(
                        entry_id=entry["entry_id"],
                        model_name=model_name,
                        embedding=embeddings[0],
                        conn=conn,
                    )
                else:
                    errors.append(f"{model_name}: empty embedding result")

            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for entry {entry.get('entry_id')} "
                    f"with model {model_config.get('name')}: {e}"
                )
                errors.append(f"{model_config.get('name')}: {e}")
                continue

        if errors and len(errors) == len(self._models):
            raise RuntimeError(
                f"All embedding models failed for entry {entry.get('entry_id')}: "
                + "; ".join(errors)
            )

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
            """,
            [entry_id, embedding],
        )

    async def health_check(self) -> HealthResult:
        """Check if module is ready.

        Asks the embedding provider for its verdict on the first configured
        model (the adapter's own health model when none is), off the event loop.

        Returns:
            The adapter's ``check_embedding_health`` verdict, unchanged; a probe
            that raises reports the reason its exception classifies to, else
            ``unreachable``.
        """
        return await run_blocking(self._probe_health)

    def _probe_health(self) -> HealthResult:
        """The blocking half of :meth:`health_check`."""
        from osprey.models.providers.health import failure_reason

        try:
            provider = self._get_provider()
            return provider.check_embedding_health(
                api_key=self._resolution().api_key,
                base_url=self._base_url(refresh=True),
                model_id=self._models[0]["name"] if self._models else None,
            )
        except Exception as e:
            return HealthResult(False, str(e), failure_reason(e) or "unreachable")
