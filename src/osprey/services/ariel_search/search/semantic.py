"""ARIEL semantic search module.

This module provides embedding-based similarity search using pgvector.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.search.base import (
    ModuleOutput,
    ParameterDescriptor,
    SearchToolDescriptor,
    module_result,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.search.base import QueryExpansion

logger = get_logger("ariel")

DEFAULT_SIMILARITY_THRESHOLD = 0.5

#: Config block the semantic knobs are read from.
_SETTINGS_PREFIX = "search_modules.semantic.settings"


@dataclass(frozen=True)
class SemanticSearchSettings:
    """Query knobs read from ``search_modules.semantic.settings``.

    Attributes:
        similarity_threshold: Minimum cosine similarity a row must reach to be
            returned. Defaults to :data:`DEFAULT_SIMILARITY_THRESHOLD`.
    """

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    @classmethod
    def from_ariel_config(cls, config: ARIELConfig | None) -> SemanticSearchSettings:
        """Read the module's ``settings`` block, defaults filled in.

        An absent block is the normal case and yields the defaults. A *present*
        key of the wrong type is refused rather than defaulted, for the same
        reason the keyword and hybrid modules refuse one: a threshold written
        as ``"0.8"`` used to travel unexamined into the query and into the
        capabilities slider, where it reads as an empty box rather than as the
        configuration error it is. Naming the key is the cheaper diagnosis.

        Args:
            config: The loaded ARIEL configuration, or ``None``.

        Returns:
            The resolved settings.

        Raises:
            ValueError: If ``similarity_threshold`` is present but not a number
                within ``[0, 1]``.
        """
        module = config.search_modules.get("semantic") if config is not None else None
        settings = module.settings if module is not None else None
        if not isinstance(settings, dict):
            return cls()

        threshold = settings.get("similarity_threshold", cls.similarity_threshold)
        # ``bool`` is a subclass of ``int``, so ``True`` would otherwise pass as
        # the number 1 — a spelling that means nothing here and is refused.
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                f"{_SETTINGS_PREFIX}.similarity_threshold must be a float in [0, 1], "
                f"got {threshold!r}"
            )

        return cls(similarity_threshold=float(threshold))


async def semantic_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    embedder: BaseEmbeddingProvider,
    *,
    max_results: int = 10,
    similarity_threshold: float | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    query_expansion: QueryExpansion | None = None,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float]] | ModuleOutput:
    """Execute semantic similarity search.

    Generates an embedding for the query and finds similar entries
    using cosine similarity.

    Args:
        query: Natural language query
        repository: ARIEL database repository
        config: ARIEL configuration
        embedder: Embedding provider (Ollama or other)
        max_results: Maximum entries to return (default: 10)
        similarity_threshold: Minimum similarity score (default: 0.5).
            Can be overridden per-query, then falls back to config,
            then to hardcoded default.
        start_date: Filter entries after this time
        end_date: Filter entries before this time
        author: Filter by author name (ILIKE match)
        source_system: Filter by source system (exact match)
        query_expansion: Resolved vocabulary expansion for this query, passed by
            the service only when one was actually resolved. When present, its
            `flattened_text` is embedded in place of `query` -- the whole query
            is the matching text for semantic search, and it is never truncated.

    Returns:
        Without `query_expansion`: the list of (entry, similarity_score) tuples
        sorted by similarity -- the bare-list contract every direct caller
        relies on today, returned unchanged.

        With `query_expansion`: a `ModuleOutput` carrying those same tuples as
        `entries` and the applied expansion groups as `expansion`. The shape is
        conditional so that the many direct callers of this function keep the
        return value they have always received; the service unwraps either form.
    """
    if not query.strip():
        return module_result([], query_expansion)

    logger.info(
        f"semantic_search: query={query!r}, max_results={max_results}, "
        f"threshold={similarity_threshold}, start_date={start_date}, end_date={end_date}"
    )

    semantic_config = config.search_modules.get("semantic")

    threshold = similarity_threshold
    if threshold is None:
        threshold = SemanticSearchSettings.from_ariel_config(config).similarity_threshold

    model_name = config.get_search_model()
    if not model_name:
        logger.warning("No semantic search model configured")
        return module_result([], query_expansion)

    # Priority: search module provider > embedding provider > default
    provider_name = (
        (semantic_config.provider if semantic_config else None)
        or config.embedding.provider
        or "ollama"
    )

    try:
        from osprey.models.config import get_provider_config

        provider_config = get_provider_config(provider_name)
    except FileNotFoundError:
        logger.debug(f"No config.yml found, using empty provider config for '{provider_name}'")
        provider_config = {}

    base_url = provider_config.get("base_url") or embedder.default_base_url
    api_key = provider_config.get("api_key")

    embed_text = query_expansion.flattened_text if query_expansion else query

    try:
        embeddings = embedder.execute_embedding(
            texts=[embed_text],
            model_id=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        if not embeddings or not embeddings[0]:
            logger.error("Failed to generate query embedding")
            return module_result([], query_expansion)

        query_embedding = embeddings[0]

        # Get expected dimension from config if available
        if semantic_config and semantic_config.settings:
            expected_dim = semantic_config.settings.get("embedding_dimension")
            if expected_dim and len(query_embedding) != expected_dim:
                logger.warning(
                    f"Embedding dimension mismatch: query embedding has "
                    f"{len(query_embedding)} dimensions but config expects "
                    f"{expected_dim}. This may cause incorrect similarity scores."
                )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return module_result([], query_expansion)

    results = await repository.semantic_search(
        query_embedding=query_embedding,
        model_name=model_name,
        max_results=max_results,
        similarity_threshold=threshold,
        start_date=start_date,
        end_date=end_date,
        author=author,
        source_system=source_system,
    )

    if len(results) == 0:
        try:
            embedding_tables = await repository.get_embedding_tables()
            all_empty = all(t.entry_count == 0 for t in embedding_tables)
            if not embedding_tables or all_empty:
                logger.warning(
                    "Semantic search found 0 results: no embeddings exist. "
                    "Run 'osprey ariel ingest' to generate embeddings."
                )
        except Exception:
            pass  # Don't let diagnostic check break the search path

    logger.info(f"semantic_search: returning {len(results)} results")
    return module_result(results, query_expansion)


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""

    query: str = Field(description="Natural language description of what to find")
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )
    expand_query: bool | None = Field(
        default=None,
        description=(
            "Apply the facility vocabulary expansion (shorthand/acronyms). "
            "None = the configured default (capabilities().vocabulary.expand_by_default)"
        ),
    )


def format_semantic_result(
    entry: EnhancedLogbookEntry,
    similarity: float,
) -> dict[str, Any]:
    """Format a semantic search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        similarity: Cosine similarity score

    Returns:
        Formatted dict for agent
    """
    from osprey.services.ariel_search.models import _format_entry_base

    return {**_format_entry_base(entry), "similarity": similarity}


def get_parameter_descriptors(config: ARIELConfig | None = None) -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API.

    The default reported here is the deployment's, not the shipped one: a panel
    that opened its slider on ``0.5`` while the deployment searched at ``0.8``
    would invite an operator to "leave the default alone" and get a looser
    search than the deployment's own. Reading it from the same
    ``search_modules.semantic.settings`` block the query path reads keeps the
    two in step.

    Describing the module never fails on bad config. ``/api/capabilities``, the
    search page and the MCP capabilities tool all walk this function, and a
    deployment with one malformed key should still see a described, usable
    module — a refusal propagating from here would leave the panel with a
    slider that has no default at all. The key itself is reported by startup
    validation, which is the surface that can explain it. So a refusal from the
    settings parser falls back to the shipped defaults here.

    Args:
        config: The loaded ARIEL configuration. ``None`` — the case for a
            caller that has no config in hand — yields the shipped defaults.

    Returns:
        One descriptor per tunable knob, in panel order.
    """
    try:
        settings = SemanticSearchSettings.from_ariel_config(config)
    except ValueError:
        settings = SemanticSearchSettings()

    return [
        ParameterDescriptor(
            name="similarity_threshold",
            label="Similarity Threshold",
            description="Minimum cosine similarity score for results (0-1)",
            param_type="float",
            default=settings.similarity_threshold,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            section="Retrieval",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="semantic_search",
        description=(
            "Find conceptually related entries using text embeddings. "
            "Use for queries describing concepts, situations, or events "
            "where exact words may not match."
        ),
        search_mode="semantic",
        args_schema=SemanticSearchInput,
        execute=semantic_search,
        format_result=format_semantic_result,
        needs_embedder=True,
        accepts_expansion=True,
    )
