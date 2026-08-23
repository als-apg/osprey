"""Vocabulary-expansion behaviour of the ARIEL semantic search module.

Semantic search matches on the whole query — there is nothing to parse and
nothing to truncate — so expansion here is one substitution: the embedder sees
`QueryExpansion.flattened_text` instead of the raw query. These tests pin that
substitution, the untruncated raw-query path, and the return shape each path
produces.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    QueryExpansion,
)
from osprey.services.ariel_search.search.semantic import get_tool_descriptor, semantic_search

AMBIGUOUS_GROUPS = (ExpansionGroup(original="ts", alternatives=("troubleshoot", "timing system")),)
AMBIGUOUS_EXPANSION = QueryExpansion(
    groups=AMBIGUOUS_GROUPS,
    flattened_text="ts fault troubleshoot timing system",
)


@pytest.fixture
def config() -> ARIELConfig:
    """A minimal config with semantic search enabled."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {"semantic": {"enabled": True, "model": "nomic-embed-text"}},
            "embedding": {"provider": "ollama", "base_url": "http://localhost:11434"},
        }
    )


@pytest.fixture
def repository(config):
    """A repository that answers every semantic query with no rows."""
    repo = MagicMock()
    repo.config = config
    repo.semantic_search = AsyncMock(return_value=[])
    repo.get_embedding_tables = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def embedder():
    """An embedding provider that records the texts it was handed."""
    provider = MagicMock()
    provider.default_base_url = "http://localhost:11434"
    provider.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return provider


def embedded_texts(embedder) -> list[str]:
    """Return the `texts` argument of the single embedding call."""
    embedder.execute_embedding.assert_called_once()
    return embedder.execute_embedding.call_args.kwargs["texts"]


class TestExpansionReachesTheEmbedder:
    """What the embedder is handed, with and without an expansion."""

    @pytest.mark.asyncio
    async def test_flattened_text_is_embedded(self, repository, config, embedder):
        """With an expansion, the embedder sees the flattened text, not the query."""
        await semantic_search(
            "ts fault",
            repository,
            config,
            embedder,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert embedded_texts(embedder) == ["ts fault troubleshoot timing system"]

    @pytest.mark.asyncio
    async def test_raw_query_is_embedded_without_expansion(self, repository, config, embedder):
        """Without an expansion the call is byte-identical to today's."""
        await semantic_search("ts fault", repository, config, embedder)

        assert embedded_texts(embedder) == ["ts fault"]

    @pytest.mark.asyncio
    async def test_long_query_reaches_the_embedder_untruncated(self, repository, config, embedder):
        """The 1000-char cap is keyword-only: semantic queries are never truncated."""
        query = "beam loss " * 120  # 1200 characters
        assert len(query) == 1200

        await semantic_search(query, repository, config, embedder)

        (embedded,) = embedded_texts(embedder)
        assert embedded == query
        assert len(embedded) == 1200

    @pytest.mark.asyncio
    async def test_long_expanded_query_reaches_the_embedder_untruncated(
        self, repository, config, embedder
    ):
        """Nor is the flattened text truncated when expansion is active."""
        query = "beam loss " * 120
        expansion = QueryExpansion(
            groups=(ExpansionGroup(original="bpm", alternatives=("beam position monitor",)),),
            flattened_text=f"{query} beam position monitor",
        )

        await semantic_search(query, repository, config, embedder, query_expansion=expansion)

        (embedded,) = embedded_texts(embedder)
        assert embedded == expansion.flattened_text
        assert len(embedded) > 1200


class TestReturnShape:
    """The conditional return shape: bare list today, ModuleOutput on expansion."""

    @pytest.mark.asyncio
    async def test_without_expansion_a_bare_list_is_returned(self, repository, config, embedder):
        """Direct callers keep the list they have always received."""
        result = await semantic_search("ts fault", repository, config, embedder)

        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    async def test_with_expansion_a_module_output_carries_the_groups(
        self, repository, config, embedder
    ):
        """The applied expansion comes back for the service to surface."""
        result = await semantic_search(
            "ts fault",
            repository,
            config,
            embedder,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.expansion == AMBIGUOUS_GROUPS
        assert result.entries == []
        assert result.diagnostics == ()

    @pytest.mark.asyncio
    async def test_module_output_entries_are_the_search_results(self, repository, config, embedder):
        """`entries` holds exactly what the bare-list return would have held."""
        rows = [({"entry_id": "1"}, 0.9), ({"entry_id": "2"}, 0.7)]
        repository.semantic_search = AsyncMock(return_value=rows)

        result = await semantic_search(
            "ts fault",
            repository,
            config,
            embedder,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.entries == rows

    @pytest.mark.asyncio
    async def test_early_returns_keep_the_shape(self, repository, config, embedder):
        """A path that produces no results still answers in the requested shape."""
        result = await semantic_search(
            "   ",
            repository,
            config,
            embedder,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.entries == []
        embedder.execute_embedding.assert_not_called()


class TestDescriptor:
    """What the module advertises to the service."""

    def test_descriptor_opts_into_expansion(self):
        descriptor = get_tool_descriptor()

        assert descriptor.accepts_expansion is True

    def test_descriptor_declares_no_query_parser(self):
        """Semantic search matches whole text — there is nothing to parse."""
        assert get_tool_descriptor().query_parser is None
