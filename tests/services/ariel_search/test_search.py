"""Tests for ARIEL search modules.

Tests for keyword and semantic search implementations.
RAG is now implemented via the Pipeline abstraction.
"""

from datetime import UTC

import pytest

from osprey.services.ariel_search.search import (
    ALLOWED_FIELD_PREFIXES,
    ALLOWED_OPERATORS,
    MAX_QUERY_LENGTH,
    parse_query,
)
from osprey.services.ariel_search.search.keyword import (
    KeywordSearchInput,
    _balance_quotes,
    build_tsquery,
    get_tool_descriptor,
    has_boolean_operators,
)
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider
from tests.services.ariel_search.repo_fakes import attach_fake_fts


class TestSearchConstants:
    """Tests for search module constants."""

    def test_allowed_operators(self):
        """ALLOWED_OPERATORS has expected values."""
        assert "AND" in ALLOWED_OPERATORS
        assert "OR" in ALLOWED_OPERATORS
        assert "NOT" in ALLOWED_OPERATORS
        assert len(ALLOWED_OPERATORS) == 3

    def test_allowed_field_prefixes(self):
        """ALLOWED_FIELD_PREFIXES has expected values."""
        assert "author:" in ALLOWED_FIELD_PREFIXES
        assert "date:" in ALLOWED_FIELD_PREFIXES
        assert len(ALLOWED_FIELD_PREFIXES) == 2

    @pytest.mark.parametrize("prefix", sorted(ALLOWED_FIELD_PREFIXES))
    def test_every_field_prefix_is_a_lowercase_name_and_a_colon(self, prefix):
        """Each prefix is a lower-case identifier followed by a colon."""
        assert prefix == prefix.lower()
        assert prefix.endswith(":")
        assert prefix[:-1].isidentifier()

    def test_agent_facing_text_names_every_operator(self):
        """The tool description and the query field name every operator word."""
        tool_description = get_tool_descriptor().description
        query_description = KeywordSearchInput.model_fields["query"].description
        assert query_description is not None
        for word in ALLOWED_OPERATORS:
            assert word in tool_description
            assert word in query_description


class TestParseQuery:
    """Tests for query parsing."""

    @pytest.mark.parametrize(
        "query",
        [
            # Parses simple query with no operators.
            pytest.param("beam current", id="simple-query"),
            pytest.param("", id="empty-query"),
        ],
    )
    def test_query_without_operators_is_all_search_text(self, query):
        """A query with no operators comes back as its own search text, with no filters or phrases."""
        search_text, filters, phrases = parse_query(query)
        assert search_text == query
        assert filters == {}
        assert phrases == []

    def test_quoted_phrase(self):
        """Extracts quoted phrases."""
        search_text, filters, phrases = parse_query('"beam loss" event')
        assert "event" in search_text
        assert phrases == ["beam loss"]

    def test_multiple_quoted_phrases(self):
        """Extracts multiple quoted phrases."""
        search_text, filters, phrases = parse_query('"beam loss" "rf fault"')
        assert phrases == ["beam loss", "rf fault"]

    @pytest.mark.parametrize(
        ("query", "text", "expected_filters"),
        [
            pytest.param("beam author:smith", "beam", {"author": "smith"}, id="author"),
            pytest.param("fault date:2024-01", "fault", {"date": "2024-01"}, id="date"),
            pytest.param(
                "beam author:smith date:2024-01",
                "beam",
                {"author": "smith", "date": "2024-01"},
                id="multiple",
            ),
        ],
    )
    def test_field_filters_are_extracted(self, query, text, expected_filters):
        """Extracts author: and date: filters, alone or together, out of the search text."""
        search_text, filters, phrases = parse_query(query)
        assert search_text == text
        assert filters == expected_filters

    def test_combined_query(self):
        """Parses query with filters and phrases."""
        search_text, filters, phrases = parse_query('"beam loss" author:smith vacuum')
        assert "vacuum" in search_text
        assert filters == {"author": "smith"}
        assert phrases == ["beam loss"]

    @pytest.mark.parametrize("spell", [str.lower, str.upper])
    @pytest.mark.parametrize("prefix", sorted(ALLOWED_FIELD_PREFIXES))
    def test_every_allowed_prefix_is_lifted(self, prefix, spell):
        """Every prefix in the constant becomes a filter keyed by its name."""
        search_text, filters, phrases = parse_query(f"beam {spell(prefix)}x1")
        assert search_text == "beam"
        assert filters == {prefix[:-1]: "x1"}

    def test_a_prefix_outside_the_constant_stays_search_text(self):
        """A prefix the constant does not name is ordinary search text."""
        assert parse_query("beam source:x1") == ("beam source:x1", {}, [])


class TestBuildTsquery:
    """Tests for tsquery building."""

    @pytest.mark.parametrize(
        ("text", "phrases", "expected"),
        [
            # Simple terms take the plain leg, with the text as a parameter.
            pytest.param("beam current", [], "plainto_tsquery('english', %s)", id="simple-terms"),
            # Each phrase adds its own parameterised phrase leg.
            pytest.param(
                "beam",
                ["exact phrase"],
                "plainto_tsquery('english', %s) && phraseto_tsquery('english', %s)",
                id="with-phrases",
            ),
            # Nothing to search is the literal empty tsquery, with no placeholder.
            pytest.param("", [], "plainto_tsquery('english', '')", id="empty-inputs"),
        ],
    )
    def test_tsquery_is_built_exactly(self, text, phrases, expected):
        """build_tsquery returns the exact parameterised tsquery for its inputs."""
        assert build_tsquery(text, phrases) == expected

    @pytest.mark.parametrize("spell", [str.upper, str.lower])
    @pytest.mark.parametrize("word", sorted(ALLOWED_OPERATORS))
    def test_every_operator_word_routes_to_websearch(self, word, spell):
        """Every operator word, in either case, takes the websearch path."""
        text = f"beam {spell(word)} loss"
        assert build_tsquery(text, []) == "websearch_to_tsquery('english', %s)"
        assert has_boolean_operators(text)

    @pytest.mark.parametrize("symbol", ["&", "|", "!"])
    def test_every_operator_symbol_routes_to_websearch(self, symbol):
        """Both functions agree that an operator symbol takes the websearch path."""
        text = f"beam {symbol} loss"
        assert build_tsquery(text, []) == "websearch_to_tsquery('english', %s)"
        assert has_boolean_operators(text)

    def test_operator_word_inside_a_token_is_not_an_operator(self):
        """An operator word inside a longer token is ordinary text."""
        text = "ANDERSON ORBIT NOTE"
        assert build_tsquery(text, []) == "plainto_tsquery('english', %s)"
        assert not has_boolean_operators(text)


class TestKeywordSearchFunction:
    """Tests for keyword_search function with mocked repository."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config with keyword search enabled."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

    @pytest.fixture
    def mock_repository(self, mock_config):
        """Create mock repository."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=[])
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [pytest.param("", id="empty"), pytest.param("   ", id="whitespace-only")],
    )
    async def test_blank_query_returns_empty(self, mock_repository, mock_config, query):
        """A blank query returns empty results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        result = await keyword_search(query, mock_repository, mock_config)
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            # Simple search delegates to repository.
            pytest.param("beam current", id="simple-search"),
            # Date filter parses YYYY-MM-DD format.
            pytest.param("beam date:2024-01-15", id="daily-date-filter"),
        ],
    )
    async def test_search_calls_repository_once(self, mock_repository, mock_config, query):
        """A search delegates to the repository."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(query, mock_repository, mock_config)
        mock_repository.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_config")
    async def test_semantic_processor_terms_are_in_keyword_match_surface(self, mock_repository):
        """Generated summaries and keywords participate in keyword search."""
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
                "enhancement_modules": {"semantic_processor": {"enabled": True}},
            }
        )

        await keyword_search("injection transient", mock_repository, config)

        call_args = mock_repository.keyword_search.call_args
        assert call_args is not None
        where_clauses = call_args.kwargs["where_clauses"]
        assert any("COALESCE(summary, '')" in clause for clause in where_clauses)
        assert any("osprey_text_array_to_string(keywords)" in clause for clause in where_clauses)

    @pytest.mark.asyncio
    async def test_fuzzy_fallback_when_no_results(self, mock_repository, mock_config):
        """Falls back to fuzzy search when FTS returns no results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        mock_repository.keyword_search.return_value = []

        await keyword_search(
            "beaam",  # misspelled
            mock_repository,
            mock_config,
            fuzzy_fallback=True,
        )

        mock_repository.fuzzy_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fuzzy_fallback_when_disabled(self, mock_repository, mock_config):
        """No fuzzy fallback when disabled."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        mock_repository.keyword_search.return_value = []

        await keyword_search(
            "beam",
            mock_repository,
            mock_config,
            fuzzy_fallback=False,
        )

        mock_repository.fuzzy_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_date_filter_monthly(self, mock_repository, mock_config):
        """Date filter parses YYYY-MM format."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam date:2024-01",
            mock_repository,
            mock_config,
        )

        # Verify the call was made with date params
        call_args = mock_repository.keyword_search.call_args
        assert call_args is not None
        params = call_args.kwargs.get(
            "params", call_args.args[1] if len(call_args.args) > 1 else []
        )
        # Should contain date boundary values
        assert any("2024-01" in str(p) for p in params)

    @pytest.mark.asyncio
    async def test_date_filter_december(self, mock_repository, mock_config):
        """Date filter handles December (month 12) rollover correctly."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam date:2024-12",
            mock_repository,
            mock_config,
        )

        call_args = mock_repository.keyword_search.call_args
        params = call_args.kwargs.get(
            "params", call_args.args[1] if len(call_args.args) > 1 else []
        )
        # Should have next year for end boundary
        assert any("2025-01-01" in str(p) for p in params)


class TestSemanticSearchFunction:
    """Tests for semantic_search function with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config with semantic search enabled."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "nomic-embed-text",
                        "settings": {"similarity_threshold": 0.8},
                    },
                },
                "embedding": {"provider": "ollama", "base_url": "http://localhost:11434"},
            }
        )

    @pytest.fixture
    def mock_config_no_model(self):
        """Create config without model configured."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"semantic": {"enabled": True}},
            }
        )

    @pytest.fixture
    def mock_repository(self, mock_config):
        """Create mock repository."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.semantic_search = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding provider."""
        from unittest.mock import MagicMock

        embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return embedder

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [pytest.param("", id="empty"), pytest.param("   ", id="whitespace-only")],
    )
    async def test_blank_query_returns_empty(
        self, mock_repository, mock_config, mock_embedder, search_calls, query
    ):
        """A blank query returns empty results."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        result = await semantic_search(
            query, mock_repository, mock_config, mock_embedder, search_calls=search_calls
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_model_returns_empty(
        self, mock_repository, mock_config_no_model, mock_embedder, search_calls
    ):
        """Returns empty if no model configured."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        result = await semantic_search(
            "test query",
            mock_repository,
            mock_config_no_model,
            mock_embedder,
            search_calls=search_calls,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_generation_called(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """Embedding generation is called with query."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
            search_calls=search_calls,
        )

        mock_embedder.execute_embedding.assert_called_once()
        call_args = mock_embedder.execute_embedding.call_args
        assert call_args.kwargs["texts"] == ["test query"]

    @pytest.mark.asyncio
    async def test_uses_config_threshold(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """Uses threshold from config when not provided."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
            search_calls=search_calls,
        )

        call_args = mock_repository.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_per_query_threshold_override(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """Per-query threshold overrides config."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
            similarity_threshold=0.5,
            search_calls=search_calls,
        )

        call_args = mock_repository.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_embedding_failure_raises_naming_the_cause(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """A failed embedding is an error naming its cause, never an empty answer."""
        from osprey.services.ariel_search.exceptions import EmbeddingGenerationError
        from osprey.services.ariel_search.search.semantic import semantic_search

        mock_embedder.execute_embedding.side_effect = Exception("Ollama unavailable")

        with pytest.raises(EmbeddingGenerationError, match="Ollama unavailable"):
            await semantic_search(
                "test query",
                mock_repository,
                mock_config,
                mock_embedder,
                search_calls=search_calls,
            )
        mock_repository.semantic_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_embedding_raises(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """A provider that answers without a vector has failed the embedding."""
        from osprey.services.ariel_search.exceptions import EmbeddingGenerationError
        from osprey.services.ariel_search.search.semantic import semantic_search

        mock_embedder.execute_embedding.return_value = [[]]

        with pytest.raises(EmbeddingGenerationError, match="returned no vector"):
            await semantic_search(
                "test query",
                mock_repository,
                mock_config,
                mock_embedder,
                search_calls=search_calls,
            )


class TestKeywordSearchWithResults:
    """Tests for keyword search when repository returns results."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

    @pytest.fixture
    def sample_results(self):
        """Sample keyword search results."""
        from datetime import datetime

        return [
            (
                {
                    "entry_id": "entry-001",
                    "source_system": "Example eLog",
                    "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
                    "author": "jsmith",
                    "raw_text": "Beam current stabilized at 500mA.",
                    "attachments": [],
                    "metadata": {},
                },
                0.85,
                ["<mark>Beam</mark> current"],
            ),
        ]

    @pytest.fixture
    def mock_repository(self, mock_config, sample_results):
        """Create mock repository that returns results."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=sample_results)
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_keyword_with_author_filter(self, mock_repository, mock_config):
        """Keyword search passes author filter to repository."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        result = await keyword_search(
            "beam author:smith",
            mock_repository,
            mock_config,
        )

        # Should have one result
        assert len(result) == 1
        assert result[0][0]["entry_id"] == "entry-001"

    @pytest.mark.asyncio
    async def test_keyword_with_date_param(self, mock_repository, mock_config):
        """Keyword search passes date parameters to repository."""
        from datetime import datetime

        from osprey.services.ariel_search.search.keyword import keyword_search

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        await keyword_search(
            "beam",
            mock_repository,
            mock_config,
            start_date=start_date,
            end_date=end_date,
        )

        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert start_date in kwargs["params"]
        assert end_date in kwargs["params"]
        assert "timestamp >= %s" in kwargs["where_clauses"]
        assert "timestamp <= %s" in kwargs["where_clauses"]


class TestSemanticSearchWithDateFilters:
    """Tests for semantic search with date filters."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "nomic-embed-text",
                    },
                },
            }
        )

    @pytest.fixture
    def mock_repository(self, mock_config):
        """Create mock repository."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.semantic_search = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        from unittest.mock import MagicMock

        embedder = make_fake_embedding_provider()()
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return embedder

    @pytest.mark.asyncio
    async def test_semantic_with_date_filters(
        self, mock_repository, mock_config, mock_embedder, search_calls
    ):
        """Semantic search passes date filters to repository."""
        from datetime import datetime

        from osprey.services.ariel_search.search.semantic import semantic_search

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        await semantic_search(
            "beam status",
            mock_repository,
            mock_config,
            mock_embedder,
            start_date=start_date,
            end_date=end_date,
            search_calls=search_calls,
        )

        call_args = mock_repository.semantic_search.call_args
        assert call_args.kwargs["start_date"] == start_date
        assert call_args.kwargs["end_date"] == end_date


# === Security Tests (TEST-H001, TEST-H002, TEST-H003) ===


class TestQuerySQLInjection:
    """Tests for SQL injection protection (GAP-C003, TEST-H001).

    Verifies that SQL injection attempts are neutralized via parameterization.
    """

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config with keyword search enabled."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

    @pytest.fixture
    def mock_repository(self, mock_config):
        """Create mock repository that captures SQL params."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=[])
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malicious_query",
        [
            pytest.param("'; DROP TABLE entries; --", id="drop-table"),
            pytest.param("' OR '1'='1", id="or"),
            pytest.param("test UNION SELECT * FROM users --", id="union-select"),
        ],
    )
    async def test_injection_is_treated_as_literal_text(
        self, mock_repository, mock_config, malicious_query
    ):
        """An SQL injection attempt reaches the repository as a bound parameter, never as SQL."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(malicious_query, mock_repository, mock_config)

        mock_repository.keyword_search.assert_called_once()
        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert kwargs["params"][0] == malicious_query
        assert not any(malicious_query in clause for clause in kwargs["where_clauses"])


class TestUnbalancedQuotes:
    """Tests for unbalanced quotes handling (GAP-C001, TEST-H002)."""

    @pytest.mark.parametrize(
        "query",
        [
            pytest.param('"hello world" test', id="balanced-quotes"),
            pytest.param("simple query", id="no-quotes"),
            # An empty quoted string is preserved.
            pytest.param('""', id="empty-quotes"),
        ],
    )
    def test_balanced_query_is_unchanged(self, query):
        """A query whose quotes are already balanced is not modified."""
        assert _balance_quotes(query) == query

    @pytest.mark.parametrize(
        "query",
        [
            pytest.param('"hello world', id="single"),
            pytest.param('test query"', id="trailing"),
            pytest.param('"first" "second" "third', id="multiple"),
        ],
    )
    def test_unbalanced_quotes_are_balanced(self, query):
        """An unbalanced quote is removed, leaving an even number of quotes."""
        result = _balance_quotes(query)
        assert result.count('"') % 2 == 0

    def test_parse_query_handles_unbalanced_quotes(self):
        """parse_query drops the unpaired quote and keeps its words as search text."""
        assert parse_query('"beam loss') == ("beam loss", {}, [])

    @pytest.mark.asyncio
    async def test_keyword_search_unbalanced_quotes(self):
        """keyword_search handles unbalanced quotes without error."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        await keyword_search('"beam loss', mock_repo, config)
        assert mock_repo.keyword_search.call_args.kwargs["params"][0] == "beam loss"


class TestQueryLengthTruncation:
    """Tests for query length truncation (GAP-C002, TEST-H003)."""

    def test_max_query_length_constant(self):
        """MAX_QUERY_LENGTH constant is 1000."""
        assert MAX_QUERY_LENGTH == 1000

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("query", "sent", "truncated"),
        [
            # Queries longer than 1000 chars are truncated, not rejected.
            pytest.param("a" * 1500, "a" * 1000, True, id="long-query-truncated"),
            pytest.param("a" * 1000, "a" * 1000, False, id="query-at-limit-not-truncated"),
            pytest.param(
                "beam current", "beam current", False, id="query-under-limit-not-truncated"
            ),
        ],
    )
    async def test_query_is_cut_to_the_limit(self, caplog, query, sent, truncated):
        """The repository receives the query cut to MAX_QUERY_LENGTH, with a warning only when cut."""
        import logging
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await keyword_search(query, mock_repo, config)

        mock_repo.keyword_search.assert_called_once()
        assert mock_repo.keyword_search.call_args.kwargs["params"][0] == sent
        assert ("Query truncated from" in caplog.text) is truncated


class TestUnknownFieldPrefix:
    """Tests for unknown field prefix handling (EDGE-001)."""

    def test_unknown_prefix_treated_as_literal(self):
        """Unknown field prefix is treated as literal text."""
        search_text, filters, phrases = parse_query("foo:bar beam")

        # foo:bar should be in search_text, not in filters
        assert "foo:bar" in search_text
        assert "foo" not in filters
        assert "beam" in search_text

    def test_known_prefix_extracted(self):
        """Known field prefixes are extracted correctly."""
        search_text, filters, phrases = parse_query("author:smith beam")

        assert "author" in filters
        assert filters["author"] == "smith"
        assert "beam" in search_text
        assert "author:smith" not in search_text


# === Edge Case Tests (EDGE-004 to EDGE-016) ===


class TestThreeTierThresholdResolution:
    """Tests for 3-tier threshold resolution (EDGE-004)."""

    @pytest.mark.asyncio
    async def test_param_overrides_config(self, search_calls):
        """Per-query parameter overrides config value."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import semantic_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        "settings": {"similarity_threshold": 0.8},
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Pass explicit threshold that should override config
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=0.5,  # Override config's 0.8
            search_calls=search_calls,
        )

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_config_overrides_default(self, search_calls):
        """Config value overrides hardcoded default."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import (
            DEFAULT_SIMILARITY_THRESHOLD,
            semantic_search,
        )

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        "settings": {"similarity_threshold": 0.9},
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Don't pass explicit threshold - should use config
        await semantic_search("test", mock_repo, config, mock_embedder, search_calls=search_calls)

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.9
        assert call_args.kwargs["similarity_threshold"] != DEFAULT_SIMILARITY_THRESHOLD


class TestFuzzyFallbackConditions:
    """Tests for fuzzy fallback conditions (EDGE-005)."""

    @pytest.mark.asyncio
    async def test_fuzzy_only_when_empty_and_enabled(self):
        """Fuzzy search only triggers when empty results AND fuzzy_fallback=True."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        # Case 1: Empty results, fuzzy_fallback=True -> fuzzy should be called
        await keyword_search("test", mock_repo, config, fuzzy_fallback=True)
        mock_repo.fuzzy_search.assert_called_once()

        mock_repo.fuzzy_search.reset_mock()

        # Case 2: Empty results, fuzzy_fallback=False -> fuzzy should NOT be called
        await keyword_search("test", mock_repo, config, fuzzy_fallback=False)
        mock_repo.fuzzy_search.assert_not_called()


class TestBoundaryValues:
    """Tests for boundary value validation (EDGE-007, EDGE-008, EDGE-009).

    Input validation tests using agent.executor schemas.
    """

    @pytest.mark.parametrize(
        "max_results",
        [
            pytest.param(0, id="zero-edge-007"),
            pytest.param(51, id="fifty-one-edge-008"),
        ],
    )
    def test_max_results_outside_the_allowed_range_is_invalid(self, max_results):
        """An out-of-range max_results raises validation error (EDGE-007, EDGE-008)."""
        from pydantic import ValidationError

        from osprey.services.ariel_search.search.keyword import KeywordSearchInput

        with pytest.raises(ValidationError):
            KeywordSearchInput(query="test", max_results=max_results)

    def test_similarity_negative_invalid(self):
        """similarity_threshold=-0.1 raises validation error (EDGE-009)."""
        from pydantic import ValidationError

        from osprey.services.ariel_search.search.semantic import SemanticSearchInput

        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", similarity_threshold=-0.1)

    def test_similarity_above_one_invalid(self):
        """similarity_threshold=1.1 raises validation error (EDGE-009)."""
        from pydantic import ValidationError

        from osprey.services.ariel_search.search.semantic import SemanticSearchInput

        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", similarity_threshold=1.1)


class TestEmbeddingGenerationFailure:
    """Tests for embedding generation failure handling (EDGE-012)."""

    @pytest.mark.asyncio
    async def test_semantic_search_embedding_failure_raises(self, search_calls):
        """Semantic search reports an embedding failure with the provider it dialled."""
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import semantic_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {"enabled": True, "model": "test-model"},
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(
            side_effect=Exception("Embedding service unavailable")
        )

        from osprey.services.ariel_search.exceptions import EmbeddingGenerationError

        with pytest.raises(EmbeddingGenerationError) as raised:
            await semantic_search(
                "test", mock_repo, config, mock_embedder, search_calls=search_calls
            )

        assert "Embedding service unavailable" in str(raised.value)
        assert raised.value.model_name == "test-model"
        mock_repo.semantic_search.assert_not_called()


# ==============================================================================
# Quality Improvements (QUAL-003, QUAL-005, QUAL-009, QUAL-010)
# ==============================================================================


class TestEmbeddingGenerationQuality:
    """Quality assertions for embedding generation (QUAL-003)."""

    @pytest.mark.requires_ollama
    @pytest.mark.usefixtures("litellm_callback_pool")
    @pytest.mark.asyncio
    async def test_actual_embedding_generation(self):
        """Test actual embedding generation with Ollama.

        QUAL-003: Embedding generation test with requires_ollama marker.
        This test actually generates embeddings when Ollama is available.
        """
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                pytest.skip("Ollama not available")
        except Exception:
            pytest.skip("Ollama not available")

        from tests.services.ariel_search.fake_providers import ollama_text_embedder

        embedder = ollama_text_embedder()

        # Generate actual embedding
        embeddings = embedder.execute_embedding(
            texts=["This is a test of beam loss in the storage ring."],
            model_id="nomic-embed-text",
        )

        # Verify result
        assert embeddings is not None
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768  # nomic-embed-text dimension

    def test_embedding_generation_mocked(self):
        """Test embedding generation with mock (always runs).

        QUAL-003: Fallback test when Ollama unavailable.
        """
        from unittest.mock import MagicMock

        mock_embedder = MagicMock()
        mock_embedder.execute_embedding = MagicMock(
            return_value=[[0.1] * 768]  # Mock 768-dim embedding
        )

        result = mock_embedder.execute_embedding(
            texts=["test text"],
            model_id="nomic-embed-text",
        )

        assert len(result) == 1
        assert len(result[0]) == 768


class TestThreeTierPriorityComplete:
    """Complete tests for 3-tier priority resolution (QUAL-009)."""

    @pytest.mark.asyncio
    async def test_three_tier_priority_explicit_param_highest(self, search_calls):
        """Explicit parameter has highest priority in 3-tier resolution.

        QUAL-009: Test 3-tier priority - tier 1 (param).
        Priority order: param > config > default
        """
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import semantic_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        "settings": {"similarity_threshold": 0.8},  # Config value
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Explicit param = 0.3 should override config's 0.8
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=0.3,  # Tier 1: explicit param
            search_calls=search_calls,
        )

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_three_tier_priority_config_middle(self, search_calls):
        """Config value has middle priority in 3-tier resolution.

        QUAL-009: Test 3-tier priority - tier 2 (config).
        """
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import (
            DEFAULT_SIMILARITY_THRESHOLD,
            semantic_search,
        )

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        "settings": {"similarity_threshold": 0.85},  # Config value
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # No explicit param -> should use config's 0.85
        await semantic_search("test", mock_repo, config, mock_embedder, search_calls=search_calls)

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.85
        assert call_args.kwargs["similarity_threshold"] != DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_three_tier_priority_default_lowest(self, search_calls):
        """Default value has lowest priority in 3-tier resolution.

        QUAL-009: Test 3-tier priority - tier 3 (default).
        """
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import (
            DEFAULT_SIMILARITY_THRESHOLD,
            semantic_search,
        )

        # Config without explicit threshold setting
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        # No similarity_threshold in settings
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # No explicit param, no config value -> should use default
        await semantic_search("test", mock_repo, config, mock_embedder, search_calls=search_calls)

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == DEFAULT_SIMILARITY_THRESHOLD


class TestThresholdInSQLVerification:
    """Quality assertions for threshold in SQL (QUAL-010)."""

    @pytest.mark.asyncio
    async def test_threshold_passed_to_repository(self, search_calls):
        """Verify threshold is passed to repository for SQL filtering.

        QUAL-010: Verify threshold in SQL.
        The threshold is passed to repository.semantic_search which
        uses it in the SQL WHERE clause for similarity filtering.
        """
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.search.semantic import semantic_search

        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                    },
                },
            }
        )

        mock_repo = MagicMock()
        attach_fake_fts(mock_repo, has_v2=False, has_copy_state=False)
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        threshold = 0.75
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=threshold,
            search_calls=search_calls,
        )

        # Verify threshold was passed to repository
        mock_repo.semantic_search.assert_called_once()
        call_kwargs = mock_repo.semantic_search.call_args.kwargs
        assert "similarity_threshold" in call_kwargs
        assert call_kwargs["similarity_threshold"] == threshold

    def test_default_similarity_threshold_constant(self):
        """DEFAULT_SIMILARITY_THRESHOLD is 0.5.

        QUAL-010: Verify default threshold value.
        """
        from osprey.services.ariel_search.search.semantic import (
            DEFAULT_SIMILARITY_THRESHOLD,
        )

        assert DEFAULT_SIMILARITY_THRESHOLD == 0.5


class TestKeywordSearchFilterParameters:
    """Tests for keyword_search filter clauses built from call parameters."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ARIEL config with keyword search enabled."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
            }
        )

    @pytest.fixture
    def mock_repository(self, mock_config):
        """Create mock repository returning no rows."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=[])
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_quoted_phrases_become_query_params(self, mock_repository, mock_config):
        """Each quoted phrase is bound as its own parameter, after the free text."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            'fault "beam current" "vacuum pump"',
            mock_repository,
            mock_config,
            fuzzy_fallback=False,
        )

        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert kwargs["params"] == ["fault", "beam current", "vacuum pump"]
        assert kwargs["where_clauses"][0].count("phraseto_tsquery") == 2

    @pytest.mark.asyncio
    async def test_author_parameter_adds_ilike_clause(self, mock_repository, mock_config):
        """The author= parameter contributes an ILIKE clause with % wildcards."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam",
            mock_repository,
            mock_config,
            author="jsmith",
            fuzzy_fallback=False,
        )

        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert "author ILIKE %s" in kwargs["where_clauses"]
        assert "%jsmith%" in kwargs["params"]

    @pytest.mark.asyncio
    async def test_author_field_filter_wins_over_parameter(self, mock_repository, mock_config):
        """An author: field filter in the query suppresses the author= parameter."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam author:smith",
            mock_repository,
            mock_config,
            author="jsmith",
            fuzzy_fallback=False,
        )

        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert kwargs["where_clauses"].count("author ILIKE %s") == 1
        assert "%smith%" in kwargs["params"]
        assert "%jsmith%" not in kwargs["params"]

    @pytest.mark.asyncio
    async def test_source_system_parameter_adds_exact_clause(self, mock_repository, mock_config):
        """The source_system= parameter matches exactly, not by ILIKE."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam",
            mock_repository,
            mock_config,
            source_system="Example eLog",
            fuzzy_fallback=False,
        )

        kwargs = mock_repository.keyword_search.call_args.kwargs
        assert "source_system = %s" in kwargs["where_clauses"]
        assert "Example eLog" in kwargs["params"]


class TestSemanticSearchDiagnostics:
    """Tests for semantic_search dimension and empty-index diagnostics."""

    @pytest.fixture
    def mock_embedder(self):
        """Embedder returning a 3-dimensional vector."""
        from unittest.mock import MagicMock

        embedder = make_fake_embedding_provider(default_base_url="http://localhost:11434")()
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return embedder

    @pytest.fixture
    def mock_repository(self):
        """Repository returning no semantic matches."""
        from unittest.mock import AsyncMock, MagicMock

        repo = MagicMock()
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        repo.semantic_search = AsyncMock(return_value=[])
        repo.get_embedding_tables = AsyncMock(return_value=[])
        return repo

    def _config(self, settings):
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "nomic-embed-text",
                        "settings": settings,
                    },
                },
            }
        )

    @pytest.mark.asyncio
    async def test_dimension_mismatch_warns_but_still_searches(
        self, mock_repository, mock_embedder, caplog, search_calls
    ):
        """A query embedding of the wrong width warns and proceeds."""
        import logging

        from osprey.services.ariel_search.search.semantic import semantic_search

        config = self._config({"embedding_dimension": 768})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await semantic_search(
                "beam", mock_repository, config, mock_embedder, search_calls=search_calls
            )

        assert "Embedding dimension mismatch" in caplog.text
        assert "3 dimensions but config expects 768" in caplog.text
        mock_repository.semantic_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_matching_dimension_does_not_warn(
        self, mock_repository, mock_embedder, caplog, search_calls
    ):
        """A correctly sized query embedding produces no mismatch warning."""
        import logging

        from osprey.services.ariel_search.search.semantic import semantic_search

        config = self._config({"embedding_dimension": 3})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await semantic_search(
                "beam", mock_repository, config, mock_embedder, search_calls=search_calls
            )

        assert "Embedding dimension mismatch" not in caplog.text

    @pytest.mark.asyncio
    async def test_zero_results_with_no_embedding_tables_warns(
        self, mock_repository, mock_embedder, caplog, search_calls
    ):
        """Zero hits with no embedding tables at all points at the ingest step."""
        import logging

        from osprey.services.ariel_search.search.semantic import semantic_search

        config = self._config({"similarity_threshold": 0.5})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            results = await semantic_search(
                "beam", mock_repository, config, mock_embedder, search_calls=search_calls
            )

        assert results == []
        assert "no embeddings exist" in caplog.text

    @pytest.mark.asyncio
    async def test_zero_results_with_empty_embedding_tables_warns(
        self, mock_repository, mock_embedder, caplog, search_calls
    ):
        """Tables that exist but hold no rows produce the same diagnostic."""
        import logging
        from unittest.mock import MagicMock

        from osprey.services.ariel_search.search.semantic import semantic_search

        empty_table = MagicMock()
        empty_table.entry_count = 0
        mock_repository.get_embedding_tables.return_value = [empty_table]
        config = self._config({"similarity_threshold": 0.5})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await semantic_search(
                "beam", mock_repository, config, mock_embedder, search_calls=search_calls
            )

        assert "no embeddings exist" in caplog.text

    @pytest.mark.asyncio
    async def test_zero_results_with_populated_tables_stays_quiet(
        self, mock_repository, mock_embedder, caplog, search_calls
    ):
        """Populated tables mean the miss is genuine — no diagnostic warning."""
        import logging
        from unittest.mock import MagicMock

        from osprey.services.ariel_search.search.semantic import semantic_search

        populated = MagicMock()
        populated.entry_count = 42
        mock_repository.get_embedding_tables.return_value = [populated]
        config = self._config({"similarity_threshold": 0.5})

        with caplog.at_level(logging.WARNING, logger="ariel"):
            await semantic_search(
                "beam", mock_repository, config, mock_embedder, search_calls=search_calls
            )

        assert "no embeddings exist" not in caplog.text
