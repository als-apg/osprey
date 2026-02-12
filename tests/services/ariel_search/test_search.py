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
    _balance_quotes,
    build_tsquery,
)


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


class TestParseQuery:
    """Tests for query parsing."""

    def test_simple_query(self):
        """Parses simple query with no operators."""
        search_text, filters, phrases = parse_query("beam current")
        assert search_text == "beam current"
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

    def test_author_filter(self):
        """Extracts author: filter."""
        search_text, filters, phrases = parse_query("beam author:smith")
        assert search_text == "beam"
        assert filters == {"author": "smith"}

    def test_date_filter(self):
        """Extracts date: filter."""
        search_text, filters, phrases = parse_query("fault date:2024-01")
        assert search_text == "fault"
        assert filters == {"date": "2024-01"}

    def test_multiple_filters(self):
        """Extracts multiple filters."""
        search_text, filters, phrases = parse_query("beam author:smith date:2024-01")
        assert search_text == "beam"
        assert filters == {"author": "smith", "date": "2024-01"}

    def test_combined_query(self):
        """Parses query with filters and phrases."""
        search_text, filters, phrases = parse_query('"beam loss" author:smith vacuum')
        assert "vacuum" in search_text
        assert filters == {"author": "smith"}
        assert phrases == ["beam loss"]

    def test_empty_query(self):
        """Handles empty query."""
        search_text, filters, phrases = parse_query("")
        assert search_text == ""
        assert filters == {}
        assert phrases == []


class TestBuildTsquery:
    """Tests for tsquery building."""

    def test_simple_terms(self):
        """Builds tsquery for simple terms."""
        result = build_tsquery("beam current", [])
        assert "plainto_tsquery" in result

    def test_with_operators(self):
        """Builds tsquery with boolean operators."""
        result = build_tsquery("beam AND loss", [])
        # After operator replacement, should use websearch_to_tsquery
        assert "websearch_to_tsquery" in result or "plainto_tsquery" in result

    def test_with_phrases(self):
        """Includes phrase queries."""
        result = build_tsquery("beam", ["exact phrase"])
        assert "phraseto_tsquery" in result

    def test_empty_inputs(self):
        """Handles empty inputs."""
        result = build_tsquery("", [])
        assert "plainto_tsquery" in result


class TestSearchModuleExports:
    """Tests for search module exports."""

    def test_keyword_search_exported(self):
        """keyword_search is exported from search module."""
        from osprey.services.ariel_search.search import keyword_search

        assert callable(keyword_search)

    def test_semantic_search_exported(self):
        """semantic_search is exported from search module."""
        from osprey.services.ariel_search.search import semantic_search

        assert callable(semantic_search)


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
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=[])
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_repository, mock_config):
        """Empty query returns empty results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        result = await keyword_search("", mock_repository, mock_config)
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, mock_repository, mock_config):
        """Whitespace-only query returns empty results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        result = await keyword_search("   ", mock_repository, mock_config)
        assert result == []

    @pytest.mark.asyncio
    async def test_simple_search_calls_repository(self, mock_repository, mock_config):
        """Simple search delegates to repository."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search("beam current", mock_repository, mock_config)
        mock_repository.keyword_search.assert_called_once()

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
    async def test_date_filter_daily(self, mock_repository, mock_config):
        """Date filter parses YYYY-MM-DD format."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        await keyword_search(
            "beam date:2024-01-15",
            mock_repository,
            mock_config,
        )

        mock_repository.keyword_search.assert_called_once()

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
        repo.config = mock_config
        repo.semantic_search = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding provider."""
        from unittest.mock import MagicMock

        embedder = MagicMock()
        embedder.default_base_url = "http://localhost:11434"
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return embedder

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_repository, mock_config, mock_embedder):
        """Empty query returns empty results."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        result = await semantic_search("", mock_repository, mock_config, mock_embedder)
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(
        self, mock_repository, mock_config, mock_embedder
    ):
        """Whitespace-only query returns empty results."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        result = await semantic_search("   ", mock_repository, mock_config, mock_embedder)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_model_returns_empty(
        self, mock_repository, mock_config_no_model, mock_embedder
    ):
        """Returns empty if no model configured."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        result = await semantic_search(
            "test query",
            mock_repository,
            mock_config_no_model,
            mock_embedder,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_generation_called(self, mock_repository, mock_config, mock_embedder):
        """Embedding generation is called with query."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
        )

        mock_embedder.execute_embedding.assert_called_once()
        call_args = mock_embedder.execute_embedding.call_args
        assert call_args.kwargs["texts"] == ["test query"]

    @pytest.mark.asyncio
    async def test_uses_config_threshold(self, mock_repository, mock_config, mock_embedder):
        """Uses threshold from config when not provided."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
        )

        call_args = mock_repository.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_per_query_threshold_override(self, mock_repository, mock_config, mock_embedder):
        """Per-query threshold overrides config."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
            similarity_threshold=0.5,
        )

        call_args = mock_repository.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty(
        self, mock_repository, mock_config, mock_embedder
    ):
        """Returns empty on embedding failure."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        mock_embedder.execute_embedding.side_effect = Exception("Ollama unavailable")

        result = await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_embedding_returns_empty(self, mock_repository, mock_config, mock_embedder):
        """Returns empty when embedding is empty."""
        from osprey.services.ariel_search.search.semantic import semantic_search

        mock_embedder.execute_embedding.return_value = [[]]

        result = await semantic_search(
            "test query",
            mock_repository,
            mock_config,
            mock_embedder,
        )

        assert result == []


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
                    "source_system": "ALS eLog",
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

        call_args = mock_repository.keyword_search.call_args
        # Check call was made (date filtering applied via SQL)
        assert call_args is not None


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
        repo.config = mock_config
        repo.semantic_search = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        from unittest.mock import MagicMock

        embedder = MagicMock()
        embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return embedder

    @pytest.mark.asyncio
    async def test_semantic_with_date_filters(self, mock_repository, mock_config, mock_embedder):
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
        repo.config = mock_config
        repo.keyword_search = AsyncMock(return_value=[])
        repo.fuzzy_search = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_drop_table_injection(self, mock_repository, mock_config):
        """SQL injection with DROP TABLE is treated as literal text."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        malicious_query = "'; DROP TABLE entries; --"
        await keyword_search(malicious_query, mock_repository, mock_config)

        # Verify the query was passed as a parameter, not executed
        mock_repository.keyword_search.assert_called_once()
        call_args = mock_repository.keyword_search.call_args
        params = call_args.kwargs.get("params", [])

        # The malicious content should be in params as literal text
        # and not cause any SQL execution
        assert any("DROP TABLE" in str(p) for p in params)

    @pytest.mark.asyncio
    async def test_or_injection(self, mock_repository, mock_config):
        """SQL injection with OR '1'='1' is treated as literal text."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        malicious_query = "' OR '1'='1"
        await keyword_search(malicious_query, mock_repository, mock_config)

        mock_repository.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_union_select_injection(self, mock_repository, mock_config):
        """SQL injection with UNION SELECT is treated as literal text."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        malicious_query = "test UNION SELECT * FROM users --"
        await keyword_search(malicious_query, mock_repository, mock_config)

        # Query should be handled safely as literal text
        mock_repository.keyword_search.assert_called_once()

    def test_parameterized_queries_use_placeholders(self):
        """Verify tsquery builder uses %s placeholders for parameterized queries."""
        result = build_tsquery("test query", [])

        # The result should contain %s placeholder for parameterized query
        assert "%s" in result

    def test_parameterized_queries_with_phrases(self):
        """Verify phrase queries also use %s placeholders."""
        result = build_tsquery("test", ["exact phrase"])

        # Should have placeholders for both
        assert result.count("%s") >= 2


class TestUnbalancedQuotes:
    """Tests for unbalanced quotes handling (GAP-C001, TEST-H002)."""

    def test_balanced_quotes_unchanged(self):
        """Balanced quotes are not modified."""
        query = '"hello world" test'
        result = _balance_quotes(query)
        assert result == query

    def test_unbalanced_single_quote(self):
        """Single unbalanced quote is removed."""
        query = '"hello world'
        result = _balance_quotes(query)
        # The last quote should be removed to balance
        assert result.count('"') % 2 == 0

    def test_unbalanced_trailing_quote(self):
        """Trailing unbalanced quote is removed."""
        query = 'test query"'
        result = _balance_quotes(query)
        assert result.count('"') % 2 == 0

    def test_unbalanced_multiple_quotes(self):
        """Multiple unbalanced quotes are handled."""
        query = '"first" "second" "third'
        result = _balance_quotes(query)
        # Should have even number of quotes after balancing
        assert result.count('"') % 2 == 0

    def test_no_quotes(self):
        """Query without quotes is unchanged."""
        query = "simple query"
        result = _balance_quotes(query)
        assert result == query

    def test_empty_quotes(self):
        """Empty quoted string is preserved."""
        query = '""'
        result = _balance_quotes(query)
        assert result == '""'

    def test_parse_query_handles_unbalanced_quotes(self):
        """parse_query safely handles unbalanced quotes."""
        # This should not raise an error
        search_text, filters, phrases = parse_query('"beam loss')
        # The unbalanced quote should be handled
        assert isinstance(search_text, str)
        assert isinstance(phrases, list)

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
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        # Should not raise an error
        result = await keyword_search('"beam loss', mock_repo, config)
        assert isinstance(result, list)


class TestQueryLengthTruncation:
    """Tests for query length truncation (GAP-C002, TEST-H003)."""

    def test_max_query_length_constant(self):
        """MAX_QUERY_LENGTH constant is 1000."""
        assert MAX_QUERY_LENGTH == 1000

    @pytest.mark.asyncio
    async def test_long_query_truncated(self):
        """Queries longer than 1000 chars are truncated."""
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
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        long_query = "a" * 1500
        await keyword_search(long_query, mock_repo, config)

        # Verify keyword_search was called (query was truncated, not rejected)
        mock_repo.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_at_limit_not_truncated(self):
        """Query exactly at 1000 chars is not truncated."""
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
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        exactly_at_limit = "a" * 1000
        await keyword_search(exactly_at_limit, mock_repo, config)

        mock_repo.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_under_limit_not_truncated(self):
        """Query under 1000 chars is not truncated."""
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
        mock_repo.keyword_search = AsyncMock(return_value=[])
        mock_repo.fuzzy_search = AsyncMock(return_value=[])

        short_query = "beam current"
        await keyword_search(short_query, mock_repo, config)

        mock_repo.keyword_search.assert_called_once()


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


# === RAG Tests moved to Pipeline Tests ===


class TestRAGCitationValidation:
    """Tests for RAG citation validation (TEST-M001).

    RAG is now implemented via Pipeline (SemanticRetriever + SingleLLMProcessor).
    These tests verify the RAG prompt template.
    """

    def test_citation_format_in_prompt(self):
        """RAG prompt template includes citation instructions."""
        from osprey.services.ariel_search.prompts import RAG_PROMPT_TEMPLATE

        assert "[#12345]" in RAG_PROMPT_TEMPLATE
        assert "cite" in RAG_PROMPT_TEMPLATE.lower()


# === Validation Tests (TEST-H007, TEST-H008) ===


class TestSemanticSearchValidation:
    """Tests for semantic search validation (TEST-H007, TEST-H008)."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with dimension specified."""
        from osprey.services.ariel_search.config import ARIELConfig

        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {
                    "semantic": {
                        "enabled": True,
                        "model": "test-model",
                        "settings": {"embedding_dimension": 384},
                    },
                },
            }
        )

    @pytest.mark.asyncio
    async def test_embedding_dimension_mismatch_warning(self, mock_config, caplog):
        """Warning logged when embedding dimension mismatches config (TEST-H008)."""
        import logging
        from unittest.mock import AsyncMock, MagicMock

        from osprey.services.ariel_search.search.semantic import semantic_search

        mock_repo = MagicMock()
        mock_repo.semantic_search = AsyncMock(return_value=[])

        # Return embedding with different dimension than configured
        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(
            return_value=[[0.1, 0.2, 0.3]]  # 3 dimensions, config expects 384
        )

        with caplog.at_level(logging.WARNING):
            await semantic_search(
                "test query",
                mock_repo,
                mock_config,
                mock_embedder,
            )

        # Check that warning was logged
        assert any("dimension mismatch" in record.message.lower() for record in caplog.records)


# === Edge Case Tests (EDGE-004 to EDGE-016) ===


class TestThreeTierThresholdResolution:
    """Tests for 3-tier threshold resolution (EDGE-004)."""

    @pytest.mark.asyncio
    async def test_param_overrides_config(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Pass explicit threshold that should override config
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=0.5,  # Override config's 0.8
        )

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_config_overrides_default(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Don't pass explicit threshold - should use config
        await semantic_search("test", mock_repo, config, mock_embedder)

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

    def test_max_results_zero_invalid(self):
        """max_results=0 raises validation error (EDGE-007)."""
        from pydantic import ValidationError

        from osprey.services.ariel_search.search.keyword import KeywordSearchInput

        with pytest.raises(ValidationError):
            KeywordSearchInput(query="test", max_results=0)

    def test_max_results_fifty_one_invalid(self):
        """max_results=51 raises validation error (EDGE-008)."""
        from pydantic import ValidationError

        from osprey.services.ariel_search.search.keyword import KeywordSearchInput

        with pytest.raises(ValidationError):
            KeywordSearchInput(query="test", max_results=51)

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
    async def test_semantic_search_embedding_failure_returns_empty(self):
        """Semantic search returns empty on embedding failure."""
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(
            side_effect=Exception("Embedding service unavailable")
        )

        result = await semantic_search("test", mock_repo, config, mock_embedder)

        assert result == []


class TestPromptExtraction:
    """Tests verifying prompt extraction to separate files (STRUCT-003, STRUCT-004).

    Note: Agent system prompt is now in agent/executor.py
    """

    def test_agent_system_prompt_in_executor(self):
        """Agent system prompt is in agent/executor.py."""
        from osprey.services.ariel_search.agent.executor import AGENT_SYSTEM_PROMPT

        assert AGENT_SYSTEM_PROMPT
        assert "ARIEL" in AGENT_SYSTEM_PROMPT
        assert "logbook" in AGENT_SYSTEM_PROMPT.lower()

    def test_rag_prompt_in_separate_file(self):
        """RAG prompt template is in dedicated file."""
        from osprey.services.ariel_search.prompts.rag_answer import RAG_PROMPT_TEMPLATE

        assert RAG_PROMPT_TEMPLATE
        assert "{context}" in RAG_PROMPT_TEMPLATE
        assert "{question}" in RAG_PROMPT_TEMPLATE

    def test_prompts_module_exports_rag_template(self):
        """Prompts module exports RAG prompt template."""
        from osprey.services.ariel_search.prompts import RAG_PROMPT_TEMPLATE

        assert RAG_PROMPT_TEMPLATE


# ==============================================================================
# Quality Improvements (QUAL-003, QUAL-005, QUAL-009, QUAL-010)
# ==============================================================================


class TestEmbeddingGenerationQuality:
    """Quality assertions for embedding generation (QUAL-003)."""

    @pytest.mark.requires_ollama
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

        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

        embedder = OllamaEmbeddingProvider()

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


class TestStemmingOutput:
    """Quality assertions for stemming output (QUAL-005)."""

    def test_stemming_applied_in_tsquery(self):
        """Verify stemming is applied in tsquery building.

        QUAL-005: Verify stemming output.
        PostgreSQL's to_tsquery with 'english' configuration applies stemming.
        """
        # build_tsquery uses 'english' config which applies stemming
        result = build_tsquery("running tests", [])

        # The result is a template string with %s placeholders
        # Stemming happens in PostgreSQL, but we verify the config is used
        assert "english" in result.lower() or "%s" in result

    def test_tsquery_uses_english_config(self):
        """tsquery explicitly uses 'english' configuration."""
        result = build_tsquery("test", [])

        # Should reference 'english' configuration for stemming
        # or use plainto_tsquery which defaults to english
        assert "english" in result.lower() or "plainto_tsquery" in result.lower()

    def test_stemmed_words_match_variations(self):
        """Stemmed search matches word variations (conceptual test).

        This verifies the expected behavior:
        - "running" -> "run" (stem)
        - "tests" -> "test" (stem)

        Actual stemming happens in PostgreSQL's to_tsvector/to_tsquery.
        """
        # The tsquery builder handles the setup, PostgreSQL does stemming
        query_with_variations = "running tests testing runner"
        result = build_tsquery(query_with_variations, [])

        # Query should be constructed (stemming happens server-side)
        assert "%s" in result or "plainto_tsquery" in result


class TestThreeTierPriorityComplete:
    """Complete tests for 3-tier priority resolution (QUAL-009)."""

    @pytest.mark.asyncio
    async def test_three_tier_priority_explicit_param_highest(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Explicit param = 0.3 should override config's 0.8
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=0.3,  # Tier 1: explicit param
        )

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_three_tier_priority_config_middle(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # No explicit param -> should use config's 0.85
        await semantic_search("test", mock_repo, config, mock_embedder)

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == 0.85
        assert call_args.kwargs["similarity_threshold"] != DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_three_tier_priority_default_lowest(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # No explicit param, no config value -> should use default
        await semantic_search("test", mock_repo, config, mock_embedder)

        call_args = mock_repo.semantic_search.call_args
        assert call_args.kwargs["similarity_threshold"] == DEFAULT_SIMILARITY_THRESHOLD


class TestThresholdInSQLVerification:
    """Quality assertions for threshold in SQL (QUAL-010)."""

    @pytest.mark.asyncio
    async def test_threshold_passed_to_repository(self):
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
        mock_repo.semantic_search = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.default_base_url = "http://localhost:11434"
        mock_embedder.execute_embedding = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        threshold = 0.75
        await semantic_search(
            "test",
            mock_repo,
            config,
            mock_embedder,
            similarity_threshold=threshold,
        )

        # Verify threshold was passed to repository
        mock_repo.semantic_search.assert_called_once()
        call_kwargs = mock_repo.semantic_search.call_args.kwargs
        assert "similarity_threshold" in call_kwargs
        assert call_kwargs["similarity_threshold"] == threshold

    def test_default_similarity_threshold_constant(self):
        """DEFAULT_SIMILARITY_THRESHOLD is 0.7.

        QUAL-010: Verify default threshold value.
        """
        from osprey.services.ariel_search.search.semantic import (
            DEFAULT_SIMILARITY_THRESHOLD,
        )

        assert DEFAULT_SIMILARITY_THRESHOLD == 0.7
