"""Tests for ARIEL RAG pipeline.

Tests for RAGPipeline: retrieval, RRF fusion, context assembly,
LLM generation, and citation extraction.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.rag import RAGPipeline, RAGResult


def _make_entry(entry_id: str, text: str = "Test content", author: str = "jsmith") -> dict:
    """Create a mock EnhancedLogbookEntry dict."""
    return {
        "entry_id": entry_id,
        "source_system": "ALS eLog",
        "timestamp": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
        "author": author,
        "raw_text": text,
        "attachments": [],
        "metadata": {"title": f"Entry {entry_id}"},
        "created_at": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
        "updated_at": datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
    }


def _make_config(keyword_enabled=True, semantic_enabled=False, pipelines=None) -> ARIELConfig:
    """Create a minimal ARIELConfig for testing."""
    modules = {}
    if keyword_enabled:
        modules["keyword"] = {"enabled": True}
    if semantic_enabled:
        modules["semantic"] = {"enabled": True, "model": "test-model"}
    config_dict: dict = {
        "database": {"uri": "postgresql://localhost:5432/test"},
        "search_modules": modules,
    }
    if pipelines:
        config_dict["pipelines"] = pipelines
    return ARIELConfig.from_dict(config_dict)


def _make_pipeline(config=None, keyword_results=None, semantic_results=None, **kwargs):
    """Create a RAGPipeline with mocked search functions.

    Returns (pipeline, mock_repository).
    """
    config = config or _make_config()
    mock_repo = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder_loader = MagicMock(return_value=mock_embedder)

    pipeline = RAGPipeline(
        repository=mock_repo,
        config=config,
        embedder_loader=mock_embedder_loader,
        **kwargs,
    )

    return pipeline, mock_repo, mock_embedder_loader


class TestRAGResult:
    """Tests for RAGResult dataclass."""

    def test_frozen(self):
        """RAGResult is immutable."""
        result = RAGResult(answer="test")
        with pytest.raises(AttributeError):
            result.answer = "changed"  # type: ignore[misc]

    def test_defaults(self):
        """RAGResult has correct defaults."""
        result = RAGResult(answer="test")
        assert result.entries == ()
        assert result.citations == ()
        assert result.retrieval_count == 0
        assert result.context_truncated is False

    def test_full_construction(self):
        """RAGResult with all fields."""
        result = RAGResult(
            answer="The answer",
            entries=({"entry_id": "1"},),
            citations=("1",),
            retrieval_count=5,
            context_truncated=True,
        )
        assert result.answer == "The answer"
        assert len(result.entries) == 1
        assert result.citations == ("1",)
        assert result.retrieval_count == 5
        assert result.context_truncated is True


class TestCitationExtraction:
    """Tests for RAGPipeline._extract_citations."""

    def test_extracts_citations(self):
        """Extracts [#id] patterns from text."""
        citations = RAGPipeline._extract_citations("Found in [#001] and [#002] with details.")
        assert citations == ["001", "002"]

    def test_deduplicates_citations(self):
        """Deduplicates citations preserving order."""
        citations = RAGPipeline._extract_citations("See [#001], also [#002], and again [#001].")
        assert citations == ["001", "002"]

    def test_empty_text(self):
        """Returns empty list for empty text."""
        assert RAGPipeline._extract_citations("") == []
        assert RAGPipeline._extract_citations(None) == []

    def test_no_citations(self):
        """Returns empty list when no citations found."""
        assert RAGPipeline._extract_citations("No citations here.") == []

    def test_alphanumeric_ids(self):
        """Handles alphanumeric entry IDs."""
        citations = RAGPipeline._extract_citations("Entry [#DEMO_001] found.")
        assert citations == ["DEMO_001"]


class TestRRFFusion:
    """Tests for RAGPipeline._fuse (RRF fusion)."""

    def test_keyword_only(self):
        """Keyword-only results pass through."""
        pipeline, _, _ = _make_pipeline()
        kw = [
            (_make_entry("A"), 0.9, ["highlight"]),
            (_make_entry("B"), 0.7, []),
        ]
        fused = pipeline._fuse(kw, [], max_results=10)
        assert len(fused) == 2
        assert fused[0]["entry_id"] == "A"
        assert fused[1]["entry_id"] == "B"

    def test_semantic_only(self):
        """Semantic-only results pass through."""
        pipeline, _, _ = _make_pipeline()
        sem = [
            (_make_entry("C"), 0.95),
            (_make_entry("D"), 0.80),
        ]
        fused = pipeline._fuse([], sem, max_results=10)
        assert len(fused) == 2
        assert fused[0]["entry_id"] == "C"

    def test_both_sources_merges_duplicates(self):
        """Entries found by both get higher RRF score."""
        pipeline, _, _ = _make_pipeline()
        kw = [
            (_make_entry("A"), 0.9, []),
            (_make_entry("B"), 0.7, []),
        ]
        sem = [
            (_make_entry("A"), 0.95),  # Duplicate: boosted
            (_make_entry("C"), 0.80),
        ]
        fused = pipeline._fuse(kw, sem, max_results=10)
        ids = [e["entry_id"] for e in fused]

        # A appears in both, should be ranked first
        assert ids[0] == "A"
        assert set(ids) == {"A", "B", "C"}

    def test_respects_max_results(self):
        """Fused results are limited to max_results."""
        pipeline, _, _ = _make_pipeline()
        kw = [(_make_entry(f"E{i}"), 0.9 - i * 0.1, []) for i in range(5)]
        fused = pipeline._fuse(kw, [], max_results=3)
        assert len(fused) == 3

    def test_empty_inputs(self):
        """Empty inputs return empty list."""
        pipeline, _, _ = _make_pipeline()
        assert pipeline._fuse([], [], max_results=10) == []


class TestContextAssembly:
    """Tests for RAGPipeline._assemble_context."""

    def test_basic_assembly(self):
        """Assembles entries into context string."""
        pipeline, _, _ = _make_pipeline()
        entries = [_make_entry("001", "Content for entry 001")]
        text, included, truncated = pipeline._assemble_context(entries)

        assert "ENTRY #001" in text
        assert "Content for entry 001" in text
        assert len(included) == 1
        assert truncated is False

    def test_entry_format(self):
        """Context uses ENTRY #id | timestamp | Author: name format."""
        pipeline, _, _ = _make_pipeline()
        entry = _make_entry("042", "Test text", author="Dr. Jones")
        text, _, _ = pipeline._assemble_context([entry])

        assert "ENTRY #042" in text
        assert "Author: Dr. Jones" in text
        assert "Entry 042" in text  # title from metadata

    def test_truncation_flag(self):
        """Context truncation is flagged when exceeding limit."""
        pipeline, _, _ = _make_pipeline(max_context_chars=100)
        entries = [
            _make_entry("001", "A" * 200),
            _make_entry("002", "B" * 200),
        ]
        _, included, truncated = pipeline._assemble_context(entries)

        assert truncated is True

    def test_per_entry_truncation(self):
        """Individual entries are truncated at max_chars_per_entry."""
        pipeline, _, _ = _make_pipeline(max_chars_per_entry=50)
        entry = _make_entry("001", "X" * 200)
        text, _, _ = pipeline._assemble_context([entry])

        # Content should be truncated (header + 50 chars + "...")
        assert "..." in text

    def test_empty_entries(self):
        """Empty entries list returns empty context."""
        pipeline, _, _ = _make_pipeline()
        text, included, truncated = pipeline._assemble_context([])

        assert text == ""
        assert included == []
        assert truncated is False

    def test_separator(self):
        """Entries are separated by ---."""
        pipeline, _, _ = _make_pipeline()
        entries = [_make_entry("001", "A"), _make_entry("002", "B")]
        text, _, _ = pipeline._assemble_context(entries)

        assert "\n---\n" in text


class TestRAGPipelineExecute:
    """Tests for RAGPipeline.execute end-to-end (mocked)."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Empty query returns no-context answer."""
        pipeline, _, _ = _make_pipeline()
        result = await pipeline.execute("")

        assert "don't have enough information" in result.answer
        assert result.entries == ()

    @pytest.mark.asyncio
    async def test_keyword_only_retrieval(self):
        """RAG with keyword-only config retrieves via keyword search."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline, mock_repo, _ = _make_pipeline(config=config)

        kw_results = [
            (_make_entry("001", "RF cavity tripped"), 0.9, ["<mark>RF</mark>"]),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ) as mock_kw,
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="The RF cavity tripped [#001].",
            ),
        ):
            result = await pipeline.execute("RF cavity")

        mock_kw.assert_called_once()
        assert result.retrieval_count == 1
        assert result.citations == ("001",)
        assert "RF cavity" in result.answer

    @pytest.mark.asyncio
    async def test_semantic_only_retrieval(self):
        """RAG with semantic-only config retrieves via semantic search."""
        config = _make_config(keyword_enabled=False, semantic_enabled=True)
        pipeline, mock_repo, _ = _make_pipeline(config=config)

        sem_results = [
            (_make_entry("002", "Beam loss event"), 0.92),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                return_value=sem_results,
            ) as mock_sem,
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="A beam loss event occurred [#002].",
            ),
        ):
            result = await pipeline.execute("beam loss")

        mock_sem.assert_called_once()
        assert result.retrieval_count == 1
        assert "002" in result.citations

    @pytest.mark.asyncio
    async def test_dual_retrieval_with_fusion(self):
        """RAG with both modules fuses results via RRF."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [(_make_entry("A", "Keyword hit"), 0.9, [])]
        sem_results = [
            (_make_entry("A", "Keyword hit"), 0.92),  # Same entry
            (_make_entry("B", "Semantic only"), 0.85),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                return_value=sem_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer based on [#A] and [#B].",
            ),
        ):
            result = await pipeline.execute("test query")

        # Entry A found by both should be boosted
        assert result.entries[0]["entry_id"] == "A"
        assert result.retrieval_count == 2  # 2 unique entries after fusion

    @pytest.mark.asyncio
    async def test_no_results_returns_no_context_answer(self):
        """No retrieval results returns informative answer."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline, _, _ = _make_pipeline(config=config)

        with patch(
            "osprey.services.ariel_search.search.keyword.keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await pipeline.execute("nonexistent topic")

        assert "don't have enough information" in result.answer
        assert result.retrieval_count == 0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error_message(self):
        """LLM failure returns error message instead of crashing."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [(_make_entry("001", "Some entry"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                side_effect=RuntimeError("API connection failed"),
            ),
        ):
            result = await pipeline.execute("test query")

        assert "Error generating answer" in result.answer
        assert result.entries  # Entries still present even when LLM fails

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_keyword(self):
        """Embedding failure gracefully falls back to keyword-only."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        mock_repo = MagicMock()
        mock_embedder_loader = MagicMock(side_effect=RuntimeError("No Ollama"))

        pipeline = RAGPipeline(
            repository=mock_repo,
            config=config,
            embedder_loader=mock_embedder_loader,
        )

        kw_results = [(_make_entry("001", "Found via keyword"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer from keyword results [#001].",
            ),
        ):
            result = await pipeline.execute("test query")

        assert result.retrieval_count == 1
        assert "001" in result.citations

    @pytest.mark.asyncio
    async def test_citation_fallback_to_all_entries(self):
        """When LLM doesn't cite entries, all context entries become citations."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [
            (_make_entry("X1", "First entry"), 0.9, []),
            (_make_entry("X2", "Second entry"), 0.8, []),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="An answer without any citations.",
            ),
        ):
            result = await pipeline.execute("test query")

        # All entries should be listed as citations
        assert "X1" in result.citations
        assert "X2" in result.citations


class TestPartialRetrievalFailure:
    """Tests for partial retrieval failure handling."""

    @pytest.mark.asyncio
    async def test_keyword_fails_semantic_succeeds(self):
        """When keyword search raises, semantic results are still used."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline, _, _ = _make_pipeline(config=config)

        sem_results = [(_make_entry("S1", "Semantic hit"), 0.9)]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Keyword DB error"),
            ),
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                return_value=sem_results,
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer from semantic [#S1].",
            ),
        ):
            result = await pipeline.execute("test query")

        assert result.retrieval_count == 1
        assert result.entries[0]["entry_id"] == "S1"

    @pytest.mark.asyncio
    async def test_semantic_fails_keyword_succeeds(self):
        """When semantic search raises, keyword results are still used."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [(_make_entry("K1", "Keyword hit"), 0.9, ["hit"])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Embedding service down"),
            ),
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer from keyword [#K1].",
            ),
        ):
            result = await pipeline.execute("test query")

        assert result.retrieval_count == 1
        assert result.entries[0]["entry_id"] == "K1"


class TestGenerateImportError:
    """Tests for _generate ImportError branch."""

    @pytest.mark.asyncio
    async def test_import_error_returns_fallback_message(self):
        """ImportError in _generate returns 'LLM not available' message."""
        config = _make_config(keyword_enabled=True, semantic_enabled=False)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [(_make_entry("001", "Some entry"), 0.9, [])]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ),
            patch(
                "osprey.services.ariel_search.rag.get_chat_completion",
                side_effect=ImportError("No module named 'osprey.models.completion'"),
                create=True,
            ),
            patch.dict("sys.modules", {"osprey.models.completion": None}),
        ):
            result = await pipeline.execute("test query")

        assert "LLM not available" in result.answer
        assert result.entries  # Entries still present


class TestWhitespaceQuery:
    """Tests for whitespace-only queries."""

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self):
        """Whitespace-only query returns no-context answer."""
        pipeline, _, _ = _make_pipeline()
        result = await pipeline.execute("   ")

        assert "don't have enough information" in result.answer
        assert result.entries == ()
        assert result.retrieval_count == 0

    @pytest.mark.asyncio
    async def test_tabs_and_newlines_only(self):
        """Tabs and newlines only returns no-context answer."""
        pipeline, _, _ = _make_pipeline()
        result = await pipeline.execute("\t\n  \t")

        assert "don't have enough information" in result.answer
        assert result.entries == ()


class TestFormatEntryEdgeCases:
    """Tests for _format_entry with edge-case data."""

    def test_missing_metadata_key(self):
        """Entry without 'metadata' key still formats."""
        pipeline, _, _ = _make_pipeline()
        entry = {
            "entry_id": "no-meta",
            "author": "jsmith",
            "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
            "raw_text": "Content here",
        }
        text = pipeline._format_entry(entry)

        assert "ENTRY #no-meta" in text
        assert "Author: jsmith" in text
        assert "Content here" in text

    def test_missing_timestamp(self):
        """Entry with None timestamp uses 'Unknown'."""
        pipeline, _, _ = _make_pipeline()
        entry = _make_entry("no-ts", "Content")
        entry["timestamp"] = None
        text = pipeline._format_entry(entry)

        assert "Unknown" in text
        assert "ENTRY #no-ts" in text

    def test_empty_raw_text(self):
        """Entry with empty raw_text formats without error."""
        pipeline, _, _ = _make_pipeline()
        entry = _make_entry("empty-text", "")
        text = pipeline._format_entry(entry)

        assert "ENTRY #empty-text" in text


class TestContextAssemblyBoundary:
    """Tests for context assembly at exact boundary."""

    def test_entry_exactly_fills_limit(self):
        """Entry that exactly fills max_context_chars is not truncated."""
        pipeline, _, _ = _make_pipeline(max_context_chars=50000, max_chars_per_entry=50000)

        # Create a single entry and measure its formatted size
        entry = _make_entry("exact", "x" * 100)
        formatted = pipeline._format_entry(entry)
        exact_limit = len(formatted)

        # Create pipeline with limit exactly matching the formatted entry
        pipeline2, _, _ = _make_pipeline(max_context_chars=exact_limit, max_chars_per_entry=50000)
        text, included, truncated = pipeline2._assemble_context([entry])

        assert len(included) == 1
        assert truncated is False
        assert text == formatted

    def test_entry_one_char_over_limit_truncates(self):
        """Entry one char over max_context_chars triggers truncation for next entry."""
        pipeline, _, _ = _make_pipeline(max_context_chars=50000, max_chars_per_entry=50000)

        entry1 = _make_entry("first", "a" * 100)
        formatted1 = pipeline._format_entry(entry1)
        entry2 = _make_entry("second", "b" * 100)

        # Set limit to exactly fit first entry but not second
        pipeline2, _, _ = _make_pipeline(
            max_context_chars=len(formatted1) + 50, max_chars_per_entry=50000
        )
        _, included, truncated = pipeline2._assemble_context([entry1, entry2])

        # Second entry won't fully fit
        assert truncated is True


class TestRAGPipelineConfig:
    """Tests for RAGPipeline configuration."""

    def test_custom_fusion_k(self):
        """Custom fusion_k parameter is used."""
        pipeline, _, _ = _make_pipeline(fusion_k=30)
        assert pipeline._fusion_k == 30

    def test_custom_context_limits(self):
        """Custom context limits are used."""
        pipeline, _, _ = _make_pipeline(
            max_context_chars=5000,
            max_chars_per_entry=500,
        )
        assert pipeline._max_context_chars == 5000
        assert pipeline._max_chars_per_entry == 500


class TestPipelineRetrievalModules:
    """Tests for RAG pipeline with configured retrieval_modules."""

    @pytest.mark.asyncio
    async def test_retrieve_uses_configured_retrieval_modules(self):
        """RAG with retrieval_modules: [keyword] only calls keyword search."""
        config = _make_config(
            keyword_enabled=True,
            semantic_enabled=True,
            pipelines={"rag": {"enabled": True, "retrieval_modules": ["keyword"]}},
        )
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [
            (_make_entry("001", "Found via keyword"), 0.9, ["<mark>Found</mark>"]),
        ]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ) as mock_kw,
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
            ) as mock_sem,
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer [#001].",
            ),
        ):
            result = await pipeline.execute("test query")

        mock_kw.assert_called_once()
        mock_sem.assert_not_called()
        assert result.retrieval_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_falls_back_to_enabled_modules(self):
        """No pipeline config falls back to enabled search modules."""
        config = _make_config(keyword_enabled=True, semantic_enabled=True)
        pipeline, _, _ = _make_pipeline(config=config)

        kw_results = [(_make_entry("001", "Keyword hit"), 0.9, [])]
        sem_results = [(_make_entry("002", "Semantic hit"), 0.85)]

        with (
            patch(
                "osprey.services.ariel_search.search.keyword.keyword_search",
                new_callable=AsyncMock,
                return_value=kw_results,
            ) as mock_kw,
            patch(
                "osprey.services.ariel_search.search.semantic.semantic_search",
                new_callable=AsyncMock,
                return_value=sem_results,
            ) as mock_sem,
            patch(
                "osprey.models.completion.get_chat_completion",
                return_value="Answer [#001] [#002].",
            ),
        ):
            result = await pipeline.execute("test query")

        mock_kw.assert_called_once()
        mock_sem.assert_called_once()
        assert result.retrieval_count == 2
