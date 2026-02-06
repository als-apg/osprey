"""Unit tests for ARIEL RAP pipeline components.

Tests for retrievers, assemblers, processors, and formatters.
See 05_RAP_ABSTRACTION.md Section 7 for test requirements.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey.services.ariel_search.pipeline import (
    AssembledContext,
    AssemblyConfig,
    Pipeline,
    PipelineBuilder,
    ProcessedResult,
    ProcessorConfig,
    RetrievalConfig,
    RetrievedItem,
)
from osprey.services.ariel_search.pipeline.assemblers import (
    ContextWindowAssembler,
    TopKAssembler,
)
from osprey.services.ariel_search.pipeline.formatters import (
    CitationFormatter,
    JSONFormatter,
)
from osprey.services.ariel_search.pipeline.processors import (
    IdentityProcessor,
    SingleLLMProcessor,
)
from osprey.services.ariel_search.pipeline.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    RRFFusion,
    SemanticRetriever,
    WeightedFusion,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_entry() -> EnhancedLogbookEntry:
    """Create a sample logbook entry for testing."""
    return {
        "entry_id": "test-001",
        "source_system": "test",
        "timestamp": datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        "author": "test_user",
        "raw_text": "This is a test entry about beam alignment procedures.",
        "attachments": [],
        "metadata": {"title": "Beam Alignment"},
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


@pytest.fixture
def sample_entries() -> list[EnhancedLogbookEntry]:
    """Create multiple sample entries for testing."""
    return [
        {
            "entry_id": f"test-{i:03d}",
            "source_system": "test",
            "timestamp": datetime(2025, 1, 15, 10 + i, 0, 0, tzinfo=UTC),
            "author": f"user_{i}",
            "raw_text": f"Test entry {i} about topic {i % 3}.",
            "attachments": [],
            "metadata": {"title": f"Entry {i}"},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_retrieved_items(sample_entries: list[EnhancedLogbookEntry]) -> list[RetrievedItem]:
    """Create sample RetrievedItem instances."""
    return [
        RetrievedItem(
            entry=entry,
            score=1.0 - (i * 0.1),
            source="test",
            metadata={"test_index": i},
        )
        for i, entry in enumerate(sample_entries)
    ]


@pytest.fixture
def mock_ariel_config() -> MagicMock:
    """Create a mock ARIEL config."""
    config = MagicMock()
    config.search_modules = {}
    config.get_search_model.return_value = "nomic-embed-text"
    return config


@pytest.fixture
def mock_repository() -> MagicMock:
    """Create a mock repository."""
    repo = MagicMock()
    repo.keyword_search = AsyncMock(return_value=[])
    repo.semantic_search = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.execute_embedding = MagicMock(return_value=[[0.1] * 768])
    embedder.default_base_url = "http://localhost:11434"
    return embedder


# ============================================================================
# Retriever Tests
# ============================================================================


class TestKeywordRetriever:
    """Tests for KeywordRetriever."""

    @pytest.mark.asyncio
    async def test_name_property(self, mock_repository, mock_ariel_config):
        """Test retriever name property."""
        retriever = KeywordRetriever(mock_repository, mock_ariel_config)
        assert retriever.name == "keyword"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_repository, mock_ariel_config):
        """Test empty query returns empty list."""
        retriever = KeywordRetriever(mock_repository, mock_ariel_config)
        result = await retriever.retrieve("", RetrievalConfig())
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, mock_repository, mock_ariel_config):
        """Test whitespace-only query returns empty list."""
        retriever = KeywordRetriever(mock_repository, mock_ariel_config)
        result = await retriever.retrieve("   ", RetrievalConfig())
        assert result == []

    @pytest.mark.asyncio
    async def test_normalizes_return_type(self, mock_repository, mock_ariel_config, sample_entry):
        """Test that results are normalized to RetrievedItem."""
        with patch("osprey.services.ariel_search.search.keyword.keyword_search") as mock_search:
            mock_search.return_value = [(sample_entry, 0.9, ["highlighted"])]

            retriever = KeywordRetriever(mock_repository, mock_ariel_config)
            result = await retriever.retrieve("beam", RetrievalConfig())

            assert len(result) == 1
            assert isinstance(result[0], RetrievedItem)
            assert result[0].entry == sample_entry
            assert result[0].score == 0.9
            assert result[0].source == "keyword"
            assert result[0].metadata["highlights"] == ["highlighted"]


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""

    @pytest.mark.asyncio
    async def test_name_property(self, mock_repository, mock_ariel_config, mock_embedder):
        """Test retriever name property."""
        retriever = SemanticRetriever(mock_repository, mock_ariel_config, mock_embedder)
        assert retriever.name == "semantic"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(
        self, mock_repository, mock_ariel_config, mock_embedder
    ):
        """Test empty query returns empty list."""
        retriever = SemanticRetriever(mock_repository, mock_ariel_config, mock_embedder)
        result = await retriever.retrieve("", RetrievalConfig())
        assert result == []

    @pytest.mark.asyncio
    async def test_normalizes_return_type(
        self, mock_repository, mock_ariel_config, mock_embedder, sample_entry
    ):
        """Test that results are normalized to RetrievedItem."""
        with patch("osprey.services.ariel_search.search.semantic.semantic_search") as mock_search:
            mock_search.return_value = [(sample_entry, 0.85)]

            retriever = SemanticRetriever(mock_repository, mock_ariel_config, mock_embedder)
            result = await retriever.retrieve("alignment", RetrievalConfig())

            assert len(result) == 1
            assert isinstance(result[0], RetrievedItem)
            assert result[0].entry == sample_entry
            assert result[0].score == 0.85
            assert result[0].source == "semantic"
            assert result[0].metadata["similarity"] == 0.85


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.mark.asyncio
    async def test_name_property(self):
        """Test retriever name property."""
        retriever = HybridRetriever([])
        assert retriever.name == "hybrid"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        """Test empty query returns empty list."""
        retriever = HybridRetriever([])
        result = await retriever.retrieve("", RetrievalConfig())
        assert result == []

    @pytest.mark.asyncio
    async def test_no_retrievers_returns_empty(self):
        """Test that empty retrievers list returns empty."""
        retriever = HybridRetriever([])
        result = await retriever.retrieve("test", RetrievalConfig())
        assert result == []

    @pytest.mark.asyncio
    async def test_combines_results(self, sample_entries):
        """Test that results from multiple retrievers are combined."""
        # Create mock retrievers
        mock_retriever1 = MagicMock()
        mock_retriever1.name = "keyword"
        mock_retriever1.retrieve = AsyncMock(
            return_value=[
                RetrievedItem(sample_entries[0], 0.9, "keyword", {}),
                RetrievedItem(sample_entries[1], 0.8, "keyword", {}),
            ]
        )

        mock_retriever2 = MagicMock()
        mock_retriever2.name = "semantic"
        mock_retriever2.retrieve = AsyncMock(
            return_value=[
                RetrievedItem(sample_entries[1], 0.85, "semantic", {}),
                RetrievedItem(sample_entries[2], 0.75, "semantic", {}),
            ]
        )

        retriever = HybridRetriever([mock_retriever1, mock_retriever2])
        result = await retriever.retrieve("test", RetrievalConfig(max_results=10))

        # Should have 3 unique entries (entry 1 appears in both)
        assert len(result) == 3

        # Entry that appears in both should have higher score
        entry_ids = [r.entry["entry_id"] for r in result]
        assert "test-001" in entry_ids  # From keyword only
        assert "test-000" in entry_ids  # From both (should be ranked higher)
        assert "test-002" in entry_ids  # From semantic only

    @pytest.mark.asyncio
    async def test_handles_retriever_errors(self, sample_entries):
        """Test graceful handling of retriever failures."""
        mock_good = MagicMock()
        mock_good.name = "good"
        mock_good.retrieve = AsyncMock(
            return_value=[RetrievedItem(sample_entries[0], 0.9, "good", {})]
        )

        mock_bad = MagicMock()
        mock_bad.name = "bad"
        mock_bad.retrieve = AsyncMock(side_effect=Exception("Retriever failed"))

        retriever = HybridRetriever([mock_good, mock_bad])
        result = await retriever.retrieve("test", RetrievalConfig())

        # Should still get results from the good retriever
        assert len(result) == 1


class TestRRFFusion:
    """Tests for RRF fusion strategy."""

    def test_fuses_single_list(self, sample_retrieved_items):
        """Test fusion with single retriever results."""
        fusion = RRFFusion()
        result = fusion.fuse([sample_retrieved_items[:3]])

        assert len(result) == 3
        # Order should be preserved
        assert result[0].entry["entry_id"] == "test-000"

    def test_fuses_multiple_lists(self, sample_entries):
        """Test fusion with multiple retriever results."""
        list1 = [
            RetrievedItem(sample_entries[0], 0.9, "r1", {}),
            RetrievedItem(sample_entries[1], 0.8, "r1", {}),
        ]
        list2 = [
            RetrievedItem(sample_entries[1], 0.85, "r2", {}),
            RetrievedItem(sample_entries[2], 0.75, "r2", {}),
        ]

        fusion = RRFFusion()
        result = fusion.fuse([list1, list2])

        # Entry 1 should have highest score (appears in both)
        assert result[0].entry["entry_id"] == "test-001"
        assert "r1" in result[0].metadata.get("sources", [])
        assert "r2" in result[0].metadata.get("sources", [])


class TestWeightedFusion:
    """Tests for weighted fusion strategy."""

    def test_equal_weights(self, sample_entries):
        """Test fusion with equal weights."""
        list1 = [RetrievedItem(sample_entries[0], 0.9, "r1", {})]
        list2 = [RetrievedItem(sample_entries[0], 0.8, "r2", {})]

        fusion = WeightedFusion()
        result = fusion.fuse([list1, list2])

        assert len(result) == 1
        # Average of 0.9 and 0.8
        assert abs(result[0].score - 0.85) < 0.01

    def test_custom_weights(self, sample_entries):
        """Test fusion with custom weights."""
        list1 = [RetrievedItem(sample_entries[0], 0.9, "r1", {})]
        list2 = [RetrievedItem(sample_entries[0], 0.8, "r2", {})]

        fusion = WeightedFusion(weights={"r1": 2.0, "r2": 1.0})
        result = fusion.fuse([list1, list2])

        assert len(result) == 1
        # Weighted average: (0.9 * 2 + 0.8 * 1) / 3 = 0.867
        assert abs(result[0].score - 0.867) < 0.01


# ============================================================================
# Assembler Tests
# ============================================================================


class TestTopKAssembler:
    """Tests for TopKAssembler."""

    def test_empty_items(self):
        """Test assembly with no items."""
        assembler = TopKAssembler()
        result = assembler.assemble([], AssemblyConfig())

        assert result.items == []
        assert result.text == ""
        assert result.total_chars == 0
        assert result.truncated is False

    def test_selects_top_k(self, sample_retrieved_items):
        """Test that top K items are selected."""
        assembler = TopKAssembler()
        config = AssemblyConfig(max_items=3)
        result = assembler.assemble(sample_retrieved_items, config)

        assert len(result.items) == 3
        # Should be sorted by score
        assert result.items[0].score >= result.items[1].score >= result.items[2].score

    def test_marks_truncated(self, sample_retrieved_items):
        """Test truncated flag when items exceed max."""
        assembler = TopKAssembler()
        config = AssemblyConfig(max_items=2)
        result = assembler.assemble(sample_retrieved_items, config)

        assert len(result.items) == 2
        assert result.truncated is True


class TestContextWindowAssembler:
    """Tests for ContextWindowAssembler."""

    def test_empty_items(self):
        """Test assembly with no items."""
        assembler = ContextWindowAssembler()
        result = assembler.assemble([], AssemblyConfig())

        assert result.items == []
        assert result.text == ""
        assert result.total_chars == 0
        assert result.truncated is False

    def test_formats_entries_correctly(self, sample_retrieved_items):
        """Test that entries are formatted in ENTRY #id format."""
        assembler = ContextWindowAssembler()
        result = assembler.assemble(sample_retrieved_items[:1], AssemblyConfig())

        assert "ENTRY #test-000" in result.text
        assert "Author:" in result.text

    def test_respects_max_chars(self, sample_retrieved_items):
        """Test that total character limit is respected."""
        assembler = ContextWindowAssembler()
        config = AssemblyConfig(max_chars=100, max_items=10)
        result = assembler.assemble(sample_retrieved_items, config)

        assert result.total_chars <= 100 + 10  # Allow for truncation suffix

    def test_respects_max_chars_per_item(self, sample_entry):
        """Test that per-item character limit is respected."""
        # Create entry with long text
        long_entry = dict(sample_entry)
        long_entry["raw_text"] = "A" * 5000

        item = RetrievedItem(long_entry, 0.9, "test", {})
        assembler = ContextWindowAssembler()
        config = AssemblyConfig(max_chars_per_item=100)
        result = assembler.assemble([item], config)

        # Should be truncated with "..."
        assert "..." in result.text


# ============================================================================
# Processor Tests
# ============================================================================


class TestIdentityProcessor:
    """Tests for IdentityProcessor."""

    @pytest.mark.asyncio
    async def test_processor_type(self):
        """Test processor type property."""
        processor = IdentityProcessor()
        assert processor.processor_type == "identity"

    @pytest.mark.asyncio
    async def test_passes_through_items(self, sample_retrieved_items):
        """Test that items are passed through unchanged."""
        processor = IdentityProcessor()
        context = AssembledContext(
            items=sample_retrieved_items[:2],
            text="context text",
            total_chars=100,
            truncated=False,
        )

        result = await processor.process("query", context, ProcessorConfig())

        assert result.answer is None
        assert result.items == sample_retrieved_items[:2]
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_extracts_citations(self, sample_retrieved_items):
        """Test that entry IDs are extracted as citations."""
        processor = IdentityProcessor()
        context = AssembledContext(
            items=sample_retrieved_items[:2],
            text="context",
            total_chars=100,
            truncated=False,
        )

        result = await processor.process("query", context, ProcessorConfig())

        assert "test-000" in result.citations
        assert "test-001" in result.citations


class TestSingleLLMProcessor:
    """Tests for SingleLLMProcessor."""

    @pytest.mark.asyncio
    async def test_processor_type(self):
        """Test processor type property."""
        processor = SingleLLMProcessor()
        assert processor.processor_type == "single_llm"

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test handling of empty context."""
        processor = SingleLLMProcessor()
        context = AssembledContext(items=[], text="", total_chars=0, truncated=False)

        result = await processor.process("query", context, ProcessorConfig())

        assert "don't have enough information" in result.answer
        assert result.items == []

    @pytest.mark.asyncio
    async def test_generates_answer(self, sample_retrieved_items):
        """Test that answer is generated."""
        with patch("osprey.models.completion.get_chat_completion") as mock_llm:
            mock_llm.return_value = "The answer is [#test-000] based on the entries."

            processor = SingleLLMProcessor()
            context = AssembledContext(
                items=sample_retrieved_items[:2],
                text="context text",
                total_chars=100,
                truncated=False,
            )

            result = await processor.process("query", context, ProcessorConfig())

            assert result.answer is not None
            assert "test-000" in result.citations

    def test_extract_citations(self):
        """Test citation extraction from text."""
        processor = SingleLLMProcessor()

        text = "Based on [#123] and [#456], the answer is clear."
        citations = processor._extract_citations(text)

        assert citations == ["123", "456"]

    def test_extract_citations_dedupes(self):
        """Test that duplicate citations are removed."""
        processor = SingleLLMProcessor()

        text = "[#123] confirms [#456] and [#123] also mentions."
        citations = processor._extract_citations(text)

        assert citations == ["123", "456"]


# ============================================================================
# Formatter Tests
# ============================================================================


class TestCitationFormatter:
    """Tests for CitationFormatter."""

    def test_formats_answer(self, sample_retrieved_items):
        """Test formatting with answer."""
        result = ProcessedResult(
            answer="The answer is [#test-000].",
            items=sample_retrieved_items[:1],
            citations=["test-000"],
        )

        formatter = CitationFormatter()
        response = formatter.format(result, {})

        assert response.format_type == "text"
        assert response.content == "The answer is [#test-000]."
        assert "test-000" in response.metadata["citations"]

    def test_formats_without_answer(self, sample_retrieved_items):
        """Test formatting when no answer (identity processor)."""
        result = ProcessedResult(
            answer=None,
            items=sample_retrieved_items[:2],
            citations=["test-000", "test-001"],
        )

        formatter = CitationFormatter()
        response = formatter.format(result, {})

        assert "[#test-000]" in response.content
        assert "[#test-001]" in response.content


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_formats_as_dict(self, sample_retrieved_items):
        """Test that output is a dictionary."""
        result = ProcessedResult(
            answer=None,
            items=sample_retrieved_items[:2],
            citations=["test-000", "test-001"],
        )

        formatter = JSONFormatter()
        response = formatter.format(result, {})

        assert response.format_type == "json"
        assert isinstance(response.content, dict)
        assert "entries" in response.content
        assert len(response.content["entries"]) == 2

    def test_includes_answer(self, sample_retrieved_items):
        """Test that answer is included when present."""
        result = ProcessedResult(
            answer="The answer is here.",
            items=sample_retrieved_items[:1],
            citations=["test-000"],
        )

        formatter = JSONFormatter()
        response = formatter.format(result, {})

        assert response.content["answer"] == "The answer is here."

    def test_truncates_text(self, sample_entry):
        """Test text truncation."""
        long_entry = dict(sample_entry)
        long_entry["raw_text"] = "A" * 1000

        item = RetrievedItem(long_entry, 0.9, "test", {})
        result = ProcessedResult(answer=None, items=[item], citations=[])

        formatter = JSONFormatter(max_text_length=100)
        response = formatter.format(result, {})

        assert len(response.content["entries"][0]["text"]) == 100


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipeline:
    """Tests for Pipeline composition."""

    @pytest.mark.asyncio
    async def test_executes_all_stages(self, sample_entries):
        """Test that all pipeline stages execute."""
        # Create mock retriever
        mock_retriever = MagicMock()
        mock_retriever.name = "test"
        mock_retriever.retrieve = AsyncMock(
            return_value=[RetrievedItem(sample_entries[0], 0.9, "test", {})]
        )

        pipeline = Pipeline(
            retriever=mock_retriever,
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )

        result = await pipeline.execute("test query")

        assert result.retrieval_count == 1
        assert result.assembly_count == 1
        assert result.processor_type == "identity"

    @pytest.mark.asyncio
    async def test_default_components(self, sample_entries):
        """Test that default components are used when not specified."""
        mock_retriever = MagicMock()
        mock_retriever.name = "test"
        mock_retriever.retrieve = AsyncMock(
            return_value=[RetrievedItem(sample_entries[0], 0.9, "test", {})]
        )

        # Only specify retriever
        pipeline = Pipeline(retriever=mock_retriever)

        result = await pipeline.execute("test")

        # Should use defaults
        assert result.processor_type == "identity"
        assert result.response.format_type == "json"


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_builds_pipeline(self, sample_entries):
        """Test building pipeline with builder."""
        mock_retriever = MagicMock()
        mock_retriever.name = "test"

        pipeline = (
            PipelineBuilder()
            .with_retriever(mock_retriever)
            .with_assembler(TopKAssembler())
            .with_processor(IdentityProcessor())
            .with_formatter(JSONFormatter())
            .build()
        )

        assert pipeline._retriever == mock_retriever

    def test_requires_retriever(self):
        """Test that builder requires retriever."""
        with pytest.raises(ValueError, match="requires a retriever"):
            PipelineBuilder().build()


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


class TestProtocolConformance:
    """Tests that implementations conform to protocols."""

    def test_keyword_retriever_is_retriever(self, mock_repository, mock_ariel_config):
        """Test KeywordRetriever conforms to Retriever protocol."""
        from osprey.services.ariel_search.pipeline.protocols import Retriever

        retriever = KeywordRetriever(mock_repository, mock_ariel_config)
        assert isinstance(retriever, Retriever)

    def test_semantic_retriever_is_retriever(
        self, mock_repository, mock_ariel_config, mock_embedder
    ):
        """Test SemanticRetriever conforms to Retriever protocol."""
        from osprey.services.ariel_search.pipeline.protocols import Retriever

        retriever = SemanticRetriever(mock_repository, mock_ariel_config, mock_embedder)
        assert isinstance(retriever, Retriever)

    def test_topk_assembler_is_assembler(self):
        """Test TopKAssembler conforms to Assembler protocol."""
        from osprey.services.ariel_search.pipeline.protocols import Assembler

        assembler = TopKAssembler()
        assert isinstance(assembler, Assembler)

    def test_identity_processor_is_processor(self):
        """Test IdentityProcessor conforms to Processor protocol."""
        from osprey.services.ariel_search.pipeline.protocols import Processor

        processor = IdentityProcessor()
        assert isinstance(processor, Processor)

    def test_json_formatter_is_formatter(self):
        """Test JSONFormatter conforms to Formatter protocol."""
        from osprey.services.ariel_search.pipeline.protocols import Formatter

        formatter = JSONFormatter()
        assert isinstance(formatter, Formatter)
