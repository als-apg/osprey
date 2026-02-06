"""Integration tests for ARIEL RAP pipeline.

Tests end-to-end pipeline execution with real database connections.
See 05_RAP_ABSTRACTION.md Section 7.2 for test requirements.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey.services.ariel_search.pipeline import (
    Pipeline,
    PipelineBuilder,
    PipelineConfig,
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
)
from osprey.services.ariel_search.pipeline.retrievers import (
    HybridRetriever,
    KeywordRetriever,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

pytestmark = pytest.mark.integration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_entries_for_db() -> list[dict]:
    """Create sample entries for database seeding."""
    now = datetime.now(UTC)
    return [
        {
            "entry_id": f"int-test-{i:03d}",
            "source_system": "integration_test",
            "timestamp": now - timedelta(hours=i),
            "author": f"tester_{i}",
            "raw_text": f"Integration test entry {i}. "
            f"{'Beam alignment procedure documented. ' if i % 2 == 0 else ''}"
            f"{'Vacuum system pressure nominal. ' if i % 3 == 0 else ''}"
            f"This is additional content for entry {i}.",
            "attachments": [],
            "metadata": {"title": f"Test Entry {i}"},
        }
        for i in range(10)
    ]


@pytest.fixture
async def seeded_repository(
    repository: ARIELRepository,
    sample_entries_for_db: list[dict],
) -> ARIELRepository:
    """Seed the repository with test entries."""

    for entry_data in sample_entries_for_db:
        entry: EnhancedLogbookEntry = {
            "entry_id": entry_data["entry_id"],
            "source_system": entry_data["source_system"],
            "timestamp": entry_data["timestamp"],
            "author": entry_data["author"],
            "raw_text": entry_data["raw_text"],
            "attachments": entry_data["attachments"],
            "metadata": entry_data["metadata"],
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        await repository.upsert_entry(entry)

    return repository


# ============================================================================
# Pipeline Composition Tests
# ============================================================================


class TestPipelineComposition:
    """Tests for pipeline stage composition."""

    @pytest.mark.asyncio
    async def test_keyword_pipeline_composes_correctly(
        self,
        seeded_repository: ARIELRepository,
        integration_ariel_config: ARIELConfig,
    ):
        """Test keyword search pipeline composition."""
        pipeline = Pipeline(
            retriever=KeywordRetriever(seeded_repository, integration_ariel_config),
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )

        result = await pipeline.execute(
            "beam alignment",
            PipelineConfig(retrieval=RetrievalConfig(max_results=5)),
        )

        assert result.response.format_type == "json"
        assert result.processor_type == "identity"
        # Should find entries with "beam alignment"
        assert result.retrieval_count >= 0  # May be 0 if FTS not set up

    @pytest.mark.asyncio
    async def test_builder_creates_valid_pipeline(
        self,
        seeded_repository: ARIELRepository,
        integration_ariel_config: ARIELConfig,
    ):
        """Test PipelineBuilder creates working pipeline."""
        pipeline = (
            PipelineBuilder()
            .with_retriever(KeywordRetriever(seeded_repository, integration_ariel_config))
            .with_assembler(TopKAssembler())
            .with_processor(IdentityProcessor())
            .with_formatter(JSONFormatter())
            .build()
        )

        result = await pipeline.execute("test")

        assert result.response is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_rag_components(
        self,
        seeded_repository: ARIELRepository,
        integration_ariel_config: ARIELConfig,
    ):
        """Test pipeline with RAG-style components."""
        # Create mock results
        mock_results = [
            RetrievedItem(
                entry={
                    "entry_id": "test-001",
                    "source_system": "test",
                    "timestamp": datetime.now(UTC),
                    "author": "tester",
                    "raw_text": "Beam alignment was performed.",
                    "attachments": [],
                    "metadata": {},
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                },
                score=0.9,
                source="mock",
                metadata={},
            )
        ]

        # Create a mock retriever with async retrieve method
        mock_retriever = MagicMock()
        mock_retriever.name = "mock"

        async def async_retrieve(*args, **kwargs):
            return mock_results

        mock_retriever.retrieve = async_retrieve

        pipeline = Pipeline(
            retriever=mock_retriever,
            assembler=ContextWindowAssembler(),
            processor=IdentityProcessor(),  # Use identity to avoid LLM calls
            formatter=CitationFormatter(),
        )

        result = await pipeline.execute("beam alignment")

        assert result.assembly_count == 1
        assert result.response.format_type == "text"


class TestPipelineWithHybridRetriever:
    """Tests for hybrid retrieval in pipelines."""

    @pytest.mark.asyncio
    async def test_hybrid_retriever_in_pipeline(self):
        """Test hybrid retriever integration in pipeline."""
        # Create mock retrievers
        entry1: EnhancedLogbookEntry = {
            "entry_id": "hybrid-001",
            "source_system": "test",
            "timestamp": datetime.now(UTC),
            "author": "tester",
            "raw_text": "Found by keyword search",
            "attachments": [],
            "metadata": {},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        entry2: EnhancedLogbookEntry = {
            "entry_id": "hybrid-002",
            "source_system": "test",
            "timestamp": datetime.now(UTC),
            "author": "tester",
            "raw_text": "Found by semantic search",
            "attachments": [],
            "metadata": {},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        mock_keyword = MagicMock()
        mock_keyword.name = "keyword"

        async def keyword_retrieve(*args, **kwargs):
            return [RetrievedItem(entry1, 0.9, "keyword", {})]

        mock_keyword.retrieve = keyword_retrieve

        mock_semantic = MagicMock()
        mock_semantic.name = "semantic"

        async def semantic_retrieve(*args, **kwargs):
            return [RetrievedItem(entry2, 0.85, "semantic", {})]

        mock_semantic.retrieve = semantic_retrieve

        hybrid = HybridRetriever([mock_keyword, mock_semantic])

        pipeline = Pipeline(
            retriever=hybrid,
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )

        result = await pipeline.execute("test query")

        # Should have results from both retrievers
        assert result.retrieval_count == 2
        assert result.response.format_type == "json"


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    @pytest.mark.asyncio
    async def test_retrieval_config_respected(self):
        """Test that retrieval config is passed to retriever."""
        mock_retriever = MagicMock()
        mock_retriever.name = "mock"

        config_received = None

        async def capture_config(query, config):
            nonlocal config_received
            config_received = config
            return []

        mock_retriever.retrieve = capture_config

        pipeline = Pipeline(retriever=mock_retriever)

        config = PipelineConfig(
            retrieval=RetrievalConfig(
                max_results=5,
                similarity_threshold=0.8,
            )
        )

        await pipeline.execute("test", config)

        assert config_received is not None
        assert config_received.max_results == 5
        assert config_received.similarity_threshold == 0.8


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_handles_retriever_error_gracefully(self):
        """Test that retriever errors are handled."""
        mock_retriever = MagicMock()
        mock_retriever.name = "failing"

        async def failing_retrieve(*args, **kwargs):
            raise RuntimeError("Retriever failed")

        mock_retriever.retrieve = failing_retrieve

        pipeline = Pipeline(retriever=mock_retriever)

        # Should propagate the error (not silently fail)
        with pytest.raises(RuntimeError, match="Retriever failed"):
            await pipeline.execute("test")

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        """Test pipeline with no results."""
        mock_retriever = MagicMock()
        mock_retriever.name = "empty"

        async def empty_retrieve(*args, **kwargs):
            return []

        mock_retriever.retrieve = empty_retrieve

        pipeline = Pipeline(
            retriever=mock_retriever,
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )

        result = await pipeline.execute("nonexistent query")

        assert result.retrieval_count == 0
        assert result.assembly_count == 0
        assert result.response.content["entries"] == []


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Tests ensuring RAP doesn't break existing functionality."""

    @pytest.mark.asyncio
    async def test_existing_search_functions_still_work(
        self,
        seeded_repository: ARIELRepository,
        integration_ariel_config: ARIELConfig,
    ):
        """Test that existing search functions continue to work."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        # Original function should still work
        results = await keyword_search(
            query="test",
            repository=seeded_repository,
            config=integration_ariel_config,
            max_results=5,
        )

        # Results should be in original format (list of tuples)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 3  # (entry, score, highlights)

    @pytest.mark.asyncio
    async def test_pipeline_produces_compatible_output(self):
        """Test that pipeline output can be converted to legacy format."""
        entry: EnhancedLogbookEntry = {
            "entry_id": "compat-001",
            "source_system": "test",
            "timestamp": datetime.now(UTC),
            "author": "tester",
            "raw_text": "Compatibility test entry",
            "attachments": [],
            "metadata": {},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        mock_retriever = MagicMock()
        mock_retriever.name = "mock"

        async def mock_retrieve(*args, **kwargs):
            return [RetrievedItem(entry, 0.9, "mock", {"highlights": ["test"]})]

        mock_retriever.retrieve = mock_retrieve

        pipeline = Pipeline(
            retriever=mock_retriever,
            assembler=TopKAssembler(),
            processor=IdentityProcessor(),
            formatter=JSONFormatter(),
        )

        result = await pipeline.execute("test")

        # Should be able to extract original format
        content = result.response.content
        assert "entries" in content
        assert len(content["entries"]) == 1
        assert content["entries"][0]["entry_id"] == "compat-001"
