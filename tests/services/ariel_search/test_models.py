"""Tests for ARIEL core models."""

from datetime import UTC, datetime

import pytest

from osprey.services.ariel_search.models import (
    ARIELSearchRequest,
    ARIELSearchResult,
    ARIELStatusResult,
    EmbeddingTableInfo,
    IngestionEntryError,
    IngestionProgress,
    IngestionResult,
    MetadataSchema,
    SearchMode,
    enhanced_entry_from_row,
    resolve_time_range,
)


class TestSearchMode:
    """Tests for SearchMode enumeration."""

    def test_mode_values(self) -> None:
        """Test that all expected modes exist."""
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.RAG.value == "rag"
        assert SearchMode.AGENT.value == "agent"


class TestARIELSearchRequest:
    """Tests for ARIELSearchRequest."""

    def test_basic_creation(self) -> None:
        """Test basic request creation."""
        request = ARIELSearchRequest(query="test query")
        assert request.query == "test query"
        assert request.modes == [SearchMode.RAG]
        assert request.time_range is None
        assert request.facility is None
        assert request.max_results == 10
        assert request.include_images is False

    def test_with_all_fields(self) -> None:
        """Test request with all fields."""
        now = datetime.now(UTC)
        time_range = (now, now)
        request = ARIELSearchRequest(
            query="test query",
            modes=[SearchMode.KEYWORD, SearchMode.SEMANTIC],
            time_range=time_range,
            facility="ALS",
            max_results=50,
            include_images=True,
            capability_context_data={"key": "value"},
        )
        assert request.modes == [SearchMode.KEYWORD, SearchMode.SEMANTIC]
        assert request.time_range == time_range
        assert request.facility == "ALS"
        assert request.max_results == 50
        assert request.include_images is True
        assert request.capability_context_data == {"key": "value"}

    def test_empty_query_raises(self) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="query is required"):
            ARIELSearchRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="query is required"):
            ARIELSearchRequest(query="   ")

    def test_max_results_clamped_low(self) -> None:
        """Test that max_results is clamped to minimum 1."""
        request = ARIELSearchRequest(query="test", max_results=0)
        assert request.max_results == 1

    def test_max_results_clamped_high(self) -> None:
        """Test that max_results is clamped to maximum 100."""
        request = ARIELSearchRequest(query="test", max_results=500)
        assert request.max_results == 100


class TestARIELSearchResult:
    """Tests for ARIELSearchResult."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = ARIELSearchResult(entries=())
        assert result.entries == ()
        assert result.answer is None
        assert result.sources == ()
        assert result.search_modes_used == ()
        assert result.reasoning == ""

    def test_immutability(self) -> None:
        """Test that result is immutable (frozen)."""
        result = ARIELSearchResult(entries=())
        with pytest.raises(AttributeError):
            result.answer = "changed"  # type: ignore[misc]


class TestEmbeddingTableInfo:
    """Tests for EmbeddingTableInfo."""

    def test_basic_creation(self) -> None:
        """Test basic table info creation."""
        info = EmbeddingTableInfo(
            table_name="text_embeddings_nomic_embed_text",
            entry_count=127500,
        )
        assert info.table_name == "text_embeddings_nomic_embed_text"
        assert info.entry_count == 127500
        assert info.dimension is None
        assert info.is_active is False

    def test_with_all_fields(self) -> None:
        """Test table info with all fields."""
        info = EmbeddingTableInfo(
            table_name="text_embeddings_nomic_embed_text",
            entry_count=127500,
            dimension=768,
            is_active=True,
        )
        assert info.dimension == 768
        assert info.is_active is True


class TestARIELStatusResult:
    """Tests for ARIELStatusResult."""

    def test_basic_creation(self) -> None:
        """Test basic status result creation."""
        result = ARIELStatusResult(
            healthy=True,
            database_connected=True,
            database_uri="postgresql://***@localhost:5432/ariel",
            entry_count=127500,
            embedding_tables=[],
            active_embedding_model="nomic-embed-text",
            enabled_search_modules=["keyword", "semantic"],
            enabled_pipelines=["rag", "agent"],
            enabled_enhancement_modules=["text_embedding"],
            last_ingestion=None,
            errors=[],
        )
        assert result.healthy is True
        assert result.database_connected is True
        assert result.entry_count == 127500


class TestIngestionModels:
    """Tests for ingestion-related models."""

    def test_ingestion_entry_error(self) -> None:
        """Test IngestionEntryError creation."""
        error = IngestionEntryError(
            entry_id="123",
            error="Parse error",
            raw_data='{"malformed":',
        )
        assert error.entry_id == "123"
        assert error.error == "Parse error"
        assert error.raw_data == '{"malformed":'

    def test_ingestion_progress(self) -> None:
        """Test IngestionProgress creation."""
        progress = IngestionProgress(
            total=1000,
            processed=500,
            succeeded=495,
            failed=5,
        )
        assert progress.total == 1000
        assert progress.processed == 500
        assert progress.succeeded == 495
        assert progress.failed == 5

    def test_ingestion_result(self) -> None:
        """Test IngestionResult creation."""
        result = IngestionResult(
            source_system="ALS eLog",
            total_entries=1000,
            succeeded=995,
            failed=5,
            errors=[IngestionEntryError(entry_id="1", error="Parse error")],
            duration_seconds=45.5,
        )
        assert result.source_system == "ALS eLog"
        assert result.total_entries == 1000
        assert result.succeeded == 995
        assert result.failed == 5
        assert len(result.errors) == 1
        assert result.duration_seconds == 45.5


class TestEnhancedEntryFromRow:
    """Tests for enhanced_entry_from_row factory function."""

    @pytest.fixture
    def sample_row(self) -> dict:
        """Sample database row as dict."""
        return {
            "entry_id": "123",
            "source_system": "ALS eLog",
            "timestamp": datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            "author": "jdoe",
            "raw_text": "Test entry content",
            "attachments": [{"url": "http://example.com/img.png", "type": "image/png"}],
            "metadata": {"logbook": "Operations"},
            "created_at": datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
            "updated_at": datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
        }

    def test_basic_conversion(self, sample_row: dict) -> None:
        """Test basic row to entry conversion."""
        entry = enhanced_entry_from_row(sample_row)
        assert entry["entry_id"] == "123"
        assert entry["source_system"] == "ALS eLog"
        assert entry["author"] == "jdoe"
        assert entry["raw_text"] == "Test entry content"
        assert len(entry["attachments"]) == 1

    def test_with_enhancement_fields(self, sample_row: dict) -> None:
        """Test conversion with enhancement fields."""
        sample_row["summary"] = "A test summary"
        sample_row["keywords"] = ["test", "entry"]
        sample_row["enhancement_status"] = {"text_embedding": {"status": "complete"}}

        entry = enhanced_entry_from_row(sample_row)
        assert entry["summary"] == "A test summary"
        assert entry["keywords"] == ["test", "entry"]
        assert entry["enhancement_status"] == {"text_embedding": {"status": "complete"}}

    def test_missing_optional_fields(self, sample_row: dict) -> None:
        """Test conversion with missing optional fields."""
        del sample_row["author"]
        del sample_row["attachments"]
        del sample_row["metadata"]

        entry = enhanced_entry_from_row(sample_row)
        assert entry["author"] == ""
        assert entry["attachments"] == []
        assert entry["metadata"] == {}

    def test_none_enhancement_fields_excluded(self, sample_row: dict) -> None:
        """Test that None enhancement fields are excluded."""
        sample_row["summary"] = None
        sample_row["keywords"] = None

        entry = enhanced_entry_from_row(sample_row)
        assert "summary" not in entry
        assert "keywords" not in entry


class TestResolveTimeRange:
    """Tests for resolve_time_range function."""

    @pytest.fixture
    def request_with_time_range(self) -> ARIELSearchRequest:
        """Request with time_range set."""
        return ARIELSearchRequest(
            query="test",
            time_range=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            ),
        )

    @pytest.fixture
    def request_without_time_range(self) -> ARIELSearchRequest:
        """Request without time_range."""
        return ARIELSearchRequest(query="test")

    def test_tool_params_override_request(
        self, request_with_time_range: ARIELSearchRequest
    ) -> None:
        """Test that tool params override request context."""
        tool_start = datetime(2023, 1, 1, tzinfo=UTC)
        tool_end = datetime(2023, 12, 31, tzinfo=UTC)

        start, end = resolve_time_range(tool_start, tool_end, request_with_time_range)
        assert start == tool_start
        assert end == tool_end

    def test_partial_tool_params_override(
        self, request_with_time_range: ARIELSearchRequest
    ) -> None:
        """Test that partial tool params override request context."""
        tool_start = datetime(2023, 1, 1, tzinfo=UTC)

        start, end = resolve_time_range(tool_start, None, request_with_time_range)
        assert start == tool_start
        assert end is None

    def test_fallback_to_request_context(self, request_with_time_range: ARIELSearchRequest) -> None:
        """Test fallback to request context when no tool params."""
        start, end = resolve_time_range(None, None, request_with_time_range)
        assert start == datetime(2024, 1, 1, tzinfo=UTC)
        assert end == datetime(2024, 1, 31, tzinfo=UTC)

    def test_no_filtering(self, request_without_time_range: ARIELSearchRequest) -> None:
        """Test no filtering when no params and no context."""
        start, end = resolve_time_range(None, None, request_without_time_range)
        assert start is None
        assert end is None


class TestMetadataSchema:
    """Tests for MetadataSchema TypedDict."""

    def test_als_metadata(self) -> None:
        """Test ALS-specific metadata fields."""
        metadata: MetadataSchema = {
            "logbook": "Operations",
            "tag": "Injection",
            "shift": "Day",
        }
        assert metadata["logbook"] == "Operations"

    def test_jlab_metadata(self) -> None:
        """Test JLab-specific metadata fields."""
        metadata: MetadataSchema = {
            "logbook_name": "CEBAF",
            "entry_type": "Operations",
            "references": ["ref1", "ref2"],
        }
        assert metadata["logbook_name"] == "CEBAF"

    def test_ornl_metadata(self) -> None:
        """Test ORNL-specific metadata fields."""
        metadata: MetadataSchema = {
            "event_time": "2024-01-15T10:30:00Z",
            "facility_section": "SNS",
        }
        assert metadata["event_time"] == "2024-01-15T10:30:00Z"
