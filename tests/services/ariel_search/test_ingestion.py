"""Tests for ARIEL ingestion adapters.

Tests adapter functionality for ALS, JLab, ORNL, and generic JSON formats.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from osprey.services.ariel_search.config import ARIELConfig, IngestionConfig
from osprey.services.ariel_search.ingestion.adapters.als import (
    ALSLogbookAdapter,
    parse_als_categories,
    transform_als_attachments,
)
from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
from osprey.services.ariel_search.ingestion.adapters.jlab import JLabLogbookAdapter
from osprey.services.ariel_search.ingestion.adapters.ornl import ORNLLogbookAdapter
from osprey.services.ariel_search.ingestion import get_adapter, KNOWN_ADAPTERS
from osprey.services.ariel_search.exceptions import AdapterNotFoundError


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "ariel"


class TestKnownAdapters:
    """Tests for adapter discovery."""

    def test_known_adapters_has_als(self):
        """ALS adapter is registered."""
        assert "als_logbook" in KNOWN_ADAPTERS

    def test_known_adapters_has_jlab(self):
        """JLab adapter is registered."""
        assert "jlab_logbook" in KNOWN_ADAPTERS

    def test_known_adapters_has_ornl(self):
        """ORNL adapter is registered."""
        assert "ornl_logbook" in KNOWN_ADAPTERS

    def test_known_adapters_has_generic(self):
        """Generic adapter is registered."""
        assert "generic_json" in KNOWN_ADAPTERS


class TestGetAdapter:
    """Tests for get_adapter factory."""

    def test_get_adapter_no_config_raises(self):
        """Raises AdapterNotFoundError when ingestion not configured."""
        config = ARIELConfig.from_dict({"database": {"uri": "test"}})
        with pytest.raises(AdapterNotFoundError) as exc_info:
            get_adapter(config)
        assert "(none)" in str(exc_info.value.adapter_name)

    def test_get_adapter_unknown_raises(self):
        """Raises AdapterNotFoundError for unknown adapter."""
        config = ARIELConfig.from_dict({
            "database": {"uri": "test"},
            "ingestion": {"adapter": "unknown", "source_url": "/test"},
        })
        with pytest.raises(AdapterNotFoundError) as exc_info:
            get_adapter(config)
        assert exc_info.value.adapter_name == "unknown"
        assert "als_logbook" in exc_info.value.available_adapters


class TestParseAlsCategories:
    """Tests for ALS category parsing."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert parse_als_categories("") == []

    def test_single_category(self):
        """Single category is parsed."""
        assert parse_als_categories("Operations") == ["Operations"]

    def test_multiple_categories(self):
        """Multiple categories are parsed."""
        result = parse_als_categories("RF Systems,Maintenance")
        assert result == ["RF Systems", "Maintenance"]

    def test_leading_trailing_commas(self):
        """Leading/trailing commas are handled."""
        result = parse_als_categories(",Operations,RF,")
        assert result == ["Operations", "RF"]

    def test_whitespace_stripped(self):
        """Whitespace is stripped from categories."""
        result = parse_als_categories("  Operations , RF Systems  ")
        assert result == ["Operations", "RF Systems"]


class TestTransformAlsAttachments:
    """Tests for ALS attachment URL transformation."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = transform_als_attachments([], "https://example.com/")
        assert result == []

    def test_single_attachment(self):
        """Single attachment is transformed."""
        source = [{"url": "attachments/2024/01/photo.jpg"}]
        result = transform_als_attachments(source, "https://elog.als.lbl.gov/")
        assert len(result) == 1
        assert result[0]["url"] == "https://elog.als.lbl.gov/attachments/2024/01/photo.jpg"
        assert result[0]["filename"] == "photo.jpg"

    def test_trailing_slash_normalized(self):
        """Trailing slash in prefix is normalized."""
        source = [{"url": "attachments/photo.jpg"}]
        result = transform_als_attachments(source, "https://example.com")
        assert result[0]["url"] == "https://example.com/attachments/photo.jpg"

    def test_leading_slash_normalized(self):
        """Leading slash in path is normalized."""
        source = [{"url": "/attachments/photo.jpg"}]
        result = transform_als_attachments(source, "https://example.com/")
        assert result[0]["url"] == "https://example.com/attachments/photo.jpg"


class TestALSLogbookAdapter:
    """Tests for ALS adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with ALS ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "als_logbook", "source_url": source_url},
        })

    def test_source_system_name(self):
        """Source system name is correct."""
        config = self._make_config("/fake/path.jsonl")
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_system_name == "ALS eLog"

    def test_detect_file_source(self):
        """File path is detected as file source."""
        config = self._make_config("/path/to/file.jsonl")
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_type == "file"

    def test_detect_http_source(self):
        """HTTP URL is detected as HTTP source."""
        config = self._make_config("https://elog.als.lbl.gov/api")
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_type == "http"

    @pytest.mark.asyncio
    async def test_fetch_entries_from_file(self):
        """Entries are fetched from JSONL file."""
        fixture_path = FIXTURES_DIR / "sample_als_entries.jsonl"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = ALSLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=5):
            entries.append(entry)

        assert len(entries) == 5
        assert entries[0]["entry_id"] == "10001"
        assert entries[0]["author"] == "jsmith"
        assert "RF cavity" in entries[0]["raw_text"]

    @pytest.mark.asyncio
    async def test_fetch_entries_with_since_filter(self):
        """Since filter excludes older entries."""
        fixture_path = FIXTURES_DIR / "sample_als_entries.jsonl"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = ALSLogbookAdapter(config)

        # Filter to only entries after timestamp 1704080000 (Jan 1, 2024 12:00 UTC)
        since = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        entries = []
        async for entry in adapter.fetch_entries(since=since):
            entries.append(entry)

        # Should exclude entries before noon
        for entry in entries:
            assert entry["timestamp"] > since


class TestJLabLogbookAdapter:
    """Tests for JLab adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with JLab ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "jlab_logbook", "source_url": source_url},
        })

    def test_source_system_name(self):
        """Source system name is correct."""
        config = self._make_config("/fake/path.json")
        adapter = JLabLogbookAdapter(config)
        assert adapter.source_system_name == "JLab Logbook"

    @pytest.mark.asyncio
    async def test_fetch_entries_from_file(self):
        """Entries are fetched from JSON file."""
        fixture_path = FIXTURES_DIR / "sample_jlab_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = JLabLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=3):
            entries.append(entry)

        assert len(entries) == 3
        assert entries[0]["entry_id"] == "J20001"
        assert entries[0]["author"] == "operator1"
        assert "Hall A" in entries[0]["raw_text"]


class TestORNLLogbookAdapter:
    """Tests for ORNL adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with ORNL ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "ornl_logbook", "source_url": source_url},
        })

    def test_source_system_name(self):
        """Source system name is correct."""
        config = self._make_config("/fake/path.json")
        adapter = ORNLLogbookAdapter(config)
        assert adapter.source_system_name == "ORNL Logbook"

    @pytest.mark.asyncio
    async def test_fetch_entries_from_file(self):
        """Entries are fetched from JSON file."""
        fixture_path = FIXTURES_DIR / "sample_ornl_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = ORNLLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=3):
            entries.append(entry)

        assert len(entries) == 3
        assert entries[0]["entry_id"] == "SNS-2024-0001"
        assert "1.4 MW" in entries[0]["raw_text"]

    @pytest.mark.asyncio
    async def test_event_time_vs_entry_time(self):
        """Entry time is used for timestamp, event time stored in metadata."""
        fixture_path = FIXTURES_DIR / "sample_ornl_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = ORNLLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=1):
            entries.append(entry)

        # Timestamp uses entry_time (13:15 UTC), event_time should be in metadata
        assert entries[0]["timestamp"].hour == 13
        assert entries[0]["timestamp"].minute == 15
        assert "event_time" in entries[0]["metadata"]


class TestGenericJSONAdapter:
    """Tests for generic JSON adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with generic ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": source_url},
        })

    def test_source_system_name(self):
        """Source system name is correct."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        assert adapter.source_system_name == "Generic JSON"

    @pytest.mark.asyncio
    async def test_fetch_entries_from_file(self):
        """Entries are fetched from JSON file."""
        fixture_path = FIXTURES_DIR / "sample_generic_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = GenericJSONAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries():
            entries.append(entry)

        assert len(entries) == 5
        assert entries[0]["entry_id"] == "GEN-001"
        assert entries[0]["author"] == "jdoe"

    @pytest.mark.asyncio
    async def test_fetch_with_limit(self):
        """Limit parameter works correctly."""
        fixture_path = FIXTURES_DIR / "sample_generic_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = GenericJSONAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=2):
            entries.append(entry)

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_fetch_with_since_filter(self):
        """Since filter excludes older entries."""
        fixture_path = FIXTURES_DIR / "sample_generic_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = GenericJSONAdapter(config)

        # Use a date that filters some entries
        since = datetime(2024, 1, 3, tzinfo=UTC)
        entries = []
        async for entry in adapter.fetch_entries(since=since):
            entries.append(entry)

        # Should have fewer entries
        for entry in entries:
            assert entry["timestamp"] > since

    @pytest.mark.asyncio
    async def test_fetch_with_until_filter(self):
        """Until filter excludes newer entries."""
        fixture_path = FIXTURES_DIR / "sample_generic_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = GenericJSONAdapter(config)

        # Use a date that filters some entries
        until = datetime(2024, 1, 3, tzinfo=UTC)
        entries = []
        async for entry in adapter.fetch_entries(until=until):
            entries.append(entry)

        for entry in entries:
            assert entry["timestamp"] < until


class TestGenericJSONAdapterParseTimestamp:
    """Tests for GenericJSONAdapter timestamp parsing."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with generic ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": source_url},
        })

    def test_parse_unix_timestamp(self):
        """Parse Unix timestamp (int)."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        result = adapter._parse_timestamp(1704067200)
        assert result.year == 2024
        assert result.month == 1

    def test_parse_unix_timestamp_float(self):
        """Parse Unix timestamp (float)."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        result = adapter._parse_timestamp(1704067200.5)
        assert result.year == 2024

    def test_parse_iso8601(self):
        """Parse ISO 8601 timestamp."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        result = adapter._parse_timestamp("2024-01-15T10:30:00+00:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso8601_with_z(self):
        """Parse ISO 8601 timestamp with Z suffix."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        result = adapter._parse_timestamp("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_unix_string(self):
        """Parse Unix timestamp as string."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        result = adapter._parse_timestamp("1704067200")
        assert result.year == 2024

    def test_parse_invalid_raises(self):
        """Invalid timestamp raises ValueError."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)
        with pytest.raises(ValueError):
            adapter._parse_timestamp("not-a-date")


class TestGenericJSONAdapterConvertEntry:
    """Tests for GenericJSONAdapter entry conversion."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with generic ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": source_url},
        })

    def test_convert_entry_title_and_text(self):
        """Entry with both title and text combines them."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)

        data = {
            "id": "test-001",
            "timestamp": 1704067200,
            "title": "Test Title",
            "text": "Test body content",
            "author": "tester",
        }

        entry = adapter._convert_entry(data)

        assert entry["entry_id"] == "test-001"
        assert "Test Title" in entry["raw_text"]
        assert "Test body content" in entry["raw_text"]

    def test_convert_entry_title_only(self):
        """Entry with only title uses title as raw_text."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)

        data = {
            "id": "test-002",
            "timestamp": 1704067200,
            "title": "Just a title",
            "author": "tester",
        }

        entry = adapter._convert_entry(data)

        assert entry["raw_text"] == "Just a title"

    def test_convert_entry_with_attachments(self):
        """Entry with attachments parses them correctly."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)

        data = {
            "id": "test-003",
            "timestamp": 1704067200,
            "text": "Entry with attachment",
            "author": "tester",
            "attachments": [
                {"url": "http://example.com/file.pdf", "type": "pdf", "filename": "file.pdf"}
            ],
        }

        entry = adapter._convert_entry(data)

        assert len(entry["attachments"]) == 1
        assert entry["attachments"][0]["url"] == "http://example.com/file.pdf"
        assert entry["attachments"][0]["filename"] == "file.pdf"

    def test_convert_entry_with_metadata_fields(self):
        """Entry with optional metadata fields."""
        config = self._make_config("/fake/path.json")
        adapter = GenericJSONAdapter(config)

        data = {
            "id": "test-004",
            "timestamp": 1704067200,
            "text": "Entry with metadata",
            "author": "tester",
            "tags": ["tag1", "tag2"],
            "books": ["Book A"],
            "level": "INFO",
            "categories": ["Cat1"],
        }

        entry = adapter._convert_entry(data)

        assert entry["metadata"]["tags"] == ["tag1", "tag2"]
        assert entry["metadata"]["books"] == ["Book A"]
        assert entry["metadata"]["level"] == "INFO"
        assert entry["metadata"]["categories"] == ["Cat1"]


class TestJLabAdapterDetailed:
    """Additional tests for JLab adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with JLab ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "jlab_logbook", "source_url": source_url},
        })

    @pytest.mark.asyncio
    async def test_fetch_with_limit(self):
        """Limit parameter works correctly."""
        fixture_path = FIXTURES_DIR / "sample_jlab_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = JLabLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=1):
            entries.append(entry)

        assert len(entries) == 1


class TestORNLAdapterDetailed:
    """Additional tests for ORNL adapter."""

    def _make_config(self, source_url: str) -> ARIELConfig:
        """Create config with ORNL ingestion settings."""
        return ARIELConfig.from_dict({
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "ornl_logbook", "source_url": source_url},
        })

    @pytest.mark.asyncio
    async def test_fetch_with_limit(self):
        """Limit parameter works correctly."""
        fixture_path = FIXTURES_DIR / "sample_ornl_entries.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not available")

        config = self._make_config(str(fixture_path))
        adapter = ORNLLogbookAdapter(config)

        entries = []
        async for entry in adapter.fetch_entries(limit=1):
            entries.append(entry)

        assert len(entries) == 1
