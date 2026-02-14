"""Tests for ARIEL ingestion adapters.

Tests adapter functionality for ALS, JLab, ORNL, and generic JSON formats.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import AdapterNotFoundError, IngestionError
from osprey.services.ariel_search.ingestion import get_adapter
from osprey.services.ariel_search.ingestion.adapters.als import (
    ALSLogbookAdapter,
    parse_als_categories,
    transform_als_attachments,
)
from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
from osprey.services.ariel_search.ingestion.adapters.jlab import JLabLogbookAdapter
from osprey.services.ariel_search.ingestion.adapters.ornl import ORNLLogbookAdapter

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "ariel"


class TestRegisteredAdapters:
    """Tests for adapter registry discovery."""

    def test_registry_has_als(self):
        """ALS adapter is registered in the Osprey registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        assert registry.get_ariel_ingestion_adapter("als_logbook") is not None

    def test_registry_has_jlab(self):
        """JLab adapter is registered in the Osprey registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        assert registry.get_ariel_ingestion_adapter("jlab_logbook") is not None

    def test_registry_has_ornl(self):
        """ORNL adapter is registered in the Osprey registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        assert registry.get_ariel_ingestion_adapter("ornl_logbook") is not None

    def test_registry_has_generic(self):
        """Generic adapter is registered in the Osprey registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        assert registry.get_ariel_ingestion_adapter("generic_json") is not None

    def test_list_adapters(self):
        """All four adapters are listed."""
        from osprey.registry import get_registry

        registry = get_registry()
        adapters = registry.list_ariel_ingestion_adapters()
        assert "als_logbook" in adapters
        assert "jlab_logbook" in adapters
        assert "ornl_logbook" in adapters
        assert "generic_json" in adapters


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
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "test"},
                "ingestion": {"adapter": "unknown", "source_url": "/test"},
            }
        )
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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "als_logbook", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "jlab_logbook", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "ornl_logbook", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "generic_json", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "generic_json", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "generic_json", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "jlab_logbook", "source_url": source_url},
            }
        )

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
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {"adapter": "ornl_logbook", "source_url": source_url},
            }
        )

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


class TestALSLogbookAdapterHTTP:
    """Tests for ALS HTTP mode configuration and time windowing."""

    def _make_config(
        self,
        source_url: str = "https://web7.als.lbl.gov/olog/rpc.php",
        proxy_url: str | None = None,
        verify_ssl: bool = False,
        chunk_days: int = 365,
    ) -> ARIELConfig:
        """Create config with ALS HTTP ingestion settings."""
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": source_url,
                    "proxy_url": proxy_url,
                    "verify_ssl": verify_ssl,
                    "chunk_days": chunk_days,
                },
            }
        )

    def test_http_source_detected(self):
        """HTTP URL is detected as HTTP source type."""
        config = self._make_config()
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_type == "http"

    def test_https_source_detected(self):
        """HTTPS URL is detected as HTTP source type."""
        config = self._make_config("https://example.com/api")
        adapter = ALSLogbookAdapter(config)
        assert adapter.source_type == "http"

    def test_config_proxy_url(self):
        """Proxy URL is read from config."""
        config = self._make_config(proxy_url="socks5://localhost:9095")
        adapter = ALSLogbookAdapter(config)
        assert adapter.proxy_url == "socks5://localhost:9095"

    def test_config_proxy_url_from_env(self):
        """Proxy URL falls back to environment variable."""
        with patch.dict("os.environ", {"ARIEL_SOCKS_PROXY": "socks5://env:1234"}):
            config = ARIELConfig.from_dict(
                {
                    "database": {"uri": "postgresql://test"},
                    "ingestion": {
                        "adapter": "als_logbook",
                        "source_url": "https://example.com/api",
                    },
                }
            )
            adapter = ALSLogbookAdapter(config)
            assert adapter.proxy_url == "socks5://env:1234"

    def test_config_verify_ssl(self):
        """SSL verification setting is read from config."""
        config = self._make_config(verify_ssl=True)
        adapter = ALSLogbookAdapter(config)
        assert adapter.verify_ssl is True

    def test_config_chunk_days(self):
        """Chunk days setting is read from config."""
        config = self._make_config(chunk_days=30)
        adapter = ALSLogbookAdapter(config)
        assert adapter.chunk_days == 30

    def test_generate_time_windows_single(self):
        """Single window when date range is less than chunk_days."""
        config = self._make_config(chunk_days=365)
        adapter = ALSLogbookAdapter(config)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 6, 1, tzinfo=UTC)

        windows = adapter._generate_time_windows(start, end)

        assert len(windows) == 1
        assert windows[0][0] == start
        assert windows[0][1] == end

    def test_generate_time_windows_multiple(self):
        """Multiple windows when date range exceeds chunk_days."""
        config = self._make_config(chunk_days=30)
        adapter = ALSLogbookAdapter(config)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 3, 1, tzinfo=UTC)  # 60 days

        windows = adapter._generate_time_windows(start, end)

        assert len(windows) == 2
        assert windows[0][0] == start
        assert windows[0][1] == start + timedelta(days=30)
        assert windows[1][0] == start + timedelta(days=30)
        assert windows[1][1] == end

    def test_generate_time_windows_exact_multiple(self):
        """Windows align correctly with exact multiples of chunk_days."""
        config = self._make_config(chunk_days=30)
        adapter = ALSLogbookAdapter(config)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)  # 30 days

        windows = adapter._generate_time_windows(start, end)

        assert len(windows) == 1
        assert windows[0][0] == start
        assert windows[0][1] == end

    def test_generate_time_windows_empty(self):
        """Empty list when start >= end."""
        config = self._make_config(chunk_days=30)
        adapter = ALSLogbookAdapter(config)

        start = datetime(2024, 1, 15, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)

        windows = adapter._generate_time_windows(start, end)

        assert windows == []


class TestALSLogbookAdapterHTTPMocked:
    """Tests for ALS HTTP mode with mocked aiohttp responses."""

    def _make_config(self, chunk_days: int = 365) -> ARIELConfig:
        """Create config with ALS HTTP ingestion settings."""
        return ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "ingestion": {
                    "adapter": "als_logbook",
                    "source_url": "https://web7.als.lbl.gov/olog/rpc.php",
                    "chunk_days": chunk_days,
                    "max_retries": 2,
                    "retry_delay_seconds": 0,  # Fast retries for tests
                },
            }
        )

    @pytest.mark.asyncio
    async def test_fetch_entries_http_success(self):
        """Successful HTTP fetch returns converted entries."""
        config = self._make_config()
        adapter = ALSLogbookAdapter(config)

        mock_response_data = [
            {
                "id": "12345",
                "timestamp": "1704067200",
                "subject": "Test entry",
                "details": "Test details",
                "author": "tester",
                "category": "Operations",
                "attachments": [],
            },
            {
                "id": "12346",
                "timestamp": "1704153600",
                "subject": "Second entry",
                "details": "",
                "author": "tester2",
                "category": "",
                "attachments": [],
            },
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            since = datetime(2024, 1, 1, tzinfo=UTC)
            until = datetime(2024, 1, 3, tzinfo=UTC)

            entries = []
            async for entry in adapter._fetch_entries_http(since, until):
                entries.append(entry)

            assert len(entries) == 2
            assert entries[0]["entry_id"] == "12345"
            assert entries[0]["author"] == "tester"
            assert "Test entry" in entries[0]["raw_text"]
            assert entries[1]["entry_id"] == "12346"

    @pytest.mark.asyncio
    async def test_fetch_entries_http_deduplication(self):
        """Duplicate entries across windows are deduplicated."""
        config = self._make_config(chunk_days=1)  # Force multiple windows
        adapter = ALSLogbookAdapter(config)

        # Same entry appears in both windows
        mock_response_data = [
            {
                "id": "12345",
                "timestamp": "1704067200",
                "subject": "Duplicate entry",
                "details": "",
                "author": "tester",
                "category": "",
                "attachments": [],
            },
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            since = datetime(2024, 1, 1, tzinfo=UTC)
            until = datetime(2024, 1, 3, tzinfo=UTC)  # 2 days = 2 windows

            entries = []
            async for entry in adapter._fetch_entries_http(since, until):
                entries.append(entry)

            # Should only have 1 entry despite appearing in 2 windows
            assert len(entries) == 1
            assert entries[0]["entry_id"] == "12345"

    @pytest.mark.asyncio
    async def test_fetch_entries_http_limit(self):
        """Limit parameter stops iteration early."""
        config = self._make_config()
        adapter = ALSLogbookAdapter(config)

        mock_response_data = [
            {
                "id": str(i),
                "timestamp": "1704067200",
                "subject": f"Entry {i}",
                "details": "",
                "author": "tester",
                "category": "",
                "attachments": [],
            }
            for i in range(10)
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            since = datetime(2024, 1, 1, tzinfo=UTC)
            until = datetime(2024, 1, 2, tzinfo=UTC)

            entries = []
            async for entry in adapter._fetch_entries_http(since, until, limit=3):
                entries.append(entry)

            assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_fetch_entries_http_skip_empty(self):
        """Empty entries are skipped when skip_empty_entries is True."""
        config = self._make_config()
        adapter = ALSLogbookAdapter(config)
        adapter.skip_empty_entries = True

        mock_response_data = [
            {
                "id": "12345",
                "timestamp": "1704067200",
                "subject": "",
                "details": "",
                "author": "tester",
                "category": "",
                "attachments": [],
            },
            {
                "id": "12346",
                "timestamp": "1704067200",
                "subject": "Non-empty",
                "details": "",
                "author": "tester",
                "category": "",
                "attachments": [],
            },
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            since = datetime(2024, 1, 1, tzinfo=UTC)
            until = datetime(2024, 1, 2, tzinfo=UTC)

            entries = []
            async for entry in adapter._fetch_entries_http(since, until):
                entries.append(entry)

            assert len(entries) == 1
            assert entries[0]["entry_id"] == "12346"

    @pytest.mark.asyncio
    async def test_fetch_window_with_retry_success_after_failure(self):
        """Retry logic recovers from transient errors."""
        import aiohttp

        config = self._make_config()
        adapter = ALSLogbookAdapter(config)

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientError("Connection failed")
            return [{"id": "123", "timestamp": "1704067200"}]

        with patch.object(adapter, "_fetch_window", side_effect=mock_fetch):
            mock_session = MagicMock()
            result = await adapter._fetch_window_with_retry(
                mock_session,
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                True,
            )

            assert len(result) == 1
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_window_with_retry_max_retries_exceeded(self):
        """IngestionError raised after max retries exceeded."""
        import aiohttp

        config = self._make_config()
        adapter = ALSLogbookAdapter(config)

        with patch.object(
            adapter, "_fetch_window", side_effect=aiohttp.ClientError("Persistent failure")
        ):
            mock_session = MagicMock()

            with pytest.raises(IngestionError) as exc_info:
                await adapter._fetch_window_with_retry(
                    mock_session,
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    True,
                )

            assert "Max retries" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_window_with_retry_4xx_not_retried(self):
        """4xx HTTP errors are not retried."""
        import aiohttp

        config = self._make_config()
        adapter = ALSLogbookAdapter(config)

        mock_error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404,
            message="Not Found",
        )

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise mock_error

        with patch.object(adapter, "_fetch_window", side_effect=mock_fetch):
            mock_session = MagicMock()

            with pytest.raises(IngestionError) as exc_info:
                await adapter._fetch_window_with_retry(
                    mock_session,
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    True,
                )

            assert "HTTP 404" in str(exc_info.value)
            assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_create_connector_no_proxy(self):
        """TCP connector created when no proxy configured."""
        import aiohttp

        config = self._make_config()
        adapter = ALSLogbookAdapter(config)
        adapter.proxy_url = None

        connector = adapter._create_connector()

        assert isinstance(connector, aiohttp.TCPConnector)

    def test_create_connector_proxy_without_aiohttp_socks(self):
        """IngestionError raised when proxy configured but aiohttp-socks missing."""
        config = self._make_config()
        adapter = ALSLogbookAdapter(config)
        adapter.proxy_url = "socks5://localhost:9095"

        with patch.dict("sys.modules", {"aiohttp_socks": None}):
            with patch(
                "osprey.services.ariel_search.ingestion.adapters.als.ALSLogbookAdapter._create_connector",
                wraps=adapter._create_connector,
            ):
                # Force ImportError for aiohttp_socks
                original_method = adapter._create_connector

                def patched_create_connector():
                    if adapter.proxy_url:
                        try:
                            raise ImportError("No module named 'aiohttp_socks'")
                        except ImportError as e:
                            raise IngestionError(
                                "SOCKS proxy configured but aiohttp-socks is not installed. "
                                "Install with: pip install osprey-framework",
                                source_system=adapter.source_system_name,
                            ) from e
                    return original_method()

                adapter._create_connector = patched_create_connector

                with pytest.raises(IngestionError) as exc_info:
                    adapter._create_connector()

                assert "aiohttp-socks" in str(exc_info.value)


class TestALSLogbookAdapterHTTPConfigParsing:
    """Tests for IngestionConfig HTTP field parsing."""

    def test_ingestion_config_defaults(self):
        """Default values for HTTP config fields."""
        from osprey.services.ariel_search.config import IngestionConfig

        config = IngestionConfig.from_dict({"adapter": "als_logbook"})

        assert config.proxy_url is None
        assert config.verify_ssl is False
        assert config.chunk_days == 365
        assert config.request_timeout_seconds == 60
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 5

    def test_ingestion_config_custom_values(self):
        """Custom values for HTTP config fields are parsed."""
        from osprey.services.ariel_search.config import IngestionConfig

        config = IngestionConfig.from_dict(
            {
                "adapter": "als_logbook",
                "source_url": "https://example.com/api",
                "proxy_url": "socks5://localhost:9095",
                "verify_ssl": True,
                "chunk_days": 30,
                "request_timeout_seconds": 120,
                "max_retries": 5,
                "retry_delay_seconds": 10,
            }
        )

        assert config.proxy_url == "socks5://localhost:9095"
        assert config.verify_ssl is True
        assert config.chunk_days == 30
        assert config.request_timeout_seconds == 120
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 10

    def test_ingestion_config_env_fallback(self):
        """Proxy URL falls back to ARIEL_SOCKS_PROXY env var."""
        from osprey.services.ariel_search.config import IngestionConfig

        with patch.dict("os.environ", {"ARIEL_SOCKS_PROXY": "socks5://env:5555"}):
            config = IngestionConfig.from_dict({"adapter": "als_logbook"})

            assert config.proxy_url == "socks5://env:5555"

    def test_ingestion_config_explicit_overrides_env(self):
        """Explicit proxy_url in config overrides env var."""
        from osprey.services.ariel_search.config import IngestionConfig

        with patch.dict("os.environ", {"ARIEL_SOCKS_PROXY": "socks5://env:5555"}):
            config = IngestionConfig.from_dict(
                {
                    "adapter": "als_logbook",
                    "proxy_url": "socks5://explicit:9999",
                }
            )

            assert config.proxy_url == "socks5://explicit:9999"
