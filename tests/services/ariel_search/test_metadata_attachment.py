"""Tests for metadata attachment extraction utility."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osprey.services.ariel_search.ingestion.metadata_attachment import (
    extract_metadata_from_attachments,
)
from osprey.services.ariel_search.models import EnhancedLogbookEntry


def _make_entry(**overrides) -> EnhancedLogbookEntry:
    """Create a minimal EnhancedLogbookEntry for testing."""
    now = datetime.now(UTC)
    base: EnhancedLogbookEntry = {
        "entry_id": "test-001",
        "source_system": "Test",
        "timestamp": now,
        "author": "tester",
        "raw_text": "hello",
        "attachments": [],
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class TestExtractMetadataFromAttachments:
    """Tests for extract_metadata_from_attachments."""

    @pytest.mark.asyncio
    async def test_no_attachments(self):
        """Entry with no attachments is unchanged."""
        entry = _make_entry()
        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_no_metadata_json_attachment(self):
        """Attachments that are not metadata.json are ignored."""
        entry = _make_entry(
            attachments=[{"url": "/files/plot.png", "filename": "plot.png"}]
        )
        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_local_metadata_json(self, tmp_path: Path):
        """Local metadata.json is read and merged."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps({"session_id": "abc", "model": "haiku"}))

        entry = _make_entry(
            attachments=[{"url": str(meta_file), "filename": "metadata.json"}],
            metadata={"existing_key": "preserved"},
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"]["session_id"] == "abc"
        assert entry["metadata"]["model"] == "haiku"
        assert entry["metadata"]["existing_key"] == "preserved"

    @pytest.mark.asyncio
    async def test_http_metadata_json(self):
        """HTTP metadata.json is fetched and merged."""
        mock_data = {"operator": "jane", "git_branch": "main"}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = lambda *a, **kw: mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        entry = _make_entry(
            attachments=[
                {"url": "https://example.com/metadata.json", "filename": "metadata.json"}
            ],
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await extract_metadata_from_attachments(entry)

        assert entry["metadata"]["operator"] == "jane"
        assert entry["metadata"]["git_branch"] == "main"

    @pytest.mark.asyncio
    async def test_malformed_json_is_skipped(self, tmp_path: Path):
        """Non-JSON metadata.json is skipped gracefully."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text("not valid json {{{")

        entry = _make_entry(
            attachments=[{"url": str(meta_file), "filename": "metadata.json"}],
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_non_dict_json_is_skipped(self, tmp_path: Path):
        """metadata.json that parses to a non-dict is skipped."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps(["a", "b"]))

        entry = _make_entry(
            attachments=[{"url": str(meta_file), "filename": "metadata.json"}],
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_missing_local_file_is_skipped(self):
        """Missing local file is skipped gracefully."""
        entry = _make_entry(
            attachments=[
                {"url": "/nonexistent/metadata.json", "filename": "metadata.json"}
            ],
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_case_insensitive_filename(self, tmp_path: Path):
        """Filename matching is case-insensitive."""
        meta_file = tmp_path / "METADATA.JSON"
        meta_file.write_text(json.dumps({"from_upper": True}))

        entry = _make_entry(
            attachments=[{"url": str(meta_file), "filename": "METADATA.JSON"}],
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"]["from_upper"] is True

    @pytest.mark.asyncio
    async def test_empty_url_is_skipped(self):
        """Attachment with empty URL is skipped."""
        entry = _make_entry(
            attachments=[{"url": "", "filename": "metadata.json"}],
        )

        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}


class TestAdapterMetadataMerge:
    """Tests that adapters merge top-level 'metadata' from source data."""

    def test_generic_adapter_merges_metadata(self):
        """GenericJSONAdapter merges data['metadata'] into entry metadata."""
        from osprey.services.ariel_search.ingestion.adapters.generic import (
            GenericJSONAdapter,
        )

        adapter = GenericJSONAdapter.__new__(GenericJSONAdapter)
        adapter.source_url = "/dev/null"

        data = {
            "id": "1",
            "timestamp": "1700000000",
            "title": "Test",
            "metadata": {"session_id": "sess-123", "custom": "value"},
        }

        entry = adapter._convert_entry(data)
        assert entry["metadata"]["session_id"] == "sess-123"
        assert entry["metadata"]["custom"] == "value"
        assert entry["metadata"]["title"] == "Test"

    def test_generic_adapter_ignores_non_dict_metadata(self):
        """GenericJSONAdapter ignores non-dict metadata fields."""
        from osprey.services.ariel_search.ingestion.adapters.generic import (
            GenericJSONAdapter,
        )

        adapter = GenericJSONAdapter.__new__(GenericJSONAdapter)
        adapter.source_url = "/dev/null"

        data = {
            "id": "2",
            "timestamp": "1700000000",
            "title": "Test",
            "metadata": "not-a-dict",
        }

        entry = adapter._convert_entry(data)
        assert "not-a-dict" not in entry["metadata"].values()
