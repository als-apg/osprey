"""Tests for metadata attachment extraction utility."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from osprey.services.ariel_search.config import IngestionConfig
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
        entry = _make_entry(attachments=[{"url": "/files/plot.png", "filename": "plot.png"}])
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
            attachments=[{"url": "https://example.com/metadata.json", "filename": "metadata.json"}],
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
            attachments=[{"url": "/nonexistent/metadata.json", "filename": "metadata.json"}],
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


class TestAdapterDeclaredSidecarNames:
    """Which filenames count is the adapter's answer, not this module's."""

    @pytest.mark.asyncio
    async def test_default_name_still_matches_without_an_adapter(self, tmp_path: Path):
        """A caller that names no adapter behaves exactly as before."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": str(meta_file), "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry)

        assert entry["metadata"] == {"a": 1}

    @pytest.mark.asyncio
    async def test_adapter_declared_name_is_matched(self, tmp_path: Path):
        """A facility whose sidecar is not metadata.json now gets its metadata."""
        meta_file = tmp_path / "entry-meta.json"
        meta_file.write_text(json.dumps({"operator": "jane"}))
        entry = _make_entry(attachments=[{"url": str(meta_file), "filename": "entry-meta.json"}])

        adapter = SimpleNamespace(metadata_sidecar_names=("entry-meta.json",))
        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {"operator": "jane"}

    @pytest.mark.asyncio
    async def test_a_name_the_adapter_does_not_declare_is_ignored(self, tmp_path: Path):
        """Declaring names narrows as well as widens."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": str(meta_file), "filename": "metadata.json"}])

        adapter = SimpleNamespace(metadata_sidecar_names=("entry-meta.json",))
        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {}

    @pytest.mark.asyncio
    async def test_matching_is_case_insensitive(self, tmp_path: Path):
        """Facilities are not consistent about case; the match should not care."""
        meta_file = tmp_path / "Metadata.JSON"
        meta_file.write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": str(meta_file), "filename": "Metadata.JSON"}])

        await extract_metadata_from_attachments(entry)

        assert entry["metadata"] == {"a": 1}

    @pytest.mark.asyncio
    async def test_unmatched_attachments_are_logged_at_debug(self, tmp_path: Path, caplog):
        """An entry with attachments and no sidecar says so rather than staying silent."""
        entry = _make_entry(attachments=[{"url": "/files/plot.png", "filename": "plot.png"}])

        with caplog.at_level("DEBUG", logger="ariel.ingestion"):
            await extract_metadata_from_attachments(entry)

        assert "none named metadata.json" in caplog.text

    @pytest.mark.asyncio
    async def test_every_builtin_adapter_declares_the_default(self):
        """The shipped adapters keep the convention they already had."""
        from osprey.services.ariel_search.ingestion.base import FacilityAdapter

        assert FacilityAdapter.metadata_sidecar_names == ("metadata.json",)


class TestSidecarFetchHonoursIngestionSettings:
    """The sidecar fetch uses the same transport as every other request in the ingest."""

    @staticmethod
    def _mock_session():
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = AsyncMock(return_value={"operator": "jane"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        calls: dict[str, Any] = {}

        mock_session = AsyncMock()

        def _get(*args, **kwargs):
            calls["get"] = kwargs
            return mock_resp

        mock_session.get = _get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session, calls

    @pytest.mark.asyncio
    async def test_ca_bundle_reaches_the_request(self, tmp_path: Path):
        """A site CA configured for ingestion verifies the sidecar fetch too."""
        import ssl

        bundle = tmp_path / "site-ca.pem"
        bundle.write_bytes(Path(ssl.get_default_verify_paths().openssl_cafile).read_bytes())

        entry = _make_entry(
            attachments=[{"url": "https://example.com/metadata.json", "filename": "metadata.json"}],
        )
        ingestion = IngestionConfig.from_dict({"adapter": "generic_json", "ca_bundle": str(bundle)})
        session, calls = self._mock_session()

        with patch("aiohttp.ClientSession", return_value=session):
            await extract_metadata_from_attachments(entry, ingestion=ingestion)

        context = calls["get"]["ssl"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert entry["metadata"] == {"operator": "jane"}

    @pytest.mark.asyncio
    async def test_verify_ssl_opt_out_reaches_the_request(self):
        """The opt-out applies here as well; nothing is verified twice-over."""
        import ssl

        entry = _make_entry(
            attachments=[{"url": "https://example.com/metadata.json", "filename": "metadata.json"}],
        )
        ingestion = IngestionConfig.from_dict({"adapter": "generic_json", "verify_ssl": False})
        session, calls = self._mock_session()

        with patch("aiohttp.ClientSession", return_value=session):
            await extract_metadata_from_attachments(entry, ingestion=ingestion)

        context = calls["get"]["ssl"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_NONE

    @pytest.mark.asyncio
    async def test_the_adapters_proxy_connector_is_used(self):
        """A SOCKS proxy the ingest goes through carries this request too."""
        entry = _make_entry(
            attachments=[{"url": "https://example.com/metadata.json", "filename": "metadata.json"}],
        )
        sentinel = object()
        adapter = SimpleNamespace(
            metadata_sidecar_names=("metadata.json",),
            _create_connector=lambda: sentinel,
        )
        session, _calls = self._mock_session()
        seen: dict[str, Any] = {}

        def _make_session(**kwargs):
            seen.update(kwargs)
            return session

        with patch("aiohttp.ClientSession", side_effect=_make_session):
            await extract_metadata_from_attachments(entry, adapter=adapter)

        assert seen["connector"] is sentinel
