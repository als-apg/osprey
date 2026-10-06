"""Tests for metadata sidecar extraction through the attachment fetcher."""

from __future__ import annotations

import json
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.ingestion import metadata_attachment as metadata_mod
from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
from osprey.services.ariel_search.ingestion.metadata_attachment import (
    SIDECAR_FETCH_TIMEOUT,
    SIDECAR_MAX_BYTES,
    extract_metadata_from_attachments,
)
from osprey.services.ariel_search.models import EnhancedLogbookEntry

LOGGER = "ariel.ingestion"


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


def _config(
    source_url: str, *, allowed_origins: list[str] | None = None, **ingestion
) -> ARIELConfig:
    data: dict[str, Any] = {
        "database": {"uri": "postgresql://test"},
        "ingestion": {"adapter": "generic_json", "source_url": source_url, **ingestion},
    }
    if allowed_origins is not None:
        data["attachments"] = {"allowed_origins": allowed_origins}
    return ARIELConfig.from_dict(data)


def _file_adapter(tmp_path: Path) -> GenericJSONAdapter:
    """A generic file source whose entries live in ``tmp_path``."""
    return GenericJSONAdapter(_config(str(tmp_path / "logbook.json")))


def _http_adapter(source_url: str, **kwargs: Any) -> GenericJSONAdapter:
    return GenericJSONAdapter(_config(source_url, **kwargs))


def _sidecar(tmp_path: Path, name: str, content: str | bytes) -> str:
    """Write a sidecar under ``tmp_path`` and return its relative url."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return name


class _NamedFileSource(GenericJSONAdapter):
    """A file source that declares its own sidecar names."""

    def __init__(self, config: ARIELConfig, names: tuple[str, ...]) -> None:
        super().__init__(config)
        self.metadata_sidecar_names = names


# -- merging from a file source ------------------------------------------------


@pytest.mark.real_fetch
class TestFileSourceSidecars:
    """A generic file source reads relative sidecars under its source's directory."""

    async def test_no_attachments(self, tmp_path: Path):
        entry = _make_entry()
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {}

    async def test_non_sidecar_attachments_are_ignored(self, tmp_path: Path):
        _sidecar(tmp_path, "plot.png", b"png")
        entry = _make_entry(attachments=[{"url": "plot.png", "filename": "plot.png"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {}

    async def test_relative_sidecar_is_read_and_merged(self, tmp_path: Path):
        url = _sidecar(tmp_path, "metadata.json", json.dumps({"session_id": "abc", "model": "m"}))
        entry = _make_entry(
            attachments=[{"url": url, "filename": "metadata.json"}],
            metadata={"existing_key": "preserved"},
        )

        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert entry["metadata"] == {"session_id": "abc", "model": "m", "existing_key": "preserved"}

    async def test_sidecar_in_a_subdirectory_is_read(self, tmp_path: Path):
        url = _sidecar(tmp_path, "entries/7/metadata.json", json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {"a": 1}

    async def test_malformed_json_is_skipped(self, tmp_path: Path, caplog):
        url = _sidecar(tmp_path, "metadata.json", "not valid json {{{")
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert entry["metadata"] == {}
        assert "metadata.json" in caplog.text

    async def test_non_dict_json_is_skipped(self, tmp_path: Path):
        url = _sidecar(tmp_path, "metadata.json", json.dumps(["a", "b"]))
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {}

    async def test_missing_file_is_skipped_with_its_code(self, tmp_path: Path, caplog):
        entry = _make_entry(attachments=[{"url": "metadata.json", "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert entry["metadata"] == {}
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "source_gone" in warnings[0].getMessage()

    async def test_absolute_path_is_refused_even_when_it_exists(self, tmp_path: Path, caplog):
        """Only relative paths under the source's directory are followed."""
        meta = tmp_path / "metadata.json"
        meta.write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": str(meta), "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert entry["metadata"] == {}
        assert "source_refused" in caplog.text

    async def test_parent_escape_is_refused(self, tmp_path: Path, caplog):
        base = tmp_path / "logs"
        base.mkdir()
        (tmp_path / "metadata.json").write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": "../metadata.json", "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(base))

        assert entry["metadata"] == {}
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "source_refused" in warnings[0].getMessage()

    async def test_oversize_sidecar_is_refused(self, tmp_path: Path, caplog):
        body = json.dumps({"pad": "x" * SIDECAR_MAX_BYTES})
        url = _sidecar(tmp_path, "metadata.json", body)
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert entry["metadata"] == {}
        assert "size_cap" in caplog.text

    async def test_case_insensitive_filename(self, tmp_path: Path):
        url = _sidecar(tmp_path, "METADATA.JSON", json.dumps({"from_upper": True}))
        entry = _make_entry(attachments=[{"url": url, "filename": "METADATA.JSON"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {"from_upper": True}

    async def test_empty_url_is_skipped(self, tmp_path: Path):
        entry = _make_entry(attachments=[{"url": "", "filename": "metadata.json"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {}


# -- without an adapter ----------------------------------------------------------


class TestWithoutAnAdapter:
    """With no adapter there is no origin set and no file base: nothing is fetched."""

    async def test_existing_sidecar_is_not_fetched(self, tmp_path: Path, attachment_fetch):
        meta = tmp_path / "metadata.json"
        meta.write_text(json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": str(meta), "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry)

        assert entry["metadata"] == {}
        assert attachment_fetch.calls == []

    async def test_http_sidecar_is_not_fetched(self, attachment_fetch):
        entry = _make_entry(
            attachments=[{"url": "https://example.com/metadata.json", "filename": "metadata.json"}]
        )
        await extract_metadata_from_attachments(entry)
        assert entry["metadata"] == {}
        assert attachment_fetch.calls == []


# -- the fetcher seam ------------------------------------------------------------


class TestFetcherSeam:
    """The sidecar step goes through the same fetcher as every attachment."""

    async def test_the_call_carries_cap_origins_adapter_and_budget(self, attachment_fetch):
        adapter = _http_adapter("https://logbook.example/entries.json")
        attachment_fetch.respond(FetchOutcome(data=json.dumps({"operator": "jane"}).encode()))
        url = "https://logbook.example/files/metadata.json"
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {"operator": "jane"}
        (call,) = attachment_fetch.calls
        assert call["url"] == url
        assert call["cap"] == SIDECAR_MAX_BYTES == 1024 * 1024
        assert call["origins"] == frozenset({("https", "logbook.example", 443)})
        assert call["adapter"] is adapter
        assert call["total"] == SIDECAR_FETCH_TIMEOUT == 5

    async def test_a_failed_fetch_warns_once_and_merges_nothing(self, attachment_fetch, caplog):
        adapter = _http_adapter("https://logbook.example/entries.json")
        attachment_fetch.respond(FetchOutcome(code="source_gone"))
        entry = _make_entry(
            attachments=[
                {"url": "https://logbook.example/metadata.json", "filename": "metadata.json"}
            ]
        )

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {}
        assert len(attachment_fetch.calls) == 1
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "source_gone" in warnings[0].getMessage()

    async def test_an_unopted_fetch_is_recorded_and_fails_the_test(
        self, tmp_path, attachment_fetch
    ):
        """A test that triggers a fetch without opting in fails at teardown."""
        entry = _make_entry(attachments=[{"url": "metadata.json", "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert [c["url"] for c in attachment_fetch.unopted] == ["metadata.json"]
        with pytest.raises(pytest.fail.Exception, match="unopted attachment fetch"):
            attachment_fetch.verify()
        attachment_fetch.unopted.clear()

    async def test_an_unopted_copy_fetch_is_recorded(self, tmp_path, attachment_fetch):
        """The copy path's reference is the fake too, so its absorbed calls still surface."""
        out = await copy_mod.fetch_attachment_bytes(
            "https://h.example/a.png", 10, frozenset(), _file_adapter(tmp_path)
        )

        assert out.data is None
        assert [c["url"] for c in attachment_fetch.unopted] == ["https://h.example/a.png"]
        with pytest.raises(pytest.fail.Exception, match="unopted attachment fetch"):
            attachment_fetch.verify()
        attachment_fetch.unopted.clear()

    def test_the_fake_replaces_both_consumer_references(self, attachment_fetch):
        from osprey.services.ariel_search.attachments import fetch as fetch_mod

        assert copy_mod.fetch_attachment_bytes is attachment_fetch
        assert metadata_mod.fetch_attachment_bytes is attachment_fetch
        assert fetch_mod.fetch_attachment_bytes is not attachment_fetch

    @pytest.mark.real_fetch
    def test_real_fetch_leaves_the_references_real(self, attachment_fetch):
        from osprey.services.ariel_search.attachments.fetch import fetch_attachment_bytes

        assert attachment_fetch is None
        assert metadata_mod.fetch_attachment_bytes is fetch_attachment_bytes
        assert copy_mod.fetch_attachment_bytes is fetch_attachment_bytes


# -- adapter-declared names ------------------------------------------------------


@pytest.mark.real_fetch
class TestAdapterDeclaredSidecarNames:
    """Which filenames count is the adapter's answer, not this module's."""

    async def test_adapter_declared_name_is_matched(self, tmp_path: Path):
        url = _sidecar(tmp_path, "entry-meta.json", json.dumps({"operator": "jane"}))
        entry = _make_entry(attachments=[{"url": url, "filename": "entry-meta.json"}])
        adapter = _NamedFileSource(_config(str(tmp_path / "logbook.json")), ("entry-meta.json",))

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {"operator": "jane"}

    async def test_a_name_the_adapter_does_not_declare_is_ignored(self, tmp_path: Path):
        url = _sidecar(tmp_path, "metadata.json", json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])
        adapter = _NamedFileSource(_config(str(tmp_path / "logbook.json")), ("entry-meta.json",))

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {}

    async def test_matching_is_case_insensitive(self, tmp_path: Path):
        url = _sidecar(tmp_path, "Metadata.JSON", json.dumps({"a": 1}))
        entry = _make_entry(attachments=[{"url": url, "filename": "Metadata.JSON"}])
        await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))
        assert entry["metadata"] == {"a": 1}

    async def test_unmatched_attachments_are_logged_at_debug(self, tmp_path: Path, caplog):
        entry = _make_entry(attachments=[{"url": "plot.png", "filename": "plot.png"}])

        with caplog.at_level("DEBUG", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=_file_adapter(tmp_path))

        assert "none named metadata.json" in caplog.text

    def test_every_builtin_adapter_declares_the_default(self):
        from osprey.services.ariel_search.ingestion.base import FacilityAdapter

        assert FacilityAdapter.metadata_sidecar_names == ("metadata.json",)


# -- adapter metadata merge (no sidecar involved) -------------------------------


class TestAdapterMetadataMerge:
    """Adapters merge top-level 'metadata' from source data."""

    def test_generic_adapter_merges_metadata(self):
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
        adapter = GenericJSONAdapter.__new__(GenericJSONAdapter)
        adapter.source_url = "/dev/null"

        data = {"id": "2", "timestamp": "1700000000", "title": "Test", "metadata": "not-a-dict"}

        entry = adapter._convert_entry(data)
        assert "not-a-dict" not in entry["metadata"].values()


# -- http sidecars against a local server ---------------------------------------


async def _sidecar_handler(_request: web.Request) -> web.Response:
    return web.json_response({"operator": "jane", "git_branch": "main"})


@pytest.fixture
async def server() -> AsyncIterator[TestServer]:
    app = web.Application()
    app.router.add_get("/files/metadata.json", _sidecar_handler)
    srv = TestServer(app)
    await srv.start_server()
    try:
        yield srv
    finally:
        await srv.close()


def _origin(srv: TestServer) -> str:
    return f"http://{srv.host}:{srv.port}"


@pytest.mark.real_fetch
class TestHttpSidecars:
    """An http sidecar is fetched only from the adapter's origins."""

    async def test_same_origin_sidecar_parses(self, server: TestServer):
        adapter = _http_adapter(f"{_origin(server)}/entries.json")
        url = f"{_origin(server)}/files/metadata.json"
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {"operator": "jane", "git_branch": "main"}

    async def test_off_origin_sidecar_is_refused_naming_the_key(self, server: TestServer, caplog):
        adapter = _http_adapter("https://logbook.example/entries.json")
        url = f"{_origin(server)}/files/metadata.json"
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        with caplog.at_level("WARNING", logger=LOGGER):
            await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {}
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "origin_not_allowed" in message
        assert "ariel.attachments.allowed_origins" in message

    async def test_listing_the_origin_makes_it_parse(self, server: TestServer):
        adapter = _http_adapter(
            "https://logbook.example/entries.json", allowed_origins=[_origin(server)]
        )
        url = f"{_origin(server)}/files/metadata.json"
        entry = _make_entry(attachments=[{"url": url, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert entry["metadata"] == {"operator": "jane", "git_branch": "main"}


# -- transport: TLS and proxy come from the adapter -----------------------------


class _RecordingResponse:
    status = 200
    content_length = None
    headers: dict[str, str] = {}

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.content = self

    async def iter_chunked(self, _size: int):
        yield self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingSession:
    def __init__(self, body: bytes, seen: dict[str, Any], **kwargs: Any) -> None:
        self._body = body
        self.seen = seen
        seen["session"] = kwargs

    def request(self, method: str, url: str, **kwargs: Any) -> _RecordingResponse:
        self.seen["request"] = {"method": method, "url": url, **kwargs}
        return _RecordingResponse(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def recorded(monkeypatch) -> dict[str, Any]:
    """Replace aiohttp's session with one that records what the fetcher sends."""
    seen: dict[str, Any] = {}
    body = json.dumps({"operator": "jane"}).encode()
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda **kwargs: _RecordingSession(body, seen, **kwargs)
    )
    return seen


@pytest.mark.real_fetch
class TestSidecarFetchUsesTheAdaptersTransport:
    """The sidecar fetch uses the same TLS context and proxy as the adapter."""

    URL = "https://example.com/files/metadata.json"

    async def test_ca_bundle_reaches_the_request(self, tmp_path: Path, recorded, monkeypatch):
        import certifi

        bundle = tmp_path / "site-ca.pem"
        bundle.write_bytes(Path(certifi.where()).read_bytes())
        adapter = _http_adapter("https://example.com/entries.json", ca_bundle=str(bundle))
        adapter_context = adapter._ssl_context()
        monkeypatch.setattr(adapter, "_ssl_context", lambda: adapter_context)
        entry = _make_entry(attachments=[{"url": self.URL, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        context = recorded["request"]["ssl"]
        assert context is adapter_context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert entry["metadata"] == {"operator": "jane"}

    async def test_verify_ssl_opt_out_reaches_the_request(self, recorded):
        adapter = _http_adapter("https://example.com/entries.json", verify_ssl=False)
        expected = adapter._ssl_context()
        entry = _make_entry(attachments=[{"url": self.URL, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        context = recorded["request"]["ssl"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_NONE == expected.verify_mode
        assert entry["metadata"] == {"operator": "jane"}

    async def test_the_adapters_proxy_connector_is_used(self, recorded, monkeypatch):
        adapter = _http_adapter("https://example.com/entries.json")
        sentinel = object()
        monkeypatch.setattr(adapter, "_create_connector", lambda: sentinel)
        entry = _make_entry(attachments=[{"url": self.URL, "filename": "metadata.json"}])

        await extract_metadata_from_attachments(entry, adapter=adapter)

        assert recorded["session"]["connector"] is sentinel
        assert entry["metadata"] == {"operator": "jane"}
