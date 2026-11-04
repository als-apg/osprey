"""Tests for ARIEL web API routes."""

from __future__ import annotations

import copy
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.search.base import SearchToolDescriptor
from tests.interfaces.ariel._attachments import PNG_URL, attachment_entry, attachment_row
from tests.interfaces.ariel._routes_app import routes_app


def _build_mock_registry():
    """Build a mock registry that provides ARIEL search modules and pipelines."""
    registry = MagicMock()

    # Build mock search modules
    keyword_mod = types.ModuleType("keyword")
    keyword_mod.get_tool_descriptor = lambda: SearchToolDescriptor(  # type: ignore[attr-defined]
        name="keyword_search",
        description="Full-text keyword search",
        search_mode="keyword",
        args_schema=MagicMock(),
        execute=AsyncMock(),
        format_result=MagicMock(),
    )
    keyword_mod.get_parameter_descriptors = None  # type: ignore[attr-defined]

    semantic_mod = types.ModuleType("semantic")
    semantic_mod.get_tool_descriptor = lambda: SearchToolDescriptor(  # type: ignore[attr-defined]
        name="semantic_search",
        description="Semantic similarity search",
        search_mode="semantic",
        args_schema=MagicMock(),
        execute=AsyncMock(),
        format_result=MagicMock(),
        needs_embedder=True,
    )
    semantic_mod.get_parameter_descriptors = None  # type: ignore[attr-defined]

    # Registered but left disabled by the shared fixture config, so it stays out
    # of every existing test's capabilities; the hybrid tests enable it locally.
    hybrid_mod = types.ModuleType("hybrid")
    hybrid_mod.get_tool_descriptor = lambda: SearchToolDescriptor(  # type: ignore[attr-defined]
        name="hybrid_search",
        description="Hybrid retrieval with optional reranking",
        search_mode="hybrid",
        args_schema=MagicMock(),
        execute=AsyncMock(),
        format_result=MagicMock(),
    )
    hybrid_mod.get_parameter_descriptors = None  # type: ignore[attr-defined]

    registry.list_ariel_search_modules.return_value = ["keyword", "semantic", "hybrid"]
    registry.get_ariel_search_module.side_effect = lambda n: {
        "keyword": keyword_mod,
        "semantic": semantic_mod,
        "hybrid": hybrid_mod,
    }.get(n)

    return registry


def _enable_hybrid(service) -> None:
    """Give the mock service a config in which the hybrid module is enabled.

    Applied per test rather than in the shared ``mock_ariel_service`` fixture,
    whose config several other tests assert against verbatim.
    """
    service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
                "hybrid": {"enabled": True},
            },
            "default_search_mode": "keyword",
        }
    )


@pytest.fixture(autouse=True)
def _mock_registry():
    """Provide a mock registry for all ARIEL route tests."""
    registry = _build_mock_registry()
    with patch(
        "osprey.registry.get_registry",
        return_value=registry,
    ):
        yield


@pytest.fixture
def mock_ariel_service():
    """Mock ARIEL service."""
    service = AsyncMock()
    service.health_check = AsyncMock(return_value=(True, "Service healthy"))
    service.repository = AsyncMock()
    # A schema that records attachment copy state: native writers prepare pictures.
    service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(True, True))
    # A migrated store with no attachment rows; tests that need rows replace it.
    service.repository.get_attachment_rows = AsyncMock(return_value={})

    # Provide a real config so /api/capabilities works
    service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            },
        }
    )

    # Mock search result
    mock_result = MagicMock()
    mock_result.entries = []
    mock_result.answer = "Test answer"
    mock_result.sources = []
    mock_result.search_modes_used = []
    mock_result.reasoning = ""
    mock_result.total = 0
    mock_result.total_exact = True
    mock_result.has_more = False
    service.search = AsyncMock(return_value=mock_result)

    # Mock status
    mock_status = MagicMock()
    mock_status.healthy = True
    mock_status.database_connected = True
    mock_status.database_uri = "postgresql://localhost/ariel"
    mock_status.entry_count = 100
    mock_status.embedding_tables = []
    mock_status.active_embedding_model = "text-embedding-3-small"
    mock_status.enabled_search_modules = ["keyword", "semantic"]
    mock_status.enabled_enhancement_modules = []
    mock_status.last_ingestion = None
    mock_status.errors = []
    mock_status.enhancement_coverage = {}
    service.get_status = AsyncMock(return_value=mock_status)

    return service


@pytest.fixture
def test_app(mock_ariel_service):
    """Create a test FastAPI app with mocked service."""
    return routes_app(mock_ariel_service)


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


def test_search_endpoint_basic(client, mock_ariel_service):
    """Test basic search endpoint."""
    response = client.post(
        "/api/search",
        json={
            "query": "test query",
            "mode": "keyword",
            "max_results": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "answer" in data
    assert data["answer"] == "Test answer"
    assert "execution_time_ms" in data

    # Verify service was called
    mock_ariel_service.search.assert_called_once()


def test_search_endpoint_with_time_range(client, mock_ariel_service):
    """Test search with time range filter."""
    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "keyword",
            "max_results": 5,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T23:59:59",
        },
    )

    assert response.status_code == 200

    # Check that time_range was passed to service
    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["time_range"] is not None


@pytest.mark.parametrize("where", ["top_level", "advanced_params"])
def test_search_bare_end_date_runs_through_that_day(client, mock_ariel_service, where):
    """A bare end date keeps the entries written on that day."""
    dates = {"start_date": "2024-12-31", "end_date": "2024-12-31"}
    body: dict = {"query": "test", "mode": "keyword"}
    if where == "top_level":
        body.update(dates)
    else:
        body["advanced_params"] = dates

    response = client.post("/api/search", json=body)

    assert response.status_code == 200
    start, end = mock_ariel_service.search.call_args.kwargs["time_range"]
    assert (start.hour, start.minute) == (0, 0)
    assert (end.date().isoformat(), end.hour, end.minute, end.second) == (
        "2024-12-31",
        23,
        59,
        59,
    )


def test_search_accepts_epoch_second_bounds(client, mock_ariel_service):
    """A JSON number is epoch seconds, as the earlier datetime field read it."""
    response = client.post(
        "/api/search", json={"query": "test", "start_date": 1759708800, "end_date": 1759795200}
    )

    assert response.status_code == 200
    start, end = mock_ariel_service.search.call_args.kwargs["time_range"]
    assert start.isoformat() == "2025-10-06T00:00:00+00:00"
    assert end.isoformat() == "2025-10-07T00:00:00+00:00"


def test_search_accepts_epoch_seconds_in_advanced_params(client, mock_ariel_service):
    """A numeric bound in advanced_params is epoch seconds too."""
    response = client.post(
        "/api/search", json={"query": "test", "advanced_params": {"end_date": 1759795200}}
    )

    assert response.status_code == 200
    _start, end = mock_ariel_service.search.call_args.kwargs["time_range"]
    assert end.isoformat() == "2025-10-07T00:00:00+00:00"


def test_list_entries_accepts_an_epoch_second_bound(client, mock_ariel_service):
    """The entries listing reads a numeric bound as epoch seconds."""
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=0)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])

    response = client.get("/api/entries?end_date=1759795200")

    assert response.status_code == 200
    end = mock_ariel_service.repository.count_entries.call_args.kwargs["end"]
    assert end.isoformat() == "2025-10-07T00:00:00+00:00"


def test_search_rejects_a_non_iso_end_date(client, mock_ariel_service):
    """A malformed bound is a 422, not a failed search."""
    response = client.post("/api/search", json={"query": "test", "end_date": "yesterday"})

    assert response.status_code == 422
    mock_ariel_service.search.assert_not_called()


def test_list_entries_endpoint(client, mock_ariel_service):
    """Test list entries endpoint."""
    # Mock repository methods
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=100)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])

    response = client.get("/api/entries?page=1&page_size=20")

    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "total" in data
    assert data["total"] == 100
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data


def test_list_entries_bare_end_date_runs_through_that_day(client, mock_ariel_service):
    """The entries listing reads a bare end date as the whole day too."""
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=0)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])

    response = client.get("/api/entries?end_date=2024-12-31")

    assert response.status_code == 200
    end = mock_ariel_service.repository.count_entries.call_args.kwargs["end"]
    assert (end.hour, end.minute, end.second) == (23, 59, 59)


def test_list_entries_passes_pagination_and_filters(client, mock_ariel_service):
    """list_entries forwards offset (derived from page) and author/source filters
    to the repository, instead of silently dropping page/author/source_system."""
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=100)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])

    response = client.get("/api/entries?page=3&page_size=20&author=alice&source_system=ERF")

    assert response.status_code == 200
    _args, kwargs = mock_ariel_service.repository.search_by_time_range.call_args
    assert kwargs["limit"] == 20
    assert kwargs["offset"] == 40  # (page - 1) * page_size
    assert kwargs["author"] == "alice"
    assert kwargs["source_system"] == "ERF"

    # total_pages must reflect the filtered set, so count_entries gets the same
    # author/source filters (not an unfiltered whole-table count).
    _cargs, ckwargs = mock_ariel_service.repository.count_entries.call_args
    assert ckwargs["author"] == "alice"
    assert ckwargs["source_system"] == "ERF"


def test_list_entries_advertises_no_sort_order(client):
    """Every parameter the endpoint declares reaches the query.

    Ordering is newest-first and is not a parameter.
    """
    schema = client.get("/openapi.json").json()
    names = {
        param["name"] for param in schema["paths"]["/api/entries"]["get"].get("parameters", [])
    }

    assert "sort_order" not in names


def test_get_entry_endpoint(client, mock_ariel_service):
    """Test get single entry endpoint."""
    # Mock entry
    mock_entry = {
        "entry_id": "test-123",
        "source_system": "Test",
        "timestamp": datetime.now(),
        "author": "Test Author",
        "raw_text": "Test entry content",
        "attachments": [],
        "metadata": {},
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "summary": None,
        "keywords": [],
    }
    mock_ariel_service.repository.get_entry = AsyncMock(return_value=mock_entry)

    response = client.get("/api/entries/test-123")

    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "test-123"
    assert data["author"] == "Test Author"


def test_get_entry_not_found(client, mock_ariel_service):
    """Test get entry returns 404 when not found."""
    mock_ariel_service.repository.get_entry = AsyncMock(return_value=None)

    response = client.get("/api/entries/nonexistent")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_create_entry_endpoint_via_service(client, mock_ariel_service):
    """Test create entry delegates to service.create_entry()."""
    from osprey.services.ariel_search.models import (
        FacilityEntryCreateResult,
        SyncStatus,
    )

    mock_ariel_service.create_entry = AsyncMock(
        return_value=FacilityEntryCreateResult(
            entry_id="local-abc123def456",
            source_system="Generic JSON",
            sync_status=SyncStatus.LOCAL_ONLY,
            message="Entry local-abc123def456 created in Generic JSON",
        )
    )

    response = client.post(
        "/api/entries",
        json={
            "subject": "Test Entry",
            "details": "Test details",
            "author": "Test Author",
            "logbook": "Test Logbook",
            "tags": ["test", "example"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "local-abc123def456"
    assert data["sync_status"] == "local_only"
    assert data["source_system"] == "Generic JSON"
    assert "message" in data

    # Verify service.create_entry was called (not repository directly)
    mock_ariel_service.create_entry.assert_called_once()


def test_create_entry_endpoint_fallback(client, mock_ariel_service):
    """Test create entry falls back to direct DB insert when adapter doesn't support writes."""
    mock_ariel_service.create_entry = AsyncMock(
        side_effect=NotImplementedError("Adapter does not support writes")
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    response = client.post(
        "/api/entries",
        json={
            "subject": "Test Entry",
            "details": "Test details",
            "author": "Test Author",
            "logbook": "Test Logbook",
            "tags": ["test", "example"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"].startswith("ariel-")
    assert data["sync_status"] == "local_only"
    assert data["source_system"] == "ARIEL Web"
    assert "saved locally" in data["message"]

    # Verify fallback path used repository directly
    mock_ariel_service.repository.upsert_entry.assert_called_once()


def test_create_entry_auth_required_returns_401(client, mock_ariel_service):
    """Missing logbook credentials surface as 401 auth_required, NOT a local save.

    This is the core of the fix: AuthenticationRequiredError must not be conflated
    with the read-only-adapter fallback, so nothing is saved and the UI can prompt.
    """
    from osprey.services.ariel_search.exceptions import AuthenticationRequiredError

    mock_ariel_service.create_entry = AsyncMock(
        side_effect=AuthenticationRequiredError(
            "OLOG publishing requires credentials.",
            source_system="Example eLog",
        )
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    response = client.post(
        "/api/entries",
        json={"subject": "Test", "details": "Test details"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "auth_required"
    assert "credentials" in body["detail"].lower()
    # Nothing was saved locally.
    mock_ariel_service.repository.upsert_entry.assert_not_called()


def test_create_entry_publish_failure_returns_502(client, mock_ariel_service):
    """A genuine publish failure surfaces an error, NOT a silent local save."""
    from osprey.services.ariel_search.exceptions import IngestionError

    mock_ariel_service.create_entry = AsyncMock(
        side_effect=IngestionError(
            "Example olog write failed with HTTP 403: bad password",
            source_system="Example eLog",
        )
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    response = client.post(
        "/api/entries",
        json={"subject": "Test", "details": "Test details"},
    )

    assert response.status_code == 502
    assert "publish failed" in response.json()["detail"].lower()
    mock_ariel_service.repository.upsert_entry.assert_not_called()


def _upload_files():
    """A single small file payload for multipart upload tests."""
    return [("files", ("shot.png", b"\x89PNG\r\n\x1a\nfakeimage", "image/png"))]


def test_upload_auth_required_returns_401(client, mock_ariel_service):
    """Attachment-bearing submit also prompts for credentials instead of saving local."""
    from osprey.services.ariel_search.exceptions import AuthenticationRequiredError

    mock_ariel_service.create_entry = AsyncMock(
        side_effect=AuthenticationRequiredError("creds required", source_system="Example eLog")
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()
    mock_ariel_service.repository.store_attachment = AsyncMock()

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=_upload_files(),
    )

    assert response.status_code == 401
    assert response.json()["code"] == "auth_required"
    # Nothing persisted: no entry, no attachment.
    mock_ariel_service.repository.upsert_entry.assert_not_called()
    mock_ariel_service.repository.store_attachment.assert_not_called()


def test_upload_publish_failure_returns_502(client, mock_ariel_service):
    """Upload route surfaces a real publish failure instead of saving local-only."""
    from osprey.services.ariel_search.exceptions import IngestionError

    mock_ariel_service.create_entry = AsyncMock(
        side_effect=IngestionError("olog down", source_system="Example eLog")
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()
    mock_ariel_service.repository.store_attachment = AsyncMock()

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=_upload_files(),
    )

    assert response.status_code == 502
    mock_ariel_service.repository.store_attachment.assert_not_called()


def test_upload_falls_back_local_with_attachments(client, mock_ariel_service):
    """Read-only adapter: upload saves local AND stores its attachments."""
    mock_ariel_service.create_entry = AsyncMock(
        side_effect=NotImplementedError("Adapter does not support writes")
    )
    mock_ariel_service.repository.upsert_entry = AsyncMock()
    mock_ariel_service.repository.store_attachment = AsyncMock()
    mock_ariel_service.repository.get_entry = AsyncMock(
        return_value={
            "entry_id": "ariel-xyz",
            "source_system": "ARIEL Web",
            "timestamp": datetime.now(),
            "author": "Anonymous",
            "raw_text": "Test\n\nBody",
            "attachments": [],
            "metadata": {},
        }
    )

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=_upload_files(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sync_status"] == "local_only"
    assert data["attachment_count"] == 1
    mock_ariel_service.repository.insert_native_attachment.assert_called_once()
    mock_ariel_service.repository.store_attachment.assert_not_called()


def test_upload_publish_hands_the_files_to_the_service(client, mock_ariel_service):
    """The route passes each validated file to the service with the entry."""
    from osprey.services.ariel_search.models import (
        EntryAttachment,
        FacilityEntryCreateResult,
        SyncStatus,
    )

    mock_ariel_service.create_entry = AsyncMock(
        return_value=FacilityEntryCreateResult(
            entry_id="99999",
            source_system="Example eLog",
            sync_status=SyncStatus.PENDING_SYNC,
            message="Entry 99999 created in Example eLog",
            attachment_count=1,
        )
    )

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body", "auth_user": "op", "auth_password": "pw"},
        files=_upload_files(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "99999"
    assert data["sync_status"] == "pending_sync"
    assert data["attachment_count"] == 1
    assert mock_ariel_service.create_entry.await_args.kwargs["attachments"] == [
        EntryAttachment("shot.png", "image/png", b"\x89PNG\r\n\x1a\nfakeimage")
    ]
    # The logbook took no files, so the operator is told they stayed in the Logbook.
    assert "saved to this Logbook only" in data["message"]


def test_upload_publish_reports_files_published_to_the_logbook(client, mock_ariel_service):
    """When the logbook took the files, the response names it instead of the Logbook."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    mock_ariel_service.create_entry = AsyncMock(
        return_value=FacilityEntryCreateResult(
            entry_id="99999",
            source_system="Example eLog",
            sync_status=SyncStatus.SYNCED,
            message="Entry 99999 created in Example eLog",
            attachment_count=1,
            attachments_published=True,
        )
    )

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=_upload_files(),
    )

    assert response.status_code == 200
    message = response.json()["message"]
    assert "1 attachment(s) published to Example eLog" in message
    assert "Logbook only" not in message


def _mock_adapter(*, supports_write, requires_write_auth, source_system):
    adapter = MagicMock()
    adapter.supports_write = supports_write
    adapter.requires_write_auth = requires_write_auth
    adapter.source_system_name = source_system
    return adapter


@pytest.mark.usefixtures("mock_ariel_service")
def test_publish_info_requires_auth(client):
    """A write adapter that needs credentials reports requires_auth=True."""
    adapter = _mock_adapter(
        supports_write=True, requires_write_auth=True, source_system="Example eLog"
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    assert response.status_code == 200
    data = response.json()
    assert data["supports_write"] is True
    assert data["requires_auth"] is True
    assert data["source_system"] == "Example eLog"


@pytest.mark.usefixtures("mock_ariel_service")
@pytest.mark.parametrize(
    ("supports_write", "requires_write_auth", "source_system"),
    [
        # A no-auth write adapter publishes without creds.
        pytest.param(True, False, "Generic JSON", id="no-auth"),
        # A read-only adapter: credentials are irrelevant.
        pytest.param(False, True, "JLab Logbook", id="read-only"),
    ],
)
def test_publish_info_reports_requires_auth_false(
    client, supports_write, requires_write_auth, source_system
):
    """An adapter that needs no credentials to publish reports requires_auth=False."""
    adapter = _mock_adapter(
        supports_write=supports_write,
        requires_write_auth=requires_write_auth,
        source_system=source_system,
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    data = response.json()
    assert data["supports_write"] is supports_write
    assert data["requires_auth"] is False


@pytest.mark.usefixtures("mock_ariel_service")
def test_publish_info_no_adapter_configured(client):
    """No ingestion adapter configured degrades gracefully to read-only."""
    response = client.get("/api/publish-info")

    assert response.status_code == 200
    data = response.json()
    assert data["supports_write"] is False
    assert data["requires_auth"] is False


@pytest.mark.usefixtures("mock_ariel_service")
def test_status_endpoint(client):
    """Test status endpoint."""
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is True
    assert data["database_connected"] is True
    assert data["entry_count"] == 100
    assert data["active_embedding_model"] == "text-embedding-3-small"
    assert "keyword" in data["enabled_search_modules"]
    assert data["enhancement_coverage"] == {}


def test_status_endpoint_carries_enhancement_coverage(client, mock_ariel_service):
    """``enhancement_coverage`` is the service's count, in the badge-strip vocabulary."""
    coverage = {
        "text_embedding": {
            "label": "Text Embedding",
            "n/a": 0,
            "pending": 1,
            "stale": 2,
            "failed": 0,
            "complete": 3,
        }
    }
    mock_ariel_service.get_status.return_value.enhancement_coverage = coverage

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["enhancement_coverage"] == coverage


def test_entry_to_response_helper():
    """Test _entry_to_response helper function."""
    entry = {
        "entry_id": "test-123",
        "source_system": "Test",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "author": "Test Author",
        "raw_text": "Test content",
        "attachments": [],
        "metadata": {"key": "value"},
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "updated_at": datetime(2024, 1, 1, 12, 0, 0),
        "summary": "Test summary",
        "keywords": ["test"],
    }

    result = routes._entry_to_response(
        entry,
        attachment_rows=None,
        model_id=None,
        file_source=False,
        score=0.95,
        highlights=["highlight1"],
    )

    assert result.entry_id == "test-123"
    assert result.author == "Test Author"
    assert result.score == 0.95
    assert result.highlights == ["highlight1"]
    assert result.metadata == {"key": "value"}


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_search_enabled_mode_reaches_service(client, mock_ariel_service, mode):
    """An enabled module name is forwarded to the service verbatim."""
    response = client.post(
        "/api/search",
        json={"query": "test", "mode": mode, "max_results": 10},
    )

    assert response.status_code == 200
    assert mock_ariel_service.search.call_args.kwargs["mode"] == mode


def test_search_unknown_mode_rejected_with_available_modes(client, mock_ariel_service):
    """An unknown mode is a 400 listing the enabled modes, not a silent fallback.

    The API used to map anything it did not recognize onto keyword search, so a
    typo returned plausible-looking results for the wrong mode.
    """
    response = client.post(
        "/api/search",
        json={"query": "test", "mode": "keywrod", "max_results": 10},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unknown search mode 'keywrod'" in detail
    assert "keyword" in detail.split("Available modes:")[1]
    assert "semantic" in detail.split("Available modes:")[1]
    mock_ariel_service.search.assert_not_called()


def test_search_blank_mode_rejected(client, mock_ariel_service):
    """A malformed (blank) mode is rejected before the service is consulted."""
    response = client.post(
        "/api/search",
        json={"query": "test", "mode": "   ", "max_results": 10},
    )

    assert response.status_code == 400
    assert "search mode cannot be empty" in response.json()["detail"]
    mock_ariel_service.search.assert_not_called()


def test_capabilities_endpoint(client):
    """Test capabilities endpoint returns valid structure."""
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "shared_parameters" in data
    assert "direct" in data["categories"]

    # Should have keyword and semantic in direct category
    direct_names = [m["name"] for m in data["categories"]["direct"]["modes"]]
    assert "keyword" in direct_names
    assert "semantic" in direct_names

    # Shared parameters should include max_results
    param_names = [p["name"] for p in data["shared_parameters"]]
    assert "max_results" in param_names


def test_search_with_advanced_params(client, mock_ariel_service):
    """Test that advanced_params are forwarded to service."""
    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "keyword",
            "max_results": 10,
            "advanced_params": {"temperature": 0.5, "similarity_threshold": 0.8},
        },
    )

    assert response.status_code == 200

    # Verify advanced_params were forwarded
    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"] == {
        "temperature": 0.5,
        "similarity_threshold": 0.8,
    }


def test_search_defaults_to_keyword_mode(client, mock_ariel_service):
    """Test that omitting mode defaults to KEYWORD."""
    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "max_results": 10,
        },
    )

    assert response.status_code == 200

    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["mode"] == "keyword"


def test_search_honors_configured_default_mode(client, mock_ariel_service):
    """Omitting mode follows ariel.default_search_mode, not a fixed name."""
    mock_ariel_service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "test-model"},
            },
            "default_search_mode": "semantic",
        }
    )

    response = client.post("/api/search", json={"query": "test", "max_results": 10})

    assert response.status_code == 200
    assert mock_ariel_service.search.call_args.kwargs["mode"] == "semantic"


@pytest.mark.parametrize("value", ["yes", "false", 1, 0, 1.0, []])
def test_search_hybrid_rejects_non_boolean_rerank(client, mock_ariel_service, value):
    """A non-boolean ``rerank`` override is a 400, not a truthiness accident.

    The panel sends the toggle's boolean, but a hand-written caller can send
    ``"false"`` -- which is truthy everywhere downstream and would silently run
    the slow reranked path the caller asked to skip.
    """
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "hybrid",
            "max_results": 10,
            "advanced_params": {"rerank": value},
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "rerank must be a boolean" in detail
    assert repr(value) in detail
    mock_ariel_service.search.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, "40", 12.5, True])
def test_search_hybrid_rejects_bad_candidate_limit(client, mock_ariel_service, value):
    """``candidate_limit`` must be a positive int -- booleans and zero included."""
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "hybrid",
            "max_results": 10,
            "advanced_params": {"candidate_limit": value},
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "candidate_limit must be a positive integer" in detail
    assert repr(value) in detail
    mock_ariel_service.search.assert_not_called()


def test_search_hybrid_forwards_valid_overrides_verbatim(client, mock_ariel_service):
    """Well-formed overrides reach the service untouched -- ``False`` included.

    ``rerank: false`` is the whole point of the override, so it must survive as
    the boolean ``False`` rather than being dropped as falsy.
    """
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "hybrid",
            "max_results": 10,
            "advanced_params": {"rerank": False, "candidate_limit": 12},
        },
    )

    assert response.status_code == 200
    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["mode"] == "hybrid"
    assert call_kwargs["advanced_params"]["rerank"] is False
    assert call_kwargs["advanced_params"]["candidate_limit"] == 12


def test_search_hybrid_without_overrides_is_accepted(client, mock_ariel_service):
    """Absent keys mean "use the configured default" and are not rejected."""
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={"query": "test", "mode": "hybrid", "max_results": 10},
    )

    assert response.status_code == 200
    assert mock_ariel_service.search.call_args.kwargs["mode"] == "hybrid"


def test_search_hybrid_accepts_explicit_null_overrides(client, mock_ariel_service):
    """An explicit ``null`` says "no override" just as an absent key does."""
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "hybrid",
            "max_results": 10,
            "advanced_params": {"rerank": None, "candidate_limit": None},
        },
    )

    assert response.status_code == 200
    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["rerank"] is None
    assert call_kwargs["advanced_params"]["candidate_limit"] is None


@pytest.mark.parametrize("mode", ["keyword", "semantic"])
def test_search_non_hybrid_modes_ignore_hybrid_overrides(client, mock_ariel_service, mode):
    """The check is hybrid-only: other modes never see these keys as theirs.

    ``rerank`` and ``candidate_limit`` are hybrid's parameter names. Another
    module is free to give them any meaning, so validating them everywhere
    would reject requests this route has no business judging.
    """
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": mode,
            "max_results": 10,
            "advanced_params": {"rerank": "yes", "candidate_limit": 0},
        },
    )

    assert response.status_code == 200
    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["rerank"] == "yes"
    assert call_kwargs["advanced_params"]["candidate_limit"] == 0


def test_search_hybrid_leaves_expand_query_alone(client, mock_ariel_service):
    """Only the two hybrid keys are judged; ``expand_query`` passes through."""
    _enable_hybrid(mock_ariel_service)

    response = client.post(
        "/api/search",
        json={
            "query": "test",
            "mode": "hybrid",
            "max_results": 10,
            "advanced_params": {"expand_query": "yes", "rerank": True},
        },
    )

    assert response.status_code == 200
    assert mock_ariel_service.search.call_args.kwargs["advanced_params"]["expand_query"] == "yes"


@pytest.mark.usefixtures("mock_ariel_service")
def test_capabilities_advertises_default_mode(client):
    """The capabilities payload carries the mode the UI should open on."""
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert response.json()["default_mode"] == "keyword"


def test_capabilities_names_the_facility_zone(client):
    """The payload names the zone every entry timestamp is rendered in."""
    from zoneinfo import ZoneInfo

    with patch(
        "osprey.interfaces.ariel.api.routes.get_facility_timezone",
        return_value=ZoneInfo("Asia/Tokyo"),
    ):
        response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert response.json()["facility_timezone"] == "Asia/Tokyo"


def test_capabilities_no_longer_reports_embedded_search(client):
    """``ariel.embedded_search`` is retired: the panel always offers Search."""
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert "embedded_search" not in response.json()


def _attachments_config(*, image_embedding=True, hybrid=True, view=None) -> ARIELConfig:
    """An ARIEL config for one attachments-capability scenario."""
    section: dict = {
        "database": {"uri": "postgresql://localhost:5432/test"},
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": "test-model"},
            "hybrid": {"enabled": hybrid},
        },
        "enhancement_modules": {
            "image_embedding": {"enabled": image_embedding},
            "image_caption": {"enabled": True},
        },
    }
    if view is not None:
        section["attachments"] = {"view": {"enabled": view}}
    return ARIELConfig.from_dict(section)


def test_capabilities_reports_the_attachments_block(client, mock_ariel_service):
    """The web payload carries the service's attachments block unchanged."""
    from osprey.services.ariel_search.capabilities import attachments_capability

    config = _attachments_config()
    mock_ariel_service.config = config

    block = client.get("/api/capabilities").json()["attachments"]

    assert block == attachments_capability(config)
    assert set(block) == {
        "copy_on_ingest",
        "formats",
        "view",
        "captions",
        "picture_search",
        "picture_search_unavailable",
    }
    assert block["view"] is True
    assert block["picture_search"] is True


@pytest.mark.parametrize("missing", ["image_embedding", "hybrid"])
def test_capabilities_picture_search_needs_both_modules(client, mock_ariel_service, missing):
    """Picture search is false when either image embeddings or hybrid is disabled."""
    mock_ariel_service.config = _attachments_config(**{missing: False})

    block = client.get("/api/capabilities").json()["attachments"]

    assert block["picture_search"] is False


def test_capabilities_reports_view_disabled(client, mock_ariel_service):
    """``view.enabled: false`` flips only ``view``; the rest of the block is unchanged."""
    mock_ariel_service.config = _attachments_config()
    enabled = client.get("/api/capabilities").json()["attachments"]
    mock_ariel_service.config = _attachments_config(view=False)
    disabled = client.get("/api/capabilities").json()["attachments"]

    assert enabled["view"] is True
    assert disabled["view"] is False
    assert {k: v for k, v in disabled.items() if k != "view"} == {
        k: v for k, v in enabled.items() if k != "view"
    }


def test_put_config_backs_up_into_the_state_zone(client, tmp_path):
    """ARIEL's config save copies the old file into the agent-data state zone.

    Not beside ``config.yml``. That file lives in the render, which the container
    split makes root-owned: creating a *new* file next to it needs write
    permission on the render directory that the admin image will not have, and
    the backup runs before a byte of the save is written -- so the old sibling
    scheme would have turned every ARIEL config save in that image into a 500.
    Anchored on the directory the route already resolves its config path in.
    """
    from osprey.utils.config_writer import config_backup_path

    config_path = tmp_path / "config.yml"
    original = "project_name: original\n"
    config_path.write_text(original)
    client.app.state.config_path = config_path

    response = client.put("/api/config", json={"content": "project_name: updated\n"})

    assert response.status_code == 200
    assert config_path.read_text() == "project_name: updated\n"

    backup = config_backup_path(config_path)
    assert backup.read_text() == original
    assert backup.parent.name == "config-backups"
    # The point of the move: nothing new lands next to the config itself.
    assert not (tmp_path / "config.yml.bak").exists()
    assert [f.name for f in tmp_path.iterdir() if f.suffix == ".bak"] == []


def test_put_config_backup_follows_a_relocated_agent_data_root(client, tmp_path):
    """The zone is read from the config being written, never assumed.

    Resolved from the *pre-write* file, which is the only reading that makes
    sense: the backup is a copy of what is there now, so it belongs in the zone
    that config names now.

    The saved document carries ``agent_data`` through unchanged, because it has
    to: ``agent_data.*`` is in the protected set, so a body that dropped it
    would be refused before the backup ran and this would stop being a test of
    where the backup lands.
    """
    from osprey.utils.config_writer import config_backup_path

    relocated = tmp_path / "elsewhere" / "state"
    config_path = tmp_path / "config.yml"
    original = f"agent_data:\n  base_dir: {relocated}\nproject_name: original\n"
    config_path.write_text(original)
    expected = config_backup_path(config_path)
    assert expected == relocated / "config-backups" / "config.yml.bak"
    client.app.state.config_path = config_path

    response = client.put(
        "/api/config",
        json={"content": original.replace("project_name: original", "project_name: updated")},
    )

    assert response.status_code == 200
    assert expected.read_text() == original
    assert not (tmp_path / "var").exists()


# --------------------------------------------------------------------------
# PUT /api/config and the protected set
#
# ARIEL's Raw YAML save replaces the whole document, so it is the widest write
# surface onto the file that carries the write gate, the approval gate and the
# paths the safety layers derive their zones from. It is gated exactly the way
# the Web Terminal's ``PUT /api/config`` is -- same protected set, same 403,
# same ``http_config`` audit record -- because the protected set is
# consulted by *every* framework writer, not just the terminal's.
# --------------------------------------------------------------------------

_PROTECTED_DOC = (
    "agent_data:\n"
    "  base_dir: {state}\n"
    "control_system:\n"
    "  writes_enabled: false\n"
    "project_name: original\n"
)


@pytest.fixture
def audit_records(recorded_entries):
    """``audit_records() -> list[dict]``: the ``http_config`` audit bodies recorded so far."""
    from osprey.audit.protected import SURFACE_HTTP_CONFIG

    def read() -> list[dict]:
        entries = sorted(recorded_entries("audit"), key=lambda entry: entry.ts)
        return [dict(e.body) for e in entries if e.body["surface"] == SURFACE_HTTP_CONFIG]

    return read


@pytest.fixture
def gated_config(client, tmp_path):
    """A config.yml carrying protected keys, wired into the ARIEL app state."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(_PROTECTED_DOC.format(state=tmp_path / "state"))
    client.app.state.config_path = config_path
    return config_path


def test_put_config_refuses_a_removed_protected_key(client, gated_config, audit_records):
    """Dropping ``agent_data`` on the way past is a protected-key change, not a save."""
    before = gated_config.read_bytes()

    response = client.put(
        "/api/config",
        json={"content": "control_system:\n  writes_enabled: false\nproject_name: updated\n"},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "agent_data.base_dir" in detail
    assert "config.yml is unchanged" in detail
    # The operator is pointed at the channel that *can* carry the change.
    assert "`config:` block" in detail
    # Byte-identical: no write, and no backup either -- a backup is a copy of a
    # file this request may turn out not to be allowed to replace.
    assert gated_config.read_bytes() == before
    assert not (gated_config.parent / "state").exists()

    records = audit_records()
    assert len(records) == 1
    assert records[0]["surface"] == "http_config"
    assert "target=config.yml" in records[0]["detail"]
    assert records[0]["subject"] == "agent_data.base_dir"
    assert records[0]["reason"] == "protected_key"


def test_put_config_refuses_a_changed_protected_value(client, gated_config, audit_records):
    """Flipping the write gate through the YAML editor is the write that must not land."""
    before = gated_config.read_bytes()
    flipped = _PROTECTED_DOC.format(state=gated_config.parent / "state").replace(
        "writes_enabled: false", "writes_enabled: true"
    )

    response = client.put("/api/config", json={"content": flipped})

    assert response.status_code == 403
    assert "control_system.writes_enabled" in response.json()["detail"]
    assert gated_config.read_bytes() == before

    records = audit_records()
    assert [r["subject"] for r in records] == ["control_system.writes_enabled"]


@pytest.mark.usefixtures("gated_config")
def test_put_config_refusal_leaks_no_value(client, audit_records):
    """Config values are secrets; a refusal reports the key, never the value."""
    import json as _j

    sentinel = "qqzzSENTINELvalue77"
    planted = _PROTECTED_DOC.format(state=sentinel)

    response = client.put("/api/config", json={"content": planted})

    assert response.status_code == 403
    assert sentinel not in response.text
    assert sentinel not in _j.dumps(audit_records())


def test_put_config_allows_an_unprotected_edit(client, gated_config, audit_records):
    """An edit that leaves every protected key alone still saves, and still backs up."""
    from osprey.utils.config_writer import config_backup_path

    before = gated_config.read_text()
    updated = before.replace("project_name: original", "project_name: updated")

    response = client.put("/api/config", json={"content": updated})

    assert response.status_code == 200
    assert gated_config.read_text() == updated
    assert config_backup_path(gated_config).read_text() == before
    assert audit_records() == []


class TestConfigPanelTierGate:
    """``web.config_panel.enabled: false`` closes the settings editor's server surface."""

    @pytest.mark.usefixtures("gated_config")
    def test_get_config_is_refused_when_the_panel_is_disabled(self, client):
        """A read is gated too: the document carries the provider base_urls."""
        client.app.state.config_panel_enabled = False

        response = client.get("/api/config")

        assert response.status_code == 403
        assert "web.config_panel.enabled" in response.json()["detail"]

    def test_put_config_is_refused_when_the_panel_is_disabled(self, client, gated_config):
        """The write is refused before the file is read, so nothing changes."""
        before = gated_config.read_bytes()
        client.app.state.config_panel_enabled = False

        response = client.put("/api/config", json={"content": "project_name: updated\n"})

        assert response.status_code == 403
        assert "web.config_panel.enabled" in response.json()["detail"]
        assert gated_config.read_bytes() == before

    @pytest.mark.usefixtures("gated_config")
    def test_the_gate_runs_before_the_protected_set(self, client):
        """A disabled panel never gets as far as having a key to judge."""
        client.app.state.config_panel_enabled = False

        response = client.put(
            "/api/config",
            json={"content": "control_system:\n  writes_enabled: false\nproject_name: x\n"},
        )

        assert response.status_code == 403
        assert "agent_data.base_dir" not in response.json()["detail"]

    @pytest.mark.usefixtures("gated_config")
    def test_an_absent_flag_refuses_the_panel(self, client):
        """An app that never resolved the flag has made no tier decision, so it refuses."""
        del client.app.state.config_panel_enabled

        response = client.get("/api/config")

        assert response.status_code == 403
        assert "web.config_panel.enabled" in response.json()["detail"]

    def test_capabilities_reports_the_gate(self, client):
        """The frontend gets the flag it needs to drop the Settings entry."""
        client.app.state.config_panel_enabled = False

        payload = client.get("/api/capabilities").json()

        assert payload["config_panel_enabled"] is False

    def test_capabilities_reports_an_open_panel(self, client):
        """An enabled flag is reported as it stands."""
        payload = client.get("/api/capabilities").json()

        assert payload["config_panel_enabled"] is True

    def test_capabilities_reports_an_absent_flag_as_closed(self, client):
        """No flag on app.state reads as closed, matching the gate's own refusal."""
        del client.app.state.config_panel_enabled

        payload = client.get("/api/capabilities").json()

        assert payload["config_panel_enabled"] is False


# ---------------------------------------------------------------------------
# Native upload: the picture is stored with its rendition in the request
# ---------------------------------------------------------------------------


def _real_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def _local_only(service) -> dict:
    """Make ``service`` save uploads local-only and keep native rows in memory.

    Returns:
        The in-memory ``attachment_files`` rows, keyed by attachment id.
    """
    rows: dict[str, dict] = {}

    async def insert_native_attachment(entry_id, attachment_id, **kw):
        rendition = kw["rendition"]
        rows[attachment_id] = {
            "attachment_id": attachment_id,
            "entry_id": entry_id,
            "filename": kw["filename"],
            "mime_type": kw["mime_type"],
            "data": kw["data"],
            "size_bytes": len(kw["data"]),
            "copy_status": "copied",
            "skip_reason": kw["skip_reason"],
            "rendition_sha256": rendition.sha256 if rendition else None,
        }

    async def get_attachment_original(attachment_id):
        row = rows.get(attachment_id)
        if row is None or row["copy_status"] != "copied" or row["data"] is None:
            return None
        return {"filename": row["filename"], "mime_type": row["mime_type"], "data": row["data"]}

    service.create_entry = AsyncMock(side_effect=NotImplementedError("read-only"))
    service.repository.upsert_entry = AsyncMock()
    service.repository.store_attachment = AsyncMock()
    service.repository.get_entry = AsyncMock(
        return_value={
            "entry_id": "ariel-xyz",
            "source_system": "ARIEL Web",
            "timestamp": datetime.now(),
            "author": "Anonymous",
            "raw_text": "Test\n\nBody",
            "attachments": [],
            "metadata": {},
        }
    )
    service.repository.insert_native_attachment = AsyncMock(side_effect=insert_native_attachment)
    service.repository.get_attachment_original = AsyncMock(side_effect=get_attachment_original)
    return rows


def test_native_web_upload_picture_is_viewable_at_once(client, mock_ariel_service):
    """The upload request itself renders the picture; no sync runs in between."""
    from osprey.services.ariel_search.attachments.formats import is_viewable

    rows = _local_only(mock_ariel_service)
    png = _real_png()

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("beam.png", png, "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["attachment_count"] == 1
    (row,) = rows.values()
    assert row["data"] == png
    assert is_viewable(row), row
    mock_ariel_service.repository.store_attachment.assert_not_called()
    linked = mock_ariel_service.repository.upsert_entry.call_args_list[-1].args[0]
    assert linked["attachments"] == [
        {
            "url": f"/api/attachments/{row['attachment_id']}",
            "type": "image/png",
            "filename": "beam.png",
        }
    ]


def test_native_web_upload_render_unavailable_stores_without_rendition(
    client, mock_ariel_service, monkeypatch
):
    """An unavailable worker leaves a copied row the poll's render step finishes."""
    from osprey.services.ariel_search.attachments import prepare as prepare_module

    async def unavailable(*_args, **_kwargs):
        raise prepare_module.RenderUnavailable("no worker")

    monkeypatch.setattr(prepare_module, "prepare_picture", unavailable)
    rows = _local_only(mock_ariel_service)

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("beam.png", _real_png(), "image/png"))],
    )

    assert response.status_code == 200
    (row,) = rows.values()
    assert row["copy_status"] == "copied"
    assert row["skip_reason"] is None
    assert row["rendition_sha256"] is None
    assert row["mime_type"] == "image/png"


def test_native_web_upload_on_schema_without_copy_state_uses_b1_columns(
    client, mock_ariel_service, monkeypatch
):
    """A schema that predates copy state still accepts uploads, through the B1 insert."""
    from osprey.services.ariel_search.attachments import prepare as prepare_module

    async def never(*_args, **_kwargs):
        raise AssertionError("prepare_picture must not run on a schema without copy state")

    monkeypatch.setattr(prepare_module, "prepare_picture", never)
    rows = _local_only(mock_ariel_service)
    mock_ariel_service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(False, False))
    png = _real_png()

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("beam.png", png, "image/png"))],
    )

    assert response.status_code == 200
    assert response.json()["attachment_count"] == 1
    assert rows == {}
    mock_ariel_service.repository.store_attachment.assert_awaited_once()
    kwargs = mock_ariel_service.repository.store_attachment.call_args.kwargs
    assert set(kwargs) == {
        "entry_id",
        "attachment_id",
        "filename",
        "mime_type",
        "data",
        "size_bytes",
    }
    assert kwargs["data"] == png
    assert kwargs["mime_type"] == "image/png"


@pytest.mark.parametrize("mode", ["images", "none"])
def test_native_pdf_upload_keeps_data_whatever_copy_on_ingest(client, mock_ariel_service, mode):
    """A PDF is kept as a copied row with ``reserved_format`` and is served back."""
    mock_ariel_service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "attachments": {"copy_on_ingest": mode},
        }
    )
    rows = _local_only(mock_ariel_service)
    pdf = b"%PDF-1.4\n%fake pdf body\n%%EOF\n"

    response = client.post(
        "/api/entries/upload",
        data={"subject": "Test", "details": "Body"},
        files=[("files", ("report.pdf", pdf, "application/pdf"))],
    )

    assert response.status_code == 200
    (row,) = rows.values()
    assert row["data"] == pdf
    assert row["copy_status"] == "copied"
    assert row["skip_reason"] == "reserved_format"
    assert row["mime_type"] == "application/pdf"
    assert row["rendition_sha256"] is None

    served = client.get(f"/api/attachments/{row['attachment_id']}")
    assert served.status_code == 200
    assert served.content == pdf
    assert served.headers["content-type"] == "application/octet-stream"
    assert served.headers["content-disposition"] == "attachment; filename*=UTF-8''report.pdf"
    assert served.headers["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------------------
# Entry responses carry attachment summaries and a display url
# ---------------------------------------------------------------------------

_NATIVE_ID = "att-0123456789ab"
_PDF_URL = "https://logbook.invalid/doc.pdf"


def _to_response(entry, rows):
    return routes._entry_to_response(entry, attachment_rows=rows, model_id=None, file_source=False)


def test_attachment_response_fields_are_summary_keys_plus_the_web_fields():
    from osprey.interfaces.ariel.api.schemas import AttachmentResponse
    from osprey.services.ariel_search.attachments.summaries import SUMMARY_KEYS

    assert set(AttachmentResponse.model_fields) == set(SUMMARY_KEYS) | {
        "display_url",
        "skip_reason_text",
    }


def test_a_skipped_attachment_carries_the_text_of_its_skip_reason():
    from osprey.imaging.formats import skip_reason_text

    skipped = {"url": PNG_URL, "filename": "big.png", "type": "image/png"}
    shown = {"url": _PDF_URL, "filename": "doc.pdf"}
    entry = attachment_entry([skipped, shown])
    rows = [
        {
            **attachment_row("e-att", skipped, mime_type="image/png", viewable=False),
            "copy_status": "skipped",
            "skip_reason": "size_cap",
        },
        attachment_row("e-att", shown, mime_type="application/pdf", viewable=False),
    ]

    big, doc = _to_response(entry, rows).attachments

    assert big.skip_reason == "size_cap"
    assert big.skip_reason_text == skip_reason_text("size_cap")
    assert doc.skip_reason_text is None


def test_display_url_viewable_is_rendition():
    from osprey.services.ariel_search.attachments import attachment_id_for

    item = {"url": PNG_URL, "filename": "pic.png", "type": "image/png"}
    entry = attachment_entry([item])
    rows = [attachment_row("e-att", item, mime_type="image/png", viewable=True)]

    att = _to_response(entry, rows).attachments[0]

    att_id = attachment_id_for("e-att", item)
    assert att.viewable is True
    assert att.display_url == f"/attachments/{att_id}/rendition"
    assert att.url == PNG_URL


def test_display_url_copied_not_viewable_is_original():
    from osprey.services.ariel_search.attachments import attachment_id_for

    item = {"url": _PDF_URL, "filename": "doc.pdf"}
    entry = attachment_entry([item])
    rows = [attachment_row("e-att", item, mime_type="application/pdf", viewable=False)]

    att = _to_response(entry, rows).attachments[0]

    assert att.viewable is False
    assert att.copy_status == "copied"
    assert att.display_url == f"/attachments/{attachment_id_for('e-att', item)}"


def test_display_url_pending_falls_back_to_source_url():
    item = {"url": PNG_URL, "filename": "pic.png"}

    att = _to_response(attachment_entry([item]), []).attachments[0]

    assert att.copy_status == "pending"
    assert att.display_url == PNG_URL


def test_display_url_null_without_url():
    item = {"url": "relative/pic.png", "filename": "pic.png"}

    att = _to_response(attachment_entry([item]), []).attachments[0]

    assert att.url is None
    assert att.display_url is None


def test_display_url_unmigrated_native_item_is_original():
    item = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "pic.png", "type": "image/png"}

    att = _to_response(attachment_entry([item]), None).attachments[0]

    assert att.attachment_id is None
    assert att.display_url == f"/attachments/{_NATIVE_ID}"


def test_display_url_migrated_native_without_row_is_null():
    """With copy state a native item without a finished row has nowhere to show."""
    item = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "pic.png"}

    att = _to_response(attachment_entry([item]), []).attachments[0]

    assert att.display_url is None


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "file:///etc/passwd", "data:image/png;base64,AA"]
)
@pytest.mark.parametrize("rows", [None, []])
def test_unsafe_urls_are_nulled(url, rows):
    item = {"url": url, "filename": "x.png"}

    att = _to_response(attachment_entry([item]), rows).attachments[0]

    assert att.url is None
    assert att.display_url is None


def test_safe_url_rule():
    assert routes._safe_url("https://a.invalid/x") == "https://a.invalid/x"
    assert routes._safe_url("HTTP://a.invalid/x") == "HTTP://a.invalid/x"
    assert routes._safe_url("/api/attachments/att-0123456789ab") is not None
    assert routes._safe_url("/other/path") is None
    assert routes._safe_url("javascript:x") is None
    assert routes._safe_url(None) is None


def test_entry_response_carries_match_fields():
    item = {"url": PNG_URL, "filename": "pic.png"}
    entry = attachment_entry(
        [item], _matched_via=("text", "caption"), _matched_attachment_ids=("a",)
    )

    result = _to_response(entry, [])

    assert result.matched_via == ["text", "caption"]
    assert result.matched_attachment_ids == ["a"]


def test_entry_response_match_fields_default_empty():
    result = _to_response(attachment_entry([]), None)

    assert result.matched_via == []
    assert result.matched_attachment_ids == []


def test_entry_response_orders_matched_attachment_first():
    from osprey.services.ariel_search.attachments import attachment_id_for

    first = {"url": "https://logbook.invalid/a.png", "filename": "a.png"}
    second = {"url": "https://logbook.invalid/b.png", "filename": "b.png"}
    second_id = attachment_id_for("e-att", second)
    entry = attachment_entry([first, second], _matched_attachment_ids=(second_id,))
    rows = [
        attachment_row("e-att", first, mime_type="image/png", viewable=True),
        attachment_row("e-att", second, mime_type="image/png", viewable=True),
    ]

    atts = _to_response(entry, rows).attachments

    assert [a.filename for a in atts] == ["b.png", "a.png"]
    # display_url follows the item, not the position.
    assert atts[0].display_url == f"/attachments/{second_id}/rendition"


# -- routes: search, list, detail ------------------------------------------------


def _seven_attachment_entry():
    items = [
        {"url": f"https://logbook.invalid/p{i}.png", "filename": f"p{i}.png", "type": "image/png"}
        for i in range(7)
    ]
    items[0]["caption"] = "c" * 600
    return attachment_entry(items)


def _route_entries(client, service, route, entry):
    """Serve ``entry`` from the given route and return the response entries."""
    if route == "search":
        service.search.return_value.entries = [entry]
        service.search.return_value.diagnostics = []
        response = client.post("/api/search", json={"query": "q", "mode": "keyword"})
        assert response.status_code == 200, response.text
        return response.json()["entries"]
    if route == "list":
        service.repository.count_entries = AsyncMock(return_value=1)
        service.repository.search_by_time_range = AsyncMock(return_value=[entry])
        response = client.get("/api/entries")
        assert response.status_code == 200, response.text
        return response.json()["entries"]
    service.repository.get_entry = AsyncMock(return_value=entry)
    response = client.get(f"/api/entries/{entry['entry_id']}")
    assert response.status_code == 200, response.text
    return [response.json()]


_ROUTES = ["search", "list", "detail"]


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_return_every_attachment_and_whole_caption(client, mock_ariel_service, route):
    entries = _route_entries(client, mock_ariel_service, route, _seven_attachment_entry())

    atts = entries[0]["attachments"]
    assert len(atts) == 7
    captions = [a["caption"] for a in atts if a["caption"]]
    assert captions == ["c" * 600]
    assert "caption_truncated" not in atts[0]


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_read_attachment_rows_once(client, mock_ariel_service, route):
    _route_entries(client, mock_ariel_service, route, _seven_attachment_entry())

    mock_ariel_service.repository.get_attachment_rows.assert_awaited_once_with(["e-att"])


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_use_rows_for_display_url(client, mock_ariel_service, route):
    item = {"url": PNG_URL, "filename": "pic.png", "type": "image/png"}
    row = attachment_row("e-att", item, mime_type="image/png", viewable=True)
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value={"e-att": [row]})

    att = _route_entries(client, mock_ariel_service, route, attachment_entry([item]))[0][
        "attachments"
    ][0]

    assert att["display_url"] == f"/attachments/{row['attachment_id']}/rendition"
    assert set(att) == set(routes.AttachmentResponse.model_fields)


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_unmigrated_store_returns_fallback_summaries(client, mock_ariel_service, route):
    native = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "n.png", "type": "image/png"}
    remote = {"url": PNG_URL, "filename": "pic.png", "type": "image/png"}
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value=None)

    atts = _route_entries(client, mock_ariel_service, route, attachment_entry([native, remote]))[0][
        "attachments"
    ]

    assert [a["copy_status"] for a in atts] == ["pending", "pending"]
    assert all(a["viewable"] is False and a["attachment_id"] is None for a in atts)
    assert atts[0]["display_url"] == f"/attachments/{_NATIVE_ID}"
    assert atts[1]["display_url"] == PNG_URL


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_failing_reader_equals_unmigrated(client, mock_ariel_service, route):
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    native = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "n.png", "type": "image/png"}
    entry = attachment_entry([native])
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value=None)
    unmigrated = _route_entries(client, mock_ariel_service, route, dict(entry))

    mock_ariel_service.repository.get_attachment_rows = AsyncMock(
        side_effect=DatabaseQueryError("boom")
    )
    with patch(
        "osprey.services.ariel_search.database.repository.warn_attachment_schema_gap_once"
    ) as warn:
        failing = _route_entries(client, mock_ariel_service, route, dict(entry))

    warn.assert_called_once_with()
    assert failing[0]["attachments"] == unmigrated[0]["attachments"]


def test_failing_reader_logs_schema_warning_once_per_process(
    client, mock_ariel_service, monkeypatch, caplog
):
    import logging

    from osprey.services.ariel_search.database import repository as repository_module
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    monkeypatch.setattr(repository_module, "_attachment_schema_gap_warned", False)
    caplog.set_level(logging.WARNING, logger="ariel")
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(
        side_effect=DatabaseQueryError("boom")
    )

    for _ in range(2):
        _route_entries(client, mock_ariel_service, "detail", _seven_attachment_entry())

    gap = [
        r
        for r in caplog.records
        if r.getMessage() == repository_module.ATTACHMENT_SCHEMA_GAP_WARNING
    ]
    assert len(gap) == 1


def test_routes_skip_reader_without_entries(client, mock_ariel_service):
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=0)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])

    assert client.get("/api/entries").status_code == 200
    assert client.post("/api/search", json={"query": "q"}).status_code == 200

    mock_ariel_service.repository.get_attachment_rows.assert_not_awaited()


def test_unmigrated_native_display_url_is_served_by_original_route(client, mock_ariel_service):
    """The native picture stays reachable between upgrade and migrate."""
    native = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "n.png", "type": "image/png"}
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value=None)
    mock_ariel_service.repository.get_attachment_original = AsyncMock(
        return_value={"filename": "n.png", "mime_type": "image/png", "data": _real_png()}
    )

    att = _route_entries(client, mock_ariel_service, "detail", attachment_entry([native]))[0][
        "attachments"
    ][0]
    served = client.get("/api" + att["display_url"])

    assert served.status_code == 200
    mock_ariel_service.repository.get_attachment_original.assert_awaited_once_with(_NATIVE_ID)


def _fused_entry(entry_id, via):
    return attachment_entry([], entry_id=entry_id, _score=0.5, _matched_via=list(via))


def test_search_hybrid_page_and_sources_stop_at_max_results(client, mock_ariel_service):
    """A lane admitting image-only hits beyond max_results: the page holds max_results.

    The route slices the fused entries itself and names exactly the shown
    entries in ``sources``; ``total_results`` is the result's total (C-PAGE),
    not the page length.
    """
    _enable_hybrid(mock_ariel_service)
    entries = [_fused_entry(f"T{i}", ["text"]) for i in range(1, 11)] + [
        _fused_entry(f"I{i}", ["image"]) for i in range(1, 5)
    ]
    result = mock_ariel_service.search.return_value
    result.entries = tuple(entries)
    result.sources = tuple(e["entry_id"] for e in entries)
    result.search_modes_used = ("hybrid",)
    result.diagnostics = ()
    result.expanded_terms = ()
    result.total, result.total_exact, result.has_more = 14, False, True

    response = client.post("/api/search", json={"query": "q", "mode": "hybrid", "max_results": 10})

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == len(data["entries"]) == 10
    assert data["total_results"] == data["total"] == 14
    assert data["has_more"] is True
    assert data["sources"] == [e["entry_id"] for e in data["entries"]]
    assert data["sources"] == [f"T{i}" for i in range(1, 11)]


def test_search_keeps_the_service_sources_without_matched_via(client, mock_ariel_service):
    """Without a fused result the route reports the service's sources unchanged."""
    entries = [attachment_entry([], entry_id=f"e{i}", _score=0.5) for i in range(2)]
    result = mock_ariel_service.search.return_value
    result.entries = tuple(entries)
    result.sources = ("e0", "e1", "cited")
    result.diagnostics = ()
    result.expanded_terms = ()

    response = client.post("/api/search", json={"query": "q", "mode": "keyword", "max_results": 10})

    assert response.status_code == 200
    assert response.json()["sources"] == ["e0", "e1", "cited"]


# ---------------------------------------------------------------------------
# Response fields: subject, entry_url, enhancement_status (C-SUBJ, C-URL, C-ENH)
# and the row projection (C-ROW)
# ---------------------------------------------------------------------------

_FIELD_MODULES = ["image_caption", "text_embedding"]


def _field_entry(**overrides):
    """One entry seen by every route: a stored subject that differs from its first
    line, a local-only save, a caption written under an older model, a picture."""
    entry = {
        "entry_id": "E-1",
        "source_system": "ARIEL Web",
        "timestamp": datetime(2025, 10, 6, 9, 30),
        "author": "J. Smith",
        "raw_text": "First line of the text\n\nThe body of the entry.",
        "subject": "Stored subject",
        "attachments": [{"url": PNG_URL, "type": "image/png", "filename": "p.png"}],
        "metadata": {"sync_status": "local_only", "logbook": "Ops", "tags": ["rf", "Vacuum"]},
        "created_at": datetime(2025, 10, 6, 9, 30),
        "updated_at": datetime(2025, 10, 6, 9, 30),
        "summary": None,
        "keywords": ["rf"],
        "enhancement_status": {
            "image_caption": {"status": "complete", "marker": "cap-v1"},
            "text_embedding": {"status": "complete"},
        },
        "has_images": True,
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def field_service(mock_ariel_service):
    """The mock service with two enabled enhancement modules, the caption model
    now ``cap-v2``, and a registry that lists them."""
    mock_ariel_service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "enhancement_modules": {
                "image_caption": {"enabled": True, "model": {"model_id": "cap-v2"}},
                "text_embedding": {"enabled": True},
            },
        }
    )
    registry = _build_mock_registry()
    registry.list_ariel_enhancement_modules.return_value = list(_FIELD_MODULES)
    with patch("osprey.registry.get_registry", return_value=registry):
        yield mock_ariel_service


def _fetch_entry(client, service, route, entry):
    """The one entry as the given route returns it, as JSON."""
    if route == "search":
        result = service.search.return_value
        result.entries = [{**entry, "_score": 1.0, "_highlights": []}]
        result.sources = [entry["entry_id"]]
        result.diagnostics = []
        response = client.post("/api/search", json={"query": "rf", "mode": "keyword"})
        assert response.status_code == 200, response.text
        return response.json()["entries"][0]
    if route == "list":
        service.repository.count_entries = AsyncMock(return_value=1)
        service.repository.search_by_time_range = AsyncMock(return_value=[entry])
        response = client.get("/api/entries")
        assert response.status_code == 200, response.text
        return response.json()["entries"][0]
    service.repository.get_entry = AsyncMock(return_value=entry)
    response = client.get(f"/api/entries/{entry['entry_id']}")
    assert response.status_code == 200, response.text
    return response.json()


_ROUTES = ["search", "list", "detail"]


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param("subject", "Stored subject", id="field-subject-stored"),
        pytest.param("entry_url", None, id="field-entry_url-local-only-null"),
        pytest.param(
            "enhancement_status",
            {
                "image_caption": {"status": "complete", "marker": "cap-v1", "state": "stale"},
                "text_embedding": {"status": "complete", "state": "complete"},
            },
            id="field-enhancement-stale-computed",
        ),
    ],
)
def test_response_field_table(client, field_service, route, field, expected):
    """Search, list and detail carry the same new fields for one entry: the stored
    subject (not the text's first line), a null URL for a local-only save, and a
    caption written under the old model reported stale."""
    data = _fetch_entry(client, field_service, route, _field_entry())

    assert data[field] == expected


@pytest.mark.parametrize("route", _ROUTES)
def test_response_field_entry_url_rendered_for_a_logbook_entry(
    client, field_service, route, monkeypatch
):
    """An entry that is in the facility logbook gets the template URL on every route."""
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: (
            "https://logbook.example/entry/{entry_id}"
            if key == "ariel.entry_url_template"
            else default
        ),
    )
    entry = _field_entry(source_system="Example eLog", metadata={})

    data = _fetch_entry(client, field_service, route, entry)

    assert data["entry_url"] == "https://logbook.example/entry/E-1"


@pytest.mark.parametrize("route", _ROUTES)
def test_response_field_subject_derived_when_none_stored(client, field_service, route):
    """A store without the subject column still gets one subject per entry."""
    entry = _field_entry()
    del entry["subject"]

    data = _fetch_entry(client, field_service, route, entry)

    assert data["subject"] == "First line of the text"


def test_response_field_enhancement_without_pictures_is_na(client, field_service):
    """An image module on an entry with no viewable picture is n/a, not stale."""
    data = _fetch_entry(client, field_service, "detail", _field_entry(has_images=False))

    assert data["enhancement_status"]["image_caption"]["state"] == "n/a"


def test_response_field_enhancement_context_resolved_once_per_request(client, field_service):
    """Markers and enabled modules are resolved once for a page, not per entry."""
    entries = [_field_entry(entry_id=f"E-{i}") for i in range(5)]
    field_service.repository.count_entries = AsyncMock(return_value=5)
    field_service.repository.search_by_time_range = AsyncMock(return_value=entries)

    with patch(
        "osprey.services.ariel_search.enhancement.status.current_markers",
        wraps=__import__(
            "osprey.services.ariel_search.enhancement.status", fromlist=["current_markers"]
        ).current_markers,
    ) as markers:
        response = client.get("/api/entries")

    assert response.status_code == 200
    assert markers.call_count == 1


def _row_dimensions():
    from osprey.services.ariel_search.filters import BUILTIN_DIMENSIONS, FilterDimension

    return [
        *BUILTIN_DIMENSIONS,
        FilterDimension("logbook", "Logbook", "declared", "metadata->>'logbook'", multi=False),
        FilterDimension(
            "enhancement", "Enhancement", "enhancement", "", multi=False, values=("x",)
        ),
    ]


def _rows(service, entries, **kwargs):
    import asyncio

    with patch(
        "osprey.services.ariel_search.filters.filter_dimensions",
        return_value=_row_dimensions(),
    ):
        return asyncio.run(routes._row_responses(service, entries, **kwargs))


def test_row_view_carries_the_c_row_fields(field_service):
    """A row holds what the list pane renders, computed from the row data."""
    entry = _field_entry(_matched_via=["text"])

    (row,) = _rows(field_service, [entry])

    assert row.model_dump() == {
        "entry_id": "E-1",
        "timestamp": row.timestamp,
        "author": "J. Smith",
        "source_system": "ARIEL Web",
        "subject": "Stored subject",
        "snippet": "First line of the text The body of the entry.",
        "facets": {
            "author": "J. Smith",
            "source_system": "ARIEL Web",
            "tags": ["rf", "Vacuum"],
            "logbook": "Ops",
        },
        "attachment_count": 1,
        "has_images": True,
        "enhancements": {"image_caption": "stale", "text_embedding": "complete"},
        "entry_url": None,
        "score": None,
        "highlights": [],
        "keywords": ["rf"],
        "matched_via": ["text"],
    }
    assert row.timestamp and row.timestamp.startswith("2025-10-06T09:30")


def test_row_view_carries_no_raw_text(field_service):
    """The row view never ships the entry's full text or attachment summaries."""
    (row,) = _rows(field_service, [_field_entry()])

    dumped = row.model_dump()
    assert "raw_text" not in dumped
    assert "attachments" not in dumped
    assert "The body of the entry." not in dumped["subject"]


def test_row_view_makes_at_most_one_batched_read(field_service):
    """A page of rows reads no attachment rows and nothing per entry."""
    entries = [_field_entry(entry_id=f"E-{i}") for i in range(10)]

    rows = _rows(field_service, entries)

    assert len(rows) == 10
    field_service.repository.get_attachment_rows.assert_not_awaited()
    assert len(field_service.repository.mock_calls) <= 1


def test_row_view_snippet_drops_a_subject_first_line(field_service):
    """When the text opens with the subject, the preview starts after it."""
    entry = _field_entry(subject="First line of the text")

    (row,) = _rows(field_service, [entry])

    assert row.snippet == "The body of the entry."


def test_row_view_snippet_is_capped(field_service):
    """The preview is at most 300 characters of the details."""
    entry = _field_entry(raw_text="Title\n" + "x" * 1000, subject="Title")

    (row,) = _rows(field_service, [entry])

    assert row.snippet == "x" * routes.ROW_SNIPPET_CHARS


def test_row_view_snippet_prefers_highlights_and_carries_scores(field_service):
    """Search rows preview the match and carry score and highlights."""
    entry = _field_entry(_score=0.5, _highlights=["the <mark>rf</mark> trip"])

    (row,) = _rows(field_service, [entry], with_scores=True)

    assert row.snippet == "the <mark>rf</mark> trip"
    assert row.highlights == ["the <mark>rf</mark> trip"]
    assert row.score == 0.5


def test_row_view_without_attachments_or_pictures(field_service):
    """No attachments counts zero; a missing has_images is false; image modules n/a."""
    entry = _field_entry(attachments=None)
    del entry["has_images"]

    (row,) = _rows(field_service, [entry])

    assert row.attachment_count == 0
    assert row.has_images is False
    assert row.enhancements["image_caption"] == "n/a"


@pytest.mark.parametrize("route", _ROUTES)
def test_row_view_subject_matches_every_route(client, field_service, route):
    """List rows, search, list and detail carry the same stored subject."""
    entry = _field_entry()
    (row,) = _rows(field_service, [entry])

    data = _fetch_entry(client, field_service, route, entry)

    assert data["subject"] == row.subject == "Stored subject"


# --- Filters and paging on /search and /entries (C-FILTER, C-PAGE) ---------------


def _post_search(client, **body):
    return client.post("/api/search", json={"query": "rf", "mode": "keyword", **body})


def _get_entries(client, **params):
    return client.get("/api/entries", params=params)


# One request per route and refusal. /search pages by offset rather than by
# cursor, so its bad paging position is a negative offset, and its page size is
# max_results.
_REFUSALS = {
    ("search", "unknown dimension"): {"filters": {"nope": ["x"]}},
    ("search", "malformed value"): {"filters": {"author": "J. Smith"}},
    ("search", "bad cursor"): {"offset": -1},
    ("search", "page_size 101"): {"max_results": 101},
    ("list", "unknown dimension"): {"filters": '{"nope": ["x"]}'},
    ("list", "malformed value"): {"filters": '{"author": "J. Smith"}'},
    ("list", "bad cursor"): {"cursor": "not-a-cursor"},
    ("list", "page_size 101"): {"page_size": 101},
}


@pytest.mark.parametrize(("route", "case"), list(_REFUSALS))
def test_filter_and_paging_refusals_are_422(client, mock_ariel_service, route, case):
    """A bad filter, paging position or page size is the caller's error, never a 500,
    and nothing is read."""
    repo = mock_ariel_service.repository
    repo.count_entries = AsyncMock(return_value=0)
    repo.search_by_time_range = AsyncMock(return_value=[])
    params = _REFUSALS[(route, case)]

    if route == "search":
        response = _post_search(client, **params)
    else:
        response = _get_entries(client, **params)

    assert response.status_code == 422, response.text
    mock_ariel_service.search.assert_not_called()
    repo.search_by_time_range.assert_not_called()
    repo.count_entries.assert_not_called()


_FILTER_REFUSALS = [
    ("unknown dimension", "nope"),
    ("malformed value", "author"),
]


@pytest.mark.parametrize("route", ["search", "list"])
@pytest.mark.parametrize(("case", "field"), _FILTER_REFUSALS)
def test_filter_refusal_names_the_field(client, route, case, field):
    """A refused filter answers ``{"detail": {"field", "reason"}}``."""
    params = _REFUSALS[(route, case)]

    response = (
        _post_search(client, **params) if route == "search" else _get_entries(client, **params)
    )

    detail = response.json()["detail"]
    assert detail["field"] == field
    assert detail["reason"]


def test_list_bad_cursor_names_the_cursor_field(client):
    response = _get_entries(client, cursor="not-a-cursor")

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "cursor"


@pytest.mark.parametrize("page_size", [0, -1])
def test_list_page_size_below_one_is_422(client, page_size):
    assert _get_entries(client, page_size=page_size).status_code == 422


def test_list_page_and_cursor_together_is_422(client, mock_ariel_service):
    """``page`` and ``cursor`` are two paging schemes; asking for both is refused."""
    from osprey.services.ariel_search.filters import encode_cursor

    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=[])
    cursor = encode_cursor(datetime(2025, 10, 6, 9, 30, tzinfo=UTC), "E-1")

    response = _get_entries(client, page=2, cursor=cursor)

    assert response.status_code == 422
    mock_ariel_service.repository.search_by_time_range.assert_not_called()


def test_list_passes_filter_and_cursor_to_the_repository(client, mock_ariel_service):
    """The parsed filter reaches both the count and the page read, with the cursor,
    beside the unchanged legacy kwargs."""
    from osprey.services.ariel_search.filters import EntryFilter, encode_cursor

    repo = mock_ariel_service.repository
    repo.count_entries = AsyncMock(return_value=3)
    repo.search_by_time_range = AsyncMock(return_value=[])
    cursor = encode_cursor(datetime(2025, 10, 6, 9, 30, tzinfo=UTC), "E-1")

    response = _get_entries(
        client,
        filters='{"author": ["J. Smith"], "tags": ["rf"]}',
        cursor=cursor,
        page_size=10,
        source_system="ERF",
    )

    assert response.status_code == 200, response.text
    kwargs = repo.search_by_time_range.call_args.kwargs
    flt = kwargs["entry_filter"]
    assert isinstance(flt, EntryFilter)
    assert flt.values("author") == ("J. Smith",)
    assert flt.values("tags") == ("rf",)
    assert kwargs["after_cursor"] == cursor
    assert kwargs["limit"] == 11  # one surplus row decides next_cursor
    assert kwargs["offset"] == 0
    assert kwargs["source_system"] == "ERF"
    ckwargs = repo.count_entries.call_args.kwargs
    assert ckwargs["entry_filter"] == flt
    assert ckwargs["source_system"] == "ERF"
    assert response.json()["total"] == 3


def test_list_without_filters_passes_no_entry_filter(client, mock_ariel_service):
    repo = mock_ariel_service.repository
    repo.count_entries = AsyncMock(return_value=0)
    repo.search_by_time_range = AsyncMock(return_value=[])

    assert _get_entries(client).status_code == 200

    assert repo.search_by_time_range.call_args.kwargs["entry_filter"] is None
    assert repo.search_by_time_range.call_args.kwargs["after_cursor"] is None
    assert repo.count_entries.call_args.kwargs["entry_filter"] is None


def _list_entries(n):
    return [
        _field_entry(entry_id=f"E-{i}", timestamp=datetime(2025, 10, 6, 9, 30 - i, tzinfo=UTC))
        for i in range(n)
    ]


def test_list_next_cursor_points_after_the_last_row_shown(client, mock_ariel_service):
    """A surplus row means more entries: the cursor continues after the page's last row."""
    from osprey.services.ariel_search.filters import encode_cursor

    rows = _list_entries(3)
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=5)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=rows)

    data = _get_entries(client, page_size=2).json()

    assert [e["entry_id"] for e in data["entries"]] == ["E-0", "E-1"]
    assert data["next_cursor"] == encode_cursor(rows[1]["timestamp"], "E-1")
    assert data["page_size"] == 2
    assert data["total"] == 5


def test_list_last_page_has_no_next_cursor(client, mock_ariel_service):
    mock_ariel_service.repository.count_entries = AsyncMock(return_value=2)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(return_value=_list_entries(2))

    data = _get_entries(client, page_size=2).json()

    assert len(data["entries"]) == 2
    assert data["next_cursor"] is None


def test_list_legacy_page_path_is_unchanged(client, mock_ariel_service):
    """An explicit ``page`` keeps offset paging; it reads no cursor and returns none."""
    repo = mock_ariel_service.repository
    repo.count_entries = AsyncMock(return_value=100)
    repo.search_by_time_range = AsyncMock(return_value=_list_entries(20))

    data = _get_entries(client, page=2, page_size=20).json()

    kwargs = repo.search_by_time_range.call_args.kwargs
    assert (kwargs["limit"], kwargs["offset"], kwargs["after_cursor"]) == (20, 20, None)
    assert data["page"] == 2
    assert data["total_pages"] == 5
    assert data["next_cursor"] is None
    assert len(data["entries"]) == 20


def test_list_repository_cursor_refusal_is_422(client, mock_ariel_service):
    """A cursor the repository refuses is still the caller's error."""
    from osprey.services.ariel_search.filters import FilterValidationError, encode_cursor

    mock_ariel_service.repository.count_entries = AsyncMock(return_value=0)
    mock_ariel_service.repository.search_by_time_range = AsyncMock(
        side_effect=FilterValidationError("cursor", "not a valid cursor")
    )
    cursor = encode_cursor(datetime(2025, 10, 6, 9, 30, tzinfo=UTC), "E-1")

    response = _get_entries(client, cursor=cursor)

    assert response.status_code == 422
    assert response.json()["detail"] == {"field": "cursor", "reason": "not a valid cursor"}


@pytest.mark.parametrize(
    "filters", [{"author": ["J. Smith"]}, '{"author": ["J. Smith"]}'], ids=["object", "text"]
)
def test_search_passes_filter_and_offset_to_the_service(client, mock_ariel_service, filters):
    """The body filter (an object or its JSON text) and offset reach the service;
    the legacy author field keeps its advanced_params path."""
    from osprey.services.ariel_search.filters import EntryFilter

    response = _post_search(client, filters=filters, offset=20, author="alice")

    assert response.status_code == 200, response.text
    kwargs = mock_ariel_service.search.call_args.kwargs
    assert isinstance(kwargs["entry_filter"], EntryFilter)
    assert kwargs["entry_filter"].values("author") == ("J. Smith",)
    assert kwargs["offset"] == 20
    assert kwargs["advanced_params"]["author"] == "alice"


def test_search_without_filters_passes_no_entry_filter(client, mock_ariel_service):
    assert _post_search(client).status_code == 200

    kwargs = mock_ariel_service.search.call_args.kwargs
    assert kwargs["entry_filter"] is None
    assert kwargs["offset"] == 0


@pytest.mark.parametrize(
    ("total", "exact", "has_more"), [(57, True, True), (1000, False, True), (3, True, False)]
)
def test_search_reports_the_module_totals(client, mock_ariel_service, total, exact, has_more):
    """total, total_exact and has_more come from the result; total_results equals total."""
    result = mock_ariel_service.search.return_value
    result.entries = [_field_entry()]
    result.diagnostics = []
    result.total, result.total_exact, result.has_more = total, exact, has_more

    data = _post_search(client).json()

    assert data["total"] == total
    assert data["total_results"] == total
    assert data["total_exact"] is exact
    assert data["has_more"] is has_more


def test_search_without_module_totals_falls_back_to_the_page(client, mock_ariel_service):
    """A result with no total reports what was seen, inexact."""
    result = mock_ariel_service.search.return_value
    result.entries = [_field_entry()]
    result.diagnostics = []
    result.total, result.total_exact, result.has_more = None, False, None

    data = _post_search(client, offset=10).json()

    assert data["total"] == 11
    assert data["total_exact"] is False
    assert data["has_more"] is False


def test_search_service_filter_refusal_is_422(client, mock_ariel_service):
    from osprey.services.ariel_search.filters import FilterValidationError

    mock_ariel_service.search.side_effect = FilterValidationError("time", "bad bound")

    response = _post_search(client)

    assert response.status_code == 422
    assert response.json()["detail"] == {"field": "time", "reason": "bad bound"}


_ROW_KEYS = {"snippet", "facets", "attachment_count", "has_images", "enhancements"}


@pytest.mark.parametrize("route", ["search", "list"])
def test_view_row_returns_rows(client, field_service, route):
    """``view=row`` answers list rows (C-ROW) on both routes; the default is full entries."""
    entry = _field_entry()
    result = field_service.search.return_value
    result.entries = [{**entry, "_score": 1.0, "_highlights": ["the <mark>rf</mark>"]}]
    result.diagnostics = []
    field_service.repository.count_entries = AsyncMock(return_value=1)
    field_service.repository.search_by_time_range = AsyncMock(return_value=[entry])

    def fetch(**extra):
        if route == "search":
            return _post_search(client, **extra)
        return _get_entries(client, **extra)

    row = fetch(view="row").json()["entries"][0]
    full = fetch().json()["entries"][0]

    assert _ROW_KEYS <= row.keys()
    assert "raw_text" not in row
    assert row["subject"] == "Stored subject"
    assert "raw_text" in full
    assert not (_ROW_KEYS & full.keys())
    if route == "search":
        assert row["snippet"] == "the <mark>rf</mark>"
        assert row["score"] == 1.0
    field_service.repository.get_attachment_rows.assert_awaited_once()


@pytest.mark.parametrize("route", ["search", "list"])
def test_unknown_view_is_422(client, route):
    response = (
        _post_search(client, view="tile")
        if route == "search"
        else _get_entries(client, view="tile")
    )

    assert response.status_code == 422


# The facets route (C-FACETS).

_FACETS_PAYLOAD = {
    "dimensions": [
        {
            "name": "author",
            "label": "Author",
            "kind": "author",
            "values": [{"value": "J. Smith", "count": 4}, {"value": "alice", "count": 1}],
        },
        {"name": "tags", "label": "Tags", "kind": "tags", "values": []},
    ],
    "histogram": [{"day": "2025-10-06", "count": 5}],
    "exact": False,
    "tier": "partial",
}


@pytest.fixture
def facets_service(field_service):
    """The field service, whose config enables two enhancement modules, with a
    stubbed ``facets``."""
    field_service.facets = AsyncMock(return_value=copy.deepcopy(_FACETS_PAYLOAD))
    return field_service


def _get_facets(client, **params):
    return client.get("/api/facets", params=params)


def test_facets_returns_the_service_payload(client, facets_service):
    """Shape: the service's dimensions, histogram, exact and tier, plus available."""
    response = _get_facets(client, q="rf", mode="keyword")

    assert response.status_code == 200, response.text
    assert response.json() == {"available": True, **_FACETS_PAYLOAD}
    kwargs = facets_service.facets.call_args.kwargs
    assert kwargs["query"] == "rf"
    assert kwargs["mode"] == "keyword"
    assert kwargs["entry_filter"] is None
    assert kwargs["advanced_params"] == {}


def test_facets_degraded_without_a_service():
    """No service (database unavailable): 200 with available false and no counts."""
    app = routes_app(None)

    response = TestClient(app).get("/api/facets", params={"q": "rf"})

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["available"] is False
    assert data["dimensions"] == []
    assert data["histogram"] == []


def test_facets_degraded_when_the_database_drops(client, facets_service):
    from osprey.services.ariel_search.exceptions import DatabaseConnectionError

    facets_service.facets = AsyncMock(side_effect=DatabaseConnectionError("gone"))

    data = _get_facets(client).json()

    assert data["available"] is False
    assert data["dimensions"] == []


def test_facets_passes_the_whole_filter_for_self_exclusion(client, facets_service):
    """The route hands the service the full filter; the service counts each
    dimension under the filter minus its own selection, so a selected author
    still lists its siblings."""
    from osprey.services.ariel_search.filters import EntryFilter

    response = _get_facets(
        client,
        q="",
        filters='{"author": ["J. Smith"], "tags": ["rf"]}',
        advanced_params='{"fuzzy": true}',
    )

    assert response.status_code == 200, response.text
    kwargs = facets_service.facets.call_args.kwargs
    flt = kwargs["entry_filter"]
    assert isinstance(flt, EntryFilter)
    assert flt.values("author") == ("J. Smith",)
    assert flt.values("tags") == ("rf",)
    assert kwargs["advanced_params"] == {"fuzzy": True}
    author = response.json()["dimensions"][0]
    assert [v["value"] for v in author["values"]] == ["J. Smith", "alice"]


def test_facets_enhancement_filter_reaches_the_service(client, facets_service):
    response = _get_facets(client, filters='{"enhancement": ["image_caption:failed"]}')

    assert response.status_code == 200, response.text
    flt = facets_service.facets.call_args.kwargs["entry_filter"]
    assert flt.values("enhancement") == ("image_caption:failed",)


@pytest.mark.parametrize(
    ("params", "status", "field"),
    [
        pytest.param({"mode": "nonexistent"}, 400, None, id="unknown-mode"),
        pytest.param({"mode": ""}, 400, None, id="blank-mode"),
        pytest.param({"filters": '{"bogus": ["x"]}'}, 422, "bogus", id="unknown-dimension"),
        pytest.param({"filters": '{"author": "J. Smith"}'}, 422, "author", id="malformed-value"),
        pytest.param({"filters": "{not json"}, 422, "filter", id="filter-not-json"),
        pytest.param(
            {"filters": '{"enhancement": ["image_caption:complete"]}'},
            422,
            "enhancement",
            id="enhancement-state-not-failed-or-stale",
        ),
        pytest.param(
            {"filters": '{"enhancement": ["nope:failed"]}'},
            422,
            "enhancement",
            id="enhancement-unknown-module",
        ),
        pytest.param(
            {"advanced_params": "{not json"}, 422, "advanced_params", id="advanced-not-json"
        ),
        pytest.param(
            {"advanced_params": "[1, 2]"}, 422, "advanced_params", id="advanced-not-object"
        ),
    ],
)
def test_facets_refusals(client, facets_service, params, status, field):
    response = _get_facets(client, q="rf", **params)

    assert response.status_code == status, response.text
    if field is not None:
        assert response.json()["detail"]["field"] == field
    facets_service.facets.assert_not_called()


def test_facets_hybrid_overrides_are_validated_as_search_does(client, facets_service):
    _enable_hybrid(facets_service)

    response = _get_facets(client, q="rf", mode="hybrid", advanced_params='{"rerank": "false"}')

    assert response.status_code == 400
    facets_service.facets.assert_not_called()


def test_facets_service_filter_refusal_is_422(client, facets_service):
    from osprey.services.ariel_search.filters import FilterValidationError

    facets_service.facets = AsyncMock(
        side_effect=FilterValidationError("time", "start is after end")
    )

    response = _get_facets(client)

    assert response.status_code == 422
    assert response.json()["detail"] == {"field": "time", "reason": "start is after end"}


def test_facets_other_failures_are_500_not_422(client, facets_service):
    facets_service.facets = AsyncMock(side_effect=ValueError("boom"))

    assert _get_facets(client).status_code == 500


def test_search_route_filters_a_third_party_mode_that_ignores_the_filter():
    """A module that never sees the filter still lists only matching entries
    through /api/search: the real service post-filters what the module returns."""
    from tests.services.ariel_search.test_service_filter import (
        _descriptor,
        _Recorder,
        _registry,
        _service,
    )

    service = _service()
    service.repository.schema_facts = AsyncMock(return_value=SchemaFacts(True, True))
    service.repository.get_attachment_rows = AsyncMock(return_value={})
    rows = [
        (_field_entry(entry_id="E1", author="J. Smith"), 0.9),
        (_field_entry(entry_id="E2", author="J. Smithson"), 0.8),
        (_field_entry(entry_id="E3", author="Bob"), 0.7),
    ]
    recorder = _Recorder(rows)
    with patch("osprey.registry.get_registry", return_value=_registry(_descriptor(recorder))):
        response = TestClient(routes_app(service)).post(
            "/api/search",
            json={"query": "beam", "mode": "third", "filters": {"author": ["j. smith"]}},
        )

    assert response.status_code == 200, response.text
    assert [e["entry_id"] for e in response.json()["entries"]] == ["E1"]
    assert "entry_filter" not in recorder.kwargs


# ---------------------------------------------------------------------------
# C-FOLLOWS-UP and C-CREATE-RESULT on both create routes
# ---------------------------------------------------------------------------

_CREATE_ROUTES = ["json", "upload"]


def _create_result(*, attachment_count=0, attachments_published=False):
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    return FacilityEntryCreateResult(
        entry_id="e-2",
        source_system="Generic JSON",
        sync_status=SyncStatus.PENDING_SYNC,
        message="Entry e-2 created in Generic JSON",
        attachment_count=attachment_count,
        attachments_published=attachments_published,
    )


def _post_create(client, route, metadata, *, files=None):
    """POST one entry through the named create route with the given metadata."""
    import json as json_module

    if route == "json":
        return client.post(
            "/api/entries",
            json={"subject": "Follow-up", "details": "Body", "metadata": metadata},
        )
    return client.post(
        "/api/entries/upload",
        data={
            "subject": "Follow-up",
            "details": "Body",
            "metadata": json_module.dumps(metadata),
        },
        files=files or [],
    )


def _hold_entries(mock_ariel_service, *ids):
    async def get_entry(entry_id):
        return {"entry_id": entry_id} if entry_id in ids else None

    mock_ariel_service.repository.get_entry = AsyncMock(side_effect=get_entry)


@pytest.mark.parametrize("route", _CREATE_ROUTES)
def test_create_carries_follows_up_into_both_metadata_dicts(client, mock_ariel_service, route):
    """A known parent id reaches the adapter request and ARIEL's local copy."""
    _hold_entries(mock_ariel_service, "e-1")
    mock_ariel_service.create_entry = AsyncMock(return_value=_create_result())

    response = _post_create(client, route, {"follows_up": "e-1"})

    assert response.status_code == 200
    call = mock_ariel_service.create_entry.await_args
    assert call.args[0].metadata["follows_up"] == "e-1"
    assert call.kwargs["local_metadata"]["follows_up"] == "e-1"
    mock_ariel_service.repository.get_entry.assert_awaited_once_with("e-1")


@pytest.mark.parametrize("route", _CREATE_ROUTES)
def test_create_local_only_fallback_keeps_follows_up(client, mock_ariel_service, route):
    """A read-only adapter's local save still stores the follow-up."""
    _hold_entries(mock_ariel_service, "e-1")
    mock_ariel_service.create_entry = AsyncMock(side_effect=NotImplementedError("read-only"))
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    response = _post_create(client, route, {"follows_up": "e-1"})

    assert response.status_code == 200
    stored = mock_ariel_service.repository.upsert_entry.await_args.args[0]
    assert stored["metadata"]["follows_up"] == "e-1"


@pytest.mark.parametrize("route", _CREATE_ROUTES)
@pytest.mark.parametrize("metadata", [{}, {"follows_up": None}, {"follows_up": ""}])
def test_create_without_follows_up_leaves_the_key_out(client, mock_ariel_service, route, metadata):
    """A missing key, null and "" mean no follow-up: no lookup, no key in either dict."""
    mock_ariel_service.repository.get_entry = AsyncMock(return_value=None)
    mock_ariel_service.create_entry = AsyncMock(return_value=_create_result())

    response = _post_create(client, route, metadata)

    assert response.status_code == 200
    call = mock_ariel_service.create_entry.await_args
    assert "follows_up" not in call.args[0].metadata
    assert "follows_up" not in call.kwargs["local_metadata"]
    mock_ariel_service.repository.get_entry.assert_not_called()


@pytest.mark.parametrize("route", _CREATE_ROUTES)
@pytest.mark.parametrize("value", ["e-404", 7, False, ["e-1"], {"id": "e-1"}])
def test_create_refuses_a_bad_follows_up_before_any_write(client, mock_ariel_service, route, value):
    """An unknown id or a non-string value is the 422 naming follows_up; nothing is written."""
    _hold_entries(mock_ariel_service, "e-1")
    mock_ariel_service.create_entry = AsyncMock(return_value=_create_result())
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    response = _post_create(client, route, {"follows_up": value})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_entry_field"
    assert body["field"] == "follows_up"
    mock_ariel_service.create_entry.assert_not_called()
    mock_ariel_service.repository.upsert_entry.assert_not_called()


def test_json_create_reports_no_attachments_published(client, mock_ariel_service):
    """With no attachments, attachments_published is null."""
    mock_ariel_service.create_entry = AsyncMock(return_value=_create_result())

    response = _post_create(client, "json", {})

    assert response.status_code == 200
    assert response.json()["attachments_published"] is None


def test_upload_without_files_reports_no_attachments_published(client, mock_ariel_service):
    """An upload carrying no files reports null, whatever the result's default says."""
    mock_ariel_service.create_entry = AsyncMock(return_value=_create_result())

    response = _post_create(client, "upload", {})

    assert response.status_code == 200
    data = response.json()
    assert data["attachment_count"] == 0
    assert data["attachments_published"] is None


@pytest.mark.parametrize("published", [True, False])
def test_upload_reports_whether_attachments_were_published(client, mock_ariel_service, published):
    """With files, attachments_published is the service's answer: True, or False if kept local."""
    mock_ariel_service.create_entry = AsyncMock(
        return_value=_create_result(attachment_count=1, attachments_published=published)
    )

    response = _post_create(client, "upload", {}, files=_upload_files())

    assert response.status_code == 200
    data = response.json()
    assert data["attachment_count"] == 1
    assert data["attachments_published"] is published


def test_upload_local_only_fallback_reports_attachments_not_published(client, mock_ariel_service):
    """A local-only save with files reports False: the files stayed in ARIEL."""
    mock_ariel_service.create_entry = AsyncMock(side_effect=NotImplementedError("read-only"))
    mock_ariel_service.repository.upsert_entry = AsyncMock()

    with patch.object(routes, "_store_and_link_attachments", AsyncMock(return_value=1)):
        response = _post_create(client, "upload", {}, files=_upload_files())

    assert response.status_code == 200
    data = response.json()
    assert data["attachment_count"] == 1
    assert data["attachments_published"] is False


# ---------------------------------------------------------------------------
# C-THREAD: GET /api/entries/{id}/thread
# ---------------------------------------------------------------------------


def _hold_thread(service, entries, children=()):
    """The repository holds ``entries`` by id; ``get_follow_ups`` returns ``children``."""
    held = {e["entry_id"]: e for e in entries}

    async def get_entry(entry_id):
        return held.get(entry_id)

    service.repository.get_entry = AsyncMock(side_effect=get_entry)
    service.repository.get_follow_ups = AsyncMock(return_value=list(children))


def test_thread_returns_parent_and_children_oldest_first_as_rows(client, field_service):
    """A follow-up with a held parent and two follow-ups of its own returns the
    parent and both children as list rows, children in repository order."""
    parent = _field_entry(entry_id="P-1")
    entry = _field_entry(entry_id="E-1", metadata={"follows_up": "P-1"})
    first = _field_entry(entry_id="C-1", timestamp=datetime(2025, 10, 6, 10, 0))
    second = _field_entry(entry_id="C-2", timestamp=datetime(2025, 10, 6, 11, 0))
    _hold_thread(field_service, [parent, entry], children=[first, second])

    response = client.get("/api/entries/E-1/thread")

    assert response.status_code == 200
    data = response.json()
    assert data["parent_id"] == "P-1"
    assert data["parent"]["entry_id"] == "P-1"
    assert [c["entry_id"] for c in data["children"]] == ["C-1", "C-2"]
    field_service.repository.get_follow_ups.assert_awaited_once_with("E-1")
    # Same row builder as the list: every child carries the C-ROW shape.
    row = client.get("/api/entries/E-1/thread").json()["children"][0]
    assert _ROW_KEYS <= set(row)
    assert set(data["parent"]) == set(row)


def test_thread_keeps_parent_id_when_parent_is_not_held(client, field_service):
    """An unresolved parent leaves ``parent`` null and ``parent_id`` set."""
    entry = _field_entry(entry_id="E-1", metadata={"follows_up": "GONE-9"})
    _hold_thread(field_service, [entry])

    response = client.get("/api/entries/E-1/thread")

    assert response.status_code == 200
    assert response.json() == {"parent_id": "GONE-9", "parent": None, "children": []}


def test_thread_of_unthreaded_entry_is_empty(client, field_service):
    """An entry that follows nothing and has no follow-ups returns an empty thread."""
    entry = _field_entry(entry_id="E-1")
    _hold_thread(field_service, [entry])

    response = client.get("/api/entries/E-1/thread")

    assert response.status_code == 200
    assert response.json() == {"parent_id": None, "parent": None, "children": []}


def test_thread_never_lists_the_entry_as_its_own_child(client, field_service):
    """The entry itself is dropped from its children even if the store returns it."""
    entry = _field_entry(entry_id="E-1")
    other = _field_entry(entry_id="C-1")
    _hold_thread(field_service, [entry], children=[entry, other])

    data = client.get("/api/entries/E-1/thread").json()

    assert [c["entry_id"] for c in data["children"]] == ["C-1"]


def test_thread_of_unknown_entry_is_404(client, field_service):
    """An id the store does not hold returns 404."""
    _hold_thread(field_service, [])

    response = client.get("/api/entries/NOPE/thread")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    field_service.repository.get_follow_ups.assert_not_awaited()


def test_thread_over_the_cap_keeps_the_newest_oldest_first_and_flags_truncated(
    client, field_service
):
    """A17: the repository returns the newest ``limit + 1`` follow-ups oldest
    first; the route keeps the newest ``limit``, still oldest first, and sets
    ``truncated``."""
    from osprey.services.ariel_search.database.repository import FOLLOW_UPS_LIMIT

    entry = _field_entry(entry_id="E-1")
    children = [_field_entry(entry_id=f"C-{i:04d}") for i in range(FOLLOW_UPS_LIMIT + 1)]
    _hold_thread(field_service, [entry], children=children)

    data = client.get("/api/entries/E-1/thread").json()

    assert data["truncated"] is True
    ids = [c["entry_id"] for c in data["children"]]
    assert len(ids) == FOLLOW_UPS_LIMIT
    assert ids == [f"C-{i:04d}" for i in range(1, FOLLOW_UPS_LIMIT + 1)]


def test_thread_at_the_cap_is_not_truncated(client, field_service):
    """Exactly ``limit`` follow-ups is the whole thread: no ``truncated`` flag."""
    from osprey.services.ariel_search.database.repository import FOLLOW_UPS_LIMIT

    entry = _field_entry(entry_id="E-1")
    children = [_field_entry(entry_id=f"C-{i:04d}") for i in range(FOLLOW_UPS_LIMIT)]
    _hold_thread(field_service, [entry], children=children)

    data = client.get("/api/entries/E-1/thread").json()

    assert len(data["children"]) == FOLLOW_UPS_LIMIT
    assert data.get("truncated", False) is False


# ---------------------------------------------------------------------------
# A8: GET /api/entries/{id}/enhancements (the inspector's cards)
# ---------------------------------------------------------------------------


def _inspect_modules(*, caption_inspect=None, caption_configure=None):
    """Fake ``image_caption`` and ``text_embedding`` classes for the factory.

    ``image_caption`` inspects through ``caption_inspect`` (an async callable)
    and configures through ``caption_configure``; ``text_embedding`` keeps the
    base ``inspect``, which answers None.
    """
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule

    class _Caption(BaseEnhancementModule):
        runs_inline = False
        bound = None

        @property
        def name(self):
            return "image_caption"

        def configure(self, config):
            if caption_configure is not None:
                caption_configure(config)

        def bind(self, config):
            type(self).bound = config

        async def enhance(self, entry, conn):  # pragma: no cover - never driven
            raise NotImplementedError

        async def inspect(self, entry_id, repository):
            if caption_inspect is None:
                return None
            return await caption_inspect(entry_id, repository)

    class _Embedding(BaseEnhancementModule):
        @property
        def name(self):
            return "text_embedding"

        async def enhance(self, entry, conn):  # pragma: no cover - never driven
            raise NotImplementedError

    return _Caption, _Embedding


@pytest.fixture
def inspect_service(field_service):
    """The field service whose registry also hands the factory module classes."""
    from osprey.registry.base import ArielEnhancementModuleRegistration

    def install(caption_cls, embedding_cls, *, caption_label="Picture captions"):
        regs = {
            "image_caption": (
                caption_cls,
                ArielEnhancementModuleRegistration(
                    name="image_caption",
                    module_path="x",
                    class_name="C",
                    description="d",
                    execution_order=5,
                    label=caption_label,
                    feeds=("keyword", "semantic"),
                ),
            ),
            "text_embedding": (
                embedding_cls,
                ArielEnhancementModuleRegistration(
                    name="text_embedding",
                    module_path="x",
                    class_name="E",
                    description="d",
                    execution_order=20,
                ),
            ),
        }
        from osprey.registry import get_registry

        get_registry().get_ariel_enhancement_module.side_effect = regs.get

    field_service.install_modules = install
    return field_service


def _cards(client, entry_id="E-1"):
    response = client.get(f"/api/entries/{entry_id}/enhancements")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["entry_id"] == entry_id
    return {card["module"]: card for card in data["modules"]}, data


def test_enhancements_one_card_per_enabled_module_in_order(client, inspect_service):
    """Every enabled module gets one card, in execution order, carrying the view
    state, the registration's label/order/feeds, the stored fields, the re-run
    command and the module's own detail."""
    calls = []

    async def caption_inspect(entry_id, repository):
        calls.append((entry_id, repository))
        return {"captions": [{"model": "cap-v1", "text": "a beam"}]}

    caption, embedding = _inspect_modules(caption_inspect=caption_inspect)
    inspect_service.install_modules(caption, embedding)
    inspect_service.repository.get_entry = AsyncMock(return_value=_field_entry())

    cards, data = _cards(client)

    assert [c["module"] for c in data["modules"]] == ["image_caption", "text_embedding"]
    first = cards["image_caption"]
    assert first["label"] == "Picture captions"
    assert first["order"] == 5
    assert first["feeds"] == ["keyword", "semantic"]
    assert first["state"] == "stale"  # cap-v1 stored, cap-v2 current
    assert first["marker"] == "cap-v1"
    assert first["command"] == "osprey ariel enhance --module image_caption --entry E-1"
    assert first["detail"] == {"captions": [{"model": "cap-v1", "text": "a beam"}]}
    assert calls == [("E-1", inspect_service.repository)]
    # Built through the factory: configured and bound to the whole config.
    assert caption.bound is inspect_service.config
    # The view state is the one the entry routes report.
    assert cards["text_embedding"]["state"] == "complete"


def test_enhancements_status_only_without_inspect(client, inspect_service):
    """A module without its own ``inspect`` gets a status-only card; a
    registration without a label or feeds falls back to the name and empty."""
    caption, embedding = _inspect_modules()
    inspect_service.install_modules(caption, embedding)
    inspect_service.repository.get_entry = AsyncMock(return_value=_field_entry())

    cards, _ = _cards(client)

    card = cards["text_embedding"]
    assert card["detail"] is None
    assert card["label"] == "text_embedding"
    assert card["feeds"] == []
    assert card["order"] == 20
    assert card["state"] == "complete"
    assert card["command"] is None
    assert card["error"] is None and card["failed_at"] is None


def test_enhancements_failed_card_carries_stored_fields_and_command(client, inspect_service):
    """A failed module's card carries the stored failure and its re-run command."""
    caption, embedding = _inspect_modules()
    inspect_service.install_modules(caption, embedding)
    status = {
        "image_caption": {"status": "complete", "marker": "cap-v2", "completed_at": "T1"},
        "text_embedding": {
            "status": "failed",
            "failed_at": "2025-10-06 10:00:00",
            "error": "boom",
            "attempts": 2,
        },
    }
    inspect_service.repository.get_entry = AsyncMock(
        return_value=_field_entry(enhancement_status=status)
    )

    cards, _ = _cards(client)

    assert cards["image_caption"]["state"] == "complete"
    assert cards["image_caption"]["completed_at"] == "T1"
    assert cards["image_caption"]["command"] is None
    failed = cards["text_embedding"]
    assert failed["state"] == "failed"
    assert failed["failed_at"] == "2025-10-06 10:00:00"
    assert failed["error"] == "boom"
    assert failed["attempts"] == 2
    assert failed["command"] == "osprey ariel enhance --module text_embedding --entry E-1"


def test_enhancements_inspect_that_raises_gives_status_only_card(client, inspect_service):
    """An ``inspect()`` that raises gives ``detail: null`` and a warning, never a 500."""

    async def caption_inspect(_entry_id, _repository):
        raise RuntimeError("store went away")

    caption, embedding = _inspect_modules(caption_inspect=caption_inspect)
    inspect_service.install_modules(caption, embedding)
    inspect_service.repository.get_entry = AsyncMock(return_value=_field_entry())

    with patch.object(routes.logger, "warning") as warning:
        cards, _ = _cards(client)

    assert cards["image_caption"]["detail"] is None
    assert cards["image_caption"]["state"] == "stale"
    assert cards["text_embedding"]["state"] == "complete"
    assert warning.called
    assert "image_caption" in str(warning.call_args)


def test_enhancements_configure_that_raises_gives_status_only_card(client, inspect_service):
    """A module whose ``configure()`` raises still gets its status-only card,
    and the next module is still built and inspected."""

    def caption_configure(_config):
        raise ValueError("bad caption config")

    caption, embedding = _inspect_modules(caption_configure=caption_configure)
    inspected = []

    async def embedding_inspect(_self, entry_id, _repository):
        inspected.append(entry_id)
        return {"table": "text_embeddings_m"}

    embedding.inspect = embedding_inspect
    inspect_service.install_modules(caption, embedding)
    inspect_service.repository.get_entry = AsyncMock(return_value=_field_entry())

    with patch.object(routes.logger, "warning") as warning:
        cards, _ = _cards(client)

    assert cards["image_caption"]["detail"] is None
    assert cards["image_caption"]["state"] == "stale"
    assert cards["text_embedding"]["detail"] == {"table": "text_embeddings_m"}
    assert inspected == ["E-1"]
    assert warning.called


def test_enhancements_of_unknown_entry_is_404(client, inspect_service):
    """An id the store does not hold returns 404."""
    caption, embedding = _inspect_modules()
    inspect_service.install_modules(caption, embedding)
    inspect_service.repository.get_entry = AsyncMock(return_value=None)

    response = client.get("/api/entries/NOPE/enhancements")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_enhancements_builtins_carry_labels_and_feeds():
    """The five built-in registrations name a label and the searches they feed."""
    from osprey.registry.builtins import FrameworkRegistryProvider

    config = FrameworkRegistryProvider().get_registry_config()
    regs = {r.name: r for r in config.ariel_enhancement_modules}
    assert set(regs) == {
        "image_caption",
        "semantic_processor",
        "text_embedding",
        "image_embedding",
        "qmd_export",
    }
    for reg in regs.values():
        assert reg.label and reg.label != reg.name
        assert reg.feeds


def test_enhancements_base_inspect_answers_none():
    """``BaseEnhancementModule.inspect`` defaults to None (status-only card)."""
    import asyncio

    _, embedding = _inspect_modules()
    assert asyncio.run(embedding().inspect("E-1", MagicMock())) is None


# ---------------------------------------------------------------------------
# Related entries (A9)
# ---------------------------------------------------------------------------


def _related_config(*, text_embedding=True, semantic=True):
    """A config whose text_embedding module names one model, ``embed-a``."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": semantic},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": text_embedding,
                    "models": [{"name": "embed-a", "dimension": 4}],
                },
            },
        }
    )


def _related_entry(entry_id):
    return {
        "entry_id": entry_id,
        "source_system": "ARIEL Web",
        "timestamp": datetime(2025, 10, 6, 9, 30),
        "author": "J. Smith",
        "raw_text": f"Text of {entry_id}",
        "attachments": [],
        "metadata": {},
        "enhancement_status": {},
        "has_images": False,
    }


@pytest.fixture
def related_service(mock_ariel_service):
    """The mock service with text embedding on, a stored vector for E-1, and
    semantic search returning E-1 itself plus six neighbours."""
    service = mock_ariel_service
    service.config = _related_config()
    repo = service.repository
    repo.get_entry = AsyncMock(return_value=_related_entry("E-1"))
    repo.get_text_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    repo.text_input_key_column = AsyncMock(return_value=False)
    repo.get_text_build_keys = AsyncMock(return_value={})
    hits = [(_related_entry("E-1"), 1.0)] + [
        (_related_entry(f"N-{i}"), 0.9 - i / 10) for i in range(6)
    ]
    repo.semantic_search = AsyncMock(side_effect=lambda **kw: hits[: kw["max_results"]])
    return service


def test_related_returns_neighbours_without_self(related_service):
    """Default limit 5: semantic_search asked for 6, self dropped, five rows back."""
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "ok"
    ids = [row["entry_id"] for row in body["entries"]]
    assert ids == ["N-0", "N-1", "N-2", "N-3", "N-4"]
    assert "E-1" not in ids
    related_service.repository.get_text_embedding.assert_awaited_once_with(
        "text_embeddings_embed_a", "E-1"
    )
    kwargs = related_service.repository.semantic_search.await_args.kwargs
    assert kwargs["query_embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert kwargs["model_name"] == "embed-a"
    assert kwargs["max_results"] == 6
    assert kwargs["similarity_threshold"] is None


def test_related_limit_caps_rows_when_self_absent(related_service):
    """When self is not among the hits, the limit+1th hit is trimmed off."""
    hits = [(_related_entry(f"N-{i}"), 0.9) for i in range(4)]
    related_service.repository.semantic_search = AsyncMock(return_value=hits)
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related?limit=3")

    assert response.status_code == 200
    assert [r["entry_id"] for r in response.json()["entries"]] == ["N-0", "N-1", "N-2"]
    kwargs = related_service.repository.semantic_search.await_args.kwargs
    assert kwargs["max_results"] == 4


@pytest.mark.parametrize("limit", [0, 11])
def test_related_limit_out_of_range_is_refused(related_service, limit):
    """The limit is 1..10."""
    client = TestClient(routes_app(related_service))

    response = client.get(f"/api/entries/E-1/related?limit={limit}")

    assert response.status_code == 422
    related_service.repository.semantic_search.assert_not_awaited()


def test_related_max_limit_ten(related_service):
    """limit=10 is accepted and asks semantic_search for 11."""
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related?limit=10")

    assert response.status_code == 200
    assert related_service.repository.semantic_search.await_args.kwargs["max_results"] == 11
    assert len(response.json()["entries"]) == 6


def test_related_disabled_when_text_embedding_off(related_service):
    """text_embedding off: state disabled, no vector read, no search."""
    related_service.config = _related_config(text_embedding=False)
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related")

    assert response.status_code == 200
    assert response.json() == {"state": "disabled", "entries": []}
    related_service.repository.get_text_embedding.assert_not_awaited()
    related_service.repository.semantic_search.assert_not_awaited()


def test_related_disabled_when_semantic_search_off(related_service):
    """The search the route runs through refuses: reported as disabled, not a 500."""
    from osprey.services.ariel_search.exceptions import ModuleNotEnabledError

    related_service.repository.semantic_search = AsyncMock(
        side_effect=ModuleNotEnabledError("off", module_name="semantic")
    )
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related")

    assert response.status_code == 200
    assert response.json() == {"state": "disabled", "entries": []}


def test_related_no_embedding_when_entry_has_no_row(related_service):
    """No stored vector for the entry: state no_embedding, no search."""
    related_service.repository.get_text_embedding = AsyncMock(return_value=None)
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related")

    assert response.status_code == 200
    assert response.json() == {"state": "no_embedding", "entries": []}
    related_service.repository.semantic_search.assert_not_awaited()


def test_related_ok_with_no_neighbours(related_service):
    """Only self comes back: state ok with an empty list."""
    related_service.repository.semantic_search = AsyncMock(
        return_value=[(_related_entry("E-1"), 1.0)]
    )
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/E-1/related")

    assert response.status_code == 200
    assert response.json() == {"state": "ok", "entries": []}


def test_related_unknown_entry_is_404(related_service):
    """An unknown id gives 404 before any vector read."""
    related_service.repository.get_entry = AsyncMock(return_value=None)
    client = TestClient(routes_app(related_service))

    response = client.get("/api/entries/NOPE/related")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    related_service.repository.get_text_embedding.assert_not_awaited()
