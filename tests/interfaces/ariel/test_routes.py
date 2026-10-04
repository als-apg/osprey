"""Tests for ARIEL web API routes."""

from __future__ import annotations

import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.search.base import SearchToolDescriptor


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
    service.get_status = AsyncMock(return_value=mock_status)

    return service


@pytest.fixture
def test_app(mock_ariel_service):
    """Create a test FastAPI app with mocked service."""
    app = FastAPI()

    # Add the router
    app.include_router(routes.router)

    # Mock the service in app state
    app.state.ariel_service = mock_ariel_service
    # The lifespan resolves this tier flag; a routes-only app states it.
    app.state.config_panel_enabled = True

    return app


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


def test_upload_publish_success_stores_attachments_locally(client, mock_ariel_service):
    """Published entry: text goes to OLOG, files stay in ARIEL (API can't take files)."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    mock_ariel_service.create_entry = AsyncMock(
        return_value=FacilityEntryCreateResult(
            entry_id="99999",
            source_system="Example eLog",
            sync_status=SyncStatus.PENDING_SYNC,
            message="Entry 99999 created in Example eLog",
        )
    )
    mock_ariel_service.repository.store_attachment = AsyncMock()
    mock_ariel_service.repository.upsert_entry = AsyncMock()
    mock_ariel_service.repository.get_entry = AsyncMock(
        return_value={
            "entry_id": "99999",
            "source_system": "Example eLog",
            "timestamp": datetime.now(),
            "author": "op",
            "raw_text": "Test\n\nBody",
            "attachments": [],
            "metadata": {},
        }
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
    # Files were stored in ARIEL and the operator is told they were not published.
    mock_ariel_service.repository.insert_native_attachment.assert_called_once()
    assert "ariel" in data["message"].lower()


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
def test_publish_info_no_auth(client):
    """A no-auth write adapter reports requires_auth=False (publishes without creds)."""
    adapter = _mock_adapter(
        supports_write=True, requires_write_auth=False, source_system="Generic JSON"
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    data = response.json()
    assert data["supports_write"] is True
    assert data["requires_auth"] is False


@pytest.mark.usefixtures("mock_ariel_service")
def test_publish_info_read_only(client):
    """A read-only adapter reports requires_auth=False — credentials are irrelevant."""
    adapter = _mock_adapter(
        supports_write=False, requires_write_auth=True, source_system="JLab Logbook"
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    data = response.json()
    assert data["supports_write"] is False
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
def audit_zone(tmp_path, monkeypatch):
    """Redirect the audit zone. ``writer.audit_dir`` is the ledger's one seam."""
    from osprey.audit import writer

    zone = tmp_path / "audit-zone" / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    return zone


def _audit_records(zone):
    import json

    from osprey.audit.protected import SURFACE_HTTP_CONFIG
    from osprey.utils.identity import acting_identity

    path = zone / acting_identity() / f"{SURFACE_HTTP_CONFIG}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def gated_config(client, tmp_path):
    """A config.yml carrying protected keys, wired into the ARIEL app state."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(_PROTECTED_DOC.format(state=tmp_path / "state"))
    client.app.state.config_path = config_path
    return config_path


def test_put_config_refuses_a_removed_protected_key(client, gated_config, audit_zone):
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

    records = _audit_records(audit_zone)
    assert len(records) == 1
    assert records[0]["surface"] == "http_config"
    assert "target=config.yml" in records[0]["detail"]
    assert records[0]["subject"] == "agent_data.base_dir"
    assert records[0]["reason"] == "protected_key"


def test_put_config_refuses_a_changed_protected_value(client, gated_config, audit_zone):
    """Flipping the write gate through the YAML editor is the write that must not land."""
    before = gated_config.read_bytes()
    flipped = _PROTECTED_DOC.format(state=gated_config.parent / "state").replace(
        "writes_enabled: false", "writes_enabled: true"
    )

    response = client.put("/api/config", json={"content": flipped})

    assert response.status_code == 403
    assert "control_system.writes_enabled" in response.json()["detail"]
    assert gated_config.read_bytes() == before

    records = _audit_records(audit_zone)
    assert [r["subject"] for r in records] == ["control_system.writes_enabled"]


@pytest.mark.usefixtures("gated_config")
def test_put_config_refusal_leaks_no_value(client, audit_zone):
    """Config values are secrets; a refusal reports the key, never the value."""
    import json as _j

    sentinel = "qqzzSENTINELvalue77"
    planted = _PROTECTED_DOC.format(state=sentinel)

    response = client.put("/api/config", json={"content": planted})

    assert response.status_code == 403
    assert sentinel not in response.text
    assert sentinel not in _j.dumps(_audit_records(audit_zone))


def test_put_config_allows_an_unprotected_edit(client, gated_config, audit_zone):
    """An edit that leaves every protected key alone still saves, and still backs up."""
    from osprey.utils.config_writer import config_backup_path

    before = gated_config.read_text()
    updated = before.replace("project_name: original", "project_name: updated")

    response = client.put("/api/config", json={"content": updated})

    assert response.status_code == 200
    assert gated_config.read_text() == updated
    assert config_backup_path(gated_config).read_text() == before
    assert _audit_records(audit_zone) == []


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
_PNG_URL = "https://logbook.invalid/pic.png"
_PDF_URL = "https://logbook.invalid/doc.pdf"


def _att_entry(attachments, **extra):
    """An entry dict carrying the given JSONB attachment items."""
    entry = {
        "entry_id": "e-att",
        "source_system": "Test",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "author": "op",
        "raw_text": "text",
        "attachments": attachments,
        "metadata": {},
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "updated_at": datetime(2024, 1, 1, 12, 0, 0),
        "summary": None,
        "keywords": [],
    }
    entry.update(extra)
    return entry


def _row(entry_id, item, *, mime_type, viewable):
    """An ``attachment_files`` row for an item, copied, viewable or not."""
    from osprey.services.ariel_search.attachments import attachment_id_for

    return {
        "attachment_id": attachment_id_for(entry_id, item),
        "entry_id": entry_id,
        "filename": item.get("filename"),
        "mime_type": mime_type,
        "copy_status": "copied",
        "skip_reason": None,
        "source_url": item.get("url"),
        "rendition_sha256": "ab" * 32 if viewable else None,
    }


def _to_response(entry, rows):
    return routes._entry_to_response(entry, attachment_rows=rows, model_id=None, file_source=False)


def test_attachment_response_fields_are_summary_keys_plus_display_url():
    from osprey.interfaces.ariel.api.schemas import AttachmentResponse
    from osprey.services.ariel_search.attachments.summaries import SUMMARY_KEYS

    assert set(AttachmentResponse.model_fields) == set(SUMMARY_KEYS) | {"display_url"}


def test_display_url_viewable_is_rendition():
    from osprey.services.ariel_search.attachments import attachment_id_for

    item = {"url": _PNG_URL, "filename": "pic.png", "type": "image/png"}
    entry = _att_entry([item])
    rows = [_row("e-att", item, mime_type="image/png", viewable=True)]

    att = _to_response(entry, rows).attachments[0]

    att_id = attachment_id_for("e-att", item)
    assert att.viewable is True
    assert att.display_url == f"/attachments/{att_id}/rendition"
    assert att.url == _PNG_URL


def test_display_url_copied_not_viewable_is_original():
    from osprey.services.ariel_search.attachments import attachment_id_for

    item = {"url": _PDF_URL, "filename": "doc.pdf"}
    entry = _att_entry([item])
    rows = [_row("e-att", item, mime_type="application/pdf", viewable=False)]

    att = _to_response(entry, rows).attachments[0]

    assert att.viewable is False
    assert att.copy_status == "copied"
    assert att.display_url == f"/attachments/{attachment_id_for('e-att', item)}"


def test_display_url_pending_falls_back_to_source_url():
    item = {"url": _PNG_URL, "filename": "pic.png"}

    att = _to_response(_att_entry([item]), []).attachments[0]

    assert att.copy_status == "pending"
    assert att.display_url == _PNG_URL


def test_display_url_null_without_url():
    item = {"url": "relative/pic.png", "filename": "pic.png"}

    att = _to_response(_att_entry([item]), []).attachments[0]

    assert att.url is None
    assert att.display_url is None


def test_display_url_unmigrated_native_item_is_original():
    item = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "pic.png", "type": "image/png"}

    att = _to_response(_att_entry([item]), None).attachments[0]

    assert att.attachment_id is None
    assert att.display_url == f"/attachments/{_NATIVE_ID}"


def test_display_url_migrated_native_without_row_is_null():
    """With copy state a native item without a finished row has nowhere to show."""
    item = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "pic.png"}

    att = _to_response(_att_entry([item]), []).attachments[0]

    assert att.display_url is None


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "file:///etc/passwd", "data:image/png;base64,AA"]
)
@pytest.mark.parametrize("rows", [None, []])
def test_unsafe_urls_are_nulled(url, rows):
    item = {"url": url, "filename": "x.png"}

    att = _to_response(_att_entry([item]), rows).attachments[0]

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
    item = {"url": _PNG_URL, "filename": "pic.png"}
    entry = _att_entry([item], _matched_via=("text", "caption"), _matched_attachment_ids=("a",))

    result = _to_response(entry, [])

    assert result.matched_via == ["text", "caption"]
    assert result.matched_attachment_ids == ["a"]


def test_entry_response_match_fields_default_empty():
    result = _to_response(_att_entry([]), None)

    assert result.matched_via == []
    assert result.matched_attachment_ids == []


def test_entry_response_orders_matched_attachment_first():
    from osprey.services.ariel_search.attachments import attachment_id_for

    first = {"url": "https://logbook.invalid/a.png", "filename": "a.png"}
    second = {"url": "https://logbook.invalid/b.png", "filename": "b.png"}
    second_id = attachment_id_for("e-att", second)
    entry = _att_entry([first, second], _matched_attachment_ids=(second_id,))
    rows = [
        _row("e-att", first, mime_type="image/png", viewable=True),
        _row("e-att", second, mime_type="image/png", viewable=True),
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
    return _att_entry(items)


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
    item = {"url": _PNG_URL, "filename": "pic.png", "type": "image/png"}
    row = _row("e-att", item, mime_type="image/png", viewable=True)
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value={"e-att": [row]})

    att = _route_entries(client, mock_ariel_service, route, _att_entry([item]))[0]["attachments"][0]

    assert att["display_url"] == f"/attachments/{row['attachment_id']}/rendition"
    assert set(att) == set(routes.AttachmentResponse.model_fields)


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_unmigrated_store_returns_fallback_summaries(client, mock_ariel_service, route):
    native = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "n.png", "type": "image/png"}
    remote = {"url": _PNG_URL, "filename": "pic.png", "type": "image/png"}
    mock_ariel_service.repository.get_attachment_rows = AsyncMock(return_value=None)

    atts = _route_entries(client, mock_ariel_service, route, _att_entry([native, remote]))[0][
        "attachments"
    ]

    assert [a["copy_status"] for a in atts] == ["pending", "pending"]
    assert all(a["viewable"] is False and a["attachment_id"] is None for a in atts)
    assert atts[0]["display_url"] == f"/attachments/{_NATIVE_ID}"
    assert atts[1]["display_url"] == _PNG_URL


@pytest.mark.parametrize("route", _ROUTES)
def test_routes_failing_reader_equals_unmigrated(client, mock_ariel_service, route):
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    native = {"url": f"/api/attachments/{_NATIVE_ID}", "filename": "n.png", "type": "image/png"}
    entry = _att_entry([native])
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

    att = _route_entries(client, mock_ariel_service, "detail", _att_entry([native]))[0][
        "attachments"
    ][0]
    served = client.get("/api" + att["display_url"])

    assert served.status_code == 200
    mock_ariel_service.repository.get_attachment_original.assert_awaited_once_with(_NATIVE_ID)


def _fused_entry(entry_id, via):
    return _att_entry([], entry_id=entry_id, _score=0.5, _matched_via=list(via))


def test_search_hybrid_page_and_sources_stop_at_max_results(client, mock_ariel_service):
    """A lane admitting image-only hits beyond max_results: the page holds max_results.

    The route slices the fused entries itself and names exactly the shown
    entries in ``sources``, so ``total_results == len(sources)``.
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

    response = client.post("/api/search", json={"query": "q", "mode": "hybrid", "max_results": 10})

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == len(data["sources"]) == len(data["entries"]) == 10
    assert data["sources"] == [e["entry_id"] for e in data["entries"]]
    assert data["sources"] == [f"T{i}" for i in range(1, 11)]


def test_search_keeps_the_service_sources_without_matched_via(client, mock_ariel_service):
    """Without a fused result the route reports the service's sources unchanged."""
    entries = [_att_entry([], entry_id=f"e{i}", _score=0.5) for i in range(2)]
    result = mock_ariel_service.search.return_value
    result.entries = tuple(entries)
    result.sources = ("e0", "e1", "cited")
    result.diagnostics = ()
    result.expanded_terms = ()

    response = client.post("/api/search", json={"query": "q", "mode": "keyword", "max_results": 10})

    assert response.status_code == 200
    assert response.json()["sources"] == ["e0", "e1", "cited"]
