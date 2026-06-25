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
from osprey.services.ariel_search.models import SearchMode
from osprey.services.ariel_search.search.base import SearchToolDescriptor


def _build_mock_registry():
    """Build a mock registry that provides ARIEL search modules and pipelines."""
    registry = MagicMock()

    # Build mock search modules
    keyword_mod = types.ModuleType("keyword")
    keyword_mod.get_tool_descriptor = lambda: SearchToolDescriptor(  # type: ignore[attr-defined]
        name="keyword_search",
        description="Full-text keyword search",
        search_mode=SearchMode.KEYWORD,
        args_schema=MagicMock(),
        execute=AsyncMock(),
        format_result=MagicMock(),
    )
    keyword_mod.get_parameter_descriptors = None  # type: ignore[attr-defined]

    semantic_mod = types.ModuleType("semantic")
    semantic_mod.get_tool_descriptor = lambda: SearchToolDescriptor(  # type: ignore[attr-defined]
        name="semantic_search",
        description="Semantic similarity search",
        search_mode=SearchMode.SEMANTIC,
        args_schema=MagicMock(),
        execute=AsyncMock(),
        format_result=MagicMock(),
        needs_embedder=True,
    )
    semantic_mod.get_parameter_descriptors = None  # type: ignore[attr-defined]

    registry.list_ariel_search_modules.return_value = ["keyword", "semantic"]
    registry.get_ariel_search_module.side_effect = lambda n: {
        "keyword": keyword_mod,
        "semantic": semantic_mod,
    }.get(n)

    return registry


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
            source_system="ALS eLog",
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
            "ALS olog write failed with HTTP 403: bad password",
            source_system="ALS eLog",
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
        side_effect=AuthenticationRequiredError("creds required", source_system="ALS eLog")
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
        side_effect=IngestionError("olog down", source_system="ALS eLog")
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
    mock_ariel_service.repository.store_attachment.assert_called_once()


def test_upload_publish_success_stores_attachments_locally(client, mock_ariel_service):
    """Published entry: text goes to OLOG, files stay in ARIEL (API can't take files)."""
    from osprey.services.ariel_search.models import FacilityEntryCreateResult, SyncStatus

    mock_ariel_service.create_entry = AsyncMock(
        return_value=FacilityEntryCreateResult(
            entry_id="99999",
            source_system="ALS eLog",
            sync_status=SyncStatus.PENDING_SYNC,
            message="Entry 99999 created in ALS eLog",
        )
    )
    mock_ariel_service.repository.store_attachment = AsyncMock()
    mock_ariel_service.repository.upsert_entry = AsyncMock()
    mock_ariel_service.repository.get_entry = AsyncMock(
        return_value={
            "entry_id": "99999",
            "source_system": "ALS eLog",
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
    mock_ariel_service.repository.store_attachment.assert_called_once()
    assert "ariel" in data["message"].lower()


def _mock_adapter(*, supports_write, requires_write_auth, source_system):
    adapter = MagicMock()
    adapter.supports_write = supports_write
    adapter.requires_write_auth = requires_write_auth
    adapter.source_system_name = source_system
    return adapter


def test_publish_info_requires_auth(client, mock_ariel_service):
    """A write adapter that needs credentials reports requires_auth=True."""
    adapter = _mock_adapter(supports_write=True, requires_write_auth=True, source_system="ALS eLog")
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    assert response.status_code == 200
    data = response.json()
    assert data["supports_write"] is True
    assert data["requires_auth"] is True
    assert data["source_system"] == "ALS eLog"


def test_publish_info_no_auth(client, mock_ariel_service):
    """A no-auth write adapter reports requires_auth=False (publishes without creds)."""
    adapter = _mock_adapter(
        supports_write=True, requires_write_auth=False, source_system="Generic JSON"
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    data = response.json()
    assert data["supports_write"] is True
    assert data["requires_auth"] is False


def test_publish_info_read_only(client, mock_ariel_service):
    """A read-only adapter reports requires_auth=False — credentials are irrelevant."""
    adapter = _mock_adapter(
        supports_write=False, requires_write_auth=True, source_system="JLab Logbook"
    )
    with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
        response = client.get("/api/publish-info")

    data = response.json()
    assert data["supports_write"] is False
    assert data["requires_auth"] is False


def test_publish_info_no_adapter_configured(client, mock_ariel_service):
    """No ingestion adapter configured degrades gracefully to read-only."""
    response = client.get("/api/publish-info")

    assert response.status_code == 200
    data = response.json()
    assert data["supports_write"] is False
    assert data["requires_auth"] is False


def test_status_endpoint(client, mock_ariel_service):
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

    result = routes._entry_to_response(entry, score=0.95, highlights=["highlight1"])

    assert result.entry_id == "test-123"
    assert result.author == "Test Author"
    assert result.score == 0.95
    assert result.highlights == ["highlight1"]
    assert result.metadata == {"key": "value"}


def test_search_mode_mapping(client, mock_ariel_service):
    """Test that search modes are correctly mapped."""
    from osprey.services.ariel_search.models import SearchMode as ServiceSearchMode

    modes = {
        "keyword": ServiceSearchMode.KEYWORD,
        "semantic": ServiceSearchMode.SEMANTIC,
    }

    for api_mode, expected_service_mode in modes.items():
        client.post(
            "/api/search",
            json={
                "query": "test",
                "mode": api_mode,
                "max_results": 10,
            },
        )

        # Check that correct mode was passed
        call_kwargs = mock_ariel_service.search.call_args.kwargs
        assert call_kwargs["mode"] == expected_service_mode


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
    from osprey.services.ariel_search.models import SearchMode as ServiceSearchMode

    call_kwargs = mock_ariel_service.search.call_args.kwargs
    assert call_kwargs["mode"] == ServiceSearchMode.KEYWORD
