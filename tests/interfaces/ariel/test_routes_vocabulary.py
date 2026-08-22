"""Route behaviour for broken configurations and vocabulary expansion.

Companion to ``test_routes.py`` (which must keep passing unedited): it covers
the app-state fields the lifespan gained — ``config_status``,
``config_errors``, ``config_remedy``, ``config_path`` — and the search
response's ``expanded_terms``.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.interfaces.ariel.app import REMEDY_NO_CONFIG_FILE
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import VocabularyError
from osprey.services.ariel_search.search.base import SearchToolDescriptor

VOCABULARY_ERROR = "ariel.vocabulary.path: /x/vocabulary.yml — file not found"


def _build_mock_registry():
    """Build a mock registry that provides ARIEL search modules."""
    registry = MagicMock()

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

    registry.list_ariel_search_modules.return_value = ["keyword"]
    registry.get_ariel_search_module.side_effect = lambda n: {"keyword": keyword_mod}.get(n)

    return registry


@pytest.fixture(autouse=True)
def _mock_registry():
    """Provide a mock registry for all ARIEL route tests."""
    with patch("osprey.registry.get_registry", return_value=_build_mock_registry()):
        yield


def _make_app(**state) -> FastAPI:
    """Build a bare app carrying the router and the given app-state fields."""
    app = FastAPI()
    app.include_router(routes.router)
    for name, value in state.items():
        setattr(app.state, name, value)
    return app


def _make_service(search_result=None, search_error=None) -> AsyncMock:
    """Build a mock ARIEL service with a real config and a stubbed search."""
    service = AsyncMock()
    service.repository = AsyncMock()
    service.config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
        }
    )
    if search_error is not None:
        service.search = AsyncMock(side_effect=search_error)
    else:
        service.search = AsyncMock(return_value=search_result)
    return service


def _make_result(expanded_terms=()) -> MagicMock:
    """Build a search result with real (non-Mock) reportable fields."""
    result = MagicMock()
    result.entries = []
    result.answer = None
    result.sources = []
    result.search_modes_used = ["keyword"]
    result.reasoning = ""
    result.diagnostics = ()
    result.expanded_terms = expanded_terms
    return result


# ---------------------------------------------------------------------------
# GET /api/capabilities
# ---------------------------------------------------------------------------


def test_capabilities_configuration_invalid_payload_is_service_independent():
    """A vocabulary that would not load answers 200 with the degraded payload."""
    app = _make_app(
        config_status="configuration_invalid",
        config_errors=[VOCABULARY_ERROR],
        config_remedy=VocabularyError.remedy,
        ariel_service=None,
    )

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "status": "configuration_invalid",
        "config_errors": [VOCABULARY_ERROR],
        "remedy": VocabularyError.remedy,
        "categories": {"direct": {"modes": []}},
        "default_mode": None,
        "shared_parameters": [],
        "vocabulary": {"enabled": False, "concepts": 0, "expand_by_default": False},
    }
    assert "ariel.vocabulary.path" in response.json()["remedy"]


def test_capabilities_no_config_file_remedy_names_config_file():
    """The remedy follows the error class: a missing config.yml is no vocabulary fault."""
    app = _make_app(
        config_status="configuration_invalid",
        config_errors=["No config.yml found"],
        config_remedy=REMEDY_NO_CONFIG_FILE,
        ariel_service=None,
    )

    payload = TestClient(app).get("/api/capabilities").json()

    assert payload["status"] == "configuration_invalid"
    assert "CONFIG_FILE" in payload["remedy"]
    assert "ariel.vocabulary.path" not in payload["remedy"]


def test_capabilities_configuration_warning_keeps_the_normal_payload():
    """A warning has a working service: the full payload plus the three config keys."""
    app = _make_app(
        config_status="configuration_warning",
        config_errors=["search_modules.semantic.model is required"],
        config_remedy="fix the named key in config.yml and restart",
        ariel_service=_make_service(),
    )

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "configuration_warning"
    assert payload["config_errors"] == ["search_modules.semantic.model is required"]
    assert payload["remedy"] == "fix the named key in config.yml and restart"
    assert payload["categories"]["direct"]["modes"][0]["name"] == "keyword"
    assert payload["default_mode"] == "keyword"
    assert payload["vocabulary"] == {
        "enabled": False,
        "concepts": 0,
        "expand_by_default": False,
    }


def test_capabilities_without_config_state_returns_the_normal_payload():
    """The pre-existing route-test app (service only) behaves exactly as before."""
    app = _make_app(ariel_service=_make_service())

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"categories", "default_mode", "shared_parameters", "vocabulary"}
    assert "status" not in payload


def test_capabilities_warning_without_service_is_the_database_503():
    """Warning + no service means the database is down, not the config."""
    app = _make_app(
        config_status="configuration_warning",
        config_errors=["search_modules.semantic.model is required"],
        config_remedy="fix the named key in config.yml and restart",
        ariel_service=None,
    )

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 503
    assert "Database unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------


def test_search_maps_vocabulary_error_to_503_naming_key_and_remedy():
    """Database up, vocabulary broken: only the search path degrades."""
    service = _make_service(search_error=VocabularyError([VOCABULARY_ERROR]))
    app = _make_app(
        ariel_service=service,
        config_status="configuration_invalid",
        config_errors=[VOCABULARY_ERROR],
        config_remedy=VocabularyError.remedy,
    )
    client = TestClient(app)

    response = client.post("/api/search", json={"query": "ts bpm", "mode": "keyword"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "ariel.vocabulary.path" in detail
    assert VocabularyError.remedy in detail


def test_entries_and_status_stay_200_while_search_is_degraded():
    """Browse and status never depend on the vocabulary."""
    service = _make_service(search_error=VocabularyError([VOCABULARY_ERROR]))
    service.repository.count_entries = AsyncMock(return_value=0)
    service.repository.search_by_time_range = AsyncMock(return_value=[])
    status = MagicMock()
    status.healthy = True
    status.database_connected = True
    status.database_uri = "postgresql://localhost/ariel"
    status.entry_count = 0
    status.embedding_tables = []
    status.active_embedding_model = None
    status.enabled_search_modules = ["keyword"]
    status.enabled_enhancement_modules = []
    status.last_ingestion = None
    status.errors = []
    service.get_status = AsyncMock(return_value=status)
    app = _make_app(
        ariel_service=service,
        config_status="configuration_invalid",
        config_errors=[VOCABULARY_ERROR],
        config_remedy=VocabularyError.remedy,
    )
    client = TestClient(app)

    assert client.get("/api/entries").status_code == 200
    assert client.get("/api/status").status_code == 200


def test_search_passes_expand_query_through_and_reports_expanded_terms():
    """``expand_query`` rides through advanced_params; groups come back on the wire."""
    result = _make_result(
        expanded_terms=({"original": "ts", "alternatives": ["troubleshoot"]},),
    )
    service = _make_service(search_result=result)
    app = _make_app(ariel_service=service)

    response = TestClient(app).post(
        "/api/search",
        json={
            "query": "ts bpm",
            "mode": "keyword",
            "advanced_params": {"expand_query": False},
        },
    )

    assert response.status_code == 200
    assert service.search.call_args.kwargs["advanced_params"]["expand_query"] is False
    assert response.json()["expanded_terms"] == [
        {"original": "ts", "alternatives": ["troubleshoot"]}
    ]


def test_search_tolerates_a_result_without_real_expansion_groups():
    """A stub result (or a module that never expanded) reports an empty list.

    Mirrors ``test_routes.py``'s bare ``MagicMock`` result, whose
    ``expanded_terms`` attribute is itself a Mock — the route must not try to
    iterate it.
    """
    stub = MagicMock()
    stub.entries = []
    stub.answer = "Test answer"
    stub.sources = []
    stub.search_modes_used = []
    stub.reasoning = ""
    service = _make_service(search_result=stub)
    app = _make_app(ariel_service=service)

    response = TestClient(app).post("/api/search", json={"query": "ts", "mode": "keyword"})

    assert response.status_code == 200
    assert response.json()["expanded_terms"] == []


# ---------------------------------------------------------------------------
# Settings editor and _require_service
# ---------------------------------------------------------------------------


def test_config_endpoints_use_the_path_the_panel_loaded(tmp_path):
    """Editor and loader agree on the file: app.state.config_path wins."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("ariel:\n  vocabulary:\n    enabled: false\n")
    app = _make_app(ariel_service=None, config_path=config_file)
    client = TestClient(app)

    got = client.get("/api/config")
    assert got.status_code == 200
    assert got.json()["parsed"]["ariel"]["vocabulary"]["enabled"] is False

    put = client.put("/api/config", json={"content": "ariel:\n  vocabulary:\n    enabled: true\n"})
    assert put.status_code == 200
    assert put.json() == {"status": "ok", "requires_restart": True}
    assert "enabled: true" in config_file.read_text()


def test_config_path_falls_back_to_the_candidate_list(tmp_path, monkeypatch):
    """An app that never set config_path keeps today's CONFIG_FILE search."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("ariel: {}\n")
    monkeypatch.setenv("CONFIG_FILE", str(config_file))
    app = _make_app(ariel_service=None)

    response = TestClient(app).get("/api/config")

    assert response.status_code == 200
    assert response.json()["raw"] == "ariel: {}\n"


def test_require_service_detail_lists_config_errors_when_both_are_broken():
    """Database down AND a broken config: the 503 names both."""
    app = _make_app(
        ariel_service=None,
        config_status="configuration_invalid",
        config_errors=[VOCABULARY_ERROR],
        config_remedy=VocabularyError.remedy,
    )

    response = TestClient(app).get("/api/entries")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Database unavailable" in detail
    assert VOCABULARY_ERROR in detail
