"""Tests for the Logbook Entry Composer API.

Covers:
  - Validation: neither artifact_id nor context_id → 422
  - Artifact not found → 404
  - Context not found (int ID) → 404
  - Missing model config → 503
  - Successful compose with artifact (mocked LLM)
  - Successful compose with context (mocked LLM)
  - Submit creates draft JSON in workspace/drafts/
  - Submit response includes ARIEL URL with draft_id
  - Submit calls notify_panel_focus
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

_MODULE = "osprey.interfaces.artifacts.logbook"


def _make_artifact(store, title="Test Plot", artifact_type="plot_html"):
    """Create a minimal artifact for testing."""
    return store.save_file(
        file_content=b"<html>test</html>",
        filename="test.html",
        artifact_type=artifact_type,
        title=title,
        description="A test artifact",
        mime_type="text/html",
        tool_source="execute",
    )


def _llm_json_response(subject="Test Subject", details="Test details.", tags=None):
    """Build a JSON string that aget_chat_completion would return."""
    if tags is None:
        tags = ["beam", "current"]
    return json.dumps({
        "subject": subject,
        "details": details,
        "tags": tags,
    })


def _mock_model_config():
    """Return a minimal model_config dict for logbook_composition."""
    return {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001", "max_tokens": 1024}


class TestLogbookCompose:
    """Tests for POST /api/logbook/compose."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_compose_requires_id(self, app_client):
        """422 when neither artifact_id nor context_id provided."""
        resp = app_client.post("/api/logbook/compose", json={})
        assert resp.status_code == 422
        assert "at least one" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_artifact_not_found(self, app_client):
        """404 for nonexistent artifact_id."""
        with patch("osprey.utils.config.get_model_config", return_value=_mock_model_config()):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": "nonexistent"},
            )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_context_not_found(self, app_client):
        """404 for nonexistent context_id (int)."""
        with patch("osprey.utils.config.get_model_config", return_value=_mock_model_config()):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"context_id": 9999},
            )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_no_model_config(self, app_client):
        """503 with clear message when no model config available."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        # get_model_config returns {} (not raises) when key is missing
        with patch("osprey.utils.config.get_model_config", return_value={}):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )
        assert resp.status_code == 503
        assert "model config" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_success_artifact(self, app_client):
        """Mock aget_chat_completion, verify subject/details/tags returned."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Beam Current Plot")

        mock_llm = AsyncMock(return_value=_llm_json_response(
            subject="Beam current analysis",
            details="Observed stable beam current at 500 mA.",
            tags=["beam", "current", "analysis"],
        ))

        with (
            patch("osprey.utils.config.get_model_config", return_value=_mock_model_config()),
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
        ):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["subject"] == "Beam current analysis"
        assert data["details"] == "Observed stable beam current at 500 mA."
        assert "beam" in data["tags"]
        assert entry.id in data["artifact_ids"]

    @pytest.mark.unit
    def test_compose_markdown_fenced_json(self, app_client):
        """LLM wrapping JSON in ```json fences must not cause a 503."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Beam Current Plot")

        # Simulate LLM returning JSON wrapped in markdown code fences
        fenced_response = '```json\n{"subject": "Beam analysis", "details": "Stable beam.", "tags": ["beam"]}\n```'
        mock_llm = AsyncMock(return_value=fenced_response)

        with (
            patch("osprey.utils.config.get_model_config", return_value=_mock_model_config()),
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
        ):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.json()}"
        data = resp.json()
        assert data["subject"] == "Beam analysis"

    @pytest.mark.unit
    def test_compose_json_with_preamble(self, app_client):
        """LLM adding preamble text before JSON must not cause a 503."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Test Plot")

        # Simulate LLM adding conversational preamble before JSON
        preamble_response = 'Here is the logbook entry:\n{"subject": "Test", "details": "Details.", "tags": ["test"]}'
        mock_llm = AsyncMock(return_value=preamble_response)

        with (
            patch("osprey.utils.config.get_model_config", return_value=_mock_model_config()),
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
        ):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.json()}"
        data = resp.json()
        assert data["subject"] == "Test"

    @pytest.mark.unit
    def test_compose_empty_model_config_fallback(self, app_client):
        """get_model_config returning {} (missing key) should be detected, not silently used."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        # Simulate get_model_config returning {} for both roles (real behavior)
        with patch("osprey.utils.config.get_model_config", return_value={}):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        # Should get a clear 503 about missing model config, not "Unknown provider: None"
        assert resp.status_code == 503
        detail = resp.json()["detail"].lower()
        assert "model config" in detail or "logbook_composition" in detail, (
            f"Expected clear model config error, got: {resp.json()['detail']}"
        )


class TestLogbookSubmit:
    """Tests for POST /api/logbook/submit."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_submit_creates_draft(self, app_client, tmp_path):
        """Draft JSON written to workspace/drafts/."""
        with (
            patch(f"{_MODULE}.resolve_workspace_root", return_value=tmp_path),
            patch(f"{_MODULE}.notify_panel_focus"),
        ):
            resp = app_client.post(
                "/api/logbook/submit",
                json={
                    "subject": "Test entry",
                    "details": "Details here.",
                    "tags": ["test"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"].startswith("draft-")

        # Verify draft file exists
        drafts_dir = tmp_path / "drafts"
        draft_files = list(drafts_dir.glob("draft-*.json"))
        assert len(draft_files) == 1

        draft_data = json.loads(draft_files[0].read_text())
        assert draft_data["subject"] == "Test entry"
        assert draft_data["details"] == "Details here."
        assert draft_data["tags"] == ["test"]

    @pytest.mark.unit
    def test_submit_returns_ariel_url(self, app_client, tmp_path):
        """Response includes ARIEL URL with draft_id."""
        with (
            patch(f"{_MODULE}.resolve_workspace_root", return_value=tmp_path),
            patch(f"{_MODULE}.notify_panel_focus"),
        ):
            resp = app_client.post(
                "/api/logbook/submit",
                json={"subject": "Test", "details": "Details."},
            )

        data = resp.json()
        assert "url" in data
        assert data["draft_id"] in data["url"]
        assert "/#create?draft=" in data["url"]

    @pytest.mark.unit
    def test_submit_calls_panel_focus(self, app_client, tmp_path):
        """Mock notify_panel_focus, verify called."""
        with (
            patch(f"{_MODULE}.resolve_workspace_root", return_value=tmp_path),
            patch(f"{_MODULE}.notify_panel_focus") as mock_focus,
        ):
            resp = app_client.post(
                "/api/logbook/submit",
                json={"subject": "Test", "details": "Details."},
            )

        assert resp.status_code == 200
        mock_focus.assert_called_once()
        call_args = mock_focus.call_args
        assert call_args[0][0] == "ariel"
        assert "/#create?draft=" in call_args[1]["url"]
