"""Tests for the Logbook Entry Composer API.

Covers:
  - Validation: neither artifact_id nor context_id → 422
  - Artifact not found → 404
  - Context not found (int ID) → 404
  - Missing provider config → 503
  - Successful compose with artifact (mocked LLM)
  - Successful compose with context (mocked LLM)
  - Submit creates draft JSON in workspace/drafts/
  - Submit response includes ARIEL URL with draft_id
  - Submit calls notify_panel_focus
  - Prompt assembly: all Purpose × Detail combinations
  - Compose with steering fields (purpose/detail_level/nudge)
  - Compose with custom_prompt
  - Compose backward compatibility (no steering fields)
  - Assemble-prompt endpoint
  - Model tier routing (haiku/sonnet/opus)
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
    return json.dumps(
        {
            "subject": subject,
            "details": details,
            "tags": tags,
        }
    )


# Default provider config returned by get_provider_config("cborg")
_MOCK_PROVIDER_CONFIG = {
    "api_key": "test-key",
    "base_url": "https://api.cborg.lbl.gov/v1",
    "models": {
        "haiku": "anthropic/claude-haiku",
        "sonnet": "anthropic/claude-sonnet",
        "opus": "anthropic/claude-opus",
    },
}

# Default logbook.composition config returned by get_config_value
_MOCK_COMPOSITION_CONFIG = {
    "provider": "cborg",
    "model_id": "anthropic/claude-haiku",
    "default_tier": "haiku",
}


def _patch_model_resolution():
    """Context manager that patches provider/config resolution for compose tests."""
    return (
        patch("osprey.utils.config.get_config_value", return_value=_MOCK_COMPOSITION_CONFIG),
        patch("osprey.utils.config.get_provider_config", return_value=_MOCK_PROVIDER_CONFIG),
    )


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
        p1, p2 = _patch_model_resolution()
        with p1, p2:
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": "nonexistent"},
            )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_context_not_found(self, app_client):
        """404 for nonexistent context_id (int)."""
        p1, p2 = _patch_model_resolution()
        with p1, p2:
            resp = app_client.post(
                "/api/logbook/compose",
                json={"context_id": 9999},
            )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_no_provider_config(self, app_client):
        """503 with clear message when provider not found."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        with (
            patch("osprey.utils.config.get_config_value", return_value=_MOCK_COMPOSITION_CONFIG),
            patch("osprey.utils.config.get_provider_config", return_value={}),
        ):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )
        assert resp.status_code == 503
        assert "provider" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_compose_success_artifact(self, app_client):
        """Mock aget_chat_completion, verify subject/details/tags returned."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Beam Current Plot")

        mock_llm = AsyncMock(
            return_value=_llm_json_response(
                subject="Beam current analysis",
                details="Observed stable beam current at 500 mA.",
                tags=["beam", "current", "analysis"],
            )
        )

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
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

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
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

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.json()}"
        data = resp.json()
        assert data["subject"] == "Test"


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

        # Verify draft file exists (exclude the -metadata.json companion)
        drafts_dir = tmp_path / "drafts"
        draft_files = [
            f for f in drafts_dir.glob("draft-*.json") if not f.name.endswith("-metadata.json")
        ]
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

    @pytest.mark.unit
    def test_submit_creates_metadata_json_attachment(self, app_client, tmp_path):
        """Submit creates a metadata.json file and includes it in attachment_paths."""
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
        draft_id = data["draft_id"]

        # Verify metadata.json file exists
        meta_file = tmp_path / "drafts" / f"{draft_id}-metadata.json"
        assert meta_file.exists(), "metadata.json attachment file not created"

        # Verify its content has session_metadata
        meta_content = json.loads(meta_file.read_text())
        assert "session_metadata" in meta_content

        # Verify it's listed in the draft's attachment_paths
        draft_file = tmp_path / "drafts" / f"{draft_id}.json"
        draft_data = json.loads(draft_file.read_text())
        assert any(
            path.endswith(f"{draft_id}-metadata.json") for path in draft_data["attachment_paths"]
        ), "metadata.json not in attachment_paths"


class TestAssemblePrompt:
    """Tests for assemble_prompt() and POST /api/logbook/assemble-prompt."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "purpose",
        ["observation", "action_taken", "anomaly", "investigation", "routine_check", "general"],
    )
    @pytest.mark.parametrize("detail_level", ["brief", "standard", "detailed"])
    def test_assemble_prompt_all_combos(self, purpose, detail_level):
        """Every Purpose × Detail combination produces a valid prompt."""
        from osprey.interfaces.artifacts.logbook import (
            BASE_PREAMBLE,
            DETAIL_FRAGMENTS,
            JSON_FORMAT_INSTRUCTIONS,
            PURPOSE_FRAGMENTS,
            assemble_prompt,
        )

        result = assemble_prompt(purpose=purpose, detail_level=detail_level)

        assert BASE_PREAMBLE in result
        assert PURPOSE_FRAGMENTS[purpose] in result
        assert DETAIL_FRAGMENTS[detail_level] in result
        assert JSON_FORMAT_INSTRUCTIONS in result
        # No nudge → no "Additional operator guidance" line
        assert "Additional operator guidance" not in result

    @pytest.mark.unit
    def test_assemble_prompt_with_nudge(self):
        """Nudge text appears in assembled prompt."""
        from osprey.interfaces.artifacts.logbook import assemble_prompt

        result = assemble_prompt(nudge="Focus on SR current readings")
        assert "Additional operator guidance: Focus on SR current readings" in result

    @pytest.mark.unit
    def test_assemble_prompt_empty_nudge_ignored(self):
        """Whitespace-only nudge is ignored."""
        from osprey.interfaces.artifacts.logbook import assemble_prompt

        result = assemble_prompt(nudge="   ")
        assert "Additional operator guidance" not in result

    @pytest.mark.unit
    def test_assemble_prompt_unknown_purpose_falls_back(self):
        """Unknown purpose falls back to 'general'."""
        from osprey.interfaces.artifacts.logbook import PURPOSE_FRAGMENTS, assemble_prompt

        result = assemble_prompt(purpose="nonexistent")
        assert PURPOSE_FRAGMENTS["general"] in result

    @pytest.mark.unit
    def test_assemble_prompt_unknown_detail_falls_back(self):
        """Unknown detail_level falls back to 'standard'."""
        from osprey.interfaces.artifacts.logbook import DETAIL_FRAGMENTS, assemble_prompt

        result = assemble_prompt(detail_level="nonexistent")
        assert DETAIL_FRAGMENTS["standard"] in result

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_assemble_endpoint(self, app_client):
        """POST /api/logbook/assemble-prompt returns assembled prompt."""
        resp = app_client.post(
            "/api/logbook/assemble-prompt",
            json={"purpose": "anomaly", "detail_level": "brief"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data
        assert "anomaly" in data["prompt"].lower()
        assert "1-2 sentences" in data["prompt"]

    @pytest.mark.unit
    def test_assemble_endpoint_defaults(self, app_client):
        """POST /api/logbook/assemble-prompt with empty body uses defaults."""
        resp = app_client.post("/api/logbook/assemble-prompt", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data
        # Default purpose=general, detail_level=standard
        assert "factual logbook entry" in data["prompt"].lower()
        assert "1 short paragraph" in data["prompt"]

    @pytest.mark.unit
    def test_assemble_endpoint_with_nudge(self, app_client):
        """POST /api/logbook/assemble-prompt includes nudge in prompt."""
        resp = app_client.post(
            "/api/logbook/assemble-prompt",
            json={"nudge": "Mention the orbit feedback status"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Mention the orbit feedback status" in data["prompt"]


class TestComposeWithSteering:
    """Tests for compose endpoint with steering/custom_prompt fields."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    @pytest.mark.unit
    def test_compose_backward_compat(self, app_client):
        """POST with only artifact_id (no steering) still uses legacy prompt."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        # Verify the legacy SYSTEM_PROMPT was used (contains "Write concise")
        call_args = mock_llm.call_args
        system_msg = call_args.kwargs["chat_request"].messages[0].content
        assert "Write concise" in system_msg

    @pytest.mark.unit
    def test_compose_with_steering(self, app_client):
        """POST with purpose/detail_level uses assembled prompt."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={
                    "artifact_id": entry.id,
                    "purpose": "anomaly",
                    "detail_level": "detailed",
                    "nudge": "Check BPM readings",
                },
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        system_msg = call_args.kwargs["chat_request"].messages[0].content
        assert "anomaly" in system_msg.lower()
        assert "2-3 paragraphs" in system_msg
        assert "Check BPM readings" in system_msg

    @pytest.mark.unit
    def test_compose_with_custom_prompt(self, app_client):
        """POST with custom_prompt uses it directly as system prompt."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())
        custom = "You are a custom prompt. Respond with JSON."

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id, "custom_prompt": custom},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        system_msg = call_args.kwargs["chat_request"].messages[0].content
        assert system_msg == custom

    @pytest.mark.unit
    def test_compose_custom_prompt_overrides_steering(self, app_client):
        """custom_prompt takes precedence over purpose/detail_level."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())
        custom = "Custom override prompt."

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={
                    "artifact_id": entry.id,
                    "purpose": "anomaly",
                    "detail_level": "brief",
                    "custom_prompt": custom,
                },
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        system_msg = call_args.kwargs["chat_request"].messages[0].content
        assert system_msg == custom

    @pytest.mark.unit
    def test_compose_session_log_disabled(self, app_client):
        """include_session_log=False omits audit trail from user prompt."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with (
            p1,
            p2,
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
            patch(
                "osprey.mcp_server.workspace.transcript_reader.TranscriptReader"
            ) as mock_reader_cls,
        ):
            mock_reader_cls.return_value.read_current_session.return_value = [
                {"tool": "channel_read", "timestamp": "2026-02-22T10:00:00"},
            ]
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id, "include_session_log": False},
            )

        assert resp.status_code == 200
        # TranscriptReader should NOT have been instantiated
        mock_reader_cls.assert_not_called()

    @pytest.mark.unit
    def test_compose_multiple_artifacts(self, app_client):
        """artifact_ids sends multiple artifacts to the LLM context."""
        store = app_client.app.state.artifact_store
        e1 = _make_artifact(store, title="Plot A")
        e2 = _make_artifact(store, title="Plot B", artifact_type="csv")

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_ids": [e1.id, e2.id]},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        user_msg = call_args.kwargs["chat_request"].messages[1].content
        assert "Plot A" in user_msg
        assert "Plot B" in user_msg
        data = resp.json()
        assert e1.id in data["artifact_ids"]
        assert e2.id in data["artifact_ids"]

    @pytest.mark.unit
    def test_compose_all_artifacts(self, app_client):
        """artifact_ids=["all"] loads every artifact from the store."""
        store = app_client.app.state.artifact_store
        _make_artifact(store, title="Alpha")
        _make_artifact(store, title="Beta")

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_ids": ["all"]},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        user_msg = call_args.kwargs["chat_request"].messages[1].content
        assert "Alpha" in user_msg
        assert "Beta" in user_msg

    @pytest.mark.unit
    def test_compose_model_tier_routing(self, app_client):
        """model="sonnet" resolves to provider's sonnet model_id."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id, "model": "sonnet"},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        assert call_args.kwargs["provider"] == "cborg"
        assert call_args.kwargs["model_id"] == "anthropic/claude-sonnet"

    @pytest.mark.unit
    def test_compose_model_opus_routing(self, app_client):
        """model="opus" resolves to provider's opus model_id."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id, "model": "opus"},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        assert call_args.kwargs["model_id"] == "anthropic/claude-opus"

    @pytest.mark.unit
    def test_compose_default_tier_from_config(self, app_client):
        """No model= uses default_tier from logbook.composition config."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        call_args = mock_llm.call_args
        # Default tier is "haiku" per _MOCK_COMPOSITION_CONFIG
        assert call_args.kwargs["model_id"] == "anthropic/claude-haiku"


class TestUserPromptContent:
    """Verify the exact content the LLM sees in the user message.

    Each test calls ``/api/logbook/compose`` with a mocked LLM, then inspects
    ``mock_llm.call_args.kwargs["chat_request"].messages[1].content`` to
    assert on the user prompt that is actually sent.
    """

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def _get_user_prompt(self, mock_llm) -> str:
        """Extract the user message content from the mocked LLM call."""
        return mock_llm.call_args.kwargs["chat_request"].messages[1].content

    @pytest.mark.unit
    def test_prompt_includes_artifact_summary(self, app_client):
        """Artifact summary dict appears in the user prompt when populated."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Beam Data")
        entry.summary = {"mean_current": 500.2, "std": 1.3}

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "Summary:" in user_msg
        assert "500.2" in user_msg
        assert "mean_current" in user_msg

    @pytest.mark.unit
    def test_prompt_includes_artifact_category(self, app_client):
        """Artifact category appears in the user prompt when set."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="SR Data Export")
        entry.category = "data"

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "Category: data" in user_msg

    @pytest.mark.unit
    def test_prompt_no_summary_when_empty(self, app_client):
        """Empty summary dict does not produce a Summary line (avoids noise)."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store, title="Simple Plot")
        assert entry.summary == {}

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with p1, p2, patch("osprey.models.completion.aget_chat_completion", mock_llm):
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "Summary:" not in user_msg

    @pytest.mark.unit
    def test_prompt_includes_chat_history(self, app_client):
        """Chat history between user and assistant appears in the user prompt."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())
        mock_chat = [
            {
                "role": "user",
                "content": "What is the beam current?",
                "timestamp": "2026-02-22T10:00:00",
            },
            {
                "role": "assistant",
                "content": "The beam current is 500 mA.",
                "timestamp": "2026-02-22T10:00:05",
            },
        ]

        p1, p2 = _patch_model_resolution()
        with (
            p1,
            p2,
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
            patch(
                "osprey.mcp_server.workspace.transcript_reader.TranscriptReader"
            ) as mock_reader_cls,
        ):
            mock_reader_cls.return_value.read_current_session.return_value = []
            mock_reader_cls.return_value.read_current_chat_history.return_value = mock_chat
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "## Conversation log" in user_msg
        assert "USER: What is the beam current?" in user_msg
        assert "ASSISTANT: The beam current is 500 mA." in user_msg

    @pytest.mark.unit
    def test_prompt_includes_tool_arguments_and_results(self, app_client):
        """Audit trail entries include tool arguments and result summaries."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())
        mock_events = [
            {
                "type": "tool_call",
                "tool": "channel_read",
                "timestamp": "2026-02-22T10:00:00",
                "arguments": {"channels": ["SR:CURRENT"]},
                "result_summary": '{"status": "success", "value": 500.0}',
            },
        ]

        p1, p2 = _patch_model_resolution()
        with (
            p1,
            p2,
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
            patch(
                "osprey.mcp_server.workspace.transcript_reader.TranscriptReader"
            ) as mock_reader_cls,
        ):
            mock_reader_cls.return_value.read_current_session.return_value = mock_events
            mock_reader_cls.return_value.read_current_chat_history.return_value = []
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "## Recent session activity" in user_msg
        assert "channel_read" in user_msg
        assert "SR:CURRENT" in user_msg
        assert "500.0" in user_msg

    @pytest.mark.unit
    def test_prompt_chat_history_excluded_when_session_log_disabled(self, app_client):
        """include_session_log=False omits both chat history and audit trail."""
        store = app_client.app.state.artifact_store
        entry = _make_artifact(store)

        mock_llm = AsyncMock(return_value=_llm_json_response())

        p1, p2 = _patch_model_resolution()
        with (
            p1,
            p2,
            patch("osprey.models.completion.aget_chat_completion", mock_llm),
            patch(
                "osprey.mcp_server.workspace.transcript_reader.TranscriptReader"
            ) as mock_reader_cls,
        ):
            mock_reader_cls.return_value.read_current_session.return_value = [
                {"tool": "channel_read", "timestamp": "2026-02-22T10:00:00"},
            ]
            mock_reader_cls.return_value.read_current_chat_history.return_value = [
                {"role": "user", "content": "hello", "timestamp": "2026-02-22T10:00:00"},
            ]
            resp = app_client.post(
                "/api/logbook/compose",
                json={"artifact_id": entry.id, "include_session_log": False},
            )

        assert resp.status_code == 200
        user_msg = self._get_user_prompt(mock_llm)
        assert "Conversation log" not in user_msg
        assert "Recent session activity" not in user_msg
        mock_reader_cls.assert_not_called()
