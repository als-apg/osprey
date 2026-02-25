"""Tests for gather_session_metadata()."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    """Set up a fake project directory with workspace and transcript."""
    # Create workspace structure so resolve_workspace_root().parent works
    workspace = tmp_path / "osprey-workspace"
    workspace.mkdir()

    # Patch resolve_workspace_root to return our fake workspace
    monkeypatch.setattr(
        "osprey.mcp_server.common.resolve_workspace_root",
        lambda: workspace,
    )

    return tmp_path


@pytest.fixture()
def fake_transcript(fake_project):
    """Create a fake JSONL transcript in the Claude transcript directory."""
    # Create a .claude/projects/<encoded>/  transcript directory
    claude_dir = fake_project / ".claude" / "projects" / "test-project"
    claude_dir.mkdir(parents=True)

    transcript_file = claude_dir / "session-abc123.jsonl"
    entries = [
        {"sessionId": "sess-42", "timestamp": "2026-02-22T10:00:00Z", "type": "init"},
        {"sessionId": "sess-42", "timestamp": "2026-02-22T10:01:00Z", "type": "tool_use"},
    ]
    transcript_file.write_text("\n".join(json.dumps(e) for e in entries))

    return transcript_file


@pytest.fixture()
def fake_settings(fake_project):
    """Create a fake .claude/settings.json."""
    settings_dir = fake_project / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"
    settings_path.write_text(json.dumps({"model": "claude-sonnet-4-6"}))
    return settings_path


def _make_subprocess_raise(*args, **kwargs):
    raise OSError("git not available")


class TestGatherSessionMetadata:
    """Tests for gather_session_metadata()."""

    def test_all_fields_present(self, fake_project):
        """Return dict always has all 8 keys, even when values are None."""
        from osprey.mcp_server.common import gather_session_metadata

        result = gather_session_metadata("test-caller")

        expected_keys = {
            "session_id",
            "transcript_path",
            "session_start_time",
            "git_branch",
            "git_commit_short",
            "operator",
            "model_name",
            "created_via",
        }
        assert set(result.keys()) == expected_keys

    def test_created_via_always_matches(self, fake_project):
        """created_via matches the argument regardless of other failures."""
        from osprey.mcp_server.common import gather_session_metadata

        with patch("subprocess.run", side_effect=_make_subprocess_raise):
            result = gather_session_metadata("my-caller-id")

        assert result["created_via"] == "my-caller-id"

    def test_git_fallback_when_unavailable(self, fake_project):
        """git_branch and git_commit_short are None when git fails."""
        from osprey.mcp_server.common import gather_session_metadata

        with patch("subprocess.run", side_effect=_make_subprocess_raise):
            result = gather_session_metadata("test")

        assert result["git_branch"] is None
        assert result["git_commit_short"] is None

    def test_transcript_fields_when_no_transcript(self, fake_project):
        """Transcript fields are None when no transcript directory exists."""
        from osprey.mcp_server.common import gather_session_metadata

        # Patch TranscriptReader to return no transcript
        with patch(
            "osprey.mcp_server.workspace.transcript_reader.TranscriptReader",
            **{"return_value.find_current_transcript.return_value": None},
        ):
            result = gather_session_metadata("test")

        assert result["session_id"] is None
        assert result["transcript_path"] is None
        assert result["session_start_time"] is None

    def test_transcript_fields_populated(self, fake_project, fake_transcript):
        """Transcript fields are populated when a JSONL transcript exists."""
        from osprey.mcp_server.common import gather_session_metadata

        # Patch TranscriptReader to return our fake transcript
        with patch(
            "osprey.mcp_server.workspace.transcript_reader.TranscriptReader",
            **{"return_value.find_current_transcript.return_value": fake_transcript},
        ):
            result = gather_session_metadata("test")

        assert result["session_id"] == "sess-42"
        assert result["transcript_path"] == str(fake_transcript)
        assert result["session_start_time"] == "2026-02-22T10:00:00Z"

    def test_session_id_env_fallback(self, fake_project, monkeypatch):
        """session_id falls back to OSPREY_SESSION_ID env var."""
        from osprey.mcp_server.common import gather_session_metadata

        monkeypatch.setenv("OSPREY_SESSION_ID", "env-session-99")

        # Patch TranscriptReader to return no transcript
        with patch(
            "osprey.mcp_server.workspace.transcript_reader.TranscriptReader",
            **{"return_value.find_current_transcript.return_value": None},
        ):
            result = gather_session_metadata("test")

        assert result["session_id"] == "env-session-99"

    def test_settings_json_missing(self, fake_project, monkeypatch):
        """model_name is None when settings.json doesn't exist and env vars unset."""
        from osprey.mcp_server.common import gather_session_metadata

        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        monkeypatch.delenv("CLAUDE_MODEL", raising=False)

        result = gather_session_metadata("test")

        assert result["model_name"] is None

    def test_model_name_from_settings(self, fake_project, fake_settings):
        """model_name comes from .claude/settings.json when present."""
        from osprey.mcp_server.common import gather_session_metadata

        result = gather_session_metadata("test")

        assert result["model_name"] == "claude-sonnet-4-6"

    def test_model_name_env_fallback(self, fake_project, monkeypatch):
        """model_name falls back to ANTHROPIC_MODEL env var."""
        from osprey.mcp_server.common import gather_session_metadata

        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

        result = gather_session_metadata("test")

        # settings.json doesn't exist, so env var should be used
        assert result["model_name"] == "claude-haiku-4-5"

    def test_operator_from_user_env(self, fake_project, monkeypatch):
        """operator comes from USER env var."""
        from osprey.mcp_server.common import gather_session_metadata

        monkeypatch.setenv("USER", "test-operator")

        result = gather_session_metadata("test")

        assert result["operator"] == "test-operator"
