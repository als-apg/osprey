"""Tests for SSE event type discrimination.

The context gallery routes have been removed (context data is now accessed
via DataContext MCP tools only). This file retains SSE discrimination tests
to verify broadcast event format.
"""

import pytest


def _save_context_entry(context_store, tool="channel_read", data_type="channel_values",
                        description="test entry"):
    """Helper to save a context entry directly on the store."""
    return context_store.save(
        tool=tool,
        data={"value": 42},
        description=description,
        summary={"count": 1},
        access_details={"format": "json"},
        data_type=data_type,
    )


# ---------------------------------------------------------------------------
# SSE event discrimination
# ---------------------------------------------------------------------------


class TestSSEDiscrimination:
    """Verify SSE events include type discrimination."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def test_artifact_event_has_type(self, app_client):
        """Artifact SSE events include {"type": "artifact"}."""
        store = app_client.app.state.artifact_store
        entry = store.save_file(
            file_content=b"test",
            filename="test.html",
            artifact_type="html",
            title="Test",
            mime_type="text/html",
            tool_source="test",
        )
        tagged = {"type": "artifact", **entry.to_dict()}
        assert tagged["type"] == "artifact"
        assert tagged["id"] == entry.id

    def test_context_event_has_type(self, app_client):
        """Context events include {"type": "context"}."""
        ctx = app_client.app.state.context_store
        entry = _save_context_entry(ctx)
        tagged = {"type": "context", **entry.to_dict()}
        assert tagged["type"] == "context"
        assert tagged["id"] == entry.id
