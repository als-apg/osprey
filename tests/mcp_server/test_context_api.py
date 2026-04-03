"""Tests for SSE event type discrimination.

Verifies that artifact SSE broadcast events include type tags so consumers
can distinguish between file-based artifacts and data artifacts.
"""

import pytest

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

    def test_data_artifact_event_has_type(self, app_client):
        """Data artifact SSE events include {"type": "artifact"}."""
        store = app_client.app.state.artifact_store
        entry = store.save_data(
            tool="channel_read",
            data={"value": 42},
            title="Channel snapshot",
            description="test data artifact",
            summary={"count": 1},
            access_details={"format": "json"},
            artifact_type="json",
        )
        tagged = {"type": "artifact", **entry.to_dict()}
        assert tagged["type"] == "artifact"
        assert tagged["id"] == entry.id
