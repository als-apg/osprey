"""Tests for notebook artifact rendering in the gallery.

Covers:
  - Gallery render endpoint returns HTML
  - Gallery render returns 404 for missing artifacts
  - Gallery render returns 400 for non-notebook artifacts
"""

import pytest


@pytest.mark.unit
def test_gallery_notebook_render_endpoint(tmp_path):
    """The gallery /api/notebooks/{id}/rendered endpoint returns HTML."""
    import nbformat

    from osprey.mcp_server.artifact_store import ArtifactStore
    from osprey.mcp_server.notebook_renderer import create_notebook_from_code

    store = ArtifactStore(workspace_root=tmp_path)
    nb = create_notebook_from_code(code="GALLERY_RENDER_MARKER_XYZ", description="Gallery test")
    nb_bytes = nbformat.writes(nb).encode()

    entry = store.save_file(
        file_content=nb_bytes,
        filename="gallery_test.ipynb",
        artifact_type="notebook",
        title="Gallery Notebook",
        description="Test notebook for gallery",
        mime_type="application/x-ipynb+json",
        tool_source="test",
    )

    from fastapi.testclient import TestClient

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)

    response = client.get(f"/api/notebooks/{entry.id}/rendered")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GALLERY_RENDER_MARKER_XYZ" in response.text


@pytest.mark.unit
def test_gallery_notebook_render_404_for_missing(tmp_path):
    """Render endpoint returns 404 for non-existent artifact."""
    from fastapi.testclient import TestClient

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)

    response = client.get("/api/notebooks/nonexistent123/rendered")
    assert response.status_code == 404


@pytest.mark.unit
def test_gallery_notebook_render_400_for_non_notebook(tmp_path):
    """Render endpoint returns 400 for non-notebook artifact types."""
    from osprey.mcp_server.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=tmp_path)
    entry = store.save_file(
        file_content=b"<h1>Not a notebook</h1>",
        filename="not_notebook.html",
        artifact_type="html",
        title="HTML Artifact",
        mime_type="text/html",
        tool_source="test",
    )

    from fastapi.testclient import TestClient

    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)

    response = client.get(f"/api/notebooks/{entry.id}/rendered")
    assert response.status_code == 400
