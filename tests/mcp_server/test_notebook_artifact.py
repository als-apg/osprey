"""Tests for notebook artifact integration with the execute tool.

Covers:
  - execute tool auto-creates notebook artifact
  - Notebook contains executed code
  - Response includes notebook_artifact_id
  - Gallery render endpoint returns HTML
"""

import json
from unittest.mock import patch

import pytest

from tests.mcp_server.conftest import get_tool_fn


def _get_python_execute():
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return get_tool_fn(execute)


@pytest.mark.unit
async def test_python_execute_creates_notebook_artifact(tmp_path, monkeypatch):
    """The execute tool creates a notebook artifact alongside normal output."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "osprey.services.python_executor.analysis.pattern_detection"
        ".detect_control_system_operations",
        return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print(6 * 7)",
            description="notebook artifact test",
            execution_mode="readonly",
        )

    data = json.loads(result)
    assert data["status"] == "success"

    # Should have a notebook_artifact_id in the response
    assert "notebook_artifact_id" in data
    nb_id = data["notebook_artifact_id"]
    assert isinstance(nb_id, str)
    assert len(nb_id) == 12  # ArtifactStore uses 12-char hex IDs

    # Artifact should also appear in artifact_ids
    assert nb_id in data.get("artifact_ids", [])


@pytest.mark.unit
async def test_notebook_artifact_contains_code(tmp_path, monkeypatch):
    """The notebook artifact file contains the executed code."""
    monkeypatch.chdir(tmp_path)

    with patch(
        "osprey.services.python_executor.analysis.pattern_detection"
        ".detect_control_system_operations",
        return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('unique_marker_xyz')",
            description="code in notebook test",
            execution_mode="readonly",
        )

    data = json.loads(result)
    nb_id = data["notebook_artifact_id"]

    # Read the notebook from disk
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    nb_path = store.get_file_path(nb_id)
    assert nb_path is not None
    assert nb_path.exists()

    import nbformat

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Code cell should contain our code
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 1
    assert "unique_marker_xyz" in code_cells[0].source


@pytest.mark.unit
async def test_notebook_failure_does_not_break_execution(tmp_path, monkeypatch):
    """If notebook creation fails, the execute tool still succeeds."""
    monkeypatch.chdir(tmp_path)

    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.mcp_server.notebook_renderer.create_notebook_from_code",
            side_effect=RuntimeError("Notebook creation failed"),
        ),
    ):
        fn = _get_python_execute()
        result = await fn(
            code="print('still works')",
            description="resilience test",
            execution_mode="readonly",
        )

    data = json.loads(result)
    assert data["status"] == "success"
    # No notebook_artifact_id when creation fails
    assert data.get("notebook_artifact_id") is None


@pytest.mark.unit
def test_gallery_notebook_render_endpoint(tmp_path):
    """The gallery /api/notebooks/{id}/rendered endpoint returns HTML."""
    import nbformat

    from osprey.mcp_server.artifact_store import ArtifactStore
    from osprey.mcp_server.notebook_renderer import create_notebook_from_code

    store = ArtifactStore(workspace_root=tmp_path)
    nb = create_notebook_from_code(
        code="GALLERY_RENDER_MARKER_XYZ", description="Gallery test"
    )
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
