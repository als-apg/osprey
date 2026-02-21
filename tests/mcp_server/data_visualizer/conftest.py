"""Shared fixtures for data-visualizer tool tests."""

from pathlib import Path

import pytest

from osprey.mcp_server.artifact_store import initialize_artifact_store
from osprey.mcp_server.data_context import initialize_data_context


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path):
    """Set up workspace with artifact store and data context."""
    ws = tmp_path / "osprey-workspace"
    ws.mkdir()
    (ws / "data").mkdir()
    (ws / "artifacts").mkdir()
    initialize_data_context(workspace_root=ws)
    initialize_artifact_store(workspace_root=ws)
