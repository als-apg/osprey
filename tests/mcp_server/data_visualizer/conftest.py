"""Shared fixtures for data-visualizer tool tests."""

import pytest

from osprey.stores.artifact_store import initialize_artifact_store


@pytest.fixture(autouse=True)
def setup_workspace(tmp_path):
    """Set up workspace with artifact store."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "artifacts").mkdir()
    initialize_artifact_store(workspace_root=ws)
