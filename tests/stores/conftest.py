"""Fixtures shared by the artifact-store suites."""

from __future__ import annotations

import pytest

from osprey.stores.artifact_store import ArtifactStore


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """A workspace root under a real repo root, so ``data_file`` anchors resolve."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return repo_root / "state" / "agent"


@pytest.fixture
def store(agent_data_root) -> ArtifactStore:
    """An ArtifactStore with a real repo root, so ``data_file`` anchors resolve."""
    return ArtifactStore(workspace_root=agent_data_root)
