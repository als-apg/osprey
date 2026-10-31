"""Fixtures shared by the artifact-gallery suites."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app


@pytest.fixture
def app_client(tmp_path: Path) -> TestClient:
    """A client for the gallery app over a workspace rooted at ``tmp_path``."""
    return TestClient(create_app(workspace_root=tmp_path))
