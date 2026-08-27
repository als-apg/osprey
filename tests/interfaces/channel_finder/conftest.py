"""Fixtures for Channel Finder web interface tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

#: The config seam these fixtures and their tests BOTH repoint. It is patched
#: through ``monkeypatch`` on purpose: a test body that repoints it again (see
#: ``_artifact_store_dir`` in test_pending_review_api.py) must stack on the same
#: mechanism. Mixing ``mock.patch`` here with ``monkeypatch`` there makes the net
#: effect depend on which was entered first — unwound in the wrong order, the
#: fixture's mock is restored *after* the real function and leaks into every
#: later test in the process.
_CONFIG_SEAM = "osprey.utils.workspace.load_osprey_config"


def _patch_config(monkeypatch, mock_config):
    """Point the shared config seam at *mock_config* for the whole test."""
    monkeypatch.setattr(_CONFIG_SEAM, lambda: mock_config)


@pytest.fixture()
def mock_config():
    """Return a minimal channel finder config dict."""
    return {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {
                "in_context": {
                    "database": {"path": "/tmp/test_db.json", "type": "flat"},
                },
            },
        },
    }


@pytest.fixture()
def mock_registry():
    """Mock the in-context registry initialization."""
    mock_reg = MagicMock()
    mock_reg.database = MagicMock()
    mock_reg.facility_name = "TEST"

    with patch(
        "osprey.mcp_server.channel_finder_in_context.server_context.initialize_cf_ic_context",
        return_value=mock_reg,
    ) as init_mock:
        yield init_mock


@pytest.fixture()
def app(mock_config, mock_registry, monkeypatch):
    """Create a test Channel Finder FastAPI app with mocked dependencies."""
    _patch_config(monkeypatch, mock_config)
    from osprey.interfaces.channel_finder.app import create_app

    application = create_app(project_cwd="/tmp/test-project")
    yield application


@pytest.fixture()
def client(app):
    """Create a TestClient for the channel finder app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def feedback_client(mock_config, mock_registry, tmp_path, monkeypatch):
    """Create a TestClient with a real FeedbackStore on tmp_path."""
    from osprey.services.channel_finder.feedback.store import FeedbackStore

    _patch_config(monkeypatch, mock_config)
    from osprey.interfaces.channel_finder.app import create_app

    application = create_app(project_cwd="/tmp/test-project")
    with TestClient(application) as c:
        # Set after lifespan runs (lifespan sets feedback_store=None)
        application.state.feedback_store = FeedbackStore(tmp_path / "feedback.json")
        yield c


@pytest.fixture()
def pending_review_client(mock_config, mock_registry, tmp_path, monkeypatch):
    """Create a TestClient with real PendingReviewStore + FeedbackStore."""
    from osprey.services.channel_finder.feedback.pending_store import PendingReviewStore
    from osprey.services.channel_finder.feedback.store import FeedbackStore

    _patch_config(monkeypatch, mock_config)
    from osprey.interfaces.channel_finder.app import create_app

    application = create_app(project_cwd="/tmp/test-project")
    with TestClient(application) as c:
        # Set after lifespan runs (lifespan sets stores=None)
        application.state.pending_review_store = PendingReviewStore(tmp_path / "pending.json")
        application.state.feedback_store = FeedbackStore(tmp_path / "feedback.json")
        yield c
