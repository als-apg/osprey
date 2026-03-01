"""Shared fixtures for ARIEL MCP server tool tests.

Provides mock ARIEL service, entry factories, and registry management.

IMPORTANT: FastMCP's @mcp.tool() decorator wraps functions into FunctionTool
objects. To call the original async function in tests, use the `.fn` attribute:
    tool.fn(query="beam loss")
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.interfaces.ariel.mcp.registry import reset_ariel_registry
from osprey.utils.workspace import reset_config_cache
from tests.mcp_server.conftest import get_tool_fn  # noqa: F401


@pytest.fixture(autouse=True)
def _reset_registry(tmp_path):
    """Reset the ARIEL MCP registry singletons between tests."""
    yield
    reset_ariel_registry()
    reset_config_cache()


def make_mock_entry(
    entry_id="test-001",
    source_system="ALS eLog",
    author="Test User",
    raw_text="Test entry content",
    summary=None,
    timestamp=None,
    logbook=None,
    shift=None,
    tags=None,
    score=None,
    attachments=None,
):
    """Create a mock EnhancedLogbookEntry dict (TypedDict, not MagicMock).

    Returns a plain dict matching the EnhancedLogbookEntry TypedDict shape.
    """
    now = timestamp or datetime(2024, 1, 15, 10, 30, 0)
    entry = {
        "entry_id": entry_id,
        "source_system": source_system,
        "timestamp": now,
        "author": author,
        "raw_text": raw_text,
        "attachments": attachments if attachments is not None else [],
        "metadata": {
            "logbook": logbook,
            "shift": shift,
            "tags": tags or [],
        },
        "created_at": now,
        "updated_at": now,
    }
    if summary is not None:
        entry["summary"] = summary
    if score is not None:
        entry["_score"] = score
    return entry


@pytest.fixture
def mock_ariel_service():
    """Create a mock ARIELSearchService with AsyncMock methods."""
    service = AsyncMock()
    service.repository = AsyncMock()
    service.config = MagicMock()
    service.pool = AsyncMock()
    return service
