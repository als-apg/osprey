"""Fixtures shared by the Bluesky MCP tool suites."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def silence_queue_activity_emit():
    """No queue tool test posts agent activity; a test asserting the emit patches it itself."""
    with patch("osprey.mcp_server.bluesky.tools.queue.notify_agent_activity_async"):
        yield
