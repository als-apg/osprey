"""Fixtures for Hierarchical channel finder MCP tests."""

import pytest

from osprey.services.channel_finder.mcp.hierarchical.registry import reset_cf_hier_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons between tests."""
    yield
    reset_cf_hier_registry()


def get_tool_fn(tool_or_fn):
    """Extract raw function from FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn
