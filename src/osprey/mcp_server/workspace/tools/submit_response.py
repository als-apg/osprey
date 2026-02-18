"""MCP tool: submit_response — persist an agent's final synthesized result.

Sub-agents call this as their last action to save their synthesis to
DataContext so the parent session and other tools can reference it.
"""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.submit_response")


@mcp.tool()
async def submit_response(
    title: str,
    content: str,
    data_type: str = "agent_response",
    entry_ids: list[str] | None = None,
) -> str:
    """Submit your final synthesized response. Call this as your LAST action
    before responding. This persists your findings to the workspace so
    the parent session and other tools can reference them.

    Include all entry IDs, channel addresses, or other identifiers you
    cited in the entry_ids parameter for cross-referencing.

    Args:
        title: Short title for the response (e.g. "Vacuum Event Analysis").
        content: The full synthesized response text (markdown).
        data_type: Category tag for filtering (e.g. "agent_response",
            "channel_addresses", "logbook_research").
        entry_ids: List of ARIEL entry IDs or channel addresses cited,
            stored as structured metadata for cross-referencing.

    Returns:
        JSON with context_entry_id and data_file path.
    """
    if not title or not title.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "title is required and must not be empty.",
                ["Provide a short descriptive title for your response."],
            )
        )

    if not content or not content.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "content is required and must not be empty.",
                ["Provide the full synthesized response text."],
            )
        )

    try:
        from osprey.mcp_server.data_context import get_data_context

        cited = entry_ids or []

        ctx = get_data_context()
        entry = ctx.save(
            tool="submit_response",
            data={
                "title": title,
                "content": content,
                "entry_ids": cited,
                "data_type": data_type,
            },
            description=title,
            summary={
                "title": title,
                "content_length": len(content),
                "cited_entries": len(cited),
            },
            access_details={
                "format": "markdown",
                "data_type": data_type,
            },
            data_type=data_type,
        )
        return json.dumps(entry.to_tool_response(), default=str)

    except Exception as exc:
        logger.exception("submit_response failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to save response: {exc}",
                ["Check that the osprey-workspace directory is accessible."],
            )
        )
