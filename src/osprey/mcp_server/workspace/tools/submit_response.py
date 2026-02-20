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
    source_agent: str | None = None,
) -> str:
    """Submit your final synthesized response. Call this as your LAST action
    before responding. This persists your findings to the workspace so
    the parent session and other tools can reference them.

    Include all entry IDs, channel addresses, or other identifiers you
    cited in the entry_ids parameter for cross-referencing.

    Args:
        title: Short title for the response (e.g. "Vacuum Event Analysis").
        content: The full synthesized response text (markdown).
        data_type: Category tag for filtering.  Must be a registered type:
            "agent_response" (default), "channel_addresses",
            "logbook_research", "search_results", or any other key from
            the type registry.
        entry_ids: List of ARIEL entry IDs or channel addresses cited,
            stored as structured metadata for cross-referencing.
        source_agent: Name of the agent submitting the response
            (e.g. "logbook-search", "wiki-search"). Used for filtering
            and grouping results by agent.

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

    from osprey.mcp_server.type_registry import valid_data_type_keys

    valid = valid_data_type_keys()
    if data_type not in valid:
        return json.dumps(
            make_error(
                "validation_error",
                f"Unknown data_type '{data_type}'. Valid: {sorted(valid)}",
                ["Use one of the registered data_type values."],
            )
        )

    try:
        from osprey.mcp_server.data_context import get_data_context

        cited = entry_ids or []

        agent = source_agent or ""

        ctx = get_data_context()
        tool_name = agent if agent else "submit_response"
        entry = ctx.save(
            tool=tool_name,
            data={
                "title": title,
                "content": content,
                "entry_ids": cited,
                "data_type": data_type,
                "source_agent": agent,
            },
            description=title,
            summary={
                "title": title,
                "content_length": len(content),
                "cited_entries": len(cited),
                "source_agent": agent,
            },
            access_details={
                "format": "markdown",
                "data_type": data_type,
            },
            data_type=data_type,
            source_agent=agent,
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
