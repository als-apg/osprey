"""MCP tool: submit_response — persist an agent's final synthesized result.

Sub-agents call this as their last action to save their synthesis to
the artifact gallery so the parent session, other tools, and the gallery
UI can all reference it.
"""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.submit_response")

# Map well-known agent names to categories
_AGENT_CATEGORY_MAP: dict[str, str] = {
    "channel-finder": "channel_finder",
    "logbook-search": "logbook_research",
    "logbook-deep-research": "logbook_research",
}


@mcp.tool()
async def submit_response(
    title: str,
    content: str,
    data_type: str = "agent_response",
    entry_ids: list[str] | None = None,
    source_agent: str | None = None,
    skip_artifact: bool = False,
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
        skip_artifact: If True, skip creating a new artifact (use when
            the agent already created plot/dashboard artifacts and wants
            to avoid double-registration).

    Returns:
        JSON with artifact_id, gallery_url, and summary.
    """
    if not title or not title.strip():
        return make_error(
                "validation_error",
                "title is required and must not be empty.",
                ["Provide a short descriptive title for your response."],
            )

    if not content or not content.strip():
        return make_error(
                "validation_error",
                "content is required and must not be empty.",
                ["Provide the full synthesized response text."],
            )

    from osprey.stores.type_registry import valid_category_keys

    valid = valid_category_keys()
    if data_type not in valid:
        return make_error(
                "validation_error",
                f"Unknown data_type '{data_type}'. Valid: {sorted(valid)}",
                ["Use one of the registered data_type or category values."],
            )

    try:
        from osprey.stores.artifact_store import get_artifact_store

        cited = entry_ids or []
        agent = source_agent or ""

        if skip_artifact:
            return json.dumps(
                {
                    "status": "success",
                    "skipped_artifact": True,
                    "title": title,
                    "source_agent": agent,
                    "note": "Artifact creation skipped (skip_artifact=True).",
                },
                default=str,
            )

        # Determine category from agent name or data_type
        category = _AGENT_CATEGORY_MAP.get(agent, data_type)

        store = get_artifact_store()
        tool_name = agent if agent else "submit_response"
        artifact = store.save_file(
            file_content=content.encode(),
            filename=f"{tool_name}.md",
            artifact_type="markdown",
            title=title,
            description=f"{category} from {agent}" if agent else category,
            mime_type="text/markdown",
            tool_source="submit_response",
            metadata={
                "data_type": data_type,
                "source_agent": agent,
                "entry_ids": cited,
            },
        )
        # Set unified fields on the entry
        artifact = store.update_entry_metadata(
            artifact.id,
            category=category,
            source_agent=agent,
            summary={
                "title": title,
                "content_length": len(content),
                "cited_entries": len(cited),
                "source_agent": agent,
            },
        )

        response = artifact.to_tool_response()
        response["gallery_url"] = gallery_url()
        return json.dumps(response, default=str)

    except Exception as exc:
        logger.exception("submit_response failed")
        return make_error(
                "internal_error",
                f"Failed to save response: {exc}",
                ["Check that the _agent_data directory is accessible."],
            )
