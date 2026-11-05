"""MCP tool: submit_response — persist an agent's final synthesized result.

Sub-agents call this as their last action to save their synthesis to
the artifact gallery so the parent session, other tools, and the gallery
UI can all reference it.

One agent, one call, one artifact: the answer as prose. The gallery groups it
by what it is (a document), never by a tag the agent picks. An agent that
computes an array too large to write out saves it from inside the computing
call (``save_artifact``) and cites the id in its answer.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.submit_response")


#: Description of an answer whose agent did not name itself.
_UNNAMED_AGENT = "Agent response"


def _describe(agent: str) -> str:
    """The artifact's human-facing description line: the answering agent."""
    return agent or _UNNAMED_AGENT


@mcp.tool()
async def submit_response(
    title: str,
    content: str,
    entry_ids: list[str] | None = None,
    source_agent: str | None = None,
    skip_artifact: bool = False,
) -> str:
    """Submit your final synthesized response. Call this as your LAST action
    before responding. This persists your findings to the workspace so
    the parent session and other tools can reference them.

    Include all entry IDs, channel addresses, or other identifiers you
    cited in the entry_ids parameter for cross-referencing.

    Everything your answer reports belongs in ``content``: put a handful of
    values in a markdown table rather than leaving them out. An array too large
    to write out is saved from inside the computing ``execute`` call with
    ``save_artifact(...)``, and its id cited in your answer.

    Args:
        title: Short title for the response (e.g. "Vacuum Event Analysis").
        content: The full synthesized response text (markdown).
        entry_ids: List of ARIEL entry IDs or channel addresses cited,
            stored as structured metadata for cross-referencing.
        source_agent: Name of the agent submitting the response
            (e.g. "logbook-search", "pyat-specialist"). Used for filtering
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

    agent = source_agent or ""

    try:
        from osprey.stores.artifact_store import get_artifact_store

        cited = entry_ids or []

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

        store = get_artifact_store()
        tool_name = agent or "submit_response"
        artifact = store.save_file(
            file_content=content.encode(),
            filename=f"{tool_name}.md",
            artifact_type="markdown",
            title=title,
            description=_describe(agent),
            mime_type="text/markdown",
            tool_source="submit_response",
            metadata={
                "source_agent": agent,
                "entry_ids": cited,
            },
        )
        # Set unified fields on the entry
        updated = store.update_entry_metadata(
            artifact.id,
            source_agent=agent,
            summary={
                "title": title,
                "content_length": len(content),
                "cited_entries": len(cited),
                "source_agent": agent,
            },
        )
        if updated is None:
            raise RuntimeError(f"artifact {artifact.id} left the index before its metadata was set")
        response = updated.to_tool_response()
        response["gallery_url"] = gallery_url()
        return json.dumps(response, default=str)

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("submit_response failed")
        return make_error(
            "internal_error",
            f"Failed to save response: {exc}",
            ["Check that the _agent_data directory is accessible."],
        )
