"""MCP tool: submit_response — persist an agent's final synthesized result.

Sub-agents call this as their last action to save their synthesis to
the artifact gallery so the parent session, other tools, and the gallery
UI can all reference it.

An agent that computes something hands its computed quantities in here too,
as ``data``, and the tool files them as a JSON artifact in the category the
agent's own definition declares (``results_category:`` in its frontmatter).
The prose and the numbers arrive in one call, so there is no separate step
for the agent to skip; an agent that owes data and hands in none is refused,
with the fix in the error.
"""

import json
import logging

from fastmcp.exceptions import ToolError

from osprey.mcp_server.agent_frontmatter import RESULTS_CATEGORY_KEY, results_category_for
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import gallery_url
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.workspace import resolve_config_path

logger = logging.getLogger("osprey.mcp_server.tools.submit_response")


def _declared_results_category(agent: str) -> str | None:
    """The category *agent* owes a data artifact to, per its rendered definition."""
    if not agent:
        return None
    return results_category_for(agent, resolve_config_path().parent)


@mcp.tool()
async def submit_response(
    title: str,
    content: str,
    data_type: str = "agent_response",
    entry_ids: list[str] | None = None,
    source_agent: str | None = None,
    skip_artifact: bool = False,
    data: dict | None = None,
) -> str:
    """Submit your final synthesized response. Call this as your LAST action
    before responding. This persists your findings to the workspace so
    the parent session and other tools can reference them.

    Include all entry IDs, channel addresses, or other identifiers you
    cited in the entry_ids parameter for cross-referencing.

    If your agent definition declares a ``results_category``, you computed
    something, and the numbers go in ``data``: a dict of native Python values
    (floats, ints, strings, small lists) holding every quantity your prose
    reports. It is filed as a JSON artifact in that category so the parent
    session and other tools read exact values, never prose. A hand-in without
    it is refused.

    Args:
        title: Short title for the response (e.g. "Vacuum Event Analysis").
        content: The full synthesized response text (markdown).
        data_type: Category tag for the prose.  Must be a registered type:
            "agent_response" (default), "channel_addresses",
            "logbook_research", "search_results", or any other key from
            the type registry.
        entry_ids: List of ARIEL entry IDs or channel addresses cited,
            stored as structured metadata for cross-referencing.
        source_agent: Name of the agent submitting the response
            (e.g. "logbook-search", "pyat-specialist"). Used for filtering
            and grouping results by agent, and to look up what the agent
            declares it owes.
        skip_artifact: If True, skip creating a new artifact (use when
            the agent already created plot/dashboard artifacts and wants
            to avoid double-registration).
        data: The computed quantities, for an agent whose definition
            declares a ``results_category``. Filed as a JSON artifact there.

    Returns:
        JSON with artifact_id, gallery_url, and summary — plus
        data_artifact_id when data was filed.
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

    agent = source_agent or ""
    results_category = _declared_results_category(agent)

    # The contract is checked BEFORE anything is written, so a refused hand-in
    # leaves no half-filed prose behind for the parent to mistake for a result.
    if results_category is not None and results_category not in valid:
        return make_error(
            "validation_error",
            f"Agent '{agent}' declares {RESULTS_CATEGORY_KEY}='{results_category}', "
            f"which is not a registered category. Valid: {sorted(valid)}",
            [
                f"Fix {RESULTS_CATEGORY_KEY} in the agent's definition "
                f"(.claude/agents/{agent}.md) to a registered category."
            ],
        )
    if data is not None and not isinstance(data, dict):
        return make_error(
            "validation_error",
            "data must be a dict of computed quantities.",
            ["Pass data={...} with native Python values (floats, ints, strings, lists)."],
        )
    if results_category is not None and not data:
        return make_error(
            "validation_error",
            f"Agent '{agent}' owes its computed results in the '{results_category}' "
            "category, and this hand-in carries none. Call submit_response again "
            "with data={...}: a dict holding every computed quantity your prose "
            "reports, as native Python values.",
            [
                "The prose alone is not the result — the parent session and other "
                "tools read the exact values from the data artifact, never from prose.",
            ],
        )
    if data and results_category is None:
        return make_error(
            "validation_error",
            f"data was passed, but agent '{agent or '(none)'}' declares no "
            f"{RESULTS_CATEGORY_KEY}, so there is no category to file it in.",
            [
                f"Declare `{RESULTS_CATEGORY_KEY}: <registered category>` in the agent's "
                "definition frontmatter, or hand in prose only.",
                "Pass source_agent=<your agent name> so the declaration can be found.",
            ],
        )

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

        # The category is the declared data_type, never the agent's name. An
        # agent-derived category would put this prose in the same bucket as the
        # structured results filed below, leaving that bucket unable to say
        # which of the two it holds. Grouping by agent is already served by
        # source_agent.
        category = data_type

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

        if data:
            # The data sheet: the agent's computed quantities as JSON, in the
            # category it declared. Saved second so its metadata can name the
            # prose it belongs to; the prose is then pointed back at it.
            results = store.save_data(
                tool=tool_name,
                data=data,
                title=title,
                description=f"{results_category} from {agent}",
                category=results_category,
                source_agent=agent,
                summary={
                    "title": title,
                    "keys": sorted(str(k) for k in data)[:50],
                    "source_agent": agent,
                },
                metadata={
                    "source_agent": agent,
                    "response_artifact_id": artifact.id,
                },
            )
            store.update_entry_metadata(
                artifact.id,
                metadata={**artifact.metadata, "data_artifact_id": results.id},
            )
            response["data_artifact_id"] = results.id
            response["data_category"] = results_category

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
