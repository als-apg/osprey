"""MCP tools: memory_save / memory_recall / memory_update / memory_delete.

Persistent session memory with enriched data model (note/pin types, tags,
importance, cross-domain links). Data stored via MemoryStore and also
registered in the DataContext index for cross-tool completeness.

PROMPT-PROVIDER: Tool docstrings are static prompts visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_memory_extraction_prompt_builder()
  Facility-customizable: category tag examples, memory usage guidance
"""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.memory")


@mcp.tool()
async def memory_save(
    content: str,
    tags: list[str] | None = None,
    importance: str | None = None,
    linked_artifact_id: str | None = None,
    linked_context_id: int | None = None,
    category: str | None = None,
) -> str:
    """Save a piece of information to persistent OSPREY session memory.

    Creates either a Note (text-first: observations, findings, procedures) or
    a Pin (link-first: bookmark a specific artifact or context entry). Memory
    type is auto-detected: if a linked_artifact_id or linked_context_id is
    provided, the memory becomes a Pin; otherwise it's a Note.

    Args:
        content: The information to remember (note body or pin annotation).
        tags: Optional tags for organic taxonomy (e.g. ["procedure", "beam"]).
        importance: "normal" (default) or "important" for priority filtering.
        linked_artifact_id: Artifact ID to link (creates a Pin).
        linked_context_id: Context entry ID to link (creates a Pin).
        category: Deprecated. Use tags instead. Kept for backward compatibility.

    Returns:
        Confirmation JSON with the stored memory ID and type.
    """
    if not content or not content.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No content provided.",
                ["Provide the information you want to save."],
            )
        )

    importance = importance or "normal"
    if importance not in ("normal", "important"):
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid importance '{importance}'. Must be 'normal' or 'important'.",
                ["Use importance='normal' or importance='important'."],
            )
        )

    try:
        from osprey.mcp_server.memory_store import get_memory_store

        store = get_memory_store()

        # Auto-detect memory type
        has_link = linked_artifact_id is not None or linked_context_id is not None
        memory_type = "pin" if has_link else "note"

        # Validate linked targets exist and resolve labels
        linked_label = None
        if linked_artifact_id is not None:
            from osprey.mcp_server.artifact_store import get_artifact_store

            art = get_artifact_store().get_entry(linked_artifact_id)
            if not art:
                return json.dumps(
                    make_error(
                        "not_found",
                        f"Artifact '{linked_artifact_id}' not found.",
                        ["Check the artifact_id from a previous tool response."],
                    )
                )
            linked_label = art.title

        if linked_context_id is not None:
            from osprey.mcp_server.data_context import get_data_context

            ctx_entry = get_data_context().get_entry(linked_context_id)
            if not ctx_entry:
                return json.dumps(
                    make_error(
                        "not_found",
                        f"Context entry {linked_context_id} not found.",
                        ["Check the context_entry_id from a previous tool response."],
                    )
                )
            linked_label = linked_label or ctx_entry.description

        # Merge category into tags for backward compat
        effective_tags = list(tags) if tags else []
        if category and category not in effective_tags:
            effective_tags.append(category)

        entry = store.save(
            memory_type=memory_type,
            content=content,
            tags=effective_tags,
            importance=importance,
            linked_artifact_id=linked_artifact_id,
            linked_context_id=linked_context_id,
            linked_label=linked_label,
            category=category,
        )

        # Build compact summary inline
        summary = {
            "operation": "save",
            "memory_id": entry.id,
            "memory_type": memory_type,
            "tags": effective_tags,
            "importance": importance,
            "content_preview": content[:200],
        }
        if category:
            summary["category"] = category
        access_details = {
            "memory_id": entry.id,
            "memory_type": memory_type,
            "tags": effective_tags,
        }

        # Register in data context for index completeness
        from osprey.mcp_server.data_context import get_data_context

        data_ctx = get_data_context()
        ctx_entry = data_ctx.save(
            tool="memory_save",
            data=entry.to_dict(),
            description=(
                f"Saved {memory_type} #{entry.id}"
                + (f" [{', '.join(effective_tags)}]" if effective_tags else "")
            ),
            summary=summary,
            access_details=access_details,
            data_type="memory",
        )

        return json.dumps(ctx_entry.to_tool_response(), default=str)

    except Exception as exc:
        logger.exception("memory_save failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to save memory: {exc}",
                ["Check filesystem permissions for osprey-workspace/memory/."],
            )
        )


@mcp.tool()
async def memory_recall(
    query: str,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    importance: str | None = None,
) -> str:
    """Search previously saved OSPREY session memories.

    Performs substring matching across all stored memories, with optional
    filters for type, tags, and importance.

    Args:
        query: Search query to match against saved memories.
        memory_type: Filter by type — "note" or "pin".
        tags: Filter by tags (any-match semantics).
        importance: Filter by importance — "normal" or "important".

    Returns:
        JSON with matching memories sorted by recency.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No query provided.",
                ["Provide a search query to find saved memories."],
            )
        )

    try:
        from osprey.mcp_server.memory_store import get_memory_store

        store = get_memory_store()
        matches = store.list_entries(
            memory_type=memory_type,
            tags=tags,
            importance=importance,
            search=query,
        )

        # Most recent first
        matches.sort(key=lambda m: m.timestamp, reverse=True)

        memories_data = [m.to_dict() for m in matches]

        recall_result = {
            "query": query,
            "matches_found": len(matches),
            "memories": memories_data,
        }

        # Build compact summary inline
        summary = {
            "operation": "recall",
            "query": query,
            "matches_found": len(matches),
            "memories": memories_data,
        }
        access_details = {"query": query, "matches_found": len(matches)}

        # Register in data context for index completeness
        from osprey.mcp_server.data_context import get_data_context

        data_ctx = get_data_context()
        entry = data_ctx.save(
            tool="memory_recall",
            data=recall_result,
            description=f"Memory recall: '{query}' — {len(matches)} match(es)",
            summary=summary,
            access_details=access_details,
            data_type="memory",
        )

        return json.dumps(entry.to_tool_response(), default=str)

    except Exception as exc:
        logger.exception("memory_recall failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to recall memories: {exc}",
                ["Check that osprey-workspace/memory/memories.json is readable."],
            )
        )


@mcp.tool()
async def memory_update(
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    importance: str | None = None,
) -> str:
    """Update an existing OSPREY session memory.

    Modify the content, tags, or importance of a previously saved memory.

    Args:
        memory_id: ID of the memory to update.
        content: New content (if changing).
        tags: New tags list (replaces existing).
        importance: New importance level ("normal" or "important").

    Returns:
        JSON confirmation with updated memory details.
    """
    if importance is not None and importance not in ("normal", "important"):
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid importance '{importance}'. Must be 'normal' or 'important'.",
                ["Use importance='normal' or importance='important'."],
            )
        )

    try:
        from osprey.mcp_server.memory_store import get_memory_store

        store = get_memory_store()

        fields = {}
        if content is not None:
            fields["content"] = content
        if tags is not None:
            fields["tags"] = tags
        if importance is not None:
            fields["importance"] = importance

        if not fields:
            return json.dumps(
                make_error(
                    "validation_error",
                    "No fields to update.",
                    ["Provide at least one of: content, tags, importance."],
                )
            )

        entry = store.update_entry(memory_id, **fields)
        if not entry:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Memory {memory_id} not found.",
                    ["Check the memory_id from a previous memory_save response."],
                )
            )

        return json.dumps(
            {
                "status": "success",
                "memory_id": entry.id,
                "memory_type": entry.memory_type,
                "updated_fields": list(fields.keys()),
            }
        )

    except Exception as exc:
        logger.exception("memory_update failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to update memory: {exc}",
                ["Check filesystem permissions for osprey-workspace/memory/."],
            )
        )


@mcp.tool()
async def memory_delete(memory_id: int) -> str:
    """Delete a previously saved OSPREY session memory.

    Args:
        memory_id: ID of the memory to delete.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        from osprey.mcp_server.memory_store import get_memory_store

        store = get_memory_store()
        deleted = store.delete_entry(memory_id)

        if not deleted:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Memory {memory_id} not found.",
                    ["Check the memory_id from a previous memory_save response."],
                )
            )

        return json.dumps(
            {
                "status": "success",
                "memory_id": memory_id,
                "message": f"Memory {memory_id} deleted.",
            }
        )

    except Exception as exc:
        logger.exception("memory_delete failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to delete memory: {exc}",
                ["Check filesystem permissions for osprey-workspace/memory/."],
            )
        )
