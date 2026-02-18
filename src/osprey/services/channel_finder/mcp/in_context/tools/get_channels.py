"""MCP tool: cf_ic_get_channels -- retrieve channels from the in-context database.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_in_context_prompt_builder()
  Facility-customizable: tool description, chunking guidance
"""

import json
import logging

from osprey.services.channel_finder.mcp.in_context.registry import get_cf_ic_registry
from osprey.services.channel_finder.mcp.in_context.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.in_context.tools.get_channels")


@mcp.tool()
def cf_ic_get_channels(chunk_idx: int | None = None, chunk_size: int = 50) -> str:
    """Get channels from the database, optionally in chunks.

    When called without chunk_idx, returns all channels.
    When called with chunk_idx and chunk_size, returns a specific chunk
    formatted for prompt consumption.

    Args:
        chunk_idx: Optional chunk index (0-based). If None, returns all channels.
        chunk_size: Number of channels per chunk (default 50).
    """
    try:
        registry = get_cf_ic_registry()
        db = registry.database

        if chunk_idx is not None:
            chunks = db.chunk_database(chunk_size)

            if chunk_idx < 0 or chunk_idx >= len(chunks):
                return json.dumps(
                    make_error(
                        "validation_error",
                        f"chunk_idx {chunk_idx} out of range (0-{len(chunks) - 1})",
                        [f"Valid chunk indices: 0 to {len(chunks) - 1}"],
                    )
                )

            formatted = db.format_chunk_for_prompt(chunks[chunk_idx])
            return json.dumps(
                {
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunks[chunk_idx]),
                    "formatted": formatted,
                }
            )
        else:
            channels = db.get_all_channels()
            result = {
                "channels": channels,
                "total": len(channels),
            }
            return json.dumps(result, default=str)

    except ValueError as exc:
        return json.dumps(make_error("validation_error", str(exc)))
    except Exception as exc:
        logger.exception("cf_ic_get_channels failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get channels: {exc}",
                [
                    "Check that the channel database file exists and is valid JSON.",
                    "Verify config.yml channel_finder.pipelines.in_context.database settings.",
                ],
            )
        )
