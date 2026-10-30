"""MCP tool: prior_answer_read.

Returns the full text of an earlier answer in the current chat conversation.
A chat bridge replays a long earlier answer shortened, ending in a note that
names the run that answered it; this tool reads that run's answer back from
the agent record, as the dispatch worker's login sees it, narrowed to the
permitted runs.

Bounds:
    * The only runs it reads are the ones ``run_dispatch`` stamped into this
      process's environment at spawn (``OSPREY_DISPATCH_PRIOR_ANSWER_RUNS``):
      the runs whose answers this dispatch's own history replayed shortened.
      Any other run id is refused before the record is asked, and the read
      itself is narrowed to that set, so the record serves no other run.
    * Outside a dispatched run the set is empty, so every call is refused.
    * Read-only: it never writes, and it needs no approval.

Contract:
    * Returns JSON ``{run_id, total_chars, offset, text, next_offset}``; a long
      answer comes back in pages of ``PAGE_CHARS`` characters, and
      ``next_offset`` is null on the last page.
    * A permitted run the record no longer holds, or whose answer has no text,
      is reported as no longer available.
    * A record that cannot be read is reported as unavailable, and a read that
      may be missing part of the answer as incomplete; neither is ever served
      as the answer.
"""

import asyncio
import json
import logging
import os

from fastmcp.exceptions import ToolError

from osprey.agent_record.client import RecordClient
from osprey.agent_runner.artifact_resolve import record_lacks_run
from osprey.mcp_server.dispatch_worker.prior_answers import (
    PRIOR_ANSWER_RUNS_ENV,
    parse_run_ids,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.prior_answer")

# A page must stay well under the CLI's cap on one MCP tool result, and a run's
# answer can reach 256 KiB (``sdk_runner._MAX_TEXT_OUTPUT``); a 68k-character
# table is two pages.
PAGE_CHARS = 40_000


def record_client() -> RecordClient:
    """The record client this process's environment configures, resolved per call."""
    return RecordClient.from_env(os.environ)


@mcp.tool()
async def prior_answer_read(run_id: str, offset: int = 0) -> str:
    """Read the full text of an earlier answer in this conversation.

    A long earlier answer arrives in the conversation history shortened, ending
    in a note that names its run; pass that run id here. Only the runs named in
    this conversation's shortened answers can be read. A long answer comes back
    in pages: call again with `offset` set to `next_offset` until it is null.

    Args:
        run_id: The run id named in a shortened answer's note.
        offset: Character offset to start the page at.

    Returns:
        JSON with run_id, total_chars, offset, text and next_offset.
    """
    allowed = parse_run_ids(os.environ.get(PRIOR_ANSWER_RUNS_ENV))
    if run_id not in allowed:
        make_error(
            "not_permitted",
            f"Run '{run_id}' is not a shortened answer in this conversation.",
            ["Use a run id from a shortened answer's note in conversation_so_far."],
        )

    try:
        answer = await asyncio.to_thread(record_client().run, run_id, narrow_runs=allowed)
        if not answer.available:
            make_error(
                "record_unavailable",
                f"The agent record is unavailable, so that earlier answer cannot be read "
                f"right now ({answer.unavailable}).",
                [
                    "Work from the answer's shortened opening, and say the full text could "
                    "not be read."
                ],
            )
        data = answer.data if isinstance(answer.data, dict) else None
        if not answer.complete and not record_lacks_run(answer, run_id):
            make_error(
                "incomplete",
                f"The record could not yet give that earlier answer in full ({answer.reason}).",
                ["Try again shortly, or work from the answer's shortened opening."],
            )
        text = data.get("answer") if data is not None else None
        if not isinstance(text, str) or not text:
            make_error(
                "no_longer_available",
                "The full text of that earlier answer is no longer available.",
                ["Say that the full answer has expired and work from its shortened opening."],
            )

        total = len(text)
        if offset < 0 or offset > total:
            make_error(
                "bad_offset",
                f"Offset {offset} is outside the answer; use 0 to {total}.",
                ["Start at offset 0, then follow next_offset."],
            )

        end = min(offset + PAGE_CHARS, total)
        return json.dumps(
            {
                "run_id": run_id,
                "total_chars": total,
                "offset": offset,
                "text": text[offset:end],
                "next_offset": end if end < total else None,
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("prior_answer_read failed")
        make_error(
            "internal_error",
            f"Failed to read the earlier answer: {exc}",
            ["Work from the answer's shortened opening."],
        )
