"""InContext backend — single MCP tool call, no outer ReAct loop."""

from __future__ import annotations

import logging
from pathlib import Path

from osprey.services.channel_finder.benchmarks.harness import mcp_client_session
from osprey.services.channel_finder.benchmarks.sdk import ToolTrace, sdk_env

from .base import Backend, WorkflowOutput

logger = logging.getLogger(__name__)


def _extract_text(tool_result) -> str:
    """Extract plain text from a fastmcp CallToolResult."""
    try:
        blocks = tool_result.content
        if blocks:
            texts = []
            for block in blocks:
                text = getattr(block, "text", None)
                if text is not None:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
    except Exception:
        pass
    return str(tool_result)


class InContextBackend(Backend):
    """Run a single ``query_channels`` MCP tool call with no outer agent loop.

    The inner LLM call happens inside the spawned MCP subprocess; the outer
    benchmark process only sees the tool result string.

    The backend identifier is ``"direct"`` rather than ``"in_context"`` —
    that's the *paradigm* this backend can host, not the harness shape.
    Calling it ``"in_context"`` historically conflated the paradigm axis
    (``in_context`` / ``hierarchical`` / ``middle_layer``) with the harness
    axis (``sdk`` / ``react`` / ``direct``) on the dashboard.
    """

    name = "direct"

    def __init__(self, project_dir: Path, model: str) -> None:
        self.project_dir = project_dir
        self.model = model
        # sdk_env injects provider auth; OSPREY_CONFIG ensures the subprocess
        # finds the project config.yml regardless of cwd.
        self._env = sdk_env(project_dir) | {
            "OSPREY_CONFIG": str(project_dir / "config.yml"),
        }

    async def run_query(self, prompt: str, pipeline_mode: str) -> WorkflowOutput:
        async with mcp_client_session(self.project_dir, "in_context", env=self._env) as client:
            tool_result = await client.call_tool("query_channels", {"query": prompt})

        is_error = getattr(tool_result, "is_error", False)

        # Prefer structuredContent (the typed dict from query_channels) for
        # exact text + tokenizer-estimated input/output token counts. Falls
        # back to plain-text extraction if the server is older or returned
        # a string-typed result for any reason.
        structured = getattr(tool_result, "structuredContent", None) or getattr(
            tool_result, "structured_content", None
        )
        if isinstance(structured, dict) and "text" in structured:
            result_text = str(structured.get("text", ""))
            input_tokens = int(structured.get("input_tokens", 0) or 0)
            output_tokens = int(structured.get("output_tokens", 0) or 0)
        else:
            result_text = _extract_text(tool_result)
            input_tokens = 0
            output_tokens = 0

        trace = ToolTrace(
            name="query_channels",
            input={"query": prompt, "_inner_model_id": self.model},
            result=result_text,
            is_error=is_error,
        )

        return WorkflowOutput(
            response_text=result_text,
            num_turns=1,
            tool_traces=[trace],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
