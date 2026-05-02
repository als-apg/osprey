"""Backend ABC and unified result shape."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from osprey.services.channel_finder.benchmarks.sdk import ToolTrace


@dataclass
class WorkflowOutput:
    """Unified per-query output for both SDK and ReAct backends."""

    response_text: str
    tool_traces: list[ToolTrace] = field(default_factory=list)
    cost_usd: float = 0.0
    num_turns: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(ABC):
    """Strategy for executing one benchmark query end-to-end."""

    name: str = ""

    @abstractmethod
    async def run_query(self, prompt: str, pipeline_mode: str) -> WorkflowOutput:
        """Execute a single benchmark query.

        Args:
            prompt: User query text.
            pipeline_mode: ``in_context`` / ``hierarchical`` / ``middle_layer``.
                ReAct backend uses this to spawn the right MCP server;
                SDK backend ignores it (server is selected via .mcp.json).
        """
