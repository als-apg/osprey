"""SDK backend — wraps the Claude Agent SDK ``query()`` runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from osprey.services.channel_finder.benchmarks.sdk import (
    combined_text,
    run_sdk_query,
)

from .base import Backend, WorkflowOutput

if TYPE_CHECKING:
    from osprey.cli.claude_code_resolver import ClaudeCodeModelSpec


class SdkBackend(Backend):
    """Run queries via ``claude_agent_sdk.query()`` with Anthropic-native tool-use."""

    name = "sdk"

    def __init__(
        self,
        project_dir: Path,
        spec: ClaudeCodeModelSpec,
        tier: str,
        max_turns: int,
        max_budget_usd: float,
    ) -> None:
        self.project_dir = project_dir
        self.spec = spec
        self.tier = tier
        # Bare wire id — what Claude CLI's --model flag (and the upstream
        # Anthropic-compatible API behind it) expects. Prefixed slugs like
        # ``anthropic/<wire>`` are rejected by the als-apg gateway with
        # ``key_model_access_denied``.
        self.model = spec.tier_to_model[tier]
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

    async def run_query(self, prompt: str, pipeline_mode: str) -> WorkflowOutput:
        # pipeline_mode is unused — SDK selects the MCP server via .mcp.json.
        result = await run_sdk_query(
            self.project_dir,
            prompt,
            model=self.model,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
        )
        return WorkflowOutput(
            response_text=combined_text(result),
            tool_traces=result.tool_traces,
            cost_usd=result.cost_usd or 0.0,
            num_turns=result.num_turns or 1,
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
        )
