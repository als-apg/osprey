"""SDK helpers for running benchmark queries via the Claude Agent SDK.

Extracted from ``tests/e2e/sdk_helpers.py`` so that production code
(``BenchmarkRunner``) can use them without importing from the test tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# SDK imports — skip if not installed
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        query,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# ---------------------------------------------------------------------------
# Tool trace dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolTrace:
    """Lightweight record of a single tool call for observability."""

    name: str
    input: dict
    result: str | None = None
    is_error: bool = False
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class SDKWorkflowResult:
    """Aggregated result from an SDK query run."""

    tool_traces: list[ToolTrace] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    system_messages: list[Any] = field(default_factory=list)
    result: Any = None

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names that were called."""
        return [t.name for t in self.tool_traces]

    @property
    def cost_usd(self) -> float | None:
        """Total cost from the ResultMessage."""
        return self.result.total_cost_usd if self.result else None

    @property
    def num_turns(self) -> int | None:
        """Number of agentic turns from the ResultMessage."""
        return self.result.num_turns if self.result else None

    @property
    def input_tokens(self) -> int:
        """Total input tokens (raw + cache creation + cache read)."""
        if not self.result or not getattr(self.result, "usage", None):
            return 0
        u = self.result.usage
        return (
            u.get("input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
        )

    @property
    def output_tokens(self) -> int:
        """Total output tokens."""
        if not self.result or not getattr(self.result, "usage", None):
            return 0
        return self.result.usage.get("output_tokens", 0)

    @property
    def cache_read_tokens(self) -> int:
        """Cache-read input tokens (charged at reduced rate)."""
        if not self.result or not getattr(self.result, "usage", None):
            return 0
        return self.result.usage.get("cache_read_input_tokens", 0)

    @property
    def cache_creation_tokens(self) -> int:
        """Cache-creation input tokens."""
        if not self.result or not getattr(self.result, "usage", None):
            return 0
        return self.result.usage.get("cache_creation_input_tokens", 0)

    def tools_matching(self, substring: str) -> list[ToolTrace]:
        """Return all tool traces whose name contains *substring*."""
        return [t for t in self.tool_traces if substring in t.name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sdk_env(project_dir: Path | None = None) -> dict[str, str]:
    """Return env overrides for SDK subprocess.

    Bypasses nested-session guard (CLAUDECODE="") and injects provider
    auth from the project's config.yml so the CLI authenticates correctly
    against Anthropic, CBORG, or other configured providers.
    """
    import os

    env: dict[str, str] = {"CLAUDECODE": ""}

    if project_dir is not None:
        try:
            import yaml

            from osprey.cli.claude_code_resolver import (
                ClaudeCodeModelResolver,
                inject_provider_env,
            )

            config_path = project_dir / "config.yml"
            if config_path.exists():
                config = yaml.safe_load(config_path.read_text()) or {}
                cc_config = config.get("claude_code", {})
                api_providers = config.get("api", {}).get("providers", {})
                spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
                if spec:
                    # Build a copy of environ, inject provider auth, then
                    # extract only the vars that changed.
                    scratch = dict(os.environ)
                    inject_provider_env(scratch, spec, project_dir=project_dir)
                    for key in list(scratch):
                        if scratch[key] != os.environ.get(key):
                            env[key] = scratch[key]
        except Exception:
            pass  # Fall back to bare env if resolver unavailable

    return env


def combined_text(result: SDKWorkflowResult) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    provider: str = "anthropic",
    model: str = "haiku",
    channel_finder_mode: str | None = None,
) -> Path:
    """Create a project via ``osprey build --preset <template>``, return project_dir."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build

    runner = CliRunner()
    args = [
        name,
        "--preset",
        template.replace("_", "-"),
        "--skip-deps",
        "--skip-lifecycle",
        "--output-dir",
        str(tmp_path),
        "--set",
        f"provider={provider}",
        "--set",
        f"model={model}",
    ]
    if channel_finder_mode is not None:
        args.extend(["--set", f"channel_finder_mode={channel_finder_mode}"])
    result = runner.invoke(build, args)
    assert result.exit_code == 0, f"osprey build failed: {result.output}"
    project_dir = tmp_path / name
    assert project_dir.exists(), f"Project directory not created: {project_dir}"
    return project_dir


# ---------------------------------------------------------------------------
# Project preparation for benchmarks
# ---------------------------------------------------------------------------


def _read_agent_prompt(project_dir: Path) -> str | None:
    """Read the rendered channel-finder agent prompt from the project.

    Returns the body of ``.claude/agents/channel-finder.md`` (everything
    after the YAML frontmatter), or ``None`` if the file doesn't exist.
    """
    agent_path = project_dir / ".claude" / "agents" / "channel-finder.md"
    if not agent_path.exists():
        return None

    text = agent_path.read_text(encoding="utf-8")

    # Strip YAML frontmatter (delimited by --- ... ---)
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3 :].strip()

    return text.strip()


def _read_channel_finder_mcp(project_dir: Path) -> dict | None:
    """Extract the channel-finder MCP server config from ``.mcp.json``.

    Returns a dict suitable for ``ClaudeAgentOptions.mcp_servers``,
    containing only the channel-finder server entry, or ``None``.
    """
    import json

    mcp_path = project_dir / ".mcp.json"
    if not mcp_path.exists():
        return None

    mcp_data = json.loads(mcp_path.read_text(encoding="utf-8"))
    servers = mcp_data.get("mcpServers", {})
    cf_servers = {k: v for k, v in servers.items() if "channel-finder" in k}
    return cf_servers or None


# ---------------------------------------------------------------------------
# Core SDK runner
# ---------------------------------------------------------------------------


async def run_sdk_query(
    project_dir: Path,
    prompt: str,
    *,
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str = "anthropic/claude-haiku",
) -> SDKWorkflowResult:
    """Run a benchmark query as the channel-finder sub-agent.

    Launches the Claude Agent SDK configured to act as the channel-finder
    sub-agent directly:

    - **system_prompt**: The rendered channel-finder agent prompt with
      paradigm-specific navigation instructions.
    - **mcp_servers**: Only the channel-finder MCP server (no control
      system, python executor, workspace, or other servers).
    - **allowed_tools**: Restricted to ``mcp__channel-finder__*`` tools
      only — no Bash, Read, Glob, Task, Skill, or other built-ins.

    This bypasses the main orchestrator agent entirely and directly
    measures the channel-finding capability of a given information
    representation paradigm.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use (defaults to Haiku for cost-effectiveness).

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.
    """
    # Read the channel-finder agent's dedicated prompt
    agent_prompt = _read_agent_prompt(project_dir)

    # Extract only the channel-finder MCP server definition
    cf_servers = _read_channel_finder_mcp(project_dir)

    # Collect stderr lines for debugging CLI failures
    stderr_lines: list[str] = []

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(project_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=sdk_env(project_dir),
        stderr=lambda line: stderr_lines.append(line),
        # Use the channel-finder agent prompt as the system prompt
        system_prompt=agent_prompt,
        # Provide only the channel-finder MCP server
        mcp_servers=cf_servers or str(project_dir / ".mcp.json"),
        # Restrict to channel-finder MCP tools only
        allowed_tools=["mcp__channel-finder__*"],
        # Don't load project settings (agents, hooks, skills, etc.)
        setting_sources=[],
    )

    workflow = SDKWorkflowResult()

    # Map tool_use_id → ToolTrace for matching results to calls
    pending_tools: dict[str, ToolTrace] = {}

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        workflow.text_blocks.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        trace = ToolTrace(
                            name=block.name,
                            input=block.input,
                            tool_use_id=block.id,
                            parent_tool_use_id=message.parent_tool_use_id,
                        )
                        workflow.tool_traces.append(trace)
                        pending_tools[block.id] = trace
                    elif isinstance(block, ToolResultBlock):
                        # Match result to its tool call
                        matched = pending_tools.get(block.tool_use_id)
                        if matched:
                            if isinstance(block.content, str):
                                matched.result = block.content
                            elif isinstance(block.content, list):
                                # Extract text from content list
                                texts = []
                                for item in block.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        texts.append(item.get("text", ""))
                                matched.result = "\n".join(texts) if texts else str(block.content)
                            matched.is_error = bool(block.is_error)

            elif isinstance(message, SystemMessage):
                workflow.system_messages.append(message)

            elif isinstance(message, ResultMessage):
                workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    return workflow
