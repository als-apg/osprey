"""Run a benchmark query as the channel-finder agent.

``run_sdk_query`` runs the agent through :func:`osprey.agent_runner.run_query`
so that provider routing, the MCP readiness barrier and the response drain are
the ones every other launch path uses.

``ToolTrace``, ``SDKWorkflowResult``, ``sdk_env``, and ``combined_text`` are
re-exported from :mod:`osprey.agent_runner` — that module is the single source
of truth for these primitives.

Nothing here builds a project — benchmarks are handed a built repo and read it.
Project setup belongs to ``tests.e2e.sdk_helpers.init_project``, which drives
the ``osprey init`` + ``osprey build`` surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osprey.agent_runner import SDKWorkflowResult, run_query
from osprey.agent_runner import ToolTrace as ToolTrace
from osprey.agent_runner import combined_text as combined_text
from osprey.agent_runner import sdk_env as sdk_env

# ---------------------------------------------------------------------------
# Reading a built project's rendered agent surface
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


def _read_channel_finder_mcp(project_dir: Path) -> dict[str, Any] | None:
    """Extract the channel-finder MCP server config from ``.mcp.json``.

    Returns ``run_query``'s ``mcp_servers`` mapping, containing only the
    channel-finder server entry, or ``None``.
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
    model: str | None = None,
    provider: str | None = None,
) -> SDKWorkflowResult:
    """Run a benchmark query as the channel-finder sub-agent.

    Runs the agent through ``osprey.agent_runner.run_query`` configured to act
    as the channel-finder sub-agent directly:

    - **system_prompt**: The rendered channel-finder agent prompt with
      paradigm-specific navigation instructions.
    - **mcp_servers**: Only the channel-finder MCP server (no control
      system, python executor, workspace, or other servers). The first turn
      waits for that server to connect.
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
        model: Bare wire-id of the Anthropic-compatible model to use
            (e.g. ``"claude-haiku-4-5-20251001"``). Must be a wire id, not
            a LiteLLM-style ``provider/<wire>`` slug — the SDK CLI forwards
            ``--model`` verbatim to the upstream Anthropic API and gateways
            like als-apg reject prefixed slugs with ``key_model_access_denied``.
            When ``None``, the runner resolves the project's main model for
            *provider*.
        provider: Overrides the project's ``claude_code.provider`` for this
            query; ``None`` keeps the configured one.

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.

    Raises:
        RuntimeError: When the query fails; the message ends with the CLI's
            captured stderr.
    """
    cf_servers = _read_channel_finder_mcp(project_dir)
    stderr_lines: list[str] = []

    try:
        return await run_query(
            project_dir,
            prompt,
            disallowed_tools=[],
            allowed_tools=["mcp__channel-finder__*"],
            system_prompt=_read_agent_prompt(project_dir),
            mcp_servers=cf_servers or project_dir / ".mcp.json",
            await_mcp_servers=set(cf_servers) if cf_servers else None,
            setting_sources=[],
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            model=model,
            provider=provider,
            stderr=stderr_lines.append,
        )
    except RuntimeError as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"{exc}\n\nCLI stderr:\n{stderr_output}") from (exc.__cause__ or exc)
