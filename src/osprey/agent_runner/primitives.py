"""Headless agent-run primitives shared across the OSPREY package.

These symbols are extracted from ``tests/e2e/sdk_helpers.py`` so they can be
imported by production code (e.g. ``osprey query``) without depending on the
test tree.  The test module re-imports from here and deletes its own copies so
there is a single source of truth.

The SDK import block mirrors the pattern in ``sdk_helpers.py``: the whole
module remains importable even when ``claude_agent_sdk`` is not installed; only
the runtime paths that actually USE the SDK will fail in that case.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ResultMessage,
        SystemMessage,
        ToolResultBlock,
    )

# SDK imports — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import ResultMessage, SystemMessage  # type: ignore[assignment]

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# ---------------------------------------------------------------------------
# Internal support: provider / spec resolution
# ---------------------------------------------------------------------------
# These helpers back sdk_env() and resolve_default_model().  They are NOT part
# of the public __all__ but live here because they have no other production
# home yet (they were in the test tree alongside the consumers).


def _apply_e2e_overrides(spec: Any) -> Any:
    """Apply suite-wide CBORG model-matrix overrides to a resolved spec (#259).

    This is the single chokepoint every routing consumer goes through
    (``provider_env_for_project``, ``resolve_default_model``,
    and ``run_sdk_query``'s default model), so forcing the spec here keeps
    the build env, the SDK ``model=`` argument, and the proxy base URL
    mutually consistent.

    * ``OSPREY_E2E_FORCE_MODEL`` — collapse every tier onto one model id and
      rewrite the tier-model env vars to match.
    * ``OSPREY_E2E_PROXY_BASE_URL`` — point ``ANTHROPIC_BASE_URL`` at the
      in-process translation proxy (set by the session fixture in
      tests/e2e/conftest.py for OpenAI-protocol / open models). Anthropic-
      protocol CBORG models (``claude-*``) leave this unset and route direct.

    Inert (returns the spec unchanged) when neither var is set, so normal
    e2e runs are unaffected.
    """
    if spec is None:
        return None
    force_model = os.environ.get("OSPREY_E2E_FORCE_MODEL")
    proxy_base = os.environ.get("OSPREY_E2E_PROXY_BASE_URL")
    if not force_model and not proxy_base:
        return spec

    import dataclasses

    tier_to_model = dict(spec.tier_to_model)
    env_block = dict(spec.env_block)
    if force_model:
        for tier in tier_to_model:
            tier_to_model[tier] = force_model
        for key in (
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
        ):
            if key in env_block:
                env_block[key] = force_model
    if proxy_base:
        env_block["ANTHROPIC_BASE_URL"] = proxy_base
    return dataclasses.replace(spec, tier_to_model=tier_to_model, env_block=env_block)


def _resolve_project_spec(project_dir: Path, *, provider: str | None = None) -> Any:
    """Return the project's ``ClaudeCodeModelSpec`` or ``None``.

    Reads ``config.yml`` and runs the same resolver ``osprey claude chat``
    uses, so test routing matches production exactly.  Surfaces any unexpected
    error (missing config, YAML parse failure, resolver import failure) rather
    than masking it as ``None``.

    Args:
        project_dir: Path to an initialized OSPREY project.
        provider: When given, overrides ``claude_code.provider`` in the loaded
            config before resolving — used by cross-provider model sweeps in
            the benchmark runner.
    """
    from osprey.cli.claude_code_resolver import load_provider_spec

    # load_provider_spec reads config.yml and expands ${VAR} in provider config
    # (e.g. a custom provider's base_url: ${ARGO_PROD_URL}) against an
    # os.environ + project .env overlay before resolving. The e2e/benchmark
    # override is applied last so it still wins (and is inert in production).
    spec = load_provider_spec(project_dir, provider=provider)
    return _apply_e2e_overrides(spec)


def provider_env_for_project(project_dir: Path, *, provider: str | None = None) -> dict[str, str]:
    """Resolve provider env vars so the SDK routes to the project's provider.

    Without this, the bundled Claude CLI defaults to ``api.anthropic.com``
    using whatever ambient ``ANTHROPIC_API_KEY`` happens to be set — which
    is the wrong endpoint for cborg/als-apg projects and 404s on model
    aliases like ``anthropic/claude-haiku``.

    Args:
        project_dir: Path to an initialized OSPREY project.
        provider: When given, overrides ``claude_code.provider`` in the loaded
            config before resolving — used by cross-provider model sweeps in
            the benchmark runner.

    Returns:
        Env dict with ``ANTHROPIC_BASE_URL``, the auth-token var, the tier-model
        vars, and the *raw* upstream secret var (see below), populated from the
        configured provider.

    Raises:
        RuntimeError: When the project has no resolvable provider.

    Notes:
        Two distinct auth vars are propagated because the project has two LLM
        call paths:

        * The bundled Claude CLI / SDK authenticates via ``spec.auth_env_var``
          (e.g. ``ANTHROPIC_AUTH_TOKEN`` for proxy providers).
        * The in-context channel-finder benchmark spawns an MCP **stdio**
          subprocess that inherits only a tiny safe-list (HOME, PATH, …) and
          then expands ``config.yml``'s ``api_key: ${SECRET}`` against its own
          environment. So the *raw* ``spec.auth_secret_env`` var must be carried
          through explicitly, or the literal ``${SECRET}`` reaches litellm and
          401s. For proxy providers (cborg/als-apg) the two var names differ;
          for anthropic-direct they coincide.

        A project-level ``.env`` is honoured first, so a freshly-configured key
        overrides a stale shell export (mirrors ``inject_provider_env``).
    """
    spec = _resolve_project_spec(project_dir, provider=provider)
    if spec is None:
        raise RuntimeError(
            f"Project at {project_dir} has no resolvable provider in "
            "config.yml — pass provider=<als-apg|cborg|anthropic|amsc-i2|argo> "
            "to init_project()."
        )
    env: dict[str, str] = dict(spec.env_block)

    # Overlay a project-level .env so freshly-configured keys win over stale
    # shell exports; strictly a superset of reading os.environ alone.
    lookup: dict[str, str] = dict(os.environ)
    env_file = project_dir / ".env"
    if env_file.is_file():
        try:
            from dotenv import dotenv_values

            for key, value in dotenv_values(env_file).items():
                if value is not None:
                    lookup[key] = value
        except ImportError:
            pass

    if spec.auth_secret_env:
        secret = lookup.get(spec.auth_secret_env)
        if secret:
            if spec.auth_env_var:
                env[spec.auth_env_var] = secret
            # Raw secret for the MCP-subprocess ${SECRET} expansion (see Notes).
            env[spec.auth_secret_env] = secret
    return env


# ---------------------------------------------------------------------------
# Tool trace dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolTrace:
    """Lightweight record of a single tool call for observability."""

    name: str
    input: dict[str, Any]
    result: str | None = None
    is_error: bool = False
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class SDKWorkflowResult:
    """Aggregated result from an SDK query run."""

    tool_traces: list[ToolTrace] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    system_messages: list[SystemMessage] = field(default_factory=list)
    result: ResultMessage | None = None
    # Authoritative MCP server snapshot from ``ClaudeSDKClient.get_mcp_status()``
    # captured just before the prompt is sent (see ``_await_mcp_ready``). Each entry
    # is an ``McpServerStatus`` object (SDK) or raw dict: {name, status, tools, ...}.
    # Empty when the runner used the one-shot ``query()`` path with no client to poll.
    # This is the ground-truth infra-vs-model discriminator: a failure where the
    # expected tool is absent here is INFRA (server never registered); present-but-
    # unused is MODEL.
    mcp_servers: list[Any] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names that were called."""
        return [t.name for t in self.tool_traces]

    @property
    def mcp_server_status(self) -> dict[str, str]:
        """``{'controls': 'connected', 'python': 'connected', ...}`` from the
        captured ``get_mcp_status()`` snapshot. Empty if no snapshot was taken."""
        return {s.get("name", "?"): s.get("status", "unknown") for s in self.mcp_servers}

    @property
    def registered_tools(self) -> list[str]:
        """Flattened ``mcp__<server>__<tool>`` names that the MCP handshake exposed.

        Built from the captured snapshot, so it reflects what the model could
        actually call — not what it chose to call (that is ``tool_names``)."""
        out: list[str] = []
        for s in self.mcp_servers:
            server = s.get("name", "")
            for t in s.get("tools", []) or []:
                tname = t.get("name") if isinstance(t, dict) else t
                if tname:
                    out.append(f"mcp__{server}__{tname}")
        return out

    def tool_was_registered(self, tool: str) -> bool | None:
        """Was *tool* (e.g. ``mcp__controls__channel_write``) exposed by the MCP
        handshake? ``None`` if no snapshot is available (cannot tell)."""
        if not self.mcp_servers:
            return None
        return tool in self.registered_tools

    @property
    def repeated_tool_calls(self) -> dict[str, int]:
        """``{'<tool>::<input-digest>': count}`` for any call issued more than
        once. A high count on a delegation tool is the fingerprint of a
        non-convergence loop."""
        from collections import Counter

        keys = [
            f"{t.name}::{json.dumps(t.input, sort_keys=True, default=str)}"
            for t in self.tool_traces
        ]
        return {k: n for k, n in Counter(keys).items() if n > 1}

    @property
    def has_redelegation_loop(self) -> bool:
        """True if a subagent-spawning tool (``Agent``/``Task*``) was re-issued
        with identical input 3+ times — model non-convergence (a MODEL timeout),
        distinct from slow-but-progressing proxy latency (INFRA). Lets a
        ``resource_timeout`` failure be bucketed from the recorded artifact
        instead of guessed."""
        for key, n in self.repeated_tool_calls.items():
            name = key.split("::", 1)[0]
            if n >= 3 and (name == "Agent" or name.startswith("Task")):
                return True
        return False

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
        usage: dict[str, Any] | None = getattr(self.result, "usage", None)
        if not usage:
            return 0
        return (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )

    @property
    def output_tokens(self) -> int:
        """Total output tokens."""
        usage: dict[str, Any] | None = getattr(self.result, "usage", None)
        if not usage:
            return 0
        return int(usage.get("output_tokens", 0))

    @property
    def cache_read_tokens(self) -> int:
        """Cache-read input tokens (charged at reduced rate)."""
        usage: dict[str, Any] | None = getattr(self.result, "usage", None)
        if not usage:
            return 0
        return int(usage.get("cache_read_input_tokens", 0))

    @property
    def cache_creation_tokens(self) -> int:
        """Cache-creation input tokens."""
        usage: dict[str, Any] | None = getattr(self.result, "usage", None)
        if not usage:
            return 0
        return int(usage.get("cache_creation_input_tokens", 0))

    def tools_matching(self, substring: str) -> list[ToolTrace]:
        """Return all tool traces whose name contains *substring*."""
        return [t for t in self.tool_traces if substring in t.name]


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------


def resolve_default_model(project_dir: Path) -> str:
    """Resolve the haiku-tier model name for the given project.

    Reads ``config.yml`` and returns the provider's haiku-tier model id,
    falling back to the upstream Anthropic default when no spec is
    configured.

    Args:
        project_dir: Path to an initialized OSPREY project.

    Returns:
        Model identifier string suitable for passing to the Claude Agent SDK
        ``model=`` argument.
    """
    spec = _resolve_project_spec(project_dir)
    if spec is not None:
        return str(spec.tier_to_model.get("haiku", "claude-haiku-4-5-20251001"))
    return "claude-haiku-4-5-20251001"


def sdk_env(project_dir: Path | None = None, *, provider: str | None = None) -> dict[str, str]:
    """Return env overrides for the SDK subprocess.

    Always sets ``CLAUDECODE=""`` to bypass the nested-session guard
    (JavaScript treats "" as falsy). When ``project_dir`` is provided,
    also injects the project's resolved provider env block so the bundled
    CLI talks to the configured provider (cborg, als-apg, anthropic-direct)
    instead of falling through to ``api.anthropic.com``.

    Args:
        project_dir: Optional path to an initialized OSPREY project.  When
            omitted only the ``CLAUDECODE`` bypass is returned.
        provider: When given, overrides ``claude_code.provider`` in the loaded
            config before resolving — used by cross-provider model sweeps in
            the benchmark runner.  Has no effect when ``project_dir`` is
            ``None``.

    Returns:
        Env dict to merge into the SDK options ``env`` field.
    """
    env: dict[str, str] = {"CLAUDECODE": ""}
    if project_dir is not None:
        env.update(provider_env_for_project(project_dir, provider=provider))
    return env


def combined_text(result: SDKWorkflowResult) -> str:
    """Combine all text blocks and tool results into a single searchable string.

    Args:
        result: A completed SDK workflow result.

    Returns:
        Lower-cased concatenation of all assistant text blocks and every
        non-empty tool-result string, joined by spaces.
    """
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


def _ingest_tool_result(block: ToolResultBlock, pending_tools: dict[str, ToolTrace]) -> None:
    """Match a ToolResultBlock to its pending ToolTrace and populate result/is_error.

    ToolResultBlocks arrive in ``UserMessage.content`` (per Anthropic API
    contract: tool_use is assistant output, tool_result is user input back to
    the model). They may also appear in ``AssistantMessage`` when the SDK
    forwards them.

    Args:
        block: The tool result block to ingest.
        pending_tools: Map of tool_use_id to the ToolTrace awaiting its result.
    """
    matched = pending_tools.get(block.tool_use_id)
    if matched is None:
        return
    if isinstance(block.content, str):
        matched.result = block.content
    elif isinstance(block.content, list):
        texts = []
        for item in block.content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        matched.result = "\n".join(texts) if texts else str(block.content)
    matched.is_error = bool(block.is_error)


# ---------------------------------------------------------------------------
# MCP readiness barrier + instrumentation
# ---------------------------------------------------------------------------
#
# OSPREY MCP servers register ASYNCHRONOUSLY: the controls stdio subprocess does
# heavyweight cold-start work (config prime, connector registration, tool-module
# imports) and only finishes its MCP handshake ~1–1.5s after the CLI launches —
# noticeably slower than the python/osprey_workspace servers. If the agent's first
# turn fires before that handshake completes, the controls tools are simply not in
# its toolset, and a less-persistent agent reports "no controls server connected"
# and gives up. That is a cold-start race, NOT a model capability gap — yet it is
# scored as a failure, and it cannot be distinguished from a genuine model give-up
# from the persisted transcript (which does not record MCP status).
#
# Both problems are solved with the SDK's own ``ClaudeSDKClient.get_mcp_status()``:
#   * READINESS — poll it until the expected servers are ``connected`` before sending
#     the prompt, so every agent gets a ready toolset (the harness-enforced equivalent
#     of the CLI's ``WaitForMcpServers`` tool, independent of whether the model thinks
#     to call it).
#   * INSTRUMENTATION — persist the final snapshot so a missing tool is provably INFRA
#     (server never registered) vs MODEL (tool was there, agent ignored it).

# The default ceiling generously covers the measured ~1.5s controls cold start with
# headroom for a loaded box. Overridable via env for slow CI hosts.
_MCP_READY_TIMEOUT_S = float(os.environ.get("OSPREY_E2E_MCP_READY_TIMEOUT", "20"))
_MCP_READY_POLL_S = 0.3


def _expected_mcp_servers(project_dir: Path) -> set[str]:
    """The MCP server names a project declares in ``.mcp.json`` — the set the
    readiness barrier waits for. Returns an empty set if the file is unreadable."""
    try:
        cfg = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(cfg.get("mcpServers", {}).keys())


async def _await_mcp_ready(
    client: ClaudeSDKClient,
    expected: set[str],
    *,
    timeout_s: float = _MCP_READY_TIMEOUT_S,
    poll_s: float = _MCP_READY_POLL_S,
) -> list[Any]:
    """Poll ``get_mcp_status()`` until every server in *expected* reports
    ``connected`` (or *timeout_s* elapses), then return the final snapshot.

    Resilient to ``get_mcp_status()`` raising early in startup (before the stream
    is live). Never raises: on timeout it returns the last snapshot seen so the
    caller can record a genuine registration failure rather than masking it.
    """
    deadline = time.monotonic() + timeout_s
    servers: list[Any] = []
    while True:
        try:
            status = await client.get_mcp_status()
            servers = status.get("mcpServers", []) if isinstance(status, dict) else (status or [])
        except Exception:  # noqa: BLE001 — status not queryable yet; keep polling
            servers = servers or []
        if expected:
            connected = {s.get("name") for s in servers if s.get("status") == "connected"}
            if expected <= connected:
                return servers
        if time.monotonic() >= deadline:
            return servers
        await asyncio.sleep(poll_s)
