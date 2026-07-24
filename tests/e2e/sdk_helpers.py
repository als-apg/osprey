"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and project initialization
utilities used by both the functional SDK tests and the safety E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# SDK imports — skip entire module if not installed
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        PermissionResultAllow,
        PermissionResultDeny,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolPermissionContext,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

# Sub-agent transcript readers — added in SDK 0.1.46, present in 0.2.87.
# Claude Code CLI >= 2.1.x no longer streams sub-agent messages through the
# ``query()`` iterator; they are written to side files under
# ``~/.claude/projects/<proj>/<session>/subagents/agent-*.jsonl``. These
# helpers parse those files so delegation traces are observable again.
try:
    from claude_agent_sdk import get_subagent_messages, list_subagents

    HAS_SUBAGENT_READERS = True
except ImportError:
    HAS_SUBAGENT_READERS = False


# ---------------------------------------------------------------------------
# Shared primitives — imported from the production package so there is a
# single source of truth; sdk_helpers re-exports them so test modules that
# import from here continue to work unchanged.
# ---------------------------------------------------------------------------
from osprey.agent_runner import (
    SDKWorkflowResult,
    ToolTrace,
    _await_mcp_ready,
    _expected_mcp_servers,
    resolve_default_model,
    sdk_env,
)
from osprey.agent_runner import (
    combined_text as combined_text,  # re-exported for other e2e test modules
)
from osprey.agent_runner.primitives import (
    _ingest_tool_result,
    _resolve_project_spec,
)
from osprey.agent_runner.primitives import (
    provider_env_for_project as provider_env_for_project,  # re-exported for e2e tests
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is installed and functional."""
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def has_anthropic_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_als_apg_api_key() -> bool:
    """Check if ALS_APG_API_KEY is set.

    The CI-default Bedrock proxy at llm.gianlucamartino.com authenticates
    with this token; the safety/SDK E2E suite skip-gates on it because the
    Claude Code CLI subprocess uses the proxy via the project's `.env` and
    `provider=als-apg` defaults landed in 8c541cc9.
    """
    return bool(os.environ.get("ALS_APG_API_KEY"))


_DEFAULT_ARIEL_DB_URI = "postgresql://ariel:ariel@localhost:5432/ariel"


def ariel_db_skip_reason(uri: str | None = None) -> str | None:
    """Return an actionable skip reason if the ARIEL Postgres is not ready.

    The scenario E2E tests (rf_cavity correlation, sector-7 vacuum burst,
    corrector honest-refusal) drive the logbook-search sub-agent, which needs a
    live, *seeded* ARIEL Postgres. When it is absent, every ARIEL search fails
    with a 5-second pool-open timeout buried inside the sub-agent; the test then
    fails on a downstream tool-trace assertion that *looks like* a model
    capability miss ("the agent couldn't find the cavity"). That false signal
    cost hours once. This guard converts the silent prerequisite into an
    explicit, actionable skip.

    Returns ``None`` when the DB is reachable and has at least one entry,
    otherwise a human-readable reason string suitable for ``pytest.skip``.
    Override the URI with ``OSPREY_ARIEL_DB_URI``.
    """
    uri = uri or os.environ.get("OSPREY_ARIEL_DB_URI", _DEFAULT_ARIEL_DB_URI)
    try:
        import psycopg
    except ImportError:
        return "psycopg not installed — cannot verify the ARIEL DB prerequisite"
    try:
        with psycopg.connect(uri, connect_timeout=3) as conn:
            row = conn.execute("SELECT count(*) FROM enhanced_entries").fetchone()
    except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
        return (
            f"ARIEL Postgres not reachable ({exc.__class__.__name__}) — scenario "
            "tests need a live, seeded ARIEL logbook DB. Bring it up with "
            "`osprey deploy up && osprey ariel migrate && osprey ariel quickstart`."
        )
    count = row[0] if row else 0
    if count == 0:
        return "ARIEL Postgres is reachable but empty — seed it with `osprey ariel quickstart`."
    return None


def _override_ariel_db_uri(project_dir: Path) -> None:
    """Point a freshly built project at the per-cell ARIEL database.

    When ``OSPREY_ARIEL_DB_URI`` is set (the matrix runner provisions one
    database per (model, seed) cell), rewrite the rendered ``config.yml`` so the
    agent's ARIEL MCP server *and* ``apply_scenarios`` both talk to the per-cell
    DB instead of the shared default. This is what makes concurrent cells
    isolated: a scenario test in one cell purges only its own DB and can no
    longer drop another cell's ``text_embeddings_*`` tables mid-test.

    The default URI is a hardcoded literal in the template (not a profile
    variable), so ``--set ariel.database.uri`` would not reach the rendered
    config — a post-build text substitution is the reliable hook.

    Projects that do not use a real ARIEL Postgres DB (e.g. the ``hello_world``
    preset renders ``ariel: {enabled: false}`` and no DB URI) have nothing to
    redirect, so the default URI is simply absent and this is a no-op. Template
    *drift* for ARIEL-using presets is caught loudly elsewhere: the matrix
    runner's per-cell provisioning patches a freshly built control-assistant
    config and asserts the default URI is present before it can seed the DB.
    """
    override = os.environ.get("OSPREY_ARIEL_DB_URI")
    if not override or override == _DEFAULT_ARIEL_DB_URI:
        return
    config_path = project_dir / "config.yml"
    text = config_path.read_text(encoding="utf-8")
    if _DEFAULT_ARIEL_DB_URI not in text:
        return
    config_path.write_text(text.replace(_DEFAULT_ARIEL_DB_URI, override), encoding="utf-8")


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    *,
    provider: str,
    model: str = "haiku",
    channel_finder_mode: str | None = None,
    tier: int | None = None,
) -> Path:
    """Create a project via ``osprey build --preset <template>``, return project_dir.

    Tier selection follows a per-mode default: tier 1 is in_context-only, while
    ``hierarchical``/``middle_layer`` require tier 3. When ``tier`` is left
    ``None`` and a ``channel_finder_mode`` is given, the tier is derived from it
    (in_context → 1, else → 3); when neither is given, ``--tier`` is omitted and
    the build derives the tier from the preset's own paradigm. An explicit
    ``tier`` kwarg is always honored. Consequence: hierarchical/middle_layer
    callers now score the full tier-3 (2908-channel) surface, not a tier-1
    subset.

    ``provider`` is required (keyword-only) — every test callsite must name
    it explicitly. Each provider gates on different credentials (CBORG needs
    LBLnet/VPN; als-apg needs ``ALS_APG_API_KEY``; anthropic-direct needs
    ``ANTHROPIC_API_KEY``), so a kwarg default silently couples tests to one
    provider's auth and produces the local-passes-CI-fails asymmetry. Pick
    ``"als-apg"`` for GitHub Actions runners, ``"cborg"`` from LBLnet, or
    ``"anthropic"`` when you have an ``ANTHROPIC_API_KEY`` available.

    Invoked via ``subprocess`` rather than Click's ``CliRunner`` because
    ``osprey build`` instantiates ``rich.Console(force_terminal=True)``,
    which performs terminal-aware lifecycle management on the captured
    ``BytesIO`` stream that ``CliRunner`` substitutes for stdout. On
    Python ≥3.11 that closes the wrapper before Click reads it back,
    raising ``ValueError: I/O operation on closed file`` at fixture
    setup. ``CliRunner`` is also a unit-test harness; an e2e fixture
    should exercise the same entry point real users invoke.

    Suite-wide override (CBORG model-matrix, issue #259): when
    ``OSPREY_E2E_FORCE_PROVIDER`` is set it replaces the per-callsite
    ``provider`` so the *entire* tests/e2e/ suite can be pointed at one
    provider without editing each fixture. Paired with
    ``OSPREY_E2E_FORCE_MODEL`` (honored in ``_resolve_project_spec``), which
    collapses all tiers onto a single model id.
    """
    from osprey.build.build_tiers import default_tier_for_mode

    provider = os.environ.get("OSPREY_E2E_FORCE_PROVIDER", provider)
    effective_tier = tier
    if effective_tier is None and channel_finder_mode is not None:
        effective_tier = default_tier_for_mode(channel_finder_mode)
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
    if effective_tier is not None:
        args.extend(["--tier", str(effective_tier)])
    if channel_finder_mode is not None:
        args.extend(["--set", f"channel_finder_mode={channel_finder_mode}"])
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "build", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"osprey build failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    project_dir = tmp_path / name
    assert project_dir.exists(), f"Project directory not created: {project_dir}"
    _override_ariel_db_uri(project_dir)
    return project_dir


def enable_writes_in_project(project_dir: Path) -> None:
    """Ensure ``control_system.writes_enabled`` is true in the project's config.yml.

    Required for tests that exercise the approval-hook path: the
    ``osprey_writes_check.py`` PreToolUse hook denies before
    ``osprey_approval.py`` gets to return ``ask``, so the SDK's
    ``can_use_tool`` callback never fires when writes are disabled.

    The kill-switch hard-block in ``cli/templates/claude_code.py`` bakes
    ``mcp__controls__channel_write`` into ``settings.json``'s ``permissions.deny``
    at build time when ``writes_enabled`` is false (so Claude Code's permissions
    layer short-circuits before the PreToolUse hook chain runs). Flipping
    ``config.yml`` alone leaves the rendered ``settings.json`` stale, so after the
    flip we regenerate the Claude Code artifacts — exactly the production fix
    (``osprey claude regen`` / the web + CLI auto-regen) rather than a hand-patch.

    Idempotent: presets like ``control_assistant`` already ship with
    ``writes_enabled: true``; only ``hello_world`` defaults to false.
    """
    config_path = project_dir / "config.yml"
    text = config_path.read_text(encoding="utf-8")
    if "writes_enabled: true" not in text:
        updated = text.replace("writes_enabled: false", "writes_enabled: true", 1)
        if updated == text:
            raise RuntimeError(
                f"Could not enable writes in {config_path}: no writes_enabled key found."
            )
        config_path.write_text(updated, encoding="utf-8")

    # Re-render artifacts so the stale settings.json deny entry is dropped.
    from osprey.cli.templates.manager import TemplateManager

    TemplateManager().regen_if_drift(project_dir)


def activate_scenario(project_dir: Path, scenario: str) -> None:
    """Activate a single scenario's telemetry overlay (no logbook seeding).

    Writes the scenario name into ``data/simulation/active_scenarios``; the
    simulation engine re-reads the state file on mtime change and clears any
    session writes (fresh machine state). Telemetry only — for scenarios whose
    diagnosis needs a seeded logbook, use :func:`activate_scenarios`, which also
    purges and reseeds ARIEL deterministically.
    """
    state_file = project_dir / "data" / "simulation" / "active_scenarios"
    assert state_file.exists(), (
        f"active_scenarios state file missing at {state_file} — "
        "template simulation overlay incomplete?"
    )
    state_file.write_text(scenario + "\n", encoding="utf-8")


def activate_scenarios(project_dir: Path, *names: str, now=None):
    """Compose and apply scenarios, seeding their logbook into ARIEL.

    Calls :func:`osprey.simulation.apply.apply_scenarios` with
    ``seed_logbook=True``: it writes the active-scenario state (with a shared
    apply-time anchor) and purges + reseeds the ARIEL logbook from the active
    scenarios' own entries. This replaces the manual ``purge && ingest``
    pre-seed and removes the stale-DB footgun (the logbook always matches the
    active telemetry, against one clock). ``nominal`` is always implicit, so its
    ambient entries are present even for telemetry-only faults.

    Returns the :class:`~osprey.simulation.apply.ApplyResult`.
    """
    from osprey.simulation.apply import apply_scenarios

    return apply_scenarios(project_dir, list(names), seed_logbook=True, now=now)


def _default_opus_model(project_dir: Path) -> str:
    """Resolve the project's opus-tier model name.

    Use for tests that benchmark agent reasoning (diagnostic-style
    challenges) — Opus is required for the planner to converge on a
    committed conclusion instead of hedging on a data dump.
    """
    spec = _resolve_project_spec(project_dir)
    if spec is not None:
        return spec.tier_to_model.get("opus", "claude-opus-4-7")
    return "claude-opus-4-7"


def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


def find_html_files(root: Path) -> list[Path]:
    """Recursively find all .html files under *root*, excluding index.html."""
    return sorted(p for p in root.rglob("*.html") if p.name != "index.html")


def read_audit_events(project_dir: Path) -> list[dict]:
    """Read OSPREY tool-call events from Claude Code native transcripts.

    Uses TranscriptReader to extract events from the most recent transcript
    in ``~/.claude/projects/<encoded>/``.

    Returns:
        List of event dicts (tool_call, agent_start, agent_stop).
    """
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    reader = TranscriptReader(project_dir)
    return reader.read_current_session()


# ---------------------------------------------------------------------------
# MCP sidecar + core runner support helpers
# ---------------------------------------------------------------------------


def _persist_mcp_sidecar(workflow: SDKWorkflowResult, project_dir: Path) -> None:
    """Write the MCP-status snapshot to a per-test sidecar when
    ``OSPREY_E2E_INIT_SIDECAR`` is set. Off by default, so ordinary CI/local runs
    are byte-for-byte unchanged. The sidecar turns the infra-vs-model question
    into a recorded fact for post-hoc benchmark forensics."""
    if not os.environ.get("OSPREY_E2E_INIT_SIDECAR"):
        return
    try:
        out_dir = project_dir / ".osprey_e2e"
        out_dir.mkdir(exist_ok=True)
        payload = {
            "mcp_server_status": workflow.mcp_server_status,
            "registered_tools": workflow.registered_tools,
            "tools_called": workflow.tool_names,
            "num_turns": workflow.num_turns,
            "cost_usd": workflow.cost_usd,
            "repeated_tool_calls": workflow.repeated_tool_calls,
            "has_redelegation_loop": workflow.has_redelegation_loop,
        }
        (out_dir / "mcp_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # instrumentation must never fail a test


def _result_text(content: Any) -> str:
    """Flatten a tool_result ``content`` field (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(texts) if texts else str(content)
    return "" if content is None else str(content)


def _harvest_subagent_traces(
    workflow: SDKWorkflowResult,
    pending_tools: dict[str, ToolTrace],
    project_dir: Path,
) -> None:
    """Append sub-agent tool calls that never streamed through ``query()``.

    Claude Code CLI >= 2.1.x writes sub-agent transcripts to side files rather
    than streaming them through the SDK iterator. We read them back via the
    SDK's ``list_subagents`` / ``get_subagent_messages`` helpers and append any
    tool calls not already captured (deduped by ``tool_use_id``), tagging each
    with a non-``None`` ``parent_tool_use_id`` so delegation tests can tell
    sub-agent activity apart from main-agent activity.

    Best-effort: a parsing failure here must not fail an otherwise-successful
    run, so the caller still sees whatever the stream yielded.
    """
    if not HAS_SUBAGENT_READERS or workflow.result is None:
        return
    session_id = workflow.result.session_id
    if not session_id:
        return

    directory = str(project_dir)
    try:
        agent_ids = list_subagents(session_id, directory=directory)
    except Exception:
        return

    for agent_id in agent_ids:
        try:
            messages = get_subagent_messages(session_id, agent_id, directory=directory)
        except Exception:
            continue
        for sm in messages:
            msg = getattr(sm, "message", None)
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            # A sub-agent message has no parent_tool_use_id of its own at the
            # top level for tool_use blocks; fall back to the agent id so the
            # trace is always attributable to a sub-agent (non-None).
            parent_id = getattr(sm, "parent_tool_use_id", None) or agent_id
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    tool_id = block.get("id")
                    if tool_id in pending_tools:
                        continue  # already captured via the stream
                    trace = ToolTrace(
                        name=block.get("name", ""),
                        input=block.get("input", {}) or {},
                        tool_use_id=tool_id,
                        parent_tool_use_id=parent_id,
                    )
                    workflow.tool_traces.append(trace)
                    if tool_id is not None:
                        pending_tools[tool_id] = trace
                elif btype == "tool_result":
                    matched = pending_tools.get(block.get("tool_use_id"))
                    if matched is not None:
                        matched.result = _result_text(block.get("content"))
                        matched.is_error = bool(block.get("is_error"))


def e2e_budget_scale() -> float:
    """Per-query budget multiplier for the model under test.

    The base ``max_budget_usd`` caps across the e2e suite are tuned for the
    haiku-tier default model. Pricier reference models (Sonnet, Opus) cost
    several times more per token, so the same multi-step task blows the cap and
    hard-errors mid-query (``Reached maximum budget``) — a cost artifact that
    deflates their benchmark score for reasons unrelated to capability. The
    model matrix runner (``scripts/run_e2e_for_model.sh``) sets
    ``OSPREY_E2E_BUDGET_SCALE`` per model so the cap **and** the cost-ceiling
    assertions scale together. Defaults to 1.0, so CI and ordinary local runs
    are byte-for-byte unchanged.
    """
    try:
        scale = float(os.environ.get("OSPREY_E2E_BUDGET_SCALE", "1.0"))
    except ValueError:
        return 1.0
    return scale if scale > 0 else 1.0


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
    disallowed_tools: list[str] | None = None,
) -> SDKWorkflowResult:
    """Run a query via the Claude Agent SDK and collect full tool traces.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use. Defaults to the project's haiku-tier model
            resolved from ``config.yml`` (e.g. ``claude-haiku-4-5`` for
            cborg, ``claude-haiku-4-5-20251001`` for direct anthropic).
        disallowed_tools: Optional list of tool names to forbid at the SDK
            level. Forwarded to the Claude Code CLI as ``--disallowedTools``,
            which takes precedence over ``permission_mode=bypassPermissions``
            and over per-tool ``permissions_allow`` in ``.mcp.json``. Use this
            to architecturally force delegation to subagents (the main agent
            cannot call a disallowed tool even when settings would permit it).

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.
    """
    # Collect stderr lines for debugging CLI failures
    stderr_lines: list[str] = []

    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(project_dir),
        cwd=str(project_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd * e2e_budget_scale(),
        env=sdk_env(project_dir),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        disallowed_tools=disallowed_tools or [],
    )

    workflow = SDKWorkflowResult()

    # Map tool_use_id → ToolTrace for matching results to calls
    pending_tools: dict[str, ToolTrace] = {}

    try:
        # ClaudeSDKClient (streaming) rather than the one-shot ``query()`` so we can
        # poll ``get_mcp_status()`` and wait out async MCP registration before the
        # first turn — eliminating the controls cold-start race (see _await_mcp_ready).
        # Message handling is identical to the query() iterator.
        async with ClaudeSDKClient(options=options) as client:
            workflow.mcp_servers = await _await_mcp_ready(
                client, _expected_mcp_servers(project_dir)
            )
            await client.query(prompt)
            async for message in client.receive_response():
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
                            _ingest_tool_result(block, pending_tools)

                elif isinstance(message, UserMessage):
                    # Tool results land here per the Anthropic API contract.
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                _ingest_tool_result(block, pending_tools)

                elif isinstance(message, SystemMessage):
                    workflow.system_messages.append(message)

                elif isinstance(message, ResultMessage):
                    workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    # Sub-agent tool calls don't stream through query() on CLI >= 2.1.x; read
    # them from the on-disk transcripts so delegation tests can observe them.
    _harvest_subagent_traces(workflow, pending_tools, project_dir)

    _persist_mcp_sidecar(workflow, project_dir)
    return workflow


# ---------------------------------------------------------------------------
# Hook-observed SDK runner (uses can_use_tool callback)
# ---------------------------------------------------------------------------


@dataclass
class HookEvent:
    """Record of a permission callback invocation (hook returned 'ask')."""

    tool_name: str
    tool_input: dict
    decision: str  # "allow" or "deny"
    reason: str | None = None


@dataclass
class HookObservedResult(SDKWorkflowResult):
    """Extends SDKWorkflowResult with hook observability."""

    hook_events: list[HookEvent] = field(default_factory=list)


async def run_sdk_query_with_hooks(
    project_dir: Path,
    prompt: str,
    *,
    approval_policy: Callable[[str, dict[str, Any]], bool] | str = "auto_approve",
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str | None = None,
) -> HookObservedResult:
    """Run a query via the Claude Agent SDK with hooks enabled and can_use_tool callback.

    Unlike ``run_sdk_query`` (which uses bypassPermissions), this function uses
    ``permission_mode="default"`` so that file-system hooks actually execute.
    When a hook returns ``permissionDecision: "ask"``, the ``can_use_tool``
    callback is invoked instead of prompting a human.

    The ``approval_policy`` controls what happens when a hook returns "ask":
    - ``"auto_approve"`` — always approve (hooks still run, but "ask" → allow)
    - ``"auto_deny"`` — always deny (test that denial propagates correctly)
    - callable — custom ``(tool_name, tool_input) -> bool`` for fine-grained control

    Every callback invocation is recorded in ``hook_events`` for observability.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        approval_policy: How to handle "ask" decisions from hooks.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use. Defaults to the project's haiku-tier model
            resolved from ``config.yml``.

    Returns:
        HookObservedResult with tool traces, text, metadata, and hook events.
    """
    hook_events: list[HookEvent] = []
    stderr_lines: list[str] = []

    async def _can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Permission callback: record the event and apply the approval policy."""
        if approval_policy == "auto_approve":
            should_allow = True
        elif approval_policy == "auto_deny":
            should_allow = False
        elif callable(approval_policy):
            should_allow = approval_policy(tool_name, tool_input)
        else:
            raise ValueError(f"Invalid approval_policy: {approval_policy!r}")

        decision = "allow" if should_allow else "deny"
        event = HookEvent(
            tool_name=tool_name,
            tool_input=tool_input,
            decision=decision,
            reason=f"approval_policy={approval_policy!r}"
            if isinstance(approval_policy, str)
            else "custom_policy",
        )
        hook_events.append(event)

        if should_allow:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Denied by test approval policy")

    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(project_dir),
        cwd=str(project_dir),
        permission_mode="default",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd * e2e_budget_scale(),
        env=sdk_env(project_dir),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        can_use_tool=_can_use_tool,
    )

    workflow = HookObservedResult()

    # Map tool_use_id → ToolTrace for matching results to calls
    pending_tools: dict[str, ToolTrace] = {}

    try:
        # ClaudeSDKClient is required for can_use_tool (streaming mode).
        # The simple query() function does not support permission callbacks.
        async with ClaudeSDKClient(options=options) as client:
            # Wait out async MCP registration (controls cold-starts ~1.5s) so the
            # agent never races a half-built toolset. Snapshot is the authoritative
            # infra-vs-model record.
            workflow.mcp_servers = await _await_mcp_ready(
                client, _expected_mcp_servers(project_dir)
            )
            await client.query(prompt)
            async for message in client.receive_response():
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
                            _ingest_tool_result(block, pending_tools)

                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                _ingest_tool_result(block, pending_tools)

                elif isinstance(message, SystemMessage):
                    workflow.system_messages.append(message)

                elif isinstance(message, ResultMessage):
                    workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    # See run_sdk_query: sub-agent tool calls live in on-disk transcripts.
    _harvest_subagent_traces(workflow, pending_tools, project_dir)

    workflow.hook_events = hook_events
    _persist_mcp_sidecar(workflow, project_dir)
    return workflow
