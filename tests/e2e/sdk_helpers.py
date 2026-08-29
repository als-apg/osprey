"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and deployment-repo
initialization utilities used by both the functional SDK tests and the safety
E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.

**Zones.** Every public helper here takes the *deployment repo root* — what
:func:`init_project` returns and what ``osprey init`` creates — so a test
carries one handle. A repo has three zones the helpers resolve internally:

* ``<repo>/`` — SOURCE: ``profile.yml``, the operator-owned ``data/`` tree
  (including ``data/simulation/``), ``personas/``, and the ``.env`` secret store.
* ``<repo>/build/`` — the RENDER (see :func:`render_dir`): ``config.yml``,
  ``.mcp.json``, ``CLAUDE.md``, ``.claude/``, and the render's own ``data/``
  copy. This is the Claude Code agent's working directory.
* ``<repo>/var/`` — STATE: ``agent_data/`` (agent memory, artifacts, the
  simulation's ``active_scenarios`` file) and ``audit/``.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tests import ci_diagnostics

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
    await_mcp_ready,
    expected_mcp_servers,
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
from osprey.utils.workspace import (
    BUILD_DIR_NAME,
    DEFAULT_AGENT_DATA_BASE_DIR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_dir(repo: Path) -> Path:
    """The render zone of the deployment repo at *repo* — ``<repo>/build``.

    Everything ``osprey build`` produces lands here: ``config.yml``,
    ``.mcp.json``, ``CLAUDE.md``, ``.claude/`` and the render's ``data/`` copy.
    It is also the Claude Code agent's working directory. Spelled once, through
    the same constant the framework renders against, so the zone split cannot
    drift between this module and the build.
    """
    return Path(repo) / BUILD_DIR_NAME


def agent_data_dir(repo: Path) -> Path:
    """The state zone's agent-data root — ``<repo>/var/agent_data``.

    Agent memory, artifacts and the simulation's mutable ``active_scenarios``
    file live under ``var/`` so they survive the wholesale re-creation of
    ``build/`` that every build performs.
    """
    return Path(repo) / DEFAULT_AGENT_DATA_BASE_DIR


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
            "`osprey up && osprey ariel migrate && osprey ariel quickstart`."
        )
    count = row[0] if row else 0
    if count == 0:
        return "ARIEL Postgres is reachable but empty — seed it with `osprey ariel quickstart`."
    return None


def _override_ariel_db_uri(render: Path) -> None:
    """Point a freshly built deployment at the per-cell ARIEL database.

    Takes the RENDER directory (``<repo>/build``) — the rewrite targets the
    rendered ``config.yml``, which is what every MCP server is pointed at via
    ``CONFIG_FILE`` and what ``apply_scenarios`` loads.

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
    config_path = render / "config.yml"
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
    connector: str = "mock",
    archiver: str = "mock_archiver",
) -> Path:
    """Create a deployment repo at ``tmp_path/name`` and build it; return the repo root.

    Two commands, because the surface has two: ``osprey init <dir> --preset P``
    writes the repo's source zone (``profile.yml``, ``data/``, ``personas/``,
    ``.env.example``), and ``osprey build --repo <dir>`` renders ``build/`` from
    it. The directory name IS the deployment name.

    Returns the REPO ROOT, not the render: it is what
    :func:`osprey.simulation.apply.apply_scenarios` takes, it is the
    ``project_root`` the rendered config records, and ``var/agent_data`` hangs
    off it. Every helper in this module takes that same handle and resolves the
    render itself via :func:`render_dir`.

    ``--no-git`` is always passed: no test reads the repo's history and
    ``git init`` is pure latency here.

    ``connector`` is pinned to ``mock`` rather than inherited from the preset:
    the control-assistant preset baselines on its live stand-in, a deployed
    soft IOC that has to answer Channel Access — this harness runs projects
    without their containers, so the preset's production default would turn
    every channel read/write into a connection timeout. Tests that deploy a
    real stack build through their own fixtures, not this helper.

    ``archiver`` is pinned for the same reason and is the archive half of that
    same fact: the preset selects ``mongodb_archiver`` and declares the
    ``va_archiver:`` block that deploys the store it reads, so a containerless
    build would leave every ``archiver_read`` failing at connect for want of a
    store — and, before that, for want of the password ``osprey up``
    mints. Pinning both halves to the mock is not a way around the pairing rule
    in :mod:`osprey.connectors.honesty` but the case it explicitly allows: a
    mock control system with the mock archiver claims nothing is real, so
    nothing lies. Tests that want recorded history deploy a store of their own.

    ``virtual_accelerator.live_standin`` is nulled as the third of those pins.
    The preset ships a live stand-in on, which is a second VA container this
    harness never starts — and, before any read of it times out, the build
    derives the posture that container is meant to be met with: the ``epics``
    gateways move to it and limits checking goes strict, so a write to a
    channel absent from a test's own limits fixture is refused where it used
    to pass. A project that wants the stand-in deploys it.

    Tier selection follows a per-mode default: tier 1 is in_context-only, while
    every other paradigm requires tier 3. When ``tier`` is left ``None`` and a
    ``channel_finder_mode`` is given, the tier is derived from it (in_context
    → 1, else → 3); when neither is given, ``tier`` is left out of the profile
    and the build derives it from the preset's own paradigm. An explicit
    ``tier`` kwarg is always honored. Consequence: hierarchical/middle_layer
    callers score the full tier-3 (2908-channel) surface, not a tier-1 subset.
    The tier is a profile field, so it is set the same way as every other one:
    ``--set tier=N`` on ``init``.

    A paradigm whose store is a service rather than tiered database files
    (``graph``) has no tier to select, so the derived tier is dropped for it
    and the profile is written without a ``tier`` field. The rule is read from
    ``tier_mode_conflict`` rather than restated here.

    ``provider`` is required (keyword-only) — every test callsite must name
    it explicitly. Each provider gates on different credentials (CBORG needs
    LBLnet/VPN; als-apg needs ``ALS_APG_API_KEY``; anthropic-direct needs
    ``ANTHROPIC_API_KEY``), so a kwarg default silently couples tests to one
    provider's auth and produces the local-passes-CI-fails asymmetry. Pick
    ``"als-apg"`` for GitHub Actions runners, ``"cborg"`` from LBLnet, or
    ``"anthropic"`` when you have an ``ANTHROPIC_API_KEY`` available.

    Both commands are invoked via ``subprocess`` rather than Click's
    ``CliRunner`` because they instantiate ``rich.Console(force_terminal=True)``,
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
    from osprey.build.build_tiers import default_tier_for_mode, tier_mode_conflict

    provider = os.environ.get("OSPREY_E2E_FORCE_PROVIDER", provider)
    effective_tier = tier
    if effective_tier is None and channel_finder_mode is not None:
        derived = default_tier_for_mode(channel_finder_mode)
        # Pin the derived tier only where the paradigm accepts one. A paradigm
        # backed by a service rather than tiered database files has no tier to
        # select, and ``tier_mode_conflict`` is the registry's own statement of
        # which pairings hold — asking it keeps the rule in one place instead of
        # re-listing paradigms here.
        if tier_mode_conflict(derived, channel_finder_mode) is None:
            effective_tier = derived
    repo = tmp_path / name
    init_args = [
        str(repo),
        "--preset",
        template.replace("_", "-"),
        "--no-git",
        "--set",
        f"provider={provider}",
        "--set",
        f"model={model}",
        "--set",
        f"connector={connector}",
    ]
    # An override FILE, not ``--set config.archiver.type=``: the preset spells
    # the key in its literal dotted form, and a ``--set`` nested path would
    # merge a second, competing ``archiver`` mapping alongside it whose winner
    # is decided by key order — which the build refuses outright rather than
    # render. A ``-O`` layer replaces the dotted key in the spelling the
    # profile already uses.
    preset_pins = tmp_path / "_archiver-pin.yml"
    preset_pins.write_text(
        f"config:\n  archiver.type: {archiver}\nvirtual_accelerator:\n  live_standin: null\n",
        encoding="utf-8",
    )
    init_args.extend(["-O", str(preset_pins)])
    if effective_tier is not None:
        init_args.extend(["--set", f"tier={effective_tier}"])
    if channel_finder_mode is not None:
        init_args.extend(["--set", f"channel_finder_mode={channel_finder_mode}"])
    _run_osprey("init", init_args, timeout=180)
    _run_osprey(
        "build",
        ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        timeout=300,
    )

    assert repo.is_dir(), f"Deployment repo not created: {repo}"
    render = render_dir(repo)
    assert (render / "config.yml").is_file(), f"Build produced no render at {render}"
    _override_ariel_db_uri(render)
    return repo


def _run_osprey(verb: str, args: list[str], *, timeout: int) -> None:
    """Run one ``osprey`` verb, failing loudly with both streams on a non-zero exit."""
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", verb, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"osprey {verb} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def _flip_writes_enabled(text: str) -> str:
    """Set the real ``writes_enabled`` key to true, leaving comments alone.

    Returns the text unchanged when no such key line exists, which the caller
    treats as an error.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        code = line.split("#", 1)[0]
        if "writes_enabled:" in code and "false" in code:
            lines[i] = line.replace("false", "true", 1)
            return "".join(lines)
    return text


def enable_writes_in_project(repo: Path) -> None:
    """Ensure ``control_system.writes_enabled`` is true in the rendered config.

    Takes the REPO ROOT; the flip lands in the render's ``config.yml``, which is
    what the MCP servers read (``CONFIG_FILE=<repo>/build/config.yml``).

    Required for tests that exercise the approval-hook path: the
    ``osprey_writes_check.py`` PreToolUse hook denies before
    ``osprey_approval.py`` gets to return ``ask``, so the SDK's
    ``can_use_tool`` callback never fires when writes are disabled.

    The kill-switch hard-block in ``cli/templates/claude_code.py`` bakes
    ``mcp__controls__channel_write`` into ``settings.json``'s ``permissions.deny``
    at build time when ``writes_enabled`` is false (so Claude Code's permissions
    layer short-circuits before the PreToolUse hook chain runs). Flipping
    ``config.yml`` alone leaves the rendered ``settings.json`` stale, so after the
    flip we regenerate the Claude Code artifacts — exactly what a fresh
    ``osprey build`` (and the web + CLI auto-regen) does, rather than a
    hand-patch.

    Idempotent: presets like ``control_assistant`` already ship with
    ``writes_enabled: true``; only ``hello_world`` defaults to false.

    Both the "already enabled?" check and the flip read the parsed value and
    skip comments. A whole-file substring test cannot tell the live key from
    the same characters quoted in a comment, and the hello_world config
    comments quote ``control_system.writes_enabled: true`` to tell the reader
    what to uncomment in their profile — which would otherwise read as "writes
    are already on" and silently skip the flip.
    """
    render = render_dir(repo)
    config_path = render / "config.yml"
    text = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    if not (parsed.get("control_system") or {}).get("writes_enabled"):
        updated = _flip_writes_enabled(text)
        if updated == text:
            raise RuntimeError(
                f"Could not enable writes in {config_path}: no writes_enabled key found."
            )
        config_path.write_text(updated, encoding="utf-8")

    # Re-render artifacts so the stale settings.json deny entry is dropped.
    from osprey.cli.templates.manager import TemplateManager

    TemplateManager().regen_if_drift(render)


def activate_scenario(repo: Path, scenario: str) -> None:
    """Activate a single scenario's telemetry overlay (no logbook seeding).

    Takes the REPO ROOT. Writes the scenario name into
    ``<repo>/var/agent_data/simulation/active_scenarios`` — mutable state, which
    is why it lives under ``var/`` rather than in either ``data/`` tree; the
    simulation engine re-reads the state file on mtime change and clears any
    session writes (fresh machine state). Telemetry only — for scenarios whose
    diagnosis needs a seeded logbook, use :func:`activate_scenarios`, which also
    purges and reseeds ARIEL deterministically.
    """
    state_file = agent_data_dir(repo) / "simulation" / "active_scenarios"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(scenario + "\n", encoding="utf-8")


def activate_scenarios(repo: Path, *names: str, now=None):
    """Compose and apply scenarios, seeding their logbook into ARIEL.

    Takes the REPO ROOT — that is what
    :func:`osprey.simulation.apply.apply_scenarios` anchors on: the
    ``data/simulation/`` model and the ``var/agent_data/simulation/`` state both
    hang off it, and it reads the render's ``config.yml`` itself.

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

    return apply_scenarios(repo, list(names), seed_logbook=True, now=now)


# ---------------------------------------------------------------------------
# Agentic-scenario benchmark integrity
#
# A scenario benchmark asks the agent to *derive* a fault from instrument data.
# Its ground truth ships inside the deployment repo as
# ``data/simulation/scenarios/<name>/scenario.json``, whose ``description``
# names the seeded fault outright — and the agent's cwd IS the render, which
# carries its own copy of that tree. Left alone, the cheapest route to a correct
# answer is to search the tree and read the answer key, which produces a right
# answer by a route that proves nothing about the capability under test. The two
# helpers below close that route from both ends.
# ---------------------------------------------------------------------------

# Generic filesystem-search tools, forbidden at the SDK level for the duration
# of a scenario benchmark. Every framework subagent already declares exactly
# these in its own ``disallowedTools`` frontmatter; the MAIN agent is the only
# session participant that still carries them, and it is the one that goes
# looking. Repo convention is that framework agents get the python executor and
# never Bash, so removing Bash here also closes a hole the ``permissions.deny``
# list cannot: under ``permission_mode="bypassPermissions"`` (what
# :func:`run_sdk_query` uses) the deny list is bypassed, while SDK-level
# ``disallowed_tools`` still takes precedence.
#
# ``Read`` is deliberately NOT in this list: ``data-visualizer`` and
# ``pyat-specialist`` declare it for agent-data artifacts, and disallowing
# a tool strips it from subagents too. Concealing the answer key (below) is what
# makes a bare ``Read`` harmless; this list is what stops the agent from finding
# anything worth reading in the first place.
SCENARIO_INTEGRITY_DISALLOWED_TOOLS = ["Bash", "Glob", "Grep"]


def conceal_scenario_ground_truth(repo: Path, *scenarios: str) -> None:
    """Delete the named scenarios' definition bundles from the deployment repo.

    Takes the REPO ROOT and scrubs BOTH simulation trees: the operator-owned
    source at ``<repo>/data/simulation`` (what the host-side engine resolves via
    ``project_root``) and the render's copy at ``<repo>/build/data/simulation``
    (which sits inside the agent's own working directory). Leaving either would
    leave the answer key one ``Read`` away.

    Call AFTER every setup step that consumes the bundle (``activate_scenarios``
    for logbook seeding, ``render_scenario_physics_env`` + ``osprey up`` for a
    VA stack's boot-time physics) and BEFORE the agent session starts. Also drops
    the names from the live ``var/agent_data/simulation/active_scenarios`` state
    file (the location :func:`activate_scenario` writes), since the name itself
    ("orm-dual-fault") is a hint, and leaving an active name whose bundle is gone
    would only earn an "Unknown scenario ... ignoring" warning from the engine.

    ONLY valid for a scenario whose runtime effect is already materialized
    somewhere the host-side :class:`~osprey.simulation.engine.SimulationEngine`
    is not: a VA-backed physics fault lives in the container's ``VA_BPM_ERRORS``/
    ``VA_CORR_GAIN`` environment from boot, so the bundle is inert once the stack
    is up. A mock-connector telemetry/archiver scenario (``rf-thermal``,
    ``vacuum-burst``) is the opposite — its bundle IS the live overlay, so
    deleting it would delete the symptom. Those suites rely on
    :data:`SCENARIO_INTEGRITY_DISALLOWED_TOOLS` alone.

    Raises:
        AssertionError: if a named bundle is not present in either tree
            (template drift — the caller believes it concealed something it did
            not).
    """
    source_sim = Path(repo) / "data" / "simulation"
    render_sim = render_dir(repo) / "data" / "simulation"
    sim_dirs = (source_sim, render_sim)
    for name in scenarios:
        for sim_dir in sim_dirs:
            bundle = sim_dir / "scenarios" / name
            assert bundle.is_dir(), (
                f"no scenario bundle at {bundle} to conceal — template layout may have "
                "changed; the benchmark's answer key would stay readable by the agent"
            )
            shutil.rmtree(bundle)

    # The live state file activate_scenario writes; a stray copy beside either
    # machine model is scrubbed too if present. The live file must EXIST —
    # a silent skip here is how a state-file relocation once left the answer
    # key agent-readable while this helper reported success.
    state_dir = agent_data_dir(repo) / "simulation"
    live_state = state_dir / "active_scenarios"
    assert live_state.is_file(), (
        f"no active-scenarios state file at {live_state} — the state-file "
        "location moved again; update this helper or the answer key stays "
        "readable by the agent"
    )
    for state_file in (live_state, *(d / "active_scenarios" for d in sim_dirs)):
        if not state_file.is_file():
            continue
        kept = [
            line
            for line in state_file.read_text(encoding="utf-8").splitlines()
            if line.strip() not in scenarios
        ]
        state_file.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    # Self-check: prove the concealment rather than assume it. Cheap — all three
    # trees are a handful of small JSON/text files. The state dir is included
    # because Read is deliberately allowed for agent-data artifacts.
    for name in scenarios:
        leaked = [
            p
            for tree in (*sim_dirs, state_dir)
            for p in tree.rglob("*")
            if p.is_file() and name in p.read_text(encoding="utf-8", errors="ignore")
        ]
        assert not leaked, f"scenario {name!r} still readable from the agent's tree: {leaked}"


def promote_ask_to_allow(repo: Path, *tools: str) -> None:
    """Move ``tools`` from ``permissions.ask`` to ``permissions.allow`` in the
    render's ``.claude/settings.json``.

    Takes the REPO ROOT; the settings file the agent reads is the rendered one
    under ``<repo>/build/.claude/``.

    ``run_sdk_query`` runs headless with no responder for an approval prompt, so
    an ``ask``-listed tool comes back to the agent as "Claude requested
    permissions ... but you haven't granted it yet" — a hard denial, and
    ``permission_mode="bypassPermissions"`` does not override it. That silently
    removes ``mcp__python__execute`` (the sanctioned compute path for framework
    agents, which never get Bash) and ``mcp__bluesky__launch_run`` (without which
    no plan can ever run) from any headless benchmark that needs them.

    Grant only what the benchmark under test actually needs, rather than
    promoting the whole ``ask`` list: a blanket promotion also hands the agent
    ``mcp__controls__channel_write``, i.e. a hand-stepped alternative to the
    measurement the benchmark is grading.

    Call AFTER ``osprey up`` — the deploy path can re-render the Claude Code
    artifacts and would discard an earlier edit.

    Raises:
        AssertionError: if a named tool is not in ``permissions.ask`` (either it
            was already granted, or the settings renderer changed — both mean
            this call is not doing what the caller thinks).
    """
    settings_path = render_dir(repo) / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    permissions = settings.setdefault("permissions", {})
    ask = permissions.setdefault("ask", [])
    allow = permissions.setdefault("allow", [])

    for tool in tools:
        assert tool in ask, (
            f"{tool} is not in permissions.ask of {settings_path} "
            f"(ask={ask}) — the settings renderer may have changed"
        )
        ask.remove(tool)
        allow.append(tool)

    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _default_opus_model(repo: Path) -> str:
    """Resolve the deployment's opus-tier model name.

    Takes the REPO ROOT; the provider spec is resolved from the render's
    ``config.yml``.

    Use for tests that benchmark agent reasoning (diagnostic-style
    challenges) — Opus is required for the planner to converge on a
    committed conclusion instead of hedging on a data dump.
    """
    spec = _resolve_project_spec(render_dir(repo))
    if spec is not None:
        return spec.tier_to_model.get("opus", "claude-opus-4-7")
    return "claude-opus-4-7"


def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


def find_html_files(root: Path) -> list[Path]:
    """Recursively find all .html files under *root*, excluding index.html."""
    return sorted(p for p in root.rglob("*.html") if p.name != "index.html")


def read_audit_events(repo: Path) -> list[dict]:
    """Read MCP tool-call events from Claude Code native transcripts.

    Takes the REPO ROOT. Claude Code keys its transcript directory on the
    session's working directory, which is the RENDER, so that is what the reader
    is pointed at.

    Uses TranscriptReader to extract events from the most recent transcript
    in ``~/.claude/projects/<encoded>/``.

    Returns:
        List of event dicts (tool_call, agent_start, agent_stop).
    """
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    reader = TranscriptReader(render_dir(repo))
    return reader.read_current_session()


# ---------------------------------------------------------------------------
# MCP sidecar + core runner support helpers
# ---------------------------------------------------------------------------


def _persist_mcp_sidecar(workflow: SDKWorkflowResult, repo: Path) -> None:
    """Write the MCP-status snapshot to a per-test sidecar when
    ``OSPREY_E2E_INIT_SIDECAR`` is set. Off by default, so ordinary CI/local runs
    are byte-for-byte unchanged. The sidecar turns the infra-vs-model question
    into a recorded fact for post-hoc benchmark forensics.

    Written at the REPO ROOT rather than inside the render: it is forensic
    output about a run, and the render is re-created wholesale by every build."""
    if not os.environ.get("OSPREY_E2E_INIT_SIDECAR"):
        return
    try:
        out_dir = Path(repo) / ".osprey_e2e"
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
    render: Path,
) -> None:
    """Append sub-agent tool calls that never streamed through ``query()``.

    Takes the RENDER directory — the SDK's transcript lookup is keyed on the
    session's working directory, which is where the agent ran.

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

    directory = str(render)
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
    repo: Path,
    prompt: str,
    *,
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> SDKWorkflowResult:
    """Run a query via the Claude Agent SDK and collect full tool traces.

    Args:
        repo: Deployment repo root (what :func:`init_project` returns). The
            agent runs with its cwd at the render, ``<repo>/build``, where
            ``.mcp.json``, ``.claude/`` and ``CLAUDE.md`` live.
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

    render = render_dir(repo)
    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(render),
        cwd=str(render),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd * e2e_budget_scale(),
        env=sdk_env(render),
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
        # first turn — eliminating the controls cold-start race (see await_mcp_ready).
        # Message handling is identical to the query() iterator.
        async with ClaudeSDKClient(options=options) as client:
            workflow.mcp_servers = await await_mcp_ready(client, expected_mcp_servers(render))
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
    _harvest_subagent_traces(workflow, pending_tools, render)

    _persist_mcp_sidecar(workflow, repo)
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
    # The hook's own ``permissionDecisionReason``, as the SDK reports it on the
    # permission context. ``reason`` above records what the *test policy* did;
    # this records what the *hook* said, which is what a test asserting on hook
    # wording needs.
    decision_reason: str | None = None


@dataclass
class HookObservedResult(SDKWorkflowResult):
    """Extends SDKWorkflowResult with hook observability."""

    hook_events: list[HookEvent] = field(default_factory=list)


def _bind_approval_policy(
    policy: Callable[..., bool],
) -> Callable[[str, dict[str, Any], Any], bool]:
    """Normalise a custom approval policy to one uniform three-argument shape.

    A policy may be written either as ``(tool_name, tool_input) -> bool`` or as
    ``(tool_name, tool_input, context) -> bool``. Which one it is, is decided
    here by explicit signature inspection, once, before any tool call — never by
    catching ``TypeError`` from a call, which would silently swallow a
    ``TypeError`` raised *inside* a three-argument policy and turn a real bug
    into a mysterious approval.

    A policy whose signature cannot be inspected (a C-level callable, say) is
    treated as the two-argument form, which is the shape every policy predating
    the context argument has. A third positional parameter counts as the
    context slot even when it carries a default.
    """
    try:
        parameters = list(inspect.signature(policy).parameters.values())
    except (TypeError, ValueError):
        parameters = []

    takes_context = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ) or (
        len(
            [
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
        >= 3
    )

    if takes_context:
        return policy

    def _call_without_context(tool_name: str, tool_input: dict[str, Any], context: Any) -> bool:
        return policy(tool_name, tool_input)

    return _call_without_context


async def run_sdk_query_with_hooks(
    repo: Path,
    prompt: str,
    *,
    approval_policy: Callable[..., bool] | str = "auto_approve",
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> HookObservedResult:
    """Run a query via the Claude Agent SDK with hooks enabled and can_use_tool callback.

    Unlike ``run_sdk_query`` (which uses bypassPermissions), this function uses
    ``permission_mode="default"`` so that file-system hooks actually execute.
    When a hook returns ``permissionDecision: "ask"``, the ``can_use_tool``
    callback is invoked instead of prompting a human.

    The ``approval_policy`` controls what happens when a hook returns "ask":
    - ``"auto_approve"`` — always approve (hooks still run, but "ask" → allow)
    - ``"auto_deny"`` — always deny (test that denial propagates correctly)
    - callable — custom fine-grained control, written either as
      ``(tool_name, tool_input) -> bool`` or, when it needs to see why the hook
      asked, as ``(tool_name, tool_input, context) -> bool``. Which form a
      policy has is decided from its signature, so both work unchanged.

    Every callback invocation is recorded in ``hook_events`` for observability,
    including the hook's own ``permissionDecisionReason`` as
    ``HookEvent.decision_reason``.

    Args:
        repo: Deployment repo root (what :func:`init_project` returns). The
            agent runs with its cwd at the render, ``<repo>/build``, which is
            where the ``.claude/`` hooks and settings it obeys are rendered.
        prompt: The user prompt to send.
        approval_policy: How to handle "ask" decisions from hooks.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use. Defaults to the project's haiku-tier model
            resolved from ``config.yml``.
        disallowed_tools: Optional list of tool names to forbid at the SDK level.
            Forwarded to the Claude Code CLI as ``--disallowedTools``. Use this to
            force a specific route when a test must *prove* one path works: the
            agent picks between equivalent capabilities non-deterministically
            (e.g. ``mcp__python__execute`` vs ``create_static_plot`` for a plot),
            so a prompt alone cannot guarantee which one a run exercises.

    Returns:
        HookObservedResult with tool traces, text, metadata, and hook events.
    """
    hook_events: list[HookEvent] = []
    stderr_lines: list[str] = []
    # Inspect the policy's arity once, here, rather than on every tool call.
    policy_call = _bind_approval_policy(approval_policy) if callable(approval_policy) else None

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
        elif policy_call is not None:
            should_allow = policy_call(tool_name, tool_input, context)
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
            decision_reason=context.decision_reason,
        )
        hook_events.append(event)

        if should_allow:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Denied by test approval policy")

    render = render_dir(repo)
    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(render),
        cwd=str(render),
        permission_mode="default",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd * e2e_budget_scale(),
        env=sdk_env(render),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        can_use_tool=_can_use_tool,
        disallowed_tools=disallowed_tools or [],
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
            workflow.mcp_servers = await await_mcp_ready(client, expected_mcp_servers(render))
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
    _harvest_subagent_traces(workflow, pending_tools, render)

    workflow.hook_events = hook_events
    _persist_mcp_sidecar(workflow, repo)
    return workflow


# ---------------------------------------------------------------------------
# Agent transcripts as CI artifacts
# ---------------------------------------------------------------------------


#: Sub-directory of ``OSPREY_CI_DIAG_DIR`` that transcripts are written to.
#: Separate from the records :mod:`tests.ci_diagnostics` keeps in the same
#: directory, so a per-worker event log and a per-test transcript can never
#: collide on a name.
TRANSCRIPT_SUBDIR = "agent"


def _transcript_payload(name: str, result: SDKWorkflowResult) -> dict[str, Any]:
    """The serialisable form of one agent run — deliberately UNTRUNCATED.

    Truncation is the whole reason this exists. ``_to_workflow_result``
    previews every tool result at 300 characters before the judge ever sees
    it, and a ``get_run_data`` payload spends its first 300 characters on
    ``run_uid`` and ``columns``, so not one measured value survives into the
    judge's view. When the judge then reports that a response's numbers
    "cannot be verified from the execution trace", nothing anywhere records
    what the tools actually returned, and the verdict text is left as the only
    account of a response nobody can re-read.

    ``mcp_server_status`` is included because it is the infra-vs-model
    discriminator: a tool the agent never called means one thing if the
    handshake offered it and quite another if it never registered.
    """
    return {
        "name": name,
        # Every turn's prose, joined the way the judge is given it, but whole.
        "response": "\n".join(result.text_blocks).strip(),
        "text_blocks": list(result.text_blocks),
        "tool_traces": [
            {
                "name": t.name,
                "input": t.input,
                "result": t.result,
                "is_error": t.is_error,
                "tool_use_id": t.tool_use_id,
                "parent_tool_use_id": t.parent_tool_use_id,
            }
            for t in result.tool_traces
        ],
        "mcp_server_status": result.mcp_server_status,
        "registered_tools": result.registered_tools,
    }


def dump_agent_transcript(name: str, result: SDKWorkflowResult) -> Path | None:
    """Persist one agent run's full response and tool traces as a CI artifact.

    Gated on ``OSPREY_CI_DIAG_DIR`` exactly like :mod:`tests.ci_diagnostics`:
    unset — which is every local run — writes nothing and returns ``None``.
    The lanes that set it already upload that directory, and
    ``capture-ci-diagnostics`` creates its own subdirectory rather than
    clearing the tree, so what is written here survives into the artifact.

    Call this BEFORE the assertions, never after. The runs worth reading are
    the ones that fail, and a dump placed below a judge assertion never
    executes on exactly those.

    Never raises. A diagnostic that can fail the test it was only meant to
    observe would mask the very failure it exists to explain — the same reason
    every probe in the capture action ends in ``|| true``.
    """
    directory = os.environ.get(ci_diagnostics.ENV_DIR)
    if not directory:
        return None
    try:
        target_dir = Path(directory) / TRANSCRIPT_SUBDIR
        target_dir.mkdir(parents=True, exist_ok=True)
        # Test ids carry ``[]`` from parametrisation and ``/`` from paths.
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name) or "transcript"
        target = target_dir / f"{safe}.json"
        target.write_text(
            json.dumps(_transcript_payload(name, result), indent=2, default=str),
            encoding="utf-8",
        )
        return target
    except Exception:
        return None
