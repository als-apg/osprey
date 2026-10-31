"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and deployment-repo
initialization utilities used by both the functional SDK tests and the safety
E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.

**Zones.** Every public helper here takes the *deployment repo root* — what
:func:`init_project` returns and what ``osprey init`` creates — so a test
carries one handle. A repo has three zones the helpers resolve internally:

* ``<repo>/`` — SOURCE: ``profile.yml``, the operator-owned ``data/`` tree
  (including ``data/simulation/``), ``tiers/``, ``apps/``, and the ``.env`` secret store.
* ``<repo>/build/`` — the RENDER (see :func:`render_dir`): ``config.yml``,
  ``.mcp.json``, ``CLAUDE.md``, ``.claude/``, and the render's own ``data/``
  copy. This is the Claude Code agent's working directory.
* ``<repo>/var/`` — STATE: ``agent_data/`` (agent memory, artifacts, the
  simulation's ``active_scenarios`` file) and ``audit/``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.agent_runner.claude_state import is_untrusted_allow_rules_notice, seed_claude_state
from osprey.facility.views.simulator import withdraw_scenarios
from osprey.port_layout import BLOCK_SIZE
from osprey_connectors import control_context
from osprey_connectors.identity import acting_identity
from osprey_connectors.simulation.view import SimulatorView
from osprey_connectors.types import WRITES_KEY, baseline_target
from osprey_connectors_tests._harness import ci_diagnostics
from tests.e2e.profile_edits import set_pairs
from tests.e2e.provider import build_model, build_provider

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
from osprey.agent_runner.launcher import project_agent_cli_path
from osprey.agent_runner.primitives import (
    MCP_READY_TIMEOUT_S,
    _ingest_tool_result,
    _resolve_project_spec,
)
from osprey.agent_runner.primitives import (
    provider_env_for_project as provider_env_for_project,  # re-exported for e2e tests
)
from osprey.agent_runner.project_paths import claude_project_dir
from osprey_connectors.workspace import (
    BUILD_DIR_NAME,
    DEFAULT_AGENT_DATA_BASE_DIR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cli_mcp_ready_env() -> dict[str, str]:
    """Env that holds a bare ``claude --print`` run's first turn for its MCP servers.

    The CLI connects stdio servers in the background by default and proceeds
    after its own short connect deadline. A subagent's toolset is fixed when it
    is spawned, so a delegation that lands before a slow server connects runs
    the whole session without that server's tools. These are the CLI's own
    knobs for the readiness barrier the SDK path applies through
    :func:`await_mcp_ready`, sized by the same ``MCP_READY_TIMEOUT_S``.

    Returns:
        ``MCP_CONNECTION_NONBLOCKING`` off, and the connect deadline
        (``MCP_CONNECT_TIMEOUT_MS``) and stdio startup limit (``MCP_TIMEOUT``)
        both set to the readiness ceiling in milliseconds.
    """
    ready_ms = str(int(MCP_READY_TIMEOUT_S * 1000))
    return {
        "MCP_CONNECTION_NONBLOCKING": "0",
        "MCP_CONNECT_TIMEOUT_MS": ready_ms,
        "MCP_TIMEOUT": ready_ms,
    }


def render_dir(repo: Path) -> Path:
    """The render zone of the deployment repo at *repo* — ``<repo>/build``.

    Everything ``osprey build`` produces lands here: ``config.yml``,
    ``.mcp.json``, ``CLAUDE.md``, ``.claude/`` and the render's ``data/`` copy.
    It is also the Claude Code agent's working directory. Spelled once, through
    the same constant the framework renders against, so the zone split cannot
    drift between this module and the build.
    """
    return Path(repo) / BUILD_DIR_NAME


#: The Claude Code state directory the harness owns for the running test, or
#: ``None`` when the run uses the developer's own state. The e2e conftest sets
#: it, together with ``CLAUDE_CONFIG_DIR``, for every test marked
#: ``model_free``; a test that reaches a model runs as it would on the
#: developer's machine, and the harness seeds nothing into their state.
harness_config_dir: Path | None = None


def trust_render(repo: Path) -> None:
    """Trust the render as a workspace in the Claude Code state this run uses.

    Seeded the way the container entrypoint seeds a deployment's per-user
    volume (:func:`seed_claude_state`): the render is a trusted workspace, so
    its ``permissions.allow`` rules apply to the run as they do in a
    deployment, and no first-run prompt is pending. The state directory is the
    test's own (:data:`harness_config_dir`, set by the e2e conftest for a
    ``model_free`` test), never the developer's home: a caller outside that
    isolation is refused rather than let write into it. Untrusted, the CLI
    leaves the allow rules off and says so on stderr
    (:func:`is_untrusted_allow_rules_notice`); a run that seeds and still sees
    the notice is a harness fault, not a scenario.
    """
    if harness_config_dir is None:
        raise RuntimeError(
            "trust_render seeds only the harness's own Claude Code state; "
            "mark the test model_free so the e2e conftest gives it one"
        )
    seed_claude_state(render_dir(repo))


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


#: The ARIEL database a run uses when it names none: a Postgres on the host's
#: standard port, the one tests/e2e/README.md tells a developer to run.
_DEFAULT_ARIEL_DB_URI = "postgresql://ariel:ariel@localhost:5432/ariel"


def e2e_ariel_db_uri() -> str:
    """The run's one ARIEL database: what the skip guard probes and every ARIEL project names.

    ``OSPREY_ARIEL_DB_URI`` when it is set and non-empty (the model-matrix
    runner provisions one database per (model, seed) cell and exports it),
    otherwise :data:`_DEFAULT_ARIEL_DB_URI`. :func:`ariel_db_skip_reason` and
    :func:`_ariel_db_pins` both read it here, so a guard that passes has checked
    the database the agent and ``apply_scenarios`` are about to use.
    """
    return os.environ.get("OSPREY_ARIEL_DB_URI") or _DEFAULT_ARIEL_DB_URI


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
    The database is :func:`e2e_ariel_db_uri`, the one every ARIEL project
    :func:`init_project` builds names, so a pass here means the agent's
    database is ready. The reason names that database without its password.
    """
    uri = uri or e2e_ariel_db_uri()
    where = uri.rsplit("@", 1)[-1]
    try:
        import psycopg
    except ImportError:
        return "psycopg not installed — cannot verify the ARIEL DB prerequisite"
    try:
        with psycopg.connect(uri, connect_timeout=3) as conn:
            row = conn.execute("SELECT count(*) FROM enhanced_entries").fetchone()
    except Exception as exc:  # any failure means "not ready"
        return (
            f"ARIEL Postgres at {where} not reachable ({exc.__class__.__name__}) — scenario "
            "tests need a live, seeded ARIEL logbook DB. Bring it up with "
            "`osprey up && osprey ariel migrate && osprey ariel quickstart`."
        )
    count = row[0] if row else 0
    if count == 0:
        return (
            f"ARIEL Postgres at {where} is reachable but empty — seed it with "
            "`osprey ariel quickstart`."
        )
    return None


def _ariel_db_pins(template: str) -> dict[str, str]:
    """The ``config:`` pin that points a lane's deployment at the run's ARIEL database.

    Every project this module builds with ARIEL names :func:`e2e_ariel_db_uri`,
    the database :func:`ariel_db_skip_reason` probes, so a guard that passes has
    checked the database the agent's ARIEL MCP server and ``apply_scenarios``
    reach. Left to itself a render names no URI: the DSN is derived from
    ``services.postgresql`` on this worker's port block, a Postgres no project
    built here ever starts. In a model-matrix cell the database is the cell's
    own, so a scenario test's logbook purge and its drop of the
    ``text_embeddings_*`` tables cannot land in another cell's logbook test.

    The pin is ``ariel.database.uri``, stated at ``osprey init`` like every other
    lane pin, and an explicit ``uri`` wins over the derived DSN. A preset that
    configures no ARIEL gets no pin: an ``ariel:`` block in its render would make
    ``apply_scenarios`` seed a logbook the deployment does not have.

    Returns:
        ``{"ariel.database.uri": <uri>}`` when the preset configures ARIEL,
        otherwise an empty mapping.
    """
    if not _preset_configures_ariel(template):
        return {}
    return {"ariel.database.uri": e2e_ariel_db_uri()}


def _assert_render_names_ariel_db(render: Path) -> None:
    """Fail when a render that configures ARIEL does not name the run's ARIEL database.

    Takes the RENDER directory (``<repo>/build``). Its ``config.yml`` is what
    every MCP server is pointed at via ``CONFIG_FILE`` and what
    ``apply_scenarios`` loads, so it is where :func:`_ariel_db_pins` has to have
    landed. A render with no ``ariel`` section has no database to point anywhere.
    """
    uri = e2e_ariel_db_uri()
    config_path = render / "config.yml"
    ariel = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("ariel")
    if not ariel:
        return
    rendered = (ariel.get("database") or {}).get("uri")
    assert rendered == uri, (
        f"ariel.database.uri in {config_path} is {rendered!r}, not {uri!r}, "
        "the ARIEL database ariel_db_skip_reason probes"
    )


#: First port of the block the projects of xdist worker 0 bind. Each worker
#: takes the next ``BLOCK_SIZE`` block, so ``-n 4`` spans 25000-28999: clear
#: of the default 10000 block a real deployment on the host may hold, of the
#: 21xxx bands the full-stack lanes pin, and below the ephemeral range the
#: kernel hands out from 32768.
E2E_PORT_BASE = 25000


def e2e_port_base() -> int:
    """The ``deployment.port_base`` for a project built by this xdist worker.

    Every companion server's port is an offset from the block — the artifact
    gallery's among them, and the gallery auto-launches on the first artifact
    save. Two workers whose projects share a block therefore race for one
    gallery port: whichever binds first owns it, and a test on the other
    worker then reaches a gallery holding a different stack's bearer token
    and is answered 401 (#823). One block per worker ends the sharing.
    """
    return E2E_PORT_BASE + BLOCK_SIZE * ci_diagnostics.worker_index()


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    *,
    provider: str,
    model: str | None = None,
    channel_finder_mode: str | None = None,
    connector: str = "virtual_accelerator",
    archiver: str = "mock_archiver",
) -> Path:
    """Create a deployment repo at ``tmp_path/name`` and build it; return the repo root.

    Two commands, because the surface has two: ``osprey init <dir> --preset P``
    writes the repo's source zone (``profile.yml``, ``data/``, ``tiers/``, ``apps/``,
    ``.env.example``), and ``osprey build --repo <dir>`` renders ``build/`` from
    it. The directory name IS the deployment name.

    Returns the REPO ROOT, not the render: it is what
    :func:`osprey.simulation.apply.apply_scenarios` takes, it is the
    ``project_root`` the rendered config records, and ``var/agent_data`` hangs
    off it. Every helper in this module takes that same handle and resolves the
    render itself via :func:`render_dir`.

    ``--no-git`` is always passed: no test reads the repo's history and
    ``git init`` is pure latency here.

    ``connector`` is pinned to the simulator served in process rather than
    inherited from the preset: the control-assistant preset baselines on its
    live stand-in, a deployed soft IOC that has to answer Channel Access — this
    harness runs projects without their containers, so the preset's production
    default would turn every channel read/write into a connection timeout.
    Tests that deploy a real stack build through their own fixtures, not this
    helper.

    ``archiver`` is pinned for the same reason and is the archive half of that
    same fact: the preset selects ``mongodb_archiver`` and declares the
    ``va_archiver:`` block that deploys the store it reads, so a containerless
    build would leave every ``archiver_read`` failing at connect for want of a
    store — and, before that, for want of the password ``osprey up``
    mints. Pinning the mock archiver beside the simulator in process is not a
    way around the pairing rule in :mod:`osprey_connectors.honesty` but the
    case it explicitly allows: the simulator in process has no recorder, so a
    synthesized archive is the only one it can have, and nothing lies. Tests
    that want recorded history deploy a store of their own.

    ``virtual_accelerator.live_standin`` is nulled as the third of those pins,
    for presets that declare a virtual accelerator at all. The control-assistant
    preset ships a live stand-in on, which is a second VA container this
    harness never starts — and, before any read of it times out, the build
    derives the posture that container is meant to be met with: the ``epics``
    gateways move to it and limits checking goes strict, so a write to a
    channel absent from a test's own limits fixture is refused where it used
    to pass. A project that wants the stand-in deploys it. The pin is written
    only when the preset has a ``virtual_accelerator:`` block to pin a key in:
    an overlay carrying the block CREATES it on a preset that has none, and a
    hello-world project would then be a deployment running a virtual
    accelerator with no channel database to build its namespace from — which
    the build refuses, correctly.

    ``ariel.database.uri`` is pinned to :func:`e2e_ariel_db_uri` when the
    preset configures ARIEL (see :func:`_ariel_db_pins`), so the agent reaches
    the database :func:`ariel_db_skip_reason` probed, and the render is then
    checked to name it: a pin that stops reaching the render fails here rather
    than letting the agent talk to some other database.

    ``channel_finder_mode`` is pinned to ``hierarchical`` when the caller names
    no mode and the preset's own is ``graph`` — the same containerless fact a
    third time. The graph paradigm answers from the ``graphdb`` service the
    preset deploys, which this harness never starts, so every lookup would
    fail at connect and an agent asked for a channel would never reach the tool
    a test is actually about. The hierarchical database the preset also ships
    answers from a file. A test whose subject IS the graph paradigm names
    ``graph`` explicitly and stands the store up itself.

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

    The suite-wide override of that pinned choice is resolved by
    :func:`tests.e2e.provider.build_provider`, which states the precedence.

    ``model`` is always written to the profile. A call site that names none
    builds with :data:`tests.e2e.provider.E2E_MODEL` (see
    :func:`tests.e2e.provider.build_model`), because the suite's budgets are
    sized for that model and the provider's catalog default is not.
    """
    from osprey.build.modes import VALID_CHANNEL_FINDER_MODES

    provider = build_provider(provider)
    if channel_finder_mode is None and _preset_channel_finder_mode(template) == "graph":
        channel_finder_mode = "hierarchical"
    if channel_finder_mode is not None and channel_finder_mode not in VALID_CHANNEL_FINDER_MODES:
        raise ValueError(
            f"channel_finder_mode {channel_finder_mode!r} is not one of "
            f"{list(VALID_CHANNEL_FINDER_MODES)}"
        )
    repo = tmp_path / name
    init_args = [
        str(repo),
        "--preset",
        template.replace("_", "-"),
        "--no-git",
        "--set",
        f"provider={provider}",
        "--set",
        f"connector={connector}",
    ]
    init_args.extend(["--set", f"model={build_model(model)}"])
    # ``archiver.type`` is written in the literal dotted spelling the preset
    # already uses, so the edit replaces that entry instead of landing beside
    # it. The stand-in pin rides along where the preset declares a VA, and the
    # ARIEL database pin where the preset configures ARIEL.
    pins: dict[str, Any] = {"config": {"archiver.type": archiver, **_ariel_db_pins(template)}}
    if connector == "virtual_accelerator":
        pins["config"]["control_system.connector.virtual_accelerator.serving"] = "in_process"
    if _preset_declares_virtual_accelerator(template):
        pins["virtual_accelerator"] = {"live_standin": None}
    # These projects run host sessions and never a stack, so their record is the
    # host-local one, which keeps the embedded backend: a ClickHouse store runs
    # only in a stack.
    pins["agent_record"] = {"backend": "embedded"}
    init_args.extend(set_pairs(pins))
    init_args.extend(["--set", f"port_base={e2e_port_base()}"])
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
    _assert_render_names_ariel_db(render)
    return repo


def _preset_channel_finder_mode(template: str) -> str | None:
    """The ``channel_finder_mode`` the packaged preset ``template`` ships, if any.

    Read from the preset the way ``osprey init`` reads it, so the pin above
    follows the preset rather than a list of names kept here.
    """
    from osprey.cli.build_profile_presets import _load_preset_raw

    raw, _path = _load_preset_raw(template.replace("_", "-"))
    mode = raw.get("channel_finder_mode")
    return mode if isinstance(mode, str) else None


def _preset_declares_virtual_accelerator(template: str) -> bool:
    """Whether the packaged preset ``template`` names a ``virtual_accelerator:`` block.

    Read from the preset the way ``osprey init`` reads it, so the answer is the
    preset's own rather than a list of names kept here.
    """
    from osprey.cli.build_profile_presets import _load_preset_raw

    raw, _path = _load_preset_raw(template.replace("_", "-"))
    return raw.get("virtual_accelerator") is not None


def _preset_configures_ariel(template: str) -> bool:
    """Whether the packaged preset ``template``, or a preset it extends, sets ARIEL config.

    Read from the presets the way ``osprey init`` reads them, following
    ``extends:``, so the answer is the presets' own rather than a list of names
    kept here. A ``config:`` key is ARIEL config when its first dotted segment
    is ``ariel``.
    """
    from osprey.cli.build_profile_presets import _load_preset_raw

    name: str | None = template.replace("_", "-")
    seen: set[str] = set()
    while name and name not in seen:
        seen.add(name)
        raw, _path = _load_preset_raw(name)
        if any(str(key).split(".", 1)[0] == "ariel" for key in raw.get("config") or {}):
            return True
        extends = raw.get("extends")
        name = extends if isinstance(extends, str) else None
    return False


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


#: The one tier value that lets a login's switch arm anything (D9). Spelled
#: here rather than imported: the connectors package keeps it private, and the
#: profile a deployer writes spells exactly this word.
WRITES_TIER_ALLOWED = "allowed"


def set_writes_tier(repo: Path, tier: str = WRITES_TIER_ALLOWED) -> None:
    """Spell ``control_system.writes: <tier>`` in the repo's profile and rebuild.

    Takes the REPO ROOT. The edit lands in the SOURCE zone (``profile.yml``'s
    ``config:`` block, under the dotted :data:`~osprey_connectors.types.WRITES_KEY`
    the presets use), and ``osprey build`` then renders it — the same two steps
    a deployer takes, so the render, its ``settings.json`` and the hook config
    all read one tier. Any other spelling of the same leaf already in the block
    (the dotted key, or ``writes`` inside a nested ``control_system:`` mapping)
    is removed first, so the profile states the tier once. A profile still
    carrying an older posture key is not rewritten here: the build refuses the
    mix, which is the failure such a fixture deserves.
    """
    profile_path = Path(repo) / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config = profile.get("config")
    if not isinstance(config, dict):
        config = {}
    section, leaf = WRITES_KEY.split(".", 1)
    config.pop(WRITES_KEY, None)
    nested = config.get(section)
    if isinstance(nested, dict):
        nested.pop(leaf, None)
        if not nested:
            del config[section]
    config[WRITES_KEY] = tier
    profile["config"] = config
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    _run_osprey(
        "build",
        ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        timeout=300,
    )


def _render_config(repo: Path) -> dict[str, Any]:
    """The render's parsed ``config.yml`` — what the servers read."""
    loaded = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def switch_record_path(repo: Path, owner: str) -> Path:
    """Where *owner*'s control-context record lives for the deployment at *repo*.

    Resolved through the deployment's one seam for the control tree
    (:func:`osprey.deployment.compose_generator.control_target_identity_dir`),
    which anchors the render's ``agent_data.base_dir`` at the repo and refuses
    an *owner* outside the username charset, so the file written here is the
    file the verdict reads for that login, in process or bound into a container.
    """
    from osprey.deployment.compose_generator import control_target_identity_dir

    directory = control_target_identity_dir(_render_config(repo), Path(repo), owner)
    return directory / control_context.RECORD_FILENAME


def _default_switch_targets(repo: Path) -> list[str]:
    """The machine a fresh session on *repo* sits on: its baseline target."""
    section = _render_config(repo).get("control_system") or {}
    return [baseline_target(section)]


def switch_on_in_record(
    repo: Path, *, owner: str, targets: list[str] | None = None
) -> control_context.ControlContext:
    """Turn *owner*'s switch on for *targets* by writing the record directly.

    The record fixture of D9, for lanes whose subject is not the chip. It goes
    through the production writers rather than a hand-rolled payload: a missing
    or unreadable record is first written ownerless, pointed at the baseline
    target, by :func:`control_context.write_record` (which stamps a fresh epoch),
    and the switch is then moved by :func:`control_context.set_switch` — the
    person's gesture, compare-and-swap on the epoch read just before. An
    epoch-less record is stamped by the first attempt and armed by the retry,
    the same one retry the chip makes.

    *targets* defaults to the deployment's baseline target. Returns the record
    as stored.
    """
    names = sorted(targets) if targets is not None else _default_switch_targets(repo)
    path = switch_record_path(repo, owner)
    control_context.invalidate_cache()
    current, status = control_context.read_record_status(path=path)
    if current is None:
        baseline = _default_switch_targets(repo)[0]
        control_context.write_record(
            control_context.ControlContext(target=baseline, generation=1), path=path
        )
        control_context.invalidate_cache()
        current = control_context.read_record(path=path)
    epoch = (current.switch_epoch if current is not None else None) or 0
    try:
        stored = control_context.set_switch(names, "on", epoch=epoch, path=path)
    except control_context.StoppedSince as refused:
        if not refused.stamped or refused.epoch is None:
            raise
        stored = control_context.set_switch(names, "on", epoch=refused.epoch, path=path)
    control_context.invalidate_cache()
    assert stored is not None, f"the switch for {owner!r} was not stored at {path} ({status})"
    return stored


#: The header chip's write route: ``POST {target, state, epoch}`` (C-POST).
CHIP_POSTURE_PATH = "/api/terminal/posture"


@dataclass(frozen=True)
class ChipRoute:
    """How to reach one login's web terminal, to arm through its header chip.

    The owner the chip arms is whoever that terminal acts as — the route reads
    the login server-side and never from the request — so a lane that arms
    through the chip names the owner by choosing which terminal to POST to.

    Attributes:
        base_url: The terminal's origin, ``http://127.0.0.1:<port>``.
        session_id: The browser session the gesture is filed under, or ``None``
            for the session-less Lab-bar gesture.
        headers: Whatever the terminal's own gate needs on top (its secret).
    """

    base_url: str
    session_id: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def switch_on_through_chip(chip: ChipRoute, target: str) -> dict[str, Any]:
    """Turn the switch on for *target* the way the header chip does; return the 200 body.

    Reads ``switch_epoch`` off ``GET`` first and sends it back with
    ``state: "on"``, as the decimal string it was read as. A ``409 stopped_since`` that just ``stamped`` an epoch-less
    record is retried once with the epoch it carries — the chip's own single
    retry; any other refusal fails here with its body, so a lane learns the
    reason (``tier_never``, ``record_unavailable`` ...) rather than a later
    refused write.
    """
    from tests.e2e._queue_drive import request

    query = "" if chip.session_id is None else f"?session_id={chip.session_id}"
    status, body = request(chip.base_url, CHIP_POSTURE_PATH + query, "GET", headers=chip.headers)
    assert status == 200, f"GET {CHIP_POSTURE_PATH} failed: {status} {body}"
    epoch = body.get("switch_epoch") if isinstance(body, dict) else None
    for _attempt in range(2):
        payload = {"session_id": chip.session_id, "target": target, "state": "on", "epoch": epoch}
        status, body = request(
            chip.base_url, CHIP_POSTURE_PATH, "POST", payload, headers=chip.headers
        )
        detail = body.get("detail") if isinstance(body, dict) else None
        if status == 409 and isinstance(detail, dict) and detail.get("stamped"):
            epoch = detail.get("epoch")
            continue
        break
    assert status == 200, f"the chip did not turn {target!r} on: {status} {body}"
    return body


@dataclass(frozen=True)
class ArmedWrites:
    """What :func:`arm_writes` armed: whose switch, for which machines, and how."""

    owner: str
    targets: list[str]
    via: str
    switch_epoch: int | None


def arm_writes(
    repo: Path,
    *,
    owner: str,
    targets: list[str] | None = None,
    chip: ChipRoute | None = None,
) -> ArmedWrites:
    """The one e2e arming helper (D9): tier ``allowed``, rebuilt, switch on for *owner*.

    A write is allowed only when the login's tier allows it AND that login's
    switch is on for the machine written, so arming is both halves, in order:
    :func:`set_writes_tier` spells ``control_system.writes: allowed`` and
    rebuilds, and then the switch is turned on for *owner* —

    * through the header chip's ``POST`` (:func:`switch_on_through_chip`) when
      *chip* is given: lanes whose subject is the chip arm this way, and the
      terminal behind *chip* must act as *owner*;
    * otherwise through the record fixture (:func:`switch_on_in_record`),
      written where the verdict reads *owner*'s record.

    *targets* defaults to the deployment's baseline target. A queue add that
    is meant to write names the same *owner* (``_queue_drive.enqueue(...,
    owner=...)``): an ownerless add has no switch to follow.
    """
    set_writes_tier(repo)
    names = sorted(targets) if targets is not None else _default_switch_targets(repo)
    if chip is not None:
        epoch = None
        for name in names:
            epoch = switch_on_through_chip(chip, name).get("switch_epoch")
        # The route carries the epoch as a decimal string; the record holds an int.
        switch_epoch = None if epoch is None else int(epoch)
        return ArmedWrites(owner=owner, targets=names, via="chip", switch_epoch=switch_epoch)
    stored = switch_on_in_record(repo, owner=owner, targets=names)
    return ArmedWrites(owner=owner, targets=names, via="record", switch_epoch=stored.switch_epoch)


def enable_writes_in_project(repo: Path, *, owner: str | None = None) -> ArmedWrites:
    """Arm writes for an in-process run of *repo*: :func:`arm_writes` through the record.

    Takes the REPO ROOT. *owner* defaults to the identity this process acts as
    (:func:`~osprey_connectors.identity.acting_identity`), which is the login
    the controls server the agent spawns resolves too — it inherits this
    environment — so the switch turned on is the one its writes read.

    Required for tests that exercise the approval-hook path: with the tier at
    ``never`` the build bakes the write tools into ``permissions.deny`` and the
    PreToolUse write check refuses before the approval hook can ask, so the
    SDK's ``can_use_tool`` callback never fires. Idempotent: arming an armed
    repo rebuilds and finds the switch already on.
    """
    return arm_writes(repo, owner=owner if owner is not None else acting_identity())


def arm_deployed_queue_owner(repo: Path, targets: list[str] | None = None) -> str:
    """Turn this process's login's switch on in a DEPLOYED stack's record; return the login.

    For lanes that drive a deployed bridge queue over HTTP. The bridge worker's
    writes follow the add's owner (``X-Osprey-Owner``, sent by
    ``_queue_drive.enqueue(..., owner=...)``), read from the control tree the
    deploy binds into the worker, so an ownerless add, or an owner whose switch
    is off, fails its first write. The tier half is the deployment's own
    (``control_system.writes: allowed`` in the preset); this is the switch half,
    through :func:`switch_on_in_record`, after the identity directory is
    provisioned the way the deploy provisions it. Call it after ``osprey build``
    and before ``osprey up``.

    The login is :func:`~osprey_connectors.posture_store.current_owner`, the
    same ladder an in-process tool walks. *targets* defaults to the deployment's
    baseline target.
    """
    from osprey.deployment.compose_generator import ensure_control_target_dir
    from osprey_connectors import posture_store

    owner = posture_store.current_owner()
    assert isinstance(owner, str) and owner, f"this process names no owner: {owner!r}"
    ensure_control_target_dir(_render_config(repo), repo, owner, relative_to=repo)
    switch_on_in_record(repo, owner=owner, targets=targets)
    return owner


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
    ``var/agent_data/simulation/`` state hangs off it, and it reads the render's
    ``config.yml`` and simulator view itself.

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
# Its ground truth ships inside the deployment repo as the facility's
# ``data/facility/scenarios/<name>.yaml``, whose ``description`` names the
# seeded fault outright — and the agent's cwd IS the render, whose simulator
# view lists every scenario with its blocks in ``data/simulator/scenarios.json``
# and copies its attached files under ``data/simulator/scenarios/<name>/``. Left alone, the cheapest route to a correct
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
    """Delete the named scenarios' definitions from the deployment repo.

    Takes the REPO ROOT and scrubs every copy of a scenario's answer key: the
    facility's ``<name>.yaml`` and ``<name>/`` under ``<repo>/data/facility/
    scenarios`` and the render's ``build/data/facility/scenarios``, the
    simulator view's ``build/data/simulator/scenarios/<name>/`` files, and the
    scenario's entry in the view's ``scenarios.json``. Leaving any would leave
    the answer key one ``Read`` away.

    Call AFTER every setup step that consumes the scenario (``activate_scenarios``
    for logbook seeding, ``osprey up`` for a stack whose containers carry their
    own copy of the view) and BEFORE the agent session starts. Also drops the
    names from the live ``var/agent_data/simulation/active_scenarios`` state
    file (the location :func:`activate_scenario` writes), since the name itself
    is a hint.

    ONLY valid for a scenario whose runtime effect is already materialized
    somewhere the host's render is not, such as a running container's own view.
    A mock-connector scenario reads the host's view on every tick, so deleting
    it would delete the symptom. Those suites rely on
    :data:`SCENARIO_INTEGRITY_DISALLOWED_TOOLS` alone.

    Raises:
        KeyError: a named scenario is not listed in the view (template drift —
            the caller believes it concealed something it did not); nothing is
            concealed.
        AssertionError: a concealed name is still readable from the agent's tree.
    """
    facility_dirs = (
        Path(repo) / "data" / "facility" / "scenarios",
        render_dir(repo) / "data" / "facility" / "scenarios",
    )
    view = SimulatorView.of_render(render_dir(repo))
    withdraw_scenarios(view.path, scenarios)
    for name in scenarios:
        for directory in facility_dirs:
            (directory / f"{name}.yaml").unlink(missing_ok=True)
            shutil.rmtree(directory / name, ignore_errors=True)

    # The live state file activate_scenario writes. It must EXIST: a silent
    # skip would leave the names agent-readable after a state-file relocation
    # while this helper reported success.
    state_dir = agent_data_dir(repo) / "simulation"
    live_state = state_dir / "active_scenarios"
    assert live_state.is_file(), (
        f"no active-scenarios state file at {live_state} — the state-file "
        "location moved; update this helper or the answer key stays "
        "readable by the agent"
    )
    kept = [
        line
        for line in live_state.read_text(encoding="utf-8").splitlines()
        if line.strip() not in scenarios
    ]
    live_state.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    # Self-check: prove the concealment rather than assume it. The state dir is
    # included because Read is deliberately allowed for agent-data artifacts.
    still_listed = {str(entry["name"]) for entry in SimulatorView.open(view.path).scenarios()}
    assert not still_listed & set(scenarios), f"the simulator view still lists {still_listed}"
    trees = [tree for tree in (*facility_dirs, view.path, state_dir) if tree.is_dir()]
    for name in scenarios:
        leaked = [
            p
            for tree in trees
            for p in tree.rglob("*")
            if p.is_file() and name in p.read_text(encoding="utf-8", errors="ignore")
        ]
        assert not leaked, f"scenario {name!r} still readable from the agent's tree: {leaked}"


# A one-shot SDK session ends with its turn and nothing re-invokes it. A tool
# that ends the turn expecting a later wake-up, or that schedules work for a
# session that will not exist, strands the run with its work half done: the
# agent is told it will be called back and never is. These are the scheduling
# names of the headless deny floor (``READ_ONLY_DENIED_BUILTINS``), so the
# pinned CLI lists every one of them; both runners below add them under
# whatever a caller disallows.
ONE_SHOT_DISALLOWED_TOOLS = ["ScheduleWakeup", "CronCreate", "Workflow"]


def _with_one_shot_floor(disallowed_tools: list[str] | None) -> list[str]:
    """The caller's ``disallowed_tools`` with :data:`ONE_SHOT_DISALLOWED_TOOLS` under it."""
    return list(dict.fromkeys([*ONE_SHOT_DISALLOWED_TOOLS, *(disallowed_tools or [])]))


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
        return spec.alias_models.get("opus", spec.default_model_id)
    return "claude-opus-5"


def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


def find_html_files(root: Path) -> list[Path]:
    """Recursively find all .html files under *root*, excluding index.html."""
    return sorted(p for p in root.rglob("*.html") if p.name != "index.html")


#: Most timeline pages :func:`read_audit_events` reads of one session.
_AUDIT_TIMELINE_PAGES = 50


def read_audit_events(repo: Path) -> list[dict]:
    """The OSPREY tool-call events of the newest session, read from the agent record.

    Takes the REPO ROOT. Holds the host's agent record for the read
    (:func:`~osprey.agent_record.host.ensure_host_record`, which starts the
    host-local service when none runs) and reads it through
    :class:`~osprey.agent_record.client.RecordClient` with the host login: the
    newest session it lists, every page of that session's timeline, subagent
    work included.

    Returns:
        Event dicts, in timeline order: ``tool_call`` for every MCP tool call
        (``timestamp``, ``tool``, ``server``, ``full_tool_name``, ``is_error``,
        ``session_id``, ``agent_id``, ``arguments``, ``result_summary``), and an
        ``agent_start``/``agent_stop`` pair for every Agent (or Task) call.
    """
    from osprey.agent_record.client import RecordClient
    from osprey.agent_record.contract import Drain
    from osprey.agent_record.host import ensure_host_record

    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    with ensure_host_record(repo, config) as record:
        client = RecordClient(record.url, record.login, record.token)
        # Drained, because the caller reads right after a run ends: the run's
        # last transcript steps reach the record only once the tailer has
        # ingested them, and an undrained read races that.
        sessions = client.list_sessions(limit=1, wait=Drain())
        assert sessions.available, f"the agent record is unavailable: {sessions.unavailable}"
        assert sessions.complete, f"the session list is incomplete: {sessions.reason}"
        if not sessions.data:
            return []
        session_id = str(sessions.data[0]["session_id"])
        nodes: list[dict] = []
        after: str | None = None
        for _ in range(_AUDIT_TIMELINE_PAGES):
            page = client.timeline(session_id, after=after, wait=Drain(session_id=session_id))
            assert page.available, f"the agent record is unavailable: {page.unavailable}"
            assert page.complete, (
                f"the timeline page of session {session_id} is incomplete: {page.reason}"
            )
            nodes.extend(node for node in page.data or [] if isinstance(node, dict))
            after = page.next
            if after is None:
                break
        else:
            raise AssertionError(
                f"session {session_id} has more than {_AUDIT_TIMELINE_PAGES} timeline "
                "pages; refusing to return a truncated event list"
            )
    return _tool_events(nodes, session_id, agent_id=None)


def _tool_events(nodes: list[dict], session_id: str, *, agent_id: str | None) -> list[dict]:
    """The tool-call and subagent events of *nodes*, descending into each subagent."""
    from osprey.mcp_server.workspace.transcript_reader import (
        _extract_text,
        _is_error_response,
        _split_mcp_tool_name,
        _truncate,
    )

    events: list[dict] = []
    for node in nodes:
        body = node.get("body") if isinstance(node.get("body"), dict) else {}
        result = node.get("result") or {}
        result_body = result.get("body") if isinstance(result.get("body"), dict) else {}
        subagent = node.get("subagent") or {}
        if node.get("kind") == "tool_call":
            name = str(body.get("name", ""))
            match = _split_mcp_tool_name(name)
            if match:
                short_name, server, full_name = match
                content = result_body.get("content", "")
                if isinstance(content, list):
                    text = _extract_text(content)
                elif isinstance(content, dict):
                    text = json.dumps(content)
                else:
                    text = str(content)
                is_error = bool(result_body.get("is_error")) or _is_error_response(text)
                events.append(
                    {
                        "type": "tool_call",
                        "timestamp": node.get("ts"),
                        "tool": short_name,
                        "tool_name": short_name,
                        "full_tool_name": full_name,
                        "server": server,
                        "server_name": server,
                        "is_error": is_error,
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "arguments": body.get("input", {}),
                        "result_summary": _truncate(text, is_error),
                    }
                )
            elif name in ("Agent", "Task"):
                tool_input = body.get("input") if isinstance(body.get("input"), dict) else {}
                agent_type = tool_input.get("subagent_type", tool_input.get("description", ""))
                started = subagent.get("agent_id") or body.get("id", "")
                events.append(
                    {
                        "type": "agent_start",
                        "timestamp": node.get("ts"),
                        "agent_id": started,
                        "agent_type": agent_type,
                    }
                )
                events.extend(
                    _tool_events(subagent.get("timeline") or [], session_id, agent_id=started)
                )
                events.append(
                    {
                        "type": "agent_stop",
                        "timestamp": result.get("ts") or node.get("ts"),
                        "agent_id": started,
                        "agent_type": agent_type,
                    }
                )
                continue
        if subagent.get("timeline"):
            events.extend(
                _tool_events(subagent["timeline"], session_id, agent_id=subagent.get("agent_id"))
            )
    return events


# ---------------------------------------------------------------------------
# MCP sidecar + core runner support helpers
# ---------------------------------------------------------------------------


def _log_query_timing(workflow: SDKWorkflowResult) -> None:
    """Append one timing record per agent query to ``OSPREY_E2E_QUERY_LOG``.

    The benchmark matrix points the variable at
    ``results/<model>__seed<seed>.queries.jsonl``; the dashboard reads it to show
    how long each model takes and how much of that time is spent waiting on the
    model rather than in OSPREY's tools. The figures are the agent SDK's own
    accounting on the final ``ResultMessage``. Off unless the variable is set.
    """
    path = os.environ.get("OSPREY_E2E_QUERY_LOG")
    result = workflow.result
    if not path or result is None:
        return
    usage = getattr(result, "usage", None) or {}
    record = {
        "test": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
        "duration_ms": getattr(result, "duration_ms", None),
        "duration_api_ms": getattr(result, "duration_api_ms", None),
        "num_turns": getattr(result, "num_turns", None),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "is_error": getattr(result, "is_error", None),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # instrumentation must never fail a test


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


def print_run_summary(title: str, result: SDKWorkflowResult) -> None:
    """Print what a live run did, under ``title``: tools, turns, cost, approvals, results.

    Debug output only -- shown under ``pytest -s`` or with a failure -- so a run that
    went wrong is diagnosable from its log without paying for it again. Approval
    decisions are printed for a :class:`HookObservedResult`.
    """
    print(f"\n--- {title} ---")
    print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    hook_events = getattr(result, "hook_events", None)
    if hook_events is not None:
        print(f"  hook_events: {len(hook_events)}")
        for evt in hook_events:
            print(f"    {evt.tool_name}: {evt.decision}")
    for trace in result.tool_traces:
        error_flag = " [ERROR]" if trace.is_error else ""
        parent_flag = (
            f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
        )
        print(f"  tool: {trace.name}{error_flag}{parent_flag}")
        print(f"    input keys: {list(trace.input)}")
        if trace.result:
            print(f"    result preview: {trace.result[:300]}")
    for block in result.text_blocks[:3]:
        print(f"  text: {block[:200]}")


def assert_agent_ran(result: SDKWorkflowResult) -> None:
    """Fail when the agent session ended in error before its first tool call.

    A deny test passes when nothing forbidden succeeded, and a session that
    never ran (no credentials — ``Not logged in`` — or an unreachable
    upstream) meets that without the gate being reached. Such a session ends
    ``is_error`` with no tool calls. A session that hits its turn cap ends
    ``is_error`` too but has tool calls, and a model that refuses in text alone
    ends without error; both did run, so neither fails here.
    """
    assert result.result is not None, "No ResultMessage received from SDK"
    assert result.tool_traces or not result.result.is_error, (
        f"The agent never ran: {result.result.result}"
    )


def e2e_budget_scale() -> float:
    """Per-query budget multiplier for the model under test.

    The base ``max_budget_usd`` caps across the e2e suite are sized for
    :data:`tests.e2e.provider.E2E_MODEL`, which every lane builds with when its
    call site names no model. Pricier models (Sonnet, Opus) cost several times
    more per token, so the same multi-step task reaches the cap and hard-errors
    mid-query (``Reached maximum budget``) — a cost artifact that deflates their
    benchmark score for reasons unrelated to capability. The model matrix runner
    (``scripts/benchmark/run_e2e_for_model.sh``) sets ``OSPREY_E2E_BUDGET_SCALE``
    per model so the cap **and** the cost-ceiling assertions scale together.
    Unset, the scale is 1.0 and each call site's cap applies as written.

    ``inf`` lifts both: the runner sets it for a model reached through the
    OpenAI-protocol proxy, whose tokens the SDK prices at Claude rates, so any
    dollar figure for it is fiction. The turn limit and the timeout still bound
    such a query.
    """
    try:
        scale = float(os.environ.get("OSPREY_E2E_BUDGET_SCALE", "1.0"))
    except ValueError:
        return 1.0
    return scale if scale > 0 else 1.0


def e2e_budget_cap(max_budget_usd: float) -> float | None:
    """The ``max_budget_usd`` to hand the SDK for a call site's base cap.

    ``None`` when :func:`e2e_budget_scale` is infinite: the SDK then sets no
    dollar cap at all, rather than being handed an infinity it cannot parse.
    """
    cap = max_budget_usd * e2e_budget_scale()
    return None if math.isinf(cap) else cap


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
    resume: str | None = None,
) -> SDKWorkflowResult:
    """Run a query via the Claude Agent SDK and collect full tool traces.

    Args:
        repo: Deployment repo root (what :func:`init_project` returns). The
            agent runs with its cwd at the render, ``<repo>/build``, where
            ``.mcp.json``, ``.claude/`` and ``CLAUDE.md`` live.
        prompt: The user prompt to send.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use. Defaults to the project's main model resolved
            from ``config.yml``: ``claude_code.default_model``, else the
            provider entry's ``default_model``.
        disallowed_tools: Optional list of tool names to forbid at the SDK
            level. Forwarded to the Claude Code CLI as ``--disallowedTools``,
            which takes precedence over ``permission_mode=bypassPermissions``
            and over per-tool ``permissions_allow`` in ``.mcp.json``. Use this
            to architecturally force delegation to subagents (the main agent
            cannot call a disallowed tool even when settings would permit it).
        resume: Optional session id of an earlier run (its
            ``result.session_id``); the prompt is then the next turn of that
            conversation. The traces collected are this turn's only.

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.
    """
    # Collect stderr lines for debugging CLI failures
    stderr_lines: list[str] = []

    render = render_dir(repo)
    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(render),
        cwd=str(render),
        cli_path=project_agent_cli_path(render),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=e2e_budget_cap(max_budget_usd),
        env=sdk_env(render),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        disallowed_tools=_with_one_shot_floor(disallowed_tools),
        resume=resume,
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

    _log_query_timing(workflow)
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

    def _call_without_context(tool_name: str, tool_input: dict[str, Any], _context: Any) -> bool:
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
        model: Model to use. Defaults to the project's main model resolved
            from ``config.yml``.
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
    # Under the harness's own state, trusted as a deployment's render is, so the
    # project's allow rules apply here too. Under the developer's state, nothing
    # is seeded and the run is what it would be on their machine.
    trusted = harness_config_dir is not None
    if trusted:
        trust_render(repo)
    options = ClaudeAgentOptions(
        model=model if model is not None else resolve_default_model(render),
        cwd=str(render),
        cli_path=project_agent_cli_path(render),
        permission_mode="default",
        max_turns=max_turns,
        max_budget_usd=e2e_budget_cap(max_budget_usd),
        env=sdk_env(render),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        can_use_tool=_can_use_tool,
        disallowed_tools=_with_one_shot_floor(disallowed_tools),
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

    untrusted = [line for line in stderr_lines if is_untrusted_allow_rules_notice(line)]
    if trusted and untrusted:
        raise RuntimeError(f"the CLI left the render's allow rules off: {untrusted[0]}")

    # See run_sdk_query: sub-agent tool calls live in on-disk transcripts.
    _harvest_subagent_traces(workflow, pending_tools, render)

    workflow.hook_events = hook_events
    _log_query_timing(workflow)
    _persist_mcp_sidecar(workflow, repo)
    return workflow


# ---------------------------------------------------------------------------
# Hook stdout, read back out of the session transcript
# ---------------------------------------------------------------------------


#: Prefix shared by every transcript record that carries a hook's own stdout.
#:
#: ``hook_success`` — a hook that exited 0 — is the one spelling verified
#: against CLI 2.1.241. The kinds Claude Code writes when a hook fails, times
#: out, or is killed share this prefix, but their exact spelling is NOT
#: verified here, which is why this matches on the prefix rather than on a
#: literal set: a hook that died mid-approval is precisely the run someone
#: needs to read, and an allow-list of one would drop it silently.
HOOK_ATTACHMENT_PREFIX = "hook_"


def hook_attachments(result: SDKWorkflowResult, render: Path) -> list[dict[str, Any]]:
    """Every hook-stdout attachment in this session's transcript, in order.

    Claude Code does not put a hook's output on the SDK wire. It writes each
    invocation's raw stdout into the session transcript as an attachment record
    carrying ``hookName``, ``hookEvent``, the ``toolUseID`` it gated, and the
    ``stdout`` itself — so that text is recoverable exactly, and joins to a tool
    trace by id rather than by guessing at ordering. This is the only channel
    that carries it; see :func:`tests.e2e.test_target_switch_agentic.approval_prompts`
    for the SDK field that looks like it should and does not.

    Dicts come back raw and unprojected. A reader chasing an unexplained
    refusal does not know in advance which key holds the answer, and a helper
    that decides for them turns a diagnostic into a guess.

    Returns an empty list when the session id or the transcript cannot be
    resolved, and skips any line that is not a JSON object: callers assert on
    what they expected to find, and a bare "nothing here" is a clearer failure
    than an exception thrown from a path that only exists to observe.
    """
    session_id = getattr(result.result, "session_id", None)
    if not session_id:
        return []
    transcript = claude_project_dir(Path(render).resolve()) / f"{session_id}.jsonl"
    if not transcript.is_file():
        return []

    attachments: list[dict[str, Any]] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        attachment = record.get("attachment")
        if not isinstance(attachment, dict):
            continue
        if not str(attachment.get("type") or "").startswith(HOOK_ATTACHMENT_PREFIX):
            continue
        attachments.append(attachment)
    return attachments


# ---------------------------------------------------------------------------
# Agent transcripts as CI artifacts
# ---------------------------------------------------------------------------


#: How much of one tool result the LLM judge reads. A ``get_run_figure`` or
#: ``get_run_data`` payload carries its measured values thousands of characters
#: in (past the run id, columns and axis labels), so a short preview hands the
#: judge a response whose numbers it cannot check.
JUDGE_RESULT_CHARS = 8000


def judge_result_preview(text: str) -> str:
    """One tool result as the LLM judge reads it: on one line, cut at
    :data:`JUDGE_RESULT_CHARS`.

    Whitespace is collapsed so that a result is one line of the execution
    trace; the judge's excerpt keeps the first and last 50 trace lines, and a
    multi-line result would otherwise push other tool calls out of it.
    """
    flat = " ".join(text.split())
    return flat[:JUDGE_RESULT_CHARS] + ("…" if len(flat) > JUDGE_RESULT_CHARS else "")


#: Sub-directory of ``OSPREY_CI_DIAG_DIR`` that transcripts are written to.
#: Separate from the records :mod:`ci_diagnostics` keeps in the same
#: directory, so a per-worker event log and a per-test transcript can never
#: collide on a name.
TRANSCRIPT_SUBDIR = "agent"


def _transcript_payload(
    name: str, result: SDKWorkflowResult, render: Path | None = None
) -> dict[str, Any]:
    """The serialisable form of one agent run — deliberately UNTRUNCATED.

    Truncation is the whole reason this exists. ``_to_workflow_result``
    previews every tool result (:func:`judge_result_preview`) before the judge
    ever sees it, so a result longer than the preview is never seen whole.
    When the judge then reports that a response's numbers "cannot be verified
    from the execution trace", nothing anywhere records what the tools
    actually returned, and the verdict text is left as the only account of a
    response nobody can re-read.

    ``mcp_server_status`` is included because it is the infra-vs-model
    discriminator: a tool the agent never called means one thing if the
    handshake offered it and quite another if it never registered.

    ``hook_attachments`` appears only when *render* is given, so a caller with
    no render on hand keeps exactly the payload it had. Its absence and an
    empty list mean different things: no key says nobody looked, and ``[]``
    says the transcript was read and held no hook output.
    """
    payload: dict[str, Any] = {
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
    if render is not None:
        payload["hook_attachments"] = hook_attachments(result, render)
    return payload


def dump_agent_transcript(
    name: str, result: SDKWorkflowResult, *, render: Path | None = None
) -> Path | None:
    """Persist one agent run's full response and tool traces as a CI artifact.

    Gated on ``OSPREY_CI_DIAG_DIR`` exactly like :mod:`ci_diagnostics`:
    unset — which is every local run — writes nothing and returns ``None``.
    The lanes that set it already upload that directory, and
    ``capture-ci-diagnostics`` creates its own subdirectory rather than
    clearing the tree, so what is written here survives into the artifact.

    Call this BEFORE the assertions, never after. The runs worth reading are
    the ones that fail, and a dump placed below a judge assertion never
    executes on exactly those.

    Pass *render* for a run whose hooks matter. The artifact then also carries
    every hook's raw stdout (see :func:`hook_attachments`), which is the only
    record of why a call was asked about or refused — tool traces show that a
    write was gated, never what the operator was asked. Omit it and the payload
    is unchanged.

    Never raises. A diagnostic that can fail the test it was only meant to
    observe would mask the very failure it exists to explain — the same reason
    every probe in the capture action ends in ``|| true``. That covers the
    transcript read too: it happens inside the same guard, below the arming
    check, so an unarmed run never goes near the session directory.
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
        # A name dumped before in this run gets a numbered file: a rerun must
        # not erase the attempt it is retrying, which is the one worth reading.
        attempt = 2
        while target.exists():
            target = target_dir / f"{safe}.{attempt}.json"
            attempt += 1
        target.write_text(
            json.dumps(_transcript_payload(name, result, render), indent=2, default=str),
            encoding="utf-8",
        )
        return target
    except Exception:
        return None
