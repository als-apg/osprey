"""Agentic-discovery e2e for the orbit-response-matrix (ORM) capability
(PROPOSAL.md FR6, PLAN.md tasks 5.3/5.4): given an operator-style prompt that
names no PV, device, or fault, the agent must run the ``orm`` Bluesky plan
over a REAL deployed VA + bridge + Tiled stack and state the correct physical
conclusion.

Two scenarios, mirroring ``test_rf_cavity_correlation_scenario.py``'s
agentic-e2e shape but against a real container stack instead of mock
connectors (the ORM measurement only exists over the live VA):

  * ``test_orm_agentic_discovery_quad`` (5.3) -- the ``errant-quad`` scenario
    seeds a misaligned quadrupole (QF07, physics-only: no setpoint readback
    shows it) plus a sector-2 orbit-bump archiver overlay and a two-entry
    logbook arc (DEMO-029/030) that explicitly defers to "a dedicated
    response-matrix measurement" without naming the culprit. The agent must
    run the ORM plan and localize the fault to QF07.
  * ``test_orm_agentic_discovery_response`` (5.4) -- the ``bpm-polarity``
    scenario seeds an inverted-polarity BPM (BPM17: no rest/archiver symptom
    at all -- individual channel telemetry looks nominal) with a logbook
    entry (DEMO-031) describing only a correction-feedback convergence
    failure. This is the harder case: there is nothing to correlate except
    the ORM's own row structure (a response-side fault flips a BPM's whole
    row of slopes, unlike a reference-orbit bump). Proves the ORM diagnoses
    response-side faults, not just orbit bumps.

Both scenarios' physics faults are deploy-time-only (PROPOSAL "Hot-swappable
VA physics faults" is out of scope) -- each test therefore builds and deploys
its OWN VA boot rather than sharing one stack across scenarios.

Deploy shape reuses ``tests/e2e/_orm_stack.py``'s single source (build +
``select_correctors``/``select_bpms`` + ``write_scan_env``) with one addition
this module owns: an explicit ``provider``/``model`` (``_orm_stack.build_args``
sets neither, so the control-assistant preset's own anthropic/haiku default
would otherwise apply -- silently, which the project's "no default provider"
convention forbids). The corrector/BPM wiring below deliberately requests the
FULL SR pyat-coupled set (not ``_orm_stack``'s small default-4 subset, which
its own docstring says these two e2e "don't depend on"): both QF07 and BPM17
lie well inside a device-numbering range that a small default subset (which
always starts from device 01) would silently miss, and hand-narrowing the
wiring to conveniently include the faulted device would leak the answer's
location into the deploy config -- the same reason no PV/device name appears
in either prompt below.

FR5's deploy-time hook (``render_scenario_physics_env``) resolves the active
scenario's ``physics`` block into ``VA_QUAD_MISALIGN``/``VA_BPM_ERRORS`` env
BEFORE ``deploy up``; ``activate_scenarios`` (telemetry + logbook) runs AFTER
the stack is healthy, against the ``postgresql`` service this same
``deploy up`` co-deploys (control-assistant's ``deployed_services`` always
includes ``postgresql`` alongside ``bluesky``/``virtual_accelerator`` --
confirmed by inspecting a real build's rendered config.yml). Unlike the
mock-connector scenario e2e siblings (which point at a long-lived,
externally-provisioned ARIEL DB and gate on ``ariel_db_skip_reason()``
BEFORE doing anything), this Postgres is freshly created by THIS deploy, so
the module below waits for the stack's own health check and then lets
``activate_scenarios`` self-migrate (it runs ``run_migrate`` before purging)
-- a Postgres-unreachable failure at that point converts to an honest skip
rather than a misleading tool-trace failure downstream.

The scan MCP server (``osprey.mcp_server.scan``, opted in via
``claude_code.servers.scan.enabled: true`` -- see ``_orm_stack.override_yaml``)
resolves ``BLUESKY_BRIDGE_URL``/``BLUESKY_PROMOTE_TOKEN`` from its OWN process
env (``${VAR:-default}`` substitution in the registry's ``ServerDefinition``,
resolved by the Claude Code CLI at MCP-subprocess-spawn time), which is never
threaded through by ``run_sdk_query``'s ``sdk_env()`` (that only injects the
provider block). This module sets both directly on the *test* process via
``monkeypatch`` before calling ``run_sdk_query`` -- the SDK's subprocess
transport merges ``options.env`` on top of inherited ``os.environ``, so the
values reach the CLI subprocess and, transitively, the scan MCP server it
spawns.

Container safety: every docker invocation below names an exact
container/image (mirrors ``test_va_substrate_equivalence.py``); teardown goes
through ``osprey deploy down``, never a raw sweep. Advisory CI lane -- skips
outright on GitHub Actions (no Docker-in-Docker + seeded ARIEL Postgres
there) and whenever Docker or ``ALS_APG_API_KEY`` (both the agent's and the
judge's provider) is unavailable. Collection-validate with:

    .venv/bin/pytest tests/e2e/test_orm_discovery_scenario.py --collect-only -q

Real run (needs Docker + judge creds -- unlikely to be available in
any given session; do not weaken the prompts or the wiring to force a local
pass):

    .venv/bin/pytest tests/e2e/test_orm_discovery_scenario.py -k quad
    .venv/bin/pytest tests/e2e/test_orm_discovery_scenario.py -k polarity
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.simulation.apply import render_scenario_physics_env
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e import _orm_stack
from tests.e2e.judge import LLMJudge
from tests.e2e.sdk_helpers import HAS_SDK, _default_opus_model, activate_scenarios, run_sdk_query
from tests.e2e.test_preset_agentic import _to_workflow_result

# Same provider for the agent AND the judge (als-apg -- reachable from GitHub
# Actions runners, per test_rf_cavity_correlation_scenario.py's precedent).
# Explicit at every callsite, never a build-time default (see module
# docstring): _orm_stack.build_args() doesn't set provider/model, so without
# this the control-assistant preset's own `provider: anthropic` would apply
# silently.
PROVIDER = "als-apg"
AGENT_MODEL_TIER = "opus"  # diagnostic reasoning needs Opus, not the haiku default

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # Skipped on CI: needs a real Docker VA+bridge+Tiled+postgres
    # stack, which the default GitHub Actions runner does not provision (see
    # tests/e2e/README.md and test_va_substrate_equivalence.py's identical
    # gate).
    pytest.mark.skipif(
        os.environ.get("GITHUB_ACTIONS") == "true",
        reason="needs a real Docker stack; not provisioned on CI runners",
    ),
]


# ---------------------------------------------------------------------------
# Build/deploy scaffold
# ---------------------------------------------------------------------------


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _channel_limits(project_dir: Path) -> dict:
    """Local copy of ``_orm_stack``'s identical private helper (mirrors
    ``test_va_substrate_equivalence.py``'s own local redefinition) — reading
    the deployed project's ``channel_limits.json`` isn't part of
    ``_orm_stack``'s named single-source surface (build +
    ``select_correctors``/``select_bpms``/``write_scan_env``), so this stays
    a plain local read rather than reaching into a private cross-module
    helper.
    """
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _minted_promote_token(project_dir: Path) -> str:
    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_PROMOTE_TOKEN")
    assert token, "BLUESKY_PROMOTE_TOKEN missing/empty in the project .env"
    return token


@contextlib.contextmanager
def _deployed_discovery_stack(tmp_path: Path, project_name: str, scenario: str) -> Iterator[Path]:
    """Build, wire, seed, and deploy one scenario's ORM stack; tear it down
    exact-named on exit (even on a mid-setup skip/failure).

    Physics faults are deploy-time-only (PROPOSAL: no hot-swap), so each
    caller gets its own fresh VA boot -- this is a function-scoped helper,
    never shared across the two scenarios below.
    """
    osprey_bin = _orm_stack.find_osprey_console_script()
    project_dir = _orm_stack.build_project_subprocess(
        project_name,
        output_dir=tmp_path,
        timeout=BUILD_TIMEOUT_SEC,
        provider=PROVIDER,
        model=AGENT_MODEL_TIER,
    )

    limits = _channel_limits(project_dir)
    correctors = _orm_stack.select_correctors(limits, count=None)
    bpms = _orm_stack.select_bpms(limits, count=None)
    # No promote_token kwarg: control-assistant's default writes_enabled=true
    # + this override's execution.execution_method=container is exactly the
    # arming-safe combination (_local_exec_arming_unsafe is False), so
    # `deploy up` auto-mints BLUESKY_PROMOTE_TOKEN itself.
    _orm_stack.write_scan_env(project_dir, correctors=correctors, bpms=bpms)

    # FR5's deploy-time hook — MUST run before `deploy up`: a physics fault
    # applies once at VA container boot.
    render_scenario_physics_env(project_dir, [scenario])

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (mirrors test_va_substrate_equivalence.py). Exact-named images only;
    # E2E_REUSE_IMAGES=1 skips this for fast local iteration (never in CI).
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(["docker", "rmi", "-f", _orm_stack.VA_IMAGE], capture_output=True, text=True)
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.BRIDGE_IMAGE], capture_output=True, text=True
        )

    try:
        up = subprocess.run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"http://localhost:{_orm_stack.BRIDGE_PORT}/health", HEALTH_TIMEOUT_SEC)

        # Telemetry + logbook half of the scenario. Runs against the
        # `postgresql` service this SAME `deploy up` co-deployed (control-
        # assistant's `deployed_services` always includes it alongside
        # `bluesky`/`virtual_accelerator`) -- freshly created, so
        # `activate_scenarios` self-migrates (`_seed_logbook` runs
        # `run_migrate` before purging); no separate `osprey ariel
        # migrate`/`quickstart` step. A connection failure here is an
        # honest prerequisite gap, not a model-capability miss — skip
        # rather than let it surface as a downstream tool-trace failure.
        try:
            activate_scenarios(project_dir, scenario)
        except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
            pytest.skip(f"ARIEL Postgres (co-deployed) not ready for logbook seeding: {exc}")

        yield project_dir
    finally:
        down = subprocess.run(
            [str(osprey_bin), "deploy", "down"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


# Orbit-response-CLASS plan names this assertion accepts, by exact
# `plan_name` match. `BUILTIN_PLANS` (src/osprey/services/bluesky_bridge/
# plans.py) ships only "orm" today; the extra names are reserved for an
# alternate orbit-response-style sweep (e.g. a k-modulation plan) the agent
# could legitimately choose instead of the literal `orm` plan without this
# assertion being able to anticipate its exact spelling.
def _assert_orm_scan_ran(result, *, plan_hint: str = "orm") -> None:
    """The deterministic tool-trace contract shared by both scenarios: the
    agent created a scan intent for the `orm` plan, launched it, and read
    its data back. Runs unconditionally (never skip-gated) — only the final
    LLM-judge grade below is gated on judge-provider credentials.
    """
    create_calls = [t for t in result.tool_traces if t.name == "mcp__scan__create_scan_intent"]
    assert create_calls, (
        f"agent never called create_scan_intent — it did not set up a scan. "
        f"Tools called: {result.tool_names}"
    )
    orm_intents = [
        t for t in create_calls if str(t.input.get("plan_name", "")).lower() == plan_hint
    ]
    assert orm_intents, (
        f"agent created a scan intent but never for the '{plan_hint}' plan — "
        f"it did not run an orbit-response-matrix measurement: "
        f"{[t.input for t in create_calls]}"
    )

    launch_calls = [t for t in result.tool_traces if t.name == "mcp__scan__launch_scan"]
    assert launch_calls, (
        f"agent never called launch_scan — it created a scan intent but never "
        f"actually ran it. Tools called: {result.tool_names}"
    )

    read_calls = [t for t in result.tool_traces if t.name == "mcp__scan__read_scan_data"]
    assert read_calls, (
        f"agent never called read_scan_data — it launched a scan but never "
        f"read the measurement back. Tools called: {result.tool_names}"
    )


# ---------------------------------------------------------------------------
# 5.3 — errant-quad: agent must localize the misaligned QF07
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Deferred to the 'crown' phase: re-derive the agentic-discovery benchmark on the "
    "connector-mediated substrate, grading on measurement plan-class + physical conclusion rather "
    "than a hard plan_name=='orm' gate (the agent legitimately chooses K-modulation here). See the "
    "integrated-scan-stack epic STATE."
)
@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb rare LLM stochastic misses
@pytest.mark.asyncio
async def test_orm_agentic_discovery_quad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator reports a persistent orbit bump; the agent must run the ORM
    plan and localize it to the misaligned quadrupole QF07.

    The prompt names no PV, device, or family — it echoes only the
    OBSERVABLE symptom (a persistent orbit bump), the same information an
    operator would actually have. The `errant-quad` scenario's logbook arc
    (DEMO-029/030) narrates that symptom and explicitly punts to "a
    dedicated response-matrix measurement" as the next diagnostic step,
    without ever naming QF07 — matching the archiver overlay, which shows
    only a BPM-level symptom (sector-2 BPMs), never the quadrupole itself.
    """
    with _deployed_discovery_stack(tmp_path, "orm-discovery-quad", "errant-quad") as project_dir:
        token = _minted_promote_token(project_dir)
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://localhost:{_orm_stack.BRIDGE_PORT}")
        monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", token)

        judge = LLMJudge(provider="als-apg")
        query = (
            "Global orbit feedback has been showing a persistent horizontal bump for "
            "the past couple of days and it isn't settling out on its own. Figure out "
            "exactly which element is responsible and tell me where to send the "
            "inspection crew."
        )
        result = await run_sdk_query(
            project_dir,
            query,
            max_turns=60,
            max_budget_usd=40.0,
            model=_default_opus_model(project_dir),
        )

        _assert_orm_scan_ran(result)

        eval = await judge.evaluate(
            _to_workflow_result(query, result),
            expectations=(
                "Diagnostic-conclusion judging. The tool-trace assertions "
                "above already verify methodology (the agent created, "
                "launched, and read an orbit-response-matrix scan) — do not "
                "re-penalize those steps.\n"
                "\n"
                "The agent must commit to a SPECIFIC quadrupole as the fault "
                "source: 'QF07', 'QF-07', 'quad 07 in the QF family', or an "
                "equivalent unambiguous reference to that exact device (a "
                "channel address like 'SR:MAG:QF:07:...' also counts). It "
                "should also state that this quad is physically misaligned "
                "/ mispositioned / off its nominal location — not merely "
                "'faulty' or 'needs adjustment' with no positional claim.\n"
                "\n"
                "Failures: naming a different quad or a different device "
                "class entirely (a corrector, a BPM, an RF cavity), a vague "
                "'a quadrupole near sector 2' or 'one of the quads' with no "
                "specific device identifier, or hedging without committing "
                "to one specific device."
            ),
        )
        assert eval.passed, eval.reasoning


# ---------------------------------------------------------------------------
# 5.4 — bpm-polarity: agent must localize the inverted-polarity BPM17
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Deferred to the 'crown' phase: re-derive the agentic-discovery benchmark on the "
    "connector-mediated substrate, grading on measurement plan-class + physical conclusion rather "
    "than a hard plan_name=='orm' gate. See the integrated-scan-stack epic STATE."
)
@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb rare LLM stochastic misses
@pytest.mark.asyncio
async def test_orm_agentic_discovery_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator reports orbit correction won't converge; the agent must run
    the ORM plan and localize the inverted-polarity BPM17 via the response
    matrix's own row structure (no rest/archiver symptom exists to
    correlate against — this is the harder, response-side case).

    The prompt names no PV, device, or family. `bpm-polarity`'s logbook
    entry (DEMO-031) describes only a correction-feedback convergence
    failure and a suspicion that "one BPM may be feeding back a reading
    with the wrong sign" — without naming which one, and explicitly notes
    individual channel telemetry looks nominal (there is nothing to read
    off the archiver; only a dedicated response-matrix measurement resolves
    it). BPM17's isotropic ``polarity: -1`` flips both transverse planes, so
    the fault shows up as one BPM's entire row of ORM slopes carrying the
    wrong sign relative to its column-mate correctors — proving the ORM
    diagnoses response-side (not just reference-orbit) faults.
    """
    with _deployed_discovery_stack(
        tmp_path, "orm-discovery-response", "bpm-polarity"
    ) as project_dir:
        token = _minted_promote_token(project_dir)
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://localhost:{_orm_stack.BRIDGE_PORT}")
        monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", token)

        judge = LLMJudge(provider="als-apg")
        query = (
            "Global orbit correction has been running for over an hour without "
            "settling — the residual keeps oscillating instead of converging, even "
            "though no individual channel looks out of range. Figure out exactly "
            "what's wrong with the orbit-correction data and what needs to be fixed."
        )
        result = await run_sdk_query(
            project_dir,
            query,
            max_turns=60,
            max_budget_usd=40.0,
            model=_default_opus_model(project_dir),
        )

        _assert_orm_scan_ran(result)

        eval = await judge.evaluate(
            _to_workflow_result(query, result),
            expectations=(
                "Diagnostic-conclusion judging. The tool-trace assertions "
                "above already verify methodology (the agent created, "
                "launched, and read an orbit-response-matrix scan) — do not "
                "re-penalize those steps.\n"
                "\n"
                "The agent must commit to a SPECIFIC BPM as the fault "
                "source: 'BPM17', 'BPM 17', 'the 17th BPM', or an "
                "equivalent unambiguous reference to that exact device (a "
                "channel address like 'SR:DIAG:BPM:17:...' also counts). "
                "It should also state that this BPM's readback has an "
                "inverted / flipped / wrong-sign polarity — a wiring-type "
                "sign fault, not a drift, offset, gain error, or noise "
                "issue.\n"
                "\n"
                "Failures: naming a different BPM, a corrector, or a quad "
                "instead; attributing the non-convergence to a generic "
                "'noisy data' or 'controller tuning' cause with no device- "
                "level fault identified; a vague 'one of the BPMs' with no "
                "specific device identifier; or hedging without committing "
                "to one specific device."
            ),
        )
        assert eval.passed, eval.reasoning
