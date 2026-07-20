"""Agentic dual-fault-localization e2e for the orbit-response-matrix (ORM)
capability (PROPOSAL.md FR6, PLAN.md task 5.1): given ONE symptom-only
operator prompt that names neither a diagnostic method nor a device, the
agent must run a single orbit-response-class scan over a REAL deployed VA +
bridge + Tiled stack and localize BOTH independently seeded faults from that
one measurement.

This is the headline behavioral proof for the phase: earlier agentic-
discovery drafts (a now-deleted sibling module -- see below) each seeded
exactly one fault and hard-gated the tool trace on ``plan_name == "orm"``.
Both of those were deferred to this "crown" phase for
re-derivation (see the deleted draft's own ``pytest.mark.skip`` reasons and
``.claude/plans/integrated-scan-stack-p7-crown``): grade on measurement
PLAN-CLASS (an orbit-response-shaped scan: it drives correctors and reads
BPM detectors in one run) rather than a literal plan name, since a legitimate
agent could choose a differently-named but structurally equivalent plan. This
module's :func:`_assert_orbit_response_scan_ran` never compares
``plan_name`` -- only the tool-call sequence and the ``plan_args`` device-
class shape.

The ``orm-dual-fault`` scenario bundle (task 3.1,
``src/osprey/templates/apps/control_assistant/data/simulation/scenarios/orm-dual-fault/``)
declares two disjoint, single-channel-invisible physics faults: BPM17's
horizontal readback has an inverted polarity (a wiring/sign fault -- flips
that BPM's entire row of ORM slopes), and HCM01 is desensitized to half its
nominal steering gain (a degraded driver -- halves that corrector's entire
column of ORM slopes). Neither shows up as an out-of-range channel; both
only surface in the structure of a real orbit-response measurement. The
scenario carries no ``logbook.json`` (unlike ``bpm-polarity``), so
``activate_scenarios`` seeds no narrative beyond ``nominal``'s ambient
entries -- deliberate, since a logbook hint here would leak the answer into
data the agent could search instead of measuring.

Deploy shape reuses ``tests/e2e/_orm_stack.py``'s single source (build +
``override_yaml`` for the VA/container/scan-MCP flip + ``select_correctors``/
``select_bpms``/``write_scan_env``), exactly as the deleted draft did. The
corrector/BPM wiring below requests the FULL SR pyat-coupled set (``count=
None``), not a small default slice: both HCM01 and BPM17 need to be in range,
and hand-narrowing the wiring to conveniently include them would leak the
answer's location into the deploy config -- the same reason neither device
appears in the prompt below.

Grading is two-part (see PLAN.md's acceptance gate for this task):

  (a) A DETERMINISTIC structural floor -- :func:`_assert_orbit_response_scan_ran`
      -- asserting the tool trace contains ``set_draft`` ->
      ``launch_run`` -> ``get_run_data`` IN THAT ORDER, and that the
      staged ``set_draft`` call's ``plan_args_patch`` carry both a
      non-empty ``correctors`` list and a non-empty ``detectors`` list (the
      ``orm`` plan's own device-class contract, see ``plans_core/orm.py``'s
      ``PARAMS`` -- checked structurally, never by ``plan_name``).
  (b) An :class:`~tests.e2e.judge.LLMJudge` grade of the agent's final
      conclusion against :data:`_JUDGE_EXPECTATIONS`, which requires BOTH
      root causes to be named (BPM17's polarity flip AND HCM01's gain
      deficit) and is told not to re-penalize methodology the structural
      floor above already covers.

:func:`test_judge_discriminates_dual_fault_conclusions` dry-verifies (b) by
itself, offline of any deploy: it feeds the SAME ``_JUDGE_EXPECTATIONS``
contract a hand-written correct conclusion and a hand-written wrong/
incomplete control conclusion and asserts the judge passes the former and
fails the latter -- proving the grading contract actually discriminates
before spending a live Docker run on it.
:func:`test_structural_floor_accepts_orbit_response_class_sequence` and its
``_rejects_*`` siblings dry-verify (a) the same way, against hand-built
``ToolTrace`` fixtures. None of these three need Docker or a live agent run,
so they carry lighter marks than the live test below and (unlike it) are not
skipped on GitHub Actions -- see each test's own marks rather than a single
module-wide ``pytestmark`` (this file intentionally does not use one, so the
live test's Docker/CI skips don't also blind the offline grading-contract
checks).

The two skipped 5.3/5.4 single-fault discovery drafts this module supersedes
have been deleted; this module is the sole ORM agentic-discovery e2e going
forward.

Container safety: every docker invocation below names an exact
container/image (mirrors ``test_va_substrate_equivalence.py``); teardown goes
through ``osprey deploy down``, never a raw sweep.

Collection-validate with::

    .venv/bin/pytest tests/e2e/test_orm_agentic_scenario.py --collect-only -q

The offline grading-contract checks can run right now (no Docker needed,
only ``ALS_APG_API_KEY`` for the judge dry-run)::

    .venv/bin/pytest tests/e2e/test_orm_agentic_scenario.py -k "not agentic_localizes"

MANUAL ACCEPTANCE GATE -- not run by this suite, needs Docker + a multi-
minute native VA image build + ``ALS_APG_API_KEY``: deploy the
``orm-dual-fault`` VA stack, send :data:`QUERY` to a real agent, and confirm
it runs one orbit-response-class scan and names both BPM17's polarity flip
and HCM01's gain deficit::

    .venv/bin/pytest tests/e2e/test_orm_agentic_scenario.py -k agentic_localizes
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

from osprey.agent_runner import SDKWorkflowResult, ToolTrace
from osprey.simulation.apply import render_scenario_physics_env
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e import _orm_stack
from tests.e2e.judge import LLMJudge, WorkflowResult
from tests.e2e.sdk_helpers import HAS_SDK, _default_opus_model, activate_scenarios, run_sdk_query
from tests.e2e.test_preset_agentic import _to_workflow_result

# Same provider for the agent AND the judge (als-apg -- reachable from GitHub
# Actions runners). Explicit at every callsite, never a build-time default
# (this project's "no default provider" convention): _orm_stack.build_args()
# doesn't set provider/model, so without this the control-assistant preset's
# own `provider: anthropic` default would apply silently.
PROVIDER = "als-apg"
AGENT_MODEL_TIER = "opus"  # diagnostic reasoning needs Opus, not the haiku default

SCENARIO = "orm-dual-fault"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0

# MCP tool names, decoupled from any plan name -- see module docstring.
SET_DRAFT = "mcp__bluesky__set_draft"
LAUNCH_RUN = "mcp__bluesky__launch_run"
GET_RUN_DATA = "mcp__bluesky__get_run_data"


# ---------------------------------------------------------------------------
# The one symptom-only operator prompt, and the judge's grading contract for
# its conclusion. Both are module-level constants (a single source) so the
# live agentic test and the offline judge dry-run below can never drift.
# ---------------------------------------------------------------------------

# Names no plan/method/tool ("ORM", "response matrix", "grid scan", "measure
# the response", etc. never appear) and no device ("BPM17"/"HCM01" never
# appear). Two clauses, each independently salient the way an operator who
# only watches the orbit-feedback display (not individual channel readbacks)
# would actually notice them: a feedback loop that fights itself in part of
# the ring (the row-flip signature of a sign-inverted BPM) and a steering
# actuator that has lost most of its authority (the column signature of a
# desensitized corrector).
QUERY = (
    "Global orbit correction has been fighting itself all shift: the "
    "feedback residual won't settle, and when the loop pushes harder in one "
    "part of the ring, the local orbit there actually gets worse instead of "
    "better. Separately, I'm seeing one of the feedback loop's actuators "
    "come back maxed out for almost no orbit effect, like it's lost most of "
    "its authority. Both symptoms have been there since this morning and I "
    "can't run either of them down from the individual channel readbacks "
    "one at a time. Figure out exactly what's wrong and where, so I know "
    "what to send a tech out to fix."
)

_JUDGE_EXPECTATIONS = (
    "Diagnostic-conclusion judging only. A separate deterministic check "
    "already verified the METHOD (the agent created, launched, and read a "
    "single orbit-response-class scan -- one that drove a set of correctors "
    "and read a set of BPM detectors together) -- do not re-penalize that "
    "methodology; grade only whether the agent's final conclusion correctly "
    "and completely identifies the two independent faults below.\n"
    "\n"
    "The agent must commit to BOTH of the following, each naming a SPECIFIC "
    "device (not merely 'a BPM' or 'one of the correctors'):\n"
    "\n"
    "1. BPM17 (acceptable forms: 'BPM17', 'BPM 17', 'the 17th BPM', or an "
    "unambiguous channel/address reference to that exact device, e.g. "
    "'SR:DIAG:BPM:17:...') has an inverted / flipped / wrong-sign polarity "
    "on its position readback -- a wiring-type sign fault, not a drift, "
    "offset, gain error, or noise issue.\n"
    "2. HCM01 (acceptable forms: 'HCM01', 'HCM 01', 'corrector 01 in the "
    "HCM family', or an unambiguous channel/address reference to that exact "
    "device, e.g. 'SR:MAG:HCM:01:...') has reduced / weakened / desensitized "
    "steering strength -- a gain deficit (roughly half nominal), not an "
    "open circuit, a stuck setpoint, or a total failure.\n"
    "\n"
    "PASS requires both (1) and (2) to be named as specific, committed "
    "conclusions. FAIL if only one of the two faults is named, if either "
    "fault is attributed to the wrong device or the wrong device class (a "
    "quad, an RF cavity, a different BPM/corrector), if either is "
    "mischaracterized (e.g. the BPM fault called 'noisy' rather than "
    "sign-flipped, or the corrector fault called 'dead'/'stuck' rather than "
    "weak), or if either conclusion is vague/hedged with no specific device "
    "identifier."
)


# ---------------------------------------------------------------------------
# Structural floor -- deterministic, never skip-gated, never compares
# plan_name. See module docstring part (a).
# ---------------------------------------------------------------------------


def _first_index_after(traces: list[ToolTrace], after: int, predicate) -> int | None:
    """Index of the first trace beyond ``after`` satisfying ``predicate``, or
    ``None``. Small local helper so the sequence floor below reads as one
    ordered walk rather than three independent ``any()`` checks."""
    for i in range(after + 1, len(traces)):
        if predicate(traces[i]):
            return i
    return None


def _is_orbit_response_draft(trace: ToolTrace) -> bool:
    """True for a ``set_draft`` call whose ``plan_args_patch`` carry both a
    non-empty ``correctors`` list and a non-empty ``detectors`` list -- the
    ``orm`` plan's own device-class contract (``plans_core/orm.py``'s
    ``PARAMS``), checked structurally so an agent that legitimately picks a
    differently-named but structurally equivalent plan still satisfies this
    floor. Never compares ``plan_name``.
    """
    if trace.name != SET_DRAFT:
        return False
    plan_args_patch = trace.input.get("plan_args_patch")
    if not isinstance(plan_args_patch, dict):
        return False
    correctors = plan_args_patch.get("correctors")
    detectors = plan_args_patch.get("detectors")
    return (
        isinstance(correctors, list)
        and bool(correctors)
        and isinstance(detectors, list)
        and bool(detectors)
    )


def _assert_orbit_response_scan_ran(result: SDKWorkflowResult) -> None:
    """The deterministic tool-trace contract: the agent staged an orbit-
    response-class scan draft, launched it, and read its data back, IN
    THAT ORDER. Runs unconditionally (never skip-gated) -- only the judge
    grade in the live test is gated on judge-provider credentials.

    Decoupled from any literal plan name -- see :func:`_is_orbit_response_draft`.
    """
    traces = result.tool_traces

    draft_idx = _first_index_after(traces, -1, _is_orbit_response_draft)
    assert draft_idx is not None, (
        "agent never staged an orbit-response-class scan draft (a "
        "set_draft call whose plan_args_patch carry both a non-empty "
        "'correctors' list and a non-empty 'detectors' list). "
        f"set_draft calls seen: "
        f"{[t.input for t in traces if t.name == SET_DRAFT]}"
    )

    launch_idx = _first_index_after(traces, draft_idx, lambda t: t.name == LAUNCH_RUN)
    assert launch_idx is not None, (
        "agent staged an orbit-response-class scan draft but never called "
        f"launch_run afterward. Tools called: {[t.name for t in traces]}"
    )

    read_idx = _first_index_after(traces, launch_idx, lambda t: t.name == GET_RUN_DATA)
    assert read_idx is not None, (
        "agent launched the scan but never called get_run_data afterward "
        f"to read the measurement back. Tools called: {[t.name for t in traces]}"
    )


# ---------------------------------------------------------------------------
# Deploy scaffold -- single scenario, single stack (mirrors the deleted
# draft's pattern; VA + container flip lives in _orm_stack.override_yaml()).
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
    ``test_va_substrate_equivalence.py``'s own local redefinition) -- reading
    the deployed project's ``channel_limits.json`` isn't part of
    ``_orm_stack``'s named single-source surface.
    """
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _minted_launch_token(project_dir: Path) -> str:
    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env"
    return token


@contextlib.contextmanager
def _deployed_dual_fault_stack(tmp_path: Path, project_name: str) -> Iterator[Path]:
    """Build, wire, seed, and deploy the ``orm-dual-fault`` VA stack; tear it
    down exact-named on exit (even on a mid-setup skip/failure).

    A physics fault is deploy-time-only (PROPOSAL: no hot-swap), so this is a
    single function-scoped boot -- both faults land in the SAME activation,
    matching the task contract ("one activation, disjoint devices").
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
    # Full pyat-coupled set (count=None), not a small default slice: BPM17
    # and HCM01 must both be in range, and hand-narrowing the wiring to
    # conveniently include them would leak the answer's location into the
    # deploy config (see module docstring).
    correctors = _orm_stack.select_correctors(limits, count=None)
    bpms = _orm_stack.select_bpms(limits, count=None)
    # No launch_token kwarg: control-assistant's default writes_enabled=true
    # + this override's execution.execution_method=container is exactly the
    # arming-safe combination, so `deploy up` auto-mints BLUESKY_LAUNCH_TOKEN.
    _orm_stack.write_scan_env(project_dir, correctors=correctors, bpms=bpms)

    # FR5's deploy-time hook -- MUST run before `deploy up`: a physics fault
    # applies once at VA container boot. Both faults in orm-dual-fault's
    # single `physics` block render together here.
    render_scenario_physics_env(project_dir, [SCENARIO])

    # Force fresh --dev builds so the deployed containers run CURRENT source.
    # Exact-named images only; E2E_REUSE_IMAGES=1 skips this for fast local
    # iteration (never in CI).
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.va_image(project_name)],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.bridge_image(project_name)],
            capture_output=True,
            text=True,
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

        # orm-dual-fault has no logbook.json (deliberate -- see module
        # docstring), so this only seeds `nominal`'s ambient entries against
        # the co-deployed `postgresql` service. A connection failure here is
        # an honest prerequisite gap, not a model-capability miss -- skip
        # rather than let it surface as a downstream tool-trace failure.
        try:
            activate_scenarios(project_dir, SCENARIO)
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


# ---------------------------------------------------------------------------
# THE live behavioral test -- MANUAL acceptance gate (see module docstring).
# Deliberately NOT covered by a module-wide `pytestmark`: only this test
# needs Docker/a live agent run, and blinding the offline grading-contract
# checks below behind the same skips would defeat their purpose.
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_als_apg
@pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="needs a real Docker stack; not provisioned on CI runners",
)
@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb rare LLM stochastic misses
@pytest.mark.asyncio
async def test_orm_dual_fault_agentic_localizes_both_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MANUAL ACCEPTANCE GATE: given only the symptom-only :data:`QUERY`, the
    agent must run ONE orbit-response-class scan and localize BOTH BPM17's
    polarity flip and HCM01's gain deficit from that single measurement.

    Not run by CI (needs Docker + a native VA image build +
    ``ALS_APG_API_KEY``) -- see the module docstring's manual-gate command.
    """
    with _deployed_dual_fault_stack(tmp_path, "orm-dual-fault-agentic") as project_dir:
        token = _minted_launch_token(project_dir)
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", f"http://localhost:{_orm_stack.BRIDGE_PORT}")
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", token)

        result = await run_sdk_query(
            project_dir,
            QUERY,
            max_turns=60,
            max_budget_usd=40.0,
            model=_default_opus_model(project_dir),
        )

        _assert_orbit_response_scan_ran(result)

        judge = LLMJudge(provider=PROVIDER)
        eval = await judge.evaluate(
            _to_workflow_result(QUERY, result), expectations=_JUDGE_EXPECTATIONS
        )
        assert eval.passed, eval.reasoning


# ---------------------------------------------------------------------------
# Offline grading-contract checks -- exercise (a) the structural floor and
# (b) the judge against the SAME contracts the live test above uses, without
# a VA deploy. No Docker needed; the judge check needs only ALS_APG_API_KEY.
# See module docstring.
# ---------------------------------------------------------------------------


def _synthetic_dual_fault_trace() -> list[ToolTrace]:
    """A hand-built tool trace shaped like a real dual-fault agentic run:
    one orbit-response-class set_draft (correctors + detectors), then a
    launch at that revision, then a data read -- in order."""
    return [
        ToolTrace(
            name=SET_DRAFT,
            input={
                "plan_name": "orm",
                "plan_args_patch": {
                    "correctors": ["corrector_01", "corrector_02", "corrector_03"],
                    "detectors": ["bpm_01", "bpm_17", "bpm_23"],
                    "span_a": 1.0,
                    "num": 9,
                },
            },
            result='{"revision": 1, "changed": ["correctors", "detectors"], "plan_name": "orm"}',
        ),
        ToolTrace(
            name=LAUNCH_RUN,
            input={"draft_revision": 1},
            result='{"id": "run-1", "status": "completed"}',
        ),
        ToolTrace(name=GET_RUN_DATA, input={"run_id": "run-1"}, result="{...}"),
    ]


def test_structural_floor_accepts_orbit_response_class_sequence() -> None:
    """Floor is decoupled from any literal plan_name -- only the device-class
    shape (correctors + detectors) and the draft -> launch -> read order
    matter."""
    result = SDKWorkflowResult(tool_traces=_synthetic_dual_fault_trace())
    _assert_orbit_response_scan_ran(result)  # must not raise


def test_structural_floor_rejects_non_orbit_response_plan() -> None:
    """A draft whose plan_args_patch carry no correctors/detectors pair (e.g. a
    generic n-d grid_scan over unrelated axes) must NOT satisfy the floor --
    proving it isn't just "any scan ran"."""
    traces = [
        ToolTrace(
            name=SET_DRAFT,
            input={
                "plan_name": "grid_scan",
                "plan_args_patch": {"axes": ["some_motor"], "num": [5]},
            },
            result='{"revision": 1, "changed": ["axes", "num"], "plan_name": "grid_scan"}',
        ),
        ToolTrace(
            name=LAUNCH_RUN,
            input={"draft_revision": 1},
            result='{"id": "run-2", "status": "completed"}',
        ),
        ToolTrace(name=GET_RUN_DATA, input={"run_id": "run-2"}, result="{...}"),
    ]
    with pytest.raises(AssertionError):
        _assert_orbit_response_scan_ran(SDKWorkflowResult(tool_traces=traces))


def test_structural_floor_rejects_out_of_order_sequence() -> None:
    """draft/launch/read out of trace order (e.g. a read before the scan
    was ever launched) must NOT satisfy the floor."""
    traces = list(reversed(_synthetic_dual_fault_trace()))
    with pytest.raises(AssertionError):
        _assert_orbit_response_scan_ran(SDKWorkflowResult(tool_traces=traces))


# A correct dual-fault conclusion (names both devices, both fault types) and
# a wrong/incomplete control conclusion (wrong device, wrong fault type,
# missing the second fault entirely) graded against the SAME
# _JUDGE_EXPECTATIONS contract the live test uses.
_POSITIVE_CONCLUSION = (
    "The orbit-response measurement resolves both root causes. BPM17's "
    "horizontal position readback has an inverted polarity -- its entire "
    "row of response slopes came back with the wrong sign relative to "
    "every corrector, consistent with a wiring/sign fault rather than "
    "noise or drift. Separately, HCM01 is desensitized to roughly half its "
    "nominal steering gain -- its entire column of response slopes is "
    "about half the expected magnitude, consistent with a degraded driver "
    "rather than a stuck or dead corrector. Recommend re-checking BPM17's "
    "cabling polarity and inspecting HCM01's corrector driver for the gain "
    "deficiency."
)
_CONTROL_CONCLUSION = (
    "The orbit-response measurement shows the correction residual isn't "
    "converging because BPM03's position readback is noisy -- its response "
    "slopes are erratic and inconsistent across repeated measurements. "
    "Recommend flagging BPM03 for a channel health check."
)


@pytest.mark.e2e
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_judge_discriminates_dual_fault_conclusions() -> None:
    """Dry-verify the SAME ``_JUDGE_EXPECTATIONS`` grading contract the live
    test uses, against hand-written positive/control conclusions -- no VA
    deploy, just the real LLMJudge over a synthetic WorkflowResult. Proves
    the judge actually discriminates a correct dual-fault conclusion from a
    wrong/incomplete one before spending a live Docker run on it.
    """
    judge = LLMJudge(provider=PROVIDER)

    positive = WorkflowResult(
        query=QUERY,
        response=_POSITIVE_CONCLUSION,
        execution_trace="(synthetic — see test_structural_floor_* for trace-shape coverage)",
        artifacts=[],
    )
    control = WorkflowResult(
        query=QUERY,
        response=_CONTROL_CONCLUSION,
        execution_trace="(synthetic — see test_structural_floor_* for trace-shape coverage)",
        artifacts=[],
    )

    pos_eval = await judge.evaluate(positive, expectations=_JUDGE_EXPECTATIONS)
    neg_eval = await judge.evaluate(control, expectations=_JUDGE_EXPECTATIONS)

    assert pos_eval.passed, f"judge rejected a correct dual-fault conclusion: {pos_eval.reasoning}"
    assert not neg_eval.passed, f"judge passed a wrong/incomplete conclusion: {neg_eval.reasoning}"
