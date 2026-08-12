"""Agentic scan-stack e2e: does the agent drive a scan end to end on a HEALTHY
stack, and does it read the result back honestly?

The subject here is the PROCEDURE, not a diagnosis. An operator asks for a
measurement; the agent must stage a draft of the right plan class, queue it,
start the queue, and read the run's data back — over a real deployed VA +
bridge + Tiled stack — and then say what the measurement shows. Nothing in
the stack is broken, so there is no hidden answer to find: a run that
"concludes" a fault is wrong, and a run that never actually took a
measurement has nothing to conclude from.

Grading is two parts, and the split is deliberate:

  (a) A DETERMINISTIC STRUCTURAL FLOOR over the tool trace — the first half
      of this module, and the only part that runs offline. It answers "was a
      scan of the right class actually staged, launched, and read back?" with
      no model in the loop.
  (b) ONE LLM-JUDGE CRITERION over the agent's prose, covering the part a
      trace cannot see: did the agent describe the procedure it ran and
      interpret the data it got back, without inventing findings the healthy
      stack cannot support. The judge is told the floor already covered
      methodology, so it never re-penalizes it.

Why the floor is shaped the way it is
-------------------------------------

**Accumulated draft state, not one call's arguments.** The draft is a shared,
incrementally editable staging surface: ``set_draft`` PATCHes it, and an agent
may legitimately fill ``correctors`` in one call and ``detectors`` in the next
(see ``osprey/mcp_server/bluesky/tools/draft.py``). Grading a single call's
``plan_args_patch`` would fail a correct two-call assembly. So the floor folds
every successful ``set_draft`` in trace order into an accumulated state,
honoring ``remove`` (which retracts keys) and resetting the state when
``plan_name`` changes — the bridge replaces ``plan_args`` wholesale on a plan
switch, so state carried across a switch was never real.

**Anchored on successful calls, and refusal-tolerant.** Every anchor —
``set_draft`` included — is checked for ``is_error``. A refused call changed
nothing on the bridge, so it must not contribute state and must not anchor the
chain. But a refusal is also not a failure of the agent: ``queue_add`` refuses
a stale draft revision *by contract*, and recovering from that refusal is
correct behavior. So a refused-then-successful chain passes. The consequence
that makes this sound: because the add that counts is a SUCCESSFUL one, the
plan the bridge actually launched IS the accumulated state at that add.

**Plan-class predicate, never a plan name.** The floor asks what the scan
does, not what it is called. Everything above is shared; only a small
predicate over the accumulated state distinguishes one measurement class from
another — correctors driven against BPM detectors for the orbit-response
class, two or more distinct setpoint axes against detectors for the grid-scan
class. An agent that picks a differently-named but structurally equivalent
plan still passes, and the two predicates are mutually exclusive, so neither
live test can be satisfied by the other's run.

**Both queue steps.** Execution is two calls: ``queue_add`` puts the pinned
draft in the queue and moves nothing; only ``queue_start`` drains it. A floor
satisfied by the add alone would pass a run in which no measurement was ever
taken, which is the one thing this module exists to detect. ``get_run_data``
closes it, and closes it on the RIGHT run: the add's own result names the run id
the bridge assigned, and the read is required to ask for that id. Both live
tests share one deploy, so an earlier test's run stays readable — without that
binding, a run that read the PREVIOUS measurement back would look identical to
one that took its own.

Both halves are dry-verified offline, against the SAME contracts the live
tests use. The floor runs against hand-built ``ToolTrace`` fixtures — no
Docker, no API key, no agent run::

    .venv/bin/pytest tests/e2e/test_scan_stack_agentic.py -k floor

The judge runs against hand-written conclusions — a correct one that must
pass, and one control per rubric criterion that must fail — proving the
rubric discriminates before a live Docker run is ever spent on it. Needs the
judge provider's credentials (``ALS_APG_API_KEY``), nothing else::

    .venv/bin/pytest tests/e2e/test_scan_stack_agentic.py -k judge

The live half — the deployed VA + bridge + Tiled stack the agent actually
drives — is the deploy scaffold at the end of this module. It is reached only
by a test that asks for :func:`deployed_scan_stack`, so both dry commands
above stay Docker-free.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from osprey import bluesky_tool_names
from osprey.agent_runner import SDKWorkflowResult, ToolTrace
from osprey.deployment.compose_generator import resolve_project_name
from osprey.services.bluesky_bridge.queue_backend import is_queue_active
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import dead_container_logs
from tests.e2e.judge import LLMJudge, WorkflowResult
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    _default_opus_model,
    is_claude_code_available,
    promote_ask_to_allow,
    run_sdk_query,
)
from tests.e2e.test_preset_agentic import _to_workflow_result

# Same provider for the agent under test and for the judge (als-apg — reachable
# from GitHub Actions runners). Explicit at every callsite: this project has no
# default provider, so an omitted one silently falls back to whatever the built
# preset happens to declare.
JUDGE_PROVIDER = "als-apg"

# Fully-qualified MCP tool names, resolved through the single source of truth
# for the Bluesky tool surface (``bluesky_tool_names.matcher`` builds the
# ``mcp__<server>__<tool>`` form the SDK reports in a tool trace). Imported
# rather than spelled as literals so a tool rename cannot silently detach this
# grading floor from the tools it grades.
SET_DRAFT = bluesky_tool_names.matcher(bluesky_tool_names.SET_DRAFT)
QUEUE_ADD = bluesky_tool_names.matcher(bluesky_tool_names.QUEUE_ADD)
QUEUE_START = bluesky_tool_names.matcher(bluesky_tool_names.QUEUE_START)
GET_RUN_DATA = bluesky_tool_names.matcher(bluesky_tool_names.GET_RUN_DATA)


# ---------------------------------------------------------------------------
# Ports + names for the live stack (the deploy scaffold at the end of this
# module). Every published port is deliberately distinct from BOTH the preset
# defaults (8090 bridge / 8091 tiled / 8095 panels / 5064 VA / 5432 postgres /
# 5080 openobserve) and every sibling e2e module's pinned port, so this module
# can run on a shared dev machine beside an already-deployed tutorial stack —
# and beside any sibling suite — without touching, or being blocked by,
# anything it does not own.
#
# The sibling pins this module steps around, per service:
#
#   bridge      18090 test_bluesky_deploy · 18099 test_va_substrate_equivalence
#               18101 test_tiled_roundtrip · 18102 _orm_stack's default
#               18103 test_bluesky_catalog_e2e · 18104 test_grid_scan_roundtrip
#               18105 test_bluesky_sandbox_escape_e2e
#               18106 test_bluesky_panels_deploy · 18108 test_bluesky_queue_e2e
#               (18107 is test_nextcloud_talk_bridge_e2e's Nextcloud)
#   panels      18095 test_bluesky_panels_deploy · 18096 test_bluesky_queue_e2e
#   tiled       18191 test_bluesky_queue_e2e · 18192 test_tiled_roundtrip
#   VA (CA)     15064 test_bluesky_queue_e2e · 15065 test_tiled_roundtrip
#               (5064 is _orm_stack's default, which the tutorial holds)
#   postgres    25432 test_bluesky_queue_e2e · 25433 test_tiled_roundtrip
#               25434 test_bluesky_panels_deploy
#   openobserve 25080 test_bluesky_queue_e2e · 25081 test_bluesky_deploy
#               25082 test_tiled_roundtrip · 25083 test_bluesky_panels_deploy
# ---------------------------------------------------------------------------
BRIDGE_PORT = 18109
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
PANELS_PORT = 18097
TILED_PORT = 18193
VA_CA_PORT = 15066
POSTGRES_PORT = 25435
OPENOBSERVE_PORT = 25084

#: Compose project this module deploys under. Container names and locally-built
#: image tags both follow ``<project>-<service>``, so every exact-named docker
#: operation below derives from this one constant rather than repeating a
#: host-global literal.
PROJECT_NAME = "scan-agentic"


# ---------------------------------------------------------------------------
# The structural floor: accumulated draft state, plan-class predicates, and
# the ordered walk over the tool trace. Plain module-level functions so the
# live tests and every plan class share exactly one implementation.
# ---------------------------------------------------------------------------


#: A plan-class predicate takes an accumulated ``plan_args`` state and answers
#: "is this a scan of my class?" — never looking at ``plan_name``.
PlanClassPredicate = Callable[[dict[str, Any]], bool]


def accumulated_draft_states(traces: list[ToolTrace]) -> list[dict[str, Any]]:
    """Fold successful ``set_draft`` calls into the draft state each call sees.

    Returns a list parallel to ``traces``: element ``i`` is the accumulated
    ``plan_args`` state as of just BEFORE ``traces[i]`` runs — i.e. what the
    bridge's draft held when that call was made.

    Mirrors the bridge's own draft semantics:

    - a ``plan_args_patch`` merges into the state key-by-key;
    - ``remove`` deletes keys (distinct from patching a key to ``None``,
      which is a legal value for an optional field);
    - naming a DIFFERENT ``plan_name`` replaces ``plan_args`` wholesale, so
      the state resets before that call's own patch is applied;
    - a call that came back ``is_error`` changed nothing and is skipped.
    """
    states: list[dict[str, Any]] = []
    plan_name: str | None = None
    args: dict[str, Any] = {}

    for trace in traces:
        states.append(dict(args))
        if trace.name != SET_DRAFT or trace.is_error:
            continue

        new_plan = trace.input.get("plan_name")
        if isinstance(new_plan, str) and new_plan and new_plan != plan_name:
            plan_name = new_plan
            args = {}

        patch = trace.input.get("plan_args_patch")
        if isinstance(patch, dict):
            args.update(patch)

        remove = trace.input.get("remove")
        if isinstance(remove, list):
            for key in remove:
                args.pop(key, None)

    return states


def is_orbit_response_state(state: dict[str, Any]) -> bool:
    """Orbit-response plan class: the state drives a set of correctors AND
    reads a set of BPM detectors together.

    The ``orm`` plan's own device-class contract, checked structurally so a
    differently-named but equivalent plan still qualifies. Never compares
    ``plan_name``.
    """
    correctors = state.get("correctors")
    detectors = state.get("detectors")
    return (
        isinstance(correctors, list)
        and bool(correctors)
        and isinstance(detectors, list)
        and bool(detectors)
    )


def is_grid_scan_state(state: dict[str, Any]) -> bool:
    """Grid-scan plan class: the state steps at least two DISTINCT setpoint
    devices over a rectangular grid and reads a set of detectors at each point.

    The ``grid_scan`` plan's device-class contract (see
    ``services/bluesky_bridge/plans_core/grid_scan.py``'s ``PARAMS`` /
    ``GridAxis``), with one addition the floor has to make itself: that plan's
    validator accepts a single axis, and accepts the same setpoint named on two
    axes. Neither is a two-dimensional grid — a repeated axis collapses the
    grid onto itself — so a trace carrying either would satisfy a naive floor
    while measuring nothing the scan was asked for. Never compares
    ``plan_name``.
    """
    detectors = state.get("detectors")
    if not (isinstance(detectors, list) and detectors):
        return False
    axes = state.get("axes")
    if not (isinstance(axes, list) and len(axes) >= 2):
        return False
    setpoints = [a.get("setpoint") for a in axes if isinstance(a, dict)]
    if not all(isinstance(s, str) and s for s in setpoints) or len(setpoints) != len(axes):
        return False
    return len(set(setpoints)) == len(setpoints)


def _first_successful_after(
    traces: list[ToolTrace],
    after: int,
    name: str,
    *,
    matches: Callable[[ToolTrace], bool] | None = None,
) -> int | None:
    """Index of the first successful ``name`` call beyond ``after``, or ``None``.

    Refused calls are skipped, never fatal — see the module docstring on
    refusal tolerance. ``matches`` narrows the search to calls that also
    satisfy a predicate on the call itself.
    """
    for i in range(after + 1, len(traces)):
        trace = traces[i]
        if trace.name != name or trace.is_error:
            continue
        if matches is not None and not matches(trace):
            continue
        return i
    return None


def _launched_run_id(add_trace: ToolTrace) -> str | None:
    """The run id the bridge assigned to the plan a ``queue_add`` pinned.

    ``queue_add`` answers with ``json.dumps({"run_id", "revision", "item"})``,
    and ``run_id`` is OSPREY's handle for the run that add will produce — the
    same value ``get_run_data`` takes as its required argument. Returns
    ``None`` when the result is not parseable JSON or carries no usable id,
    which is what makes the binding in :func:`find_satisfying_chain` degrade
    instead of hard-failing.
    """
    try:
        body = json.loads(add_trace.result or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    run_id = body.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _reads_run(run_id: str) -> Callable[[ToolTrace], bool]:
    """Predicate: a ``get_run_data`` call that reads ``run_id``.

    A factory rather than an inline lambda so the id is bound at call time,
    not looked up from the enclosing loop when the predicate finally runs.
    """
    return lambda trace: trace.input.get("run_id") == run_id


def find_satisfying_chain(
    traces: list[ToolTrace], predicate: PlanClassPredicate
) -> tuple[int, int, int] | None:
    """Locate a launch → start → read chain that satisfies ``predicate``.

    Returns the ``(queue_add, queue_start, get_run_data)`` trace indices of the
    first satisfying chain, or ``None`` if there is none. Satisfying means: a
    SUCCESSFUL ``queue_add`` whose accumulated draft state passes ``predicate``,
    followed later by a successful ``queue_start``, followed later by a
    successful ``get_run_data`` OF THE RUN THAT ADD LAUNCHED.

    That last binding is what keeps the read honest across tests. Both live
    tests share one module deploy, so from the second one onward an EARLIER
    test's run is still readable — and an agent that queued a scan, started
    it, and then read the previous run's data back would satisfy an unbound
    floor while reporting a measurement it never took. The add's own result
    carries the ``run_id`` the bridge assigned, and ``get_run_data`` takes
    that id as a required argument, so the two can simply be required to
    agree.

    The binding degrades rather than breaks: when the add's result carries no
    usable run id (:func:`_launched_run_id` returns ``None`` — an unparseable
    body, or a response shape that moved), any successful read anchors the
    chain, exactly as before. A bridge change can cost this floor the extra
    discrimination; it can never turn a correct run red.

    Greedy first-match on the two tail anchors is exhaustive, not a shortcut:
    the earliest successful ``queue_start`` after an add leaves the widest
    window for a subsequent read, so if any start/read pair exists for that
    add, the first-match pair does too. Only the add itself needs the full
    scan, since a later add can carry state an earlier one did not.
    """
    states = accumulated_draft_states(traces)
    for add_idx, trace in enumerate(traces):
        if trace.name != QUEUE_ADD or trace.is_error:
            continue
        if not predicate(states[add_idx]):
            continue
        start_idx = _first_successful_after(traces, add_idx, QUEUE_START)
        if start_idx is None:
            continue
        run_id = _launched_run_id(trace)
        read_idx = _first_successful_after(
            traces,
            start_idx,
            GET_RUN_DATA,
            matches=None if run_id is None else _reads_run(run_id),
        )
        if read_idx is None:
            continue
        return add_idx, start_idx, read_idx
    return None


def assert_scan_executed(
    result: SDKWorkflowResult, predicate: PlanClassPredicate, *, plan_class: str
) -> None:
    """Assert the deterministic floor: the agent staged, launched, and read
    back a scan of ``plan_class``. Runs unconditionally — never skip-gated.
    """
    traces = result.tool_traces
    if find_satisfying_chain(traces, predicate) is not None:
        return

    states = accumulated_draft_states(traces)
    draft_calls = [
        f"{'REFUSED' if t.is_error else 'ok'} {t.input}" for t in traces if t.name == SET_DRAFT
    ]
    add_states = [
        f"{'REFUSED' if t.is_error else 'ok'} launched={_launched_run_id(t)!r} state={states[i]}"
        for i, t in enumerate(traces)
        if t.name == QUEUE_ADD
    ]
    read_calls = [
        f"{'REFUSED' if t.is_error else 'ok'} run_id={t.input.get('run_id')!r}"
        for t in traces
        if t.name == GET_RUN_DATA
    ]
    raise AssertionError(
        f"no {plan_class} scan was staged, launched, and read back. The floor "
        "needs a SUCCESSFUL queue_add whose accumulated draft state satisfies "
        f"the {plan_class} contract, then a successful queue_start (an item "
        "sitting in the queue is not a measurement), then a successful "
        "get_run_data OF THE RUN THAT ADD LAUNCHED (reading an earlier run "
        "back is not this measurement — compare 'launched' below against the "
        "run_id each read asked for).\n"
        f"  set_draft calls: {draft_calls or '(none)'}\n"
        f"  queue_add calls, with the draft state each saw: {add_states or '(none)'}\n"
        f"  get_run_data calls, with the run each read: {read_calls or '(none)'}\n"
        # A silent bluesky MCP outage and free-choice agent drift produce the
        # same bare failure; the server status + full call list tell them apart.
        f"  MCP server status: {result.mcp_server_status}\n"
        f"  all tools called, in order: {[t.name for t in traces]}"
    )


def assert_orbit_response_scan_executed(result: SDKWorkflowResult) -> None:
    """The orbit-response-class floor — see :func:`assert_scan_executed`."""
    assert_scan_executed(result, is_orbit_response_state, plan_class="orbit-response-class")


def assert_grid_scan_executed(result: SDKWorkflowResult) -> None:
    """The grid-scan-class floor — see :func:`assert_scan_executed`."""
    assert_scan_executed(result, is_grid_scan_state, plan_class="grid-scan-class")


# ---------------------------------------------------------------------------
# The judge criterion: one rubric per plan class, covering the part a tool
# trace cannot see. Module-level constants because the live tests and the
# offline discrimination checks below must grade against the SAME text — a
# rubric that drifts between the dry run and the live run proves nothing.
# ---------------------------------------------------------------------------


def _judge_expectations(measurement: str, data_examples: str) -> str:
    """Build a plan class's rubric from the one shared contract.

    The three criteria, the hand-off from the structural floor, and the
    failure list are identical across plan classes; only the measurement being
    asked for differs. Written as one template rather than two near-copies so
    the classes cannot drift into grading different things.
    """
    return (
        "Procedure-and-interpretation judging only. A separate deterministic "
        "check has ALREADY verified the TOOL TRACE: that the agent staged a "
        f"draft of {measurement}, queued it, started the queue, and read the "
        "run's data back. Do not re-grade that sequence, do not require "
        "particular tool names, and do not penalize a tool call that was "
        "refused and then successfully retried — recovering from a refusal is "
        "correct behavior.\n"
        "\n"
        "That deterministic check covers the trace and nothing else. It says "
        "nothing about whether the agent's prose actually delivers the result "
        "to the operator, which is what you are grading.\n"
        "\n"
        "WHAT YOU ARE READING. The AGENT RESPONSE section is everything the "
        "agent said across all of its turns, concatenated in order — not a "
        "single closing message. An agent that really takes a measurement has "
        "to narrate its way there, so lines like 'the scan is queued' or "
        "'starting it now' appear MID-response in a perfectly healthy run and "
        "are not evidence that it stopped there. Grade the FINAL state the "
        "response leaves the operator in: narration of that kind followed by "
        "results and a reading of them is a PASS. Only a response whose LAST "
        "word to the operator is a plan or a queued item fails on it.\n"
        "\n"
        "Judge the response against all three criteria:\n"
        "\n"
        "1. IT ANSWERS THE REQUEST. The response tells the operator that the "
        "requested measurement was carried out on the machine and what came "
        "of it. A response that ENDS with the operator still believing "
        "nothing has run — 'the scan is queued', 'ready to launch when you "
        "are', 'here is what I would run' — fails this criterion no matter "
        "what the trace showed.\n"
        "2. IT INTERPRETS THE RUN IT READ BACK. The conclusion reports "
        f"concrete findings from THIS measurement ({data_examples}) rather "
        "than restating the scan's own parameters, or making generic "
        "statements that would be equally true of any run of this kind. "
        "Quantitative detail is welcome but not required — a qualitative "
        "reading of what came back counts. IMPORTANT: the execution trace you "
        "are shown is abbreviated and its tool results are truncated, so the "
        "underlying numbers are usually NOT visible to you. Do not require "
        "them to be, and never fail a conclusion merely because you cannot "
        "re-verify its figures against the trace — you are judging whether "
        "the agent reports findings, not auditing its arithmetic.\n"
        "3. IT ENDS CLEANLY. No unhandled error, no unresolved tool failure "
        "left dangling, and no statement the response contradicts elsewhere.\n"
        "\n"
        "PASS requires all three. FAIL if the response describes results from "
        "a measurement it never took or invents numbers, if it stops at "
        "planning or queueing, if it ignores what the operator actually "
        "asked, if it ends on an unhandled error, or if it commits to a "
        "specific hardware fault or diagnosis that nothing in its own "
        "reported findings supports. Length is not a criterion: a short, "
        "grounded reading of a clean measurement is a PASS, and so is a "
        "suitably hedged one."
    )


ORM_JUDGE_EXPECTATIONS = _judge_expectations(
    "an orbit-response-class scan — one that drives a set of correctors and "
    "reads a set of BPM detectors together",
    "for example the orbit shifts the BPMs reported as each corrector was "
    "stepped, or the response of the ring to the correctors that were driven",
)

GRID_JUDGE_EXPECTATIONS = _judge_expectations(
    "a grid-scan-class scan — one that steps two or more distinct setpoint "
    "devices over a rectangular grid, reading a set of detectors at every "
    "grid point",
    "for example how the detector readings varied across the grid, or what "
    "the scanned region looked like at the points that were measured",
)


# ---------------------------------------------------------------------------
# Offline dry tests for the floor. No Docker, no API key, no agent session —
# these grade the grader, against hand-built traces. Every test here carries
# `harness_benchmark`: an agent runs in none of them, so nothing they assert
# can measure model capability.
# ---------------------------------------------------------------------------


def _draft(
    *,
    plan_name: str | None = "orm",
    patch: dict[str, Any] | None = None,
    remove: list[str] | None = None,
    is_error: bool = False,
) -> ToolTrace:
    """A ``set_draft`` trace. Omitted arguments are omitted from ``input``,
    matching how the SDK reports a call that did not pass them."""
    payload: dict[str, Any] = {}
    if plan_name is not None:
        payload["plan_name"] = plan_name
    if patch is not None:
        payload["plan_args_patch"] = patch
    if remove is not None:
        payload["remove"] = remove
    return ToolTrace(
        name=SET_DRAFT,
        input=payload,
        result=(
            '{"code": "unknown_plan"}'
            if is_error
            else '{"revision": 1, "changed": ["correctors"], "plan_name": "orm"}'
        ),
        is_error=is_error,
    )


def _add(*, is_error: bool = False, revision: int = 1, run_id: str = "run-1") -> ToolTrace:
    """A ``queue_add`` trace. The success body carries ``run_id`` because the
    real tool's does, and the floor binds the later data read to it."""
    return ToolTrace(
        name=QUEUE_ADD,
        input={"draft_revision": revision},
        result=(
            '{"code": "stale_revision"}'
            if is_error
            else f'{{"run_id": "{run_id}", "revision": 1, "item": {{"item_uid": "item-1"}}}}'
        ),
        is_error=is_error,
    )


def _start(*, is_error: bool = False) -> ToolTrace:
    return ToolTrace(
        name=QUEUE_START,
        input={},
        result='{"code": "not_armed"}' if is_error else '{"started": true, "msg": ""}',
        is_error=is_error,
    )


# A healthy orbit-response readback: diagonal-dominant response slopes, every
# corrector steering its nearest BPM hardest, nothing sign-flipped or weak.
# Small enough to survive `_to_workflow_result`'s 300-char result preview, so
# the judge dry tests below see real numbers rather than an ellipsis.
_ORM_RUN_DATA = (
    '{"run_id":"run-1","points":27,"slopes_mm_per_A":'
    '{"corrector_01":{"bpm_01":1.82,"bpm_17":0.61,"bpm_23":0.55},'
    '"corrector_02":{"bpm_01":0.58,"bpm_17":1.75,"bpm_23":0.63},'
    '"corrector_03":{"bpm_01":0.54,"bpm_17":0.60,"bpm_23":1.79}}}'
)

# A healthy grid readback: both detectors vary smoothly and monotonically
# along both axes over the 5x5 grid.
_GRID_RUN_DATA = (
    '{"run_id":"run-1","shape":[5,5],'
    '"bpm_01_mm":[[-2.1,-1,0,1.1,2.2],[-1.6,-0.5,0.5,1.6,2.7],'
    "[-1.1,0,1,2.1,3.2],[-0.5,0.5,1.6,2.6,3.7],[0,1.1,2.1,3.2,4.2]],"
    '"bpm_02_mm":[[1.9,1.4,0.9,0.4,-0.1],[1.4,0.9,0.4,-0.1,-0.6],'
    "[0.9,0.4,-0.1,-0.6,-1.1],[0.4,-0.1,-0.6,-1.1,-1.6],"
    "[-0.1,-0.6,-1.1,-1.6,-2.1]]}"
)


def _read(*, is_error: bool = False, data: str = _ORM_RUN_DATA, run_id: str = "run-1") -> ToolTrace:
    """A ``get_run_data`` trace. ``run_id`` is the run being READ — it has to
    match the one the anchoring ``queue_add`` reported for the floor to accept
    the chain."""
    return ToolTrace(
        name=GET_RUN_DATA,
        input={"run_id": run_id},
        result='{"code": "no_such_run"}' if is_error else data,
        is_error=is_error,
    )


_ORM_ARGS: dict[str, Any] = {
    "correctors": ["corrector_01", "corrector_02", "corrector_03"],
    "detectors": ["bpm_01", "bpm_17", "bpm_23"],
    "span_a": 1.0,
    "num": 9,
}


def _orm_run_trace() -> list[ToolTrace]:
    """The canonical healthy shape: one complete orbit-response draft, the add
    that pins it, the start that drains the queue, then the data read."""
    return [_draft(patch=_ORM_ARGS), _add(), _start(), _read()]


_GRID_ARGS: dict[str, Any] = {
    "detectors": ["bpm_01", "bpm_02"],
    "axes": [
        {"setpoint": "corrector_01", "start": -1.0, "stop": 1.0, "num_points": 5},
        {"setpoint": "corrector_02", "start": -1.0, "stop": 1.0, "num_points": 5},
    ],
}


def _grid_run_trace() -> list[ToolTrace]:
    """The canonical healthy grid-scan shape: one complete two-axis draft, the
    add that pins it, the start that drains the queue, then the data read."""
    return [
        _draft(plan_name="grid_scan", patch=_GRID_ARGS),
        _add(),
        _start(),
        _read(data=_GRID_RUN_DATA),
    ]


def _floor_passes(traces: list[ToolTrace]) -> bool:
    return find_satisfying_chain(traces, is_orbit_response_state) is not None


def _grid_floor_passes(traces: list[ToolTrace]) -> bool:
    return find_satisfying_chain(traces, is_grid_scan_state) is not None


@pytest.mark.harness_benchmark
def test_floor_accepts_orbit_response_class_run() -> None:
    """The healthy shape passes, and the assertion helper agrees with the
    chain walk it wraps."""
    assert _floor_passes(_orm_run_trace())
    assert_orbit_response_scan_executed(SDKWorkflowResult(tool_traces=_orm_run_trace()))


@pytest.mark.harness_benchmark
def test_floor_rejects_a_scan_of_another_plan_class() -> None:
    """A draft carrying no correctors/detectors pair — e.g. a generic n-d grid
    scan over unrelated axes — must not satisfy the orbit-response floor.
    Non-vacuity for the predicate: this is not "any scan ran"."""
    traces = [_draft(plan_name="grid_scan", patch={"axes": ["some_motor"], "num": [5]})]
    traces += [_add(), _start(), _read()]
    assert not _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_rejects_out_of_order_calls() -> None:
    """All four calls present, all successful, wrong order (the data read
    happens before anything was ever queued). The walk is ordered, so this
    must fail."""
    assert not _floor_passes(list(reversed(_orm_run_trace())))


@pytest.mark.harness_benchmark
@pytest.mark.parametrize("missing", [QUEUE_ADD, QUEUE_START])
def test_floor_requires_both_queue_steps(missing: str) -> None:
    """Non-vacuity for the two-step execution contract: dropping EITHER queue
    call must fail the floor.

    ``queue_start`` is the one that matters most. An agent that composes a
    scan and queues it has moved nothing, so a floor satisfied by the add
    alone would pass a run in which no measurement was taken. Reversing the
    whole trace (above) does not prove this — it perturbs every call at once.
    """
    assert not _floor_passes([t for t in _orm_run_trace() if t.name != missing])


@pytest.mark.harness_benchmark
def test_floor_accepts_draft_assembled_across_two_calls() -> None:
    """Correctors in one ``set_draft``, detectors in the next. The bridge draft
    is incremental, so the state the add sees is the fold of both — grading a
    single call's ``plan_args_patch`` would wrongly fail this correct run."""
    traces = [
        _draft(patch={"correctors": ["corrector_01"], "span_a": 1.0}),
        _draft(plan_name=None, patch={"detectors": ["bpm_01"], "num": 9}),
        _add(),
        _start(),
        _read(),
    ]
    assert _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_rejects_state_retracted_by_remove() -> None:
    """``remove`` deletes keys from the draft. An agent that fills a complete
    orbit-response draft and then retracts ``detectors`` before queueing
    launched a draft that no longer reads any BPM — the fold must reflect
    that, not the high-water mark."""
    traces = [
        _draft(patch=_ORM_ARGS),
        _draft(plan_name=None, remove=["detectors"]),
        _add(),
        _start(),
        _read(),
    ]
    assert not _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_rejects_state_carried_across_a_plan_name_change() -> None:
    """Naming a different plan replaces ``plan_args`` wholesale on the bridge.
    State assembled under the old plan was never real for the new one, so a
    satisfied draft followed by a plan switch and then a launch must fail —
    what got queued was the (empty) new draft."""
    traces = [
        _draft(patch=_ORM_ARGS),
        _draft(plan_name="grid_scan"),
        _add(),
        _start(),
        _read(),
    ]
    assert not _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_ignores_draft_state_from_a_refused_set_draft() -> None:
    """A refused ``set_draft`` changed nothing on the bridge, so its patch must
    not contribute state — otherwise a rejected draft could satisfy the floor
    for a launch that carried none of it."""
    traces = [_draft(patch=_ORM_ARGS, is_error=True), _add(), _start(), _read()]
    assert not _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_rejects_a_chain_of_refused_calls() -> None:
    """Right names, right order, every execution call refused. Nothing ran, so
    the floor must not be satisfied — this is the control that keeps refusal
    tolerance from degrading into "the agent reached for the tools"."""
    traces = [_draft(patch=_ORM_ARGS), _add(is_error=True), _start(is_error=True), _read()]
    assert not _floor_passes(traces)


@pytest.mark.harness_benchmark
def test_floor_binds_the_data_read_to_the_run_the_add_launched() -> None:
    """The read that closes the chain must be OF the run the add launched.

    Both live tests share one module deploy, so from the second test onward an
    earlier run is still readable on the bridge. Under an unbound floor, an
    agent that staged a scan, queued it, started it, and then read the PREVIOUS
    run's data would pass — reporting a measurement it never took, which is the
    one failure this module exists to catch.

    Third case is the graceful degrade: when the add's result carries no usable
    run id, the binding is dropped rather than failing the run, so a change in
    the bridge's response shape costs this floor discrimination and never turns
    a correct run red.
    """
    launched = [_draft(patch=_ORM_ARGS), _add(run_id="run-7"), _start(), _read(run_id="run-7")]
    assert _floor_passes(launched)

    read_an_older_run = [
        _draft(patch=_ORM_ARGS),
        _add(run_id="run-7"),
        _start(),
        _read(run_id="run-1"),
    ]
    assert not _floor_passes(read_an_older_run)

    unparseable_add_result = [
        _draft(patch=_ORM_ARGS),
        ToolTrace(name=QUEUE_ADD, input={"draft_revision": 1}, result="queued at revision 1"),
        _start(),
        _read(run_id="run-1"),
    ]
    assert _floor_passes(unparseable_add_result)


@pytest.mark.harness_benchmark
def test_floor_accepts_a_refused_add_followed_by_a_successful_one() -> None:
    """``queue_add`` refuses a stale draft revision by contract; noticing and
    re-queueing the current revision is correct agent behavior, not a failure.
    A refusal never anchors the chain, and never poisons it either."""
    traces = [
        _draft(patch=_ORM_ARGS),
        _add(is_error=True, revision=0),
        _draft(plan_name=None, patch={"num": 11}),
        _add(revision=2),
        _start(),
        _read(),
    ]
    assert _floor_passes(traces)


# --- grid-scan class -------------------------------------------------------
# Same accumulator and same ordered walk as above; only the plan-class
# predicate changes. These tests cover what is specific to that predicate.


@pytest.mark.harness_benchmark
def test_grid_floor_accepts_two_axis_grid_run() -> None:
    """The healthy two-axis shape passes, and the assertion helper agrees with
    the chain walk it wraps."""
    assert _grid_floor_passes(_grid_run_trace())
    assert_grid_scan_executed(SDKWorkflowResult(tool_traces=_grid_run_trace()))


@pytest.mark.harness_benchmark
def test_grid_floor_rejects_a_single_axis_scan() -> None:
    """One axis is a line scan, not a grid. The plan's own validator accepts it
    (``axes`` only has ``min_length=1``), so the floor is the only thing
    standing between a 1-D sweep and a passing 2-D grid run."""
    traces = [
        _draft(
            plan_name="grid_scan",
            patch={"detectors": ["bpm_01"], "axes": [_GRID_ARGS["axes"][0]]},
        ),
        _add(),
        _start(),
        _read(),
    ]
    assert not _grid_floor_passes(traces)


@pytest.mark.harness_benchmark
def test_grid_floor_rejects_two_axes_naming_the_same_setpoint() -> None:
    """Two axes over the SAME setpoint device collapse the grid onto itself —
    the second axis fights the first, and the scan measures a line at best. The
    plan's validator only checks setpoints against detectors, never against
    each other, so this reject lives here."""
    axis = _GRID_ARGS["axes"][0]
    traces = [
        _draft(
            plan_name="grid_scan",
            patch={"detectors": ["bpm_01"], "axes": [axis, dict(axis, num_points=7)]},
        ),
        _add(),
        _start(),
        _read(),
    ]
    assert not _grid_floor_passes(traces)


@pytest.mark.harness_benchmark
def test_grid_floor_accepts_a_grid_assembled_across_two_calls() -> None:
    """Detectors in one ``set_draft``, the axes in the next. Shares the
    accumulator with the orbit-response class, so an incremental grid build is
    graded on the state the add actually saw."""
    traces = [
        _draft(plan_name="grid_scan", patch={"detectors": _GRID_ARGS["detectors"]}),
        _draft(plan_name=None, patch={"axes": _GRID_ARGS["axes"]}),
        _add(),
        _start(),
        _read(),
    ]
    assert _grid_floor_passes(traces)


@pytest.mark.harness_benchmark
def test_grid_and_orbit_response_floors_do_not_accept_each_other() -> None:
    """The two plan classes must be mutually exclusive on these traces, or the
    two live tests could pass on each other's runs — an agent that ran a grid
    scan when asked for an orbit response (or the reverse) would still be
    graded as having done the right measurement."""
    assert not _grid_floor_passes(_orm_run_trace())
    assert not _floor_passes(_grid_run_trace())


# ---------------------------------------------------------------------------
# Offline discrimination checks for the judge criterion. These grade the
# GRADER: the same rubric constants the live tests use, against hand-written
# conclusions, over a synthetic healthy-run trace. No Docker and no agent
# session — only the judge provider's credentials.
#
# Every conclusion below (positive and control alike) is paired with the SAME
# successful trace, so the only thing that varies is the prose. A control that
# fails therefore proves the rubric grades what the agent SAID, not what the
# floor already checked.
#
# The operator requests here exist only to frame these conclusions; the live
# tests carry their own prompts. The rubric is the shared surface, and it is
# shared verbatim.
# ---------------------------------------------------------------------------

_DRY_ORM_REQUEST = (
    "Can you measure the orbit response of the ring for me and tell me how it "
    "looks? I want to know whether the steering is behaving the way we expect."
)

_DRY_GRID_REQUEST = (
    "Please scan the two steering magnets against each other over their usual "
    "range and tell me what the detectors see across that region."
)

_ORM_POSITIVE_CONCLUSION = (
    "I measured the orbit response and it looks healthy. Each of the three "
    "correctors was stepped through nine currents from -1 to +1 A while all "
    "three BPMs were read, and I pulled the finished run back. Every "
    "corrector steers its nearest BPM hardest — about 1.8 mm/A on the "
    "diagonal — with smaller cross-terms of 0.5 to 0.6 mm/A at the other two, "
    "and every slope came back positive. No row or column of the response is "
    "sign-flipped, and none is noticeably weaker than its neighbours. Within "
    "the resolution of this sweep the steering is behaving the way we expect, "
    "so I would not send anyone out on the strength of this measurement."
)

_GRID_POSITIVE_CONCLUSION = (
    "I ran the two-magnet grid and read the results back. The scan stepped "
    "corrector_01 and corrector_02 from -1 to +1 A in five points each, "
    "reading both BPMs at all 25 grid points. Over that square, BPM 1 climbs "
    "smoothly from about -2.1 mm in one corner to +4.2 mm in the opposite "
    "one, in roughly even steps along both axes, while BPM 2 falls the other "
    "way across the same region, from about +1.9 mm to -2.1 mm. Both surfaces "
    "are close to planar with no plateaus, jumps, or dead corners, which is "
    "what two independent, linear steering magnets should look like."
)

# Each control breaks exactly ONE of the rubric's three criteria, so a control
# that the judge passes points at a specific hole in the rubric.
_ORM_CONTROLS: dict[str, str] = {
    "planning_only": (
        "I have put together an orbit-response scan for you: three correctors "
        "stepped over nine currents from -1 to +1 A, reading all three BPMs at "
        "each point. The draft is staged and the item is sitting in the queue. "
        "Say the word and I will start it, and once it finishes I can pull the "
        "data and walk you through what the steering is doing."
    ),
    "ungrounded_diagnosis": (
        "Your steering problem is BPM 17 — its horizontal readback polarity is "
        "inverted, which is why the orbit correction has been fighting itself. "
        "I would also have HCM01 looked at; correctors of that vintage lose "
        "roughly half their gain as the driver ages. Get a tech on the BPM 17 "
        "cabling first and the rest should settle down."
    ),
    "unhandled_error": (
        "I tried to pull the orbit-response data but the read came back with "
        "an error I could not get past:\n"
        "  KeyError: 'primary'\n"
        "I am not sure whether the scan wrote anything usable. You may want to "
        "check the data server before we try this again."
    ),
}

_GRID_CONTROLS: dict[str, str] = {
    "planning_only": (
        "I have staged the two-magnet grid: corrector_01 and corrector_02, "
        "five points each from -1 to +1 A, reading both BPMs at every grid "
        "point. It is queued and ready whenever you want it launched — once "
        "it runs I can pull the data and describe the region for you."
    ),
    "ungrounded_diagnosis": (
        "The grid confirms what I suspected: corrector_02 is saturating at the "
        "top of its range and dragging the whole upper half of the scan with "
        "it. That magnet's power supply is on its way out and should be "
        "swapped at the next opportunity."
    ),
    "unhandled_error": (
        "The grid scan ran but reading it back failed:\n"
        "  TimeoutError: no response from the data server after 30s\n"
        "I could not retrieve any of the detector readings, so I have nothing "
        "to tell you about the scanned region yet."
    ),
}


def _judged(request: str, conclusion: str, traces: list[ToolTrace]) -> WorkflowResult:
    """Package a hand-written conclusion the way the live tests package a real
    one — through ``_to_workflow_result``, so the judge sees the same shape."""
    return _to_workflow_result(
        request, SDKWorkflowResult(tool_traces=traces, text_blocks=[conclusion])
    )


@pytest.mark.harness_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_judge_accepts_a_grounded_orbit_response_conclusion() -> None:
    """The rubric must pass a conclusion that reports the measurement it ran
    and reads the data it got back."""
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _judged(_DRY_ORM_REQUEST, _ORM_POSITIVE_CONCLUSION, _orm_run_trace()),
        expectations=ORM_JUDGE_EXPECTATIONS,
    )
    assert eval.passed, f"judge rejected a grounded orbit-response conclusion: {eval.reasoning}"


@pytest.mark.harness_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
@pytest.mark.parametrize("control", list(_ORM_CONTROLS), ids=list(_ORM_CONTROLS))
async def test_judge_rejects_a_failing_orbit_response_conclusion(control: str) -> None:
    """Each control breaks one criterion and must fail — otherwise the rubric
    would pass a run that planned but never reported, invented a diagnosis, or
    ended on an unresolved error."""
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _judged(_DRY_ORM_REQUEST, _ORM_CONTROLS[control], _orm_run_trace()),
        expectations=ORM_JUDGE_EXPECTATIONS,
    )
    assert not eval.passed, f"judge passed the '{control}' control: {eval.reasoning}"


@pytest.mark.harness_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_judge_accepts_a_grounded_grid_scan_conclusion() -> None:
    """Same contract, grid class."""
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _judged(_DRY_GRID_REQUEST, _GRID_POSITIVE_CONCLUSION, _grid_run_trace()),
        expectations=GRID_JUDGE_EXPECTATIONS,
    )
    assert eval.passed, f"judge rejected a grounded grid-scan conclusion: {eval.reasoning}"


@pytest.mark.harness_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
@pytest.mark.parametrize("control", list(_GRID_CONTROLS), ids=list(_GRID_CONTROLS))
async def test_judge_rejects_a_failing_grid_scan_conclusion(control: str) -> None:
    """Same controls, grid class."""
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _judged(_DRY_GRID_REQUEST, _GRID_CONTROLS[control], _grid_run_trace()),
        expectations=GRID_JUDGE_EXPECTATIONS,
    )
    assert not eval.passed, f"judge passed the '{control}' control: {eval.reasoning}"


# ===========================================================================
# The live stack. Everything above runs offline; everything below deploys real
# containers, and is reached ONLY by a test that asks for
# ``deployed_scan_stack`` — so the dry commands in the module docstring stay
# Docker-free even though the fixtures live in the same file.
#
# CONTAINER SAFETY: every docker invocation below names an EXACT image
# (``<project>-bluesky-bridge:local`` / ``-va:local`` / ``-bluesky-panels:local``,
# all derived from ``PROJECT_NAME``) — never a wildcard, never a prune, never a
# volume operation. Teardown goes through ``osprey down``.
# ===========================================================================

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
# Cold-cache `osprey up` budget. This stack is the FULL queue shape — VA +
# bridge + queueserver + Redis + Tiled + the panels sidecar — so it is sized
# against test_bluesky_queue_e2e.py's 2400 s (same service set), not against
# test_orm_roundtrip.py's 1200 s (no queueserver/panels images to build).
# Measured, not guessed: at 1200 s a cold run on an Apple Silicon host timed
# out with the VA and bridge images built and the rest still going. The VA
# image is linux/amd64 ONLY (its Dockerfile refuses aarch64 outright, since no
# pcaspy wheel exists for it), so on an arm64 host the whole thing compiles
# PyAT/pcaspy from source under emulation — the slowest path this fixture has
# to survive. E2E_REUSE_IMAGES=1 skips the rebuild entirely and lands in
# well under a minute.
DEPLOY_UP_TIMEOUT_SEC = 2400
HEALTH_TIMEOUT_SEC = 300.0
DOWN_TIMEOUT_SEC = 300
# How long the queue-hygiene fixture waits for the manager to leave an active
# state after an abort. Matches the post-abort settle budget
# test_bluesky_queue_e2e.py measured against this same stack: the Run Engine
# pauses, unwinds the plan, and only then reports idle.
ABORT_SETTLE_TIMEOUT_SEC = 240.0

BRIDGE_IMAGE = _orm_stack.bridge_image(PROJECT_NAME)
VA_IMAGE = _orm_stack.va_image(PROJECT_NAME)
PANELS_IMAGE = _orm_stack.panels_image(PROJECT_NAME)

#: Everything this module deep-merges into ``_orm_stack.override_yaml()``:
#: host-port moves ``init_args`` has no ``--set`` hook for, plus the approval
#: policy the headless agent needs. Note the two ALTITUDES, which is the
#: whole reason this is one dict and not sibling entries:
#:
#: - ``services.postgresql.port_host`` / ``services.openobserve.port`` are
#:   CONFIG keys — those services are in the preset's config template, so a
#:   dotted key under ``config:`` edits them in place.
#: - ``bluesky.tiled_port`` / ``bluesky_panels.port`` are BUILD-PROFILE keys
#:   (see ``profiles/presets/control-assistant.yml``). Their ``services.*``
#:   entries are synthesized by the build injectors AFTER the config overlay
#:   runs, so a dotted ``config:`` key for either is silently discarded — no
#:   error, no stray key, just the default port. They must be set at profile
#:   altitude, as top-level blocks, which is where the override file already
#:   speaks.
#:
#: The ``bluesky`` block here carries ONLY ``tiled_port``: ``bluesky.port`` and
#: ``bluesky.tiled_enabled`` come from ``init_args``' own ``--set`` flags, so
#: dropping those flags there would silently take the Tiled sidecar out of this
#: stack rather than fail loudly.
#:
#: The two ``approval.tools`` keys arm the approval hook for exactly the two
#: queue tools ``_REQUIRED_TOOLS`` already promotes, and only in this rendered
#: test project. They are a SECOND gate, independent of the ``settings.json``
#: promotion below: ``promote_ask_to_allow`` moves a tool between
#: ``permissions.ask`` and ``permissions.allow``, while
#: ``.claude/hooks/osprey_approval.py`` reads its per-tool policy from
#: ``config.yml`` and defaults unlisted tools to ``always`` — so without these
#: the hook returns ``permissionDecision: "ask"`` on every ``queue_add``, and a
#: headless session with no responder takes that as a hard denial. Set here
#: rather than in ``_orm_stack.override_yaml()`` because that helper is shared
#: with both round-trip e2es and the Docker-free render gate, which must keep
#: the shipped default policy.
#:
#: ``execute`` is deliberately NOT armed even though ``_REQUIRED_TOOLS``
#: promotes it. Its shipped ``selective`` policy already allows a readonly
#: execute — the analysis path — while asking on write mode or write patterns.
#: Setting it to ``skip`` would hand the agent a ``caput``-shaped way to
#: hand-step the measurement the floor grades, which is the same hole that
#: keeps ``mcp__controls__channel_write`` off the promoted list.
_EXTRA_CONFIG: dict[str, Any] = {
    "config": {
        "services.postgresql.port_host": POSTGRES_PORT,
        "services.openobserve.port": OPENOBSERVE_PORT,
        "approval.tools.queue_add": "skip",
        "approval.tools.queue_start": "skip",
    },
    "bluesky": {"tiled_port": TILED_PORT},
    "bluesky_panels": {"port": PANELS_PORT},
}

#: Tools promoted from ``permissions.ask`` to ``permissions.allow`` on the
#: deployed project. Every one of these sits in the rendered ``ask`` list, which
#: a headless session has no responder for — so each would come back to the
#: agent as a hard denial that ``bypassPermissions`` does not override.
#:
#: Deliberately NOT a blanket promotion. ``mcp__controls__channel_write`` stays
#: gated: with it the agent has a hand-stepped substitute for the very
#: measurement this module grades, and the floor would be satisfiable without a
#: scan ever running. The two queue steps are both required (an add without a
#: start moves nothing), and the python executor is the sanctioned compute path
#: — framework agents never get Bash, so without it there is no way to work
#: over a response matrix at all.
_REQUIRED_TOOLS = (
    bluesky_tool_names.matcher(bluesky_tool_names.QUEUE_ADD),
    bluesky_tool_names.matcher(bluesky_tool_names.QUEUE_START),
    "mcp__python__execute",
)


@dataclass
class DeployedScanStack:
    """Everything a live test needs about the one deployed project.

    ``correctors``/``bpms`` are the device names wired into the bridge worker
    (``write_scan_env``), so a test that composes a plan names exactly the
    devices the deployed worker registered.
    """

    repo: Path
    correctors: dict[str, tuple[str, str]]
    bpms: dict[str, str]
    limits: dict[str, Any]
    token: str


def _dead_container_logs() -> str:
    """Logs from every container of this deployment that is not running."""
    return dead_container_logs(resolve_project_name({"project_name": PROJECT_NAME}))


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """One subprocess call with the environment ``osprey`` expects from a test.

    ``CLAUDECODE=''`` is what tells the CLI it is not running inside a Claude
    Code session; left set, the build/deploy commands take their interactive
    path.
    """
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


@pytest.fixture(scope="module")
def deployed_scan_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedScanStack]:
    """Build and ``osprey up --dev`` the scan stack; tear it down after.

    One stack for the whole module: nothing here is per-test state (the queue
    is, and that is what ``clean_queue`` below handles), and the VA image build
    is minutes on a cold cache.

    The provider is pinned explicitly rather than left to the preset's own
    default — this project has no default provider, and an omitted one would
    silently run the agent under whatever the control-assistant preset happens
    to declare. The MODEL is deliberately not pinned at build time: each live
    test chooses its own at ``run_sdk_query``, and the profile's tier map (what
    ``sdk_helpers._default_opus_model`` reads) is written regardless.
    """
    base = tmp_path_factory.mktemp("scan_stack_agentic_build")
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME,
        output_dir=base,
        bridge_port=BRIDGE_PORT,
        va_port=VA_CA_PORT,
        timeout=BUILD_TIMEOUT_SEC,
        provider=JUDGE_PROVIDER,
        extra_config=_EXTRA_CONFIG,
    )

    # Correctors and BPMs come from the BUILT project's own channel_limits.json
    # — never a hardcoded preset channel. The default 4+4 slice is deliberate:
    # these scenarios ask for a measurement on a healthy stack, so no particular
    # device has to be in range, and a small device count keeps a real scan to
    # seconds rather than minutes.
    # The render's copy, not the operator-owned source under <repo>/data/ —
    # build/data is the file the deployed containers actually read.
    limits = _orm_stack.channel_limits(repo / "build")
    correctors = _orm_stack.select_correctors(limits)
    bpms = _orm_stack.select_bpms(limits)
    _orm_stack.write_scan_env(repo, correctors=correctors, bpms=bpms)

    _orm_stack.force_image_rebuild(BRIDGE_IMAGE, VA_IMAGE, PANELS_IMAGE)

    osprey_bin = _orm_stack.find_osprey_console_script()
    try:
        try:
            up = _run(
                [str(osprey_bin), "up", "-d", "--dev"],
                cwd=repo,
                timeout=DEPLOY_UP_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as exc:
            # The likeliest failure on a cold cache (see DEPLOY_UP_TIMEOUT_SEC),
            # and the one that used to report nothing but the deadline. What was
            # still building, and which service never came up, lives in the
            # captured output and the container logs -- so say it here, while the
            # containers still exist. `TimeoutExpired` carries whatever had been
            # written before the deadline, as bytes when the child was not opened
            # in text mode.
            stdout = (
                exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
            )
            stderr = (
                exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            )
            pytest.fail(
                f"osprey up -d --dev timed out after {DEPLOY_UP_TIMEOUT_SEC}s "
                "(a cold-cache image build that did not finish in budget, or a service "
                "whose health chain never settled):\n"
                f"--- stdout so far ---\n{stdout}\n--- stderr so far ---\n{stderr}\n"
                f"--- containers that are not running ---\n{_dead_container_logs()}"
            )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}\n"
                f"--- containers that are not running ---\n{_dead_container_logs()}"
            )
        try:
            _orm_stack.wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n--- containers that are not running ---\n{_dead_container_logs()}")

        # AFTER `osprey up`, never before: the deploy path can re-render the Claude
        # Code artifacts and would discard an earlier edit to settings.json.
        promote_ask_to_allow(repo, *_REQUIRED_TOOLS)

        yield DeployedScanStack(
            repo=repo,
            correctors=correctors,
            bpms=bpms,
            limits=limits,
            token=_orm_stack.minted_launch_token(repo),
        )
    finally:
        # Best-effort by construction: this runs on the failure path too, where
        # an exception is already propagating. A teardown that raised here would
        # REPLACE that exception -- the fixture would report "osprey down timed
        # out" for a run that actually failed on a health timeout, hiding the
        # real cause. So every teardown failure is reported and swallowed.
        try:
            down = _run([str(osprey_bin), "down"], cwd=repo, timeout=DOWN_TIMEOUT_SEC)
            if down.returncode != 0:
                print(  # noqa: T201 - surface teardown issues in CI logs
                    f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
                )
        except (OSError, subprocess.SubprocessError) as exc:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down could not complete ({type(exc).__name__}: {exc}) -- "
                f"containers of project {PROJECT_NAME!r} may still be running"
            )


def _queue_snapshot() -> dict[str, Any]:
    status, body = _queue_drive.request(BRIDGE_URL, "/queue", "GET")
    assert status == 200, f"GET /queue failed: {status} {body}"
    assert isinstance(body, dict), f"GET /queue returned a non-object body: {body!r}"
    return body


def _wait_for_settled_manager(timeout: float) -> dict[str, Any]:
    """Poll ``GET /queue`` until no plan is in motion; return the last snapshot.

    "Settled" is the manager reporting a non-active state AND no running item.
    The active-state set is imported from ``queue_backend`` rather than spelled
    as literals here, so a state added there cannot leave this wait thinking an
    unwinding manager is idle.
    """
    deadline = time.monotonic() + timeout
    snapshot: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = _queue_snapshot()
        if not is_queue_active(snapshot["status"]) and not snapshot.get("running_item"):
            return snapshot
        time.sleep(1.0)
    raise AssertionError(
        f"the queue manager was still active {timeout:.0f}s after an abort "
        f"(state={snapshot.get('status', {}).get('manager_state')!r}, "
        f"running_item={snapshot.get('running_item')!r}) — a scan from an earlier "
        "test is still on the hardware, so this one cannot start from a known state"
    )


@pytest.fixture(autouse=True)
def clean_queue(request: pytest.FixtureRequest) -> None:
    """Leave the manager idle and its queue EMPTY before every live test.

    Autouse, but inert for the offline tests: it acts only when the test being
    set up actually asks for ``deployed_scan_stack`` (directly or through
    another fixture), so nothing here can drag a Docker deploy into the dry
    runs. ``getfixturevalue`` is what deploys the stack on the first live test.

    Why per-test rather than once per module. The queue is Redis-backed and
    that Redis lives in a compose NAMED VOLUME keyed on the project name, which
    ``osprey down`` does NOT remove — so a rerun inherits whatever the
    previous attempt left queued. And these tests are agentic: a rerun (each
    live test carries ``flaky``) follows an attempt that may have left a scan
    mid-flight. An agent that then queues its own work and arms the queue would
    put the PREVIOUS attempt's plan on the hardware too, and read back a run it
    never launched — a floor satisfied by someone else's scan.

    The three steps, and why each is needed:

    1. ``POST /queue/abort`` — the only surface that stops a plan already
       moving hardware (``POST /queue/stop`` merely halts the queue AFTER the
       running item finishes). Ungated by design, so no token is sent. A 409
       ``nothing_running`` is the normal, healthy answer on a quiet stack.
    2. Wait for the manager to settle. The abort returns while the Run Engine
       is still unwinding, and a delete against a moving queue is not the clean
       slate this fixture promises.
    3. ``DELETE /draft`` — clear the shared plan draft. The draft is bridge
       state, not queue state, so nothing above touches it, and it outlives
       both a finished test and a whole ``osprey down``/``up`` of nothing.
       Left standing it breaks the floor's central assumption: the accumulator
       replays the draft from EMPTY, so a run whose agent patched only the
       keys the leftover draft was missing would be graded against a state
       poorer than the one the bridge actually launched. The error is
       one-directional — the floor sees less than really happened, so this
       shows up as a false FAIL on the rerun path, never as a false pass.
    4. Delete every remaining item. This is where the requeued copy of an
       aborted plan is caught: upstream does not discard an interrupted plan,
       it pushes a copy back to the FRONT of the queue under a new ``item_uid``
       (and ``POST /queue/start`` then refuses outright with
       ``interrupted_item_in_queue``). Left behind, it would refuse the next
       test's arming step rather than merely dirty the queue. Removal goes
       through the bridge's own ``DELETE /queue/items/{uid}`` — never into
       Redis, which is a different path from the one an operator has.
    """
    if "deployed_scan_stack" not in request.fixturenames:
        return
    request.getfixturevalue("deployed_scan_stack")

    status, body = _queue_drive.request(BRIDGE_URL, "/queue/abort", "POST", timeout=180.0)
    assert status in (200, 409), (
        f"POST /queue/abort answered {status}: {body}. Expected 200 (a plan was "
        "aborted) or 409 nothing_running (the healthy quiet case) — anything else "
        "means the manager could not be brought to a stop, and this test would "
        "run against a stack that is still moving"
    )

    snapshot = _wait_for_settled_manager(ABORT_SETTLE_TIMEOUT_SEC)

    # The documented answer is 200 on both paths — the route is idempotent and
    # reports `cleared: false` when there was no draft to clear. 404/409 are
    # tolerated for the same reason the abort tolerates 409: they would still
    # mean "nothing to clear", and this fixture's job is to reach a clean
    # slate, not to police the shape of a bridge that says it is already there.
    status, body = _queue_drive.request(BRIDGE_URL, "/draft", "DELETE")
    assert status in (200, 404, 409), (
        f"DELETE /draft answered {status}: {body}. A draft left standing is "
        "state the next test's agent did not create, and the floor grades the "
        "draft it replays from empty"
    )

    # Re-snapshot and re-delete rather than deleting one list once: the abort's
    # requeue lands asynchronously, so an item can appear between the snapshot
    # and the delete, and a uid can go stale under us (the manager withdrew it
    # first). Both are lost races, not faults — the fixture's job is to reach an
    # empty queue, so it retries instead of failing the test that follows it. A
    # DELETE that 404s on an item already gone is success by any measure the
    # caller cares about; only a still-dirty queue after every attempt is a
    # failure worth reporting.
    remaining = snapshot["status"]["items_in_queue"]
    for _attempt in range(3):
        for item in snapshot.get("items") or []:
            uid = item.get("item_uid")
            if not isinstance(uid, str):
                continue
            _queue_drive.request(BRIDGE_URL, f"/queue/items/{uid}", "DELETE")
        snapshot = _queue_snapshot()
        remaining = snapshot["status"]["items_in_queue"]
        if remaining == 0:
            break

    assert remaining == 0, (
        f"the queue still holds {remaining} item(s) after three drain attempts — "
        "this test's armed start would drain work it did not enqueue. Items: "
        f"{[i.get('item_uid') for i in snapshot.get('items') or []]}"
    )


# ---------------------------------------------------------------------------
# The live tests. One operator request each, run against the one deployed
# stack, graded by the floor and then by the rubric — both defined at the top
# of this module and both already dry-verified there.
#
# THE REQUESTS ARE OPERATOR LANGUAGE, DELIBERATELY. Each one names the
# MEASUREMENT an operator wants and nothing else: no tool or server names, no
# plan name, no device names, no mention of drafts, queues, or where the data
# lands. Finding the tools and choosing the devices IS the task — a request
# that spelled either would grade the agent on following instructions rather
# than on running a measurement, and would keep passing after the wiring that
# lets it discover them broke.
#
# Markers are stacked per-test rather than declared in a module ``pytestmark``:
# everything above this point runs offline, and blinding those grading-contract
# checks behind the same Docker/SDK/credential skips would defeat the point of
# dry-verifying the grader before a live run is spent on it.
# ---------------------------------------------------------------------------

ORM_OPERATOR_REQUEST = (
    "Please run an orbit-response measurement on the machine for me: I want to "
    "see how the beam orbit moves when the steering correctors are driven. "
    "Once it has run, tell me what the measurement shows about the way the "
    "ring responds to the steering."
)


@pytest.mark.e2e
@pytest.mark.slow
# dockerbuild: builds the VA/bridge/panels images and deploys the full queue
# stack — runs in its own CI job, never the shared e2e-tests lane (the
# marker -> --ignore pairing is enforced by
# tests/deployment/test_ci_workflow_wiring.py).
@pytest.mark.dockerbuild
@pytest.mark.agentic_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
# Reruns absorb LLM sampling, nothing else. ``only_rerun`` keeps that honest:
# the floor and the judge both fail as AssertionError, while a stack that never
# came up fails as something else and so is reported on the first attempt
# instead of costing three full agent runs to say the same thing. Reruns are
# cheap here — ``deployed_scan_stack`` is module-scoped and survives them, and
# ``clean_queue`` re-empties the queue before each attempt, so a rerun cannot
# inherit a scan the previous attempt left mid-flight.
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_agent_measures_orbit_response_on_a_healthy_stack(
    deployed_scan_stack: DeployedScanStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked for an orbit-response measurement in plain operator language, the
    agent must actually take one on the deployed stack and report what it read
    back.

    The floor answers "did a scan of that class get staged, launched, and read
    back?" deterministically; the judge covers only what a trace cannot see —
    whether the prose delivers the result rather than stopping at a plan or
    inventing findings the healthy stack cannot support.
    """
    repo = deployed_scan_stack.repo
    # The bluesky MCP server runs host-side in the agent's session and reaches
    # the deployed bridge over these two: the URL is this module's own pinned
    # port, and the token is the one `osprey up` minted — the arming step on
    # the queue is gated by exactly that value, so without it the agent could
    # stage and queue a scan but never start one.
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", BRIDGE_URL)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", deployed_scan_stack.token)

    result = await run_sdk_query(
        repo,
        ORM_OPERATOR_REQUEST,
        max_turns=60,
        max_budget_usd=10.0,
        # Opus-tier: composing a scan, waiting it out, and then committing to a
        # reading of the data is the multi-step reasoning this lane measures.
        model=_default_opus_model(repo),
        # No Bash/Glob/Grep. The agent's compute path is the sanctioned python
        # executor; shell and filesystem search would let it inspect the
        # deployed project instead of measuring the machine it was asked about.
        disallowed_tools=SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    )

    assert_orbit_response_scan_executed(result)

    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _to_workflow_result(ORM_OPERATOR_REQUEST, result),
        expectations=ORM_JUDGE_EXPECTATIONS,
    )
    assert eval.passed, eval.reasoning


GRID_OPERATOR_REQUEST = (
    "I would like a two-dimensional map on the machine. Pick two of the "
    "steering correctors, step them together over a grid of settings across "
    "the range they are allowed, and take a beam-position reading at every "
    "point of that grid. Once it has run, tell me what the map looks like "
    "over the region you covered."
)


@pytest.mark.e2e
@pytest.mark.slow
# dockerbuild: see the orbit-response test above — same stack, same CI job.
@pytest.mark.dockerbuild
@pytest.mark.agentic_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_agent_maps_a_two_axis_grid_on_a_healthy_stack(
    deployed_scan_stack: DeployedScanStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked in plain operator language for a two-dimensional map, the agent
    must run a grid-scan-class measurement on the deployed stack and report
    what the region it covered looks like.

    The second measurement class, on the same stack and the same deploy as the
    orbit-response test above — which is the point of running it here rather
    than in a module of its own. The two floors are mutually exclusive (proved
    offline in ``test_grid_and_orbit_response_floors_do_not_accept_each_other``),
    so neither test can be satisfied by the other's run, and ``clean_queue``
    leaves this one an empty queue no matter what the earlier test left behind.
    """
    repo = deployed_scan_stack.repo
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", BRIDGE_URL)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", deployed_scan_stack.token)

    result = await run_sdk_query(
        repo,
        GRID_OPERATOR_REQUEST,
        max_turns=60,
        max_budget_usd=10.0,
        model=_default_opus_model(repo),
        disallowed_tools=SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    )

    assert_grid_scan_executed(result)

    judge = LLMJudge(provider=JUDGE_PROVIDER)
    eval = await judge.evaluate(
        _to_workflow_result(GRID_OPERATOR_REQUEST, result),
        expectations=GRID_JUDGE_EXPECTATIONS,
    )
    assert eval.passed, eval.reasoning
