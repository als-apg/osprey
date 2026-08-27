"""E2E tests for the ``pyat-specialist`` framework subagent.

Two halves share one deployment-build path:

- **Delegation** (:func:`test_pyat_specialist_delegation`): an operator-style,
  unambiguously *computational* lattice question must route to the
  ``pyat-specialist`` subagent, which computes via ``mcp__python__execute``.
  Asserts the orchestrator never runs the lattice computation itself and that
  every subagent ``execute`` call is readonly-or-unset (the template pins
  ``execution_mode="readonly"`` for the in-memory simulation).

- **Grounding** (:func:`test_pyat_specialist_grounding`): grades the subagent's
  answer — the artifact it files and returns — against ground truth computed
  in-test from ``build_ring()`` with the *identical* 4D recipe (no pinned
  numeric literals). One LLM judge checks both halves: every requested quantity
  present and within tolerance, and the answer labeled as computed from the
  simulated design lattice. The judge reads the numbers wherever the answer put
  them, in prose or in a table.

These tests use real API calls via the Claude Agent SDK — zero mocking.

**LOCAL-ONLY E2E.** Skipped in CI; the pyAT compute path is not provisioned on
GitHub Actions runners. To run locally you need:

- Claude Code CLI installed (``brew install claude``)
- ``claude_agent_sdk`` Python package installed
- ``ALS_APG_API_KEY`` (the CI-default als-apg provider)
- ``accelerator-toolbox`` (pyAT) and ``osprey-framework`` importable — both
  core deps of the worktree venv (imported lazily inside the grounding test so
  collection never fails on a host that lacks them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.judge import LLMJudge, WorkflowResult
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SDKWorkflowResult,
    e2e_budget_scale,
    init_project,
    is_claude_code_available,
    run_sdk_query_with_hooks,
)

# ---------------------------------------------------------------------------
# Module-level markers
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agentic_benchmark,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]

# The pyAT compute path: the ``python`` MCP server's ``execute`` tool.
_PY_EXEC_TOOL = "mcp__python__execute"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_agent_traces(result: SDKWorkflowResult) -> list:
    """Tool traces that belong to a sub-agent (have a parent_tool_use_id)."""
    return [t for t in result.tool_traces if t.parent_tool_use_id is not None]


def _execute_traces(result: SDKWorkflowResult) -> list:
    """All ``mcp__python__execute`` tool traces, orchestrator and sub-agent."""
    return [t for t in result.tool_traces if t.name == _PY_EXEC_TOOL]


def _print_trace_debug(test_name: str, result: SDKWorkflowResult) -> None:
    """Print standard debug output for a pyat-specialist test."""
    print(f"\n--- {test_name} ---")
    print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    for t in result.tool_traces:
        parent_flag = f" (sub-agent: {t.parent_tool_use_id})" if t.parent_tool_use_id else ""
        mode = t.input.get("execution_mode") if t.name == _PY_EXEC_TOOL else None
        mode_flag = f" [execution_mode={mode!r}]" if mode is not None else ""
        print(f"  tool: {t.name}{parent_flag}{mode_flag}")


# ---------------------------------------------------------------------------
# Delegation half (Task 3.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_pyat_specialist_delegation(tmp_path: Path) -> None:
    """A computational lattice question routes to pyat-specialist, not the orchestrator.

    The prompt asks for beta functions at every BPM and the fractional tunes —
    quantities that can only come from running optics code on the lattice
    model, so the orchestrator must hand off to the ``pyat-specialist``
    subagent rather than answering from training data or computing them itself.

    Delegation is forced by the CLAUDE.md directive ("For ANY lattice/optics
    computation … delegate to pyat-specialist; do NOT compute lattice
    quantities via mcp__python__execute yourself"), not by an SDK-level tool
    block (which would propagate to the subagent and starve the very tool it
    needs). The regression signals under test:

    - The orchestrator never calls ``mcp__python__execute`` for the lattice
      computation directly (``parent_tool_use_id is None``).
    - Every subagent ``execute`` call is readonly-or-unset — a ``readwrite``
      would mean the readonly pin was ignored and would risk a write-approval
      block in headless mode.
    """
    repo = init_project(tmp_path, "pyat_del", template="control_assistant", provider="als-apg")
    prompt = (
        "For the storage ring, compute the horizontal and vertical beta "
        "functions at every beam position monitor and report the fractional "
        "betatron tunes."
    )

    # ``run_sdk_query_with_hooks`` (permission_mode="default" + auto-approve
    # callback) is required: ``mcp__python__execute`` fires a PreToolUse
    # approval "ask" hook, and the plain ``run_sdk_query`` (bypassPermissions,
    # no callback) would DENY it — the subagent's compute would never run.
    result = await run_sdk_query_with_hooks(
        repo, prompt, approval_policy="auto_approve", max_turns=25, max_budget_usd=2.0
    )

    _print_trace_debug("pyat-specialist delegation", result)

    # Session completed without an SDK-level error.
    assert result.result is not None, "No ResultMessage received"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # A sub-agent was invoked at all.
    sa_traces = _sub_agent_traces(result)
    assert sa_traces, (
        "No sub-agent tool calls observed. The CLAUDE.md delegation directive "
        "('For ANY lattice/optics computation … delegate to pyat-specialist') "
        f"was not followed. Tools called: {result.tool_names}"
    )

    # The lattice computation ran inside the subagent: mcp__python__execute
    # called with a non-None parent_tool_use_id.
    sub_exec = [t for t in _execute_traces(result) if t.parent_tool_use_id is not None]
    assert sub_exec, (
        f"pyat-specialist never called {_PY_EXEC_TOOL} from a sub-agent "
        "context. Either the orchestrator answered from training data or the "
        f"compute never ran. Tools called: {result.tool_names}"
    )

    # Every subagent execute call is readonly-or-unset (never readwrite).
    bad_mode = [t for t in sub_exec if t.input.get("execution_mode") not in (None, "readonly")]
    assert not bad_mode, (
        "Subagent ran mcp__python__execute in a non-readonly mode "
        f"{[t.input.get('execution_mode') for t in bad_mode]} — the template's "
        "readonly pin ('Always pass execution_mode=readonly') was ignored."
    )

    # The orchestrator itself never computes the lattice question directly.
    direct_exec = [t for t in _execute_traces(result) if t.parent_tool_use_id is None]
    assert not direct_exec, (
        f"Orchestrator called {_PY_EXEC_TOOL} directly ({len(direct_exec)}x) "
        "instead of delegating to pyat-specialist. The CLAUDE.md delegation "
        "prohibition ('do NOT compute lattice quantities … yourself') was "
        "ignored — the silent non-delegation regression this test exists to catch."
    )

    # Cost under budget.
    if result.cost_usd is not None:
        budget = 2.0 * e2e_budget_scale()
        assert result.cost_usd < budget, (
            f"Test cost ${result.cost_usd:.4f} — exceeded ${budget:.2f} budget"
        )


# ---------------------------------------------------------------------------
# Grounding half (Task 3.2)
# ---------------------------------------------------------------------------

# Tolerances (pinned — do not loosen to pass):
#   tunes:         compared modulo 1, ABSOLUTE 1e-3 (tests/simulation/test_fidelity.py convention)
#   circumference: 1e-6 RELATIVE
#   beta:          1% RELATIVE at named elements
_TUNE_ABS_TOL = 1e-3
_CIRCUMFERENCE_REL_TOL = 1e-6
_BETA_REL_TOL = 0.01

_X_TOKENS = ("x", "h", "horiz", "horizontal")
_Y_TOKENS = ("y", "v", "vert", "vertical")


def _ground_truth() -> dict:
    """Compute the reference quantities in-test with the template's 4D recipe.

    Identical recipe to the agent template: ``build_ring()`` → ``deepcopy`` →
    ``disable_6d()`` → ``at.get_optics(...)``. No pinned numeric literals — the
    truth is recomputed from the shared lattice at use time so the test never
    rots against a hand-copied constant. Imported lazily so module collection
    never fails on a host that lacks accelerator-toolbox / osprey-framework.
    """
    import copy

    import at

    from osprey.simulation.lattice import build_ring

    ring = build_ring()
    ring4d = copy.deepcopy(ring)
    ring4d.disable_6d()
    _, ringdata, elemdata = at.get_optics(ring4d, refpts=range(len(ring4d)))

    def beta_at(name: str) -> list[float]:
        idx = [i for i, el in enumerate(ring4d) if el.FamName == name]
        assert idx, f"named element {name!r} not present in the lattice"
        return [float(elemdata.beta[idx[0]][0]), float(elemdata.beta[idx[0]][1])]

    return {
        "tune": (float(ringdata.tune[0]), float(ringdata.tune[1])),
        "circumference": float(ring4d.circumference),
        "beta": {"BPM01": beta_at("BPM01"), "BPM03": beta_at("BPM03")},
    }


def _to_workflow_result(query: str, sdk_result: SDKWorkflowResult) -> WorkflowResult:
    """Convert an ``SDKWorkflowResult`` into the plain-text shape the judge reads."""
    response = "\n".join(sdk_result.text_blocks).strip()
    trace_lines: list[str] = []
    for t in sdk_result.tool_traces:
        trace_lines.append(f"TOOL: {t.name}")
        if t.result:
            preview = t.result[:300] + ("…" if len(t.result) > 300 else "")
            trace_lines.append(f"  result: {preview}")
    return WorkflowResult(
        query=query,
        response=response,
        execution_trace="\n".join(trace_lines),
        artifacts=[],
    )


@pytest.mark.asyncio
# The retries absorb the LLM's turn-to-turn variance and nothing else. A verdict
# that fails on every attempt is a real regression in what the subagent computes
# or reports — widen neither the retries nor the tolerances.
@pytest.mark.flaky(reruns=2)
async def test_pyat_specialist_grounding(tmp_path: Path) -> None:
    """The numbers in the subagent's answer match ground truth, and it says where
    they came from.

    The answer is the deliverable, so the answer is what gets graded: the judge
    is handed the reference values — recomputed in-test with the template's own
    4D recipe, no pinned literals — and the tolerance each must hold to. It
    fails the response for a wrong number as readily as for a missing
    provenance statement, wherever in the prose or its tables the value appears.
    """
    repo = init_project(tmp_path, "pyat_grd", template="control_assistant", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    prompt = (
        "For the storage ring, compute and report the fractional betatron "
        "tunes, the ring circumference in meters, and the horizontal and "
        "vertical beta functions at BPM01 and BPM03."
    )

    # See the delegation test: the executor's approval "ask" hook needs the
    # auto-approve callback, or the compute never runs.
    result = await run_sdk_query_with_hooks(
        repo, prompt, approval_policy="auto_approve", max_turns=25, max_budget_usd=2.0
    )

    _print_trace_debug("pyat-specialist grounding", result)

    assert result.result is not None, "No ResultMessage received"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # Cost under budget.
    if result.cost_usd is not None:
        budget = 2.0 * e2e_budget_scale()
        assert result.cost_usd < budget, (
            f"Test cost ${result.cost_usd:.4f} — exceeded ${budget:.2f} budget"
        )

    truth = _ground_truth()
    nu_x, nu_y = truth["tune"]
    bpm01_x, bpm01_y = truth["beta"]["BPM01"]
    bpm03_x, bpm03_y = truth["beta"]["BPM03"]

    result_eval = await judge.evaluate(
        _to_workflow_result(prompt, result),
        expectations=(
            "Grade the response on TWO things.\n\n"
            "(1) NUMERIC CORRECTNESS. The reference values below were computed "
            "from the same lattice with the same recipe and are correct. Find "
            "each quantity in the response — it may appear in a sentence or in "
            "a table, under any reasonable name or symbol (nu_x/Qx/horizontal "
            "tune; beta_x/βx) and in any order — and compare it to the "
            "reference. FAIL if any is missing, or differs by more than its "
            "tolerance. Tunes may be reported with or without the integer part; "
            "compare only the FRACTIONAL part, modulo 1.\n"
            f"  - horizontal tune: {nu_x!r} (fractional part; tolerance "
            f"{_TUNE_ABS_TOL:.0e} absolute)\n"
            f"  - vertical tune:   {nu_y!r} (fractional part; tolerance "
            f"{_TUNE_ABS_TOL:.0e} absolute)\n"
            f"  - circumference:   {truth['circumference']!r} m (tolerance "
            f"{_CIRCUMFERENCE_REL_TOL:.0e} relative)\n"
            f"  - beta at BPM01:   x={bpm01_x!r} m, y={bpm01_y!r} m (tolerance "
            f"{_BETA_REL_TOL:.0%} relative)\n"
            f"  - beta at BPM03:   x={bpm03_x!r} m, y={bpm03_y!r} m (tolerance "
            f"{_BETA_REL_TOL:.0%} relative)\n\n"
            "(2) PROVENANCE. The response explicitly states that the reported "
            "quantities were COMPUTED from the simulated ALS-U Accumulator Ring "
            "(AR) design lattice — simulation-derived from the lattice/optics "
            "model, not a live machine reading or measured data.\n\n"
            "FAIL on an unhandled error. Do not reward a confident tone: a "
            "number outside tolerance fails no matter how it is presented. "
            "In your reasoning, quote each value you found and the reference "
            "you compared it to."
        ),
    )
    assert result_eval.passed, result_eval.reasoning
