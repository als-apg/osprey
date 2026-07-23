"""E2E tests for the ``pyat-specialist`` framework subagent.

Two halves share one project-build path:

- **Delegation** (:func:`test_pyat_specialist_delegation`): an operator-style,
  unambiguously *computational* lattice question must route to the
  ``pyat-specialist`` subagent, which computes via ``mcp__python__execute``.
  Asserts the orchestrator never runs the lattice computation itself and that
  every subagent ``execute`` call is readonly-or-unset (the template pins
  ``execution_mode="readonly"`` for the in-memory simulation).

- **Grounding** (:func:`test_pyat_specialist_grounding`): reads the exact
  floats the subagent saved to its results JSON artifact and checks them
  against ground truth computed in-test from ``build_ring()`` with the
  *identical* 4D recipe (no pinned numeric literals). An LLM judge confirms
  provenance and simulation-derived labeling in the prose only — numbers are
  never parsed out of prose.

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

import json
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
    project = init_project(tmp_path, "pyat_del", template="control_assistant", provider="als-apg")
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
        project, prompt, approval_policy="auto_approve", max_turns=25, max_budget_usd=2.0
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


def _leaves(obj, prefix: str = ""):
    """Yield ``(lowercased_path, value)`` for every scalar leaf of a JSON tree.

    Nested dicts extend the path with ``.key`` and lists with ``[i]`` so a value
    can be located by any substring of its full path — this is what makes the
    result-key matching tolerant of how the subagent chose to nest/name keys
    (``beta.BPM01.x`` and ``BPM01.beta_x`` both carry ``beta``/``bpm01``/``x``).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}.{k}".lower())
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _coerce_float(value, label: str) -> float:
    """Coerce a saved value to float, failing loudly on a numpy display-repr string.

    The template requires native ``float()`` coercion before ``save_artifact``;
    a value like ``'np.float64(0.22)'`` means that step was skipped. Surface it
    as a clear, actionable error rather than an opaque ``ValueError``.
    """
    if isinstance(value, bool):  # bool is an int subclass — never a physics scalar
        raise AssertionError(f"{label}: got a boolean {value!r}, expected a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{label}: expected a numeric value but got {value!r} — likely an "
            "un-coerced numpy display-repr string. The subagent must convert to "
            "native float()/int() before save_artifact (per the template's "
            "'Returning Results' recipe)."
        ) from exc


def _has_token(path: str, tokens) -> bool:
    """True if *path* carries a plane token as a delimited segment (avoids
    matching an incidental character inside a longer word)."""
    return any(
        f".{t}" in path or f"_{t}" in path or path.endswith(t) or f"[{t}]" in path for t in tokens
    )


def _resolve_two_plane(cands: list, label: str) -> tuple[float, float]:
    """Resolve (x, y) from candidate ``(path, value)`` leaves.

    Prefers explicit plane tokens (``_x``/``_y``, ``horizontal``/``vertical``);
    falls back to sorted order when exactly two candidates exist (covers a
    2-element list saved under one key, e.g. ``tune[0]``/``tune[1]``).
    """
    xs = [(p, v) for p, v in cands if _has_token(p, _X_TOKENS)]
    ys = [(p, v) for p, v in cands if _has_token(p, _Y_TOKENS)]
    if xs and ys:
        return _coerce_float(xs[0][1], f"{label} (x)"), _coerce_float(ys[0][1], f"{label} (y)")
    if len(cands) == 2:
        ordered = sorted(cands)
        return (
            _coerce_float(ordered[0][1], f"{label} (x)"),
            _coerce_float(ordered[1][1], f"{label} (y)"),
        )
    raise AssertionError(f"could not resolve horizontal/vertical {label} from saved keys: {cands}")


def _load_results_artifacts(project_dir: Path) -> list[tuple[dict, dict]]:
    """Return ``[(index_entry, parsed_json)]`` for the subagent's results artifacts.

    SDK e2e runs the artifact store **unscoped** (``OSPREY_SESSION_ID`` unset),
    so the shared root ``<project_dir>/_agent_data/artifacts/`` holds every
    artifact this run produced. Filter to ``artifact_type == 'json'`` AND
    ``category != 'code_output'`` — the latter drops the executor's auto-saved
    code+stdout wrapper (also stored as json), leaving the ``save_artifact``
    results dicts.
    """
    index_path = project_dir / "_agent_data" / "artifacts" / "artifacts.json"
    assert index_path.exists(), (
        "No artifact index at "
        f"{index_path} — the pyat-specialist never saved a results artifact. "
        "save_artifact failures are swallowed upstream (non-fatal), so a missing "
        "index means the subagent's compute/save pipeline did not complete."
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts_dir = index_path.parent
    out: list[tuple[dict, dict]] = []
    for entry in index.get("entries", []):
        if entry.get("artifact_type") != "json" or entry.get("category") == "code_output":
            continue
        fp = artifacts_dir / entry.get("filename", "")
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, (dict, list)):
            out.append((entry, data))
    return out


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
@pytest.mark.flaky(reruns=2)
async def test_pyat_specialist_grounding(tmp_path: Path) -> None:
    """The subagent's saved numbers match ground truth; prose carries provenance.

    Numeric correctness is checked against exact floats read from the results
    JSON artifact (never parsed out of prose) versus ground truth recomputed
    in-test with the identical 4D recipe. The LLM judge only confirms the prose
    labels the answer as simulation-derived — it never sees or grades a number.
    """
    project = init_project(tmp_path, "pyat_grd", template="control_assistant", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    prompt = (
        "For the storage ring, compute and report the fractional betatron "
        "tunes, the ring circumference in meters, and the horizontal and "
        "vertical beta functions at BPM01 and BPM03."
    )

    # See the delegation test: the executor's approval "ask" hook needs the
    # auto-approve callback, or the compute (and its saved artifact) never runs.
    result = await run_sdk_query_with_hooks(
        project, prompt, approval_policy="auto_approve", max_turns=25, max_budget_usd=2.0
    )

    _print_trace_debug("pyat-specialist grounding", result)

    assert result.result is not None, "No ResultMessage received"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # --- Numeric grounding: read exact floats from the results artifact(s) ---

    truth = _ground_truth()
    results = _load_results_artifacts(project)
    assert results, (
        "No results JSON artifact found (artifact_type=='json' and "
        "category!='code_output'). The subagent must save its computed "
        "quantities via save_artifact(dict, ...). Index entries seen: "
        f"{[(e.get('artifact_type'), e.get('category')) for e, _ in _all_index_entries(project)]}"
    )

    # Union the leaves across every results artifact so the check is robust to a
    # subagent that split its output across more than one save_artifact call.
    leaves: list[tuple[str, object]] = []
    for _entry, data in results:
        leaves.extend(_leaves(data))

    def _artifact_dump() -> str:
        return json.dumps([data for _e, data in results], indent=2, default=str)[:2000]

    # Tunes — compared modulo 1 with absolute tolerance. Match "tune" only:
    # a raw "nu" substring would false-match keys like "number_of_bpms".
    tune_cands = [(p, v) for p, v in leaves if "tune" in p]
    assert tune_cands, f"No tune keys in the results artifact(s):\n{_artifact_dump()}"
    got_nux, got_nuy = _resolve_two_plane(tune_cands, "tune")
    for label, got, ref in (
        ("nu_x", got_nux, truth["tune"][0]),
        ("nu_y", got_nuy, truth["tune"][1]),
    ):
        d = abs((got - ref) % 1.0)
        d = min(d, 1.0 - d)  # circular distance on the unit circle
        assert d < _TUNE_ABS_TOL, (
            f"{label} mismatch: got {got} (frac {got % 1.0:.6f}), "
            f"truth {ref} (frac {ref % 1.0:.6f}), |Δ mod 1| {d:.2e} ≥ {_TUNE_ABS_TOL:.0e}"
        )

    # Circumference — relative tolerance.
    circ_cands = [(p, v) for p, v in leaves if "circ" in p]
    assert circ_cands, f"No circumference key in the results artifact(s):\n{_artifact_dump()}"
    got_circ = _coerce_float(circ_cands[0][1], "circumference")
    rel = abs(got_circ - truth["circumference"]) / truth["circumference"]
    assert rel < _CIRCUMFERENCE_REL_TOL, (
        f"circumference mismatch: got {got_circ}, truth {truth['circumference']}, "
        f"rel {rel:.2e} ≥ {_CIRCUMFERENCE_REL_TOL:.0e}"
    )

    # Beta at named elements — 1% relative, per plane.
    for elem in ("BPM01", "BPM03"):
        beta_cands = [(p, v) for p, v in leaves if "beta" in p and elem.lower() in p]
        assert beta_cands, (
            f"No beta keys for {elem} in the results artifact(s) — the operator "
            f"asked for beta at {elem}.\n{_artifact_dump()}"
        )
        got_bx, got_by = _resolve_two_plane(beta_cands, f"beta at {elem}")
        ref_bx, ref_by = truth["beta"][elem]
        for label, got, ref in (
            (f"beta_x @ {elem}", got_bx, ref_bx),
            (f"beta_y @ {elem}", got_by, ref_by),
        ):
            rel = abs(got - ref) / ref
            assert rel < _BETA_REL_TOL, (
                f"{label} mismatch: got {got}, truth {ref}, rel {rel:.2e} ≥ {_BETA_REL_TOL:.0%}"
            )

    # Cost under budget.
    if result.cost_usd is not None:
        budget = 2.0 * e2e_budget_scale()
        assert result.cost_usd < budget, (
            f"Test cost ${result.cost_usd:.4f} — exceeded ${budget:.2f} budget"
        )

    # --- Provenance judged in PROSE only (numbers already verified above) ---

    result_eval = await judge.evaluate(
        _to_workflow_result(prompt, result),
        expectations=(
            "The response explicitly states that the reported quantities "
            "(tunes, circumference, beta functions) were COMPUTED from the "
            "simulated ALS-U Accumulator Ring (AR) design lattice — i.e. they "
            "are simulation-derived from the lattice/optics model, not a live "
            "machine reading or measured data. Do NOT grade the numeric values "
            "themselves; judge only whether the answer is clearly labeled as "
            "computed from the simulated design lattice and free of unhandled "
            "errors."
        ),
    )
    assert result_eval.passed, result_eval.reasoning


def _all_index_entries(project_dir: Path) -> list[tuple[dict, dict]]:
    """Every json artifact index entry (unfiltered) — used only to enrich a
    failure message when no results artifact is found."""
    index_path = project_dir / "_agent_data" / "artifacts" / "artifacts.json"
    if not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [(e, {}) for e in index.get("entries", [])]
