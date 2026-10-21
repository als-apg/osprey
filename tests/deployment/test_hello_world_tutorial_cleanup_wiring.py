"""The agentic lane's cleanup step removes exactly what the tutorial test deploys.

``test_hello_world_tutorial_query_through_gateway`` deploys the telemetry store
and gateway under one fixed compose project and tears it down in its own
``finally``. The lane carries an ``if: always()`` step that removes the same
containers and volumes by exact name, for a run killed before that ``finally``
ran. The two must name the same project, and the test must stay selected by
the lane's ``-k``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"
TEST_MODULE = REPO_ROOT / "tests/e2e/test_preset_agentic.py"
JOB = "agentic-per-preset"
TEST_NAME = "test_hello_world_tutorial_query_through_gateway"


def _module_constant(name: str) -> str:
    for node in ast.parse(TEST_MODULE.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    raise AssertionError(f"{TEST_MODULE.name} declares no {name}")


def _steps() -> list[dict]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][JOB]["steps"]


def _cleanup_step(project: str) -> dict:
    found = [
        step
        for step in _steps()
        if step.get("if") == "always()" and project in (step.get("run") or "")
    ]
    assert len(found) == 1, f"{JOB} has {len(found)} always() steps naming {project!r}"
    return found[0]


def test_the_cleanup_step_removes_by_exact_name_only() -> None:
    # Which names the step removes is pinned by TELEMETRY_CLEANUP_STEPS in
    # test_ci_workflow_wiring.py; this guards against the broad forms.
    run = _cleanup_step(_module_constant("TUTORIAL_PROJECT_NAME"))["run"]
    for broad in ("prune", " -a", "--all", "*", "xargs"):
        assert broad not in run, f"the cleanup step uses {broad!r}; names must be exact"


def test_the_cleanup_runs_after_the_failure_capture() -> None:
    steps = _steps()
    project = _module_constant("TUTORIAL_PROJECT_NAME")
    cleanup = steps.index(_cleanup_step(project))
    capture = next(
        i for i, step in enumerate(steps) if "capture-ci-diagnostics" in (step.get("uses") or "")
    )
    assert capture < cleanup


def test_the_lane_selects_the_tutorial_test() -> None:
    preset = "hello-world"
    assert (
        preset
        in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][JOB]["strategy"]["matrix"][
            "preset"
        ]
    )
    assert preset.replace("-", "_") in TEST_NAME
    names = {
        node.name
        for node in ast.parse(TEST_MODULE.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert TEST_NAME in names
