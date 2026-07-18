"""Assertions against the parsed `.github/workflows/ci.yml`, proving the two
secret-free Docker-stack e2e lanes (Tiled roundtrip, VA substrate
equivalence) are correctly wired.

Everything here loads ci.yml with ``yaml.safe_load`` and asserts against the
parsed structure — never a text/regex match, so re-flowing YAML style can't
fool a check. YAML 1.1 parses the bare `on:` workflow-trigger key to the
Python boolean ``True``; this module never indexes ``workflow["on"]``.

Two secret-free jobs exist because two e2e files needed one each:

* ``test_tiled_roundtrip.py`` (Task 4.1) is new in this phase and never had
  a lane at all.
* ``test_va_substrate_equivalence.py`` (Phase 1) had a CI bug: it carried no
  dedicated job and was only ever swept up by ``e2e-tests``' glob over
  ``tests/e2e/`` — contradicting its own module docstring ("never collected
  by the fast lane"). Fixing that means giving it a real lane, not just
  removing its only lane via ``--ignore``.

Every assertion is paired with a mutation test: a fresh, in-memory-mutated
copy of the parsed workflow reintroduces exactly the bug the assertion
exists to catch, and the same assertion must then fail. ci.yml itself is
never edited by this module.
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

TILED_JOB = "tiled-roundtrip-e2e"
VA_JOB = "va-substrate-equivalence-e2e"
UNIT_TEST_JOB = "test"
E2E_TESTS_JOB = "e2e-tests"
GATE_JOB = "all-checks-passed"
SECRET_TOKEN = "secrets.ALS_APG_API_KEY"
TILED_TEST_FILE = "tests/e2e/test_tiled_roundtrip.py"
VA_TEST_FILE = "tests/e2e/test_va_substrate_equivalence.py"


def _load_workflow() -> dict[str, Any]:
    with CI_YML.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded is not None, f"{CI_YML} parsed to None"
    return loaded


@pytest.fixture()
def workflow() -> dict[str, Any]:
    return _load_workflow()


def _jobs(wf: dict[str, Any]) -> dict[str, Any]:
    return wf["jobs"]


def _job_declares_secret(wf: dict[str, Any], job_name: str, token: str) -> bool:
    """Search a job's fully serialized form for a secret-expression token.

    Serializing the whole job (not just top-level keys) so the search also
    covers step-level ``env:``/``run:`` blocks, matching how
    ``ALS_APG_API_KEY`` actually appears in the secret-gated jobs (e.g.
    ``e2e-tests``'s pre-flight probe step).
    """
    return token in json.dumps(_jobs(wf)[job_name])


def _find_named_step(wf: dict[str, Any], job_name: str, step_name: str) -> dict[str, Any]:
    for step in _jobs(wf)[job_name]["steps"]:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"job '{job_name}' has no step named '{step_name}'")


# ---------------------------------------------------------------------------
# (a) tiled-roundtrip-e2e job exists
# ---------------------------------------------------------------------------


def test_tiled_roundtrip_job_exists(workflow: dict[str, Any]) -> None:
    assert TILED_JOB in _jobs(workflow)


def test_tiled_roundtrip_job_exists__mutation_drops_job() -> None:
    """Dropping the job from the parsed dict must fail the existence check."""
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][TILED_JOB]
    with pytest.raises(AssertionError):
        assert TILED_JOB in _jobs(mutated)


# ---------------------------------------------------------------------------
# (a) va-substrate-equivalence-e2e job exists (the Phase-1 gap fix)
# ---------------------------------------------------------------------------


def test_va_substrate_equivalence_job_exists(workflow: dict[str, Any]) -> None:
    assert VA_JOB in _jobs(workflow)


def test_va_substrate_equivalence_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][VA_JOB]
    with pytest.raises(AssertionError):
        assert VA_JOB in _jobs(mutated)


# ---------------------------------------------------------------------------
# (b) tiled-roundtrip-e2e declares no ALS_APG_API_KEY secret anywhere in it
# ---------------------------------------------------------------------------


def test_tiled_roundtrip_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    assert not _job_declares_secret(workflow, TILED_JOB, SECRET_TOKEN)


def test_tiled_roundtrip_job_has_no_llm_secret__mutation_adds_secret() -> None:
    """Injecting the secret into a step env must flip the check to failing."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][TILED_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        assert not _job_declares_secret(mutated, TILED_JOB, SECRET_TOKEN)


def test_tiled_roundtrip_job_has_no_llm_secret__mutation_survives_lookalike() -> None:
    """A differently-spelled secret expression must NOT trip the assertion —
    proves the check matches the real token, not any 'secrets.' substring."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][TILED_JOB]["steps"].append(
        {"name": "unrelated", "env": {"CODECOV_TOKEN": "${{ secrets.CODECOV_TOKEN }}"}}
    )
    assert not _job_declares_secret(mutated, TILED_JOB, SECRET_TOKEN)


# ---------------------------------------------------------------------------
# (b) va-substrate-equivalence-e2e declares no ALS_APG_API_KEY secret either
# ---------------------------------------------------------------------------


def test_va_substrate_equivalence_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    assert not _job_declares_secret(workflow, VA_JOB, SECRET_TOKEN)


def test_va_substrate_equivalence_job_has_no_llm_secret__mutation_adds_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][VA_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        assert not _job_declares_secret(mutated, VA_JOB, SECRET_TOKEN)


# ---------------------------------------------------------------------------
# (c) e2e-tests' run step --ignores both new-lane files
# ---------------------------------------------------------------------------


def test_e2e_tests_ignores_both_new_lane_files(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, E2E_TESTS_JOB, "Run E2E tests")
    run_text = step["run"]
    assert f"--ignore={TILED_TEST_FILE}" in run_text
    assert f"--ignore={VA_TEST_FILE}" in run_text


def _drop_ignore_line(run_text: str, target_file: str) -> str:
    """Remove the `--ignore=<target_file>` line, whichever way it happens to
    be terminated (line-continuation backslash mid-list, or bare newline on
    the last entry)."""
    lines = run_text.splitlines(keepends=True)
    kept = [line for line in lines if f"--ignore={target_file}" not in line]
    assert len(kept) == len(lines) - 1, f"expected exactly one line dropped for {target_file}"
    return "".join(kept)


def test_e2e_tests_ignores_both_new_lane_files__mutation_drops_tiled_ignore() -> None:
    """Removing only the tiled-roundtrip ignore line must fail — proves the
    two ignore assertions are independent, not satisfied by either alone."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TILED_TEST_FILE)
    assert f"--ignore={VA_TEST_FILE}" in step["run"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert f"--ignore={TILED_TEST_FILE}" in step["run"]


def test_e2e_tests_ignores_both_new_lane_files__mutation_drops_va_ignore() -> None:
    """Removing only the VA-substrate ignore line must fail the same way,
    confirming the two files are checked independently in the real assertion."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], VA_TEST_FILE)
    assert f"--ignore={TILED_TEST_FILE}" in step["run"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert f"--ignore={VA_TEST_FILE}" in step["run"]


# ---------------------------------------------------------------------------
# (d) all-checks-passed depends on the new job(s)
# ---------------------------------------------------------------------------


def test_all_checks_passed_needs_tiled_roundtrip(workflow: dict[str, Any]) -> None:
    assert TILED_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_all_checks_passed_needs_tiled_roundtrip__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    needs = _jobs(mutated)[GATE_JOB]["needs"]
    needs.remove(TILED_JOB)
    with pytest.raises(AssertionError):
        assert TILED_JOB in _jobs(mutated)[GATE_JOB]["needs"]


def test_all_checks_passed_needs_va_substrate_equivalence(workflow: dict[str, Any]) -> None:
    assert VA_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_all_checks_passed_needs_va_substrate_equivalence__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    needs = _jobs(mutated)[GATE_JOB]["needs"]
    needs.remove(VA_JOB)
    with pytest.raises(AssertionError):
        assert VA_JOB in _jobs(mutated)[GATE_JOB]["needs"]


def _needs_contains_both_new_jobs(wf: dict[str, Any]) -> bool:
    """Deliberately `all(...)`, not `any(...)`. An `any`-shaped check would
    pass with only one of the two new jobs wired into the gate — exactly the
    silent-partial-fix shape this phase has repeatedly caught elsewhere."""
    needs = _jobs(wf)[GATE_JOB]["needs"]
    return all(job in needs for job in (TILED_JOB, VA_JOB))


def test_all_checks_passed_needs_both_new_jobs(workflow: dict[str, Any]) -> None:
    """(g) A single job nobody depends on can go red forever without
    blocking a merge; both new e2e lanes must actually gate the merge."""
    assert _needs_contains_both_new_jobs(workflow)


def test_all_checks_passed_needs_both_new_jobs__mutation_drops_only_tiled() -> None:
    """Dropping just tiled-roundtrip-e2e (VA stays) must still fail the
    'both' check — proves it isn't secretly an `any`."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(TILED_JOB)
    assert VA_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _needs_contains_both_new_jobs(mutated)


def test_all_checks_passed_needs_both_new_jobs__mutation_drops_only_va() -> None:
    """Dropping just va-substrate-equivalence-e2e (tiled stays) must also
    fail the 'both' check, confirming both entries are independently load-bearing."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(VA_JOB)
    assert TILED_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _needs_contains_both_new_jobs(mutated)


# ---------------------------------------------------------------------------
# The unit-test job must install the extras its tests import
# ---------------------------------------------------------------------------


def _unit_test_install_cmd(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, UNIT_TEST_JOB, "Install dependencies")["run"]


def test_unit_test_job_installs_required_extras(workflow: dict[str, Any]) -> None:
    """tests/va/* import softioc unguarded and error at collection without the
    `virtual-accelerator` extra (an empty back-compat alias today, but pinned so
    older-version builds keep resolving); `dev` carries pytest itself."""
    cmd = _unit_test_install_cmd(workflow)
    for extra in ("dev", "virtual-accelerator"):
        assert f"--extra {extra}" in cmd, (
            f"unit-test job must `uv sync --extra {extra}`; got: {cmd}"
        )


def test_unit_test_job_installs_required_extras__mutation_drops_extra() -> None:
    """Dropping a pinned extra must fail — otherwise the guard is decorative."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Install dependencies")
    step["run"] = step["run"].replace(" --extra virtual-accelerator", "")
    with pytest.raises(AssertionError):
        test_unit_test_job_installs_required_extras(mutated)


def test_bluesky_stack_is_a_core_dependency() -> None:
    """The bridge's unit tests guard their imports with `pytest.importorskip`,
    so a missing bluesky stack does not error — it SKIPS, and the scanner,
    RunEngine-integration and Tiled fault-isolation guards vanish from CI
    silently, inside a green check. Bluesky is a core part of OSPREY: pin the
    stack in [project] dependencies so plain `uv sync` always installs it and
    those tests can never resume skipping unnoticed."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    core_deps = pyproject["project"]["dependencies"]
    for stack_dep in ("bluesky", "ophyd-async", "tiled"):
        assert any(dep.startswith(stack_dep) for dep in core_deps), (
            f"{stack_dep} must be a core dependency"
        )


# ---------------------------------------------------------------------------
# YAML 1.1 `on:` gotcha regression guard
# ---------------------------------------------------------------------------


def test_workflow_on_key_parses_to_bool_true_not_string(workflow: dict[str, Any]) -> None:
    """Documents the YAML 1.1 footgun this module deliberately avoids: the
    bare `on:` trigger key parses to the Python bool True, not the string
    'on'. Indexing workflow['on'] would KeyError; every fixture/helper above
    only ever indexes workflow['jobs']."""
    assert True in workflow
    assert "on" not in workflow
