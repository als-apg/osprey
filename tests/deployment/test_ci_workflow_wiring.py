"""Assertions against the parsed `.github/workflows/ci.yml`, proving the
secret-free Docker-stack e2e lanes (Tiled roundtrip, VA substrate
equivalence, control-assistant demo) are correctly wired — and that every
``dockerbuild``-marked e2e file stays out of the shared e2e-tests lane.

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
DEMO_JOB = "multi-user-demo-e2e"
DEMO_TEST_FILE = "tests/e2e/test_multi_user_demo.py"
LIFECYCLE_TEST_FILE = "tests/e2e/test_deploy_lifecycle.py"
ORM_JOB = "orm-roundtrip-e2e"
ORM_TEST_FILE = "tests/e2e/test_orm_roundtrip.py"
OVERLAY_JOB = "dispatch-overlay-e2e"
OVERLAY_TEST_FILE = "tests/e2e/test_dispatch_overlay_visibility.py"
CATALOG_JOB = "bluesky-catalog-e2e"
SANDBOX_JOB = "bluesky-sandbox-escape-e2e"
BENCHMARKS_JOB = "channel-finder-benchmarks"
BENCHMARKS_TEST_FILE = "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py"


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
# (e) multi-user-demo-e2e lane + the dockerbuild --ignore guard
# ---------------------------------------------------------------------------


def test_multi_user_demo_job_exists(workflow: dict[str, Any]) -> None:
    assert DEMO_JOB in _jobs(workflow)


def test_multi_user_demo_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][DEMO_JOB]
    with pytest.raises(AssertionError):
        assert DEMO_JOB in _jobs(mutated)


def test_multi_user_demo_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    assert not _job_declares_secret(workflow, DEMO_JOB, SECRET_TOKEN)


def test_all_checks_passed_needs_multi_user_demo(workflow: dict[str, Any]) -> None:
    assert DEMO_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_all_checks_passed_needs_multi_user_demo__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(DEMO_JOB)
    with pytest.raises(AssertionError):
        assert DEMO_JOB in _jobs(mutated)[GATE_JOB]["needs"]


def _dockerbuild_marked_e2e_files() -> list[str]:
    """Every ``tests/e2e/`` file whose source carries the ``dockerbuild``
    marker. Text scan, not collection: importing the files would need their
    (heavy, optional) e2e dependencies, and the marker is always spelled
    literally at module or test level."""
    e2e_dir = CI_YML.parents[2] / "tests" / "e2e"
    return sorted(
        (p.relative_to(e2e_dir.parents[1])).as_posix()
        for p in e2e_dir.rglob("test_*.py")
        if "pytest.mark.dockerbuild" in p.read_text(encoding="utf-8")
    )


def _run_step_ignores_all(wf: dict[str, Any], files: list[str]) -> list[str]:
    """Return the subset of ``files`` MISSING from the e2e-tests run step's
    ``--ignore`` list (empty = fully guarded)."""
    run_text = _find_named_step(wf, E2E_TESTS_JOB, "Run E2E tests")["run"]
    return [f for f in files if f"--ignore={f}" not in run_text]


def test_every_dockerbuild_marked_file_is_ignored_in_e2e_lane(workflow: dict[str, Any]) -> None:
    """A ``dockerbuild``-marked e2e file runs a real image build + full-stack
    deploy; swept into the shared e2e-tests lane it either double-executes
    (if it has its own job) or leaves host-global residue — fixed
    ``<prefix>-web-<user>``/``<prefix>-nginx`` container names and the
    host-global openobserve data volume (root creds pinned on first init) —
    that breaks later tests on the same runner. There is no marker-expression
    equivalent (``-m "not dockerbuild"`` would also drop legit in-lane
    ``slow``-marked tests sharing files), so the curated ``--ignore`` list IS
    the mechanism; this guard makes it total: every marked file, present or
    future, must be ignored here and given its own job."""
    files = _dockerbuild_marked_e2e_files()
    assert files, "expected at least one dockerbuild-marked e2e file (marker scan broke?)"
    missing = _run_step_ignores_all(workflow, files)
    assert missing == [], (
        f"dockerbuild-marked e2e file(s) not --ignored in the '{E2E_TESTS_JOB}' lane: "
        f"{missing} — add an --ignore AND a dedicated job for each"
    )


def test_every_dockerbuild_marked_file_is_ignored__mutation_drops_one_ignore() -> None:
    """Removing a single marked file's ignore line must fail the guard."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], LIFECYCLE_TEST_FILE)
    assert _run_step_ignores_all(mutated, [DEMO_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, _dockerbuild_marked_e2e_files()) == [LIFECYCLE_TEST_FILE]


def test_every_dockerbuild_marked_file_is_ignored__mutation_new_marked_file() -> None:
    """A future dockerbuild-marked file with no ignore entry must be reported
    missing — the exact 'leaked into the shared lane' shape this guard exists
    to catch before it costs a 50-minute red run."""
    workflow = _load_workflow()
    phantom = "tests/e2e/test_future_dockerbuild_stack.py"
    assert _run_step_ignores_all(workflow, [*_dockerbuild_marked_e2e_files(), phantom]) == [phantom]


# ---------------------------------------------------------------------------
# (f) e2e-lane slimming: orm-roundtrip-e2e + dispatch-overlay-e2e extractions,
# the nightly channel-finder benchmarks, and the no-advisory-tier gate
# ---------------------------------------------------------------------------


def test_orm_roundtrip_job_exists(workflow: dict[str, Any]) -> None:
    assert ORM_JOB in _jobs(workflow)


def test_orm_roundtrip_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][ORM_JOB]
    with pytest.raises(AssertionError):
        assert ORM_JOB in _jobs(mutated)


def test_orm_roundtrip_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    """The ORM roundtrip drives the bridge HTTP API directly — an LLM secret
    appearing in its job would mean the lane's scope silently grew."""
    assert not _job_declares_secret(workflow, ORM_JOB, SECRET_TOKEN)


def test_orm_roundtrip_job_has_no_llm_secret__mutation_adds_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ORM_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        assert not _job_declares_secret(mutated, ORM_JOB, SECRET_TOKEN)


def test_dispatch_overlay_job_exists(workflow: dict[str, Any]) -> None:
    assert OVERLAY_JOB in _jobs(workflow)


def test_dispatch_overlay_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][OVERLAY_JOB]
    with pytest.raises(AssertionError):
        assert OVERLAY_JOB in _jobs(mutated)


def test_dispatch_overlay_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Inverse of the secret-free checks: the overlay test runs a REAL agent
    turn, and its fixture skips outright without ALS_APG_API_KEY — a job
    missing the secret would green-wash the lane by skipping its only test."""
    assert _job_declares_secret(workflow, OVERLAY_JOB, SECRET_TOKEN)


def test_dispatch_overlay_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][OVERLAY_JOB] = json.loads(
        json.dumps(mutated["jobs"][OVERLAY_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, OVERLAY_JOB, SECRET_TOKEN)


def test_benchmarks_job_is_dispatch_gated_and_lane_ignores_it(workflow: dict[str, Any]) -> None:
    """The channel-finder benchmarks are a statistical quality score, not a
    per-PR correctness gate: they must be ignored by the shared e2e-tests
    lane AND still exist as a manually-dispatched job behind the
    ``run_benchmarks`` input (otherwise the --ignore silently deletes the
    only benchmark signal). The e2e-tests lane must honor the same input in
    the opposite direction, so the benchmark button doesn't also burn a full
    ~19-min LLM lane run."""
    assert _run_step_ignores_all(workflow, [BENCHMARKS_TEST_FILE]) == []
    job_if = _jobs(workflow)[BENCHMARKS_JOB]["if"]
    assert "workflow_dispatch" in job_if and "run_benchmarks" in job_if, (
        f"'{BENCHMARKS_JOB}' must gate on the run_benchmarks dispatch input; got: {job_if!r}"
    )
    lane_if = _jobs(workflow)[E2E_TESTS_JOB]["if"]
    assert "run_benchmarks" in lane_if, (
        f"'{E2E_TESTS_JOB}' must exclude run_benchmarks dispatches; got: {lane_if!r}"
    )


def test_benchmarks_job_is_dispatch_gated__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], BENCHMARKS_TEST_FILE)
    assert _run_step_ignores_all(mutated, [BENCHMARKS_TEST_FILE]) == [BENCHMARKS_TEST_FILE]


def _gate_run_text(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, GATE_JOB, "Check all jobs status")["run"]


def test_gate_checks_every_needed_job(workflow: dict[str, Any]) -> None:
    """Completeness: every job listed in the gate's ``needs`` must have its
    ``needs.<job>.result`` examined by the gate script. A needs entry the
    script never reads is decorative — the job could go red forever inside a
    green check (the exact shape the old advisory tier had)."""
    run_text = _gate_run_text(workflow)
    unchecked = [
        job for job in _jobs(workflow)[GATE_JOB]["needs"] if f"needs.{job}.result" not in run_text
    ]
    assert unchecked == [], f"gate never examines: {unchecked}"


def test_gate_checks_every_needed_job__mutation_adds_unchecked_need() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].append("phantom-lane")
    run_text = _gate_run_text(mutated)
    unchecked = [
        job for job in _jobs(mutated)[GATE_JOB]["needs"] if f"needs.{job}.result" not in run_text
    ]
    assert unchecked == ["phantom-lane"]


def test_gate_has_no_advisory_tier(workflow: dict[str, Any]) -> None:
    """No lane result may be waved through as 'non-blocking': every checked
    lane either passes, is legitimately skipped (its ``if:`` didn't match the
    event), or fails the gate. The literal advisory marker phrase from the
    old gate must never reappear."""
    run_text = _gate_run_text(workflow)
    assert "non-blocking" not in run_text
    assert "exit 1" in run_text


def _gating_e2e_jobs(wf: dict[str, Any]) -> list[str]:
    needs = _jobs(wf)[GATE_JOB]["needs"]
    return [j for j in (ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB) if j in needs]


def test_all_checks_passed_needs_promoted_and_new_lanes(workflow: dict[str, Any]) -> None:
    """The two extracted lanes AND the two previously-advisory bluesky lanes
    must all gate the merge. Deliberately `all`, not `any` — the same
    silent-partial-fix guard shape as ``_needs_contains_both_new_jobs``."""
    assert _gating_e2e_jobs(workflow) == [ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB]


def test_all_checks_passed_needs_promoted_lanes__mutation_drops_sandbox() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(SANDBOX_JOB)
    assert CATALOG_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _gating_e2e_jobs(mutated) == [ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB]


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
