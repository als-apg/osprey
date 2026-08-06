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

One check reads a source file rather than the parsed workflow: the unit
lane's ``--dist loadgroup`` flag is only safe because ``tests/conftest.py``
overrides ``pytest_xdist_make_scheduler``, and that override lives in
Python, not in YAML. Its *behaviour* is covered by
``tests/infrastructure/test_xdist_scheduler.py``; what is pinned here is the
pairing — the flag in ci.yml must never outlive the scheduler behind it.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
import sys
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
NEXTCLOUD_JOB = "nextcloud-talk-bridge-e2e"
NEXTCLOUD_TEST_FILE = "tests/e2e/test_nextcloud_talk_bridge_e2e.py"
NEXTCLOUD_DOCKERFILE = "tests/e2e/fixtures/Dockerfile.nextcloud_talk"

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
PARALLEL_FLAGS = ("-n 4", "--dist loadgroup")
SCHEDULER_HOOK = "pytest_xdist_make_scheduler"
SCHEDULER_CLASS = "FileOrGroupScheduling"


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


def test_lume_pyat_is_a_core_dependency() -> None:
    """`model.pyat` imports lume_pyat at module level — PyATRingModel subclasses
    LUMEPyATModel — so the VA cannot boot without it. Placement, not just
    presence, is what this guards: the dev-wheel channel builds the VA image's
    dependency manifest from the wheel's base `Requires-Dist` only
    (`wheel_build._wheel_base_requirements` drops every entry gated behind an
    extra). A pin parked in an optional extra would therefore vanish from
    `osprey-local-requirements.txt` silently, and the image would build green
    and fail at runtime on the first model import."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    core_deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("lume-pyat") for dep in core_deps), (
        "lume-pyat must be a core dependency, not an extra"
    )


# ---------------------------------------------------------------------------
# The unit-test job runs in parallel, and the scheduler that makes it safe
# still exists
# ---------------------------------------------------------------------------


def _unit_test_pytest_line(wf: dict[str, Any]) -> str:
    """The single ``pytest tests/`` invocation from the unit-test job's run
    block, isolated from the surrounding ``COV_ARGS`` shell conditional.

    Everything from an unquoted ``#`` onward is dropped first, so a flag named
    in a shell comment — whole-line or *trailing*, as in
    ``... $COV_ARGS  # TODO restore -n 4 --dist loadgroup`` — cannot satisfy
    the pin while the lane actually runs serial. Truncating at a ``#`` inside a
    quoted argument would only ever make the pin stricter, so the naive split
    is the safe direction to be wrong in.
    """
    run_text = _find_named_step(wf, UNIT_TEST_JOB, "Run unit tests")["run"]
    commands = [line.split("#", 1)[0] for line in run_text.splitlines()]
    lines = [cmd for cmd in commands if "pytest tests/" in cmd]
    assert len(lines) == 1, f"expected exactly one pytest invocation; got: {lines}"
    return lines[0]


def _missing_parallel_flags(cmd: str) -> list[str]:
    """Which of the parallel flags are absent (empty = fully wired). Both are
    load-bearing and checked independently: `-n 4` without `--dist loadgroup`
    falls back to xdist's default per-test `load` scheduling, which scatters a
    file's tests across workers and reinstates the module-state bleed."""
    return [flag for flag in PARALLEL_FLAGS if flag not in cmd]


def test_unit_test_job_runs_xdist_parallel(workflow: dict[str, Any]) -> None:
    """The unit lane was serial for as long as the suite had cross-test
    global-state bleed. That debt is paid, so the lane must actually collect
    the win — a silent revert to serial would cost ~3x wall clock on every
    push without failing anything."""
    line = _unit_test_pytest_line(workflow)
    missing = _missing_parallel_flags(line)
    assert missing == [], (
        f"the '{UNIT_TEST_JOB}' job in .github/workflows/ci.yml must invoke pytest with "
        f"{' '.join(PARALLEL_FLAGS)} — missing {missing} in: {line.strip()!r}. "
        f"Flags are matched as literal substrings (exact spelling, single spaces) on the "
        f"one `pytest tests/` line, so reflowing the command across backslash-continued "
        f"lines (the e2e lane's style) means updating _unit_test_pytest_line too."
    )


def test_unit_test_job_runs_xdist_parallel__mutation_flags_only_in_trailing_comment() -> None:
    """The hole a reviewer probe found: flags parked in a TRAILING shell
    comment while the lane runs serial. `# TODO restore ...` is exactly how a
    temporary revert gets written, so it must fail the pin, not satisfy it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(
        " -n 4 --dist loadgroup $COV_ARGS",
        " $COV_ARGS  # TODO restore -n 4 --dist loadgroup",
    )
    assert step["run"] != original, "parallel invocation not found — mutation is stale"
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == list(PARALLEL_FLAGS)


def test_unit_test_job_runs_xdist_parallel__mutation_drops_worker_count() -> None:
    """Dropping `-n 4` (leaving `--dist loadgroup`) must be reported: without
    a worker count xdist does not parallelize at all and the dist mode is
    inert."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    step["run"] = step["run"].replace(" -n 4", "")
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == ["-n 4"]


def test_unit_test_job_runs_xdist_parallel__mutation_drops_dist_mode() -> None:
    """Dropping `--dist loadgroup` (leaving `-n 4`) must be reported too — the
    dangerous half of the revert, since the lane still looks parallelized
    while the scheduler override goes inert."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    step["run"] = step["run"].replace(" --dist loadgroup", "")
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == ["--dist loadgroup"]


def _conftest_source() -> str:
    return CONFTEST.read_text(encoding="utf-8")


def _hook_source(source: str) -> str:
    """Source of the top-level ``pytest_xdist_make_scheduler`` definition, or
    ``""`` if it is not defined.

    Parsed with ``ast`` rather than grepped because the explanatory comment
    above the hook also spells "loadgroup": a whole-file substring search
    would stay green after the guard itself was deleted.
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == SCHEDULER_HOOK:
            return ast.get_source_segment(source, node) or ""
    return ""


def _missing_scheduler_parts(source: str) -> list[str]:
    """Which parts of the scheduler override are absent (empty = all present)."""
    hook = _hook_source(source)
    present = {
        f"def {SCHEDULER_HOOK}": bool(hook),
        f"class {SCHEDULER_CLASS}": f"class {SCHEDULER_CLASS}(" in source,
        "loadgroup guard": 'getoption("dist"' in hook and "loadgroup" in hook,
        f"returns {SCHEDULER_CLASS}": f"{SCHEDULER_CLASS}(config, log)" in hook,
    }
    return [label for label, ok in present.items() if not ok]


def test_conftest_defines_the_loadgroup_scheduler_override() -> None:
    """Source-level counterpart to the ci.yml pin above. `--dist loadgroup` is
    only safe for this suite because ``tests/conftest.py`` replaces stock
    ``LoadGroupScheduling`` — which gives every unmarked nodeid its own scope
    — with one that falls back to file scope. Deleting the override while
    leaving the CI flag in place would parallelize the lane per-test again,
    silently, and the failures would land as intermittent reds in unrelated
    PRs."""
    assert _missing_scheduler_parts(_conftest_source()) == []


def test_conftest_scheduler_override__mutation_renames_hook() -> None:
    """A hook pytest never calls is the same as no hook at all: renaming the
    definition must be reported, not papered over by the class still being
    defined below it."""
    source = _conftest_source()
    mutated = source.replace(f"def {SCHEDULER_HOOK}(", f"def _disabled_{SCHEDULER_HOOK}(")
    assert mutated != source, f"no `def {SCHEDULER_HOOK}(` in {CONFTEST} — mutation is stale"
    assert _missing_scheduler_parts(mutated) == [
        f"def {SCHEDULER_HOOK}",
        "loadgroup guard",
        f"returns {SCHEDULER_CLASS}",
    ]


def test_conftest_scheduler_override__mutation_drops_loadgroup_guard() -> None:
    """The guard returns None for every dist mode but ``loadgroup``, which is
    what keeps the e2e lane's ``--dist loadfile`` (and ad-hoc ``load`` /
    ``worksteal`` runs) on stock xdist. Deleting it would silently widen the
    override's reach beyond the lane it was written for."""
    source = _conftest_source()
    mutated = "\n".join(line for line in source.splitlines() if 'getoption("dist"' not in line)
    assert mutated != source, f"no dist-option guard in {CONFTEST} — mutation is stale"
    assert _missing_scheduler_parts(mutated) == ["loadgroup guard"]


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
# (h) nextcloud-talk-bridge-e2e lane
# ---------------------------------------------------------------------------


def test_nextcloud_talk_bridge_job_exists(workflow: dict[str, Any]) -> None:
    assert NEXTCLOUD_JOB in _jobs(workflow)


def test_nextcloud_talk_bridge_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][NEXTCLOUD_JOB]
    with pytest.raises(AssertionError):
        assert NEXTCLOUD_JOB in _jobs(mutated)


def test_e2e_lane_ignores_nextcloud_talk_bridge(workflow: dict[str, Any]) -> None:
    """The lane runs a real Nextcloud container plus a dispatcher/worker pair;
    swept into the shared e2e-tests lane it would double-execute against its
    own fixed host port and exact-named container."""
    assert _run_step_ignores_all(workflow, [NEXTCLOUD_TEST_FILE]) == []


def test_e2e_lane_ignores_nextcloud_talk_bridge__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], NEXTCLOUD_TEST_FILE)
    assert _run_step_ignores_all(mutated, [BENCHMARKS_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, [NEXTCLOUD_TEST_FILE]) == [NEXTCLOUD_TEST_FILE]


def test_all_checks_passed_needs_nextcloud_talk_bridge(workflow: dict[str, Any]) -> None:
    assert NEXTCLOUD_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_all_checks_passed_needs_nextcloud_talk_bridge__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(NEXTCLOUD_JOB)
    with pytest.raises(AssertionError):
        assert NEXTCLOUD_JOB in _jobs(mutated)[GATE_JOB]["needs"]


def test_gate_checks_nextcloud_talk_bridge_result(workflow: dict[str, Any]) -> None:
    """A ``needs`` entry the gate script never reads is decorative — the
    generic completeness guard above catches that, but this lane gets its own
    named check so a partial removal names the right lane in the failure."""
    assert f"needs.{NEXTCLOUD_JOB}.result" in _gate_run_text(workflow)


def test_gate_checks_nextcloud_talk_bridge_result__mutation_drops_check_line() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if NEXTCLOUD_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    with pytest.raises(AssertionError):
        assert f"needs.{NEXTCLOUD_JOB}.result" in _gate_run_text(mutated)


# ---------------------------------------------------------------------------
# (i) Nextcloud fixture image-pin drift guard
# ---------------------------------------------------------------------------
#
# .github/workflows/build-nextcloud-fixture.yml already cross-checks the four
# values that live INSIDE the Dockerfile and refuses to publish on any
# disagreement. What no workflow can see is the fifth value: the ghcr tag the
# e2e module asks for at run time. That constant is what binds the published
# image to the pins it claims to carry, so a bump that moves `FROM`/`ARG`
# without moving the tag (or the reverse) yields an image whose name lies about
# its contents. The comparison below is therefore always against the
# authoritative `FROM`/`ARG`, never against the `# ..._PIN=` comments — a guard
# that validated a comment against a comment would constrain nothing.

_FROM_RE = re.compile(r"^FROM\s+nextcloud:(\S+)-apache\s*$", re.MULTILINE)
_ARG_RE = re.compile(r"^ARG\s+SPREED_VERSION=(\S+)\s*$", re.MULTILINE)
_NEXTCLOUD_COMMENT_RE = re.compile(r"^#\s*NEXTCLOUD_PIN=(\S+)\s*$", re.MULTILINE)
_SPREED_COMMENT_RE = re.compile(r"^#\s*SPREED_PIN=(\S+)\s*$", re.MULTILINE)


def _dockerfile_text() -> str:
    return (CI_YML.parents[2] / NEXTCLOUD_DOCKERFILE).read_text(encoding="utf-8")


def _sole_match(pattern: re.Pattern[str], text: str, label: str) -> str:
    found = pattern.findall(text)
    assert len(found) == 1, f"expected exactly one {label} in {NEXTCLOUD_DOCKERFILE}; got {found}"
    return found[0]


def _e2e_fixture_image() -> str:
    """The image reference constant from the e2e module, imported not regexed.

    The module imports cleanly with no container runtime present: its only
    module-level runtime touch is ``shutil.which`` inside a ``skipif`` marker,
    and the daemon/image checks are ``pytest.skip`` calls inside a fixture
    body. So the real object is available, and a source parse would only be a
    second, weaker spelling of the same fact. Loaded under a private module
    name so it can never collide with pytest's own collection of that file.
    """
    spec = importlib.util.spec_from_file_location(
        "_nextcloud_e2e_pin_probe", CI_YML.parents[2] / NEXTCLOUD_TEST_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises AttributeError without it.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return str(module.NEXTCLOUD_FIXTURE_IMAGE)
    finally:
        sys.modules.pop(spec.name, None)


def _assert_fixture_pins_agree(dockerfile_text: str, image_reference: str) -> None:
    """The whole pin contract in one place, parameterized on its two inputs so
    a mutation can feed it a drifted copy of either side."""
    nextcloud = _sole_match(_FROM_RE, dockerfile_text, "FROM nextcloud:<ver>-apache")
    spreed = _sole_match(_ARG_RE, dockerfile_text, "ARG SPREED_VERSION=<ver>")
    nextcloud_comment = _sole_match(_NEXTCLOUD_COMMENT_RE, dockerfile_text, "# NEXTCLOUD_PIN=")
    spreed_comment = _sole_match(_SPREED_COMMENT_RE, dockerfile_text, "# SPREED_PIN=")

    assert nextcloud == nextcloud_comment, (
        f"Dockerfile FROM pins nextcloud {nextcloud} but the NEXTCLOUD_PIN comment "
        f"says {nextcloud_comment}"
    )
    assert spreed == spreed_comment, (
        f"Dockerfile ARG pins spreed {spreed} but the SPREED_PIN comment says {spreed_comment}"
    )

    expected_tag = f"{nextcloud}-spreed{spreed}"
    actual_tag = image_reference.rsplit(":", 1)[-1]
    assert actual_tag == expected_tag, (
        f"{NEXTCLOUD_TEST_FILE}'s NEXTCLOUD_FIXTURE_IMAGE is tagged {actual_tag!r}, but the "
        f"authoritative FROM/ARG in {NEXTCLOUD_DOCKERFILE} build {expected_tag!r} — the "
        f"published image would advertise contents it does not have"
    )


def test_nextcloud_fixture_pins_agree_across_dockerfile_and_e2e_module() -> None:
    _assert_fixture_pins_agree(_dockerfile_text(), _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_bumps_from_without_the_tag() -> None:
    """The real drift shape: someone bumps the base image (and dutifully the
    NEXTCLOUD_PIN comment with it) but leaves the module's ghcr tag behind."""
    text = _dockerfile_text()
    mutated = text.replace("FROM nextcloud:33.0.7-apache", "FROM nextcloud:33.0.8-apache").replace(
        "NEXTCLOUD_PIN=33.0.7", "NEXTCLOUD_PIN=33.0.8"
    )
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_bumps_spreed_arg_without_the_tag() -> None:
    """Same failure from the other authoritative value, proving both halves of
    the derived tag are independently load-bearing."""
    text = _dockerfile_text()
    mutated = text.replace("ARG SPREED_VERSION=23.0.9", "ARG SPREED_VERSION=23.1.0").replace(
        "SPREED_PIN=23.0.9", "SPREED_PIN=23.1.0"
    )
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_comment_drifts_from_authoritative_from() -> None:
    """Bumping only the comment must fail on the comment/authoritative check —
    and must NOT be waved through by the tag check, which is deliberately
    derived from FROM/ARG and would still agree here."""
    text = _dockerfile_text()
    mutated = text.replace("NEXTCLOUD_PIN=33.0.7", "NEXTCLOUD_PIN=33.0.8")
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="NEXTCLOUD_PIN comment"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_tag_drifts_from_dockerfile() -> None:
    """And the inverse: the module's tag moves while the Dockerfile stands
    still. Feeding a drifted reference through the same helper proves the tag
    comparison, not just the comment comparison, is doing work."""
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(
            _dockerfile_text(), "ghcr.io/als-apg/nextcloud-talk-fixture:33.0.7-spreed23.1.0"
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
