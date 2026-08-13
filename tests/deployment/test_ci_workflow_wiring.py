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
from packaging.requirements import Requirement

CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

TILED_JOB = "tiled-roundtrip-e2e"
VA_JOB = "va-substrate-equivalence-e2e"
UNIT_TEST_JOB = "test"
E2E_TESTS_JOB = "e2e-tests"
GATE_JOB = "all-checks-passed"
SECRET_TOKEN = "secrets.ALS_APG_API_KEY"
TILED_TEST_FILE = "tests/e2e/test_tiled_roundtrip.py"
VA_TEST_FILE = "tests/e2e/test_va_substrate_equivalence.py"
LIFECYCLE_TEST_FILE = "tests/e2e/test_deploy_lifecycle.py"
ORM_JOB = "orm-roundtrip-e2e"
ORM_TEST_FILE = "tests/e2e/test_orm_roundtrip.py"
GRID_TEST_FILE = "tests/e2e/test_grid_scan_roundtrip.py"
QUEUE_JOB = "bluesky-queue-e2e"
QUEUE_TEST_FILE = "tests/e2e/test_bluesky_queue_e2e.py"
SCAN_AGENTIC_JOB = "scan-agentic-e2e"
SCAN_AGENTIC_TEST_FILE = "tests/e2e/test_scan_stack_agentic.py"
SCAN_AGENTIC_SKIP_GATE_STEP = "Fail the lane on any skipped test"
ARCHIVER_JOB = "archiver-world-e2e"
ARCHIVER_TEST_FILE = "tests/e2e/test_archiver_world_e2e.py"
OVERLAY_JOB = "dispatch-overlay-e2e"
OVERLAY_TEST_FILE = "tests/e2e/test_dispatch_overlay_visibility.py"
CATALOG_JOB = "bluesky-catalog-e2e"
SANDBOX_JOB = "bluesky-sandbox-escape-e2e"
BENCHMARKS_JOB = "channel-finder-benchmarks"
BENCHMARKS_TEST_FILE = "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py"
FLOOR_JOB = "dependency-floor"
NEXTCLOUD_JOB = "nextcloud-talk-bridge-e2e"
NEXTCLOUD_TEST_FILE = "tests/e2e/test_nextcloud_talk_bridge_e2e.py"
NEXTCLOUD_DOCKERFILE = "tests/e2e/fixtures/Dockerfile.nextcloud_talk"
GCHAT_JOB = "gchat-bridge-e2e"
GCHAT_TEST_FILE = "tests/e2e/test_gchat_bridge_e2e.py"
GCHAT_SMOKE_FILE = "tests/e2e/fixtures/test_gchat_fixture_smoke.py"
GCHAT_EXTRA = "gchat"
GCHAT_SKIP_GATE_STEP = "Fail the lane on any skipped test"
PUBSUB_FIXTURE_NAME = "pubsub_emulator"

ALS_APG_BASE_URL_ENV = "ALS_APG_BASE_URL"
PROBE_BASE_VAR = "ALS_APG_PROBE_BASE"
PROBE_KEY_ENV = "ALS_APG_API_KEY"
PROBE_TARGET_PATH = "/v1/messages"
# `ALS_APG_PROBE_BASE="${ALS_APG_BASE_URL:-https://…}"` — the `:-default` is optional.
PROBE_BASE_ASSIGNMENT = re.compile(rf'{PROBE_BASE_VAR}="\$\{{{ALS_APG_BASE_URL_ENV}(:-[^}}]*)?\}}"')
# Guards against silent under-discovery: if probe steps are renamed or reshaped
# so the finder stops matching them, the count check fails loudly instead of
# passing over an empty list. Raise this when probe lanes are added.
EXPECTED_MIN_ALS_APG_PROBES = 8

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


@pytest.fixture()
def pyproject() -> dict[str, Any]:
    return _load_pyproject()


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
    """The `virtual-accelerator` extra carries the serving stack the live
    tests/va suites need — pcaspy, and `lume-pva-apg[ca,pva]` which brings p4p
    with it. Those suites guard their imports, so without the extra they SKIP
    rather than error and the lane reports green having never touched Channel
    Access; `dev` carries pytest itself."""
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
# The serving package is pinned in the `virtual-accelerator` extra, with both
# transports and the platform marker its Channel Access half needs
# ---------------------------------------------------------------------------
#
# The placement is the opposite of lume-pyat's above, and deliberately so.
# lume-pyat has to be core because the VA cannot boot without it on any host;
# the serving stack can only be installed where pcaspy loads, so it belongs in
# the extra that already carries that constraint. Both Dockerfiles install
# `[virtual-accelerator]` rather than naming the package, so this extra is the
# only place either image can learn the pin — which is what these guards
# protect.


#: The one platform with a loadable Channel Access server wheel, and therefore
#: the marker both serving entries must carry. Spelled with DOUBLE quotes
#: because this is compared against ``str(Requirement(...).marker)``, which is
#: packaging's normalised rendering — pyproject.toml itself writes the same
#: marker with single quotes, and both parse to this.
LIVE_CA_MARKER = 'sys_platform == "linux" and platform_machine == "x86_64"'


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())


def _va_extra(pyproject: dict[str, Any]) -> list[str]:
    return pyproject["project"]["optional-dependencies"]["virtual-accelerator"]


def _requirement(deps: list[str], name: str) -> Requirement | None:
    """The single requirement in ``deps`` naming ``name``, or None.

    Parsed rather than string-matched: every property asserted below (extras,
    specifier, marker) lives in a different part of the grammar, and a
    ``startswith`` check would be satisfied by a line that carries none of
    them.
    """
    matches = [req for dep in deps if (req := Requirement(dep)).name == name]
    assert len(matches) <= 1, f"{name} declared more than once: {matches}"
    return matches[0] if matches else None


def test_lume_pva_is_pinned_in_the_va_extra(pyproject: dict[str, Any]) -> None:
    """The extra is what puts the serving stack in both VA images: neither
    Dockerfile names lume-pva-apg, so an entry that is not here reaches no
    image at all. Core placement would be worse than absent — the `ca` extra
    requires pcaspy unconditionally, so a core pin would drag the Channel
    Access server onto every plain `uv sync`, including the macOS hosts the
    pcaspy marker beside it exists to keep clear of a source build."""
    assert _requirement(_va_extra(pyproject), "lume-pva-apg") is not None, (
        "lume-pva-apg must be pinned in the `virtual-accelerator` extra"
    )
    assert _requirement(pyproject["project"]["dependencies"], "lume-pva-apg") is None, (
        "lume-pva-apg must not be a core dependency — see the docstring"
    )


def test_lume_pva_is_pinned_in_the_va_extra__mutation_drops_the_pin() -> None:
    """Dropping the entry must fail — otherwise the guard is decorative."""
    mutated = _load_pyproject()
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep for dep in _va_extra(mutated) if Requirement(dep).name != "lume-pva-apg"
    ]
    with pytest.raises(AssertionError):
        test_lume_pva_is_pinned_in_the_va_extra(mutated)


def test_lume_pva_is_pinned_in_the_va_extra__mutation_also_declares_it_in_core() -> None:
    """A core declaration must fail even with the extra entry left intact.

    The extra is deliberately NOT disturbed here. Removing it as well would
    trip the presence half of the assertion and the mutation would pass for a
    reason that has nothing to do with placement — leaving the core-exclusion
    half never exercised. Copying instead of moving is what aims this mutation
    at the assertion it is named for."""
    mutated = _load_pyproject()
    copied = [dep for dep in _va_extra(mutated) if Requirement(dep).name == "lume-pva-apg"]
    assert copied, "no lume-pva-apg in the extra — mutation is stale"
    mutated["project"]["dependencies"] = [*mutated["project"]["dependencies"], *copied]
    with pytest.raises(AssertionError, match="must not be a core dependency"):
        test_lume_pva_is_pinned_in_the_va_extra(mutated)


def test_lume_pva_pin_carries_both_transports(pyproject: dict[str, Any]) -> None:
    """`serving/runner.py` imports pcaspy, lume_pva_apg AND p4p at module
    scope, and the base package pulls neither transport — pcaspy arrives only
    through the `ca` extra and p4p only through `pva`. Either one missing is
    not a degraded serve, it is an ImportError at runner import, long after a
    green image build."""
    req = _requirement(_va_extra(pyproject), "lume-pva-apg")
    assert req is not None and req.extras == {"ca", "pva"}, (
        f"lume-pva-apg must carry both the `ca` and `pva` extras; got {req}"
    )


@pytest.mark.parametrize("dropped", ["ca", "pva"])
def test_lume_pva_pin_carries_both_transports__mutation_drops_one(dropped: str) -> None:
    """Dropping either transport must fail — a guard that only noticed the
    loss of both would let the p4p-typed value layer go missing silently."""
    kept = "pva" if dropped == "ca" else "ca"
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.replace("[ca,pva]", f"[{kept}]") if Requirement(dep).name == "lume-pva-apg" else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the extras spelling moved"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_carries_both_transports(mutated)


def test_lume_pva_pin_shares_the_pcaspy_platform_marker(pyproject: dict[str, Any]) -> None:
    """The two entries must be marked identically or the marker on pcaspy is
    decorative: `lume-pva-apg[ca]` requires pcaspy with no marker of its own,
    so an unmarked serving pin re-introduces it on exactly the hosts the
    pcaspy line excludes — verified, not assumed (resolving
    `lume-pva-apg[ca,pva]` for macOS at 3.13 selects pcaspy 0.8.1, whose only
    artifact there is the sdist, i.e. the EPICS source build)."""
    extra = _va_extra(pyproject)
    pcaspy = _requirement(extra, "pcaspy")
    lume_pva = _requirement(extra, "lume-pva-apg")
    # Anchored absolutely, not only to each other. Parity alone would be
    # satisfied by two entries edited together to some other platform, which
    # is the one way this pair can drift and still agree.
    assert pcaspy is not None and str(pcaspy.marker) == LIVE_CA_MARKER, (
        f"pcaspy must stay marked {LIVE_CA_MARKER!r} — the one platform with a "
        f"loadable Channel Access server wheel; got {pcaspy.marker}"
    )
    assert lume_pva is not None and str(lume_pva.marker) == str(pcaspy.marker), (
        f"lume-pva-apg must carry pcaspy's marker ({pcaspy.marker}); got {lume_pva.marker}"
    )


def test_lume_pva_pin_shares_the_pcaspy_platform_marker__mutation_unmarks_it() -> None:
    """An unmarked serving pin must fail."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.split(";", 1)[0].strip() if Requirement(dep).name == "lume-pva-apg" else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the marker is already absent"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_shares_the_pcaspy_platform_marker(mutated)


def test_lume_pva_pin_shares_the_pcaspy_platform_marker__mutation_moves_both_to_another_platform() -> (
    None
):
    """Retargeting BOTH entries together must fail.

    This is the mutation the parity check alone could not catch: two markers
    edited in step still agree with each other, so only the absolute anchor
    rejects them. Darwin is the pointed choice — it is where pcaspy's wheels
    exist but do not load, so a plausible edit could land here."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        f"{dep.split(';', 1)[0].strip()}; sys_platform == 'darwin'" for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the markers are already not the pinned pair"
    )
    with pytest.raises(AssertionError, match="pcaspy must stay marked"):
        test_lume_pva_pin_shares_the_pcaspy_platform_marker(mutated)


def test_lume_pva_pin_is_exact(pyproject: dict[str, Any]) -> None:
    """Exact-pinned for the same reason as lume-base and lume-pyat: the runner
    subclass is written against one contract of a package whose extension
    seams are not yet a stable API, and a floor would let a resolve pick up a
    surface it was never written against. The version itself is not asserted —
    only that a bump is a deliberate edit rather than a resolver's choice."""
    req = _requirement(_va_extra(pyproject), "lume-pva-apg")
    assert req is not None
    assert [spec.operator for spec in req.specifier] == ["=="], (
        f"lume-pva-apg must be exact-pinned; got {req.specifier}"
    )


def test_lume_pva_pin_is_exact__mutation_relaxes_to_a_floor() -> None:
    """A `>=` floor must fail."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.replace("==", ">=") if Requirement(dep).name == "lume-pva-apg" else dep for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the pin is already not exact"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_is_exact(mutated)


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
# (e) the dockerbuild --ignore guard
# ---------------------------------------------------------------------------


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
    assert _run_step_ignores_all(mutated, [ORM_TEST_FILE]) == []  # others survive
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


def test_orm_roundtrip_job_runs_the_grid_scan_module(workflow: dict[str, Any]) -> None:
    """The grid-scan roundtrip's ``dockerbuild`` marker takes it OUT of the
    shared lane; this job is where it went. The blanket marker->--ignore guard
    proves it left, and nothing else proves it arrived — delete the ``Run grid
    scan roundtrip`` step and every other wiring test still passes while the
    module runs nowhere at all."""
    assert GRID_TEST_FILE in json.dumps(_jobs(workflow)[ORM_JOB]["steps"])


def test_orm_roundtrip_job_runs_the_grid_scan_module__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[ORM_JOB]["steps"]
    kept = [step for step in steps if GRID_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the grid module"
    _jobs(mutated)[ORM_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_orm_roundtrip_job_runs_the_grid_scan_module(mutated)


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
# dependency-floor lane: the declared pandas/numpy minimums are actually run
# ---------------------------------------------------------------------------


def test_dependency_floor_job_exists(workflow: dict[str, Any]) -> None:
    assert FLOOR_JOB in _jobs(workflow)


def test_dependency_floor_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][FLOOR_JOB]
    with pytest.raises(AssertionError):
        assert FLOOR_JOB in _jobs(mutated)


def test_dependency_floor_job_pins_the_versions_pyproject_declares() -> None:
    """The lane certifies nothing if its pin drifts from the declared floor."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    declared = {
        dep.split(">=")[0]: dep.split(">=")[1]
        for dep in pyproject["project"]["dependencies"]
        if dep.startswith(("pandas>=", "numpy>="))
    }
    assert declared, "pyproject no longer declares pandas/numpy floors"

    job = json.dumps(_jobs(_load_workflow())[FLOOR_JOB])
    for package, floor in declared.items():
        assert f"{package}=={floor}" in job, (
            f"{FLOOR_JOB} does not pin {package}=={floor}; pyproject declares >={floor}"
        )


def test_dependency_floor_job_keeps_the_pin_when_running_tests() -> None:
    """A bare ``uv run`` re-syncs and silently restores the newest pandas,
    so every ``uv run`` in the job must carry ``--no-sync``.
    """
    steps = _jobs(_load_workflow())[FLOOR_JOB]["steps"]
    uv_runs = [s["run"] for s in steps if "uv run" in s.get("run", "")]
    assert uv_runs, f"{FLOOR_JOB} has no `uv run` step"
    for run in uv_runs:
        assert "uv run --no-sync" in run, f"`uv run` without --no-sync would undo the pin: {run!r}"


def test_dependency_floor_gates_the_merge(workflow: dict[str, Any]) -> None:
    assert FLOOR_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_dependency_floor_gates_the_merge__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(FLOOR_JOB)
    with pytest.raises(AssertionError):
        assert FLOOR_JOB in _jobs(mutated)[GATE_JOB]["needs"]


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
# (j) gchat-bridge-e2e lane
# ---------------------------------------------------------------------------


def test_gchat_bridge_job_exists(workflow: dict[str, Any]) -> None:
    assert GCHAT_JOB in _jobs(workflow)


def test_gchat_bridge_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][GCHAT_JOB]
    with pytest.raises(AssertionError):
        assert GCHAT_JOB in _jobs(mutated)


def _emulator_container_e2e_files() -> list[str]:
    """Every ``tests/e2e/`` file that reaches the fixed-name Pub/Sub emulator
    container, found by the fixture's own name.

    The scan token is the FIXTURE name, not the module name, and deliberately
    so: ``tests/e2e/fixtures/conftest.py`` registers ``pubsub_emulator``
    directory-wide, so a file there can reach the container by taking it as a
    test argument without importing anything. ``pubsub_emulator`` is a
    substring of ``gchat_pubsub_emulator``, so this one token covers the
    import spelling and the fixture-argument spelling at once.

    Text scan rather than collection, for the same reason as the
    ``dockerbuild`` scan above: importing these files needs their heavy
    optional e2e dependencies, and both spellings are always literal.
    """
    e2e_dir = CI_YML.parents[2] / "tests" / "e2e"
    return sorted(
        p.relative_to(e2e_dir.parents[1]).as_posix()
        for p in e2e_dir.rglob("test_*.py")
        if PUBSUB_FIXTURE_NAME in p.read_text(encoding="utf-8")
    )


def test_shared_lane_ignores_every_emulator_container_file(workflow: dict[str, Any]) -> None:
    """The collision this guard exists for: ``PubSubEmulator.start()`` pre-cleans
    by ``rm -f``-ing the exact container name it is about to use, and every file
    below reaches that same name. The shared e2e lane runs ``-n 4 --dist
    loadfile``, which keeps a file's tests on one worker but runs different
    FILES concurrently — so two of these landing in that lane means one file's
    pre-clean kills the other's live emulator mid-test. Ignoring only the
    bridge module would fix today's pair and leave the hazard armed for the
    next file to use the fixture, so the guard is over the whole scan."""
    files = _emulator_container_e2e_files()
    assert set(files) >= {GCHAT_SMOKE_FILE, GCHAT_TEST_FILE}, (
        f"emulator-fixture scan found {files} — expected at least the two Google Chat "
        f"files; the fixture module was renamed or the scan broke"
    )
    missing = _run_step_ignores_all(workflow, files)
    assert missing == [], (
        f"file(s) using the fixed-name Pub/Sub emulator container not --ignored in the "
        f"'{E2E_TESTS_JOB}' lane: {missing} — they would run concurrently there and "
        f"tear down each other's emulator"
    )


def test_shared_lane_ignores_emulator_files__mutation_drops_bridge_ignore() -> None:
    """Dropping only the bridge module's ignore must fail — the smoke file
    surviving is exactly the half-fix this guard rejects."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], GCHAT_TEST_FILE)
    assert _run_step_ignores_all(mutated, [GCHAT_SMOKE_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_every_emulator_container_file(mutated)


def test_shared_lane_ignores_emulator_files__mutation_drops_smoke_ignore() -> None:
    """And the mirror image: leaving the fixture smoke test in the parallel
    lane is just as fatal, since it starts the same container by the same name."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], GCHAT_SMOKE_FILE)
    assert _run_step_ignores_all(mutated, [GCHAT_TEST_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_every_emulator_container_file(mutated)


_XDIST_WORKERS_RE = re.compile(r"(?<!\S)-n\s+\d")


def _gchat_pytest_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """The gchat job's pytest invocations, in the order Actions will run them.

    Matched on the full ``uv run pytest`` invocation rather than a bare
    ``pytest`` substring, so a step that only MENTIONS pytest — in a comment,
    or in a message naming the command a developer should run locally — is not
    mistaken for one that runs it. A false positive there would be read as an
    unguarded emulator user and fail the sequencing check for no reason.
    """
    return [s for s in _jobs(wf)[GCHAT_JOB]["steps"] if "uv run pytest " in s.get("run", "")]


def _emulator_files_per_step(wf: dict[str, Any]) -> list[list[str]]:
    """For each pytest step, which emulator-using files it selects."""
    files = _emulator_container_e2e_files()
    return [sorted(f for f in files if f in step["run"]) for step in _gchat_pytest_steps(wf)]


def test_emulator_files_run_one_per_sequential_step(workflow: dict[str, Any]) -> None:
    """Having moved both files out of the parallel lane, the job they moved
    INTO must not reintroduce the same overlap. Two properties make that
    provable rather than incidental: one file per pytest step (Actions runs
    steps strictly in order, in separate processes, so two steps can never
    overlap), and no xdist flags (which would parallelize within a step). A
    single pytest invocation naming both files would satisfy neither — it puts
    two module-scoped emulator fixtures in one process, where teardown order
    decides whether they overlap."""
    per_step = _emulator_files_per_step(workflow)
    assert [len(selected) for selected in per_step] == [1] * len(per_step), (
        f"each pytest step in '{GCHAT_JOB}' must select exactly one emulator-using file; "
        f"got {per_step}"
    )
    assert sorted(f for selected in per_step for f in selected) == (
        _emulator_container_e2e_files()
    ), f"'{GCHAT_JOB}' must run every emulator-using file exactly once; got {per_step}"
    for step in _gchat_pytest_steps(workflow):
        assert not _XDIST_WORKERS_RE.search(step["run"]), (
            f"step {step.get('name')!r} runs pytest with xdist workers; the emulator's "
            f"fixed container name cannot survive parallel workers"
        )
        assert "--dist" not in step["run"], (
            f"step {step.get('name')!r} sets an xdist dist mode; this job must stay serial"
        )


def test_emulator_files_run_one_per_step__mutation_merges_both_into_one_step() -> None:
    """The tempting simplification — one pytest call listing both files — must
    fail, because it makes non-overlap depend on fixture teardown order rather
    than on the job's structure."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _gchat_pytest_steps(mutated)
    assert len(steps) == 2, f"expected two pytest steps to merge; got {len(steps)}"
    steps[0]["run"] = steps[0]["run"].replace(
        GCHAT_SMOKE_FILE, f"{GCHAT_SMOKE_FILE} {GCHAT_TEST_FILE}"
    )
    _jobs(mutated)[GCHAT_JOB]["steps"].remove(steps[1])
    with pytest.raises(AssertionError):
        test_emulator_files_run_one_per_sequential_step(mutated)


def test_emulator_files_run_one_per_step__mutation_adds_xdist_workers() -> None:
    """Speeding the job up with ``-n 2`` reinstates the collision inside a
    single step, so the pin must reject it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _gchat_pytest_steps(mutated)[0]
    step["run"] = step["run"].replace("-v --tb=short", "-v --tb=short -n 2")
    with pytest.raises(AssertionError, match="xdist workers"):
        test_emulator_files_run_one_per_sequential_step(mutated)


def _gchat_install_cmd(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, GCHAT_JOB, "Install osprey")["run"]


def test_gchat_job_installs_the_gchat_extra(workflow: dict[str, Any]) -> None:
    """The e2e module drives a REAL ``google-cloud-pubsub`` client against the
    emulator and guards that with a module-level ``skipif``, so a job that
    syncs only ``dev`` skips the entire file and reports success — a green
    check over zero executed tests, confirmed live before this lane existed."""
    cmd = _gchat_install_cmd(workflow)
    assert f"--extra {GCHAT_EXTRA}" in cmd, (
        f"'{GCHAT_JOB}' must `uv sync --extra {GCHAT_EXTRA}`; got: {cmd}"
    )


def test_gchat_job_installs_the_gchat_extra__mutation_drops_extra() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GCHAT_JOB, "Install osprey")
    step["run"] = step["run"].replace(f" --extra {GCHAT_EXTRA}", "")
    with pytest.raises(AssertionError):
        test_gchat_job_installs_the_gchat_extra(mutated)


_JUNIT_RE = re.compile(r"--junitxml=(\S+)")


def _junit_reports_written(wf: dict[str, Any]) -> list[str]:
    return [r for step in _gchat_pytest_steps(wf) for r in _JUNIT_RE.findall(step["run"])]


def test_gchat_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """The extra above is the fix; this is the backstop that makes the whole
    class of failure loud. Every skip condition this lane has — absent
    container runtime, absent emulator image, absent ``gchat`` extra, absent
    provider key — is a CI misconfiguration that pytest reports as success, so
    the job reads the junit reports and fails on any skip (and on an empty
    collection, which would otherwise pass a zero-skip check trivially). Every
    pytest step must write a report the gate actually reads: a run whose report
    nothing inspects is back to skipping its way to green."""
    reports = _junit_reports_written(workflow)
    assert len(reports) == len(_gchat_pytest_steps(workflow)), (
        f"every pytest step in '{GCHAT_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, GCHAT_JOB, GCHAT_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{GCHAT_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_gchat_job_fails_on_any_skipped_test__mutation_drops_a_junit_report() -> None:
    """A pytest step that writes no report is invisible to the gate — the
    silent-partial-fix shape, one lane guarded and one not."""
    mutated = copy.deepcopy(_load_workflow())
    step = _gchat_pytest_steps(mutated)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_gchat_job_fails_on_any_skipped_test(mutated)


def test_gchat_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GCHAT_JOB, GCHAT_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_gchat_job_fails_on_any_skipped_test(mutated)


def test_gchat_bridge_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Same shape as the dispatch-overlay check, for the same reason: the
    agentic tier in this module runs a REAL agent turn and is marked
    ``requires_als_apg``, which the root conftest turns into a skip when the
    key is absent. The zero-skip gate would catch that, but only as "something
    skipped"; this assertion names the cause, and fails at the one place a
    reviewer would look for it."""
    assert _job_declares_secret(workflow, GCHAT_JOB, SECRET_TOKEN)


def test_gchat_bridge_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][GCHAT_JOB] = json.loads(
        json.dumps(mutated["jobs"][GCHAT_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, GCHAT_JOB, SECRET_TOKEN)


def test_all_checks_passed_needs_gchat_bridge(workflow: dict[str, Any]) -> None:
    """``needs:`` alone is not a gate — the roll-up runs ``if: always()``, so a
    needed job that failed still lets it start; the ``check_pr_lane`` line is
    what turns the result into an exit code. Both halves are pinned."""
    assert GCHAT_JOB in _jobs(workflow)[GATE_JOB]["needs"]
    assert f"needs.{GCHAT_JOB}.result" in _gate_run_text(workflow)


def test_all_checks_passed_needs_gchat_bridge__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(GCHAT_JOB)
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_gchat_bridge(mutated)


def test_all_checks_passed_needs_gchat_bridge__mutation_drops_check_pr_lane_line() -> None:
    """The dangerous half: the ``needs`` entry stays (so the gate waits for the
    job) while the line that reads its result is gone — the lane could go red
    forever inside a green check."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if GCHAT_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert GCHAT_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the needs entry survives
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_gchat_bridge(mutated)


# ---------------------------------------------------------------------------
# ALS-APG endpoint-override drift guard
# ---------------------------------------------------------------------------


def _als_apg_probe_steps(wf: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every step that preflight-probes the ALS-APG gateway, as
    ``(job, step name, run text)``.

    Discovery keys on what a probe *does* — POST an authenticated request to
    ``/v1/messages`` — not on whether it happens to define
    ``ALS_APG_PROBE_BASE``. That distinction is the whole point: a newly added
    probe that hardcodes its URL defines no such variable, so keying on the
    variable would skip exactly the step the guard exists to catch.
    """
    found: list[tuple[str, str, str]] = []
    for job_name, job in _jobs(wf).items():
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str) and PROBE_TARGET_PATH in run and PROBE_KEY_ENV in run:
                found.append((job_name, step.get("name", "<unnamed>"), run))
    return found


def test_workflow_exports_the_als_apg_base_url_override(workflow: dict[str, Any]) -> None:
    """The workflow-level ``env:`` block is what lets a single repository
    variable retarget every LLM-touching job at once. Without it the probes
    below would each fall back to the baked-in default and the override would
    silently do nothing — green locally, wrong in CI."""
    env = workflow.get("env") or {}
    assert env.get(ALS_APG_BASE_URL_ENV) == "${{ vars.ALS_APG_BASE_URL }}", (
        f"ci.yml must export {ALS_APG_BASE_URL_ENV} from the repository variable "
        f"at workflow level; found {env.get(ALS_APG_BASE_URL_ENV)!r}"
    )


def test_workflow_exports_the_als_apg_base_url_override__mutation_drops_export() -> None:
    mutated = copy.deepcopy(_load_workflow())
    (mutated.get("env") or {}).pop(ALS_APG_BASE_URL_ENV, None)
    with pytest.raises(AssertionError, match="must export"):
        test_workflow_exports_the_als_apg_base_url_override(mutated)


def test_every_als_apg_probe_honors_the_base_url_override(workflow: dict[str, Any]) -> None:
    """Each probe must derive its base from ``$ALS_APG_BASE_URL`` (a baked-in
    default after ``:-`` is fine) and must aim curl at that derived variable.

    Both halves are load-bearing and fail independently: a probe can read the
    override into ``ALS_APG_PROBE_BASE`` and then still curl a literal URL,
    which is how a hardcoded endpoint survived review once already. Such a
    lane fails against a perfectly healthy gateway, and reads as an outage."""
    probes = _als_apg_probe_steps(workflow)
    assert len(probes) >= EXPECTED_MIN_ALS_APG_PROBES, (
        f"expected at least {EXPECTED_MIN_ALS_APG_PROBES} ALS-APG probe steps in ci.yml, "
        f"found {len(probes)} — discovery has drifted and this guard is now vacuous"
    )
    for job_name, step_name, run in probes:
        where = f"{job_name} / {step_name}"
        assert PROBE_BASE_ASSIGNMENT.search(run), (
            f"{where}: ALS-APG probe must derive its base from "
            f'${{{ALS_APG_BASE_URL_ENV}}} (e.g. ALS_APG_PROBE_BASE="${{{ALS_APG_BASE_URL_ENV}'
            ':-https://default}"), so the repository variable can retarget it'
        )
        target_lines = [line for line in run.splitlines() if PROBE_TARGET_PATH in line]
        assert target_lines, f"{where}: probe target line vanished"
        for line in target_lines:
            assert f"${PROBE_BASE_VAR}" in line, (
                f"{where}: probe must curl $"
                + PROBE_BASE_VAR
                + f", not a literal URL: {line.strip()}"
            )


def test_every_als_apg_probe_honors_the_base_url_override__mutation_hardcodes_base() -> None:
    """The original bug's exact shape: the assignment stops reading the
    override and pins a URL instead."""
    mutated = copy.deepcopy(_load_workflow())
    job_name, step_name, _ = _als_apg_probe_steps(mutated)[0]
    step = _find_named_step(mutated, job_name, step_name)
    step["run"] = PROBE_BASE_ASSIGNMENT.sub(
        f'{PROBE_BASE_VAR}="https://hardcoded.example"', step["run"], count=1
    )
    with pytest.raises(AssertionError, match="must derive its base"):
        test_every_als_apg_probe_honors_the_base_url_override(mutated)


def test_every_als_apg_probe_honors_the_base_url_override__mutation_curls_literal_url() -> None:
    """The subtler half: the override is still read into the variable — so a
    reader skimming the assignment sees the right thing — while curl ignores
    it and hits a literal endpoint."""
    mutated = copy.deepcopy(_load_workflow())
    job_name, step_name, _ = _als_apg_probe_steps(mutated)[0]
    step = _find_named_step(mutated, job_name, step_name)
    step["run"] = step["run"].replace(
        f'"${PROBE_BASE_VAR}{PROBE_TARGET_PATH}"', f'"https://hardcoded.example{PROBE_TARGET_PATH}"'
    )
    assert PROBE_BASE_ASSIGNMENT.search(step["run"]), "the assignment must survive this mutation"
    with pytest.raises(AssertionError, match="not a literal URL"):
        test_every_als_apg_probe_honors_the_base_url_override(mutated)


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


# ---------------------------------------------------------------------------
# (h) multi-user login-flow browser test: registered in the lane, and the lane
# is actually triggered by the code it covers
# ---------------------------------------------------------------------------

BROWSER_JOB = "visual-theming"
BROWSER_RUN_STEP = "Run behavioral theming suite"
BROWSER_FILTER_STEP = "Detect design-system / dispatch dashboard changes"
AUTH_BROWSER_TEST_FILE = "tests/interfaces/web_terminal/test_auth_login_browser.py"
AUTH_SERVING_TEST_FILE = "tests/deployment/web_terminals/test_auth_serving.py"
AUTH_PERIMETER_PATHS = (
    "src/osprey/services/auth_sidecar/**",
    "src/osprey/deployment/web_terminals/**",
    "src/osprey/templates/modules/web_terminals/**",
    AUTH_SERVING_TEST_FILE,
)


def _browser_lane_files(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, BROWSER_JOB, BROWSER_RUN_STEP)["run"]


def _browser_lane_filter_paths(wf: dict[str, Any]) -> list[str]:
    """The `theming` filter's path list, parsed out of the step's `filters` blob.

    dorny/paths-filter takes its filters as a YAML *string*, so the outer
    safe_load leaves this as text and it has to be parsed a second time —
    still structurally, never by substring matching.
    """
    step = _find_named_step(wf, BROWSER_JOB, BROWSER_FILTER_STEP)
    return yaml.safe_load(step["with"]["filters"])["theming"]


def test_login_flow_browser_test_runs_in_the_browser_lane(workflow: dict[str, Any]) -> None:
    """The suite has to be NAMED in the lane's pytest invocation.

    The lane runs an explicit file list, not a directory glob, so a new
    browser-marked file that nobody adds here is never collected anywhere: the
    unit lane skips it (no chromium), and this lane never hears of it. It
    reports green by running nothing at all."""
    assert AUTH_BROWSER_TEST_FILE in _browser_lane_files(workflow)


def test_login_flow_browser_test_runs_in_the_browser_lane__mutation_drops_the_file() -> None:
    """Removing the file from the invocation must fail the guard — the exact
    vacuous-green shape above."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BROWSER_JOB, BROWSER_RUN_STEP)
    step["run"] = step["run"].replace(f"{AUTH_BROWSER_TEST_FILE} \\\n", "")
    assert AUTH_BROWSER_TEST_FILE not in _browser_lane_files(mutated)


def test_browser_lane_is_triggered_by_the_perimeter_it_covers(workflow: dict[str, Any]) -> None:
    """Every path the login-flow test actually exercises must arm the lane.

    The filter was written for `interfaces/` and `dispatch/`, but this suite
    drives the auth sidecar, the rendered nginx config and the shared container
    fixture — none of which live there. Left unlisted, a PR that changes the
    login page or the perimeter template skips every gated step in this job,
    which GitHub reports as a job SUCCESS. The proof would be silently gone at
    exactly the moment it mattered."""
    paths = _browser_lane_filter_paths(workflow)
    missing = [path for path in AUTH_PERIMETER_PATHS if path not in paths]
    assert missing == [], (
        f"the '{BROWSER_JOB}' lane is not triggered by {missing}, so a change there would "
        f"green-skip {AUTH_BROWSER_TEST_FILE} instead of running it"
    )


def test_browser_lane_is_triggered_by_the_perimeter__mutation_drops_the_sidecar_path() -> None:
    """Dropping the sidecar source from the filter must be reported missing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BROWSER_JOB, BROWSER_FILTER_STEP)
    filters = yaml.safe_load(step["with"]["filters"])
    filters["theming"] = [
        path for path in filters["theming"] if path != "src/osprey/services/auth_sidecar/**"
    ]
    step["with"]["filters"] = yaml.safe_dump(filters)
    paths = _browser_lane_filter_paths(mutated)
    assert [p for p in AUTH_PERIMETER_PATHS if p not in paths] == [
        "src/osprey/services/auth_sidecar/**"
    ]


# ---------------------------------------------------------------------------
# (i) bluesky-queue-e2e: the queue stack's own lane, secret-free
# ---------------------------------------------------------------------------


def test_bluesky_queue_job_exists(workflow: dict[str, Any]) -> None:
    """``test_bluesky_queue_e2e.py`` is ``--ignore``'d by the shared e2e-tests
    lane (it deploys the whole queue stack twice), so this dedicated job is the
    only place it runs at all — without it the file is collected nowhere. The
    job existing is not enough: one of its steps has to actually name the
    module, or the lane reports green having run nothing."""
    assert QUEUE_JOB in _jobs(workflow)
    assert QUEUE_TEST_FILE in json.dumps(_jobs(workflow)[QUEUE_JOB])


def test_bluesky_queue_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][QUEUE_JOB]
    with pytest.raises(AssertionError):
        test_bluesky_queue_job_exists(mutated)


def test_bluesky_queue_job_exists__mutation_drops_the_run_step() -> None:
    """The vacuous-green half: the job survives, but nothing in it names the
    e2e module any more."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[QUEUE_JOB]["steps"]
    kept = [step for step in steps if QUEUE_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the e2e module"
    _jobs(mutated)[QUEUE_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_bluesky_queue_job_exists(mutated)


def test_bluesky_queue_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    """Every stage drives the bridge HTTP API and the osprey CLI directly — an
    LLM secret appearing here would mean the lane's scope silently grew."""
    assert not _job_declares_secret(workflow, QUEUE_JOB, SECRET_TOKEN)


def test_bluesky_queue_job_has_no_llm_secret__mutation_adds_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][QUEUE_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        assert not _job_declares_secret(mutated, QUEUE_JOB, SECRET_TOKEN)


def test_e2e_lane_ignores_bluesky_queue(workflow: dict[str, Any]) -> None:
    """The guards above catch "runs nowhere"; this one catches "runs twice".

    The module carries no ``dockerbuild`` marker, so the blanket
    marker->--ignore guard does not reach it and this bespoke check is the
    only thing holding the ``--ignore`` in place. Drop that line and the whole
    queueserver stack — bridge, Redis, Tiled, VA, panels sidecar, on fixed
    host ports — boots a second time inside the shared lane's ``-n 4`` run,
    concurrently with its own job. Nothing would fail; it would just get slow
    and flaky in a way nobody could attribute."""
    assert _run_step_ignores_all(workflow, [QUEUE_TEST_FILE]) == []


def test_e2e_lane_ignores_bluesky_queue__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], QUEUE_TEST_FILE)
    assert _run_step_ignores_all(mutated, [ORM_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, [QUEUE_TEST_FILE]) == [QUEUE_TEST_FILE]


def test_all_checks_passed_needs_bluesky_queue(workflow: dict[str, Any]) -> None:
    """``needs:`` alone is not a gate — the roll-up runs ``if: always()``, so a
    needed job that failed still lets it start; the ``check_pr_lane`` line is
    what turns the result into an exit code. Both halves are pinned.

    These two assertions travel with the lane: if ``bluesky-queue-e2e`` cannot
    go green in the PR, the job, its gate entry AND this guard pair are dropped
    together and re-landed together in the follow-up. A gate entry left behind
    without its lane blocks every merge; a lane left behind without its gate
    entry is unwatched."""
    assert QUEUE_JOB in _jobs(workflow)[GATE_JOB]["needs"]
    assert f"needs.{QUEUE_JOB}.result" in _gate_run_text(workflow)


def test_all_checks_passed_needs_bluesky_queue__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(QUEUE_JOB)
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_bluesky_queue(mutated)


def test_all_checks_passed_needs_bluesky_queue__mutation_drops_check_pr_lane_line() -> None:
    """The dangerous half: the ``needs`` entry stays (so the gate waits for the
    job) while the line that reads its result is gone — the lane could go red
    forever inside a green check."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if QUEUE_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert QUEUE_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the needs entry survives
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_bluesky_queue(mutated)


# ---------------------------------------------------------------------------
# (k) scan-agentic-e2e: the agent-driven scan lane, secret-gated
# ---------------------------------------------------------------------------


def test_scan_agentic_job_exists(workflow: dict[str, Any]) -> None:
    """``test_scan_stack_agentic.py`` is ``--ignore``'d by the shared e2e-tests
    lane (its two live tests deploy the VA/bridge/queue stack on fixed host
    ports), so this dedicated job is the only place any of it runs — the
    offline floor and judge checks included, since they live in the same
    module. The job existing is not enough: one of its steps has to name the
    module, or the lane reports green having run nothing.

    Unlike the queue lane above, nothing bespoke holds that ``--ignore`` in
    place: the module's live tests carry the ``dockerbuild`` marker, so the
    blanket guard in section (e)
    (``test_every_dockerbuild_marked_file_is_ignored_in_e2e_lane``) is what
    pins it. This check is the other half of that pair — (e) proves the module
    left the shared lane, and only this one proves it arrived somewhere."""
    assert SCAN_AGENTIC_JOB in _jobs(workflow)
    assert SCAN_AGENTIC_TEST_FILE in json.dumps(_jobs(workflow)[SCAN_AGENTIC_JOB])


def test_scan_agentic_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][SCAN_AGENTIC_JOB]
    with pytest.raises(AssertionError):
        test_scan_agentic_job_exists(mutated)


def test_scan_agentic_job_exists__mutation_drops_the_run_step() -> None:
    """The vacuous-green half: the job survives, but nothing in it names the
    e2e module any more."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[SCAN_AGENTIC_JOB]["steps"]
    kept = [step for step in steps if SCAN_AGENTIC_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the e2e module"
    _jobs(mutated)[SCAN_AGENTIC_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_scan_agentic_job_exists(mutated)


def test_scan_agentic_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Both live tests drive a real agent and then a real judge, and both are
    ``requires_als_apg``-marked — without the secret they skip, so a job
    missing it would green-wash the lane down to its offline checks."""
    assert _job_declares_secret(workflow, SCAN_AGENTIC_JOB, SECRET_TOKEN)


def test_scan_agentic_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][SCAN_AGENTIC_JOB] = json.loads(
        json.dumps(mutated["jobs"][SCAN_AGENTIC_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, SCAN_AGENTIC_JOB, SECRET_TOKEN)


def _junit_reports_written_by(steps: list[dict[str, Any]]) -> list[str]:
    """Report paths the given steps write, in step order."""
    return [r for step in steps for r in _JUNIT_RE.findall(step["run"])]


def _scan_agentic_pytest_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """This job's pytest invocations. Matched on the full ``uv run pytest``
    invocation for the same reason as the gchat helper: the zero-skip gate step
    below runs ``uv run python`` and merely names pytest in its comment, and
    counting that as a pytest step would demand a junit report it never
    writes."""
    return [s for s in _jobs(wf)[SCAN_AGENTIC_JOB]["steps"] if "uv run pytest " in s.get("run", "")]


def test_scan_agentic_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """The secret check above only proves the key is present; this proves the
    lane cannot green while its expensive half quietly does nothing.

    Every skip this module can take is a CI misconfiguration rather than a
    legitimate environment gap — absent docker, absent SDK, absent ``claude``
    CLI, absent provider key — and pytest exits 0 for all of them. The CLI one
    is not hypothetical: claude-agent-sdk bundles its own CLI and installs no
    ``claude`` console script, so ``is_claude_code_available()`` can skip both
    live tests on a runner where the SDK would have worked, leaving the offline
    floor and judge checks to carry a green check on their own. So the job
    reads the junit report and fails on any skip, and on an empty collection,
    which would otherwise satisfy a zero-skip check trivially. Every pytest
    step must write a report the gate actually reads: a run whose report
    nothing inspects is back to skipping its way to green."""
    reports = _junit_reports_written_by(_scan_agentic_pytest_steps(workflow))
    assert len(reports) == len(_scan_agentic_pytest_steps(workflow)), (
        f"every pytest step in '{SCAN_AGENTIC_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, SCAN_AGENTIC_JOB, SCAN_AGENTIC_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{SCAN_AGENTIC_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_scan_agentic_job_fails_on_any_skipped_test__mutation_drops_the_junit_report() -> None:
    """A pytest step that writes no report is invisible to the gate — the
    lane keeps running, and the gate keeps passing on a file it never reads."""
    mutated = copy.deepcopy(_load_workflow())
    step = _scan_agentic_pytest_steps(mutated)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_scan_agentic_job_fails_on_any_skipped_test(mutated)


def test_scan_agentic_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, SCAN_AGENTIC_JOB, SCAN_AGENTIC_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_scan_agentic_job_fails_on_any_skipped_test(mutated)


def test_all_checks_passed_needs_scan_agentic(workflow: dict[str, Any]) -> None:
    """``needs:`` alone is not a gate — the roll-up runs ``if: always()``, so a
    needed job that failed still lets it start; the ``check_pr_lane`` line is
    what turns the result into an exit code. Both halves are pinned.

    Same drop-together/re-land-together discipline as the queue lane: if
    ``scan-agentic-e2e`` cannot go green in the PR, the job, its gate entry AND
    this guard pair leave as a unit and come back as a unit. Splitting them
    either wedges the merge gate on a job that no longer exists or leaves the
    lane running unwatched."""
    assert SCAN_AGENTIC_JOB in _jobs(workflow)[GATE_JOB]["needs"]
    assert f"needs.{SCAN_AGENTIC_JOB}.result" in _gate_run_text(workflow)


def test_all_checks_passed_needs_scan_agentic__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(SCAN_AGENTIC_JOB)
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_scan_agentic(mutated)


def test_all_checks_passed_needs_scan_agentic__mutation_drops_check_pr_lane_line() -> None:
    """The dangerous half: the ``needs`` entry stays (so the gate waits for the
    job) while the line that reads its result is gone — the lane could go red
    forever inside a green check. Worth pinning twice over here, because this
    lane is the expensive one: nobody re-reads a lane's log once the roll-up
    is green."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if SCAN_AGENTIC_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert SCAN_AGENTIC_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the needs entry survives
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_scan_agentic(mutated)


# ---------------------------------------------------------------------------
# (l) every CLI-dependent lane exposes the SDK's bundled Claude CLI
# ---------------------------------------------------------------------------

#: Lanes that run tests gated on ``is_claude_code_available()``. That helper
#: shells ``claude --version`` off PATH, and ``claude-agent-sdk`` ships a
#: working binary but installs no console script — so on a bare runner the
#: gated tests skip and the lane greens having executed nothing. Each lane
#: therefore puts the SDK's bundled CLI on PATH before running pytest.
CLI_DEPENDENT_JOBS = (
    "agentic-per-preset",
    "e2e-tests",
    "channel-finder-benchmarks",
    SCAN_AGENTIC_JOB,
)

BUNDLED_CLI_STEP = "Put the SDK's bundled Claude CLI on PATH"


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_exposes_the_bundled_claude(
    job_name: str, workflow: dict[str, Any] | None = None
) -> None:
    """A lane running agent tests must make ``claude`` resolvable on PATH.

    Without it the lane is vacuously green: pytest exits 0 having skipped
    every test that needed an agent. Only ``scan-agentic-e2e`` carries a
    zero-skip gate to catch that at runtime, so for the other three this
    guard is the only thing standing between a silent skip and a green
    check.

    Pinned by content, not just by step name: the step must both resolve the
    SDK's ``_bundled`` directory and append it to ``GITHUB_PATH``. Appending
    something else, or resolving without exporting, would leave the skips in
    place while the step still looked present.
    """
    wf = workflow if workflow is not None else _load_workflow()
    step = _find_named_step(wf, job_name, BUNDLED_CLI_STEP)
    run = step["run"]
    assert "_bundled" in run, f"{job_name}: step must resolve the SDK's bundled CLI directory"
    assert "GITHUB_PATH" in run, f"{job_name}: step must append that directory to GITHUB_PATH"


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_exposes_the_bundled_claude__mutation_drops_step(job_name: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[job_name]["steps"]
    kept = [s for s in steps if s.get("name") != BUNDLED_CLI_STEP]
    assert len(kept) == len(steps) - 1, "expected exactly one step dropped"
    _jobs(mutated)[job_name]["steps"] = kept
    with pytest.raises(AssertionError):
        test_cli_dependent_lane_exposes_the_bundled_claude(job_name, mutated)


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_exposes_the_bundled_claude__mutation_drops_export(
    job_name: str,
) -> None:
    """The quiet half: the step stays and still resolves the directory, but
    never exports it — so PATH is unchanged and every agent test skips again
    behind a step whose name says otherwise."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, job_name, BUNDLED_CLI_STEP)
    step["run"] = "\n".join(line for line in step["run"].splitlines() if "GITHUB_PATH" not in line)
    assert "_bundled" in step["run"], "the resolve line survives"
    with pytest.raises(AssertionError):
        test_cli_dependent_lane_exposes_the_bundled_claude(job_name, mutated)


# ---------------------------------------------------------------------------
# (g) the archiver-world lane: its own job, secret-free, and gated on both halves
# ---------------------------------------------------------------------------


def test_archiver_world_job_exists(workflow: dict[str, Any]) -> None:
    assert ARCHIVER_JOB in _jobs(workflow)


def test_archiver_world_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][ARCHIVER_JOB]
    with pytest.raises(AssertionError):
        assert ARCHIVER_JOB in _jobs(mutated)


def test_archiver_world_job_has_no_llm_secret(workflow: dict[str, Any]) -> None:
    """Every assertion in that file is mechanical — seed duration, store size, a
    setpoint round-trip, a noise-band step, document counts — and the archiver
    read goes through the connector rather than an agent turn. A secret
    appearing in this job would mean the lane's scope silently grew."""
    assert not _job_declares_secret(workflow, ARCHIVER_JOB, SECRET_TOKEN)


def test_archiver_world_job_has_no_llm_secret__mutation_adds_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ARCHIVER_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        assert not _job_declares_secret(mutated, ARCHIVER_JOB, SECRET_TOKEN)


def test_archiver_world_job_installs_the_pymongo_extra(workflow: dict[str, Any]) -> None:
    """The staged bring-up preflights pymongo before any image build, so without
    the extra this lane aborts in seconds with an install hint instead of running
    — a green-looking job that tested nothing."""
    installed = json.dumps(_jobs(workflow)[ARCHIVER_JOB])
    assert "archiver-mongodb" in installed, (
        f"the '{ARCHIVER_JOB}' lane must `uv sync` the archiver-mongodb extra"
    )


def test_archiver_world_job_installs_the_pymongo_extra__mutation_drops_extra() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ARCHIVER_JOB] = json.loads(
        json.dumps(mutated["jobs"][ARCHIVER_JOB]).replace(" --extra archiver-mongodb", "")
    )
    with pytest.raises(AssertionError):
        test_archiver_world_job_installs_the_pymongo_extra(mutated)


def test_archiver_world_job_runs_pytest_unbuffered(workflow: dict[str, Any]) -> None:
    """``-s`` is what puts this lane's measured budgets in the log on a GREEN run.

    Pytest captures stdout and replays it only for failures, so without this the
    seed duration, store size and write->read latency appear exactly when nobody
    can act on them any more. The budgets are the lane's product — a run that
    hides them still passes, and the trend that precedes a breach is invisible.
    """
    steps = json.dumps(_jobs(workflow)[ARCHIVER_JOB]["steps"])
    assert ARCHIVER_TEST_FILE in steps
    assert " -s " in steps or steps.rstrip().endswith(" -s"), (
        f"the '{ARCHIVER_JOB}' lane must run pytest with -s so the MEASURED "
        "budget lines reach the log on a passing run"
    )


def test_archiver_world_job_runs_pytest_unbuffered__mutation_drops_the_flag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ARCHIVER_JOB] = json.loads(
        json.dumps(mutated["jobs"][ARCHIVER_JOB]).replace(" -v -s ", " -v ")
    )
    with pytest.raises(AssertionError):
        test_archiver_world_job_runs_pytest_unbuffered(mutated)


def test_all_checks_passed_needs_archiver_world(workflow: dict[str, Any]) -> None:
    """Both halves of the gate, for the reason spelled out on the gchat pair:
    ``needs:`` makes the roll-up wait, ``check_pr_lane`` makes it care."""
    assert ARCHIVER_JOB in _jobs(workflow)[GATE_JOB]["needs"]
    assert f"needs.{ARCHIVER_JOB}.result" in _gate_run_text(workflow)


def test_all_checks_passed_needs_archiver_world__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(ARCHIVER_JOB)
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_archiver_world(mutated)


def test_all_checks_passed_needs_archiver_world__mutation_drops_check_pr_lane_line() -> None:
    """The dangerous half: the job is still waited on, but nothing reads its
    result — the lane could go red forever inside a green check."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if ARCHIVER_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert ARCHIVER_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the needs entry survives
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_archiver_world(mutated)


# ---------------------------------------------------------------------------
# CI diagnostics: the evidence a killed lane leaves behind
# ---------------------------------------------------------------------------
#
# A lane that FAILS explains itself in the job log. A lane that is KILLED — job
# timeout, runner OOM, a wedged xdist worker held to the cap — explains nothing,
# because the signal lands while pytest's output is still buffered. Everything
# pinned below exists to make that second case readable, and every piece of it
# is silently droppable: removing a step, raising a timeout, or restoring an ini
# option costs no test failure of its own. Hence these pins.

CAPTURE_ACTION = "./.github/actions/capture-ci-diagnostics"
CAPTURE_STEP = "Capture failure diagnostics"
WRITEBACK_JOB = "deploy-writeback-e2e"
PODMAN_LIFECYCLE_JOB = "multi-user-deploy-lifecycle-e2e-podman"
#: Jobs that name a container engine without running one — the gate lists the
#: Docker lanes in its human-readable status strings.
NON_CONTAINER_JOBS = {GATE_JOB}


def _container_jobs(wf: dict[str, Any]) -> set[str]:
    """Jobs that touch a container engine, derived rather than listed.

    Derived on purpose: a hand-maintained list is exactly what a newly added
    Docker lane forgets to join. Comments cannot pollute the match because
    ``yaml.safe_load`` has already dropped them.
    """
    pattern = re.compile(r"\b(docker|podman)\b")
    return {
        name
        for name, job in _jobs(wf).items()
        if name not in NON_CONTAINER_JOBS and pattern.search(json.dumps(job))
    }


def _capture_step_index(job: dict[str, Any]) -> int | None:
    for index, step in enumerate(job.get("steps") or []):
        if step.get("uses") == CAPTURE_ACTION:
            return index
    return None


def test_every_container_job_captures_diagnostics(workflow: dict[str, Any]) -> None:
    """No Docker lane may fail without leaving its container logs behind.

    Before this action existed, the only ``if: always()`` steps on these lanes
    were teardown blocks — so a red lane deleted the containers holding the
    explanation and uploaded nothing at all.
    """
    missing = sorted(
        name
        for name in _container_jobs(workflow)
        if _capture_step_index(_jobs(workflow)[name]) is None
    )
    assert missing == [], (
        f"these jobs run containers but have no '{CAPTURE_STEP}' step: {missing}. "
        f"Add `uses: {CAPTURE_ACTION}` with `if: failure()`, positioned before any "
        f"teardown step — or add the job to NON_CONTAINER_JOBS if it only names an "
        f"engine in a status string."
    )


def test_every_container_job_captures_diagnostics__mutation_drops_a_lane() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    del job["steps"][_capture_step_index(job)]
    with pytest.raises(AssertionError):
        test_every_container_job_captures_diagnostics(mutated)


def test_capture_runs_before_teardown(workflow: dict[str, Any]) -> None:
    """Ordering is the whole point: the teardown blocks ``docker rm -f`` the very
    containers whose logs are the diagnosis. Capture after cleanup captures
    nothing."""
    late = []
    for name in sorted(_container_jobs(workflow)):
        steps = _jobs(workflow)[name]["steps"]
        capture = _capture_step_index(_jobs(workflow)[name])
        teardown = [i for i, step in enumerate(steps) if step.get("if") == "always()"]
        if capture is not None and teardown and capture > min(teardown):
            late.append(name)
    assert late == [], (
        f"'{CAPTURE_STEP}' runs after an `if: always()` teardown step in {late} — "
        f"by then the containers it reads have been removed."
    )


def test_capture_runs_before_teardown__mutation_moves_it_after_cleanup() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[WRITEBACK_JOB]["steps"]
    steps.append(steps.pop(_capture_step_index(_jobs(mutated)[WRITEBACK_JOB])))
    with pytest.raises(AssertionError):
        test_capture_runs_before_teardown(mutated)


def test_capture_steps_run_on_failure(workflow: dict[str, Any]) -> None:
    """``if: failure()`` and not the default: a step with no condition is skipped
    the moment anything upstream fails, which is the only time it matters."""
    wrong = {}
    for name in sorted(_container_jobs(workflow)):
        index = _capture_step_index(_jobs(workflow)[name])
        if index is None:
            continue
        condition = _jobs(workflow)[name]["steps"][index].get("if")
        if condition != "failure()":
            wrong[name] = condition
    assert wrong == {}, f"capture steps must carry `if: failure()`; got {wrong}"


def test_capture_steps_run_on_failure__mutation_drops_the_condition() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    del job["steps"][_capture_step_index(job)]["if"]
    with pytest.raises(AssertionError):
        test_capture_steps_run_on_failure(mutated)


def test_capture_artifact_names_are_unique(workflow: dict[str, Any]) -> None:
    """Artifact names collide silently within a run — two lanes sharing one name
    means one lane's evidence overwrites the other's."""
    names = []
    for name in sorted(_container_jobs(workflow)):
        index = _capture_step_index(_jobs(workflow)[name])
        if index is not None:
            names.append(_jobs(workflow)[name]["steps"][index]["with"]["name"])
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert duplicates == [], f"duplicate capture artifact names: {duplicates}"


def test_capture_artifact_names_are_unique__mutation_collides_two_lanes() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    job["steps"][_capture_step_index(job)]["with"]["name"] = VA_JOB
    with pytest.raises(AssertionError):
        test_capture_artifact_names_are_unique(mutated)


def test_podman_lane_captures_with_podman(workflow: dict[str, Any]) -> None:
    """The podman lane has no ``docker`` binary; asking for one captures nothing."""
    job = _jobs(workflow)[PODMAN_LIFECYCLE_JOB]
    step = job["steps"][_capture_step_index(job)]
    assert step["with"]["engine"] == "podman"


def test_podman_lane_captures_with_podman__mutation_asks_for_docker() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[PODMAN_LIFECYCLE_JOB]
    job["steps"][_capture_step_index(job)]["with"]["engine"] = "docker"
    with pytest.raises(AssertionError):
        test_podman_lane_captures_with_podman(mutated)


# ---------------------------------------------------------------------------
# The unit lane's own survivability
# ---------------------------------------------------------------------------


def test_unit_test_step_timeout_is_below_the_job_cap(workflow: dict[str, Any]) -> None:
    """A JOB timeout is a hard cancel — the collection steps never dependably
    run, which is how a frozen lane ends up recorded as ``cancelled`` with
    nothing to read. A STEP timeout fails cleanly and leaves the rest of the
    job's time for the summary and upload steps."""
    job = _jobs(workflow)[UNIT_TEST_JOB]
    step = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests")
    step_cap = step.get("timeout-minutes")
    assert step_cap is not None, (
        "the 'Run unit tests' step needs its own `timeout-minutes`; without one a "
        "wedged worker is killed by the job cap and takes the evidence with it"
    )
    assert step_cap < job["timeout-minutes"], (
        f"step cap ({step_cap}m) must leave headroom under the job cap "
        f"({job['timeout-minutes']}m) for the diagnostics steps to run"
    )


def test_unit_test_step_timeout_is_below_the_job_cap__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_unit_test_step_timeout_is_below_the_job_cap(mutated)


def test_unit_test_step_timeout_is_below_the_job_cap__mutation_raises_it_to_the_cap() -> None:
    """Equal is as bad as absent: no headroom means no collection phase."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[UNIT_TEST_JOB]
    _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["timeout-minutes"] = job[
        "timeout-minutes"
    ]
    with pytest.raises(AssertionError):
        test_unit_test_step_timeout_is_below_the_job_cap(mutated)


def test_unit_lane_arms_the_diagnostics_recorder(workflow: dict[str, Any]) -> None:
    """``tests/ci_diagnostics.py`` installs nothing unless this variable is set,
    so without it the lane runs exactly as blind as before."""
    step = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests")
    assert step.get("env", {}).get("OSPREY_CI_DIAG_DIR"), (
        "the unit lane must set OSPREY_CI_DIAG_DIR on the pytest step to enable the "
        "per-worker event log and stack dumps"
    )


def test_unit_lane_arms_the_diagnostics_recorder__mutation_drops_the_env() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["env"]["OSPREY_CI_DIAG_DIR"]
    with pytest.raises(AssertionError):
        test_unit_lane_arms_the_diagnostics_recorder(mutated)


def test_unit_lane_uploads_its_diagnostics(workflow: dict[str, Any]) -> None:
    """``always()``, not ``failure()``: a lane killed by the job cap is recorded
    as ``cancelled``, and a ``failure()`` step would not run for it."""
    upload = _find_named_step(workflow, UNIT_TEST_JOB, "Upload lane diagnostics")
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == "ci-diag"
    summary = _find_named_step(workflow, UNIT_TEST_JOB, "Summarise lane diagnostics")
    assert summary["if"] == "always()"


def test_unit_lane_uploads_its_diagnostics__mutation_narrows_to_failure() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, UNIT_TEST_JOB, "Upload lane diagnostics")["if"] = "failure()"
    with pytest.raises(AssertionError):
        test_unit_lane_uploads_its_diagnostics(mutated)


def test_unit_lane_does_not_set_faulthandler_timeout(workflow: dict[str, Any]) -> None:
    """The subtle one: the ini option arms a watchdog that kills workers.

    pytest implements ``faulthandler_timeout`` with
    ``faulthandler.dump_traceback_later``, whose watchdog walks every thread's
    frames without holding the GIL. A sample landing inside one of the parsers
    this suite lives in — PyYAML, ruamel, Jinja2 — reads a frame that is being
    freed and takes the worker down with it, which xdist reports only as ``node
    down: Not properly terminated``. ``tests/ci_diagnostics.py`` samples from an
    ordinary Python thread for that reason, and setting this option would put
    the unsafe watchdog back, on a per-test timer.

    Its dumps would also reach only the job log, which is exactly what a
    cancelled job truncates.
    """
    line = _unit_test_pytest_line(workflow)
    assert "faulthandler_timeout" not in line, (
        "the unit lane must not pass `-o faulthandler_timeout`: pytest implements it "
        "with faulthandler.dump_traceback_later, whose GIL-free watchdog segfaults "
        "workers that are inside a YAML or Jinja parser. See the docstring of "
        "tests/ci_diagnostics.py."
    )


def _sole_heavy_step(job: dict[str, Any]) -> dict[str, Any] | None:
    """The lane's single pytest-running step, or None if there isn't exactly one.

    Lanes with several heavy steps are excluded on purpose: capping them means
    splitting one job budget between them, which is a per-lane runtime judgement
    rather than something derivable, and a wrong split reds a healthy run.
    """
    heavy = [s for s in (job.get("steps") or []) if "uv run pytest" in str(s.get("run", ""))]
    return heavy[0] if len(heavy) == 1 else None


def test_capped_container_lanes_cap_their_heavy_step(workflow: dict[str, Any]) -> None:
    """Same argument as the unit lane, applied to the e2e lanes.

    Without a step cap the hang is killed by the JOB cap, which is a hard cancel
    — and the capture step never runs, so the containers are torn down by the
    runner with their logs unread. That is the exact blind spot the capture step
    exists to close, so a capped lane must not leave it open.
    """
    missing = []
    for name in sorted(_container_jobs(workflow)):
        job = _jobs(workflow)[name]
        job_cap = job.get("timeout-minutes")
        step = _sole_heavy_step(job)
        if not job_cap or step is None:
            continue
        step_cap = step.get("timeout-minutes")
        if step_cap is None or step_cap >= job_cap:
            missing.append((name, step.get("name"), step_cap, job_cap))
    assert missing == [], (
        "these lanes declare a job cap and have one heavy step, so that step needs its "
        f"own smaller `timeout-minutes`: {missing}"
    )


def test_capped_container_lanes_cap_their_heavy_step__mutation_drops_a_step_cap() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _sole_heavy_step(_jobs(mutated)[TILED_JOB])["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_capped_container_lanes_cap_their_heavy_step(mutated)


def test_capped_container_lanes_cap_their_heavy_step__mutation_raises_it_to_the_job_cap() -> None:
    """No headroom is the same as no cap — the capture step still never runs."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    _sole_heavy_step(job)["timeout-minutes"] = job["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_capped_container_lanes_cap_their_heavy_step(mutated)


def test_unit_lane_does_not_set_faulthandler_timeout__mutation_restores_the_ini_option() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    step["run"] = step["run"].replace(
        " --dist loadgroup $COV_ARGS",
        " --dist loadgroup $COV_ARGS -o faulthandler_timeout=300",
    )
    with pytest.raises(AssertionError):
        test_unit_lane_does_not_set_faulthandler_timeout(mutated)
