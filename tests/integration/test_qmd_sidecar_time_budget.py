"""The qmd sidecar modules' timeout marks, against the waits and the check that bound them.

A sidecar test can chain waits whose ceilings sum past the full check's
per-test cap, so each sidecar module carries its own ``timeout`` mark. That
mark has to sit between two bounds: above the longest chain of waits one test
can hit, so a slow boot fails with the wait's own message and the container's
log rather than the cap's kill, and below the check's ``bound_s``, so the check
still ends. These tests start no container: they recompute each chain from the
named ceilings and read both bounds from the declarations that state them.

The CI steps that run those marks are bounded too: each step cap outlasts the
marks it runs, and the job cap outlasts its step caps, so a slow sidecar ends
with the wait's own message rather than a cancelled step. Every
``wait_until_ready`` site that passes a ``ceiling=`` is declared below, against
the mark that bounds it or the reason none has to.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import docker.constants
import pytest
import yaml

from tests import _container_support
from tests.e2e import test_deploy_lifecycle
from tests.integration import (
    _qmd_ariel_support,
    _qmd_okf_support,
    test_qmd_ariel_search_sidecar,
    test_qmd_okf_search_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

QMD_JOB = "qmd-sidecar-e2e"
SIDECAR_STEP = "Run qmd sidecar E2E"
DEPLOY_QMD_STEP = "Run the multi-user shared-bundle sidecar E2E"

#: The ``dockerbuild``, non-``wallclock`` qmd integration modules the sidecar
#: step selects that carry no timeout mark, each with why.
SIDECAR_STEP_EXEMPT = {
    "test_qmd_models_mount": "skips in this step; its own step runs it",
    "test_qmd_ariel_resync_paths": "starts no sidecar and carries no timeout mark",
}


@pytest.fixture(scope="module")
def pytest_check(monkeypatch_module: pytest.MonkeyPatch) -> Any:
    """The declared ``pytest`` check, read from ``scripts/check.py``.

    scripts/ is not a package: the declaration is loaded by path, under a name
    no import uses, and registered in sys.modules before exec so @dataclass can
    resolve annotations through cls.__module__.
    """
    name = "_check_for_sidecar_budget"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "check.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch_module.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return {c.id: c for c in module.CHECKS}["pytest"]


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    """A monkeypatch whose changes are undone when the module ends."""
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def _per_test_cap(pytest_check: Any) -> float:
    """The full check's ``--timeout=`` per-test cap."""
    caps = [
        float(flag.removeprefix("--timeout="))
        for flag in pytest_check.tier_flags["full"]
        if flag.startswith("--timeout=")
    ]
    assert len(caps) == 1, pytest_check.tier_flags["full"]
    return caps[0]


def _timeout_mark(module: ModuleType) -> float:
    """The seconds a module's ``pytestmark`` gives each of its tests."""
    marks = [mark for mark in module.pytestmark if mark.name == "timeout"]
    assert len(marks) == 1, f"{module.__name__} carries {len(marks)} timeout marks"
    return float(marks[0].args[0])


def _deploy_qmd_mark() -> float:
    """The seconds the deploy lifecycle's qmd test carries on its own mark."""
    test = test_deploy_lifecycle.test_deploy_lifecycle_shared_bundle_and_qmd_sidecar
    marks = [mark for mark in test.pytestmark if mark.name == "timeout"]
    assert len(marks) == 1, f"{test.__name__} carries {len(marks)} timeout marks"
    return float(marks[0].args[0])


@pytest.fixture(scope="module")
def ci_workflow() -> dict[str, Any]:
    """The CI workflow, as GitHub reads it."""
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def _job_cap(workflow: dict[str, Any], job: str) -> float:
    """A job's ``timeout-minutes``, in seconds."""
    return 60.0 * workflow["jobs"][job]["timeout-minutes"]


def _step_cap(workflow: dict[str, Any], job: str, step_name: str) -> float:
    """A named step's ``timeout-minutes``, in seconds."""
    steps = [s for s in workflow["jobs"][job]["steps"] if s.get("name") == step_name]
    assert len(steps) == 1, f"{job} has {len(steps)} steps named {step_name!r}"
    return 60.0 * steps[0]["timeout-minutes"]


def _sidecar_step_modules() -> dict[str, ModuleType]:
    """The qmd integration modules the sidecar step selects, by name.

    The step runs ``-m "dockerbuild and not wallclock" -k qmd`` over
    tests/integration/; the module-level marks decide the set.
    """
    selected = {}
    for path in sorted((REPO_ROOT / "tests" / "integration").glob("test_*qmd*.py")):
        module = importlib.import_module(f"tests.integration.{path.stem}")
        names = {mark.name for mark in getattr(module, "pytestmark", [])}
        if "dockerbuild" in names and "wallclock" not in names:
            selected[path.stem] = module
    return selected


def _postgres_start_bound() -> float:
    """The longest the lane's Postgres start can take before it fails.

    ``start_or_fail`` retries the start; each attempt runs testcontainers' own
    readiness wait, bounded by its configured timeout, with a pause between.
    """
    from testcontainers.core.config import testcontainers_config

    attempts = _container_support.CONTAINER_START_ATTEMPTS
    return (
        attempts * testcontainers_config.timeout
        + (attempts - 1) * _container_support.CONTAINER_START_BACKOFF_S
    )


def test_the_ariel_sidecar_mark_is_its_longest_chain_of_waits(pytest_check):
    """The first test to request the sidecar world pays for the whole chain.

    The Postgres start, the sidecar's ``/health`` wait, the world's own
    re-index wait, then one more re-index wait in the test body; each
    ``wait_until_ready`` wait may run one round past its ceiling.
    """
    s = _qmd_ariel_support
    chain = (
        _postgres_start_bound()
        + s.SIDECAR_HEALTH_CEILING
        + 2 * s.SIDECAR_REINDEX_CEILING
        + 3 * s.SIDECAR_ROUND_S
    )
    budget = _timeout_mark(test_qmd_ariel_search_sidecar)

    assert budget == s.SIDECAR_ARIEL_TEST_TIMEOUT
    assert chain == budget
    assert budget < pytest_check.bound_s


def test_the_okf_sidecar_mark_is_its_longest_chain_of_waits(pytest_check):
    """One sidecar boot, then the drafting test's re-index wait.

    Each of the two may run one round past its ceiling.
    """
    s = _qmd_ariel_support
    chain = s.SIDECAR_HEALTH_CEILING + s.SIDECAR_REINDEX_CEILING
    budget = _timeout_mark(test_qmd_okf_search_sidecar)

    assert budget == _qmd_okf_support.SIDECAR_OKF_TEST_TIMEOUT
    assert chain + 2 * s.SIDECAR_ROUND_S == budget
    assert budget < pytest_check.bound_s


@pytest.mark.parametrize(
    "ceiling",
    ["SIDECAR_HEALTH_CEILING", "SIDECAR_REINDEX_CEILING"],
)
def test_each_sidecar_ceiling_outlasts_the_per_test_cap(pytest_check, ceiling):
    """Why the marks exist: one wait alone would be killed by the cap."""
    assert getattr(_qmd_ariel_support, ceiling) > _per_test_cap(pytest_check)


def test_one_round_counts_every_step_a_wait_round_takes():
    """The slower probe, the pause and two liveness reads."""
    from osprey.services.qmd.client import DEFAULT_TIMEOUT_SECONDS

    s = _qmd_ariel_support
    assert s.SIDECAR_ROUND_S == (
        max(
            s.SIDECAR_PROBE_TIMEOUT,
            DEFAULT_TIMEOUT_SECONDS + docker.constants.DEFAULT_TIMEOUT_SECONDS,
        )
        + s.SIDECAR_POLL_INTERVAL
        + 2 * _container_support.CONTAINER_STATE_TIMEOUT
    )


def test_the_marker_retry_is_a_cadence_inside_the_reindex_ceiling():
    """A retry happens within the wait's rounds, so it adds no term to a chain."""
    s = _qmd_ariel_support
    assert s.SIDECAR_POLL_INTERVAL < s.SIDECAR_MARKER_RETRY_INTERVAL < s.SIDECAR_REINDEX_CEILING


def test_the_sidecar_step_cap_outlasts_the_marks_it_runs(ci_workflow):
    """The step runs every selected module back to back, each booting its own sidecar."""
    modules = _sidecar_step_modules()
    assert set(SIDECAR_STEP_EXEMPT) <= set(modules), sorted(modules)
    marks = {
        name: _timeout_mark(module)
        for name, module in modules.items()
        if name not in SIDECAR_STEP_EXEMPT
    }
    cap = _step_cap(ci_workflow, QMD_JOB, SIDECAR_STEP)

    assert sum(marks.values()) < cap, (
        f"step {SIDECAR_STEP!r} caps at {cap:.0f} s, under the {sum(marks.values()):.0f} s "
        f"its modules' marks sum to: {marks}"
    )


def test_the_deploy_lifecycle_qmd_step_cap_outlasts_its_mark(ci_workflow):
    mark = _deploy_qmd_mark()
    cap = _step_cap(ci_workflow, QMD_JOB, DEPLOY_QMD_STEP)

    assert mark < cap, f"step {DEPLOY_QMD_STEP!r} caps at {cap:.0f} s, under its {mark:.0f} s mark"


def test_the_qmd_job_cap_outlasts_its_step_caps(ci_workflow):
    """A step cap above the job's remaining time never fires."""
    steps = {
        step["name"]: 60.0 * step["timeout-minutes"]
        for step in ci_workflow["jobs"][QMD_JOB]["steps"]
        if "timeout-minutes" in step
    }
    cap = _job_cap(ci_workflow, QMD_JOB)

    assert sum(steps.values()) < cap, (
        f"job {QMD_JOB!r} caps at {cap:.0f} s, under the {sum(steps.values()):.0f} s "
        f"its step caps sum to: {steps}"
    )


def test_the_deploy_lifecycle_qmd_mark_is_its_longest_chain_of_waits(pytest_check):
    """The ``osprey up``, the sidecar's ``/health`` wait, then the re-index wait.

    Each of the two ``wait_until_ready`` waits may run one round past its
    ceiling.
    """
    s = _qmd_ariel_support
    chain = (
        test_deploy_lifecycle.DEPLOY_UP_TIMEOUT_SEC
        + s.SIDECAR_HEALTH_CEILING
        + s.SIDECAR_REINDEX_CEILING
        + 2 * s.SIDECAR_ROUND_S
    )
    budget = _deploy_qmd_mark()

    assert budget == test_deploy_lifecycle.DEPLOY_LIFECYCLE_QMD_TIMEOUT
    assert chain == budget
    assert budget < pytest_check.bound_s
