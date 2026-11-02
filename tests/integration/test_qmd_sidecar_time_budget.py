"""The qmd sidecar modules' timeout marks, against the waits and the check that bound them.

A sidecar test can chain waits whose ceilings sum past the full check's
per-test cap, so each sidecar module carries its own ``timeout`` mark. That
mark has to sit between two bounds: above the longest chain of waits one test
can hit, so a slow boot fails with the wait's own message and the container's
log rather than the cap's kill, and below the check's ``bound_s``, so the check
still ends. These tests start no container: they recompute each chain from the
named ceilings and read both bounds from the declarations that state them.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests import _container_support
from tests.integration import (
    _qmd_ariel_support,
    _qmd_okf_support,
    test_qmd_ariel_search_sidecar,
    test_qmd_okf_search_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """One sidecar boot, then the drafting test's poll for its concept.

    Each of the two may run one round past its bound.
    """
    s = _qmd_ariel_support
    chain = s.SIDECAR_HEALTH_CEILING + _qmd_okf_support.SIDECAR_REINDEX_TIMEOUT
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
