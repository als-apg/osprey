"""Unit tests for the runner's cost-class deadline machinery."""

from __future__ import annotations

import asyncio

import pytest

from osprey.health.config import CategoryRecord, CheckSpec, Cost
from osprey.health.models import CheckResult, Status
from osprey.health.runner import run_health_suite

# --- Fixtures / helpers -----------------------------------------------------


@pytest.fixture
def runtime():
    from osprey.health.runtime import HealthRuntime

    return HealthRuntime({})


async def _fake_probe(spec, ctx):
    """Controllable stand-in probe: ``hang`` sleeps forever, ``sleep`` delays,
    ``result_status`` sets the outcome (default ok)."""
    if spec.get("hang"):
        await asyncio.sleep(3600)
    sleep = spec.get("sleep")
    if sleep:
        await asyncio.sleep(float(sleep))
    return CheckResult(spec["name"], spec["category"], Status(spec.get("result_status", "ok")), "x")


@pytest.fixture(autouse=True)
def patch_probe(monkeypatch):
    monkeypatch.setattr("osprey.health.runner.get_probe", lambda _type: _fake_probe)


def _check(name, *, requires=(), timeout_s=5.0, **params):
    return CheckSpec(
        name=name,
        type="fake",
        params=params,
        timeout_s=timeout_s,
        timeout_status=Status.ERROR,
        requires=tuple(requires),
    )


def _decl(name, checks, *, cost=Cost.POLL, timeout_s=30.0):
    return CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, checks=checks)


def _call(name, func, *, cost=Cost.POLL, timeout_s=5.0):
    return CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, func=func)


def _by_name(report):
    return {r.name: r for r in report.results}


# --- Declarative deadline synthesis -----------------------------------------


async def test_deadline_synthesizes_error_row(runtime) -> None:
    records = [_decl("c", [_check("slow", hang=True)])]
    report = await run_health_suite(records, runtime=runtime, suite_timeout_s=0.3)
    assert report.deadline_hit is True
    row = report.results[0]
    assert row.status is Status.ERROR
    assert row.message == "suite deadline exceeded"


async def test_row_count_invariant_under_deadline(runtime) -> None:
    # One check completes, two hang; every configured check still yields a row.
    checks = [_check("fast", result_status="ok"), _check("h1", hang=True), _check("h2", hang=True)]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime, suite_timeout_s=0.3)
    by = _by_name(report)
    assert set(by) == {"fast", "h1", "h2"}
    assert by["fast"].status is Status.OK
    assert by["h1"].status is Status.ERROR
    assert by["h2"].status is Status.ERROR
    assert by["h1"].message == "suite deadline exceeded"


async def test_deadline_hit_false_within_budget(runtime) -> None:
    report = await run_health_suite([_decl("c", [_check("a")])], runtime=runtime, suite_timeout_s=5)
    assert report.deadline_hit is False
    assert report.results[0].status is Status.OK


# --- requires precedence at expiry ------------------------------------------


async def test_deadline_A_still_running_B_skips(runtime) -> None:
    # A never completes; at expiry A → deadline error, B (requires A) → skip.
    checks = [_check("A", hang=True), _check("B", requires=["A"])]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime, suite_timeout_s=0.3)
    by = _by_name(report)
    assert by["A"].status is Status.ERROR
    assert by["A"].message == "suite deadline exceeded"
    assert by["B"].status is Status.SKIP
    assert "A" in by["B"].message


async def test_deadline_A_already_failed_B_skips_requires_wins(runtime) -> None:
    # A resolves as error before expiry; an unrelated check hangs to force the
    # deadline. At expiry B (requires A) → skip (requires rule wins over the
    # deadline-error branch), while the eligible hung check → deadline error.
    checks = [
        _check("A", result_status="error"),
        _check("slow", hang=True),
        _check("B", requires=["A"]),
    ]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime, suite_timeout_s=0.3)
    by = _by_name(report)
    assert by["A"].status is Status.ERROR
    assert "A" not in by["A"].message  # A is a real error, not a synthesized deadline row
    assert by["slow"].status is Status.ERROR
    assert by["slow"].message == "suite deadline exceeded"
    assert by["B"].status is Status.SKIP
    assert "dependency 'A'" in by["B"].message


async def test_synthesized_deadline_error_cascades_to_dependents(runtime) -> None:
    checks = [_check("A", hang=True), _check("B", requires=["A"]), _check("C", requires=["B"])]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime, suite_timeout_s=0.3)
    by = _by_name(report)
    assert by["A"].status is Status.ERROR
    assert by["B"].status is Status.SKIP
    assert by["C"].status is Status.SKIP


# --- Callable-backed category -----------------------------------------------


async def test_callable_deadline_one_error_row(runtime) -> None:
    async def _hang():
        await asyncio.sleep(3600)
        return []

    report = await run_health_suite(
        [_call("mycat", _hang, timeout_s=30)], runtime=runtime, suite_timeout_s=0.3
    )
    assert len(report.results) == 1
    assert report.results[0].name == "mycat"
    assert report.results[0].status is Status.ERROR
    assert report.deadline_hit is True


# --- Budget independence ----------------------------------------------------


async def test_poll_deadline_does_not_bound_on_demand(runtime) -> None:
    # Poll category hangs (bounded by suite_timeout_s=0.3); on_demand category
    # completes because it has its own, generous budget.
    records = [
        _decl("poll_cat", [_check("p", hang=True)], cost=Cost.POLL),
        _decl("od_cat", [_check("d", sleep=0.5)], cost=Cost.ON_DEMAND),
    ]
    report = await run_health_suite(
        records, runtime=runtime, full=True, suite_timeout_s=0.3, on_demand_timeout_s=5.0
    )
    by = _by_name(report)
    assert by["p"].status is Status.ERROR  # poll deadline synthesized
    assert by["p"].message == "suite deadline exceeded"
    assert by["d"].status is Status.OK  # on_demand unaffected by poll deadline


async def test_on_demand_deadline_does_not_bound_poll(runtime) -> None:
    records = [
        _decl("poll_cat", [_check("p", sleep=0.4)], cost=Cost.POLL),
        _decl("od_cat", [_check("d", hang=True)], cost=Cost.ON_DEMAND),
    ]
    report = await run_health_suite(
        records, runtime=runtime, full=True, suite_timeout_s=5.0, on_demand_timeout_s=0.3
    )
    by = _by_name(report)
    assert by["p"].status is Status.OK  # poll unaffected by on_demand deadline
    assert by["d"].status is Status.ERROR
    assert by["d"].message == "suite deadline exceeded"


async def test_on_demand_timeout_defaults_to_sum_of_budgets(runtime) -> None:
    # No explicit on_demand_timeout_s: default = sum of on_demand category
    # budgets (0.3 + 0.3 = 0.6). Both hang, so both are synthesized within a
    # bounded time rather than running to the poll suite budget.
    records = [
        _decl("od1", [_check("a", hang=True)], cost=Cost.ON_DEMAND, timeout_s=0.3),
        _decl("od2", [_check("b", hang=True)], cost=Cost.ON_DEMAND, timeout_s=0.3),
    ]
    report = await run_health_suite(records, runtime=runtime, full=True, suite_timeout_s=30)
    by = _by_name(report)
    assert by["a"].status is Status.ERROR
    assert by["b"].status is Status.ERROR
    assert report.deadline_hit is True
    # Bounded well under the (irrelevant) 30s poll budget.
    assert report.elapsed_ms < 3000
