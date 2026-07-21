"""Unit tests for the async health-suite runner (core execution)."""

from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from osprey.health.config import CategoryRecord, CheckSpec, Cost
from osprey.health.models import CheckResult, Status
from osprey.health.runner import run_health_suite

# --- Fixtures / helpers -----------------------------------------------------


@pytest.fixture
def runtime():
    from osprey.health.runtime import HealthRuntime

    # No channel_read checks in these tests, so the connector is never built.
    return HealthRuntime({})


async def _fake_probe(spec, ctx):
    """Controllable stand-in probe driven by check params.

    Params recognized: ``result_status`` (default ok), ``sleep`` (seconds),
    ``hang`` (sleep effectively forever), ``raise`` (raise RuntimeError).
    """
    if spec.get("hang"):
        await asyncio.sleep(3600)
    sleep = spec.get("sleep")
    if sleep:
        await asyncio.sleep(float(sleep))
    if spec.get("raise"):
        raise RuntimeError("boom")
    status = Status(spec.get("result_status", "ok"))
    return CheckResult(spec["name"], spec["category"], status, "fake", value=spec.get("value", ""))


@pytest.fixture(autouse=True)
def patch_probe(monkeypatch):
    monkeypatch.setattr("osprey.health.runner.get_probe", lambda _type: _fake_probe)


def _check(name, *, requires=(), timeout_s=5.0, timeout_status=Status.ERROR, **params):
    return CheckSpec(
        name=name,
        type="fake",
        params=params,
        timeout_s=timeout_s,
        timeout_status=timeout_status,
        requires=tuple(requires),
    )


def _decl(name, checks, *, cost=Cost.POLL, timeout_s=30.0):
    return CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, checks=checks)


def _call(name, func, *, cost=Cost.POLL, timeout_s=5.0):
    return CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, func=func)


def _by_name(report):
    return {r.name: r for r in report.results}


# --- Basic execution --------------------------------------------------------


async def test_single_declarative_check_ok(runtime) -> None:
    report = await run_health_suite([_decl("c", [_check("a")])], runtime=runtime)
    assert [r.status for r in report.results] == [Status.OK]
    assert report.results[0].name == "a"
    assert report.results[0].category == "c"


async def test_results_preserve_category_then_declared_order(runtime) -> None:
    records = [
        _decl("cat1", [_check("a"), _check("b")]),
        _decl("cat2", [_check("c")]),
    ]
    report = await run_health_suite(records, runtime=runtime)
    assert [r.name for r in report.results] == ["a", "b", "c"]


async def test_declared_order_preserved_despite_completion_order(runtime) -> None:
    # 'first' sleeps longer than 'second' but must still appear first.
    checks = [_check("first", sleep=0.15), _check("second", sleep=0.01)]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    assert [r.name for r in report.results] == ["first", "second"]


# --- Isolation --------------------------------------------------------------


async def test_raising_check_becomes_error_and_siblings_run(runtime) -> None:
    checks = [_check("boom", **{"raise": True}), _check("ok_one")]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    by = _by_name(report)
    assert by["boom"].status is Status.ERROR
    assert "RuntimeError" in by["boom"].message
    assert by["ok_one"].status is Status.OK


async def test_raising_category_does_not_abort_others(runtime) -> None:
    records = [
        _decl("bad", [_check("x", **{"raise": True})]),
        _decl("good", [_check("y")]),
    ]
    report = await run_health_suite(records, runtime=runtime)
    by = _by_name(report)
    assert by["x"].status is Status.ERROR
    assert by["y"].status is Status.OK


# --- Per-check timeout / timeout_status -------------------------------------


async def test_hung_check_times_out_as_error_by_default(runtime) -> None:
    checks = [_check("slow", timeout_s=0.1, hang=True)]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    assert report.results[0].status is Status.ERROR
    assert "timed out" in report.results[0].message


async def test_hung_check_timeout_status_warning_opt_in(runtime) -> None:
    checks = [_check("slow", timeout_s=0.1, timeout_status=Status.WARNING, hang=True)]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    assert report.results[0].status is Status.WARNING


# --- requires gating + skip cascade -----------------------------------------


async def test_requires_failed_dependency_skips_dependent(runtime) -> None:
    checks = [
        _check("a", result_status="error"),
        _check("b", requires=["a"]),
    ]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    by = _by_name(report)
    assert by["a"].status is Status.ERROR
    assert by["b"].status is Status.SKIP
    assert "a" in by["b"].message


async def test_requires_warning_dependency_passes(runtime) -> None:
    # A warning-classified dependency still "passes" (ok|warning).
    checks = [
        _check("a", result_status="warning"),
        _check("b", requires=["a"]),
    ]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    by = _by_name(report)
    assert by["a"].status is Status.WARNING
    assert by["b"].status is Status.OK


async def test_requires_warning_timeout_passes_dependent(runtime) -> None:
    # timeout_status=warning composes literally with the requires pass rule.
    checks = [
        _check("a", timeout_s=0.1, timeout_status=Status.WARNING, hang=True),
        _check("b", requires=["a"]),
    ]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    by = _by_name(report)
    assert by["a"].status is Status.WARNING
    assert by["b"].status is Status.OK


async def test_requires_skip_cascades(runtime) -> None:
    checks = [
        _check("a", result_status="error"),
        _check("b", requires=["a"]),
        _check("c2", requires=["b"]),
    ]
    report = await run_health_suite([_decl("c", checks)], runtime=runtime)
    by = _by_name(report)
    assert by["a"].status is Status.ERROR
    assert by["b"].status is Status.SKIP
    assert by["c2"].status is Status.SKIP


# --- Concurrency overlap ----------------------------------------------------


async def test_categories_run_concurrently(runtime) -> None:
    records = [
        _decl("c1", [_check("a", sleep=0.3)]),
        _decl("c2", [_check("b", sleep=0.3)]),
    ]
    t0 = perf_counter()
    report = await run_health_suite(records, runtime=runtime)
    elapsed = perf_counter() - t0
    assert elapsed < 0.5  # concurrent ~0.3s, not serial 0.6s
    assert all(r.status is Status.OK for r in report.results)


async def test_independent_checks_in_category_overlap(runtime) -> None:
    checks = [_check("a", sleep=0.3), _check("b", sleep=0.3)]  # no requires → same batch
    t0 = perf_counter()
    await run_health_suite([_decl("c", checks)], runtime=runtime)
    elapsed = perf_counter() - t0
    assert elapsed < 0.5


# --- Cost gating (--full) ---------------------------------------------------


async def test_on_demand_declarative_skipped_without_full(runtime) -> None:
    records = [_decl("od", [_check("a"), _check("b")], cost=Cost.ON_DEMAND)]
    report = await run_health_suite(records, runtime=runtime, full=False)
    assert [r.status for r in report.results] == [Status.SKIP, Status.SKIP]
    assert all("--full" in r.message for r in report.results)


async def test_on_demand_declarative_runs_with_full(runtime) -> None:
    records = [_decl("od", [_check("a")], cost=Cost.ON_DEMAND)]
    report = await run_health_suite(records, runtime=runtime, full=True)
    assert report.results[0].status is Status.OK


async def test_poll_category_runs_without_full(runtime) -> None:
    report = await run_health_suite([_decl("p", [_check("a")])], runtime=runtime, full=False)
    assert report.results[0].status is Status.OK


async def test_category_selection_never_elevates_on_demand(runtime) -> None:
    # Selecting an on_demand category without --full still yields skip rows.
    records = [_decl("od", [_check("a")], cost=Cost.ON_DEMAND)]
    report = await run_health_suite(records, runtime=runtime, full=False, categories=["od"])
    assert report.results[0].status is Status.SKIP


# --- Category selection -----------------------------------------------------


async def test_categories_filter_selects_subset(runtime) -> None:
    records = [
        _decl("a", [_check("x")]),
        _decl("b", [_check("y")]),
        _decl("c", [_check("z")]),
    ]
    report = await run_health_suite(records, runtime=runtime, categories=["a", "c"])
    assert [r.name for r in report.results] == ["x", "z"]


async def test_empty_categories_filter_runs_nothing(runtime) -> None:
    records = [_decl("a", [_check("x")])]
    report = await run_health_suite(records, runtime=runtime, categories=[])
    assert report.results == []
    assert report.exit_code == 0


# --- Callable categories ----------------------------------------------------


async def test_sync_callable_category_offloaded(runtime) -> None:
    def _sync_cat():
        return [CheckResult("s1", "mycat", Status.OK, "sync ok")]

    report = await run_health_suite([_call("mycat", _sync_cat)], runtime=runtime)
    assert report.results[0].name == "s1"
    assert report.results[0].status is Status.OK


async def test_async_callable_category(runtime) -> None:
    async def _async_cat():
        return [CheckResult("a1", "mycat", Status.WARNING, "async warn")]

    report = await run_health_suite([_call("mycat", _async_cat)], runtime=runtime)
    assert report.results[0].status is Status.WARNING


async def test_raising_callable_becomes_single_error_row(runtime) -> None:
    def _boom():
        raise ValueError("nope")

    report = await run_health_suite([_call("mycat", _boom)], runtime=runtime)
    assert len(report.results) == 1
    row = report.results[0]
    assert row.name == "mycat"
    assert row.status is Status.ERROR
    assert "nope" in row.details


async def test_timing_out_callable_becomes_error_row(runtime) -> None:
    async def _slow():
        await asyncio.sleep(3600)
        return []

    report = await run_health_suite([_call("mycat", _slow, timeout_s=0.1)], runtime=runtime)
    assert len(report.results) == 1
    assert report.results[0].status is Status.ERROR
    assert "budget" in report.results[0].message


async def test_on_demand_callable_skip_row_named_after_category(runtime) -> None:
    report = await run_health_suite(
        [_call("mycat", lambda: [], cost=Cost.ON_DEMAND)], runtime=runtime, full=False
    )
    assert len(report.results) == 1
    assert report.results[0].name == "mycat"
    assert report.results[0].status is Status.SKIP


# --- Report shape -----------------------------------------------------------


async def test_report_has_elapsed_and_deadline_false(runtime) -> None:
    report = await run_health_suite([_decl("c", [_check("a")])], runtime=runtime)
    assert report.elapsed_ms >= 0.0
    assert report.deadline_hit is False
