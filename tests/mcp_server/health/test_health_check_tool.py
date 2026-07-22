"""Unit tests for the ``health_check`` poll-tier MCP tool.

The tool is a thin serving layer over
:meth:`HealthServerContext.get_poll_report`: it validates the requested category
set against the cached config's records, filters the cached report's rows on a
category selection (with CLI-parity rules for plugin rows and config faults), and
wraps the report in the locked wire envelope plus the ``cached`` / ``age_s`` /
``refresh_suppressed`` serve flags. These tests replace the server context with a
lightweight stub so no config, connector, or suite is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from osprey.health.models import CheckReport, CheckResult, Status
from osprey.mcp_server.health.server_context import PollReportResult
from osprey.mcp_server.health.tools import health_check as hc
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

pytestmark = pytest.mark.unit

health_check_fn = get_tool_fn(hc.health_check)


# --- stubs ------------------------------------------------------------------


@dataclass
class _StubRecord:
    """Stand-in for a ``CategoryRecord`` — only ``name`` is read by the tool."""

    name: str


class _StubLoaded:
    """Minimal ``LoadedHealthConfig`` surface the tool reads."""

    def __init__(
        self,
        *,
        record_names: list[str],
        extra_rows: list[CheckResult] | None = None,
        config_ok: bool = True,
    ) -> None:
        self.records = [_StubRecord(name) for name in record_names]
        self.extra_rows: list[CheckResult] = extra_rows or []
        self.config_ok = config_ok


class _StubContext:
    """Returns a preconfigured ``PollReportResult`` or raises a preset error."""

    def __init__(self, result: Any = None, *, raises: BaseException | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[Any] = []

    async def get_poll_report(self, categories: Any = None) -> PollReportResult:
        self.calls.append(categories)
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture(autouse=True)
def _reset_health_singleton():
    """Match the sibling context tests' autouse reset (defensive; tool stubs the getter)."""
    from osprey.mcp_server.health.server_context import reset_server_context

    reset_server_context()
    yield
    reset_server_context()


def _install(monkeypatch, ctx: _StubContext) -> _StubContext:
    """Point the tool's ``get_server_context`` at *ctx*."""
    monkeypatch.setattr(hc, "get_server_context", lambda: ctx)
    return ctx


def _result(
    report: CheckReport,
    *,
    loaded: _StubLoaded,
    cached: bool = False,
    age_s: float = 0.0,
    refresh_suppressed: bool = False,
) -> PollReportResult:
    return PollReportResult(
        loaded=loaded,
        report=report,
        cached_at=0.0,
        cached=cached,
        age_s=age_s,
        refresh_suppressed=refresh_suppressed,
    )


# --- wire contract ----------------------------------------------------------


async def test_envelope_carries_report_shape_and_serve_flags(monkeypatch):
    report = CheckReport(
        results=[
            CheckResult("epics.ping", "connectivity", Status.OK, "up"),
            CheckResult("cfg.load", "configuration", Status.WARNING, "degraded"),
        ],
        elapsed_ms=42.0,
        deadline_hit=False,
    )
    loaded = _StubLoaded(record_names=["connectivity", "configuration"])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded, cached=True, age_s=3.5)))

    envelope = extract_response_dict(await health_check_fn())

    for key in (
        "summary",
        "ok",
        "warnings",
        "errors",
        "skips",
        "total",
        "elapsed_ms",
        "deadline_hit",
        "results",
    ):
        assert key in envelope
    assert envelope["cached"] is True
    assert envelope["age_s"] == pytest.approx(3.5)
    assert envelope["refresh_suppressed"] is False
    assert envelope["total"] == 2
    assert envelope["elapsed_ms"] == pytest.approx(42.0)


async def test_fresh_flags_pass_through(monkeypatch):
    report = CheckReport(results=[CheckResult("a", "cat", Status.OK, "ok")])
    loaded = _StubLoaded(record_names=["cat"])
    _install(
        monkeypatch,
        _StubContext(
            _result(report, loaded=loaded, cached=False, age_s=0.0, refresh_suppressed=False)
        ),
    )

    envelope = extract_response_dict(await health_check_fn())
    assert envelope["cached"] is False
    assert envelope["age_s"] == 0.0
    assert envelope["refresh_suppressed"] is False


async def test_refresh_suppressed_flag_passes_through(monkeypatch):
    report = CheckReport(results=[CheckResult("a", "cat", Status.OK, "ok")])
    loaded = _StubLoaded(record_names=["cat"])
    _install(
        monkeypatch,
        _StubContext(
            _result(report, loaded=loaded, cached=True, age_s=99.0, refresh_suppressed=True)
        ),
    )

    envelope = extract_response_dict(await health_check_fn())
    assert envelope["refresh_suppressed"] is True
    assert envelope["cached"] is True


# --- category validation ----------------------------------------------------


async def test_unknown_category_errors_with_valid_names(monkeypatch):
    report = CheckReport(results=[CheckResult("a", "connectivity", Status.OK, "ok")])
    loaded = _StubLoaded(record_names=["connectivity", "configuration"])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    with assert_raises_error(error_type="unknown_category") as ctx:
        await health_check_fn(categories=["bogus"])
    msg = ctx["envelope"]["error_message"]
    assert "bogus" in msg
    assert "connectivity" in msg and "configuration" in msg


async def test_empty_derived_category_is_valid_and_yields_zero_rows(monkeypatch):
    # 'idle' is a known record with no checks: valid name, no rows produced.
    report = CheckReport(
        results=[CheckResult("a", "connectivity", Status.OK, "ok")],
        elapsed_ms=5.0,
    )
    loaded = _StubLoaded(record_names=["connectivity", "idle"])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    envelope = extract_response_dict(await health_check_fn(categories=["idle"]))
    assert envelope["total"] == 0
    assert envelope["results"] == []
    # Carries the underlying run's elapsed_ms even with zero selected rows.
    assert envelope["elapsed_ms"] == pytest.approx(5.0)


# --- filtered serve ---------------------------------------------------------


async def test_filtered_serve_selects_by_category_and_carries_snapshot_meta(monkeypatch):
    report = CheckReport(
        results=[
            CheckResult("a", "connectivity", Status.OK, "up"),
            CheckResult("b", "providers", Status.WARNING, "slow"),
            CheckResult("c", "connectivity", Status.ERROR, "down"),
        ],
        elapsed_ms=77.0,
        deadline_hit=True,
    )
    loaded = _StubLoaded(record_names=["connectivity", "providers"])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    envelope = extract_response_dict(await health_check_fn(categories=["connectivity"]))
    cats = {r["category"] for r in envelope["results"]}
    assert cats == {"connectivity"}
    assert envelope["total"] == 2
    # Snapshot-level fields describe the underlying unfiltered run.
    assert envelope["elapsed_ms"] == pytest.approx(77.0)
    assert envelope["deadline_hit"] is True


async def test_filtered_serve_excludes_plugin_extra_rows(monkeypatch):
    plugin_row = CheckResult("plugin.diag", "providers", Status.WARNING, "plugin load issue")
    report = CheckReport(
        results=[
            CheckResult("b", "providers", Status.OK, "up"),
            plugin_row,
        ]
    )
    # The same object instance is the registered plugin extra row.
    loaded = _StubLoaded(record_names=["providers"], extra_rows=[plugin_row])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    envelope = extract_response_dict(await health_check_fn(categories=["providers"]))
    names = {r["name"] for r in envelope["results"]}
    assert names == {"b"}
    assert "plugin.diag" not in names


async def test_filtered_serve_force_includes_configuration_when_config_not_ok(monkeypatch):
    report = CheckReport(
        results=[
            CheckResult("cfg.fault", "configuration", Status.ERROR, "config broke"),
            CheckResult("net", "connectivity", Status.OK, "up"),
        ]
    )
    loaded = _StubLoaded(
        record_names=["configuration", "connectivity"],
        config_ok=False,
    )
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    # Ask only for connectivity, but the config-fault rows must ride along.
    envelope = extract_response_dict(await health_check_fn(categories=["connectivity"]))
    cats = {r["category"] for r in envelope["results"]}
    assert cats == {"connectivity", "configuration"}


async def test_filtered_serve_does_not_force_configuration_when_config_ok(monkeypatch):
    report = CheckReport(
        results=[
            CheckResult("cfg.ok", "configuration", Status.OK, "loaded"),
            CheckResult("net", "connectivity", Status.OK, "up"),
        ]
    )
    loaded = _StubLoaded(record_names=["configuration", "connectivity"], config_ok=True)
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    envelope = extract_response_dict(await health_check_fn(categories=["connectivity"]))
    cats = {r["category"] for r in envelope["results"]}
    assert cats == {"connectivity"}


# --- unfiltered serve -------------------------------------------------------


async def test_unfiltered_serve_keeps_plugin_rows(monkeypatch):
    plugin_row = CheckResult("plugin.diag", "providers", Status.WARNING, "plugin load issue")
    report = CheckReport(
        results=[
            CheckResult("b", "providers", Status.OK, "up"),
            plugin_row,
        ]
    )
    loaded = _StubLoaded(record_names=["providers"], extra_rows=[plugin_row])
    _install(monkeypatch, _StubContext(_result(report, loaded=loaded)))

    envelope = extract_response_dict(await health_check_fn())
    names = {r["name"] for r in envelope["results"]}
    assert names == {"b", "plugin.diag"}


# --- suppression ------------------------------------------------------------


async def test_cold_and_suppressed_returns_health_suppressed_error(monkeypatch):
    from osprey.mcp_server.health.server_context import HealthRefreshSuppressedError

    _install(monkeypatch, _StubContext(raises=HealthRefreshSuppressedError(3)))

    with assert_raises_error(error_type="health_suppressed") as ctx:
        await health_check_fn()
    envelope = ctx["envelope"]
    assert "3" in envelope["error_message"]
    assert envelope["details"]["wedged_count"] == 3
