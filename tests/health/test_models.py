"""Unit tests for the health-check result models.

The ``to_dict`` wire-shape tests are the P2/P3 contract: they pin the exact key
set, ordering, and conditional-key behavior emitted by ``CheckResult`` and
``CheckReport``. Treat a change here as a deliberate contract revision.
"""

from __future__ import annotations

import pytest

from osprey.health import STATUS_ICONS, CheckReport, CheckResult, Status

# --- Status -----------------------------------------------------------------


def test_status_values() -> None:
    assert Status.OK == "ok"
    assert Status.WARNING == "warning"
    assert Status.ERROR == "error"
    assert Status.SKIP == "skip"
    assert {s.value for s in Status} == {"ok", "warning", "error", "skip"}


def test_status_icons_cover_all_statuses() -> None:
    assert set(STATUS_ICONS) == {"ok", "warning", "error", "skip"}


# --- CheckResult signature & defaults ---------------------------------------


def test_check_result_signature_defaults() -> None:
    r = CheckResult(name="n", category="c", status=Status.OK, message="m")
    assert r.value == ""
    assert r.latency_ms == 0.0
    assert r.details == ""


def test_check_result_positional_signature() -> None:
    # Locked positional order: name, category, status, message, value, latency_ms, details
    r = CheckResult("n", "c", Status.WARNING, "m", "42 mA", 12.34, "extra")
    assert r.name == "n"
    assert r.category == "c"
    assert r.status is Status.WARNING
    assert r.message == "m"
    assert r.value == "42 mA"
    assert r.latency_ms == 12.34
    assert r.details == "extra"


# --- CheckResult.to_dict conditional keys -----------------------------------


def test_result_to_dict_minimal_omits_optional_keys() -> None:
    r = CheckResult("n", "c", Status.OK, "m")
    assert r.to_dict() == {
        "name": "n",
        "category": "c",
        "status": "ok",
        "message": "m",
    }


def test_result_to_dict_status_serialized_as_string() -> None:
    d = CheckResult("n", "c", Status.ERROR, "m").to_dict()
    assert d["status"] == "error"
    assert isinstance(d["status"], str)


def test_result_to_dict_includes_value_when_truthy() -> None:
    assert CheckResult("n", "c", Status.OK, "m", value="401.2 mA").to_dict()["value"] == "401.2 mA"


def test_result_to_dict_omits_empty_value() -> None:
    assert "value" not in CheckResult("n", "c", Status.OK, "m", value="").to_dict()


def test_result_to_dict_rounds_latency_to_one_decimal() -> None:
    assert CheckResult("n", "c", Status.OK, "m", latency_ms=12.3456).to_dict()["latency_ms"] == 12.3


def test_result_to_dict_omits_zero_latency() -> None:
    assert "latency_ms" not in CheckResult("n", "c", Status.OK, "m", latency_ms=0.0).to_dict()


def test_result_to_dict_includes_details_when_truthy() -> None:
    assert CheckResult("n", "c", Status.ERROR, "m", details="boom").to_dict()["details"] == "boom"


def test_result_to_dict_omits_empty_details() -> None:
    assert "details" not in CheckResult("n", "c", Status.OK, "m", details="").to_dict()


def test_result_to_dict_all_optional_keys_present() -> None:
    r = CheckResult("n", "c", Status.OK, "m", value="v", latency_ms=1.0, details="d")
    assert r.to_dict() == {
        "name": "n",
        "category": "c",
        "status": "ok",
        "message": "m",
        "value": "v",
        "latency_ms": 1.0,
        "details": "d",
    }


# --- CheckReport counts -----------------------------------------------------


def _mixed_report() -> CheckReport:
    return CheckReport(
        results=[
            CheckResult("a", "c", Status.OK, "m"),
            CheckResult("b", "c", Status.OK, "m"),
            CheckResult("c", "c", Status.WARNING, "m"),
            CheckResult("d", "c", Status.ERROR, "m"),
            CheckResult("e", "c", Status.SKIP, "m"),
        ]
    )


def test_report_counts() -> None:
    rep = _mixed_report()
    assert rep.ok_count == 2
    assert rep.warning_count == 1
    assert rep.error_count == 1
    assert rep.skip_count == 1
    assert rep.total == 5


def test_report_empty_counts() -> None:
    rep = CheckReport()
    assert rep.total == 0
    assert rep.ok_count == 0
    assert rep.exit_code == 0


# --- exit_code --------------------------------------------------------------


def test_exit_code_all_ok() -> None:
    rep = CheckReport(results=[CheckResult("a", "c", Status.OK, "m")])
    assert rep.exit_code == 0


def test_exit_code_warnings_only() -> None:
    rep = CheckReport(results=[CheckResult("a", "c", Status.WARNING, "m")])
    assert rep.exit_code == 1


def test_exit_code_errors_dominate_warnings() -> None:
    rep = CheckReport(
        results=[
            CheckResult("a", "c", Status.WARNING, "m"),
            CheckResult("b", "c", Status.ERROR, "m"),
        ]
    )
    assert rep.exit_code == 2


def test_exit_code_skips_do_not_affect_exit() -> None:
    rep = CheckReport(
        results=[
            CheckResult("a", "c", Status.OK, "m"),
            CheckResult("b", "c", Status.SKIP, "m"),
        ]
    )
    assert rep.exit_code == 0


# --- summary_line -----------------------------------------------------------


def test_summary_line_all_passed() -> None:
    rep = CheckReport(results=[CheckResult("a", "c", Status.OK, "m")])
    assert rep.summary_line() == "1/1 checks passed"


def test_summary_line_skip_clause() -> None:
    # 10 ok + 5 skip => 15 total, per the locked example wording.
    results = [CheckResult(f"ok{i}", "c", Status.OK, "m") for i in range(10)]
    results += [CheckResult(f"sk{i}", "c", Status.SKIP, "m") for i in range(5)]
    rep = CheckReport(results=results)
    assert rep.summary_line() == "10/15 checks passed (5 skipped)"


def test_summary_line_warning_singular_and_plural() -> None:
    one = CheckReport(results=[CheckResult("a", "c", Status.WARNING, "m")])
    assert one.summary_line() == "0/1 checks passed (1 warning)"
    two = CheckReport(
        results=[
            CheckResult("a", "c", Status.WARNING, "m"),
            CheckResult("b", "c", Status.WARNING, "m"),
        ]
    )
    assert two.summary_line() == "0/2 checks passed (2 warnings)"


def test_summary_line_error_singular_and_plural() -> None:
    one = CheckReport(results=[CheckResult("a", "c", Status.ERROR, "m")])
    assert one.summary_line() == "0/1 checks passed (1 error)"


def test_summary_line_warnings_errors_skips_order() -> None:
    rep = _mixed_report()
    assert rep.summary_line() == "2/5 checks passed (1 warning, 1 error, 1 skipped)"


# --- CheckReport.to_dict LOCKED wire shape ----------------------------------


def test_report_to_dict_wire_shape_keys_and_order() -> None:
    """The report wire shape is the P2/P3 contract — pin keys and their order."""
    d = CheckReport().to_dict()
    assert list(d.keys()) == [
        "summary",
        "ok",
        "warnings",
        "errors",
        "skips",
        "total",
        "elapsed_ms",
        "deadline_hit",
        "results",
    ]


def test_report_to_dict_exact_payload() -> None:
    rep = CheckReport(
        results=[
            CheckResult("a", "c", Status.OK, "ok msg"),
            CheckResult("b", "c", Status.WARNING, "warn msg", value="1.0", latency_ms=2.34),
            CheckResult("c", "c", Status.ERROR, "err msg", details="boom"),
            CheckResult("d", "c", Status.SKIP, "skip msg"),
        ],
        elapsed_ms=123.456,
        deadline_hit=True,
    )
    assert rep.to_dict() == {
        "summary": "1/4 checks passed (1 warning, 1 error, 1 skipped)",
        "ok": 1,
        "warnings": 1,
        "errors": 1,
        "skips": 1,
        "total": 4,
        "elapsed_ms": 123.5,
        "deadline_hit": True,
        "results": [
            {"name": "a", "category": "c", "status": "ok", "message": "ok msg"},
            {
                "name": "b",
                "category": "c",
                "status": "warning",
                "message": "warn msg",
                "value": "1.0",
                "latency_ms": 2.3,
            },
            {
                "name": "c",
                "category": "c",
                "status": "error",
                "message": "err msg",
                "details": "boom",
            },
            {"name": "d", "category": "c", "status": "skip", "message": "skip msg"},
        ],
    }


def test_report_to_dict_defaults() -> None:
    d = CheckReport().to_dict()
    assert d["elapsed_ms"] == 0.0
    assert d["deadline_hit"] is False
    assert d["results"] == []


def test_report_to_dict_is_json_serializable() -> None:
    import json

    payload = json.dumps(_mixed_report().to_dict())
    assert json.loads(payload)["total"] == 5


# --- package __init__: eager exports + lazy __getattr__ ---------------------


def test_package_eager_exports() -> None:
    import osprey.health as health

    assert health.Status is Status
    assert health.CheckResult is CheckResult
    assert health.CheckReport is CheckReport
    assert health.STATUS_ICONS is STATUS_ICONS


def test_package_lazy_names_declared_in_all() -> None:
    import osprey.health as health

    assert "run_health_suite" in health.__all__
    assert "HealthRuntime" in health.__all__


def test_package_getattr_unknown_raises() -> None:
    import osprey.health as health

    with pytest.raises(AttributeError):
        _ = health.does_not_exist
