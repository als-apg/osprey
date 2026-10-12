"""Tests for the core ``telemetry_gateway`` health category.

Every rule is driven from canned Collector counter pages (Prometheus text) —
the stateless tier from one page, the host tier from a stored sample plus a
page — with an injected clock, so each rule, each tier and the re-baseline are
pinned without a running gateway.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.health.core.telemetry_gateway import (
    BASELINE_FILE,
    CATEGORY,
    COLLECTOR_FILE,
    Baseline,
    evaluate_trend,
    fed_identities,
    metrics_url,
    non_regular_ledgers,
    parse_metrics,
    read_baseline,
    stateless_rows,
    telemetry_gateway,
    write_baseline,
)
from osprey.health.models import CheckResult, Status

T0 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC)
DEPLOYED = {"deployed_services": ["telemetry_store", "telemetry_gateway"]}
LEDGER = "clickhouse/ledger"
OTLP = "clickhouse/otlp"
GIB = 1024 * 1024 * 1024


def page(
    *,
    sent: float = 100,
    failed: float = 0,
    refused: float = 0,
    enqueue_failed: float = 0,
    queue: float = 0,
    capacity: float | None = GIB,
    otlp_sent: float = 50,
    total_suffix: bool = True,
) -> str:
    """A Collector counters page with the series the category reads."""
    t = "_total" if total_suffix else ""
    lines = [
        "# HELP otelcol_exporter_sent_log_records Number of log records sent.",
        "# TYPE otelcol_exporter_sent_log_records counter",
        f'otelcol_exporter_sent_log_records{t}{{exporter="{LEDGER}",service_instance_id="a"}} {sent}',
        f'otelcol_exporter_sent_log_records{t}{{exporter="{OTLP}",service_instance_id="a"}} {otlp_sent}',
        f'otelcol_exporter_send_failed_log_records{t}{{exporter="{LEDGER}"}} {failed}',
        f'otelcol_exporter_enqueue_failed_log_records{t}{{exporter="{LEDGER}"}} {enqueue_failed}',
        f'otelcol_receiver_refused_log_records{t}{{receiver="otlp",transport="http"}} {refused}',
        f'otelcol_exporter_queue_size{{data_type="logs",exporter="{LEDGER}"}} {queue}',
        'otelcol_process_uptime_total{service_instance_id="a"} 12.5',
    ]
    if capacity is not None:
        lines.append(f'otelcol_exporter_queue_capacity{{exporter="{LEDGER}"}} {capacity}')
    return "\n".join(lines) + "\n"


def by_name(rows: list[CheckResult]) -> dict[str, CheckResult]:
    return {row.name: row for row in rows}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [True, False])
def test_parse_reads_counters_with_and_without_the_total_suffix(suffix):
    sample = parse_metrics(page(sent=7, failed=2, refused=3, enqueue_failed=1, total_suffix=suffix))
    assert sample.per_exporter("otelcol_exporter_sent_") == {LEDGER: 7, OTLP: 50}
    assert sample.per_exporter("otelcol_exporter_send_failed_") == {LEDGER: 2}
    assert sample.total("otelcol_exporter_enqueue_failed_", "otelcol_receiver_refused_") == 4
    assert sample.queue_capacity == {LEDGER: GIB}


def test_parse_ignores_comments_garbage_and_unrelated_series():
    sample = parse_metrics('# x\nnot a sample\nfoo{a="b"} nope\nup 1\n')
    assert sample.counters == {} and sample.queue_size == {}


# --------------------------------------------------------------------------- #
# Stateless tier
# --------------------------------------------------------------------------- #


def test_stateless_ok_page():
    rows = by_name(stateless_rows(parse_metrics(page())))
    assert rows["telemetry_gateway_queue"].status == Status.OK
    assert rows["telemetry_gateway_refusals"].status == Status.OK
    assert rows["telemetry_gateway_refusals"].message == "0 refusals since gateway start"


def test_queue_over_eighty_percent_degrades():
    rows = by_name(stateless_rows(parse_metrics(page(queue=0.81 * GIB))))
    assert rows["telemetry_gateway_queue"].status == Status.WARNING
    assert LEDGER in rows["telemetry_gateway_queue"].message


def test_queue_at_eighty_percent_is_ok():
    rows = by_name(stateless_rows(parse_metrics(page(queue=0.8 * GIB))))
    assert rows["telemetry_gateway_queue"].status == Status.OK


def test_queue_without_a_capacity_series_uses_the_rendered_queue_size():
    rows = by_name(stateless_rows(parse_metrics(page(queue=0.9 * GIB, capacity=None))))
    assert rows["telemetry_gateway_queue"].status == Status.WARNING


def test_a_full_logs_queue_is_not_masked_by_empty_queues_of_other_signals():
    other_signals = (
        f'otelcol_exporter_queue_size{{data_type="metrics",exporter="{LEDGER}"}} 0\n'
        f'otelcol_exporter_queue_size{{data_type="traces",exporter="{LEDGER}"}} 0\n'
    )
    sample = parse_metrics(page(queue=0.9 * GIB) + other_signals)
    assert sample.queue_size == {LEDGER: 0.9 * GIB}
    row = by_name(stateless_rows(sample))["telemetry_gateway_queue"]
    assert row.status == Status.WARNING and f"{LEDGER} logs" in row.message


def test_one_past_refusal_does_not_degrade_the_stateless_tier():
    rows = by_name(stateless_rows(parse_metrics(page(refused=1))))
    assert rows["telemetry_gateway_refusals"].status == Status.OK
    assert rows["telemetry_gateway_refusals"].message == "1 refusal since gateway start"
    assert all(row.status == Status.OK for row in rows.values())


def test_refusal_row_counts_both_refusal_families():
    rows = by_name(stateless_rows(parse_metrics(page(refused=2, enqueue_failed=3))))
    assert rows["telemetry_gateway_refusals"].value == "5"


def test_non_regular_ledgers_are_found_with_dotfiles_and_without_following_links(tmp_path):
    audit = tmp_path / "var" / "audit"
    (audit / "bob").mkdir(parents=True)
    (audit / "bob" / "ok.jsonl").write_text("{}\n")
    (audit / "bob" / "note.txt").symlink_to("/dev/null")
    (audit / "bob" / ".hidden.jsonl").symlink_to(audit / "bob" / "ok.jsonl")
    (audit / "eve").mkdir()
    (audit / "eve" / "zero.jsonl").symlink_to("/dev/zero")
    os.mkfifo(audit / "eve" / "pipe.jsonl")
    (audit / "loose.jsonl").write_text("")
    found = [p.relative_to(audit).as_posix() for p in non_regular_ledgers(audit)]
    assert found == ["bob/.hidden.jsonl", "eve/pipe.jsonl", "eve/zero.jsonl"]


def test_fed_identities_unescape_the_include_globs():
    rendered = {
        "receivers": {
            "otlp": {"include": ["/audit/nobody/*.jsonl"]},
            "file_log/d1": {"include": ["/audit/bob/*.jsonl"]},
            "file_log/d2": {"include": [r"/audit/ops team/*.jsonl"]},
            "file_log/d3": {"include": ["/audit/EXPR[(]x[)]/*.jsonl"]},
            "file_log/d4": {"include": [r"/audit/a\+b/*.jsonl"]},
        }
    }
    assert fed_identities(rendered) == {"bob", "ops team", "EXPR(x)", "a+b"}


# --------------------------------------------------------------------------- #
# Host tier
# --------------------------------------------------------------------------- #


def baseline_of(text: str, ts: datetime, since: datetime | None = None) -> Baseline:
    sample = parse_metrics(text)
    sent = sample.per_exporter("otelcol_exporter_sent_")
    return Baseline(dict(sample.counters), ts, dict.fromkeys(sent, since or ts))


def test_no_previous_sample_takes_a_baseline():
    row, stored = evaluate_trend(parse_metrics(page()), None, T0)
    assert (row.status, row.message) == (Status.OK, "Baseline taken")
    assert stored is not None and stored.ts == T0


def test_counter_below_its_previous_value_re_baselines():
    previous = baseline_of(page(sent=500, refused=4), T0)
    row, stored = evaluate_trend(
        parse_metrics(page(sent=3, refused=0)), previous, T0 + timedelta(seconds=60)
    )
    assert row.status == Status.OK and "restarted" in row.message
    assert stored is not None and stored.ts == T0 + timedelta(seconds=60)


def test_steady_counters_are_ok_and_replace_the_baseline():
    previous = baseline_of(page(sent=100), T0)
    now = T0 + timedelta(seconds=45)
    row, stored = evaluate_trend(parse_metrics(page(sent=140)), previous, now)
    assert row.status == Status.OK
    assert stored is not None and stored.ts == now and stored.sent_since[LEDGER] == now


def test_send_failures_rising_while_nothing_is_sent_degrades():
    previous = baseline_of(page(sent=100, failed=1), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=100, failed=9)), previous, T0 + timedelta(seconds=31)
    )
    assert row.status == Status.WARNING and "send failures" in row.message


def test_send_failures_rising_while_sends_still_succeed_is_ok():
    previous = baseline_of(page(sent=100, failed=1), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=120, failed=9)), previous, T0 + timedelta(seconds=31)
    )
    assert row.status == Status.OK


@pytest.mark.parametrize("field", ["refused", "enqueue_failed"])
def test_a_refusal_rising_since_the_sample_degrades(field):
    previous = baseline_of(page(**{field: 1}), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=150, **{field: 2})), previous, T0 + timedelta(seconds=40)
    )
    assert row.status == Status.WARNING and "refusal" in row.message


def test_an_old_refusal_that_did_not_rise_is_ok_on_the_host():
    previous = baseline_of(page(refused=1), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=150, refused=1)), previous, T0 + timedelta(seconds=40)
    )
    assert row.status == Status.OK


def test_a_sample_younger_than_thirty_seconds_is_not_compared_and_is_kept():
    previous = baseline_of(page(sent=100, failed=0, refused=0), T0)
    row, stored = evaluate_trend(
        parse_metrics(page(sent=100, failed=5, refused=5)), previous, T0 + timedelta(seconds=10)
    )
    assert row.status == Status.OK
    assert stored is None


def test_queue_non_empty_with_sent_unchanged_for_three_hundred_seconds_degrades():
    previous = baseline_of(page(sent=100), T0 + timedelta(seconds=240), since=T0)
    row, stored = evaluate_trend(
        parse_metrics(page(sent=100, queue=4096)), previous, T0 + timedelta(seconds=300)
    )
    assert row.status == Status.WARNING and "sent nothing for 300 s" in row.message
    assert stored is not None and stored.sent_since[LEDGER] == T0


def test_stall_sees_data_in_one_signal_queue_when_the_others_are_empty():
    previous = baseline_of(page(sent=100), T0 + timedelta(seconds=240), since=T0)
    empty_traces = f'otelcol_exporter_queue_size{{data_type="traces",exporter="{LEDGER}"}} 0\n'
    row, _ = evaluate_trend(
        parse_metrics(page(sent=100, queue=4096) + empty_traces),
        previous,
        T0 + timedelta(seconds=300),
    )
    assert row.status == Status.WARNING and "sent nothing for 300 s" in row.message


def test_queue_non_empty_but_sending_is_ok():
    previous = baseline_of(page(sent=100), T0 + timedelta(seconds=240), since=T0)
    row, stored = evaluate_trend(
        parse_metrics(page(sent=101, queue=4096)), previous, T0 + timedelta(seconds=300)
    )
    assert row.status == Status.OK
    assert stored is not None and stored.sent_since[LEDGER] == T0 + timedelta(seconds=300)


def test_queue_stalled_under_three_hundred_seconds_is_ok():
    previous = baseline_of(page(sent=100), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=100, queue=4096)), previous, T0 + timedelta(seconds=299)
    )
    assert row.status == Status.OK


def test_backup_rename_in_the_log_degrades():
    previous = baseline_of(page(), T0)
    row, _ = evaluate_trend(
        parse_metrics(page(sent=120)), previous, T0 + timedelta(seconds=60), backup_in_log=True
    )
    assert row.status == Status.WARNING and ".backup" in row.message


def test_baseline_round_trips_atomically(tmp_path):
    path = tmp_path / BASELINE_FILE
    stored = baseline_of(page(sent=3), T0)
    write_baseline(path, stored)
    assert read_baseline(path) == stored
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    assert set(json.loads(path.read_text())) == {"counters", "ts", "sent_since"}


def test_unreadable_baseline_reads_as_none(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{not json")
    assert read_baseline(path) is None
    path.write_text('{"counters": 3, "ts": "x"}')
    assert read_baseline(path) is None


# --------------------------------------------------------------------------- #
# The category end to end
# --------------------------------------------------------------------------- #


def _transport(text: str, seen: list[str] | None = None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status, text=text)

    return httpx.MockTransport(handler)


def _log(text: str | None = ""):
    calls: list[datetime] = []

    async def reader(since: datetime) -> str | None:
        calls.append(since)
        return text

    reader.calls = calls  # type: ignore[attr-defined]
    return reader


async def _run(tmp_path, text, *, now=T0, contained=False, environ=None, log=None, **kw):
    return await telemetry_gateway(
        DEPLOYED,
        cwd=tmp_path,
        transport=_transport(text, **kw),
        environ=environ or {},
        now=lambda: now,
        in_container=lambda: contained,
        log_reader=log or _log(),
    )()


async def test_not_deployed_contributes_no_rows(tmp_path):
    rows = await telemetry_gateway(
        {"deployed_services": ["telemetry_store"]}, cwd=tmp_path, in_container=lambda: False
    )()
    assert rows == []


async def test_host_run_takes_then_compares_a_baseline(tmp_path):
    first = by_name(await _run(tmp_path, page(sent=100)))
    assert first["telemetry_gateway_trend"].message == "Baseline taken"
    assert (tmp_path / BASELINE_FILE).is_file()
    later = T0 + timedelta(seconds=60)
    second = by_name(await _run(tmp_path, page(sent=100, failed=4), now=later))
    assert second["telemetry_gateway_trend"].status == Status.WARNING
    assert {r.category for r in second.values()} == {CATEGORY}


async def test_host_run_reads_the_log_since_the_stored_sample(tmp_path):
    await _run(tmp_path, page())
    log = _log("2026-10-11 bolt: renamed state to state.backup\n")
    rows = by_name(await _run(tmp_path, page(sent=200), now=T0 + timedelta(seconds=60), log=log))
    assert log.calls == [T0]
    assert rows["telemetry_gateway_trend"].status == Status.WARNING


async def test_host_tier_is_off_when_the_runtime_does_not_answer(tmp_path):
    rows = by_name(await _run(tmp_path, page(), log=_log(None)))
    assert "telemetry_gateway_trend" not in rows
    assert not (tmp_path / "var" / "health").exists()


async def test_in_container_path_runs_without_var_health(tmp_path):
    seen: list[str] = []
    environ = {tc.GATEWAY_HOST_ENV: tc.GATEWAY_SERVICE, tc.GATEWAY_PORT_ENV: "4318"}
    rows = by_name(
        await _run(tmp_path, page(refused=1), contained=True, environ=environ, seen=seen)
    )
    assert seen == [f"http://telemetry_gateway:{tc.GATEWAY_METRICS_CONTAINER_PORT}/metrics"]
    assert set(rows) == {
        "telemetry_gateway_metrics",
        "telemetry_gateway_queue",
        "telemetry_gateway_refusals",
    }
    assert all(row.status == Status.OK for row in rows.values())
    assert not (tmp_path / "var").exists()


async def test_unreachable_counters_page_is_one_warning_row(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rows = await telemetry_gateway(
        DEPLOYED,
        cwd=tmp_path,
        transport=httpx.MockTransport(handler),
        environ={},
        in_container=lambda: False,
        log_reader=_log(),
    )()
    assert [(r.name, r.status) for r in rows] == [("telemetry_gateway_metrics", Status.WARNING)]
    assert not (tmp_path / "var").exists()


async def test_host_launch_reads_the_loopback_metrics_slot(tmp_path):
    seen: list[str] = []
    await _run(tmp_path, page(), seen=seen)
    assert seen == [f"http://127.0.0.1:{tc.metrics_port(DEPLOYED)}/metrics"]
    assert metrics_url(DEPLOYED, {tc.GATEWAY_HOST_ENV: "10.0.0.5"}) == seen[0]


async def test_audit_rows_flag_non_regular_files_and_list_unfed_directories(tmp_path):
    audit = tmp_path / "var" / "audit"
    for name in ("bob", "carol", "new-user"):
        (audit / name).mkdir(parents=True)
    os.mkfifo(audit / "carol" / "x.jsonl")
    collector = tmp_path / COLLECTOR_FILE
    collector.parent.mkdir(parents=True)
    collector.write_text(
        yaml.safe_dump(
            {
                "receivers": {
                    "file_log/da": {"include": ["/audit/bob/*.jsonl"]},
                    "file_log/db": {"include": ["/audit/carol/*.jsonl"]},
                }
            }
        )
    )
    rows = by_name(await _run(tmp_path, page()))
    files = rows["telemetry_gateway_audit_files"]
    assert files.status == Status.WARNING and str(audit / "carol" / "x.jsonl") in files.details
    unfed = rows["telemetry_gateway_unfed"]
    assert unfed.status == Status.OK
    assert unfed.message == "Not fed until the next build: new-user"


async def test_audit_rows_absent_where_var_audit_is_not_visible(tmp_path):
    rows = by_name(await _run(tmp_path, page(), contained=True))
    assert "telemetry_gateway_audit_files" not in rows
    assert "telemetry_gateway_unfed" not in rows


async def test_unfed_row_absent_without_a_rendered_collector_file(tmp_path):
    (tmp_path / "var" / "audit" / "bob").mkdir(parents=True)
    rows = by_name(await _run(tmp_path, page(), contained=True))
    assert rows["telemetry_gateway_audit_files"].status == Status.OK
    assert "telemetry_gateway_unfed" not in rows


def test_baseline_file_lives_under_var_health():
    assert Path("var/health/telemetry_gateway.json") == BASELINE_FILE
