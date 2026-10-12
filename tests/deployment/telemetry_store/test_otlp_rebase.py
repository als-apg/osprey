"""The OTLP replay's time rebase: one offset, every time field, inside the clamp window."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from osprey.deployment import telemetry_config as tc
from tests.e2e._otlp_replay import NEWEST_BEFORE_NOW, otlp_times, rebase_otlp_times

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
)
NOW = datetime.datetime(2026, 10, 11, 12, 0, tzinfo=datetime.UTC)
NS = 1_000_000_000
#: The gateway's clamp: older than the retention window, or more than an hour
#: ahead, is dropped (``telemetry_gateway._clamp_processor``).
FUTURE_SKEW_NS = 3600 * NS


def _ns(when: datetime.datetime) -> int:
    return int(when.timestamp()) * NS


def _every_kind(base: int) -> dict:
    """One request of each signal, carrying every OTLP time field once."""
    return {
        "requests": [
            {
                "signal": "logs",
                "body": {
                    "resourceLogs": [
                        {
                            "scopeLogs": [
                                {
                                    "logRecords": [
                                        {
                                            "timeUnixNano": str(base),
                                            "observedTimeUnixNano": str(base + 7),
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
            },
            {
                "signal": "traces",
                "body": {
                    "resourceSpans": [
                        {
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "startTimeUnixNano": str(base + 100),
                                            "endTimeUnixNano": str(base + 900),
                                            "events": [{"timeUnixNano": str(base + 500)}],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
            },
            {
                "signal": "metrics",
                "body": {
                    "resourceMetrics": [
                        {
                            "scopeMetrics": [
                                {
                                    "metrics": [
                                        {
                                            "sum": {
                                                "dataPoints": [
                                                    {
                                                        "startTimeUnixNano": str(base - 50),
                                                        "timeUnixNano": str(base + 1000),
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
            },
        ]
    }


def _fields(payload: dict) -> dict[str, int]:
    logs, traces, metrics = (r["body"] for r in payload["requests"])
    record = logs["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    span = traces["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    point = metrics["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["sum"]["dataPoints"][0]
    return {
        "log.time": int(record["timeUnixNano"]),
        "log.observed": int(record["observedTimeUnixNano"]),
        "span.start": int(span["startTimeUnixNano"]),
        "span.end": int(span["endTimeUnixNano"]),
        "span.event": int(span["events"][0]["timeUnixNano"]),
        "point.start": int(point["startTimeUnixNano"]),
        "point.time": int(point["timeUnixNano"]),
    }


def test_every_time_field_moves_by_one_offset_and_the_newest_lands_five_minutes_back() -> None:
    base = _ns(NOW - datetime.timedelta(days=900))
    before = _fields(_every_kind(base))
    after = _fields(rebase_otlp_times(_every_kind(base), NOW))
    offsets = {name: after[name] - before[name] for name in before}
    assert len(set(offsets.values())) == 1, offsets
    assert offsets["log.time"] != 0
    assert max(after.values()) == _ns(NOW) - int(NEWEST_BEFORE_NOW.total_seconds()) * NS


def test_values_keep_their_type_and_an_unset_time_stays_unset() -> None:
    payload = {"logRecords": [{"timeUnixNano": "0", "observedTimeUnixNano": 1_000}]}
    rebased = rebase_otlp_times(payload, NOW)
    record = rebased["logRecords"][0]
    assert record["timeUnixNano"] == "0"
    assert isinstance(record["observedTimeUnixNano"], int)
    assert payload["logRecords"][0]["observedTimeUnixNano"] == 1_000, "the input was changed"


def test_a_payload_without_a_time_is_refused() -> None:
    with pytest.raises(ValueError, match="no OTLP time"):
        rebase_otlp_times({"resourceLogs": []}, NOW)


def test_the_capture_dated_two_years_back_lands_inside_the_clamp_window() -> None:
    capture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    original = otlp_times(capture)
    assert original, "the capture carries no OTLP time"
    two_years_back = NOW - datetime.timedelta(days=730)
    aged = rebase_otlp_times(capture, two_years_back)
    assert max(otlp_times(aged)) < _ns(NOW - datetime.timedelta(days=700))

    replayed = otlp_times(rebase_otlp_times(aged, NOW))
    oldest_kept = _ns(NOW) - tc.retention_days({}) * 86_400 * NS
    assert all(oldest_kept < t <= _ns(NOW) + FUTURE_SKEW_NS for t in replayed)
    assert max(replayed) == _ns(NOW) - int(NEWEST_BEFORE_NOW.total_seconds()) * NS
    deltas = [t - min(original) for t in original]
    assert [t - min(replayed) for t in replayed] == deltas, "an interval changed"
