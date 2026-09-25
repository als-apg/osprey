"""Completed days of the telemetry store are exported once, per type and stream."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from osprey.services.archive import run as archive_run
from osprey.services.archive.manifest import MANIFEST_NAME, load_state
from osprey.services.archive.run import run_pass
from osprey.services.archive.telemetry_export import (
    PAGE_SIZE,
    TelemetryExporter,
    pending_days,
)

ENV = {"ZO_INGEST_USER_EMAIL": "ingest@osprey.local", "ZO_INGEST_SA_TOKEN": "tok"}
# 2026-09-24 02:00Z: 2026-09-23 is complete and past the one-hour grace.
NOW = datetime(2026, 9, 24, 2, 0, 0, tzinfo=UTC)


class FakeStore:
    """An in-memory telemetry store answering the two endpoints the export reads."""

    def __init__(self, streams: dict[str, dict[str, int]], fail_stream: str | None = None):
        # type -> stream -> hits per hour
        self.streams = streams
        self.fail_stream = fail_stream
        self.searches: list[dict] = []
        self.auth: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("Authorization", ""))
        kind = request.url.params["type"]
        if request.method == "GET" and request.url.path == "/api/default/streams":
            names = sorted(self.streams.get(kind, {}))
            return httpx.Response(200, json={"list": [{"name": n} for n in names]})
        if request.method == "POST" and request.url.path == "/api/default/_search":
            query = json.loads(request.content)["query"]
            self.searches.append({"type": kind, **query})
            stream = query["sql"].split('"')[1]
            if stream == self.fail_stream:
                return httpx.Response(500, json={"error": "boom"})
            per_hour = self.streams[kind][stream]
            remaining = max(per_hour - query["from"], 0)
            count = min(remaining, query["size"])
            hits = [
                {"stream": stream, "start": query["start_time"], "n": query["from"] + i}
                for i in range(count)
            ]
            return httpx.Response(200, json={"hits": hits, "total": count})
        return httpx.Response(404)


def _exporter(store: FakeStore, **kwargs) -> TelemetryExporter:
    return TelemetryExporter(
        url="http://openobserve:5080",
        environ=kwargs.pop("environ", ENV),
        transport=httpx.MockTransport(store.handler),
        **kwargs,
    )


@pytest.fixture
def trees(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    os.chmod(dest, 0o700)
    return sources, dest


def _lines(dest: Path, day: str = "2026-09-24") -> list[dict]:
    text = (dest / day / MANIFEST_NAME).read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_a_completed_day_exports_logs_and_traces_per_stream(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 1, "claude_code": 2}, "traces": {"default": 1}})

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == ["2026-09-23"]
    assert result.errors == []
    base = dest / "2026-09-24/openobserve/2026-09-23"
    assert len((base / "logs/default.jsonl").read_text().splitlines()) == 24
    assert len((base / "logs/claude_code.jsonl").read_text().splitlines()) == 48
    assert len((base / "traces/default.jsonl").read_text().splitlines()) == 24
    day_line = next(r for r in _lines(dest) if r["kind"] == "telemetry_day")
    assert day_line["rows"] == {
        "logs/claude_code": 48,
        "logs/default": 24,
        "traces/default": 24,
    }
    assert {s["type"] for s in store.searches} == {"logs", "traces"}


def test_pages_until_a_short_page(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": PAGE_SIZE + 5}})

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    first_hour = [s for s in store.searches if s["start_time"] == store.searches[0]["start_time"]]
    assert [s["from"] for s in first_hour] == [0, PAGE_SIZE]
    assert all(s["size"] == PAGE_SIZE for s in store.searches)
    exported = dest / "2026-09-24/openobserve/2026-09-23/logs/default.jsonl"
    assert len(exported.read_text().splitlines()) == 24 * (PAGE_SIZE + 5)


def test_the_open_day_is_not_exported():
    just_before_grace = datetime(2026, 9, 24, 0, 59, 0, tzinfo=UTC)

    assert pending_days(just_before_grace, set(), 1) == [datetime(2026, 9, 22).date()]
    assert pending_days(NOW, set(), 1) == [datetime(2026, 9, 23).date()]


def test_an_exported_day_is_never_exported_again(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 1}})

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))
    searches = len(store.searches)
    second = run_pass(
        sources,
        dest,
        now=datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC),
        telemetry=_exporter(store, backfill_days=1),
    )

    assert second.telemetry_days == []
    assert len(store.searches) == searches
    assert load_state(dest).telemetry_days == {"2026-09-23"}


def test_a_failed_stream_leaves_the_day_unrecorded(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 1, "broken": 1}}, fail_stream="broken")

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == []
    assert [e["source"] for e in result.errors] == ["openobserve/2026-09-23"]
    assert load_state(dest).telemetry_days == set()
    assert not (dest / "2026-09-24/openobserve").exists()
    assert list((dest / ".incoming").iterdir()) == []


def test_a_day_that_cannot_be_placed_is_recorded_as_an_error_and_left_unrecorded(
    trees, monkeypatch
):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 1, "other": 1}})

    def _full_disk(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(archive_run.ArchiveTree, "place", _full_disk)

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))

    assert result.telemetry_days == []
    assert [e["source"] for e in result.errors] == ["openobserve/2026-09-23"]
    assert load_state(dest).telemetry_days == set()
    assert list((dest / ".incoming").iterdir()) == []


def test_backfill_stops_at_the_limit(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 0}})

    result = run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=3))

    assert result.telemetry_days == ["2026-09-21", "2026-09-22", "2026-09-23"]


def test_credentials_come_from_the_environment_only(trees):
    sources, dest = trees
    store = FakeStore({"logs": {"default": 0}})

    run_pass(sources, dest, now=NOW, telemetry=_exporter(store, backfill_days=1))
    expected = "Basic " + base64.b64encode(b"ingest@osprey.local:tok").decode()
    assert store.auth == {expected}

    store_without = FakeStore({"logs": {"default": 0}})
    result = run_pass(
        sources,
        dest,
        now=datetime(2026, 9, 25, 2, 0, 0, tzinfo=UTC),
        telemetry=_exporter(store_without, backfill_days=1, environ={}),
    )

    assert store_without.searches == []
    assert result.telemetry_days == []
    assert "ZO_INGEST_SA_TOKEN" in result.errors[0]["error"]
