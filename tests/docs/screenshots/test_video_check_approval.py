"""Tests for the approval-log checks of the demo video recorder.

Pure parsing over record-shaped hook debug entries, plus the record read with
the client faked; safe for CI.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from docs.screenshots import video_check_approval as vca
from docs.screenshots.video_check_approval import Ask

from osprey.agent_record.client import ClientAnswer
from tests.agent_record.fakes import FakeRecordClient

MARK = datetime(2026, 9, 24, 10, 0, 0, tzinfo=UTC)


def _record(tool, status: str, detail: str | None = None, tool_use_id: str | None = None):
    rec = {"source": "hook", "ts": "2026-09-24T10:00:00", "hook": "approval", "status": status}
    if tool is not None:
        rec["tool"] = tool
    if tool_use_id is not None:
        rec["tool_use_id"] = tool_use_id
    if detail is not None:
        rec["detail"] = detail
    return rec


def _log(*records):
    """A HookLog answering *records* whatever the mark."""
    return lambda since: list(records)


def _item(body, ts: datetime):
    return {"kind": "debug", "ts": ts.isoformat().replace("+00:00", "Z"), "body": body}


def test_log_mark_is_now_in_utc():
    before = datetime.now(UTC)
    mark = vca.log_mark()
    assert before <= mark <= datetime.now(UTC)
    assert mark.tzinfo is not None


def test_find_asks_on_an_empty_log_is_empty():
    assert vca.find_asks(_log(), MARK) == []
    assert vca.ask_record_count(_log(), MARK) == 0


def test_find_asks_passes_the_mark_to_the_log():
    seen = []

    def log(since):
        seen.append(since)
        return []

    vca.find_asks(log, MARK)
    assert seen == [MARK]


def test_entry_create_then_entry_publish():
    log = _log(
        _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "toolu_c"),
        _record("mcp__ariel__entry_create", "allow", None, "toolu_x"),
        _record("mcp__ariel__entry_publish", "ask", "policy=always tool=entry_publish", "toolu_p"),
    )
    asks = vca.find_asks(log, MARK)
    assert [a.name for a in asks] == ["entry_create", "entry_publish"]
    assert [a.tool_use_id for a in asks] == ["toolu_c", "toolu_p"]


def test_execute_selective_takes_name_from_tool_field():
    log = _log(_record("mcp__python__execute", "ask", "execute_selective", "toolu_e"))
    assert vca.find_asks(log, MARK) == [Ask(tool_use_id="toolu_e", name="execute")]


def test_channel_write_selective_takes_name_from_tool_field():
    log = _log(_record("mcp__controls__channel_write", "ask", "channel_write_selective", "toolu_w"))
    assert vca.find_asks(log, MARK) == [Ask(tool_use_id="toolu_w", name="channel_write")]


def test_non_mcp_tool_name_is_kept():
    assert vca.find_asks(_log(_record("Bash", "ask", None, "toolu_b")), MARK) == [
        Ask(tool_use_id="toolu_b", name="Bash")
    ]


def test_missing_tool_use_id_is_none():
    log = _log(_record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create"))
    assert vca.find_asks(log, MARK) == [Ask(tool_use_id=None, name="entry_create")]


def test_unnamed_asks_are_counted_but_not_named():
    log = _log(
        _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "toolu_ok"),
        {"status": "ask", "tool": 42},
        {"status": "ask"},
        "not a record",
    )
    assert vca.find_asks(log, MARK) == [Ask(tool_use_id="toolu_ok", name="entry_create")]
    assert vca.ask_record_count(log, MARK) == 3


def test_hook_bodies_keeps_hook_lines_since_the_mark_oldest_first():
    old = _record("Bash", "ask", None, "old")
    new = _record("Write", "ask", None, "new")
    newer = _record("Edit", "ok", None, "newer")
    transcript = {"type": "progress", "raw": "{}"}
    items = [
        _item(newer, MARK + timedelta(seconds=2)),
        _item(transcript, MARK + timedelta(seconds=1)),
        _item(new, MARK),
        _item(old, MARK - timedelta(seconds=1)),
        {"kind": "debug", "ts": "not a time", "body": dict(new)},
        "garbage",
    ]
    assert vca.hook_bodies(items, MARK) == [new, newer]
    assert vca.hook_bodies(None, MARK) == []


@pytest.fixture
def project(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text("project_name: demo\n")
    return tmp_path


def _hold_record(monkeypatch, fake, held):
    from osprey.agent_record import client as client_mod
    from osprey.agent_record import host

    @contextmanager
    def ensure_host_record(repo, config):
        held.append((repo, config))
        yield SimpleNamespace(url="http://127.0.0.1:1", login="host", token="t")

    monkeypatch.setattr(host, "ensure_host_record", ensure_host_record)
    monkeypatch.setattr(client_mod, "RecordClient", lambda url, login, token: fake)


def test_record_log_reads_hook_debug_entries_since_the_mark(monkeypatch, project):
    from osprey.agent_record.contract import Drain

    ask = _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "tu")
    fake = FakeRecordClient(
        {"entries": ClientAnswer(data=[_item(ask, MARK + timedelta(seconds=1))], complete=True)}
    )
    held: list = []
    _hold_record(monkeypatch, fake, held)

    asks = vca.find_asks(vca.record_log(project), MARK)

    assert asks == [Ask(tool_use_id="tu", name="entry_create")]
    assert held == [(project, {"project_name": "demo"})]
    (call,) = fake.calls_to("entries")
    assert call["kind"] == "debug"
    assert call["source"] == "hook"
    assert call["since"] == MARK
    assert call["wait"] == Drain()


@pytest.mark.parametrize(
    "fake",
    [
        FakeRecordClient(unavailable="connection refused"),
        FakeRecordClient({"entries": ClientAnswer(data=[], complete=False, reason="scan")}),
    ],
    ids=["unavailable", "incomplete"],
)
def test_record_log_refuses_an_unanswered_read(monkeypatch, project, fake):
    _hold_record(monkeypatch, fake, [])

    with pytest.raises(vca.RecordUnavailable):
        vca.find_asks(vca.record_log(project), MARK)
