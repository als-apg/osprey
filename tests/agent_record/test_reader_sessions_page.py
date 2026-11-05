"""``list_sessions`` as the session list reads it: activity, prompts, pages.

Pins Decision 5's record rules: a session's activity time (its ``last_ts`` and
sort place) comes from its transcript rows only, so the OTLP rows a chat child
keeps writing under the key after ``/clear`` never move the older transcript up;
the first prompt is the first one :func:`is_bookkeeping_prompt` passes (the
``/clear`` echo is not it); ``message_count`` counts main-stream prompts and
answers; ``before`` pages back through a backend bound, so it reaches sessions
older than the newest scan of rows and never repeats a session an earlier page
listed; and a session a cut scan reached with no prompt is kept ``truncated``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import reader as rd
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.mcp_server import session as session_record
from tests.agent_record.test_identities import _config

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
ME = "alice"
T1 = "11111111-aaaa-4aaa-8aaa-111111111111"  # the key's first transcript (= the key)
T2 = "22222222-bbbb-4bbb-8bbb-222222222222"  # the transcript after /clear
OLD = "33333333-cccc-4ccc-8ccc-333333333333"


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _cc(session: str, n: int, kind: str, t: float, body: dict | None = None, **fields):
    return c.Entry(
        key=f"cc:{ME}:{session}:u{n}:0",
        kind=kind,
        ts=_at(t),
        identity=ME,
        body=body if body is not None else {},
        inserted_at=_at(t),
        session_id=session,
        **fields,
    )


def _otel(session: str, n: int, t: float) -> c.Entry:
    return c.Entry(
        key=f"otel:{ME}:{session}:api:{n}",
        kind=c.KIND_USAGE,
        ts=_at(t),
        identity=ME,
        body={"tokens": n},
        inserted_at=_at(t),
        session_id=session,
    )


def _reader(backend, **limits) -> rd.RecordReader:
    return rd.RecordReader(backend, _config(), limits=rd.Limits(**limits))


@pytest.fixture
def backend(tmp_path: Path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        yield b


def _ids(answer: rd.ReadAnswer) -> list[str]:
    return [row["session_id"] for row in answer.data]


# --- the record rules ------------------------------------------------------------------


def test_the_real_first_prompt_after_clear_and_the_message_count(backend):
    backend.append(
        [
            _cc(T1, 1, c.KIND_PROMPT, 1, {"text": "tune the ring"}),
            _cc(T1, 2, c.KIND_ANSWER, 2, {"text": "tuned"}),
            # After /clear: the command echo, its output and a meta prompt come first.
            _cc(T2, 1, c.KIND_PROMPT, 10, {"text": "<command-name>/clear</command-name>"}),
            _cc(
                T2, 2, c.KIND_PROMPT, 11, {"text": "<local-command-stdout></local-command-stdout>"}
            ),
            _cc(T2, 3, c.KIND_PROMPT, 12, {"text": "caveat", "is_meta": True}),
            _cc(T2, 4, c.KIND_PROMPT, 13, {"text": "[Request interrupted by user]"}),
            _cc(T2, 5, c.KIND_PROMPT, 14, {"text": "read the BPMs"}),
            _cc(T2, 6, c.KIND_THINKING, 15, {"text": "hm"}),
            _cc(T2, 7, c.KIND_TOOL_CALL, 16, {"id": "tu1", "name": "Read"}, tool_use_id="tu1"),
            _cc(T2, 8, c.KIND_TOOL_RESULT, 17, {"content": "x"}, tool_use_id="tu1"),
            _cc(T2, 9, c.KIND_ANSWER, 18, {"text": "here"}),
            # A subagent's prompt and answer are not the session's messages.
            _cc(T2, 10, c.KIND_PROMPT, 19, {"text": "sub"}, agent_id="ag1"),
            _cc(T2, 11, c.KIND_ANSWER, 20, {"text": "sub done"}, agent_id="ag1"),
            _cc(T2, 12, c.KIND_PROMPT, 21, {"text": "and the tunes"}),
        ]
    )
    rows = {r["session_id"]: r for r in _reader(backend).list_sessions(c.All()).data}
    assert rows[T2]["prompt"] == "read the BPMs"
    assert rows[T2]["message_count"] == 3  # two prompts, one answer
    assert rows[T2]["truncated"] is False
    assert rows[T1]["prompt"] == "tune the ring" and rows[T1]["message_count"] == 2
    assert "entries" not in rows[T1]


def test_otel_rows_under_the_key_after_clear_do_not_move_the_old_transcript(backend):
    backend.append(
        [
            _cc(T1, 1, c.KIND_PROMPT, 1, {"text": "first"}),
            _cc(T1, 2, c.KIND_ANSWER, 2, {"text": "ok"}),
            _cc(T2, 1, c.KIND_PROMPT, 10, {"text": "second"}),
            _cc(T2, 2, c.KIND_ANSWER, 11, {"text": "ok"}),
            # The chat child's OTLP rows carry the key, T1, long after T1's last turn.
            _otel(T1, 1, 50),
            _otel(T1, 2, 60),
        ]
    )
    rows = _reader(backend).list_sessions(c.All()).data
    assert [r["session_id"] for r in rows] == [T2, T1]
    assert rows[1]["last_ts"] == _at(2)
    assert rows[1]["first_ts"] == _at(1)


def test_a_promptless_run_still_sorts_by_its_rows(backend):
    accept = c.Entry(
        key=c.run_key("run-9", c.RUN_ACCEPT),
        kind=c.KIND_RUN,
        ts=_at(30),
        identity="dispatch-worker-1",
        body={"phase": c.RUN_ACCEPT, "run_id": "run-9", "session_id": OLD},
        inserted_at=_at(30),
        session_id=OLD,
        run_id="run-9",
    )
    backend.append([_cc(T1, 1, c.KIND_PROMPT, 1, {"text": "first"}), accept])
    rows = _reader(backend).list_sessions(c.All()).data
    assert [r["session_id"] for r in rows] == [OLD, T1]
    assert rows[0]["prompt"] is None and rows[0]["run_ids"] == ["run-9"]
    assert rows[0]["truncated"] is False


# --- before ----------------------------------------------------------------------------


def _busy_newer_session(start: float, n: int) -> list[c.Entry]:
    rows = [_cc(T2, 0, c.KIND_PROMPT, start, {"text": "newer"})]
    rows += [_cc(T2, i, c.KIND_ANSWER, start + i, {"text": f"a{i}"}) for i in range(1, n)]
    return rows


def test_before_reaches_sessions_older_than_the_newest_scan(backend):
    backend.append(
        [
            _cc(OLD, 1, c.KIND_PROMPT, 1, {"text": "the old one"}),
            _cc(OLD, 2, c.KIND_ANSWER, 2, {"text": "ok"}),
            *_busy_newer_session(100, 30),
        ]
    )
    reader = _reader(backend, scan_rows=10)
    newest = reader.list_sessions(c.All())
    assert not newest.complete and "scan limit" in newest.reason
    assert _ids(newest) == [T2]
    # The cut reached T2 before its first prompt: kept, flagged, never named.
    assert newest.data[0]["prompt"] is None and newest.data[0]["truncated"] is True

    older = reader.list_sessions(c.All(), before=_at(100))
    assert older.complete
    assert _ids(older) == [OLD]
    assert older.data[0]["prompt"] == "the old one"
    assert older.data[0]["truncated"] is False


def test_before_never_repeats_a_session_whose_activity_continues_past_it(backend):
    backend.append(
        [
            _cc(OLD, 1, c.KIND_PROMPT, 1, {"text": "old"}),
            _cc(T1, 1, c.KIND_PROMPT, 5, {"text": "spans the bound"}),
            _cc(T1, 2, c.KIND_ANSWER, 50, {"text": "late"}),
            # T2 ends before the bound; only its key's OTLP rows are after it.
            _cc(T2, 1, c.KIND_PROMPT, 8, {"text": "ends before"}),
            _otel(T2, 1, 60),
        ]
    )
    page = _reader(backend).list_sessions(c.All(), before=_at(20))
    assert _ids(page) == [T2, OLD]


def test_before_pages_the_list_with_limit(backend):
    sessions = [f"{n:08d}-0000-4000-8000-000000000000" for n in range(5)]
    backend.append(
        [_cc(s, 1, c.KIND_PROMPT, 10 * (i + 1), {"text": s}) for i, s in enumerate(sessions)]
    )
    reader = _reader(backend)
    first = reader.list_sessions(c.All(), limit=2)
    assert first.more and _ids(first) == sessions[:2:-1][:2]
    boundary = first.data[-1]["last_ts"]
    second = reader.list_sessions(c.All(), limit=2, before=boundary)
    assert _ids(second) == [sessions[2], sessions[1]] and second.more
    third = reader.list_sessions(c.All(), limit=2, before=second.data[-1]["last_ts"])
    assert _ids(third) == [sessions[0]] and not third.more


def test_is_bookkeeping_prompt_is_one_rule_re_exported_for_the_session_readers():
    assert session_record.is_bookkeeping_prompt is rd.is_bookkeeping_prompt
    assert rd.is_bookkeeping_prompt({"text": "<command-message>skill</command-message>"})
    assert rd.is_bookkeeping_prompt({"text": "x", "is_meta": True})
    assert not rd.is_bookkeeping_prompt({"text": "hello"})
    assert not rd.is_bookkeeping_prompt({"text": ""})


def test_the_bookkeeping_rule_is_the_transcript_readers_rule():
    from osprey.mcp_server.workspace import transcript_reader as tr

    assert rd.INTERRUPT_MARKER == tr.INTERRUPT_MARKER
    assert set(rd.BOOKKEEPING_PREFIXES) == set(tr._BOOKKEEPING_PREFIXES)


def test_session_ids_lists_only_the_sessions_named(backend):
    backend.append(
        [
            _cc(T1, 1, c.KIND_PROMPT, 1, {"text": "one"}),
            _cc(T2, 1, c.KIND_PROMPT, 2, {"text": "two"}),
            _cc(OLD, 1, c.KIND_PROMPT, 3, {"text": "three"}),
        ]
    )
    reader = _reader(backend)
    unknown = "44444444-dddd-4ddd-8ddd-444444444444"
    answer = reader.list_sessions(c.All(), session_ids={T1, OLD, unknown})
    assert answer.complete and _ids(answer) == [OLD, T1]
    assert _ids(reader.list_sessions(c.All(), session_ids=())) == []


def test_an_unreadable_prompt_is_not_counted_and_flags_the_session(backend):
    backend.append(
        [
            # Chunk 1 of 2 without chunk 0: the prompt's body cannot be read.
            _cc(T1, 1, c.KIND_PROMPT, 1, {c.CHUNK_PART_FIELD: "xx"}, chunk=c.Chunk(1, 2)),
            _cc(T1, 2, c.KIND_ANSWER, 2, {"text": "ok"}),
        ]
    )
    row = _reader(backend).list_sessions(c.All()).data[0]
    assert row["prompt"] is None and row["truncated"] is True
    assert row["message_count"] == 1  # the answer only


def test_truncated_follows_the_cut_not_the_wording_of_its_reason(backend, monkeypatch):
    backend.append(_busy_newer_session(100, 30))
    reader = _reader(backend, scan_rows=10)
    original = rd._Ctx.incomplete
    monkeypatch.setattr(rd._Ctx, "incomplete", lambda self, reason: original(self, "reworded"))
    answer = reader.list_sessions(c.All())
    assert answer.reason == "reworded"
    assert answer.data[0]["prompt"] is None and answer.data[0]["truncated"] is True
