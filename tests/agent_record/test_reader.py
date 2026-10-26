"""The one reader, on the embedded backend.

Pins every read-API method (``list_sessions`` with promptless runs,
``session_users``, ``timeline`` with thinking inline, subagents nested under
their Agent call and each call joined to its decision and result,
``agent_timeline``, ``full_entry`` with chunk reassembly, ``run``,
``run_timeline``, ``recent_audit``, ``bridge_history``, ``entries``) under
each scope (``Own``, ``All``, ``Reader`` kinds and days back, ``Source`` with
the runs its own entries name and both ``narrow`` forms); the copied preflight
matching (60 s window, same person) and the own-view filter; the row, scan,
byte and time limits; a wait mark not yet passed (a drain, a transcript mark,
and the drain of each run's session at read time) answering ``complete: false``
within the 10 s bound; a run asked for or named by a bridge turn with no run
entry yet answering incomplete; and a backend that raises answering ``unavailable``,
never a partial answer marked complete.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import reader as rd
from osprey.agent_record.backends.embedded import EmbeddedBackend
from tests.agent_record.test_identities import _config

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
NOW = T0 + timedelta(hours=1)

S1 = "sess-alice-1"  # alice, terminal transcript with a subagent and a queue submit
S2 = "sess-bob-1"  # bob, terminal transcript holding an incomplete chunk set
S3 = "sess-run-1"  # dispatch-worker-1 session of run R1 (bob), asked over teams
S4 = "sess-run-2"  # dispatch-worker-2 promptless run R2 (alice), still running
R1 = "run-1"
R2 = "run-2"
W1 = "dispatch-worker-1"
W2 = "dispatch-worker-2"
BRIDGE = "teams-bridge"
QS = "queueserver"
S3_FILE = f"proj/{S3}.jsonl"


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _e(
    identity: str,
    key: str,
    kind: str,
    t: float,
    body: dict | None = None,
    *,
    arrived: float | None = None,
    **fields,
) -> c.Entry:
    return c.Entry(
        key=key,
        kind=kind,
        ts=_at(t),
        identity=identity,
        body=body if body is not None else {},
        inserted_at=_at(t if arrived is None else arrived),
        **fields,
    )


def _cc(identity: str, session: str, n: int, kind: str, t: float, body=None, **fields):
    return _e(
        identity, f"cc:{identity}:{session}:u{n}:0", kind, t, body, session_id=session, **fields
    )


def _accept(identity: str, run_id: str, session: str, requester: str | None, t: float):
    body = {"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session}
    if requester:
        body["requester"] = requester
    return _e(
        identity,
        c.run_key(run_id, c.RUN_ACCEPT),
        c.KIND_RUN,
        t,
        body,
        session_id=session,
        run_id=run_id,
    )


def _terminal(identity: str, run_id: str, t: float):
    body: dict = {"phase": c.RUN_TERMINAL, "status": "done", "cost": 0.1}
    return _e(identity, c.run_key(run_id, c.RUN_TERMINAL), c.KIND_RUN, t, body, run_id=run_id)


BIG_TEXT = "the long answer " * 200


def _seed() -> list[c.Entry]:
    a = "alice"
    big = _cc(a, S1, 90, c.KIND_ANSWER, 90, {"text": BIG_TEXT})
    chunks = c.split_entry(big, 400)
    assert len(chunks) > 2
    broken = c.split_entry(_cc("bob", S2, 3, c.KIND_ANSWER, 3, {"text": BIG_TEXT}), 400)
    return [
        # alice S1: prompt, thinking, answer, an Agent call with a subagent stream,
        # a queue submit joined to its decision and result, then a final answer.
        _cc(a, S1, 1, c.KIND_PROMPT, 1, {"text": "scan the ring"}),
        _cc(a, S1, 2, c.KIND_THINKING, 2, {"text": "let me think"}),
        _cc(a, S1, 3, c.KIND_ANSWER, 3, {"text": "on it"}),
        _cc(
            a,
            S1,
            4,
            c.KIND_TOOL_CALL,
            4,
            {"id": "tu-agent", "name": "Agent"},
            tool_use_id="tu-agent",
        ),
        _cc(a, S1, 5, c.KIND_PROMPT, 5, {"text": "sub task"}, agent_id="ag1"),
        _cc(a, S1, 6, c.KIND_THINKING, 6, {"text": "sub thinks"}, agent_id="ag1"),
        _cc(
            a,
            S1,
            7,
            c.KIND_TOOL_CALL,
            7,
            {"id": "tu-read", "name": "Read"},
            tool_use_id="tu-read",
            agent_id="ag1",
        ),
        _cc(
            a,
            S1,
            8,
            c.KIND_TOOL_RESULT,
            8,
            {"content": "file"},
            tool_use_id="tu-read",
            agent_id="ag1",
        ),
        _cc(a, S1, 9, c.KIND_ANSWER, 9, {"text": "sub done"}, agent_id="ag1"),
        _cc(
            a,
            S1,
            10,
            c.KIND_SUBAGENT,
            10,
            {"agent_id": "ag1"},
            tool_use_id="tu-agent",
            agent_id="ag1",
        ),
        _cc(
            a,
            S1,
            20,
            c.KIND_TOOL_CALL,
            20,
            {"id": "tu-q", "name": "mcp__bluesky__queue_add"},
            tool_use_id="tu-q",
        ),
        _e(
            a,
            f"otel:{a}:{S1}:tool_decision:tu-q",
            c.KIND_DECISION,
            20.5,
            {"decision": "accept"},
            session_id=S1,
            tool_use_id="tu-q",
        ),
        _cc(a, S1, 21, c.KIND_TOOL_RESULT, 21, {"content": "queued"}, tool_use_id="tu-q"),
        *chunks,
        # queueserver preflight on behalf of alice: one within 60 s of her submit,
        # one with no submit before it.
        _e(
            QS,
            f"sp:{QS}:pf1",
            c.KIND_AUDIT,
            50,
            {"surface": "preflight", "plan": "a"},
            on_behalf_of=a,
        ),
        _e(
            QS,
            f"sp:{QS}:pf2",
            c.KIND_AUDIT,
            500,
            {"surface": "preflight", "plan": "b"},
            on_behalf_of=a,
        ),
        _e(a, f"sp:{a}:fb1", c.KIND_FEEDBACK, 30, {"text": "nice"}),
        _e(a, f"sp:{a}:fb2", c.KIND_FEEDBACK, 40, {"text": "nicer"}),
        _e(a, f"sp:{a}:fb3", c.KIND_FEEDBACK, 45, {"text": "nicest"}, arrived=-5 * 86400),
        # bob S2: a prompt and an answer whose chunk 1 never arrived.
        _cc("bob", S2, 1, c.KIND_PROMPT, 1, {"text": "hello"}),
        *[ch for ch in broken if ch.chunk.index != 1],
        # run R1: accepted for bob on worker 1, asked over teams, finished.
        _accept(W1, R1, S3, "bob", 100),
        _cc(W1, S3, 1, c.KIND_PROMPT, 101, {"text": "<rendered payload>"}),
        _cc(W1, S3, 2, c.KIND_TOOL_CALL, 102, {"id": "tu-9", "name": "Bash"}, tool_use_id="tu-9"),
        _cc(W1, S3, 3, c.KIND_TOOL_RESULT, 103, {"content": "ok"}, tool_use_id="tu-9"),
        _cc(W1, S3, 4, c.KIND_ANSWER, 104, {"text": "final answer"}),
        _terminal(W1, R1, 105),
        _e(
            BRIDGE,
            f"sp:{BRIDGE}:t1",
            c.KIND_BRIDGE_TURN,
            99,
            {"conversation": "c1", "run_id": R1, "asked_by": "bob", "question": "status?"},
            run_id=R1,
        ),
        # run R2: promptless, accepted for alice on worker 2, no terminal yet.
        _accept(W2, R2, S4, "alice", 200),
    ]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class _Drains:
    """A drainer that records each drain it is asked for; one passes once *ready* says so."""

    def __init__(self, ready=lambda call: False) -> None:
        self.calls: list[tuple] = []
        self.ready = ready

    def __call__(self, scope, session_id, also):
        call = (scope, session_id, also)
        self.calls.append(call)
        return SimpleNamespace(passed=lambda: self.ready(call))


def _draining(backend, config, drain, clock=None) -> tuple[rd.RecordReader, _Clock]:
    clock = clock or _Clock()
    reader = rd.RecordReader(
        backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW, drain=drain
    )
    return reader, clock


@pytest.fixture
def config() -> dict:
    return _config(workers=2, bridges=("teams_bridge",), lanes=True)


@pytest.fixture
def backend(tmp_path: Path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        b.append(_seed())
        yield b


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def reader(backend, config, clock) -> rd.RecordReader:
    return rd.RecordReader(backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW)


def _keys(items) -> list[str]:
    return [i.key for i in items]


def _flat(nodes) -> list[str]:
    out = []
    for node in nodes:
        out.extend(i.key for i in node.walk())
    return out


# --- timeline ------------------------------------------------------------------------


def test_timeline_keeps_thinking_inline_nests_subagents_and_joins_calls(reader):
    answer = reader.timeline(c.Own("alice"), S1)
    assert answer.available and answer.complete, answer.reason
    nodes = answer.data
    kinds = [n.item.kind for n in nodes]
    assert kinds == [
        c.KIND_PROMPT,
        c.KIND_THINKING,
        c.KIND_ANSWER,
        c.KIND_TOOL_CALL,
        c.KIND_TOOL_CALL,
        c.KIND_ANSWER,
    ]
    agent = nodes[3]
    assert agent.result.kind == c.KIND_SUBAGENT and agent.agent_id == "ag1"
    sub = agent.children
    assert [n.item.kind for n in sub] == [
        c.KIND_PROMPT,
        c.KIND_THINKING,
        c.KIND_TOOL_CALL,
        c.KIND_ANSWER,
    ]
    assert sub[2].result.body == {"content": "file"}
    submit = nodes[4]
    assert submit.decision.kind == c.KIND_DECISION
    assert submit.decision.body == {"decision": "accept"}
    assert submit.result.body == {"content": "queued"}
    # The chunked answer is reassembled byte-identical, once.
    assert nodes[5].item.body == {"text": BIG_TEXT}
    assert nodes[5].item.entry.chunk == c.UNSPLIT
    json_nodes = answer.to_json()["data"]
    assert json_nodes[3]["subagent"]["agent_id"] == "ag1"
    assert json_nodes[4]["decision"]["tool_use_id"] == "tu-q"


def test_timeline_of_another_persons_session_is_empty_under_own(reader):
    answer = reader.timeline(c.Own("bob"), S1)
    assert answer.available and answer.complete
    assert answer.data == []


def test_a_subagent_with_no_call_yet_is_its_own_group(backend, config, clock):
    backend.append([_cc("alice", S1, 300, c.KIND_ANSWER, 300, {"text": "late"}, agent_id="ag9")])
    reader = rd.RecordReader(backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW)
    nodes = reader.timeline(c.Own("alice"), S1).data
    orphan = nodes[-1]
    assert orphan.item is None and orphan.agent_id == "ag9"
    assert orphan.to_json()["orphan"] is True
    assert [n.item.body["text"] for n in orphan.children] == ["late"]


def test_agent_timeline_returns_the_subagent_stream(reader):
    answer = reader.agent_timeline(c.Own("alice"), S1, "ag1")
    assert answer.complete
    assert [n.item.kind for n in answer.data] == [
        c.KIND_PROMPT,
        c.KIND_THINKING,
        c.KIND_TOOL_CALL,
        c.KIND_ANSWER,
    ]
    assert answer.data[2].result.kind == c.KIND_TOOL_RESULT


def test_timeline_row_limit_pages(reader):
    answer = reader.timeline(c.Own("alice"), S1, limit=3)
    assert answer.more and answer.complete
    assert [n.item.kind for n in answer.data] == [c.KIND_PROMPT, c.KIND_THINKING, c.KIND_ANSWER]


def test_an_incomplete_chunk_set_is_flagged_and_the_answer_is_not_complete(reader):
    answer = reader.timeline(c.All(), S2)
    assert answer.available and not answer.complete
    assert "chunk" in answer.reason
    broken = [n.item for n in answer.data if n.item.kind == c.KIND_ANSWER]
    assert len(broken) == 1 and broken[0].incomplete and broken[0].body is None


# --- full_entry ------------------------------------------------------------------------


def test_full_entry_reassembles_and_scopes(reader):
    key = "cc:alice:sess-alice-1:u90:0"
    answer = reader.full_entry(c.Own("alice"), key)
    assert answer.complete and answer.data.body == {"text": BIG_TEXT}
    hidden = reader.full_entry(c.Own("bob"), key)
    assert hidden.available and hidden.data is None


def test_full_entry_of_an_incomplete_set_is_unavailable(reader):
    answer = reader.full_entry(c.All(), "cc:bob:sess-bob-1:u3:0")
    assert not answer.available and not answer.complete
    assert answer.to_json()["status"] == "unavailable"


# --- list_sessions / session_users -----------------------------------------------------


def test_list_sessions_under_own_includes_promptless_runs(reader):
    answer = reader.list_sessions(c.Own("alice"))
    assert answer.complete
    rows = {r["session_id"]: r for r in answer.data}
    assert set(rows) == {S1, S4}
    assert rows[S1]["prompt"] == "scan the ring"
    assert rows[S4]["prompt"] is None and rows[S4]["run_ids"] == [R2]
    assert rows[S4]["persons"] == ["alice"]


def test_list_sessions_under_all_and_its_limit(reader):
    rows = reader.list_sessions(c.All()).data
    assert [r["session_id"] for r in rows] == [S4, S3, S1, S2]
    assert rows[1]["persons"] == ["bob"] and rows[1]["prompt"] == "<rendered payload>"
    page = reader.list_sessions(c.All(), limit=2)
    assert page.more and [r["session_id"] for r in page.data] == [S4, S3]


def test_session_list_cap_defaults_to_the_old_page_limits():
    assert rd.PAGE_ROWS == 500 and rd.SESSION_ROWS == 200 and rd.WAIT_SECONDS == 10.0


def test_session_users(reader):
    users = reader.session_users(c.All()).data
    assert [(u["person"], u["sessions"]) for u in users] == [("alice", 2), ("bob", 2)]
    assert [u["person"] for u in reader.session_users(c.Own("bob")).data] == ["bob"]


# --- Reader scope ------------------------------------------------------------------------


def test_reader_scope_keeps_its_kinds_and_days_back(reader):
    scope = c.Reader("ops", kinds=frozenset({c.KIND_FEEDBACK}), days_back=2)
    answer = reader.entries(scope, c.KIND_FEEDBACK)
    assert _keys(answer.data) == ["sp:alice:fb2", "sp:alice:fb1"]
    assert reader.entries(scope, c.KIND_AUDIT).data == []
    wide = c.Reader("ops", kinds=frozenset({c.KIND_FEEDBACK}))
    assert len(reader.entries(wide, c.KIND_FEEDBACK).data) == 3
    assert reader.timeline(scope, S1).data == []


# --- entries / recent_audit / preflight -------------------------------------------------


def test_entries_newest_first_with_since_and_limit(reader):
    answer = reader.entries(c.Own("alice"), c.KIND_FEEDBACK)
    assert _keys(answer.data) == ["sp:alice:fb3", "sp:alice:fb2", "sp:alice:fb1"]
    page = reader.entries(c.Own("alice"), c.KIND_FEEDBACK, since=_at(35), limit=1)
    assert _keys(page.data) == ["sp:alice:fb3"] and page.more
    assert reader.entries(c.Own("bob"), c.KIND_FEEDBACK).data == []
    with pytest.raises(ValueError):
        reader.entries(c.All(), "telemetry")


def test_entries_source_matches_the_body_before_the_limit(reader, backend):
    backend.append(
        [
            _e("alice", "sp:alice:dbg1", c.KIND_DEBUG, 60, {"source": "hook", "n": 1}),
            _e("alice", "sp:alice:dbg2", c.KIND_DEBUG, 61, {"source": "sdk", "n": 2}),
            _e("alice", "sp:alice:dbg3", c.KIND_DEBUG, 62, {"source": "hook", "n": 3}),
            _e("alice", "sp:alice:dbg4", c.KIND_DEBUG, 63, {"n": 4}),
        ]
    )
    hook = reader.entries(c.Own("alice"), c.KIND_DEBUG, source="hook")
    assert _keys(hook.data) == ["sp:alice:dbg3", "sp:alice:dbg1"] and not hook.more
    newest = reader.entries(c.Own("alice"), c.KIND_DEBUG, source="hook", limit=1)
    assert _keys(newest.data) == ["sp:alice:dbg3"] and newest.more
    # The newest debug entry is not a hook's, so a limit counted before the match
    # would have answered it.
    first = reader.entries(c.Own("alice"), c.KIND_DEBUG, source="sdk", limit=1)
    assert _keys(first.data) == ["sp:alice:dbg2"] and not first.more
    every = reader.entries(c.Own("alice"), c.KIND_DEBUG)
    assert len(every.data) == 4
    assert reader.entries(c.Own("alice"), c.KIND_DEBUG, source="nobody").data == []


def test_recent_audit_written_by_keeps_only_that_identity(reader, backend):
    backend.append([_e("alice", "sp:alice:aud1", c.KIND_AUDIT, 70, {"surface": "tool"})])
    shown = reader.recent_audit(c.Own("alice"))
    assert {f"sp:{QS}:pf1", "sp:alice:aud1"} <= set(_keys(shown.data))
    own = reader.recent_audit(c.Own("alice"), written_by="alice")
    # The queueserver's preflight on alice's behalf is in her view but not hers.
    assert _keys(own.data) == ["sp:alice:aud1"]
    assert reader.recent_audit(c.All(), written_by=QS, surface="preflight").data
    assert reader.recent_audit(c.Own("alice"), written_by="bob").data == []


def test_own_view_hides_unmatched_preflight_other_views_show_all(reader):
    own = reader.recent_audit(c.Own("alice"), surface="preflight")
    assert _keys(own.data) == [f"sp:{QS}:pf1"]
    assert own.data[0].person == "alice"
    everything = reader.recent_audit(c.All(), surface="preflight")
    assert _keys(everything.data) == [f"sp:{QS}:pf2", f"sp:{QS}:pf1"]


def _item(kind, t, body, person="alice", key=None):
    entry = c.Entry(key=key or f"k{t}{kind}", kind=kind, ts=_at(t), identity="x", body=body)
    return rd.RecordItem(entry, person)


def test_preflight_matching_window_and_person():
    submit = _item(c.KIND_TOOL_CALL, 0, {"name": "mcp__lane__queue_start"})
    other = _item(c.KIND_TOOL_CALL, 0, {"name": "queue_add"}, person="bob", key="kb")
    read = _item(c.KIND_TOOL_CALL, 0, {"name": "Read"}, key="kr")
    timed = _item(c.KIND_TOOL_CALL, 30, {"name": "queue_add", "duration_ms": 25_000}, key="kt")
    pre_in = _item(c.KIND_AUDIT, 60, {"surface": "preflight"}, key="p1")
    pre_out = _item(c.KIND_AUDIT, 61, {"surface": "preflight"}, key="p2")
    pre_before = _item(c.KIND_AUDIT, -1, {"surface": "preflight"}, key="p3")
    matches = rd.match_preflight([pre_in, pre_out, pre_before], [submit, other, read])
    assert [m.matched for m in matches] == [True, False, False]
    assert matches[0].submit is submit
    # A call's start is its time less its duration; the nearest start wins.
    later = rd.match_preflight([pre_in], [submit, timed])
    assert later[0].submit is timed
    assert [m.item.key for m in rd.visible_preflight(matches, own=True)] == ["p1"]
    assert len(rd.visible_preflight(matches, own=False)) == 3
    assert rd.is_queue_submit("mcp__x__queue_add") and not rd.is_queue_submit("queue_addx")


# --- runs and bridges ----------------------------------------------------------------------


def test_run_resolves_answer_question_and_tool_count(reader):
    answer = reader.run(c.Own("bob"), R1)
    assert answer.complete, answer.reason
    run = answer.data
    assert run["session_id"] == S3 and run["requester"] == "bob"
    assert run["status"] == "done" and run["answer"] == "final answer"
    # The bridge turn is the bridge's own fact: no person's scope holds it.
    assert run["question"] is None
    assert reader.run(c.All(), R1).data["question"] == "status?"
    assert run["num_tool_calls"] == 1 and run["tool_count"] == 1
    assert "timeline" not in run
    assert answer.to_json()["data"]["terminal"]["body"]["cost"] == 0.1
    assert reader.run(c.Own("alice"), R1).data is None


def test_run_timeline_resolves_the_session(reader):
    answer = reader.run_timeline(c.All(), R1)
    assert answer.complete
    assert [n.item.kind for n in answer.data] == [
        c.KIND_RUN,
        c.KIND_PROMPT,
        c.KIND_TOOL_CALL,
        c.KIND_ANSWER,
    ]
    assert answer.data[2].result.body == {"content": "ok"}


def test_a_run_whose_session_has_not_drained_is_not_complete(backend, config):
    drains = _Drains()
    reader, clock = _draining(backend, config, drains)
    for answer in (reader.run(c.All(), R1), reader.run_timeline(c.All(), R1)):
        assert answer.available and not answer.complete
        assert "did not drain" in answer.reason
    assert clock.sleeps and sum(clock.sleeps) <= 2 * rd.WAIT_SECONDS
    assert max(clock.sleeps) <= rd.WAIT_POLL_SECONDS
    # The session is drained under the identity that accepted the run.
    assert (c.All(), S3, frozenset({W1})) in drains.calls


def test_a_run_drain_that_passes_rereads_the_run_entries(backend, config):
    later = "run-later"
    backend.append([_accept(W1, later, "sess-later", "bob", 300)])

    polls = []

    def ready(_call):
        # Pending at the first check; ingest commits the terminal while the read waits.
        polls.append(1)
        if len(polls) == 2:
            backend.append([_terminal(W1, later, 301)])
        return len(polls) >= 2

    reader, _ = _draining(backend, config, _Drains(ready))
    answer = reader.run(c.All(), later)
    assert answer.data["status"] == "done", "the terminal the drain committed is read"
    assert answer.complete, answer.reason


def test_a_run_with_no_run_entry_yet_is_not_complete(reader):
    answer = reader.run(c.All(), "run-unknown")
    assert answer.available and not answer.complete and answer.data is None
    assert "no accept or terminal" in answer.reason
    assert not reader.run_timeline(c.All(), "run-unknown").complete


def test_a_bridge_turn_whose_run_entries_are_not_consumed_is_incomplete_then_complete(
    backend, reader
):
    run_id = "run-pending"
    backend.append(
        [
            _e(
                BRIDGE,
                f"sp:{BRIDGE}:t-pending",
                c.KIND_BRIDGE_TURN,
                400,
                {"conversation": "c2", "run_id": run_id, "question": "later?"},
                run_id=run_id,
            )
        ]
    )
    first = reader.bridge_history(c.Source(BRIDGE), conversation="c2")
    assert first.available and not first.complete
    assert f"run {run_id} has no accept or terminal" in first.reason
    backend.append(
        [
            _accept(W1, run_id, "sess-pending", "bob", 401),
            _cc(W1, "sess-pending", 1, c.KIND_ANSWER, 402, {"text": "now"}),
            _terminal(W1, run_id, 403),
        ]
    )
    then = reader.bridge_history(c.Source(BRIDGE), conversation="c2")
    assert then.complete, then.reason
    assert then.data[0]["run"]["answer"] == "now"


def test_a_running_run_is_not_complete(reader):
    answer = reader.run(c.Own("alice"), R2)
    assert not answer.complete and "terminal" in answer.reason
    assert answer.data["status"] == rd.RUN_RUNNING and answer.data["requester"] == "alice"
    assert answer.data["answer"] is None and answer.data["num_tool_calls"] == 0


def test_bridge_history_under_its_source_scope(reader):
    answer = reader.bridge_history(c.Source(BRIDGE), conversation="c1")
    assert answer.complete, answer.reason
    (turn,) = answer.data
    assert turn["turn"].body["question"] == "status?"
    assert turn["run"]["answer"] == "final answer" and turn["run"]["status"] == "done"
    assert reader.bridge_history(c.Source(BRIDGE), conversation="other").data == []


def test_bridge_history_drains_each_runs_session(backend, config):
    drains = _Drains()
    reader, _ = _draining(backend, config, drains)
    answer = reader.bridge_history(c.Source(BRIDGE))
    assert answer.available and not answer.complete
    assert (c.Source(BRIDGE), S3, frozenset({W1})) in drains.calls


# --- Source scope -----------------------------------------------------------------------------


def test_source_sees_its_own_entries_its_runs_and_their_sessions(reader):
    rows = {r["session_id"] for r in reader.list_sessions(c.Source(BRIDGE)).data}
    assert rows == {S3}
    assert reader.timeline(c.Source(BRIDGE), S1).data == []
    assert reader.run(c.Source(BRIDGE), R2).data is None
    timeline = reader.timeline(c.Source(BRIDGE), S3).data
    assert any(n.item.kind == c.KIND_ANSWER for n in timeline)
    worker = {r["session_id"] for r in reader.list_sessions(c.Source(W1)).data}
    assert worker == {S3}


def test_source_narrow_to_a_session(reader):
    scope = c.Source(W1, c.Narrow(session_id=S3))
    assert len(reader.timeline(scope, S3).data) == 4
    elsewhere = c.Source(W1, c.Narrow(session_id="sess-other"))
    assert reader.timeline(elsewhere, S3).data == []
    assert reader.run(elsewhere, R1).data is None


def test_source_narrow_to_run_ids(reader):
    allowed = c.Source(W1, c.Narrow(run_ids=frozenset({R1})))
    assert reader.run(allowed, R1).data["answer"] == "final answer"
    other = c.Source(W1, c.Narrow(run_ids=frozenset({R2})))
    assert reader.run(other, R1).data is None
    assert reader.run(other, R2).data is None
    bridge = c.Source(BRIDGE, c.Narrow(run_ids=frozenset({R2})))
    assert reader.bridge_history(bridge).data == []


# --- wait marks ---------------------------------------------------------------------------------


def test_a_drain_not_yet_passed_answers_incomplete_within_the_bound(backend, config):
    drains = _Drains()
    reader, clock = _draining(backend, config, drains)
    answer = reader.entries(c.Own("alice"), c.KIND_FEEDBACK, wait=c.WaitMark(drain=c.Drain()))
    assert answer.available and not answer.complete
    assert "wait mark" in answer.reason
    assert len(answer.data) == 3
    assert rd.WAIT_SECONDS - rd.WAIT_POLL_SECONDS <= sum(clock.sleeps) <= rd.WAIT_SECONDS
    # The service resolves the drain from the caller's scope; the caller names only a session.
    assert drains.calls == [(c.Own("alice"), None, frozenset())]


def test_a_passed_drain_answers_complete_without_waiting(backend, config):
    drains = _Drains(lambda call: True)
    reader, clock = _draining(backend, config, drains)
    wait = c.WaitMark(drain=c.Drain(S1))
    assert reader.timeline(c.Own("alice"), S1, wait=wait).complete
    assert drains.calls == [(c.Own("alice"), S1, frozenset())]
    assert clock.sleeps == []


def test_a_reader_without_a_drainer_has_nothing_to_drain(reader, clock):
    answer = reader.entries(c.All(), c.KIND_FEEDBACK, wait=c.WaitMark(drain=c.Drain()))
    assert answer.complete and clock.sleeps == []


def test_a_transcript_mark_is_checked_against_the_scopes_identity(backend, reader):
    backend.put_state(c.transcript_state_name(W1, S3_FILE), c.TranscriptOffset(0, 100).to_state())
    mark = c.WaitMark(transcript=c.TranscriptMark(S3, {S3_FILE: (0, 100)}))
    assert reader.timeline(c.Source(W1), S3, wait=mark).complete
    assert not reader.timeline(c.Own("bob"), S3, wait=mark).complete
    assert reader.timeline(c.All(), S3, wait=mark).complete


def test_a_drain_that_passes_while_waiting_answers_complete(backend, config):
    clock = _Clock()

    def sleep(seconds):
        clock.sleep(seconds)
        backend.append([_e("alice", "sp:alice:fb-later", c.KIND_FEEDBACK, 60, {"text": "x"})])

    def landed(_call):
        return "sp:alice:fb-later" in _keys(reader.entries(c.All(), c.KIND_FEEDBACK).data)

    reader = rd.RecordReader(
        backend, config, clock=clock, sleep=sleep, now=lambda: NOW, drain=_Drains(landed)
    )
    answer = reader.entries(c.All(), c.KIND_FEEDBACK, wait=c.WaitMark(drain=c.Drain()))
    assert answer.complete and len(clock.sleeps) == 1
    assert "sp:alice:fb-later" in _keys(answer.data)


# --- limits -------------------------------------------------------------------------------------


def test_scan_byte_and_time_limits_answer_incomplete(backend, config):
    clock = _Clock()
    scan = rd.RecordReader(
        backend,
        config,
        limits=rd.Limits(scan_rows=2),
        clock=clock,
        sleep=clock.sleep,
        now=lambda: NOW,
    )
    answer = scan.entries(c.All(), c.KIND_FEEDBACK)
    assert answer.available and not answer.complete and "scan limit" in answer.reason
    small = rd.RecordReader(
        backend, config, limits=rd.Limits(bytes=30), clock=clock, sleep=clock.sleep, now=lambda: NOW
    )
    answer = small.entries(c.All(), c.KIND_FEEDBACK)
    assert not answer.complete and "byte limit" in answer.reason and len(answer.data) == 1
    ticks = iter(range(1000))
    slow = rd.RecordReader(
        backend,
        config,
        limits=rd.Limits(seconds=2),
        clock=lambda: next(ticks),
        sleep=lambda s: None,
        now=lambda: NOW,
    )
    answer = slow.entries(c.All(), c.KIND_FEEDBACK)
    assert not answer.complete and "time limit" in answer.reason


# --- an unavailable backend ------------------------------------------------------------------------


class _Down:
    """A backend whose every call fails."""

    def _fail(self, *args, **kwargs):
        raise c.RecordBackendUnavailable("connection refused")

    append = select = get_state = put_state = delete_arrived_before = export_parquet = _fail
    health = _fail


class _FailsLater:
    """A backend that answers the first *ok* selects, then fails."""

    def __init__(self, backend, ok: int) -> None:
        self._backend = backend
        self._ok = ok

    def select(self, scope_filter, query):
        if self._ok <= 0:
            raise c.RecordBackendError("lost the connection")
        self._ok -= 1
        return self._backend.select(scope_filter, query)

    def get_state(self, name):
        return self._backend.get_state(name)


def _calls():
    return [
        lambda r, s: r.list_sessions(s),
        lambda r, s: r.session_users(s),
        lambda r, s: r.timeline(s, S1),
        lambda r, s: r.agent_timeline(s, S1, "ag1"),
        lambda r, s: r.full_entry(s, "cc:alice:sess-alice-1:u1:0"),
        lambda r, s: r.run(s, R1),
        lambda r, s: r.run_timeline(s, R1),
        lambda r, s: r.recent_audit(s),
        lambda r, s: r.bridge_history(s),
        lambda r, s: r.entries(s, c.KIND_FEEDBACK),
    ]


@pytest.mark.parametrize("call", _calls())
@pytest.mark.parametrize("scope", [c.Own("alice"), c.All(), c.Reader("ops"), c.Source(BRIDGE)])
def test_a_backend_that_raises_answers_unavailable(config, call, scope):
    clock = _Clock()
    reader = rd.RecordReader(_Down(), config, clock=clock, sleep=clock.sleep, now=lambda: NOW)
    answer = call(reader, scope)
    assert not answer.available and not answer.complete and answer.data is None
    assert answer.to_json() == {
        "status": "unavailable",
        "complete": False,
        "reason": answer.reason,
    }
    assert "connection refused" in answer.reason


@pytest.mark.parametrize("ok", [1, 2, 3])
def test_a_backend_failing_mid_read_never_yields_a_partial_complete_answer(backend, config, ok):
    clock = _Clock()
    reader = rd.RecordReader(
        _FailsLater(backend, ok), config, clock=clock, sleep=clock.sleep, now=lambda: NOW
    )
    answer = reader.run(c.Source(BRIDGE), R1)
    assert not answer.available and answer.data is None


def test_a_drain_on_a_failing_backend_is_unavailable(config):
    def down(_call):
        raise c.RecordBackendUnavailable("down")

    reader = rd.RecordReader(_Down(), config, drain=_Drains(down))
    answer = reader.entries(c.All(), c.KIND_FEEDBACK, wait=c.WaitMark(drain=c.Drain()))
    assert not answer.available


def test_every_method_answers_under_every_scope(reader):
    for scope in (
        c.Own("alice"),
        c.All(),
        c.Reader("ops"),
        c.Source(BRIDGE),
        c.Source(W1, c.Narrow(session_id=S3)),
    ):
        for call in _calls():
            answer = call(reader, scope)
            assert answer.available, (scope, answer.reason)
            answer.to_json()


def test_arguments_that_are_not_valid_raise(reader):
    with pytest.raises(ValueError):
        reader.timeline(c.All(), "no spaces allowed")
    with pytest.raises(ValueError):
        reader.run(c.All(), "bad:run")
    with pytest.raises(ValueError):
        reader.full_entry(c.All(), "")
    with pytest.raises(ValueError):
        reader.entries("everyone", c.KIND_FEEDBACK)


# --- budgets on a clock that keeps moving -------------------------------------------------------


class _TickingClock(_Clock):
    """A clock that also advances 1 ms on every reading, as a real one does."""

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


def _ticking(backend, config, **kwargs) -> tuple[rd.RecordReader, _TickingClock]:
    clock = _TickingClock()
    reader = rd.RecordReader(
        backend, config, clock=clock, sleep=clock.sleep, now=lambda: NOW, **kwargs
    )
    return reader, clock


def test_an_unpassed_wait_mark_still_answers_the_entries_that_exist(backend, config):
    reader, clock = _ticking(backend, config, drain=_Drains())
    mark = c.WaitMark(drain=c.Drain())
    answer = reader.entries(c.Own("alice"), c.KIND_FEEDBACK, wait=mark)
    assert answer.available and not answer.complete
    assert "wait mark" in answer.reason and "time limit" not in answer.reason
    assert _keys(answer.data) == ["sp:alice:fb3", "sp:alice:fb2", "sp:alice:fb1"]
    assert clock.t >= rd.WAIT_SECONDS
    timeline = reader.timeline(c.Own("alice"), S1, wait=mark)
    assert len(timeline.data) == 6 and "time limit" not in timeline.reason


def test_a_runs_drain_does_not_use_up_the_read_budget(backend, config):
    reader, _ = _ticking(backend, config, drain=_Drains())
    run = reader.run(c.All(), R1)
    assert not run.complete and "did not drain" in run.reason and "time limit" not in run.reason
    assert run.data["answer"] == "final answer" and run.data["num_tool_calls"] == 1
    timeline = reader.run_timeline(c.All(), R1)
    assert len(timeline.data) == 4 and "time limit" not in timeline.reason
    history = reader.bridge_history(c.Source(BRIDGE), wait=c.WaitMark(drain=c.Drain()))
    assert history.data and history.data[0]["run"]["answer"] == "final answer"
    assert "time limit" not in history.reason


def test_the_time_limit_is_checked_while_loading(backend, config):
    reader, _ = _ticking(backend, config, limits=rd.Limits(seconds=0.002))
    for answer in (reader.list_sessions(c.All()), reader.session_users(c.All())):
        assert answer.available and not answer.complete and "time limit" in answer.reason


# --- whole-record scans keep the newest rows --------------------------------------------------


def test_a_scan_cut_keeps_the_newest_rows(backend, config, clock):
    reader = rd.RecordReader(
        backend,
        config,
        limits=rd.Limits(scan_rows=3),
        clock=clock,
        sleep=clock.sleep,
        now=lambda: NOW,
    )
    answer = reader.list_sessions(c.All())
    assert not answer.complete and "scan limit" in answer.reason
    assert [r["session_id"] for r in answer.data] == [S4]


# --- byte limits on every output path ---------------------------------------------------------


def _limited(backend, config, clock, **limits) -> rd.RecordReader:
    return rd.RecordReader(
        backend,
        config,
        limits=rd.Limits(**limits),
        clock=clock,
        sleep=clock.sleep,
        now=lambda: NOW,
    )


def test_list_sessions_counts_its_rows_toward_the_byte_limit(backend, config, clock, reader):
    full = reader.list_sessions(c.All()).data
    small = _limited(backend, config, clock, bytes=rd._measure(full[0])[0])
    answer = small.list_sessions(c.All())
    assert not answer.complete and "byte limit" in answer.reason
    assert [r["session_id"] for r in answer.data] == [full[0]["session_id"]]


def test_a_run_that_does_not_fit_keeps_only_its_small_facts(backend, config, clock):
    answer = _limited(backend, config, clock, bytes=10).run(c.All(), R1)
    assert answer.available and not answer.complete and "byte limit" in answer.reason
    assert answer.data["status"] == "done" and answer.data["num_tool_calls"] == 1
    assert answer.data["answer"] is None and answer.data["terminal"] is None


def test_bridge_history_counts_turns_and_run_facts(backend, config, clock):
    answer = _limited(backend, config, clock, bytes=10).bridge_history(c.All())
    assert not answer.complete and "byte limit" in answer.reason and answer.data == []
    turn_only = _limited(backend, config, clock, bytes=100).bridge_history(c.All())
    assert not turn_only.complete and "byte limit" in turn_only.reason
    (item,) = turn_only.data
    assert item["run"]["status"] == "done" and item["run"]["answer"] is None


# --- bridge history query count ---------------------------------------------------------------


class _Counting:
    def __init__(self, backend) -> None:
        self._backend = backend
        self.selects = 0

    def select(self, scope_filter, query):
        self.selects += 1
        return self._backend.select(scope_filter, query)

    def get_state(self, name):
        return self._backend.get_state(name)


def test_bridge_history_queries_do_not_grow_with_its_turns(backend, config, clock):
    counting = _Counting(backend)
    reader = rd.RecordReader(counting, config, clock=clock, sleep=clock.sleep, now=lambda: NOW)
    assert len(reader.bridge_history(c.All()).data) == 1
    one = counting.selects
    backend.append(
        [
            _e(
                BRIDGE,
                f"sp:{BRIDGE}:t{n}",
                c.KIND_BRIDGE_TURN,
                300 + n,
                {"conversation": "c1", "run_id": run_id, "question": f"q{n}"},
                run_id=run_id,
            )
            for n, run_id in enumerate([R1, R2] * 10, start=2)
        ]
    )
    counting.selects = 0
    answer = reader.bridge_history(c.All())
    assert len(answer.data) == 21
    assert counting.selects == one
    assert reader.bridge_history(c.All(), limit=3).more


# --- an incomplete chunk set keeps its person -------------------------------------------------


def test_an_incomplete_on_behalf_entry_stays_in_its_owners_view(backend, reader):
    entry = _e(
        QS,
        f"sp:{QS}:pf3",
        c.KIND_AUDIT,
        25,
        {"surface": "preflight", "plan": BIG_TEXT},
        on_behalf_of="alice",
    )
    backend.append([ch for ch in c.split_entry(entry, 400) if ch.chunk.index != 1])
    answer = reader.recent_audit(c.Own("alice"))
    broken = [i for i in answer.data if i.key == f"sp:{QS}:pf3"]
    assert len(broken) == 1 and broken[0].incomplete and broken[0].person == "alice"
    assert not answer.complete and "chunk" in answer.reason
    assert rd._leading_fields('{"surface":"preflight","plan":"ab') == {"surface": "preflight"}
    assert rd._leading_fields('{"n":12') == {}


# --- timeline paging keeps joins together -----------------------------------------------------


def test_a_timeline_page_keeps_each_call_with_its_result(reader):
    answer = reader.timeline(c.Own("alice"), S1, limit=5)
    assert answer.more and len(answer.data) == 5
    submit = answer.data[4]
    assert submit.result is not None and submit.decision is not None
    assert answer.data[3].children


# --- a forged run accept on a foreign session --------------------------------------------------


def test_a_forged_accept_on_a_foreign_session_opens_nothing(backend, reader):
    r9 = "run-9"
    backend.append(
        [
            _accept(W2, r9, S1, "mallory", 400),
            _e(
                BRIDGE,
                f"sp:{BRIDGE}:t9",
                c.KIND_BRIDGE_TURN,
                401,
                {"conversation": "c9", "run_id": r9, "question": "leak?"},
                run_id=r9,
            ),
        ]
    )
    alice_keys = set(_flat(reader.timeline(c.Own("alice"), S1).data))
    for scope in (c.Source(W2), c.Source(BRIDGE)):
        seen = set(_flat(reader.timeline(scope, S1).data))
        assert not seen & alice_keys, scope
        history = reader.bridge_history(scope, conversation="c9").data
        for item in history:
            assert item["run"] is None or item["run"]["answer"] is None
    assert reader.timeline(c.Own("alice"), S1).complete
    assert set(_flat(reader.timeline(c.Own("alice"), S1).data)) == alice_keys
    assert len(alice_keys) > 10
