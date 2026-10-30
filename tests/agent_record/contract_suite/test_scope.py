"""The reader's scopes on every backend: own, all, named-reader and source
views; a dispatch run on someone's behalf; a run entry that lands after its
session's entries; a cron run and its ``session_log`` under
``Source(dispatch, narrow=session)``; a multi-asker ``prior_answer_read``
under the allow-listed narrow; and ``Source(bridge)`` reading a turn whose run
and transcript came from a dispatch identity.
"""

from __future__ import annotations

import pytest

from osprey.agent_record import contract as c

from . import _kit as k

S1 = "sess-alice"  # alice's terminal
S2 = "sess-bob"  # bob's terminal
S3 = "sess-r1"  # W1, run R1 for bob, asked by bob in conversation c1
S5 = "sess-r3"  # W1, run R3 for alice, asked by alice in conversation c1
S6 = "sess-r4"  # W1, run R4 for bob, asked nowhere
S4 = "sess-cron"  # W2, cron run R2: no requester
S7 = "sess-r5"  # W2, run R5 for alice
R1, R2, R3, R4, R5 = "run-1", "run-2", "run-3", "run-4", "run-5"

DAY = 86400


def _run(worker: str, run_id: str, session: str, requester: str | None, t: float, answer: str):
    return [
        k.accept(worker, run_id, session, requester, t),
        k.cc(worker, session, 1, c.KIND_PROMPT, t + 1, {"text": f"<payload {run_id}>"}),
        k.cc(
            worker,
            session,
            2,
            c.KIND_TOOL_CALL,
            t + 2,
            {"id": f"tu-{run_id}", "name": "Bash"},
            tool_use_id=f"tu-{run_id}",
        ),
        k.cc(
            worker,
            session,
            3,
            c.KIND_TOOL_RESULT,
            t + 3,
            {"content": "ok"},
            tool_use_id=f"tu-{run_id}",
        ),
        k.cc(worker, session, 4, c.KIND_ANSWER, t + 4, {"text": answer}),
        k.terminal(worker, run_id, t + 5),
    ]


def _seed() -> list[c.Entry]:
    return [
        k.cc(k.ALICE, S1, 1, c.KIND_PROMPT, 1, {"text": "alice asks"}),
        k.cc(k.ALICE, S1, 2, c.KIND_ANSWER, 2, {"text": "alice is answered"}),
        k.cc(k.BOB, S2, 1, c.KIND_PROMPT, 1, {"text": "bob asks"}),
        k.cc(k.BOB, S2, 2, c.KIND_ANSWER, 2, {"text": "bob is answered"}),
        k.turn(1, R1, 99, asked_by=k.BOB, question="status?"),
        *_run(k.W1, R1, S3, k.BOB, 100, "answer one"),
        k.turn(3, R3, 199, asked_by=k.ALICE, question="and now?"),
        *_run(k.W1, R3, S5, k.ALICE, 200, "answer three"),
        *_run(k.W1, R4, S6, k.BOB, 300, "answer four"),
        *_run(k.W2, R2, S4, None, 400, "cron answer"),
        *_run(k.W2, R5, S7, k.ALICE, 500, "answer five"),
        k.entry(k.ALICE, "sp:alice:fb1", c.KIND_FEEDBACK, 30, {"text": "recent"}),
        k.entry(k.ALICE, "sp:alice:fb2", c.KIND_FEEDBACK, 31, {"text": "old"}, arrived=-5 * DAY),
    ]


@pytest.fixture
def seeded(backend, settle):
    backend.append(_seed())
    settle()
    return backend


@pytest.fixture
def reader(seeded):
    return k.reader(seeded, now=k.NOW)


def _texts(nodes, kind: str) -> list[str]:
    return [n.item.body["text"] for n in nodes if n.item.kind == kind]


# --- own / all / named reader / source ----------------------------------------------------


def test_own_sees_its_terminal_and_the_runs_asked_on_its_behalf(reader):
    alice = reader.list_sessions(c.Own(k.ALICE))
    assert alice.complete, alice.reason
    assert k.session_ids(alice) == {S1, S5, S7}
    assert k.session_ids(reader.list_sessions(c.Own(k.BOB))) == {S2, S3, S6}
    for session in (S2, S3, S6, S4):
        assert reader.timeline(c.Own(k.ALICE), session).data == [], session
    assert _texts(reader.timeline(c.Own(k.ALICE), S1).data, c.KIND_ANSWER) == ["alice is answered"]


def test_all_sees_every_session_and_names_each_person(reader):
    answer = reader.list_sessions(c.All())
    assert answer.complete, answer.reason
    persons = {r["session_id"]: r["persons"] for r in answer.data}
    assert persons == {
        S1: [k.ALICE],
        S2: [k.BOB],
        S3: [k.BOB],
        S5: [k.ALICE],
        S6: [k.BOB],
        S4: [],
        S7: [k.ALICE],
    }
    users = {u["person"]: u["sessions"] for u in reader.session_users(c.All()).data}
    assert users == {k.ALICE: 3, k.BOB: 3}


def test_a_named_reader_keeps_its_kinds_and_days_back(reader):
    recent = c.Reader("ops", kinds=frozenset({c.KIND_FEEDBACK}), days_back=2)
    assert [i.key for i in reader.entries(recent, c.KIND_FEEDBACK).data] == ["sp:alice:fb1"]
    every_day = c.Reader("ops", kinds=frozenset({c.KIND_FEEDBACK}))
    assert {i.key for i in reader.entries(every_day, c.KIND_FEEDBACK).data} == {
        "sp:alice:fb1",
        "sp:alice:fb2",
    }
    assert reader.list_sessions(recent).data == []
    assert reader.timeline(recent, S1).data == []
    assert reader.run(recent, R1).data is None


def test_a_source_sees_what_it_ingested_and_the_runs_it_names(reader):
    assert k.session_ids(reader.list_sessions(c.Source(k.W1))) == {S3, S5, S6}
    assert k.session_ids(reader.list_sessions(c.Source(k.W2))) == {S4, S7}
    assert k.session_ids(reader.list_sessions(c.Source(k.BRIDGE))) == {S3, S5}
    assert reader.timeline(c.Source(k.W2), S3).data == []
    assert reader.entries(c.Source(k.W1), c.KIND_FEEDBACK).data == []


# --- dispatch runs ------------------------------------------------------------------------


def test_a_dispatch_run_on_someones_behalf_is_theirs(reader):
    answer = reader.run(c.Own(k.BOB), R1)
    assert answer.complete, answer.reason
    run = answer.data
    assert run["requester"] == k.BOB and run["session_id"] == S3
    assert run["answer"] == "answer one" and run["status"] == "done"
    assert run["num_tool_calls"] == 1
    assert reader.run(c.Own(k.ALICE), R1).data is None
    timeline = reader.run_timeline(c.Own(k.BOB), R1)
    assert timeline.complete
    assert _texts(timeline.data, c.KIND_ANSWER) == ["answer one"]


def test_a_run_entry_that_lands_after_its_session_entries_attributes_them(seeded, settle):
    session, run_id = "sess-late", "run-late"
    entries = [
        k.cc(k.W1, session, 1, c.KIND_PROMPT, 600, {"text": "early"}, arrived=600),
        k.cc(k.W1, session, 2, c.KIND_ANSWER, 601, {"text": "late answer"}, arrived=601),
    ]
    seeded.append(entries)
    settle()
    reader = k.reader(seeded, now=k.NOW)
    assert reader.timeline(c.Own(k.BOB), session).data == []
    assert session not in k.session_ids(reader.list_sessions(c.Own(k.BOB)))
    seeded.append([k.accept(k.W1, run_id, session, k.BOB, 599, arrived=900)])
    settle()
    timeline = reader.timeline(c.Own(k.BOB), session)
    assert timeline.complete
    assert {e.key for e in entries} <= set(k.flat(timeline.data))
    assert session in k.session_ids(reader.list_sessions(c.Own(k.BOB)))
    assert reader.run(c.Own(k.BOB), run_id).data["answer"] == "late answer"
    assert reader.timeline(c.Own(k.ALICE), session).data == []


def test_a_cron_run_names_no_person(reader):
    answer = reader.run(c.All(), R2)
    assert answer.complete, answer.reason
    assert answer.data["requester"] is None and answer.data["answer"] == "cron answer"
    for person in (k.ALICE, k.BOB):
        assert reader.run(c.Own(person), R2).data is None
        assert S4 not in k.session_ids(reader.list_sessions(c.Own(person)))
        assert reader.timeline(c.Own(person), S4).data == []
    assert reader.run(c.Source(k.W2), R2).data["answer"] == "cron answer"
    rows = {r["session_id"]: r for r in reader.list_sessions(c.All()).data}
    assert rows[S4]["run_ids"] == [R2] and rows[S4]["prompt"] == f"<payload {R2}>"


def test_a_cron_runs_session_log_under_its_narrowed_source(reader):
    session_log = c.Source(k.W2, c.Narrow(session_id=S4))
    answer = reader.timeline(session_log, S4)
    assert answer.complete, answer.reason
    assert _texts(answer.data, c.KIND_ANSWER) == ["cron answer"]
    assert _texts(answer.data, c.KIND_PROMPT) == [f"<payload {R2}>"]
    assert reader.timeline(session_log, S7).data == []
    assert reader.run(session_log, R5).data is None
    assert k.session_ids(reader.list_sessions(session_log)) == {S4}


def test_prior_answer_read_of_a_multi_asker_conversation(reader):
    allowed = c.Source(k.W1, c.Narrow(run_ids=frozenset({R1, R3})))
    first = reader.run(allowed, R1)
    assert first.complete, first.reason
    assert first.data["answer"] == "answer one" and first.data["requester"] == k.BOB
    second = reader.run(allowed, R3).data
    assert second["answer"] == "answer three" and second["requester"] == k.ALICE
    assert reader.run(allowed, R4).data is None
    assert reader.timeline(allowed, S6).data == []
    assert k.session_ids(reader.list_sessions(allowed)) == {S3, S5}


def test_source_bridge_reads_the_dispatch_answer_and_no_unrelated_session(reader):
    history = reader.bridge_history(c.Source(k.BRIDGE), conversation="c1")
    assert history.complete, history.reason
    answers = {t["turn"].body["question"]: t["run"]["answer"] for t in history.data}
    assert answers == {"status?": "answer one", "and now?": "answer three"}
    timeline = reader.timeline(c.Source(k.BRIDGE), S3).data
    assert _texts(timeline, c.KIND_ANSWER) == ["answer one"]
    for session in (S1, S2, S6, S4, S7):
        assert reader.timeline(c.Source(k.BRIDGE), session).data == [], session
    assert reader.run(c.Source(k.BRIDGE), R4).data is None
    assert reader.bridge_history(c.Source(k.BRIDGE), conversation="other").data == []
