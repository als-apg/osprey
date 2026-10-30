"""Tests for :mod:`osprey.bridges.core.history` over an in-memory record.

Covers the ``bridge_turn`` body a settled turn writes (answered turns name their
run, failed turns name none), the read back through ``bridge_history`` with a
drain (oldest first, each answered turn's answer taken from its run, a failed
turn replayed with the marker and no run lookup), a read the record cannot
answer or answers incomplete yielding no turns and the reason, the per-turn
artifact cap, who asked, and :func:`replay_window`'s turn cap, character budget
and shortened replay of long answers. The real writer, service and drain are
covered by ``test_history_record.py``.
"""

from __future__ import annotations

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import KIND_BRIDGE_TURN, Drain
from osprey.bridges.core.history import (
    FAILED_TURN_ANSWER,
    MAX_ARTIFACTS_PER_TURN,
    RECORD_INCOMPLETE,
    RECORD_UNAVAILABLE,
    TURN_ANSWERED,
    TURN_FAILED,
    ConversationHistory,
    replay_window,
    shorten_answer,
)
from tests.agent_record.fakes import FakeRecordClient
from tests.bridges.record_double import MemoryRecord, RecordHistory

ALICE = {"id": "users/111", "name": "Alice"}
RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"


def _turns(*pairs: tuple[str, str]) -> list[dict]:
    return [
        {"question": q, "answer": a, "ts": float(i), "run_id": RUN, "artifacts": []}
        for i, (q, a) in enumerate(pairs)
    ]


# --- the turn written -------------------------------------------------------


def test_an_answered_turn_writes_the_turn_body_naming_its_run():
    h = RecordHistory()
    arts = [{"entry_id": "e1", "filename": "plot.png", "delivered_mime": "image/png"}]

    assert h.append("k", "q", run_id="run-1", artifacts=arts, asked_by=ALICE) == "turn-1"

    (turn,) = h.record.turns
    assert turn["kind"] == KIND_BRIDGE_TURN and turn["run_id"] == "run-1"
    assert turn["body"] == {
        "conversation": "k",
        "run_id": "run-1",
        "question": "q",
        "artifacts": arts,
        "status": TURN_ANSWERED,
        "asked_by": ALICE,
    }


def test_a_failed_turn_names_no_run_but_keeps_the_one_it_tried():
    h = RecordHistory()

    h.append_failed("k", "q", ALICE, run_id="run-9")

    (turn,) = h.record.turns
    assert turn["run_id"] is None
    assert turn["body"]["status"] == TURN_FAILED
    assert turn["body"]["run_id"] == "run-9"
    assert turn["body"]["asked_by"] == ALICE


def test_nothing_is_written_without_a_key_or_a_failed_question():
    h = RecordHistory()
    assert h.append("", "q", run_id="run-1") is None
    assert h.append_failed("k", "") is None
    assert h.append_failed("", "q") is None
    assert h.record.turns == []


def test_an_asker_naming_no_one_is_not_written():
    h = RecordHistory()
    h.append("k", "q", run_id="run-1", asked_by={"id": "", "name": None})
    h.append("k", "q", run_id="run-1", asked_by={"id": 7, "name": "Alice"})
    assert ["asked_by" in body for body in h.record.bodies()] == [False, False]


def test_artifacts_are_capped_keeping_the_tail():
    h = RecordHistory()
    arts = [{"entry_id": f"e{i}"} for i in range(MAX_ARTIFACTS_PER_TURN + 5)]
    h.append("k", "q", run_id="run-1", artifacts=arts)
    written = h.record.bodies()[0]["artifacts"]
    assert len(written) == MAX_ARTIFACTS_PER_TURN
    assert written[0]["entry_id"] == "e5"


def test_the_default_writer_is_the_spool_writer(monkeypatch):
    from osprey.agent_record import writer

    calls = []
    monkeypatch.setattr(writer, "write", lambda kind, body, **ids: calls.append(ids) or "e1")
    h = ConversationHistory(FakeRecordClient())
    assert h.append("k", "q", run_id="run-1") == "e1"
    assert h.append_failed("k", "q") == "e1"
    assert calls == [{"run_id": "run-1"}, {}]


def test_the_default_constructor_opens_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_AGENT_RECORD_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    h = ConversationHistory()
    h.append_failed("k", "q")
    assert not list(tmp_path.rglob("history.json"))
    # With no record configured, a read is unavailable, not empty-and-complete.
    assert h.read("k").record == RECORD_UNAVAILABLE


# --- the read back ------------------------------------------------------------


def test_turns_read_back_oldest_first_with_their_runs_answers():
    h = RecordHistory()
    h.seed("k", "q1", "a1", run_id="run-1", asked_by=ALICE)
    h.seed("k", "q2", "a2", run_id="run-2")
    h.seed("other", "q3", "a3", run_id="run-3")

    read = h.read("k")

    assert read.record is None
    assert [(t["question"], t["answer"], t["run_id"]) for t in read.turns] == [
        ("q1", "a1", "run-1"),
        ("q2", "a2", "run-2"),
    ]
    assert read.turns[0]["asked_by"] == ALICE and "asked_by" not in read.turns[1]
    assert set(read.turns[1]) == {"question", "answer", "ts", "run_id", "artifacts"}
    assert read.turns[0]["ts"] < read.turns[1]["ts"]


def test_a_read_asks_for_one_conversation_with_a_drain_up_to_the_turn_cap():
    h = RecordHistory(max_turns=7)
    h.read("k")
    assert h.record.reads == [{"conversation": "k", "limit": 7, "wait": Drain()}]


def test_a_failed_turn_replays_the_marker_without_a_run_lookup():
    h = RecordHistory()
    h.append_failed("k", "q", run_id="run-9")

    (turn,) = h.read("k").turns

    assert turn["answer"] == FAILED_TURN_ANSWER
    assert turn["run_id"] is None and turn["artifacts"] == []


def test_descriptors_round_trip_opaquely():
    h = RecordHistory()
    weird = [{"anything": {"nested": [1, 2, 3]}, "x": None}]
    h.seed("k", "q", "a", run_id="run-1", artifacts=weird)
    assert h.read("k").turns[0]["artifacts"] == weird


def test_a_read_the_record_cannot_answer_has_no_turns_and_says_so():
    h = RecordHistory()
    h.seed("k", "q", "a", run_id="run-1")
    h.record.unavailable = "agent-record is unreachable"

    read = h.read("k")

    assert (read.turns, read.record) == ([], RECORD_UNAVAILABLE)
    assert "unreachable" in read.reason


def test_an_incomplete_read_has_no_turns_and_says_so():
    h = RecordHistory()
    h.append("k", "q", run_id="run-not-yet-recorded")

    read = h.read("k")

    assert (read.turns, read.record) == ([], RECORD_INCOMPLETE)
    assert "run-not-yet-recorded" in read.reason


def test_a_raising_client_reads_as_unavailable():
    class Broken:
        def bridge_history(self, **_kw):
            raise RuntimeError("boom")

    read = ConversationHistory(Broken(), write=lambda *a, **k: None).read("k")
    assert read.record == RECORD_UNAVAILABLE and "boom" in read.reason


def test_rows_the_reader_cannot_use_are_skipped():
    answer = ClientAnswer(
        data=[
            "not a row",
            {"turn": None},
            {"turn": {"body": None}},
            {"turn": {"ts": "2026-10-23T12:00:00+00:00", "body": {"question": "q"}}},
        ],
        complete=True,
    )
    h = ConversationHistory(FakeRecordClient({"bridge_history": answer}))
    (turn,) = h.read("k").turns
    assert turn["answer"] == FAILED_TURN_ANSWER and turn["ts"] is not None


def test_an_empty_key_reads_nothing():
    h = RecordHistory()
    assert h.read("").turns == [] and h.record.reads == []


# --- the replay window ------------------------------------------------------


def test_the_turn_cap_keeps_the_newest():
    turns = _turns(*[(f"q{i}", f"a{i}") for i in range(6)])
    assert [t["question"] for t in replay_window(turns, max_turns=3)] == ["q3", "q4", "q5"]


def test_the_char_budget_drops_oldest():
    turns = _turns(("x" * 100, "y" * 100), ("q2", "a2"), ("q3", "a3"))
    assert [t["question"] for t in replay_window(turns, max_chars=250)] == ["q2", "q3"]


def test_the_char_budget_counts_the_full_serialized_turn():
    turns = _turns(("q1", "a1"), ("q2", "a2"))
    turns[1]["artifacts"] = [{"blob": "z" * 500}]
    assert [t["question"] for t in replay_window(turns, max_chars=300)] == ["q2"]


def test_the_newest_turn_survives_even_over_budget():
    turns = _turns(("q1", "a1"), ("q2", "a2"))
    assert [t["question"] for t in replay_window(turns, max_chars=1)] == ["q2"]


def _table(chars: int) -> str:
    """A markdown-ish table of exactly ``chars`` characters, one row per line."""
    row = "| PV:NAME | 1.2345 | ok |\n"
    return (row * (chars // len(row) + 1))[:chars]


def test_without_a_limit_every_answer_is_replayed_in_full():
    (turn,) = replay_window(_turns(("q", _table(10_000))))
    assert len(turn["answer"]) == 10_000 and "answer_chars" not in turn


def test_a_long_answer_is_replayed_as_its_opening_and_a_note_naming_its_run():
    original = _table(10_000)
    (turn,) = replay_window(_turns(("q", original)), shorten_over=3000)

    opening, note = turn["answer"].rsplit("\n", 1)
    assert original.startswith(opening)
    assert len(turn["answer"]) < 3000
    assert RUN in note and f"{len(original):,}" in note
    assert turn["answer_chars"] == len(original)


def test_an_answer_at_the_limit_is_replayed_whole():
    (turn,) = replay_window(_turns(("q", "x" * 3000)), shorten_over=3000)
    assert turn["answer"] == "x" * 3000 and "answer_chars" not in turn


def test_the_opening_ends_at_a_line_break_in_its_second_half():
    answer = "a" * 100 + "\n" + "b" * 299 + "\n" + "c" * 5000
    shortened = shorten_answer(answer, RUN, 1000)
    assert shortened is not None
    assert shortened.split("\n")[:2] == ["a" * 100, "b" * 299]
    assert shortened.split("\n")[2].startswith("[Answer shortened")

    early = "a" * 100 + "\n" + "b" * 5000
    shortened = shorten_answer(early, RUN, 1000)
    assert shortened is not None
    assert shortened.startswith("a" * 100 + "\n" + "b" * 399 + "\n[Answer shortened")


def test_a_long_answer_without_a_run_id_is_replayed_whole():
    turns = _turns(("q", _table(10_000)))
    turns[0]["run_id"] = None
    (turn,) = replay_window(turns, shorten_over=3000)
    assert "answer_chars" not in turn


def test_shortening_never_changes_the_read():
    turns = _turns(("q", _table(10_000)))
    replay_window(turns, shorten_over=3000)
    assert len(turns[0]["answer"]) == 10_000 and "answer_chars" not in turns[0]


def test_shortening_comes_before_the_budget_so_more_turns_survive():
    turns = _turns(("q0", _table(60_000)), ("q1", "a1"), ("q2", "a2"), ("q3", "a3"))
    assert [t["question"] for t in replay_window(turns, max_chars=20_000)] == ["q1", "q2", "q3"]
    assert [t["question"] for t in replay_window(turns, max_chars=20_000, shorten_over=3000)] == [
        "q0",
        "q1",
        "q2",
        "q3",
    ]


def test_a_tiny_limit_never_lengthens_an_answer():
    assert shorten_answer("x" * 50, RUN, 10) is None


def test_shortening_keeps_who_asked():
    turns = _turns(("q", _table(10_000)))
    turns[0]["asked_by"] = ALICE
    (turn,) = replay_window(turns, shorten_over=3000)
    assert "answer_chars" in turn and turn["asked_by"] == ALICE


def test_the_history_cut_uses_its_own_budgets():
    h = ConversationHistory(MemoryRecord(), write=lambda *a, **k: None, max_turns=2)
    turns = _turns(("q1", "a1"), ("q2", "a2"), ("q3", "a3"))
    assert [t["question"] for t in h.cut(turns)] == ["q2", "q3"]
