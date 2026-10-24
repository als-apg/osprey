"""State, wait marks, retention and export on every backend: state is last
write wins and never an entry; a ``WaitMark`` passes exactly when ingest has
passed it; arrival-day delete drops whole days and keeps state; the Parquet
export holds each ``(key, chunk.index)`` of its day once.
"""

from __future__ import annotations

from datetime import timedelta

from osprey.agent_record import contract as c

from . import _kit as k

DAY = 86400
FILE = "-data-workspace/sess-1.jsonl"


def test_state_is_last_write_wins_and_never_an_entry(backend, settle, tmp_path):
    assert backend.get_state("cursor") is None
    backend.put_state("cursor", {"offset": 1})
    backend.put_state("cursor", {"offset": 2, "note": "später"})
    backend.put_state("other", {"offset": 9})
    settle()
    assert backend.get_state("cursor") == {"offset": 2, "note": "später"}
    assert backend.get_state("other") == {"offset": 9}
    assert k.rows(backend) == []
    assert backend.health().entries == 0
    assert k.export_rows(backend, k.T0.date(), tmp_path / "x.parquet") == []
    backend.append([k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, arrived=-30 * DAY)])
    assert backend.delete_arrived_before(k.T0.date()) == 1
    assert backend.get_state("cursor") == {"offset": 2, "note": "später"}


def test_a_spool_mark_passes_once_its_line_is_committed_or_refused(backend, settle):
    mark = c.WaitMark(spool=((k.ALICE, "e1"), (k.ALICE, "e2")))
    assert not mark.passed(backend, backend)
    backend.append([k.entry(k.ALICE, c.spool_key(k.ALICE, "e1"), c.KIND_FEEDBACK, 1)])
    settle()
    assert not mark.passed(backend, backend)
    backend.put_state(c.refused_state_name(k.ALICE, "e2"), c.refused_state_value("audit", "no"))
    assert mark.passed(backend, backend)
    # Another identity's line with the same entry id does not pass it.
    other = c.WaitMark(spool=((k.BOB, "e1"),))
    assert not other.passed(backend, backend)


def test_a_transcript_mark_passes_at_its_offset_or_a_fully_read_later_generation(backend):
    mark = c.WaitMark(transcript=c.TranscriptMark("sess-1", {FILE: (1, 100)}))
    state = c.transcript_state_name(k.W1, FILE)
    assert not mark.passed(backend, backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(1, 99).to_state())
    assert not mark.passed(backend, backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(1, 100).to_state())
    assert mark.passed(backend, backend, transcript_identity=k.W1)
    assert not mark.passed(backend, backend, transcript_identity=k.W2)
    backend.put_state(state, c.TranscriptOffset(2, 10).to_state())
    assert not mark.passed(backend, backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(2, 10, fully_read=True).to_state())
    assert mark.passed(backend, backend, transcript_identity=k.W1)


def test_the_reader_answers_a_wait_mark_passed_and_not_passed(backend, settle):
    backend.append([k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, {"text": "x"})])
    settle()
    clock = k.Clock()
    reader = k.reader(backend, now=k.NOW, clock=clock)
    passed = reader.entries(
        c.Own(k.ALICE), c.KIND_FEEDBACK, wait=c.WaitMark(spool=((k.ALICE, "e1"),))
    )
    assert passed.complete and clock.sleeps == []
    pending = reader.entries(
        c.Own(k.ALICE), c.KIND_FEEDBACK, wait=c.WaitMark(spool=((k.ALICE, "e9"),))
    )
    assert pending.available and not pending.complete and "wait mark" in pending.reason
    assert [i.key for i in pending.data] == ["sp:alice:e1"]


def test_arrival_day_delete_drops_whole_days(backend, settle):
    backend.append(
        [
            k.entry(k.ALICE, "sp:alice:d0", c.KIND_FEEDBACK, 1, arrived=0),
            k.entry(k.ALICE, "sp:alice:d0b", c.KIND_FEEDBACK, 2, arrived=11 * 3600),
            k.entry(k.ALICE, "sp:alice:d1", c.KIND_FEEDBACK, 3, arrived=DAY + 1),
            k.entry(k.ALICE, "sp:alice:d2", c.KIND_FEEDBACK, 4, arrived=2 * DAY + 1),
        ]
    )
    settle()
    assert backend.delete_arrived_before(k.T0.date()) == 0
    second_day = (k.T0 + timedelta(days=1)).date()
    assert backend.delete_arrived_before(second_day) == 2
    settle()
    assert {e.key for e in k.rows(backend)} == {"sp:alice:d1", "sp:alice:d2"}
    health = backend.health()
    assert health.entries == 2
    assert health.oldest_arrival_day == second_day
    assert health.newest_arrival_day == (k.T0 + timedelta(days=2)).date()


def test_the_parquet_export_holds_each_fact_of_its_day_once(backend, settle, tmp_path):
    big = k.cc(k.ALICE, "sess-1", 1, c.KIND_ANSWER, 1, {"text": "chunked body " * 300})
    chunks = c.split_entry(big, 512)
    small = k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 2, {"text": "once"})
    backend.append([*chunks, small, small])
    backend.append([*chunks, small])
    backend.append([k.entry(k.BOB, "sp:bob:e1", c.KIND_FEEDBACK, 3, arrived=DAY + 3)])
    settle()
    exported = k.export_rows(backend, k.T0.date(), tmp_path / "day.parquet")
    pairs = [(r["key"], r["chunk_index"]) for r in exported]
    assert len(pairs) == len(set(pairs)) == len(chunks) + 1
    assert {r["key"] for r in exported} == {big.key, small.key}
    by_pair = {(r["key"], r["chunk_index"]): r for r in exported}
    assert by_pair[(small.key, 0)]["body"] == c.serialize_body(small.body)
    assert by_pair[(small.key, 0)]["identity"] == k.ALICE
    for chunk in chunks:
        row = by_pair[(big.key, chunk.chunk.index)]
        assert row["chunk_total"] == len(chunks)
        assert row["body"] == c.serialize_body(chunk.body)
    next_day = k.export_rows(backend, (k.T0 + timedelta(days=1)).date(), tmp_path / "n.parquet")
    assert [r["key"] for r in next_day] == ["sp:bob:e1"]
