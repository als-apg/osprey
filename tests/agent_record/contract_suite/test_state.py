"""State, wait marks, retention and export on every backend: state is last
write wins and never an entry; ``count_state`` counts a name prefix, with a
time only the values stamped at or after it; a drain passes exactly when ingest has
committed past it, and a transcript ``WaitMark`` when the tailer has; arrival-day delete drops whole days and keeps state; the Parquet
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


def test_count_state_counts_a_prefix_and_only_times_since(backend, settle):
    old, recent = k.T0 - timedelta(hours=25), k.T0 - timedelta(hours=1)
    backend.put_state(c.refused_state_name(k.ALICE, "e1"), c.refused_state_value("run", "x", old))
    backend.put_state(
        c.refused_state_name(k.ALICE, "e2"), c.refused_state_value("run", "x", recent)
    )
    backend.put_state(c.refused_state_name(k.ALICE, "e3"), c.refused_state_value("run", "x", k.T0))
    backend.put_state(c.refused_state_name(k.BOB, "e1"), c.refused_state_value("run", "x", k.T0))
    backend.put_state(f"{c.REFUSED_STATE_PREFIX}:{k.ALICE}:untimed", {"kind": "run"})
    backend.put_state("cursor", {"offset": 1, c.STATE_TIME_FIELD: c.state_time(k.T0)})
    settle()
    alice = f"{c.REFUSED_STATE_PREFIX}:{k.ALICE}:"
    assert backend.count_state(alice) == 4
    assert backend.count_state(alice, k.T0 - timedelta(hours=24)) == 2
    assert backend.count_state(alice, k.T0) == 1, "a time equal to since counts"
    assert backend.count_state(f"{c.REFUSED_STATE_PREFIX}:") == 5
    assert backend.count_state("nothing:") == 0
    # Last write wins: re-stamping a refusal old ages it out.
    backend.put_state(c.refused_state_name(k.ALICE, "e3"), c.refused_state_value("run", "x", old))
    settle()
    assert backend.count_state(alice, k.T0 - timedelta(hours=24)) == 1


def test_a_drain_passes_once_its_lines_are_committed_or_refused(backend, settle, tmp_path):
    root = tmp_path / "spool"
    fine = k.spool_line(c.KIND_FEEDBACK, when=k.T0)
    refused = k.spool_line(c.KIND_RUN, {"phase": c.RUN_ACCEPT}, when=k.T0, run_id="r1")
    k.spool(root, k.ALICE, fine, refused)
    k.spool(root, k.BOB, k.spool_line(c.KIND_FEEDBACK, when=k.T0))
    drain = k.drainer(backend, root, tmp_path / "mounts")
    pending = drain(c.Own(k.ALICE), None, frozenset())
    assert not pending.passed()
    k.consume(backend, root)
    settle()
    assert pending.passed()
    assert backend.get_state(c.refused_state_name(k.ALICE, refused.entry_id)) is not None


def test_a_transcript_mark_passes_at_its_offset_or_a_fully_read_later_generation(backend):
    mark = c.WaitMark(transcript=c.TranscriptMark("sess-1", {FILE: (1, 100)}))
    state = c.transcript_state_name(k.W1, FILE)
    assert not mark.passed(backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(1, 99).to_state())
    assert not mark.passed(backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(1, 100).to_state())
    assert mark.passed(backend, transcript_identity=k.W1)
    assert not mark.passed(backend, transcript_identity=k.W2)
    backend.put_state(state, c.TranscriptOffset(2, 10).to_state())
    assert not mark.passed(backend, transcript_identity=k.W1)
    backend.put_state(state, c.TranscriptOffset(2, 10, fully_read=True).to_state())
    assert mark.passed(backend, transcript_identity=k.W1)


def test_the_reader_answers_a_drain_passed_and_not_passed(backend, settle, tmp_path):
    root = tmp_path / "spool"
    backend.append([k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, {"text": "x"})])
    settle()
    clock = k.Clock()
    reader = k.reader(
        backend, now=k.NOW, clock=clock, drain=k.drainer(backend, root, tmp_path / "mounts")
    )
    wait = c.WaitMark(drain=c.Drain())
    passed = reader.entries(c.Own(k.ALICE), c.KIND_FEEDBACK, wait=wait)
    assert passed.complete and clock.sleeps == []
    k.spool(root, k.ALICE, k.spool_line(c.KIND_FEEDBACK, when=k.T0))
    pending = reader.entries(c.Own(k.ALICE), c.KIND_FEEDBACK, wait=wait)
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
