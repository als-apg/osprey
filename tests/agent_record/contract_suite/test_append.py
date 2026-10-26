"""Append on every backend: first write wins on ``(key, chunk.index)``, the same
row survives on every backend and after the backend settles, a duplicate re-sent
on a later arrival day is stored and exported nowhere, and neither a raw-id
collision across identities nor a resumed session loses a fact.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from osprey.agent_record import contract as c

from . import _kit as k


def test_first_write_wins_and_the_first_row_survives_settling(backend, settle):
    first = k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, {"v": 1})
    rival = k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 2, {"v": 2}, arrived=3)
    assert backend.append([first]) == c.AppendResult(appended=1, duplicates=0)
    assert backend.append([rival]) == c.AppendResult(appended=0, duplicates=1)
    # Within one batch the earlier copy is the one stored.
    a = k.entry(k.BOB, "sp:bob:e2", c.KIND_FEEDBACK, 5, {"v": "a"})
    b = k.entry(k.BOB, "sp:bob:e2", c.KIND_FEEDBACK, 6, {"v": "b"})
    assert backend.append([a, b]) == c.AppendResult(appended=1, duplicates=1)
    settle()
    stored = {e.key: e for e in k.rows(backend)}
    assert len(k.rows(backend)) == 2
    assert stored["sp:alice:e1"].body == {"v": 1}
    assert stored["sp:alice:e1"].ts == k.at(1)
    assert stored["sp:alice:e1"].inserted_at == k.at(1)
    assert stored["sp:bob:e2"].body == {"v": "a"}
    assert backend.health().entries == 2


def test_a_duplicate_resent_on_a_later_day_is_stored_and_exported_nowhere(
    backend, settle, tmp_path
):
    day = 86400
    original = k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, {"v": 1})
    resent = k.entry(k.ALICE, "sp:alice:e1", c.KIND_FEEDBACK, 1, {"v": 1}, arrived=day + 1)
    backend.append([original])
    assert backend.append([resent]) == c.AppendResult(appended=0, duplicates=1)
    settle()
    [stored] = k.rows(backend)
    assert stored.inserted_at == k.at(1)
    first_day = k.T0.date()
    later_day = (k.T0 + timedelta(days=1)).date()
    assert k.export_rows(backend, later_day, tmp_path / "later.parquet") == []
    exported = k.export_rows(backend, first_day, tmp_path / "first.parquet")
    assert [(r["key"], r["chunk_index"]) for r in exported] == [("sp:alice:e1", 0)]
    health = backend.health()
    assert health.entries == 1
    assert health.oldest_arrival_day == health.newest_arrival_day == first_day


def test_the_same_raw_id_from_two_identities_loses_no_fact(backend, tmp_path):
    when = datetime.now(UTC) - timedelta(hours=1)
    spool_root = tmp_path / "spool"
    for identity in (k.ALICE, k.BOB):
        k.spool(
            spool_root,
            identity,
            k.spool_line(c.KIND_FEEDBACK, {"by": identity}, when=when, entry_id="e-same"),
        )
    k.consume(backend, spool_root)
    mounts = tmp_path / "mounts"
    session = "5f0c9a3e-1111-4222-8333-944455556666"
    for identity in (k.ALICE, k.BOB):
        k.write_transcript(
            mounts, identity, session, k.user_line(session, identity, uuid="u-same", when=when)
        )
    k.tail(backend, mounts)
    stored = k.rows(backend)
    assert len(stored) == 4
    for identity in (k.ALICE, k.BOB):
        mine = [e for e in stored if e.identity == identity]
        assert {e.kind for e in mine} == {c.KIND_FEEDBACK, c.KIND_PROMPT}
        assert {e.key for e in mine} >= {c.spool_key(identity, "e-same")}
    feedback = {e.identity: e.body for e in stored if e.kind == c.KIND_FEEDBACK}
    assert feedback == {k.ALICE: {"by": k.ALICE}, k.BOB: {"by": k.BOB}}
    prompts = {e.identity: e.body["text"] for e in stored if e.kind == c.KIND_PROMPT}
    assert prompts == {k.ALICE: k.ALICE, k.BOB: k.BOB}


def test_a_resumed_session_loses_no_fact(backend, tmp_path):
    when = datetime.now(UTC) - timedelta(hours=1)
    session = "6a1d0b4f-2222-4333-8444-a55566667777"
    mounts = tmp_path / "mounts"
    first = k.user_line(session, "start", uuid="u-1", when=when) + k.text_line(
        session, "done", uuid="a-1", when=when + timedelta(seconds=1)
    )
    k.write_transcript(mounts, k.ALICE, session, first)
    k.tail(backend, mounts)
    # A new process resumes the same session: same session id, same file, new
    # lines, one of them repeating an earlier text under a new uuid.
    resumed = k.user_line(session, "start", uuid="u-2", when=when + timedelta(minutes=5))
    resumed += k.text_line(session, "done again", uuid="a-2", when=when + timedelta(minutes=6))
    k.write_transcript(mounts, k.ALICE, session, resumed)
    k.tail(backend, mounts)
    # Its spool lines come from a new process's segment and repeat no entry id.
    spool_root = tmp_path / "spool"
    k.spool(spool_root, k.ALICE, k.spool_line(c.KIND_FEEDBACK, when=when, session_id=session))
    k.spool(spool_root, k.ALICE, k.spool_line(c.KIND_FEEDBACK, when=when, session_id=session))
    k.consume(backend, spool_root)
    stored = [e for e in k.rows(backend) if e.session_id == session]
    assert len(stored) == 6
    assert len({e.key for e in stored}) == 6
    texts = [e.body["text"] for e in stored if e.kind in (c.KIND_PROMPT, c.KIND_ANSWER)]
    assert sorted(texts) == ["done", "done again", "start", "start"]
    timeline = k.reader(backend).timeline(c.Own(k.ALICE), session)
    assert timeline.complete and len(timeline.data) == 6


def test_appends_are_stamped_in_commit_order_when_the_clock_steps_back(backend, monkeypatch):
    from osprey.agent_record.backends import clickhouse, embedded

    times = iter([k.at(0), k.at(100), k.at(50), k.at(10)])
    clock = lambda: next(times)  # noqa: E731
    monkeypatch.setattr(embedded, "_now", clock)
    monkeypatch.setattr(clickhouse, "_now", clock)
    backend.append([k.entry(k.ALICE, "sp:alice:given", c.KIND_FEEDBACK, 1, {}, arrived=1)])
    for n in range(3):
        fresh = c.Entry(
            key=f"sp:alice:s{n}", kind=c.KIND_FEEDBACK, ts=k.at(1), identity=k.ALICE, body={}
        )
        backend.append([fresh])
    stamps = {e.key: e.inserted_at for e in k.rows(backend)}
    tick = timedelta(microseconds=1)
    assert stamps["sp:alice:given"] == k.at(1)
    assert stamps["sp:alice:s0"] == k.at(100)
    assert stamps["sp:alice:s1"] == k.at(100) + tick
    assert stamps["sp:alice:s2"] == k.at(100) + 2 * tick
