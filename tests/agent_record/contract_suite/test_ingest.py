"""What ingest stores, on every backend, through the real ingest path: the spool
consumer and the transcript tailer over the parametrized backend, never
``backend.append``.

Pinned: a forged run entry (from a non-dispatch identity, a rival accept on a
won session, and a second accept on a foreign session) is refused or opens
nothing; the trusted on-behalf cases (the sidecar's login row and a matched
queueserver preflight row are the person's, a non-preflight queueserver row
and an unmatched preflight are not, a spoofed ``on_behalf_of`` from any other
sender is dropped); a secret in a body is absent from the stored rows, the
Parquet export and every read; and ``run``/``bridge_history`` answer
``complete: false`` until the run's stored transcript mark has passed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import reader as rd

from . import _kit as k

SECRET = "s3cr3t-token-value-0123456789"
SESSION = "7b2e1c50-3333-4444-8555-b66677778888"
R1 = "run-1"


@pytest.fixture
def base() -> datetime:
    return datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)


@pytest.fixture
def spool_root(tmp_path) -> Path:
    return tmp_path / "spool"


@pytest.fixture
def mounts(tmp_path) -> Path:
    return tmp_path / "mounts"


def _accept(run_id: str, session: str, requester: str | None, when: datetime) -> c.SpoolLine:
    body: dict = {"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session}
    if requester:
        body["requester"] = requester
    return k.spool_line(c.KIND_RUN, body, when=when, run_id=run_id, session_id=session)


def _terminal(run_id: str, session: str, when: datetime):
    body: dict = {"phase": c.RUN_TERMINAL, "status": "done"}
    return k.spool_line(c.KIND_RUN, body, when=when, run_id=run_id, session_id=session)


def _turn(run_id: str, when: datetime, *, conversation: str, question: str) -> c.SpoolLine:
    body = {"conversation": conversation, "run_id": run_id, "question": question}
    return k.spool_line(c.KIND_BRIDGE_TURN, body, when=when, run_id=run_id)


def _bob_run(backend, spool_root, mounts, base) -> list[str]:
    """Run R1 on worker 1 for bob, its transcript tailed and its entries consumed."""
    k.spool(
        spool_root,
        k.W1,
        _accept(R1, SESSION, k.BOB, base),
        _terminal(R1, SESSION, base + timedelta(seconds=9)),
    )
    data = k.user_line(SESSION, "<payload>", uuid="u-1", when=base + timedelta(seconds=1))
    data += k.text_line(SESSION, "final answer", uuid="a-1", when=base + timedelta(seconds=2))
    k.write_transcript(mounts, k.W1, SESSION, data)
    k.tail(backend, mounts)
    k.consume(backend, spool_root)
    return [e.key for e in k.rows(backend) if e.session_id == SESSION and e.kind != c.KIND_RUN]


# --- forged run entries -------------------------------------------------------------------


def test_a_run_entry_from_a_non_dispatch_identity_is_refused(backend, spool_root, base):
    forged = _accept("run-forged", SESSION, "mallory", base)
    k.spool(spool_root, k.ALICE, forged)
    report = k.consume(backend, spool_root)
    assert report.refused == 1 and report.appended == 0
    assert k.rows(backend) == []
    assert backend.get_state(c.refused_state_name(k.ALICE, forged.entry_id)) is not None
    assert k.reader(backend).run(c.All(), "run-forged").data is None


def test_a_rival_accept_on_a_won_session_is_refused(backend, spool_root, mounts, base):
    _bob_run(backend, spool_root, mounts, base)
    rival = _accept("run-rival", SESSION, "mallory", base + timedelta(seconds=20))
    k.spool(spool_root, k.W1, rival)
    report = k.consume(backend, spool_root)
    assert report.refused == 1 and report.appended == 0
    keys = {e.key for e in k.rows(backend)}
    assert c.run_key("run-rival", c.RUN_ACCEPT) not in keys
    reader = k.reader(backend)
    assert reader.run(c.All(), "run-rival").data is None
    run = reader.run(c.Own(k.BOB), R1)
    assert run.complete, run.reason
    assert run.data["requester"] == k.BOB and run.data["answer"] == "final answer"


def test_a_second_run_entry_on_a_foreign_session_opens_nothing(backend, spool_root, mounts, base):
    session_keys = _bob_run(backend, spool_root, mounts, base)
    assert len(session_keys) == 2
    later = base + timedelta(seconds=30)
    k.spool(spool_root, k.W2, _accept("run-9", SESSION, "mallory", later))
    k.spool(spool_root, k.BRIDGE, _turn("run-9", later, conversation="c9", question="leak?"))
    k.consume(backend, spool_root)
    reader = k.reader(backend)
    for scope in (c.Source(k.W2), c.Source(k.BRIDGE), c.Own("mallory")):
        seen = set(k.flat(reader.timeline(scope, SESSION).data))
        assert not seen & set(session_keys), scope
        for item in reader.bridge_history(scope, conversation="c9").data:
            assert item["run"] is None or item["run"]["answer"] is None, scope
        forged = reader.run(scope, "run-9").data
        assert forged is None or forged["answer"] is None, scope
    own = reader.timeline(c.Own(k.BOB), SESSION)
    assert own.complete and set(session_keys) <= set(k.flat(own.data))
    assert reader.run(c.Own(k.BOB), R1).data["answer"] == "final answer"


# --- on behalf ----------------------------------------------------------------------------


def test_the_trusted_on_behalf_cases(backend, spool_root, mounts, base):
    submit = k.assistant_line(
        "8c3f2d61-4444-4555-8666-c77788889999",
        [{"type": "tool_use", "id": "tu-q", "name": "mcp__bluesky__queue_add", "input": {}}],
        uuid="a-q",
        when=base + timedelta(seconds=10),
    )
    k.write_transcript(mounts, k.ALICE, "8c3f2d61-4444-4555-8666-c77788889999", submit)
    k.tail(backend, mounts)
    login = k.spool_line(
        c.KIND_AUDIT, {"surface": "login"}, when=base, on_behalf_of=k.ALICE, entry_id="login"
    )
    matched = k.spool_line(
        c.KIND_AUDIT,
        {"surface": "preflight", "plan": "scan"},
        when=base + timedelta(seconds=30),
        on_behalf_of=k.ALICE,
        entry_id="pf-matched",
    )
    unmatched = k.spool_line(
        c.KIND_AUDIT,
        {"surface": "preflight", "plan": "later"},
        when=base + timedelta(seconds=900),
        on_behalf_of=k.ALICE,
        entry_id="pf-unmatched",
    )
    lane_chat = k.spool_line(
        c.KIND_AUDIT,
        {"surface": "chat"},
        when=base + timedelta(seconds=31),
        on_behalf_of=k.ALICE,
        entry_id="qs-chat",
    )
    spoof = k.spool_line(
        c.KIND_AUDIT, {"surface": "chat"}, when=base, on_behalf_of=k.BOB, entry_id="spoof"
    )
    bridge_spoof = k.spool_line(
        c.KIND_BRIDGE_TURN,
        {"conversation": "c1", "question": "hi"},
        when=base,
        on_behalf_of=k.BOB,
        entry_id="bridge-spoof",
    )
    k.spool(spool_root, k.SIDECAR, login)
    k.spool(spool_root, k.QS, matched, unmatched, lane_chat)
    k.spool(spool_root, k.ALICE, spoof)
    k.spool(spool_root, k.BRIDGE, bridge_spoof)
    report = k.consume(backend, spool_root)
    assert report.appended == 6 and report.refused == 0
    stored = {e.key: e.on_behalf_of for e in k.rows(backend) if e.kind != c.KIND_TOOL_CALL}
    assert stored == {
        c.spool_key(k.SIDECAR, "login"): k.ALICE,
        c.spool_key(k.QS, "pf-matched"): k.ALICE,
        c.spool_key(k.QS, "pf-unmatched"): k.ALICE,
        c.spool_key(k.QS, "qs-chat"): "",
        c.spool_key(k.ALICE, "spoof"): "",
        c.spool_key(k.BRIDGE, "bridge-spoof"): "",
    }
    reader = k.reader(backend)
    alice = reader.recent_audit(c.Own(k.ALICE))
    assert alice.complete, alice.reason
    assert {i.key for i in alice.data} == {
        c.spool_key(k.SIDECAR, "login"),
        c.spool_key(k.QS, "pf-matched"),
        c.spool_key(k.ALICE, "spoof"),
    }
    assert {i.person for i in alice.data} == {k.ALICE}
    preflight = reader.recent_audit(c.Own(k.ALICE), surface="preflight").data
    assert [i.key for i in preflight] == [c.spool_key(k.QS, "pf-matched")]
    assert reader.recent_audit(c.Own(k.BOB)).data == []
    assert reader.entries(c.Own(k.BOB), c.KIND_BRIDGE_TURN).data == []
    everything = {i.key for i in reader.recent_audit(c.All()).data}
    assert c.spool_key(k.QS, "pf-unmatched") in everything
    assert c.spool_key(k.QS, "qs-chat") in everything


# --- secrets ------------------------------------------------------------------------------


def test_a_secret_in_a_body_is_absent_from_rows_export_and_reads(
    backend, spool_root, mounts, base, tmp_path
):
    session = "9d4a3e72-5555-4666-8777-d88899990000"
    line = k.spool_line(
        c.KIND_AUDIT,
        {"text": f"token={SECRET}", "nested": [{"v": SECRET}], "surface": "chat"},
        when=base,
        session_id=session,
    )
    k.spool(spool_root, k.ALICE, line)
    k.consume(backend, spool_root, secrets=[SECRET])
    transcript = k.user_line(session, f"my key is {SECRET}", uuid="u-1", when=base)
    transcript += k.text_line(session, f"echo {SECRET}", uuid="a-1", when=base)
    k.write_transcript(mounts, k.ALICE, session, transcript)
    k.tail(backend, mounts, secret_set={SECRET})
    stored = k.rows(backend)
    assert len(stored) == 3
    for entry in stored:
        assert SECRET not in c.serialize_body(entry.body)
        assert SECRET not in repr(entry)
    masked = [e for e in stored if c.SECRET_MASK in c.serialize_body(e.body)]
    assert len(masked) == 3
    audit = next(e for e in stored if e.kind == c.KIND_AUDIT)
    assert audit.body["text"] == f"token={c.SECRET_MASK}"
    assert audit.body["nested"] == [{"v": c.SECRET_MASK}]

    days = {e.inserted_at.astimezone(UTC).date() for e in stored}
    exported = []
    for day in days:
        exported += k.export_rows(backend, day, tmp_path / f"{day}.parquet")
    assert len(exported) == 3
    for row in exported:
        for value in row.values():
            assert SECRET not in str(value)
    assert sum(c.SECRET_MASK in row["body"] for row in exported) == 3

    reader = k.reader(backend)
    reads = [
        reader.timeline(c.Own(k.ALICE), session),
        reader.entries(c.All(), c.KIND_AUDIT),
        reader.recent_audit(c.Own(k.ALICE)),
        reader.list_sessions(c.All()),
    ]
    for answer in reads:
        assert answer.complete, answer.reason
        assert SECRET not in json.dumps(answer.to_json(), default=str)


# --- drains at read time ------------------------------------------------------------------


def test_run_and_bridge_history_drain_the_runs_session_at_read_time(
    backend, spool_root, mounts, base
):
    prompt = k.user_line(SESSION, "<payload>", uuid="u-1", when=base + timedelta(seconds=1))
    answer = k.text_line(SESSION, "final answer", uuid="a-1", when=base + timedelta(seconds=2))
    k.spool(
        spool_root,
        k.W1,
        _accept(R1, SESSION, k.BOB, base),
        _terminal(R1, SESSION, base + timedelta(seconds=3)),
    )
    k.spool(spool_root, k.BRIDGE, _turn(R1, base, conversation="c1", question="status?"))
    k.consume(backend, spool_root)
    k.write_transcript(mounts, k.W1, SESSION, prompt)
    k.tail(backend, mounts)
    # The answer reaches the transcript just before the reads; the tailer has not run.
    k.write_transcript(mounts, k.W1, SESSION, answer)

    clock = k.Clock()
    reader = k.reader(backend, clock=clock, drain=k.drainer(backend, spool_root, mounts))
    for result in (reader.run(c.All(), R1), reader.run_timeline(c.All(), R1)):
        assert result.available and not result.complete
        assert "did not drain" in result.reason
    history = reader.bridge_history(c.Source(k.BRIDGE), conversation="c1")
    assert history.available and not history.complete
    assert clock.sleeps and max(clock.sleeps) <= rd.WAIT_POLL_SECONDS
    assert sum(clock.sleeps) <= 3 * rd.WAIT_SECONDS

    # The tailer, woken by the drain, commits the answer while the read waits.
    woken = k.Clock()

    def tail_once(seconds: float) -> None:
        woken.sleep(seconds)
        k.tail(backend, mounts)

    reader = rd.RecordReader(
        backend,
        k.CONFIG,
        clock=woken,
        sleep=tail_once,
        drain=k.drainer(backend, spool_root, mounts),
    )
    run = reader.run(c.Own(k.BOB), R1)
    assert run.complete, run.reason
    assert run.data["answer"] == "final answer" and run.data["requester"] == k.BOB
    history = reader.bridge_history(c.Source(k.BRIDGE), conversation="c1")
    assert history.complete, history.reason
    [item] = history.data
    assert item["turn"].body["question"] == "status?"
    assert item["run"]["answer"] == "final answer" and item["run"]["status"] == "done"
    timeline = reader.timeline(c.Source(k.BRIDGE), SESSION)
    assert [n.item.kind for n in timeline.data if n.item.kind != c.KIND_RUN] == [
        c.KIND_PROMPT,
        c.KIND_ANSWER,
    ]
