"""The drain wait mark, resolved by the service (P3-D1, A2-1).

Pins, through the running service: a ``run:`` line and an OTLP batch written
just before a drained read are in its answer, in the shared workspace mode too
(the line in the second worker's spool directory); a transcript step appended
just before a drained ``timeline`` read is in it; an operator's drain covers a
line pending in a terminal's spool; a stalled consumer answers
``complete: false`` within the reader's bound; a caller's drain covers only
what its own scope sees, so it cannot drain (or wait on) another identity.
Also the consumer's and the tailer's snapshot/committed queries the drain is
built on, and the tailer's wake.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import identities as ids
from osprey.agent_record import reader as rd
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.ingest.spool import ConsumeReport, SpoolConsumer
from osprey.agent_record.ingest.tailer import TranscriptTailer
from tests.agent_record.contract_suite import _kit as k
from tests.agent_record.test_identities import _config
from tests.agent_record.test_otlp_receiver import _decision, _request, _usage
from tests.agent_record.test_service import (
    _auth,
    _b64,
    _eventually,
    _fresh_writer,  # noqa: F401  (autouse: no writer state leaks between tests)
    _serving,
    _token,
)
from tests.agent_record.test_spool_consumer import _line, _sealed

SECRET = "s3cr3t-token-value-0123456789"
NOW = datetime.now(UTC).replace(microsecond=0)
OTLP_SESSION = "bb1af18d-a6d1-4ce8-a1f6-b088b297e9b8"  # the OTLP builders' session
SESSION = "sess-drain-1"
PROJECT = "-data-workspace"
DRAIN = json.dumps({"drain": {"session_id": None}})


def _drain(session_id: str) -> str:
    return json.dumps({"drain": {"session_id": session_id}})


def _env(tmp_path: Path, config: dict) -> dict[str, str]:
    from osprey.agent_record import credentials as creds

    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    return {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps([SECRET])),
        service.SPOOL_ROOT_ENV: str(tmp_path / "spool"),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
    }


def _cfg(tmp_path: Path, *, shared: bool = False) -> dict:
    cfg = _config(workers=2, shared=shared, bridges=("teams_bridge",))
    cfg["project_root"] = str(tmp_path)
    return cfg


@pytest.fixture
def backend(tmp_path):
    with EmbeddedBackend(tmp_path / "injected.duckdb") as b:
        yield b


@pytest.fixture
def slow_ingest(monkeypatch):
    """Ingest that, unwoken, would not run again within any read's wait bound."""
    monkeypatch.setattr(service, "SPOOL_INTERVAL_S", 60.0)
    monkeypatch.setattr(service, "POLL_INTERVAL_S", 5.0)


@pytest.fixture
def stalled(monkeypatch):
    """A consumer that never commits anything, and a reader that waits one second."""
    monkeypatch.setattr(SpoolConsumer, "consume", lambda self: ConsumeReport())
    monkeypatch.setattr(
        service, "RecordReader", functools.partial(rd.RecordReader, wait_seconds=1.0)
    )


def _accept(run_id: str, session: str) -> c.SpoolLine:
    body = {"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session, "requester": "bob"}
    return _line(c.KIND_RUN, body, run_id=run_id, session_id=session)


def _terminal(run_id: str, session: str) -> c.SpoolLine:
    body = {"phase": c.RUN_TERMINAL, "status": "done"}
    return _line(c.KIND_RUN, body, run_id=run_id, session_id=session)


def _transcript(tmp_path: Path, root: str, session: str, data: bytes) -> Path:
    path = tmp_path / "transcripts" / root / PROJECT / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(data)
    return path


async def _ingested(client, login: str, kind: str, what: str) -> None:
    """Wait (untimed) until ingest has run once and *what* is readable by *login*."""

    async def seen():
        answer = await client.get("/v1/read/entries", params={"kind": kind}, headers=_auth(login))
        return any(i["body"].get("what") == what for i in answer.json().get("data") or [])

    await _eventually(seen)


def _what(answer: dict) -> set:
    return {item["body"].get("what") for item in answer["data"]}


# --- through the service ---------------------------------------------------------------


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.usefixtures("slow_ingest")
async def test_a_run_line_written_just_before_a_read_is_in_the_answer(tmp_path, backend, shared):
    config = _cfg(tmp_path, shared=shared)
    worker_dir = "dispatch-worker-2"  # the second worker's spool directory
    login = ids.SHARED_DISPATCH_IDENTITY if shared else worker_dir
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    spool = tmp_path / "spool"
    _sealed(spool / worker_dir, _line(c.KIND_AUDIT, {"what": "warm-up"}))
    async with _serving(app) as client:
        await _ingested(client, login, c.KIND_AUDIT, "warm-up")
        _sealed(spool / worker_dir, _accept("run-d1", SESSION), _terminal("run-d1", SESSION))
        started = time.monotonic()
        for who in (login, "operator"):
            answer = (
                await client.get("/v1/read/run", params={"run_id": "run-d1"}, headers=_auth(who))
            ).json()
            assert answer["status"] == "ok"
            assert answer["complete"], answer["reason"]
            assert answer["data"]["status"] == "done" and answer["data"]["session_id"] == SESSION
        assert time.monotonic() - started < service.SPOOL_INTERVAL_S / 2, "the drain woke ingest"


async def test_an_otlp_batch_received_just_before_a_read_is_in_the_answer(tmp_path, backend):
    config = _cfg(tmp_path)
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    async with _serving(app) as client:
        payload = _request(_decision("toolu_drain"), _usage())
        response = await client.post(
            "/v1/logs",
            content=json.dumps(payload).encode(),
            headers=_auth("alice") | {"content-type": "application/json"},
        )
        assert response.status_code == 200
        answer = (
            await client.get(
                "/v1/read/timeline",
                params={"session_id": OTLP_SESSION, "wait": _drain(OTLP_SESSION)},
                headers=_auth("alice"),
            )
        ).json()
        assert answer["complete"], answer["reason"]
        assert "toolu_drain" in json.dumps(answer["data"])


@pytest.mark.usefixtures("slow_ingest")
async def test_a_transcript_step_appended_just_before_a_drained_timeline_read_is_in_it(
    tmp_path, backend
):
    config = _cfg(tmp_path)
    root = "alice-claude-projects"
    first = k.user_line(SESSION, "first step", uuid="u-1", when=NOW - timedelta(seconds=5))
    _transcript(tmp_path, root, SESSION, first)
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    async with _serving(app) as client:

        async def first_polled():
            answer = await client.get(
                "/v1/read/timeline", params={"session_id": SESSION}, headers=_auth("alice")
            )
            return answer.json()["data"]

        await _eventually(first_polled)
        second = k.text_line(SESSION, "second step", uuid="a-1", when=NOW)
        _transcript(tmp_path, root, SESSION, second)
        started = time.monotonic()
        answer = (
            await client.get(
                "/v1/read/timeline",
                params={"session_id": SESSION, "wait": _drain(SESSION)},
                headers=_auth("alice"),
            )
        ).json()
        assert answer["complete"], answer["reason"]
        assert "second step" in json.dumps(answer["data"])
        assert time.monotonic() - started < service.POLL_INTERVAL_S - 1, "the drain woke the tailer"


@pytest.mark.usefixtures("slow_ingest")
async def test_an_operator_drain_covers_a_line_pending_in_a_terminals_spool(tmp_path, backend):
    config = _cfg(tmp_path)
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    spool = tmp_path / "spool"
    _sealed(spool / "alice", _line(c.KIND_AUDIT, {"what": "warm-up"}))
    async with _serving(app) as client:
        await _ingested(client, "operator", c.KIND_AUDIT, "warm-up")
        _sealed(spool / "alice", _line(c.KIND_FEEDBACK, {"what": "pending"}))
        answer = (
            await client.get(
                "/v1/read/entries",
                params={"kind": c.KIND_FEEDBACK, "wait": DRAIN},
                headers=_auth("operator"),
            )
        ).json()
        assert answer["complete"], answer["reason"]
        assert _what(answer) == {"pending"}


@pytest.mark.usefixtures("stalled")
async def test_a_stalled_consumer_answers_incomplete_within_the_bound(tmp_path, backend):
    config = _cfg(tmp_path)
    _sealed(tmp_path / "spool" / "alice", _line(c.KIND_FEEDBACK, {"what": "stuck"}))
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    async with _serving(app) as client:
        started = time.monotonic()
        answer = (
            await client.get(
                "/v1/read/entries",
                params={"kind": c.KIND_FEEDBACK, "wait": DRAIN},
                headers=_auth("alice"),
            )
        ).json()
        elapsed = time.monotonic() - started
        assert answer["status"] == "ok" and answer["complete"] is False
        assert "wait mark" in answer["reason"]
        assert 1.0 <= elapsed < 5.0


@pytest.mark.usefixtures("stalled")
async def test_a_caller_cannot_drain_another_identity(tmp_path, backend):
    config = _cfg(tmp_path)
    # Bob's line is stuck; alice's drain does not cover it, so she does not wait on it.
    _sealed(tmp_path / "spool" / "bob", _line(c.KIND_FEEDBACK, {"what": "bobs"}))
    _sealed(tmp_path / "spool" / "dispatch-worker-1", _line(c.KIND_AUDIT, {"what": "workers"}))
    app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
    async with _serving(app) as client:
        for login in ("alice", "teams-bridge"):
            answer = (
                await client.get(
                    "/v1/read/entries",
                    params={"kind": c.KIND_FEEDBACK, "wait": DRAIN},
                    headers=_auth(login),
                )
            ).json()
            assert answer["complete"], (login, answer["reason"])
        # A wait mark cannot name whose spool to drain.
        named = await client.get(
            "/v1/read/entries",
            params={
                "kind": c.KIND_FEEDBACK,
                "wait": json.dumps({"drain": {"session_id": None, "identity": "bob"}}),
            },
            headers=_auth("alice"),
        )
        assert named.status_code == 400
        removed = await client.get(
            "/v1/read/entries",
            params={"kind": c.KIND_FEEDBACK, "wait": json.dumps({"spool": [["bob", "e1"]]})},
            headers=_auth("alice"),
        )
        assert removed.status_code == 400
        # Bob's own drain does cover it, and waits.
        bobs = (
            await client.get(
                "/v1/read/entries",
                params={"kind": c.KIND_FEEDBACK, "wait": DRAIN},
                headers=_auth("bob"),
            )
        ).json()
        assert bobs["complete"] is False


# --- what a drain covers ------------------------------------------------------------------


def test_drain_visible_follows_the_scope():
    config = _config(workers=2, shared=True, bridges=("teams_bridge",))
    assert service.drain_visible(config, c.All()) is None
    assert service.drain_visible(config, c.Reader("ops")) is None
    own = service.drain_visible(config, c.Own("alice"))
    assert own("alice") and not own("bob") and not own(ids.SHARED_DISPATCH_IDENTITY)
    worker = service.drain_visible(config, c.Source("dispatch-worker-1"))
    assert worker(ids.SHARED_DISPATCH_IDENTITY) and not worker("alice")
    bridge = service.drain_visible(
        config, c.Source("teams-bridge"), frozenset({"dispatch-worker-2"})
    )
    assert bridge("teams-bridge") and bridge(ids.SHARED_DISPATCH_IDENTITY) and not bridge("bob")


def test_a_drain_covers_the_otlp_dirs_of_logins_mapping_to_the_identity(tmp_path, backend):
    config = _config(workers=2, shared=True)
    root = tmp_path / "spool"
    for directory in (".otlp/dispatch-worker-shared", ".otlp/alice", "dispatch-worker-1", "bob"):
        _sealed(root / directory, _line(c.KIND_AUDIT))
    consumer = SpoolConsumer(backend, root, config=config)
    marks = consumer.snapshot(service.drain_visible(config, c.Source("dispatch-worker-2")))
    assert sorted(str(m.directory.relative_to(root)) for m in marks) == [
        ".otlp/dispatch-worker-shared",
        "dispatch-worker-1",
    ]
    assert len(consumer.snapshot()) == 4


# --- the consumer's committed-offset queries -------------------------------------------------


def _open_segment(directory: Path, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / c.SegmentName("h", 7, time.time_ns(), sealed=False).name
    path.write_bytes(data)
    return path


def test_an_open_segment_is_drained_to_its_last_complete_line(tmp_path, backend):
    root = tmp_path / "spool"
    done = (_line(c.KIND_AUDIT).to_json() + "\n").encode()
    path = _open_segment(root / "alice", done + b'{"partial')
    consumer = SpoolConsumer(backend, root, config=_config())
    [mark] = consumer.snapshot()
    assert mark.offset == len(done)
    assert not consumer.drained([mark])
    consumer.consume()
    assert consumer.drained([mark]), "a writer mid-line holds no drain"
    # Sealing renames the segment; what its open name committed still counts.
    sealed = path.with_name(path.name[: -len(c.OPEN_SUFFIX)] + c.SEALED_SUFFIX)
    os.rename(path, sealed)
    assert consumer.drained([mark])


def test_a_deleted_segment_counts_as_drained(tmp_path, backend):
    root = tmp_path / "spool"
    path = _sealed(root / "alice", _line(c.KIND_AUDIT))
    consumer = SpoolConsumer(backend, root, config=_config())
    marks = consumer.snapshot()
    path.unlink()
    assert consumer.drained(marks)


def test_a_drain_on_a_failing_backend_raises(tmp_path):
    root = tmp_path / "spool"
    _sealed(root / "alice", _line(c.KIND_AUDIT))

    class Down:
        def get_state(self, _name):
            raise c.RecordBackendUnavailable("down")

    consumer = SpoolConsumer(Down(), root, config=_config())  # type: ignore[arg-type]
    with pytest.raises(c.RecordBackendError):
        consumer.drained(consumer.snapshot())


# --- the tailer's committed-offset queries and wake --------------------------------------------


def _tailer(backend, mounts: Path, config=None) -> TranscriptTailer:
    roots = {name: mounts / name for name in k.ROOTS.values()}
    return TranscriptTailer(backend, roots, config or k.CONFIG)


def test_a_session_snapshot_covers_its_subagents_under_visible_roots_only(tmp_path, backend):
    mounts = tmp_path / "mounts"
    main = k.write_transcript(
        mounts, k.ALICE, SESSION, k.user_line(SESSION, "hi", uuid="u-1", when=NOW)
    )
    sub = main.parent / SESSION / "subagents" / "agent-a1.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_bytes(k.user_line(SESSION, "sub", uuid="u-2", when=NOW) + b'{"partial')
    k.write_transcript(mounts, k.BOB, SESSION, k.user_line(SESSION, "bob", uuid="u-3", when=NOW))
    k.write_transcript(
        mounts, k.ALICE, "sess-other", k.user_line("sess-other", "x", uuid="u-4", when=NOW)
    )
    tailer = _tailer(backend, mounts)
    marks = tailer.snapshot(SESSION, service.drain_visible(k.CONFIG, c.Own(k.ALICE)))
    assert sorted(m.relpath for m in marks) == [
        f"{PROJECT}/{SESSION}.jsonl",
        f"{PROJECT}/{SESSION}/subagents/agent-a1.jsonl",
    ]
    assert {m.relpath: m.offset for m in marks}[
        f"{PROJECT}/{SESSION}/subagents/agent-a1.jsonl"
    ] == (len(k.user_line(SESSION, "sub", uuid="u-2", when=NOW)))
    assert {m.identity for m in tailer.snapshot(SESSION)} == {k.ALICE, k.BOB}
    assert not tailer.drained(marks)
    tailer.poll_once()
    assert tailer.drained(marks)


def test_a_replaced_transcript_is_drained_only_in_its_new_generation(tmp_path, backend):
    mounts = tmp_path / "mounts"
    path = k.write_transcript(
        mounts, k.ALICE, SESSION, k.user_line(SESSION, "one", uuid="u-1", when=NOW)
    )
    tailer = _tailer(backend, mounts)
    tailer.poll_once()
    replacement = path.with_name("replacement")
    replacement.write_bytes(k.user_line(SESSION, "two", uuid="u-2", when=NOW))
    os.replace(replacement, path)
    [mark] = tailer.snapshot(SESSION)
    assert mark.generation == 1
    assert not tailer.drained([mark])
    tailer.poll_once()
    assert tailer.drained([mark])


def test_wake_starts_the_next_poll_before_the_interval(tmp_path, backend):
    tailer = _tailer(backend, tmp_path / "mounts")
    polls: list[float] = []
    tailer.poll_once = lambda now=None: polls.append(time.monotonic())  # type: ignore[method-assign]
    tailer.wake()  # not running: nothing to wake, never raises

    async def scenario():
        task = asyncio.create_task(tailer.run(interval=5.0))
        while not polls:
            await asyncio.sleep(0.01)
        await asyncio.to_thread(tailer.wake)  # from another thread, as a read does
        while len(polls) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(scenario(), timeout=3.0))
    assert polls[1] - polls[0] < 2.0
