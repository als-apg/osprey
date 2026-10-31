"""A host session's transcripts forwarded to the stack's record through the spool.

Pinned here: a forwarded line keeps the transcript key of its own identity, and
its ``agent_id`` and ``parent_key``; the consumer ingests it under the
transcript route from a host source only, and a key naming another identity is
no forwarded line; a forwarded transcript, tailed through :class:`SpoolForward`
and consumed, is stored exactly as the record's own tailer stores it, and the
record files it once whichever arrives first; a batch is one sealed segment or
nothing; the forward state survives a new forwarder and refuses a corrupt file;
the stack's record has no host root.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import service, writer
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.ingest.forward import ForwardState, SpoolForward
from osprey.agent_record.ingest.spool import SpoolConsumer
from osprey.agent_record.ingest.tailer import TranscriptTailer
from tests.agent_record.test_identities import _config

HOST = "builder"
CONFIG = _config(host=HOST)
SESSION = "5f0c9a3e-1111-4222-8333-944455556666"
PROJECT = "-repo-build"
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


def _when() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _turn(tag: str) -> bytes:
    """A prompt, then a tool call with its result, then a subagent's answer."""
    lines = [
        {
            "parentUuid": None,
            "sessionId": SESSION,
            "type": "user",
            "message": {"role": "user", "content": f"read the {tag} channel"},
            "uuid": f"u1-{tag}",
            "timestamp": _when(),
        },
        {
            "parentUuid": f"u1-{tag}",
            "sessionId": SESSION,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "tool_use", "id": f"toolu-{tag}", "name": "read", "input": {}}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            "uuid": f"a1-{tag}",
            "timestamp": _when(),
        },
        {
            "parentUuid": f"a1-{tag}",
            "sessionId": SESSION,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"toolu-{tag}", "content": "0.41"}
                ],
            },
            "uuid": f"u2-{tag}",
            "timestamp": _when(),
        },
    ]
    return b"".join((json.dumps(line) + "\n").encode() for line in lines)


def _subagent(tag: str) -> bytes:
    line = {
        "parentUuid": None,
        "isSidechain": True,
        "sessionId": SESSION,
        "agentId": "a7",
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": f"the subagent read {tag}"}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        "uuid": f"s1-{tag}",
        "timestamp": _when(),
    }
    return (json.dumps(line) + "\n").encode()


@pytest.fixture
def host_projects(tmp_path) -> Path:
    """The host's ``projects/<encoded build dir>/``, with one session and its subagent."""
    root = tmp_path / "claude" / "projects" / PROJECT
    (root / SESSION / "subagents").mkdir(parents=True)
    (root / f"{SESSION}.jsonl").write_bytes(_turn("first"))
    (root / SESSION / "subagents" / "agent-a7.jsonl").write_bytes(_subagent("first"))
    return root


@pytest.fixture
def spool_root(tmp_path) -> Path:
    path = tmp_path / "spool"
    (path / HOST).mkdir(parents=True)
    return path


def _tailer(sink, host_projects: Path) -> TranscriptTailer:
    return TranscriptTailer(sink, {"host": host_projects}, CONFIG, prefixes={"host": f"{PROJECT}/"})


def _forwarder(tmp_path: Path, spool_root: Path) -> SpoolForward:
    return SpoolForward(spool_root / HOST, ForwardState(tmp_path / "forwarded.json"))


def _entries(backend) -> dict[str, c.Entry]:
    return {e.key: e for e in backend.select(c.ScopeFilter(), c.Query())}


def _comparable(entry: c.Entry) -> tuple:
    return (
        entry.kind,
        entry.identity,
        dict(entry.body),
        entry.session_id,
        entry.agent_id,
        entry.parent_key,
        entry.tool_use_id,
    )


# --- the contract ------------------------------------------------------------------


def _forwarded_line(key: str, **ids) -> c.SpoolLine:
    return c.SpoolLine(
        entry_id="e1", kind=c.KIND_TOOL_CALL, ts=NOW.isoformat(), body={"x": 1}, key=key, **ids
    )


def test_a_forwarded_line_keeps_its_own_transcript_key():
    key = c.transcript_key(HOST, SESSION, "u1", 0)
    line = _forwarded_line(key, agent_id="a7", parent_key="cc:builder:s:p:0")
    assert c.forwarded_transcript(line, HOST)
    assert c.spool_entry_key(line, HOST, receiver_owned=False) == key
    entry = c.entry_from_spool_line(
        line, identity=HOST, receiver_owned=False, on_behalf_of_trusted=False
    )
    assert (entry.key, entry.agent_id, entry.parent_key) == (key, "a7", "cc:builder:s:p:0")


def test_a_key_naming_another_identity_is_no_forwarded_line():
    line = _forwarded_line(c.transcript_key("alice", SESSION, "u1", 0), agent_id="a7")
    assert not c.forwarded_transcript(line, HOST)
    assert c.spool_entry_key(line, HOST, receiver_owned=False) == c.spool_key(HOST, "e1")
    entry = c.entry_from_spool_line(
        line, identity=HOST, receiver_owned=False, on_behalf_of_trusted=False
    )
    assert entry.agent_id == ""


def test_a_spool_line_round_trips_its_agent_and_parent():
    line = _forwarded_line(
        c.transcript_key(HOST, SESSION, "u1", 0), agent_id="a7", parent_key="cc:x:y:z:0"
    )
    again = c.parse_spool_line(line.to_json())
    assert (again.agent_id, again.parent_key) == ("a7", "cc:x:y:z:0")
    with pytest.raises(c.SpoolLineInvalid):
        c.SpoolLine(entry_id="e", kind=c.KIND_AUDIT, ts=NOW.isoformat(), body={}, agent_id=7)  # type: ignore[arg-type]


# --- the consumer ------------------------------------------------------------------


def _segment(directory: Path, *lines: c.SpoolLine) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assert writer.write_batch(directory, lines) is not None


def test_a_host_source_forwards_transcript_kinds(tmp_path, spool_root):
    key = c.transcript_key(HOST, SESSION, "u1", 0)
    _segment(spool_root / HOST, _forwarded_line(key, session_id=SESSION))
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        report = SpoolConsumer(backend, spool_root, config=CONFIG).consume()
        stored = _entries(backend)
    assert (report.appended, report.refused) == (1, 0)
    assert stored[key].kind == c.KIND_TOOL_CALL
    assert stored[key].identity == HOST


def test_a_terminal_may_not_forward_transcript_entries(tmp_path, spool_root):
    key = c.transcript_key("alice", SESSION, "u1", 0)
    _segment(spool_root / "alice", _forwarded_line(key))
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        consumer = SpoolConsumer(backend, spool_root, config=CONFIG)
        report = consumer.consume()
        stored = _entries(backend)
    assert (report.appended, report.refused) == (0, 1)
    assert stored == {}
    (alice,) = [s for s in consumer.health().sources.values() if s.identity == "alice"]
    (refused,) = alice.refused
    assert "may not forward" in refused.reason


def test_a_transcript_kind_without_its_key_stays_refused_over_the_spool(tmp_path, spool_root):
    _segment(spool_root / HOST, _forwarded_line(c.transcript_key("alice", SESSION, "u1", 0)))
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        report = SpoolConsumer(backend, spool_root, config=CONFIG).consume()
    assert (report.appended, report.refused) == (0, 1)


# --- forwarding a transcript -------------------------------------------------------


def test_a_forwarded_transcript_is_stored_as_the_record_tails_it(
    tmp_path, host_projects, spool_root
):
    with EmbeddedBackend(tmp_path / "direct.duckdb") as direct:
        tailed = _tailer(direct, host_projects).poll_once()
        expected = _entries(direct)
    forwarded = _tailer(_forwarder(tmp_path, spool_root), host_projects).poll_once()
    assert forwarded.errors == {}
    assert forwarded.appended == tailed.appended == len(expected)
    with EmbeddedBackend(tmp_path / "stack.duckdb") as stack:
        SpoolConsumer(stack, spool_root, config=CONFIG).consume()
        stored = _entries(stack)
    assert stored.keys() == expected.keys()
    assert {k: _comparable(e) for k, e in stored.items()} == {
        k: _comparable(e) for k, e in expected.items()
    }
    assert any(e.agent_id == "a7" for e in stored.values())
    assert any(e.parent_key for e in stored.values())


def test_the_record_files_a_forwarded_entry_once_whichever_arrives_first(
    tmp_path, host_projects, spool_root
):
    with EmbeddedBackend(tmp_path / "r.duckdb") as backend:
        # The host-local record tailed the file before the takeover.
        first = _tailer(backend, host_projects).poll_once()
        # The host session forwards it all again from a position of its own.
        _tailer(_forwarder(tmp_path, spool_root), host_projects).poll_once()
        consumed = SpoolConsumer(backend, spool_root, config=CONFIG).consume()
        assert (consumed.appended, consumed.duplicates) == (0, first.appended)

        # A turn written after the takeover reaches the record once, forwarded.
        with (host_projects / f"{SESSION}.jsonl").open("ab") as handle:
            handle.write(_turn("second"))
        resumed = _tailer(_forwarder(tmp_path, spool_root), host_projects).poll_once()
        assert resumed.appended == 3
        assert SpoolConsumer(backend, spool_root, config=CONFIG).consume().appended == 3

        # After `osprey down` the host-local record reads the file again: nothing new.
        back = _tailer(backend, host_projects).poll_once()
        assert (back.appended, back.duplicates) == (0, 3)


def test_a_failed_batch_commits_no_position(tmp_path, host_projects, spool_root, monkeypatch):
    monkeypatch.setattr(writer, "write_batch", lambda directory, lines: None)
    forwarder = _forwarder(tmp_path, spool_root)
    report = _tailer(forwarder, host_projects).poll_once()
    assert report.appended == 0
    assert report.errors
    assert (
        forwarder.state.get_state(c.transcript_state_name(HOST, f"{PROJECT}/{SESSION}.jsonl"))
        is None
    )


# --- the batch writer and the forward state ----------------------------------------


def test_a_batch_is_one_sealed_segment(tmp_path):
    directory = tmp_path / "spool" / HOST
    directory.mkdir(parents=True)
    lines = [_forwarded_line(c.transcript_key(HOST, SESSION, f"u{i}", 0)) for i in range(3)]
    assert writer.write_batch(directory, lines) == ["e1", "e1", "e1"]
    (segment,) = directory.iterdir()
    assert segment.name.endswith(c.SEALED_SUFFIX)
    assert [c.parse_spool_line(raw).key for raw in segment.read_bytes().splitlines()] == [
        line.key for line in lines
    ]
    assert writer.write_batch(directory, []) == []
    assert len(list(directory.iterdir())) == 1


def test_a_failed_batch_leaves_no_partial_segment(tmp_path, monkeypatch):
    directory = tmp_path / "spool" / HOST
    directory.mkdir(parents=True)

    def refuse(fd):  # noqa: ARG001 - mirrors os.fsync
        raise OSError("fsync failed")

    monkeypatch.setattr(writer.os, "fsync", refuse)
    line = _forwarded_line(c.transcript_key(HOST, SESSION, "u1", 0))
    assert writer.write_batch(directory, [line]) is None
    assert [p.name for p in directory.iterdir()] == [c.WRITER_ERRORS_FILE]


def test_the_forward_state_survives_a_new_forwarder(tmp_path):
    path = tmp_path / "forwarded.json"
    ForwardState(path).put_state("transcript:builder:a.jsonl", {"generation": 0, "offset": 9})
    assert ForwardState(path).get_state("transcript:builder:a.jsonl") == {
        "generation": 0,
        "offset": 9,
    }
    assert os.stat(path).st_mode & 0o777 == 0o664


def test_a_corrupt_forward_state_is_refused(tmp_path):
    path = tmp_path / "forwarded.json"
    path.write_text("[1, 2]")
    with pytest.raises(c.RecordBackendError, match="not a JSON object"):
        ForwardState(path)
    path.write_text("{")
    with pytest.raises(c.RecordBackendError, match="unreadable"):
        ForwardState(path)


# --- the stack's record -------------------------------------------------------------


def test_the_stacks_record_has_no_host_root(tmp_path):
    settings = service._settings(
        {**CONFIG, "services": {}}, {service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "mounts")}
    )
    roots, prefixes = service._transcript_roots(CONFIG, settings)
    assert "host" not in roots
    assert prefixes == {}
    assert roots  # the terminals' volumes stay roots
