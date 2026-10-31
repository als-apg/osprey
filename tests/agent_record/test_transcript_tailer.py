"""The transcript tailer: mounted ``projects/**/*.jsonl`` into the record.

Pinned here: offsets keyed by identity and path relative to ``projects/``; a
rewrite (new inode) or compaction (shrink) bumps the generation and re-reads
with duplicates dropped; a late subagent file is picked up; an unreadable
root or file is that source's error; lines older than retention are skipped
and counted (with and without an arrival day); symlinks are never followed;
a restart resumes from the committed offset with no duplicate; a partial line
waits for its newline; secrets never reach the stored rows; the poll interval
is at most five seconds.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.ingest import tailer as tl

ROOT = "alice-claude-projects"
IDENTITY = "alice"
SESSION = "5f0c9a3e-1111-4222-8333-944455556666"
PROJECT = "-data-workspace"
MAIN = f"{PROJECT}/{SESSION}.jsonl"
SUBAGENT = f"{PROJECT}/{SESSION}/subagents/agent-a7.jsonl"
CONFIG: dict = {"deployment": {"host_identity": "builder"}}


def _ts(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _user(text: str, *, uuid: str, when: datetime | None = None) -> bytes:
    obj = {
        "parentUuid": None,
        "isSidechain": False,
        "userType": "external",
        "cwd": "/data/workspace",
        "sessionId": SESSION,
        "version": "2.1.300",
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": uuid,
        "timestamp": _ts(when or datetime.now(UTC)),
    }
    return (json.dumps(obj) + "\n").encode("utf-8")


def _assistant(text: str, *, uuid: str, agent_id: str | None = None) -> bytes:
    obj = {
        "parentUuid": "u-1",
        "isSidechain": agent_id is not None,
        "sessionId": SESSION,
        "version": "2.1.300",
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        "uuid": uuid,
        "timestamp": _ts(datetime.now(UTC)),
    }
    if agent_id is not None:
        obj["agentId"] = agent_id
    return (json.dumps(obj) + "\n").encode("utf-8")


@pytest.fixture
def backend(tmp_path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        yield b


@pytest.fixture
def projects(tmp_path) -> Path:
    root = tmp_path / "mounts" / ROOT
    (root / PROJECT).mkdir(parents=True)
    return root


def _write(root: Path, relpath: str, data: bytes, *, mode: str = "wb") -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        fh.write(data)
    return path


def _tailer(backend, root: Path, **kw) -> tl.TranscriptTailer:
    return tl.TranscriptTailer(backend, {ROOT: root}, CONFIG, **kw)


def _rows(backend) -> list[c.Entry]:
    return backend.select(c.ScopeFilter(), c.Query())


def _state(backend, relpath: str = MAIN) -> dict:
    value = backend.get_state(c.transcript_state_name(IDENTITY, relpath))
    assert value is not None
    return dict(value)


def test_lines_are_appended_and_the_offset_is_committed(backend, projects):
    data = _user("hello", uuid="u-1") + _assistant("hi", uuid="a-1")
    _write(projects, MAIN, data)
    report = _tailer(backend, projects).poll_once()
    assert report.appended == 2
    assert not report.errors
    assert {e.kind for e in _rows(backend)} == {c.KIND_PROMPT, c.KIND_ANSWER}
    assert all(e.identity == IDENTITY for e in _rows(backend))
    state = _state(backend)
    assert c.TranscriptOffset.from_state(state) == c.TranscriptOffset(0, len(data), True)
    assert state[tl.INODE_FIELD] == (projects / MAIN).stat().st_ino


def test_a_restart_resumes_from_the_committed_offset_with_no_duplicate(backend, projects):
    _write(projects, MAIN, _user("one", uuid="u-1"))
    _tailer(backend, projects).poll_once()
    _write(projects, MAIN, _assistant("two", uuid="a-1"), mode="ab")
    report = _tailer(backend, projects).poll_once()
    assert (report.appended, report.duplicates) == (1, 0)
    assert len(_rows(backend)) == 2


def test_an_unchanged_file_is_not_reread(backend, projects):
    _write(projects, MAIN, _user("one", uuid="u-1"))
    tailer = _tailer(backend, projects)
    tailer.poll_once()
    report = tailer.poll_once()
    assert (report.appended, report.duplicates, report.files_read) == (0, 0, 0)


def test_a_partial_line_waits_for_its_newline(backend, projects):
    whole = _user("one", uuid="u-1")
    partial = _assistant("two", uuid="a-1")
    _write(projects, MAIN, whole + partial[:20])
    tailer = _tailer(backend, projects)
    assert tailer.poll_once().appended == 1
    state = c.TranscriptOffset.from_state(_state(backend))
    assert state == c.TranscriptOffset(0, len(whole), False)
    _write(projects, MAIN, partial[20:], mode="ab")
    assert tailer.poll_once().appended == 1
    assert c.TranscriptOffset.from_state(_state(backend)).fully_read


def test_a_rewrite_bumps_the_generation_and_drops_duplicates(backend, projects):
    first = _user("one", uuid="u-1") + _assistant("two", uuid="a-1")
    _write(projects, MAIN, first)
    tailer = _tailer(backend, projects)
    tailer.poll_once()
    replacement = projects / f"{MAIN}.new"
    replacement.write_bytes(first + _assistant("three", uuid="a-2"))
    os.rename(replacement, projects / MAIN)
    report = tailer.poll_once()
    assert (report.appended, report.duplicates) == (1, 2)
    assert c.TranscriptOffset.from_state(_state(backend)).generation == 1
    assert len(_rows(backend)) == 3


def test_compaction_in_place_bumps_the_generation(backend, projects):
    _write(projects, MAIN, _user("one", uuid="u-1") + _assistant("two" * 50, uuid="a-1"))
    tailer = _tailer(backend, projects)
    tailer.poll_once()
    compacted = _user("one", uuid="u-1")
    with open(projects / MAIN, "r+b") as fh:
        fh.truncate(0)
        fh.write(compacted)
    report = tailer.poll_once()
    assert (report.appended, report.duplicates) == (0, 1)
    state = c.TranscriptOffset.from_state(_state(backend))
    assert state == c.TranscriptOffset(1, len(compacted), True)


def test_a_late_and_growing_subagent_file_is_picked_up(backend, projects):
    _write(projects, MAIN, _user("go", uuid="u-1"))
    tailer = _tailer(backend, projects)
    tailer.poll_once()
    _write(projects, SUBAGENT, _assistant("sub one", uuid="s-1", agent_id="a7"))
    assert tailer.poll_once().appended == 1
    _write(projects, SUBAGENT, _assistant("sub two", uuid="s-2", agent_id="a7"), mode="ab")
    assert tailer.poll_once().appended == 1
    sub = [e for e in _rows(backend) if e.agent_id == "a7"]
    assert len(sub) == 2
    assert c.TranscriptOffset.from_state(_state(backend, SUBAGENT)).fully_read


def test_an_unreadable_file_is_that_sources_error(backend, projects):
    path = _write(projects, MAIN, _user("one", uuid="u-1"))
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("running as a user that reads mode-000 files")
        tailer = _tailer(backend, projects)
        report = tailer.poll_once()
        assert f"{ROOT}:{MAIN}" in report.errors
        assert MAIN in tailer.health[ROOT].file_errors
        assert not tailer.health[ROOT].ok
    finally:
        path.chmod(0o644)
    report = tailer.poll_once()
    assert not report.errors
    assert tailer.health[ROOT].ok
    assert report.appended == 1


def test_a_missing_root_is_that_sources_error(backend, tmp_path):
    tailer = _tailer(backend, tmp_path / "absent")
    report = tailer.poll_once()
    assert ROOT in report.errors
    assert tailer.health[ROOT].error is not None


def test_an_unknown_root_name_is_an_error_not_silence(backend, projects):
    tailer = tl.TranscriptTailer(backend, {"not-a-root": projects}, CONFIG)
    assert "not-a-root" in tailer.poll_once().errors
    assert tailer.health["not-a-root"].identity is None


def test_the_host_root_maps_to_the_host_identity(backend, projects):
    _write(projects, MAIN, _user("one", uuid="u-1"))
    tl.TranscriptTailer(backend, {"host": projects}, CONFIG).poll_once()
    assert {e.identity for e in _rows(backend)} == {"builder"}


def test_lines_older_than_retention_are_skipped_on_a_fresh_record(backend, projects):
    now = datetime.now(UTC)
    old = _user("old", uuid="u-old", when=now - timedelta(days=30))
    new = _user("new", uuid="u-new", when=now - timedelta(days=1))
    _write(projects, MAIN, old + new)
    tailer = _tailer(backend, projects, retention_days=14)
    report = tailer.poll_once(now=now)
    assert (report.appended, report.skipped_retention) == (1, 1)
    assert tailer.health[ROOT].skipped_retention == 1
    assert c.TranscriptOffset.from_state(_state(backend)).fully_read


def test_a_line_older_than_the_oldest_arrival_day_is_not_reinserted(backend, projects):
    # Once retention reaches past the oldest arrival day, that day is the floor:
    # a line from before it was expired and must not come back on a re-read.
    real_now = datetime.now(UTC)
    _write(projects, f"{PROJECT}/other.jsonl", _user("seed", uuid="u-seed"))
    tailer = _tailer(backend, projects, retention_days=60)
    tailer.poll_once(now=real_now)
    assert backend.health().oldest_arrival_day is not None
    later = real_now + timedelta(days=90)
    stale = _user("stale", uuid="u-stale", when=real_now - timedelta(days=2))
    _write(projects, MAIN, stale)
    report = tailer.poll_once(now=later)
    assert (report.appended, report.skipped_retention) == (0, 1)
    assert all(e.body.get("text") != "stale" for e in _rows(backend))


def test_a_young_record_still_takes_an_unexpired_backlog_line_on_retry(backend, projects):
    now = datetime.now(UTC)
    _write(projects, f"{PROJECT}/other.jsonl", _user("seed", uuid="u-seed"))
    _write(projects, MAIN, _user("backlog", uuid="u-back", when=now - timedelta(days=3)))
    failing = _tailer(_FailingAppend(backend), projects, retention_days=14)
    failing.poll_once(now=now)
    tailer = _tailer(backend, projects, retention_days=14)
    tailer.poll_once(now=now)
    assert backend.health().oldest_arrival_day is not None
    _write(
        projects,
        f"{PROJECT}/late.jsonl",
        _user("late", uuid="u-late", when=now - timedelta(days=5)),
    )
    report = tailer.poll_once(now=now)
    assert (report.appended, report.skipped_retention) == (1, 0)
    assert {"backlog", "late", "seed"} <= {e.body.get("text") for e in _rows(backend)}


def test_a_symlink_never_escapes_the_root(backend, projects, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_file = outside / "leak.jsonl"
    secret_file.write_bytes(_user("from outside", uuid="u-out"))
    (projects / PROJECT / "linked.jsonl").symlink_to(secret_file)
    (projects / "linked-dir").symlink_to(outside, target_is_directory=True)
    _write(projects, MAIN, _user("inside", uuid="u-in"))
    report = _tailer(backend, projects).poll_once()
    assert report.appended == 1
    assert not report.errors
    assert [e.body.get("text") for e in _rows(backend)] == ["inside"]


def test_a_symlinked_root_is_refused(backend, projects, tmp_path):
    link = tmp_path / "link-root"
    link.symlink_to(projects, target_is_directory=True)
    _write(projects, MAIN, _user("inside", uuid="u-in"))
    report = _tailer(backend, link).poll_once()
    assert ROOT in report.errors
    assert _rows(backend) == []


def test_only_jsonl_regular_files_are_read(backend, projects):
    (projects / PROJECT / "notes.txt").write_bytes(_user("txt", uuid="u-txt"))
    os.mkfifo(projects / PROJECT / "pipe.jsonl")
    _write(projects, MAIN, _user("inside", uuid="u-in"))
    report = _tailer(backend, projects).poll_once()
    assert report.appended == 1
    assert not report.errors


def test_a_secret_in_a_line_is_absent_from_the_stored_rows(backend, projects):
    secret = "sk-ant-api03-supersecretvalue"
    _write(projects, MAIN, _user(f"my key is {secret}", uuid="u-1"))
    _tailer(backend, projects, secret_set={secret}).poll_once()
    rows = _rows(backend)
    assert rows
    stored = json.dumps([dict(e.body) for e in rows])
    assert secret not in stored
    assert c.SECRET_MASK in stored


class _FailingAppend:
    """A backend whose state works but whose append refuses."""

    def __init__(self, inner):
        self.inner = inner

    def append(self, _entries):
        raise c.RecordBackendUnavailable("down")

    def get_state(self, name):
        return self.inner.get_state(name)

    def put_state(self, name, value):
        self.inner.put_state(name, value)

    def health(self):
        return self.inner.health()


def test_a_failed_append_commits_no_offset(backend, projects):
    _write(projects, MAIN, _user("one", uuid="u-1"))
    report = _tailer(_FailingAppend(backend), projects).poll_once()
    assert f"{ROOT}:{MAIN}" in report.errors
    assert backend.get_state(c.transcript_state_name(IDENTITY, MAIN)) is None
    assert _tailer(backend, projects).poll_once().appended == 1


def test_last_seen_moves_only_when_lines_are_committed(backend, projects):
    tailer = _tailer(backend, projects)
    t0 = datetime(2026, 10, 23, 12, tzinfo=UTC)
    tailer.poll_once(now=t0)
    assert tailer.health[ROOT].last_seen is None
    _write(projects, MAIN, _user("one", uuid="u-1"))
    tailer.poll_once(now=t0 + timedelta(seconds=5))
    assert tailer.health[ROOT].last_seen == t0 + timedelta(seconds=5)
    tailer.poll_once(now=t0 + timedelta(seconds=10))
    assert tailer.health[ROOT].last_seen == t0 + timedelta(seconds=5)
    assert tailer.health[ROOT].last_polled == t0 + timedelta(seconds=10)


def test_it_polls_at_least_every_five_seconds(backend, projects):
    assert 0 < tl.POLL_INTERVAL_S <= tl.MAX_POLL_INTERVAL_S <= 5
    tailer = _tailer(backend, projects)
    with pytest.raises(ValueError):
        asyncio.run(tailer.run(interval=6))


def test_a_file_the_parser_refuses_is_that_files_error_and_the_others_still_land(backend, projects):
    bad = f"{PROJECT}/a:b.jsonl"
    _write(projects, bad, b"not json\n")
    _write(projects, MAIN, _user("good", uuid="u-good"))
    tailer = _tailer(backend, projects)
    report = tailer.poll_once()
    assert f"{ROOT}:{bad}" in report.errors
    assert bad in tailer.health[ROOT].file_errors
    assert [e.body.get("text") for e in _rows(backend)] == ["good"]


def test_run_survives_a_failed_poll_and_records_it(backend, projects):
    tailer = _tailer(backend, projects)
    calls = []

    def boom(now=None):
        calls.append(now)
        if len(calls) >= 2:
            raise asyncio.CancelledError
        raise RuntimeError("kaboom")

    tailer.poll_once = boom  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tailer.run(interval=0.01))
    assert len(calls) == 2
    assert "kaboom" in (tailer.health[ROOT].error or "")


def test_an_in_place_rewrite_of_equal_size_bumps_the_generation(backend, projects):
    first = _user("aaaa", uuid="u-1")
    _write(projects, MAIN, first)
    _tailer(backend, projects).poll_once()
    path = projects / MAIN
    replacement = _user("bbbbbbbb", uuid="u-2")
    with open(path, "r+b") as fh:
        fh.write(replacement)
    report = _tailer(backend, projects).poll_once()
    assert report.appended == 1
    assert c.TranscriptOffset.from_state(_state(backend)).generation == 1
    assert {e.body.get("text") for e in _rows(backend)} == {"aaaa", "bbbbbbbb"}


def test_a_directory_nested_too_deep_is_reported(backend, projects):
    deep = "/".join(f"d{i}" for i in range(tl._MAX_DEPTH + 1))
    _write(projects, f"{deep}/x.jsonl", _user("deep", uuid="u-deep"))
    report = _tailer(backend, projects).poll_once()
    assert any("deeper than" in why for why in report.errors.values())
