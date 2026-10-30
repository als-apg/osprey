"""The spool consumer: identity from the directory, the allow-list, trusted
``on_behalf_of``, run first-accept-wins, checkpoints by segment name and inode,
deletion of committed ``.sealed`` segments only, an outage that loses nothing,
scrubbed bodies, and hostile files set aside.
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import writer
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.ingest.spool import QUARANTINE_SUFFIX, SpoolConsumer
from tests.agent_record.test_identities import _config

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
SECRET = "s3cr3t-token-value-0123456789"


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


@pytest.fixture
def backend(tmp_path):
    with EmbeddedBackend(tmp_path / "record.duckdb") as b:
        yield b


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "spool"
    path.mkdir()
    return path


CONFIG = _config(workers=1, lanes=True)


def _consumer(backend, root, **kwargs) -> SpoolConsumer:
    kwargs.setdefault("config", CONFIG)
    return SpoolConsumer(backend, root, **kwargs)


_counter = iter(range(1, 10**9))


def _line(kind: str = c.KIND_AUDIT, body: dict | None = None, **ids) -> c.SpoolLine:
    ids.setdefault("entry_id", f"e{next(_counter)}")
    return c.SpoolLine(
        kind=kind, ts=ids.pop("ts", NOW.isoformat()), body=body or {"what": "x"}, **ids
    )


def _sealed(directory: Path, *lines: c.SpoolLine | bytes, start_ns: int | None = None) -> Path:
    """A sealed segment holding *lines*, written by hand."""
    directory.mkdir(parents=True, exist_ok=True)
    name = c.SegmentName("h", 7, start_ns or next(_counter), sealed=True).name
    path = directory / name
    path.write_bytes(
        b"".join(
            raw if isinstance(raw, bytes) else (raw.to_json() + "\n").encode() for raw in lines
        )
    )
    return path


def _all(backend) -> list[c.Entry]:
    return backend.select(c.ScopeFilter(), c.Query())


def _by_key(backend) -> dict[str, c.Entry]:
    return {e.key: e for e in _all(backend)}


def _segments(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if c.parse_segment_name(p.name))


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Flaky:
    """A backend that fails every call while ``down``."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.down = False

    def _check(self) -> None:
        if self.down:
            raise c.RecordBackendUnavailable("backend is down")

    def append(self, entries):
        self._check()
        return self.inner.append(entries)

    def select(self, scope_filter, query):
        self._check()
        return self.inner.select(scope_filter, query)

    def get_state(self, name):
        self._check()
        return self.inner.get_state(name)

    def put_state(self, name, value):
        self._check()
        return self.inner.put_state(name, value)


# --- identity and keys ---------------------------------------------------------------


def test_identity_comes_from_the_directory_and_lines_become_sp_keys(backend, root):
    line = _line(session_id="s-1")
    path = _sealed(root / "alice", line)
    report = _consumer(backend, root).consume()
    assert report.appended == 1
    entry = _by_key(backend)[c.spool_key("alice", line.entry_id)]
    assert entry.identity == "alice"
    assert entry.session_id == "s-1"
    assert not path.exists(), "a fully committed sealed segment is deleted"


def test_a_line_key_is_honoured_only_under_otlp(backend, root):
    otlp = _line(c.KIND_DECISION, key="otel:alice:abc")
    forged = _line(c.KIND_AUDIT, key="otel:bob:forged")
    _sealed(root / c.OTLP_SPOOL_DIR / "alice", otlp)
    _sealed(root / "alice", forged)
    _consumer(backend, root).consume()
    keys = _by_key(backend)
    assert keys["otel:alice:abc"].identity == "alice"
    assert c.spool_key("alice", forged.entry_id) in keys
    assert "otel:bob:forged" not in keys


# --- the source's four cases ---------------------------------------------------------


def test_a_concurrent_append_during_consume_loses_no_line(backend, root, monkeypatch):
    directory = root / "alice"
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(directory))
    consumer = _consumer(backend, root)
    written: list[str] = []
    done = threading.Event()

    def produce() -> None:
        for i in range(300):
            entry_id = writer.write(c.KIND_AUDIT, {"i": i})
            assert entry_id
            written.append(entry_id)
            if i % 50 == 0:
                writer.seal_all()
        done.set()

    thread = threading.Thread(target=produce)
    thread.start()
    while not done.is_set():
        consumer.consume()
    thread.join()
    writer.seal_all()
    consumer.consume()
    keys = _by_key(backend)
    assert {c.spool_key("alice", e) for e in written} <= set(keys)
    assert _segments(directory) == []


def test_a_writer_in_another_pid_namespace_idles_past_the_bound_then_appends(
    backend, root, monkeypatch
):
    directory = root / "alice"
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(directory))
    clock = _Clock()
    monkeypatch.setattr(writer, "_monotonic", clock)
    monkeypatch.setattr(writer, "_pid", lambda: 1)  # pid 1 in its own namespace
    monkeypatch.setattr(writer, "_host", lambda: "other-ns")
    now = [NOW]
    consumer = _consumer(backend, root, now=lambda: now[0])

    first = writer.write(c.KIND_AUDIT, {"n": 1})
    [segment] = _segments(directory)
    assert segment.endswith(c.OPEN_SUFFIX)
    consumer.consume()
    assert c.spool_key("alice", first) in _by_key(backend)

    # The writer goes quiet for two hours; its idle timer never fires here.
    old = (NOW - timedelta(hours=2)).timestamp()
    os.utime(directory / segment, (old, old))
    consumer.consume()
    health = consumer.health().sources[str(directory)]
    assert health.abandoned == (segment,)
    assert _segments(directory) == [segment], "an open segment is never sealed or deleted"

    # It wakes up (its clock has not moved) and appends to the same segment …
    second = writer.write(c.KIND_AUDIT, {"n": 2})
    assert _segments(directory) == [segment]
    consumer.consume()
    # … and once its own idle bound passes it seals and starts a new one.
    clock.now += c.SEGMENT_IDLE_S + 1
    third = writer.write(c.KIND_AUDIT, {"n": 3})
    writer.seal_all()
    report = consumer.consume()
    keys = _by_key(backend)
    for entry_id in (first, second, third):
        assert c.spool_key("alice", entry_id) in keys
    assert report.duplicates == 0, "a sealed segment resumes from its open name's checkpoint"
    assert _segments(directory) == []


def test_a_kind_outside_its_sources_allow_list_is_refused_and_consumed(backend, root):
    bridge = _line(c.KIND_BRIDGE_TURN)
    decision = _line(c.KIND_DECISION)  # an OTLP kind, sent over a spool directory
    fine = _line(c.KIND_FEEDBACK)
    _sealed(root / "alice", bridge, decision, fine)
    consumer = _consumer(backend, root)
    pending = consumer.snapshot()
    assert pending and not consumer.drained(pending)
    report = consumer.consume()
    assert report.refused == 2
    keys = _by_key(backend)
    assert set(keys) == {c.spool_key("alice", fine.entry_id)}
    for line in (bridge, decision):
        assert backend.get_state(c.refused_state_name("alice", line.entry_id)) is not None
    assert consumer.drained(pending), "refused lines count as consumed"
    assert _segments(root / "alice") == []


def test_a_forged_run_from_a_non_dispatch_identity_is_refused_and_shown(backend, root):
    forged = _line(
        c.KIND_RUN, {"phase": c.RUN_ACCEPT, "requester": "mallory"}, run_id="r1", session_id="s1"
    )
    _sealed(root / "alice", forged)
    consumer = _consumer(backend, root)
    consumer.consume()
    assert _all(backend) == []
    shown = consumer.health().sources[str(root / "alice")]
    assert shown.refused_total == 1
    assert shown.refused[0].entry_id == forged.entry_id
    assert shown.refused[0].kind == c.KIND_RUN
    assert "terminal" in shown.refused[0].reason


def test_a_forged_on_behalf_of_is_dropped(backend, root):
    forged = _line(on_behalf_of="bob")
    sidecar = _line(body={"surface": "chat"}, on_behalf_of="carol")
    preflight = _line(body={"surface": "preflight"}, on_behalf_of="dave")
    other = _line(body={"surface": "chat"}, on_behalf_of="erin")
    _sealed(root / "alice", forged)
    _sealed(root / "sidecar", sidecar)
    _sealed(root / "queueserver", preflight, other)
    _consumer(backend, root).consume()
    keys = _by_key(backend)
    assert keys[c.spool_key("alice", forged.entry_id)].on_behalf_of == ""
    assert keys[c.spool_key("sidecar", sidecar.entry_id)].on_behalf_of == "carol"
    assert keys[c.spool_key("queueserver", preflight.entry_id)].on_behalf_of == "dave"
    assert keys[c.spool_key("queueserver", other.entry_id)].on_behalf_of == ""


# --- run entries ---------------------------------------------------------------------


def test_run_entries_follow_first_accept_wins_per_session(backend, root):
    worker = root / "dispatch-worker-1"
    first = _line(
        c.KIND_RUN, {"phase": c.RUN_ACCEPT, "requester": "alice"}, run_id="r1", session_id="s1"
    )
    terminal = _line(c.KIND_RUN, {"phase": c.RUN_TERMINAL}, run_id="r1", session_id="s1")
    _sealed(worker, first, terminal)
    consumer = _consumer(backend, root)
    consumer.consume()
    rival = _line(
        c.KIND_RUN, {"phase": c.RUN_ACCEPT, "requester": "mallory"}, run_id="r2", session_id="s1"
    )
    again = _line(
        c.KIND_RUN, {"phase": c.RUN_ACCEPT, "requester": "alice"}, run_id="r1", session_id="s1"
    )
    _sealed(worker, rival, again)
    # A fresh consumer reads the winner back from the backend.
    consumer = _consumer(backend, root)
    report = consumer.consume()
    keys = _by_key(backend)
    assert c.run_key("r1", c.RUN_ACCEPT) in keys
    assert c.run_key("r1", c.RUN_TERMINAL) in keys
    assert c.run_key("r2", c.RUN_ACCEPT) not in keys
    assert report.refused == 1 and report.duplicates == 1
    [shown] = consumer.health().sources[str(worker)].refused
    assert shown.entry_id == rival.entry_id and "first accept wins" in shown.reason


def test_rival_accepts_in_one_segment_keep_the_first(backend, root):
    worker = root / "dispatch-worker-1"
    a = _line(c.KIND_RUN, {"phase": c.RUN_ACCEPT, "session_id": "s9"}, run_id="ra")
    b = _line(c.KIND_RUN, {"phase": c.RUN_ACCEPT, "session_id": "s9"}, run_id="rb")
    _sealed(worker, a, b)
    _consumer(backend, root).consume()
    keys = _by_key(backend)
    assert c.run_key("ra", c.RUN_ACCEPT) in keys
    assert c.run_key("rb", c.RUN_ACCEPT) not in keys


# --- secrets ---------------------------------------------------------------------------


def test_a_secret_is_absent_from_the_stored_rows(backend, root):
    line = _line(body={"text": f"token={SECRET}", "nested": [{"v": SECRET}]})
    _sealed(root / "alice", line)
    _consumer(backend, root, secrets=[SECRET]).consume()
    raw = backend._connection().execute("SELECT body FROM entries").fetchall()
    assert raw and all(SECRET not in row[0] for row in raw)
    [entry] = _all(backend)
    assert entry.body == {"text": "token=***", "nested": [{"v": "***"}]}


# --- segments --------------------------------------------------------------------------


def test_an_open_segment_is_read_to_its_last_complete_line_and_never_removed(backend, root):
    directory = root / "alice"
    directory.mkdir()
    done = _line()
    partial = _line()
    path = directory / c.segment_name("h", 3, 5)
    raw = (partial.to_json() + "\n").encode()
    path.write_bytes((done.to_json() + "\n").encode() + raw[:10])
    consumer = _consumer(backend, root)
    consumer.consume()
    assert set(_by_key(backend)) == {c.spool_key("alice", done.entry_id)}
    assert path.exists()
    with path.open("ab") as handle:
        handle.write(raw[10:])
    consumer.consume()
    assert c.spool_key("alice", partial.entry_id) in _by_key(backend)
    assert path.exists()
    inode = path.stat().st_ino
    committed = backend.get_state(c.spool_state_name("alice", path.name, inode))
    assert committed == {"offset": path.stat().st_size}


def test_an_unparseable_line_is_refused_and_the_segment_still_drains(backend, root):
    good = _line()
    path = _sealed(root / "alice", b"not json\n", b'{"entry_id": "bad1", "kind": "audit"}\n', good)
    consumer = _consumer(backend, root)
    report = consumer.consume()
    assert report.refused == 2
    assert set(_by_key(backend)) == {c.spool_key("alice", good.entry_id)}
    assert backend.get_state(c.refused_state_name("alice", "bad1")) is not None
    assert not path.exists()


# --- outage ------------------------------------------------------------------------------


def test_a_backend_outage_keeps_the_segments_and_recovery_loses_nothing(backend, root):
    flaky = _Flaky(backend)
    directory = root / "alice"
    lines = [_line(body={"i": i}) for i in range(5)]
    first = _sealed(
        directory, *lines[:3], start_ns=int((NOW - timedelta(minutes=10)).timestamp() * 1e9)
    )
    second = _sealed(
        directory, *lines[3:], start_ns=int((NOW - timedelta(minutes=1)).timestamp() * 1e9)
    )
    (directory / c.WRITER_ERRORS_FILE).write_text(
        '{"ts": "2026-10-23T11:00:00+00:00", "kind": "audit", "error": "OSError: disk full"}\n'
    )
    consumer = _consumer(flaky, root, now=lambda: NOW)
    flaky.down = True
    report = consumer.consume()
    assert report.appended == 0 and "down" in report.backend_error
    assert first.exists() and second.exists()
    health = consumer.health()
    assert health.backend_error
    source = health.sources[str(directory)]
    assert source.backlog_lines == 5
    assert source.oldest_backlog_age_s == pytest.approx(600, abs=1)
    assert len(source.writer_errors) == 1 and "disk full" in source.writer_errors[0]

    flaky.down = False
    report = consumer.consume()
    assert report.appended == 5 and not report.backend_error
    assert {c.spool_key("alice", ln.entry_id) for ln in lines} == set(_by_key(backend))
    assert not first.exists() and not second.exists()
    assert consumer.health().sources[str(directory)].backlog_lines == 0


def test_an_outage_mid_pass_commits_nothing_it_did_not_store(backend, root):
    flaky = _Flaky(backend)
    lines = [_line(body={"i": i}) for i in range(3)]
    path = _sealed(root / "alice", *lines)
    real_append = flaky.append

    def failing_append(entries):
        flaky.down = True
        return real_append(entries)

    flaky.append = failing_append
    consumer = _consumer(flaky, root)
    consumer.consume()
    assert path.exists()
    flaky.down = False
    flaky.append = real_append
    consumer.consume()
    assert len(_all(backend)) == 3 and not path.exists()


# --- hostile files -----------------------------------------------------------------------


def test_a_fifo_and_a_symlink_are_quarantined(backend, root, tmp_path):
    directory = root / "alice"
    directory.mkdir()
    fifo = directory / c.SegmentName("h", 1, 1, sealed=True).name
    os.mkfifo(fifo)
    link = directory / c.segment_name("h", 1, 2)
    link.symlink_to("/dev/zero")
    errors_link = directory / c.WRITER_ERRORS_FILE
    errors_link.symlink_to(tmp_path / "elsewhere")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "bob").symlink_to(outside)
    good = _line()
    _sealed(directory, good)
    consumer = _consumer(backend, root, now=lambda: NOW)
    consumer.consume()
    stamp = NOW.strftime("%Y%m%dT%H%M%SZ")
    for path in (fifo, link, errors_link, root / "bob"):
        assert not os.path.lexists(path)
        assert os.path.lexists(path.with_name(f"{path.name}{QUARANTINE_SUFFIX}{stamp}"))
    assert len(consumer.health().quarantined) == 4
    assert set(_by_key(backend)) == {c.spool_key("alice", good.entry_id)}
    # A second pass leaves the evidence alone.
    consumer.consume()
    assert len(consumer.health().quarantined) == 4
