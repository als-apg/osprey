"""The spool writer client: where a line lands, the segment protocol, receiver mode.

``write()`` appends one line into the acting identity's own spool directory
(``$OSPREY_SPOOL_DIR``, or ``<project root>/var/spool/<identity>`` on the host)
through ``<host>-<pid>-<start ns>.open`` segments that it seals by rename on
size, age, idleness and exit. ``write_received()`` is agent-record's receiver
mode: a fsynced, sealed segment under ``<spool root>/.otlp/<login>/``. Neither
ever raises: a failure is ``None`` plus a line in the directory's ``.errors``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.agent_record import writer
from osprey.agent_record.contract import (
    WRITER_ERRORS_FILE,
    is_token,
    parse_segment_name,
    parse_spool_line,
)


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    """Each test starts with no open segment and the real thresholds."""
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR", raising=False)
    yield
    writer.seal_all()


@pytest.fixture
def spool(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "spool" / "alice"
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(directory))
    return directory


def _segments(directory: Path, suffix: str) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.name.endswith(suffix))


def _lines(path: Path) -> list:
    return [parse_spool_line(raw) for raw in path.read_bytes().splitlines()]


def _all_lines(directory: Path) -> list:
    out = []
    for path in sorted(directory.iterdir()):
        if path.name.endswith((".open", ".sealed")):
            out += _lines(path)
    return out


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Where a line lands
# ---------------------------------------------------------------------------


def test_env_names_the_identity_directory_itself(spool: Path):
    entry_id = writer.write(
        "audit",
        {"decision": "allow"},
        session_id="s-1",
        terminal_session="t-1",
        run_id="r-1",
        tool_use_id="tu-1",
        on_behalf_of="bob",
    )

    assert entry_id and is_token(entry_id)
    (segment,) = _segments(spool, ".open")
    parts = parse_segment_name(segment.name)
    assert parts is not None and not parts.sealed
    assert parts.pid == os.getpid()
    (line,) = _lines(segment)
    assert line.entry_id == entry_id
    assert line.kind == "audit"
    assert line.body == {"decision": "allow"}
    assert (line.session_id, line.terminal_session, line.run_id, line.tool_use_id) == (
        "s-1",
        "t-1",
        "r-1",
        "tu-1",
    )
    assert line.on_behalf_of == "bob"
    assert line.key is None
    raw = json.loads(segment.read_bytes())
    assert "identity" not in raw


def test_env_wins_over_the_identity_ladder(spool: Path, monkeypatch):
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "mallory")
    assert writer.write("audit", {})
    assert writer.spool_dir() == spool
    assert not (spool / "mallory").exists()


def test_host_form_uses_project_root_and_acting_identity(tmp_path, monkeypatch):
    import osprey_connectors.workspace as workspace

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(workspace, "resolve_project_root", lambda config=None: tmp_path)
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "carol")

    entry_id = writer.write("feedback", {"text": "hi"})

    directory = tmp_path / "var" / "spool" / "carol"
    assert writer.spool_dir() == directory
    assert entry_id
    assert [line.entry_id for line in _all_lines(directory)] == [entry_id]


def test_blank_env_falls_back_to_the_host_form(tmp_path, monkeypatch):
    import osprey_connectors.workspace as workspace

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(workspace, "resolve_project_root", lambda config=None: tmp_path)
    monkeypatch.setenv("OSPREY_SPOOL_DIR", "  ")
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "dave")
    assert writer.spool_dir() == tmp_path / "var" / "spool" / "dave"


def test_env_is_read_per_call(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(first))
    assert writer.write("audit", {"n": 1})
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(second))
    assert writer.write("audit", {"n": 2})
    assert [line.body["n"] for line in _all_lines(first)] == [1]
    assert [line.body["n"] for line in _all_lines(second)] == [2]


def test_created_directory_is_group_shared(spool: Path):
    assert writer.write("audit", {})
    assert spool.stat().st_mode & 0o7777 == writer.SPOOL_DIR_MODE


def test_given_entry_id_and_ts_are_kept(spool: Path):
    ts = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    assert writer.write("audit", {}, entry_id="e-42", ts=ts) == "e-42"
    (line,) = _all_lines(spool)
    assert line.entry_id == "e-42"
    assert datetime.fromisoformat(line.ts) == ts


@pytest.mark.usefixtures("spool")
def test_entry_ids_are_unique():
    ids = {writer.write("audit", {"i": i}) for i in range(50)}
    assert len(ids) == 50 and None not in ids


# ---------------------------------------------------------------------------
# Segment protocol
# ---------------------------------------------------------------------------


def test_lines_append_to_one_open_segment(spool: Path):
    for i in range(3):
        assert writer.write("audit", {"i": i})
    (segment,) = _segments(spool, ".open")
    assert [line.body["i"] for line in _lines(segment)] == [0, 1, 2]


def test_a_removed_segment_starts_a_new_one(spool: Path):
    assert writer.write("audit", {"i": 0})
    (first,) = _segments(spool, ".open")
    first.unlink()

    assert writer.write("audit", {"i": 1})

    (second,) = _segments(spool, ".open")
    assert second.name != first.name
    assert [line.body["i"] for line in _lines(second)] == [1]
    assert not first.exists()


def test_a_renamed_segment_is_never_appended_to(spool: Path):
    """Once the segment is not there by name, the writer must not follow it."""
    assert writer.write("audit", {"i": 0})
    (first,) = _segments(spool, ".open")
    aside = spool / "aside"
    first.rename(aside)

    assert writer.write("audit", {"i": 1})

    assert [line.body["i"] for line in _lines(aside)] == [0]


def test_seals_on_size(spool: Path, monkeypatch):
    monkeypatch.setattr(writer, "SEGMENT_MAX_BYTES", 200)
    assert writer.write("audit", {"pad": "x" * 300})
    (sealed,) = _segments(spool, ".sealed")
    assert _segments(spool, ".open") == []
    assert parse_segment_name(sealed.name).sealed

    assert writer.write("audit", {"i": 1})
    assert len(_segments(spool, ".open")) == 1
    assert len(_segments(spool, ".sealed")) == 1


def test_seals_before_a_write_into_a_full_segment(spool: Path, monkeypatch):
    assert writer.write("audit", {"i": 0})
    monkeypatch.setattr(writer, "SEGMENT_MAX_BYTES", 10)  # now over the bound
    assert writer.write("audit", {"i": 1})
    sealed = _segments(spool, ".sealed")
    assert [line.body["i"] for line in _lines(sealed[0])] == [0]


def test_seals_at_age_before_the_write(spool: Path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(writer, "_monotonic", clock)
    assert writer.write("audit", {"i": 0})
    clock.now += 30
    assert writer.write("audit", {"i": 1})
    assert _segments(spool, ".sealed") == []

    clock.now += 31  # 61 s since the segment started, 31 s since the last write
    assert writer.write("audit", {"i": 2})

    (sealed,) = _segments(spool, ".sealed")
    assert [line.body["i"] for line in _lines(sealed)] == [0, 1]
    (opened,) = _segments(spool, ".open")
    assert [line.body["i"] for line in _lines(opened)] == [2]


def test_idle_timer_seals(spool: Path, monkeypatch):
    monkeypatch.setattr(writer, "SEGMENT_IDLE_S", 0.05)
    assert writer.write("audit", {"i": 0})
    deadline = time.monotonic() + 5
    while not _segments(spool, ".sealed") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(_segments(spool, ".sealed")) == 1
    assert _segments(spool, ".open") == []


def test_idle_timer_waits_for_the_last_write(spool: Path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(writer, "_monotonic", clock)
    assert writer.write("audit", {"i": 0})
    (segment,) = writer._segments.values()
    clock.now += 59
    writer._idle_check(segment)
    assert _segments(spool, ".sealed") == []
    clock.now += 1
    writer._idle_check(segment)
    assert len(_segments(spool, ".sealed")) == 1


def test_seal_all_seals_every_open_segment(tmp_path, monkeypatch):
    for name in ("a", "b"):
        monkeypatch.setenv("OSPREY_SPOOL_DIR", str(tmp_path / name))
        assert writer.write("audit", {})
    writer.seal_all()
    for name in ("a", "b"):
        assert _segments(tmp_path / name, ".open") == []
        assert len(_segments(tmp_path / name, ".sealed")) == 1


def test_seals_at_interpreter_exit(tmp_path):
    directory = tmp_path / "exit"
    env = {**os.environ, "OSPREY_SPOOL_DIR": str(directory)}
    proc = subprocess.run(  # fixed argv, test-authored code
        [
            sys.executable,
            "-c",
            "from osprey.agent_record.writer import write; assert write('audit', {'at': 'exit'})",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert _segments(directory, ".open") == []
    (sealed,) = _segments(directory, ".sealed")
    assert [line.body for line in _lines(sealed)] == [{"at": "exit"}]


def test_another_pid_never_touches_the_inherited_segment(spool: Path, monkeypatch):
    assert writer.write("audit", {"who": "parent"})
    (parent_segment,) = _segments(spool, ".open")

    monkeypatch.setattr(writer, "_pid", lambda: os.getpid() + 1)
    assert writer.write("audit", {"who": "child"})
    writer.seal_all()

    assert parent_segment.exists()  # neither appended to nor sealed by the child
    assert [line.body["who"] for line in _lines(parent_segment)] == ["parent"]
    (child_segment,) = _segments(spool, ".sealed")
    assert parse_segment_name(child_segment.name).pid == os.getpid() + 1


def test_concurrent_writers_lose_no_line(spool: Path):
    def worker(n: int) -> None:
        for i in range(50):
            assert writer.write("audit", {"n": n, "i": i})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    seen = {(line.body["n"], line.body["i"]) for line in _all_lines(spool)}
    assert seen == {(n, i) for n in range(8) for i in range(50)}


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------


def _errors(directory: Path) -> list[dict]:
    path = directory / WRITER_ERRORS_FILE
    return [json.loads(raw) for raw in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    ("kind", "body", "ids"),
    [
        ("not-a-kind", {}, {}),
        ("audit", {"x": object()}, {}),
        ("audit", ["not", "a", "mapping"], {}),
        ("audit", {}, {"entry_id": "has space"}),
        ("audit", {}, {"ts": "yesterday"}),
        ("audit", {}, {"identity": "mallory"}),
    ],
)
def test_a_bad_line_returns_none_and_records_the_error(spool: Path, kind, body, ids):
    assert writer.write(kind, body, **ids) is None
    (error,) = _errors(spool)
    assert error["kind"] == kind
    assert error["error"]
    assert _all_lines(spool) == []


def test_an_unwritable_directory_returns_none(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(blocker / "alice"))
    assert writer.write("audit", {}) is None


def test_a_failing_append_is_recorded(spool: Path, monkeypatch):
    assert writer.write("audit", {"i": 0})

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_all", refuse)
    assert writer.write("audit", {"i": 1}) is None
    (error,) = _errors(spool)
    assert "disk full" in error["error"]


def test_an_unresolvable_root_returns_none(monkeypatch):
    import osprey_connectors.workspace as workspace

    def boom(config=None):  # noqa: ARG001 - mirrors resolve_project_root
        raise RuntimeError("no root")

    monkeypatch.setattr(workspace, "resolve_project_root", boom)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    assert writer.write("audit", {}) is None


def test_short_write_is_terminated_and_fails(spool: Path, monkeypatch):
    real_write = os.write
    calls = []

    def short(fd, data):
        calls.append(data)
        if len(calls) == 1:
            return real_write(fd, data[:5])
        return real_write(fd, data)

    assert writer.write("audit", {"i": 0})
    monkeypatch.setattr(writer.os, "write", short)
    assert writer.write("audit", {"i": 1}) is None
    monkeypatch.setattr(writer.os, "write", real_write)
    assert writer.write("audit", {"i": 2})
    (segment,) = _segments(spool, ".open")
    raw = segment.read_bytes().splitlines()
    assert len(raw) == 3
    assert parse_spool_line(raw[0]).body == {"i": 0}
    assert parse_spool_line(raw[2]).body == {"i": 2}


# ---------------------------------------------------------------------------
# Receiver mode
# ---------------------------------------------------------------------------


def _batch():
    return [
        writer.ReceivedLine("decision", {"tool": "Bash"}, "otel:alice:s:1", {"session_id": "s"}),
        writer.ReceivedLine("usage", {"tokens": 3}, "otel:alice:s:2"),
    ]


def test_receiver_writes_a_sealed_segment_under_otlp_login(tmp_path, monkeypatch):
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(tmp_path / "elsewhere"))
    root = tmp_path / "spool"

    ids = writer.write_received(root, "alice", _batch())

    directory = root / ".otlp" / "alice"
    assert _segments(directory, ".open") == []
    (sealed,) = _segments(directory, ".sealed")
    lines = _lines(sealed)
    assert ids == [line.entry_id for line in lines]
    assert [line.key for line in lines] == ["otel:alice:s:1", "otel:alice:s:2"]
    assert lines[0].session_id == "s"
    assert not (tmp_path / "elsewhere").exists()


def test_receiver_fsyncs_file_and_directory(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync

    def spy(fd):
        synced.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(writer.os, "fsync", spy)
    assert writer.write_received(tmp_path, "alice", _batch())
    import stat

    assert [stat.S_ISREG(mode) for mode in synced] == [True, False]
    assert stat.S_ISDIR(synced[1])


def test_receiver_each_batch_is_its_own_segment(tmp_path):
    assert writer.write_received(tmp_path, "alice", _batch())
    assert writer.write_received(tmp_path, "alice", _batch())
    assert len(_segments(tmp_path / ".otlp" / "alice", ".sealed")) == 2


def test_receiver_empty_batch_writes_nothing(tmp_path):
    assert writer.write_received(tmp_path, "alice", []) == []
    assert not (tmp_path / ".otlp").exists()


@pytest.mark.parametrize("login", ["", "..", "a/b", "a:b"])
def test_receiver_refuses_a_bad_login(tmp_path, login):
    assert writer.write_received(tmp_path, login, _batch()) is None
    assert not (tmp_path / ".otlp" / "b").exists()


def test_receiver_refuses_a_line_without_a_key(tmp_path):
    batch = [*_batch(), writer.ReceivedLine("usage", {}, "")]
    assert writer.write_received(tmp_path, "alice", batch) is None
    directory = tmp_path / ".otlp" / "alice"
    assert _segments(directory, ".sealed") == []
    assert _segments(directory, ".open") == []
    assert _errors(directory)


def test_receiver_failed_write_leaves_no_partial_segment(tmp_path, monkeypatch):
    def refuse(fd):  # noqa: ARG001 - mirrors os.fsync
        raise OSError("fsync failed")

    monkeypatch.setattr(writer.os, "fsync", refuse)
    assert writer.write_received(tmp_path, "alice", _batch()) is None
    directory = tmp_path / ".otlp" / "alice"
    assert _segments(directory, ".open") == []
    assert _segments(directory, ".sealed") == []
    assert "fsync failed" in _errors(directory)[0]["error"]


# ---------------------------------------------------------------------------
# Stdlib-only
# ---------------------------------------------------------------------------


def test_writer_imports_only_the_stdlib_and_its_allowed_helpers():
    tree = ast.parse(Path(writer.__file__).read_text(encoding="utf-8"))
    allowed = {
        "osprey.agent_record.contract",
        "osprey_connectors.identity",
        "osprey_connectors.workspace",
        # The repo layout's names, where the spool root is spelled.
        "osprey.utils.workspace",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top in sys.stdlib_module_names or top == "__future__" or name in allowed, name
