"""One guarded run per process: a thread or task the run starts runs under it.

The lock a process holds, the journaled run it has open and the reports it
filed are the process's, kept in one place behind one lock. A helper thread or
an asyncio task inside a run journals into the run's one journal and is
restored by its one exit path; a write in flight finishes before the run ends
and none is accepted after; a forked child holds nothing and leaves its
parent's journal alone. Channels are a dict served through ``osprey.runtime``;
no control system is involved. Threads are ordered by ``threading.Event`` and
joined with a timeout.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
import osprey.runtime._run_state as _run_state
import osprey.runtime.guarded_run as guarded_run
import osprey.runtime.journal as journal_mod
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import (
    LOCK_FILE_NAME,
    RESTORE_REPORT_TAG,
    OspreyRunBusy,
    journaled_run,
    lock,
    restore_reports,
)
from osprey.runtime.journal import OspreyWriteRefused, guarded_write, read_pending_journal
from tests.runtime._guarded_run_fixtures import (
    GENERATION,
    fresh_run_state,  # noqa: F401 - used through pytestmark
    journal_path,
    write_config,
)

pytestmark = pytest.mark.usefixtures("fresh_run_state")

#: How long a thread or child may take to reach a point or finish; a hang
#: guard, never a statement about speed.
TIMEOUT_S = 60.0

#: The refusal of a guarded write with no run open.
NOT_JOURNALED = "guarded writes run only inside a journaled guarded run"


class _Channels:
    """A dict of channel values; every read and write is recorded, every write lands."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []
        self.events: list[str] = []

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        self.events.append("read")
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.writes.append((address, value))
        self.values[address] = value

    def channel_limits(self, _address: str) -> None:
        return None

    def guarded(self, address: str, value: Any, before: Callable[[], None] | None = None) -> None:
        """One guarded write of ``value`` to ``address``, running ``before`` inside the write."""

        def write() -> None:
            if before is not None:
                before()
            self.write_channel(address, value)

        guarded_write([address], write, lambda exc: exc)


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline that asks nobody, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    return root


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    """``Q`` at 3.0 and ``S`` at 5.0, neither with a limits record."""
    fake = _Channels({"Q": 3.0, "S": 5.0})
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


class _Thread:
    """A thread running ``target``; what escapes it is kept in ``error``."""

    def __init__(self, target: Callable[[], None]) -> None:
        self.error: BaseException | None = None

        def run() -> None:
            try:
                target()
            except BaseException as exc:  # kept for the test to assert on
                self.error = exc

        self._thread = threading.Thread(target=run)
        self._thread.start()

    def join(self) -> BaseException | None:
        self._thread.join(TIMEOUT_S)
        assert not self._thread.is_alive(), "the thread did not finish"
        return self.error


class _WatchedLock:
    """A re-entrant lock that sets ``contended`` when a thread has to wait for it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self) -> _WatchedLock:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()


def _holder(directory: Path) -> dict[str, Any] | None:
    text = (directory / LOCK_FILE_NAME).read_text(encoding="utf-8")
    return json.loads(text) if text else None


def _tagged(text: str) -> list[dict[str, Any]]:
    prefix = RESTORE_REPORT_TAG + " "
    return [
        json.loads(line[len(prefix) :]) for line in text.splitlines() if line.startswith(prefix)
    ]


def _headers(raw: bytes) -> int:
    return sum(1 for line in raw.splitlines() if "header" in json.loads(line))


def test_a_tool_run_from_a_thread_under_the_runs_lock_journals_and_restores(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    on_disk: list[dict[str, Any]] = []
    raw_during: list[bytes] = []
    holder_during: list[dict[str, Any] | None] = []

    def on_disk_now() -> None:
        pending = read_pending_journal(journal_path())
        assert pending is not None
        on_disk.append(dict(pending.values))

    with lock(None) as directory:
        holder_before = _holder(directory)

        def tool() -> None:
            with journaled_run(None):
                channels.guarded("Q", 2.5, on_disk_now)
                channels.guarded("S", 6.0, on_disk_now)
                raw_during.append(journal_path().read_bytes())
                holder_during.append(_holder(directory))
                raise RuntimeError("the tool failed")

        error = _Thread(tool).join()

    assert isinstance(error, RuntimeError), error
    assert on_disk == [{"Q": 3.0}, {"Q": 3.0, "S": 5.0}], "each line is on disk before its write"
    assert _headers(raw_during[0]) == 1
    assert channels.values == {"Q": 3.0, "S": 5.0}, "both addresses are put back"
    (report,) = _tagged(capsys.readouterr().err)
    assert report["aborted"] is True and sorted(report["restored"]) == ["Q", "S"]
    assert error.restore_report.aborted is True  # type: ignore[attr-defined]
    assert holder_before is not None and set(holder_before) == {"pid", "started"}
    assert holder_during == [holder_before], "a thread joining the hold writes no holder record"
    assert journal_path().read_bytes() == b""


def test_a_run_on_the_main_thread_runs_under_a_hold_a_helper_thread_took(
    channels: _Channels,
) -> None:
    taken = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with lock("live"):
            taken.set()
            assert release.wait(TIMEOUT_S)

    thread = _Thread(holder)
    try:
        assert taken.wait(TIMEOUT_S)
        with journaled_run(None) as journal:
            channels.guarded("Q", 2.5)
    finally:
        release.set()
    assert thread.join() is None
    assert journal.values == {"Q": 3.0}
    assert channels.writes == [("Q", 2.5)]


def test_a_thread_asking_for_the_lock_is_busy_like_any_second_run() -> None:
    with lock("live") as directory:
        error = _Thread(lambda: lock("live").__enter__()).join()

    assert isinstance(error, OspreyRunBusy)
    assert error.root == str(directory)
    assert error.pid == os.getpid()


def test_a_lock_on_another_target_is_refused_while_one_is_held(repo: Path) -> None:
    refused = "this process holds the run lock on live; it cannot take one on va"
    with lock("live"):
        with pytest.raises(RuntimeError, match=refused):
            with lock("va"):
                pytest.fail("a second target's lock is never taken")

    assert not (repo / "var" / "guarded_run" / "va").exists(), "nothing was opened for va"
    with lock("va"):
        pass


def test_a_threads_guarded_write_is_journaled_into_the_one_journal(channels: _Channels) -> None:
    on_disk: list[dict[str, Any]] = []

    def on_disk_now() -> None:
        pending = read_pending_journal(journal_path())
        assert pending is not None
        on_disk.append(dict(pending.values))

    with journaled_run(None) as journal:
        error = _Thread(lambda: channels.guarded("Q", 2.5, on_disk_now)).join()

    assert error is None
    assert on_disk == [{"Q": 3.0}]
    assert journal.values == {"Q": 3.0}
    assert channels.writes == [("Q", 2.5)]


def test_two_threads_writing_one_address_leave_one_journal_line(channels: _Channels) -> None:
    start = threading.Barrier(2, timeout=TIMEOUT_S)

    def writer(value: float) -> Callable[[], None]:
        def write() -> None:
            start.wait()
            channels.guarded("Q", value)

        return write

    with journaled_run(None):
        threads = [_Thread(writer(1.0)), _Thread(writer(2.0))]
        assert [thread.join() for thread in threads] == [None, None]
        raw = journal_path().read_bytes()

    lines = [json.loads(line) for line in raw.splitlines()]
    assert _headers(raw) == 1
    assert [line for line in lines if "address" in line] == [{"address": "Q", "value": 3.0}]


def test_a_context_copied_inside_a_run_is_refused_after_it(channels: _Channels) -> None:
    with journaled_run(None):
        copied = contextvars.copy_context()

    with pytest.raises(OspreyWriteRefused, match=NOT_JOURNALED):
        copied.run(channels.guarded, "Q", 2.5)

    assert channels.writes == []


def test_a_thread_still_in_the_run_when_the_lock_is_released_is_restored(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    inside = threading.Event()
    leave = threading.Event()
    refusals: list[OspreyWriteRefused] = []

    def tool() -> None:
        with journaled_run(None):
            channels.guarded("Q", 2.5)
            inside.set()
            assert leave.wait(TIMEOUT_S)
            try:
                channels.guarded("S", 6.0)
            except OspreyWriteRefused as refused:
                refusals.append(refused)

    with lock(None):
        thread = _Thread(tool)
        assert inside.wait(TIMEOUT_S)
    restored = dict(channels.values)
    leave.set()

    assert thread.join() is None
    assert restored == {"Q": 3.0, "S": 5.0}, "the release put Q back before the thread went on"
    assert [refused.reason for refused in refusals] == [NOT_JOURNALED]
    (report,) = _tagged(capsys.readouterr().err)
    assert report["aborted"] is True and report["restored"] == ["Q"]
    assert restore_reports() == [report]
    assert journal_path().read_bytes() == b""


def test_a_tool_failing_in_a_thread_stops_the_run(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    def tool() -> None:
        with journaled_run(None):
            channels.guarded("Q", 2.5)
            raise ValueError("the tool broke\nat its second step")

    with pytest.raises(ValueError, match="the tool broke") as caught:
        with journaled_run(None):
            assert isinstance(_Thread(tool).join(), ValueError)
            with pytest.raises(OspreyWriteRefused) as refused:
                channels.guarded("S", 6.0)

    assert refused.value.reason == "the guarded run was stopped: the tool broke"
    assert channels.writes[-1] == ("Q", 3.0), "the outer exit put Q back"
    assert channels.values == {"Q": 3.0, "S": 5.0}
    assert caught.value.restore_report.aborted is True  # type: ignore[attr-defined]
    (report,) = _tagged(capsys.readouterr().err)
    assert report["aborted"] is True and report["restored"] == ["Q"]


def test_a_write_in_flight_finishes_before_the_run_is_restored(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched = _WatchedLock()
    monkeypatch.setattr(_run_state, "LOCK", watched)
    in_write = threading.Event()
    ended = threading.Event()
    refusals: list[OspreyWriteRefused] = []

    def hold_the_write() -> None:
        in_write.set()
        assert watched.contended.wait(TIMEOUT_S), "the run's end never waited for the write"
        channels.events.append("write returned")

    def writer() -> None:
        channels.guarded("Q", 2.5, hold_the_write)
        assert ended.wait(TIMEOUT_S)
        try:
            channels.guarded("S", 6.0)
        except OspreyWriteRefused as refused:
            refusals.append(refused)

    thread: _Thread | None = None
    with pytest.raises(RuntimeError, match="the body failed"):
        with journaled_run(None):
            thread = _Thread(writer)
            assert in_write.wait(TIMEOUT_S)
            raise RuntimeError("the body failed")
    ended.set()

    assert thread is not None and thread.join() is None
    assert channels.events == ["read", "write returned", "read"], (
        "the restore's read comes after the write in flight returned"
    )
    assert channels.values == {"Q": 3.0, "S": 5.0}
    assert [refused.reason for refused in refusals] == [NOT_JOURNALED]


def test_an_asyncio_task_runs_under_the_run(channels: _Channels) -> None:
    async def main() -> Any:
        with journaled_run(None) as journal:

            async def task_write() -> None:
                channels.guarded("Q", 2.5)

            await asyncio.create_task(task_write())
            await asyncio.to_thread(channels.guarded, "S", 6.0)
        return journal

    journal = asyncio.run(main())

    assert journal.values == {"Q": 3.0, "S": 5.0}
    assert channels.writes == [("Q", 2.5), ("S", 6.0)]


def test_a_task_outliving_the_run_is_refused(channels: _Channels) -> None:
    async def main() -> None:
        go = asyncio.Event()

        async def late() -> None:
            await go.wait()
            channels.guarded("Q", 2.5)

        with journaled_run(None):
            task = asyncio.create_task(late())
        go.set()
        await task

    with pytest.raises(OspreyWriteRefused, match=NOT_JOURNALED):
        asyncio.run(main())

    assert channels.writes == []


def test_a_thread_after_the_run_ended_takes_its_own_run(channels: _Channels) -> None:
    with lock(None) as directory:
        pass

    def tool() -> None:
        with journaled_run(None):
            channels.guarded("Q", 2.5)
            assert _holder(directory) is not None, "the thread holds the lock itself"

    assert _Thread(tool).join() is None
    assert channels.writes == [("Q", 2.5)]
    assert _holder(directory) is None
    assert journal_path().read_bytes() == b""
    with lock(None):
        pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_a_child_forked_inside_a_run_leaves_the_parents_journal(repo: Path) -> None:
    script = textwrap.dedent(
        """
        import os
        import osprey.runtime
        from osprey.runtime.guarded_run import journaled_run
        from osprey.runtime.journal import guarded_write

        values = {"Q": 3.0}
        osprey.runtime.read_channels = lambda addresses, **_: [values[a] for a in addresses]
        osprey.runtime.write_channel = lambda address, value, **_: values.update({address: value})
        osprey.runtime.channel_limits = lambda _address: None

        def run(path):
            with journaled_run("live"):
                guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 2.5), Exception)
                before = open(path, "rb").read()
                child = os.fork()
                if child == 0:
                    return True
                os.waitpid(child, 0)
                print("parent", open(path, "rb").read() == before, len(before) > 0, flush=True)
            return False

        if run(os.environ["JOURNAL"]):
            print("child left the run", flush=True)
            os._exit(0)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env={**os.environ, "JOURNAL": str(journal_path())},
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["child left the run", "parent True True"], done.stdout
    (report,) = _tagged(done.stderr)
    assert report["aborted"] is False, "only the parent's own exit reported"


def test_the_run_state_is_one_object_no_module_binds() -> None:
    def context_vars(module: Any) -> list[str]:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = []
        for node in tree.body:
            value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "ContextVar"
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names += [t.id for t in targets if isinstance(t, ast.Name)]
        return names

    def bound_by_import(module: Any) -> list[str]:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        return [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in {"STATE", "LOCK"}
        ]

    assert context_vars(guarded_run) == []
    assert context_vars(journal_mod) == ["_ACTIVE"]
    assert bound_by_import(guarded_run) == bound_by_import(journal_mod) == []
