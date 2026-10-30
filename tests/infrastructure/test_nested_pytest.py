"""The bound on a nested pytest run: its own progress, not a clock.

A child that keeps writing runs to its end however long it takes; a child that
writes nothing for the stall window is stopped and its output reported. A
finished run says how long it sat silent before it exited. A
nested run's temporary directories, and the pruning pytest does of them at
exit, stay inside a root of its own.

The children here take no time of their own. Each says its words one at a time
and waits after each to be told to go on, and the run is measured on a
``_ConversationClock`` that moves only between a word and that go-ahead. What
the run counts as elapsed is then fixed by the test, not by how loaded the host
is or how late a poll sees a write.
"""

from __future__ import annotations

import getpass
import os
import signal
import sys
from pathlib import Path

import pytest

from osprey.deployment.subprocess_capture import StopOutcome, observe_exit, stop_child
from tests import _nested_pytest
from tests._nested_pytest import run_nested_pytest

#: Says each argument after the first on its own line, records how many it has
#: said, and its pid, in the file the first names, then waits for ``SIGUSR1``
#: before the next. It exits after the last go-ahead. The wait is on an event
#: the signal sets, which keeps a go-ahead that comes early and, unlike
#: ``signal.sigwait`` on macOS, still dies of ``SIGABRT`` with its stacks.
_CHILD = """\
import os, signal, sys, threading
go = threading.Event()
signal.signal(signal.SIGUSR1, lambda *_: go.set())
said = sys.argv[1]
for count, word in enumerate(sys.argv[2:], 1):
    print(word, flush=True)
    with open(said + ".new", "w") as record:
        record.write(f"{os.getpid()} {count}")
    os.replace(said + ".new", said)
    go.wait()
    go.clear()
"""


class _ConversationClock:
    """A clock for ``run_nested_pytest`` that moves with what ``_CHILD`` says.

    It stands still until the child has said a new word, and for one reading
    more: the run checks the streams before it reads the clock, so by that
    second reading it has seen the word. Each reading after is ``step`` later,
    until ``gap`` after the word, when the child is told to go on and the clock
    stands still again. With no ``gap`` the child is never told.
    """

    def __init__(self, said: Path, *, gap: float | None, step: float = 1.0) -> None:
        self.now = 0.0
        self._said = said
        self._gap = gap
        self._step = step
        self._heard = 0
        self._spoken_at: float | None = None
        self._held = False

    def __call__(self) -> float:
        record = self._said.read_text().split() if self._said.exists() else None
        if record is not None and int(record[1]) != self._heard:
            self._heard = int(record[1])
            self._spoken_at = self.now
            self._held = True
        elif self._held:
            self._held = False
        elif self._spoken_at is not None:
            self.now += self._step
            if self._gap is not None and self.now >= self._spoken_at + self._gap:
                os.kill(int(record[0]), signal.SIGUSR1)
                self._spoken_at = None
        return self.now


def _say(tmp_path: Path, *words: str) -> list[str]:
    return [sys.executable, "-c", _CHILD, str(tmp_path / "said"), *words]


def test_a_run_that_keeps_writing_is_not_stopped_by_its_length(tmp_path):
    clock = _ConversationClock(tmp_path / "said", gap=1.0)

    result = run_nested_pytest(
        _say(tmp_path, *"." * 3), cwd=None, env=None, stall_seconds=1.5, clock=clock
    )

    assert result.returncode == 0
    assert result.stdout.count(".") == 3
    assert clock.now == 3.0


def test_a_finished_run_reports_its_silence_before_exit(tmp_path):
    result = run_nested_pytest(
        _say(tmp_path, "done"),
        cwd=None,
        env=None,
        clock=_ConversationClock(tmp_path / "said", gap=5.0),
    )

    assert result.returncode == 0
    assert result.quiet_before_exit == 5.0


def test_a_run_that_writes_nothing_for_the_stall_window_is_stopped_with_its_stacks(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="no new output for 1.0 s") as stopped:
        run_nested_pytest(
            _say(tmp_path, "started"),
            cwd=None,
            env=None,
            stall_seconds=1.0,
            clock=_ConversationClock(tmp_path / "said", gap=None),
        )
    report = str(stopped.value)
    assert "started" in report
    assert "most recent call first" in report


def test_a_nested_run_leaves_the_shared_temporary_root_alone(tmp_path):
    shared = tmp_path / "shared"
    numbered = shared / f"pytest-of-{getpass.getuser()}"
    for name in ("pytest-1", "pytest-2", "pytest-3"):
        (numbered / name).mkdir(parents=True)
    (numbered / "pytest-1" / "left-by-a-finished-run").write_text("")
    work = tmp_path / "work"
    work.mkdir()
    module = work / "test_uses_tmp_path.py"
    module.write_text(
        "def test_writes_into_tmp_path(tmp_path):\n    (tmp_path / 'out').write_text('x')\n"
    )

    result = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            os.devnull,
            "--rootdir",
            str(work),
            str(module),
            "-q",
        ],
        cwd=work,
        env={
            **os.environ,
            "PYTEST_DEBUG_TEMPROOT": str(shared),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert sorted(p.name for p in numbered.iterdir()) == ["pytest-1", "pytest-2", "pytest-3"]
    assert (numbered / "pytest-1" / "left-by-a-finished-run").exists()


def test_the_leftover_stop_runs_before_the_leader_is_reaped(monkeypatch):
    seen: list[int] = []

    def spy(process, **kwargs):
        # The leader has exited but is still a zombie nobody has reaped.
        assert observe_exit(process) == 0
        assert process.returncode is None
        outcome = stop_child(process, **kwargs)
        assert process.returncode == 0, "the stop did not reap the leader"
        seen.append(process.pid)
        return outcome

    monkeypatch.setattr(_nested_pytest, "stop_child", spy)
    result = run_nested_pytest([sys.executable, "-c", "pass"], cwd=None, env=None)
    assert result.returncode == 0
    assert len(seen) == 1


def test_a_nested_run_whose_table_could_not_be_read_fails_saying_so(monkeypatch):
    def stop_that_could_not_look(process, **kwargs):
        stop_child(process, **kwargs)
        return StopOutcome(ended=(process.pid,), unknown=(process.pid,))

    monkeypatch.setattr(_nested_pytest, "stop_child", stop_that_could_not_look)
    with pytest.raises(
        pytest.fail.Exception,
        match="could not tell whether the nested run left processes behind: "
        "process table unreadable: group",
    ):
        run_nested_pytest([sys.executable, "-c", "pass"], cwd=None, env=None)
