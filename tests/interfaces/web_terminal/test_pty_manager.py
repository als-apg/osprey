"""Tests for PTY manager: real children behind ``PtySession`` and ``PtyRegistry``."""

from __future__ import annotations

import itertools
import logging
import os
import signal
import subprocess
import sys
import termios

import pytest

from osprey.deployment import process_table, subprocess_capture
from osprey.deployment.subprocess_capture import group_is_gone
from osprey.interfaces.web_terminal import process_tree
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, PtySession
from tests._live import wait_until
from tests._processes import kill_group, kill_quietly, wait_gone
from tests.interfaces.web_terminal._pty_child import (
    CHILD_HANG_CEILING,
    SENTINEL,
    detaching_child_script,
    read_answer,
    sleeper_script,
    wait_for_exit,
    wait_for_pids,
    wait_for_report,
)


def _stubborn_child_script(pid_file, grandchild) -> str:
    """Return the source of a PTY child that ignores SIGHUP and SIGTERM.

    It starts *grandchild* in its own process group, waits for its ready line,
    publishes its own pid and then the grandchild's to *pid_file* under a
    temporary name renamed into place, and sleeps.
    """
    return f"""
import os, signal, subprocess, sys, time
for signum in (signal.SIGHUP, signal.SIGTERM):
    signal.signal(signum, signal.SIG_IGN)
proc = subprocess.Popen(
    [sys.executable, {str(grandchild)!r}],
    start_new_session=False,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
)
proc.stdout.readline()
tmp = {str(pid_file) + ".tmp"!r}
with open(tmp, "w") as fh:
    fh.write(f"{{os.getpid()}}\\n{{proc.pid}}\\n")
os.replace(tmp, {str(pid_file)!r})
time.sleep(120)
"""


def _stubborn_session(tmp_path) -> tuple[PtySession, list[int]]:
    """Start a PTY child whose own group ignores the hang-up and SIGTERM; return its pids."""
    pid_file = tmp_path / "pids"
    grandchild = sleeper_script(tmp_path, "mcp_helper.py", ignore=(signal.SIGHUP, signal.SIGTERM))
    session = PtySession([sys.executable, "-c", _stubborn_child_script(pid_file, grandchild)])
    session.start()
    return session, wait_for_pids(session, pid_file, 2)


def _winsize(session: PtySession) -> tuple[int, int]:
    return termios.tcgetwinsize(session._master_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
class TestPtySession:
    def test_terminate_cleans_up(self):
        session = PtySession("/bin/sh")
        session.start()
        assert session.is_alive
        assert session.exit_code is None
        session.terminate()
        assert not session.is_alive
        assert session.exit_code is not None

    def test_write_and_read(self):
        session = PtySession("/bin/sh")
        session.start()
        try:
            output = read_answer(session, b'echo hello_test_""marker')
            assert b"hello_test_marker" in output
        finally:
            session.terminate()

    def test_start_with_custom_dimensions(self):
        """PTY should be created with the specified dimensions."""
        session = PtySession("/bin/sh")
        session.start(initial_rows=50, initial_cols=132)
        try:
            assert _winsize(session) == (50, 132)
        finally:
            session.terminate()

    def test_resize(self):
        """A resize reaches the terminal the child reads its size from."""
        session = PtySession("/bin/sh")
        session.start()
        try:
            session.resize(40, 120)
            assert _winsize(session) == (40, 120)
        finally:
            session.terminate()

    def test_start_with_cwd(self, tmp_path):
        """Child process must run in the supplied working directory.

        ``osprey web --project X`` launched from another directory must spawn
        the terminal's Claude with cwd=X so it finds X/.mcp.json.
        """
        target = tmp_path / "project"
        target.mkdir()

        session = PtySession("/bin/sh")
        session.start(cwd=str(target))
        try:
            output = read_answer(session, b"pwd -P")
            expected = os.path.realpath(target).encode()
            assert expected in output
        finally:
            session.terminate()

    def test_sigwinch_delivered_on_resize(self, tmp_path):
        """SIGWINCH reaches the child when the PTY is resized.

        The child installs a SIGWINCH handler that writes a marker file and
        announces the handler first, so a resize never lands on a disposition
        that ignores it. On macOS delivery needs the controlling terminal the
        preexec sets up with TIOCSCTTY.
        """
        marker = tmp_path / "sigwinch"
        ready = tmp_path / "sigwinch_ready"
        child_script = "\n".join(
            [
                "import signal, pathlib",
                f"signal.signal(signal.SIGWINCH, lambda *_: pathlib.Path({str(marker)!r}).write_text('ok'))",
                f"pathlib.Path({str(ready)!r}).write_text('ok')",
                "while True: signal.pause()",
            ]
        )

        def _drain(fd: int) -> bytes:
            out = b""
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except (BlockingIOError, OSError):
                    return out
                if not chunk:
                    return out
                out += chunk

        session = PtySession([sys.executable, "-c", child_script])
        session.start()
        try:

            def handler_installed() -> bool:
                assert session.is_alive, (
                    f"the child exited with code {session.exit_code} before installing its "
                    f"handler: {_drain(session._master_fd)!r}"
                )
                return ready.exists()

            wait_until(
                handler_installed, timeout=CHILD_HANG_CEILING, what="the child's SIGWINCH handler"
            )

            # TIOCSWINSZ raises SIGWINCH only on a size that changed, so the
            # two sizes alternate, one resize a second until the child answers.
            sizes = itertools.cycle(((40, 120), (24, 80)))

            def delivered_or_dead() -> bool:
                if marker.exists() or not session.is_alive:
                    return True
                session.resize(*next(sizes))
                return False

            wait_until(
                delivered_or_dead,
                timeout=CHILD_HANG_CEILING,
                interval=1.0,
                what="SIGWINCH to reach the child",
            )

            assert session.is_alive, (
                f"the child exited with code {session.exit_code} before SIGWINCH was "
                f"delivered: {_drain(session._master_fd)!r}"
            )
            assert marker.exists(), "SIGWINCH was not delivered to the child process"
        finally:
            session.terminate()

    def test_terminate_ends_and_logs_the_process_groups_the_child_started(self, tmp_path, caplog):
        pid_file = tmp_path / "pids"
        scripts = [
            sleeper_script(tmp_path, "magnet_scan.py"),
            sleeper_script(tmp_path, "orbit_poll.py"),
        ]
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=scripts)]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 3)
            with caplog.at_level(logging.INFO):
                session.terminate()
            assert not session.is_alive
            for pid in pids[:2]:
                assert group_is_gone(pid, within=5.0), (
                    f"process {pid} the child started is still running"
                )
            assert group_is_gone(session.pid, within=5.0), (
                "the helper in the child's group is still running"
            )
            assert any(
                "magnet_scan.py" in record.getMessage() and str(pids[0]) in record.getMessage()
                for record in caplog.records
            ), [record.getMessage() for record in caplog.records]
        finally:
            session.terminate()
            kill_quietly(pids)

    def test_terminate_ends_a_helper_that_outlives_the_hang_up(self, tmp_path):
        pid_file = tmp_path / "pids"
        source = detaching_child_script(pid_file, scripts=[], helper_ignores_hup=True)
        session = PtySession([sys.executable, "-c", source])
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 1)
            session.terminate()
            assert group_is_gone(session.pid, within=5.0), (
                "the helper that ignores SIGHUP is still running"
            )
        finally:
            session.terminate()
            kill_quietly(pids)

    def test_terminate_ends_a_started_process_that_ignores_sigterm(self, tmp_path):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py", ignore=(signal.SIGTERM, signal.SIGHUP))
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            session.terminate()
            assert group_is_gone(pids[0], within=10.0), (
                "the SIGTERM-ignoring process is still running"
            )
        finally:
            session.terminate()
            kill_quietly(pids)

    def test_terminate_ends_what_the_child_started_when_the_first_process_table_read_fails(
        self, tmp_path, monkeypatch
    ):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py", ignore=(signal.SIGHUP,))
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            real = process_table.snapshot
            reads: list[None] = []

            def spy():
                reads.append(None)
                if len(reads) == 1:
                    raise process_table.ProcessTableUnreadable(OSError("process table"))
                return real()

            monkeypatch.setattr(process_table, "snapshot", spy)
            monkeypatch.setattr(process_table, "_READ_RETRY_S", 0.0)
            session.terminate()
            assert group_is_gone(pids[0], within=5.0), (
                "the command the child started is still running"
            )
        finally:
            monkeypatch.undo()
            session.terminate()
            kill_quietly(pids)

    def test_terminate_warns_and_ends_its_own_group_when_the_table_stays_unreadable(
        self, tmp_path, monkeypatch, caplog
    ):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py", ignore=(signal.SIGHUP,))
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            child = session.pid
            reads: list[None] = []

            def unreadable():
                reads.append(None)
                raise process_table.ProcessTableUnreadable(OSError("process table"))

            monkeypatch.setattr(process_table, "snapshot", unreadable)
            monkeypatch.setattr(process_tree, "READ_WAIT_S", 0.3)
            with caplog.at_level(logging.WARNING):
                session.terminate()

            assert len(reads) > 1
            assert not session.is_alive
            warnings = [
                r.getMessage()
                for r in caplog.records
                if r.levelno == logging.WARNING
                and "what it started is left running" in r.getMessage()
            ]
            assert len(warnings) == 1, warnings
            assert f"PTY child {child} " in warnings[0]
        finally:
            monkeypatch.undo()
            session.terminate()
            kill_quietly(pids)

    def test_terminate_of_a_dead_child_looks_for_nothing_it_started(self, monkeypatch):
        session = PtySession(["/bin/sh", "-c", "exit 0"])
        session.start()
        wait_for_exit(session)
        calls: list[None] = []

        def spy(*_args, **_kwargs) -> None:
            calls.append(None)
            return None

        monkeypatch.setattr(process_tree, "read_while_alive", spy)
        session.terminate()
        assert calls == []
        assert session.exit_code == 0

    def test_is_alive_never_reaps_the_child(self):
        session = PtySession([sys.executable, "-c", "pass"])
        session.start()
        try:
            assert wait_for_exit(session) == 0
            pid = session.pid
            assert pid is not None
            look = os.WEXITED | os.WNOHANG | os.WNOWAIT
            assert subprocess_capture._waitid(pid, look) is not None, "the probe reaped the child"
            session.terminate()
            assert session.exit_code == 0
            with pytest.raises(ChildProcessError):
                subprocess_capture._waitid(pid, look)
        finally:
            session.terminate()

    def test_terminate_stops_the_childs_own_group_including_a_member_that_ignores_the_hang_up(
        self, tmp_path
    ):
        session, (child, grandchild) = _stubborn_session(tmp_path)
        try:
            assert session.pid == child
            session.terminate()
            assert subprocess_capture.group_is_gone(child), (
                f"the child's group (pids {child} {grandchild}) is still running"
            )
            assert not session.is_alive
        finally:
            kill_group(child)
            session.terminate()

    def test_terminate_names_the_survivors_of_its_own_group(self, tmp_path, monkeypatch, caplog):
        real_stop = subprocess_capture.stop_child
        real_end = process_tree.end_groups
        ended: list[int] = []

        def stop_reporting_a_survivor(process, **kwargs):
            real_stop(process, **kwargs)
            return subprocess_capture.StopOutcome(ended=(process.pid,), survivors=(4242,))

        def end_spy(groups):
            ended.extend(group.pgid for group in groups)
            return real_end(groups)

        monkeypatch.setattr(subprocess_capture, "stop_child", stop_reporting_a_survivor)
        monkeypatch.setattr(process_tree, "end_groups", end_spy)
        session, (child, _) = _stubborn_session(tmp_path)
        try:
            with caplog.at_level(logging.WARNING):
                session.terminate()
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            named = [w for w in warnings if "4242" in w]
            assert len(named) == 1, warnings
            assert f"PTY child {child}" in named[0]
            assert child not in ended, "the child's group was stopped twice"
        finally:
            kill_group(child)
            session.terminate()

    def test_terminate_says_the_table_was_unreadable_not_a_pid(self, tmp_path, monkeypatch, caplog):
        real_stop = subprocess_capture.stop_child

        def stop_that_could_not_look(process, **kwargs):
            real_stop(process, **kwargs)
            return subprocess_capture.StopOutcome(ended=(process.pid,), unknown=(process.pid,))

        monkeypatch.setattr(subprocess_capture, "stop_child", stop_that_could_not_look)
        session, (child, _) = _stubborn_session(tmp_path)
        try:
            with caplog.at_level(logging.WARNING):
                session.terminate()
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any(
                f"process table unreadable: group {child} may still be running" in w
                for w in warnings
            ), warnings
            assert not any("pids" in w for w in warnings), warnings
        finally:
            kill_group(child)
            session.terminate()

    def test_terminate_says_started_groups_are_unknown_when_the_table_fails_after_the_first_read(
        self, tmp_path, monkeypatch, caplog
    ):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py", ignore=(signal.SIGHUP,))
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            sleeper = os.getpgid(pids[0])
            real_snapshot = process_table.snapshot
            real_killpg = os.killpg
            reads: list[None] = []
            targets: list[int] = []

            def first_read_only():
                reads.append(None)
                if len(reads) > 1:
                    raise process_table.ProcessTableUnreadable(PermissionError("process table"))
                return real_snapshot()

            def killpg(pgid: int, sig: int) -> None:
                targets.append(pgid)
                real_killpg(pgid, sig)

            with monkeypatch.context() as patch, caplog.at_level(logging.WARNING):
                patch.setattr(process_table, "snapshot", first_read_only)
                patch.setattr(os, "killpg", killpg)
                session.terminate()

            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            unknown = [
                w
                for w in warnings
                if "process table unreadable: started group(s) may still be running" in w
            ]
            assert len(unknown) == 1, warnings
            assert f"pgid {sleeper} " in unknown[0], unknown
            assert not any("survived SIGKILL" in w for w in warnings), warnings
            assert sleeper not in targets, targets
        finally:
            monkeypatch.undo()
            session.terminate()
            kill_quietly(pids)

    def test_started_commands_lists_only_groups_of_their_own(self, tmp_path):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py")
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            grandchild, helper = pids

            commands = session.started_commands()

            assert [c.label for c in commands] == ["magnet_scan.py"]
            assert commands[0].pgid == grandchild
            assert helper not in {pid for c in commands for pid, _ in c.members}
        finally:
            session.terminate()
            kill_quietly(pids)

    def test_started_commands_of_a_dead_child_is_empty(self, monkeypatch):
        session = PtySession(["/bin/sh", "-c", "exit 0"])
        session.start()
        wait_for_exit(session)
        calls: list[None] = []

        def spy() -> None:
            calls.append(None)
            return None

        monkeypatch.setattr(process_table, "snapshot", spy)
        try:
            assert session.started_commands() == []
            assert calls == []
        finally:
            monkeypatch.undo()
            session.terminate()

    def test_started_commands_raises_when_the_table_cannot_be_read(self, monkeypatch):
        def unreadable():
            raise process_table.ProcessTableUnreadable(PermissionError("process table"))

        monkeypatch.setattr(process_table, "snapshot", unreadable)
        session = PtySession(["/bin/sh", "-c", "sleep 60"])
        session.start()
        try:
            with pytest.raises(process_table.ProcessTableUnreadable):
                session.started_commands()
        finally:
            session.terminate()

        dead = PtySession(["/bin/sh", "-c", "exit 0"])
        dead.start()
        wait_for_exit(dead)
        try:
            assert dead.started_commands() == []
        finally:
            dead.terminate()


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
class TestPtyChildWaits:
    def test_read_answer_returns_the_executed_answer(self):
        session = PtySession("/bin/sh")
        session.start()
        try:
            output = read_answer(session, b'echo left_""right')
            assert b"left_right" in output
            assert SENTINEL in output
            assert output.index(b"left_right") < output.rindex(SENTINEL)
        finally:
            session.terminate()

    def test_read_answer_fails_on_a_dead_child_naming_its_exit_code(self):
        session = PtySession(["/bin/sh", "-c", "exit 3"])
        session.start()
        try:
            wait_for_exit(session)
            with pytest.raises(AssertionError, match=r"exited with code 3 before answering"):
                read_answer(session, b"echo hi")
        finally:
            session.terminate()

    def test_wait_for_report_fails_on_a_dead_reporter(self, tmp_path):
        session = PtySession(["/bin/sh", "-c", "exit 4"])
        session.start()
        try:
            wait_for_exit(session)
            with pytest.raises(AssertionError, match=r"exited with code 4 with 0 of 1 line"):
                wait_for_report(tmp_path / "never.txt", 1, session)
        finally:
            session.terminate()

    def test_wait_for_report_accepts_a_line_written_just_before_exit(self, tmp_path):
        report = tmp_path / "report.txt"
        session = PtySession(["/bin/sh", "-c", f'printf "x\\n" >> "{report}"'])
        session.start()
        try:
            wait_for_exit(session)
            assert wait_for_report(report, 1, session) == ["x"]
        finally:
            session.terminate()

    async def test_wait_gone_returns_once_every_pid_has_exited(self):
        procs = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range(2)]
        try:
            # Unreaped on purpose: an exited process nobody has waited for counts as gone.
            await wait_gone([proc.pid for proc in procs], within=CHILD_HANG_CEILING)
        finally:
            for proc in procs:
                proc.kill()
                proc.wait(timeout=30)

    async def test_wait_gone_fails_naming_the_pid_still_running(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            with pytest.raises(
                AssertionError, match=rf"pid {proc.pid} \(ps state '\S+'\) is still"
            ):
                await wait_gone([proc.pid], within=0.1)
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_hang_ceiling_sits_below_the_per_test_timeout(self):
        assert CHILD_HANG_CEILING < 300


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
class TestPtyRegistry:
    def test_stale_cleanup_does_not_kill_replacement(self):
        """A stale owner's teardown kills only its own process.

        The first child has died and a replacement is pooled under the same
        key; the first handler's cleanup must leave the replacement alone.
        """
        registry = PtyRegistry()
        try:
            session_1, _ = registry.get_or_create_session("default", ["/bin/sh", "-c", "exit 0"])
            wait_for_exit(session_1)
            assert not session_1.is_alive

            session_2, reused = registry.get_or_create_session("default", "/bin/sh")
            assert reused is False
            assert session_2.is_alive

            registry.terminate_session_if_owner("default", session_1)

            assert session_2.is_alive
            assert registry.get_session("default") is session_2
        finally:
            registry.cleanup_all()

    @pytest.mark.filterwarnings("error::ResourceWarning")
    def test_a_dead_session_found_at_reattach_is_reaped(self):
        registry = PtyRegistry()
        try:
            dead, _ = registry.get_or_create_session("default", ["/bin/sh", "-c", "exit 0"])
            wait_for_exit(dead)
            pid = dead.pid
            assert pid is not None

            replacement, reused = registry.get_or_create_session("default", "/bin/sh")

            assert reused is False
            assert replacement is not dead
            assert dead.exit_code == 0
            with pytest.raises(ChildProcessError):
                subprocess_capture._waitid(pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        finally:
            registry.cleanup_all()

    def test_get_or_create_session_passes_initial_dimensions(self):
        """The registry forwards the requested size to the spawn."""
        registry = PtyRegistry()
        try:
            session, _ = registry.get_or_create_session("test-dims", "/bin/sh", rows=48, cols=160)
            assert _winsize(session) == (48, 160)
        finally:
            registry.cleanup_all()

    def test_owner_terminate_works_when_still_owner(self):
        """When the owning handler tears down, the session is killed and forgotten."""
        registry = PtyRegistry()
        try:
            session, _ = registry.get_or_create_session("default", "/bin/sh")
            assert session.is_alive

            registry.terminate_session_if_owner("default", session)
            assert not session.is_alive
            assert registry.get_session("default") is None
        finally:
            registry.cleanup_all()

    def test_get_or_create_session_passes_cwd(self, tmp_path):
        """The interactive terminal's spawn path forwards cwd."""
        target = tmp_path / "proj"
        target.mkdir()

        registry = PtyRegistry()
        try:
            session, _reused = registry.get_or_create_session(
                "term-cwd", "/bin/sh", cwd=str(target)
            )
            output = read_answer(session, b"pwd -P")
            expected = os.path.realpath(target).encode()
            assert expected in output
        finally:
            registry.cleanup_all()

    def test_eviction_ends_the_evicted_terminals_started_processes(self, tmp_path):
        pid_file = tmp_path / "pids"
        script = detaching_child_script(
            pid_file, scripts=[sleeper_script(tmp_path, "magnet_scan.py")]
        )
        registry = PtyRegistry(max_background=1)
        pids: list[int] = []
        try:
            session, _ = registry.get_or_create_session("a", [sys.executable, "-c", script])
            pids = wait_for_pids(session, pid_file, 2)
            registry.get_or_create_session("b", "/bin/sh")
            assert registry.get_session("a") is None
            assert group_is_gone(pids[0], within=5.0), (
                "the evicted terminal's started process is still running"
            )
        finally:
            registry.cleanup_all()
            kill_quietly(pids)

    def test_cleanup_all_ends_every_terminals_started_processes(self, tmp_path):
        registry = PtyRegistry()
        pids: list[int] = []
        try:
            for key in ("a", "b"):
                directory = tmp_path / key
                directory.mkdir()
                pid_file = directory / "pids"
                script = detaching_child_script(
                    pid_file, scripts=[sleeper_script(directory, "magnet_scan.py")]
                )
                session, _ = registry.get_or_create_session(key, [sys.executable, "-c", script])
                pids += wait_for_pids(session, pid_file, 2)
            registry.cleanup_all()
            grandchildren = pids[0::2]
            for pid in grandchildren:
                assert group_is_gone(pid, within=5.0), f"started process {pid} is still running"
        finally:
            registry.cleanup_all()
            kill_quietly(pids)
