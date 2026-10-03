"""Tests for PTY manager: real children behind ``PtySession`` and ``PtyRegistry``."""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import signal
import struct
import sys
import termios
import time

import pytest

from osprey.interfaces.web_terminal import process_tree
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, PtySession
from tests.interfaces.web_terminal._pty_child import (
    CHILD_HANG_CEILING,
    SENTINEL,
    detaching_child_script,
    kill_quietly,
    pid_gone,
    read_answer,
    sleeper_script,
    wait_for_exit,
    wait_for_pids,
    wait_for_report,
)

needs_ps = pytest.mark.skipif(
    shutil.which("ps") is None, reason="needs ps to read the process tree"
)


def _winsize(session: PtySession) -> tuple[int, int]:
    buf = fcntl.ioctl(session._master_fd, termios.TIOCGWINSZ, b"\x00" * 8)
    rows, cols = struct.unpack("HHHH", buf)[:2]
    return rows, cols


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
            deadline = time.monotonic() + CHILD_HANG_CEILING
            while not ready.exists():
                assert session.is_alive, (
                    f"the child exited with code {session.exit_code} before installing its "
                    f"handler: {_drain(session._master_fd)!r}"
                )
                assert time.monotonic() < deadline, "the child never installed its handler"
                time.sleep(0.05)

            # TIOCSWINSZ raises SIGWINCH only on a size that changed, so the
            # two sizes alternate.
            sizes = ((40, 120), (24, 80))
            attempt = 0
            deadline = time.monotonic() + CHILD_HANG_CEILING
            while not marker.exists() and session.is_alive and time.monotonic() < deadline:
                session.resize(*sizes[attempt % 2])
                attempt += 1
                round_ends = time.monotonic() + 1
                while not marker.exists() and time.monotonic() < round_ends:
                    time.sleep(0.05)

            assert session.is_alive, (
                f"the child exited with code {session.exit_code} before SIGWINCH was "
                f"delivered: {_drain(session._master_fd)!r}"
            )
            assert marker.exists(), "SIGWINCH was not delivered to the child process"
        finally:
            session.terminate()

    @needs_ps
    def test_terminate_ends_the_process_groups_the_child_started(self, tmp_path):
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
            session.terminate()
            assert not session.is_alive
            for pid in pids:
                assert pid_gone(pid), f"process {pid} the child started is still running"
        finally:
            session.terminate()
            kill_quietly(pids)

    @needs_ps
    def test_terminate_ends_a_helper_that_outlives_the_hang_up(self, tmp_path):
        pid_file = tmp_path / "pids"
        source = detaching_child_script(pid_file, scripts=[], helper_ignores_hup=True)
        session = PtySession([sys.executable, "-c", source])
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 1)
            session.terminate()
            assert pid_gone(pids[0]), "the helper that ignores SIGHUP is still running"
        finally:
            session.terminate()
            kill_quietly(pids)

    @needs_ps
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
            assert pid_gone(pids[0], within=10.0), "the SIGTERM-ignoring process is still running"
        finally:
            session.terminate()
            kill_quietly(pids)

    @needs_ps
    def test_terminate_logs_what_it_ended(self, tmp_path, caplog):
        pid_file = tmp_path / "pids"
        script = sleeper_script(tmp_path, "magnet_scan.py")
        session = PtySession(
            [sys.executable, "-c", detaching_child_script(pid_file, scripts=[script])]
        )
        session.start()
        pids: list[int] = []
        try:
            pids = wait_for_pids(session, pid_file, 2)
            with caplog.at_level(logging.INFO):
                session.terminate()
            assert any(
                "magnet_scan.py" in record.getMessage() and str(pids[0]) in record.getMessage()
                for record in caplog.records
            ), [record.getMessage() for record in caplog.records]
        finally:
            session.terminate()
            kill_quietly(pids)

    def test_terminate_of_a_dead_child_looks_for_nothing(self, monkeypatch):
        session = PtySession(["/bin/sh", "-c", "exit 0"])
        session.start()
        wait_for_exit(session)
        calls: list[None] = []

        def spy() -> None:
            calls.append(None)
            return None

        monkeypatch.setattr(process_tree, "snapshot", spy)
        session.terminate()
        assert calls == []


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

    @needs_ps
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
            assert pid_gone(pids[0]), "the evicted terminal's started process is still running"
        finally:
            registry.cleanup_all()
            kill_quietly(pids)

    @needs_ps
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
                assert pid_gone(pid), f"started process {pid} is still running"
        finally:
            registry.cleanup_all()
            kill_quietly(pids)
