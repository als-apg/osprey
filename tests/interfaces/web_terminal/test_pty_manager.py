"""Tests for PTY manager: real children behind ``PtySession`` and ``PtyRegistry``."""

from __future__ import annotations

import fcntl
import os
import select
import struct
import sys
import termios
import time

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, PtySession

#: How long a real shell may take to answer under a loaded parallel run.
READ_TIMEOUT_S = 15.0


def _read_until(session: PtySession, marker: bytes, timeout: float = READ_TIMEOUT_S) -> bytes:
    """Read the PTY master until *marker* appears or *timeout* passes.

    The shell echoes the command line first and answers later, so a reader
    that stops early sees only the echo. Only the deadline ends the wait.
    """
    output = b""
    deadline = time.monotonic() + timeout
    while marker not in output and time.monotonic() < deadline:
        readable, _, _ = select.select([session._master_fd], [], [], 0.1)
        if readable:
            try:
                output += os.read(session._master_fd, 4096)
            except BlockingIOError:
                continue
    return output


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
            session.write_input(b"echo hello_test_marker\n")
            assert b"hello_test_marker" in _read_until(session, b"hello_test_marker")
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
            session.write_input(b"pwd -P\n")
            expected = os.path.realpath(target).encode()
            assert expected in _read_until(session, expected)
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
        child_script = (
            "import signal, time, pathlib; "
            f"signal.signal(signal.SIGWINCH, lambda *_: pathlib.Path({str(marker)!r}).write_text('ok')); "
            f"pathlib.Path({str(ready)!r}).write_text('ok'); "
            "time.sleep(60)"
        )

        session = PtySession([sys.executable, "-c", child_script])
        session.start()
        try:
            deadline = time.monotonic() + 30
            while not ready.exists():
                assert time.monotonic() < deadline, "the child never installed its handler"
                time.sleep(0.05)

            # TIOCSWINSZ raises SIGWINCH only on a size that changed, so the
            # two sizes alternate.
            sizes = ((40, 120), (24, 80))
            attempt = 0
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline:
                session.resize(*sizes[attempt % 2])
                attempt += 1
                round_ends = time.monotonic() + 1
                while not marker.exists() and time.monotonic() < round_ends:
                    time.sleep(0.05)

            assert marker.exists(), "SIGWINCH was not delivered to the child process"
        finally:
            session.terminate()


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
            deadline = time.monotonic() + 5
            while session_1.is_alive and time.monotonic() < deadline:
                time.sleep(0.02)
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
            session.write_input(b"pwd -P\n")
            expected = os.path.realpath(target).encode()
            assert expected in _read_until(session, expected)
        finally:
            registry.cleanup_all()
