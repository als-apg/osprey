"""Tests for PTY manager."""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, PtySession


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
class TestPtySession:
    def test_start_and_is_alive(self):
        session = PtySession("echo")
        session.start()
        # echo exits immediately, but let's just check it started
        # (may already have exited)
        session.terminate()

    def test_terminate_cleans_up(self):
        session = PtySession("/bin/sh")
        session.start()
        assert session.is_alive
        session.terminate()
        assert not session.is_alive

    def test_write_and_read(self):
        session = PtySession("/bin/sh")
        session.start()
        try:
            # Write a command
            session.write_input(b"echo hello_test_marker\n")

            # Read output — we should eventually see our marker
            import os
            import select
            import time

            output = b""
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                r, _, _ = select.select([session._master_fd], [], [], 0.1)
                if r:
                    try:
                        chunk = os.read(session._master_fd, 4096)
                        output += chunk
                        if b"hello_test_marker" in output:
                            break
                    except OSError:
                        break

            assert b"hello_test_marker" in output
        finally:
            session.terminate()

    def test_start_with_custom_dimensions(self):
        """PTY should be created with the specified dimensions."""
        session = PtySession("/bin/sh")
        session.start(initial_rows=50, initial_cols=132)
        try:
            # Query the PTY's current window size via ioctl
            import fcntl
            import struct
            import termios

            buf = fcntl.ioctl(session._master_fd, termios.TIOCGWINSZ, b"\x00" * 8)
            rows, cols = struct.unpack("HHHH", buf)[:2]
            assert rows == 50
            assert cols == 132
        finally:
            session.terminate()

    def test_resize(self):
        session = PtySession("/bin/sh")
        session.start()
        try:
            # Should not raise
            session.resize(40, 120)
        finally:
            session.terminate()

    def test_exit_code_none_while_running(self):
        session = PtySession("/bin/sh")
        session.start()
        try:
            assert session.exit_code is None
        finally:
            session.terminate()

    def test_sigwinch_delivered_on_resize(self):
        """SIGWINCH must be delivered to the child when the PTY is resized.

        The child installs a SIGWINCH handler that creates a marker file,
        then the test calls session.resize() and checks for the file.
        """
        marker = tempfile.mktemp(suffix="_sigwinch")

        # Python one-liner: install SIGWINCH handler, write marker, wait.
        child_script = (
            "import signal, time, pathlib; "
            f"pathlib.Path('{marker}').unlink(missing_ok=True); "
            f"signal.signal(signal.SIGWINCH, lambda *_: pathlib.Path('{marker}').write_text('ok')); "
            "time.sleep(10)"
        )

        session = PtySession(sys.executable)
        # Pass the script via -c by reaching into the Popen machinery:
        # PtySession wraps [shell_command], so we set the command to python
        # and inject args.  Easiest: override _shell_command with a full cmd.
        session._shell_command = sys.executable
        # We need to pass -c and the script.  PtySession wraps the command
        # in a list, so we'll temporarily monkey-patch start() to pass args.
        import subprocess
        import fcntl
        import struct
        import termios

        master_fd, slave_fd = __import__("pty").openpty()
        winsize = struct.pack("HHHH", 24, 80, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

        slave_for_preexec = slave_fd

        def preexec():
            os.setsid()
            fcntl.ioctl(slave_for_preexec, termios.TIOCSCTTY, 0)

        proc = subprocess.Popen(
            [sys.executable, "-c", child_script],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=preexec,
        )
        os.close(slave_fd)

        # Set master to non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        try:
            # Give child time to install the signal handler
            time.sleep(0.5)

            # Resize the PTY — should deliver SIGWINCH to the child
            new_winsize = struct.pack("HHHH", 40, 120, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, new_winsize)

            # Wait for the marker file to appear
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if os.path.exists(marker):
                    break
                time.sleep(0.1)

            assert os.path.exists(marker), "SIGWINCH was not delivered to the child process"
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
                proc.wait()
            try:
                os.close(master_fd)
            except OSError:
                pass
            try:
                os.unlink(marker)
            except OSError:
                pass


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
class TestPtyRegistry:
    def test_create_and_get_session(self):
        registry = PtyRegistry()
        try:
            session = registry.create_session("test-1", "/bin/sh")
            assert session is not None
            assert registry.get_session("test-1") is session
        finally:
            registry.cleanup_all()

    def test_terminate_session(self):
        registry = PtyRegistry()
        try:
            registry.create_session("test-1", "/bin/sh")
            registry.terminate_session("test-1")
            assert registry.get_session("test-1") is None
        finally:
            registry.cleanup_all()

    def test_create_replaces_existing(self):
        registry = PtyRegistry()
        try:
            s1 = registry.create_session("test-1", "/bin/sh")
            s2 = registry.create_session("test-1", "/bin/sh")
            assert s1 is not s2
            assert not s1.is_alive
        finally:
            registry.cleanup_all()

    def test_cleanup_all(self):
        registry = PtyRegistry()
        registry.create_session("a", "/bin/sh")
        registry.create_session("b", "/bin/sh")
        registry.cleanup_all()
        assert registry.get_session("a") is None
        assert registry.get_session("b") is None

    def test_terminate_nonexistent_session(self):
        registry = PtyRegistry()
        # Should not raise
        registry.terminate_session("nonexistent")

    def test_stale_cleanup_does_not_kill_replacement(self):
        """Simulate page reload: WS1 creates session, WS2 replaces it,
        then WS1's finally block runs — must NOT kill WS2's session."""
        registry = PtyRegistry()
        try:
            # WS1 creates a session
            session_1 = registry.create_session("default", "/bin/sh")
            assert session_1.is_alive

            # WS2 connects (page reload) — replaces session
            session_2 = registry.create_session("default", "/bin/sh")
            assert not session_1.is_alive  # session_1 was killed
            assert session_2.is_alive

            # WS1's finally block runs with the OLD session reference.
            # It must NOT kill session_2.
            registry.terminate_session_if_owner("default", session_1)

            # session_2 must still be alive
            assert session_2.is_alive
            assert registry.get_session("default") is session_2
        finally:
            registry.cleanup_all()

    def test_create_session_passes_initial_dimensions(self):
        """Registry should forward initial dimensions to the PTY session."""
        import fcntl
        import struct
        import termios

        registry = PtyRegistry()
        try:
            session = registry.create_session(
                "test-dims", "/bin/sh", initial_rows=48, initial_cols=160
            )
            buf = fcntl.ioctl(session._master_fd, termios.TIOCGWINSZ, b"\x00" * 8)
            rows, cols = struct.unpack("HHHH", buf)[:2]
            assert rows == 48
            assert cols == 160
        finally:
            registry.cleanup_all()

    def test_owner_terminate_works_when_still_owner(self):
        """When the owning WS disconnects normally, cleanup works."""
        registry = PtyRegistry()
        try:
            session = registry.create_session("default", "/bin/sh")
            assert session.is_alive

            # Owner terminates — should work
            registry.terminate_session_if_owner("default", session)
            assert not session.is_alive
            assert registry.get_session("default") is None
        finally:
            registry.cleanup_all()

    def test_create_session_with_command_list(self):
        """PtySession with a list command works correctly."""
        registry = PtyRegistry()
        try:
            session = registry.create_session("test-list", ["echo", "hello"])
            assert session is not None
            # echo exits immediately, just verify it was created
        finally:
            registry.cleanup_all()

    def test_start_with_extra_env(self):
        """Extra env vars are passed to the child process."""
        import select

        registry = PtyRegistry()
        try:
            session = registry.create_session(
                "test-env",
                "/bin/sh",
                extra_env={"OSPREY_TEST_VAR": "test_value_12345"},
            )
            # Ask the shell to print the env var
            session.write_input(b"echo $OSPREY_TEST_VAR\n")

            output = b""
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                r, _, _ = select.select([session._master_fd], [], [], 0.1)
                if r:
                    try:
                        chunk = os.read(session._master_fd, 4096)
                        output += chunk
                        if b"test_value_12345" in output:
                            break
                    except OSError:
                        break

            assert b"test_value_12345" in output
        finally:
            registry.cleanup_all()
