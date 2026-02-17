"""PTY session management using stdlib pty + asyncio.

Provides PtySession (single terminal process) and PtyRegistry (multi-session
manager with cleanup) for the OSPREY Web Terminal.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import signal
import struct
import subprocess
import termios
from collections.abc import AsyncIterator

from osprey.utils.logger import get_logger

logger = get_logger("pty_manager")


class PtySession:
    """Manages a single PTY-backed subprocess."""

    def __init__(self, shell_command: str) -> None:
        self._shell_command = shell_command
        self._master_fd: int | None = None
        self._process: subprocess.Popen | None = None

    def start(self, initial_rows: int = 24, initial_cols: int = 80) -> None:
        """Spawn the shell process attached to a new PTY.

        Args:
            initial_rows: Initial terminal row count (default 24).
            initial_cols: Initial terminal column count (default 80).
        """
        master_fd, slave_fd = pty.openpty()

        # Set initial terminal size BEFORE spawning — a 0x0 PTY causes
        # many TUI programs (including Claude Code) to exit immediately.
        winsize = struct.pack("HHHH", initial_rows, initial_cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

        # Build a clean environment for the child process.
        # Strip Claude Code internal session variables (nesting detection,
        # entrypoint tracking, beta flags).
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))
        }

        # When token-based auth is configured (e.g. CBORG proxy at LBNL),
        # strip ANTHROPIC_API_KEY to prevent the "auth conflict" warning.
        # The .env in the project dir may contain a stale API key that
        # Claude Code loads automatically, conflicting with the token.
        if env.get("ANTHROPIC_AUTH_TOKEN"):
            env.pop("ANTHROPIC_API_KEY", None)

        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"

        # Capture for closure — preexec runs in the child after fork().
        slave_for_preexec = slave_fd

        def _child_preexec() -> None:
            """Set up the child's session and controlling terminal.

            setsid() creates a new session (detaching from the parent's
            controlling terminal).  On macOS the inherited slave fd does NOT
            automatically become the controlling terminal, so we must call
            TIOCSCTTY explicitly.  Without a controlling terminal the kernel
            has no process group to deliver SIGWINCH to when the master's
            window size changes.
            """
            os.setsid()
            fcntl.ioctl(slave_for_preexec, termios.TIOCSCTTY, 0)

        self._process = subprocess.Popen(
            [self._shell_command],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=_child_preexec,
            env=env,
        )

        # Close slave in parent — only the child uses it
        os.close(slave_fd)

        # Set master to non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        self._master_fd = master_fd

    async def read_output(self) -> AsyncIterator[bytes]:
        """Yield chunks of PTY output as they arrive.

        Continues reading after the process exits to drain any
        remaining buffered output before signalling completion.
        """
        if self._master_fd is None:
            return

        loop = asyncio.get_event_loop()
        fd = self._master_fd

        while True:
            try:
                data = await loop.run_in_executor(None, self._blocking_read, fd)
                if data:
                    yield data
                elif not self.is_alive:
                    # Process exited and no more data in buffer
                    break
            except OSError:
                break

    @staticmethod
    def _blocking_read(fd: int) -> bytes:
        """Blocking read with short timeout for cancellation responsiveness."""
        import select

        readable, _, _ = select.select([fd], [], [], 0.1)
        if readable:
            try:
                return os.read(fd, 4096)
            except OSError:
                return b""
        return b""

    def write_input(self, data: bytes) -> None:
        """Write raw bytes to the PTY (keystrokes from the client)."""
        if self._master_fd is not None:
            os.write(self._master_fd, data)

    def resize(self, rows: int, cols: int) -> None:
        """Notify the PTY of a terminal size change."""
        if self._master_fd is not None:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)

    def terminate(self) -> None:
        """Terminate the subprocess and close the PTY."""
        if self._process is not None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "PTY process %d did not exit after SIGKILL — orphaned",
                        self._process.pid,
                    )

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

    @property
    def is_alive(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Return exit code if process has terminated, else None."""
        if self._process is None:
            return None
        return self._process.poll()


class PtyRegistry:
    """Manages multiple PTY sessions keyed by session ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, PtySession] = {}

    def create_session(
        self,
        session_id: str,
        shell_command: str,
        initial_rows: int = 24,
        initial_cols: int = 80,
    ) -> PtySession:
        """Create and start a new PTY session."""
        if session_id in self._sessions:
            self._sessions[session_id].terminate()

        session = PtySession(shell_command)
        session.start(initial_rows=initial_rows, initial_cols=initial_cols)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> PtySession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def terminate_session(self, session_id: str) -> None:
        """Terminate and remove a session."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.terminate()

    def terminate_session_if_owner(self, session_id: str, owner: PtySession) -> None:
        """Terminate only if the caller still owns the session.

        Prevents a stale WebSocket's cleanup from killing a newer session
        that replaced it (e.g. on page reload or reconnection).
        """
        current = self._sessions.get(session_id)
        if current is owner:
            self.terminate_session(session_id)
        elif owner is not None:
            # Stale session — just terminate the process directly,
            # don't touch the registry (it has a newer session).
            owner.terminate()

    def cleanup_all(self) -> None:
        """Terminate all sessions (called during shutdown)."""
        for session_id in list(self._sessions):
            self.terminate_session(session_id)
