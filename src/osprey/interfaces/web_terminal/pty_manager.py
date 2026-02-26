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
from collections import OrderedDict
from collections.abc import AsyncIterator

from osprey.utils.logger import get_logger

logger = get_logger("pty_manager")


class PtySession:
    """Manages a single PTY-backed subprocess."""

    def __init__(self, shell_command: str | list[str]) -> None:
        if isinstance(shell_command, str):
            self._command_list = [shell_command]
        else:
            self._command_list = list(shell_command)
        self._master_fd: int | None = None
        self._process: subprocess.Popen | None = None
        self._last_rows: int = 24
        self._last_cols: int = 80

    def start(
        self,
        initial_rows: int = 24,
        initial_cols: int = 80,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Spawn the shell process attached to a new PTY.

        Args:
            initial_rows: Initial terminal row count (default 24).
            initial_cols: Initial terminal column count (default 80).
            extra_env: Additional environment variables to set in the child process.
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
            k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))
        }

        # When token-based auth is configured (e.g. CBORG proxy at LBNL),
        # strip ANTHROPIC_API_KEY to prevent the "auth conflict" warning.
        # The .env in the project dir may contain a stale API key that
        # Claude Code loads automatically, conflicting with the token.
        if env.get("ANTHROPIC_AUTH_TOKEN"):
            env.pop("ANTHROPIC_API_KEY", None)

        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"

        if extra_env:
            env.update(extra_env)

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
            self._command_list,
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
        self._last_rows = rows
        self._last_cols = cols

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
    """Manages multiple PTY sessions with LRU pool semantics.

    Sessions are kept alive in the background after detach, enabling
    near-instant reattach when switching between Claude sessions.
    """

    def __init__(self, max_background: int = 5) -> None:
        self._sessions: OrderedDict[str, PtySession] = OrderedDict()
        self._attached: set[str] = set()
        self._max_background = max_background

    # ---- Pool methods ---- #

    def get_or_create_session(
        self,
        session_key: str,
        command: str | list[str],
        rows: int = 24,
        cols: int = 80,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[PtySession, bool]:
        """Get existing session or create a new one.

        Returns:
            (session, was_reused) — True if an existing live session was reattached.
        """
        existing = self._sessions.get(session_key)
        if existing is not None:
            if existing.is_alive:
                # LRU bump — move to end
                self._sessions.move_to_end(session_key)
                existing.resize(rows, cols)
                return existing, True
            else:
                # Dead — remove silently, respawn below
                self._sessions.pop(session_key, None)
                self._attached.discard(session_key)

        # Evict if at capacity
        self._evict_lru()

        session = self._spawn_session(command, rows, cols, extra_env)
        self._sessions[session_key] = session
        return session, False

    def attach_session(self, session_key: str) -> bool:
        """Mark session as actively consumed by a WebSocket.

        Returns False if already attached or not in pool.
        """
        if session_key not in self._sessions:
            return False
        if session_key in self._attached:
            return False
        self._attached.add(session_key)
        return True

    def detach_session(self, session_key: str) -> None:
        """Remove from attached set without terminating.

        LRU-bumps the session so it's less likely to be evicted.
        """
        self._attached.discard(session_key)
        if session_key in self._sessions:
            self._sessions.move_to_end(session_key)

    def rekey_session(self, old_key: str, new_key: str) -> None:
        """Rename a session entry (e.g. after UUID discovery)."""
        if old_key not in self._sessions:
            return
        session = self._sessions.pop(old_key)
        self._sessions[new_key] = session
        if old_key in self._attached:
            self._attached.discard(old_key)
            self._attached.add(new_key)

    def _evict_lru(self) -> None:
        """Evict the oldest non-attached session if at capacity."""
        if len(self._sessions) < self._max_background:
            return
        # Find oldest non-attached
        for key in list(self._sessions):
            if key not in self._attached:
                evicted = self._sessions.pop(key)
                evicted.terminate()
                logger.info("Evicted LRU session %s", key)
                return

    def _spawn_session(
        self,
        command: str | list[str],
        rows: int,
        cols: int,
        extra_env: dict[str, str] | None,
    ) -> PtySession:
        """Create and start a new PtySession."""
        session = PtySession(command)
        session.start(initial_rows=rows, initial_cols=cols, extra_env=extra_env)
        return session

    # ---- Session methods (kept for operator sessions and tests) ---- #

    def create_session(
        self,
        session_id: str,
        shell_command: str | list[str],
        initial_rows: int = 24,
        initial_cols: int = 80,
        extra_env: dict[str, str] | None = None,
    ) -> PtySession:
        """Create and start a new PTY session."""
        if session_id in self._sessions:
            self._sessions[session_id].terminate()

        session = PtySession(shell_command)
        session.start(initial_rows=initial_rows, initial_cols=initial_cols, extra_env=extra_env)
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
        self._attached.discard(session_id)

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
        self._attached.clear()
