"""PTY session management using stdlib pty + asyncio.

Provides PtySession (single terminal process) and PtyRegistry (multi-session
manager with cleanup) for the OSPREY Web Terminal.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import pty
import signal
import struct
import subprocess
import termios
from collections import OrderedDict
from collections.abc import AsyncIterator

from osprey.agent_runner.clean_env import build_base_child_env
from osprey.utils.logger import get_logger

logger = get_logger("pty_manager")


def build_pty_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for the PTY child process.

    Layers the PTY-specific keys on top of :func:`build_base_child_env` (which
    strips Claude Code session vars while preserving the telemetry master switch,
    drops the sensitive credentials named by
    :mod:`osprey.utils.sensitive_env`, resolves the auth-token conflict, and
    augments ``PATH``): sets the terminal type variables, then applies any
    caller-supplied ``extra_env`` last.

    This is the one launch path where the credential deny step is real removal
    rather than a no-op: the result becomes the PTY child's *complete*
    ``env=``, so a dropped name is gone from the agent session and from the MCP
    servers it spawns. The SDK paths overlay their dict onto ``os.environ``
    instead and get no such guarantee — see
    :mod:`osprey.agent_runner.clean_env` for which names that leaves open where.

    ``extra_env`` is applied last, after the strip, and is therefore the seam
    through which a caller can deliberately re-introduce a credential the base
    helper removed. That is intentional — a launcher that has decided a
    particular child may hold a particular token says so here — but it means
    the deny step is a default, not an invariant: read the caller's
    ``extra_env`` before concluding a name cannot reach the child. Today the
    real caller,
    :func:`osprey.interfaces.web_terminal.routes.websocket._build_extra_env`,
    re-introduces exactly one: ``OSPREY_PANEL_TOKEN``, the panel-tier-only
    credential the agent's panel tools and hooks would otherwise be answered
    401 for. It never re-introduces the operator secret.

    Args:
        extra_env: Additional environment variables to overlay last. Wins over
            everything the base helper resolved, including its credential strip.

    Returns:
        The fully resolved environment dict for the child process.
    """
    env = build_base_child_env()

    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"

    if extra_env:
        env.update(extra_env)

    return env


#: Environment names deliberately EXCLUDED from the pool env fingerprint.
#:
#: The fingerprint decides whether a warm pooled PTY may be reattached or has
#: to be killed and respawned, so its scope is a safety/liveness trade-off and
#: is set here as a *deny* list rather than an allow list: every name a caller
#: passes counts unless it is named below. A privilege-bearing variable added
#: later is therefore covered by default — the worst a name nobody thought
#: about can do is force a respawn, never let a stale child outlive the
#: privilege change that was supposed to reach it.
#:
#: **It is no longer the session posture's backstop, on purpose.** A
#: per-target posture now lands in the store and is read at write time, so
#: nothing about it belongs in a spawn env — do not re-add the stamp. What
#: this list still protects is every OTHER env change: a deployment-wide
#: readonly marker, a rotated panel token, a later privilege name.
#:
#: The exclusions are the names that legitimately differ between two
#: connections to the *same pool key*, as built by
#: :func:`osprey.interfaces.web_terminal.routes.websocket._build_extra_env`:
#:
#: * ``OSPREY_SESSION_ID`` — set only when the handler knows the Claude session
#:   id, and then equal to the pool key itself. A session spawned fresh (id
#:   dictated on the CLI, ``claude_session_id`` still ``None``) carries no such
#:   variable; switching back to that same session later resumes it by id and
#:   does. Absent-or-equal-to-the-key: it names the session the pool already
#:   keyed on, so it carries no privilege the key does not.
#: * ``OSPREY_TELEMETRY_SESSION_ID`` — same shape. The spawn call site passes a
#:   telemetry id, the ``switch_session`` call site does not, and when present
#:   it is also the pool key.
#: * ``OSPREY_TELEMETRY_SESSION_START`` — a wall-clock timestamp, minted anew on
#:   every connection. Fingerprinting it would respawn every reattach.
#: * ``OSPREY_POSTURE_SESSION`` — the posture-store key the child's posture was
#:   read under, and by construction the pool key itself (``_build_extra_env``
#:   computes ``claude_session_id or telemetry_session_id``, the same
#:   expression the handler keys the pool on). Absent-or-equal-to-the-key, the
#:   same shape as ``OSPREY_SESSION_ID``: it names the session the pool already
#:   keyed on and carries no privilege the key does not. Fingerprinting it
#:   would kill a live child on rekey, when a session's key moves from its
#:   telemetry id to its discovered Claude UUID. Its companion
#:   ``OSPREY_POSTURE_SOURCE`` is deliberately *not* excluded — it is constant
#:   (``live``) on this seam, so it never forces a respawn, and leaving it in
#:   keeps the deny list to names that provably differ per connection.
#:
#: Fingerprinting any of these would make a mere reconnect or tab-switch kill a
#: running agent session, which is the liveness half of this contract.
POOL_FINGERPRINT_EXCLUDED_ENV = frozenset(
    {
        "OSPREY_SESSION_ID",
        "OSPREY_TELEMETRY_SESSION_ID",
        "OSPREY_TELEMETRY_SESSION_START",
        "OSPREY_POSTURE_SESSION",
    }
)


def env_fingerprint(extra_env: dict[str, str] | None = None) -> str:
    """Fingerprint the spawn-relevant part of a child's ``extra_env``.

    Two calls that would produce the same child environment — up to the
    per-connection names in :data:`POOL_FINGERPRINT_EXCLUDED_ENV` — produce the
    same fingerprint. :meth:`PtyRegistry.get_or_create_session` compares the
    caller's fingerprint against the one recorded when the pooled child was
    spawned, and respawns on a mismatch.

    The digest, rather than the dict, is what the registry keeps: ``extra_env``
    carries the panel token, and a hash cannot be logged, repr'd into a
    traceback, or dumped by a debugger as a live credential.

    Args:
        extra_env: The environment overlay a caller would hand to
            :meth:`PtySession.start`. ``None`` and ``{}`` fingerprint alike.

    Returns:
        A hex SHA-256 digest of the name/value pairs that matter, sorted.
    """
    payload = "\x00".join(
        f"{name}\x1f{value}"
        for name, value in sorted((extra_env or {}).items())
        if name not in POOL_FINGERPRINT_EXCLUDED_ENV
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Fingerprint of a child spawned with no ``extra_env`` overlay at all.
EMPTY_ENV_FINGERPRINT = env_fingerprint(None)


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
        cwd: str | None = None,
    ) -> None:
        """Spawn the shell process attached to a new PTY.

        Args:
            initial_rows: Initial terminal row count (default 24).
            initial_cols: Initial terminal column count (default 80).
            extra_env: Additional environment variables to set in the child process.
            cwd: Working directory for the child process. When set, the spawned
                process runs in this directory so Claude Code resolves
                ``.mcp.json`` (and config/.env) relative to the project rather
                than the launch directory (issue #313). When ``None`` the child
                inherits the parent's cwd.
        """
        master_fd, slave_fd = pty.openpty()

        # Set initial terminal size BEFORE spawning — a 0x0 PTY causes
        # many TUI programs (including Claude Code) to exit immediately.
        winsize = struct.pack("HHHH", initial_rows, initial_cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

        # Build a clean environment for the child process.
        env = build_pty_env(extra_env)

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
            cwd=cwd,
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
        # Close master fd FIRST — the kernel sends SIGHUP to the entire
        # session (all process groups under this session leader), which is the
        # standard Unix mechanism for cleaning up terminal sessions.  Shells
        # handle SIGHUP by terminating their children.
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        if self._process is not None:
            # Give the SIGHUP from PTY close a moment to propagate.
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

            if self._process.poll() is None:
                # Still alive — send SIGTERM to the process group.
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Last resort — SIGKILL.
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

    @property
    def pid(self) -> int | None:
        """The PTY child's process id, or ``None`` before it is started.

        Read-only and public because one thing outside this class legitimately
        needs it: the control-target chip in the header asks which control-system target the session
        is on, and the controls MCP server publishes that against the pid chain
        of the Claude Code process running inside this PTY. That pid is the only
        handle the web server has on the session's process tree.
        """
        if self._process is None:
            return None
        return self._process.pid

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
        # Fingerprint of the extra_env each pooled child was spawned with, kept
        # in lockstep with _sessions. Read by get_or_create_session to decide
        # whether a warm entry may be reattached; see env_fingerprint().
        self._env_fingerprints: dict[str, str] = {}
        # Current pool key -> the key that child was SPAWNED under, recorded
        # only where a rekey has moved a live session off its spawn key. An
        # absent key resolves to itself; see audit_session_key().
        self._audit_keys: dict[str, str] = {}
        self._max_background = max_background

    # ---- Pool methods ---- #

    def get_or_create_session(
        self,
        session_key: str,
        command: str | list[str],
        rows: int = 24,
        cols: int = 80,
        extra_env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[PtySession, bool]:
        """Get existing session or create a new one.

        A warm pooled child is reattached only when the caller's ``extra_env``
        fingerprints identically to the one it was spawned with. A child's
        environment is fixed at ``execvp`` time and cannot be amended
        afterwards, so an env change that matters can only be delivered by
        killing the child and spawning a new one. Reusing the warm entry after
        such a change would leave the server believing it had launched a child
        under an environment that child never saw, which is precisely the state
        this comparison exists to make unreachable.

        The session's write posture is **not** among those changes any more:
        it is read live from the posture store at write time, so a flip reaches
        a running agent without a respawn (see
        :data:`POOL_FINGERPRINT_EXCLUDED_ENV`). What is left is a stale warm
        entry, a rotated credential, or a caller that changes the launch env
        without knowing it must terminate first. It fails towards a respawn,
        never towards a stale child.

        Only :data:`POOL_FINGERPRINT_EXCLUDED_ENV` is ignored in that
        comparison — the names that legitimately differ between two connections
        to one session. Everything else counts, so a reconnect keeps its
        session alive while a privilege change never fails to reach the child.

        Args:
            cwd: Working directory for the spawned process (issue #313). Only
                used when a new session is created; reused live sessions keep
                the directory they were spawned in.

        Returns:
            (session, was_reused) — True if an existing live session was reattached.
        """
        fingerprint = env_fingerprint(extra_env)
        existing = self._sessions.get(session_key)
        if existing is not None:
            # An entry with no recorded fingerprint never came through this
            # registry's own spawn path (every insertion site records one), so
            # the only thing that can be assumed about its child is the base
            # environment — no overlay, and therefore no sandbox marker.
            recorded = self._env_fingerprints.get(session_key, EMPTY_ENV_FINGERPRINT)
            if existing.is_alive and recorded == fingerprint:
                # LRU bump — move to end
                self._sessions.move_to_end(session_key)
                existing.resize(rows, cols)
                return existing, True

            if existing.is_alive:
                # Launch env changed under a live child. Values are never
                # logged — extra_env carries the panel token.
                logger.info(
                    "Launch env changed for session %s — terminating the warm PTY "
                    "so the new environment reaches a fresh child",
                    session_key,
                )
                self.terminate_session(session_key)
            else:
                # Dead — remove silently, respawn below
                self._sessions.pop(session_key, None)
                self._env_fingerprints.pop(session_key, None)
                self._attached.discard(session_key)

        # Evict if at capacity
        self._evict_lru()

        session = self._spawn_session(command, rows, cols, extra_env, cwd)
        self._sessions[session_key] = session
        self._env_fingerprints[session_key] = fingerprint
        # A fresh child under this key exports this key as its own posture
        # marker, so any alias inherited from the child it replaces would
        # misfile every record the new one produces.
        self._audit_keys.pop(session_key, None)
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
        """Rename a session entry (e.g. after UUID discovery).

        A PTY spawns under the telemetry id and is renamed to the Claude UUID
        the moment discovery finds it. Two things move with it, in opposite
        directions:

        * **The env fingerprint.** It describes the child, not the key, and
          dropping it would leave the renamed session looking unrecorded —
          which the next :meth:`get_or_create_session` would read as "spawned
          with no overlay" and could hand back a sandboxed child to a caller
          asking for a writable one.
        * **The audit join key**, in the opposite direction. The live child's
          ``OSPREY_POSTURE_SESSION`` was fixed at ``execvp`` time and still
          names the spawn key; it is in
          :data:`POOL_FINGERPRINT_EXCLUDED_ENV` precisely so this rename does
          not kill the child to update it. An alias is recorded so a
          server-side emitter holding only the new key can resolve back to the
          key that child's own records carry — see :meth:`audit_session_key`.

        **The posture store is not touched here, on purpose.** A rekey fires
        moments after the spawn, before any entry for this session can exist,
        so there is nothing to move; and every entry the web server writes goes
        under *both* the current key and the spawn key
        (``routes.websocket.persist_or_raise``), which is what keeps the
        running child — reading the telemetry id it was spawned with — and a
        post-restart reattach under the Claude UUID on one narrowing. Moving an
        entry here would have to choose which of those two readers to take it
        away from.

        Args:
            old_key: The key the session is pooled under now.
            new_key: The key it moves to.
        """
        if old_key not in self._sessions:
            return
        session = self._sessions.pop(old_key)
        self._sessions[new_key] = session
        fingerprint = self._env_fingerprints.pop(old_key, None)
        if fingerprint is not None:
            self._env_fingerprints[new_key] = fingerprint
        if old_key in self._attached:
            self._attached.discard(old_key)
            self._attached.add(new_key)

        # Chained renames collapse to the FIRST key — that is the one the
        # running child exported. An alias that would name the key itself is
        # dropped rather than stored, so "absent" stays the single spelling of
        # "this key is its own spawn key" (a rekey back to the original, and a
        # same-key rekey, both land here).
        original = self._audit_keys.pop(old_key, old_key)
        if original == new_key:
            self._audit_keys.pop(new_key, None)
        else:
            self._audit_keys[new_key] = original

    def audit_session_key(self, session_key: str) -> str:
        """The posture-store key an audit record about *session_key* joins on.

        The seam between a server-side audit emitter — which knows a session
        only by its current pool key — and the key that session's own child
        stamps into every record it emits. They differ for exactly as long as
        one live child outlives a rekey: the child's
        ``OSPREY_POSTURE_SESSION`` cannot be rewritten without killing it, so
        the *server* does the resolving instead.

        A toggle event recorded under the current key would split one session
        into two unrelated actors in the ledger: the toggle under the Claude
        UUID, every tool call it governs under the telemetry id.

        The posture store reads it for the same reason, one step earlier:
        ``routes.websocket.persist_or_raise`` records every narrowing under
        this key as well as the current one, so the running child — which
        looks its posture up under the key it exported — finds the entry a
        route wrote under the Claude UUID it knows nothing about.

        Any server-side recorder that names a PTY session must pass the id it
        was handed through here before putting it in an envelope's ``session``
        field, or the join it writes will be to a key no child's records carry.
        ``HttpAuditMiddleware`` is not such a recorder: it files every
        ``http_mutation`` envelope with ``session: null``, because an HTTP
        request belongs to no session (it is stamped ``posture_source=app`` for
        the same reason).

        Returns:
            The original spawn key when this session has been rekeyed,
            otherwise *session_key* unchanged — which is the answer for every
            session that never moved, a resumed one and a chat key included.
        """
        return self._audit_keys.get(session_key, session_key)

    def _evict_lru(self) -> None:
        """Evict the oldest non-attached session if at capacity."""
        if len(self._sessions) < self._max_background:
            return
        # Find oldest non-attached
        for key in list(self._sessions):
            if key not in self._attached:
                evicted = self._sessions.pop(key)
                self._env_fingerprints.pop(key, None)
                self._audit_keys.pop(key, None)
                evicted.terminate()
                logger.info("Evicted LRU session %s", key)
                return

    def _spawn_session(
        self,
        command: str | list[str],
        rows: int,
        cols: int,
        extra_env: dict[str, str] | None,
        cwd: str | None = None,
    ) -> PtySession:
        """Create and start a new PtySession."""
        session = PtySession(command)
        session.start(initial_rows=rows, initial_cols=cols, extra_env=extra_env, cwd=cwd)
        return session

    # ---- Session methods (kept for operator sessions and tests) ---- #

    def create_session(
        self,
        session_id: str,
        shell_command: str | list[str],
        initial_rows: int = 24,
        initial_cols: int = 80,
        extra_env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> PtySession:
        """Create and start a new PTY session."""
        if session_id in self._sessions:
            self._sessions[session_id].terminate()

        session = PtySession(shell_command)
        session.start(
            initial_rows=initial_rows,
            initial_cols=initial_cols,
            extra_env=extra_env,
            cwd=cwd,
        )
        self._sessions[session_id] = session
        self._env_fingerprints[session_id] = env_fingerprint(extra_env)
        self._audit_keys.pop(session_id, None)
        return session

    def get_session(self, session_id: str) -> PtySession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def terminate_session(self, session_id: str) -> None:
        """Terminate and remove a session.

        Drops the entry's recorded env fingerprint and audit alias with it, so
        the key is fully forgotten and a later spawn under the same key records
        its own.
        """
        session = self._sessions.pop(session_id, None)
        self._env_fingerprints.pop(session_id, None)
        self._audit_keys.pop(session_id, None)
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
        self._env_fingerprints.clear()
        self._audit_keys.clear()
        self._attached.clear()
