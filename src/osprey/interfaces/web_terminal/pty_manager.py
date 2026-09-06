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
#: * ``OSPREY_SESSION_ID`` — stamped on every spawn path, always equal to the
#:   pool key (``claude_session_id or telemetry_session_id``): it names the
#:   session the pool already keyed on, so it carries no privilege the key
#:   does not.
#: * ``OSPREY_TELEMETRY_SESSION_ID`` — same shape. The spawn call site passes a
#:   telemetry id, the ``switch_session`` call site does not, and when present
#:   it is also the pool key.
#: * ``OSPREY_TELEMETRY_SESSION_START`` — a wall-clock timestamp, minted anew on
#:   every connection. Fingerprinting it would respawn every reattach.
#: * ``OSPREY_POSTURE_SESSION`` — the audit session id the child's posture was
#:   read under, and by construction the pool key itself (``_build_extra_env``
#:   computes ``claude_session_id or telemetry_session_id``, the same
#:   expression the handler keys the pool on). Absent-or-equal-to-the-key, the
#:   same shape as ``OSPREY_SESSION_ID``: it names the session the pool already
#:   keyed on and carries no privilege the key does not. Its companion
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
        """Terminate the subprocess and close the PTY.

        Blocking, and best-effort: it hangs up the terminal, then escalates
        SIGTERM to SIGKILL, waiting between steps, so it can occupy the
        calling thread for about seven seconds. A caller that must not block
        that long runs it in a worker thread.

        It may also return with the child still running — the SIGKILL wait can
        expire, which is logged and then let go. Returning is therefore not
        proof of death: :attr:`is_alive` is, and it is the probe anything that
        needs to *know* the child is gone must poll.
        """
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
        needs it: the control-target chip in the header asks which
        control-system target the deployment is on, and the controls MCP
        server publishes that against the pid chain
        of the Claude Code process running inside this PTY. That pid is the only
        handle the web server has on the session's process tree.
        """
        if self._process is None:
            return None
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        """Whether the subprocess is still running.

        The public death probe: :meth:`terminate` can return with the child
        alive, so a caller that must observe the process gone polls this.
        """
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
        # Pool key -> the token of the caller currently consuming it. The
        # token, not the key, is what identifies an attachment: two callers
        # can meet on one key, and only the one holding the token may release
        # it. See attach_session().
        self._attached: dict[str, object] = {}
        # Pool keys a hand-off in flight will re-fill. Unlike an attachment a
        # reservation names no owner and grants no exclusivity; it only keeps
        # the eviction pass off a key whose consumer has left and whose next
        # consumer has not arrived. See reserve().
        self._reserved: set[str] = set()
        # Fingerprint of the extra_env each pooled child was spawned with, kept
        # in lockstep with _sessions. Read by get_or_create_session to decide
        # whether a warm entry may be reattached; see env_fingerprint().
        self._env_fingerprints: dict[str, str] = {}
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
                self._attached.pop(session_key, None)

        # Evict if at capacity
        self._evict_lru()

        session = self._spawn_session(command, rows, cols, extra_env, cwd)
        self._sessions[session_key] = session
        self._env_fingerprints[session_key] = fingerprint
        return session, False

    def attach_session(self, session_key: str, owner: object) -> bool:
        """Mark a pooled session as actively consumed by one caller.

        One consumer per key: two readers on a single PTY file descriptor
        split the child's output between them, so a key that is already
        attached is refused rather than shared. The caller that wins holds the
        attachment until it releases it with the same token.

        Args:
            session_key: The pool key to attach.
            owner: A token identifying the caller — any object, compared by
                identity. Only the holder of this token can detach the key
                again, which is what keeps a departing caller from releasing
                an attachment a newer one has since taken over.

        Returns:
            True when the attachment was taken. False when the key is not in
            the pool, or when it is already attached — by another caller or by
            this one.
        """
        if session_key not in self._sessions:
            return False
        if session_key in self._attached:
            return False
        self._attached[session_key] = owner
        return True

    def detach_session(self, session_key: str, owner: object) -> None:
        """Release an attachment without terminating the session.

        LRU-bumps the session so it's less likely to be evicted.

        A detach from anyone but the current holder is a no-op — including a
        detach of a key that is not attached at all. A caller whose session
        went away and was replaced under the same key therefore tears its own
        state down without clearing the attachment the replacement holds.

        Args:
            session_key: The pool key to release.
            owner: The token :meth:`attach_session` was given for this key.
        """
        if session_key not in self._attached or self._attached[session_key] is not owner:
            return
        del self._attached[session_key]
        if session_key in self._sessions:
            self._sessions.move_to_end(session_key)

    def is_attached(self, session_key: str) -> bool:
        """Whether some caller currently holds *session_key*.

        The authoritative answer, and the one to ask before evicting or
        popping a key: :meth:`attached_owner` cannot tell an unattached key
        from one attached with a falsy token.
        """
        return session_key in self._attached

    def reserve(self, session_key: str) -> None:
        """Hold *session_key* against eviction while a hand-off is in flight.

        A hand-off leaves its key attached to nobody: the outgoing surface
        releases the key before the incoming one takes it, and in that gap a
        pooled entry looks exactly like the cold background session the
        eviction pass exists to reclaim. A reservation says the gap is
        deliberate and a consumer is on its way, so :meth:`_evict_lru` steps
        over the key the way it steps over an attached one.

        A reservation is not an attachment. It carries no owner token, grants
        no exclusive right to read the child, and is taken and dropped by the
        same hand-off, so reserving a key twice is reserving it once. A key
        that holds no session may be reserved — a hand-off reserves before it
        pops, and the entry it protects may not exist yet.

        Every reservation must be released in a ``finally``: a key left
        reserved is a key the pool can never reclaim.
        """
        self._reserved.add(session_key)

    def unreserve(self, session_key: str) -> None:
        """Release a hand-off reservation.

        A no-op for a key that holds none, so a ``finally`` can call it
        without knowing whether :meth:`reserve` was reached.
        """
        self._reserved.discard(session_key)

    def is_reserved(self, session_key: str) -> bool:
        """Whether a hand-off currently holds *session_key*."""
        return session_key in self._reserved

    def attached_owner(self, session_key: str) -> object | None:
        """The token currently holding *session_key*, or None if it is free."""
        return self._attached.get(session_key)

    def pop_lru_victim(self) -> PtySession | None:
        """Remove and return the session a spawn at capacity would evict, unkilled.

        The selection half of :meth:`_evict_lru`, for a caller on an event
        loop that is about to call :meth:`get_or_create_session` and cannot
        afford the blocking kill that eviction performs there: it takes the
        victim out of the pool here, kills it wherever it likes, and the spawn
        that follows finds room. Nothing is popped below capacity.

        A key is held either by a consumer reading its child
        (:meth:`attach_session`) or by a hand-off about to re-fill it
        (:meth:`reserve`). Both are stepped over, so a pool whose every entry
        is held grows past ``max_background`` rather than killing a session
        somebody is using — capacity is a target, not a guarantee.

        Returns:
            The oldest unheld session, forgotten by the pool exactly as
            :meth:`pop_session` forgets one, or None when the pool is below
            capacity or every entry is held.
        """
        if len(self._sessions) < self._max_background:
            return None
        for key in list(self._sessions):
            if not self.is_attached(key) and not self.is_reserved(key):
                logger.info("Evicted LRU session %s", key)
                return self.pop_session(key)
        return None

    def _evict_lru(self) -> None:
        """Evict the oldest unheld session if at capacity (see :meth:`pop_lru_victim`)."""
        victim = self.pop_lru_victim()
        if victim is not None:
            victim.terminate()

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
        return session

    def get_session(self, session_id: str) -> PtySession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def pop_session(self, session_id: str) -> PtySession | None:
        """Remove a session from the pool and hand it to the caller, unkilled.

        Everything :meth:`terminate_session` does *except* the kill. The pool
        entry goes, and with it the recorded env fingerprint, the audit alias
        and any attachment, so the key is fully forgotten and a later spawn
        under it records its own.

        The split exists because the two halves belong on different threads.
        The bookkeeping is a handful of dict operations and is safe on an
        event loop; :meth:`PtySession.terminate` blocks for seconds waiting on
        signals and is not. Once the entry is out of the pool no new consumer
        can reach the child, so the caller is free to kill it wherever it
        likes — and must, since nothing else holds a reference any more.

        Returns:
            The removed session, or None when *session_id* names no pooled
            session. The bookkeeping runs either way, and for an unknown key
            is a no-op.
        """
        session = self._sessions.pop(session_id, None)
        self._env_fingerprints.pop(session_id, None)
        self._attached.pop(session_id, None)
        return session

    def reinsert(self, session_id: str, session: PtySession) -> bool:
        """Put a session :meth:`pop_session` removed back into the pool, unattached.

        The path for a kill that did not take: a hand-off pops the entry,
        terminates the child, and finds it still alive after the death wait.
        Dropping the object then would orphan a running process nothing can
        reach; parking it in a side list would give the pool a second
        registry to drift from. Putting it back lets the next acquire meet it
        as an ordinary holder and run the kill again.

        No launch fingerprint is recorded — the pop dropped it and a survivor
        of a kill is not a child to reattach warm — and no attachment or
        reservation is taken, so the entry is exactly as evictable as any
        other background session.

        Returns:
            True when the session was put back. False, with the pool left
            alone, when *session_id* is occupied: something else filled the
            key in the meantime and the survivor is the caller's to deal with.
        """
        if session_id in self._sessions:
            return False
        self._sessions[session_id] = session
        return True

    def terminate_session(self, session_id: str) -> None:
        """Terminate and remove a session.

        The pool bookkeeping is :meth:`pop_session`; the kill that follows
        blocks the calling thread for as long as the child takes to die, and
        may return with it still alive (see :meth:`PtySession.terminate`). A
        caller on an event loop that cannot afford either uses the two halves
        separately.
        """
        session = self.pop_session(session_id)
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
        self._env_fingerprints.clear()
        self._attached.clear()
        self._reserved.clear()
