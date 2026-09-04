"""Writes the session binding a notebook kernel reads at start-up.

The document says which PTY session a kernel should join: the one **most
recently attached** in the browser. The web terminal is the only writer, and it
writes identity only — a session key, the PTY child's pid, and the shared
agent-data root — never a posture or a control target. Those are looked up
live by the kernel from the stores those two identifiers address, so a binding
cannot go stale into a *wrong* answer; the worst it can be is a session that no
longer exists, which the kernel resolves to "run sandboxed".

The path constant and the reader live in :mod:`osprey.jupyter_kernel`, which is
standard library only: a kernel process reads the binding without importing the
web terminal, and this module imports the constant from there so the path has a
single producer.

Chat-pool children are never bound. They have no PTY, so there is no pid to
name and nothing for a kernel to join.
"""

from __future__ import annotations

import logging
import os

from osprey.interfaces.web_terminal._json_store import read_json_object, write_json_atomic
from osprey.jupyter_kernel import binding_path

logger = logging.getLogger(__name__)

__all__ = ["clear_binding", "write_binding"]


def write_binding(
    shared_root: str | os.PathLike[str],
    session_id: str,
    pty_pid: int | None,
    agent_data_root: str,
) -> None:
    """Point the binding at *session_id*, replacing whatever it named before.

    Failure is swallowed: a binding that cannot be written costs the operator
    a notebook that runs sandboxed, and must never cost them the terminal
    attach that triggered the write.

    Args:
        shared_root: The shared agent-data root the binding lives under.
        session_id: The posture-store key the session's own child stamps —
            ``PtyRegistry.audit_session_key`` of the pool key, so a rekeyed
            session is named by the id its records carry.
        pty_pid: The PTY child's process id, or ``None`` if it has none yet.
        agent_data_root: The agent-data root the kernel should stamp, so it
            reads the same stores the terminal writes.
    """
    path = binding_path(shared_root)
    document = {
        "session_id": session_id,
        "pty_pid": pty_pid,
        "agent_data_root": agent_data_root,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, document)
    except OSError:
        logger.warning("Could not write the notebook session binding %s", path, exc_info=True)


def clear_binding(shared_root: str | os.PathLike[str], session_id: str) -> None:
    """Remove the binding, but only while it still names *session_id*.

    A terminated session that was superseded before it died must not take the
    live binding with it — the ownership check is what keeps a background
    session's eviction from unbinding the session the operator is looking at.

    Args:
        shared_root: The shared agent-data root the binding lives under.
        session_id: The session being torn down, in the same key space
            :func:`write_binding` recorded.
    """
    path = binding_path(shared_root)
    document = read_json_object(path)
    if document is None or document.get("session_id") != session_id:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove the notebook session binding %s", path, exc_info=True)
