"""Canonical list of environment variables agent-run code must never see.

Several places in Osprey hand a copy of the parent process's environment to
something an agent can influence: the PTY child that runs the agent session
(``agent_runner.clean_env``), and the subprocesses that execute
agent-generated Python (``mcp_server.python_executor.executor``,
``mcp_server.workspace.execution.sandbox_executor``,
``services.bluesky_bridge.plan_validation``). Each of those hand-offs must
drop the same set of credentials, for the same reason: possession of one of
these values lets code call a write-armed or session-authenticating endpoint
directly, from outside the tool layer whose in-tool re-checks are the actual
safety authority.

This module is the single place that set is written down. It deliberately
imports nothing from ``osprey`` — only the standard library — so that both
``agent_runner`` and ``mcp_server`` can depend on it without creating an
import cycle between those two packages.

It also holds the one primitive a process that *keeps* such a credential needs:
:func:`guard_process_memory`. Stripping a name from a child's environment
closes inheritance; it does nothing about the parent's own memory, which on
Linux a process running as the same uid can read through ``/proc/<pid>/mem``,
``/proc/<pid>/environ`` (the exec-time environment block, which outlives any
``os.environ.pop``) or ``ptrace``. The web server and the dispatch worker each
take their bearer out of the environment into a holder and call the guard
before they serve or spawn, so for what a holder keeps from the agent — the
operator secret, the event-dispatcher bearer, the worker bearer — the two
halves of the promise are made in the same place: the agent inherits nothing,
and it cannot read the holder back. The guard says nothing about the rest of
this list. The panel token reaches the agent's own tooling by design, and a
``*_LAUNCH_TOKEN`` stays in the environment of the server that injects it
upstream; those names are here so no spawn site passes them on to agent-run
code, not because any holder closes them.

Two match shapes are supported:

* :data:`SENSITIVE_ENV_EXACT` — full variable names, matched exactly.
* :data:`SENSITIVE_ENV_SUFFIXES` — name suffixes, so a future credential that
  follows an established naming convention (a second bridge's
  ``*_LAUNCH_TOKEN``, say) is covered the day it is introduced rather than
  the day someone remembers to add it here.

Both forms are case-sensitive, matching the case-sensitive environment
semantics of the POSIX platforms Osprey runs its sandboxes on.
"""

import ctypes
import os
import sys
from collections.abc import Mapping

#: Exact variable names to remove. ``OSPREY_TERMINAL_SECRET`` authenticates a
#: web-terminal session, ``OSPREY_PANEL_TOKEN`` authorizes panel-backend calls,
#: ``EVENT_DISPATCHER_TOKEN`` authenticates callers to the event-dispatcher API,
#: and ``DISPATCH_WORKER_TOKEN`` is that token's peer on the other leg — it
#: authenticates the dispatcher to a worker's ``/dispatch`` API, which is what
#: actually starts an agent run. The two are issued, rendered and deployed
#: together everywhere (``deployment.container_lifecycle``,
#: ``cli.templates.scaffolding``, every service compose template), so a set that
#: held one and not the other would be an oversight rather than a policy. Each
#: is a credential that would let agent code act as the server that launched it.
#:
#: ``OSPREY_AUDIT_IDENTITY`` must never be added here or to the suffixes below.
#: It is not a credential: it is how a child resolves the directory its own
#: records are filed under (see ``osprey_connectors.identity``), and stripping
#: it makes that child answer under a name no reader looks for rather than fail.
SENSITIVE_ENV_EXACT: tuple[str, ...] = (
    "OSPREY_TERMINAL_SECRET",
    "OSPREY_PANEL_TOKEN",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
)

#: Name suffixes to remove. ``*_LAUNCH_TOKEN`` arms a bridge's plan-launch
#: endpoint (e.g. ``BLUESKY_LAUNCH_TOKEN``); matching on the suffix means a
#: second bridge adopting the same convention is covered without a code change.
SENSITIVE_ENV_SUFFIXES: tuple[str, ...] = ("_LAUNCH_TOKEN",)


def is_sensitive(name: str) -> bool:
    """Return whether *name* is a credential agent-run code must not see.

    True when *name* appears in :data:`SENSITIVE_ENV_EXACT` or ends with one
    of :data:`SENSITIVE_ENV_SUFFIXES`. Exposed alongside
    :func:`strip_sensitive` for callers that decide one name at a time — for
    instance popping a single variable out of ``os.environ`` — so the match
    rule is not re-implemented per call site.
    """
    return name in SENSITIVE_ENV_EXACT or name.endswith(SENSITIVE_ENV_SUFFIXES)


def strip_sensitive(env: Mapping[str, str]) -> dict[str, str]:
    """Return a new dict holding every entry of *env* except the sensitive ones.

    *env* is never mutated, so a caller may pass ``os.environ`` itself and use
    the result as a child process's ``env=`` without disturbing its own
    process environment. Entries are dropped when :func:`is_sensitive` accepts
    their name; every other key keeps its original value.
    """
    return {key: value for key, value in env.items() if not is_sensitive(key)}


#: ``prctl(2)`` option that sets the process's dumpable flag; ``0`` makes the
#: process non-dumpable.
PR_SET_DUMPABLE = 4

#: ``prctl(2)`` option that reads the dumpable flag back, as the call's result.
PR_GET_DUMPABLE = 3


def _prctl(option: int, arg: int) -> int:
    """Call ``prctl(option, arg, 0, 0, 0)`` through libc and return its raw result.

    The seam :func:`guard_process_memory` goes through, so a test can stand in
    for the kernel without a second libc binding.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    return int(prctl(option, arg, 0, 0, 0))


def guard_process_memory() -> bool:
    """Make this process's memory unreadable to other processes of the same uid.

    On Linux this marks the process non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``).
    The kernel then owns ``/proc/<pid>/environ``, ``/proc/<pid>/mem``, the
    ``fd/`` directory and the other per-process files as root, and refuses
    ``ptrace`` from any caller that does not hold ``CAP_SYS_PTRACE`` in the
    process's user namespace — which a peer running as the same unprivileged
    uid does not. It is the same measure ``ssh-agent`` takes for the keys it
    holds, and it is what closes the half of the promise that stripping a
    child's environment cannot: ``os.environ.pop`` edits the interpreter's
    mapping, not the exec-time environment block the kernel recorded, and a
    secret popped into memory is still in memory.

    **Children are unaffected.** ``execve`` resets the flag, so every process
    this one spawns — the agent CLI, its MCP servers, a sandboxed run, a
    sidecar — starts dumpable and debuggable as before. The process keeps
    access to itself: ``/proc/self/fd``, ``exe`` and ``cwd`` still resolve for
    the owning task. What it gives up is core dumps and being attached to by a
    debugger of its own uid, which is the point.

    **Call it as soon as the credential is in hand, and before anything the
    credential must be kept from exists.** A holder pops its carriers out of
    ``os.environ`` first and calls this second, so an error here never leaves
    a carrier behind for a child to inherit. The window the order leaves is
    the one between this process's ``exec`` and this call — in OSPREY, the
    length of the CLI's own start-up — during which the exec-time block is
    readable to a same-uid peer. In the container shape no agent process
    exists yet in that window, because this process is what spawns it. A holder of
    ``CAP_SYS_PTRACE`` in the container's user namespace (root under
    ``--privileged`` or ``--cap-add SYS_PTRACE``, or container root in a
    userns-remapped engine) can still read the process; root in a default
    Docker container cannot.

    Returns:
        ``True`` when the flag was set. ``False`` where the platform has no
        such flag and no ``/proc`` to close — macOS and Windows expose no
        exec-time environment file, and a same-uid process cannot read another
        process's memory there without a privilege the agent does not have.

    Raises:
        RuntimeError: when the kernel refuses to set the flag. A holder that
            cannot keep its credential from a same-uid peer must not go on to
            serve as though it could, so this fails closed rather than logging
            and continuing.
    """
    if not sys.platform.startswith("linux"):
        return False
    if _prctl(PR_SET_DUMPABLE, 0) != 0:
        errno = ctypes.get_errno()
        reason = os.strerror(errno) if errno else "unknown error"
        raise RuntimeError(
            f"prctl(PR_SET_DUMPABLE, 0) failed: {reason}. Refusing to hold a credential "
            "this process cannot keep from another process of the same uid."
        )
    return True
