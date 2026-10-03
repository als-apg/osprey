"""The process tree of a PTY child, and how to end it.

A PTY child's *process tree* is every process descending from the child,
grouped by process group. The agent CLI runs each shell command in a process
group and session of its own, so ending the child's own group does not reach
them; :func:`tree_groups` finds every group a descendant belongs to and
:func:`end_groups` ends them. A chat child runs in the server's own process
group, where no group signal may be sent, so its descendants in that group are
listed by :func:`server_group_members` and ended one by one with
:func:`end_processes`.

The tree is read from ``ps``: macOS has no ``/proc`` and psutil is not a
dependency, while ``ps`` is present on every host the server runs on. Without
``ps``, or when ``ps`` fails, nothing is found, so ``PtySession.terminate``
reaches only the child's own group.

A group is the one a snapshot saw only while at least one of its snapshotted
members, with the same pid, the same start time and the same group, is still
in it. A process group id the kernel has since reused is therefore never
signalled as the group that was seen.

A group is named by what the agent launched: its leader, which is the shell
wrapper the agent CLI starts a command through, reduced to the command line
the wrapper was given (:func:`agent_command`). The name never comes from a
process the command started in turn, so a loop is not named after whichever
of its children is alive at the moment of a look, and the same command reads
the same on every look while the group lives.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from osprey.utils.logger import get_logger

logger = get_logger("process_tree")

# Every process with its parent, group, state, full start time and command line, no header.
_PS_COMMAND = ["ps", "-A", "-o", "pid=,ppid=,pgid=,stat=,lstart=,command="]
# Programs that are shells: the agent CLI starts every command through one of them.
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh"})
# The script the agent CLI's shell wrapper runs: setup pieces joined by ``&&``, then
# ``eval`` of the command as one word, then a record of the working directory.
_WRAPPER_EVAL = " && eval "
_WRAPPER_CWD = " && pwd -P >| "
_WRAPPER_STDIN = " < /dev/null"
# How long started processes get to exit on SIGTERM before SIGKILL.
END_TERM_WAIT_S = 3.0
# How long started processes get to disappear after SIGKILL.
END_KILL_WAIT_S = 2.0
# Interval between snapshots while waiting for groups to end.
_POLL_S = 0.05
# Longest label kept for a group.
_LABEL_MAX = 40
# Longest command line sent to the browser for a group.
_COMMAND_MAX = 160

# A file name with an extension, such as ``magnet_scan.py``.
_FILE_NAME = re.compile(r"^[\w.-]+\.[A-Za-z0-9]{1,5}$")
# What makes a command line shell syntax rather than one program and its arguments: a
# separator, a pipe, a redirection, a substitution, quoting, or a leading assignment.
_SHELL_SYNTAX = re.compile(r"[;|&<>()$`'\"\\]|^\S+=")


@dataclass(frozen=True)
class ProcessRow:
    """One process as ``ps`` reported it."""

    pid: int
    ppid: int
    pgid: int
    started: float  # epoch seconds
    command: str
    state: str  # ``ps`` state; a leading ``Z`` is an exited process nobody has reaped

    @property
    def exited(self) -> bool:
        return self.state.startswith("Z")


def snapshot() -> dict[int, ProcessRow] | None:
    """Return every process on the host by pid, or None when ``ps`` cannot be read."""
    try:
        result = subprocess.run(
            _PS_COMMAND,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Cannot read the process table: %s", exc)
        return None
    rows: dict[int, ProcessRow] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 9)
        if len(parts) != 10:
            continue
        try:
            pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
            started = time.mktime(time.strptime(" ".join(parts[4:9]), "%a %b %d %H:%M:%S %Y"))
        except (ValueError, OverflowError):
            continue
        rows[pid] = ProcessRow(pid, ppid, pgid, started, parts[9], parts[3])
    return rows


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _program(command: str) -> str:
    tokens = command.split()
    return os.path.basename(tokens[0]).lstrip("-") if tokens else ""


def _file_name(tokens: Sequence[str]) -> str | None:
    """The first argument after the program that is not an option and names a file."""
    for arg in tokens[1:]:
        if arg.startswith("-"):
            continue
        name = os.path.basename(arg)
        if _FILE_NAME.match(name):
            return name
    return None


def name_for(command: str) -> str:
    """Return the name an operator knows a command line by.

    That is the first argument after the program that is not an option and
    whose basename is a file name, else the program's basename::

        python /x/magnet_scan.py --sector 3  ->  magnet_scan.py
        uv run python scan.py                ->  scan.py
        sleep 60                             ->  sleep

    Never more than one file or program name: it carries none of the
    command's arguments, so it is what a log line names a process by.
    """
    tokens = command.split()
    if not tokens:
        return ""
    return _truncate(_file_name(tokens) or _program(command), _LABEL_MAX)


def label_for(command: str) -> str:
    """Return what the operator is shown a command line as.

    The file name the command runs when it runs one, else the program's
    basename when the command is one program and its arguments, else the
    command line itself — shell syntax such as a loop, a pipeline or an
    assignment, which no one program names — cut to a fixed length::

        python /x/magnet_scan.py --sector 3        ->  magnet_scan.py
        sleep 60                                   ->  sleep
        while true; do date >> /tmp/beat; done     ->  while true; do date >> /tmp/beat; done
        bash -c 'while true; do sleep 1; done'     ->  bash -c 'while true; do sleep 1; done'
    """
    tokens = command.split()
    if not tokens:
        return ""
    name = _file_name(tokens)
    if name is not None:
        return _truncate(name, _LABEL_MAX)
    if _SHELL_SYNTAX.search(command):
        return _truncate(command.strip(), _LABEL_MAX)
    return _truncate(_program(command), _LABEL_MAX)


def _unquote(word: str) -> str:
    """Return the text the wrapper's shell reads *word* as.

    The agent CLI quotes a command as one bare word or as one single-quoted
    word in which a single quote is written ``'"'"'``.
    """
    if not word.startswith("'"):
        return word
    body = word[1:-1] if len(word) > 1 and word.endswith("'") else word[1:]
    return body.replace("'\"'\"'", "'")


def agent_command(command: str) -> str | None:
    """Return the command line *command* runs through the agent CLI's shell wrapper.

    The agent CLI runs each command as ``<shell> -c <script>``: setup pieces
    joined by ``&&``, then ``eval`` of the command as one quoted word, with
    its standard input from ``/dev/null`` unless the command redirects it,
    then a record of the working directory. The command line is that word,
    unquoted. ``None`` when *command* is not such a wrapper.
    """
    parts = command.split(None, 2)
    if len(parts) != 3 or parts[1] != "-c" or os.path.basename(parts[0]) not in _SHELLS:
        return None
    script = parts[2]
    start = script.find(_WRAPPER_EVAL)
    end = script.rfind(_WRAPPER_CWD)
    if start < 0 or end < start:
        return None
    word = script[start + len(_WRAPPER_EVAL) : end].removesuffix(_WRAPPER_STDIN)
    return _unquote(word)


@dataclass(frozen=True)
class ProcessGroup:
    """A process group seen in a PTY child's tree.

    ``command`` is the command line the agent launched the group with, held
    whole so that a log line names the group by its file or program name
    (:func:`name_for`) whatever the path length, and ``label`` is what the
    operator is shown it as (:func:`label_for`). :meth:`to_json` cuts the
    command line to a fixed length for the browser.
    """

    pgid: int
    members: tuple[tuple[int, float], ...]  # (pid, started) of every member seen
    label: str
    command: str

    def to_json(self) -> dict[str, str]:
        return {"label": self.label, "command": _truncate(self.command, _COMMAND_MAX)}


def _group(pgid: int, rows: Sequence[ProcessRow]) -> ProcessGroup:
    """Return the group *rows* make up, named by what the agent launched.

    That is the group's leader, the process whose pid is the group id: the
    one the agent CLI started for the command. Once the leader has exited
    the earliest-started member stands in. A leader that is the agent CLI's
    shell wrapper is reduced to the command line it was given. A process the
    command started in turn never names the group.
    """
    ordered = sorted(rows, key=lambda row: (row.started, row.pid))
    leader = next((row for row in ordered if row.pid == pgid), ordered[0])
    command = agent_command(leader.command) or leader.command
    return ProcessGroup(
        pgid=pgid,
        members=tuple((row.pid, row.started) for row in ordered),
        label=label_for(command),
        command=command,
    )


def tree_groups(root: int) -> list[ProcessGroup]:
    """Return every process group a process descending from *root* belongs to.

    *root* is included. The server's own group and groups 0 and 1 are never
    returned. A group's members are every process in it, including one already
    reparented out of the tree, since a signal to the group reaches it.
    Ordered by earliest member start, then by pgid: ``ps`` reports start times
    in whole seconds, so groups started within one second tie on time, and the
    pgid keeps their order the same from one look to the next.

    Must run while *root* is alive: a process that put itself in a new session
    keeps its parent link only until its parent exits.
    """
    rows = snapshot()
    if rows is None or root not in rows:
        return []
    children: dict[int, list[int]] = {}
    for row in rows.values():
        if row.pid != row.ppid:
            children.setdefault(row.ppid, []).append(row.pid)
    seen = {root}
    queue = deque([root])
    while queue:
        for child in children.get(queue.popleft(), ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    own = os.getpgrp()
    pgids = {rows[pid].pgid for pid in seen} - {own}
    groups = [
        _group(pgid, [row for row in rows.values() if row.pgid == pgid])
        for pgid in pgids
        if pgid > 1
    ]
    return sorted(groups, key=lambda g: (min(started for _, started in g.members), g.pgid))


def started_groups(root: int) -> list[ProcessGroup]:
    """Return the process groups *root*'s descendants split off for their commands.

    That is :func:`tree_groups` without *root*'s own group: the groups an
    agent runs its shell commands in. Blocking (it runs ``ps``), and like
    :func:`tree_groups` it must run while *root* is alive.
    """
    try:
        own = os.getpgid(root)
    except OSError:
        return []
    return [group for group in tree_groups(root) if group.pgid != own]


def server_group_members(root: int) -> list[ProcessRow]:
    """Return *root*'s descendants that are in the server's own process group.

    A process in the server's own group cannot be ended through its group,
    which :func:`tree_groups` never returns and :func:`end_groups` never
    signals, so these are ended one by one with :func:`end_processes`.
    *root* itself and the server are never listed. Ordered by start time, then pid.
    Must run while *root* is alive, for the same reason as :func:`tree_groups`.
    """
    rows = snapshot()
    if rows is None or root not in rows:
        return []
    children: dict[int, list[int]] = {}
    for row in rows.values():
        if row.pid != row.ppid:
            children.setdefault(row.ppid, []).append(row.pid)
    seen = {root}
    queue = deque([root])
    while queue:
        for child in children.get(queue.popleft(), ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    own_group, own_pid = os.getpgrp(), os.getpid()
    members = [rows[pid] for pid in seen - {root} if rows[pid].pgid == own_group and pid != own_pid]
    return sorted(members, key=lambda row: (row.started, row.pid))


def still_running(
    groups: Sequence[ProcessGroup], rows: dict[int, ProcessRow] | None = None
) -> list[ProcessGroup]:
    """Return the groups that still hold one of their snapshotted members.

    A member that has exited but is not yet reaped (a zombie, which a server
    running as PID 1 without an init leaves behind) no longer counts.

    *rows* defaults to a fresh :func:`snapshot`. When the table cannot be
    read every group is returned: an unknown group is never claimed gone.
    """
    if rows is None:
        rows = snapshot()
        if rows is None:
            return list(groups)
    alive = []
    for group in groups:
        for pid, started in group.members:
            row = rows.get(pid)
            if (
                row is not None
                and not row.exited
                and row.started == started
                and row.pgid == group.pgid
            ):
                alive.append(group)
                break
    return alive


def _signal_groups(groups: Sequence[ProcessGroup], signum: int) -> None:
    own = os.getpgrp()
    for group in groups:
        if group.pgid <= 1 or group.pgid == own:
            continue
        try:
            os.killpg(group.pgid, signum)
        except (ProcessLookupError, PermissionError):
            pass


def _wait_gone(groups: list[ProcessGroup], within: float) -> list[ProcessGroup]:
    deadline = time.monotonic() + within
    while groups and time.monotonic() < deadline:
        time.sleep(_POLL_S)
        groups = still_running(groups)
    return groups


def end_groups(
    groups: Sequence[ProcessGroup],
) -> tuple[list[ProcessGroup], list[ProcessGroup]]:
    """End *groups*: SIGTERM, then SIGKILL what remains. Never raises.

    Only groups that still hold a snapshotted member are signalled, never the
    server's own group and never group 0 or 1.

    Returns:
        ``(ended, survivors)``: the groups that were signalled, and those the
        last look still found after SIGKILL.
    """
    try:
        live = still_running(groups)
        _signal_groups(live, signal.SIGTERM)
        remaining = _wait_gone(list(live), END_TERM_WAIT_S)
        if remaining:
            _signal_groups(remaining, signal.SIGKILL)
            remaining = _wait_gone(remaining, END_KILL_WAIT_S)
        return live, remaining
    except Exception:  # teardown finishes whatever the process table reports
        logger.warning("Ending the PTY child's process groups failed", exc_info=True)
        return [], list(groups)


def _live_rows(rows: Sequence[ProcessRow]) -> list[ProcessRow]:
    """The *rows* whose process is still the one snapshotted: same start time and group.

    Every row is returned when the table cannot be read: an unknown process is
    never claimed gone.
    """
    current = snapshot()
    if current is None:
        return list(rows)
    live = []
    for row in rows:
        now = current.get(row.pid)
        if (
            now is not None
            and not now.exited
            and now.started == row.started
            and now.pgid == row.pgid
        ):
            live.append(row)
    return live


def _signal_processes(rows: Sequence[ProcessRow], signum: int) -> None:
    own = os.getpid()
    for row in rows:
        if row.pid <= 1 or row.pid == own:
            continue
        try:
            os.kill(row.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass


def _wait_processes_gone(rows: list[ProcessRow], within: float) -> list[ProcessRow]:
    deadline = time.monotonic() + within
    while rows and time.monotonic() < deadline:
        time.sleep(_POLL_S)
        rows = _live_rows(rows)
    return rows


def end_processes(
    rows: Sequence[ProcessRow],
) -> tuple[list[ProcessRow], list[ProcessRow]]:
    """End the processes *rows* name, one by one: SIGTERM, then SIGKILL. Never raises.

    The per-process twin of :func:`end_groups`, with the same waits. Only a
    process still the one snapshotted (same pid, start time and group) is
    signalled, never the server and never pid 0 or 1.

    Returns:
        ``(ended, survivors)``: the processes that were signalled, and those
        the last look still found after SIGKILL.
    """
    try:
        live = [row for row in _live_rows(rows) if row.pid > 1 and row.pid != os.getpid()]
        _signal_processes(live, signal.SIGTERM)
        remaining = _wait_processes_gone(list(live), END_TERM_WAIT_S)
        if remaining:
            _signal_processes(remaining, signal.SIGKILL)
            remaining = _wait_processes_gone(remaining, END_KILL_WAIT_S)
        return live, remaining
    except Exception:  # teardown finishes whatever the process table reports
        logger.warning("Ending the agent's processes failed", exc_info=True)
        return [], list(rows)


def describe_processes(rows: Sequence[ProcessRow]) -> str:
    """Return one line naming the processes *rows* name.

    A process is named by :func:`name_for`, a file or program name and never
    the command line. Names come from processes the agent started and may
    carry any text, so they are logged through ``repr``.
    """
    return "; ".join(f"pid {row.pid} {name_for(row.command)!r}" for row in rows)


def describe(groups: Sequence[ProcessGroup]) -> str:
    """Return one line naming *groups*.

    A group is named by :func:`name_for` of its command, a file or program
    name and never the command line. Names come from processes the agent
    started and may carry any text, so they are logged through ``repr``.
    """
    return "; ".join(
        f"pgid {g.pgid} {name_for(g.command)!r} (pids {', '.join(str(p) for p, _ in g.members)})"
        for g in groups
    )
