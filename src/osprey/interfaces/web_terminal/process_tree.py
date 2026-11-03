"""The process tree of a PTY child, and how to end it.

A PTY child's *process tree* is every process descending from the child,
grouped by process group. The agent CLI runs each shell command in a process
group and session of its own, so ending the child's own group does not reach
them; :func:`tree_groups` finds every group a descendant belongs to and
:func:`end_groups` ends them. A chat child runs in the server's own process
group, where no group signal may be sent, so its descendants in that group are
listed by :func:`server_group_members` and ended one by one with
:func:`end_processes`.

The tree is read in-process with psutil, so a read never forks and has no
timeout: a busy host makes it slower, never unknown. A tree is read through
one reader, :func:`read_while_alive`, which raises
:class:`ProcessTableUnreadable` when it could not look, so no caller takes an
unknown tree for one that started nothing. The tree can only be read while
its root lives, so that is the only time the reader retries a failed read.
An ending never signals what it could not re-identify: what an unreadable
table hides is reported as unknown, never as gone and never as a survivor.

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

import contextlib
import os
import re
import signal
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from osprey.deployment import process_table
from osprey.deployment.process_table import ProcessRow, ProcessTableUnreadable
from osprey_connectors.logger import get_logger

logger = get_logger("process_tree")

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
# How long a read that decides what to look for, end or keep retries an
# unreadable table: a tree read while its root lives, and each read of an ending.
READ_WAIT_S = 5.0
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


def read_while_alive(alive: Callable[[], bool]) -> dict[int, ProcessRow]:
    """Return the process table, read while the root of a tree is alive.

    *alive* says whether the root still runs. A tree can only be read while it
    does, so every read, the first included, is preceded by that question, and
    a failed read is retried only while the answer is yes, for up to
    :data:`READ_WAIT_S` (:func:`~osprey.deployment.process_table.snapshot_within`).

    Raises:
        ProcessTableUnreadable: No read succeeded while the root was alive:
            what it started is unknown, never empty. Raised at once for a
            failure no retry can answer, and with :class:`ProcessLookupError`
            as the cause when the root had exited before a read was taken.
    """
    return process_table.snapshot_within(READ_WAIT_S, alive=alive)


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


_T = TypeVar("_T", ProcessGroup, ProcessRow)


@dataclass(frozen=True)
class Ending(Generic[_T]):
    """What an ending did about the groups or processes it was given.

    ``ended`` holds those that were signalled. ``survivors`` holds those a
    read of the process table still found after SIGKILL. ``unknown`` holds
    those that could not be re-identified because the table could not be
    read: they were not signalled again and may still be running. Unknown is
    never folded into survivors, since nobody looked.
    """

    ended: tuple[_T, ...] = ()
    survivors: tuple[_T, ...] = ()
    unknown: tuple[_T, ...] = ()


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


def tree_groups(root: int, rows: dict[int, ProcessRow]) -> list[ProcessGroup]:
    """Return every process group a process descending from *root* belongs to.

    *root* is included. The server's own group and groups 0 and 1 are never
    returned. A group's members are every process in it, including one already
    reparented out of the tree, since a signal to the group reaches it.
    Ordered by earliest member start, then by pgid: the pgid keeps groups whose
    starts are equal in the same order from one look to the next.

    *rows* is the table :func:`read_while_alive` read while *root* was alive:
    a process that put itself in a new session keeps its parent link only
    until its parent exits.
    """
    if root not in rows:
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


def started_groups(root: int, alive: Callable[[], bool]) -> list[ProcessGroup]:
    """Return the process groups *root*'s descendants split off for their commands.

    That is :func:`tree_groups` without *root*'s own group: the groups an
    agent runs its shell commands in. The table is read through
    :func:`read_while_alive`, with *alive* saying whether *root* still runs.
    Blocking (it reads the process table).

    Returns:
        The groups, empty when *root* started nothing or no longer exists.

    Raises:
        ProcessTableUnreadable: The table could not be read while *root* was
            alive: what it started is unknown.
    """
    try:
        own = os.getpgid(root)
    except OSError:
        return []
    rows = read_while_alive(alive)
    return [group for group in tree_groups(root, rows) if group.pgid != own]


def server_group_members(root: int, rows: dict[int, ProcessRow]) -> list[ProcessRow]:
    """Return *root*'s descendants that are in the server's own process group.

    A process in the server's own group cannot be ended through its group,
    which :func:`tree_groups` never returns and :func:`end_groups` never
    signals, so these are ended one by one with :func:`end_processes`.
    *root* itself and the server are never listed. Ordered by start time, then pid.
    *rows* is the table :func:`read_while_alive` read while *root* was alive,
    for the same reason as in :func:`tree_groups`.
    """
    if root not in rows:
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
    groups: Sequence[ProcessGroup], rows: dict[int, ProcessRow]
) -> list[ProcessGroup]:
    """Return the groups that still hold one of their snapshotted members.

    A member that has exited but is not yet reaped (a zombie, which a server
    running as PID 1 without an init leaves behind) no longer counts.

    *rows* is a table read after the groups were snapshotted; an ending reads
    it through :func:`~osprey.deployment.process_table.snapshot_within`.
    """
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
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group.pgid, signum)


def _wait_gone(
    items: list[_T],
    within: float,
    alive: Callable[[Sequence[_T], dict[int, ProcessRow]], list[_T]],
) -> tuple[list[_T], bool]:
    """Poll the table until *items* are gone or *within* seconds have passed.

    A failed poll keeps the last known set. When the wait ends on a failed
    poll with something left, one verdict read is taken, retried for up to
    :data:`READ_WAIT_S`.

    Returns:
        ``(remaining, read)``: what the latest successful read still found,
        and whether *remaining* comes from a read at all; when it does not,
        *remaining* could not be re-identified.
    """
    deadline = time.monotonic() + within
    read = True
    while items and time.monotonic() < deadline:
        time.sleep(_POLL_S)
        try:
            rows = process_table.snapshot()
        except ProcessTableUnreadable:
            read = False
            continue
        read = True
        items = alive(items, rows)
    if items and not read:
        try:
            rows = process_table.snapshot_within(READ_WAIT_S)
        except ProcessTableUnreadable:
            return items, False
        return alive(items, rows), True
    return items, read


def _end(
    items: Sequence[_T],
    alive: Callable[[Sequence[_T], dict[int, ProcessRow]], list[_T]],
    send: Callable[[Sequence[_T], int], None],
) -> Ending[_T]:
    """SIGTERM what a read re-identifies in *items*, then SIGKILL what a read still finds.

    Nothing a read did not re-identify is signalled; it is reported as unknown.
    """
    try:
        rows = process_table.snapshot_within(READ_WAIT_S)
    except ProcessTableUnreadable:
        return Ending(unknown=tuple(items))
    live = alive(items, rows)
    send(live, signal.SIGTERM)
    remaining, read = _wait_gone(list(live), END_TERM_WAIT_S, alive)
    if remaining and read:
        send(remaining, signal.SIGKILL)
        remaining, read = _wait_gone(remaining, END_KILL_WAIT_S, alive)
    if read:
        return Ending(ended=tuple(live), survivors=tuple(remaining))
    return Ending(ended=tuple(live), unknown=tuple(remaining))


def end_groups(groups: Sequence[ProcessGroup]) -> Ending[ProcessGroup]:
    """End *groups*: SIGTERM, then SIGKILL what remains. Never raises.

    Only groups a read of the process table re-identifies (they still hold a
    snapshotted member) are signalled, never the server's own group and never
    group 0 or 1. A group the table could not be read to re-identify is not
    signalled. Each read that decides a step is retried for up to
    :data:`READ_WAIT_S`.

    Returns:
        The :class:`Ending`: ``ended`` the groups that were signalled,
        ``survivors`` those a read still found after SIGKILL, and ``unknown``
        those that could not be re-identified because the table could not be
        read.
    """
    try:
        return _end(groups, still_running, _signal_groups)
    except Exception:  # teardown finishes whatever the process table reports
        logger.warning("Ending the PTY child's process groups failed", exc_info=True)
        return Ending(unknown=tuple(groups))


def _live_rows(rows: Sequence[ProcessRow], current: dict[int, ProcessRow]) -> list[ProcessRow]:
    """The *rows* whose process *current* still shows as the one snapshotted.

    That is the same pid, start time and group, not yet exited.
    """
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
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(row.pid, signum)


def _signallable_rows(
    rows: Sequence[ProcessRow], current: dict[int, ProcessRow]
) -> list[ProcessRow]:
    own = os.getpid()
    return [row for row in _live_rows(rows, current) if row.pid > 1 and row.pid != own]


def end_processes(rows: Sequence[ProcessRow]) -> Ending[ProcessRow]:
    """End the processes *rows* name, one by one: SIGTERM, then SIGKILL. Never raises.

    The per-process twin of :func:`end_groups`, with the same reads and waits.
    Only a process a read re-identifies (same pid, start time and group) is
    signalled, never the server and never pid 0 or 1.

    Returns:
        The :class:`Ending`: ``ended`` the processes that were signalled,
        ``survivors`` those a read still found after SIGKILL, and ``unknown``
        those that could not be re-identified because the table could not be
        read.
    """
    try:
        return _end(rows, _signallable_rows, _signal_processes)
    except Exception:  # teardown finishes whatever the process table reports
        logger.warning("Ending the agent's processes failed", exc_info=True)
        return Ending(unknown=tuple(rows))


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
