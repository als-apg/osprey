#!/usr/bin/env python3
"""Run a command under a wall-clock bound that stops its whole process group.

Usage::

    run_bounded.py [--dump SECONDS] [--grace SECONDS] SECONDS -- COMMAND [ARG ...]

The command starts in a new process group, so everything it spawns (``uv``,
pytest, pytest-xdist workers) is reached by the bound, not only the direct
child. Contract:

* The command's exit code is passed through; a command killed by signal N
  exits ``128 + N``.
* The command runs with ``PYTHONFAULTHANDLER=1``, so every Python process in
  the group writes the stack of each of its threads to stderr when it receives
  a fatal signal. Only the fatal-signal handler is installed; nothing samples
  a live process.
* A command still running after ``SECONDS`` is stopped as a group by
  ``stop_group``: SIGABRT first, so each Python process prints its stacks and
  the stop names what was holding the run; then, if any process of the group
  is still running after ``--dump`` seconds (default 5), SIGTERM; then
  SIGKILL to whatever is left after ``--grace`` seconds (default 15). Both
  waits are on the whole group, not only the command's own process. The
  wrapper exits 124, the coreutils ``timeout`` convention, so a caller can
  tell a bound from a failure.
* SIGINT, SIGTERM and SIGHUP delivered to the wrapper are forwarded to the
  group. After a forwarded SIGINT the wrapper dies of SIGINT itself once the
  command has exited, so a calling shell stops on Ctrl-C as it would without
  the wrapper.
* Outside a timeout the group is never signalled: stragglers left by a command
  that exited on its own are not this wrapper's business.

Standard library only, because stock macOS ships no ``timeout`` and this runs
before any project dependency is known to be installed. Importing the module
installs no signal handler, so ``stop_group`` can be loaded by path elsewhere;
only ``main`` installs them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from types import FrameType

#: The exit status of a command stopped by its bound.
TIMEOUT_EXIT = 124

_FORWARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

#: How often a stopping group is checked for members still running.
_POLL_SECONDS = 0.05


def _positive_seconds(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text}")
    return value


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_bounded.py",
        description="Run COMMAND; stop its process group after SECONDS and exit 124.",
    )
    parser.add_argument("--dump", type=float, default=5.0, help="SIGABRT-to-SIGTERM wait")
    parser.add_argument("--grace", type=float, default=15.0, help="SIGTERM-to-SIGKILL wait")
    parser.add_argument("seconds", type=_positive_seconds, help="wall-clock bound")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _group_is_gone(pgid: int, wait: Callable[[float | None], object]) -> bool:
    """Whether every process of the group ``pgid`` has exited.

    The leader is reaped first: an unreaped leader stays in the group as a
    zombie, while members it leaves behind are reparented and reaped by init.
    """
    try:
        wait(0)
    except subprocess.TimeoutExpired:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_group(pgid: int, seconds: float, wait: Callable[[float | None], object]) -> bool:
    """Wait up to ``seconds`` for the group to exit; return whether it did."""
    deadline = time.monotonic() + seconds
    while not _group_is_gone(pgid, wait):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def stop_group(
    pgid: int,
    *,
    dump: float,
    grace: float,
    wait: Callable[[float | None], object],
) -> None:
    """Stop the process group ``pgid``, printing its Python stacks first.

    The group is sent SIGABRT, which makes every process with ``faulthandler``
    enabled write each thread's stack to its stderr before it dies; if any
    member is still running ``dump`` seconds later, the group is sent SIGTERM,
    and if any is still running ``grace`` seconds after that, SIGKILL. Each
    wait is on every member of the group, not only its leader: the leader is
    often a wrapper (``uv run``) that dies of SIGABRT at once while the Python
    processes beneath it are still writing.

    Args:
        pgid: The group to stop; its leader's pid.
        dump: How long the processes get to write their stacks.
        grace: How long SIGTERM gets before SIGKILL.
        wait: Waits for the group's leader for at most the given seconds
            (``None``: until it exits), raising ``subprocess.TimeoutExpired``
            when it is still running; normally ``Popen.wait``.
    """
    _signal_group(pgid, signal.SIGABRT)
    if _wait_group(pgid, dump, wait):
        return
    _signal_group(pgid, signal.SIGTERM)
    if not _wait_group(pgid, grace, wait):
        _signal_group(pgid, signal.SIGKILL)
    wait(None)


def main(argv: list[str] | None = None) -> int:
    """Run the command and return the wrapper's exit status."""
    args = _parse(sys.argv[1:] if argv is None else argv)
    last_forwarded: list[int] = []
    group: list[int] = []

    def forward(signum: int, _frame: FrameType | None) -> None:
        last_forwarded.append(signum)
        for pgid in group:
            _signal_group(pgid, signum)

    # Installed before the child starts, so no signal falls between the two.
    for sig in _FORWARDED:
        signal.signal(sig, forward)
    env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
    child = subprocess.Popen(args.command, process_group=0, env=env)
    group.append(child.pid)
    if last_forwarded:
        _signal_group(child.pid, last_forwarded[-1])

    try:
        returncode = child.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        print(
            f"run_bounded: still running after {args.seconds:g} s, stopping: "
            f"{shlex.join(args.command)}",
            file=sys.stderr,
            flush=True,
        )
        stop_group(child.pid, dump=args.dump, grace=args.grace, wait=child.wait)
        return TIMEOUT_EXIT

    if last_forwarded and last_forwarded[-1] == signal.SIGINT:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    return 128 - returncode if returncode < 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
