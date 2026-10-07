#!/usr/bin/env python3
"""Run a command under a wall-clock bound that stops its whole process group.

Usage::

    run_bounded.py [--grace SECONDS] SECONDS -- COMMAND [ARG ...]

The command starts in a new process group, so everything it spawns (``uv``,
pytest, pytest-xdist workers) is reached by the bound, not only the direct
child. Contract:

* The command's exit code is passed through; a command killed by signal N
  exits ``128 + N``.
* A command still running after ``SECONDS`` is sent SIGTERM as a group, then
  SIGKILL after ``--grace`` seconds (default 15) if it is still running, and
  the wrapper exits 124, the coreutils ``timeout`` convention, so a caller can
  tell a bound from a failure.
* SIGINT, SIGTERM and SIGHUP delivered to the wrapper are forwarded to the
  group. After a forwarded SIGINT the wrapper dies of SIGINT itself once the
  command has exited, so a calling shell stops on Ctrl-C as it would without
  the wrapper.
* Outside a timeout the group is never signalled: stragglers left by a command
  that exited on its own are not this wrapper's business.

Standard library only, because stock macOS ships no ``timeout`` and this runs
before any project dependency is known to be installed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from types import FrameType

#: The exit status of a command stopped by its bound.
TIMEOUT_EXIT = 124

_FORWARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


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
    except ProcessLookupError:
        pass


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
    child = subprocess.Popen(args.command, process_group=0)
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
        _signal_group(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=args.grace)
        except subprocess.TimeoutExpired:
            _signal_group(child.pid, signal.SIGKILL)
            child.wait()
        return TIMEOUT_EXIT

    if last_forwarded and last_forwarded[-1] == signal.SIGINT:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    return 128 - returncode if returncode < 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
