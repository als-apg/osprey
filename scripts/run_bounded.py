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
  ``stop_child``: SIGABRT first, so each Python process prints its stacks and
  the stop names what was holding the run; then, if any process of the group
  is still running after ``--dump`` seconds (default 5), SIGTERM; then
  SIGKILL to whatever is left after ``--grace`` seconds (default 15), and a
  last wait for the group to be gone. Every wait is on the whole group, not
  only the command's own process, the stop reaps the command, and the
  wrapper prints the stop's own account on stderr when the group was not
  proven gone. The wrapper exits 124, the coreutils ``timeout``
  convention, so a caller can tell a bound from a failure.
* SIGINT, SIGTERM and SIGHUP delivered to the wrapper are forwarded to the
  group. After a forwarded SIGINT the wrapper dies of SIGINT itself once the
  command has exited, so a calling shell stops on Ctrl-C as it would without
  the wrapper.
* Outside a timeout the group is never signalled: stragglers left by a command
  that exited on its own are not this wrapper's business.

The wrapper runs under the project environment (``uv run``, as every gate
script and CI step calls it) and takes its group stop from the product's
``osprey.deployment.subprocess_capture``. Importing the module installs no
signal handler; only ``main`` installs them.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import signal
import subprocess
import sys
from types import FrameType

from osprey.deployment.subprocess_capture import stop_child

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
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


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
        outcome = stop_child(child, dump=args.dump, grace=args.grace)
        if not outcome.gone:
            print(f"run_bounded: {outcome.describe()}", file=sys.stderr, flush=True)
        return TIMEOUT_EXIT

    if last_forwarded and last_forwarded[-1] == signal.SIGINT:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    return 128 - returncode if returncode < 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
