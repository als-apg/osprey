#!/usr/bin/env python3
"""Find the Claude Code this tree pins and print the directory to put first on PATH.

Run under the project venv (``uv run python scripts/pinned_claude.py``): the pin
and the version probe are the product's own, so a gate and a headless run agree
on what "reports the pin" means.

    pinned_claude.py --pin   print the pinned version
    pinned_claude.py         print a directory whose ``claude`` reports the pin

Candidates, in order: every executable ``claude`` in each ``PATH`` directory, then
the native installer's per-version binary ``~/.local/share/claude/versions/<pin>``.
The first that reports exactly the pin wins. A winner named ``claude`` is printed
as its own directory; any other is reached through a ``claude`` symlink in
``${XDG_CACHE_HOME:-~/.cache}/osprey/claude/<pin>/``, keyed by the pin so it never
needs re-pointing.

Exit 1 when nothing reports the pin, with the product's own refusal on stderr
(it names the install command), or when a candidate does not answer
``--version`` in time, naming that binary.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from osprey.agent_runner.errors import AgentCliUnavailableError
from osprey.agent_runner.launcher import CLI_NAME, _probe_cli_version, agent_cli_path
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION


def _candidates(pin: str) -> Iterator[Path]:
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry) / CLI_NAME
        if path.is_file() and os.access(path, os.X_OK):
            yield path
    versioned = Path.home() / ".local" / "share" / "claude" / "versions" / pin
    if versioned.is_file() and os.access(versioned, os.X_OK):
        yield versioned


def _shim_dir(pin: str) -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "osprey" / "claude" / pin


def _directory_for(binary: Path, pin: str) -> Path:
    """Return a directory whose ``claude`` is *binary*."""
    binary = Path(os.path.abspath(binary))
    if binary.name == CLI_NAME:
        return binary.parent
    shim_dir = _shim_dir(pin)
    shim_dir.mkdir(parents=True, exist_ok=True)
    temp = shim_dir / f".{CLI_NAME}.{os.getpid()}"
    with contextlib.suppress(FileNotFoundError):
        temp.unlink()
    temp.symlink_to(binary)
    os.replace(temp, shim_dir / CLI_NAME)
    return shim_dir


def resolve(pin: str) -> int:
    for candidate in _candidates(pin):
        try:
            reported = _probe_cli_version(str(candidate))
        except subprocess.TimeoutExpired:
            print(
                f"`{candidate} --version` did not answer in time, so it could not be "
                f"checked against the pinned Claude Code {pin}.",
                file=sys.stderr,
            )
            return 1
        if reported == pin:
            print(_directory_for(candidate, pin))
            return 0
    try:
        found = agent_cli_path({"cli_version": pin})
    except AgentCliUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1
    # The product's own lookup also searches the user-local bin directories.
    print(_directory_for(Path(found), pin))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="print the pinned version")
    args = parser.parse_args(argv)
    if args.pin:
        print(_DEFAULT_CLAUDE_CLI_VERSION)
        return 0
    return resolve(_DEFAULT_CLAUDE_CLI_VERSION)


if __name__ == "__main__":
    sys.exit(main())
