"""Resolve shell commands that may live outside the default PATH.

Non-login processes (e.g. ``osprey web`` started from a lifecycle hook)
often inherit a stripped PATH that excludes user-local bin directories.
This module provides helpers to find executables in well-known locations
and to augment the child environment so *their* subprocesses can too.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Well-known user-local bin directories, checked in order.
_USER_BIN_CANDIDATES = [
    Path.home() / ".local" / "bin",
    Path.home() / ".cargo" / "bin",
    Path("/usr/local/bin"),
]


def user_bin_dirs() -> list[str]:
    """Return existing user-local bin directories not already on PATH."""
    current = set(os.environ.get("PATH", "").split(os.pathsep))
    return [
        str(d) for d in _USER_BIN_CANDIDATES if d.is_dir() and str(d) not in current
    ]


def resolve_shell_command(command: str) -> str:
    """Resolve a command name to an absolute path.

    Search order:
    1. If *command* is already an absolute path, validate it exists.
    2. Look up *command* on the current ``PATH`` via :func:`shutil.which`.
    3. Look up *command* on an augmented ``PATH`` that includes user-local
       bin directories.

    Returns the absolute path to the executable.

    Raises:
        FileNotFoundError: If the command cannot be found anywhere, with
            a message that includes install instructions and a config hint.
    """
    # Absolute path — just validate.
    if os.path.isabs(command):
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command
        raise FileNotFoundError(
            f"{command!r} does not exist or is not executable. "
            f"Check the path or set web_terminal.shell in config.yml."
        )

    # Normal PATH lookup.
    found = shutil.which(command)
    if found:
        return found

    # Augmented PATH lookup — add user-local bin dirs.
    extra = user_bin_dirs()
    if extra:
        augmented = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")
        found = shutil.which(command, path=augmented)
        if found:
            return found

    raise FileNotFoundError(
        f"{command!r} not found on PATH or in common install locations "
        f"({', '.join(str(d) for d in _USER_BIN_CANDIDATES)}). "
        f"Install it or set web_terminal.shell to an absolute path in config.yml."
    )
