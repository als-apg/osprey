"""Resolve shell commands that may live outside the default PATH.

Non-login processes (e.g. ``osprey web`` started from a lifecycle hook)
often inherit a stripped PATH that excludes user-local bin directories.
This module provides helpers to find executables in well-known locations
and to augment the child environment so *their* subprocesses can too.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

#: Bin directories under the invoking user's home, relative to it.
_HOME_RELATIVE_BINS = (Path(".local") / "bin", Path(".cargo") / "bin")

#: Bin directories that exist independently of any account.
_SYSTEM_BINS = (Path("/usr/local/bin"),)


def _user_bin_candidates() -> list[Path]:
    """Well-known user-local bin directories, in search order.

    Built on call rather than at import, and tolerant of an account with no
    home: a uid with no passwd entry and no ``HOME`` — ordinary under a
    random-uid cluster policy — makes ``Path.home()`` raise, and doing that at
    import turned a shorter PATH into an ImportError in every module that
    imports this one at module scope. The same degrade-not-raise posture
    :func:`osprey.utils.identity.resolve_identity` takes.

    Returns:
        The home-relative directories followed by the system ones, or only the
        system ones when no home resolves.
    """
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return list(_SYSTEM_BINS)
    return [home / relative for relative in _HOME_RELATIVE_BINS] + list(_SYSTEM_BINS)


def user_bin_dirs() -> list[str]:
    """Return existing user-local bin directories not already on PATH."""
    current = set(os.environ.get("PATH", "").split(os.pathsep))
    return [str(d) for d in _user_bin_candidates() if d.is_dir() and str(d) not in current]


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
            f"Check the path, or set web_terminal.shell under `config:` in "
            f"profile.yml — a command name, an absolute path, or an argv list "
            f"whose first element is one of those — and run `osprey build`."
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
        f"({', '.join(str(d) for d in _user_bin_candidates())}). "
        f"Install it, or set web_terminal.shell under `config:` in profile.yml "
        f"to an absolute path — or to an argv list starting with one — and run "
        f"`osprey build`."
    )


def normalize_shell_command(value: str | list[str]) -> list[str]:
    """Normalize a configured shell command into argv, resolving only argv[0].

    ``web_terminal.shell`` is argv, and it may be written either way: a single
    string (``"claude"``, ``"/opt/harness/run --profile ops"``, quoting
    honoured by :func:`shlex.split`) or a YAML list
    (``["/opt/harness/run", "--profile", "ops"]``). Both reach the PTY as the
    same argv, so a facility whose harness needs arguments no longer has to
    hide them in a wrapper script.

    Only the first element is resolved to an absolute path — the arguments are
    the harness's own and are passed through untouched.

    Args:
        value: The configured command, as a string or an argv list.

    Returns:
        argv with an absolute executable at index 0.

    Raises:
        FileNotFoundError: If argv[0] cannot be found (see
            :func:`resolve_shell_command`).
        ValueError: If *value* is neither a string nor a list, is empty, or
            parses to no words at all.
    """
    # YAML admits shapes the key does not — a bare number, a mapping, a `shell:`
    # written with no value at all. Each names the key, like the empty case
    # below; the alternative is whatever shlex.split raises on a non-string.
    if isinstance(value, list):
        parts = [str(part) for part in value]
    elif isinstance(value, str):
        parts = shlex.split(value)
    else:
        raise ValueError(
            f"web_terminal.shell must be a command string or an argv list, not "
            f"{type(value).__name__}. Set it to a command, an absolute path, or "
            f"an argv list, or remove the key to use the default launcher."
        )
    if not parts:
        raise ValueError(
            "web_terminal.shell is empty. Set it to a command, an absolute path, "
            "or an argv list, or remove the key to use the default launcher."
        )
    return [resolve_shell_command(parts[0]), *parts[1:]]
