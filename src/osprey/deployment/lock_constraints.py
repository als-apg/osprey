"""The checkout's ``uv.lock``, exported as a constraints file for builds from source.

A build made from an osprey source checkout — a ``--dev`` image, or a project
venv installing the editable checkout — installs every third-party package the
checkout's ``uv.lock`` holds at the version the lock holds. Without this the
installers resolve those packages fresh from the index on every build, so a
dev image (and every CI lane that builds one) tests the checkout against
whatever was newest that day rather than against the versions the repo's own
test suite ran with.

The lock reaches the installers as a pip/uv constraints file: ``uv export``
renders it as ``name==version ; marker`` lines, which pip and uv both read with
``-c`` (or ``PIP_CONSTRAINT`` / ``UV_CONSTRAINT``). A constraint only bounds a
package that something else asks for, so one file serves every image whatever
subset of the workspace it installs. The workspace's own distributions are left
out (``--no-emit-workspace``): those are the wheels built from the checkout, and
a pin on them would refuse the very build being installed.

Release builds never come through here; their images resolve from the index as
they always have.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: The constraints file a ``--dev`` build stages into every build context, the
#: same bytes in each. Its presence is what switches the Dockerfiles onto the
#: lock-driven install; ``OSPREY_DEV`` keeps its own meaning.
LOCK_CONSTRAINTS_FILENAME = "osprey-lock-constraints.txt"

#: ``uv export`` with every package the lock resolves for any extra and any
#: member, so one export constrains every image. ``--no-dev`` drops only the
#: dependency groups; the ``dev`` *extra* stays, harmlessly — a constraint
#: installs nothing. The header and the ``# via`` annotations are dropped so the
#: file changes only when a pin does: it is COPYed into the deps layer, and any
#: byte of it that moves rebuilds that layer.
EXPORT_ARGS: tuple[str, ...] = (
    "export",
    "--locked",
    "--no-hashes",
    "--no-dev",
    "--all-packages",
    "--all-extras",
    "--no-emit-workspace",
    "--no-header",
    "--no-annotate",
    "--quiet",
)

_EXPORT_TIMEOUT_S = 300

# One export per checkout per process: a deploy stages the file into many
# contexts, and they must all receive the same bytes. Reset alongside the
# dev-wheel memo (wheel_build._reset_wheel_build_cache).
_export_cache: dict[str, str] = {}


class LockExportError(Exception):
    """The checkout's lock could not be exported; carries a reason and a remedy."""

    def __init__(self, reason: str, remedy: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.remedy = remedy


def find_uv() -> str | None:
    """The uv a build installs with, found as ``osprey build`` finds it."""
    return os.environ.get("UV") or shutil.which("uv")


def checkout_lock_root(spec: str | os.PathLike[str]) -> Path | None:
    """The checkout root behind a local osprey install *spec*, if it holds a lock.

    *spec* is what :func:`osprey.cli.build_environment._resolve_osprey_spec`
    returns. A filesystem path naming a directory with both ``pyproject.toml``
    and ``uv.lock`` is a source checkout; anything else (a ``==`` pin, a
    ``git+`` URL, a source tree without a lock) is not, and gets ``None``.
    """
    text = os.fspath(spec)
    if not text.startswith((".", "~", "/")):
        return None
    root = Path(text).expanduser().resolve()
    if (root / "pyproject.toml").is_file() and (root / "uv.lock").is_file():
        return root
    return None


def reset_export_cache() -> None:
    """Forget every memoized export (tests, and the dev-wheel reset hook)."""
    _export_cache.clear()


def export_lock_constraints(checkout_root: Path) -> str:
    """The checkout's ``uv.lock`` as constraints text, exported once per process.

    :param checkout_root: The directory holding the workspace ``pyproject.toml``
        and ``uv.lock``.
    :raises LockExportError: If there is no ``uv.lock``, no ``uv`` to export
        it with, or the lock is out of date with ``pyproject.toml`` (``uv
        export --locked`` refuses a stale lock rather than re-resolving, which
        is the point: a re-resolve would be exactly the fresh-from-the-index
        install this file exists to prevent).
    """
    root = Path(checkout_root).resolve()
    key = str(root)
    if key in _export_cache:
        return _export_cache[key]
    if not (root / "uv.lock").is_file():
        raise LockExportError(
            f"the osprey checkout at {root} has no uv.lock",
            "A build from source installs third-party packages at the versions "
            "the checkout's uv.lock holds, and refuses to resolve them fresh.\n"
            "Create the lock with:  uv lock",
        )
    uv = find_uv()
    if uv is None:
        raise LockExportError(
            "uv is not installed, and it is what exports the checkout's uv.lock",
            "Install uv (https://docs.astral.sh/uv/), or set UV to its path.",
        )
    try:
        result = subprocess.run(
            [uv, *EXPORT_ARGS, "--project", str(root)],
            capture_output=True,
            text=True,
            timeout=_EXPORT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LockExportError(
            f"exporting {root / 'uv.lock'} with {uv} failed ({e})",
            "Check that uv runs: uv --version",
        ) from e
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        remedy = "Fix the error below, then re-run."
        if "--locked" in detail or "needs to be updated" in detail:
            remedy = (
                "The lock is out of date with pyproject.toml. Update it with "
                "`uv lock` (and commit it), then re-run."
            )
        raise LockExportError(
            f"uv could not export the checkout's uv.lock at {root}",
            f"{remedy}\n\n{detail}",
        )
    text = result.stdout
    if not text.endswith("\n"):
        text += "\n"
    _export_cache[key] = text
    return text
