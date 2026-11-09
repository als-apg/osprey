"""A machine repository: one facility, versioned on its own, with a VA-only setup.

A machine repository holds a facility directory at ``data/facility`` and, beside
it, ``profile.yml`` — the VA-only setup it builds and serves on its own (the
file in this package, included verbatim by the docs). A deployment mounts the
repository (as a submodule or a nested clone) and names its facility directory
with ``machine: <mount>/data/facility``.

The facility is the bundled ``example`` facility without the deployment's
content directories (``knowledge/``, ``ariel/``, ``benchmarks/``), so it has
models, decks and scenarios. One scenario is marked ``start: true``: the
machine's start set.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import osprey
from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS
from osprey.utils.workspace import PROFILE_FILENAME

#: This package: the VA-only profile file sits here.
FIXTURE_DIR = Path(__file__).parent

#: The machine repository's own profile, the D14 VA-only setup.
PROFILE = FIXTURE_DIR / PROFILE_FILENAME

#: The bundled facility the fixture machine is made from.
EXAMPLE_FACILITY = Path(osprey.__file__).parent / "templates" / "facilities" / "example"

#: Where a machine repository keeps its facility, relative to its top.
FACILITY_PATH = "data/facility"

#: The scenario the fixture machine marks as its start set.
MARKED_SCENARIO = "rf-thermal"

#: The files a working copy holds at its top that are never part of the machine.
REPOSITORY_FILES = {
    "build/out.txt": "a previous build's output\n",
    "var/state.txt": "a deployment's state\n",
    ".env": "SECRET=machine-repository-secret\n",
}


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd*, failing loudly; local file transport is allowed."""
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def git_init(directory: Path) -> None:
    """``git init`` *directory* with a committer identity of its own."""
    directory.mkdir(parents=True, exist_ok=True)
    git("init", "-q", cwd=directory)
    git("config", "user.email", "test@example.com", cwd=directory)
    git("config", "user.name", "Test", cwd=directory)


def commit_all(directory: Path, message: str = "commit") -> None:
    """Stage and commit everything in *directory*."""
    git("add", "-A", cwd=directory)
    git("commit", "-qm", message, cwd=directory)


def add_repository_files(top: Path) -> None:
    """Write :data:`REPOSITORY_FILES` at *top*, untracked, as a working copy has them."""
    for rel, text in REPOSITORY_FILES.items():
        path = top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def make_machine_repository(top: Path) -> Path:
    """Materialize the fixture machine repository at *top* and commit it.

    The committed repository holds ``profile.yml`` and ``data/facility``; after
    the commit, ``build/``, ``var/`` and ``.env`` are written at its top, so the
    working copy carries the files a real one accumulates.

    Returns:
        *top*.
    """
    top.mkdir(parents=True)
    shutil.copy2(PROFILE, top / PROFILE_FILENAME)
    facility = top / FACILITY_PATH
    shutil.copytree(
        EXAMPLE_FACILITY,
        facility,
        ignore=lambda directory, names: (
            [name for name in names if name in DEPLOYMENT_CONTENT_DIRS]
            if Path(directory) == EXAMPLE_FACILITY
            else []
        ),
    )
    scenario = facility / "scenarios" / f"{MARKED_SCENARIO}.yaml"
    scenario.write_text(
        scenario.read_text(encoding="utf-8").rstrip("\n") + "\nstart: true\n", encoding="utf-8"
    )
    git_init(top)
    commit_all(top, "the machine")
    add_repository_files(top)
    return top
