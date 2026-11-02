"""What a checks module builds its records from.

A check is declared by the code that owns it, in a ``repo_checks.py`` module
beside that code, and ``scripts/check.py`` collects every such module. This
module is what they share: the records a check and a CI lane are declared
with, the facts of a run a check's hooks are handed, and the helpers that
start a process, read git and match a path against a scope.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIERS = ("quick", "full")
KINDS = ("invariant", "static", "tests", "build")

#: Exit statuses.
CLEAN, FAILED_EXIT, REFUSED_EXIT, ENVIRONMENT_EXIT = 0, 1, 2, 3

BOTH = frozenset(TIERS)
FULL = frozenset({"full"})

WHOLE, PATHS, NOT_TOUCHED = "whole", "paths", "not touched"

#: Every uv run a check or need starts carries --frozen: a plain uv run rewrites
#: a stale uv.lock, and the lockfile check must judge it as the tree left it.
UV_RUN = ("uv", "run", "--frozen")

#: Marks a field of a record that is behaviour, not declaration: ``list
#: --json`` prints the declaration and leaves the hooks out, and two records
#: that declare the same are equal.
HOOK = {"hook": True}


class Refused(Exception):
    """The command cannot judge; the message says what to do about it."""


class EnvironmentFault(Exception):
    """This machine cannot judge: a need raises it and the run stops there."""

    def __init__(self, probe: str, fact: str) -> None:
        super().__init__(f"{probe}: {fact}")
        self.probe = probe
        self.fact = fact


@dataclass(frozen=True)
class Selection:
    """What a tier does with one check: run it whole, on these paths, or not at all."""

    state: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Context:
    """The facts of one run that the checks share."""

    tier: str = "full"
    base: str | None = None
    shard: str | None = None
    #: Per check id; ``"*"`` holds a ``run`` caller's flags for every pytest check.
    exec_flags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The directory the runners write their reports into, one subdirectory per check.
    reports: str = "{reports}"
    #: The range ends at the working tree, not at HEAD.
    worktree: bool = False
    #: The checks were named by a ``run`` caller, so an install a tier does
    #: without is a refusal.
    named: bool = False


@dataclass(frozen=True)
class Install:
    """An install a check proves something only against.

    A tier does without it: when it is absent the check is named as not run
    here. A ``run`` caller asked for the proof, so there its absence is a
    refusal, and so is any install other than the proven one, everywhere.
    """

    #: Exit 0 when the proven install is there, ``ABSENT_EXIT`` when nothing
    #: is, anything else when another one is; the last line it prints says which.
    probe: tuple[str, ...]
    #: What to install, for a refusal.
    instruction: Callable[[], str]
    #: The verdict's line for a run against the proven install, given whether
    #: the check passed.
    attestation: Callable[[bool], str]


#: A probe's exit status when what it looks for is not installed at all.
ABSENT_EXIT = 2


@dataclass(frozen=True)
class Venue:
    """A platform a check can only run on."""

    #: Whether this machine is one.
    probe: Callable[[], bool]
    #: The requirement in words, for the line that names an unmet one.
    words: str


#: Picks a runner's tests from the changed paths the range still holds, given
#: the repository's file list and the scripts every check runs through.
Selector = Callable[[Sequence[str], Sequence[str], frozenset[str]], Selection]


@dataclass(frozen=True)
class Check:
    """One check: the one name every caller uses, the one spelling it runs, and
    everything the command does with it that differs from another check's."""

    id: str
    kind: str
    tiers: frozenset[str]
    #: The command; ``{base}`` is filled from ``--changed`` or ``--base``. Empty
    #: for a check implemented by its ``builtin``.
    argv: tuple[str, ...]
    #: Path globs that make the check relevant to a range; empty means always.
    scope: tuple[str, ...] = ()
    #: What must be present to judge; unmet is a refusal.
    needs: tuple[str, ...] = ()
    #: A check that judges a range declares how it judges the working tree: the
    #: command lines that replace ``argv`` under ``--worktree``. ``{base}`` is
    #: filled as in ``argv``; a line holding ``{untracked}`` is repeated once per
    #: untracked path and dropped when there is none.
    worktree: tuple[tuple[str, ...], ...] = ()
    #: A platform the check can only run on, a key of the collected venues;
    #: unmet is named, not failed.
    venue: str | None = None
    #: Wall-clock bound through ``run_bounded.py``; 0 means unbounded.
    bound_s: int = 0
    #: The machine-readable report the runner writes and the command reads back,
    #: a key of ``REPORT_FLAGS`` in ``scripts/check.py``; ``None`` for a check
    #: that writes none.
    report: str | None = None
    #: What ``--fix`` runs to repair what this check reports; it writes files.
    fix: tuple[tuple[str, ...], ...] = ()
    #: The needs that depend on what was selected, probed after ``needs``.
    selected_needs: Callable[[Selection, Context], tuple[str, ...]] | None = field(
        default=None, compare=False, metadata=HOOK
    )
    #: The directory the command runs in, relative to the repository root.
    workdir: str | None = field(default=None, compare=False, metadata=HOOK)
    #: A ``run`` caller may pass it execution flags.
    exec_flags: bool = field(default=False, compare=False, metadata=HOOK)
    #: The execution flags each tier passes it.
    tier_flags: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict, compare=False, metadata=HOOK
    )
    #: Appended to the command on GitHub Actions, so failures print as annotations.
    annotate: tuple[str, ...] = field(default=(), compare=False, metadata=HOOK)
    #: The runner's exit status for a selection whose tests the marker
    #: expression all deselects; such a run is named, not failed.
    empty_exit: int | None = field(default=None, compare=False, metadata=HOOK)
    #: The shards a ``run`` caller may split it into with ``--shard``.
    shards: tuple[str, ...] = field(default=(), compare=False, metadata=HOOK)
    #: The suffixes of the test files a caller may add to it after ``--``.
    named_suffixes: tuple[str, ...] = field(default=(), compare=False, metadata=HOOK)
    #: A caller may add a node id (``file::node``) of such a file after ``--``.
    named_node_ids: bool = field(default=False, compare=False, metadata=HOOK)
    #: Whether a file name is one of its test files; a caller may then add,
    #: after ``--``, a directory that holds one at any depth.
    named_dirs: Callable[[str], bool] | None = field(default=None, compare=False, metadata=HOOK)
    #: Picks the tests a range selects; without one the check runs whole or
    #: not at all, by ``scope``. The selected paths follow ``argv``.
    selector: Selector | None = field(default=None, compare=False, metadata=HOOK)
    #: Computes the scope from the tree being judged; ``scope`` is then empty.
    derive_scope: Callable[[Sequence[str]], tuple[str, ...]] | None = field(
        default=None, compare=False, metadata=HOOK
    )
    #: Builds the command lines from the selection, the run, the execution flags
    #: and the report flags, where the default spelling does not hold.
    command: Callable[..., list[list[str]]] | None = field(
        default=None, compare=False, metadata=HOOK
    )
    #: Runs before the command; a failure fails the check.
    prepare: Callable[[Context], int] | None = field(default=None, compare=False, metadata=HOOK)
    #: The check, or the part of one, implemented in Python; runs after the
    #: command, if there is one, passed.
    builtin: Callable[[Context], int] | None = field(default=None, compare=False, metadata=HOOK)
    #: The install the check proves something only against.
    install: Install | None = field(default=None, compare=False, metadata=HOOK)


@dataclass(frozen=True)
class Lane:
    """A CI job this command never runs and always names."""

    job: str
    why: str
    #: Path globs; a range that touches one is told to run the job on CI.
    scope: tuple[str, ...]


# ---------------------------------------------------------------------------
# Processes


def say(line: str) -> None:
    print(f"check: {line}", flush=True)


def run(argv: Sequence[str], cwd: Path | None = None) -> int:
    """Run *argv* with this process's stdout and stderr, and return its exit status.

    The child's output is never captured, redirected or discarded: what a check
    prints is part of what it reports.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return subprocess.run(list(argv), cwd=cwd or ROOT).returncode
    except FileNotFoundError:
        print(f"check: {argv[0]}: command not found", file=sys.stderr, flush=True)
        return 127


def capture(argv: Sequence[str]) -> tuple[int, str]:
    """Run a probe and return its exit status and stdout."""
    try:
        done = subprocess.run(list(argv), cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, ""
    return done.returncode, done.stdout


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def git_paths(subcommand: str, *args: str) -> list[str]:
    """The paths a git listing names, read NUL-separated so no path is split or quoted."""
    return [path for path in git(subcommand, "-z", *args).split("\0") if path]


def bounded(seconds: int, argv: Sequence[str]) -> list[str]:
    """*argv* stopped at *seconds* by ``run_bounded.py``."""
    return [*UV_RUN, "python", "scripts/run_bounded.py", str(seconds), "--", *argv]


# ---------------------------------------------------------------------------
# Scopes


def matches(path: str, pattern: str) -> bool:
    """Whether *path* matches the scope glob *pattern*.

    ``**`` crosses directories and ``*`` does not. A pattern without a slash is
    matched against the file name at any depth.
    """
    target = path if "/" in pattern else path.rsplit("/", 1)[-1]
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.fullmatch("".join(parts), target) is not None


def touches(paths: Iterable[str], scope: Sequence[str]) -> bool:
    """Whether any of *paths* falls in *scope*; an empty scope is every change."""
    if not scope:
        return True
    return any(matches(path, pattern) for path in paths for pattern in scope)
