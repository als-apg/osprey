"""What a test run did to the roots it must leave alone, used by the conftest guards.

Two forms of root are guarded. ``<repo>/var/agent_data`` and its siblings are
single directories: each is the mark left behind when something resolved a
runtime root to the checkout instead of to a tmp path. The packaged templates
are a tree: tests copy it while other tests run beside them, so a write into it
surfaces as a copy error somewhere else. The guards in ``_guards`` fail a
session that changed either.

This module reads leak markers, never subjects. What it compares is what git
does not track — new entries, removed entries, a moved mtime — so it reads the
disk itself and never the tracked-file listing.

Existence is the wrong question to ask of a marker directory. A developer who
has actually run OSPREY in this checkout owns it, and what is inside it is that
deployment's business — but the commonest way to own it is to have leaked it:
a previous run created it, was interrupted before its own teardown could
speak, and left behind exactly the state that reads as "a developer owns this".
Keyed on existence, the guard is therefore silent from the first leak onwards,
on the machine where the leak is easiest to reproduce. So what is compared here
is a *snapshot*: the directory's own entries, plus its modification time, taken
before any test and compared after the last one.

For the single-directory form, one case is out of reach by construction — a
write *inside* an entry the snapshot already held. Finding that would mean
walking a real deployment's run folders on every test, which is a cost the
guard is not worth. What is caught is an entry appearing at the top level,
which is the shape every leak observed so far has had. The tree form has no
such gap: its root is bounded and shipped, so it snapshots every directory in
it and reaches a write at any depth, including one made and deleted again in a
directory that holds no Python module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_SKIPPED = "__pycache__"
_SOURCE_SUFFIX = ".py"


@dataclass(frozen=True)
class DirectoryState:
    """A directory at one moment: whether it is there, its entries, its mtime."""

    exists: bool
    mtime_ns: int | None = None
    entries: frozenset[str] = frozenset()


def snapshot(path: Path) -> DirectoryState:
    """The state of ``path`` now, or an absent one if it cannot be read.

    A directory that cannot be stat'ed or listed is reported as absent, which
    is the same answer ``Path.exists()`` already gave for that case: a
    permission error is not evidence of a leak, and a guard that treated it as
    one would fail runs for a reason the run did not cause.
    """
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return DirectoryState(exists=False)
    try:
        entries = frozenset(entry.name for entry in path.iterdir())
    except OSError:
        entries = frozenset()
    return DirectoryState(exists=True, mtime_ns=mtime_ns, entries=entries)


def what_this_run_did(baseline: DirectoryState, path: Path) -> str | None:
    """How ``path`` changed since ``baseline``, as a clause, or ``None``.

    The clause reads as the predicate of "the test run …", which is how the
    guard's message is phrased, so it drops straight into it.

    The entry diff is what *names* the leak, and it is tried first. The mtime
    comparison only adds the case where an entry appeared and was removed again
    before anyone looked, and it is the weaker signal of the two: a filesystem
    whose timestamps are coarse can leave the mtime untouched across a create
    and a delete inside one tick. It must therefore never be the gate in front
    of the listing — a directory whose mtime did not move can still have gained
    an entry.
    """
    now = snapshot(path)
    if not now.exists:
        return None
    if not baseline.exists:
        return f"created {path}"
    appeared = sorted(now.entries - baseline.entries)
    if appeared:
        return f"added {', '.join(appeared)} to {path}"
    if now.mtime_ns != baseline.mtime_ns:
        return f"added and removed entries in {path}"
    return None


def tree_snapshot(root: Path) -> dict[str, DirectoryState]:
    """Every directory under ``root``, keyed by its root-relative POSIX path.

    The root itself is ``"."``. Byte-code caches are not walked, but each
    directory's ``entries`` are its raw listing and so still name its
    ``__pycache__``. A root that is not there has no directories: ``{}``.
    """
    states: dict[str, DirectoryState] = {}
    for directory, dirnames, _ in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != _SKIPPED]
        here = Path(directory)
        states[here.relative_to(root).as_posix()] = snapshot(here)
    return states


def what_this_run_did_to_tree(baseline: dict[str, DirectoryState], root: Path) -> list[str]:
    """How the tree at ``root`` changed since ``baseline``, as sorted clauses.

    Each clause reads as the predicate of "the test run …", as the clause of
    ``what_this_run_did`` does, and an unchanged tree gives an empty list.

    Per directory, the names that appeared and the names that went are the
    clauses that *name* the change; a directory that went is named by its
    parent's ``removed`` clause. A moved mtime with no name delta is a write
    that came and went. ``__pycache__`` is never a change, and neither is the
    mtime of a directory that holds a Python module: any interpreter that
    imports the module can create its ``__pycache__`` there, and a bytecode
    prune can remove it again, at any point in the session and in either
    order, so that mtime says nothing about what a test wrote. A transient
    write into a directory that holds a Python module therefore goes unseen;
    one that leaves an entry behind is still named.
    """
    now = tree_snapshot(root)
    if not now:
        return [f"removed {root}"] if baseline else []
    if not baseline:
        return [f"created {root}"]
    clauses: list[str] = []
    for key in sorted(baseline.keys() & now.keys()):
        then, current = baseline[key], now[key]
        directory = root / key
        then_names = then.entries - {_SKIPPED}
        now_names = current.entries - {_SKIPPED}
        appeared = sorted(now_names - then_names)
        gone = sorted(then_names - now_names)
        if appeared:
            clauses.append(f"added {', '.join(appeared)} to {directory}")
        if gone:
            clauses.append(f"removed {', '.join(gone)} from {directory}")
        if (
            not appeared
            and not gone
            and current.mtime_ns != then.mtime_ns
            and current.entries ^ then.entries != {_SKIPPED}
            and not _holds_python_source(then_names | now_names)
        ):
            clauses.append(f"added and removed entries in {directory}")
    return clauses


def _holds_python_source(names: frozenset[str]) -> bool:
    """Whether a directory with these entries is one an import byte-compiles into."""
    return any(name.endswith(_SOURCE_SUFFIX) for name in names)
