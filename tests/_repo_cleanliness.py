"""What a test run did to ``<repo>/var/agent_data``, used by ``tests/conftest.py``.

That directory is the mark left behind when something resolved the agent-data
root to the checkout instead of to a tmp path, and the guard in
``tests/conftest.py`` fails a session that produced one.

Existence is the wrong question to ask of it. A developer who has actually run
OSPREY in this checkout owns that directory, and what is inside it is that
deployment's business — but the commonest way to own it is to have leaked it:
a previous run created it, was interrupted before its own teardown could
speak, and left behind exactly the state that reads as "a developer owns this".
Keyed on existence, the guard is therefore silent from the first leak onwards,
on the machine where the leak is easiest to reproduce. So what is compared here
is a *snapshot*: the directory's own entries, plus its modification time, taken
before any test and compared after the last one.

One case is out of reach by construction — a write *inside* an entry the
snapshot already held. Finding that would mean walking a real deployment's run
folders on every test, which is a cost the guard is not worth. What is caught is
an entry appearing at the top level, which is the shape every leak observed so
far has had.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
