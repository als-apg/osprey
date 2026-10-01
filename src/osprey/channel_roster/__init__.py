"""The authoritative enumeration of a facility's channels.

One producer of "which channels exist": the facility file a build writes at the
root of every render -- never the write-limits projection
``channel_limits.json``, which gates a subset. Consumers (plan-device
derivation, the channel snapshot, the build's fact lines, the channel-finder
web routes) derive from this package rather than enumerating a source of their
own.

:func:`registered_channels` is that one producer's front door, and the only
entry point a consumer needs: it resolves the facility file and reads its
channel records (:mod:`osprey.channel_roster.sources`), each setpoint carrying
the readback the file pairs it with. The submodules stay importable for the
tests that exercise one stage, but a consumer that assembles the stages itself
is a second producer, which is the thing this package removes.

Absence is data: when no roster can be built the reason travels as a
:class:`~osprey.channel_roster.records.RosterAbsence` that every consumer
renders identically. See :mod:`osprey.channel_roster.records`.

**Reading the source is memoized per build process.** Several consumers ask the
same question during one build -- both bridge lanes render from it and the
channel snapshot is written from it -- and the answer is a file on disk. It
is read once per (source path, mtime, size), so a build pays for it once and a
source that changed on disk is re-read rather than served stale. See
:func:`registered_channels` for what is deliberately never cached.

Import-graph constraints this package keeps: it must not import
:mod:`osprey.services.facility_knowledge` (which pulls qmd), and it is
host/build-side only -- nothing inside the bridge container imports it.
Importing this package costs a consumer nothing it does not use: every
heavyweight dependency a reader needs is imported inside the function that
needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import read_database_roster, resolve_limits_path
from .graph import read_graph_roster
from .pairing import assign_readbacks
from .records import (
    ABSENCE_TEMPLATES,
    SOURCE_LABELS,
    ChannelDirection,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
    RosterSourceKind,
)
from .sources import (
    RosterSourceResolution,
    read_facility_roster,
    resolve_roster_source,
)

#: The package's front door: what a consumer outside it needs to call
#: :func:`registered_channels`, hold what it returns, and render an absence.
#: The other names the imports above bring in stay importable for the tests
#: that exercise one stage, but they are the package's own vocabulary.
__all__ = [
    "ChannelRecord",
    "RosterAbsence",
    "RosterAbsenceReason",
    "RosterResult",
    "RosterSource",
    "RosterSourceKind",
    "registered_channels",
    "resolve_roster_source",
]

#: Rosters already read, keyed by everything the read depends on (see
#: :func:`registered_channels`). Module-level rather than an argument because the point
#: is to be shared across consumers that never meet: the compose generator's
#: two lane renders and the channel snapshot each call
#: :func:`registered_channels` with their own copy of the config.
#:
#: This is the test seam. A test that wants a cold read clears it; a test that
#: wants to observe a cache hit counts calls to a monkeypatched reader.
_roster_cache: dict[tuple, RosterResult] = {}


def registered_channels(config: dict[str, Any]) -> RosterResult:
    """Enumerate every channel this project's facility has.

    The one call a consumer makes. Source resolution and reading happen behind
    it, so that "which channels exist" has exactly one answer per build no
    matter who asks.

    Memoization is keyed on the resolved source path together with its mtime
    and size -- a rewritten source is a different key, so the cache cannot
    serve a roster the file no longer holds.

    A source whose path cannot be stat'ed is deliberately never cached, and is
    read every time: "not there" during a build can mean "not there yet", and
    caching the miss would pin every later caller to a failure the build has
    since fixed.

    Args:
        config: Full project configuration dictionary, as the build holds it.

    Returns:
        A :class:`~osprey.channel_roster.records.RosterResult` holding one
        record per channel record of the facility file, each settable one
        carrying the readback the file pairs it with; or one carrying only a
        :class:`~osprey.channel_roster.records.RosterAbsence` saying why there
        is no roster.
    """
    resolution = resolve_roster_source(config)
    source = resolution.source
    if source is None:
        return RosterResult(absence=resolution.absence)

    stamp = _stat_fingerprint(source.path)
    key = None if stamp is None else (source.kind, stamp)
    if key is not None:
        cached = _roster_cache.get(key)
        if cached is not None:
            return cached

    result = read_facility_roster(source)

    if key is not None:
        # One key at a time: a build reads one roster, so the second key is a
        # different project rather than a second question about this one, and
        # holding both would keep a facility's records alive for nobody.
        _roster_cache.clear()
        _roster_cache[key] = result
    return result


def _stat_fingerprint(path: Path) -> tuple[str, int, int] | None:
    """Return ``(path, mtime_ns, size)``, or None when there is no readable file."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)
