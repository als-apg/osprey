"""The authoritative enumeration of a facility's channels.

One producer of "which channels exist": the facility knowledge graph or a
channel-finder paradigm database, selected per ``detect_pipeline_config`` --
never the write-limits projection ``channel_limits.json``, which gates a
subset. Consumers (plan-device derivation, the channel snapshot, the build's
fact lines, the channel-finder web routes) derive from this package rather
than enumerating a source of their own.

:func:`registered_channels` is that one producer's front door, and the only
entry point a consumer needs: it resolves the source
(:mod:`osprey.channel_roster.sources`), dispatches to the reader that source
kind belongs to (:mod:`osprey.channel_roster.graph`,
:mod:`osprey.channel_roster.database`) and pairs each settable channel with its
readback (:mod:`osprey.channel_roster.pairing`). The submodules stay importable
for the tests that exercise one stage, but a consumer that assembles the stages
itself is a second producer, which is the thing this package removes.

Absence is data: when no roster can be built the reason travels as a
:class:`~osprey.channel_roster.records.RosterAbsence` that every consumer
renders identically. See :mod:`osprey.channel_roster.records`.

**Reading a source is memoized per build process.** Several consumers ask the
same question during one build -- both bridge lanes render from it and the
channel snapshot is written from it -- and the corpus behind the answer is a
multi-megabyte Turtle file. It is parsed once per (source path, mtime, size),
so a build pays for it once and a source that changed on disk is re-read rather
than served stale. See :func:`registered_channels` for what else the key
carries and what is deliberately never cached.

Import-graph constraints this package keeps: it must not import
:mod:`osprey.services.facility_knowledge` (which pulls qmd -- the corpus is
read with literal IRIs and a lazily imported rdflib, as ``channel_snapshot``
does), and it is host/build-side only -- nothing inside the bridge container
imports it. Importing this package therefore costs a consumer nothing it does
not use: every heavyweight dependency the readers need -- rdflib, the channel
finder's database classes, the pipeline detection, the limits validator -- is
imported inside the function that needs it, so a build that resolves to an
absence never loads a parser.
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
    resolve_corpus_path,
    resolve_database_path,
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
#: :func:`_cache_key`). Module-level rather than an argument because the point
#: is to be shared across consumers that never meet: the compose generator's
#: two lane renders and the channel snapshot each call
#: :func:`registered_channels` with their own copy of the config.
#:
#: This is the test seam. A test that wants a cold read clears it; a test that
#: wants to observe a cache hit counts calls to a monkeypatched reader.
_roster_cache: dict[tuple, RosterResult] = {}


def registered_channels(config: dict[str, Any]) -> RosterResult:
    """Enumerate every channel this project's facility has.

    The one call a consumer makes. Source resolution, reading and readback
    pairing happen behind it, so that "which channels exist" has exactly one
    answer per build no matter who asks.

    Memoization is keyed on the resolved source path together with its mtime
    and size -- a rewritten source is a different key, so the cache cannot
    serve a roster the file no longer holds. The key also carries the paradigm
    that reads the file and, for a database source, the fingerprint of the
    channel limits file: those decide the *directions* the same database file
    yields, and keying on the file alone would let one project's answer be
    served to another whose limits differ.

    Two things are deliberately never cached. An absence from resolution has no
    path to key on -- a project that configures no source, or graph mode naming
    no corpus, is a config question that costs nothing to re-answer. And a
    source whose path cannot be stat'ed is read every time: "not there" during a
    build can mean "not there yet", and caching the miss would pin every later
    caller to a failure the build has since fixed.

    Args:
        config: Full project configuration dictionary, as the build holds it.

    Returns:
        A :class:`~osprey.channel_roster.records.RosterResult` holding one
        record per enumerated channel, each settable one carrying the readback
        the roster has for it; or one carrying only a
        :class:`~osprey.channel_roster.records.RosterAbsence` saying why there
        is no roster. Absence travels untouched -- pairing is not applied to a
        result that has no records, and the direction-underivable absence a
        database reader returns alongside real records survives pairing.

    Raises:
        PipelineModeError: If ``channel_finder.pipeline_mode`` names a paradigm
            that does not exist. A typo'd mode is a configuration mistake with
            a fix, and reporting it as "this facility has no channels" would
            hide it behind a plausible state.
    """
    resolution = resolve_roster_source(config)
    source = resolution.source
    if source is None:
        return RosterResult(absence=resolution.absence)

    key = _cache_key(config, resolution, source)
    if key is not None:
        cached = _roster_cache.get(key)
        if cached is not None:
            return cached

    result = _paired(_read(config, resolution, source))

    if key is not None:
        # One key at a time: a build reads one roster, so the second key is a
        # different project rather than a second question about this one, and
        # holding both would keep a corpus's records alive for nobody.
        _roster_cache.clear()
        _roster_cache[key] = result
    return result


def _read(
    config: dict[str, Any], resolution: RosterSourceResolution, source: RosterSource
) -> RosterResult:
    """Dispatch a resolved source to the reader that reads that kind."""
    if source.kind is RosterSourceKind.GRAPH:
        return read_graph_roster(source)
    return read_database_roster(
        config, source, resolution.paradigm or "", resolution.db_config or {}
    )


def _paired(result: RosterResult) -> RosterResult:
    """Give the result's settable channels their readbacks, source and absence intact."""
    if not result.records:
        return result
    return RosterResult(
        records=assign_readbacks(result.records),
        source=result.source,
        absence=result.absence,
    )


def _cache_key(
    config: dict[str, Any], resolution: RosterSourceResolution, source: RosterSource
) -> tuple | None:
    """Fingerprint everything a read of this resolution depends on.

    Returns:
        The key, or None when the source cannot be stat'ed and so must be read
        every time.
    """
    stamp = _stat_fingerprint(source.path)
    if stamp is None:
        return None
    if source.kind is RosterSourceKind.GRAPH:
        return (source.kind, stamp, resolution.paradigm)
    return (
        source.kind,
        stamp,
        resolution.paradigm,
        (resolution.db_config or {}).get("type"),
        _stat_fingerprint(resolve_limits_path(config)),
    )


def _stat_fingerprint(path: Path | None) -> tuple[str, int, int] | None:
    """Return ``(path, mtime_ns, size)``, or None when there is no readable file.

    A configured-but-unreadable limits file and no limits file at all fingerprint
    the same, which is correct: the database reader treats them the same way, by
    falling through to the address grammar.
    """
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)
