"""Materializing a deployment's archive: the base seed and its fingerprint.

A deployment that serves simulated channels needs history for them, and the
honest way to have history is to *store* it. This module computes that stored
history and writes it: one document per timestamp carrying every channel, over
a two-tier grid reaching back the configured retention, in the schema
``MongoDBArchiverConnector`` already reads.

The values are not this module's invention. Every channel is synthesized by the
same code that answers a live archiver query — ``SimulationEngine`` for the
channels a machine model describes, :mod:`osprey_connectors.simulation.procedural` for the
rest — evaluated at each document's own absolute timestamp. That is the whole
reason those generators are pure functions of ``(channel, epoch seconds)``: a
document written here at time T holds exactly what a query for T would compute,
so seeded history and synthesized history are one world rather than two
plausible ones.

Three pieces of the design are worth stating outright:

* **Two tiers, one grid.** The coarse tier spans the full retention window; the
  dense tier adds samples *between* its points over the recent span. They are
  built from one epoch-aligned grid and the dense tier skips timestamps the
  coarse tier already owns, so no timestamp is written twice — the schema is one
  document per timestamp and a duplicate would be a second, competing sample.

* **Retention is per document, not per collection.** Each document carries an
  ``expireAt`` that a TTL index acts on, stamped by the tier that produced it.
  A coarse sample lives for the retention span, a dense one until it ages out
  of the dense span, and a document with no stamp at all never expires — which
  is what lets a later scenario rewrite protect the windows it touches by
  simply removing the field.

* **Reseeding is a decision, not a default.** A seed manifest records the knobs
  the store was built with, and :func:`compare_fingerprint` reports match,
  mismatch or absence. The seed instant is stored but deliberately excluded from
  the comparison — it differs on every deploy, and including it would make every
  deploy a reseed.

This module never imports ``pymongo``: the functions that talk to a store take
a collection handle their caller opened. That keeps an optional dependency
optional, and leaves the grid, the fingerprint and the synthesis — the parts
worth reasoning about — pure and testable with no store at all.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from osprey_connectors.logger import get_logger
from osprey_connectors.simulation.engine import SimulationEngine, engine_serves
from osprey_connectors.simulation.procedural import (
    DEFAULT_NOISE_LEVEL,
    baseline_value,
    generate_series,
)
from osprey_connectors.simulation.series import epoch_seconds_array

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymongo.collection import Collection

logger = get_logger("archiver_seed")

__all__ = [
    "MANIFEST_ID",
    "SEED_SCHEMA_VERSION",
    "FingerprintComparison",
    "SeedKnobs",
    "SeedReport",
    "SeedState",
    "compare_fingerprint",
    "oldest_sample",
    "prepare_collection",
    "seed_base",
    "seed_fingerprint",
    "seed_grid",
    "synthesize_documents",
    "tier_expiry",
    "write_manifest",
]

#: ``_id`` of the seed-manifest document. A fixed id makes writing it an upsert
#: and reading it a primary-key lookup, and makes a second manifest impossible.
MANIFEST_ID = "osprey:seed_manifest"

#: Bumped when the meaning of a stored document changes in a way that makes an
#: existing store wrong rather than merely differently configured. It is part of
#: the fingerprint, so a bump reseeds every deployment on upgrade.
SEED_SCHEMA_VERSION = 1

#: Field naming the instant a document's sample was taken. The connector queries
#: and sorts on it; the manifest document deliberately has none, which is what
#: keeps it out of every archiver read without needing a filter.
DATE_FIELD = "date"

#: Field the TTL index acts on. A document without it never expires.
EXPIRE_FIELD = "expireAt"

#: Manifest record types, restated rather than imported: this module belongs to
#: the simulation package and the names live in the Virtual Accelerator service
#: package, which a host-side seeder must not need installed. They are a wire
#: vocabulary — the strings appear verbatim in every generated
#: ``channel_manifest.json`` — so restating them costs no coupling.
RECORD_TYPE_ANALOG = "ai"
RECORD_TYPE_BINARY = "bi"
RECORD_TYPE_MBB = "mbbi"
_TEXT_RECORD_TYPES = ("stringin", "longstringin")

#: Timestamps synthesized in one pass. Sized so the value matrix of a full
#: channel set stays in the tens of megabytes: the whole point of chunking is
#: that a month of history for three thousand channels does not fit in memory
#: at once.
DEFAULT_CHUNK_SIZE = 2000

#: Concurrent insert workers. The synthesis runs on the calling thread and the
#: inserts overlap it, which is the parallelism that matters here — the round
#: trip to the store, not the arithmetic.
DEFAULT_WORKERS = 4


class SeedState(Enum):
    """What a store's manifest says about the seed it holds."""

    ABSENT = "absent"
    """No manifest document: the store has never been seeded (or was wiped)."""

    MATCH = "match"
    """The store was built with the knobs now in force. Nothing to do."""

    MISMATCH = "mismatch"
    """The store was built with different knobs, so its coverage no longer
    describes what the profile asks for. The store has to be rebuilt."""


@dataclass(frozen=True)
class FingerprintComparison:
    """The outcome of comparing a store's manifest against current knobs."""

    state: SeedState
    differences: tuple[tuple[str, Any, Any], ...] = ()
    """``(key, stored, expected)`` for each field that moved. Empty unless the
    state is :attr:`SeedState.MISMATCH`. Carried so the deploy step can *report*
    what changed rather than announcing an unexplained reseed."""

    seeded_at: datetime | None = None
    """The instant the stored seed was anchored at, when there is one. Metadata:
    it is never compared (see the module docstring)."""

    def describe(self) -> str:
        """One line per changed knob, for a deploy-time report."""
        return "\n".join(
            f"  {key}: {stored!r} -> {expected!r}" for key, stored, expected in self.differences
        )


@dataclass(frozen=True)
class SeedKnobs:
    """The shape of the archive, as the profile's ``va_archiver:`` block sets it.

    Defaults mirror :class:`~osprey.cli.build_profile_archiver.VAArchiverConfig`
    so a caller holding no config still gets the shipped archive rather than an
    arbitrary one. The block validates ranges and the cadence-divisibility rule
    at build time; :meth:`validate` repeats the ones this module's arithmetic
    depends on, because a store can also be seeded from a hand-written config
    that never went through a profile.
    """

    retention_days: int = 30
    hot_span_hours: int = 48
    hot_cadence_sec: int = 10
    tail_cadence_sec: int = 60

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SeedKnobs:
        """Read the knobs out of a rendered project config.

        Args:
            config: The loaded ``config.yml`` as a mapping. Its ``va_archiver``
                subtree supplies the knobs; any absent one keeps its default,
                so a config predating a knob still seeds.

        Returns:
            The parsed knobs.

        Raises:
            ValueError: If a knob is present but not a positive integer, or the
                set is internally inconsistent (see :meth:`validate`).
        """
        block = config.get("va_archiver") or {}
        if not isinstance(block, Mapping):
            raise ValueError(f"config 'va_archiver' must be a mapping (got {type(block).__name__})")

        values: dict[str, int] = {}
        for name in ("retention_days", "hot_span_hours", "hot_cadence_sec", "tail_cadence_sec"):
            if name in block:
                raw = block[name]
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise ValueError(f"va_archiver.{name} must be an integer (got {raw!r})")
                values[name] = raw

        knobs = cls(**values)
        knobs.validate()
        return knobs

    def validate(self) -> None:
        """Refuse knobs this module's grid arithmetic cannot honor.

        Raises:
            ValueError: If any knob is below one, the dense span reaches past
                retention, or the coarse cadence is not a whole multiple of the
                dense one — that last is what makes the coarse grid a subset of
                the dense one rather than a second grid interleaved with it.
        """
        for name in ("retention_days", "hot_span_hours", "hot_cadence_sec", "tail_cadence_sec"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"va_archiver.{name} must be >= 1 (got {value})")
        if self.hot_span_hours > self.retention_days * 24:
            raise ValueError(
                f"va_archiver.hot_span_hours ({self.hot_span_hours}) exceeds "
                f"retention_days ({self.retention_days} days)"
            )
        if self.tail_cadence_sec % self.hot_cadence_sec:
            raise ValueError(
                f"va_archiver.tail_cadence_sec ({self.tail_cadence_sec}) must be a whole "
                f"multiple of hot_cadence_sec ({self.hot_cadence_sec})"
            )

    @property
    def retention_s(self) -> int:
        """Retention span in seconds."""
        return self.retention_days * 86400

    @property
    def hot_span_s(self) -> int:
        """Dense-tier span in seconds."""
        return self.hot_span_hours * 3600


@dataclass
class SeedReport:
    """What one seeding run wrote."""

    documents: int = 0
    channels: int = 0
    start: datetime | None = None
    end: datetime | None = None
    elapsed_s: float = 0.0
    chunks: int = 0

    def describe(self) -> str:
        """A one-line summary for a deploy-time progress report."""
        span = (
            f"{self.start:%Y-%m-%d %H:%M} to {self.end:%Y-%m-%d %H:%M} UTC"
            if self.start and self.end
            else "empty window"
        )
        return (
            f"seeded {self.documents:,} documents x {self.channels:,} channels "
            f"({span}) in {self.elapsed_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def seed_grid(knobs: SeedKnobs, t0: datetime) -> tuple[np.ndarray, np.ndarray]:
    """The timestamps to seed and the instant each one expires.

    The grid is epoch-aligned rather than anchored on ``t0``: both tiers land on
    whole multiples of their cadence since the epoch, which is the only way the
    coarse tier can be a subset of the dense one no matter what second ``t0``
    happens to fall on. The dense tier contributes only the timestamps the
    coarse tier does not already cover, so every timestamp appears exactly once.

    Expiry is stamped by the tier that produced a timestamp: a coarse sample
    lives out the retention span, a dense one lives until it ages out of the
    dense span. Both are measured from the sample's own time, so the oldest
    seeded documents are already at the end of their life when they are written
    — which is what makes a seeded archive age like a recorded one instead of
    surviving intact and then vanishing all at once.

    Args:
        knobs: The archive's shape.
        t0: The instant the window ends — the seed anchor. Naive values are read
            as UTC, matching the store's own convention.

    Returns:
        ``(epoch_seconds, expire_epoch_seconds)``, both float64, ascending by
        time. Empty only if the window is degenerate.
    """
    knobs.validate()
    end = _epoch_seconds(t0)

    tail_start = end - knobs.retention_s
    hot_start = end - knobs.hot_span_s

    tail = _aligned_range(tail_start, end, knobs.tail_cadence_sec)
    hot = _aligned_range(hot_start, end, knobs.hot_cadence_sec)
    # The coarse cadence is a whole multiple of the dense one, so a dense point
    # belongs to the coarse tier exactly when it is a multiple of the coarse
    # cadence. Testing that is cheaper and exact, where a set difference over
    # floats would be a float-equality comparison.
    hot_only = hot[np.mod(hot, knobs.tail_cadence_sec) != 0]

    times = np.concatenate([tail, hot_only])
    order = np.argsort(times, kind="stable")
    times = times[order]
    return times, tier_expiry(knobs, times)


def tier_expiry(knobs: SeedKnobs, epoch_s: np.ndarray) -> np.ndarray:
    """When each of these samples expires, by the tier its timestamp belongs to.

    A timestamp on the coarse cadence belongs to the coarse tier and lives out
    the retention span; anything else is a dense-tier sample and lives until it
    ages out of the dense span. Membership is read off the timestamp rather than
    tracked alongside it, which is what lets a *later* writer — a scenario
    rewrite restoring a window it once protected — re-stamp a document correctly
    without knowing how it was originally produced.
    """
    dense = np.mod(epoch_s, knobs.tail_cadence_sec) != 0
    lifetime = np.where(dense, float(knobs.hot_span_s), float(knobs.retention_s))
    return np.asarray(epoch_s + lifetime)


def _aligned_range(start_s: float, end_s: float, cadence_s: int) -> np.ndarray:
    """Epoch-aligned timestamps in ``[start_s, end_s]`` at ``cadence_s``."""
    first = np.ceil(start_s / cadence_s) * cadence_s
    last = np.floor(end_s / cadence_s) * cadence_s
    if last < first:
        return np.empty(0, dtype=np.float64)
    count = int(round((last - first) / cadence_s)) + 1
    return np.asarray(first + np.arange(count, dtype=np.float64) * cadence_s)


def _epoch_seconds(moment: datetime) -> float:
    """Epoch seconds for one datetime, reading a naive value as UTC."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.timestamp()


def _as_utc_datetimes(epoch_s: np.ndarray) -> list[datetime]:
    """Epoch seconds back to the timezone-aware datetimes the store stores.

    pymongo reads a naive datetime as UTC, so writing aware ones costs nothing
    and removes the question entirely.
    """
    return [datetime.fromtimestamp(float(value), UTC) for value in epoch_s]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def synthesize_documents(
    channels: Sequence[Mapping[str, Any]],
    epoch_s: np.ndarray,
    *,
    engine: SimulationEngine | None = None,
    boot_values: Mapping[str, float] | None = None,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    value_transform: Callable[[str, Sequence[Any]], Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """One document per timestamp, carrying every channel's value at it.

    This is the single definition of "what the archive holds at time T", and it
    is deliberately exported: the base seed writes it, a scenario rewrite
    restores to it, and an equivalence test compares a live query against it.
    A second implementation of this arithmetic anywhere would be a second
    opinion about the machine's past.

    ``value_transform`` is the one seam in that: the arithmetic stays here, and
    a caller may only *declare* a transform over its result — never compute the
    values itself. It exists for the deployment whose ``live`` target is a
    stand-in, whose readout carries systematic offsets the machine this module
    models does not have. Because a transform changes what the store contains,
    a caller applying one must describe it to :func:`seed_fingerprint` through
    ``transform_fingerprint``; a store seeded with offsets would otherwise
    compare MATCH against one seeded without them.

    Channel values come from three places, matching what the deployment's live
    half serves:

    * a channel the machine model describes goes through the engine, events and
      all;
    * an analog channel it does not goes through the procedural generator,
      anchored on the value the Virtual Accelerator boots it at;
    * a channel served as a flag, an enumeration or text carries a *constant* —
      the live machine holds those still, and noise on a status flag would be
      history no operator could read as anything but a fault.

    Args:
        channels: Manifest channel entries. Each needs ``address``; ``record_type``
            selects the constant path when present (absent is read as analog).
        epoch_s: Absolute epoch seconds, one per document.
        engine: The machine model's engine, or ``None`` when the project has
            none and every channel is procedural.
        boot_values: ``{address: value}`` from the machine model, threaded into
            :func:`~osprey_connectors.simulation.procedural.baseline_value` so procedural
            channels anchor where the VA boots them.
        noise_level: Relative noise for the procedural channels.
        value_transform: Called once per channel with ``(address, values)``
            after the values are synthesized and before they are scattered into
            the documents, returning the values to store. It must return one
            value per timestamp, in the same order; a shorter or longer sequence
            raises rather than silently shifting a channel's history. ``None``
            stores what this module computed, which is the only behaviour a
            deployment without a stand-in ever sees.

    Returns:
        A list of ``{date: datetime, <address>: value}`` documents, ascending in
        time. No ``expireAt`` — stamping is the caller's, because the tier a
        timestamp belongs to is a property of the grid, not of the values.
    """
    stamps = _as_utc_datetimes(epoch_s)
    documents: list[dict[str, Any]] = [{DATE_FIELD: stamp} for stamp in stamps]
    if not documents:
        return documents

    # Synthesize at the epoch seconds a *reader* of these documents derives from
    # their stored dates, not at the ones the grid was built from. Datetimes
    # carry microseconds, so a grid time with finer resolution than that would
    # otherwise be stored as one instant and valued as another — a discrepancy
    # no test of the shipped whole-second cadences would ever surface, and one
    # that would break bit-equality for any caller that seeds a finer grid.
    reader_epoch_s = epoch_seconds_array(stamps)
    assert reader_epoch_s is not None  # noqa: S101 - datetimes always convert

    for channel in channels:
        address = str(channel["address"])
        values = _channel_values(
            channel,
            address,
            stamps,
            reader_epoch_s,
            engine=engine,
            boot_values=boot_values,
            noise_level=noise_level,
        )
        if value_transform is not None:
            values = value_transform(address, values)
        # ``strict`` is the length contract: a transform that returned the wrong
        # number of values would otherwise leave the tail of a channel's window
        # unwritten, which reads back as a channel that simply stops.
        for document, value in zip(documents, values, strict=True):
            document[address] = value

    return documents


def _channel_values(
    channel: Mapping[str, Any],
    address: str,
    stamps: list[datetime],
    epoch_s: np.ndarray,
    *,
    engine: SimulationEngine | None,
    boot_values: Mapping[str, float] | None,
    noise_level: float,
) -> Sequence[Any]:
    """One channel's values across the chunk, by the rules above."""
    record_type = str(channel.get("record_type", RECORD_TYPE_ANALOG))

    if engine_serves(engine, address) and engine is not None:
        # Even a flag goes through synthesis when the model describes it: a
        # scenario is entitled to step STATUS:FAULT true partway through the
        # window, and a constant would erase exactly the event worth archiving.
        return [_coerce(record_type, value) for value in engine.synthesize_series(address, stamps)]

    if record_type == RECORD_TYPE_ANALOG:
        return [
            float(value)
            for value in generate_series(
                address,
                epoch_s,
                noise_level=noise_level,
                baseline=baseline_value(address, boot_values),
            )
        ]

    # A discrete or textual channel with no model entry: nothing moves it. The
    # manifest forbids declaring these record types noisy, and the VA serves
    # them as a bare coerced baseline on every poll tick — so one value covers
    # the window, and computing a noisy series to round it away would be both
    # slower and a claim about motion that never happens.
    return [_coerce(record_type, baseline_value(address, boot_values))] * len(stamps)


def _coerce(record_type: str, value: Any) -> Any:
    """One value in the type the channel is served as.

    Mirrors the serving layer's own coercion table. The machine model stores
    every channel as a number, even the ones served as flags, and a raw float
    on a flag channel would be archived as a value no client could ever have
    read back.
    """
    if record_type == RECORD_TYPE_BINARY:
        return bool(value)
    if record_type == RECORD_TYPE_MBB:
        return int(value)
    if record_type in _TEXT_RECORD_TYPES:
        return str(value)
    return float(value)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def seed_fingerprint(
    knobs: SeedKnobs,
    channel_addresses: Iterable[str],
    *,
    compression: str,
    transform_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The knobs a store's coverage depends on, as a comparable dict.

    Everything here changes what the store *contains*: the window's depth and
    density, which channels are in it, how it is compressed on disk, the
    document schema itself, and any transform the caller applied to the values.
    Nothing here is an instant or a count — those move on every deploy and would
    make the comparison a coin flip.

    The transform belongs here for the same reason the channel set does: a store
    seeded with a stand-in's systematic offsets holds different numbers than one
    seeded without them, and a fingerprint blind to that would report MATCH
    across the very change that makes the stored past belong to another machine.

    Args:
        knobs: The archive's shape.
        channel_addresses: The seeded channel set. Order does not matter; the
            hash is taken over the sorted, deduplicated names.
        compression: The collection's block compressor.
        transform_fingerprint: A JSON-able description of the value transform
            passed to :func:`synthesize_documents`, or ``None`` when the values
            are stored as this module computed them. It is folded in as
            canonical JSON — key order and tuple-versus-list spelling therefore
            do not move the fingerprint — under the ``value_transform`` field,
            whose ``None`` is also what a manifest predating the field reads as,
            so an untransformed store seeded by an older version still compares
            MATCH rather than reseeding for a knob nobody changed.

    Returns:
        The fingerprint, JSON-serializable and safe to store verbatim.
    """
    names = sorted(set(channel_addresses))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "retention_days": knobs.retention_days,
        "hot_span_hours": knobs.hot_span_hours,
        "hot_cadence_sec": knobs.hot_cadence_sec,
        "tail_cadence_sec": knobs.tail_cadence_sec,
        "compression": compression,
        "channel_count": len(names),
        "channel_set_sha256": digest,
        "value_transform": (
            None
            if transform_fingerprint is None
            else json.dumps(transform_fingerprint, sort_keys=True, separators=(",", ":"))
        ),
    }


def write_manifest(
    collection: Collection,
    fingerprint: Mapping[str, Any],
    *,
    seeded_at: datetime,
    report: SeedReport | None = None,
) -> None:
    """Record what this store was built with, replacing any previous manifest.

    The manifest lives in the sample collection under a fixed ``_id`` and
    carries no ``date`` field, so no archiver query can ever match it — the
    connector's window filter requires one. Keeping it here rather than in a
    sidecar collection is what makes "the store" one thing to create, wipe and
    reason about.

    The touched-window ledger starts empty. It belongs to the scenario rewrite,
    which needs to know which windows it has to restore before applying a new
    set; a fresh base has none, and a reseed clears it precisely because the
    windows it named no longer exist.
    """
    document = {
        "_id": MANIFEST_ID,
        "fingerprint": dict(fingerprint),
        "seeded_at": seeded_at,
        "touched_windows": [],
    }
    if report is not None:
        document["coverage"] = {
            "documents": report.documents,
            "channels": report.channels,
            "start": report.start,
            "end": report.end,
        }
    collection.replace_one({"_id": MANIFEST_ID}, document, upsert=True)


def compare_fingerprint(
    collection: Collection, fingerprint: Mapping[str, Any]
) -> FingerprintComparison:
    """Whether the store in front of us was built with the knobs now in force.

    Args:
        collection: The sample collection.
        fingerprint: What :func:`seed_fingerprint` says the knobs are now.

    Returns:
        The comparison. ``MISMATCH`` carries every field that moved, including
        ones the stored manifest lacks entirely (reported as ``None``), so a
        store seeded by an older version reseeds with a readable reason rather
        than silently.
    """
    stored = collection.find_one({"_id": MANIFEST_ID})
    if stored is None:
        return FingerprintComparison(SeedState.ABSENT)

    seeded_at = stored.get("seeded_at")
    held = stored.get("fingerprint") or {}
    differences = tuple(
        (key, held.get(key), expected)
        for key, expected in sorted(fingerprint.items())
        if held.get(key) != expected
    )
    if differences:
        return FingerprintComparison(SeedState.MISMATCH, differences, seeded_at)
    return FingerprintComparison(SeedState.MATCH, (), seeded_at)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def prepare_collection(collection: Collection) -> None:
    """Create the two indexes the store cannot work without.

    ``{date: 1}`` is not an optimization: every archiver read sorts on ``date``,
    and an unindexed sort is capped at 32 MB of documents in memory — a limit a
    real window crosses long before a real deployment notices it in testing.

    The TTL index is declared with ``expireAfterSeconds=0`` so each document's
    own ``expireAt`` *is* its deadline, which is what gives the two tiers
    different lifetimes from one index. Documents without the field — the
    manifest, and any window a scenario rewrite has protected — are ignored by
    it entirely.

    Both are idempotent: re-creating an identical index is a no-op, so this is
    safe on every deploy rather than only on the first.
    """
    collection.create_index(DATE_FIELD)
    collection.create_index(EXPIRE_FIELD, expireAfterSeconds=0)


def seed_base(
    collection: Collection,
    channels: Sequence[Mapping[str, Any]],
    knobs: SeedKnobs,
    *,
    t0: datetime,
    engine: SimulationEngine | None = None,
    boot_values: Mapping[str, float] | None = None,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    compression: str = "zstd",
    workers: int = DEFAULT_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[SeedReport], None] | None = None,
    value_transform: Callable[[str, Sequence[Any]], Sequence[Any]] | None = None,
    transform_fingerprint: Mapping[str, Any] | None = None,
) -> SeedReport:
    """Build the store: indexes, the whole base series, then the manifest.

    The order is the contract. Indexes come first so the documents land into an
    indexed collection instead of being indexed afterwards; the manifest comes
    *last*, so a run that dies halfway leaves a store with no manifest — which
    the next deploy reads as ``ABSENT`` and rebuilds. A manifest written first
    would make a half-seeded store indistinguishable from a complete one.

    Synthesis runs on the calling thread and inserts overlap it on a small pool.
    That split is deliberate: the arithmetic is numpy and the engine is not
    contractually thread-safe, while the round trip to the store is the part
    worth overlapping. Submission blocks once the pool is saturated, so memory
    stays bounded at a few chunks rather than the whole window.

    Args:
        collection: The sample collection. Existing samples are not removed —
            a caller reseeding is expected to drop first, and one that is
            extending a store deliberately is not second-guessed here.
        channels: Manifest channel entries to seed.
        knobs: The archive's shape.
        t0: The seed anchor; the window ends here.
        engine: The machine model's engine, or ``None``.
        boot_values: Machine-model values, for procedural baseline anchoring.
        noise_level: Relative noise for the procedural channels.
        compression: The collection's block compressor, recorded in the
            manifest because changing it changes the store's size on disk.
        workers: Concurrent insert workers.
        chunk_size: Timestamps synthesized and inserted per batch.
        progress: Called after each chunk with the running report, for a
            deploy-time progress line.
        value_transform: Applied per channel to every chunk's values (see
            :func:`synthesize_documents`), or ``None`` to store the synthesized
            values unchanged.
        transform_fingerprint: The transform's declared identity, recorded in
            the manifest so a store built with it does not compare MATCH against
            one built without. Passing a ``value_transform`` and leaving this
            ``None`` is how a reseed would silently be skipped, so a caller
            applying one is expected to describe it.

    Returns:
        The completed report.

    Raises:
        ValueError: If ``chunk_size`` or ``workers`` is below one, or the knobs
            are inconsistent.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers})")

    times, expiry = seed_grid(knobs, t0)
    addresses = [str(channel["address"]) for channel in channels]
    started = time.monotonic()

    report = SeedReport(channels=len(addresses))
    if len(times):
        report.start = datetime.fromtimestamp(float(times[0]), UTC)
        report.end = datetime.fromtimestamp(float(times[-1]), UTC)

    prepare_collection(collection)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="archiver-seed") as pool:
        pending: set[Future[None]] = set()
        for chunk_times, chunk_expiry in _chunks(times, expiry, chunk_size):
            documents = synthesize_documents(
                channels,
                chunk_times,
                engine=engine,
                boot_values=boot_values,
                noise_level=noise_level,
                value_transform=value_transform,
            )
            for document, expire_at in zip(documents, _as_utc_datetimes(chunk_expiry), strict=True):
                document[EXPIRE_FIELD] = expire_at

            pending = _submit_bounded(pool, pending, collection, documents, workers)

            report.documents += len(documents)
            report.chunks += 1
            report.elapsed_s = time.monotonic() - started
            if progress is not None:
                progress(report)

        for future in pending:
            future.result()

    report.elapsed_s = time.monotonic() - started
    write_manifest(
        collection,
        seed_fingerprint(
            knobs,
            addresses,
            compression=compression,
            transform_fingerprint=transform_fingerprint,
        ),
        seeded_at=t0,
        report=report,
    )
    logger.debug(report.describe())
    return report


def _submit_bounded(
    pool: ThreadPoolExecutor,
    pending: set[Future[None]],
    collection: Collection,
    documents: list[dict[str, Any]],
    limit: int,
) -> set[Future[None]]:
    """Queue one insert, first draining until fewer than ``limit`` are in flight.

    Draining is what bounds memory: an unbounded submit would let synthesis run
    ahead of the store and hold the entire window's documents at once. Results
    are collected as part of draining so an insert failure surfaces here rather
    than being swallowed by a future nobody reads.
    """
    while len(pending) >= limit:
        done = {future for future in pending if future.done()}
        if not done:
            next(iter(pending)).result()
            continue
        for future in done:
            future.result()
        pending -= done

    pending.add(pool.submit(_insert_chunk, collection, documents))
    return pending


def _insert_chunk(collection: Collection, documents: list[dict[str, Any]]) -> None:
    """Insert one batch, unordered so the server may parallelize it internally."""
    if documents:
        collection.insert_many(documents, ordered=False)


def _chunks(
    times: np.ndarray, expiry: np.ndarray, size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Split the grid into contiguous batches of at most ``size`` timestamps."""
    for start in range(0, len(times), size):
        stop = start + size
        yield times[start:stop], expiry[start:stop]


def oldest_sample(collection: Collection) -> datetime | None:
    """The instant of the oldest stored sample, or ``None`` for an empty store.

    This is what honest archiver metadata reports as the start of coverage. The
    filter on ``date`` skips the manifest, which has none — and skipping it is
    the reason the manifest was given no ``date`` in the first place.
    """
    document = collection.find_one({DATE_FIELD: {"$exists": True}}, sort=[(DATE_FIELD, 1)])
    if document is None:
        return None
    stamp = document.get(DATE_FIELD)
    if not isinstance(stamp, datetime):
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)
