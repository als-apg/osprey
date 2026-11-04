"""CANOPY's Mole adapter for OSPREY's MongoDB archiver-recorder.

Loaded by its ``module:Class`` path (:data:`RECORDER_TYPE`), as CANOPY loads an out-of-tree adapter.

OSPREY's archiver-recorder stores one document per instant in a single
collection: a tz-aware ``date`` (BSON Date), one float field per recorded
channel, and an ``expireAt`` TTL stamp. The scenario seeder also leaves one
undated manifest document in the same collection. This adapter maps that
shape onto Mole's ``ArchiverAdapter`` contract.

Configuration (``ArchiverInstanceConfig``):

* ``url`` -- a ``mongodb://`` URI whose default database is the archive
  database (required); the URI carries the user but no password;
* ``api_key_env`` -- name of the env var holding the password (required);
* ``extras['collection']`` -- the archive collection (required);
* ``extras['max_raw']``, ``extras['max_binned_scan']``,
  ``extras['last_before_lookback_s']`` -- read caps, coerced with ``int()``.

A misconfigured adapter never raises at build time and never connects: it
reports unhealthy with the configuration message, and every read raises
``ArchiverUnavailable`` carrying that same message, so canopy-api always boots.

The channel set is the field names of the oldest and newest dated documents,
minus the bookkeeping fields, cached for 30 s. The index reads those two
documents only: a channel recorded solely in between is not listed. Every
operation that takes a PV checks membership first.

``search`` takes a regular expression (at most 128 characters from
``[A-Za-z0-9_:-.*^$[]]``, at most three ``*``) and returns the sorted channel
names it fully matches; anything else is a 400. ``metadata`` reports the PV and
archiver only: the recorder stores no units, precision or limits.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bson
import numpy as np
from canopy.contracts.sample import SEVERITY_ORDINAL, AlarmSeverity, BinnedPoint, Sample
from canopy.core.errors import BadRequest
from canopy.mole.adapters.base import (
    SEED_SAMPLE_BUDGET,
    AdapterHealth,
    ArchiveBlock,
    ArchiverBadResponse,
    ArchiverError,
    ArchiverTimeout,
    ArchiverUnavailable,
    BinningNotSupported,
    PVNotFoundInArchiver,
    fetch_window_budgeted_via,
)
from canopy.mole.config import ArchiverInstanceConfig
from canopy.mole.schemas.pv import PVMetadata
from pymongo import AsyncMongoClient
from pymongo import errors as mongo_errors
from pymongo.uri_parser import parse_uri

#: The ``type`` CANOPY's Mole config names this adapter by.
RECORDER_TYPE = "osprey.integrations.canopy.mongo_recorder:OspreyMongoRecorderAdapter"


def _decode_raw_batches(batches: list[bytes], pv: str) -> ArchiveBlock:
    """``bson.decode_all`` plus the numpy build, meant for a worker thread."""
    docs = [doc for batch in batches for doc in bson.decode_all(batch)]
    ts = np.fromiter(
        (datetime_to_ns(doc[DATE_FIELD]) for doc in docs), dtype=np.int64, count=len(docs)
    )
    values = np.empty(len(docs), dtype=object)
    for i, doc in enumerate(docs):
        values[i] = _map_value(doc[pv])
    # The recorder keeps no alarm state: every sample is NO_ALARM (see ``_sample``).
    severities = np.full(len(docs), SEVERITY_ORDINAL[AlarmSeverity.NO_ALARM], dtype=np.int8)
    return ArchiveBlock(ts, values, severities)


#: Field holding each document's sample instant.
DATE_FIELD = "date"

#: Fields of a recorder document that are not channels.
EXCLUDED_FIELDS: frozenset[str] = frozenset({"_id", DATE_FIELD, "expireAt", "osprey_densified"})

#: Read caps and their defaults (FR8/FR9).
DEFAULT_CAPS: dict[str, int] = {
    "max_raw": 50_000,
    "max_binned_scan": 1_000_000,
    "last_before_lookback_s": 86_400,
}

#: Search pattern limits: length, alphabet, and the number of ``*``.
MAX_PATTERN_LEN = 128
MAX_PATTERN_STARS = 3
_PATTERN_ALPHABET = re.compile(r"[A-Za-z0-9_:\-.*^$\[\]]*")
_BAD_PATTERN = "pattern is a regular expression (use .* not *)"

#: How long the channel index is reused before it is queried again.
INDEX_TTL_S = 30.0

#: Bound on the health probe, below the registry's 2 s window so the
#: adapter's own message surfaces instead of "health probe timed out".
HEALTH_TIMEOUT_S = 1.5

#: Server-selection bound handed to the client.
SERVER_SELECTION_TIMEOUT_MS = 2000

#: OperationFailure codes that mean "not allowed in": Unauthorized, AuthenticationFailed.
_AUTH_CODES = frozenset({13, 18})


def map_mongo_error(exc: BaseException, archiver: str) -> ArchiverError:
    """Map a pymongo (or asyncio timeout) failure onto Mole's archiver errors.

    The checks run in the FR9 order, which matters because of pymongo's class
    hierarchy: ``ServerSelectionTimeoutError`` and ``NetworkTimeout`` both
    subclass ``AutoReconnect`` and carry ``timeout=True``, and
    ``ExecutionTimeout`` subclasses ``OperationFailure``.

    1. ``ServerSelectionTimeoutError`` -> 503
    2. ``NetworkTimeout``, ``ExecutionTimeout``, ``asyncio.TimeoutError`` or any
       ``PyMongoError`` with ``.timeout`` true -> 504
    3. ``AutoReconnect`` -> 503
    4. ``OperationFailure`` with code 13 or 18 -> 503
    5. any other ``PyMongoError`` -> 502
    """
    prefix = f"MongoDB archiver '{archiver}'"
    if isinstance(exc, mongo_errors.ServerSelectionTimeoutError):
        return ArchiverUnavailable(f"{prefix} unreachable: {exc}")
    if isinstance(
        exc,
        (mongo_errors.NetworkTimeout, mongo_errors.ExecutionTimeout, asyncio.TimeoutError),
    ) or (isinstance(exc, mongo_errors.PyMongoError) and getattr(exc, "timeout", False)):
        return ArchiverTimeout(f"{prefix} timed out: {exc or type(exc).__name__}")
    if isinstance(exc, mongo_errors.AutoReconnect):
        return ArchiverUnavailable(f"{prefix} connection lost: {exc}")
    if isinstance(exc, mongo_errors.OperationFailure) and exc.code in _AUTH_CODES:
        return ArchiverUnavailable(f"{prefix} refused access: {exc}")
    return ArchiverBadResponse(f"{prefix} failed: {exc}")


_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
_NS_PER_US = 1_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _ms_to_datetime(ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=ms)


def ns_floor_ms(ts_ns: int) -> datetime:
    """``ts_ns`` as a tz-aware UTC datetime, floored to the millisecond (integer math)."""
    return _ms_to_datetime(ts_ns // _NS_PER_MS)


def ns_ceil_ms(ts_ns: int) -> datetime:
    """``ts_ns`` as a tz-aware UTC datetime, ceiled to the millisecond (integer math)."""
    return _ms_to_datetime(-(-ts_ns // _NS_PER_MS))


def datetime_to_ns(value: datetime) -> int:
    """A BSON date (tz-aware, or naive meaning UTC) as integer ns since the epoch."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * _NS_PER_S + delta.microseconds * _NS_PER_US


def _map_value(value: Any) -> float | str | bytes | None:
    """Recorder values onto ``Sample.value``: bool/int -> float, str/bytes raw."""
    if isinstance(value, bool | int):  # bool subclasses int; both become float
        return float(value)
    if isinstance(value, float | str | bytes) or value is None:
        return value
    return str(value)  # other BSON types (Decimal128, ...) surface as text


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Validate and compile a search pattern, or raise ``BadRequest``."""
    if (
        len(pattern) > MAX_PATTERN_LEN
        or not _PATTERN_ALPHABET.fullmatch(pattern)
        or pattern.count("*") > MAX_PATTERN_STARS
    ):
        raise BadRequest(_BAD_PATTERN)
    try:
        return re.compile(pattern)
    except re.error:
        raise BadRequest(_BAD_PATTERN) from None


def _is_valid_channel_name(pv: str) -> bool:
    return bool(pv) and "." not in pv and not pv.startswith("$") and pv not in EXCLUDED_FIELDS


class OspreyMongoRecorderAdapter:
    """Adapter for one OSPREY archiver-recorder collection."""

    type_name: str = "osprey_mongodb_recorder"
    # A polled recorder: every channel lands in each slot, no alarm state, a
    # channel that did not answer is simply absent.
    event_faithful: bool = False

    def __init__(
        self,
        name: str,
        url: str,
        *,
        password: str | None,
        database: str | None,
        collection: str | None,
        timeout_s: float = 30.0,
        caps: dict[str, int] | None = None,
        config_error: str | None = None,
    ) -> None:
        self.name = name
        self._url = url
        self._password = password
        self.database = database
        self.collection_name = collection
        self._timeout_s = timeout_s
        self.caps: dict[str, int] = dict(DEFAULT_CAPS if caps is None else caps)
        self.config_error = config_error
        self._client: AsyncMongoClient | None = None
        self._index: frozenset[str] | None = None
        self._index_at = 0.0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, database={self.database!r}, "
            f"collection={self.collection_name!r})"
        )

    # --- construction ----------------------------------------------------- #

    @classmethod
    def from_config(cls, name: str, cfg: ArchiverInstanceConfig) -> OspreyMongoRecorderAdapter:
        """Build from config; never raises, never connects."""
        database: str | None = None
        collection: str | None = None
        password: str | None = None
        caps = dict(DEFAULT_CAPS)
        problems: list[str] = []

        try:
            database = parse_uri(cfg.url)["database"] or None
        except mongo_errors.PyMongoError as exc:
            problems.append(f"invalid MongoDB URI: {exc}")
        else:
            if database is None:
                problems.append("the URI names no default database")

        raw_collection = cfg.extras.get("collection")
        if isinstance(raw_collection, str) and raw_collection:
            collection = raw_collection
        else:
            problems.append("extras['collection'] is not set")

        if not cfg.api_key_env:
            problems.append("api_key_env is not set (it names the password env var)")
        else:
            password = os.environ.get(cfg.api_key_env) or None
            if password is None:
                problems.append(f"password env var {cfg.api_key_env} is unset or empty")

        for key, default in DEFAULT_CAPS.items():
            raw = cfg.extras.get(key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                problems.append(f"extras['{key}'] must be an integer, got {raw!r}")
                continue
            if value <= 0:
                problems.append(f"extras['{key}'] must be positive, got {value}")
                continue
            caps[key] = value

        config_error = None
        if problems:
            config_error = f"MongoDB archiver '{name}' misconfigured: " + "; ".join(problems)
        return cls(
            name=name,
            url=cfg.url,
            password=password,
            database=database,
            collection=collection,
            timeout_s=cfg.timeout_s,
            caps=caps,
            config_error=config_error,
        )

    def client_options(self) -> dict[str, Any]:
        """Keyword options for ``AsyncMongoClient`` (FR9)."""
        return {
            "password": self._password,
            "serverSelectionTimeoutMS": SERVER_SELECTION_TIMEOUT_MS,
            "timeoutMS": int(self._timeout_s * 1000),
            "tz_aware": True,
        }

    # --- lifecycle -------------------------------------------------------- #

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def health(self) -> AdapterHealth:
        """Ping plus a channel-index load, bounded at 1.5 s."""
        if self.config_error is not None:
            return AdapterHealth(
                healthy=False, last_error=self.config_error, last_check_ts_ns=time.time_ns()
            )
        start = time.monotonic()
        error: str | None = None
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT_S):
                with self._mongo_errors():
                    await self._ping()
                index = await self.channel_index()
            if not index:
                error = f"{self.database}.{self.collection_name} has no dated documents"
        except TimeoutError:
            error = f"MongoDB archiver '{self.name}' did not answer within {HEALTH_TIMEOUT_S} s"
        except ArchiverError as exc:
            error = exc.detail
        return AdapterHealth(
            healthy=error is None,
            latency_ms=(time.monotonic() - start) * 1000,
            last_error=error,
            last_check_ts_ns=time.time_ns(),
        )

    # --- channel index ---------------------------------------------------- #

    async def channel_index(self) -> frozenset[str]:
        """Channel names in the collection, cached for ``INDEX_TTL_S``."""
        self._require_configured()
        now = time.monotonic()
        if self._index is not None and now - self._index_at < INDEX_TTL_S:
            return self._index
        with self._mongo_errors():
            index = await self._query_channel_index()
        if index:  # an empty collection is re-checked on every call
            self._index, self._index_at = index, now
        return index

    async def _query_channel_index(self) -> frozenset[str]:
        coll = self._collection()
        dated = {DATE_FIELD: {"$exists": True}}
        names: set[str] = set()
        for direction in (1, -1):
            doc = await coll.find_one(dated, sort=[(DATE_FIELD, direction)])
            if doc is not None:
                names.update(doc.keys())
        return frozenset(n for n in names - EXCLUDED_FIELDS if _is_valid_channel_name(n))

    async def _require_channel(self, pv: str) -> None:
        """Configured, well-formed and indexed, or raise."""
        self._require_configured()
        if not _is_valid_channel_name(pv):
            raise PVNotFoundInArchiver(f"'{pv}' is not a channel name in archiver '{self.name}'")
        if pv not in await self.channel_index():
            raise PVNotFoundInArchiver(f"PV '{pv}' not in archiver '{self.name}'")

    # --- reads ------------------------------------------ #

    async def fetch_window(
        self,
        pv: str,
        t0_ns: int,
        t1_ns: int,
        *,
        max_points: int | None = None,  # noqa: ARG002 - the adapter contract; never truncated
    ) -> list[Sample]:
        """Raw samples in ``[t0_ns, t1_ns]``; more than ``max_raw`` is a 400, never truncated.

        ``max_points`` is ignored: the service decimates.
        """
        await self._require_channel(pv)
        if t1_ns < t0_ns:
            return []
        max_raw = self.caps["max_raw"]
        docs = await self._find_samples(pv, ns_floor_ms(t0_ns), ns_ceil_ms(t1_ns), 1, max_raw + 1)
        if len(docs) > max_raw:
            raise BadRequest(
                f"more than max_raw={max_raw} raw samples of '{pv}' in the requested window; "
                "narrow the range or request binned data"
            )
        samples = (self._sample(doc, pv) for doc in docs)
        return [s for s in samples if t0_ns <= s.ts_ns <= t1_ns]

    async def fetch_binned(
        self,
        pv: str,
        t0_ns: int,
        t1_ns: int,
        *,
        binning: Literal["auto", "time"],
        target_buckets: int | None,
        bin_size_ns: int | None,
    ) -> list[BinnedPoint]:
        """Buckets of ``[t0_ns, t1_ns]`` anchored on ``t0_ns``, computed in Python.

        ``auto`` clamps the target to ``MAX_BUCKETS`` and uses the width
        ``ceil((t1-t0)/target)``; ``time`` uses ``bin_size_ns`` and is a 400
        above ``MAX_BUCKETS`` buckets. The scan is capped at
        ``max_binned_scan`` documents: more is a 400, never a truncation.
        Empty buckets are omitted; non-numeric samples are not binned.
        """
        await self._require_channel(pv)
        if binning == "auto":
            if target_buckets is None or target_buckets < 1:
                raise BinningNotSupported("binning='auto' requires target_buckets >= 1")
        elif binning == "time":
            if bin_size_ns is None or bin_size_ns < 1:
                raise BinningNotSupported("binning='time' requires bin_size_ns >= 1")
        else:
            raise BinningNotSupported(f"Unsupported binning mode: {binning!r}")
        if t1_ns < t0_ns:
            return []
        width, n = _bucket_grid(t0_ns, t1_ns, binning, target_buckets, bin_size_ns)

        max_scan = self.caps["max_binned_scan"]
        docs = await self._find_samples(pv, ns_floor_ms(t0_ns), ns_ceil_ms(t1_ns), 1, max_scan + 1)
        if len(docs) > max_scan:
            raise BadRequest(
                f"more than max_binned_scan={max_scan} samples of '{pv}' in the requested window; narrow the range"
            )

        buckets: dict[int, list[float]] = {}
        for doc in docs:
            sample = self._sample(doc, pv)
            if not (t0_ns <= sample.ts_ns <= t1_ns) or not isinstance(sample.value, float):
                continue
            index = min((sample.ts_ns - t0_ns) // width, n - 1)
            buckets.setdefault(index, []).append(sample.value)
        return [
            _binned_point(t0_ns + index * width, width, buckets[index]) for index in sorted(buckets)
        ]

    async def fetch_window_budgeted(
        self, pv: str, t0_ns: int, t1_ns: int, *, budget: int = SEED_SAMPLE_BUDGET
    ) -> ArchiveBlock:
        """Seed-path read over ``find_raw_batches`` on the async client.

        The cursor's network waits stay on the event loop; only ``bson.decode_all``
        and the numpy build run in a worker thread. A window over ``max_raw``
        is halved and retried rather than refused.
        """
        await self._require_channel(pv)
        return await fetch_window_budgeted_via(self._fetch_block, pv, t0_ns, t1_ns, budget=budget)

    async def _fetch_block(
        self, pv: str, lo_ns: int, hi_ns: int, include_last_before: bool
    ) -> ArchiveBlock | None:
        max_raw = self.caps["max_raw"]
        query: dict[str, Any] = {
            DATE_FIELD: {"$gte": ns_floor_ms(lo_ns), "$lte": ns_ceil_ms(hi_ns)},
            pv: {"$exists": True},
        }
        block = await self._raw_block(pv, query, 1, max_raw + 1)
        if len(block) > max_raw:
            return None
        mask = (block.ts_ns >= lo_ns) & (block.ts_ns <= hi_ns)
        block = ArchiveBlock(block.ts_ns[mask], block.values[mask], block.severities[mask])
        if include_last_before:
            lookback_ns = self.caps["last_before_lookback_s"] * _NS_PER_S
            prior_query = {
                DATE_FIELD: {"$gte": ns_ceil_ms(lo_ns - lookback_ns), "$lt": ns_ceil_ms(lo_ns)},
                pv: {"$exists": True},
            }
            prior = await self._raw_block(pv, prior_query, -1, 1)
            mask = prior.ts_ns < lo_ns
            if mask.any():
                block = ArchiveBlock.concat(
                    [
                        ArchiveBlock(prior.ts_ns[mask], prior.values[mask], prior.severities[mask]),
                        block,
                    ]
                )
        return block

    async def _raw_block(
        self, pv: str, query: dict[str, Any], direction: int, limit: int
    ) -> ArchiveBlock:
        with self._mongo_errors():
            cursor = (
                self._collection()
                .find_raw_batches(query, {"_id": 0, DATE_FIELD: 1, pv: 1})
                .sort(DATE_FIELD, direction)
                .limit(limit)
            )
            batches = [batch async for batch in cursor]
        return await asyncio.to_thread(_decode_raw_batches, batches, pv)

    async def fetch_last_before(self, pv: str, ts_ns: int) -> Sample | None:
        """Newest sample at or before ``ts_ns`` within ``last_before_lookback_s``, else None."""
        await self._require_channel(pv)
        lookback_ns = self.caps["last_before_lookback_s"] * _NS_PER_S
        docs = await self._find_samples(
            pv, ns_ceil_ms(ts_ns - lookback_ns), ns_floor_ms(ts_ns), -1, 1
        )
        for doc in docs:
            sample = self._sample(doc, pv)
            if sample.ts_ns <= ts_ns:
                return sample
        return None

    async def metadata(self, pv: str) -> PVMetadata:
        """The PV and archiver name; the recorder keeps no other metadata."""
        await self._require_channel(pv)
        return PVMetadata(pv=pv, archiver=self.name)

    async def search(self, pattern: str, *, limit: int) -> list[str]:
        """Sorted channel names that fully match the regex ``pattern``, at most ``limit``."""
        self._require_configured()
        rx = _compile_pattern(pattern)
        return sorted(n for n in await self.channel_index() if rx.fullmatch(n))[:limit]

    # --- internals -------------------------------------------------------- #

    @staticmethod
    def _sample(doc: dict[str, Any], pv: str) -> Sample:
        return Sample(
            ts_ns=datetime_to_ns(doc[DATE_FIELD]),
            value=_map_value(doc[pv]),
            severity=AlarmSeverity.NO_ALARM,
            source="archive",
        )

    async def _find_samples(
        self, pv: str, start: datetime, end: datetime, direction: int, limit: int
    ) -> list[dict[str, Any]]:
        """Documents carrying ``pv`` dated in ``[start, end]``, sorted by date, at most ``limit``."""
        query = {DATE_FIELD: {"$gte": start, "$lte": end}, pv: {"$exists": True}}
        with self._mongo_errors():
            cursor = (
                self._collection()
                .find(query, {"_id": 0, DATE_FIELD: 1, pv: 1})
                .sort(DATE_FIELD, direction)
                .limit(limit)
            )
            rows: list[dict[str, Any]] = await cursor.to_list(None)
            return rows

    def _require_configured(self) -> None:
        if self.config_error is not None:
            raise ArchiverUnavailable(self.config_error)

    def _collection(self):
        if self._client is None:
            self._client = AsyncMongoClient(self._url, **self.client_options())
        assert self.database is not None and self.collection_name is not None
        return self._client[self.database][self.collection_name]

    async def _ping(self) -> None:
        self._collection()  # ensures the client exists
        assert self._client is not None
        assert self.database is not None
        await self._client[self.database].command("ping")

    @contextmanager
    def _mongo_errors(self) -> Iterator[None]:
        try:
            yield
        except (TimeoutError, mongo_errors.PyMongoError) as exc:
            raise map_mongo_error(exc, self.name) from exc


#: Most buckets a binned read returns: the ``auto`` clamp and the ``time`` ceiling.
MAX_BUCKETS = 5_000


def _bucket_grid(
    t0_ns: int,
    t1_ns: int,
    binning: str,
    target_buckets: int | None,
    bin_size_ns: int | None,
) -> tuple[int, int]:
    """Bucket width and count for ``t0_ns <= t1_ns``; a 400 for too many ``time`` buckets."""
    span = t1_ns - t0_ns
    if binning == "auto":
        assert target_buckets is not None
        target = min(int(target_buckets), MAX_BUCKETS)
        width = max(1, -(-span // target))
        return width, min(max(1, -(-(span + 1) // width)), target)
    assert bin_size_ns is not None
    width = int(bin_size_ns)
    n = max(1, -(-(span + 1) // width))
    if n > MAX_BUCKETS:
        raise BadRequest(
            f"binning='time' would produce {n} buckets, more than {MAX_BUCKETS}; "
            "use a larger bin size or a narrower range"
        )
    return width, n


def _binned_point(ts_ns: int, width: int, values: list[float]) -> BinnedPoint:
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((v - mean) ** 2 for v in values) / count
    return BinnedPoint(
        ts_ns=ts_ns,
        bin_ns=width,
        mean=mean,
        std_dev=math.sqrt(variance) if count > 1 else 0.0,
        min=min(values),
        max=max(values),
        count=count,
        severity=AlarmSeverity.NO_ALARM,
    )
