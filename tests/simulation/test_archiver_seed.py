"""The base seed: what it writes, what it promises, and what a reader gets back.

Three questions decide whether a deployment's archive is honest, and this file
answers each against something other than the seeder's own opinion:

* **Does the store hold what a query would compute?** The roundtrip tests seed a
  real MongoDB and read it back through ``MongoDBArchiverConnector`` — the same
  connector a deployed agent uses — and compare against the generators directly.
  A seeder that agreed only with itself would pass a mock and fail a deployment.

* **Does history age like history?** The two tiers carry different lifetimes,
  stamped per document from the sample's own time. The expiry tests check the
  stamps rather than waiting on MongoDB's TTL sweeper, which runs on its own
  minute-scale schedule; the sweeper's own behaviour is the retention contract
  test's subject, not this file's.

* **Is a reseed a decision?** The fingerprint tests pin what does and does not
  provoke one. The seed instant is the interesting case: it moves on every
  deploy, so counting it would make every deploy rebuild the store, and
  *forgetting* to exclude it is a mistake nothing else would catch.

Container-backed tests skip cleanly without Docker. The grid, synthesis and
fingerprint tests need no container and carry the contract on their own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from osprey.simulation.archiver_seed import (
    DATE_FIELD,
    EXPIRE_FIELD,
    MANIFEST_ID,
    FingerprintComparison,
    SeedKnobs,
    SeedState,
    compare_fingerprint,
    oldest_sample,
    prepare_collection,
    seed_base,
    seed_fingerprint,
    seed_grid,
    synthesize_documents,
    write_manifest,
)
from osprey.simulation.procedural import baseline_value, generate_series
from osprey.simulation.series import epoch_seconds_array
from tests._container_support import is_docker_available, start_or_skip, stop_quietly

# A fixed anchor: every expectation below is a function of it, and a failure
# should be reproducible tomorrow.
T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

# A small archive — an hour of coarse history with ten minutes of dense — so a
# container test seeds in under a second. The tier *arithmetic* is what is under
# test, and it does not know how big the numbers are.
SMALL = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)

CHANNELS = [
    {"address": "SR:VAC:IP07:PRESSURE", "record_type": "ai"},
    {"address": "SR:RF:CAV01:TEMP:BODY", "record_type": "ai"},
    {"address": "SR:DIAG:BPM:12:POSITION:X", "record_type": "ai"},
    {"address": "SR:STATUS:VALID", "record_type": "bi"},
]
ADDRESSES = [channel["address"] for channel in CHANNELS]


# ---------------------------------------------------------------------------
# Container fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongo_store():
    """A MongoDB container for this module, yielding connection parameters."""
    if not is_docker_available():
        pytest.skip("Docker not available — needed to seed a real store.")
    try:
        from testcontainers.mongodb import MongoDbContainer
    except ImportError:
        pytest.skip("testcontainers[mongodb] not installed")

    username, password = "seeduser", "seedpass123"
    container = start_or_skip(
        lambda: MongoDbContainer("mongo:7", username=username, password=password),
        label="mongodb-seed",
    )
    try:
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(27017)),
            "username": username,
            "password": password,
            "auth_db": "admin",
            "database": "seed_test_db",
            "collection": "pv_history",
        }
    finally:
        stop_quietly(container)


@pytest.fixture
def collection(mongo_store):
    """An empty collection, dropped again afterwards.

    Dropped rather than emptied: the indexes are part of what the seeder
    creates, so a test that inherited them would not be testing a fresh store.
    """
    from pymongo import MongoClient

    client = MongoClient(
        host=mongo_store["host"],
        port=mongo_store["port"],
        username=mongo_store["username"],
        password=mongo_store["password"],
        authSource=mongo_store["auth_db"],
    )
    handle = client[mongo_store["database"]][mongo_store["collection"]]
    handle.drop()
    try:
        yield handle
    finally:
        handle.drop()
        client.close()


@pytest.fixture
def read_back(mongo_store, monkeypatch):
    """Read a window through the connector a deployed agent would use.

    The connector takes its password from the environment by name, as it does
    in a deployment; ``monkeypatch`` is what keeps that variable from outliving
    the test.
    """
    password_env = "SEED_TEST_MONGO_PASSWORD"
    monkeypatch.setenv(password_env, mongo_store["password"])

    async def query(pv_names, start, end):
        from osprey.connectors.archiver.mongodb_archiver_connector import (
            MongoDBArchiverConnector,
        )

        connector = MongoDBArchiverConnector()
        await connector.connect(
            {
                "host": mongo_store["host"],
                "port": mongo_store["port"],
                "name": mongo_store["database"],
                "collection": mongo_store["collection"],
                "auth": mongo_store["auth_db"],
                "username": mongo_store["username"],
                "password_env": password_env,
                "timeout": 10,
            }
        )
        try:
            return await connector.get_data(
                pv_list=list(pv_names), start_date=start, end_date=end, precision_ms=0
            )
        finally:
            await connector.disconnect()

    return query


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


class TestSeedGrid:
    """One grid, two densities, no timestamp written twice."""

    def test_the_window_reaches_back_the_full_retention(self):
        times, _ = seed_grid(SeedKnobs(), T0)

        start = datetime.fromtimestamp(float(times[0]), UTC)
        end = datetime.fromtimestamp(float(times[-1]), UTC)
        assert T0 - end < timedelta(minutes=1)
        assert timedelta(days=29, hours=23) < T0 - start <= timedelta(days=30)

    def test_no_timestamp_appears_twice(self):
        """The schema is one document per timestamp; a duplicate would be a
        second, competing sample for the same instant."""
        times, _ = seed_grid(SeedKnobs(), T0)

        assert len(np.unique(times)) == len(times)

    def test_the_coarse_tier_is_a_subset_of_the_dense_grid(self):
        """Both tiers land on multiples of their cadence since the epoch.

        Anchoring on ``t0`` instead would put the tiers on interleaved
        timestamps whenever the seed instant was not itself a whole cadence —
        which is every deploy but a lucky one.
        """
        knobs = SeedKnobs()
        times, _ = seed_grid(knobs, T0)

        assert np.all(np.mod(times, knobs.hot_cadence_sec) == 0)
        assert T0.second != 0, "the anchor must not accidentally be cadence-aligned"

    def test_the_tiers_have_the_declared_cadences(self):
        knobs = SeedKnobs()
        times, _ = seed_grid(knobs, T0)
        gaps = set(np.diff(times).tolist())

        assert gaps == {float(knobs.hot_cadence_sec), float(knobs.tail_cadence_sec)}

    def test_each_tier_carries_its_own_lifetime(self):
        """Measured from the sample's own time, so a seeded archive ages like a
        recorded one instead of surviving intact and then vanishing at once."""
        knobs = SeedKnobs()
        times, expiry = seed_grid(knobs, T0)
        dense = np.mod(times, knobs.tail_cadence_sec) != 0

        assert np.all(expiry[dense] - times[dense] == knobs.hot_span_s)
        assert np.all(expiry[~dense] - times[~dense] == knobs.retention_s)

    def test_the_oldest_dense_samples_are_already_at_the_end_of_their_life(self):
        """A dense sample from the start of the dense span expires about at t0.

        This is the property that keeps steady state bounded: the seed does not
        hand a fresh deployment 48 hours of dense history that then lives 48
        hours longer than it should. "About" is one coarse cadence — the oldest
        *dense-only* timestamp is the first cadence-aligned point inside the
        span that the coarse tier does not already own.
        """
        knobs = SeedKnobs()
        times, expiry = seed_grid(knobs, T0)
        dense = np.mod(times, knobs.tail_cadence_sec) != 0

        assert expiry[dense].min() <= _epoch(T0) + knobs.tail_cadence_sec

    def test_a_shallow_archive_still_produces_a_grid(self):
        """The CI lanes run with retention cut to hours, not a month."""
        times, _ = seed_grid(SMALL, T0)

        assert len(times) > 0
        assert len(np.unique(times)) == len(times)


class TestSeedKnobs:
    """Knobs the grid arithmetic cannot honour are refused, not approximated."""

    def test_defaults_match_the_shipped_profile_block(self):
        """The two default sets are written out separately — in the profile
        block and here — so a deployment seeded without a config gets the
        archive the profile describes rather than a second opinion about it."""
        from osprey.cli.build_profile_archiver import VAArchiverConfig

        block = VAArchiverConfig()
        knobs = SeedKnobs()

        assert knobs.retention_days == block.retention_days
        assert knobs.hot_span_hours == block.hot_span_hours
        assert knobs.hot_cadence_sec == block.hot_cadence_sec
        assert knobs.tail_cadence_sec == block.tail_cadence_sec

    def test_from_config_reads_the_va_archiver_subtree(self):
        knobs = SeedKnobs.from_config(
            {"va_archiver": {"retention_days": 2, "hot_span_hours": 6, "port_host": 27017}}
        )

        assert knobs.retention_days == 2
        assert knobs.hot_span_hours == 6
        # Knobs this module does not use are simply not its business.
        assert knobs.hot_cadence_sec == SeedKnobs().hot_cadence_sec

    def test_a_config_without_the_block_seeds_the_default_archive(self):
        assert SeedKnobs.from_config({}) == SeedKnobs()

    @pytest.mark.parametrize(
        "block",
        [
            {"retention_days": 0},
            {"hot_cadence_sec": -1},
            {"retention_days": 1, "hot_span_hours": 25},
            {"hot_cadence_sec": 7, "tail_cadence_sec": 60},
            {"retention_days": "30"},
            {"retention_days": True},
        ],
    )
    def test_inconsistent_knobs_are_refused(self, block):
        with pytest.raises(ValueError):
            SeedKnobs.from_config({"va_archiver": block})

    def test_a_non_dividing_cadence_names_both_knobs(self):
        """The fix needs both numbers; naming one leaves the reader guessing."""
        with pytest.raises(ValueError, match="tail_cadence_sec.*hot_cadence_sec"):
            SeedKnobs(hot_cadence_sec=7, tail_cadence_sec=60).validate()


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeDocuments:
    """What a document holds, and why it is that and not something plausible."""

    def test_a_document_carries_every_channel_at_its_own_timestamp(self):
        times, _ = seed_grid(SMALL, T0)
        documents = synthesize_documents(CHANNELS, times[:5])

        assert len(documents) == 5
        for document, epoch in zip(documents, times[:5], strict=True):
            assert document[DATE_FIELD] == datetime.fromtimestamp(float(epoch), UTC)
            assert set(document) == {DATE_FIELD, *ADDRESSES}

    def test_analog_values_equal_the_generator_at_the_same_instant(self):
        """The point of the whole design: a stored document and a live query
        are one computation, not two that usually agree."""
        pv = "SR:VAC:IP07:PRESSURE"
        times, _ = seed_grid(SMALL, T0)
        documents = synthesize_documents(CHANNELS, times[:32])

        expected = generate_series(pv, times[:32], baseline=baseline_value(pv))
        assert [document[pv] for document in documents] == expected.tolist()

    def test_values_are_computed_at_the_timestamp_actually_stored(self):
        """A datetime carries microseconds, so a sub-microsecond grid time would
        be stored as one instant and valued at another. Seeding a grid finer
        than a microsecond is the only way to catch that, and a seeder for a
        higher-rate facility would do exactly that."""
        pv = "SR:VAC:IP07:PRESSURE"
        epoch = np.array([1_800_000_000.0 + 1e-9 * i for i in range(4)])

        documents = synthesize_documents(CHANNELS, epoch)

        stored = epoch_seconds_array([document[DATE_FIELD] for document in documents])
        expected = generate_series(pv, stored, baseline=baseline_value(pv))
        assert [document[pv] for document in documents] == expected.tolist()

    def test_a_flag_channel_is_stored_as_a_flag(self):
        """The VA serves these as booleans. A float on a status channel is a
        value no client could ever have read back — and once archived, it is
        indistinguishable from one that was."""
        times, _ = seed_grid(SMALL, T0)
        documents = synthesize_documents(CHANNELS, times[:8])

        values = {document["SR:STATUS:VALID"] for document in documents}
        assert values == {True}
        assert all(isinstance(document["SR:STATUS:VALID"], bool) for document in documents)

    def test_a_modelless_flag_does_not_move(self):
        """Nothing drives it — the manifest forbids declaring these noisy, and
        the VA serves a bare coerced baseline on every tick."""
        channels = [{"address": "SR:STATUS:READY", "record_type": "bi"}]
        times, _ = seed_grid(SMALL, T0)

        documents = synthesize_documents(channels, times)

        assert len({document["SR:STATUS:READY"] for document in documents}) == 1

    def test_boot_values_anchor_the_procedural_baseline(self):
        """The seeder holds the machine model; without threading it through, a
        setpoint would be archived at the taxonomy's guess for its name."""
        pv = "SR:MAG:HCM:01:CURRENT:SP"
        channels = [{"address": pv, "record_type": "ai"}]
        times, _ = seed_grid(SMALL, T0)

        anchored = synthesize_documents(channels, times[:16], boot_values={pv: 0.25})
        unanchored = synthesize_documents(channels, times[:16])

        assert np.mean([d[pv] for d in anchored]) == pytest.approx(0.25, abs=0.05)
        assert np.mean([d[pv] for d in unanchored]) == pytest.approx(150.0, rel=0.1)

    def test_an_empty_window_produces_no_documents(self):
        assert synthesize_documents(CHANNELS, np.empty(0)) == []


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


class TestFingerprint:
    """What provokes a reseed, and what must not."""

    def _fingerprint(self, knobs=SMALL, addresses=ADDRESSES, compression="zstd"):
        return seed_fingerprint(knobs, addresses, compression=compression)

    def test_the_channel_set_is_hashed_order_independently(self):
        """The manifest and the config can list channels in any order; only
        membership changes what the store contains."""
        forwards = self._fingerprint(addresses=ADDRESSES)
        backwards = self._fingerprint(addresses=list(reversed(ADDRESSES)))

        assert forwards == backwards

    def test_adding_a_channel_changes_the_fingerprint(self):
        assert self._fingerprint() != self._fingerprint(addresses=[*ADDRESSES, "SR:NEW:CHANNEL"])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("retention_days", 2),
            ("hot_span_hours", 2),
            ("hot_cadence_sec", 5),
            ("tail_cadence_sec", 120),
        ],
    )
    def test_every_knob_that_changes_coverage_changes_the_fingerprint(self, field, value):
        changed = SeedKnobs(**{**SMALL.__dict__, field: value})

        assert self._fingerprint(knobs=changed) != self._fingerprint()

    def test_compression_is_part_of_it(self):
        """It does not change which samples are stored, but it changes the size
        of the store on disk — which is the budget the knob exists to hold."""
        assert self._fingerprint(compression="snappy") != self._fingerprint()


class TestFingerprintComparison:
    """Match, mismatch, absent — and the trap in between."""

    def test_an_unseeded_store_reports_absent(self, collection):
        assert compare_fingerprint(collection, self._current()).state is SeedState.ABSENT

    def test_a_store_seeded_with_the_same_knobs_matches(self, collection):
        write_manifest(collection, self._current(), seeded_at=T0)

        comparison = compare_fingerprint(collection, self._current())

        assert comparison.state is SeedState.MATCH
        assert comparison.differences == ()

    def test_a_later_deploy_of_an_unchanged_profile_still_matches(self, collection):
        """The trap: the seed instant moves on every deploy. Counting it would
        make every deploy rebuild the store, and the mistake would look like
        working code — a reseed is not an error, just hours of wasted CI."""
        write_manifest(collection, self._current(), seeded_at=T0)

        comparison = compare_fingerprint(collection, self._current())

        assert comparison.state is SeedState.MATCH
        assert comparison.seeded_at is not None

    def test_a_changed_knob_reports_what_moved(self, collection):
        write_manifest(collection, self._current(), seeded_at=T0)

        comparison = compare_fingerprint(
            collection, seed_fingerprint(SeedKnobs(retention_days=7), ADDRESSES, compression="zstd")
        )

        assert comparison.state is SeedState.MISMATCH
        moved = {key for key, _, _ in comparison.differences}
        assert "retention_days" in moved
        assert "7" in comparison.describe()

    def test_a_manifest_from_an_older_version_mismatches_readably(self, collection):
        """A field the stored manifest never had reads as ``None`` rather than
        being skipped — skipping would let an older store pass as current."""
        collection.replace_one(
            {"_id": MANIFEST_ID},
            {"_id": MANIFEST_ID, "fingerprint": {"retention_days": SMALL.retention_days}},
            upsert=True,
        )

        comparison = compare_fingerprint(collection, self._current())

        assert comparison.state is SeedState.MISMATCH
        assert any(stored is None for _, stored, _ in comparison.differences)

    def test_the_manifest_starts_with_an_empty_ledger(self, collection):
        """The scenario rewrite owns it; a fresh base has touched nothing."""
        write_manifest(collection, self._current(), seeded_at=T0)

        assert collection.find_one({"_id": MANIFEST_ID})["touched_windows"] == []

    @staticmethod
    def _current():
        return seed_fingerprint(SMALL, ADDRESSES, compression="zstd")


# ---------------------------------------------------------------------------
# Writing to a real store
# ---------------------------------------------------------------------------


class TestSeedBase:
    """The whole run against MongoDB, read back through the real connector."""

    def test_it_writes_one_document_per_grid_timestamp(self, collection):
        times, _ = seed_grid(SMALL, T0)

        report = seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        assert report.documents == len(times)
        assert collection.count_documents({DATE_FIELD: {"$exists": True}}) == len(times)
        assert report.channels == len(CHANNELS)

    def test_the_indexes_the_store_cannot_work_without_exist(self, collection):
        """An unindexed sort on ``date`` is capped at 32 MB of documents in
        memory, which a real window crosses long before a test does."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        indexes = collection.index_information()
        keyed = {tuple(spec["key"][0]) for spec in indexes.values()}
        assert (DATE_FIELD, 1) in keyed
        ttl = [spec for spec in indexes.values() if "expireAfterSeconds" in spec]
        assert len(ttl) == 1
        assert ttl[0]["expireAfterSeconds"] == 0
        assert tuple(ttl[0]["key"][0]) == (EXPIRE_FIELD, 1)

    def test_documents_carry_their_tier_s_expiry(self, collection):
        """Checked on the stamps, not by waiting: MongoDB's TTL sweeper runs on
        its own minute-scale schedule, so a test that waited for it would be
        testing the sweeper's timer."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        for document in collection.find({DATE_FIELD: {"$exists": True}}):
            stored = document[DATE_FIELD].replace(tzinfo=UTC)
            lifetime = (document[EXPIRE_FIELD].replace(tzinfo=UTC) - stored).total_seconds()
            dense = int(stored.timestamp()) % SMALL.tail_cadence_sec != 0
            assert lifetime == (SMALL.hot_span_s if dense else SMALL.retention_s)

    def test_the_manifest_is_written_and_is_invisible_to_an_archiver_read(self, collection):
        """It lives in the sample collection, which is only safe because it has
        no ``date``: the connector's window filter requires one, so no query can
        ever return it as a sample."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        manifest = collection.find_one({"_id": MANIFEST_ID})
        assert manifest is not None
        assert DATE_FIELD not in manifest
        assert (
            compare_fingerprint(
                collection, seed_fingerprint(SMALL, ADDRESSES, compression="zstd")
            ).state
            is SeedState.MATCH
        )

    def test_a_half_written_store_reads_as_absent(self, collection):
        """The manifest is written last on purpose. A run that dies partway
        leaves no manifest, so the next deploy rebuilds instead of trusting a
        store that is missing most of its history."""
        times, _ = seed_grid(SMALL, T0)
        prepare_collection(collection)
        collection.insert_many(synthesize_documents(CHANNELS, times[:10]))

        comparison = compare_fingerprint(
            collection, seed_fingerprint(SMALL, ADDRESSES, compression="zstd")
        )

        assert comparison.state is SeedState.ABSENT

    def test_progress_is_reported_per_chunk(self, collection):
        """A three-minute step with no output reads as a hang."""
        seen: list[int] = []

        report = seed_base(
            collection,
            CHANNELS,
            SMALL,
            t0=T0,
            chunk_size=64,
            progress=lambda r: seen.append(r.documents),
        )

        assert len(seen) == report.chunks > 1
        assert seen == sorted(seen)
        assert seen[-1] == report.documents

    def test_the_oldest_sample_is_the_start_of_the_window(self, collection):
        """What honest metadata reports as the start of coverage — no more
        'archived since the year 2000'."""
        times, _ = seed_grid(SMALL, T0)
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        assert oldest_sample(collection) == datetime.fromtimestamp(float(times[0]), UTC)

    def test_an_empty_store_has_no_oldest_sample(self, collection):
        prepare_collection(collection)

        assert oldest_sample(collection) is None

    @pytest.mark.parametrize("workers", [1, 4])
    def test_the_worker_count_does_not_change_what_is_stored(self, collection, workers):
        """Inserts are unordered and overlap the synthesis; concurrency is a
        throughput decision and must not be a correctness one."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=32, workers=workers)

        times, _ = seed_grid(SMALL, T0)
        assert collection.count_documents({DATE_FIELD: {"$exists": True}}) == len(times)
        pv = "SR:VAC:IP07:PRESSURE"
        document = collection.find_one({DATE_FIELD: {"$exists": True}}, sort=[(DATE_FIELD, 1)])
        assert (
            document[pv]
            == generate_series(pv, np.asarray([float(times[0])]), baseline=baseline_value(pv))[0]
        )


class TestConnectorRoundtrip:
    """Read the seeded store back the way a deployed agent does."""

    @pytest.mark.asyncio
    async def test_a_query_returns_the_values_the_generator_computes(self, collection, read_back):
        """The end-to-end claim: history in the store equals history a query
        would have synthesized, through the real connector and its real schema."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)
        pv = "SR:VAC:IP07:PRESSURE"
        window_end = T0.replace(second=0, microsecond=0)
        window_start = window_end - timedelta(minutes=5)

        frame = await read_back([pv], window_start, window_end)

        rows = frame[frame["channel"] == pv]
        assert len(rows) > 0
        stamps = epoch_seconds_array(list(rows["timestamp"]))
        expected = generate_series(pv, stamps, baseline=baseline_value(pv))
        assert np.allclose(rows["value"].to_numpy(), expected, rtol=0, atol=0)

    @pytest.mark.asyncio
    async def test_a_window_before_the_archive_starts_returns_nothing(self, collection, read_back):
        """Honest emptiness. The mock archiver's ten-point floor is exactly the
        fabrication a stored archive exists to replace."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)
        ancient = T0 - timedelta(days=400)

        frame = await read_back(["SR:VAC:IP07:PRESSURE"], ancient, ancient + timedelta(hours=1))

        assert len(frame) == 0

    @pytest.mark.asyncio
    async def test_a_channel_outside_the_seeded_set_returns_nothing(self, collection, read_back):
        """Sparse documents are legal in this schema, so an unseeded channel
        contributes no samples rather than an error or a plausible series."""
        seed_base(collection, CHANNELS, SMALL, t0=T0, chunk_size=64)

        frame = await read_back(["SR:NOT:SEEDED"], T0 - timedelta(minutes=10), T0)

        assert len(frame) == 0


def _epoch(moment: datetime) -> float:
    return moment.timestamp()


def test_the_module_never_imports_pymongo():
    """An optional dependency has to stay optional.

    The seeder is imported by deployment code that runs on hosts where the
    ``archiver-mongodb`` extra was never selected — the fingerprint check has to
    be able to say "no store here" without one.
    """
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys, osprey.simulation.archiver_seed as m;"
            "assert 'pymongo' not in sys.modules, sorted(sys.modules);"
            "print('CLEAN')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CLEAN" in result.stdout


def test_the_manifest_document_is_json_serializable_apart_from_its_dates():
    """The fingerprint is stored verbatim and compared field by field, so every
    value in it has to be a plain scalar rather than something BSON encodes by
    its own rules."""
    fingerprint = seed_fingerprint(SMALL, ADDRESSES, compression="zstd")

    assert json.loads(json.dumps(fingerprint)) == fingerprint


def test_a_comparison_describes_nothing_when_nothing_moved():
    assert FingerprintComparison(SeedState.MATCH).describe() == ""
