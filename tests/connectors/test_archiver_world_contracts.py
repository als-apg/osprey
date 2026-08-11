"""Four contracts that decide whether a deployment has one archive or two.

A project that serves simulated channels has a live half and a stored half, and
the whole point of seeding the store from the same generators the live half uses
is that the two are one world. That claim is easy to state and easy to lose: a
second epoch conversion, a grid anchored on the seed instant, a scenario rewrite
that forgets a stretch of history — each of them leaves a store that is merely
*plausible* next to what a query would compute. These tests are the acceptance
surface for the claim, checked end to end against a real MongoDB through the
same connectors a deployed agent uses.

* **Equivalence.** A value read out of the store equals, bit for bit, the value
  the mock archiver synthesizes for the same instant — on engine-served channels
  and on procedural ones, and across the hot/tail boundary where the seed grid
  changes density. Once a scenario is applied, no sample inside its event window
  still reads clean: a gap there would be a stretch of calm history sitting
  inside a fault the agent is being asked to diagnose.

* **Retention.** History ages like history. A forced pass of mongod's own TTL
  sweeper takes the aged samples of *both* tiers and leaves the event windows a
  scenario protected standing, because the evidence a fault is diagnosed from
  must never be the thing retention prunes first. The block compressor the
  manifest claims is the one the server actually used.

* **Knob change.** Moving ``retention_days`` in the rendered ``config.yml`` —
  one YAML value, no code — makes the stored fingerprint mismatch, and the
  reseed that follows really does reach further back.

* **Scenario-switch honesty.** Applying an eventful scenario and then an
  eventless one leaves the store bit-identical to its base, read back through
  ``MongoDBArchiverConnector``. The module-level counterpart of this contract is
  ``tests/simulation/test_archiver_scenario_seed.py::
  test_switching_back_leaves_no_trace_of_the_previous_scenario``, which pins the
  same property on the collection directly; this one is the connector's view of
  it, so a divergence that only the read path could introduce has somewhere to
  show up.

Everything here needs Docker and skips cleanly without it. There is no LLM in
this file and no network beyond the container.

The container is this module's own rather than the shared session fixture in
``conftest.py``, for two reasons that are part of what is under test. The block
compressor is a *server start flag* (``--wiredTigerCollectionBlockCompressor``,
exactly as the ``mongodb`` compose template renders it), so a default server
could only confirm mongod's default. And mongod's TTL monitor is switched off at
startup here and turned on for the one test that wants a sweep — every other
anchor in this file is deliberately in the past, and a live sweeper would delete
those stores out from under their assertions.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from osprey.simulation.apply import (
    DENSIFIED_FIELD,
    apply_scenarios,
    event_subwindows,
    event_window,
)
from osprey.simulation.archiver_seed import (
    DATE_FIELD,
    EXPIRE_FIELD,
    MANIFEST_ID,
    SeedKnobs,
    SeedState,
    compare_fingerprint,
    oldest_sample,
    seed_base,
    seed_fingerprint,
    synthesize_documents,
    tier_expiry,
)
from osprey.simulation.procedural import DEFAULT_NOISE_LEVEL
from osprey.simulation.series import epoch_seconds_array
from tests._container_support import is_docker_available, start_or_skip, stop_quietly

# ---------------------------------------------------------------------------
# The world under test
# ---------------------------------------------------------------------------

#: Fixed anchor for every test but the retention one, whose sweep has to be
#: measured against the server's real clock. A failure here reproduces tomorrow,
#: and the odd second is load-bearing: the seed grid is epoch-aligned, so an
#: anchor that happened to fall on a cadence would hide a grid phased on T0.
T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

#: A day of coarse history with an hour of dense, so a container test seeds in
#: about a second. The tier arithmetic does not know how big the numbers are.
KNOBS = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)

#: The block compressor the profile's ``va_archiver`` block defaults to, which
#: the build renders into ``services.mongodb.compression``.
COMPRESSION = "zstd"

#: One noise level for both halves of the world. A deployment renders a single
#: value into the live connector and the seeder alike; stating it once here is
#: what makes the equivalence assertion about the *generators* rather than about
#: two differently-tuned copies of them.
NOISE = DEFAULT_NOISE_LEVEL

# Channels the machine model describes: both halves synthesize these through the
# engine.
CAVITY_TEMP = "SR:RF:CAVITY:01:TEMPERATURE:RB"
CAVITY_POWER = "SR:RF:CAVITY:01:POWER:REV"
CAVITY_VALID = "SR:RF:CAVITY:01:STATUS:VALID"

# Channels it does not: both halves fall back to the procedural generator on the
# taxonomy baseline. Neither ends in ``:SP``/``:RB``, so the seeder's
# boot-value lookup and the mock's baseline-free call resolve to the same
# number — which is the condition under which the two are comparable at all.
BPM_X = "SR:DIAG:BPM:12:POSITION:X"
VAC_PRESSURE = "SR:VAC:IP07:PRESSURE"
READY = "SR:STATUS:READY"

ENGINE_ANALOG = (CAVITY_TEMP, CAVITY_POWER)
PROCEDURAL_ANALOG = (BPM_X, VAC_PRESSURE)
ANALOG = ENGINE_ANALOG + PROCEDURAL_ANALOG
DISCRETE = (CAVITY_VALID, READY)

#: The build-generated channel manifest, in the shape the seeder reads.
CHANNELS: list[dict[str, Any]] = [
    {"address": CAVITY_TEMP, "record_type": "ai"},
    {"address": CAVITY_POWER, "record_type": "ai"},
    {"address": CAVITY_VALID, "record_type": "bi"},
    {"address": BPM_X, "record_type": "ai"},
    {"address": VAC_PRESSURE, "record_type": "ai"},
    {"address": READY, "record_type": "bi"},
]
ADDRESSES = [str(channel["address"]) for channel in CHANNELS]

# The excursions, positioned to make three geometries testable at once. The
# temperature spike sits on the hot/tail boundary, so its window has a coarse
# half and a dense half — where a rewrite that handled one grid and not the
# other would leave clean samples behind. The power spike is offset from it, so
# the two windows OVERLAP without being the same window: a rewrite that decided
# anything per region rather than per channel would smear one channel's event
# across the other's history. The later scenario's spike is far enough after
# both to leave an untouched stretch between them.
SPIKE_WIDTH_S = 120.0
SPIKE_OFFSET_S = -KNOBS.hot_span_s
POWER_OFFSET_S = SPIKE_OFFSET_S + 300
TRIP_OFFSET_S = SPIKE_OFFSET_S + 2400


def _machine() -> dict:
    """A machine model whose channels are the engine-served half of the world."""
    return {
        "name": "Archiver world rig",
        "description": "Three modelled channels; everything else is procedural",
        "channels": {
            CAVITY_TEMP: {
                "value": 23.0,
                "units": "degC",
                "noise": 0.01,
                "description": "Cavity 1 body temperature",
            },
            CAVITY_POWER: {
                "value": 12.0,
                "units": "kW",
                "noise": 0.02,
                "description": "Cavity 1 reflected power",
            },
            CAVITY_VALID: {
                "value": 1.0,
                "units": "",
                "noise": 0.0,
                "description": "Cavity 1 interlock valid",
            },
        },
    }


def _rf_thermal() -> dict:
    """A cavity thermal excursion, in the shape the shipped bundle uses.

    Anchor-relative (``at_offset``) spikes on the cavity's temperature and
    reflected power, sized for a container test rather than for a 30-day
    archive. The shipped ``rf-thermal`` bundle's own composition is pinned by
    ``tests/simulation/test_control_assistant_scenarios.py``; what matters here
    is the mechanism it exercises, not its amplitudes.
    """
    return {
        "description": "A cavity-1 thermal excursion.",
        "archiver": [
            {
                "channel": CAVITY_TEMP,
                "events": [
                    {
                        "shape": "spike",
                        "at_offset": SPIKE_OFFSET_S,
                        "amplitude": 7.0,
                        "width": SPIKE_WIDTH_S,
                    }
                ],
            },
            {
                "channel": CAVITY_POWER,
                "events": [
                    {
                        "shape": "spike",
                        "at_offset": POWER_OFFSET_S,
                        "amplitude": 80.0,
                        "width": SPIKE_WIDTH_S,
                    }
                ],
            },
        ],
    }


def _cavity_trip() -> dict:
    """A second, later excursion — a scenario to switch *to*, not back from.

    Switching between two eventful scenarios is the case a switch back to
    ``nominal`` cannot reach: the two sets' windows are disjoint, so the stretch
    between them belongs to neither and has to end up under ordinary retention
    again. It touches the same channel as ``rf-thermal`` on purpose, so the two
    windows land in one channel's span rather than in two independent ones.
    """
    return {
        "description": "A later cavity trip.",
        "archiver": [
            {
                "channel": CAVITY_TEMP,
                "events": [
                    {
                        "shape": "spike",
                        "at_offset": TRIP_OFFSET_S,
                        "amplitude": 5.0,
                        "width": SPIKE_WIDTH_S,
                    }
                ],
            }
        ],
    }


def _script(scenario: dict, channel: str) -> list[dict]:
    """One channel's event script out of a scenario bundle."""
    for entry in scenario["archiver"]:
        if entry["channel"] == channel:
            return entry["events"]
    raise AssertionError(f"{channel} is not in this scenario")


def _within(windows, moment: float) -> bool:
    """Whether an instant falls inside any of these spans."""
    return any(start <= moment <= end for start, end in windows)


# ---------------------------------------------------------------------------
# Container and project fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongo():
    """A MongoDB started the way the deployment's compose template starts it.

    The compressor flag is the template's, verbatim. ``ttlMonitorEnabled=false``
    is this module's own: see the module docstring for why the sweeper has to be
    off by default and on only where a test asks for it. The one-second sleep is
    what makes that sweep quick when it is enabled — the shipped default is 60s,
    which is a timer, not a contract.
    """
    if not is_docker_available():
        pytest.skip("Docker not available — needed to seed and read a real store.")
    try:
        from testcontainers.mongodb import MongoDbContainer
    except ImportError:
        pytest.skip("testcontainers[mongodb] not installed")

    username, password = "worlduser", "worldpass123"
    command = [
        "--wiredTigerCollectionBlockCompressor",
        COMPRESSION,
        "--setParameter",
        "ttlMonitorEnabled=false",
        "--setParameter",
        "ttlMonitorSleepSecs=1",
    ]
    container = start_or_skip(
        lambda: MongoDbContainer("mongo:7", username=username, password=password).with_command(
            command
        ),
        label="mongodb-world-contracts",
    )
    try:
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(27017)),
            "username": username,
            "password": password,
            "auth_db": "admin",
            "database": "archiver_world_db",
            "collection": "pv_history",
        }
    finally:
        stop_quietly(container)


@pytest.fixture(scope="module")
def mongo_client(mongo):
    """One client for the module, used for setup and for admin commands."""
    from pymongo import MongoClient

    client: Any = MongoClient(
        host=mongo["host"],
        port=mongo["port"],
        username=mongo["username"],
        password=mongo["password"],
        authSource=mongo["auth_db"],
    )
    try:
        yield client
    finally:
        client.close()


class World:
    """A built project wired to a real store, and the two ways to read it back.

    The process never chdirs into the project: every path the code under test
    resolves has to come from the project root or from ``CONFIG_FILE``, not from
    where the caller happens to be.
    """

    def __init__(self, root: Path, collection, store: dict, password_env: str):
        self.root = root
        self.collection = collection
        self._store = store
        self._password_env = password_env

    # -- the project on disk -------------------------------------------------

    @property
    def machine_path(self) -> Path:
        return self.root / "data" / "simulation" / "machine.json"

    def config(self) -> dict:
        """The rendered config, read off disk every time.

        Deliberately not through ``load_config``: the knob-change contract turns
        on re-reading a file that a human just edited, and a cached loader would
        make the test pass for the wrong reason.
        """
        return yaml.safe_load((self.root / "config.yml").read_text())

    def knobs(self) -> SeedKnobs:
        return SeedKnobs.from_config(self.config())

    def fingerprint(self) -> dict:
        return seed_fingerprint(self.knobs(), ADDRESSES, compression=COMPRESSION)

    def set_retention_days(self, days: int) -> None:
        """Edit one value in the rendered config — the whole of a knob change."""
        config = self.config()
        config["va_archiver"]["retention_days"] = days
        (self.root / "config.yml").write_text(yaml.safe_dump(config))

    # -- the engines ---------------------------------------------------------

    def _engine(self, state_dir: Path):
        from osprey.simulation.engine import SimulationEngine

        return SimulationEngine.from_file(self.machine_path, state_dir=state_dir)

    def seed_engine(self):
        """The engine a deploy seeds with: the project's own scenario state."""
        from osprey.simulation.engine import resolve_state_dir

        return self._engine(resolve_state_dir(self.config(), self.root))

    def boot_values(self) -> dict[str, float]:
        """The map the Virtual Accelerator boots its records from.

        Read through the loader ``osprey deploy up`` uses, so the seeded
        procedural baselines are anchored by the same rule the live half applies
        rather than by this test's reading of the machine file.
        """
        from osprey.services.virtual_accelerator.manifest.loaders import (
            load_machine_json_channels,
        )

        return {
            address: entry["value"]
            for address, entry in load_machine_json_channels(self.machine_path).items()
            if "value" in entry
        }

    # -- writing the store ---------------------------------------------------

    def seed(self, t0: datetime, knobs: SeedKnobs | None = None):
        """Build the base archive, exactly as the deploy step builds it."""
        return seed_base(
            self.collection,
            CHANNELS,
            knobs or self.knobs(),
            t0=t0,
            engine=self.seed_engine(),
            boot_values=self.boot_values(),
            noise_level=NOISE,
            compression=COMPRESSION,
            chunk_size=512,
        )

    def reseed(self, t0: datetime):
        """Drop and rebuild — what a fingerprint mismatch provokes."""
        self.collection.drop()
        return self.seed(t0)

    def apply(self, names: list[str], *, at: datetime = T0):
        """Activate a scenario set; the project has no ARIEL, so no logbook."""
        return apply_scenarios(self.root, names, seed_logbook=False, now=at)

    # -- reading it back -----------------------------------------------------

    async def stored(self, channels, start: datetime, end: datetime) -> pd.DataFrame:
        """Read the store through the connector a deployed agent uses."""
        from osprey.connectors.archiver.mongodb_archiver_connector import (
            MongoDBArchiverConnector,
        )

        connector = MongoDBArchiverConnector()
        await connector.connect(
            {
                "host": self._store["host"],
                "port": self._store["port"],
                "name": self._store["database"],
                "collection": self._store["collection"],
                "auth": self._store["auth_db"],
                "username": self._store["username"],
                "password_env": self._password_env,
                "timeout": 20,
            }
        )
        try:
            # precision_ms=0 is full resolution: every stored sample at its own
            # timestamp, with no binning between the store and the assertion.
            return await connector.get_data(
                pv_list=list(channels), start_date=start, end_date=end, precision_ms=0
            )
        finally:
            await connector.disconnect()

    async def mocked(
        self, channels, start: datetime, end: datetime, precision_ms: int
    ) -> pd.DataFrame:
        """Synthesize the same window through the mock archiver.

        The connector is handed no ``simulation_file``: deriving it from the
        control-system config is the shipped behaviour, and a test that named
        the file explicitly would not be reading the same machine model the
        store was seeded from for the same reason a deployment does.
        """
        from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector

        connector = MockArchiverConnector()
        await connector.connect({"noise_level": NOISE, "sample_rate_hz": 1.0})
        try:
            return await connector.get_data(
                pv_list=list(channels),
                start_date=start,
                end_date=end,
                precision_ms=precision_ms,
            )
        finally:
            await connector.disconnect()

    def base_values(self, channels, stamps: list[datetime]) -> dict[str, list]:
        """What the archive holds at these instants with no scenario active.

        Computed through :func:`synthesize_documents`, the single definition of
        an archived value at time T, on an engine that has never had a scenario
        activated — so "clean" here means what the base seed wrote, not what
        this test thinks it wrote.
        """
        nominal_state = self.root / "_nominal_state"
        nominal_state.mkdir(exist_ok=True)
        epochs = epoch_seconds_array(stamps)
        assert epochs is not None
        documents = synthesize_documents(
            CHANNELS,
            epochs,
            engine=self._engine(nominal_state),
            boot_values=self.boot_values(),
            noise_level=NOISE,
        )
        return {pv: [document[pv] for document in documents] for pv in channels}


@pytest.fixture
def world(tmp_path, mongo, mongo_client, monkeypatch):
    """A freshly built project on an empty collection.

    Function-scoped on purpose: the scenario state file lives in the project,
    so a shared project would let one test's active set decide what the next
    test's mock archiver synthesizes.
    """
    root = tmp_path / "world"
    scenarios = root / "data" / "simulation" / "scenarios"
    scenarios.mkdir(parents=True)
    (root / "data" / "simulation" / "machine.json").write_text(json.dumps(_machine()))
    for name, bundle in (("rf-thermal", _rf_thermal()), ("cavity-trip", _cavity_trip())):
        (scenarios / name).mkdir()
        (scenarios / name / "scenario.json").write_text(json.dumps(bundle))

    password_env = "ARCHIVER_WORLD_MONGO_PASSWORD"
    config = {
        "project_name": "archiver-world-contracts",
        "project_root": str(root),
        "control_system": {
            "type": "mock",
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
        },
        "archiver": {
            "type": "mongodb_archiver",
            "mongodb_archiver": {
                "host": mongo["host"],
                "port": mongo["port"],
                "name": mongo["database"],
                "collection": mongo["collection"],
                "auth": mongo["auth_db"],
                "username": mongo["username"],
                "password_env": password_env,
                "timeout": 20,
            },
        },
        "services": {"mongodb": {"compression": COMPRESSION}},
        "va_archiver": {
            "retention_days": KNOBS.retention_days,
            "hot_span_hours": KNOBS.hot_span_hours,
            "hot_cadence_sec": KNOBS.hot_cadence_sec,
            "tail_cadence_sec": KNOBS.tail_cadence_sec,
        },
    }
    (root / "config.yml").write_text(yaml.safe_dump(config))
    (root / ".env").write_text(f"{password_env}={mongo['password']}\n")

    # The mock archiver resolves its machine model, and its scenario state, out
    # of the ambient project config; the MongoDB connector reads its password
    # from the environment by name. Both are set the way a deployment sets them.
    monkeypatch.setenv("CONFIG_FILE", str(root / "config.yml"))
    monkeypatch.setenv(password_env, mongo["password"])

    collection = mongo_client[mongo["database"]][mongo["collection"]]
    collection.drop()
    try:
        yield World(root, collection, mongo, password_env)
    finally:
        collection.drop()


@pytest.fixture
def ttl_sweeper(mongo_client):
    """Turn mongod's own TTL monitor on for one test, and off again after.

    Yields a callable that starts the sweeper. This is a real pass of the
    server's sweeper rather than a delete emulating its predicate: the predicate
    is not the interesting part — whether the documents a scenario protected
    survive one is.
    """

    def start() -> None:
        mongo_client.admin.command("setParameter", 1, ttlMonitorEnabled=True)

    try:
        yield start
    finally:
        mongo_client.admin.command("setParameter", 1, ttlMonitorEnabled=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _aligned(moment: datetime, cadence_s: int) -> datetime:
    """The seeded timestamp at or before ``moment``."""
    epoch = moment.timestamp() // cadence_s * cadence_s
    return datetime.fromtimestamp(epoch, UTC)


def _probe(start: datetime, cadence_s: int, intervals: int) -> tuple[datetime, datetime, int]:
    """A mock query whose own grid lands on seeded timestamps.

    ``MockArchiverConnector`` takes ``int(duration / precision)`` points and
    spreads them evenly from the window's start to its end, so asking for
    exactly ``intervals + 1`` points across ``intervals`` cadences puts every
    one of them on a timestamp the seed grid already holds. The comparison
    asserts the returned stamps really are that grid, so a change to the mock's
    arithmetic fails loudly here instead of quietly comparing nothing.

    Returns ``(start, end, precision_ms)``.
    """
    duration = intervals * cadence_s
    precision_ms, remainder = divmod(1000 * duration, intervals + 1)
    assert remainder == 0, "choose an (intervals, cadence) pair with a whole-ms precision"
    assert intervals + 1 >= 10, "the mock floors a window at ten points"
    return start, start + timedelta(seconds=duration), precision_ms


def _documents_between(collection, start_s: float, end_s: float) -> list[dict]:
    """Stored samples in a span, ascending, read off the collection directly.

    Expiry is not something an archiver query returns — it is not history, it is
    how long history lives — so the retention assertions read the documents
    rather than the connector's frame.
    """
    return list(
        collection.find(
            {
                DATE_FIELD: {
                    "$gte": datetime.fromtimestamp(start_s, UTC),
                    "$lte": datetime.fromtimestamp(end_s, UTC),
                }
            }
        ).sort(DATE_FIELD, 1)
    )


def _series(frame: pd.DataFrame, channel: str) -> pd.Series:
    """One channel's samples out of a canonical long frame, indexed by time."""
    rows = frame[frame["channel"] == channel]
    return pd.Series(rows["value"].to_numpy(), index=pd.DatetimeIndex(rows["timestamp"]))


async def _assert_bit_equal(world: World, channels, start, end, precision_ms) -> int:
    """Every timestamp the mock reports must be in the store, holding its value.

    Returns the number of samples compared, so a caller can refuse a vacuous
    pass.
    """
    stored = await world.stored(channels, start, end)
    mocked = await world.mocked(channels, start, end, precision_ms)

    compared = 0
    for channel in channels:
        stored_series = _series(stored, channel)
        mocked_series = _series(mocked, channel)
        assert not mocked_series.empty, f"the mock synthesized nothing for {channel}"

        missing = mocked_series.index.difference(stored_series.index)
        assert missing.empty, (
            f"{channel}: the probe grid left the seeded grid at {list(missing)[:3]} — "
            "the two halves are being compared at instants only one of them has"
        )
        np.testing.assert_array_equal(
            stored_series.loc[mocked_series.index].to_numpy(dtype=float),
            mocked_series.to_numpy(dtype=float),
            err_msg=f"{channel}: stored history and synthesized history disagree",
        )
        compared += len(mocked_series)
    return compared


# ---------------------------------------------------------------------------
# 1. Equivalence: the store holds what a query would compute
# ---------------------------------------------------------------------------


class TestSeederMockEquivalence:
    """Analog channels only — see :class:`TestServedTypes` for the rest."""

    @pytest.mark.asyncio
    async def test_the_dense_tier_equals_what_the_mock_synthesizes(self, world):
        """Inside the hot span, where the store holds a sample every cadence."""
        world.seed(T0)
        start = _aligned(T0 - timedelta(minutes=30), KNOBS.hot_cadence_sec)

        compared = await _assert_bit_equal(world, ANALOG, *_probe(start, KNOBS.hot_cadence_sec, 9))

        assert compared == len(ANALOG) * 10

    @pytest.mark.asyncio
    async def test_the_coarse_tier_equals_what_the_mock_synthesizes(self, world):
        """Outside the hot span the grid is sparser, and nothing else changes.

        The values are a function of the timestamp alone; a generator that had
        picked up the sample *rate* anywhere would agree in one tier and not the
        other.
        """
        world.seed(T0)
        start = _aligned(T0 - timedelta(hours=6), KNOBS.tail_cadence_sec)

        compared = await _assert_bit_equal(world, ANALOG, *_probe(start, KNOBS.tail_cadence_sec, 9))

        assert compared == len(ANALOG) * 10

    @pytest.mark.asyncio
    async def test_a_window_straddling_the_hot_boundary_equals_it_throughout(self, world):
        """The seam between the two tiers, checked on the cadence they share.

        The coarse cadence is a whole multiple of the dense one, so coarse
        timestamps exist on both sides of the boundary and one probe grid spans
        it. What must not happen is the values changing character where the
        density does.
        """
        world.seed(T0)
        boundary = T0 - timedelta(seconds=KNOBS.hot_span_s)
        start = _aligned(boundary - timedelta(seconds=270), KNOBS.tail_cadence_sec)
        window = _probe(start, KNOBS.tail_cadence_sec, 9)

        compared = await _assert_bit_equal(world, ANALOG, *window)

        assert compared == len(ANALOG) * 10
        # And the window really did straddle it: the seeded store is denser on
        # the recent side, which is the whole reason this window is interesting.
        stored = await world.stored([CAVITY_TEMP], window[0], window[1])
        stamps = _series(stored, CAVITY_TEMP).index
        older = stamps[stamps < boundary]
        newer = stamps[stamps >= boundary]
        assert len(newer) > len(older) > 0, "the probe did not cross the hot/tail boundary"

    @pytest.mark.asyncio
    async def test_no_clean_sample_survives_inside_an_applied_event_window(self, world):
        """A gap here is a stretch of calm history inside an active fault.

        The excursion is positioned on the hot/tail boundary, so its window has
        a coarse half and a dense half; a rewrite that handled one grid and not
        the other would leave clean samples in the other one and nothing else in
        this file would notice.
        """
        world.seed(T0)
        result = world.apply(["rf-thermal"])
        assert result.archiver.skipped is None

        horizon = oldest_sample(world.collection).timestamp()
        start_s, end_s = event_window(_script(_rf_thermal(), CAVITY_TEMP), T0.timestamp(), horizon)
        start = datetime.fromtimestamp(start_s, UTC)
        end = datetime.fromtimestamp(end_s, UTC)
        boundary = T0 - timedelta(seconds=KNOBS.hot_span_s)
        assert start < boundary < end, "the excursion's window must cross the tier boundary"

        stored = await world.stored([CAVITY_TEMP], start, end)
        series = _series(stored, CAVITY_TEMP)
        assert len(series) > 0

        stamps = [stamp.to_pydatetime() for stamp in series.index]
        # Densification steps the same epoch-aligned grid the base seed uses and
        # skips coarse multiples, so an insert can never land on a timestamp the
        # base seed already wrote. One sample per instant is the schema.
        assert len(set(stamps)) == len(stamps), "a dense insert duplicated a base-seed sample"

        clean = world.base_values([CAVITY_TEMP], stamps)[CAVITY_TEMP]
        untouched = [
            stamp
            for stamp, stored_value, clean_value in zip(
                stamps, series.to_numpy(), clean, strict=True
            )
            if stored_value == clean_value
        ]
        assert not untouched, (
            f"{len(untouched)} of {len(stamps)} samples inside the event window still read "
            f"clean, first at {untouched[0]}"
        )

    @pytest.mark.asyncio
    async def test_history_outside_the_event_window_is_untouched(self, world):
        """The other half of the same contract: an event reaches its window and
        no further, so the store still equals the mock everywhere else."""
        world.seed(T0)
        world.apply(["rf-thermal"])
        start = _aligned(T0 - timedelta(hours=12), KNOBS.tail_cadence_sec)

        compared = await _assert_bit_equal(world, ANALOG, *_probe(start, KNOBS.tail_cadence_sec, 9))

        assert compared == len(ANALOG) * 10


class TestServedTypes:
    """Discrete channels, which are outside the equivalence contract above.

    A ``bi`` record is served — and therefore stored — as a boolean, and the
    manifest is what says so. The mock archiver has no manifest: it synthesizes
    from a channel name and a clock, so it cannot know a record type and returns
    a number for these. That is an accepted divergence, not a defect to assert
    around, so what gets pinned is the store's side of it.
    """

    @pytest.mark.asyncio
    async def test_discrete_channels_are_stored_as_the_type_they_are_served_as(self, world):
        world.seed(T0)

        document = world.collection.find_one({DATE_FIELD: {"$exists": True}})

        for channel in DISCRETE:
            assert isinstance(document[channel], bool), (
                f"{channel} is served as a flag; a float here is a value no client "
                f"could ever have read back"
            )

    @pytest.mark.asyncio
    async def test_the_mock_cannot_know_a_record_type_and_says_so_in_numbers(self, world):
        """Documents the divergence rather than papering over it."""
        world.seed(T0)
        start = _aligned(T0 - timedelta(minutes=30), KNOBS.hot_cadence_sec)
        window_start, window_end, precision_ms = _probe(start, KNOBS.hot_cadence_sec, 9)

        frame = await world.mocked([READY], window_start, window_end, precision_ms)

        values = _series(frame, READY)
        assert not values.empty
        assert not any(isinstance(value, bool) for value in values)


# ---------------------------------------------------------------------------
# 2. Retention: history ages, evidence does not
# ---------------------------------------------------------------------------


def _tier(stamp: datetime, knobs: SeedKnobs) -> str:
    """Which tier a stored timestamp belongs to, read off the timestamp."""
    return "coarse" if int(stamp.timestamp()) % knobs.tail_cadence_sec == 0 else "dense"


class TestRetention:
    """A forced pass of mongod's own sweeper, and what it is allowed to take."""

    #: How far the anchor is backdated. Every sample older than
    #: ``anchor - lifetime`` is then already past its expiry when it is written,
    #: which is what gives the sweep something to do in both tiers.
    AGE_S = 1800

    def _anchor(self) -> datetime:
        return datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=self.AGE_S)

    @pytest.mark.asyncio
    async def test_a_ttl_pass_takes_the_aged_tiers_and_leaves_the_evidence(
        self, world, ttl_sweeper
    ):
        anchor = self._anchor()
        world.seed(anchor)
        # The excursion is anchored deep in the coarse stretch that is already
        # past its expiry, so the protection it writes is the only thing
        # standing between those documents and the sweeper.
        expired_offset = -(KNOBS.retention_s - self.AGE_S // 2)
        world.root.joinpath("data/simulation/scenarios/rf-thermal/scenario.json").write_text(
            json.dumps(_aged_rf_thermal(expired_offset))
        )
        result = world.apply(["rf-thermal"], at=anchor)
        assert result.archiver.skipped is None
        assert result.archiver.updated > 0

        collection = world.collection
        protected = list(
            collection.find({EXPIRE_FIELD: {"$exists": False}, DATE_FIELD: {"$exists": True}})
        )
        assert protected, "the scenario protected nothing — there is no contract to test"

        cutoff = datetime.now(UTC)
        doomed = list(collection.find({EXPIRE_FIELD: {"$lte": cutoff}}, {DATE_FIELD: 1}))
        assert doomed, "nothing was due — the anchor is not old enough to age anything"
        tiers = {_tier(doc[DATE_FIELD].replace(tzinfo=UTC), KNOBS) for doc in doomed}
        assert tiers == {"dense", "coarse"}, f"only {tiers} aged; both tiers must"

        ttl_sweeper()
        _await_sweep(collection, [doc["_id"] for doc in doomed])

        surviving = {doc["_id"] for doc in collection.find({}, {"_id": 1})}
        assert {doc["_id"] for doc in protected} <= surviving, (
            "retention pruned the stretch of history the active scenario exists to be "
            "diagnosed from"
        )
        assert MANIFEST_ID in surviving, "the manifest carries no expiry and must survive"

    def test_the_manifest_records_the_compressor_the_server_actually_used(self, world, mongo):
        """The fingerprint's ``compression`` is a claim about the store on disk.

        It is what a knob change compares against, so a value that merely
        travelled through the config without reaching WiredTiger would make
        every reseed decision about the wrong thing.
        """
        world.seed(T0)

        stats = world.collection.database.command("collStats", mongo["collection"])
        creation = stats["wiredTiger"]["creationString"]
        recorded = world.collection.find_one({"_id": MANIFEST_ID})["fingerprint"]["compression"]

        assert f"block_compressor={COMPRESSION}" in creation
        assert recorded == COMPRESSION


def _aged_rf_thermal(at_offset: float) -> dict:
    """The excursion, repositioned into a stretch that has already expired."""
    scenario = _rf_thermal()
    for entry in scenario["archiver"]:
        for event in entry["events"]:
            event["at_offset"] = at_offset
    return scenario


def _await_sweep(collection, ids, timeout_s: float = 90.0) -> None:
    """Block until mongod has removed every one of these documents."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = collection.count_documents({"_id": {"$in": ids}})
        if remaining == 0:
            return
        time.sleep(0.5)
    pytest.fail(
        f"the TTL monitor left {remaining} of {len(ids)} expired documents after {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# 3. A knob change is a config edit, and it reaches the store
# ---------------------------------------------------------------------------


class TestKnobChange:
    def test_moving_retention_in_the_config_mismatches_and_reseeds_deeper(self, world):
        """One YAML value moves; nothing else in the project does.

        The fingerprint is what makes a reseed a reported decision rather than a
        silent rebuild, and coverage is what makes the decision worth taking —
        so both halves are checked, not just the comparison.
        """
        world.seed(T0)
        assert compare_fingerprint(world.collection, world.fingerprint()).state is SeedState.MATCH
        shallow_start = oldest_sample(world.collection)

        world.set_retention_days(KNOBS.retention_days + 2)

        comparison = compare_fingerprint(world.collection, world.fingerprint())
        assert comparison.state is SeedState.MISMATCH
        assert ("retention_days", KNOBS.retention_days, KNOBS.retention_days + 2) in (
            comparison.differences
        )
        assert "retention_days" in comparison.describe()

        world.reseed(T0)

        assert compare_fingerprint(world.collection, world.fingerprint()).state is SeedState.MATCH
        deep_start = oldest_sample(world.collection)
        assert deep_start < shallow_start
        assert (shallow_start - deep_start) > timedelta(days=2) - timedelta(minutes=1)

    def test_an_unchanged_config_does_not_provoke_a_reseed(self, world):
        """The other side of the same knob: a redeploy that changed nothing must
        not rebuild a month of history to arrive back where it started."""
        world.seed(T0)

        assert compare_fingerprint(world.collection, world.fingerprint()).state is SeedState.MATCH


# ---------------------------------------------------------------------------
# 4. Scenario-switch honesty, seen from the connector
# ---------------------------------------------------------------------------


class TestScenarioSwitchHonesty:
    """Apply an eventful set, then an eventless one; the world must be the base.

    ``tests/simulation/test_archiver_scenario_seed.py::
    test_switching_back_leaves_no_trace_of_the_previous_scenario`` pins this on
    the collection itself, document by document. This is the same contract seen
    through ``MongoDBArchiverConnector``, which is where a leftover dense insert
    or a re-typed value would actually reach an agent.
    """

    @pytest.mark.asyncio
    async def test_switching_back_leaves_the_archive_bit_identical(self, world):
        world.seed(T0)
        horizon = oldest_sample(world.collection) - timedelta(seconds=1)
        end = T0 + timedelta(seconds=1)

        base = await world.stored(ADDRESSES, horizon, end)
        assert not base.empty

        world.apply(["rf-thermal"])
        during = await world.stored(ADDRESSES, horizon, end)
        assert not during.equals(base), "the scenario never reached the archive"

        world.apply(["nominal"])
        after = await world.stored(ADDRESSES, horizon, end)

        pd.testing.assert_frame_equal(after, base)
        assert world.collection.count_documents({DENSIFIED_FIELD: True}) == 0

    @pytest.mark.asyncio
    async def test_switching_between_two_scenarios_puts_the_gap_back_under_retention(self, world):
        """The case a switch back to ``nominal`` cannot reach.

        Two eventful sets in a row leave three stretches: the old window, the
        new one, and the calm between them. Only the middle one is subtle. A
        rewrite that decided expiry over the *span* it recomputed — the union of
        the old window and the new one — would strip the stamp from that middle
        stretch and never put it back, because the ledger it writes names only
        the new window and no later apply would come back for it. The result is
        a wedge of history that outlives its retention forever, which is the
        same dishonesty as history that vanishes early, told backwards.

        So the assertion is made per document against the live windows of the
        set now in force: inside one, no stamp; outside every one, the stamp its
        tier says it should have.
        """
        world.seed(T0)
        collection = world.collection
        anchor_s = T0.timestamp()
        horizon = oldest_sample(collection).timestamp()

        old_window = event_window(_script(_rf_thermal(), CAVITY_TEMP), anchor_s, horizon)
        new_window = event_window(_script(_cavity_trip(), CAVITY_TEMP), anchor_s, horizon)
        assert old_window[1] < new_window[0], "the two windows must leave a gap between them"
        span = (old_window[0], new_window[1])
        gap = (old_window[1], new_window[0])
        before = _documents_between(collection, *gap)
        assert before, "there is no history between the windows to put back under retention"

        world.apply(["rf-thermal"])
        world.apply(["cavity-trip"])

        live = {
            pv: event_subwindows(_script(_cavity_trip(), pv), anchor_s, horizon)
            for pv in (CAVITY_TEMP,)
        }
        stranded, mis_stamped = [], []
        for document in _documents_between(collection, *span):
            stamp = document[DATE_FIELD].replace(tzinfo=UTC)
            epoch = stamp.timestamp()
            held = document.get(EXPIRE_FIELD)
            if _within(live[CAVITY_TEMP], epoch):
                assert held is None, f"the live window lost its protection at {stamp}"
                continue
            if held is None:
                stranded.append(stamp)
                continue
            wanted = float(tier_expiry(KNOBS, np.asarray([epoch]))[0])
            if abs(held.replace(tzinfo=UTC).timestamp() - wanted) > 0.001:
                mis_stamped.append(stamp)

        assert not stranded, (
            f"{len(stranded)} document(s) between the two scenarios' windows are still "
            f"unexpiring and no ledger names them again, first at {stranded[0]}"
        )
        assert not mis_stamped, (
            f"{len(mis_stamped)} document(s) came back under an expiry that is not their "
            f"tier's, first at {mis_stamped[0]}"
        )

        # And the stretch is intact, not merely correctly stamped: same
        # timestamps as the base seed wrote, with no dense inserts left behind
        # from the set that is no longer active.
        after = _documents_between(collection, *gap)
        assert [d[DATE_FIELD] for d in after] == [d[DATE_FIELD] for d in before]
        assert not any(d.get(DENSIFIED_FIELD) for d in after)

    @pytest.mark.asyncio
    async def test_the_old_scenarios_window_reads_clean_again(self, world):
        """Switching to another fault has to erase the first one, not stack it."""
        world.seed(T0)
        anchor_s = T0.timestamp()
        horizon = oldest_sample(world.collection).timestamp()
        old_window = event_window(_script(_rf_thermal(), CAVITY_TEMP), anchor_s, horizon)
        start = datetime.fromtimestamp(old_window[0], UTC)
        end = datetime.fromtimestamp(old_window[1], UTC)

        world.apply(["rf-thermal"])
        world.apply(["cavity-trip"])

        series = _series(await world.stored([CAVITY_TEMP], start, end), CAVITY_TEMP)
        stamps = [stamp.to_pydatetime() for stamp in series.index]
        clean = world.base_values([CAVITY_TEMP], stamps)[CAVITY_TEMP]

        np.testing.assert_array_equal(
            series.to_numpy(dtype=float),
            np.asarray(clean, dtype=float),
            err_msg="the previous scenario's excursion is still in the archive",
        )


class TestOverlappingWindows:
    """Two channels whose event windows overlap without being the same window.

    Everything the rewrite does — which values it recomputes, which documents it
    protects, which instants it densifies — has to be decided per channel and
    per instant. Two channels sharing a region is where a decision made once for
    the whole region shows up: one channel's excursion smeared across the
    other's history, or a stretch protected because some *other* channel had an
    event there.
    """

    @pytest.mark.asyncio
    async def test_each_channel_is_rewritten_over_its_own_window_only(self, world):
        world.seed(T0)
        anchor_s = T0.timestamp()
        horizon = oldest_sample(world.collection).timestamp()
        temp = event_subwindows(_script(_rf_thermal(), CAVITY_TEMP), anchor_s, horizon)
        power = event_subwindows(_script(_rf_thermal(), CAVITY_POWER), anchor_s, horizon)
        assert len(temp) == len(power) == 1
        assert temp[0][0] < power[0][0] < temp[0][1] < power[0][1], (
            "the two windows must overlap without being the same window"
        )

        world.apply(["rf-thermal"])

        start = datetime.fromtimestamp(temp[0][0], UTC)
        end = datetime.fromtimestamp(power[0][1], UTC)
        stored = await world.stored([CAVITY_TEMP, CAVITY_POWER], start, end)

        seen = {"temp-only": 0, "power-only": 0, "both": 0}
        for label, channel, windows, other in (
            ("temp-only", CAVITY_TEMP, temp, power),
            ("power-only", CAVITY_POWER, power, temp),
        ):
            series = _series(stored, channel)
            stamps = [stamp.to_pydatetime() for stamp in series.index]
            assert stamps, f"{channel} has no samples across the two windows"
            clean = world.base_values([channel], stamps)[channel]
            for stamp, value, base in zip(stamps, series.to_numpy(), clean, strict=True):
                epoch = stamp.timestamp()
                if not _within(windows, epoch):
                    assert value == base, (
                        f"{channel} was rewritten at {stamp}, which is outside its own window "
                        f"and inside the other channel's"
                    )
                    continue
                assert value != base, f"{channel} was not rewritten at {stamp}"
                seen["both" if _within(other, epoch) else label] += 1

        assert seen["both"] > 0, "the windows never actually overlapped in stored history"
        assert seen["temp-only"] > 0 and seen["power-only"] > 0, (
            "no document fell in exactly one window, so nothing distinguished the two"
        )

    @pytest.mark.asyncio
    async def test_a_dense_insert_carries_only_the_channels_live_at_its_instant(self, world):
        """Sparse documents are legal, so an insert outside a channel's window
        must simply omit it rather than give it invented resolution."""
        world.seed(T0)
        anchor_s = T0.timestamp()
        horizon = oldest_sample(world.collection).timestamp()
        power = event_subwindows(_script(_rf_thermal(), CAVITY_POWER), anchor_s, horizon)

        world.apply(["rf-thermal"])

        inserts = list(world.collection.find({DENSIFIED_FIELD: True}))
        assert inserts, "the coarse half of the window was never densified"
        outside_power = [
            document
            for document in inserts
            if not _within(power, document[DATE_FIELD].replace(tzinfo=UTC).timestamp())
        ]
        assert outside_power, "every insert fell inside both windows — nothing was distinguished"
        for document in outside_power:
            assert CAVITY_TEMP in document
            assert CAVITY_POWER not in document, (
                "an insert outside the power channel's window still carried it"
            )
