"""Rewriting stored history when the active scenario changes.

A stored archive is the deployment's memory, and activating a scenario has to
change what that memory says — otherwise the agent reads a month of calm
history while the live machine shows a fault, which is exactly the two-world
split this feature exists to close. Rewriting stored data is also the most
destructive thing ``osprey sim apply`` does, so the contracts are sharp:

* **Switching away leaves nothing behind.** Apply a scenario with events, then
  apply an eventless one, and the store has to equal the base everywhere — same
  values, same expiry stamps, no leftover dense inserts. A stale event window
  surviving a scenario switch would be a fault the agent can find and the
  operator did not cause, and it is the failure this file's headline test
  exists to catch.

* **Only the event windows move.** Channels no scenario touches, and stretches
  of time outside an event's reach, are not rewritten at all.

* **The credentials come from the project, never from the ambient shell.** An
  apply is routinely run from a temporary directory or another project, and
  picking up a stray ``MONGO_ROOT_PASSWORD`` would at best fail confusingly and
  at worst rewrite a different deployment's archive.

Container-backed tests skip cleanly without Docker; the window arithmetic and
the config resolution carry their own weight without one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from osprey.simulation.apply import (
    DENSIFIED_FIELD,
    active_archiver_events,
    apply_scenarios,
    archiver_store_config,
    event_subwindows,
    event_window,
    persisted_scenario_anchor,
    preflight_archive_rewrite,
)
from osprey.simulation.archiver_seed import (
    DATE_FIELD,
    EXPIRE_FIELD,
    MANIFEST_ID,
    SeedKnobs,
    seed_base,
)
from tests._container_support import is_docker_available, start_or_skip, stop_quietly

PRESSURE = "SR:VAC:IP07:PRESSURE"
TEMPERATURE = "SR:RF:CAV01:TEMP:BODY"
FAULT = "SR:VAC:IP07:STATUS:FAULT"

T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

# Two hours of coarse history with ten minutes of dense, so a container test
# seeds in well under a second. The event sits inside the coarse stretch on
# purpose: densification is only interesting where the grid is coarse.
KNOBS = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)

SPIKE_OFFSET_S = -7200  # two hours before the anchor: coarse-grid territory
LATE_OFFSET_S = -5400  # ninety minutes back: still coarse, and clear of the first
SPIKE_WIDTH_S = 120.0
# Four sigmas either side, matching the rewrite's own cut (see ``event_window``).
SPIKE_REACH_S = 4 * SPIKE_WIDTH_S

CHANNELS = [
    {"address": PRESSURE, "record_type": "ai"},
    {"address": TEMPERATURE, "record_type": "ai"},
    {"address": FAULT, "record_type": "bi"},
]


def _spike(offset: float, amplitude: float = 5e-9) -> dict:
    return {
        "shape": "spike",
        "at_offset": offset,
        "amplitude": amplitude,
        "width": SPIKE_WIDTH_S,
    }


def _machine() -> dict:
    """A three-channel model with one eventful scenario per shape worth testing."""
    return {
        "name": "Rewrite rig",
        "description": "Three channels and a scenario per event shape",
        "channels": {
            PRESSURE: {
                "value": 1e-9,
                "units": "Torr",
                "noise": 0.0,
                "description": "Ion pump pressure",
            },
            TEMPERATURE: {
                "value": 25.0,
                "units": "degC",
                "noise": 0.0,
                "description": "Cavity body temperature",
            },
            FAULT: {
                "value": 0,
                "units": "",
                "noise": 0.0,
                "description": "Ion pump fault flag",
            },
        },
        "scenarios": {
            "nominal": {"description": "All systems nominal."},
            "burst": {
                "description": "A vacuum burst two hours ago.",
                "archiver": [{"channel": PRESSURE, "events": [_spike(SPIKE_OFFSET_S)]}],
            },
            "late-burst": {
                "description": "The same channel, disturbed at a different hour.",
                "archiver": [{"channel": PRESSURE, "events": [_spike(LATE_OFFSET_S)]}],
            },
            "twin-burst": {
                "description": "Two channels whose windows overlap but do not coincide.",
                "archiver": [
                    {
                        "channel": PRESSURE,
                        "events": [_spike(SPIKE_OFFSET_S), _spike(LATE_OFFSET_S)],
                    },
                    {"channel": TEMPERATURE, "events": [_spike(LATE_OFFSET_S, amplitude=3.0)]},
                ],
            },
            "flagged": {
                "description": "A flag channel raised for a while.",
                "archiver": [{"channel": FAULT, "events": [_spike(SPIKE_OFFSET_S, amplitude=1.0)]}],
            },
            "nightly": {
                "description": "A disturbance that recurs at the same time every day.",
                "archiver": [
                    {
                        "channel": PRESSURE,
                        "events": [
                            {
                                "shape": "spike",
                                "at_time": "03:00:00",
                                "amplitude": 5e-9,
                                "width": SPIKE_WIDTH_S,
                            }
                        ],
                    }
                ],
            },
        },
    }


@pytest.fixture(scope="module")
def mongo_store():
    """A MongoDB container for this module."""
    if not is_docker_available():
        pytest.skip("Docker not available — needed to rewrite a real store.")
    try:
        from testcontainers.mongodb import MongoDbContainer
    except ImportError:
        pytest.skip("testcontainers[mongodb] not installed")

    username, password = "rewriteuser", "rewritepass123"
    container = start_or_skip(
        lambda: MongoDbContainer("mongo:7", username=username, password=password),
        label="mongodb-rewrite",
    )
    try:
        yield {
            "host": container.get_container_host_ip(),
            "port": int(container.get_exposed_port(27017)),
            "username": username,
            "password": password,
            "database": "rewrite_db",
            "collection": "pv_history",
        }
    finally:
        stop_quietly(container)


def _write_project(root: Path, store: dict | None, *, password: str | None) -> Path:
    """Lay out a built project on disk: model, config, and its ``.env``."""
    (root / "data" / "simulation").mkdir(parents=True, exist_ok=True)
    (root / "data" / "simulation" / "machine.json").write_text(json.dumps(_machine()))

    config: dict = {
        "project_name": "rewrite-project",
        "project_root": str(root),
        "control_system": {
            "type": "mock",
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
        },
        "va_archiver": {
            "retention_days": KNOBS.retention_days,
            "hot_span_hours": KNOBS.hot_span_hours,
            "hot_cadence_sec": KNOBS.hot_cadence_sec,
            "tail_cadence_sec": KNOBS.tail_cadence_sec,
        },
    }
    if store is not None:
        config["archiver"] = {
            "type": "mongodb_archiver",
            "mongodb_archiver": {
                "host": store["host"],
                "port": store["port"],
                "name": store["database"],
                "collection": store["collection"],
                "auth": "admin",
                "username": store["username"],
                "password_env": "MONGO_ROOT_PASSWORD",
                "timeout": 10,
            },
        }
    (root / "config.yml").write_text(yaml.safe_dump(config))
    if password is not None:
        (root / ".env").write_text(f"MONGO_ROOT_PASSWORD={password}\n")
    return root


def _engine(root: Path):
    """The project's engine, resolved exactly as the product resolves it."""
    from osprey.simulation.engine import SimulationEngine, resolve_state_dir

    config = yaml.safe_load((root / "config.yml").read_text())
    return SimulationEngine.from_file(
        root / "data" / "simulation" / "machine.json",
        state_dir=resolve_state_dir(config, root),
    )


@pytest.fixture
def project(tmp_path, mongo_store):
    """A built project wired to a freshly seeded archive.

    The process never chdirs into it: every path the code under test resolves
    has to come from ``project_dir``, not from where the caller happens to be.
    """
    from pymongo import MongoClient

    root = _write_project(tmp_path / "proj", mongo_store, password=mongo_store["password"])

    client = MongoClient(
        host=mongo_store["host"],
        port=mongo_store["port"],
        username=mongo_store["username"],
        password=mongo_store["password"],
        authSource="admin",
    )
    collection = client[mongo_store["database"]][mongo_store["collection"]]
    collection.drop()
    # Seeded through the same engine the rewrite will use. This is what a
    # deployment does, and it is load-bearing rather than incidental: a base
    # seeded procedurally for a channel the machine model describes would
    # disagree with every later recompute, and the disagreement would look
    # exactly like a scenario that failed to restore.
    seed_base(collection, CHANNELS, KNOBS, t0=T0, chunk_size=256, engine=_engine(root))
    try:
        yield root, collection
    finally:
        collection.drop()
        client.close()


def _snapshot(collection) -> dict:
    """Every stored sample, keyed by timestamp, for exact before/after compares."""
    return {
        doc[DATE_FIELD]: {key: value for key, value in doc.items() if key not in ("_id",)}
        for doc in collection.find({DATE_FIELD: {"$exists": True}})
    }


def _apply(root: Path, names: list[str], *, at: datetime = T0):
    """Activate a set with the logbook left alone (no ARIEL in this project)."""
    return apply_scenarios(root, names, seed_logbook=False, now=at)


def _window(offset: float, *, shrink: float = 0.0) -> tuple[datetime, datetime]:
    """The absolute bounds of one spike's window, optionally pulled in a little.

    Shrinking is for assertions about the *interior* of a window: the grid is
    epoch-aligned and the window edges are not, so the outermost grid point of a
    window is a boundary case that says nothing about the property under test.
    """
    centre = T0 + timedelta(seconds=offset)
    reach = timedelta(seconds=SPIKE_REACH_S - shrink)
    return centre - reach, centre + reach


def _stamps_of(collection, channel: str, low: datetime, high: datetime) -> list[float]:
    """Ascending epoch seconds of every stored sample carrying ``channel``."""
    return sorted(
        document[DATE_FIELD].replace(tzinfo=UTC).timestamp()
        for document in collection.find(
            {DATE_FIELD: {"$gte": low, "$lte": high}, channel: {"$exists": True}}
        )
    )


def _gaps(collection, channel: str, low: datetime, high: datetime) -> list[float]:
    """The intervals between consecutive samples of one channel over a stretch."""
    stamps = _stamps_of(collection, channel, low, high)
    return [later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False)]


# ---------------------------------------------------------------------------
# Where the credentials come from
# ---------------------------------------------------------------------------


class TestStoreResolution:
    def test_a_project_without_a_mongodb_archive_has_nothing_to_rewrite(self, tmp_path):
        root = _write_project(tmp_path / "plain", None, password=None)

        assert (
            archiver_store_config(yaml.safe_load((root / "config.yml").read_text()), root) is None
        )

    def test_the_password_comes_from_the_project_env_not_the_ambient_one(
        self, tmp_path, monkeypatch
    ):
        """The foreign-cwd case, which is the normal one for a test or a harness.

        A stray ``MONGO_ROOT_PASSWORD`` in the environment belongs to whatever
        deployment exported it. Reaching for it here would, in the bad case,
        authenticate against someone else's archive and rewrite it.
        """
        root = _write_project(
            tmp_path / "proj",
            {
                "host": "127.0.0.1",
                "port": 27017,
                "database": "db",
                "collection": "c",
                "username": "u",
            },
            password="the-project-password",
        )
        monkeypatch.setenv("MONGO_ROOT_PASSWORD", "some-other-deployments-password")
        monkeypatch.chdir(tmp_path)

        store = archiver_store_config(yaml.safe_load((root / "config.yml").read_text()), root)

        assert store["password"] == "the-project-password"

    def test_a_missing_password_is_reported_rather_than_guessed(self, tmp_path):
        root = _write_project(
            tmp_path / "proj",
            {
                "host": "127.0.0.1",
                "port": 27017,
                "database": "db",
                "collection": "c",
                "username": "u",
            },
            password=None,
        )

        store = archiver_store_config(yaml.safe_load((root / "config.yml").read_text()), root)
        assert store is not None
        assert store["password"] is None


# ---------------------------------------------------------------------------
# Which stretch of history an event reaches
# ---------------------------------------------------------------------------


class TestEventWindow:
    ANCHOR = T0.timestamp()
    HORIZON = T0.timestamp() - 86400

    def test_a_spike_is_local_to_its_own_bump(self):
        start, end = event_window(
            [{"shape": "spike", "at_offset": -3600, "amplitude": 1.0, "width": 60.0}],
            self.ANCHOR,
            self.HORIZON,
        )

        assert self.ANCHOR - end > 3000, "the window must not run to the anchor"
        assert end - start == pytest.approx(2 * 4.0 * 60.0)

    def test_a_step_reaches_the_end_of_the_archive(self):
        """It changes every sample after it, so its window has to as well."""
        start, end = event_window(
            [{"shape": "step", "at_offset": -3600, "to": 5.0}], self.ANCHOR, self.HORIZON
        )

        assert start == pytest.approx(self.ANCHOR - 3600)
        assert end == pytest.approx(self.ANCHOR)

    def test_a_daily_event_reaches_the_whole_archive(self):
        """It recurs on every date the archive covers."""
        assert event_window(
            [{"shape": "step", "at_time": "03:00:00", "to": 5.0}], self.ANCHOR, self.HORIZON
        ) == (self.HORIZON, self.ANCHOR)

    def test_an_event_older_than_the_archive_is_clamped_to_it(self):
        start, _ = event_window(
            [{"shape": "step", "at_offset": -86400 * 10, "to": 5.0}], self.ANCHOR, self.HORIZON
        )

        assert start == self.HORIZON

    def test_a_window_fraction_event_is_refused_by_name(self):
        """A fraction names a position in the reader's window, not an instant —
        there is no timestamp to store it at, and silently placing it somewhere
        plausible would be inventing history."""
        with pytest.raises(ValueError, match="at_offset"):
            event_window([{"shape": "step", "at": 0.5, "to": 5.0}], self.ANCHOR, self.HORIZON)

    def test_several_events_span_from_the_earliest_to_the_latest(self):
        start, end = event_window(
            [
                {"shape": "spike", "at_offset": -7200, "amplitude": 1.0, "width": 60.0},
                {"shape": "spike", "at_offset": -3600, "amplitude": 1.0, "width": 60.0},
            ],
            self.ANCHOR,
            self.HORIZON,
        )

        assert start < self.ANCHOR - 7200 < self.ANCHOR - 3600 < end


class TestEventSubwindows:
    """Which stretches an event is *evidence* over, as opposed to reaches.

    The two are not the same thing, and conflating them is expensive in both
    directions: a daily event's values reach across the whole archive, but its
    evidence is a few minutes a day, and protecting the reach would leave an
    entire deployment unexpiring while densifying it would insert a month of
    samples nobody asked for.
    """

    ANCHOR = T0.timestamp()
    HORIZON = T0.timestamp() - 86400 * 3

    def test_a_daily_event_contributes_one_bump_per_date_it_covers(self):
        windows = event_subwindows(
            [{"shape": "spike", "at_time": "03:00:00", "amplitude": 1.0, "width": 60.0}],
            self.ANCHOR,
            self.HORIZON,
            tz=ZoneInfo("UTC"),
        )

        assert len(windows) == 3, "three dates covered, three bumps"
        assert all(end - start == pytest.approx(2 * 4.0 * 60.0) for start, end in windows)
        assert sum(end - start for start, end in windows) < (self.ANCHOR - self.HORIZON) / 100

    def test_separate_spikes_stay_separate_bumps(self):
        """A three-spike script must not protect the calm hours between them."""
        windows = event_subwindows(
            [
                {"shape": "spike", "at_offset": -7200, "amplitude": 1.0, "width": 60.0},
                {"shape": "spike", "at_offset": -3600, "amplitude": 1.0, "width": 60.0},
            ],
            self.ANCHOR,
            self.HORIZON,
        )

        assert len(windows) == 2
        assert windows[0][1] < windows[1][0], "the quiet stretch between them is not covered"

    def test_overlapping_bumps_are_merged(self):
        """Disjoint output is what lets the rewrite fill each instant once."""
        windows = event_subwindows(
            [
                {"shape": "spike", "at_offset": -3700, "amplitude": 1.0, "width": 60.0},
                {"shape": "spike", "at_offset": -3600, "amplitude": 1.0, "width": 60.0},
            ],
            self.ANCHOR,
            self.HORIZON,
        )

        assert len(windows) == 1

    def test_a_persistent_event_is_evidence_to_the_end_of_the_archive(self):
        """A step really does change every later sample, so it protects them."""
        assert event_subwindows(
            [{"shape": "step", "at_offset": -3600, "to": 5.0}], self.ANCHOR, self.HORIZON
        ) == [(self.ANCHOR - 3600, self.ANCHOR)]

    def test_an_occurrence_older_than_the_archive_contributes_nothing(self):
        assert (
            event_subwindows(
                [{"shape": "spike", "at_offset": -86400 * 30, "amplitude": 1.0, "width": 60.0}],
                self.ANCHOR,
                self.HORIZON,
            )
            == []
        )

    def test_a_window_fraction_event_is_refused_here_too(self):
        with pytest.raises(ValueError, match="at_offset"):
            event_subwindows([{"shape": "step", "at": 0.5, "to": 5.0}], self.ANCHOR, self.HORIZON)

    @pytest.mark.parametrize("resolve", [event_window, event_subwindows])
    def test_an_empty_script_names_no_window(self, resolve):
        """Answering with an inverted span would put a start > end pair in the
        ledger, and every later apply would recompute an empty stretch forever."""
        with pytest.raises(ValueError, match="empty"):
            resolve([], self.ANCHOR, self.HORIZON)


class TestPersistedAnchor:
    """A re-apply has to land on the timeline the world already has.

    A deploy-time reseed re-applies the active set so the rebuilt store carries
    its events. If that re-apply minted a fresh anchor, the running VA's
    ``at_offset`` events, the seeded logbook and the archive's event windows
    would all slide to a new T0 — a timeline change as a side effect of an
    operation whose whole point was to leave the world alone.
    """

    def test_the_anchor_written_by_an_apply_is_readable_afterwards(self, tmp_path):
        root = _write_project(tmp_path / "proj", None, password=None)
        config = yaml.safe_load((root / "config.yml").read_text())

        assert persisted_scenario_anchor(config, root) is None, "nothing activated yet"

        _apply(root, ["burst"], at=T0)

        anchor = persisted_scenario_anchor(config, root)
        assert anchor is not None
        assert anchor == T0

    def test_a_project_that_never_activated_anything_has_no_anchor(self, tmp_path):
        """``None`` means "no established timeline", which is a caller's cue to
        mint one — not an error to paper over with wall-clock now."""
        root = _write_project(tmp_path / "proj", None, password=None)

        assert (
            persisted_scenario_anchor(yaml.safe_load((root / "config.yml").read_text()), root)
            is None
        )

    def test_the_anchor_survives_a_second_apply_at_a_different_instant(self, tmp_path):
        """Re-applying moves it, deliberately — this reads what is *current*,
        so a caller that wants the old timeline has to read it before applying."""
        root = _write_project(tmp_path / "proj", None, password=None)
        config = yaml.safe_load((root / "config.yml").read_text())
        later = T0 + timedelta(hours=3)

        _apply(root, ["burst"], at=T0)
        _apply(root, ["nominal"], at=later)

        assert persisted_scenario_anchor(config, root) == later


class TestComposedEvents:
    def test_the_active_set_s_scripts_are_composed(self, tmp_path):
        root = _write_project(tmp_path / "proj", None, password=None)
        machine = root / "data" / "simulation" / "machine.json"

        assert active_archiver_events(machine, ["nominal"]) == {}
        assert list(active_archiver_events(machine, ["burst"])) == [PRESSURE]

    def test_an_unknown_scenario_is_refused(self, tmp_path):
        root = _write_project(tmp_path / "proj", None, password=None)

        with pytest.raises(ValueError, match="Unknown scenario"):
            active_archiver_events(root / "data" / "simulation" / "machine.json", ["nope"])


# ---------------------------------------------------------------------------
# The rewrite itself
# ---------------------------------------------------------------------------


class TestRewrite:
    def test_applying_an_eventless_set_to_a_fresh_archive_changes_nothing(self, project):
        """ "Restore then nothing" has to be genuinely nothing: the benchmark
        matrix and the doc-screenshot capture both apply ``nominal`` routinely,
        and neither should pay for, or risk, a rewrite."""
        root, collection = project
        before = _snapshot(collection)

        result = _apply(root, ["nominal"])

        assert result.archiver.skipped is None
        assert result.archiver.updated == 0
        assert result.archiver.inserted == 0
        assert _snapshot(collection) == before

    def test_an_event_rewrites_its_own_window_and_leaves_the_rest_alone(self, project):
        root, collection = project
        before = _snapshot(collection)

        result = _apply(root, ["burst"])

        assert PRESSURE in result.archiver.channels
        assert result.archiver.updated > 0

        after = _snapshot(collection)
        changed = [
            stamp
            for stamp, doc in after.items()
            if stamp in before and doc.get(PRESSURE) != before[stamp].get(PRESSURE)
        ]
        assert changed, "the event did not reach the archive"

        window = timedelta(seconds=4 * SPIKE_WIDTH_S + 60)
        centre = T0 + timedelta(seconds=SPIKE_OFFSET_S)
        assert all(abs(stamp.replace(tzinfo=UTC) - centre) <= window for stamp in changed)

    def test_the_untouched_channel_keeps_every_sample(self, project):
        """A scenario declares the channels it affects; the archive must not
        quietly re-noise the ones it does not."""
        root, collection = project
        before = _snapshot(collection)

        _apply(root, ["burst"])

        after = _snapshot(collection)
        assert all(after[stamp][TEMPERATURE] == doc[TEMPERATURE] for stamp, doc in before.items())

    def test_the_event_window_is_protected_from_retention(self, project):
        """Retention must never prune the evidence a scenario exists to be
        diagnosed from, so the touched documents lose their expiry outright."""
        root, collection = project

        _apply(root, ["burst"])

        centre = T0 + timedelta(seconds=SPIKE_OFFSET_S)
        in_window = collection.count_documents(
            {
                DATE_FIELD: {
                    "$gte": centre - timedelta(seconds=SPIKE_WIDTH_S),
                    "$lte": centre + timedelta(seconds=SPIKE_WIDTH_S),
                },
                EXPIRE_FIELD: {"$exists": True},
            }
        )
        assert in_window == 0

    def test_the_coarse_stretch_is_densified_for_the_event_channel_only(self, project):
        """A spike sampled once a minute is a shape nobody can read. The dense
        inserts carry only the channels the event touches — the others keep the
        coarse history they genuinely have."""
        root, collection = project

        result = _apply(root, ["burst"])

        assert result.archiver.inserted > 0
        inserted = list(collection.find({DENSIFIED_FIELD: True}))
        assert len(inserted) == result.archiver.inserted
        for document in inserted:
            assert PRESSURE in document
            assert TEMPERATURE not in document
            assert EXPIRE_FIELD not in document

    def test_dense_inserts_land_on_the_shared_epoch_grid(self, project):
        """The insert phase must share the recorder's clock, not just the seeder's.

        The recorder floors every sample to an integer multiple of its cadence
        since the Unix epoch. If densification placed samples at offsets from
        the apply anchor instead, an event window would end up holding two
        interleaved timelines — the recorder's and this one's — half a cadence
        apart, which reads as a doubled sampling rate rather than as a denser
        one, and no equivalence assertion could tell the two grids apart.
        """
        root, collection = project

        _apply(root, ["burst"])

        inserted = list(collection.find({DENSIFIED_FIELD: True}))
        assert inserted
        for document in inserted:
            epoch = document[DATE_FIELD].replace(tzinfo=UTC).timestamp()
            assert epoch % KNOBS.hot_cadence_sec == 0, "insert is off the dense grid"
            # Strictly between coarse samples: a dense insert on a coarse
            # timestamp would duplicate a sample the base seed already wrote.
            assert epoch % KNOBS.tail_cadence_sec != 0

    def test_the_anchor_second_does_not_leak_into_the_grid(self, project):
        """Applying at a non-cadence instant must not shift the inserts.

        T0 here has a second of 53, so an implementation that offset its grid
        from the anchor would put every insert 3 seconds off the shared clock —
        and would still pass every other test in this file.
        """
        root, collection = project
        assert T0.second % KNOBS.hot_cadence_sec != 0, "the anchor must be off-grid"

        _apply(root, ["burst"], at=T0)

        stamps = [
            document[DATE_FIELD].replace(tzinfo=UTC).timestamp()
            for document in collection.find({DENSIFIED_FIELD: True})
        ]
        assert stamps
        assert all(stamp % KNOBS.hot_cadence_sec == 0 for stamp in stamps)

    def test_switching_back_leaves_no_trace_of_the_previous_scenario(self, project):
        """The headline contract.

        Apply an eventful set, then an eventless one, and the store has to be
        the base again — values, expiry stamps and document count alike. A
        surviving event window would be a fault the agent can find and the
        operator never caused, which is worse than having no history at all.
        """
        root, collection = project
        base = _snapshot(collection)

        _apply(root, ["burst"])
        assert _snapshot(collection) != base, "the rewrite did nothing to undo"

        result = _apply(root, ["nominal"])

        assert result.archiver.restored > 0
        assert result.archiver.removed > 0
        assert collection.count_documents({DENSIFIED_FIELD: True}) == 0
        assert _snapshot(collection) == base

    def test_the_ledger_records_what_was_touched_and_clears_again(self, project):
        """The restore can only find the old windows if the apply wrote them
        down; an empty ledger after an eventless set is what keeps the next
        apply from re-restoring stretches nobody has touched."""
        root, collection = project

        _apply(root, ["burst"])
        ledger = collection.find_one({"_id": MANIFEST_ID})["touched_windows"]
        assert [entry["channels"] for entry in ledger] == [[PRESSURE]]

        _apply(root, ["nominal"])
        assert collection.find_one({"_id": MANIFEST_ID})["touched_windows"] == []

    def test_re_applying_the_same_set_is_idempotent(self, project):
        """A second apply must land on the same store, not stack a second
        rewrite on top of the first.

        Applied to the two-channel set on purpose: with one channel there is
        only one region to get right, and every way two regions can interfere —
        one claiming a shared timestamp, one undoing the other's expiry — is
        invisible.
        """
        root, collection = project

        _apply(root, ["twin-burst"])
        once = _snapshot(collection)
        result = _apply(root, ["twin-burst"])

        assert _snapshot(collection) == once
        assert result.archiver.updated == 0, "a re-apply that changed nothing must not claim it"


class TestOverlappingWindows:
    """Two channels whose event windows overlap without coinciding.

    This is the shape the shipped bundles actually have — a scenario perturbs a
    handful of channels, and their events do not all sit at the same instant —
    and it is the one where a rewrite that works region by region comes apart:
    whichever region reaches a shared timestamp first inserts the document, the
    second finds it already there and inserts nothing, and the channels it was
    carrying are simply missing from that stretch. Which region goes first is a
    set-iteration accident, so the archive comes out differently run to run.
    """

    def test_the_shared_window_is_dense_for_both_channels(self, project):
        root, collection = project

        _apply(root, ["twin-burst"])

        low, high = _window(LATE_OFFSET_S, shrink=KNOBS.hot_cadence_sec)
        for channel in (PRESSURE, TEMPERATURE):
            gaps = _gaps(collection, channel, low, high)
            assert gaps, f"{channel} has no samples in the shared window at all"
            assert max(gaps) <= KNOBS.hot_cadence_sec, f"{channel} has a hole in its own window"

    def test_each_channel_is_dense_only_inside_its_own_windows(self, project):
        """The other half of the same contract: a channel with no event in a
        stretch keeps the coarse history it genuinely has, rather than being
        handed resolution because a neighbour needed it."""
        root, collection = project

        _apply(root, ["twin-burst"])

        low, high = _window(SPIKE_OFFSET_S, shrink=KNOBS.hot_cadence_sec)
        assert max(_gaps(collection, PRESSURE, low, high)) <= KNOBS.hot_cadence_sec
        assert min(_gaps(collection, TEMPERATURE, low, high)) == KNOBS.tail_cadence_sec

    def test_a_shared_instant_is_one_document_carrying_both_channels(self, project):
        """Not two competing samples of the same instant, and not one sample
        that quietly dropped a channel."""
        root, collection = project

        _apply(root, ["twin-burst"])

        low, high = _window(LATE_OFFSET_S, shrink=KNOBS.hot_cadence_sec)
        inserted = list(
            collection.find({DATE_FIELD: {"$gte": low, "$lte": high}, DENSIFIED_FIELD: True})
        )
        assert inserted
        assert all(PRESSURE in document and TEMPERATURE in document for document in inserted)
        stamps = [document[DATE_FIELD] for document in inserted]
        assert len(set(stamps)) == len(stamps), "an instant was written twice"

    def test_a_stretch_the_store_does_not_cover_is_not_manufactured(self, project):
        """Densification adds resolution to history that exists; it does not
        invent history that does not.

        A recorder outage, an expired stretch, or a window reaching past the
        newest sample all look the same from here: samples are absent. Filling
        them would be exactly the fabrication this feature exists to remove,
        told backwards — so the rewrite leaves them empty and says how many.
        """
        root, collection = project
        low, high = _window(LATE_OFFSET_S, shrink=SPIKE_REACH_S / 2)
        collection.delete_many({DATE_FIELD: {"$gte": low, "$lte": high}})

        result = _apply(root, ["twin-burst"])

        assert collection.count_documents({DATE_FIELD: {"$gte": low, "$lte": high}}) == 0
        assert result.archiver.uncovered > 0, "the empty stretch was not reported"
        assert "uncovered" in result.archiver.describe()


class TestScenarioSwitch:
    """Switching from one eventful scenario to another, not back to nominal.

    Switching back to ``nominal`` is the easy direction — every window the old
    set touched is a window the new set does not, so a restore that recomputes
    the union lands right by accident. Switching to a *different* eventful set
    is where the two halves come apart: the union of the old and the new window
    is live at one end and dead at the other, and the ledger the apply leaves
    behind names only the new one. Anything the old window keeps here, it keeps
    forever.
    """

    def _unexpiring_between_the_windows(self, collection) -> tuple[int, int]:
        """``(total, unexpiring)`` samples in the stretch neither window covers."""
        between = {
            DATE_FIELD: {"$gte": _window(SPIKE_OFFSET_S)[1], "$lte": _window(LATE_OFFSET_S)[0]}
        }
        return (
            collection.count_documents(between),
            collection.count_documents({**between, EXPIRE_FIELD: {"$exists": False}}),
        )

    def test_the_stretch_between_the_two_windows_ages_again(self, project):
        root, collection = project
        _apply(root, ["burst"])

        _apply(root, ["late-burst"])

        total, unexpiring = self._unexpiring_between_the_windows(collection)
        assert total > 0, "the two windows do not straddle a stretch to check"
        assert unexpiring == 0, f"{unexpiring}/{total} samples were left unexpiring"

    def test_a_leak_here_would_be_unrecoverable(self, project):
        """Why this defect is the critical one rather than merely wrong.

        The ledger a switch leaves behind names only the *new* window. Anything
        the stretch between the two windows keeps, no later apply ever looks at
        again — not even applying ``nominal``, which is the operator's one
        obvious way to put the world back. So the check is not just that the
        stretch ages again, but that a full restore afterwards agrees: a fix
        that merely deferred the repair to the next apply would satisfy the
        first assertion and still leave history that outlives its retention
        forever.
        """
        root, collection = project

        _apply(root, ["burst"])
        _apply(root, ["late-burst"])
        _apply(root, ["nominal"])

        total, unexpiring = self._unexpiring_between_the_windows(collection)
        assert total > 0
        assert unexpiring == 0, f"{unexpiring}/{total} samples survived a restore to nominal"

    def test_one_span_carries_both_outcomes_at_once(self, project):
        """Expiry is decided per document, and a single pass proves it.

        The recomputed span is the union of the old and the new window: it is
        live at one end and dead at the other, *in the same pass*. One boolean
        over that span cannot produce both answers — whichever way it points,
        either the new window keeps an expiry retention will prune or the old
        one loses the expiry it must get back. Asserting the two halves
        together is what a per-region fix cannot satisfy.
        """
        root, collection = project
        _apply(root, ["burst"])

        _apply(root, ["late-burst"])

        live_low, live_high = _window(LATE_OFFSET_S, shrink=KNOBS.hot_cadence_sec)
        live = {DATE_FIELD: {"$gte": live_low, "$lte": live_high}}
        protected = collection.count_documents({**live, EXPIRE_FIELD: {"$exists": False}})
        assert protected > 0, "the new window is not protected from retention"
        assert collection.count_documents({**live, EXPIRE_FIELD: {"$exists": True}}) == 0

        _, unexpiring = self._unexpiring_between_the_windows(collection)
        assert unexpiring == 0, "the same pass left the old stretch unexpiring"

    def test_the_old_window_goes_back_to_base(self, project):
        root, collection = project
        base = _snapshot(collection)

        _apply(root, ["burst"])
        _apply(root, ["late-burst"])

        low, high = _window(SPIKE_OFFSET_S)
        for stamp, document in _snapshot(collection).items():
            if low <= stamp.replace(tzinfo=UTC) <= high and stamp in base:
                assert document == base[stamp], f"the previous scenario survived at {stamp}"

    def test_the_ledger_converges_on_the_new_window(self, project):
        root, collection = project

        _apply(root, ["burst"])
        _apply(root, ["late-burst"])

        ledger = collection.find_one({"_id": MANIFEST_ID})["touched_windows"]
        assert [entry["channels"] for entry in ledger] == [[PRESSURE]]
        low, high = _window(LATE_OFFSET_S)
        assert ledger[0]["start"] == pytest.approx(low.timestamp())
        assert ledger[0]["end"] == pytest.approx(high.timestamp())

    def test_a_switch_reports_the_documents_it_restored_separately(self, project):
        """The count the CLI prints as "rewrote N samples" is the new set's
        story, not the union it had to recompute to get there."""
        root, collection = project
        _apply(root, ["burst"])

        result = _apply(root, ["late-burst"])

        low, high = _window(LATE_OFFSET_S)
        live = collection.count_documents({DATE_FIELD: {"$gte": low, "$lte": high}})
        assert 0 < result.archiver.updated <= live
        assert result.archiver.restored > 0, "the old window stayed out of retention"

    def test_the_ledger_names_the_union_before_the_rewrite_starts(self, project, monkeypatch):
        """Crash safety.

        The rewrite writes event values into the union of the old and the new
        window. A process that dies part way through has already marked
        documents in the stretch only the old window covered — and if the ledger
        still named just that old window, or already named just the new one,
        nothing would ever come back for them. So the union goes down first and
        is narrowed only after the rewrite returns.
        """
        from osprey.simulation import apply as apply_module

        root, collection = project
        _apply(root, ["burst"])

        def _die(*args, **kwargs):
            raise RuntimeError("killed mid-rewrite")

        monkeypatch.setattr(apply_module, "_rewrite_documents", _die)
        with pytest.raises(RuntimeError, match="killed mid-rewrite"):
            _apply(root, ["late-burst"])

        ledger = collection.find_one({"_id": MANIFEST_ID})["touched_windows"]
        assert [entry["channels"] for entry in ledger] == [[PRESSURE]]
        assert ledger[0]["start"] == pytest.approx(_window(SPIKE_OFFSET_S)[0].timestamp())
        assert ledger[0]["end"] == pytest.approx(_window(LATE_OFFSET_S)[1].timestamp())


class TestCrashRecovery:
    """Dying part way through a rewrite must leave a recoverable store.

    The rewrite marks documents across the union of the old and the new windows
    and inserts dense samples inside the new ones. Every one of those marks is
    findable again only through the touched-window ledger, so the ledger is
    written *before* the first document moves and narrowed only after the last
    one has. The two tests here kill the process at each of the two write
    phases; in both cases applying ``nominal`` afterwards — the operator's one
    obvious way to put the world back — has to reach the base exactly.
    """

    @staticmethod
    def _crash_after(real):
        """Run the real step, then die — a crash that has already written.

        Dying *before* the step is the easy case: nothing moved, so any ledger
        recovers it. The case that needs the ordering is a process that got its
        writes in and then lost the narrowing write, which is what these
        wrappers reproduce.
        """

        def _wrapped(*args, **kwargs):
            real(*args, **kwargs)
            raise RuntimeError("killed mid-rewrite")

        return _wrapped

    def test_a_crash_after_recomputing_leaves_the_marks_findable(self, project, monkeypatch):
        """The switch case: values written across the union, ledger lost.

        The recompute reaches the whole union of the old and the new window. If
        the ledger still named only the *old* one, the new window would keep
        this apply's event values and its lifted expiry with nothing recording
        that it had been touched — and the restore below would put back only
        half the store.
        """
        from osprey.simulation import apply as apply_module

        root, collection = project
        base = _snapshot(collection)
        _apply(root, ["burst"])

        monkeypatch.setattr(
            apply_module, "_rewrite_documents", self._crash_after(apply_module._rewrite_documents)
        )
        with pytest.raises(RuntimeError, match="killed mid-rewrite"):
            _apply(root, ["late-burst"])
        monkeypatch.undo()

        _apply(root, ["nominal"])
        assert collection.count_documents({DENSIFIED_FIELD: True}) == 0
        assert _snapshot(collection) == base

    def test_a_crash_while_densifying_on_a_fresh_store_is_recoverable(self, project, monkeypatch):
        """The first eventful apply on a deployment nobody has run a scenario on.

        This is the case a ledger written only when it "differs from the final
        one" would miss: with no previous windows the union *equals* the new
        windows, so an implementation that treats that as nothing to record
        writes marks and dense samples under an empty ledger. Nothing would ever
        come back for them — not even a restore to nominal, which is what makes
        it unrecoverable rather than merely untidy.
        """
        from osprey.simulation import apply as apply_module

        root, collection = project
        base = _snapshot(collection)
        real_densify = apply_module._densify

        def _die_having_already_inserted(*args, **kwargs):
            real_densify(*args, **kwargs)
            raise RuntimeError("killed mid-rewrite")

        monkeypatch.setattr(apply_module, "_densify", _die_having_already_inserted)
        with pytest.raises(RuntimeError, match="killed mid-rewrite"):
            _apply(root, ["burst"])
        monkeypatch.undo()

        assert collection.count_documents({DENSIFIED_FIELD: True}) > 0, "nothing was inserted"
        ledger = collection.find_one({"_id": MANIFEST_ID})["touched_windows"]
        assert [entry["channels"] for entry in ledger] == [[PRESSURE]], (
            "the crashed apply left its marks under an empty ledger"
        )

        _apply(root, ["nominal"])
        assert collection.count_documents({DENSIFIED_FIELD: True}) == 0
        assert _snapshot(collection) == base


class TestRecurringEvents:
    """A daily event is evidence for minutes a day, not for the whole archive."""

    def test_it_protects_its_own_bumps_and_nothing_else(self, project):
        root, collection = project

        _apply(root, ["nightly"])

        total = collection.count_documents({DATE_FIELD: {"$exists": True}})
        unexpiring = collection.count_documents(
            {DATE_FIELD: {"$exists": True}, EXPIRE_FIELD: {"$exists": False}}
        )
        assert 0 < unexpiring < total / 4, "a daily event de-expired the deployment"

    def test_it_densifies_its_own_bumps_and_nothing_else(self, project):
        root, collection = project
        base = collection.count_documents({DATE_FIELD: {"$exists": True}})

        result = _apply(root, ["nightly"])

        # One bump a day at four sigmas either side, filled at the dense
        # cadence: a small, countable number. Densifying the whole span its
        # values reach over would multiply the store several times instead.
        assert 0 < result.archiver.inserted < base / 4


class TestStoredTypes:
    def test_dense_inserts_carry_the_type_the_channel_is_stored_as(self, project):
        """A flag channel's history must not change type part way through.

        The updates already coerce to the stored type; an insert has no document
        of its own to read that from, and writing the engine's raw float would
        leave a boolean channel holding ``0.7`` beside its ``True`` — which reads
        as a different instrument rather than a different value.
        """
        root, collection = project

        _apply(root, ["flagged"])

        inserted = [
            document for document in collection.find({DENSIFIED_FIELD: True}) if FAULT in document
        ]
        assert inserted, "the flag channel's window was not densified"
        assert all(isinstance(document[FAULT], bool) for document in inserted)

    def test_the_archive_is_not_rewritten_when_the_caller_opts_out(self, project):
        root, collection = project
        before = _snapshot(collection)

        result = apply_scenarios(root, ["burst"], seed_logbook=False, seed_archive=False, now=T0)

        assert result.archiver is None
        assert _snapshot(collection) == before


class TestRefusals:
    def test_an_unseeded_store_is_reported_not_silently_skipped(self, project):
        """ "The scenario is active but its history is not" must never be
        something a caller has to infer from a zero count."""
        root, collection = project
        collection.delete_one({"_id": MANIFEST_ID})

        result = _apply(root, ["burst"])

        assert "not been seeded" in result.archiver.skipped

    def test_a_project_with_no_store_says_so(self, tmp_path):
        root = _write_project(tmp_path / "plain", None, password=None)

        result = _apply(root, ["burst"])

        assert result.archiver.skipped == "project declares no MongoDB archive"

    def test_a_configured_store_with_no_password_refuses_loudly(self, tmp_path, mongo_store):
        """Failing here is the point: the scenario is now active, and history
        that silently contradicts it is the exact defect being removed."""
        root = _write_project(tmp_path / "proj", mongo_store, password=None)

        with pytest.raises(RuntimeError, match="MONGO_ROOT_PASSWORD"):
            _apply(root, ["burst"])


class TestPreflight:
    """Deciding the rewrite before the scenario is activated.

    Every refusal below is one a caller can act on: fix the ``.env``, fix the
    bundle, and run the command again. Discovering them from inside the rewrite
    would leave the world half applied — telemetry live, narrative reseeded,
    history untouched and contradicting both — which is the divergence the
    feature exists to close, arrived at by a different route.
    """

    def _machine_path(self, root: Path) -> Path:
        return root / "data" / "simulation" / "machine.json"

    def _config(self, root: Path) -> dict:
        return yaml.safe_load((root / "config.yml").read_text())

    def test_a_missing_password_is_refused_with_nothing_activated(self, tmp_path, mongo_store):
        root = _write_project(tmp_path / "proj", mongo_store, password=None)

        with pytest.raises(RuntimeError, match="MONGO_ROOT_PASSWORD"):
            preflight_archive_rewrite(root, self._config(root), self._machine_path(root), ["burst"])

        assert persisted_scenario_anchor(self._config(root), root) is None

    def test_a_window_fraction_event_is_refused_before_anything_is_written(self, tmp_path):
        root = _write_project(tmp_path / "plain", None, password=None)
        machine = json.loads(self._machine_path(root).read_text())
        machine["scenarios"]["burst"]["archiver"][0]["events"] = [
            {"shape": "step", "at": 0.5, "to": 5.0}
        ]
        self._machine_path(root).write_text(json.dumps(machine))

        with pytest.raises(ValueError, match="at_offset"):
            preflight_archive_rewrite(root, self._config(root), self._machine_path(root), ["burst"])

    def test_an_unknown_scenario_is_refused(self, tmp_path):
        root = _write_project(tmp_path / "plain", None, password=None)

        with pytest.raises(ValueError, match="Unknown scenario"):
            preflight_archive_rewrite(root, self._config(root), self._machine_path(root), ["nope"])

    def test_a_project_with_no_store_has_nothing_to_decide(self, tmp_path):
        root = _write_project(tmp_path / "plain", None, password=None)

        assert (
            preflight_archive_rewrite(root, self._config(root), self._machine_path(root), ["burst"])
            is None
        )

    def test_a_healthy_project_returns_the_store_it_would_rewrite(self, tmp_path, mongo_store):
        root = _write_project(tmp_path / "proj", mongo_store, password=mongo_store["password"])

        store = preflight_archive_rewrite(
            root, self._config(root), self._machine_path(root), ["burst"]
        )

        assert store is not None
        assert store["database"] == mongo_store["database"]
