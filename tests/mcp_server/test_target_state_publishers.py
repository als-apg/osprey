"""Tests for the state-file primitives the header chip reads and writes.

The state file grew three publication blocks and gained a neighbour, and this
file covers the half of ``target_state`` that serves the chip rather than the
prompt line:

  - switch requests: the one file in the directory the WEB server writes —
    write / read / remove, the stamped ``created_at``, the TTL both readers
    share, and the dead-addressee sweep
  - the three publishers (``last_switch``, ``reachability``,
    ``last_posture_realign``): each merges without clobbering a sibling block,
    each is synchronous, and ``write_on_start`` resets all three
  - the in-flight marker reader in its new home, still importable from
    ``tools/control_target.py`` under its old name

The publishers being SYNCHRONOUS is a correctness property, not a style choice:
the reconciler and the endpoint prober publish from the same event loop, and an
``await`` between ``_update``'s read and its write would let one of them write
back a record built before the other's change. It is pinned here.
"""

import inspect
import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.tools import control_target

TARGETS_META = {
    "live": {"label": "ALS storage ring", "endpoint": "gw:5064", "real_machine": True},
    "va": {"label": "Virtual accelerator", "endpoint": "localhost:5074", "real_machine": False},
    "standin": {"label": "Live stand-in", "endpoint": "localhost:5084", "real_machine": False},
}


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state directory in tmp_path instead of a real deployment.

    The environment stamp is cleared as well as the config derivation patched,
    so this fixture pins the directory whichever of the two resolution rules
    ``state_dir`` is applying.
    """
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def started(state_root):
    """A server whose record exists, so a merge has something to merge into."""
    target_state.write_on_start("va", TARGETS_META, server_pid=os.getpid(), owner_ppid=99)
    return os.getpid()


def _dead_pid(monkeypatch, dead):
    """Make ``is_process_alive`` report *dead* as gone and everything else alive."""
    real = os.kill

    def fake_kill(pid, sig):
        if pid in dead:
            raise ProcessLookupError
        return real(os.getpid(), 0)

    monkeypatch.setattr(target_state.os, "kill", fake_kill)


# ---------------------------------------------------------------------------
# Switch requests
# ---------------------------------------------------------------------------


class TestRequestFileContract:
    """Named for the server it addresses, so a successor never inherits it."""

    def test_path_is_named_for_the_addressed_server(self, state_root):
        assert target_state.request_file_path(4321) == (
            state_root / target_state.STATE_DIR_NAME / "request_4321.json"
        )

    def test_path_defaults_to_this_process(self, state_root):
        assert target_state.request_file_path().name == f"request_{os.getpid()}.json"

    def test_glob_matches_the_file_the_writer_produces(self, state_root):
        target_state.write_request({"request_id": "r1", "target": "live", "server_pid": 4321})

        directory = state_root / target_state.STATE_DIR_NAME
        assert [p.name for p in directory.glob(target_state.REQUEST_FILE_GLOB)] == [
            "request_4321.json"
        ]

    def test_request_glob_never_matches_a_state_file(self, started):
        directory = target_state.state_dir()
        assert list(directory.glob(target_state.REQUEST_FILE_GLOB)) == []

    def test_state_glob_never_matches_a_request_file(self, state_root):
        target_state.write_request({"request_id": "r1", "target": "live", "server_pid": 4321})

        directory = state_root / target_state.STATE_DIR_NAME
        assert list(directory.glob(target_state.STATE_FILE_GLOB)) == []


class TestWriteRequest:
    def test_round_trips_the_record(self, state_root):
        record = {
            "request_id": "req-1",
            "target": "standin",
            "server_pid": 4321,
            "created_at": "2026-08-30T10:00:00+00:00",
            "requested_by": "operator",
        }

        path = target_state.write_request(record)

        assert json.loads(path.read_text(encoding="utf-8")) == record
        assert target_state.read_request(4321) == record

    def test_created_at_is_stamped_when_the_caller_omits_it(self, state_root):
        """A request that cannot be aged could never expire, so it is never written."""
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

        record = target_state.read_request(4321)
        assert target_state.is_request_fresh(record)
        datetime.fromisoformat(record["created_at"])  # parseable, not just present

    def test_creates_the_state_directory(self, state_root):
        assert not (state_root / target_state.STATE_DIR_NAME).exists()

        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

        assert (state_root / target_state.STATE_DIR_NAME).is_dir()

    def test_a_second_request_replaces_the_first(self, state_root):
        target_state.write_request({"request_id": "one", "target": "va", "server_pid": 4321})
        target_state.write_request({"request_id": "two", "target": "live", "server_pid": 4321})

        assert target_state.read_request(4321)["request_id"] == "two"

    @pytest.mark.parametrize(
        "record",
        [
            {"request_id": "r", "target": "va"},
            {"request_id": "r", "target": "va", "server_pid": "not a pid"},
            {"request_id": "r", "target": "va", "server_pid": None},
            {"request_id": "r", "target": "va", "server_pid": 0},
            "not a mapping",
        ],
    )
    def test_a_request_addressed_to_nobody_is_a_programming_error(self, record, state_root):
        with pytest.raises(ValueError):
            target_state.write_request(record)

    def test_a_failed_write_leaves_no_temp_file(self, state_root, monkeypatch):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)

        def fail(*args, **kwargs):
            raise OSError("rename refused")

        monkeypatch.setattr(target_state.os, "replace", fail)

        with pytest.raises(OSError):
            target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

        assert list(directory.iterdir()) == []


class TestReadAndRemoveRequest:
    def test_absent_request_reads_as_none(self, state_root):
        assert target_state.read_request(4321) is None

    def test_corrupt_request_reads_as_none(self, state_root):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "request_4321.json").write_text("{not json", encoding="utf-8")

        assert target_state.read_request(4321) is None

    def test_read_defaults_to_this_process(self, state_root):
        target_state.write_request(
            {"request_id": "mine", "target": "va", "server_pid": os.getpid()}
        )

        assert target_state.read_request()["request_id"] == "mine"

    def test_remove_deletes_the_request(self, state_root):
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

        target_state.remove_request(4321)

        assert target_state.read_request(4321) is None

    def test_remove_is_idempotent(self, state_root):
        target_state.remove_request(4321)
        target_state.remove_request(4321)  # no exception is the assertion

    def test_remove_only_touches_the_addressed_request(self, state_root):
        target_state.write_request({"request_id": "a", "target": "va", "server_pid": 4321})
        target_state.write_request({"request_id": "b", "target": "va", "server_pid": 4322})

        target_state.remove_request(4321)

        assert target_state.read_request(4322)["request_id"] == "b"


class TestRequestFreshness:
    """One TTL spelling for the route that refuses a duplicate and the
    reconciler that expires a late one: the window shown is the window kept."""

    def test_ttl_is_thirty_seconds(self):
        assert target_state.REQUEST_TTL_S == 30

    def test_a_just_written_request_is_fresh(self, state_root):
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

        assert target_state.is_request_fresh(target_state.read_request(4321)) is True

    def test_a_request_older_than_the_ttl_is_not_fresh(self):
        created = datetime.now(UTC) - timedelta(seconds=target_state.REQUEST_TTL_S + 1)

        assert not target_state.is_request_fresh({"created_at": created.isoformat()})

    def test_the_boundary_is_inclusive(self):
        now = datetime.now(UTC)
        record = {"created_at": (now - timedelta(seconds=30)).isoformat()}

        assert target_state.is_request_fresh(record, now=now.timestamp())

    def test_a_naive_stamp_is_read_as_utc(self):
        naive = datetime.now(UTC).replace(tzinfo=None)

        assert target_state.is_request_fresh({"created_at": naive.isoformat()})

    def test_a_small_clock_skew_into_the_future_is_tolerated(self):
        created = datetime.now(UTC) + timedelta(seconds=5)

        assert target_state.is_request_fresh({"created_at": created.isoformat()})

    @pytest.mark.parametrize(
        "record",
        [
            None,
            {},
            "not a mapping",
            {"created_at": ""},
            {"created_at": "yesterday"},
            {"created_at": 5},
        ],
    )
    def test_a_request_that_cannot_be_aged_is_never_fresh(self, record):
        """Fail-closed: acting on an unaged request is the surprise the TTL prevents."""
        assert target_state.is_request_fresh(record) is False


class TestRequestSweep:
    def test_sweep_removes_a_request_addressed_to_a_dead_server(self, state_root, monkeypatch):
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})
        _dead_pid(monkeypatch, {4321})

        target_state.sweep_stale(server_pid=os.getpid())

        assert target_state.read_request(4321) is None

    def test_sweep_leaves_a_request_for_a_live_server_alone(self, state_root, monkeypatch):
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})
        _dead_pid(monkeypatch, set())

        target_state.sweep_stale(server_pid=os.getpid())

        assert target_state.read_request(4321)["request_id"] == "r"

    def test_sweep_removes_a_request_whose_name_encodes_no_pid(self, state_root):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        junk = directory / "request_nonsense.json"
        junk.write_text("{}", encoding="utf-8")

        target_state.sweep_stale(server_pid=os.getpid())

        assert not junk.exists()

    def test_a_swept_request_contributes_no_orphans(self, state_root, monkeypatch):
        """Requests are not state files: they own no connector-host children."""
        target_state.write_request(
            {"request_id": "r", "target": "va", "server_pid": 4321, "children": [777]}
        )
        _dead_pid(monkeypatch, {4321})

        assert target_state.sweep_stale(server_pid=os.getpid()) == []

    def test_sweep_does_not_touch_execution_markers(self, state_root, monkeypatch):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / f"{target_state.INFLIGHT_FILE_PREFIX}4321_abc.json"
        marker.write_text(json.dumps({"pid": 4321}), encoding="utf-8")
        _dead_pid(monkeypatch, {4321})

        target_state.sweep_stale(server_pid=os.getpid())

        assert marker.exists(), "the marker's own reader sweeps it; the state sweep must not"

    def test_write_on_start_drops_a_request_addressed_to_its_own_pid(self, state_root):
        """Only a dead predecessor can have left one: nothing knows this server yet."""
        target_state.write_request(
            {"request_id": "stale", "target": "live", "server_pid": os.getpid()}
        )

        target_state.write_on_start("va", TARGETS_META, server_pid=os.getpid())

        assert target_state.read_request() is None


# ---------------------------------------------------------------------------
# The publication blocks
# ---------------------------------------------------------------------------


class TestPublishersAreSynchronous:
    """A coroutine here could interleave inside ``_update``'s read-then-write."""

    @pytest.mark.parametrize(
        "name",
        [
            "publish_last_switch",
            "publish_reachability",
            "publish_posture_realign",
            "publish_switch",
        ],
    )
    def test_publisher_is_not_a_coroutine_function(self, name):
        assert not inspect.iscoroutinefunction(getattr(target_state, name))


class TestPublishLastSwitch:
    def test_publishes_the_terminus(self, started):
        outcome = {
            "request_id": "req-1",
            "target": "live",
            "status": "success",
            "reason": None,
            "detail": "switched to live",
            "at": "2026-08-30T10:00:00+00:00",
        }

        assert target_state.publish_last_switch(outcome) is True
        assert target_state.read()["last_switch"] == outcome

    def test_the_target_travels_with_the_outcome(self, started):
        """The popover's roster is one row per machine, and the row it puts an
        outcome on is the one the request named — so the terminus has to say
        which target it was aimed at, not only which request it answered."""
        target_state.publish_last_switch(
            {"request_id": "r", "target": "standin", "status": "refused", "reason": "no"}
        )

        assert target_state.read()["last_switch"]["target"] == "standin"

    def test_at_is_stamped_when_the_caller_omits_it(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "refused"})

        block = target_state.read()["last_switch"]
        datetime.fromisoformat(block["at"])

    def test_a_refusal_reason_travels_verbatim(self, started):
        """The vocabulary is the switch lifecycle's; this module never edits it."""
        target_state.publish_last_switch(
            {"request_id": "r", "status": "refused", "reason": "execution_in_flight"}
        )

        assert target_state.read()["last_switch"]["reason"] == "execution_in_flight"

    def test_none_clears_the_block(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "success"})

        target_state.publish_last_switch(None)

        assert target_state.read()["last_switch"] is None

    def test_preserves_identity_and_display_metadata(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "success"})

        record = target_state.read()
        assert record["target"] == "va"
        assert record["targets"] == TARGETS_META
        assert record["server_pid"] == started
        assert record["owner_ppid"] == 99

    def test_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_last_switch({"request_id": "r"}) is False
        assert target_state.read() is None


class TestPublishReachability:
    ROWS = {
        "live": {"epics": {"state": "reached", "probed_at": "2026-08-30T10:00:00+00:00"}},
        "va": {
            "epics": {
                "state": "down",
                "probed_at": "2026-08-30T10:00:00+00:00",
                "detail": "refused",
            },
            "pva": {"state": "not_applicable", "probed_at": "2026-08-30T10:00:00+00:00"},
        },
    }

    def test_publishes_every_role_of_every_target(self, started):
        assert target_state.publish_reachability(self.ROWS) is True

        published = target_state.read()["reachability"]["targets"]
        assert published["live"]["epics"]["state"] == "reached"
        assert published["va"]["epics"]["detail"] == "refused"

    def test_not_applicable_is_preserved_not_collapsed(self, started):
        """It is a decision from configuration, not a prober that failed to look."""
        target_state.publish_reachability(self.ROWS)

        assert target_state.read()["reachability"]["targets"]["va"]["pva"]["state"] == (
            "not_applicable"
        )

    def test_probed_at_survives_so_a_reader_can_compute_age(self, started):
        """The reader is in another process, so the block carries an instant, not an age."""
        probed_at = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        target_state.publish_reachability(
            {"live": {"epics": {"state": "down", "probed_at": probed_at}}}
        )

        row = target_state.read()["reachability"]["targets"]["live"]["epics"]
        age_s = (datetime.now(UTC) - datetime.fromisoformat(row["probed_at"])).total_seconds()
        assert 5 <= age_s < 60

    def test_the_sweep_is_stamped(self, started):
        target_state.publish_reachability(self.ROWS)

        stamp = target_state.read()["reachability"]["published_at"]
        assert (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() < 60

    def test_each_sweep_replaces_the_last(self, started):
        target_state.publish_reachability(self.ROWS)
        target_state.publish_reachability({"live": {"epics": {"state": "down"}}})

        published = target_state.read()["reachability"]["targets"]
        assert published == {"live": {"epics": {"state": "down"}}}

    def test_advancing_probes_advance_the_published_stamp(self, started):
        target_state.publish_reachability(self.ROWS)
        first = target_state.read()["reachability"]["published_at"]

        target_state.publish_reachability(self.ROWS)
        second = target_state.read()["reachability"]["published_at"]

        assert second >= first

    @pytest.mark.parametrize(
        "rows",
        [
            {},
            None,
            "not a mapping",
            {"live": "not a mapping"},
            {"live": {"epics": {"probed_at": "2026-08-30T10:00:00+00:00"}}},
            {"live": {"epics": {"state": ""}}},
        ],
    )
    def test_a_sweep_that_measured_nothing_clears_the_block(self, rows, started):
        """Absence renders ``unknown``; an empty row would render a false verdict."""
        target_state.publish_reachability(self.ROWS)

        target_state.publish_reachability(rows)

        assert target_state.read()["reachability"] is None

    def test_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_reachability(self.ROWS) is False


class TestPublishPostureRealign:
    def test_publishes_pending_then_done(self, started):
        assert target_state.publish_posture_realign({"state": "pending"}) is True
        assert target_state.read()["last_posture_realign"]["state"] == "pending"

        target_state.publish_posture_realign({"state": "done"})
        assert target_state.read()["last_posture_realign"]["state"] == "done"

    def test_at_is_stamped_when_the_caller_omits_it(self, started):
        target_state.publish_posture_realign({"state": "pending"})

        datetime.fromisoformat(target_state.read()["last_posture_realign"]["at"])

    def test_a_supplied_at_is_kept(self, started):
        target_state.publish_posture_realign({"state": "done", "at": "2026-08-30T10:00:00+00:00"})

        assert target_state.read()["last_posture_realign"]["at"] == "2026-08-30T10:00:00+00:00"

    def test_none_clears_the_block(self, started):
        target_state.publish_posture_realign({"state": "pending"})

        target_state.publish_posture_realign(None)

        assert target_state.read()["last_posture_realign"] is None

    def test_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_posture_realign({"state": "pending"}) is False


class TestBlocksDoNotClobberOneAnother:
    """Three publishers merge into one file; each must be blind to the others."""

    def test_every_block_survives_every_other_publisher(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "success"})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "pending"})
        target_state.publish_switch("live", 3, children=[901])
        target_state.record_child_pids([902])

        record = target_state.read()
        assert record["last_switch"]["request_id"] == "r"
        assert record["reachability"]["targets"]["live"]["epics"]["state"] == "reached"
        assert record["last_posture_realign"]["state"] == "pending"
        assert record["target"] == "live"
        assert record["generation"] == 3
        assert record["children"] == [902]
        assert record["targets"] == TARGETS_META

    def test_publish_switch_does_not_clear_the_blocks(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "success"})

        target_state.publish_switch("live", 1)

        assert target_state.read()["last_switch"]["request_id"] == "r"


class TestWriteOnStartResetsThePublications:
    def test_a_fresh_record_carries_three_null_blocks(self, state_root):
        target_state.write_on_start("va", TARGETS_META, server_pid=1234)

        record = target_state.read(1234)
        assert record["last_switch"] is None
        assert record["reachability"] is None
        assert record["last_posture_realign"] is None

    def test_a_restart_drops_the_predecessors_publications(self, started):
        target_state.publish_last_switch({"request_id": "r", "status": "success"})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "pending"})

        target_state.write_on_start("live", TARGETS_META, server_pid=os.getpid())

        record = target_state.read()
        assert record["last_switch"] is None
        assert record["reachability"] is None
        assert record["last_posture_realign"] is None


class TestReadersTolerateTheNewKeys:
    """Every reader takes the record with ``.get()``; none enumerates its keys."""

    def test_the_banner_matcher_reads_a_published_record(self, started, monkeypatch):
        from osprey.mcp_server.control_system import target_banner

        target_state.publish_last_switch({"request_id": "r", "status": "success"})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "done"})
        monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: None)

        record = target_banner._matched_record(99)

        assert record is not None
        assert record["target"] == "va"

    def test_the_hook_projection_ignores_the_new_keys(self, started):
        """The stdlib hook reader is a replica; it must not learn these blocks."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "osprey_target_state_replica",
            "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        target_state.publish_last_switch({"request_id": "r", "status": "success"})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})

        resolved = module._resolve_record(target_state.read(), "va")

        assert resolved == {
            "target": "va",
            "generation": 0,
            "display": resolved["display"],
            "fallback": None,
            "reason": None,
        }

    def test_the_stale_sweep_reads_a_published_record(self, state_root, monkeypatch):
        target_state.write_on_start("va", TARGETS_META, server_pid=4321, owner_ppid=99)
        target_state.publish_last_switch({"request_id": "r", "status": "success"}, server_pid=4321)
        target_state.publish_switch("live", 2, children=[777], server_pid=4321)
        _dead_pid(monkeypatch, {4321})

        assert target_state.sweep_stale(server_pid=os.getpid()) == [777]
        assert target_state.read(4321) is None


# ---------------------------------------------------------------------------
# The in-flight marker reader, in its new home
# ---------------------------------------------------------------------------


class TestInFlightExecutionsMoved:
    def _write_marker(self, pid, target="va"):
        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{target_state.INFLIGHT_FILE_PREFIX}{pid}_abc.json"
        path.write_text(json.dumps({"pid": pid, "target": target}), encoding="utf-8")
        return path

    def test_the_switch_tool_still_exports_the_names(self):
        """Every existing importer keeps working; only the definitions moved."""
        assert control_target.INFLIGHT_FILE_PREFIX is target_state.INFLIGHT_FILE_PREFIX
        assert control_target.INFLIGHT_FILE_SUFFIX is target_state.INFLIGHT_FILE_SUFFIX
        assert control_target.INFLIGHT_FILE_GLOB is target_state.INFLIGHT_FILE_GLOB
        assert control_target.in_flight_executions is target_state.in_flight_executions

    def test_a_live_marker_is_reported(self, state_root):
        self._write_marker(os.getpid())

        live = target_state.in_flight_executions()

        assert [row["pid"] for row in live] == [os.getpid()]

    def test_a_dead_writers_marker_is_swept(self, state_root, monkeypatch):
        path = self._write_marker(4321)
        _dead_pid(monkeypatch, {4321})

        assert target_state.in_flight_executions() == []
        assert not path.exists()

    def test_an_unreadable_marker_is_neither_reported_nor_deleted(self, state_root):
        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        junk = directory / f"{target_state.INFLIGHT_FILE_PREFIX}nonsense.json"
        junk.write_text("{not json", encoding="utf-8")

        assert target_state.in_flight_executions() == []
        assert junk.exists()

    def test_a_missing_state_dir_reads_as_no_executions(self, state_root):
        assert target_state.in_flight_executions() == []

    def test_a_request_file_is_not_mistaken_for_a_marker(self, state_root):
        target_state.write_request({"request_id": "r", "target": "va", "server_pid": os.getpid()})

        assert target_state.in_flight_executions() == []
