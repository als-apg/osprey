"""Tests for the report a controls server publishes about itself.

``target_state`` writes one ``server_<pid>.json`` per controls server — what
that server is doing, never what the deployment is on — and this file covers
the half of the module that serves the header chip and the convergence readers
rather than the prompt line:

  - switch requests: the one file family here a controls server does not own —
    named for the process that ASKED, consumed by the record's owner — write /
    read / remove, the stamped ``requested_at``, the TTL both readers share,
    the read-back that tells a consumed request from a superseded one, and the
    dead-requester sweep
  - ``write_server_record``, which resets the report at start with
    ``applied_target`` / ``applied_generation`` NULL: a server that has launched
    nothing has reached nothing, and a start-time guess published as an
    observation would let a reader call an unconverged fleet converged
  - ``publish_switch``, the one publisher that ends that null — it is called
    once a child has answered its init frame
  - the three publication blocks (``last_switch``, ``reachability``,
    ``last_posture_realign``): each merges without clobbering a sibling, each
    is synchronous, and an ``applying`` block carries the ``expires_at`` only
    the publishing server can compute
  - ``publish_targets``, which re-renders the display metadata a narrowed
    session outgrew, on the same merge terms
  - the in-flight marker reader in its new home, still importable from
    ``tools/control_target.py`` under its old name

The publishers being SYNCHRONOUS is a correctness property, not a style choice:
the reconciler and the endpoint prober publish from the same event loop, and an
``await`` between ``_update``'s read and its write would let one of them write
back a report built before the other's change. It is pinned here.

Every payload this module writes is asserted to round-trip through
``control_context.parse_report``: the library is the only reader, and a writer
that drifted from the ten fields it parses would degrade silently.
"""

import inspect
import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors import control_context

TARGETS_META = {
    "live": {"label": "ALS storage ring", "endpoint": "gw:5064", "real_machine": True},
    "va": {"label": "Virtual accelerator", "endpoint": "localhost:5074", "real_machine": False},
    "standin": {"label": "Live stand-in", "endpoint": "localhost:5084", "real_machine": False},
}

#: The report's whole vocabulary, in ``ServerReport``'s terms. A writer that
#: adds a key here writes something no reader parses; one that drops a key
#: writes a report a reader has to guess at.
REPORT_FIELDS = {
    "server_pid",
    "session",
    "applied_target",
    "applied_generation",
    "children",
    "reachability",
    "last_switch",
    "last_posture_realign",
    "targets",
    "updated_at",
}


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state directory in tmp_path instead of a real deployment.

    The environment stamp is cleared as well as the config derivation patched,
    so this fixture pins the directory whichever of the two resolution rules
    ``state_dir`` is applying. ``OSPREY_POSTURE_SESSION`` is cleared too: the
    session a report carries is read from the environment, and a test that
    inherited the runner's would assert against the machine it ran on.
    """
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def started(state_root):
    """A server whose report exists, so a merge has something to merge into."""
    target_state.write_server_record(TARGETS_META, server_pid=os.getpid(), session="sess-1")
    return os.getpid()


def _dead_pid(monkeypatch, dead):
    """Make ``is_process_alive`` report *dead* as gone and everything else alive."""
    real = os.kill

    def fake_kill(pid, sig):
        if pid in dead:
            raise ProcessLookupError
        return real(os.getpid(), 0)

    monkeypatch.setattr(target_state.os, "kill", fake_kill)


def _reparsed(server_pid=None):
    """This server's report as the library parses it back off disk."""
    return control_context.read_report(target_state.report_file_path(server_pid))


# ---------------------------------------------------------------------------
# Switch requests
# ---------------------------------------------------------------------------


class TestRequestFileContract:
    """Named for the process that ASKED, not the one that answers.

    Polarity inverted: any process may ask, the record's owner consumes, and
    the pid in the name is the requester's own — so a request is a slot each
    requester owns, its residue is swept when that requester dies, and no
    successor at an answering server's pid can inherit a gesture made at
    another one.
    """

    def test_path_is_named_for_the_requester(self, state_root):
        assert target_state.request_file_path(4321) == (
            state_root / target_state.STATE_DIR_NAME / "switch_request_4321.json"
        )

    def test_path_defaults_to_this_process(self, state_root):
        assert target_state.request_file_path().name == f"switch_request_{os.getpid()}.json"

    def test_glob_matches_the_file_the_writer_produces(self, state_root):
        target_state.write_request({"request_id": "r1", "target": "live", "requested_by_pid": 4321})

        directory = state_root / target_state.STATE_DIR_NAME
        assert [p.name for p in directory.glob(target_state.REQUEST_FILE_GLOB)] == [
            "switch_request_4321.json"
        ]

    def test_request_glob_never_matches_a_report(self, started):
        directory = target_state.state_dir()
        assert list(directory.glob(target_state.REQUEST_FILE_GLOB)) == []

    def test_report_glob_never_matches_a_request_file(self, state_root):
        target_state.write_request({"request_id": "r1", "target": "live", "requested_by_pid": 4321})

        directory = state_root / target_state.STATE_DIR_NAME
        assert list(directory.glob(target_state.REPORT_FILE_GLOB)) == []


class TestWriteRequest:
    def test_round_trips_the_record(self, state_root):
        record = {
            "request_id": "req-1",
            "target": "standin",
            "requested_at": "2026-08-30T10:00:00+00:00",
            "session": "sess-1",
            "requested_by_pid": 4321,
        }

        path = target_state.write_request(record)

        assert json.loads(path.read_text(encoding="utf-8")) == record
        assert target_state.read_file(target_state.request_file_path(4321)) == record

    def test_a_session_less_requester_writes_a_null_session(self, state_root):
        """A bare ``claude`` asks the same way every other process does."""
        target_state.write_request(
            {"request_id": "r", "target": "va", "session": None, "requested_by_pid": 4321}
        )

        assert target_state.read_file(target_state.request_file_path(4321))["session"] is None

    def test_requested_at_is_stamped_when_the_caller_omits_it(self, state_root):
        """A request that cannot be aged could never expire, so it is never written."""
        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})

        record = target_state.read_file(target_state.request_file_path(4321))
        assert target_state.is_request_fresh(record)
        datetime.fromisoformat(record["requested_at"])  # parseable, not just present

    def test_creates_the_state_directory(self, state_root):
        assert not (state_root / target_state.STATE_DIR_NAME).exists()

        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})

        assert (state_root / target_state.STATE_DIR_NAME).is_dir()

    def test_a_second_request_replaces_the_first(self, state_root):
        target_state.write_request({"request_id": "one", "target": "va", "requested_by_pid": 4321})
        target_state.write_request(
            {"request_id": "two", "target": "live", "requested_by_pid": 4321}
        )

        assert target_state.read_file(target_state.request_file_path(4321))["request_id"] == "two"

    @pytest.mark.parametrize(
        "record",
        [
            {"request_id": "r", "target": "va"},
            {"request_id": "r", "target": "va", "requested_by_pid": "not a pid"},
            {"request_id": "r", "target": "va", "requested_by_pid": None},
            {"request_id": "r", "target": "va", "requested_by_pid": 0},
            "not a mapping",
        ],
    )
    def test_a_request_from_nobody_is_a_programming_error(self, record, state_root):
        with pytest.raises(ValueError):
            target_state.write_request(record)

    def test_the_addressed_server_is_no_longer_part_of_the_record(self, state_root):
        """A request says what was asked for, never which server must answer it."""
        with pytest.raises(ValueError):
            target_state.write_request({"request_id": "r", "target": "va", "server_pid": 4321})

    def test_a_failed_write_leaves_no_temp_file(self, state_root, monkeypatch):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)

        def fail(*args, **kwargs):
            raise OSError("rename refused")

        monkeypatch.setattr(target_state.os, "replace", fail)

        with pytest.raises(OSError):
            target_state.write_request(
                {"request_id": "r", "target": "va", "requested_by_pid": 4321}
            )

        assert list(directory.iterdir()) == []


class TestRequestReadBack:
    """What the read-back means once the OWNER is the one that unlinks.

    The slot belongs to the requester, so only two things can happen between
    the write and the read-back: the owner consumes the request — which is the
    request succeeding, at once — or this same process writes a second one over
    it. Only the second is a supersession.
    """

    def test_a_request_consumed_before_the_read_back_still_landed(self, state_root, monkeypatch):
        real_read = target_state.read_file

        def consume(path):
            path.unlink(missing_ok=True)
            return real_read(path)

        monkeypatch.setattr(target_state, "read_file", consume)

        path = target_state.write_request(
            {"request_id": "r", "target": "va", "requested_by_pid": 4321}
        )

        assert path == target_state.request_file_path(4321)
        assert not path.exists()

    def test_a_different_request_id_in_the_slot_is_superseded(self, state_root, monkeypatch):
        monkeypatch.setattr(target_state, "read_file", lambda path: {"request_id": "theirs"})

        with pytest.raises(target_state.RequestSuperseded):
            target_state.write_request(
                {"request_id": "mine", "target": "va", "requested_by_pid": 4321}
            )

    def test_an_unreadable_slot_is_not_superseded(self, state_root, monkeypatch):
        """Nothing readable says another id won; the caller waits on the record."""
        monkeypatch.setattr(target_state, "read_file", lambda path: None)

        target_state.write_request({"request_id": "mine", "target": "va", "requested_by_pid": 4321})


class TestReadAndRemoveRequest:
    def test_absent_request_reads_as_none(self, state_root):
        assert target_state.read_file(target_state.request_file_path(4321)) is None

    def test_corrupt_request_reads_as_none(self, state_root):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "switch_request_4321.json").write_text("{not json", encoding="utf-8")

        assert target_state.read_file(target_state.request_file_path(4321)) is None

    def test_read_defaults_to_this_process(self, state_root):
        target_state.write_request(
            {"request_id": "mine", "target": "va", "requested_by_pid": os.getpid()}
        )

        assert target_state.read_file(target_state.request_file_path())["request_id"] == "mine"

    def test_remove_deletes_the_request(self, state_root):
        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})

        target_state.remove_request(4321)

        assert target_state.read_file(target_state.request_file_path(4321)) is None

    def test_remove_is_idempotent(self, state_root):
        target_state.remove_request(4321)
        target_state.remove_request(4321)  # no exception is the assertion

    def test_remove_only_touches_the_named_requester(self, state_root):
        target_state.write_request({"request_id": "a", "target": "va", "requested_by_pid": 4321})
        target_state.write_request({"request_id": "b", "target": "va", "requested_by_pid": 4322})

        target_state.remove_request(4321)

        assert target_state.read_file(target_state.request_file_path(4322))["request_id"] == "b"


class TestRequestFreshness:
    """One TTL spelling for the route that refuses a duplicate and the
    reconciler that expires a late one: the window shown is the window kept."""

    def test_ttl_is_thirty_seconds(self):
        assert target_state.REQUEST_TTL_S == 30

    def test_a_just_written_request_is_fresh(self, state_root):
        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})

        pending = target_state.read_file(target_state.request_file_path(4321))
        assert target_state.is_request_fresh(pending) is True

    def test_a_request_older_than_the_ttl_is_not_fresh(self):
        asked = datetime.now(UTC) - timedelta(seconds=target_state.REQUEST_TTL_S + 1)

        assert not target_state.is_request_fresh({"requested_at": asked.isoformat()})

    def test_the_boundary_is_inclusive(self):
        now = datetime.now(UTC)
        record = {"requested_at": (now - timedelta(seconds=30)).isoformat()}

        assert target_state.is_request_fresh(record, now=now.timestamp())

    def test_a_naive_stamp_is_read_as_utc(self):
        naive = datetime.now(UTC).replace(tzinfo=None)

        assert target_state.is_request_fresh({"requested_at": naive.isoformat()})

    def test_a_small_clock_skew_into_the_future_is_tolerated(self):
        asked = datetime.now(UTC) + timedelta(seconds=5)

        assert target_state.is_request_fresh({"requested_at": asked.isoformat()})

    @pytest.mark.parametrize(
        "record",
        [
            None,
            {},
            "not a mapping",
            {"requested_at": ""},
            {"requested_at": "yesterday"},
            {"requested_at": 5},
            {"created_at": "2026-08-30T10:00:00+00:00"},
        ],
    )
    def test_a_request_that_cannot_be_aged_is_never_fresh(self, record):
        """Fail-closed: acting on an unaged request is the surprise the TTL prevents."""
        assert target_state.is_request_fresh(record) is False


class TestRequestSweep:
    def test_sweep_removes_a_request_from_a_dead_requester(self, state_root, monkeypatch):
        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})
        _dead_pid(monkeypatch, {4321})

        target_state.sweep_stale(server_pid=os.getpid())

        assert target_state.read_file(target_state.request_file_path(4321)) is None

    def test_sweep_leaves_a_request_from_a_live_requester_alone(self, state_root, monkeypatch):
        target_state.write_request({"request_id": "r", "target": "va", "requested_by_pid": 4321})
        _dead_pid(monkeypatch, set())

        target_state.sweep_stale(server_pid=os.getpid())

        assert target_state.read_file(target_state.request_file_path(4321))["request_id"] == "r"

    def test_sweep_removes_a_request_whose_name_encodes_no_pid(self, state_root):
        directory = state_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        junk = directory / "switch_request_nonsense.json"
        junk.write_text("{}", encoding="utf-8")

        target_state.sweep_stale(server_pid=os.getpid())

        assert not junk.exists()

    def test_a_swept_request_contributes_no_orphans(self, state_root, monkeypatch):
        """Requests are not reports: they own no connector-host children."""
        target_state.write_request(
            {"request_id": "r", "target": "va", "requested_by_pid": 4321, "children": [777]}
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

        assert marker.exists(), "the marker's own reader sweeps it; the report sweep must not"

    def test_writing_the_report_drops_a_request_this_pid_left_behind(self, state_root):
        """Only a dead predecessor can have left one: this server has asked for nothing."""
        target_state.write_request(
            {"request_id": "stale", "target": "live", "requested_by_pid": os.getpid()}
        )

        target_state.write_server_record(TARGETS_META, server_pid=os.getpid())

        assert target_state.read_file(target_state.request_file_path()) is None


# ---------------------------------------------------------------------------
# The report file itself
# ---------------------------------------------------------------------------


class TestReportFileContract:
    """One file per server, named for its PID, in the family the library owns."""

    def test_the_family_is_the_librarys_own(self):
        assert target_state.REPORT_FILE_PREFIX is control_context.REPORT_FILE_PREFIX
        assert target_state.REPORT_FILE_SUFFIX is control_context.REPORT_FILE_SUFFIX
        assert target_state.REPORT_FILE_GLOB is control_context.REPORT_FILE_GLOB

    def test_path_is_named_for_the_reporting_server(self, state_root):
        assert target_state.report_file_path(4321) == (
            state_root / target_state.STATE_DIR_NAME / "server_4321.json"
        )

    def test_path_defaults_to_this_process(self, state_root):
        assert target_state.report_file_path().name == f"server_{os.getpid()}.json"

    def test_the_writer_and_the_library_name_the_same_file(self, started):
        assert target_state.report_file_path(started) == control_context.report_path_under(
            target_state.state_dir().parent, started
        )

    def test_glob_matches_the_file_the_writer_produces(self, started):
        directory = target_state.state_dir()

        assert [p.name for p in directory.glob(target_state.REPORT_FILE_GLOB)] == [
            f"server_{started}.json"
        ]


class TestWriteServerRecord:
    def test_writes_exactly_the_fields_the_library_parses(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert set(target_state.read(1234)) == REPORT_FIELDS

    def test_the_payload_round_trips_through_parse_report(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234, session="s")

        payload = target_state.read(1234)
        assert control_context.parse_report(payload).to_payload() == payload

    def test_a_fresh_report_has_reached_nothing(self, state_root):
        """Null is not the baseline: this server has launched no child yet, and a
        reader that took a start-time guess for an observation would count an
        unconverged fleet as converged."""
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        report = target_state.read(1234)
        assert report["applied_target"] is None
        assert report["applied_generation"] is None

    def test_the_report_names_its_own_server(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.read(1234)["server_pid"] == 1234

    def test_server_pid_defaults_to_this_process(self, state_root):
        target_state.write_server_record(TARGETS_META)

        assert target_state.read()["server_pid"] == os.getpid()

    def test_the_session_comes_from_the_environment(self, state_root, monkeypatch):
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "sess-from-env")

        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.read(1234)["session"] == "sess-from-env"

    def test_an_explicit_session_wins(self, state_root, monkeypatch):
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "sess-from-env")

        target_state.write_server_record(TARGETS_META, server_pid=1234, session="explicit")

        assert target_state.read(1234)["session"] == "explicit"

    def test_a_bare_claude_reports_no_session(self, state_root):
        """A server nobody stamped reports the same way every other one does."""
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.read(1234)["session"] is None

    def test_every_target_slot_is_always_present(self, state_root):
        target_state.write_server_record({"live": {"label": "Live"}}, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert set(targets) == set(target_state.TARGET_NAMES)
        assert targets["va"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_children_known_at_start_are_recorded(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234, children=[901, 901, 902])

        assert target_state.read(1234)["children"] == [901, 902]

    def test_updated_at_is_stamped(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        stamp = target_state.read(1234)["updated_at"]
        assert (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() < 60

    def test_creates_the_state_directory(self, state_root):
        assert not (state_root / target_state.STATE_DIR_NAME).exists()

        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert (state_root / target_state.STATE_DIR_NAME).is_dir()


class TestPublishSwitch:
    """The one publisher that ends the null: a child has answered its init frame."""

    def test_publishes_what_the_child_reached(self, started):
        assert target_state.publish_switch("live", 4) is True

        report = target_state.read()
        assert report["applied_target"] == "live"
        assert report["applied_generation"] == 4

    def test_it_can_carry_the_child_pids(self, started):
        target_state.publish_switch("live", 1, children=[901, 902])

        assert target_state.read()["children"] == [901, 902]

    def test_the_children_are_left_alone_when_the_caller_says_nothing(self, started):
        target_state.record_child_pids([901])

        target_state.publish_switch("live", 1)

        assert target_state.read()["children"] == [901]

    def test_a_standin_binding_round_trips(self, started):
        target_state.publish_switch("standin", 2)

        assert _reparsed().applied_target == "standin"

    def test_updated_at_advances(self, started):
        before = target_state.read()["updated_at"]

        target_state.publish_switch("live", 1)

        assert target_state.read()["updated_at"] >= before

    def test_without_a_report_writes_nothing(self, state_root):
        assert target_state.publish_switch("live", 1) is False
        assert target_state.read() is None


class TestPublishersAreSynchronous:
    """A coroutine here could interleave inside ``_update``'s read-then-write."""

    @pytest.mark.parametrize(
        "name",
        [
            "publish_last_switch",
            "publish_reachability",
            "publish_posture_realign",
            "publish_switch",
            "publish_targets",
            "record_child_pids",
            "write_server_record",
        ],
    )
    def test_publisher_is_not_a_coroutine_function(self, name):
        assert not inspect.iscoroutinefunction(getattr(target_state, name))


class TestApplyingBound:
    """The deadline no reader can compute: the timeouts are this server's."""

    def test_the_spawn_and_probe_pair_counts_twice(self, state_root):
        """A probe failure is retried once through the read-only gateway."""
        bound = target_state.applying_bound_s(
            spawn_timeout_s=10, probe_timeout_s=4, drain_timeout_s=5
        )

        assert bound == 5 + 2 * (10 + 4)

    def test_a_launch_that_cannot_retry_counts_the_pair_once(self, state_root):
        bound = target_state.applying_bound_s(
            spawn_timeout_s=10, probe_timeout_s=4, drain_timeout_s=5, fallback_retry=False
        )

        assert bound == 5 + 10 + 4


class TestPublishLastSwitch:
    def test_publishes_a_terminus(self, started):
        outcome = {
            "generation": 4,
            "status": target_state.SWITCH_APPLIED,
            "reason": None,
            "detail": "switched to live",
            "at": "2026-08-30T10:00:00+00:00",
        }

        assert target_state.publish_last_switch(outcome) is True
        assert target_state.read()["last_switch"] == outcome

    def test_the_generation_travels_with_the_block(self, started):
        """A reader matches a server's progress by generation, never by target:
        two servers can be on the same target at different generations."""
        target_state.publish_last_switch({"generation": 9, "status": target_state.SWITCH_FAILED})

        assert target_state.read()["last_switch"]["generation"] == 9

    def test_at_is_stamped_when_the_caller_omits_it(self, started):
        target_state.publish_last_switch({"generation": 1, "status": target_state.SWITCH_FAILED})

        block = target_state.read()["last_switch"]
        datetime.fromisoformat(block["at"])

    def test_a_failure_reason_travels_verbatim(self, started):
        """The vocabulary is the switch lifecycle's; this module never edits it."""
        target_state.publish_last_switch(
            {
                "generation": 2,
                "status": target_state.SWITCH_FAILED,
                "reason": "spawn_failed",
                "detail": "the child never answered",
            }
        )

        block = target_state.read()["last_switch"]
        assert block["reason"] == "spawn_failed"
        assert block["detail"] == "the child never answered"

    def test_an_applying_block_carries_its_own_deadline(self, started):
        """Every other process only ever sees this file, so the bound is written
        into it by the one process that knows its own timeouts."""
        target_state.publish_last_switch(
            {
                "generation": 3,
                "status": target_state.SWITCH_APPLYING,
                "at": "2026-08-30T10:00:00+00:00",
            },
            expires_in_s=33,
        )

        block = target_state.read()["last_switch"]
        assert block["expires_at"] == "2026-08-30T10:00:33+00:00"

    def test_the_deadline_is_measured_from_the_stamp_it_publishes(self, started):
        target_state.publish_last_switch(
            {"generation": 3, "status": target_state.SWITCH_APPLYING}, expires_in_s=30
        )

        block = target_state.read()["last_switch"]
        span = datetime.fromisoformat(block["expires_at"]) - datetime.fromisoformat(block["at"])
        assert span == timedelta(seconds=30)

    def test_a_caller_supplied_deadline_is_kept(self, started):
        target_state.publish_last_switch(
            {
                "generation": 3,
                "status": target_state.SWITCH_APPLYING,
                "expires_at": "2027-01-01T00:00:00+00:00",
            },
            expires_in_s=30,
        )

        assert target_state.read()["last_switch"]["expires_at"] == "2027-01-01T00:00:00+00:00"

    def test_an_applying_block_without_a_bound_stays_unbounded(self, started, caplog):
        """A reader with no deadline keeps waiting — the fail-closed outcome —
        and this module does not invent timeouts it does not hold."""
        with caplog.at_level("WARNING"):
            target_state.publish_last_switch(
                {"generation": 3, "status": target_state.SWITCH_APPLYING}
            )

        assert "expires_at" not in target_state.read()["last_switch"]
        assert any("expires_at" in record.message for record in caplog.records)

    @pytest.mark.parametrize("status", ["applied", "failed"])
    def test_a_terminus_gets_no_deadline(self, started, status):
        target_state.publish_last_switch({"generation": 3, "status": status}, expires_in_s=30)

        assert "expires_at" not in target_state.read()["last_switch"]

    def test_none_clears_the_block(self, started):
        target_state.publish_last_switch({"generation": 1, "status": target_state.SWITCH_APPLIED})

        target_state.publish_last_switch(None)

        assert target_state.read()["last_switch"] is None

    def test_preserves_identity_and_display_metadata(self, started):
        target_state.publish_last_switch({"generation": 1, "status": target_state.SWITCH_APPLIED})

        report = target_state.read()
        assert report["server_pid"] == started
        assert report["session"] == "sess-1"
        assert report["targets"] == TARGETS_META

    def test_without_a_report_writes_nothing(self, state_root):
        assert target_state.publish_last_switch({"generation": 1}) is False
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
        """Empty, not null: the field always holds the mapping a reader parses,
        and "this server has not probed" renders ``unknown`` rather than down."""
        target_state.publish_reachability(self.ROWS)

        target_state.publish_reachability(rows)

        assert target_state.read()["reachability"] == {}
        assert _reparsed().reachability == {}

    def test_without_a_report_writes_nothing(self, state_root):
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

    def test_without_a_report_writes_nothing(self, state_root):
        assert target_state.publish_posture_realign({"state": "pending"}) is False


class TestPublishTargets:
    """Display metadata is re-rendered by the writer, never by a reader."""

    NARROWED = {
        "live": {
            "label": "ALS storage ring",
            "endpoint": "gw:5065",
            "real_machine": True,
            "selected_role": "read_only",
        },
        "va": {"label": "Virtual accelerator", "endpoint": "localhost:5074"},
        "standin": {"label": "Live stand-in", "endpoint": "localhost:5084"},
    }

    def test_publishes_the_new_block(self, started):
        assert target_state.publish_targets(self.NARROWED) is True

        targets = target_state.read()["targets"]
        assert targets["live"]["endpoint"] == "gw:5065"
        assert targets["live"]["selected_role"] == "read_only"

    def test_the_sibling_blocks_survive(self, started):
        target_state.publish_last_switch({"generation": 1, "status": target_state.SWITCH_APPLIED})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "done"})
        target_state.record_child_pids([901])

        target_state.publish_targets(self.NARROWED)

        report = target_state.read()
        assert report["last_switch"]["generation"] == 1
        assert report["reachability"]["targets"]["live"]["epics"]["state"] == "reached"
        assert report["last_posture_realign"]["state"] == "done"
        assert report["children"] == [901]

    def test_the_binding_is_untouched(self, started):
        """This publisher says what a target IS, never which one a child reached."""
        target_state.publish_switch("live", 4)

        target_state.publish_targets(self.NARROWED)

        report = target_state.read()
        assert report["applied_target"] == "live"
        assert report["applied_generation"] == 4

    def test_an_empty_selected_role_is_dropped(self, started):
        target_state.publish_targets({"live": {"label": "Live", "selected_role": ""}})

        assert "selected_role" not in target_state.read()["targets"]["live"]

    def test_without_a_report_writes_nothing(self, state_root):
        assert target_state.publish_targets(self.NARROWED) is False
        assert target_state.read() is None


class TestBlocksDoNotClobberOneAnother:
    """Several publishers merge into one file; each must be blind to the others."""

    def test_every_block_survives_every_other_publisher(self, started):
        target_state.publish_last_switch({"generation": 3, "status": target_state.SWITCH_APPLIED})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "pending"})
        target_state.publish_switch("live", 3, children=[901])
        target_state.record_child_pids([902])

        report = target_state.read()
        assert report["last_switch"]["generation"] == 3
        assert report["reachability"]["targets"]["live"]["epics"]["state"] == "reached"
        assert report["last_posture_realign"]["state"] == "pending"
        assert report["applied_target"] == "live"
        assert report["applied_generation"] == 3
        assert report["children"] == [902]
        assert report["targets"] == TARGETS_META
        assert set(report) == REPORT_FIELDS

    def test_publish_switch_does_not_clear_the_blocks(self, started):
        target_state.publish_last_switch(
            {"generation": 1, "status": target_state.SWITCH_APPLYING}, expires_in_s=30
        )

        target_state.publish_switch("live", 1)

        assert target_state.read()["last_switch"]["generation"] == 1


class TestWriteServerRecordResetsThePublications:
    def test_a_fresh_report_carries_no_publications(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        report = target_state.read(1234)
        assert report["last_switch"] is None
        assert report["reachability"] == {}
        assert report["last_posture_realign"] is None

    def test_a_restart_drops_the_predecessors_publications(self, started):
        target_state.publish_last_switch({"generation": 1, "status": target_state.SWITCH_APPLIED})
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "pending"})
        target_state.publish_switch("live", 1)

        target_state.write_server_record(TARGETS_META, server_pid=os.getpid())

        report = target_state.read()
        assert report["last_switch"] is None
        assert report["reachability"] == {}
        assert report["last_posture_realign"] is None
        assert report["applied_target"] is None
        assert report["applied_generation"] is None


class TestTheLibraryReadsWhatThisWrites:
    """``control_context`` is the only reader of these files. Every state this
    module can publish has to survive its parser."""

    def test_a_published_report_parses(self, started):
        target_state.publish_switch("live", 5, children=[901])
        target_state.publish_last_switch(
            {"generation": 6, "status": target_state.SWITCH_APPLYING}, expires_in_s=30
        )
        target_state.publish_reachability({"live": {"epics": {"state": "reached"}}})
        target_state.publish_posture_realign({"state": "pending"})

        report = _reparsed()

        assert report.server_pid == started
        assert report.session == "sess-1"
        assert report.applied_target == "live"
        assert report.applied_generation == 5
        assert report.children == (901,)
        assert report.last_switch["status"] == target_state.SWITCH_APPLYING
        assert report.last_switch["expires_at"]
        assert report.last_posture_realign["state"] == "pending"
        assert report.reachability["targets"]["live"]["epics"]["state"] == "reached"
        assert report.updated_at

    def test_a_fresh_report_parses_as_having_reached_nothing(self, started):
        report = _reparsed()

        assert report.applied_target is None
        assert report.applied_generation is None

    def test_the_fleet_reader_sees_a_live_server(self, started):
        reports = control_context.live_reports(
            target_state.state_dir().glob(target_state.REPORT_FILE_GLOB)
        )

        assert [report.server_pid for report in reports] == [started]

    def test_a_dead_servers_report_is_swept_and_its_children_salvaged(
        self, state_root, monkeypatch
    ):
        target_state.write_server_record(TARGETS_META, server_pid=4321, session="gone")
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
        target_state.write_request(
            {"request_id": "r", "target": "va", "requested_by_pid": os.getpid()}
        )

        assert target_state.in_flight_executions() == []

    def test_a_report_is_not_mistaken_for_a_marker(self, started):
        assert target_state.in_flight_executions() == []
