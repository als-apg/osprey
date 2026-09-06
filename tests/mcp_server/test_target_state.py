"""Tests for the single-writer per-server report file.

Covers:
  - the path contract a stdlib-only hook has to be able to restate
  - write_server_record: reset at start, PID capture, display metadata
  - publish_switch / publish_targets / record_child_pids merges
  - the optional display keys (probe_channel, selected_role): preserved when
    real, absent when the caller has none
  - fail-closed reads (absent, corrupt, non-object)
  - stale-PID sweep: deletion, orphan child PIDs, own file preserved
  - delete_on_shutdown idempotence
  - atomic writes leaving no temp litter when the dump fails
  - server start: the claim over an ownerless record, and the first child's
    target coming from that record rather than from the deployment baseline
"""

import json
import logging
import os

import pytest

from osprey.mcp_server.control_system import server, server_context, target_state
from osprey_connectors import control_context

TARGETS_META = {
    "live": {
        "label": "ALS storage ring",
        "endpoint": "gateway.example.com:5064",
        "real_machine": True,
        "probe_channel": "SR:BeamCurrent",
    },
    "va": {
        "label": "Virtual accelerator",
        "endpoint": "localhost:5074",
        "real_machine": False,
        "probe_channel": "VA:BeamCurrent",
    },
    "standin": {
        "label": "Live stand-in",
        "endpoint": "localhost:5084",
        "real_machine": False,
        "probe_channel": "STANDIN:BeamCurrent",
    },
}


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state directory in tmp_path instead of a real deployment."""
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path


def _write_foreign(state_root, pid, *, children=None, target="live"):
    """Drop a report that looks like another server's, bypassing the API."""
    directory = state_root / target_state.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target_state.REPORT_FILE_PREFIX}{pid}{target_state.REPORT_FILE_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "server_pid": pid,
                "session": f"sess-{pid}",
                "applied_target": target,
                "applied_generation": 3,
                "targets": TARGETS_META,
                "children": list(children or []),
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Path contract
# ---------------------------------------------------------------------------


class TestPathContract:
    """The spelling a stdlib-only hook mirrors: root / control_target / PID file."""

    def test_state_dir_is_fixed_subdir_of_shared_root(self, state_root):
        assert target_state.state_dir() == state_root / "control_target"

    def test_report_is_named_for_the_server_pid(self, state_root):
        assert target_state.report_file_path(4321) == (
            state_root / "control_target" / "server_4321.json"
        )

    def test_report_defaults_to_this_process(self, state_root):
        assert target_state.report_file_path().name == f"server_{os.getpid()}.json"

    def test_glob_matches_the_file_the_writer_produces(self, state_root):
        path = _write_foreign(state_root, 4321)
        matched = list(target_state.state_dir().glob(target_state.REPORT_FILE_GLOB))
        assert matched == [path]


# ---------------------------------------------------------------------------
# write_server_record
# ---------------------------------------------------------------------------


class TestWriteServerRecord:
    def test_writes_the_report(self, state_root, monkeypatch):
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "sess-1")

        target_state.write_server_record(TARGETS_META, server_pid=1234)

        record = json.loads(target_state.report_file_path(1234).read_text(encoding="utf-8"))
        assert record == {
            "server_pid": 1234,
            "session": "sess-1",
            "applied_target": None,
            "applied_generation": None,
            "targets": TARGETS_META,
            "children": [],
            "reachability": {},
            "last_switch": None,
            "last_posture_realign": None,
            "updated_at": record["updated_at"],
        }

    def test_captures_own_pid_by_default(self, state_root):
        target_state.write_server_record(TARGETS_META)

        assert target_state.read()["server_pid"] == os.getpid()

    def test_a_restart_reports_having_reached_nothing_again(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 7, server_pid=1234)

        target_state.write_server_record(TARGETS_META, server_pid=1234)

        record = target_state.read(1234)
        assert record["applied_target"] is None
        assert record["applied_generation"] is None

    def test_every_target_slot_always_present(self, state_root):
        target_state.write_server_record({"live": {"label": "Live"}}, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert set(targets) == set(target_state.TARGET_NAMES)
        assert targets["live"] == {"label": "Live", "endpoint": "", "real_machine": False}
        assert targets["va"] == {"label": "", "endpoint": "", "real_machine": False}
        assert targets["standin"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_target_names_has_three_slots(self):
        assert target_state.TARGET_NAMES == ("live", "va", "standin")
        assert target_state.TARGET_STANDIN == "standin"

    def test_unconfigured_standin_is_absent_as_empty_like_va(self, state_root):
        """A deployment with no stand-in still carries the slot, empty."""
        meta = {"live": {"label": "Live", "endpoint": "gw:5064", "real_machine": True}}
        target_state.write_server_record(meta, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert targets["standin"] == targets["va"]
        assert targets["standin"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_creates_the_state_directory(self, state_root):
        assert not (state_root / "control_target").exists()
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        assert (state_root / "control_target").is_dir()


# ---------------------------------------------------------------------------
# probe_channel round-tripping
# ---------------------------------------------------------------------------


class TestProbeChannel:
    """The approval describer names the probe channel from this file alone."""

    def test_present_probe_channel_is_preserved(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert targets["live"]["probe_channel"] == "SR:BeamCurrent"
        assert targets["va"]["probe_channel"] == "VA:BeamCurrent"
        assert targets["standin"]["probe_channel"] == "STANDIN:BeamCurrent"

    def test_absent_probe_channel_stays_absent(self, state_root):
        meta = {
            "live": {"label": "Live", "endpoint": "gw:5064", "real_machine": True},
            "va": {"label": "VA", "endpoint": "localhost:5074", "real_machine": False},
            "standin": {"label": "Stand-in", "endpoint": "localhost:5084", "real_machine": False},
        }
        target_state.write_server_record(meta, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert "probe_channel" not in targets["live"]
        assert "probe_channel" not in targets["va"]
        assert "probe_channel" not in targets["standin"]

    @pytest.mark.parametrize("bogus", ["", None, 5064, ["SR:BeamCurrent"]])
    def test_unusable_probe_channel_is_dropped_never_stringified(self, state_root, bogus):
        meta = {"live": {"label": "Live", "probe_channel": bogus}}
        target_state.write_server_record(meta, server_pid=1234)

        assert "probe_channel" not in target_state.read(1234)["targets"]["live"]

    def test_probe_channel_survives_a_switch(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        assert target_state.read(1234)["targets"] == TARGETS_META


# ---------------------------------------------------------------------------
# selected_role round-tripping
# ---------------------------------------------------------------------------


class TestSelectedRole:
    """The role whose gateway the endpoint belongs to travels with the endpoint."""

    ROLED_META = {
        "live": {
            "label": "ALS storage ring",
            "endpoint": "gateway.example.com:5064",
            "real_machine": True,
            "selected_role": "read_only",
        },
        "va": {
            "label": "Virtual accelerator",
            "endpoint": "localhost:5074",
            "real_machine": False,
            "selected_role": "writes",
        },
        "standin": {
            "label": "Live stand-in",
            "endpoint": "localhost:5084",
            "real_machine": False,
            "selected_role": "read_only",
        },
    }

    def test_present_selected_role_is_preserved(self, state_root):
        target_state.write_server_record(self.ROLED_META, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert targets["live"]["selected_role"] == "read_only"
        assert targets["va"]["selected_role"] == "writes"
        assert targets["standin"]["selected_role"] == "read_only"

    def test_absent_selected_role_stays_absent(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert "selected_role" not in targets["live"]
        assert "selected_role" not in targets["va"]
        assert "selected_role" not in targets["standin"]

    @pytest.mark.parametrize("bogus", ["", None, 5064, ["read_only"]])
    def test_unusable_selected_role_is_dropped_never_stringified(self, state_root, bogus):
        meta = {"live": {"label": "Live", "selected_role": bogus}}
        target_state.write_server_record(meta, server_pid=1234)

        assert "selected_role" not in target_state.read(1234)["targets"]["live"]

    def test_selected_role_survives_a_switch(self, state_root):
        target_state.write_server_record(self.ROLED_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        assert target_state.read(1234)["targets"] == self.ROLED_META

    def test_selected_role_survives_a_republish(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.publish_targets(self.ROLED_META, server_pid=1234) is True

        assert target_state.read(1234)["targets"] == self.ROLED_META

    def test_an_empty_role_is_dropped_on_a_republish_too(self, state_root):
        target_state.write_server_record(self.ROLED_META, server_pid=1234)

        target_state.publish_targets(
            {"live": {"label": "Live", "endpoint": "gw:5064", "selected_role": ""}},
            server_pid=1234,
        )

        assert "selected_role" not in target_state.read(1234)["targets"]["live"]


# ---------------------------------------------------------------------------
# publish_targets
# ---------------------------------------------------------------------------


class TestPublishTargets:
    """The one publisher that moves the display metadata written at start."""

    NARROWED = {
        "live": {
            "label": "ALS storage ring",
            "endpoint": "gateway.example.com:5065",
            "real_machine": True,
            "probe_channel": "SR:BeamCurrent",
            "selected_role": "read_only",
        },
    }

    def test_republishing_replaces_the_block(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.publish_targets(self.NARROWED, server_pid=1234) is True

        targets = target_state.read(1234)["targets"]
        assert targets["live"]["endpoint"] == "gateway.example.com:5065"
        assert targets["live"]["selected_role"] == "read_only"

    def test_an_omitted_slot_is_written_empty_not_dropped(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        target_state.publish_targets(self.NARROWED, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert set(targets) == set(target_state.TARGET_NAMES)
        assert targets["va"] == {"label": "", "endpoint": "", "real_machine": False}
        assert targets["standin"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_identity_and_pids_are_untouched(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 2, children=[5001], server_pid=1234)

        target_state.publish_targets(self.NARROWED, server_pid=1234)

        record = target_state.read(1234)
        assert record["applied_target"] == "va"
        assert record["applied_generation"] == 2
        assert record["children"] == [5001]
        assert record["server_pid"] == 1234

    def test_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_targets(TARGETS_META, server_pid=1234) is False
        assert target_state.read(1234) is None


# ---------------------------------------------------------------------------
# publish / children
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_updates_the_binding(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.publish_switch("va", 1, server_pid=1234) is True

        record = target_state.read(1234)
        assert record["applied_target"] == "va"
        assert record["applied_generation"] == 1

    def test_publish_preserves_display_metadata_and_the_pid(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["targets"] == TARGETS_META
        assert record["server_pid"] == 1234

    def test_standin_round_trips_as_a_first_binding(self, state_root):
        """``standin`` is a target like any other: it survives the first launch."""
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        target_state.publish_switch(target_state.TARGET_STANDIN, 0, server_pid=1234)

        record = target_state.read(1234)
        assert record["applied_target"] == "standin"
        assert record["applied_generation"] == 0
        assert record["targets"] == TARGETS_META

    def test_standin_round_trips_through_a_switch(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.publish_switch(target_state.TARGET_STANDIN, 2, server_pid=1234) is True

        record = target_state.read(1234)
        assert record["applied_target"] == "standin"
        assert record["applied_generation"] == 2
        assert record["targets"]["standin"]["endpoint"] == "localhost:5084"
        assert record["targets"] == TARGETS_META

    def test_switching_away_from_standin_back_to_live(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch(target_state.TARGET_STANDIN, 0, server_pid=1234)

        target_state.publish_switch("live", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["applied_target"] == "live"
        assert record["targets"] == TARGETS_META

    def test_publish_can_carry_child_pids(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, children=[5001, 5002], server_pid=1234)

        assert target_state.read(1234)["children"] == [5001, 5002]

    def test_publish_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_switch("va", 1, server_pid=1234) is False
        assert target_state.read(1234) is None

    def test_record_child_pids_sets_and_clears(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert target_state.record_child_pids([5001, 5001, 0, "x"], server_pid=1234) is True
        assert target_state.read(1234)["children"] == [5001]

        assert target_state.record_child_pids([], server_pid=1234) is True
        assert target_state.read(1234)["children"] == []


# ---------------------------------------------------------------------------
# Fail-closed reads
# ---------------------------------------------------------------------------


class TestRead:
    def test_absent_file_reads_as_none(self, state_root):
        assert target_state.read(1234) is None

    def test_corrupt_json_reads_as_none(self, state_root):
        path = _write_foreign(state_root, 1234)
        path.write_text("{not json", encoding="utf-8")

        assert target_state.read(1234) is None

    def test_non_object_json_reads_as_none(self, state_root):
        path = _write_foreign(state_root, 1234)
        path.write_text("[1, 2, 3]", encoding="utf-8")

        assert target_state.read(1234) is None

    def test_unreadable_file_reads_as_none(self, state_root):
        directory = state_root / "control_target"
        directory.mkdir(parents=True)
        # A directory where a file is expected: OSError, not a crash.
        (directory / "server_1234.json").mkdir()

        assert target_state.read(1234) is None


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class TestSweep:
    @staticmethod
    def _kill_with_dead(dead_pids):
        def fake_kill(pid, sig):
            if pid in dead_pids:
                raise ProcessLookupError(pid)
            return None

        return fake_kill

    def test_deletes_dead_owner_file_and_returns_its_children(self, state_root, monkeypatch):
        dead = _write_foreign(state_root, 4321, children=[5001, 5002])
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321}))

        orphans = target_state.sweep_stale(server_pid=1234)

        assert orphans == [5001, 5002]
        assert not dead.exists()

    def test_leaves_live_foreign_files_alone(self, state_root, monkeypatch):
        alive = _write_foreign(state_root, 4321, children=[5001])
        monkeypatch.setattr(os, "kill", self._kill_with_dead(set()))

        assert target_state.sweep_stale(server_pid=1234) == []
        assert alive.exists()

    def test_leaves_own_file_alone_without_probing_it(self, state_root, monkeypatch):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        # Even claiming our own PID is dead must not delete our file.
        monkeypatch.setattr(os, "kill", self._kill_with_dead({1234}))

        assert target_state.sweep_stale(server_pid=1234) == []
        assert target_state.read(1234) is not None

    def test_write_server_record_returns_the_orphans_it_swept(self, state_root, monkeypatch):
        dead = _write_foreign(state_root, 4321, children=[5001])
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321}))

        orphans = target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert orphans == [5001]
        assert not dead.exists()
        assert target_state.read(1234)["server_pid"] == 1234

    def test_corrupt_dead_file_is_removed_without_orphans(self, state_root, monkeypatch):
        dead = _write_foreign(state_root, 4321, children=[5001])
        dead.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321}))

        assert target_state.sweep_stale(server_pid=1234) == []
        assert not dead.exists()

    def test_file_with_unparseable_pid_is_swept(self, state_root):
        directory = state_root / "control_target"
        directory.mkdir(parents=True)
        junk = directory / "server_notapid.json"
        junk.write_text("{}", encoding="utf-8")

        assert target_state.sweep_stale(server_pid=1234) == []
        assert not junk.exists()

    def test_orphans_are_deduplicated_across_files(self, state_root, monkeypatch):
        _write_foreign(state_root, 4321, children=[5001, 5002])
        _write_foreign(state_root, 4322, children=[5002, 5003])
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321, 4322}))

        assert target_state.sweep_stale(server_pid=1234) == [5001, 5002, 5003]

    def test_missing_state_dir_sweeps_to_empty(self, state_root):
        assert target_state.sweep_stale(server_pid=1234) == []


class TestIsProcessAlive:
    def test_this_process_is_alive(self):
        assert target_state.is_process_alive(os.getpid()) is True

    def test_dead_pid_is_not_alive(self, monkeypatch):
        monkeypatch.setattr(os, "kill", TestSweep._kill_with_dead({4321}))
        assert target_state.is_process_alive(4321) is False

    def test_permission_error_counts_as_alive(self, monkeypatch):
        def denied(pid, sig):
            raise PermissionError(pid)

        monkeypatch.setattr(os, "kill", denied)
        assert target_state.is_process_alive(4321) is True

    def test_non_positive_pids_never_reach_os_kill(self, monkeypatch):
        def explode(pid, sig):  # pragma: no cover - must not be called
            raise AssertionError("os.kill called with a process-group pid")

        monkeypatch.setattr(os, "kill", explode)
        assert target_state.is_process_alive(0) is False
        assert target_state.is_process_alive(-1) is False

    @pytest.mark.parametrize("value", [True, False, "4321", 4321.0, None, [4321]])
    def test_anything_but_an_int_names_no_process(self, monkeypatch, value):
        """``True`` is ``1`` to ``os.kill``, and PID 1 is always alive."""

        def explode(pid, sig):  # pragma: no cover - must not be called
            raise AssertionError(f"os.kill called with {pid!r}")

        monkeypatch.setattr(os, "kill", explode)
        assert target_state.is_process_alive(value) is False


# ---------------------------------------------------------------------------
# The records that describe a session: one matcher for every child process
# ---------------------------------------------------------------------------


def _record(pid, *, owner_ppid=None, target="va", generation=3, **extra) -> dict:
    payload = {
        "target": target,
        "generation": generation,
        "server_pid": pid,
        "owner_ppid": os.getppid() if owner_ppid is None else owner_ppid,
        **extra,
    }
    path = target_state.report_file_path(pid if isinstance(pid, int) else 99999)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _entries():
    return sorted(target_state.state_dir().glob(target_state.REPORT_FILE_GLOB))


class TestRecordPid:
    @pytest.mark.parametrize("value", [True, False, "4321", 4321.0, None, 0, -1])
    def test_only_a_positive_int_is_a_pid(self, value):
        assert target_state.record_pid({"server_pid": value}, "server_pid") is None

    def test_a_positive_int_is_returned_as_is(self):
        assert target_state.record_pid({"server_pid": 4321}, "server_pid") == 4321

    def test_a_missing_field_is_none(self):
        assert target_state.record_pid({}, "server_pid") is None


class TestLiveRecords:
    def test_a_running_owner_keeps_its_record(self, state_root):
        _record(os.getpid())

        assert [r["server_pid"] for r in target_state.live_records(_entries())] == [os.getpid()]

    def test_a_dead_owner_is_residue(self, state_root, monkeypatch):
        _record(4321)
        monkeypatch.setattr(os, "kill", TestSweep._kill_with_dead({4321}))

        assert target_state.live_records(_entries()) == []

    def test_a_bool_server_pid_is_not_a_live_server(self, state_root, monkeypatch):
        """``True`` would reach ``os.kill(1, 0)`` and read as alive forever."""
        _record(True)

        def explode(pid, sig):  # pragma: no cover - must not be called
            raise AssertionError(f"os.kill called with {pid!r}")

        monkeypatch.setattr(os, "kill", explode)
        assert target_state.live_records(_entries()) == []

    def test_an_unreadable_entry_is_skipped(self, state_root):
        _record(os.getpid())
        target_state.report_file_path(4321).write_text("{not json", encoding="utf-8")

        assert len(target_state.live_records(_entries())) == 1


class TestSessionRecord:
    def test_the_record_our_parent_owns_is_ours(self, state_root):
        _record(os.getpid())

        record = target_state.session_record(_entries(), os.getppid())

        assert record is not None
        assert record["target"] == "va"

    def test_another_parents_record_is_not_ours(self, state_root):
        _record(os.getpid(), owner_ppid=os.getppid() + 100000)

        assert target_state.session_record(_entries(), os.getppid()) is None

    def test_equality_is_strict_int(self, state_root):
        """A record spelling the parent as a string or a bool matches nothing."""
        _record(os.getpid(), owner_ppid=str(os.getppid()))

        assert target_state.session_record(_entries(), os.getppid()) is None
        assert target_state.session_record(_entries(), 1) is None

    def test_a_bool_owner_never_matches_pid_one(self, state_root):
        _record(os.getpid(), owner_ppid=True)

        assert target_state.session_record(_entries(), 1) is None

    def test_a_dead_server_record_is_residue(self, state_root, monkeypatch):
        _record(4321)
        monkeypatch.setattr(os, "kill", TestSweep._kill_with_dead({4321}))

        assert target_state.session_record(_entries(), os.getppid()) is None

    def test_an_unknown_target_is_no_answer(self, state_root):
        _record(os.getpid(), target="production")

        assert target_state.session_record(_entries(), os.getppid()) is None

    def test_two_records_sharing_the_parent_are_ambiguous(self, state_root, caplog):
        _record(os.getpid())
        _record(os.getppid(), target="live")

        with caplog.at_level("WARNING"):
            assert target_state.session_record(_entries(), os.getppid()) is None

        assert "share owner_ppid" in caplog.text

    def test_generation_is_only_required_when_asked(self, state_root):
        _record(os.getpid(), generation="3")

        assert target_state.session_record(_entries(), os.getppid()) is not None
        assert (
            target_state.session_record(_entries(), os.getppid(), require_generation=True) is None
        )

    def test_a_bool_generation_does_not_pin_a_run(self, state_root):
        _record(os.getpid(), generation=True)

        assert (
            target_state.session_record(_entries(), os.getppid(), require_generation=True) is None
        )

    def test_no_entries_is_none(self, state_root):
        assert target_state.session_record([], os.getppid()) is None


# ---------------------------------------------------------------------------
# Shutdown + durability
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_delete_removes_the_file(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        target_state.delete_on_shutdown(server_pid=1234)

        assert not target_state.report_file_path(1234).exists()

    def test_delete_is_idempotent(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        target_state.delete_on_shutdown(server_pid=1234)
        target_state.delete_on_shutdown(server_pid=1234)  # missing file is fine

        assert target_state.read(1234) is None


class TestAtomicWrite:
    def test_successful_write_leaves_only_the_state_file(self, state_root):
        target_state.write_server_record(TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        directory = state_root / "control_target"
        assert [p.name for p in directory.iterdir()] == ["server_1234.json"]
        assert json.loads((directory / "server_1234.json").read_text(encoding="utf-8"))

    def test_failed_dump_leaves_no_temp_file_and_no_state_file(self, state_root, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(target_state.json, "dump", boom)

        with pytest.raises(RuntimeError):
            target_state.write_server_record(TARGETS_META, server_pid=1234)

        assert list((state_root / "control_target").iterdir()) == []

    def test_failed_dump_does_not_corrupt_the_previous_record(self, state_root, monkeypatch):
        target_state.write_server_record(TARGETS_META, server_pid=1234)

        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        with monkeypatch.context() as patched:
            patched.setattr(target_state.json, "dump", boom)
            with pytest.raises(RuntimeError):
                target_state.publish_switch("va", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["applied_target"] is None
        assert record["targets"] == TARGETS_META
        assert [p.name for p in (state_root / "control_target").iterdir()] == ["server_1234.json"]


# ---------------------------------------------------------------------------
# Server start: the report, the reaping, and the claim
# ---------------------------------------------------------------------------


def _dead_pid():
    """A PID that names nothing, so a claim over it is a claim over a corpse."""
    pid = 2
    while pid < 100000:
        if not control_context.is_process_alive(pid):
            return pid
        pid += 1
    raise AssertionError("every PID in the search range is alive")


class _FakeManager:
    """A connector-host supervisor that only records how it was asked to start."""

    def __init__(self, *, started=False, proxy="proxy"):
        self.started = started
        self.proxy = proxy
        self.ensure_started_calls = []

    def is_started(self):
        return self.started

    async def ensure_started(self, target=None):
        self.ensure_started_calls.append(target)
        self.started = True
        return True

    def active_proxy(self):
        return self.proxy

    def active_target(self):
        return "live"

    def active_generation(self):
        return 0


def _switching_context(manager):
    """A server context on the switch-capable path, serving from *manager*."""
    context = server_context.ControlSystemContext()
    context._config = server_context.MCPServerConfig(raw={})
    context._switch_capable = True
    context._connector_hosts = manager
    context._connectors["control_system"] = server_context.ConnectorEntry(
        config={}, connector_type="control_system"
    )
    return context


class TestServerStartClaimsTheRecord:
    """A fresh server owns the record only when nobody live already does."""

    def test_server_start_claims_an_ownerless_record_without_moving_it(
        self, control_context_root, write_control_context
    ):
        write_control_context(control_context_root, target="va", generation=7, owned_by=None)

        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert (claimed.target, claimed.generation) == ("va", 7)
        assert claimed.owner is not None
        assert claimed.owner.kind == control_context.OWNER_CONTROLS_SERVER
        assert claimed.owner.pid == os.getpid()
        assert claimed.owner.port is None

    def test_server_start_claim_keeps_the_posture_it_found(
        self, control_context_root, write_control_context
    ):
        write_control_context(
            control_context_root,
            target="va",
            generation=7,
            posture={"va": "sandbox"},
            owned_by=None,
        )

        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert claimed.posture == {"va": "sandbox"}

    def test_server_start_claims_over_a_dead_owner(
        self, control_context_root, write_control_context
    ):
        from tests._control_context_fixtures import owner as make_owner

        write_control_context(
            control_context_root,
            target="standin",
            generation=4,
            owned_by=make_owner(control_context.OWNER_WEB_TERMINAL, pid=_dead_pid(), port=8443),
        )

        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert claimed.owner is not None
        assert claimed.owner.kind == control_context.OWNER_CONTROLS_SERVER
        assert (claimed.target, claimed.generation) == ("standin", 4)

    def test_server_start_claims_over_an_owner_too_degraded_to_name_a_process(
        self, control_context_root
    ):
        from tests._control_context_fixtures import write_payload

        write_payload(
            control_context.record_path_under(control_context_root),
            {
                "schema": 1,
                "owner": "whoever",
                "target": "va",
                "generation": 7,
                "posture": {},
                "last_switch": None,
            },
        )

        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert claimed.owner is not None
        assert claimed.owner.pid == os.getpid()
        assert (claimed.target, claimed.generation) == ("va", 7)

    def test_server_start_follows_a_live_web_terminal_owner(
        self, control_context_root, write_control_context
    ):
        path = write_control_context(control_context_root, target="va", generation=7)
        before = path.read_text(encoding="utf-8")

        followed = server_context.claim_control_context(baseline="live")

        assert followed is not None
        assert followed.owner is not None
        assert followed.owner.kind == control_context.OWNER_WEB_TERMINAL
        assert path.read_text(encoding="utf-8") == before

    def test_server_start_follows_a_live_controls_server_owner(
        self, control_context_root, write_control_context
    ):
        from tests._control_context_fixtures import owner as make_owner

        path = write_control_context(
            control_context_root,
            target="va",
            generation=7,
            owned_by=make_owner(control_context.OWNER_CONTROLS_SERVER),
        )
        before = path.read_text(encoding="utf-8")

        followed = server_context.claim_control_context(baseline="live")

        assert followed is not None
        assert followed.owner is not None
        assert followed.owner.pid == os.getpid()
        assert path.read_text(encoding="utf-8") == before

    def test_server_start_writes_a_baseline_record_when_there_is_none(self, control_context_root):
        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert (claimed.target, claimed.generation, claimed.posture) == ("live", 0, {})
        assert claimed.owner is not None
        assert claimed.owner.kind == control_context.OWNER_CONTROLS_SERVER
        assert control_context.record_path_under(control_context_root).exists()

    def test_server_start_writes_a_baseline_record_over_an_unreadable_one(
        self, control_context_root
    ):
        from tests._control_context_fixtures import write_payload

        write_payload(control_context.record_path_under(control_context_root), ["not", "a record"])

        claimed = server_context.claim_control_context(baseline="live")

        assert claimed is not None
        assert (claimed.target, claimed.generation) == ("live", 0)

    def test_server_start_claim_names_the_pid_it_was_given(
        self, control_context_root, write_control_context
    ):
        write_control_context(control_context_root, target="va", generation=7, owned_by=None)

        claimed = server_context.claim_control_context(baseline="live", server_pid=4321)

        assert claimed is not None
        assert claimed.owner is not None
        assert claimed.owner.pid == 4321

    def test_server_start_claim_survives_an_unresolvable_agent_data_root(self, monkeypatch):
        from osprey_connectors import session_store

        monkeypatch.setattr(session_store, "agent_data_root", lambda: None)
        control_context.invalidate_cache()

        assert server_context.claim_control_context(baseline="live") is None

    def test_server_start_claim_survives_an_unwritable_record(
        self, control_context_root, write_control_context, monkeypatch
    ):
        write_control_context(control_context_root, target="va", generation=7, owned_by=None)

        def boom(*args, **kwargs):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(control_context, "write_record", boom)

        assert server_context.claim_control_context(baseline="live") is None


class TestServerStartLaunchTarget:
    """The first child's target comes from the record, the baseline only otherwise."""

    def test_server_start_takes_the_launch_target_from_the_record(
        self, control_context_root, write_control_context
    ):
        write_control_context(control_context_root, target="va", generation=7)

        assert server_context.launch_target_from_record() == "va"

    def test_server_start_launch_target_is_none_without_a_record(self, control_context_root):
        assert server_context.launch_target_from_record() is None

    def test_server_start_launch_target_is_none_when_the_record_is_unreadable(
        self, control_context_root
    ):
        from tests._control_context_fixtures import write_payload

        write_payload(
            control_context.record_path_under(control_context_root),
            {"schema": 1, "target": "", "generation": 7},
        )

        assert server_context.launch_target_from_record() is None

    async def test_server_start_launches_the_first_child_on_the_records_target(
        self, control_context_root, write_control_context
    ):
        write_control_context(control_context_root, target="va", generation=7)
        manager = _FakeManager()

        await _switching_context(manager).control_system()

        assert manager.ensure_started_calls == ["va"]

    async def test_server_start_launches_on_the_baseline_when_there_is_no_record(
        self, control_context_root
    ):
        manager = _FakeManager()

        await _switching_context(manager).control_system()

        assert manager.ensure_started_calls == [None]

    async def test_server_start_stops_consulting_the_record_once_a_child_runs(
        self, control_context_root, write_control_context
    ):
        write_control_context(control_context_root, target="va", generation=7)
        manager = _FakeManager(started=True)

        await _switching_context(manager).control_system()

        assert manager.ensure_started_calls == [None]


class TestServerStartStep:
    """What ``create_server`` runs: the report, the reaping, and the claim."""

    def test_server_start_step_claims_the_record_and_reaps_orphans(
        self, control_context_root, write_control_context, monkeypatch, caplog
    ):
        write_control_context(control_context_root, target="va", generation=7, owned_by=None)
        manager = _FakeManager()
        manager.reset_state = lambda: [4321]
        context = _switching_context(manager)
        monkeypatch.setattr(server_context, "get_server_context", lambda: context)

        with caplog.at_level(logging.WARNING):
            server._start_from_record()

        record = control_context.read_record()
        assert record is not None
        assert record.owner is not None
        assert record.owner.kind == control_context.OWNER_CONTROLS_SERVER
        assert (record.target, record.generation) == ("va", 7)
        assert "orphaned connector-host" in caplog.text

    def test_server_start_step_claims_even_when_the_report_cannot_be_written(
        self, control_context_root, write_control_context, monkeypatch
    ):
        write_control_context(control_context_root, target="va", generation=7, owned_by=None)

        def boom():
            raise RuntimeError("no data root")

        manager = _FakeManager()
        manager.reset_state = boom
        context = _switching_context(manager)
        monkeypatch.setattr(server_context, "get_server_context", lambda: context)

        server._start_from_record()

        record = control_context.read_record()
        assert record is not None
        assert record.owner is not None
        assert record.owner.pid == os.getpid()

    def test_server_start_step_never_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("no server context")

        monkeypatch.setattr(server_context, "get_server_context", boom)

        server._start_from_record()
