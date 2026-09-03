"""Tests for the single-writer control-system target state file.

Covers:
  - the path contract a stdlib-only hook has to be able to restate
  - write_on_start: baseline reset, PID capture, display metadata
  - publish_switch / publish_targets / record_child_pids merges
  - the optional display keys (probe_channel, selected_role): preserved when
    real, absent when the caller has none
  - fail-closed reads (absent, corrupt, non-object)
  - stale-PID sweep: deletion, orphan child PIDs, own file preserved
  - delete_on_shutdown idempotence
  - atomic writes leaving no temp litter when the dump fails
"""

import json
import os

import pytest

from osprey.mcp_server.control_system import target_state

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
    """Drop a state file that looks like another server's, bypassing the API."""
    directory = state_root / target_state.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target_state.STATE_FILE_PREFIX}{pid}{target_state.STATE_FILE_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 3,
                "server_pid": pid,
                "owner_ppid": pid - 1,
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

    def test_state_file_is_named_for_the_server_pid(self, state_root):
        assert target_state.state_file_path(4321) == (
            state_root / "control_target" / "target_state_4321.json"
        )

    def test_state_file_defaults_to_this_process(self, state_root):
        assert target_state.state_file_path().name == f"target_state_{os.getpid()}.json"

    def test_glob_matches_the_file_the_writer_produces(self, state_root):
        path = _write_foreign(state_root, 4321)
        matched = list(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))
        assert matched == [path]


# ---------------------------------------------------------------------------
# write_on_start
# ---------------------------------------------------------------------------


class TestWriteOnStart:
    def test_writes_baseline_record(self, state_root):
        target_state.write_on_start("va", TARGETS_META, server_pid=1234, owner_ppid=99)

        record = json.loads(target_state.state_file_path(1234).read_text(encoding="utf-8"))
        assert record == {
            "target": "va",
            "generation": 0,
            "server_pid": 1234,
            "owner_ppid": 99,
            "targets": TARGETS_META,
            "children": [],
            "last_switch": None,
            "reachability": None,
            "last_posture_realign": None,
        }

    def test_captures_own_pid_and_parent_pid_by_default(self, state_root):
        target_state.write_on_start("live", TARGETS_META)

        record = target_state.read()
        assert record["server_pid"] == os.getpid()
        assert record["owner_ppid"] == os.getppid()

    def test_resets_a_previous_selection_to_the_baseline(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)
        target_state.publish_switch("va", 7, server_pid=1234)

        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)

        record = target_state.read(1234)
        assert record["target"] == "live"
        assert record["generation"] == 0

    def test_every_target_slot_always_present(self, state_root):
        target_state.write_on_start("live", {"live": {"label": "Live"}}, server_pid=1234)

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
        target_state.write_on_start("live", meta, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert targets["standin"] == targets["va"]
        assert targets["standin"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_creates_the_state_directory(self, state_root):
        assert not (state_root / "control_target").exists()
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)
        assert (state_root / "control_target").is_dir()


# ---------------------------------------------------------------------------
# probe_channel round-tripping
# ---------------------------------------------------------------------------


class TestProbeChannel:
    """The approval describer names the probe channel from this file alone."""

    def test_present_probe_channel_is_preserved(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

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
        target_state.write_on_start("live", meta, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert "probe_channel" not in targets["live"]
        assert "probe_channel" not in targets["va"]
        assert "probe_channel" not in targets["standin"]

    @pytest.mark.parametrize("bogus", ["", None, 5064, ["SR:BeamCurrent"]])
    def test_unusable_probe_channel_is_dropped_never_stringified(self, state_root, bogus):
        meta = {"live": {"label": "Live", "probe_channel": bogus}}
        target_state.write_on_start("live", meta, server_pid=1234)

        assert "probe_channel" not in target_state.read(1234)["targets"]["live"]

    def test_probe_channel_survives_a_switch(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)
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
        target_state.write_on_start("live", self.ROLED_META, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert targets["live"]["selected_role"] == "read_only"
        assert targets["va"]["selected_role"] == "writes"
        assert targets["standin"]["selected_role"] == "read_only"

    def test_absent_selected_role_stays_absent(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert "selected_role" not in targets["live"]
        assert "selected_role" not in targets["va"]
        assert "selected_role" not in targets["standin"]

    @pytest.mark.parametrize("bogus", ["", None, 5064, ["read_only"]])
    def test_unusable_selected_role_is_dropped_never_stringified(self, state_root, bogus):
        meta = {"live": {"label": "Live", "selected_role": bogus}}
        target_state.write_on_start("live", meta, server_pid=1234)

        assert "selected_role" not in target_state.read(1234)["targets"]["live"]

    def test_selected_role_survives_a_switch(self, state_root):
        target_state.write_on_start("live", self.ROLED_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        assert target_state.read(1234)["targets"] == self.ROLED_META

    def test_selected_role_survives_a_republish(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        assert target_state.publish_targets(self.ROLED_META, server_pid=1234) is True

        assert target_state.read(1234)["targets"] == self.ROLED_META

    def test_an_empty_role_is_dropped_on_a_republish_too(self, state_root):
        target_state.write_on_start("live", self.ROLED_META, server_pid=1234)

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
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        assert target_state.publish_targets(self.NARROWED, server_pid=1234) is True

        targets = target_state.read(1234)["targets"]
        assert targets["live"]["endpoint"] == "gateway.example.com:5065"
        assert targets["live"]["selected_role"] == "read_only"

    def test_an_omitted_slot_is_written_empty_not_dropped(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        target_state.publish_targets(self.NARROWED, server_pid=1234)

        targets = target_state.read(1234)["targets"]
        assert set(targets) == set(target_state.TARGET_NAMES)
        assert targets["va"] == {"label": "", "endpoint": "", "real_machine": False}
        assert targets["standin"] == {"label": "", "endpoint": "", "real_machine": False}

    def test_identity_and_pids_are_untouched(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)
        target_state.publish_switch("va", 2, children=[5001], server_pid=1234)

        target_state.publish_targets(self.NARROWED, server_pid=1234)

        record = target_state.read(1234)
        assert record["target"] == "va"
        assert record["generation"] == 2
        assert record["children"] == [5001]
        assert record["server_pid"] == 1234
        assert record["owner_ppid"] == 99

    def test_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_targets(TARGETS_META, server_pid=1234) is False
        assert target_state.read(1234) is None


# ---------------------------------------------------------------------------
# publish / children
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_updates_target_and_generation(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)

        assert target_state.publish_switch("va", 1, server_pid=1234) is True

        record = target_state.read(1234)
        assert record["target"] == "va"
        assert record["generation"] == 1

    def test_publish_preserves_display_metadata_and_pids(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)
        target_state.publish_switch("va", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["targets"] == TARGETS_META
        assert record["server_pid"] == 1234
        assert record["owner_ppid"] == 99

    def test_standin_round_trips_as_a_baseline(self, state_root):
        """``standin`` is a target like any other: it survives write_on_start."""
        target_state.write_on_start(
            target_state.TARGET_STANDIN, TARGETS_META, server_pid=1234, owner_ppid=99
        )

        record = target_state.read(1234)
        assert record["target"] == "standin"
        assert record["generation"] == 0
        assert record["targets"] == TARGETS_META

    def test_standin_round_trips_through_a_switch(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)

        assert target_state.publish_switch(target_state.TARGET_STANDIN, 2, server_pid=1234) is True

        record = target_state.read(1234)
        assert record["target"] == "standin"
        assert record["generation"] == 2
        assert record["targets"]["standin"]["endpoint"] == "localhost:5084"
        assert record["targets"] == TARGETS_META

    def test_switching_away_from_standin_back_to_live(self, state_root):
        target_state.write_on_start(
            target_state.TARGET_STANDIN, TARGETS_META, server_pid=1234, owner_ppid=99
        )
        target_state.publish_switch("live", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["target"] == "live"
        assert record["targets"] == TARGETS_META

    def test_publish_can_carry_child_pids(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, children=[5001, 5002], server_pid=1234)

        assert target_state.read(1234)["children"] == [5001, 5002]

    def test_publish_without_a_record_writes_nothing(self, state_root):
        assert target_state.publish_switch("va", 1, server_pid=1234) is False
        assert target_state.read(1234) is None

    def test_record_child_pids_sets_and_clears(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

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
        (directory / "target_state_1234.json").mkdir()

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
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)
        # Even claiming our own PID is dead must not delete our file.
        monkeypatch.setattr(os, "kill", self._kill_with_dead({1234}))

        assert target_state.sweep_stale(server_pid=1234) == []
        assert target_state.read(1234) is not None

    def test_write_on_start_returns_the_orphans_it_swept(self, state_root, monkeypatch):
        dead = _write_foreign(state_root, 4321, children=[5001])
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321}))

        orphans = target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        assert orphans == [5001]
        assert not dead.exists()
        assert target_state.read(1234)["target"] == "live"

    def test_corrupt_dead_file_is_removed_without_orphans(self, state_root, monkeypatch):
        dead = _write_foreign(state_root, 4321, children=[5001])
        dead.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(os, "kill", self._kill_with_dead({4321}))

        assert target_state.sweep_stale(server_pid=1234) == []
        assert not dead.exists()

    def test_file_with_unparseable_pid_is_swept(self, state_root):
        directory = state_root / "control_target"
        directory.mkdir(parents=True)
        junk = directory / "target_state_notapid.json"
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
    path = target_state.state_file_path(pid if isinstance(pid, int) else 99999)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _entries():
    return sorted(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))


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
        target_state.state_file_path(4321).write_text("{not json", encoding="utf-8")

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
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        target_state.delete_on_shutdown(server_pid=1234)

        assert not target_state.state_file_path(1234).exists()

    def test_delete_is_idempotent(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        target_state.delete_on_shutdown(server_pid=1234)
        target_state.delete_on_shutdown(server_pid=1234)  # missing file is fine

        assert target_state.read(1234) is None


class TestAtomicWrite:
    def test_successful_write_leaves_only_the_state_file(self, state_root):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234)
        target_state.publish_switch("va", 1, server_pid=1234)

        directory = state_root / "control_target"
        assert [p.name for p in directory.iterdir()] == ["target_state_1234.json"]
        assert json.loads((directory / "target_state_1234.json").read_text(encoding="utf-8"))

    def test_failed_dump_leaves_no_temp_file_and_no_state_file(self, state_root, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(target_state.json, "dump", boom)

        with pytest.raises(RuntimeError):
            target_state.write_on_start("live", TARGETS_META, server_pid=1234)

        assert list((state_root / "control_target").iterdir()) == []

    def test_failed_dump_does_not_corrupt_the_previous_record(self, state_root, monkeypatch):
        target_state.write_on_start("live", TARGETS_META, server_pid=1234, owner_ppid=99)

        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        with monkeypatch.context() as patched:
            patched.setattr(target_state.json, "dump", boom)
            with pytest.raises(RuntimeError):
                target_state.publish_switch("va", 1, server_pid=1234)

        record = target_state.read(1234)
        assert record["target"] == "live"
        assert [p.name for p in (state_root / "control_target").iterdir()] == [
            "target_state_1234.json"
        ]
