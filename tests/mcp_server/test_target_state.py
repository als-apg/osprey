"""Tests for the single-writer control-system target state file.

Covers:
  - the path contract a stdlib-only hook has to be able to restate
  - write_on_start: baseline reset, PID capture, display metadata
  - publish_switch / record_child_pids merges
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
