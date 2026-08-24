"""Tests for the stdlib-only target-state reader ``osprey_target_state``.

This module is imported by hooks and the status line rather than invoked as a
script, so it is tested by direct import — the ``osprey_hook_log`` precedent.

The whole contract under test is "never surprise the caller": every failure mode
must arrive as the same baseline-fallback marker, and nothing must ever raise.
The tests therefore lean on two seams instead of on real process trees:
``ancestor_pids`` (monkeypatched to a synthesized chain) and ``resolve_state_dir``
(pointed at ``tmp_path``). The ancestor walker itself is unit-tested separately
against monkeypatched ``/proc`` and ``ps`` shapes.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

#: A PID that is on the synthesized ancestor chain but is not this process.
OWNER_PPID = 424242

#: A PID that is on no chain at all.
STRANGER_PPID = 999001


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the reader at an empty temp state directory."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(directory))
    return directory


@pytest.fixture
def synthetic_chain(monkeypatch):
    """Replace the ancestor walk with a fixed chain containing OWNER_PPID."""
    chain = [os.getpid(), OWNER_PPID, 300]
    monkeypatch.setattr(reader, "ancestor_pids", lambda *a, **k: list(chain))
    return chain


@pytest.fixture
def alive_everything(monkeypatch):
    """Treat every PID as alive unless a test says otherwise."""
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)


def write_state(directory, server_pid, **overrides):
    """Write a well-formed state file for *server_pid*, with overrides applied."""
    record = {
        "target": "va",
        "generation": 3,
        "server_pid": server_pid,
        "owner_ppid": OWNER_PPID,
        "targets": {
            "live": {"label": "ALS storage ring", "endpoint": "epics://", "real_machine": True},
            "va": {
                "label": "Virtual accelerator",
                "endpoint": "pva://vasrv",
                "real_machine": False,
            },
        },
        "children": [],
    }
    record.update(overrides)
    path = directory / f"target_state_{server_pid}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_target_generation_and_display(
    state_dir, synthetic_chain, alive_everything
):
    write_state(state_dir, 5150)

    result = reader.read_session_target()

    assert not reader.is_baseline(result)
    assert result["fallback"] is None
    assert result["reason"] is None
    assert result["target"] == "va"
    assert result["generation"] == 3
    assert result["display"] == {
        "label": "Virtual accelerator",
        "endpoint": "pva://vasrv",
        "real_machine": False,
    }


def test_display_comes_from_the_selected_target_not_the_other_one(
    state_dir, synthetic_chain, alive_everything
):
    """Switching ``target`` must switch which ``targets`` entry is rendered."""
    write_state(state_dir, 5151, target="live")

    result = reader.read_session_target()

    assert result["target"] == "live"
    assert result["display"]["label"] == "ALS storage ring"
    assert result["display"]["real_machine"] is True


def test_result_always_carries_all_five_contract_keys(state_dir, synthetic_chain, alive_everything):
    write_state(state_dir, 5152)
    resolved = reader.read_session_target()

    expected = {"target", "generation", "display", "fallback", "reason"}
    assert set(resolved) == expected
    assert set(reader.baseline_result()) == expected


# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------


def test_dead_server_pid_is_ignored(state_dir, synthetic_chain, monkeypatch):
    """A file whose owning server is gone must not answer for this session."""
    write_state(state_dir, 5153)
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: False)

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_NO_STATE


def test_dead_server_file_is_never_deleted(state_dir, synthetic_chain, monkeypatch):
    """The reader is read-only: sweeping stale files is the writer's job."""
    path = write_state(state_dir, 5154)
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: False)

    reader.read_session_target()

    assert path.exists()


def test_live_file_wins_over_a_dead_sibling(state_dir, synthetic_chain, monkeypatch):
    write_state(state_dir, 5155, target="live")
    write_state(state_dir, 5156, target="va")
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: int(pid) == 5156)

    result = reader.read_session_target()

    assert result["target"] == "va"


def test_is_process_alive_treats_permission_error_as_alive(monkeypatch):
    """Another user's server is unreachable, not dead — discarding it loses state."""

    def _boom(pid, sig):
        raise PermissionError

    monkeypatch.setattr(reader.os, "kill", _boom)
    assert reader._is_process_alive(4321) is True


def test_is_process_alive_rejects_non_positive_and_garbage_pids():
    assert reader._is_process_alive(0) is False
    assert reader._is_process_alive(-1) is False
    assert reader._is_process_alive(None) is False
    assert reader._is_process_alive("not-a-pid") is False


# ---------------------------------------------------------------------------
# fail-closed fallbacks
# ---------------------------------------------------------------------------


def test_zero_files_yields_baseline_marker(state_dir, synthetic_chain, alive_everything):
    result = reader.read_session_target()

    assert result == {
        "target": None,
        "generation": None,
        "display": None,
        "fallback": reader.FALLBACK_BASELINE,
        "reason": reader.REASON_NO_STATE,
    }


def test_missing_state_directory_yields_baseline_marker(tmp_path, monkeypatch, synthetic_chain):
    absent = tmp_path / "never-created"
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(absent))

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_NO_STATE


def test_unresolvable_repo_root_yields_baseline_marker(monkeypatch, synthetic_chain):
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: None)

    assert reader.is_baseline(reader.read_session_target())


def test_two_live_matching_files_are_ambiguous(state_dir, synthetic_chain, alive_everything):
    """Two sessions sharing a checkout must not be guessed between (CC-3)."""
    write_state(state_dir, 5157, target="va")
    write_state(state_dir, 5158, target="live")

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_AMBIGUOUS
    assert result["target"] is None


def test_other_sessions_file_is_not_mine(state_dir, synthetic_chain, alive_everything):
    """A live file owned by a PID off our chain belongs to another session."""
    write_state(state_dir, 5159, owner_ppid=STRANGER_PPID)

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_NO_STATE


def test_only_the_matching_file_is_selected_among_several(
    state_dir, synthetic_chain, alive_everything
):
    write_state(state_dir, 5160, owner_ppid=STRANGER_PPID, target="live")
    write_state(state_dir, 5161, owner_ppid=OWNER_PPID, target="va", generation=7)

    result = reader.read_session_target()

    assert result["target"] == "va"
    assert result["generation"] == 7


def test_corrupt_json_yields_baseline_marker(state_dir, synthetic_chain, alive_everything):
    (state_dir / "target_state_5162.json").write_text("{not json", encoding="utf-8")

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_UNREADABLE


def test_non_dict_payload_is_corruption(state_dir, synthetic_chain, alive_everything):
    (state_dir / "target_state_5163.json").write_text("[1, 2, 3]", encoding="utf-8")

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_UNREADABLE


def test_corrupt_file_does_not_hide_a_good_sibling(state_dir, synthetic_chain, alive_everything):
    (state_dir / "target_state_5164.json").write_text("", encoding="utf-8")
    write_state(state_dir, 5165, target="live")

    result = reader.read_session_target()

    assert result["target"] == "live"


def test_missing_target_field_is_unreadable(state_dir, synthetic_chain, alive_everything):
    """There is no safe guess between live and va, so an absent target fails closed."""
    write_state(state_dir, 5166, target="")

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_UNREADABLE


def test_unparseable_owner_ppid_is_skipped(state_dir, synthetic_chain, alive_everything):
    write_state(state_dir, 5167, owner_ppid="not-a-pid")

    assert reader.is_baseline(reader.read_session_target())


def test_unrelated_files_in_the_directory_are_ignored(state_dir, synthetic_chain, alive_everything):
    (state_dir / "README.md").write_text("not state", encoding="utf-8")
    (state_dir / "target_state_notanumber.json").write_text("{}", encoding="utf-8")
    write_state(state_dir, 5168)

    result = reader.read_session_target()

    assert result["target"] == "va"


# ---------------------------------------------------------------------------
# schema tolerance
# ---------------------------------------------------------------------------


def test_missing_targets_metadata_degrades_without_raising(
    state_dir, synthetic_chain, alive_everything
):
    write_state(state_dir, 5169, targets={})

    result = reader.read_session_target()

    assert not reader.is_baseline(result)
    assert result["target"] == "va"
    # Label degrades to the target name rather than to a blank identity.
    assert result["display"] == {"label": "va", "endpoint": "", "real_machine": False}


def test_targets_of_the_wrong_type_degrades_without_raising(
    state_dir, synthetic_chain, alive_everything
):
    write_state(state_dir, 5170, targets="not-a-mapping")

    result = reader.read_session_target()

    assert result["display"]["label"] == "va"


def test_partial_target_metadata_fills_the_missing_keys(
    state_dir, synthetic_chain, alive_everything
):
    write_state(state_dir, 5171, targets={"va": {"label": "VA only"}})

    result = reader.read_session_target()

    assert result["display"] == {"label": "VA only", "endpoint": "", "real_machine": False}


def test_missing_generation_degrades_to_zero(state_dir, synthetic_chain, alive_everything):
    """Generation is softer than target: a bad one must not discard a good target."""
    write_state(state_dir, 5172, generation="seven")

    result = reader.read_session_target()

    assert result["target"] == "va"
    assert result["generation"] == 0


def test_unknown_extra_fields_are_tolerated(state_dir, synthetic_chain, alive_everything):
    write_state(state_dir, 5173, future_field={"written": "by a newer server"})

    assert reader.read_session_target()["target"] == "va"


# ---------------------------------------------------------------------------
# never raises
# ---------------------------------------------------------------------------


def test_an_exploding_state_dir_still_returns_the_marker(monkeypatch):
    def _boom(hook_input=None):
        raise RuntimeError("filesystem on fire")

    monkeypatch.setattr(reader, "resolve_state_dir", _boom)

    result = reader.read_session_target()

    assert reader.is_baseline(result)
    assert result["reason"] == reader.REASON_UNREADABLE


def test_is_baseline_tolerates_non_dict_input():
    assert reader.is_baseline(None) is True
    assert reader.is_baseline("nonsense") is True


# ---------------------------------------------------------------------------
# ancestor pid chain
# ---------------------------------------------------------------------------


def test_ancestor_pids_includes_self_and_walks_parents(monkeypatch):
    parents = {10: 9, 9: 8, 8: 1}
    monkeypatch.setattr(reader, "parent_pid", lambda pid: parents.get(pid))

    assert reader.ancestor_pids(10) == [10, 9, 8]


def test_ancestor_pids_stops_when_a_parent_is_unknown(monkeypatch):
    monkeypatch.setattr(reader, "parent_pid", lambda pid: None)

    assert reader.ancestor_pids(77) == [77]


def test_ancestor_pids_breaks_a_cycle(monkeypatch):
    parents = {5: 6, 6: 5}
    monkeypatch.setattr(reader, "parent_pid", lambda pid: parents.get(pid))

    assert reader.ancestor_pids(5) == [5, 6]


def test_ancestor_pids_is_bounded(monkeypatch):
    """A pathological tree must not spin: the hop bound is a hard stop."""
    monkeypatch.setattr(reader, "parent_pid", lambda pid: pid + 1)

    assert len(reader.ancestor_pids(1000, max_hops=8)) == 8


def test_ancestor_pids_rejects_garbage_start():
    assert reader.ancestor_pids("not-a-pid") == []
    assert reader.ancestor_pids(1) == []


def test_ancestor_pids_real_tree_contains_this_process_and_its_parent():
    """Sanity check against the actual process tree, on whichever platform."""
    chain = reader.ancestor_pids()

    assert chain[0] == os.getpid()
    assert os.getppid() in chain


# -- the ps fallback shape --------------------------------------------------


class _Completed:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_ppid_from_ps_parses_padded_output(monkeypatch):
    """``ps -o ppid=`` right-pads its column; the parse must survive that."""
    seen = {}

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Completed(stdout="  4200\n")

    monkeypatch.setattr(reader.subprocess, "run", _run)

    assert reader._ppid_from_ps(4321) == 4200
    assert seen["cmd"] == ["ps", "-o", "ppid=", "-p", "4321"]


def test_ppid_from_ps_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(reader.subprocess, "run", lambda cmd, **kw: _Completed("", 1))
    assert reader._ppid_from_ps(4321) is None


def test_ppid_from_ps_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(reader.subprocess, "run", lambda cmd, **kw: _Completed("PPID\n"))
    assert reader._ppid_from_ps(4321) is None


def test_ppid_from_ps_returns_none_when_ps_is_missing(monkeypatch):
    def _missing(cmd, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(reader.subprocess, "run", _missing)
    assert reader._ppid_from_ps(4321) is None


def test_ppid_from_ps_returns_none_on_timeout(monkeypatch):
    def _slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, reader.PS_TIMEOUT_S)

    monkeypatch.setattr(reader.subprocess, "run", _slow)
    assert reader._ppid_from_ps(4321) is None


def test_ppid_from_proc_parses_a_comm_containing_spaces_and_parens(tmp_path, monkeypatch):
    """The ``comm`` field is untrusted text; fields are taken after the LAST ')'."""
    proc = tmp_path / "proc" / "4321"
    proc.mkdir(parents=True)
    (proc / "stat").write_text("4321 (weird ) name) S 4200 4321 4321 0 -1 4194304\n")

    real_open = open

    def _fake_open(path, *args, **kwargs):
        if str(path) == "/proc/4321/stat":
            return real_open(proc / "stat", *args, **kwargs)
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _fake_open)
    result = reader._ppid_from_proc(4321)
    # Restore before asserting: pytest's own failure reporting opens files.
    monkeypatch.undo()

    assert result == 4200


def test_ppid_from_proc_returns_none_without_proc(monkeypatch):
    """macOS has no ``/proc``; the read must fail quietly, not raise."""

    def _missing(path, *args, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", _missing)
    result = reader._ppid_from_proc(4321)
    monkeypatch.undo()

    assert result is None


def test_parent_pid_falls_back_to_ps_when_proc_is_absent(monkeypatch):
    monkeypatch.setattr(reader, "_ppid_from_proc", lambda pid: None)
    monkeypatch.setattr(reader, "_ppid_from_ps", lambda pid: 4200)

    assert reader.parent_pid(4321) == 4200


def test_parent_pid_prefers_proc_and_skips_the_subprocess(monkeypatch):
    def _never(pid):
        raise AssertionError("ps must not be spawned when /proc answered")

    monkeypatch.setattr(reader, "_ppid_from_proc", lambda pid: 4200)
    monkeypatch.setattr(reader, "_ppid_from_ps", _never)

    assert reader.parent_pid(4321) == 4200


# ---------------------------------------------------------------------------
# path contract
# ---------------------------------------------------------------------------


def test_resolve_state_dir_anchors_on_repo_root_and_agent_data(monkeypatch, tmp_path):
    monkeypatch.setattr(reader, "get_repo_root", lambda hook_input=None: str(tmp_path))

    resolved = reader.resolve_state_dir()

    assert resolved == os.path.join(str(tmp_path), reader._AGENT_DATA_BASE_DIR, "control_target")
    assert reader.STATE_DIR_NAME == "control_target"
    assert reader.STATE_FILE_GLOB == "target_state_*.json"


def test_resolve_state_dir_returns_none_when_repo_root_is_empty(monkeypatch):
    monkeypatch.setattr(reader, "get_repo_root", lambda hook_input=None: "")

    assert reader.resolve_state_dir() is None


def test_module_imports_no_third_party_dependencies():
    """Hooks run outside the venv: every guaranteed path must be stdlib only."""
    import ast
    from pathlib import Path

    source = Path(reader.__file__).read_text()
    tree = ast.parse(source)

    # Only unguarded, module-level imports count: the one osprey import is
    # deliberately inside a try/except with a literal fallback.
    top_level_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top_level_roots.add(node.module.split(".")[0])

    # `osprey_hook_log` is the sibling helper; everything else must be stdlib.
    assert top_level_roots <= {
        "__future__",
        "json",
        "os",
        "subprocess",
        "sys",
        "osprey_hook_log",
    }, top_level_roots
