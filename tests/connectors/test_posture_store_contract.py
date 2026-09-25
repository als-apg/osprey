"""Contract tests for the write-posture reader.

The posture lives in the control-context record now — one ``posture`` field on
one file per deployment, holding per-target narrowings in the grammar the
retired ``session-postures.json`` used per session. These tests pin the parts
every reader has to agree on: where the file is, what its shapes mean, and how
a lookup combines with the deployment ceiling, a read-only run and the launch
pin.

Not to be confused with ``tests/interfaces/test_session_store.py``, which
covers the web terminal's own session store.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import acting_identity
from osprey_connectors.types import CONTROL_TARGETS
from tests._control_context_fixtures import write_control_context

# --- config sections -------------------------------------------------------
# A switch-capable deployment: live is EPICS, va and standin both configured.
# Write posture is spelled per connector block so a test can arm one machine
# and leave the others alone, which is the whole point of the ceiling.


def _section(*, live_writes: bool = False, va_writes: bool = False, deployment: bool = False):
    return {
        "type": "epics",
        "writes_enabled": deployment,
        "connector": {
            "epics": {"prefix": "X:", "writes_enabled": live_writes},
            "virtual_accelerator": {"host": "localhost", "writes_enabled": va_writes},
            "live_standin": {"prefix": "S:"},
        },
    }


ARMED = _section(live_writes=True, va_writes=True)
UNARMED = _section()


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Stamp ``OSPREY_AGENT_DATA_ROOT`` at a scratch root, caches cleared."""
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    posture_store.invalidate_cache()
    yield tmp_path
    posture_store.invalidate_cache()


def _write_record(root: Path, posture, *, target: str = "live", generation: int = 3) -> Path:
    """A complete record whose ``posture`` field is *posture*, plus a cache drop.

    The record is written by :func:`write_control_context`, so this suite pins
    what the reader does with a payload rather than restating what one is.
    """
    path = write_control_context(
        root, target=target, generation=generation, posture=posture, owned_by=None
    )
    posture_store.invalidate_cache()
    return path


# --- path resolution -------------------------------------------------------


def test_root_prefers_the_env_stamp(data_root):
    assert posture_store.agent_data_root() == data_root
    assert posture_store.state_dir() == data_root / "control_target" / acting_identity()


def test_root_falls_back_to_the_shared_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(posture_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    posture_store.invalidate_cache()
    assert posture_store.state_dir() == tmp_path / "var" / "control_target" / acting_identity()


def test_blank_env_stamp_is_not_a_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, "   ")
    monkeypatch.setattr(posture_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    posture_store.invalidate_cache()
    assert posture_store.state_dir() == tmp_path / "var" / "control_target" / acting_identity()


def test_unresolvable_root_answers_none(monkeypatch):
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)

    def _boom():
        raise OSError("no project root")

    monkeypatch.setattr(posture_store, "resolve_shared_data_root", _boom)
    posture_store.invalidate_cache()
    assert posture_store.agent_data_root() is None
    assert posture_store.state_dir() is None
    # A caller that cannot resolve the root sees no narrowings, not a crash.
    assert posture_store.recorded_posture() == {}


@pytest.mark.usefixtures("data_root")
def test_the_posture_sits_beside_the_target_state_file():
    """FR9: co-sited with the state file — one directory, not two."""
    from osprey.mcp_server.control_system import target_state

    assert posture_store.STATE_DIR_NAME == target_state.STATE_DIR_NAME
    assert control_context.record_path().parent == posture_store.state_dir()


def test_the_retired_store_file_is_gone(data_root):
    """``session-postures.json`` is not written, not read, and not spelled here."""
    assert not hasattr(posture_store, "STORE_FILENAME")
    for retired in ("store_path", "load_store", "session_map"):
        assert not hasattr(posture_store, retired), retired

    stale = data_root / posture_store.STATE_DIR_NAME / "session-postures.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"s1": {"live": "sandbox"}}), encoding="utf-8")
    posture_store.invalidate_cache()

    assert posture_store.recorded_posture() == {}
    assert posture_store.store_permits("live") is True


# --- parse_posture_value: the entry grammar ----------------------------------------
#
# The record carries ONE posture value of this grammar and applies this filter
# to it (``control_context.parse_posture``). What survives it decides whether a
# real machine is written to.


def test_bare_sandbox_expands_to_every_target():
    assert posture_store.parse_posture_value("sandbox") == dict.fromkeys(CONTROL_TARGETS, "sandbox")


def test_bare_writes_is_dropped():
    assert posture_store.parse_posture_value("writes") == {}


def test_unknown_values_are_dropped():
    assert posture_store.parse_posture_value("readonly") == {}
    assert posture_store.parse_posture_value(7) == {}
    assert posture_store.parse_posture_value(None) == {}


def test_per_target_values_are_validated_the_same_way():
    parsed = posture_store.parse_posture_value(
        {"live": "sandbox", "va": "writes", "standin": "nonsense", 7: "sandbox"}
    )
    assert parsed == {"live": "sandbox"}


def test_a_map_that_narrows_nothing_is_dropped():
    assert posture_store.parse_posture_value({"live": "writes"}) == {}
    assert posture_store.parse_posture_value({}) == {}


def test_a_shape_the_grammar_does_not_know_is_tolerated():
    assert posture_store.parse_posture_value([1, 2, 3]) == {}
    assert posture_store.parse_posture_value({7: "sandbox"}) == {}


def test_the_record_applies_this_very_filter(data_root):
    """One grammar, one implementation: the record's field is filtered here."""
    _write_record(data_root, {"live": "sandbox", "va": "writes", "nonsense": "sandbox"})
    assert posture_store.recorded_posture() == {"live": "sandbox", "nonsense": "sandbox"}
    record = control_context.read_record()
    assert record is not None
    assert record.posture == posture_store.recorded_posture()


# --- lookups ---------------------------------------------------------------


@pytest.mark.usefixtures("data_root")
def test_no_record_is_no_narrowing():
    assert posture_store.recorded_posture() == {}
    assert posture_store.target_posture("live") is None
    assert posture_store.store_permits("live") is True


def test_recorded_posture_and_target_posture(data_root):
    _write_record(data_root, {"live": "sandbox"})
    assert posture_store.recorded_posture() == {"live": "sandbox"}
    assert posture_store.target_posture("live") == "sandbox"
    assert posture_store.target_posture("va") is None
    assert posture_store.target_posture(None) is None
    assert posture_store.target_posture("") is None


def test_a_bare_sandbox_in_the_record_narrows_every_target(data_root):
    _write_record(data_root, "sandbox")
    for target in CONTROL_TARGETS:
        assert posture_store.target_posture(target) == "sandbox"


def test_a_recorded_writes_narrows_nothing(data_root):
    _write_record(data_root, {"live": "writes"})
    assert posture_store.recorded_posture() == {}
    assert posture_store.store_permits("live") is True


def test_a_degraded_record_narrows_nothing(data_root):
    """Every unusable record is "no record", and no record narrows anything."""
    path = control_context.record_path_under(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    for payload in (
        "{not json at all",
        json.dumps({"schema": 2, "target": "live", "generation": 0, "posture": "sandbox"}),
        json.dumps({"schema": 1, "target": "elsewhere", "generation": 0, "posture": "sandbox"}),
        json.dumps({"schema": 1, "target": "live", "generation": -1, "posture": "sandbox"}),
    ):
        path.write_text(payload, encoding="utf-8")
        posture_store.invalidate_cache()
        assert posture_store.recorded_posture() == {}, payload
        assert posture_store.store_permits("live") is True, payload


def test_an_undecodable_record_is_no_narrowing_not_an_exception(data_root):
    """A record that is not UTF-8 degrades like a corrupt one, rather than raising.

    Every reader here sits on a write path — the connector's reference monitor
    asks on each write — so an exception from one mis-encoded file would be
    raised into the write itself instead of leaving the ceiling in charge.
    """
    path = control_context.record_path_under(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe{\x00k\x00: sandbox}")
    posture_store.invalidate_cache()

    assert posture_store.recorded_posture() == {}
    assert posture_store.store_permits("live") is True


def test_a_raising_reader_leaves_the_ceiling_in_charge(data_root, monkeypatch):
    """``store_permits`` is called inside write paths and never raises."""
    _write_record(data_root, {"live": "sandbox"})

    def _boom(**_kwargs):
        raise RuntimeError("the state directory is on fire")

    monkeypatch.setattr(control_context, "read_record", _boom)
    assert posture_store.recorded_posture() == {}
    assert posture_store.store_permits("live") is True
    assert posture_store.effective_writes(ARMED, "live") is True


# --- cache -----------------------------------------------------------------


def test_two_narrowings_within_one_second_are_both_seen(data_root):
    """A coarse filesystem clock must not hide the second flip.

    Both writes go through the atomic temp+rename the owner uses, so the
    signature moves on size and inode even when ``st_mtime_ns`` does not.
    """
    path = control_context.record_path_under(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic(posture):
        payload = {
            "schema": 1,
            "owner": None,
            "target": "live",
            "generation": 3,
            "posture": posture,
            "last_switch": None,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    _atomic({"live": "sandbox"})
    assert posture_store.target_posture("live") == "sandbox"
    _atomic({"va": "sandbox"})
    assert posture_store.target_posture("live") is None
    assert posture_store.target_posture("va") == "sandbox"
    _atomic({"live": "sandbox", "va": "sandbox"})
    assert posture_store.target_posture("live") == "sandbox"


def test_the_record_appearing_and_disappearing_is_seen(data_root):
    assert posture_store.recorded_posture() == {}
    path = _write_record(data_root, "sandbox")
    assert posture_store.target_posture("live") == "sandbox"
    path.unlink()
    assert posture_store.recorded_posture() == {}


def test_a_moved_root_is_seen(data_root, tmp_path, monkeypatch):
    _write_record(data_root, "sandbox")
    assert posture_store.target_posture("live") == "sandbox"
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(other))
    assert posture_store.recorded_posture() == {}


def test_invalidate_cache_forgets_the_records_cache(data_root, monkeypatch):
    """One cache, held by the record reader; this module's hook drops that one."""
    _write_record(data_root, {"live": "sandbox"})
    assert posture_store.target_posture("live") == "sandbox"

    dropped: list[bool] = []
    monkeypatch.setattr(control_context, "invalidate_cache", lambda: dropped.append(True))
    posture_store.invalidate_cache()
    assert dropped == [True]


# --- effective_writes ------------------------------------------------------


@pytest.mark.parametrize("armed", [True, False])
@pytest.mark.parametrize("entry", [None, "sandbox", "writes"])
@pytest.mark.parametrize("readonly", [True, False])
def test_effective_writes_truth_table(data_root, monkeypatch, armed, entry, readonly):
    section = ARMED if armed else UNARMED
    if entry is not None:
        _write_record(data_root, {"live": entry})
    if readonly:
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    expected = armed and not readonly and entry != "sandbox"
    assert posture_store.effective_writes(section, "live") is expected


@pytest.mark.usefixtures("data_root")
def test_effective_writes_uses_the_targets_own_ceiling():
    section = _section(live_writes=False, va_writes=True)
    assert posture_store.effective_writes(section, "live") is False
    assert posture_store.effective_writes(section, "va") is True


def test_no_target_takes_the_most_restrictive_narrowing(data_root):
    _write_record(data_root, {"va": "sandbox"})
    # Any narrowing refuses when the caller holds no target of its own.
    assert posture_store.effective_writes(ARMED, None) is False
    assert posture_store.effective_writes(ARMED) is False


def test_no_target_with_nothing_narrowed_leaves_the_ceiling_in_charge(data_root):
    _write_record(data_root, {})
    assert posture_store.effective_writes(ARMED, None) is True
    assert posture_store.effective_writes(UNARMED, None) is False


@pytest.mark.usefixtures("data_root")
def test_connector_type_ceiling_beats_the_deployment_wide_key():
    """Mixed config: deployment armed, the EPICS block explicitly unarmed."""
    section = {
        "type": "epics",
        "writes_enabled": True,
        "connector": {"epics": {"prefix": "X:", "writes_enabled": False}},
    }
    assert posture_store.effective_writes(section, None, connector_type="epics") is False
    armed = {
        "type": "epics",
        "writes_enabled": False,
        "connector": {"epics": {"prefix": "X:", "writes_enabled": True}},
    }
    assert posture_store.effective_writes(armed, None, connector_type="epics") is True


def test_connector_type_ceiling_with_a_target_indexes_the_posture(data_root):
    _write_record(data_root, {"va": "sandbox"})
    section = _section(live_writes=True, va_writes=True)
    # The ceiling stays the connector TYPE's; the posture is indexed by target.
    assert (
        posture_store.effective_writes(section, "va", connector_type="virtual_accelerator") is False
    )
    assert posture_store.effective_writes(section, "live", connector_type="epics") is True


def test_a_bare_sandbox_refuses_every_target(data_root):
    _write_record(data_root, "sandbox")
    for target in CONTROL_TARGETS:
        assert posture_store.effective_writes(ARMED, target) is False
    assert posture_store.effective_writes(ARMED, None) is False


def test_an_unresolvable_root_leaves_the_ceiling_in_charge(monkeypatch):
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)

    def _boom():
        raise OSError("no project root")

    monkeypatch.setattr(posture_store, "resolve_shared_data_root", _boom)
    posture_store.invalidate_cache()
    assert posture_store.effective_writes(ARMED, "live") is True


# --- store_permits: the clause on its own ----------------------------------
#
# The connector's reference monitor reads a deployment ceiling keyed on the
# connector TYPE, which ``effective_writes`` cannot derive from a target, so it
# ANDs its own ceiling with this clause instead of restating the combining
# terms. That makes ``store_permits`` public API and not an implementation
# detail: rule 3 keeps exactly two implementations, this one and the hook's.


def test_store_permits_is_public_api():
    assert "store_permits" in posture_store.__all__
    assert callable(posture_store.store_permits)


def test_store_permits_takes_only_a_target():
    """The posture is the deployment's now: no session key indexes it."""
    assert list(inspect.signature(posture_store.store_permits).parameters) == ["target"]
    assert list(inspect.signature(posture_store.target_posture).parameters) == ["target"]


def test_store_permits_is_the_clause_effective_writes_uses(data_root):
    """The clause refuses → the rule refuses; the clause permits → the ceiling decides."""
    _write_record(data_root, {"live": "sandbox"})

    # The clause alone refuses; the whole rule refuses for the same reason.
    assert posture_store.store_permits("live") is False
    assert posture_store.effective_writes(ARMED, "live") is False
    # And where the clause permits, only the ceiling can still refuse.
    assert posture_store.store_permits("va") is True
    assert posture_store.effective_writes(ARMED, "va") is True
    assert posture_store.effective_writes(UNARMED, "va") is False


def test_store_permits_carries_the_no_target_rule(data_root):
    """A caller that cannot name a machine gets the most restrictive answer."""
    _write_record(data_root, {"standin": "sandbox"})

    assert posture_store.store_permits(None) is False
    assert posture_store.store_permits("") is False
    assert posture_store.store_permits("va") is True


def test_store_permits_never_consults_the_execution_mode(data_root, monkeypatch):
    """The readonly run is a separate term of rule 3, ANDed by the caller."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    _write_record(data_root, {"standin": "sandbox"})

    assert posture_store.store_permits("live") is True
    assert posture_store.effective_writes(ARMED, "live") is False


# --- the launch pin --------------------------------------------------------
#
# Unchanged by the move: it is a fact about the RUN, read from the environment,
# and it is ANDed in ahead of the record read so a sandbox that launched narrow
# refuses without touching the disk.


@pytest.mark.usefixtures("data_root")
def test_the_launch_pin_refuses_ahead_of_the_record(monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    def _explode(**_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the record was read behind a launch-pinned refusal")

    monkeypatch.setattr(control_context, "read_record", _explode)
    assert posture_store.store_permits("live") is False
    assert posture_store.effective_writes(ARMED, "live") is False


@pytest.mark.usefixtures("data_root")
def test_the_launch_pin_leaves_other_targets_alone(monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    assert posture_store.store_permits("va") is True
    assert posture_store.launch_narrowed_target() == "live"


def test_a_launch_stamp_cannot_widen_a_recorded_narrowing(data_root, monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=writes")
    _write_record(data_root, {"live": "sandbox"})
    assert posture_store.store_permits("live") is False


@pytest.mark.usefixtures("data_root")
def test_an_unstamped_process_is_unaffected_by_the_launch_clause(monkeypatch):
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    assert posture_store.launch_permits("live") is True
    assert posture_store.launch_narrowed_target() is None
    assert posture_store.store_permits("live") is True


# --- the read-only tree (a lane's queueserver, the dispatch worker) --------
#
# A container that holds no chip of its own reads ONE owner's record out of a
# read-only bind of the whole tree. Its reader is reason-returning and fails
# CLOSED, which is the whole difference from the host readers above: an
# unprovisioned bind, a group the container could not join or a record written
# 0600 would otherwise read as "nobody narrowed anything" and run an owned plan
# at the deployment ceiling — the one outcome the chip exists to prevent. Only
# an absent record is "no narrowing", because that is what it means.


def _provision_tree(root: Path) -> Path:
    """A marked, readable tree at ``<root>/control_target`` — what the build leaves."""
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _write_tree_record(tree: Path, owner: str, posture, *, target: str = "live") -> Path:
    """One owner's record inside the tree, written the way its owner writes it."""
    path = tree / owner / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=3, posture=dict(posture)),
        path=path,
    )
    posture_store.invalidate_cache()
    return path


def test_the_tree_marker_filename_agrees_with_the_build():
    """The reader restates the build's filename; a drift here fails the tree open."""
    from osprey.deployment.compose_generator import CONTROL_TREE_MARKER_NAME

    assert posture_store.CONTROL_TREE_MARKER_NAME == CONTROL_TREE_MARKER_NAME
    assert "CONTROL_TREE_MARKER_NAME" in posture_store.__all__


def test_a_record_in_the_tree_narrows_its_own_owner(tmp_path):
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", {"live": "sandbox"})
    _write_tree_record(tree, "bob", {})

    assert posture_store._read_tree_record(tree, "alice") == ({"live": "sandbox"}, None)
    # Bob's own record narrows nothing, and Alice's never reaches him.
    assert posture_store._read_tree_record(tree, "bob") == ({}, None)


def test_no_record_in_the_tree_for_this_owner_is_no_narrowing(tmp_path):
    """ENOENT is the one failure that is not unavailable: nobody narrowed anything."""
    tree = _provision_tree(tmp_path)
    narrowed, unavailable = posture_store._read_tree_record(tree, "carol")
    assert narrowed == {}
    assert unavailable is None


def test_a_record_in_the_tree_on_another_target_narrows_nothing_here(tmp_path):
    """Compare is target-only, and the same rule the host readers use."""
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", {"va": "sandbox"}, target="va")

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert unavailable is None
    assert narrowed == {"va": "sandbox"}
    assert posture_store._permits(narrowed, "live") is True
    assert posture_store._permits(narrowed, "va") is False


def test_an_unmarked_tree_is_unavailable(tmp_path):
    """A bind whose source the runtime auto-created empty must not fail open."""
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir()
    _write_tree_record(tree, "alice", {})

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(tree) in unavailable
    assert "not provisioned" in unavailable
    assert "osprey build" in unavailable


def test_a_missing_tree_is_unavailable(tmp_path):
    tree = tmp_path / posture_store.STATE_DIR_NAME

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert "not provisioned" in unavailable
    assert "osprey build" in unavailable


@pytest.mark.skipif(os.geteuid() == 0, reason="root traverses a directory at mode 0000")
def test_an_unreadable_tree_names_the_group_join(tmp_path):
    """The remedy is the container's, not the deployment's: it names the join."""
    tree = _provision_tree(tmp_path)
    tree.chmod(0o000)
    try:
        narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    finally:
        tree.chmod(0o2770)

    assert narrowed == {}
    assert unavailable is not None
    assert str(tree) in unavailable
    assert "not readable by this container" in unavailable
    assert "group join" in unavailable
    assert "Docker Desktop" in unavailable


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file at mode 0000")
def test_an_unreadable_record_in_the_tree_is_unavailable_naming_the_file(tmp_path):
    """EACCES on a PRESENT record is not "nobody narrowed anything".

    The mode is the real one a 0600 record leaves a reader running as another
    uid, rather than a patched reader: the open this reader does is the seam
    that has to refuse, and it is not the one a ``read_text`` patch would cover.
    """
    tree = _provision_tree(tmp_path)
    path = _write_tree_record(tree, "alice", {"live": "sandbox"})
    path.chmod(0o000)
    try:
        narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    finally:
        path.chmod(0o640)

    assert narrowed == {}
    assert unavailable is not None
    assert str(path) in unavailable
    assert "check the file's mode and group" in unavailable


def test_a_directory_where_the_tree_record_belongs_is_unavailable(tmp_path):
    """EISDIR is a failure on a present path, not an absent record."""
    tree = _provision_tree(tmp_path)
    (tree / "alice" / control_context.RECORD_FILENAME).mkdir(parents=True)

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert "check the file's mode and group" in unavailable


def test_a_corrupt_or_foreign_record_in_the_tree_is_unavailable(tmp_path):
    """Where a host reader answers "no narrowing", this one answers unavailable."""
    tree = _provision_tree(tmp_path)
    path = tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)

    for payload in (
        "{not json at all",
        json.dumps({"schema": 2, "target": "live", "generation": 0, "posture": "sandbox"}),
        json.dumps({"schema": 1, "target": "elsewhere", "generation": 0, "posture": "sandbox"}),
    ):
        path.write_text(payload, encoding="utf-8")
        posture_store.invalidate_cache()
        narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
        assert narrowed == {}, payload
        assert unavailable is not None, payload
        assert str(path) in unavailable, payload


def test_an_undecodable_record_in_the_tree_is_unavailable(tmp_path):
    tree = _provision_tree(tmp_path)
    path = tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe{\x00k\x00: sandbox}")
    posture_store.invalidate_cache()

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(path) in unavailable


def test_an_owner_the_tree_cannot_hold_as_a_directory_is_unavailable(tmp_path):
    """A name that is not one path segment cannot address a record — and must not permit."""
    tree = _provision_tree(tmp_path)

    for bogus in ("", "   ", ".", "..", "../alice", "al/ice", "al\\ice", "ali\x00ce"):
        narrowed, unavailable = posture_store._read_tree_record(tree, bogus)
        assert narrowed == {}, bogus
        assert unavailable is not None, bogus


def test_the_tree_reader_does_not_route_through_the_host_readers(tmp_path, monkeypatch):
    """Neither ``read_record`` nor ``recorded_posture``: both fold failure into permitted."""
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", {"live": "sandbox"})

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the tree reader went through a host reader")

    monkeypatch.setattr(control_context, "read_record", _boom)
    monkeypatch.setattr(posture_store, "recorded_posture", _boom)
    assert posture_store._read_tree_record(tree, "alice").narrowed == {"live": "sandbox"}


def test_the_tree_bind_is_read_through_one_normalisation(tmp_path, monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, f"  {tmp_path}  ")
    assert posture_store._control_context_tree() == tmp_path

    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "   ")
    assert posture_store._control_context_tree() is None

    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    assert posture_store._control_context_tree() is None


def test_a_relative_tree_or_dir_bind_is_no_bind(monkeypatch, caplog):
    """A bind names a path in the container's own filesystem; relative is a mistake."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "control_target")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, "state/mine")

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
        assert posture_store._control_context_tree() is None
        assert posture_store.bound_state_dir() is None

    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in caplog.text
    assert posture_store.CONTROL_CONTEXT_DIR_ENV_VAR in caplog.text


def test_a_tree_or_dir_bind_expands_a_tilde(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, "~/state/control_target/alice")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "~/state/control_target")

    assert posture_store.bound_state_dir() == Path.home() / "state/control_target/alice"
    assert posture_store._control_context_tree() == Path.home() / "state/control_target"


def test_a_tree_or_dir_bind_naming_an_unknown_user_does_not_raise(monkeypatch):
    """``Path.expanduser`` raises for ``~nosuchuser``; the ``os.path`` form does not."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, "~nosuchuser0/state")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "~nosuchuser0/control_target")
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, "~nosuchuser0/agent_data")

    # Left unexpanded it is not absolute, so it is no bind — and never an exception.
    assert posture_store.bound_state_dir() is None
    assert posture_store._control_context_tree() is None
    assert posture_store.stamped_agent_data_root() == Path("~nosuchuser0/agent_data")


# The tree root is group-writable (2770) and its group is shared with accounts
# other than the one reading, so any of them can plant a link under another
# identity's name. Followed, that link decides whether a real machine is written
# to from a file its supposed owner never wrote — so nothing here follows one,
# and a link is unavailable rather than a record or an absence.


def test_a_symlinked_record_in_the_tree_is_unavailable(tmp_path):
    tree = _provision_tree(tmp_path)
    planted = _write_tree_record(_provision_tree(tmp_path / "elsewhere"), "alice", {})
    record = tree / "alice" / control_context.RECORD_FILENAME
    record.parent.mkdir(parents=True)
    record.symlink_to(planted)

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(record) in unavailable
    assert "symbolic link" in unavailable


def test_a_symlinked_owner_directory_in_the_tree_is_unavailable(tmp_path):
    """``O_NOFOLLOW`` guards the record's own name; the directory above it is checked too."""
    tree = _provision_tree(tmp_path)
    elsewhere = _provision_tree(tmp_path / "elsewhere")
    _write_tree_record(elsewhere, "alice", {})
    (tree / "alice").symlink_to(elsewhere / "alice", target_is_directory=True)

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(tree / "alice") in unavailable
    assert "symbolic link" in unavailable


def test_a_symlinked_tree_marker_is_unavailable(tmp_path):
    """A link here would let any account in the group declare an unbuilt tree built."""
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir()
    elsewhere = tmp_path / "somebodys-file"
    elsewhere.write_text("not the build's marker\n", encoding="utf-8")
    marker = tree / posture_store.CONTROL_TREE_MARKER_NAME
    marker.symlink_to(elsewhere)

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(marker) in unavailable
    assert "symbolic link" in unavailable


def test_a_tree_marker_that_is_not_a_file_is_unavailable(tmp_path):
    tree = tmp_path / posture_store.STATE_DIR_NAME
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).mkdir(parents=True)

    narrowed, unavailable = posture_store._read_tree_record(tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert "not provisioned" in unavailable


# --- store_verdict: the answer, and the reason it answered that way --------
#
# A bool says a write was refused; only a name says what to put in front of an
# operator. The verdict is what a consumer branches on, so the reason a refusal
# message names and the reason the code took cannot drift apart. Three answers,
# because three things are findable here: nothing narrows this target, somebody
# narrowed it, or the narrowing could not be read at all — which the tree
# reader treats as a refusal rather than an absence.


@pytest.fixture
def bound_tree(tmp_path, monkeypatch):
    """A provisioned read-only tree bound in, with no owner named anywhere yet."""
    tree = _provision_tree(tmp_path)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    posture_store.invalidate_cache()
    yield tree
    posture_store.invalidate_cache()


def test_store_verdict_is_public_api():
    """The reason travels by name; consumers import the enum, not the wording."""
    assert "StoreVerdict" in posture_store.__all__
    assert "store_verdict" in posture_store.__all__

    assert [member.value for member in posture_store.StoreVerdict] == [
        "permitted",
        "narrowing",
        "control_context_unavailable",
    ]
    # A ``StrEnum``: the value crosses the connector IPC as a bare string, so a
    # far-side consumer compares the word and an identity check would fail.
    assert posture_store.StoreVerdict.NARROWING == "narrowing"


def test_store_verdict_takes_a_target_and_an_owner():
    """An owner is WHO the work belongs to; a session key was an INDEX into the posture.

    The retired session key made one browser tab's narrowing a different
    narrowing from another's, which is why no reader here takes one: the
    posture is the deployment's, filed per identity. The owner is the opposite
    kind of parameter — it does not select among several postures for one
    person, it names the person whose one posture governs this write, which is
    the only question a container holding every identity's records can ask.
    """
    assert list(inspect.signature(posture_store.store_verdict).parameters) == ["target", "owner"]


@pytest.mark.usefixtures("data_root")
def test_nothing_narrowed_is_permitted():
    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.PERMITTED
    assert posture_store.store_permits("live") is True


def test_a_recorded_narrowing_answers_narrowing(data_root):
    _write_record(data_root, {"live": "sandbox"})

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.NARROWING
    assert posture_store.store_permits("live") is False


def test_a_narrowing_on_another_target_permits_this_one(data_root):
    """Compare is target-only for every reader — the same rule the tree reader uses."""
    _write_record(data_root, {"va": "sandbox"}, target="va")

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.PERMITTED
    assert posture_store.store_verdict("va") is posture_store.StoreVerdict.NARROWING


@pytest.mark.usefixtures("data_root")
def test_a_launch_pin_naming_a_target_answers_narrowing(monkeypatch):
    """Somebody had this machine read-only when the run started — a decision."""
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.NARROWING
    assert posture_store.store_verdict("va") is posture_store.StoreVerdict.PERMITTED


@pytest.mark.usefixtures("data_root")
def test_a_launch_pin_on_every_target_answers_control_context_unavailable(monkeypatch):
    """Nobody decided this: the executor could resolve neither target nor record.

    Reporting it as a narrowing would send an operator to a chip to undo a
    decision they never made, so the reason word forks where the message
    already forks.
    """
    monkeypatch.setenv(
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.launch_posture_stamp(None, posture_store.POSTURE_SANDBOX),
    )

    verdict = posture_store.store_verdict("live")
    assert verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert posture_store.store_permits("live") is False
    assert posture_store.store_verdict(None) is verdict


@pytest.mark.usefixtures("data_root")
def test_the_launch_pin_is_taken_before_any_record_is_read(monkeypatch):
    """The pin is one environment read, so a run that launched narrow touches no disk."""

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the launch pin let a record read happen")

    monkeypatch.setattr(posture_store, "recorded_posture", _boom)
    monkeypatch.setattr(posture_store, "_read_tree_record", _boom)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.NARROWING


@pytest.mark.usefixtures("bound_tree")
def test_work_that_belongs_to_nobody_reads_no_record(monkeypatch):
    """The ceiling alone governs an owner-less plan, so there is no record to read.

    Passing the sentinel is naming an owner — "this belongs to nobody" — and it
    must not become a refusal: a container holding the tree cannot look up a
    record for a name it was never given, and the deployment ceiling is what
    stands in for one.
    """

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("an owner-less verdict read a record")

    monkeypatch.setattr(posture_store, "_read_tree_record", _boom)
    monkeypatch.setattr(posture_store, "recorded_posture", _boom)

    assert (
        posture_store.store_verdict("live", posture_store.NO_OWNER)
        is posture_store.StoreVerdict.PERMITTED
    )


def test_the_tree_reader_answers_for_the_owner_a_plan_carries(bound_tree):
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    _write_tree_record(bound_tree, "bob", {})

    assert posture_store.store_verdict("live", "alice") is posture_store.StoreVerdict.NARROWING
    # One owner's chip never reaches another's plan, and never another target.
    assert posture_store.store_verdict("live", "bob") is posture_store.StoreVerdict.PERMITTED
    assert posture_store.store_verdict("va", "alice") is posture_store.StoreVerdict.PERMITTED


@pytest.mark.usefixtures("bound_tree")
def test_an_owner_with_no_record_in_the_tree_is_permitted():
    """The one absence that is an answer: nobody narrowed anything for this owner."""
    assert posture_store.store_verdict("live", "carol") is posture_store.StoreVerdict.PERMITTED


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file at mode 0000")
def test_a_present_but_unreadable_record_in_the_tree_refuses(bound_tree, monkeypatch, caplog):
    """Where the host readers answer "no narrowing", the tree reader refuses.

    A record written 0600 to a reader running as another uid is a narrowing
    that may exist and cannot be read — the one outcome that must not run an
    owned plan at the deployment ceiling. The remedy the read composed is
    logged, because the verdict carries the reason and not the sentence.
    """
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    path = _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    path.chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
            verdict = posture_store.store_verdict("live")
        # The bool spelling refuses on the same read, which is what keeps the
        # reference monitor from proceeding on an unreadable narrowing.
        permitted = posture_store.store_permits("live")
    finally:
        path.chmod(0o640)

    assert verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert permitted is False
    assert str(path) in caplog.text


def test_an_unprovisioned_tree_bind_refuses_rather_than_permits(tmp_path, monkeypatch):
    """A bind the runtime auto-created empty is not "nobody narrowed anything"."""
    monkeypatch.setenv(
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / posture_store.STATE_DIR_NAME)
    )
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)

    verdict = posture_store.store_verdict("live", "alice")
    assert verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    # Nobody named: the ceiling governs, and no unreadable bind is consulted.
    assert posture_store.store_verdict("live", posture_store.NO_OWNER) is (
        posture_store.StoreVerdict.PERMITTED
    )


@pytest.mark.usefixtures("bound_tree")
def test_an_owner_the_tree_cannot_hold_refuses():
    """A name nobody can look up is a lookup that did not happen, not one that found nothing."""
    assert (
        posture_store.store_verdict("live", "../alice")
        is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    )


def test_the_bound_owner_decides_which_record_the_verdict_reads(bound_tree):
    """The rung that makes a lane's plan obey the chip of whoever queued it."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    with posture_store.bind_owner({posture_store.RESERVED_OWNER_KWARG: "alice"}):
        assert posture_store.store_verdict("live") is posture_store.StoreVerdict.NARROWING
    # Outside the block the container holds no chip of its own.
    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.PERMITTED


def test_the_owner_stamped_in_the_environment_decides_a_dispatch_job(bound_tree, monkeypatch):
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.NARROWING
    assert posture_store.store_permits("live") is False


def test_without_a_tree_bind_the_host_record_decides(data_root, monkeypatch):
    """An owner does not redirect a host read: the record there is the reader's own."""
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    _write_record(data_root, {"live": "sandbox"})

    assert posture_store.store_verdict("live", "alice") is posture_store.StoreVerdict.NARROWING


@pytest.mark.usefixtures("data_root")
def test_a_raising_host_reader_still_leaves_the_ceiling_in_charge(monkeypatch):
    """The host rung keeps its fail-open; only the tree rung refuses on a bad read."""
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)

    def _boom(*_args, **_kwargs):
        raise OSError("the record could not be read")

    monkeypatch.setattr(control_context, "read_record", _boom)
    posture_store.invalidate_cache()

    assert posture_store.store_verdict("live") is posture_store.StoreVerdict.PERMITTED


# The two rungs that answer PERMITTED without reading anything are the only
# fail-open branches in the verdict, so each one takes exactly the value it is
# for and nothing else: the sentinel, and a process that was handed no tree.
# Everything else that is not a name, or is a tree that cannot be opened, is a
# lookup that did not happen — the tree reader's own classification.


def test_an_owner_that_is_not_a_name_refuses(bound_tree, caplog):
    """The ceiling-only rung is for the sentinel alone, never for a type.

    An owner can arrive from a decoded payload — a queue item's reserved kwarg
    is JSON — so a list or a number is a shape the wire can carry where a name
    was meant. Read as "nobody owns this" it would run an owned plan at the
    deployment ceiling, which is the outcome an owner exists to prevent.
    """
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
        for junk in (b"alice", 17, Path("alice"), ["alice"], object()):
            assert (
                posture_store.store_verdict("live", junk)
                is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
            ), junk
    assert "not a name" in caplog.text


@pytest.mark.usefixtures("bound_tree")
def test_an_owner_named_with_no_name_refuses():
    """A caller that named an owner spelled "" named a directory nobody can hold."""
    assert (
        posture_store.store_verdict("live", "   ")
        is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    )


def test_a_padded_owner_addresses_the_same_record(bound_tree):
    """One person, one directory: every rung of the ladder strips the name it answers.

    A rendered compose value and a hand-built call both produce padding, and an
    unstripped name is an ENOENT on ``<tree>/ alice/`` — the one absence the
    reader is allowed to read as "this owner narrowed nothing".
    """
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    assert posture_store.current_owner("  alice  ") == "alice"
    assert posture_store.store_verdict("live", " alice") is posture_store.StoreVerdict.NARROWING
    assert posture_store.store_verdict("live", "alice ") is posture_store.StoreVerdict.NARROWING


@pytest.mark.parametrize(
    "bind",
    ["control_target", "~nosuchuser0/control_target"],
    ids=["relative", "unknown-user"],
)
def test_a_bind_that_is_set_but_unusable_refuses(bind, monkeypatch, caplog):
    """A bind is bound or it is not — one answer, so it cannot be a tree here and a host there.

    A relative path, or one through a ``~user`` this container has no passwd
    entry for, names no usable directory — a deployment mistake of the same
    class as an unprovisioned mount. Falling back to the host reader would send
    a container that has no record of its own to the reader whose every way of
    not knowing is "nothing is narrowed".
    """

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("an unusable tree bind fell through to the host reader")

    monkeypatch.setattr(posture_store, "recorded_posture", _boom)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)

    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, bind)
    # One spelling of the question: the owner ladder calls this a container
    # holding the tree, and the verdict must not call it a host.
    assert posture_store.current_owner() is posture_store.NO_OWNER
    with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
        assert (
            posture_store.store_verdict("live", "alice")
            is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
        )
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in caplog.text


def test_an_oversized_record_in_the_tree_is_unavailable(bound_tree):
    """The read is bounded: a record is a few hundred bytes, and the tree is shared.

    The file is re-read on every gated write, so an unbounded read would let the
    account whose plan is running choose how much the container allocates per
    write. Over-long is neither a narrowing nor an absence — it is a record that
    could not be read.
    """
    path = bound_tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    posture_store.invalidate_cache()

    narrowed, unavailable = posture_store._read_tree_record(bound_tree, "alice")
    assert narrowed == {}
    assert unavailable is not None
    assert str(path) in unavailable
    # Its own remedy, not the unreadable record's: this file's mode and group
    # permitted the read, so an operator sent to check them finds both correct.
    assert "larger than" in unavailable
    assert "mode and group" not in unavailable
    assert (
        posture_store.store_verdict("live", "alice")
        is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    )


def test_a_record_of_ordinary_size_is_still_read(bound_tree):
    """The bound is headroom, not a limit a real record approaches."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    assert posture_store.store_verdict("live", "alice") is posture_store.StoreVerdict.NARROWING


# --- store_verdict_detail: the verdict, and the sentence for the one arm ----
#
# Only ``control_context_unavailable`` has a remedy the refusing process cannot
# compose for itself: the sentence names the path that could not be read, the
# variable that named no usable one, or the owner that was not a name — facts
# that live in this reader and nowhere else. The other two verdicts are a grant
# and somebody's decision, and a consumer words those from the verdict and the
# target alone, so they carry no reason.


def test_store_verdict_detail_is_public_api():
    """The reason reaches a consumer through a named pair, not through the log."""
    assert "StoreVerdictDetail" in posture_store.__all__
    assert "store_verdict_detail" in posture_store.__all__

    assert posture_store.StoreVerdictDetail._fields == ("verdict", "reason")
    assert list(inspect.signature(posture_store.store_verdict_detail).parameters) == [
        "target",
        "owner",
    ]


def test_a_permitted_or_narrowed_answer_carries_no_reason(bound_tree):
    """A grant and a decision are both fully worded from the verdict and the target."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    permitted = posture_store.store_verdict_detail("va", "alice")
    narrowing = posture_store.store_verdict_detail("live", "alice")

    assert permitted == (posture_store.StoreVerdict.PERMITTED, None)
    assert narrowing == (posture_store.StoreVerdict.NARROWING, None)


def test_one_record_read_per_verdict(bound_tree, monkeypatch):
    """The short spelling asks the detail for its answer, so neither reads twice.

    The record is re-read on every gated write — that is what lands a chip flip
    on a session already mid-conversation — so a second read inside one
    decision is a second chance for the answer to change underneath it.
    """
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    reads: list[str] = []
    real = posture_store._read_tree_record

    def _counting(tree, owner):
        reads.append(owner)
        return real(tree, owner)

    monkeypatch.setattr(posture_store, "_read_tree_record", _counting)

    assert posture_store.store_verdict("live", "alice") is posture_store.StoreVerdict.NARROWING
    assert reads == ["alice"]
    assert posture_store.store_verdict_detail("live", "alice").verdict is (
        posture_store.StoreVerdict.NARROWING
    )
    assert reads == ["alice", "alice"]


@pytest.mark.usefixtures("data_root")
def test_a_pin_on_every_target_hands_over_the_launch_remedy(monkeypatch):
    """Nobody decided this one, so its remedy is the run's and not the chip's."""
    monkeypatch.setenv(
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.launch_posture_stamp(None, posture_store.POSTURE_SANDBOX),
    )

    detail = posture_store.store_verdict_detail("live")

    assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert detail.reason == "pinned everywhere at launch — re-run the script"


def test_record_verdict_detail_is_public_api():
    """The record clause without the pin is a name a refusal can ask by."""
    assert "record_verdict_detail" in posture_store.__all__
    assert list(inspect.signature(posture_store.record_verdict_detail).parameters) == [
        "target",
        "owner",
    ]


def test_without_a_pin_the_store_and_the_record_answer_alike(bound_tree):
    """With no launch pin the store's clause IS the record's."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    for target in ("live", "va", None):
        assert posture_store.store_verdict_detail(
            target, "alice"
        ) == posture_store.record_verdict_detail(target, "alice")


def test_a_pin_hides_the_record_from_the_store_but_not_from_the_record_clause(
    bound_tree, monkeypatch
):
    """The pin answers the store without a read; the record clause spends that read."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    monkeypatch.setenv(
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.launch_posture_stamp("live", posture_store.POSTURE_SANDBOX),
    )
    reads: list[str] = []
    real = posture_store._read_tree_record

    def _counting(tree, owner):
        reads.append(owner)
        return real(tree, owner)

    monkeypatch.setattr(posture_store, "_read_tree_record", _counting)

    assert posture_store.store_verdict_detail("live", "alice") == (
        posture_store.StoreVerdict.NARROWING,
        None,
    )
    assert reads == []
    assert posture_store.record_verdict_detail("live", "alice") == (
        posture_store.StoreVerdict.NARROWING,
        None,
    )
    assert reads == ["alice"]


def test_the_record_clause_reports_an_unreadable_record_under_a_pin(monkeypatch):
    """A pin answers narrowing; the record behind it may still be unreadable."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "control_target")
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    monkeypatch.setenv(
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.launch_posture_stamp("live", posture_store.POSTURE_SANDBOX),
    )

    assert posture_store.store_verdict_detail("live").verdict is (
        posture_store.StoreVerdict.NARROWING
    )
    detail = posture_store.record_verdict_detail("live")
    assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in detail.reason


@pytest.mark.usefixtures("data_root")
def test_a_pin_naming_a_target_carries_no_remedy(monkeypatch):
    """A named pin is an operator's decision, and its wording is the caller's."""
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    assert posture_store.store_verdict_detail("live") == (
        posture_store.StoreVerdict.NARROWING,
        None,
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file at mode 0000")
def test_an_unreadable_record_hands_over_the_sentence_it_logs(bound_tree, monkeypatch, caplog):
    """One sentence, two destinations: the deployment's log and the refused write.

    The log reaches whoever fixes the bind; the returned sentence reaches
    whoever ran the write, who may be the only person who ever sees it.
    """
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    path = _write_tree_record(bound_tree, "alice", {"live": "sandbox"})
    path.chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
            detail = posture_store.store_verdict_detail("live")
    finally:
        path.chmod(0o640)

    assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert str(path) in detail.reason
    assert "check the file's mode and group" in detail.reason
    assert detail.reason in caplog.text


def test_an_unusable_bind_hands_over_the_variable_that_names_no_path(monkeypatch, caplog):
    """The remedy is a deployment's: the bind, not anybody's chip."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "control_target")
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
        detail = posture_store.store_verdict_detail("live", "alice")

    assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in detail.reason
    assert detail.reason in caplog.text


def test_an_owner_that_is_not_a_name_hands_over_a_sentence_saying_so(bound_tree, caplog):
    """A refusal nobody can act on at a chip still has to say what happened."""
    _write_tree_record(bound_tree, "alice", {"live": "sandbox"})

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.posture_store"):
        detail = posture_store.store_verdict_detail("live", ["alice"])

    assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE
    assert "not a name" in detail.reason
    assert detail.reason in caplog.text


def test_every_unavailable_answer_carries_a_reason(bound_tree, monkeypatch):
    """The pairing is the contract: an unavailable verdict with no sentence is a
    refusal handed to an operator with nothing to do about it."""
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(bound_tree / "unbuilt"))

    for owner in ("alice", "../alice", 17):
        detail = posture_store.store_verdict_detail("live", owner)
        assert detail.verdict is posture_store.StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE, owner
        assert detail.reason, owner
