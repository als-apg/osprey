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
import os
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
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
    assert posture_store.state_dir() == data_root / "control_target"


def test_root_falls_back_to_the_shared_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(posture_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    posture_store.invalidate_cache()
    assert posture_store.state_dir() == tmp_path / "var" / "control_target"


def test_blank_env_stamp_is_not_a_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, "   ")
    monkeypatch.setattr(posture_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    posture_store.invalidate_cache()
    assert posture_store.state_dir() == tmp_path / "var" / "control_target"


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


def test_the_posture_sits_beside_the_target_state_file(data_root):
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


def test_no_record_is_no_narrowing(data_root):
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


def test_effective_writes_uses_the_targets_own_ceiling(data_root):
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


def test_connector_type_ceiling_beats_the_deployment_wide_key(data_root):
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
    """Same function, not a copy — a divergence here is a divergence there."""
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


def test_the_launch_pin_refuses_ahead_of_the_record(data_root, monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    def _explode(**_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the record was read behind a launch-pinned refusal")

    monkeypatch.setattr(control_context, "read_record", _explode)
    assert posture_store.store_permits("live") is False
    assert posture_store.effective_writes(ARMED, "live") is False


def test_the_launch_pin_leaves_other_targets_alone(data_root, monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    assert posture_store.store_permits("va") is True
    assert posture_store.launch_narrowed_target() == "live"


def test_a_launch_stamp_cannot_widen_a_recorded_narrowing(data_root, monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=writes")
    _write_record(data_root, {"live": "sandbox"})
    assert posture_store.store_permits("live") is False


def test_an_unstamped_process_is_unaffected_by_the_launch_clause(data_root, monkeypatch):
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    assert posture_store.launch_permits("live") is True
    assert posture_store.launch_narrowed_target() is None
    assert posture_store.store_permits("live") is True
