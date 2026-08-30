"""Contract tests for the per-(session, target) write-posture store.

This is the store half of the control-target header chip: the file the web
server writes when an operator narrows one target, and the three readers
(`osprey_connectors.session_store`, the controls server, the stdlib hook)
answer from. The tests here pin the parts every reader has to agree on —
where the file is, what its shapes mean, and how a lookup combines with the
deployment ceiling and a read-only run.

Not to be confused with ``tests/interfaces/test_session_store.py``, which
covers the web terminal's own session store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osprey_connectors import session_store
from osprey_connectors.types import CONTROL_TARGETS

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
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    session_store.invalidate_cache()
    yield tmp_path
    session_store.invalidate_cache()


def _write_store(root: Path, payload) -> Path:
    path = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_legacy(root: Path, payload) -> Path:
    path = root / session_store.STORE_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- path resolution -------------------------------------------------------


def test_store_path_prefers_the_env_stamp(data_root):
    assert session_store.agent_data_root() == data_root
    assert session_store.state_dir() == data_root / "control_target"
    assert session_store.store_path() == data_root / "control_target" / "session-postures.json"


def test_store_path_falls_back_to_the_shared_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    session_store.invalidate_cache()
    assert (
        session_store.store_path() == tmp_path / "var" / "control_target" / "session-postures.json"
    )


def test_blank_env_stamp_is_not_a_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, "   ")
    monkeypatch.setattr(session_store, "resolve_shared_data_root", lambda: tmp_path / "var")
    session_store.invalidate_cache()
    assert (
        session_store.store_path() == tmp_path / "var" / "control_target" / "session-postures.json"
    )


def test_unresolvable_root_answers_none(monkeypatch):
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)

    def _boom():
        raise OSError("no project root")

    monkeypatch.setattr(session_store, "resolve_shared_data_root", _boom)
    session_store.invalidate_cache()
    assert session_store.agent_data_root() is None
    assert session_store.state_dir() is None
    assert session_store.store_path() is None
    # A caller that cannot resolve the store sees an empty map, not a crash;
    # surfacing ``store_unavailable`` is the route's job, off ``store_path()``.
    assert session_store.load_store() == {}


def test_store_sits_beside_the_target_state_file(data_root):
    """FR9: co-sited with the state file — one directory, not two."""
    from osprey.mcp_server.control_system import target_state

    assert session_store.STATE_DIR_NAME == target_state.STATE_DIR_NAME


# --- parse_store -----------------------------------------------------------


def test_bare_sandbox_expands_to_every_target():
    parsed = session_store.parse_store({"s1": "sandbox"})
    assert parsed == {"s1": dict.fromkeys(CONTROL_TARGETS, "sandbox")}


def test_bare_writes_is_dropped():
    assert session_store.parse_store({"s1": "writes"}) == {}


def test_unknown_values_are_dropped():
    parsed = session_store.parse_store(
        {"s1": "readonly", "s2": 1, "s3": None, "s4": ["sandbox"], "s5": "sandbox"}
    )
    assert parsed == {"s5": dict.fromkeys(CONTROL_TARGETS, "sandbox")}


def test_per_target_values_are_validated_the_same_way():
    parsed = session_store.parse_store(
        {"s1": {"live": "sandbox", "va": "writes", "standin": "bogus", "other": 3}}
    )
    assert parsed == {"s1": {"live": "sandbox"}}


def test_a_map_that_narrows_nothing_is_dropped():
    assert session_store.parse_store({"s1": {"live": "writes"}}) == {}


def test_operator_keys_are_retained():
    """The drop-on-restore rule lives only in the web server's startup load."""
    parsed = session_store.parse_store({"operator-abc12345": "sandbox"})
    assert "operator-abc12345" in parsed


def test_non_mapping_and_bad_keys_are_tolerated():
    assert session_store.parse_store(["sandbox"]) == {}
    assert session_store.parse_store(None) == {}
    assert session_store.parse_store({3: "sandbox"}) == {}


def test_parse_store_accepts_raw_json_text():
    parsed = session_store.parse_store('{"s1": {"va": "sandbox"}}')
    assert parsed == {"s1": {"va": "sandbox"}}


def test_corrupt_json_is_an_empty_store():
    assert session_store.parse_store("{not json") == {}


# --- load / lookups --------------------------------------------------------


def test_missing_file_is_an_empty_map_not_an_error(data_root):
    assert session_store.load_store() == {}
    assert session_store.session_map("s1") == {}
    assert session_store.target_posture("s1", "live") is None


def test_an_undecodable_file_is_an_empty_store_not_an_exception(data_root):
    """A store that is not UTF-8 degrades like a corrupt one, rather than raising.

    ``UnicodeDecodeError`` is a ``ValueError``, so it slips past the ``OSError``
    guard on the read and has to be named explicitly. Every reader here sits on
    a write path — the connector's reference monitor asks on each write — so an
    exception from one mis-encoded file would be raised into the write itself
    instead of leaving the deployment ceiling in charge.
    """
    path = data_root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe{\x00k\x00: sandbox}")
    session_store.invalidate_cache()

    assert session_store.load_store() == {}
    assert session_store.session_map("k") == {}
    assert session_store.store_permits("k", "live") is True


def test_session_map_and_target_posture(data_root):
    _write_store(data_root, {"s1": {"live": "sandbox"}, "s2": "sandbox"})
    assert session_store.session_map("s1") == {"live": "sandbox"}
    assert session_store.target_posture("s1", "live") == "sandbox"
    assert session_store.target_posture("s1", "va") is None
    assert session_store.target_posture("s2", "va") == "sandbox"
    assert session_store.session_map("nobody") == {}
    assert session_store.session_map(None) == {}
    assert session_store.target_posture(None, "live") is None


def test_legacy_path_store_is_read_through_when_the_new_one_is_absent(data_root):
    _write_legacy(data_root, {"s1": "sandbox"})
    assert session_store.target_posture("s1", "live") == "sandbox"
    # The new path wins the moment it exists.
    _write_store(data_root, {"s1": {"va": "sandbox"}})
    assert session_store.target_posture("s1", "live") is None
    assert session_store.target_posture("s1", "va") == "sandbox"


# --- cache -----------------------------------------------------------------


def test_cache_is_invalidated_by_two_flips_within_one_second(data_root):
    """A coarse filesystem clock must not hide the second flip.

    Both writes go through the atomic temp+rename the web server uses, so the
    signature moves on size and inode even when ``st_mtime_ns`` does not.
    """
    path = session_store.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic(payload):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    _atomic({"s1": {"live": "sandbox"}})
    assert session_store.target_posture("s1", "live") == "sandbox"
    _atomic({"s1": {"va": "sandbox"}})
    assert session_store.target_posture("s1", "live") is None
    assert session_store.target_posture("s1", "va") == "sandbox"
    _atomic({"s1": {"live": "sandbox", "va": "sandbox"}})
    assert session_store.target_posture("s1", "live") == "sandbox"


def test_cache_notices_the_file_appearing_and_disappearing(data_root):
    assert session_store.load_store() == {}
    path = _write_store(data_root, {"s1": "sandbox"})
    assert session_store.session_map("s1")
    path.unlink()
    assert session_store.load_store() == {}


def test_cache_notices_a_moved_root(data_root, tmp_path, monkeypatch):
    _write_store(data_root, {"s1": "sandbox"})
    assert session_store.session_map("s1")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(other))
    assert session_store.session_map("s1") == {}


def test_repeated_reads_do_not_reparse(data_root, monkeypatch):
    _write_store(data_root, {"s1": "sandbox"})
    assert session_store.session_map("s1")
    calls = []
    original = session_store.parse_store
    monkeypatch.setattr(
        session_store,
        "parse_store",
        lambda raw: (calls.append(raw), original(raw))[1],
    )
    session_store.load_store()
    session_store.load_store()
    assert calls == []


# --- effective_writes ------------------------------------------------------


@pytest.mark.parametrize("armed", [True, False])
@pytest.mark.parametrize("entry", [None, "sandbox", "writes"])
@pytest.mark.parametrize("readonly", [True, False])
def test_effective_writes_truth_table(data_root, monkeypatch, armed, entry, readonly):
    section = ARMED if armed else UNARMED
    if entry is not None:
        _write_store(data_root, {"s1": {"live": entry}})
    if readonly:
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    expected = armed and not readonly and entry != "sandbox"
    assert session_store.effective_writes(section, "s1", "live") is expected


def test_effective_writes_uses_the_targets_own_ceiling(data_root):
    section = _section(live_writes=False, va_writes=True)
    assert session_store.effective_writes(section, "s1", "live") is False
    assert session_store.effective_writes(section, "s1", "va") is True


def test_without_a_session_key_the_store_is_not_consulted(data_root):
    _write_store(data_root, {"s1": "sandbox"})
    assert session_store.effective_writes(ARMED, None, "live") is True
    assert session_store.effective_writes(ARMED, "", "live") is True


def test_no_target_takes_the_most_restrictive_entry(data_root):
    _write_store(data_root, {"s1": {"va": "sandbox"}})
    # Any sandbox entry for this key refuses when the caller holds no target.
    assert session_store.effective_writes(ARMED, "s1", None) is False
    # A key whose entries narrow nothing leaves the ceiling in charge.
    assert session_store.effective_writes(ARMED, "s2", None) is True


def test_connector_type_ceiling_beats_the_deployment_wide_key(data_root):
    """Mixed config: deployment armed, the EPICS block explicitly unarmed."""
    section = {
        "type": "epics",
        "writes_enabled": True,
        "connector": {"epics": {"prefix": "X:", "writes_enabled": False}},
    }
    assert session_store.effective_writes(section, None, None, connector_type="epics") is False
    armed = {
        "type": "epics",
        "writes_enabled": False,
        "connector": {"epics": {"prefix": "X:", "writes_enabled": True}},
    }
    assert session_store.effective_writes(armed, None, None, connector_type="epics") is True


def test_connector_type_ceiling_with_a_target_stamp_indexes_the_store(data_root):
    _write_store(data_root, {"s1": {"va": "sandbox"}})
    section = _section(live_writes=True, va_writes=True)
    # The ceiling stays the connector TYPE's; the store is indexed by target.
    assert (
        session_store.effective_writes(section, "s1", "va", connector_type="virtual_accelerator")
        is False
    )
    assert session_store.effective_writes(section, "s1", "live", connector_type="epics") is True


def test_legacy_sandbox_entry_refuses_every_target(data_root):
    _write_store(data_root, {"s1": "sandbox"})
    for target in CONTROL_TARGETS:
        assert session_store.effective_writes(ARMED, "s1", target) is False
    assert session_store.effective_writes(ARMED, "s1", None) is False


def test_operator_key_is_enforced_by_the_reader(data_root):
    _write_store(data_root, {"operator-abc12345": {"live": "sandbox"}})
    assert session_store.effective_writes(ARMED, "operator-abc12345", "live") is False


def test_unresolvable_store_leaves_the_ceiling_in_charge(monkeypatch):
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    def _boom():
        raise OSError("no project root")

    monkeypatch.setattr(session_store, "resolve_shared_data_root", _boom)
    session_store.invalidate_cache()
    assert session_store.effective_writes(ARMED, "s1", "live") is True


# --- store_permits: the clause on its own ----------------------------------
#
# The connector's reference monitor reads a deployment ceiling keyed on the
# connector TYPE, which ``effective_writes`` cannot derive from a target, so it
# ANDs its own ceiling with this clause instead of restating the four combining
# terms. That makes ``store_permits`` public API and not an implementation
# detail: rule 3 keeps exactly two implementations, this one and the hook's.


def test_store_permits_is_public_api():
    assert "store_permits" in session_store.__all__
    assert callable(session_store.store_permits)


def test_store_permits_is_the_clause_effective_writes_uses(data_root):
    """Same function, not a copy — a divergence here is a divergence there."""
    _write_store(data_root, {"s1": {"live": "sandbox"}})

    # The clause alone refuses; the whole rule refuses for the same reason.
    assert session_store.store_permits("s1", "live") is False
    assert session_store.effective_writes(ARMED, "s1", "live") is False
    # And where the clause permits, only the ceiling can still refuse.
    assert session_store.store_permits("s1", "va") is True
    assert session_store.effective_writes(ARMED, "s1", "va") is True
    assert session_store.effective_writes(UNARMED, "s1", "va") is False


def test_store_permits_carries_the_no_key_and_no_target_rules(data_root):
    """The two rules a caller must not re-derive: no key permits, and no target
    takes the most restrictive entry."""
    _write_store(data_root, {"s1": {"standin": "sandbox"}})

    assert session_store.store_permits(None, "standin") is True
    assert session_store.store_permits("", "standin") is True
    assert session_store.store_permits("s1", None) is False
    assert session_store.store_permits("s2", None) is True
    assert session_store.store_permits("s1", "va") is True


def test_store_permits_never_consults_the_execution_mode(data_root, monkeypatch):
    """The readonly run is a separate term of rule 3, ANDed by the caller."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    _write_store(data_root, {"s1": {"standin": "sandbox"}})

    assert session_store.store_permits("s1", "live") is True
    assert session_store.effective_writes(ARMED, "s1", "live") is False
