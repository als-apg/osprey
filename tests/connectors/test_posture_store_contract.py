"""Contract tests for the write-posture reader.

The write state lives in the control-context record — one ``switch`` per
machine, per identity — and the record's ``posture`` field is only a mirror of
it, kept in the grammar the retired ``session-postures.json`` used. These tests
pin the parts every reader has to agree on: where the file is, what the
posture field's shapes mean to an old reader, when a change is seen, and what
the fail-closed tree reader answers. The verdict itself — switch, tier, owner
and launch pin in one order — is pinned by the case table
(``osprey_connectors_tests/_harness/fixtures/write_verdict_cases.json``).

Not to be confused with ``tests/interfaces/test_session_store.py``, which
covers the web terminal's own session store.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import acting_identity
from osprey_connectors.types import CONTROL_TARGETS
from osprey_connectors_tests._harness import write_control_context, write_payload

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


def _write_record(
    root: Path, posture, *, target: str = "live", generation: int = 3, switch=None
) -> Path:
    """A complete record whose ``posture`` field is *posture* VERBATIM, plus a cache drop.

    An old ``sandbox`` value, as a file written before the mirror carries it: no writer
    stores one any more, so it is laid over the written payload by hand. The
    record is otherwise written by :func:`write_control_context`, so this suite
    pins what the reader does with a payload rather than restating what one is.
    """
    path = write_control_context(
        root, target=target, generation=generation, owned_by=None, switch=switch
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["posture"] = posture
    write_payload(path, payload)
    posture_store.invalidate_cache()
    return path


def _switch_record(root: Path, *switched_on: str, target: str = "live") -> Path:
    """A record with exactly *switched_on* turned on, the way the person's gesture stores it."""
    path = write_control_context(
        root, target=target, owned_by=None, switch=dict.fromkeys(switched_on, "asked")
    )
    posture_store.invalidate_cache()
    return path


#: Every machine on except ``live``.
LIVE_OFF = ("va", "standin")
ALL_ON = tuple(CONTROL_TARGETS)


def _mirror() -> dict[str, str]:
    """The record's ``posture`` mirror as the record reader parses it; ``{}`` for no record."""
    record = control_context.read_record()
    return {} if record is None else record.posture


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
    # A caller that cannot resolve the root reads no record, not a crash.
    assert control_context.read_record() is None


def test_the_next_read_answers_once_the_root_is_back(tmp_path, monkeypatch):
    """Nothing is cached against a root that did not resolve.

    A transient config failure answers "no record"; a cached answer would keep
    saying so after the root came back, hiding the record that is on disk.
    """
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)

    def _boom():
        raise RuntimeError("no project root")

    monkeypatch.setattr(posture_store, "resolve_shared_data_root", _boom)
    posture_store.invalidate_cache()
    assert _mirror() == {}

    root = tmp_path / "recovered"
    _write_record(root, {"standin": posture_store.POSTURE_SANDBOX})
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))

    posture_store.invalidate_cache()
    assert _mirror() == {"standin": posture_store.POSTURE_SANDBOX}


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
    _switch_record(data_root, "live")

    record = control_context.read_record()
    assert record is not None
    assert record.switch == {"live": "asked"}
    assert "live" not in record.posture


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
    record = control_context.read_record()
    assert record is not None
    assert record.posture == {"live": "sandbox", "nonsense": "sandbox"}
    assert record.posture == posture_store.parse_posture_value(
        {"live": "sandbox", "va": "writes", "nonsense": "sandbox"}
    )


# --- cache -----------------------------------------------------------------


def test_two_flips_within_one_second_are_both_seen(data_root):
    """A coarse filesystem clock must not hide the second flip.

    Both writes go through the atomic temp+rename the owner uses, so the
    signature moves on size and inode even when ``st_mtime_ns`` does not.
    """
    path = control_context._record_path_under(data_root)
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
    assert _mirror().get("live") == "sandbox"
    _atomic({"va": "sandbox"})
    assert _mirror().get("live") is None
    assert _mirror().get("va") == "sandbox"
    _atomic({"live": "sandbox", "va": "sandbox"})
    assert _mirror().get("live") == "sandbox"


def test_the_record_appearing_and_disappearing_is_seen(data_root):
    assert _mirror() == {}
    path = _write_record(data_root, "sandbox")
    assert _mirror().get("live") == "sandbox"
    path.unlink()
    assert _mirror() == {}


def test_a_moved_root_is_seen(data_root, tmp_path, monkeypatch):
    _write_record(data_root, "sandbox")
    assert _mirror().get("live") == "sandbox"
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(other))
    assert _mirror() == {}


def test_invalidate_cache_forgets_the_records_cache(data_root, monkeypatch):
    """One cache, held by the record reader; this module's hook drops that one."""
    _write_record(data_root, {"live": "sandbox"})
    assert _mirror().get("live") == "sandbox"

    dropped: list[bool] = []
    monkeypatch.setattr(control_context, "invalidate_cache", lambda: dropped.append(True))
    posture_store.invalidate_cache()
    assert dropped == [True]


# --- the read-only tree (a lane's queueserver, the dispatch worker) --------
#
# A container that holds no chip of its own reads ONE owner's record out of a
# read-only bind of the whole tree. Its reader is reason-returning and fails
# CLOSED, which is the whole difference from the host record reader: an
# unprovisioned bind, a group the container could not join or a record written
# 0600 must not be answered as the owner's switch. Only an absent record is an
# answer — every switch off — because that is what it means.


def _provision_tree(root: Path) -> Path:
    """A marked, readable tree at ``<root>/control_target`` — what the build leaves."""
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _write_tree_record(tree: Path, owner: str, switched_on=(), *, target: str = "live") -> Path:
    """One owner's record inside the tree, with exactly *switched_on* turned on."""
    path = tree / owner / control_context.RECORD_FILENAME
    record = control_context.ControlContext(
        target=target,
        generation=3,
        switch=dict.fromkeys(switched_on, "asked"),
        switch_epoch=1,
    )
    write_payload(path, record.to_payload())
    posture_store.invalidate_cache()
    return path


def _read_tree(tree: Path, owner: str):
    """``(record, unavailable)`` — the tree reader's answer for *owner*."""
    read = posture_store._read_tree_context(tree, owner)
    return read.record, read.unavailable


def test_the_tree_marker_filename_agrees_with_the_build():
    """The reader restates the build's filename; a drift here fails the tree open."""
    from osprey.deployment.compose_generator import CONTROL_TREE_MARKER_NAME

    assert posture_store.CONTROL_TREE_MARKER_NAME == CONTROL_TREE_MARKER_NAME
    assert "CONTROL_TREE_MARKER_NAME" in posture_store.__all__


def test_a_record_in_the_tree_is_its_own_owners(tmp_path):
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", LIVE_OFF)
    _write_tree_record(tree, "bob", ALL_ON)

    alice, unavailable = _read_tree(tree, "alice")
    assert unavailable is None
    assert alice.switch == dict.fromkeys(LIVE_OFF, "asked")
    # Bob's own record arms every machine, and Alice's never reaches him.
    bob, unavailable = _read_tree(tree, "bob")
    assert unavailable is None
    assert bob.switch == dict.fromkeys(ALL_ON, "asked")


def test_no_record_in_the_tree_for_this_owner_is_an_answer(tmp_path):
    """ENOENT is the one failure that is not unavailable: no record, every switch off."""
    tree = _provision_tree(tmp_path)
    record, unavailable = _read_tree(tree, "carol")
    assert record is None
    assert unavailable is None


def test_a_record_in_the_tree_pointed_at_another_target_still_answers_per_machine(tmp_path):
    """The switch is per machine, whatever target the record points at."""
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", ("live", "standin"), target="va")

    record, unavailable = _read_tree(tree, "alice")
    assert unavailable is None
    assert record is not None
    assert record.switch == {"live": "asked", "standin": "asked"}
    assert record.switch.get("va") != "asked"


def test_an_unmarked_tree_is_unavailable(tmp_path):
    """A bind whose source the runtime auto-created empty must not fail open."""
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir()
    _write_tree_record(tree, "alice", ALL_ON)

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert str(tree) in unavailable
    assert "not provisioned" in unavailable
    assert "osprey build" in unavailable


def test_a_missing_tree_is_unavailable(tmp_path):
    tree = tmp_path / posture_store.STATE_DIR_NAME

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert "not provisioned" in unavailable
    assert "osprey build" in unavailable


@pytest.mark.skipif(os.geteuid() == 0, reason="root traverses a directory at mode 0000")
def test_an_unreadable_tree_names_the_group_join(tmp_path):
    """The remedy is the container's, not the deployment's: it names the join."""
    tree = _provision_tree(tmp_path)
    tree.chmod(0o000)
    try:
        record, unavailable = _read_tree(tree, "alice")
    finally:
        tree.chmod(0o2770)

    assert record is None
    assert unavailable is not None
    assert str(tree) in unavailable
    assert "not readable by this container" in unavailable
    assert "group join" in unavailable
    assert "Docker Desktop" in unavailable


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file at mode 0000")
def test_an_unreadable_record_in_the_tree_is_unavailable_naming_the_file(tmp_path):
    """EACCES on a PRESENT record is not "every switch off".

    The mode is the real one a 0600 record leaves a reader running as another
    uid, rather than a patched reader: the open this reader does is the seam
    that has to refuse, and it is not the one a ``read_text`` patch would cover.
    """
    tree = _provision_tree(tmp_path)
    path = _write_tree_record(tree, "alice", LIVE_OFF)
    path.chmod(0o000)
    try:
        record, unavailable = _read_tree(tree, "alice")
    finally:
        path.chmod(0o640)

    assert record is None
    assert unavailable is not None
    assert str(path) in unavailable
    assert "check the file's mode and group" in unavailable


def test_a_directory_where_the_tree_record_belongs_is_unavailable(tmp_path):
    """EISDIR is a failure on a present path, not an absent record."""
    tree = _provision_tree(tmp_path)
    (tree / "alice" / control_context.RECORD_FILENAME).mkdir(parents=True)

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert "check the file's mode and group" in unavailable


def test_a_corrupt_or_foreign_record_in_the_tree_is_unavailable(tmp_path):
    """Where the host record reader answers "no record", this one answers unavailable."""
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
        record, unavailable = _read_tree(tree, "alice")
        assert record is None, payload
        assert unavailable is not None, payload
        assert str(path) in unavailable, payload


def test_an_undecodable_record_in_the_tree_is_unavailable(tmp_path):
    tree = _provision_tree(tmp_path)
    path = tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe{\x00k\x00: sandbox}")
    posture_store.invalidate_cache()

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert str(path) in unavailable


def test_an_owner_the_tree_cannot_hold_as_a_directory_is_unavailable(tmp_path):
    """A name that is not one path segment cannot address a record — and must not permit."""
    tree = _provision_tree(tmp_path)

    for bogus in ("", "   ", ".", "..", "../alice", "al/ice", "al\\ice", "ali\x00ce"):
        record, unavailable = _read_tree(tree, bogus)
        assert record is None, bogus
        assert unavailable is not None, bogus


def test_the_tree_reader_does_not_route_through_the_host_readers(tmp_path, monkeypatch):
    """No host reader: the host rung is not the tree rung, whatever either answers."""
    tree = _provision_tree(tmp_path)
    _write_tree_record(tree, "alice", LIVE_OFF)

    def _boom(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the tree reader went through a host reader")

    monkeypatch.setattr(control_context, "read_record", _boom)
    monkeypatch.setattr(control_context, "read_record_status", _boom)
    assert _read_tree(tree, "alice")[0].switch == dict.fromkeys(LIVE_OFF, "asked")


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
    planted = _write_tree_record(_provision_tree(tmp_path / "elsewhere"), "alice", ALL_ON)
    link = tree / "alice" / control_context.RECORD_FILENAME
    link.parent.mkdir(parents=True)
    link.symlink_to(planted)

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert str(link) in unavailable
    assert "symbolic link" in unavailable


def test_a_symlinked_owner_directory_in_the_tree_is_unavailable(tmp_path):
    """``O_NOFOLLOW`` guards the record's own name; the directory above it is checked too."""
    tree = _provision_tree(tmp_path)
    elsewhere = _provision_tree(tmp_path / "elsewhere")
    _write_tree_record(elsewhere, "alice", ALL_ON)
    (tree / "alice").symlink_to(elsewhere / "alice", target_is_directory=True)

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
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

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert str(marker) in unavailable
    assert "symbolic link" in unavailable


def test_a_tree_marker_that_is_not_a_file_is_unavailable(tmp_path):
    tree = tmp_path / posture_store.STATE_DIR_NAME
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).mkdir(parents=True)

    record, unavailable = _read_tree(tree, "alice")
    assert record is None
    assert unavailable is not None
    assert "not provisioned" in unavailable


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


def test_an_oversized_record_in_the_tree_is_unavailable(bound_tree):
    """The read is bounded: a record is a few hundred bytes, and the tree is shared.

    The file is re-read on every gated write, so an unbounded read would let the
    account whose plan is running choose how much the container allocates per
    write. Over-long is neither a switch nor an absence — it is a record that
    could not be read.
    """
    path = bound_tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    posture_store.invalidate_cache()

    record, unavailable = _read_tree(bound_tree, "alice")
    assert record is None
    assert unavailable is not None
    assert str(path) in unavailable
    # Its own remedy, not the unreadable record's: this file's mode and group
    # permitted the read, so an operator sent to check them finds both correct.
    assert "larger than" in unavailable
    assert "mode and group" not in unavailable


def test_a_record_of_ordinary_size_is_still_read(bound_tree):
    """The bound is headroom, not a limit a real record approaches."""
    _write_tree_record(bound_tree, "alice", LIVE_OFF)

    record, unavailable = _read_tree(bound_tree, "alice")
    assert unavailable is None
    assert record is not None
    assert record.switch == dict.fromkeys(LIVE_OFF, "asked")
