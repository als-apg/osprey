"""The control-context record's lock, and what a write keeps under it.

Rule 3 of :mod:`osprey_connectors.control_context`: the record is written only
under an exclusive ``flock`` on ``<record path>.lock``, and a write that is not
the person's own switch gesture keeps the ``switch`` and ``switch_epoch`` the
file on disk carries. The cross-process tests here run real interpreters, because
the race they pin is between processes: a stop racing a refusal terminus must not
come out with the machine switched back on, and a target write racing a turn-on
must not switch it off again.

The stop and the turn-on are spelled here with the module's private lock and
writer, the two halves the person route is built from; the non-person writers
are the public ``write_terminus`` and ``write_record``.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from pathlib import Path

import pytest

from osprey_connectors import control_context
from osprey_connectors_tests._harness import (
    data_root,
    race_record_writers,
    write_control_context,
)

__all__ = ["data_root"]


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- the lock file -----------------------------------------------------------


def test_the_lock_sits_beside_the_record_it_guards(data_root):
    path = write_control_context(data_root)
    assert path.with_name(path.name + ".lock").is_file()


@pytest.mark.usefixtures("data_root")
def test_the_lock_follows_an_explicit_path(tmp_path):
    elsewhere = tmp_path / "elsewhere" / "record.json"
    control_context.write_record(
        control_context.ControlContext(target="va", generation=0), path=elsewhere
    )
    assert elsewhere.is_file()
    assert (tmp_path / "elsewhere" / "record.json.lock").is_file()
    assert not control_context.state_dir().exists()


def test_a_symlinked_lock_refuses_the_write_and_leaves_its_target_alone(data_root):
    """A link planted at ``<record>.lock`` is not followed: the write fails closed."""
    path = write_control_context(data_root, generation=3)
    before = path.read_bytes()
    lock = control_context._lock_path(path)
    lock.unlink()
    target = data_root / "planted-target"
    target.write_text("", encoding="utf-8")
    target.chmod(0o600)
    lock.symlink_to(target)

    with pytest.raises(OSError) as exc:
        control_context.write_record(
            control_context.ControlContext(target="va", generation=4), path=path
        )
    assert exc.value.errno == errno.ELOOP

    assert (target.stat().st_mode & 0o777) == 0o600
    assert lock.is_symlink()
    assert path.read_bytes() == before


def test_a_dangling_symlinked_lock_creates_nothing_at_its_target(data_root, tmp_path):
    path = write_control_context(data_root, generation=3)
    lock = control_context._lock_path(path)
    lock.unlink()
    target = tmp_path / "nowhere"
    lock.symlink_to(target)

    with pytest.raises(OSError) as exc:
        control_context.write_record(
            control_context.ControlContext(target="va", generation=4), path=path
        )
    assert exc.value.errno == errno.ELOOP

    assert not target.exists()


def test_an_existing_regular_lock_is_reused_and_not_chmodded(data_root):
    path = write_control_context(data_root, generation=3)
    lock = control_context._lock_path(path)
    lock.chmod(0o600)
    inode = lock.stat().st_ino

    control_context.write_record(
        control_context.ControlContext(target="va", generation=4), path=path
    )

    assert lock.stat().st_ino == inode
    assert (lock.stat().st_mode & 0o777) == 0o600
    assert _payload(path)["generation"] == 4


@pytest.mark.usefixtures("data_root")
def test_a_fresh_lock_is_created_at_0o660_under_a_umask(tmp_path):
    record = tmp_path / "fresh" / "record.json"
    previous = os.umask(0o077)
    try:
        control_context.write_record(
            control_context.ControlContext(target="va", generation=0), path=record
        )
    finally:
        os.umask(previous)
    assert (control_context._lock_path(record).stat().st_mode & 0o777) == 0o660


def _hold(lock_file: Path):
    """Take the lock from a second open file description, as another writer would."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o660)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def test_a_held_lock_refuses_the_write_after_a_bounded_wait(data_root, monkeypatch):
    path = write_control_context(data_root, generation=3)
    before = path.read_bytes()
    monkeypatch.setattr(control_context, "_LOCK_TIMEOUT_S", 0.2)
    fd = _hold(path.with_name(path.name + ".lock"))
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="stayed locked"):
            control_context.write_record(
                control_context.ControlContext(target="va", generation=4), path=path
            )
        assert time.monotonic() - started >= 0.2
    finally:
        _release(fd)
    assert path.read_bytes() == before


def test_a_held_lock_makes_a_terminus_report_that_it_did_not_stick(data_root, monkeypatch):
    path = write_control_context(data_root)
    monkeypatch.setattr(control_context, "_LOCK_TIMEOUT_S", 0.1)
    record = control_context.read_record()
    updated = control_context.terminus(
        record,
        request_id="r-locked",
        target="va",
        requested_at=0,
        requested_by="pid:1",
        status=control_context.SWITCH_REFUSED,
        reason="test",
        detail="refused",
        generation=None,
    )
    fd = _hold(path.with_name(path.name + ".lock"))
    try:
        assert control_context.write_terminus(updated, "r-locked") is False
    finally:
        _release(fd)


@pytest.mark.usefixtures("data_root")
def test_the_lock_is_keyed_on_the_path_written(tmp_path, monkeypatch):
    """A lock held on one record does not stall a write to another."""
    monkeypatch.setattr(control_context, "_LOCK_TIMEOUT_S", 0.1)
    elsewhere = tmp_path / "elsewhere" / "record.json"
    fd = _hold(elsewhere.with_name(elsewhere.name + ".lock"))
    try:
        with pytest.raises(RuntimeError):
            control_context.write_record(
                control_context.ControlContext(target="va", generation=0), path=elsewhere
            )
        control_context.write_record(control_context.ControlContext(target="va", generation=0))
    finally:
        _release(fd)
    assert control_context.record_path().is_file()


def test_the_lock_is_released_after_every_write(data_root):
    path = write_control_context(data_root)
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
    finally:
        _release(fd)


# --- what a write keeps --------------------------------------------------------


def test_a_write_keeps_the_switch_and_epoch_on_disk(data_root):
    path = write_control_context(data_root, switch={"live": "asked"})
    epoch = _payload(path)["switch_epoch"]
    stale = control_context.ControlContext(
        target="va", generation=2, switch={"standin": "asked"}, switch_epoch=1
    )
    control_context.write_record(stale)
    stored = control_context.read_record()
    assert stored.target == "va" and stored.generation == 2
    assert stored.switch == {"live": "asked"}
    assert stored.switch_epoch == epoch


@pytest.mark.usefixtures("data_root")
def test_a_write_over_an_epoch_less_record_keeps_it_epoch_less():
    """A non-person write never stamps an epoch onto a readable record."""
    from osprey_connectors_tests._harness import write_payload

    path = control_context.record_path()
    write_payload(
        path, {"schema": 1, "target": "live", "generation": 1, "switch": {"live": "asked"}}
    )
    control_context.write_record(control_context.ControlContext(target="live", generation=2))
    assert _payload(path)["switch_epoch"] is None
    assert _payload(path)["switch"] == {"live": "asked"}


@pytest.mark.parametrize("damage", ["missing", "corrupt", "foreign_schema"])
def test_a_write_over_a_lost_record_writes_the_switch_empty_with_a_fresh_epoch(data_root, damage):
    path = write_control_context(data_root, switch={"live": "asked"})
    old_epoch = _payload(path)["switch_epoch"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{not json", encoding="utf-8")
    else:
        path.write_text(json.dumps({**_payload(path), "schema": 2}), encoding="utf-8")
    control_context.invalidate_cache()
    before = time.time_ns()
    control_context.write_record(
        control_context.ControlContext(target="live", generation=1, switch={"live": "asked"})
    )
    payload = _payload(path)
    assert payload["switch"] == {}
    assert payload["switch_epoch"] >= before > old_epoch
    assert payload["switch_epoch"] != 0


# --- two real processes ----------------------------------------------------------

#: One child process's body, run after the shared race prelude. ``stop`` /
#: ``turn_on`` play the person: wait for a signal, take the lock, signal
#: ``locked``, hold it for a while, write, signal ``done``.
#: ``terminus`` / ``write`` play every other writer: read the record (so what
#: they write is built from a view the person is about to change), signal
#: ``read``, wait for a signal, write through the public API.
_CHILD = """
    path = cc.record_path()
    me = cc.Owner(kind=cc.OWNER_CONTROLS_SERVER, pid=os.getpid())
    if mode in ("stop", "turn_on"):
        wait(wait_for)
        with cc._record_lock(path):
            current, _status = cc._read(path)
            if mode == "stop":
                new = replace(
                    current,
                    switch={},
                    switch_epoch=max(current.switch_epoch + 1, time.time_ns()),
                )
            else:
                new = replace(current, switch={**current.switch, "va": "asked"})
            mark("locked")
            time.sleep(hold)
            cc._write_record_unlocked(new, path)
        mark("done")
        print(json.dumps({"epoch": new.switch_epoch}))
    else:
        seen = cc.read_record()
        mark("read")
        wait(wait_for)
        if mode == "terminus":
            updated = cc.terminus(
                replace(seen, owner=me),
                request_id="req-1",
                target=seen.target,
                requested_at=0,
                requested_by="pid:1",
                status=cc.SWITCH_REFUSED,
                reason="test",
                detail="refused",
                generation=None,
            )
            print(json.dumps({"confirmed": cc.write_terminus(updated, "req-1")}))
        else:
            cc.write_record(replace(seen, owner=me, generation=seen.generation + 1))
            print(json.dumps({"confirmed": True}))
"""


@pytest.mark.parametrize("interleaved", [True, False], ids=["interleaved", "after"])
def test_a_stop_and_a_terminus_leave_the_switch_empty(data_root, tmp_path, interleaved):
    path = write_control_context(data_root, switch={"live": "asked", "va": "asked"})
    old_epoch = _payload(path)["switch_epoch"]

    stopped, answered = race_record_writers(
        _CHILD, data_root, tmp_path, "stop", "terminus", interleaved
    )

    record = control_context.read_record()
    assert record.switch == {}
    assert record.switch_epoch == stopped["epoch"] > old_epoch
    # The terminus was not lost either: it landed, after the stop, and stuck.
    assert answered["confirmed"] is True
    assert record.last_switch["request_id"] == "req-1"


@pytest.mark.parametrize("interleaved", [True, False], ids=["interleaved", "after"])
def test_a_non_person_write_keeps_a_turn_on_and_its_epoch(data_root, tmp_path, interleaved):
    path = write_control_context(data_root, generation=4)
    epoch = _payload(path)["switch_epoch"]

    turned_on, _written = race_record_writers(
        _CHILD, data_root, tmp_path, "turn_on", "write", interleaved
    )

    record = control_context.read_record()
    assert record.switch == {"va": "asked"}
    assert record.switch_epoch == epoch == turned_on["epoch"]
    # And the non-person write itself landed, after the turn-on.
    assert record.generation == 5
