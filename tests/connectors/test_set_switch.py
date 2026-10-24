"""The person's switch gesture: :func:`osprey_connectors.control_context.set_switch`.

Rule 3 of :mod:`osprey_connectors.control_context`: the person's gesture is the
only write that moves ``switch`` and ``switch_epoch``, and it reads, compares and
writes in one hold of the record lock. A turn-on names the epoch the person saw
and is refused with :class:`~osprey_connectors.control_context.StoppedSince` when
a stop, or a lost record, came after it; a turn-off never refuses; turning every
machine off clears every key and moves the epoch forward.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store, refusals
from osprey_connectors.control_context import StoppedSince, set_switch
from osprey_connectors_tests._harness._control_context_fixtures import (
    data_root,
    race_record_writers,
    write_control_context,
    write_payload,
)

__all__ = ["data_root"]


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _epoch(path: Path) -> int:
    epoch = _payload(path)["switch_epoch"]
    assert isinstance(epoch, int) and epoch > 0
    return epoch


# --- per target ----------------------------------------------------------------


@pytest.mark.parametrize("target", ["live", "va", "standin"])
def test_a_turn_on_arms_one_target_and_keeps_the_epoch(data_root, target):
    path = write_control_context(data_root)
    epoch = _epoch(path)

    stored = set_switch([target], "on", epoch=epoch)

    assert stored is not None and stored.switch == {target: "on"}
    assert stored.switch_epoch == epoch
    assert _payload(path)["switch"] == {target: "on"}
    assert _payload(path)["switch_epoch"] == epoch


@pytest.mark.parametrize("target", ["live", "va", "standin"])
def test_a_turn_off_disarms_one_target_and_leaves_the_others(data_root, target):
    path = write_control_context(data_root, switch={"live": "on", "va": "on", "standin": "on"})
    epoch = _epoch(path)

    stored = set_switch([target], "off")

    expected = {t: "on" for t in ("live", "va", "standin") if t != target}
    assert stored is not None and stored.switch == expected
    assert _payload(path)["switch"] == expected
    assert _payload(path)["switch_epoch"] == epoch


def test_a_turn_on_adds_to_what_is_already_on(data_root):
    path = write_control_context(data_root, switch={"live": "on"})
    set_switch(["va", "standin"], "on", epoch=_epoch(path))
    assert _payload(path)["switch"] == {"live": "on", "va": "on", "standin": "on"}


def test_a_turn_off_of_a_target_that_is_off_is_a_no_op(data_root):
    path = write_control_context(data_root, switch={"live": "on"})
    epoch = _epoch(path)
    set_switch(["va"], "off")
    assert _payload(path)["switch"] == {"live": "on"}
    assert _payload(path)["switch_epoch"] == epoch


def test_a_turn_on_with_a_newer_epoch_than_the_record_is_accepted(data_root):
    path = write_control_context(data_root)
    epoch = _epoch(path)
    set_switch(["va"], "on", epoch=epoch + 10)
    assert _payload(path)["switch"] == {"va": "on"}
    assert _payload(path)["switch_epoch"] == epoch


def test_the_gesture_keeps_every_other_field(data_root):
    path = write_control_context(data_root, target="va", generation=7, posture={"live": "sandbox"})
    before = _payload(path)
    set_switch(["live"], "on", epoch=before["switch_epoch"])
    after = _payload(path)
    assert {k: v for k, v in after.items() if k != "switch"} == {
        k: v for k, v in before.items() if k != "switch"
    }


@pytest.mark.usefixtures("data_root")
def test_the_gesture_follows_an_explicit_path(tmp_path):
    elsewhere = tmp_path / "elsewhere" / "record.json"
    control_context.write_record(
        control_context.ControlContext(target="va", generation=0), path=elsewhere
    )
    set_switch(["va"], "on", epoch=_epoch(elsewhere), path=elsewhere)
    assert _payload(elsewhere)["switch"] == {"va": "on"}
    assert not control_context.record_path().exists()


def test_a_read_after_the_gesture_sees_it(data_root):
    path = write_control_context(data_root)
    assert control_context.read_record().switch == {}
    set_switch(["live"], "on", epoch=_epoch(path))
    assert control_context.read_record().switch == {"live": "on"}


# --- every machine off -----------------------------------------------------------


def test_all_off_clears_every_key_and_bumps_the_epoch(data_root):
    path = write_control_context(data_root, switch={"live": "on", "va": "on"})
    payload = _payload(path)
    # A key outside CONTROL_TARGETS survives parse; all-off must clear it too.
    payload["switch"]["retired-machine"] = "on"
    write_payload(path, payload)
    old_epoch = payload["switch_epoch"]
    assert "retired-machine" in control_context.read_record().switch

    before = time.time_ns()
    stored = set_switch("all", "off")

    assert stored is not None and stored.switch == {}
    assert _payload(path)["switch"] == {}
    new_epoch = _payload(path)["switch_epoch"]
    assert new_epoch > old_epoch and new_epoch >= before
    assert stored.switch_epoch == new_epoch


def test_all_off_moves_the_epoch_forward_even_when_the_clock_does_not(data_root, monkeypatch):
    path = write_control_context(data_root, switch={"live": "on"})
    old_epoch = _epoch(path)
    monkeypatch.setattr(control_context.time, "time_ns", lambda: old_epoch - 1000)
    set_switch("all", "off")
    assert _payload(path)["switch_epoch"] == old_epoch + 1


def test_a_turn_on_begun_before_all_off_is_refused(data_root):
    path = write_control_context(data_root)
    seen = _epoch(path)
    set_switch("all", "off")
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "on", epoch=seen)
    assert caught.value.epoch == _epoch(path) > seen
    assert _payload(path)["switch"] == {}


def test_all_off_on_an_empty_switch_still_bumps_the_epoch(data_root):
    path = write_control_context(data_root)
    old_epoch = _epoch(path)
    set_switch("all", "off")
    assert _payload(path)["switch"] == {}
    assert _payload(path)["switch_epoch"] > old_epoch


# --- refusals ----------------------------------------------------------------------


def test_a_turn_on_with_an_older_epoch_is_refused_and_writes_nothing(data_root):
    path = write_control_context(data_root, switch={"live": "on"})
    epoch = _epoch(path)
    before = path.read_bytes()

    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "on", epoch=epoch - 1)

    assert caught.value.epoch == epoch
    assert caught.value.reason == "stopped_since"
    assert str(caught.value) == refusals.refusal_phrase("stopped_since")
    assert path.read_bytes() == before


def _write_epoch_less(path: Path, switch: dict | None = None) -> None:
    payload = {"schema": 1, "target": "live", "generation": 1}
    if switch is not None:
        payload["switch"] = switch
    write_payload(path, payload)


@pytest.mark.parametrize("offered", [0, 1, 2**62], ids=["zero", "one", "huge"])
@pytest.mark.usefixtures("data_root")
def test_a_missing_epoch_never_compares_as_older(offered):
    path = control_context.record_path()
    _write_epoch_less(path)
    with pytest.raises(StoppedSince):
        set_switch(["live"], "on", epoch=offered)
    assert _payload(path)["switch"] == {}


@pytest.mark.usefixtures("data_root")
def test_a_turn_on_against_an_epoch_less_record_stamps_one_and_the_retry_succeeds():
    path = control_context.record_path()
    _write_epoch_less(path, switch={"va": "on"})
    before = time.time_ns()

    with pytest.raises(StoppedSince) as caught:
        set_switch(["live"], "on", epoch=before)

    stamped = _payload(path)["switch_epoch"]
    assert isinstance(stamped, int) and stamped >= before and stamped != 0
    assert caught.value.epoch == stamped
    # The refused gesture armed nothing; what was on stays on.
    assert _payload(path)["switch"] == {"va": "on"}

    stored = set_switch(["live"], "on", epoch=caught.value.epoch)
    assert stored is not None and stored.switch == {"va": "on", "live": "on"}
    assert _payload(path)["switch_epoch"] == stamped


@pytest.mark.usefixtures("data_root")
def test_a_turn_off_on_an_epoch_less_record_stamps_one():
    path = control_context.record_path()
    _write_epoch_less(path, switch={"va": "on", "live": "on"})
    before = time.time_ns()
    stored = set_switch(["va"], "off")
    assert stored is not None and stored.switch == {"live": "on"}
    stamped = _payload(path)["switch_epoch"]
    assert isinstance(stamped, int) and stamped >= before


# --- argument errors -------------------------------------------------------------------


def test_a_turn_on_without_an_epoch_is_a_value_error(data_root):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="epoch"):
        set_switch(["va"], "on")
    assert path.read_bytes() == before


def test_turning_every_machine_on_is_a_value_error(data_root):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="all"):
        set_switch("all", "on", epoch=_epoch(path))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("targets", "state"),
    [
        (["va"], "maybe"),
        ("va", "on"),  # a bare name, not a sequence of names
        ([], "off"),
        (["nowhere"], "on"),
        ([7], "off"),
    ],
    ids=["unknown-state", "bare-string", "empty", "unknown-target-on", "non-string-target"],
)
def test_malformed_arguments_are_value_errors(data_root, targets, state):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        set_switch(targets, state, epoch=_epoch(path))
    assert path.read_bytes() == before


def test_a_turn_off_of_an_unknown_key_clears_it(data_root):
    path = write_control_context(data_root)
    payload = _payload(path)
    payload["switch"] = {"retired-machine": "on", "live": "on"}
    write_payload(path, payload)
    set_switch(["retired-machine"], "off")
    assert _payload(path)["switch"] == {"live": "on"}


def test_no_agent_data_root_is_a_runtime_error(monkeypatch):
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(control_context, "record_path", lambda: None)
    with pytest.raises(RuntimeError):
        set_switch("all", "off")


# --- missing and lost records --------------------------------------------------------------


@pytest.mark.usefixtures("data_root")
@pytest.mark.parametrize(
    ("targets", "state"), [(["va"], "off"), ("all", "off")], ids=["off", "all"]
)
def test_off_on_a_missing_record_writes_nothing_and_returns_none(targets, state):
    path = control_context.record_path()
    assert set_switch(targets, state) is None
    assert not path.exists()


@pytest.mark.usefixtures("data_root")
def test_on_on_a_missing_record_is_refused():
    path = control_context.record_path()
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "on", epoch=time.time_ns())
    assert caught.value.epoch is None
    assert not path.exists()


@pytest.mark.parametrize("damage", ["corrupt", "foreign_schema"])
def test_off_on_an_unreadable_record_writes_nothing(data_root, damage):
    path = write_control_context(data_root, switch={"live": "on"})
    _damage(path, damage)
    before = path.read_bytes()
    assert set_switch("all", "off") is None
    assert set_switch(["live"], "off") is None
    assert path.read_bytes() == before
    with pytest.raises(StoppedSince):
        set_switch(["live"], "on", epoch=time.time_ns())
    assert path.read_bytes() == before


def _damage(path: Path, damage: str) -> None:
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{not json", encoding="utf-8")
    else:
        path.write_text(json.dumps({**_payload(path), "schema": 2}), encoding="utf-8")
    control_context.invalidate_cache()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "foreign_schema"])
def test_a_turn_on_begun_before_the_record_was_lost_is_refused(data_root, damage):
    path = write_control_context(data_root, switch={"live": "on"})
    seen = _epoch(path)
    _damage(path, damage)
    # A non-person writer rewrites the record; it stamps a fresh epoch.
    control_context.write_record(control_context.ControlContext(target="live", generation=2))
    assert _epoch(path) > seen

    with pytest.raises(StoppedSince) as caught:
        set_switch(["live"], "on", epoch=seen)
    assert _payload(path)["switch"] == {}
    # The gesture retried with the new epoch goes through.
    set_switch(["live"], "on", epoch=caught.value.epoch)
    assert _payload(path)["switch"] == {"live": "on"}


# --- the lock ---------------------------------------------------------------------------------


def test_the_gesture_waits_for_the_lock_and_refuses_after_a_bounded_wait(data_root, monkeypatch):
    import fcntl

    path = write_control_context(data_root, switch={"live": "on"})
    before = path.read_bytes()
    monkeypatch.setattr(control_context, "_LOCK_TIMEOUT_S", 0.2)
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(RuntimeError, match="locked"):
            set_switch("all", "off")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert path.read_bytes() == before


def test_the_gesture_writes_through_the_shared_private_writer(data_root, monkeypatch):
    path = write_control_context(data_root)
    calls: list[Path] = []
    original = control_context._write_record_unlocked

    def spy(record, target_path):
        calls.append(target_path)
        original(record, target_path)

    monkeypatch.setattr(control_context, "_write_record_unlocked", spy)
    set_switch(["va"], "on", epoch=_epoch(path))
    assert calls == [path]


# --- two real processes ------------------------------------------------------------------------

#: One child process's body, run after the shared race prelude. ``stop`` is the
#: person: wait for a signal, then call the public ``set_switch("all", "off")``
#: with the private writer slowed down so the lock is held across the other
#: writer's attempt; signal ``locked`` from inside the hold and ``done`` after.
#: ``write`` is every other writer: read the record (a view the stop is about to
#: change), signal ``read``, wait, write through the public ``write_record``.
_CHILD = """
    if mode == "stop":
        original = cc._write_record_unlocked

        def slow(record, path):
            mark("locked")
            time.sleep(hold)
            original(record, path)

        cc._write_record_unlocked = slow
        wait(wait_for)
        stored = cc.set_switch("all", "off")
        mark("done")
        print(json.dumps({"epoch": stored.switch_epoch}))
    else:
        seen = cc.read_record()
        assert seen.switch, "the other writer must start from a view with machines on"
        mark("read")
        wait(wait_for)
        me = cc.Owner(kind=cc.OWNER_CONTROLS_SERVER, pid=os.getpid())
        cc.write_record(replace(seen, owner=me, generation=seen.generation + 1))
        print(json.dumps({"confirmed": True}))
"""


@pytest.mark.parametrize("interleaved", [True, False], ids=["interleaved", "after"])
def test_all_off_racing_a_non_person_write_leaves_the_switch_empty(
    data_root, tmp_path, interleaved
):
    path = write_control_context(data_root, generation=4, switch={"live": "on", "va": "on"})
    old_epoch = _epoch(path)

    stopped, written = race_record_writers(
        _CHILD, data_root, tmp_path, "stop", "write", interleaved
    )
    record = control_context.read_record()
    assert record.switch == {}
    assert record.switch_epoch == stopped["epoch"] > old_epoch
    # The non-person write landed too, after the stop, without undoing it.
    assert written["confirmed"] is True
    assert record.generation == 5


# --- the fixture ---------------------------------------------------------------------------------


def test_the_fixture_switch_goes_through_the_gesture(data_root, monkeypatch):
    calls: list[tuple] = []
    original = control_context.set_switch

    def spy(targets, state, **kwargs):
        calls.append((list(targets) if targets != "all" else targets, state))
        return original(targets, state, **kwargs)

    monkeypatch.setattr(control_context, "set_switch", spy)
    path = write_control_context(data_root, switch={"va": "on"})
    assert ("va" in calls[-1][0]) and calls[-1][1] == "on"
    epoch = _epoch(path)
    # A rewrite that names a different switch turns off what it no longer names.
    write_control_context(data_root, switch={"live": "on"})
    assert _payload(path)["switch"] == {"live": "on"}
    assert _payload(path)["switch_epoch"] == epoch


def test_stopped_since_carries_its_epoch_and_reason():
    """A route answers a stale gesture from attributes, never by parsing text."""
    error = StoppedSince(epoch=42)
    assert error.epoch == 42 and error.reason == "stopped_since"
    # Not a RuntimeError: "store unavailable" handlers must not swallow it.
    assert not isinstance(error, RuntimeError)
