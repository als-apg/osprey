"""The person's switch gesture: :func:`osprey_connectors.control_context.set_switch`.

Rule 3 of :mod:`osprey_connectors.control_context`: the person's gesture is the
only write that moves ``switch`` and ``switch_epoch``, and it reads, compares and
writes in one hold of the record lock. A machine stands at off, asked or allowed.
A widening names the epoch the person saw and is refused with
:class:`~osprey_connectors.control_context.StoppedSince` when a stop, or a lost
record, came after it; a narrowing or a no-op never refuses; turning every machine
off clears every key and moves the epoch forward. What is widening is decided
inside the lock against the map on disk, not by what the person thought they saw.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store, refusals
from osprey_connectors.control_context import StoppedSince, set_switch, switch_position
from osprey_connectors_tests._harness import (
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

    stored = set_switch([target], "asked", epoch=epoch)

    assert stored is not None and stored.switch == {target: "asked"}
    assert stored.switch_epoch == epoch
    assert _payload(path)["switch"] == {target: "asked"}
    assert _payload(path)["switch_epoch"] == epoch


@pytest.mark.parametrize("target", ["live", "va", "standin"])
def test_a_turn_off_disarms_one_target_and_leaves_the_others(data_root, target):
    path = write_control_context(
        data_root, switch={"live": "asked", "va": "asked", "standin": "asked"}
    )
    epoch = _epoch(path)

    stored = set_switch([target], "off")

    expected = {t: "asked" for t in ("live", "va", "standin") if t != target}
    assert stored is not None and stored.switch == expected
    assert _payload(path)["switch"] == expected
    assert _payload(path)["switch_epoch"] == epoch


def test_a_turn_on_adds_to_what_is_already_on(data_root):
    path = write_control_context(data_root, switch={"live": "asked"})
    set_switch(["va", "standin"], "asked", epoch=_epoch(path))
    assert _payload(path)["switch"] == {"live": "asked", "va": "asked", "standin": "asked"}


def test_a_turn_off_of_a_target_that_is_off_is_a_no_op(data_root):
    path = write_control_context(data_root, switch={"live": "asked"})
    epoch = _epoch(path)
    set_switch(["va"], "off")
    assert _payload(path)["switch"] == {"live": "asked"}
    assert _payload(path)["switch_epoch"] == epoch


def test_a_turn_on_with_a_newer_epoch_than_the_record_is_refused_and_writes_nothing(data_root):
    """Equality, not order: an epoch the record never carried was not what the person saw."""
    path = write_control_context(data_root, switch={"live": "asked"})
    epoch = _epoch(path)
    before = path.read_bytes()

    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "asked", epoch=epoch + 10)

    assert caught.value.epoch == epoch
    assert caught.value.stamped is False
    assert path.read_bytes() == before


def test_a_turn_on_with_the_equal_epoch_is_accepted(data_root):
    path = write_control_context(data_root, switch={"live": "asked"})
    epoch = _epoch(path)
    stored = set_switch(["va"], "asked", epoch=epoch)
    assert stored is not None and stored.switch == {"live": "asked", "va": "asked"}
    assert _payload(path)["switch"] == {"live": "asked", "va": "asked"}
    assert _payload(path)["switch_epoch"] == epoch


def test_the_gesture_keeps_every_other_field(data_root):
    """Only ``switch`` and its ``posture`` mirror move; the mirror follows the switch."""
    path = write_control_context(data_root, target="va", generation=7, switch={"va": "asked"})
    before = _payload(path)
    set_switch(["live"], "asked", epoch=before["switch_epoch"])
    after = _payload(path)
    assert after["switch"] == {"va": "asked", "live": "asked"}
    assert after["posture"] == control_context.posture_mirror(after["switch"])
    assert {k: v for k, v in after.items() if k not in ("switch", "posture")} == {
        k: v for k, v in before.items() if k not in ("switch", "posture")
    }


@pytest.mark.usefixtures("data_root")
def test_the_gesture_follows_an_explicit_path(tmp_path):
    elsewhere = tmp_path / "elsewhere" / "record.json"
    control_context.write_record(
        control_context.ControlContext(target="va", generation=0), path=elsewhere
    )
    set_switch(["va"], "asked", epoch=_epoch(elsewhere), path=elsewhere)
    assert _payload(elsewhere)["switch"] == {"va": "asked"}
    assert not control_context.record_path().exists()


def test_a_read_after_the_gesture_sees_it(data_root):
    path = write_control_context(data_root)
    assert control_context.read_record().switch == {}
    set_switch(["live"], "asked", epoch=_epoch(path))
    assert control_context.read_record().switch == {"live": "asked"}


# --- three positions --------------------------------------------------------------


@pytest.mark.parametrize(
    ("before_position", "state"),
    [(None, "asked"), (None, "allowed"), ("asked", "allowed")],
    ids=["off-asked", "off-allowed", "asked-allowed"],
)
class TestEachWidening:
    @staticmethod
    def _record(data_root, before_position):
        switch = {} if before_position is None else {"va": before_position}
        return write_control_context(data_root, switch=switch)

    @pytest.mark.parametrize("offset", [-1, 1], ids=["older", "newer"])
    def test_a_stale_epoch_is_refused(self, data_root, before_position, state, offset):
        path = self._record(data_root, before_position)
        epoch = _epoch(path)
        before = path.read_bytes()
        with pytest.raises(StoppedSince) as caught:
            set_switch(["va"], state, epoch=epoch + offset)
        assert caught.value.epoch == epoch
        assert path.read_bytes() == before

    def test_no_epoch_is_refused(self, data_root, before_position, state):
        path = self._record(data_root, before_position)
        epoch = _epoch(path)
        before = path.read_bytes()
        with pytest.raises(StoppedSince) as caught:
            set_switch(["va"], state)
        assert caught.value.epoch == epoch
        assert path.read_bytes() == before

    def test_the_seen_epoch_succeeds(self, data_root, before_position, state):
        path = self._record(data_root, before_position)
        epoch = _epoch(path)
        stored = set_switch(["va"], state, epoch=epoch)
        assert stored is not None and stored.switch == {"va": state}
        assert switch_position(stored, "va") == state
        assert _payload(path)["switch"] == {"va": state}
        assert _payload(path)["switch_epoch"] == epoch


@pytest.mark.parametrize(
    ("before_position", "state"),
    [
        ("allowed", "asked"),
        ("allowed", "off"),
        ("asked", "off"),
        ("asked", "asked"),
        ("allowed", "allowed"),
    ],
    ids=["allowed-asked", "allowed-off", "asked-off", "asked-asked", "allowed-allowed"],
)
@pytest.mark.parametrize("offered", [None, 1], ids=["no-epoch", "stale-epoch"])
def test_a_narrowing_or_no_op_needs_no_epoch(data_root, before_position, state, offered):
    path = write_control_context(data_root, switch={"va": before_position, "live": "asked"})
    epoch = _epoch(path)
    stored = set_switch(["va"], state, epoch=offered)
    expected = {"live": "asked"} if state == "off" else {"va": state, "live": "asked"}
    assert stored is not None and stored.switch == expected
    assert _payload(path)["switch"] == expected
    assert _payload(path)["switch_epoch"] == epoch


def test_a_move_believed_to_be_a_narrowing_that_raced_all_off_is_refused(data_root):
    """Seen at allowed, moved down to asked: the stop in between makes it a widening."""
    path = write_control_context(data_root, switch={"va": "allowed"})
    seen = _epoch(path)
    set_switch("all", "off")
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "asked")
    assert caught.value.epoch == _epoch(path) > seen
    assert caught.value.stamped is False
    assert _payload(path)["switch"] == {}


def test_one_widening_name_gates_the_whole_call(data_root):
    path = write_control_context(data_root, switch={"live": "allowed"})
    epoch = _epoch(path)
    before = path.read_bytes()
    with pytest.raises(StoppedSince):
        set_switch(["live", "va"], "asked")
    assert path.read_bytes() == before

    stored = set_switch(["live", "va"], "asked", epoch=epoch)
    assert stored is not None and stored.switch == {"live": "asked", "va": "asked"}


def test_a_call_that_narrows_every_name_it_names_needs_no_epoch(data_root):
    write_control_context(data_root, switch={"live": "allowed", "va": "asked"})
    stored = set_switch(["live", "va"], "asked")
    assert stored is not None and stored.switch == {"live": "asked", "va": "asked"}


# --- the reader --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("switch", "position"),
    [
        ({"live": "asked"}, "asked"),
        ({"live": "allowed"}, "allowed"),
        ({"live": "on"}, "asked"),
        ({}, "off"),
        ({"va": "allowed"}, "off"),
        ({"live": "off"}, "off"),
        ({"live": "Allowed"}, "off"),
        ({"live": "asked "}, "off"),
        ({"live": True}, "off"),
        ({"live": 1}, "off"),
        ({"live": None}, "off"),
    ],
    ids=[
        "asked",
        "allowed",
        "legacy-on",
        "absent",
        "another-machine",
        "off-word",
        "capitalised",
        "trailing-space",
        "bool",
        "int",
        "none",
    ],
)
def test_switch_position_reads_a_built_record(switch, position):
    record = control_context.ControlContext(target="live", generation=1, switch=switch)
    assert switch_position(record, "live") == position


def test_switch_position_of_no_record_is_off():
    assert switch_position(None, "live") == "off"


@pytest.mark.parametrize(
    "value", ["yes", "ON", "Asked", "allowed ", "off", True, 1, None, ["asked"], {"a": 1}]
)
def test_an_unknown_stored_value_reads_as_off(data_root, value):
    path = write_control_context(data_root)
    payload = _payload(path)
    payload["switch"] = {"live": value, "va": "allowed"}
    write_payload(path, payload)
    record = control_context.read_record()
    assert record.switch == {"va": "allowed"}
    assert switch_position(record, "live") == "off"


@pytest.mark.parametrize(
    ("switch", "sandboxed"),
    [
        ({}, {"live", "va", "standin"}),
        ({"live": "asked"}, {"va", "standin"}),
        ({"live": "allowed"}, {"va", "standin"}),
        ({"live": "on"}, {"va", "standin"}),
        ({"live": "maybe"}, {"live", "va", "standin"}),
        ({"live": "asked", "va": "allowed", "standin": "on"}, set()),
    ],
    ids=["empty", "asked", "allowed", "legacy-on", "unknown", "every-machine"],
)
def test_the_mirror_refuses_only_the_machines_at_off(switch, sandboxed):
    mirror = control_context.posture_mirror(switch)
    assert mirror == dict.fromkeys(sandboxed, posture_store.POSTURE_SANDBOX)


# --- a map still spelling on -------------------------------------------------------


def _write_legacy(path: Path, switch: dict) -> None:
    payload = _payload(path)
    payload["switch"] = switch
    write_payload(path, payload)


def test_a_map_spelling_on_reads_asked(data_root):
    path = write_control_context(data_root)
    _write_legacy(path, {"live": "on"})
    record = control_context.read_record()
    assert record.switch == {"live": "asked"}
    assert switch_position(record, "live") == "asked"


def test_a_set_rewrites_a_map_spelling_on_in_the_new_words(data_root):
    path = write_control_context(data_root)
    _write_legacy(path, {"live": "on", "va": "on", "standin": "on"})
    set_switch(["standin"], "off")
    assert _payload(path)["switch"] == {"live": "asked", "va": "asked"}


def test_a_set_to_where_a_map_spelling_on_already_is_rewrites_it(data_root):
    path = write_control_context(data_root)
    _write_legacy(path, {"live": "on"})
    set_switch(["live"], "asked")
    assert _payload(path)["switch"] == {"live": "asked"}


def test_on_to_allowed_is_a_widening(data_root):
    path = write_control_context(data_root)
    epoch = _epoch(path)
    _write_legacy(path, {"live": "on"})
    with pytest.raises(StoppedSince):
        set_switch(["live"], "allowed")
    set_switch(["live"], "allowed", epoch=epoch)
    assert _payload(path)["switch"] == {"live": "allowed"}


def test_a_non_person_write_rewrites_a_map_spelling_on_too(data_root):
    path = write_control_context(data_root, target="va")
    epoch = _epoch(path)
    _write_legacy(path, {"live": "on"})
    control_context.write_record(control_context.ControlContext(target="va", generation=2))
    assert _payload(path)["switch"] == {"live": "asked"}
    assert _payload(path)["switch_epoch"] == epoch


# --- every machine off -----------------------------------------------------------


def test_all_off_clears_every_key_and_bumps_the_epoch(data_root):
    path = write_control_context(data_root, switch={"live": "asked", "va": "asked"})
    payload = _payload(path)
    # A key outside CONTROL_TARGETS survives parse; all-off must clear it too.
    payload["switch"]["retired-machine"] = "asked"
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
    path = write_control_context(data_root, switch={"live": "asked"})
    old_epoch = _epoch(path)
    monkeypatch.setattr(control_context.time, "time_ns", lambda: old_epoch - 1000)
    set_switch("all", "off")
    assert _payload(path)["switch_epoch"] == old_epoch + 1


def test_a_turn_on_begun_before_all_off_is_refused(data_root):
    path = write_control_context(data_root)
    seen = _epoch(path)
    set_switch("all", "off")
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "asked", epoch=seen)
    assert caught.value.epoch == _epoch(path) > seen
    # Not a stamp: the chip re-renders and needs a new click, so the turn-on stays off.
    assert caught.value.stamped is False
    assert _payload(path)["switch"] == {}


def test_all_off_on_an_empty_switch_still_bumps_the_epoch(data_root):
    path = write_control_context(data_root)
    old_epoch = _epoch(path)
    set_switch("all", "off")
    assert _payload(path)["switch"] == {}
    assert _payload(path)["switch_epoch"] > old_epoch


# --- refusals ----------------------------------------------------------------------


def test_a_turn_on_with_an_older_epoch_is_refused_and_writes_nothing(data_root):
    path = write_control_context(data_root, switch={"live": "asked"})
    epoch = _epoch(path)
    before = path.read_bytes()

    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "asked", epoch=epoch - 1)

    assert caught.value.epoch == epoch
    assert caught.value.stamped is False
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
        set_switch(["live"], "asked", epoch=offered)
    assert _payload(path)["switch"] == {}


@pytest.mark.usefixtures("data_root")
def test_a_turn_on_against_an_epoch_less_record_stamps_one_and_the_retry_succeeds():
    path = control_context.record_path()
    _write_epoch_less(path, switch={"va": "asked"})
    before = time.time_ns()

    with pytest.raises(StoppedSince) as caught:
        set_switch(["live"], "asked", epoch=before)

    stamped = _payload(path)["switch_epoch"]
    assert isinstance(stamped, int) and stamped >= before and stamped != 0
    assert caught.value.epoch == stamped
    # Only a just-stamped record lets the chip retry on its own.
    assert caught.value.stamped is True
    # The refused gesture armed nothing; what was on stays on.
    assert _payload(path)["switch"] == {"va": "asked"}

    stored = set_switch(["live"], "asked", epoch=caught.value.epoch)
    assert stored is not None and stored.switch == {"va": "asked", "live": "asked"}
    assert _payload(path)["switch_epoch"] == stamped


@pytest.mark.usefixtures("data_root")
def test_a_widening_with_no_epoch_on_an_epoch_less_record_stamps_one():
    path = control_context.record_path()
    _write_epoch_less(path)
    with pytest.raises(StoppedSince) as caught:
        set_switch(["live"], "allowed")
    assert caught.value.stamped is True
    assert caught.value.epoch == _epoch(path)
    assert _payload(path)["switch"] == {}


@pytest.mark.usefixtures("data_root")
def test_a_narrowing_on_an_epoch_less_record_stamps_one():
    path = control_context.record_path()
    _write_epoch_less(path, switch={"va": "allowed"})
    before = time.time_ns()
    stored = set_switch(["va"], "asked")
    assert stored is not None and stored.switch == {"va": "asked"}
    assert _epoch(path) >= before


@pytest.mark.usefixtures("data_root")
def test_a_turn_off_on_an_epoch_less_record_stamps_one():
    path = control_context.record_path()
    _write_epoch_less(path, switch={"va": "asked", "live": "asked"})
    before = time.time_ns()
    stored = set_switch(["va"], "off")
    assert stored is not None and stored.switch == {"live": "asked"}
    stamped = _payload(path)["switch_epoch"]
    assert isinstance(stamped, int) and stamped >= before


# --- argument errors -------------------------------------------------------------------


@pytest.mark.parametrize("state", ["asked", "allowed"])
def test_a_widening_without_an_epoch_is_stopped_since_and_writes_nothing(data_root, state):
    """A missing epoch is judged inside the lock, as a stale one, not as an argument error."""
    path = write_control_context(data_root)
    epoch = _epoch(path)
    before = path.read_bytes()
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], state)
    assert caught.value.epoch == epoch
    assert caught.value.stamped is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("state", ["asked", "allowed"])
def test_turning_every_machine_on_is_a_value_error(data_root, state):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="all"):
        set_switch("all", state, epoch=_epoch(path))
    assert path.read_bytes() == before


def test_the_legacy_gesture_word_on_is_taken_as_asked_and_never_stored(data_root):
    """A framework released before the three positions sends ``on``."""
    path = write_control_context(data_root)
    stored = set_switch(["va"], "on", epoch=_epoch(path))
    assert stored is not None and stored.switch == {"va": "asked"}
    assert _payload(path)["switch"] == {"va": "asked"}


def test_the_legacy_gesture_word_on_widens_like_asked(data_root):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(StoppedSince):
        set_switch(["va"], "on")
    assert path.read_bytes() == before


def test_the_legacy_gesture_word_on_narrows_allowed_to_asked(data_root):
    write_control_context(data_root, switch={"va": "allowed"})
    stored = set_switch(["va"], "on")
    assert stored is not None and stored.switch == {"va": "asked"}


def test_the_legacy_gesture_word_on_cannot_name_all(data_root):
    write_control_context(data_root)
    with pytest.raises(ValueError, match="one at a time"):
        set_switch("all", "on")


@pytest.mark.parametrize("epoch", [True, "1", 1.5], ids=["bool", "str", "float"])
def test_an_epoch_that_is_not_an_integer_is_a_value_error(data_root, epoch):
    path = write_control_context(data_root)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="epoch"):
        set_switch(["va"], "asked", epoch=epoch)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("targets", "state"),
    [
        (["va"], "maybe"),
        ("va", "asked"),  # a bare name, not a sequence of names
        ([], "off"),
        (["nowhere"], "asked"),
        (["nowhere"], "allowed"),
        ([7], "off"),
    ],
    ids=[
        "unknown-state",
        "bare-string",
        "empty",
        "unknown-target-asked",
        "unknown-target-allowed",
        "non-string-target",
    ],
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
    payload["switch"] = {"retired-machine": "asked", "live": "asked"}
    write_payload(path, payload)
    set_switch(["retired-machine"], "off")
    assert _payload(path)["switch"] == {"live": "asked"}


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
        set_switch(["va"], "asked", epoch=time.time_ns())
    assert caught.value.epoch is None
    assert caught.value.stamped is False
    assert not path.exists()


@pytest.mark.usefixtures("data_root")
@pytest.mark.parametrize("state", ["asked", "allowed"])
@pytest.mark.parametrize("offered", [None, 1], ids=["no-epoch", "any-epoch"])
def test_a_widening_on_a_missing_record_is_refused(state, offered):
    path = control_context.record_path()
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], state, epoch=offered)
    assert caught.value.epoch is None
    assert not path.exists()


@pytest.mark.parametrize("damage", ["corrupt", "foreign_schema"])
def test_off_on_an_unreadable_record_writes_nothing(data_root, damage):
    path = write_control_context(data_root, switch={"live": "asked"})
    _damage(path, damage)
    before = path.read_bytes()
    assert set_switch("all", "off") is None
    assert set_switch(["live"], "off") is None
    assert path.read_bytes() == before
    with pytest.raises(StoppedSince):
        set_switch(["live"], "asked", epoch=time.time_ns())
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
    path = write_control_context(data_root, switch={"live": "asked"})
    seen = _epoch(path)
    _damage(path, damage)
    # A non-person writer rewrites the record; it stamps a fresh epoch.
    control_context.write_record(control_context.ControlContext(target="live", generation=2))
    assert _epoch(path) > seen

    with pytest.raises(StoppedSince) as caught:
        set_switch(["live"], "asked", epoch=seen)
    assert _payload(path)["switch"] == {}
    # The gesture retried with the new epoch goes through.
    set_switch(["live"], "asked", epoch=caught.value.epoch)
    assert _payload(path)["switch"] == {"live": "asked"}


# --- the lock ---------------------------------------------------------------------------------


def test_the_gesture_waits_for_the_lock_and_refuses_after_a_bounded_wait(data_root, monkeypatch):
    import fcntl

    path = write_control_context(data_root, switch={"live": "asked"})
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
    set_switch(["va"], "asked", epoch=_epoch(path))
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
    path = write_control_context(data_root, generation=4, switch={"live": "asked", "va": "asked"})
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
    path = write_control_context(data_root, switch={"va": "asked"})
    assert ("va" in calls[-1][0]) and calls[-1][1] == "asked"
    epoch = _epoch(path)
    # A rewrite that names a different switch turns off what it no longer names.
    write_control_context(data_root, switch={"live": "asked"})
    assert _payload(path)["switch"] == {"live": "asked"}
    assert _payload(path)["switch_epoch"] == epoch


def test_stopped_since_carries_its_epoch_and_reason():
    """A route answers a stale gesture from attributes, never by parsing text."""
    error = StoppedSince(epoch=42)
    assert error.epoch == 42 and error.reason == "stopped_since"
    assert error.stamped is False
    assert StoppedSince(epoch=42, stamped=True).stamped is True
    # Not a RuntimeError: "store unavailable" handlers must not swallow it.
    assert not isinstance(error, RuntimeError)


@pytest.mark.parametrize("offset", [-1, 1], ids=["older", "newer"])
def test_the_fixture_retry_helpers_still_arm_after_a_refusal(data_root, offset):
    """The retry the fixtures use — re-offer the refusal's epoch — lands under equality."""
    path = write_control_context(data_root)
    epoch = _epoch(path)
    with pytest.raises(StoppedSince) as caught:
        set_switch(["va"], "asked", epoch=epoch + offset)
    assert caught.value.epoch == epoch
    set_switch(["va"], "asked", epoch=caught.value.epoch)
    assert _payload(path)["switch"] == {"va": "asked"}
