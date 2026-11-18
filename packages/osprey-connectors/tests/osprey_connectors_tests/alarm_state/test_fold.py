"""The fold from topic messages to what each PV's alarm is now."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from osprey_connectors.alarm_state import AlarmSeverity, AlarmStates
from osprey_connectors.alarm_state._fold import (
    FeedSnapshot,
    config_record,
    fold_message,
    state_record,
    state_rows,
)
from osprey_connectors.errors import NaiveDatetimeError
from osprey_connectors.service_answers import NotListed

PV = "DEMO:RF1:CAV:Temp-I"
PATH = f"/Demo/RF/Cavity1/{PV}"
STATE_KEY = f"state:{PATH}".encode()
CONFIG_KEY = f"config:{PATH}".encode()

CONFIG = {
    "user": "opsuser",
    "host": "alarm-srv.demo.lab",
    "description": "Cavity 1 over-temperature",
    "enabled": True,
    "latching": True,
    "guidance": [{"title": "Action", "details": "Check cooling water flow", "delay": 0}],
    "displays": [{"title": "Cavity", "details": "cavity.bob"}],
    "actions": [{"title": "Mail", "details": "mailto:someone@demo.lab"}],
}


def _state(severity: str, current: str, message: str = "HIHI_ALARM", current_msg: str = "NO_ALARM"):
    return json.dumps(
        {
            "severity": severity,
            "message": message,
            "value": "48.7",
            "time": {"seconds": 1793243647, "nano": 120000000},
            "current_severity": current,
            "current_message": current_msg,
        }
    ).encode()


def _folded(*messages) -> FeedSnapshot:
    snapshot = FeedSnapshot()
    for key, value in messages:
        fold_message(snapshot, key, value)
    return snapshot


def test_a_recorded_sequence_ends_in_the_latest_state():
    snapshot = _folded(
        (CONFIG_KEY, json.dumps(CONFIG).encode()),
        (STATE_KEY, _state("OK", "OK", "NO_ALARM")),
        (STATE_KEY, _state("MAJOR", "MAJOR", current_msg="HIHI_ALARM")),
        (STATE_KEY, _state("MAJOR", "OK")),
    )
    record = state_record(snapshot, PV, stale=False)
    assert record.severity == "MAJOR"
    assert record.message == "HIHI_ALARM"
    assert record.status == "NO_ALARM"
    assert record.latched is True
    assert record.acknowledged is False
    assert record.stale is False
    assert snapshot.messages == 4


def test_the_recorded_time_becomes_an_instant_in_utc():
    record = state_record(_folded((STATE_KEY, _state("MAJOR", "MAJOR"))), PV, stale=False)
    assert record.event_at == datetime.fromisoformat("2026-10-29T03:14:07.120Z")


def test_an_active_alarm_is_not_latched():
    record = state_record(_folded((STATE_KEY, _state("MAJOR", "MAJOR"))), PV, stale=False)
    assert record.latched is False


def test_an_alarm_that_only_eased_is_still_latched():
    record = state_record(_folded((STATE_KEY, _state("MAJOR", "MINOR"))), PV, stale=False)
    assert record.latched is True


def test_an_ack_severity_means_acknowledged_with_its_plain_severity():
    record = state_record(_folded((STATE_KEY, _state("MAJOR_ACK", "MAJOR"))), PV, stale=False)
    assert record.severity == "MAJOR"
    assert record.acknowledged is True
    assert record.latched is False


def test_an_acknowledged_alarm_that_recovered_is_not_latched():
    record = state_record(_folded((STATE_KEY, _state("MINOR_ACK", "OK"))), PV, stale=False)
    assert record.acknowledged is True
    assert record.latched is False


def test_a_cleared_alarm_reads_ok():
    snapshot = _folded(
        (STATE_KEY, _state("MAJOR", "OK")),
        (STATE_KEY, _state("OK", "OK", "NO_ALARM")),
    )
    record = state_record(snapshot, PV, stale=False)
    assert record.severity == "OK"
    assert record.latched is False
    assert record.acknowledged is False


def test_an_unknown_severity_reads_as_undefined_and_never_as_recovered():
    record = state_record(_folded((STATE_KEY, _state("WEIRD", "ALSO_WEIRD"))), PV, stale=False)
    assert record.severity is AlarmSeverity.UNDEFINED
    assert record.latched is False


def test_the_stale_flag_is_the_callers():
    record = state_record(_folded((STATE_KEY, _state("OK", "OK"))), PV, stale=True)
    assert record.stale is True


def test_a_state_tombstone_removes_the_state():
    snapshot = _folded((STATE_KEY, _state("MAJOR", "MAJOR")), (STATE_KEY, None))
    assert state_record(snapshot, PV, stale=False) == NotListed(address=PV)
    assert snapshot.messages == 2


def test_an_empty_value_is_a_tombstone():
    snapshot = _folded((STATE_KEY, _state("MAJOR", "MAJOR")), (STATE_KEY, b""))
    assert state_record(snapshot, PV, stale=False) == NotListed(address=PV)


def test_a_config_tombstone_takes_the_pv_out_of_the_tree():
    snapshot = _folded((CONFIG_KEY, json.dumps(CONFIG).encode()), (CONFIG_KEY, None))
    assert config_record(snapshot, PV).in_tree == "no"


def test_a_delete_marker_removes_like_a_tombstone():
    delete = json.dumps({"user": "opsuser", "host": "h", "delete": "removed"}).encode()
    snapshot = _folded((CONFIG_KEY, json.dumps(CONFIG).encode()), (CONFIG_KEY, delete))
    assert config_record(snapshot, PV).in_tree == "no"


def test_a_tombstone_of_one_path_leaves_the_other():
    other_key = f"state:/Demo/Other/{PV}".encode()
    snapshot = _folded(
        (other_key, _state("MINOR", "MINOR")),
        (STATE_KEY, _state("MAJOR", "MAJOR")),
        (STATE_KEY, None),
    )
    assert state_record(snapshot, PV, stale=False).severity == "MINOR"


def test_config_is_the_live_configuration_without_the_actions():
    record = config_record(_folded((CONFIG_KEY, json.dumps(CONFIG).encode())), PV)
    assert record.in_tree == "yes"
    assert record.path == PATH
    assert record.guidance == CONFIG["guidance"]
    assert record.displays == CONFIG["displays"]
    assert record.enabled is True
    assert record.latching is True
    assert "mailto" not in repr(record)


def test_a_pv_no_message_names_is_not_in_the_tree():
    record = config_record(_folded((CONFIG_KEY, json.dumps(CONFIG).encode())), "OTHER:PV")
    assert record.in_tree == "no"


def test_a_state_without_config_is_unknown_with_its_path():
    record = config_record(_folded((STATE_KEY, _state("OK", "OK"))), PV)
    assert record.in_tree == "unknown"
    assert record.path == PATH


def test_tree_names_are_read_as_the_pv_they_watch():
    escaped = rb"state:/Demo/Group/ca:\/\/DEMO:Escaped"
    snapshot = _folded((escaped, _state("MINOR", "MINOR")))
    assert state_record(snapshot, "DEMO:Escaped", stale=False).severity == "MINOR"
    assert state_record(snapshot, "ca://DEMO:Escaped", stale=False).severity == "MINOR"


def test_keys_that_say_nothing_about_a_pv_still_count_as_the_feed_speaking():
    snapshot = _folded(
        (b"command:/Demo/RF", b'{"command": "acknowledge"}'),
        (None, b"{}"),
        (b"state", b"{}"),
    )
    assert snapshot.messages == 3
    assert snapshot.states == {}
    assert snapshot.configs == {}


def test_an_unparseable_value_leaves_the_last_good_one():
    snapshot = _folded(
        (STATE_KEY, _state("MAJOR", "MAJOR")),
        (STATE_KEY, b"{not json"),
        (STATE_KEY, b"\xff\xfe"),
        (STATE_KEY, b"[1, 2]"),
    )
    assert state_record(snapshot, PV, stale=False).severity == "MAJOR"
    assert snapshot.messages == 4


def test_str_keys_and_mapping_values_fold_like_the_bytes():
    from_bytes = _folded((STATE_KEY, _state("MAJOR", "OK")))
    from_objects = _folded((STATE_KEY.decode(), json.loads(_state("MAJOR", "OK"))))
    assert from_objects == from_bytes


def test_a_given_time_replaces_the_recorded_one():
    snapshot = FeedSnapshot()
    given = datetime(2026, 11, 6, tzinfo=UTC)
    fold_message(snapshot, STATE_KEY, _state("MAJOR", "MAJOR"), event_at=given)
    assert state_record(snapshot, PV, stale=False).event_at == datetime.fromisoformat(
        "2026-11-06T00:00:00.000Z"
    )


@pytest.mark.parametrize(
    "recorded", [{"seconds": "soon"}, "2026-10-29T03:14:07.120", "soon", 1793243647]
)
def test_an_unusable_or_unzoned_recorded_time_is_none(recorded: object):
    value = json.dumps({"severity": "OK", "time": recorded}).encode()
    assert state_record(_folded((STATE_KEY, value)), PV, stale=False).event_at is None


def test_a_recorded_time_with_a_zone_is_read_as_that_instant():
    value = json.dumps({"severity": "OK", "time": "2026-10-28T20:14:07-07:00"}).encode()
    record = state_record(_folded((STATE_KEY, value)), PV, stale=False)
    assert record.event_at == datetime(2026, 10, 29, 3, 14, 7, tzinfo=UTC)


def test_a_naive_given_time_is_refused():
    with pytest.raises(NaiveDatetimeError):
        fold_message(
            FeedSnapshot(), STATE_KEY, _state("MAJOR", "MAJOR"), event_at=datetime(2026, 11, 6)
        )


# -- the list of every channel's state ------------------------------------------


def _key(pv: str, group: str = "Demo/RF") -> bytes:
    return f"state:/{group}/{pv}".encode()


def _rows(snapshot: FeedSnapshot, *, stale: bool = False) -> dict[str, object]:
    return {row.address: row for row in state_rows(snapshot, stale=stale)}


def test_a_tree_node_state_is_not_a_row():
    snapshot = _folded(
        (b"state:/Demo/SR", json.dumps({"severity": "MAJOR"}).encode()),
        (STATE_KEY, _state("MAJOR", "MAJOR")),
    )
    assert list(_rows(snapshot)) == [PV]


def test_each_row_is_the_channels_record_under_its_latest_path():
    other = f"state:/Demo/Other/{PV}".encode()
    snapshot = _folded((STATE_KEY, _state("MINOR", "MINOR")), (other, _state("MAJOR", "OK")))
    row = _rows(snapshot, stale=True)[PV]
    assert row.path == f"/Demo/Other/{PV}"
    assert row.condition == state_record(snapshot, PV, stale=True)
    assert row.condition.latched is True
    assert row.condition.stale is True


def test_an_acknowledged_alarm_lists_acknowledged():
    row = _rows(_folded((STATE_KEY, _state("MAJOR_ACK", "MAJOR"))))[PV]
    assert row.condition.severity == "MAJOR"
    assert row.condition.acknowledged is True


def test_a_tombstoned_channel_is_gone():
    snapshot = _folded((STATE_KEY, _state("MAJOR", "MAJOR")), (STATE_KEY, None))
    assert _rows(snapshot) == {}


def test_the_counts_equal_a_hand_count():
    snapshot = _folded(
        (_key("A:1"), _state("MAJOR", "MAJOR")),
        (_key("A:2"), _state("MAJOR_ACK", "MAJOR")),
        (_key("A:3"), _state("MINOR", "MINOR")),
        (_key("A:4"), _state("OK", "OK", "NO_ALARM")),
        (b"state:/Demo/RF", json.dumps({"severity": "MAJOR"}).encode()),
    )
    states = AlarmStates(rows=state_rows(snapshot, stale=False), stale=False, last_contact_at=None)
    assert dict(states.counts) == {"MAJOR": 2, "MINOR": 1, "OK": 1}


#: The ranks of the severity table before ``AlarmSeverity``: an unknown word ranked above all.
_OLD_RANK = {"OK": 0, "MINOR": 1, "MAJOR": 2, "INVALID": 3, "UNDEFINED": 4}


def _old_latched(severity: object, current: object) -> bool:
    """The latched rule as it read the recorded words before ``AlarmSeverity``."""

    def rank(word: object) -> int:
        return _OLD_RANK.get(word, 5) if isinstance(word, str) else 5

    recorded = None if severity is None else str(severity)
    bare = recorded.removesuffix("_ACK") if recorded is not None else None
    return bare == recorded and rank(current) < rank(bare)


_WORDS = (None, "OK", "MINOR", "MAJOR", "INVALID", "UNDEFINED", "BOGUS", 3)


@pytest.mark.parametrize("severity", (*_WORDS, "MAJOR_ACK", "UNDEFINED_ACK", "BOGUS_ACK"))
@pytest.mark.parametrize("current", _WORDS)
def test_latched_reads_every_pair_of_recorded_words_as_before(severity, current):
    body = {"severity": severity, "current_severity": current}
    snapshot = _folded((STATE_KEY, json.dumps(body).encode()))

    assert state_record(snapshot, PV, stale=False).latched is _old_latched(severity, current)


def test_a_severity_that_is_not_text_is_read_as_its_text():
    body = json.loads(_state("MAJOR", "MAJOR"))
    body["severity"] = 3
    snapshot = _folded((STATE_KEY, json.dumps(body).encode()))
    assert state_record(snapshot, PV, stale=False).severity is AlarmSeverity.UNDEFINED
    states = AlarmStates(rows=state_rows(snapshot, stale=False), stale=False, last_contact_at=None)
    assert dict(states.counts) == {"UNDEFINED": 1}
