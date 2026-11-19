"""Tests for the connectors shim: the names it binds and its payload helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pytest

from osprey.services.channel_directory import _connectors
from osprey.services.channel_directory.vocabulary import EPICS_WORDS, NEUTRAL_WORDS

_CONTRACT_NAMES = (
    "APPLIANCE_TARGETS",
    "ENGINEER",
    "SERVED_SIGNAL_PROPERTY",
    "SERVER_HEALTH_SIGNALS",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
    "AlarmStateRow",
    "AlarmStates",
    "BackendSpec",
    "ChannelDirectory",
    "DirectoryPage",
    "DirectoryQuery",
    "DirectoryRecord",
    "DirectoryRow",
    "HistoryPage",
    "Outcome",
    "PutLog",
    "PutRow",
    "Source",
    "Unavailable",
    "has_simulator",
    "service_targets",
)


class _Kind(StrEnum):
    ROW = "row"


@dataclass(frozen=True)
class _Inner:
    kind: _Kind
    names: tuple[str, ...]


@dataclass(frozen=True)
class _Outer:
    inner: _Inner
    rows: list[_Inner] = field(default_factory=list)
    extra: dict[str, _Kind] = field(default_factory=dict)
    note: str | None = None


def test_unavailable_payload_is_the_no_source_form() -> None:
    payload = _connectors.unavailable_payload(
        "not available for this target", target="standin", words={"standin": EPICS_WORDS}
    )

    assert payload == {
        "outcome": "unavailable",
        "source": {"target": "standin", "service": "none", "label": "All PVs", "rehearsal": True},
        "reason": "not available for this target",
    }


def test_without_words_the_no_source_form_is_labelled_in_the_neutral_words() -> None:
    payload = _connectors.unavailable_payload("not available for this target", target="live")

    assert payload["source"]["label"] == NEUTRAL_WORDS.directory == "Channels"


def test_as_payload_converts_nested_dataclasses_and_enums() -> None:
    inner = _Inner(kind=_Kind.ROW, names=("A", "B"))

    payload = _connectors.as_payload(_Outer(inner=inner, rows=[inner], extra={"k": _Kind.ROW}))

    assert payload == {
        "inner": {"kind": "row", "names": ["A", "B"]},
        "rows": [{"kind": "row", "names": ["A", "B"]}],
        "extra": {"k": "row"},
        "note": None,
    }


def test_with_the_modules_every_name_is_bound() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    from osprey_connectors import service_answers, simulator
    from osprey_connectors.simulation import view

    assert _connectors.Source is service_answers.Source
    assert _connectors.SERVED_SIGNAL_PROPERTY is view.SERVED_SIGNAL_PROPERTY
    assert _connectors.has_simulator is simulator.has_simulator
    assert all(getattr(_connectors, name) is not None for name in _CONTRACT_NAMES)


def test_with_the_modules_the_listing_is_bound() -> None:
    directory = pytest.importorskip("osprey_connectors.directory")

    assert _connectors.ServerRow is directory.ServerRow


def test_as_payload_of_a_contract_record() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    source = _connectors.Source(
        kind="directory", target="live", service="channelfinder", rehearsal=False
    )
    page = _connectors.HistoryPage(
        outcome=_connectors.Outcome.UNAVAILABLE, source=source, reason="login page"
    )

    assert _connectors.as_payload(page, {"live": EPICS_WORDS}) == {
        "outcome": "unavailable",
        "source": {
            "target": "live",
            "service": "channelfinder",
            "label": "All PVs",
            "rehearsal": False,
        },
        "rows": [],
        "truncated": False,
        "retention_known": False,
        "reason": "login page",
    }


def test_a_source_is_served_with_the_label_the_framework_names() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    directory = _connectors.Source(
        kind="directory", target="standin", service="channelfinder", rehearsal=True
    )
    put_log = _connectors.Source(kind="put_log", target="live", service="caputlog", rehearsal=False)
    words = {"standin": EPICS_WORDS, "live": EPICS_WORDS}

    assert _connectors.as_payload(directory, words) == {
        "target": "standin",
        "service": "channelfinder",
        "label": "All PVs",
        "rehearsal": True,
    }
    assert _connectors.as_payload(directory)["label"] == "Channels"
    assert _connectors.as_payload(put_log, words)["label"] == "caputlog"


def test_rows_are_served_under_the_epics_words() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    put = _connectors.PutRow(at="t", channel="SR:BPM1.HIHI", field=".HIHI", old="1", new="2")
    row = _connectors.DirectoryRow(name="SR:BPM1", channel_type="ai", server="ioc-a")

    assert _connectors.as_payload(put) == {
        "at": "t",
        "channel": "SR:BPM1.HIHI",
        "field": ".HIHI",
        "old": "1",
        "new": "2",
        "user": None,
        "client": None,
    }
    assert _connectors.as_payload(row) == {
        "name": "SR:BPM1",
        "status": None,
        "description": None,
        "record_type": "ai",
        "server": "ioc-a",
        "host": None,
        "last_announced": None,
    }
