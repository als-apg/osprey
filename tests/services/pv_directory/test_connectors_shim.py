"""Tests for the connectors shim, with the PV directory modules present and blocked.

Blocking is the connectors floor: a release without those modules. The shim then
binds every contract name to ``None`` and its payload helpers still answer.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from types import ModuleType

import pytest

from osprey.services.pv_directory import _connectors, resolve

_NEW_MODULES = (
    "osprey_connectors.service_answers",
    "osprey_connectors.service_targets",
    "osprey_connectors._service_http",
    "osprey_connectors.directory",
    "osprey_connectors.directory._contract",
    "osprey_connectors.alarm_state",
    "osprey_connectors.alarm_state._contract",
    "osprey_connectors.alarm_log",
    "osprey_connectors.alarm_log._contract",
    "osprey_connectors.put_log",
    "osprey_connectors.put_log._contract",
)

_CONTRACT_NAMES = (
    "APPLIANCE_TARGETS",
    "ENGINEER",
    "IOC_HEALTH_SIGNALS",
    "SERVED_SIGNAL_PROPERTY",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
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


@pytest.fixture
def floor() -> Iterator[ModuleType]:
    """The shim re-imported with the PV directory modules unimportable."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        yield importlib.reload(_connectors)
    importlib.reload(_connectors)


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
    assert _connectors.unavailable_payload("not available for this target", target="standin") == {
        "outcome": "unavailable",
        "source": {"target": "standin", "service": "none", "label": "All PVs", "rehearsal": True},
        "reason": "not available for this target",
    }


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

    assert _connectors.DIRECTORY_CONNECTORS_AVAILABLE is True
    assert _connectors.Source is service_answers.Source
    assert _connectors.SERVED_SIGNAL_PROPERTY is view.SERVED_SIGNAL_PROPERTY
    assert _connectors.has_simulator is simulator.has_simulator
    assert all(getattr(_connectors, name) is not None for name in _CONTRACT_NAMES)


def test_as_payload_of_a_contract_record() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    source = _connectors.Source(
        kind="directory", target="live", service="channelfinder", rehearsal=False
    )
    page = _connectors.HistoryPage(
        outcome=_connectors.Outcome.UNAVAILABLE, source=source, reason="login page"
    )

    assert _connectors.as_payload(page) == {
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

    assert _connectors.as_payload(directory) == {
        "target": "standin",
        "service": "channelfinder",
        "label": "All PVs",
        "rehearsal": True,
    }
    assert _connectors.as_payload(put_log)["label"] == "caputlog"


def test_rows_are_served_under_the_epics_words() -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    put = _connectors.PutRow(at="t", channel="SR:BPM1.HIHI", field=".HIHI", old="1", new="2")
    row = _connectors.DirectoryRow(name="SR:BPM1", channel_type="ai", server="ioc-a")

    assert _connectors.as_payload(put) == {
        "at": "t",
        "pv": "SR:BPM1.HIHI",
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
        "ioc": "ioc-a",
        "host": None,
        "last_announced": None,
    }


def test_floor_binds_every_contract_name_to_none(floor: ModuleType) -> None:
    assert floor.DIRECTORY_CONNECTORS_AVAILABLE is False
    assert {name: getattr(floor, name) for name in _CONTRACT_NAMES} == dict.fromkeys(
        _CONTRACT_NAMES
    )


def test_floor_payload_helpers_still_work(floor: ModuleType) -> None:
    payload = floor.unavailable_payload(floor.FLOOR_REASON, target="live")

    assert payload["source"]["service"] == "none"
    assert (
        payload["reason"] == "installed osprey-connectors has no PV directory support (upgrade it)"
    )
    assert floor.as_payload(_Inner(kind=_Kind.ROW, names=())) == {"kind": "row", "names": []}


def test_floor_section_resolves_with_one_warning(floor: ModuleType) -> None:
    resolved = resolve.resolve_section({"control_system": {"type": "epics"}, "pv_directory": {}})

    assert resolved is not None
    assert resolved.warnings == (f"pv_directory: {floor.FLOOR_REASON}",)
