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
    "osprey_connectors.pv_directory",
    "osprey_connectors.pv_directory.contract",
    "osprey_connectors.pv_directory.targets",
)

_CONTRACT_NAMES = (
    "ALARM_STALE_WINDOW_S",
    "DEFAULT_DESCRIPTION_PROPERTY",
    "ENGINEER",
    "IOC_HEALTH_SIGNALS",
    "ROW_FIELD_PROPERTIES",
    "SERVED_SIGNAL_PROPERTY",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
    "BackendSpec",
    "DirectoryPage",
    "DirectoryRecord",
    "DirectoryRow",
    "HistoryPage",
    "Outcome",
    "PVDirectory",
    "PutLog",
    "PutRow",
    "SearchQuery",
    "Source",
    "Unavailable",
    "has_simulator",
    "pv_directory_targets",
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
    pytest.importorskip("osprey_connectors.pv_directory")
    from osprey_connectors.pv_directory import contract, targets

    assert _connectors.DIRECTORY_CONNECTORS_AVAILABLE is True
    assert _connectors.Source is contract.Source
    assert _connectors.ROW_FIELD_PROPERTIES is contract.ROW_FIELD_PROPERTIES
    assert _connectors.has_simulator is targets.has_simulator
    assert all(getattr(_connectors, name) is not None for name in _CONTRACT_NAMES)


def test_as_payload_of_a_contract_record() -> None:
    pytest.importorskip("osprey_connectors.pv_directory")
    source = _connectors.Source(
        target="live", service="channelfinder", label="All PVs", rehearsal=False
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
