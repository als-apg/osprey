"""The limits view: ``limits.yaml``'s records as ``data/channel_limits.json``.

The view writes ``_version: "4.0"`` and one entry per limits record, each
stating ``writable`` and ``confirm``. It writes no ``defaults`` block and no
entry for a channel without a record: ``control_system.limits_checking.mode``
decides such a channel at write time. Every render carries the file, and it
replaces a ``data/channel_limits.json`` the project's ``data/`` tree copied in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.errors import ChannelLimitsViolationError
from osprey.facility.views.limits import limits_document
from tests.facility._limits_render import render_limits, validator_under
from tests.facility.test_cf_view_parity import load_golden

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

LIMITS = "data/channel_limits.json"

QF = "SR:MAG:QF:01:CURRENT:SP"
QD = "SR:MAG:QD:01:CURRENT:SP"
CORR = "SR:MAG:CORR:01:CURRENT:SP"
BEAM = "SR:BEAM:CURRENT"

SLOTS = ("min_value", "max_value", "max_step", "writable", "confirm")


def _doc(channels: dict[str, str], records: list[dict[str, Any]] | None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "channels": [{"id": id_, "role": role} for id_, role in channels.items()]
    }
    if records is not None:
        doc["limits"] = {"records": records}
    return doc


# --- one record, one entry ---------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "record", "entry"),
    [
        (
            "setpoint",
            {"min_value": 0.0, "max_value": 5.0},
            {"min_value": 0.0, "max_value": 5.0, "writable": True, "confirm": True},
        ),
        (
            "setpoint",
            {"min_value": 0.0, "max_value": 5.0, "max_step": 0.5, "confirm": False},
            {
                "min_value": 0.0,
                "max_value": 5.0,
                "max_step": 0.5,
                "writable": True,
                "confirm": False,
            },
        ),
        (
            "setpoint",
            {"min_value": 0.0, "max_value": 5.0, "writable": False},
            {"min_value": 0.0, "max_value": 5.0, "writable": False, "confirm": True},
        ),
        ("setpoint", {"max_value": 5.0}, {"max_value": 5.0, "writable": False, "confirm": True}),
        ("setpoint", {"min_value": 0.0}, {"min_value": 0.0, "writable": False, "confirm": True}),
        ("setpoint", {}, {"writable": False, "confirm": True}),
        ("setpoint", {"writable": False}, {"writable": False, "confirm": True}),
        (
            "readback",
            {"min_value": 0.0, "max_value": 5.0},
            {"min_value": 0.0, "max_value": 5.0, "writable": False, "confirm": True},
        ),
        ("readback", {"writable": False}, {"writable": False, "confirm": True}),
        ("setpoint", {"writable": True}, {"writable": False, "confirm": True}),
    ],
)
def test_a_record_resolves_to_one_entry(
    role: str, record: dict[str, Any], entry: dict[str, Any]
) -> None:
    document = limits_document(_doc({"A:B": role}, [{"address": "A:B", **record}]))

    assert document == {"_version": "4.0", "A:B": entry}


def test_a_channel_without_a_record_has_no_entry() -> None:
    doc = _doc(
        {"A:SP": "setpoint", "B:SP": "setpoint", "B:RB": "readback"},
        [{"address": "A:SP", "min_value": 0.0, "max_value": 1.0}],
    )

    assert sorted(limits_document(doc)) == ["A:SP", "_version"]


@pytest.mark.parametrize("records", [None, []])
def test_no_records_give_the_version_alone(records: list[dict[str, Any]] | None) -> None:
    doc = _doc({"A:SP": "setpoint", "A:RB": "readback"}, records)

    assert limits_document(doc) == {"_version": "4.0"}


def test_entries_follow_the_version_sorted_by_address() -> None:
    channels = {"C:SP": "setpoint", "A:SP": "setpoint", "B:SP": "setpoint"}
    records = [{"address": address, "writable": False} for address in channels]

    assert list(limits_document(_doc(channels, records))) == ["_version", "A:SP", "B:SP", "C:SP"]


# --- the control-assistant build ---------------------------------------------------


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
class TestTheControlAssistantBuild:
    def test_every_render_carries_the_view(self, built_control_assistant: BuiltProject) -> None:
        expected = json.dumps(limits_document(built_control_assistant.facility), indent=2) + "\n"

        assert built_control_assistant.outputs
        for outputs in built_control_assistant.outputs:
            assert outputs.files[LIMITS] == expected.encode("utf-8")

    def test_the_rendered_file_is_the_view_not_the_copied_file(
        self, built_control_assistant: BuiltProject
    ) -> None:
        rendered = (built_control_assistant.build_dir / LIMITS).read_bytes()

        assert rendered == built_control_assistant.outputs[0].files[LIMITS]
        assert "defaults" not in json.loads(rendered)

    def test_the_entries_are_the_limits_golden(self, built_control_assistant: BuiltProject) -> None:
        document = json.loads((built_control_assistant.build_dir / LIMITS).read_bytes())
        version = document.pop("_version")
        resolved = {
            address: {slot: entry.get(slot) for slot in SLOTS}
            for address, entry in document.items()
        }

        assert version == "4.0"
        assert all(set(entry) <= set(SLOTS) for entry in document.values())
        assert resolved == load_golden("limits.json")["channels"]


# --- hello-world -------------------------------------------------------------------


def test_hello_world_holds_its_three_records(tmp_path: Path) -> None:
    document = json.loads(render_limits(tmp_path, "hello_world").read_bytes())

    assert document == {
        "_version": "4.0",
        BEAM: {"writable": False, "confirm": True},
        QD: {"min_value": 0.0, "max_value": 250.0, "writable": True, "confirm": True},
        QF: {"min_value": 0.0, "max_value": 300.0, "writable": True, "confirm": True},
    }


def test_hello_world_runs_optional() -> None:
    from osprey.cli.build_profile import resolve_build_profile

    profile, _preset_dir = resolve_build_profile(None, preset="hello-world")

    assert profile.config["control_system.limits_checking.mode"] == "optional"


@pytest.mark.parametrize("mode", ["exclusive", "optional"])
class TestHelloWorldRecordsUnderEitherMode:
    def test_every_write_is_confirmed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
    ) -> None:
        validator = validator_under(monkeypatch, render_limits(tmp_path, "hello_world"), mode)

        assert [validator.resolve_confirm(channel) for channel in (QF, QD, CORR)] == [True] * 3


class TestTheCorrectorSetpointHasNoRecord:
    def test_optional_writes_it_with_no_limits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        validator = validator_under(monkeypatch, render_limits(tmp_path, "hello_world"), "optional")

        for value in (0.0, 4.0, -1.0e9, 1.0e9):
            validator.validate(CORR, value)

    def test_exclusive_refuses_it_until_a_record_is_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        validator = validator_under(
            monkeypatch, render_limits(tmp_path, "hello_world"), "exclusive"
        )
        with pytest.raises(ChannelLimitsViolationError) as refusal:
            validator.validate(CORR, 4.0)
        assert refusal.value.violation_type == "UNLISTED_CHANNEL"

        record = {"address": CORR, "min_value": -5.0, "max_value": 5.0}
        recorded = render_limits(tmp_path, "hello_world", added=[record], name="recorded")
        validator = validator_under(monkeypatch, recorded, "exclusive")

        validator.validate(CORR, 4.0)
        with pytest.raises(ChannelLimitsViolationError) as refusal:
            validator.validate(CORR, 5.5)
        assert refusal.value.violation_type == "MAX_EXCEEDED"
