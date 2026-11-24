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

from osprey.facility.views.limits import limits_document
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_facility.records import FacilityRecords
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


def _doc(
    channels: dict[str, str],
    records: list[dict[str, Any]] | None,
    value_type: str | None = None,
) -> dict[str, Any]:
    typed = {} if value_type is None else {"value_type": value_type}
    doc: dict[str, Any] = {
        "channels": [{"id": id_, "role": role, **typed} for id_, role in channels.items()]
    }
    if records is not None:
        doc["limits"] = {"records": records}
    return doc


# --- one record, one entry ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value_type", "role", "record", "entry"),
    [
        (
            "float",
            "setpoint",
            {"min_value": 0.0, "max_value": 5.0},
            {"min_value": 0.0, "max_value": 5.0, "writable": True, "confirm": True},
        ),
        (
            "float",
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
            "float",
            "setpoint",
            {"min_value": 0.0, "max_value": 5.0, "writable": False},
            {"min_value": 0.0, "max_value": 5.0, "writable": False, "confirm": True},
        ),
        (
            "float",
            "setpoint",
            {"max_value": 5.0},
            {"max_value": 5.0, "writable": False, "confirm": True},
        ),
        (
            "float",
            "setpoint",
            {"min_value": 0.0},
            {"min_value": 0.0, "writable": False, "confirm": True},
        ),
        ("float", "setpoint", {}, {"writable": False, "confirm": True}),
        ("float", "setpoint", {"writable": False}, {"writable": False, "confirm": True}),
        (
            "float",
            "readback",
            {"min_value": 0.0, "max_value": 5.0},
            {"min_value": 0.0, "max_value": 5.0, "writable": False, "confirm": True},
        ),
        ("float", "readback", {"writable": False}, {"writable": False, "confirm": True}),
        ("float", "setpoint", {"writable": True}, {"writable": False, "confirm": True}),
        ("bool", "setpoint", {}, {"writable": False, "confirm": True}),
        ("bool", "setpoint", {"writable": False}, {"writable": False, "confirm": True}),
        ("bool", "readback", {"writable": True}, {"writable": False, "confirm": True}),
    ],
)
def test_a_record_resolves_to_one_entry(
    value_type: str, role: str, record: dict[str, Any], entry: dict[str, Any]
) -> None:
    doc = _doc({"A:B": role}, [{"address": "A:B", **record}], value_type)

    assert limits_document(FacilityRecords(doc)) == {"_version": "4.0", "A:B": entry}


@pytest.mark.parametrize("value_type", ["bool", "enum", "string", "waveform"])
def test_a_stated_writable_makes_a_setpoint_without_bounds_writable(value_type: str) -> None:
    doc = _doc({"A:GO": "setpoint"}, [{"address": "A:GO", "writable": True}], value_type)

    assert limits_document(FacilityRecords(doc))["A:GO"] == {"writable": True, "confirm": True}


def test_a_channel_without_a_record_has_no_entry() -> None:
    doc = _doc(
        {"A:SP": "setpoint", "B:SP": "setpoint", "B:RB": "readback"},
        [{"address": "A:SP", "min_value": 0.0, "max_value": 1.0}],
    )

    assert sorted(limits_document(FacilityRecords(doc))) == ["A:SP", "_version"]


@pytest.mark.parametrize("records", [None, []])
def test_no_records_give_the_version_alone(records: list[dict[str, Any]] | None) -> None:
    doc = _doc({"A:SP": "setpoint", "A:RB": "readback"}, records)

    assert limits_document(FacilityRecords(doc)) == {"_version": "4.0"}


def test_entries_follow_the_version_sorted_by_address() -> None:
    channels = {"C:SP": "setpoint", "A:SP": "setpoint", "B:SP": "setpoint"}
    records = [{"address": address, "writable": False} for address in channels]

    assert list(limits_document(FacilityRecords(_doc(channels, records)))) == [
        "_version",
        "A:SP",
        "B:SP",
        "C:SP",
    ]


# --- the demo build ---------------------------------------------------


@pytest.mark.slow
class TestTheDemoBuild:
    def test_every_render_carries_the_view(self, built_demo: BuiltProject) -> None:
        expected = (
            json.dumps(limits_document(FacilityRecords(built_demo.facility)), indent=2) + "\n"
        )

        assert built_demo.outputs
        for outputs in built_demo.outputs:
            assert outputs.files[LIMITS] == expected.encode("utf-8")

    def test_the_rendered_file_is_the_view_not_the_copied_file(
        self, built_demo: BuiltProject
    ) -> None:
        rendered = (built_demo.build_dir / LIMITS).read_bytes()

        assert rendered == built_demo.outputs[0].files[LIMITS]
        assert "defaults" not in json.loads(rendered)

    def test_the_entries_are_the_limits_golden(self, built_demo: BuiltProject) -> None:
        document = json.loads((built_demo.build_dir / LIMITS).read_bytes())
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


# --- a bool command setpoint --------------------------------------------------------

PADDLE = "BR:DIAG:TV:01:PADDLE:SP"


@pytest.mark.slow
class TestABoolCommandUnderExclusive:
    def test_a_stated_writable_record_passes_the_validator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = {"address": PADDLE, "writable": True, "confirm": False}
        validator = validator_under(
            monkeypatch, render_limits(tmp_path, "demo", added=[record]), "exclusive"
        )

        validator.validate(PADDLE, 1)
        validator.validate(PADDLE, "TRUE")
        assert validator.resolve_confirm(PADDLE) is False

    @pytest.mark.parametrize(
        ("added", "refusal"),
        [
            ([], "UNLISTED_CHANNEL"),
            ([{"address": PADDLE, "confirm": False}], "READ_ONLY_CHANNEL"),
            ([{"address": PADDLE, "writable": False}], "READ_ONLY_CHANNEL"),
        ],
        ids=["no-record", "confirm-only", "writable-false"],
    )
    def test_it_is_refused_without_a_stated_writable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        added: list[dict[str, Any]],
        refusal: str,
    ) -> None:
        validator = validator_under(
            monkeypatch, render_limits(tmp_path, "demo", added=added), "exclusive"
        )

        with pytest.raises(ChannelLimitsViolationError) as refused:
            validator.validate(PADDLE, 1)
        assert refused.value.violation_type == refusal


# --- the database path -------------------------------------------------------------

DATABASE_KEY = "control_system.limits_checking.database_path"


def _injected(tmp_path: Path, control_system: dict[str, Any] | None) -> dict[str, Any]:
    """A render's config after the build has derived the database path."""
    import yaml

    from osprey.cli.build_injectors import _inject_limits_database

    config: dict[str, Any] = {"project_name": "demo"}
    if control_system is not None:
        config["control_system"] = control_system
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _inject_limits_database(tmp_path)
    rendered: dict[str, Any] = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    return rendered


class TestTheBuildDerivesTheDatabasePath:
    def test_a_limits_block_gains_the_view_path(self, tmp_path: Path) -> None:
        rendered = _injected(tmp_path, {"limits_checking": {"enabled": True, "mode": "optional"}})

        assert rendered["control_system"]["limits_checking"] == {
            "enabled": True,
            "mode": "optional",
            "database_path": LIMITS,
        }

    def test_a_per_type_block_gains_the_deployment_wide_path(self, tmp_path: Path) -> None:
        block = {"enabled": True, "mode": "exclusive"}

        rendered = _injected(tmp_path, {"connector": {"epics": {"limits_checking": dict(block)}}})

        assert rendered["control_system"]["limits_checking"] == {"database_path": LIMITS}
        assert rendered["control_system"]["connector"]["epics"]["limits_checking"] == block

    def test_the_view_path_replaces_a_stated_one(self, tmp_path: Path) -> None:
        block = {"enabled": True, "mode": "optional", "database_path": "/srv/limits.json"}

        rendered = _injected(tmp_path, {"limits_checking": dict(block)})

        assert rendered["control_system"]["limits_checking"] == {
            "enabled": True,
            "mode": "optional",
            "database_path": LIMITS,
        }

    @pytest.mark.parametrize(
        "control_system",
        [
            None,
            {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
            },
        ],
    )
    def test_a_config_stating_no_limits_block_gains_nothing(
        self, tmp_path: Path, control_system: dict[str, Any] | None
    ) -> None:
        rendered = _injected(tmp_path, control_system)

        assert "limits_checking" not in (rendered.get("control_system") or {})

    @pytest.mark.parametrize("preset", ["demo", "hello-world"])
    def test_no_preset_states_the_path(self, preset: str) -> None:
        from osprey.cli.build_profile import resolve_build_profile

        profile, _preset_dir = resolve_build_profile(None, preset=preset)

        assert DATABASE_KEY not in profile.config

    @pytest.mark.slow
    def test_the_built_render_names_the_view(self, built_demo: BuiltProject) -> None:
        import yaml

        config = yaml.safe_load((built_demo.build_dir / "config.yml").read_text(encoding="utf-8"))

        assert config["control_system"]["limits_checking"]["database_path"] == LIMITS
        assert (built_demo.build_dir / LIMITS).is_file()


# --- findings: report lines, nothing changed ----------------------------------------


def test_findings_name_open_setpoints_and_records_short_of_writable() -> None:
    from osprey.facility.views.limits import limits_findings

    doc = _doc(
        {QF: "setpoint", QD: "setpoint", CORR: "setpoint", BEAM: "readback"},
        [
            {"address": QF, "min_value": 0.0, "max_value": 5.0},  # banded: no finding
            {"address": QD, "min_value": 0.0},  # one edge
        ],
    )
    records = FacilityRecords(doc)
    assert limits_findings(records, True) == [
        "limits: 1 setpoints have no limits record; the target serves them writable with no range",
        f"  {CORR}",  # a readback without a record is no finding: it is never writable
        "limits: 1 limits records serve their setpoint not writable; stating what each line "
        "lacks makes it writable",
        f"  {QD}: min_value 0.0, no max_value",
    ]
    # the same setpoint where the target refuses a write to a channel with no record
    refused = limits_findings(records, False)[0]
    assert refused == "limits: 1 setpoints have no limits record; the target refuses writes to them"
    # the findings change nothing: the database is what it was
    assert limits_document(records)[QD]["writable"] is False


def _stating(record: dict[str, Any], missing: tuple[str, ...]) -> dict[str, Any]:
    """The record with each slot it lacks stated: an edge inside any stated one, else `true`."""
    stated = dict(record)
    for slot in missing:
        if slot == "min_value":
            stated[slot] = stated["max_value"] - 1.0 if "max_value" in stated else 0.0
        elif slot == "max_value":
            stated[slot] = stated["min_value"] + 1.0
        else:
            stated[slot] = True
    return stated


def test_every_listed_record_becomes_writable_by_stating_what_its_line_lacks() -> None:
    from osprey.facility.views.limits import limits_findings
    from osprey_facility.validate import record_missing, record_writable

    qs = "SR:MAG:QS:01:CURRENT:SP"
    lock = "SR:MAG:QL:01:CURRENT:SP"
    rb = "SR:MAG:QD:01:CURRENT:RB"
    valve = "SR:VAC:VALVE:01:OPEN:SP"
    gate = "SR:VAC:GATE:01:OPEN:SP"
    channels = {
        QD: ("setpoint", "float"),
        qs: ("setpoint", "float"),
        lock: ("setpoint", "float"),
        QF: ("setpoint", "float"),
        rb: ("readback", "float"),
        valve: ("setpoint", "bool"),
        gate: ("setpoint", "bool"),
    }
    limits = {
        QD: {"min_value": 0.0},
        qs: {"max_step": 0.5},
        lock: {"min_value": 0.0, "writable": False},
        QF: {"min_value": 0.0, "max_value": 5.0},
        rb: {"min_value": 0.0},
        valve: {"confirm": False},
        gate: {"writable": True},
    }
    records = FacilityRecords(
        {
            "channels": [
                {"id": address, "role": role, "value_type": value_type}
                for address, (role, value_type) in channels.items()
            ],
            "limits": {
                "records": [{"address": address, **record} for address, record in limits.items()]
            },
        }
    )

    assert limits_findings(records, True) == [
        "limits: 3 limits records serve their setpoint not writable; stating what each line "
        "lacks makes it writable",
        f"  {QD}: min_value 0.0, no max_value",
        f"  {qs}: max_step 0.5, no min_value, no max_value",
        f"  {valve}: no writable: true",
    ]
    listed = {QD, qs, valve}
    for address, record in limits.items():
        role, value_type = channels[address]
        missing = record_missing(record, role, value_type)
        if address in listed:
            assert missing
            assert record_writable(_stating(record, missing), role, value_type), address
        else:
            assert record_writable(record, role, value_type) or missing is None, address


def test_findings_list_the_first_addresses_unless_every_one_is_asked_for() -> None:
    from osprey.facility.views.limits import FINDING_ADDRESSES, limits_findings

    many = {f"XR:MAG:Q:{n:02d}:SP": "setpoint" for n in range(FINDING_ADDRESSES + 3)}
    records = FacilityRecords(_doc(many, []))
    short = limits_findings(records, True)
    assert len(short) == 1 + FINDING_ADDRESSES + 1
    assert short[-1] == "  and 3 more; --all-limits lists every one"
    assert len(limits_findings(records, True, every=True)) == 1 + len(many)


def test_a_facility_with_every_setpoint_banded_has_no_finding() -> None:
    from osprey.facility.views.limits import limits_findings

    records = FacilityRecords(
        _doc({QF: "setpoint"}, [{"address": QF, "min_value": 0.0, "max_value": 5.0}])
    )
    assert limits_findings(records, True) == []
