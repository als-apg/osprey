"""Tests for the mml layer's mapping loader (``osprey.facility.layers.mml.mapping``)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    EngineBlock,
    FieldRole,
    Identity,
    ImportStop,
    MappingError,
    Problem,
    SameAs,
    WiringFamily,
    check_mapping,
    draft_mapping,
    draft_text,
    dump_mapping,
    field_roles,
    load_or_draft,
    parse_mapping,
    read_mapping,
    require_decided,
    undecided_slots,
)


def _document() -> dict[str, Any]:
    """A small, fully decided new-format mapping document."""
    return {
        "facility": {"code": "QUOKKA", "name": "Quokka", "description": "A test facility."},
        "models": {
            "SR": {
                "name": "SR",
                "description": "The stored-beam lattice.",
                "provenance": "stated",
                "wiring": {
                    "QF": {
                        "element_field": "Setpoint",
                        "engine": {"attribute": "PolynomB", "index": 1},
                        "calibration": "table",
                    },
                    "BPMx": {
                        "element_field": "Monitor",
                        "engine": {"axis": "x"},
                        "calibration": "linear",
                    },
                },
            }
        },
        "section_order": ["SR"],
        "families": {
            "QF": {
                "class": "Quadrupole",
                "aliases": ["QF"],
                "description": "Focusing quadrupoles.",
                "provenance": "stated",
                "channels": 4,
                "fields": {
                    "Setpoint": {"description": "Current setpoint.", "provenance": "stated"},
                    "Monitor": {"description": "Current readback.", "provenance": "stated"},
                },
            },
            "BPMx": {
                "class": "BeamPositionMonitor",
                "aliases": ["BPMx"],
                "description": "Horizontal beam positions.",
                "provenance": "stated",
                "channels": 2,
                "fields": {
                    "Monitor": {"description": "Horizontal position.", "provenance": "stated"},
                },
            },
        },
        "directions": {
            "QF.Setpoint": {"direction": "write", "provenance": "stated", "override": False},
            "QF.Monitor": {"direction": "read", "provenance": "stated", "override": False},
            "BPMx.Monitor": {"direction": "read", "provenance": "stated", "override": False},
        },
    }


class TestParse:
    def test_facility_block_seeds_identity_only(self) -> None:
        mapping = parse_mapping(_document())
        assert mapping.identity == Identity(
            code="QUOKKA", name="Quokka", description="A test facility."
        )

    def test_facility_block_is_optional(self) -> None:
        document = _document()
        del document["facility"]
        assert parse_mapping(document).identity is None

    def test_facility_block_takes_code_alone(self) -> None:
        document = _document()
        document["facility"] = {"code": "QUOKKA"}
        assert parse_mapping(document).identity == Identity(
            code="QUOKKA", name=None, description=None
        )

    def test_models_carry_wiring_in_engine_words(self) -> None:
        model = parse_mapping(_document()).models["SR"]
        assert model.name == "SR"
        assert model.wiring == {
            "QF": WiringFamily(
                element_field="Setpoint",
                engine=EngineBlock(attribute="PolynomB", index=1),
                calibration="table",
            ),
            "BPMx": WiringFamily(
                element_field="Monitor", engine=EngineBlock(axis="x"), calibration="linear"
            ),
        }

    def test_a_frequency_family_takes_a_voltage_in_volts(self) -> None:
        document = _document()
        document["models"]["SR"]["wiring"]["RF"] = {
            "element_field": "Setpoint",
            "engine": {"attribute": "Frequency"},
            "calibration": "linear",
            "voltage": 3000000.0,
        }
        wiring = parse_mapping(document).models["SR"].wiring
        assert wiring["RF"].voltage == 3000000.0
        assert wiring["QF"].voltage is None

    def test_the_nsls2_fixture_answers_its_cavity_voltage(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "nsls2"
        mapping = read_mapping(fixture / "imported" / "mml" / "mapping.yaml")
        assert mapping.models["StorageRing"].wiring["RF"].voltage == 3000000.0

    def test_a_model_without_wiring_wires_nothing(self) -> None:
        document = _document()
        del document["models"]["SR"]["wiring"]
        assert parse_mapping(document).models["SR"].wiring == {}

    @pytest.mark.parametrize(
        ("written", "parsed"),
        [
            ("names", "names"),
            ("address", "address"),
            (["Q_A", "Q_B", "Q_C", "Q_D"], ("Q_A", "Q_B", "Q_C", "Q_D")),
            ({"same_as": "BPMx"}, SameAs("BPMx")),
        ],
    )
    def test_a_family_says_how_its_devices_are_identified(self, written: Any, parsed: Any) -> None:
        document = _document()
        document["families"]["QF"]["devices"] = written
        families = parse_mapping(document).families
        assert (families["QF"].devices, families["QF"].devices_present) == (parsed, True)
        assert (families["BPMx"].devices, families["BPMx"].devices_present) == (None, False)

    @pytest.mark.parametrize(
        ("written", "key", "message"),
        [
            (
                "ordinal",
                "families.QF.devices",
                "must be names, address, a list of names, a same_as: entry or null, got 'ordinal'",
            ),
            (
                3,
                "families.QF.devices",
                "must be names, address, a list of names, a same_as: entry or null, got int",
            ),
            ([], "families.QF.devices", "must name at least one device"),
            (["Q_A", 2], "families.QF.devices[1]", "must be a string, got int"),
            (["Q_A", " "], "families.QF.devices[1]", "must name a device, got an empty string"),
            ({"like": "BPMx"}, "families.QF.devices.like", "unknown key"),
            ({"same_as": None}, "families.QF.devices.same_as", "must be a string, got null"),
        ],
    )
    def test_a_devices_answer_of_another_shape_is_refused(
        self, written: Any, key: str, message: str
    ) -> None:
        document = _document()
        document["families"]["QF"]["devices"] = written
        with pytest.raises(MappingError) as caught:
            parse_mapping(document)
        assert (caught.value.key, caught.value.message) == (key, message)

    def test_document_order_is_kept(self) -> None:
        mapping = parse_mapping(_document())
        assert list(mapping.families) == ["QF", "BPMx"]
        assert list(mapping.directions) == ["QF.Setpoint", "QF.Monitor", "BPMx.Monitor"]

    def test_undecided_slots_parse(self) -> None:
        document = _document()
        document["directions"]["QF.Monitor"]["direction"] = None
        document["models"]["SR"]["wiring"]["QF"]["engine"] = None
        document["families"]["QF"]["class"] = None
        mapping = parse_mapping(document)
        assert mapping.directions["QF.Monitor"].direction is None
        assert mapping.models["SR"].wiring["QF"].engine is None

    def test_judgments_parse(self) -> None:
        document = _document()
        document["judgments"] = {
            "QF": {
                "rows_beyond_devices": {"Monitor": {"SR:QF:EXTRA": {"field": "Extra"}}},
                "unbound_devices": {3: "drop", 4: None},
                "shared_pvs": {1: 2, 3: "keep_all"},
            }
        }
        judgments = parse_mapping(document).judgments["QF"]
        assert judgments.rows_beyond["Monitor"]["SR:QF:EXTRA"].name == "Extra"  # type: ignore[union-attr]
        assert judgments.unbound_devices == {3: "drop", 4: None}
        assert judgments.shared_pvs.owners == {1: 2, 3: "keep_all"}  # type: ignore[union-attr]
        assert judgments.shared_pvs_present

    def test_input_is_not_modified(self) -> None:
        document = _document()
        before = copy.deepcopy(document)
        parse_mapping(document)
        assert document == before


class TestRefusals:
    @pytest.mark.parametrize(
        ("edit", "key", "message"),
        [
            (lambda d: d.update(systems={}), "systems", "unknown key"),
            (lambda d: d.pop("models"), "models", "required key is missing"),
            (lambda d: d["facility"].update(token="X"), "facility.token", "unknown key"),
            (lambda d: d["facility"].pop("code"), "facility.code", "required key is missing"),
            (
                lambda d: d["facility"].update(code=None),
                "facility.code",
                "must be a string, got null",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(kind="strength"),
                "models.SR.wiring.QF.kind",
                "unknown key",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(verdict="couple"),
                "models.SR.wiring.QF.verdict",
                "unknown key",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(calibration="spline"),
                "models.SR.wiring.QF.calibration",
                "must be linear, table or null, got 'spline'",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"]["engine"].update(index=-1),
                "models.SR.wiring.QF.engine.index",
                "must be a non-negative integer, got -1",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"]["engine"].update(index=True),
                "models.SR.wiring.QF.engine.index",
                "must be a non-negative integer, got bool",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["BPMx"]["engine"].update(axis="z"),
                "models.SR.wiring.BPMx.engine.axis",
                "must be x or y, got 'z'",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(engine={}),
                "models.SR.wiring.QF.engine",
                "must name an attribute, an index or an axis",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"]["engine"].update(plane=0),
                "models.SR.wiring.QF.engine.plane",
                "unknown key",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(voltage=1.0),
                "models.SR.wiring.QF.voltage",
                "only a family whose engine attribute is Frequency takes a voltage",
            ),
            (
                lambda d: d["models"]["SR"]["wiring"]["QF"].update(engine=None, voltage=1.0),
                "models.SR.wiring.QF.voltage",
                "only a family whose engine attribute is Frequency takes a voltage",
            ),
            *(
                (
                    lambda d, v=voltage: d["models"]["SR"]["wiring"]["QF"].update(
                        engine={"attribute": "Frequency"}, voltage=v
                    ),
                    "models.SR.wiring.QF.voltage",
                    f"must be a positive number of volts, got {shown}",
                )
                for voltage, shown in (
                    (0, "0"),
                    (-3.0, "-3.0"),
                    (float("inf"), "inf"),
                    (True, "bool"),
                    (None, "null"),
                    ("3 MV", "'3 MV'"),
                )
            ),
            (
                lambda d: d["directions"]["QF.Setpoint"].update(direction="both"),
                "directions.QF.Setpoint.direction",
                "must be read, write or null, got 'both'",
            ),
            (
                lambda d: d["directions"].update({"QF": d["directions"]["QF.Setpoint"]}),
                "directions.QF",
                "key must be '<family>.<field>'",
            ),
            (
                lambda d: d["families"]["QF"].update(channels=True),
                "families.QF.channels",
                "must be an integer, got bool",
            ),
        ],
    )
    def test_structure_is_refused_by_path(self, edit: Any, key: str, message: str) -> None:
        document = _document()
        edit(document)
        with pytest.raises(MappingError) as caught:
            parse_mapping(document)
        assert caught.value.key == key
        assert caught.value.message == message
        assert str(caught.value) == f"{key}: {message}"

    def test_a_non_mapping_document_is_refused(self) -> None:
        with pytest.raises(MappingError, match="<document>: must be a mapping, got list"):
            parse_mapping([])  # type: ignore[arg-type]


class TestRead:
    def test_reads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "mapping.yaml"
        path.write_text(yaml.safe_dump(_document(), sort_keys=False), encoding="utf-8")
        assert read_mapping(path).models["SR"].name == "SR"

    def test_bad_yaml_is_a_mapping_error(self, tmp_path: Path) -> None:
        path = tmp_path / "mapping.yaml"
        path.write_text("models: [unclosed\n", encoding="utf-8")
        with pytest.raises(MappingError) as caught:
            read_mapping(path)
        assert caught.value.key == "<document>"
        assert caught.value.message.startswith("is not valid YAML")


class TestRoles:
    def test_write_is_a_setpoint_paired_with_the_monitor(self) -> None:
        roles = field_roles(parse_mapping(_document()))
        assert roles["QF.Setpoint"] == FieldRole(role="setpoint", pair="Monitor")
        assert roles["QF.Monitor"] == FieldRole(role="readback", pair=None)
        assert roles["BPMx.Monitor"] == FieldRole(role="readback", pair=None)

    def test_a_setpoint_without_a_monitor_is_its_own_pair(self) -> None:
        document = _document()
        del document["directions"]["QF.Monitor"]
        roles = field_roles(parse_mapping(document))
        assert roles["QF.Setpoint"] == FieldRole(role="setpoint", pair=None)

    def test_only_the_setpoint_field_takes_the_monitor(self) -> None:
        document = _document()
        document["directions"]["QF.Desired"] = {
            "direction": "write",
            "provenance": "stated",
            "override": False,
        }
        roles = field_roles(parse_mapping(document))
        assert roles["QF.Desired"] == FieldRole(role="setpoint", pair=None)
        assert roles["QF.Setpoint"] == FieldRole(role="setpoint", pair="Monitor")

    def test_a_written_monitor_is_no_pair(self) -> None:
        document = _document()
        document["directions"]["QF.Monitor"]["direction"] = "write"
        roles = field_roles(parse_mapping(document))
        assert roles["QF.Setpoint"] == FieldRole(role="setpoint", pair=None)

    def test_a_null_direction_stops_the_import(self) -> None:
        document = _document()
        document["directions"]["QF.Monitor"]["direction"] = None
        with pytest.raises(ImportStop) as caught:
            field_roles(parse_mapping(document))
        assert caught.value.format_message() == (
            "import mml: mapping-undecided: directions.QF.Monitor.direction: write read or write"
        )


class TestUndecided:
    def test_a_decided_mapping_has_no_open_slot(self) -> None:
        mapping = parse_mapping(_document())
        assert undecided_slots(mapping) == []
        require_decided(mapping)

    def test_every_null_a_reviewer_decides_is_listed_in_document_order(self) -> None:
        document = _document()
        document["models"]["SR"]["description"] = None
        document["models"]["SR"]["wiring"]["QF"]["engine"] = None
        document["models"]["SR"]["wiring"]["BPMx"]["calibration"] = None
        document["families"]["QF"]["class"] = None
        document["families"]["BPMx"]["fields"]["Monitor"]["description"] = None
        document["directions"]["QF.Setpoint"]["direction"] = None
        document["judgments"] = {
            "QF": {"unbound_devices": {3: None}, "shared_pvs": None},
            "BPMx": {"rows_beyond_devices": {"Monitor": {"SR:BPM:SUM": None}}},
        }
        keys = [key for key, _ in undecided_slots(parse_mapping(document))]
        assert keys == [
            "models.SR.description",
            "models.SR.wiring.QF.engine",
            "models.SR.wiring.BPMx.calibration",
            "families.QF.class",
            "families.BPMx.fields.Monitor.description",
            "directions.QF.Setpoint.direction",
            "judgments.QF.unbound_devices.3",
            "judgments.QF.shared_pvs",
            "judgments.BPMx.rows_beyond_devices.Monitor[SR:BPM:SUM]",
        ]

    def test_a_new_class_needs_its_branch(self) -> None:
        document = _document()
        document["families"]["QF"]["class"] = "FocusingTrim"
        document["families"]["QF"]["branch"] = None
        keys = [key for key, _ in undecided_slots(parse_mapping(document))]
        assert keys == ["families.QF.branch"]

    def test_a_vocabulary_class_needs_no_branch(self) -> None:
        document = _document()
        document["families"]["QF"]["branch"] = None
        assert undecided_slots(parse_mapping(document)) == []

    def test_a_family_with_no_channels_needs_no_class(self) -> None:
        document = _document()
        document["families"]["QF"]["class"] = None
        document["families"]["QF"]["channels"] = 0
        assert undecided_slots(parse_mapping(document)) == []

    def test_a_null_devices_answer_is_undecided(self) -> None:
        document = _document()
        document["families"]["QF"]["devices"] = None
        assert undecided_slots(parse_mapping(document)) == [
            (
                "families.QF.devices",
                "write names, address, a list of names or {same_as: <family>}",
            )
        ]

    def test_the_stop_prints_one_line_per_slot(self) -> None:
        document = _document()
        document["models"]["SR"]["wiring"]["QF"]["engine"] = None
        document["directions"]["QF.Setpoint"]["direction"] = None
        with pytest.raises(ImportStop) as caught:
            require_decided(parse_mapping(document))
        assert caught.value.exit_code == 1
        assert caught.value.format_message().splitlines() == [
            "import mml: mapping-undecided: models.SR.wiring.QF.engine: "
            "name the attribute and index, or the axis, the model wires",
            "import mml: mapping-undecided: directions.QF.Setpoint.direction: write read or write",
        ]

    def test_the_stop_is_printed_without_an_error_prefix(self) -> None:
        document = _document()
        document["directions"]["QF.Setpoint"]["direction"] = None
        mapping = parse_mapping(document)

        @click.command()
        def stop() -> None:
            require_decided(mapping)

        result = CliRunner().invoke(stop)
        assert result.exit_code == 1
        assert result.output == (
            "import mml: mapping-undecided: directions.QF.Setpoint.direction: write read or write\n"
        )


def _export() -> dict[str, Any]:
    """The merged export the small document maps: one system, two families."""
    return {
        "_import_order": ["SR"],
        "SR": {
            "QF": {
                "DeviceList": [[1, 1], [1, 2], [2, 1], [2, 2]],
                "Setpoint": {"ChannelNames": ["QF1:SP", "QF2:SP", "QF3:SP", "QF4:SP"]},
                "Monitor": {"ChannelNames": ["QF1:RB", "QF2:RB", "QF3:RB", "QF4:RB"]},
            },
            "BPMx": {
                "DeviceList": [[1, 1], [1, 2]],
                "Monitor": {"ChannelNames": ["BPM1:X", "BPM2:X"]},
            },
        },
    }


def _problems(document: dict[str, Any], ao: dict[str, Any] | None = None) -> list[str]:
    return [str(problem) for problem in check_mapping(parse_mapping(document), ao)]


class TestCheck:
    def test_a_sound_mapping_has_no_problem(self) -> None:
        assert _problems(_document()) == []
        assert _problems(_document(), _export()) == []

    def test_problem_renders_key_then_message(self) -> None:
        assert str(Problem("models.SR.name", "is wrong")) == "models.SR.name: is wrong"

    def test_the_code_is_pn_local(self) -> None:
        document = _document()
        document["facility"]["code"] = "my facility"
        assert _problems(document) == ["facility.code: 'my facility' is not PN_LOCAL"]

    def test_model_names_are_pn_local_and_distinct(self) -> None:
        document = _document()
        document["models"]["sr"] = {"name": "sr", "description": "x", "provenance": "stated"}
        document["models"]["TL"] = {"name": "1tl", "description": "x", "provenance": "stated"}
        document["section_order"] = ["SR", "sr", "1tl"]
        assert _problems(document) == [
            "models.sr.name: 'sr' is models.SR's name up to case",
            "models.TL.name: '1tl' is not PN_LOCAL",
        ]

    def test_section_order_is_a_permutation_of_the_model_names(self) -> None:
        document = _document()
        document["section_order"] = ["SR", "SR", "TL"]
        assert _problems(document) == [
            "section_order[1]: 'SR' is listed twice",
            "section_order[2]: 'TL' is no model name",
        ]
        document["section_order"] = []
        assert _problems(document) == ["section_order: leaves out the model SR"]

    def test_renames_are_pn_local_and_distinct(self) -> None:
        document = _document()
        document["families"]["QF"]["rename"] = "bpmx"
        assert _problems(document) == [
            "families.BPMx: maps to 'BPMx', which families.QF also maps to"
        ]
        document["families"]["QF"]["rename"] = "Q-F"
        assert _problems(document) == ["families.QF.rename: 'Q-F' is not PN_LOCAL"]

    def test_classes_and_branches(self) -> None:
        document = _document()
        document["families"]["QF"]["class"] = "AcceleratorDevice"
        assert _problems(document) == [
            "families.QF.class: 'AcceleratorDevice' is the vocabulary root; name a class under it"
        ]
        document["families"]["QF"]["class"] = "Quadrupole"
        document["families"]["QF"]["branch"] = "Sextupole"
        assert _problems(document) == [
            "families.QF.branch: Quadrupole is a vocabulary class under Magnet, not Sextupole"
        ]
        document["families"]["QF"]["class"] = "TrimQuad"
        document["families"]["QF"]["branch"] = "NoSuchClass"
        assert _problems(document) == [
            "families.QF.branch: 'NoSuchClass' is no vocabulary class and no declared branch"
        ]

    def test_declared_branches(self) -> None:
        document = _document()
        document["branches"] = {
            "Magnet": {"parent": "AcceleratorDevice", "description": None},
            "Loop1": {"parent": "Loop2", "description": None},
            "Loop2": {"parent": "Loop1", "description": None},
            "Orphan": {"parent": "Nowhere", "description": None},
        }
        assert _problems(document) == [
            "branches.Magnet: is a vocabulary class already",
            "branches.Loop1.parent: Loop1 extends itself through Loop2",
            "branches.Loop2.parent: Loop2 extends itself through Loop1",
            "branches.Orphan.parent: 'Nowhere' is no vocabulary class and no declared branch",
        ]

    def test_directions_name_family_fields(self) -> None:
        document = _document()
        document["directions"]["QF.Trim"] = {"direction": "write", "provenance": "stated"}
        document["directions"]["XX.Monitor"] = {"direction": "read", "provenance": "stated"}
        assert _problems(document) == [
            "directions.QF.Trim: QF has no field Trim",
            "directions.XX.Monitor: XX is no family",
        ]

    def test_wiring_names_a_family_field(self) -> None:
        document = _document()
        wiring = document["models"]["SR"]["wiring"]
        wiring["QF"]["element_field"] = "Trim"
        wiring["XX"] = {
            "element_field": "Monitor",
            "engine": {"axis": "y"},
            "calibration": "linear",
        }
        assert _problems(document) == [
            "models.SR.wiring.QF.element_field: QF has no field Trim",
            "models.SR.wiring.XX: XX is no family",
        ]

    def test_judgments_name_families(self) -> None:
        document = _document()
        document["judgments"] = {"XX": {"shared_pvs": "keep_all"}}
        assert _problems(document) == ["judgments.XX: XX is no family"]


class TestCheckAgainstTheExport:
    def test_models_name_exactly_the_exported_systems(self) -> None:
        document = _document()
        ao = _export()
        ao["TL"] = {}
        assert _problems(document, ao) == ["models: leaves out the exported system TL"]
        document["models"]["BR"] = {"name": "BR", "description": "x", "provenance": "stated"}
        document["section_order"].append("BR")
        assert _problems(document, _export()) == ["models.BR: BR is no exported system"]

    def test_a_system_the_export_lacks_is_one_problem(self) -> None:
        document = _document()
        ao = _export()
        ao["TL"] = ao.pop("SR")
        ao["_import_order"] = ["TL"]
        del ao["TL"]["BPMx"]
        assert _problems(document, ao) == [
            "models.SR: SR is no exported system",
            "models: leaves out the exported system TL",
        ]

    def test_families_name_exactly_the_exported_families(self) -> None:
        ao = _export()
        ao["SR"]["DCCT"] = {"Monitor": {"ChannelNames": ["DCCT"]}}
        assert _problems(_document(), ao) == [
            "families: leaves out the exported family DCCT",
            "directions: DCCT.Monitor carries channels and has no direction",
        ]
        del ao["SR"]["DCCT"]
        del ao["SR"]["BPMx"]
        assert _problems(_document(), ao)[:1] == ["families.BPMx: BPMx is no exported family"]

    def test_a_direction_names_an_exported_field(self) -> None:
        document = _document()
        document["families"]["QF"]["fields"]["Trim"] = {"description": "x", "provenance": "stated"}
        document["directions"]["QF.Trim"] = {"direction": "write", "provenance": "stated"}
        assert _problems(document, _export()) == [
            "directions.QF.Trim: the export carries no channels under QF.Trim"
        ]

    def test_a_field_a_judgment_creates_takes_a_direction(self) -> None:
        ao = _export()
        ao["SR"]["QF"]["Monitor"]["ChannelNames"].append("QF5:RB")
        document = _document()
        document["families"]["QF"]["fields"]["Extra"] = {"description": "x", "provenance": "stated"}
        document["directions"]["QF.Extra"] = {"direction": "read", "provenance": "stated"}
        document["judgments"] = {
            "QF": {"rows_beyond_devices": {"Monitor": {"QF5:RB": {"field": "Extra"}}}}
        }
        assert _problems(document, ao) == []
        del document["directions"]["QF.Extra"]
        assert _problems(document, ao) == [
            "judgments.QF.rows_beyond_devices.Monitor[QF5:RB]: creates the field 'Extra' of QF "
            "in SR; add families.QF.fields.Extra and directions.QF.Extra"
        ]

    def test_a_wired_family_is_the_model_system_s_and_its_field_carries_channels(self) -> None:
        document = _document()
        document["models"]["SR"]["wiring"]["BPMx"]["element_field"] = "Setpoint"
        document["families"]["BPMx"]["fields"]["Setpoint"] = {
            "description": "x",
            "provenance": "stated",
        }
        document["directions"]["BPMx.Setpoint"] = {"direction": "write", "provenance": "stated"}
        assert _problems(document, _export()) == [
            "directions.BPMx.Setpoint: the export carries no channels under BPMx.Setpoint",
            "models.SR.wiring.BPMx.element_field: SR carries no channels under BPMx.Setpoint",
        ]

    def test_a_stated_direction_agrees_with_the_export_unless_overridden(self) -> None:
        document = _document()
        document["directions"]["QF.Setpoint"]["direction"] = "read"
        assert _problems(document, _export()) == [
            "directions.QF.Setpoint: stated read, the export votes write; "
            "set override: true to keep it"
        ]
        document["directions"]["QF.Setpoint"]["override"] = True
        assert _problems(document, _export()) == []


def _pending_export() -> dict[str, Any]:
    """The small export pending one of each judgment.

    ``BPM:SUM`` is a row beyond BPMx's two devices, QF's fourth device is
    bound by no channel, and ``PS12`` supplies QF devices 1 and 2.
    """
    ao = _export()
    ao["SR"]["BPMx"]["Monitor"]["ChannelNames"] = ["BPM1:X", "BPM2:X", "BPM:SUM"]
    ao["SR"]["QF"]["Setpoint"]["ChannelNames"] = ["PS12", "PS12", "QF3:SP"]
    ao["SR"]["QF"]["Monitor"]["ChannelNames"] = ["QF1:RB", "QF2:RB", "QF3:RB"]
    return ao


def _answered() -> dict[str, Any]:
    """The small document answering every judgment :func:`_pending_export` pends."""
    document = _document()
    document["judgments"] = {
        "QF": {"unbound_devices": {4: "keep"}, "shared_pvs": {1: 1}},
        "BPMx": {"rows_beyond_devices": {"Monitor": {"BPM:SUM": "drop"}}},
    }
    return document


class TestCheckJudgments:
    def test_answers_to_every_pending_judgment_check_clean(self) -> None:
        assert _problems(_answered(), _pending_export()) == []
        require_decided(parse_mapping(_answered()), _pending_export())

    def test_a_pending_judgment_needs_an_answer_slot(self) -> None:
        document = _document()
        missing = [
            "judgments.QF.unbound_devices.4: is pending in SR and has no answer",
            "judgments.QF.shared_pvs: is pending in SR and has no answer",
            "judgments.BPMx.rows_beyond_devices.Monitor[BPM:SUM]: "
            "is pending in SR and has no answer",
        ]
        assert _problems(document, _pending_export()) == missing
        assert undecided_slots(parse_mapping(document)) == []
        with pytest.raises(ImportStop) as caught:
            require_decided(parse_mapping(document), _pending_export())
        assert caught.value.format_message().splitlines() == [
            "import mml: mapping-undecided: judgments.QF.unbound_devices.4: answer drop or keep",
            "import mml: mapping-undecided: judgments.QF.shared_pvs: "
            "answer keep_all or name each group's owner",
            "import mml: mapping-undecided: judgments.BPMx.rows_beyond_devices.Monitor[BPM:SUM]: "
            "answer drop, device or {field: <name>}",
        ]

    def test_a_devices_answer_names_a_family_or_one_word_names(self) -> None:
        document = _document()
        document["families"]["QF"]["devices"] = {"same_as": "QF"}
        document["families"]["BPMx"]["devices"] = {"same_as": "BPMz"}
        assert _problems(document) == [
            "families.QF.devices: QF cannot take its devices from itself",
            "families.BPMx.devices: BPMz is no family",
        ]
        document["families"]["QF"]["devices"] = ["1Q", "Q-2", "Q_3", "Q 4"]
        document["families"]["BPMx"]["devices"] = {"same_as": "QF"}
        assert _problems(document) == [
            "families.QF.devices[1]: 'Q-2' is not one word of letters, digits and _",
            "families.QF.devices[3]: 'Q 4' is not one word of letters, digits and _",
        ]

    def test_an_answer_names_a_judgment_the_export_pends(self) -> None:
        document = _answered()
        document["judgments"]["QF"]["unbound_devices"][3] = "drop"
        document["judgments"]["BPMx"]["rows_beyond_devices"]["Monitor"]["BPM:DIFF"] = "drop"
        assert _problems(document, _pending_export()) == [
            "judgments.QF.unbound_devices.3: names no unbound device of QF in any system",
            "judgments.BPMx.rows_beyond_devices.Monitor[BPM:DIFF]: "
            "names no row beyond the devices of BPMx in any system",
        ]

    def test_an_owner_is_a_member_of_its_supply_group(self) -> None:
        document = _answered()
        document["judgments"]["QF"]["shared_pvs"] = {1: 3, 5: 5}
        assert _problems(document, _pending_export()) == [
            "judgments.QF.shared_pvs.5: names no supply group of QF in any system",
            "judgments.QF.shared_pvs.1: device 3 is not a member of supply group 1 of QF in SR",
        ]
        document["judgments"]["QF"]["shared_pvs"] = {5: 5}
        assert _problems(document, _pending_export()) == [
            "judgments.QF.shared_pvs.5: names no supply group of QF in any system",
            "judgments.QF.shared_pvs.1: supply group 1 of QF in SR has no owner",
        ]
        document["judgments"]["QF"]["shared_pvs"] = "keep_all"
        assert _problems(document, _pending_export()) == []

    def test_an_owner_map_needs_a_shared_supply(self) -> None:
        document = _document()
        document["judgments"] = {"BPMx": {"shared_pvs": {1: 1}}}
        assert _problems(document, _export()) == [
            "judgments.BPMx.shared_pvs: names no shared supply of BPMx in any system"
        ]

    def test_a_row_answer_the_export_cannot_carry(self) -> None:
        ao = _pending_export()
        ao["SR"]["BPMx"]["Monitor"]["ChannelNames"] = ["BPM1:X", "BPM2:X", "BPM1:X"]
        document = _answered()
        document["judgments"]["BPMx"]["rows_beyond_devices"]["Monitor"] = {"BPM1:X": "device"}
        assert _problems(document, ao) == [
            "judgments.BPMx.rows_beyond_devices.Monitor[BPM1:X]: 'BPM1:X' is also bound below "
            "device 3 of BPMx in SR; answer `drop` or `field:`"
        ]
        document["judgments"]["BPMx"]["rows_beyond_devices"]["Monitor"] = {
            "BPM1:X": {"field": "Monitor"}
        }
        assert _problems(document, ao) == [
            "judgments.BPMx.rows_beyond_devices.Monitor[BPM1:X]: "
            "the field name 'Monitor' is a key BPMx already carries in SR"
        ]
        document["judgments"]["BPMx"]["rows_beyond_devices"]["Monitor"] = {
            "BPM1:X": {"field": "1st"}
        }
        assert _problems(document, ao) == [
            "judgments.BPMx.rows_beyond_devices.Monitor[BPM1:X]: "
            "the field name '1st' for BPMx in SR is not PN_LOCAL"
        ]


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


def _va() -> dict[str, Any]:
    """The export's sampled model facts for the small export's system."""
    return {
        "SR": {
            "families": {
                "QF": {
                    "nominals": ["Setpoint"],
                    "Setpoint": {"calibration": {"kind": "table"}},
                },
                "BPMx": {"nominals": ["Monitor"], "Monitor": {"calibration": {"kind": "linear"}}},
            }
        }
    }


def _typed_export() -> dict[str, Any]:
    ao = _export()
    ao["SR"]["QF"]["AT"] = {"ATType": "QUAD", "ATIndex": [3, 9, 15, 21]}
    ao["SR"]["BPMx"]["AT"] = {"ATType": "BPMx", "ATIndex": [1, 7]}
    return ao


class TestDraft:
    def test_the_draft_parses_and_checks_clean_against_its_export(self) -> None:
        ao = _typed_export()
        document = draft_mapping(ao, {"SR": {"Machine": "Quokka Light Source"}}, _va())
        mapping = parse_mapping(document)
        assert check_mapping(mapping, ao) == []
        assert undecided_slots(mapping) == [
            ("families.QF.branch", "name the class it extends"),
            ("families.BPMx.branch", "name the class it extends"),
        ]

    def test_identity_comes_from_the_accelerator_data(self) -> None:
        document = draft_mapping(_export(), {"SR": {"Machine": "Quokka Light Source"}})
        assert document["facility"] == {
            "code": "Quokka_Light_Source",
            "name": "Quokka Light Source",
            "description": (
                "The Quokka Light Source accelerator facility, as exported by its "
                "MATLAB Middle Layer."
            ),
        }

    def test_no_accelerator_data_writes_no_facility_block(self) -> None:
        assert "facility" not in draft_mapping(_export())

    def test_models_keep_the_raw_system_token_and_take_a_pn_local_name(self) -> None:
        ao = _export()
        ao["transfer-line"] = {}
        ao["_import_order"] = ["SR", "transfer-line"]
        document = draft_mapping(ao)
        assert [(raw, model["name"]) for raw, model in document["models"].items()] == [
            ("SR", "SR"),
            ("transfer-line", "transfer_line"),
        ]
        assert document["section_order"] == ["SR", "transfer_line"]

    def test_directions_come_from_the_export_s_vote(self) -> None:
        directions = draft_mapping(_export())["directions"]
        assert directions == {
            "QF.Setpoint": {"direction": "write", "provenance": "derived", "override": False},
            "QF.Monitor": {"direction": "read", "provenance": "derived", "override": False},
            "BPMx.Monitor": {"direction": "read", "provenance": "derived", "override": False},
        }

    def test_wiring_is_proposed_in_engine_words_from_the_lattice_type(self) -> None:
        wiring = draft_mapping(_typed_export(), va=_va())["models"]["SR"]["wiring"]
        assert wiring == {
            "QF": {
                "element_field": "Setpoint",
                "engine": {"attribute": "PolynomB", "index": 1},
                "calibration": "table",
            },
            "BPMx": {"element_field": "Monitor", "engine": {"axis": "x"}, "calibration": "linear"},
        }

    def test_an_unknown_lattice_type_is_left_to_the_reviewer(self) -> None:
        ao = _typed_export()
        ao["SR"]["QF"]["AT"]["ATType"] = "Wiggler"
        wiring = draft_mapping(ao, va=_va())["models"]["SR"]["wiring"]
        assert wiring["QF"]["engine"] is None

    def test_a_family_bound_to_no_element_is_not_wired(self) -> None:
        ao = _typed_export()
        ao["SR"]["QF"]["AT"]["ATIndex"] = []
        del ao["SR"]["BPMx"]["AT"]
        assert "wiring" not in draft_mapping(ao, va=_va())["models"]["SR"]

    def test_no_sampled_model_facts_propose_no_wiring(self) -> None:
        assert "wiring" not in draft_mapping(_typed_export())["models"]["SR"]

    def test_pending_judgments_are_open_slots(self) -> None:
        ao = _export()
        ao["SR"]["BPMx"]["Monitor"]["ChannelNames"] = ["BPM1:X", "BPM2:X", "BPM:SUM"]
        document = draft_mapping(ao)
        assert document["judgments"] == {
            "BPMx": {"rows_beyond_devices": {"Monitor": {"BPM:SUM": None}}}
        }

    def test_a_family_naming_every_device_is_identified_by_names(self) -> None:
        ao = _export()
        ao["SR"]["QF"]["CommonNames"] = ["QF1", "QF2", "QF3", "QF4"]
        families = draft_mapping(ao)["families"]
        assert families["QF"]["devices"] == "names"
        assert list(families["QF"])[:3] == ["branch", "class", "devices"]

    def test_a_family_naming_no_device_is_identified_by_address(self) -> None:
        ao = _export()
        ao["SR"]["QF"]["CommonNames"] = ["QF1", "QF2", "", "QF4"]
        families = draft_mapping(ao)["families"]
        assert families["QF"]["devices"] == "address"
        assert families["BPMx"]["devices"] == "address"

    def test_a_family_without_channels_takes_no_devices_slot(self) -> None:
        ao = _export()
        ao["SR"]["Screen"] = {"DeviceList": [[1, 1]], "CommonNames": ["Screen1"]}
        assert "devices" not in draft_mapping(ao)["families"]["Screen"]

    def test_the_other_axis_of_one_named_family_is_the_same_devices(self) -> None:
        ao = _export()
        ao["SR"]["BPMx"]["CommonNames"] = ["BPM1", "BPM2"]
        ao["SR"]["BPMy"] = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["BPM1:Y", "BPM2:Y"]},
        }
        families = draft_mapping(ao)["families"]
        assert families["BPMx"]["devices"] == "names"
        assert families["BPMy"]["devices"] == {"same_as": "BPMx"}

    def test_two_named_twins_leave_the_family_to_its_addresses(self) -> None:
        ao = _export()
        ao["SR"]["BPMx"]["CommonNames"] = ["BPM1", "BPM2"]
        ao["SR"]["BPMy"] = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["BPM1:Y", "BPM2:Y"]},
        }
        ao["SR"]["Spare"] = copy.deepcopy(ao["SR"]["BPMx"])
        assert draft_mapping(ao)["families"]["BPMy"]["devices"] == "address"

    def test_a_twin_in_one_system_only_is_not_the_same_devices(self) -> None:
        ao = _export()
        ao["SR"]["BPMx"]["CommonNames"] = ["BPM1", "BPM2"]
        ao["SR"]["BPMy"] = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["BPM1:Y", "BPM2:Y"]},
        }
        ao["TL"] = {"BPMy": {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["TL1:Y"]}}}
        ao["_import_order"] = ["SR", "TL"]
        assert draft_mapping(ao)["families"]["BPMy"]["devices"] == "address"

    def test_the_draft_as_written_lists_the_ids_each_address_answer_yields(self) -> None:
        ao = _export()
        ao["SR"]["QF"]["CommonNames"] = ["QF1", "QF2", "QF3", "QF4"]
        ao["SR"]["BPMx"]["Monitor"]["ChannelNames"] = ["SR:BPM1:X", "SR:BPM2:X"]
        text = draft_text(ao)
        assert yaml.safe_load(text) == draft_mapping(ao)
        lines = text.splitlines()
        at = lines.index("    devices: address")
        assert lines[at + 1] == "    # SR/BPM1, SR/BPM2"
        assert [line for line in lines if line.lstrip().startswith("#")] == [lines[at + 1]]

    def test_a_long_id_comment_wraps_at_the_page_width(self) -> None:
        ao = _export()
        count = 40
        ao["SR"]["BPMx"] = {
            "DeviceList": [[1, n] for n in range(1, count + 1)],
            "Monitor": {"ChannelNames": [f"SR:BPM{n}:X" for n in range(1, count + 1)]},
        }
        lines = draft_text(ao).splitlines()
        comment = [line for line in lines if line.startswith("    # SR/BPM")]
        assert len(comment) > 1
        assert all(len(line) <= 100 for line in comment)
        ids = ", ".join(line.removeprefix("    # ") for line in comment).replace(",,", ",")
        assert ids.split(", ") == [f"SR/BPM{n}" for n in range(1, count + 1)]

    def test_the_nsls2_export_drafts_the_devices_its_fixture_mapping_answers(self) -> None:
        from tests.facility.test_fixture_mappings import _export as fixture_export

        ao, ad, va = fixture_export("nsls2")
        drafted = draft_mapping(ao, ad, va)["families"]
        reviewed = read_mapping(FIXTURES / "nsls2" / MAPPING_FILE).families
        assert drafted["BPMy"]["devices"] == {"same_as": "BPMx"}
        assert {raw for raw, family in drafted.items() if family.get("devices") == "address"} == {
            "RF",
            "TUNE",
            "DCCT",
        }
        answered = parse_mapping(draft_mapping(ao, ad, va)).families
        assert {raw: family.devices for raw, family in answered.items()} == {
            raw: family.devices for raw, family in reviewed.items()
        }

    def test_the_spear3_export_drafts_the_devices_its_fixture_mapping_answers(self) -> None:
        from tests.facility.test_fixture_mappings import _export as fixture_export

        ao, ad, va = fixture_export("spear3")
        drafted = parse_mapping(draft_mapping(ao, ad, va)).families
        reviewed = read_mapping(FIXTURES / "spear3" / MAPPING_FILE).families
        assert {raw: family.devices for raw, family in drafted.items()} == {
            raw: family.devices for raw, family in reviewed.items()
        }
        text = draft_text(ao, ad, va)
        assert "    # StorageRing/VG01_AM1, " in text

    def test_the_dump_is_deterministic_and_round_trips(self) -> None:
        document = draft_mapping(_typed_export(), va=_va())
        text = dump_mapping(document)
        assert text == dump_mapping(draft_mapping(_typed_export(), va=_va()))
        assert yaml.safe_load(text) == document

    def test_a_fixture_export_drafts_a_mapping_that_checks_clean(self) -> None:
        tree = FIXTURES / "spear3"
        flat = json.loads((tree / "spear3.storagering.ao.json").read_text(encoding="utf-8"))
        ad = json.loads((tree / "spear3.storagering.ad.json").read_text(encoding="utf-8"))
        va = json.loads((tree / "spear3.storagering.va.json").read_text(encoding="utf-8"))
        families = {key: body for key, body in flat.items() if not key.startswith("_")}
        ao = {"_import_order": ["StorageRing"], "StorageRing": families}
        document = draft_mapping(ao, {"StorageRing": ad}, {"StorageRing": va})
        mapping = parse_mapping(yaml.safe_load(dump_mapping(document)))
        assert check_mapping(mapping, ao) == []
        assert mapping.identity is not None and mapping.identity.code == "SPEAR3"
        wiring = mapping.models["StorageRing"].wiring
        assert wiring["QF"].engine == EngineBlock(attribute="PolynomB", index=1)
        assert wiring["HCM"].engine == EngineBlock(attribute="KickAngle", index=0)
        assert wiring["BPMy"].engine == EngineBlock(axis="y")
        assert wiring["RF"].engine == EngineBlock(attribute="Frequency")


class TestLoadOrDraft:
    def test_an_absent_mapping_is_drafted_and_stops_the_import(self, tmp_path: Path) -> None:
        with pytest.raises(ImportStop) as caught:
            load_or_draft(tmp_path, _typed_export(), va=_va())
        assert caught.value.exit_code == 1
        assert caught.value.format_message() == (
            "import mml: mapping-draft: imported/mml/mapping.yaml written; review it, then re-run"
        )
        written = tmp_path / MAPPING_FILE
        assert written.read_text(encoding="utf-8") == draft_text(_typed_export(), va=_va())
        assert yaml.safe_load(written.read_text(encoding="utf-8")) == draft_mapping(
            _typed_export(), va=_va()
        )

    def test_a_present_mapping_is_read_and_must_be_decided(self, tmp_path: Path) -> None:
        path = tmp_path / MAPPING_FILE
        path.parent.mkdir(parents=True)
        document = _document()
        path.write_text(dump_mapping(document), encoding="utf-8")
        assert load_or_draft(tmp_path, _export()).models["SR"].name == "SR"

        document["directions"]["QF.Setpoint"]["direction"] = None
        path.write_text(dump_mapping(document), encoding="utf-8")
        with pytest.raises(ImportStop, match="mapping-undecided"):
            load_or_draft(tmp_path, _export())
        assert path.read_text(encoding="utf-8") == dump_mapping(document)
