"""Tests for the mml layer's mapping loader (``osprey.facility.layers.mml.mapping``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey.facility.layers.mml.mapping import (
    EngineBlock,
    FieldRole,
    Identity,
    ImportStop,
    MappingError,
    Problem,
    WiringFamily,
    check_mapping,
    field_roles,
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

    def test_a_model_without_wiring_wires_nothing(self) -> None:
        document = _document()
        del document["models"]["SR"]["wiring"]
        assert parse_mapping(document).models["SR"].wiring == {}

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
