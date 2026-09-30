"""Tests for the mml layer's mapping loader (``osprey.facility.layers.mml.mapping``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.mapping import (
    EngineBlock,
    Identity,
    MappingError,
    WiringFamily,
    parse_mapping,
    read_mapping,
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
                "class": "BPM",
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
