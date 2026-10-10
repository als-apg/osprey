"""The list layer: a CSV of channel addresses imported as facility sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.facility.layers.list import LAYER_DIR, ListSourceInvalid, import_list


def _csv(tmp_path: Path, text: str, name: str = "list.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _import(tmp_path: Path, text: str) -> tuple[Path, list[Path]]:
    facility_dir = tmp_path / "facility"
    return facility_dir, import_list(_csv(tmp_path, text), facility_dir)


def _channels(facility_dir: Path) -> list[dict[str, Any]]:
    return yaml.safe_load((facility_dir / LAYER_DIR / "channels.yaml").read_text("utf-8"))


def _refused(tmp_path: Path, text: str) -> list[str]:
    with pytest.raises(ListSourceInvalid) as stop:
        _import(tmp_path, text)
    assert not (tmp_path / "facility" / LAYER_DIR).exists()
    return [error.format_message() for error in stop.value.errors]


class TestColumns:
    def test_every_column_is_written_onto_its_channel_record(self, tmp_path: Path) -> None:
        facility_dir, written = _import(
            tmp_path,
            "address,role,pair,device,place,unit,description,tags\n"
            "Q1:SP,setpoint,Q1:RB,SR/Q1,,A,Quadrupole current,magnet; sr\n"
            "Q1:RB,,,SR/Q1,,A,,\n"
            "HALL:T,readback,,,HALL,degC,,\n",
        )

        assert written == [facility_dir / LAYER_DIR / "channels.yaml"]
        assert _channels(facility_dir) == [
            {"id": "HALL:T", "role": "readback", "on": {"place": "HALL"}, "unit": "degC"},
            {"id": "Q1:RB", "on": {"device": "SR/Q1"}, "unit": "A"},
            {
                "id": "Q1:SP",
                "role": "setpoint",
                "pair": "Q1:RB",
                "on": {"device": "SR/Q1"},
                "unit": "A",
                "description": "Quadrupole current",
                "tags": ["magnet", "sr"],
            },
        ]

    def test_the_columns_may_come_in_any_order(self, tmp_path: Path) -> None:
        facility_dir, _written = _import(tmp_path, "unit,address\nmA,DCCT:I\n")
        assert _channels(facility_dir) == [{"id": "DCCT:I", "unit": "mA"}]

    def test_a_file_without_a_header_is_one_address_per_line(self, tmp_path: Path) -> None:
        facility_dir, _written = _import(tmp_path, "LAB:TEMP:02\n\nLAB:TEMP:01\n")
        assert _channels(facility_dir) == [{"id": "LAB:TEMP:01"}, {"id": "LAB:TEMP:02"}]

    def test_a_byte_order_mark_is_not_part_of_the_header(self, tmp_path: Path) -> None:
        facility_dir, _written = _import(tmp_path, "﻿address,unit\nX:I,A\n")
        assert _channels(facility_dir) == [{"id": "X:I", "unit": "A"}]

    def test_the_layer_seeds_nothing(self, tmp_path: Path) -> None:
        facility_dir, _written = _import(tmp_path, "address\nX:I\n")
        assert sorted(p.relative_to(facility_dir).as_posix() for p in facility_dir.rglob("*")) == [
            "imported",
            "imported/list",
            "imported/list/channels.yaml",
        ]


class TestRefusals:
    def test_a_row_naming_both_a_device_and_a_place_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,device,place\nQ1:SP,SR/Q1,SR\n") == [
            "facility: source-invalid: channel Q1:SP — list.csv row 2 names both a device "
            "and a place; fix: keep one of `device` and `place` in list.csv"
        ]

    def test_an_unknown_role_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,role\nQ1:SP,write\n") == [
            "facility: source-invalid: channel Q1:SP — list.csv row 2 states role write; "
            "fix: write setpoint, readback or none, or leave it empty"
        ]

    def test_an_unknown_column_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,colour\nQ1:SP,red\n") == [
            "facility: source-invalid: path list.csv — unknown column `colour`; fix: remove "
            "the column; the columns are address, role, pair, device, place, unit, "
            "description, tags, s, model"
        ]

    def test_a_row_without_an_address_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,unit\n,A\n") == [
            "facility: source-invalid: path list.csv — row 2 has no address; fix: fill "
            "`address` or remove the row"
        ]

    def test_an_address_listed_twice_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address\nX:I\nX:I\n") == [
            "facility: source-invalid: channel X:I — list.csv rows 2 and 3 both state it; "
            "fix: keep one row per address"
        ]

    def test_a_headerless_line_of_several_cells_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "X:I,A\n") == [
            "facility: source-invalid: path list.csv — row 1 holds 2 cells and the file has "
            "no header; fix: add a header row naming `address`, or write one address per line"
        ]

    def test_a_file_that_is_not_utf8_is_source_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "list.csv"
        path.write_bytes(b"address\nT\xe9MP\n")
        with pytest.raises(ListSourceInvalid) as stop:
            import_list(path, tmp_path / "facility")
        assert [error.kind for error in stop.value.errors] == ["source-invalid"]

    def test_every_refusal_is_reported(self, tmp_path: Path) -> None:
        lines = _refused(tmp_path, "address,role,device,place\nA,write,,\nB,,D,P\n")
        assert [line.split(" — ")[0] for line in lines] == [
            "facility: source-invalid: channel A",
            "facility: source-invalid: channel B",
        ]


def _devices(facility_dir: Path) -> list[dict[str, Any]]:
    return yaml.safe_load((facility_dir / LAYER_DIR / "devices.yaml").read_text("utf-8"))


class TestPositions:
    def test_a_stated_s_and_model_write_a_device_record(self, tmp_path: Path) -> None:
        facility_dir, written = _import(
            tmp_path,
            "address,role,device,s,model\n"
            "G1:SP,setpoint,L/G1,3.5,LINE\n"
            "G1:RB,,L/G1,3.5,\n"
            "G2:RB,,L/G2,1,\n"
            "G3:RB,,L/G3,,\n",
        )

        assert written == [
            facility_dir / LAYER_DIR / "channels.yaml",
            facility_dir / LAYER_DIR / "devices.yaml",
        ]
        assert _devices(facility_dir) == [
            {"id": "L/G1", "s": 3.5, "model": "LINE"},
            {"id": "L/G2", "s": 1.0},
        ]
        assert [channel["on"] for channel in _channels(facility_dir)] == [
            {"device": "L/G1"},
            {"device": "L/G1"},
            {"device": "L/G2"},
            {"device": "L/G3"},
        ]

    def test_an_import_stating_no_s_leaves_no_devices_file(self, tmp_path: Path) -> None:
        facility_dir, _written = _import(tmp_path, "address,device,s\nG1:RB,L/G1,2\n")
        (tmp_path / "list.csv").write_text("address,device\nG1:RB,L/G1\n", encoding="utf-8")

        written = import_list(tmp_path / "list.csv", facility_dir)

        assert written == [facility_dir / LAYER_DIR / "channels.yaml"]
        assert not (facility_dir / LAYER_DIR / "devices.yaml").exists()

    def test_s_on_a_row_with_no_device_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,place,s,model\nT:RB,HALL,2,LINE\n") == [
            "facility: source-invalid: channel T:RB — list.csv row 2 states `s`, `model` with "
            "no device; a position belongs to a device; fix: fill `device`, or empty `s` "
            "and `model`"
        ]

    def test_model_on_a_row_with_no_device_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,model\nT:RB,LINE\n") == [
            "facility: source-invalid: channel T:RB — list.csv row 2 states `model` with no "
            "device; a position belongs to a device; fix: fill `device`, or empty `model`"
        ]

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "3.5m"])
    def test_an_s_that_is_not_a_finite_number_is_source_invalid(
        self, tmp_path: Path, value: str
    ) -> None:
        assert _refused(tmp_path, f"address,device,s\nG1:RB,L/G1,{value}\n") == [
            f"facility: source-invalid: channel G1:RB — list.csv row 2 states s {value}, "
            "not a finite number; fix: write `s` in metres as a number"
        ]

    def test_rows_disagreeing_on_a_devices_s_are_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(
            tmp_path, "address,device,s,model\nG1:SP,L/G1,3.5,LINE\nG1:RB,L/G1,4,LINE\n"
        ) == [
            "facility: source-invalid: device L/G1 — list.csv rows 2 and 3 state s 3.5 and 4; "
            "fix: state one `s` for the device"
        ]

    def test_rows_disagreeing_on_a_devices_model_are_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(
            tmp_path, "address,device,s,model\nG1:SP,L/G1,3.5,LINE\nG1:RB,L/G1,3.5,SR\n"
        ) == [
            "facility: source-invalid: device L/G1 — list.csv rows 2 and 3 state model LINE "
            "and SR; fix: state one `model` for the device"
        ]

    def test_a_model_without_an_s_is_source_invalid(self, tmp_path: Path) -> None:
        assert _refused(tmp_path, "address,device,model\nG1:RB,L/G1,LINE\n") == [
            "facility: source-invalid: device L/G1 — list.csv row 2 states model LINE and no "
            "row states its s; fix: fill `s` on a row of the device, or empty `model`"
        ]

    def test_a_listed_s_off_the_deck_of_a_wired_device_is_a_place_conflict(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("at")
        from osprey.facility.build import LATER_STAGES
        from osprey.facility.validate import run_stages
        from tests.facility._synthetic_trees import deck_tree, write_tree

        facility_dir = write_tree(tmp_path / "facility", deck_tree())
        listing = _csv(tmp_path, "address,device,s,model\nLQ:RB,LINE/Q1,1.5,LINE\n")
        import_list(listing, facility_dir)

        result = run_stages(facility_dir, project_name="p", later=LATER_STAGES)

        assert [error.format_message() for error in result.errors] == [
            "facility: place-conflict: device LINE/Q1 — layer list states s 1.5 in model LINE, "
            "but the deck puts the device at s 1; fix: drop `s` from the layer, or add a fix "
            "`set` of the deck's value"
        ]


class TestCommand:
    def test_import_list_writes_the_layer_under_the_repo(self, tmp_path: Path) -> None:
        from osprey.cli.main import cli
        from tests._builds import init_project

        repo = init_project(tmp_path, "hello-world", "demo")
        listing = _csv(tmp_path, "address\nLAB:TEMP:01\n")

        result = _invoke(cli, listing, repo)

        assert result.exit_code == 0, result.output
        assert result.stdout == f"wrote data/facility/{LAYER_DIR}/channels.yaml\n"
        assert _channels(repo / "data" / "facility") == [{"id": "LAB:TEMP:01"}]

    def test_import_list_prints_each_refusal_and_exits_1(self, tmp_path: Path) -> None:
        from osprey.cli.main import cli
        from tests._builds import init_project

        repo = init_project(tmp_path, "hello-world", "demo")
        listing = _csv(tmp_path, "address,device,place\nQ1:SP,SR/Q1,SR\n", name="two.csv")

        result = _invoke(cli, listing, repo)

        assert result.exit_code == 1
        assert result.stderr == (
            "facility: source-invalid: channel Q1:SP — two.csv row 2 names both a device and "
            "a place; fix: keep one of `device` and `place` in two.csv\n"
        )
        assert not (repo / "data" / "facility" / LAYER_DIR).exists()


def _invoke(cli: Any, listing: Path, repo: Path) -> Result:
    return CliRunner().invoke(
        cli, ["facility", "import", "list", str(listing), "--repo", str(repo)]
    )
