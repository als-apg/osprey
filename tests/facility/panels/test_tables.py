"""The rule tables: their schema, and the refusal that names file and line."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.facility.panels.tables import (
    COLUMNS,
    OPTIONAL_FILES,
    PanelTableError,
    read_tables,
)
from tests.facility.panels._fixture import copy_tables


def _edit(directory: Path, name: str, lines: list[str]) -> Path:
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _header(name: str) -> str:
    return ",".join(COLUMNS[name])


def test_the_schema_is_the_header_lines_the_facility_repository_writes() -> None:
    assert COLUMNS["positions.csv"] == (
        "thing,frame,x_mm,y_mm,z_mm,ax_rad,ay_rad,az_rad,s_m,side_mm,height_mm,"
        "derived_assumes,box_mm,basis,document,version,state,locator"
    ).split(",")
    assert ",".join(COLUMNS["record_types.csv"]) == (
        "channel,record_type,drawn_as,basis,source,revision,locator"
    )
    assert ",".join(COLUMNS["pairs.csv"]) == (
        "setpoint,readback,evidence,basis,source,revision,locator"
    )
    assert ",".join(COLUMNS["health.csv"]) == (
        "channel,value,state,evidence,basis,source,revision,locator"
    )
    assert ",".join(COLUMNS["excluded.csv"]) == "kind,id,reason,source,revision,locator"
    assert ",".join(COLUMNS["pictures.csv"]) == (
        "device_class,view,part,kind,x_mm,y_mm,z_mm,document,version,state,locator"
    )
    assert OPTIONAL_FILES == frozenset({"pictures.csv", "excluded.csv"})


def test_the_fixture_tables_read(panel_tables: Path) -> None:
    tables = read_tables(panel_tables)
    assert len(tables.positions) == 9
    assert len(tables.record_types) == 6
    assert len(tables.pairs) == 3
    assert len(tables.health) == 7
    assert len(tables.excluded) == 2
    assert len(tables.pictures) == 2
    rack = next(pose for pose in tables.positions if pose.thing == "XR/XR01/RK1")
    assert rack.s_m is None
    assert rack.az_rad == pytest.approx(1.570796)
    assert rack.locator == "sheet 1, row 2"
    pump = next(pose for pose in tables.positions if pose.thing == "XR01:IP1")
    assert pump.ax_rad is None
    assert pump.derived_assumes is None
    inferred = next(pair for pair in tables.pairs if pair.setpoint == "XR01:PS1:On-Cmd")
    assert inferred.basis == "inferred"
    assert inferred.revision is None
    assert {rule.value for rule in tables.health} >= {"0", "1", "!=0"}


def test_a_missing_optional_file_is_empty(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    for name in OPTIONAL_FILES:
        (directory / name).unlink()
    tables = read_tables(directory)
    assert tables.pictures == ()
    assert tables.excluded == ()


@pytest.mark.parametrize("name", ["positions.csv", "record_types.csv", "pairs.csv", "health.csv"])
def test_a_missing_required_file_is_refused(tmp_path: Path, name: str) -> None:
    directory = copy_tables(tmp_path / "panels")
    (directory / name).unlink()
    with pytest.raises(PanelTableError, match=name) as raised:
        read_tables(directory)
    assert raised.value.file == name
    assert raised.value.line is None


def test_an_extra_column_is_refused_at_line_one(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    _edit(directory, "pairs.csv", [_header("pairs.csv") + ",note"])
    with pytest.raises(PanelTableError, match=r"pairs\.csv line 1.*note") as raised:
        read_tables(directory)
    assert raised.value.line == 1


def test_a_missing_column_is_refused_at_line_one(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    columns = [column for column in COLUMNS["health.csv"] if column != "evidence"]
    _edit(directory, "health.csv", [",".join(columns)])
    with pytest.raises(PanelTableError, match=r"health\.csv line 1.*evidence"):
        read_tables(directory)


@pytest.mark.parametrize(
    ("name", "row", "word"),
    [
        ("health.csv", "XR01:IP1:State-Sts,0,green,screen,stated,s,r,l", "state"),
        ("health.csv", "XR01:IP1:State-Sts,,bad,screen,stated,s,r,l", "value"),
        ("health.csv", "XR01:IP1:State-Sts,0,bad,screen,stated,s,r", "fields"),
        ("health.csv", "XR01:IP1:State-Sts,0,bad,screen,stated,s,r,l,extra", "fields"),
        ("record_types.csv", "XR01:X,ai,gauge,stated,s,r,l", "drawn_as"),
        ("pairs.csv", "XR01:X-SP,XR01:X-RB,hardware,guessed,s,r,l", "basis"),
        ("excluded.csv", "cable,XR01:X,reason,s,r,l", "kind"),
        ("pictures.csv", "Quadrupole,top,pole 1,pole,1,2,3,d,v,s,l", "view"),
        ("pictures.csv", "Quadrupole,end,pole 1,pole,one,2,3,d,v,s,l", "x_mm"),
        (
            "positions.csv",
            "XR01:X,XR-RING,1,2,3,0,0,0,far,0,0,,,stated,d,v,s,l",
            "s_m",
        ),
        ("positions.csv", "XR01:X,,1,2,3,0,0,0,1,0,0,,,stated,d,v,s,l", "frame"),
    ],
)
def test_a_malformed_row_is_refused_naming_file_and_line(
    tmp_path: Path, name: str, row: str, word: str
) -> None:
    directory = copy_tables(tmp_path / "panels")
    original = (directory / name).read_text(encoding="utf-8").splitlines()
    _edit(directory, name, [*original[:2], row, *original[2:]])
    with pytest.raises(PanelTableError, match=rf"{re.escape(name)} line 3\b.*{word}") as raised:
        read_tables(directory)
    assert (raised.value.file, raised.value.line) == (name, 3)


def test_the_line_counts_a_quoted_field_spanning_two_lines(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    header = _header("pairs.csv")
    _edit(
        directory,
        "pairs.csv",
        [
            header,
            'XR01:A-SP,XR01:A-RB,hardware,stated,s,r,"two',
            'lines"',
            "XR01:B-SP,XR01:B-RB,hardware,maybe,s,r,l",
        ],
    )
    with pytest.raises(PanelTableError, match=r"pairs\.csv line 4\b"):
        read_tables(directory)


@pytest.mark.parametrize(
    ("name", "row"),
    [
        ("pairs.csv", "XR01:QF1:Current-SP,XR01:QD1:Current-RB,hardware,stated,s,r,l"),
        ("record_types.csv", "XR01:QF1:Current-SP,ao,setpoint,stated,s,r,l"),
        (
            "positions.csv",
            "XR01:QF1,XR-RING,1,2,3,0,0,0,1,0,0,,,stated,d,v,s,l",
        ),
    ],
)
def test_a_second_row_for_one_key_is_refused(tmp_path: Path, name: str, row: str) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / name).read_text(encoding="utf-8").splitlines()
    _edit(directory, name, [*lines, row])
    with pytest.raises(PanelTableError, match=rf"line {len(lines) + 1}\b.*line 2\b"):
        read_tables(directory)
