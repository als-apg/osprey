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
        "device_class,view,part,kind,x_mm,y_mm,z_mm,document,version,state,locator,"
        "w_mm,h_mm,angle_deg,signal,under,devices"
    )
    assert ",".join(COLUMNS["names.csv"]) == (
        "thing,short_name,scheme,basis,document,version,state,locator"
    )
    assert ",".join(COLUMNS["metrics.csv"]) == (
        "view,level,order,title,measure,signal,class,unit,caption,source,precision"
    )
    assert ",".join(COLUMNS["calibration_kinds.csv"]) == (
        "kind,title,class,shape,unit,wanted_basis,tolerance,tolerance_source,held_in,order,under,short"
    )
    assert ",".join(COLUMNS["calibrations.csv"]) == (
        "id,kind,subject,value,unit,conditions,basis,method,date,document,version,state,"
        "supersedes,standing"
    )
    assert ",".join(COLUMNS["calibration_curves.csv"]) == "id,x,y"
    assert ",".join(COLUMNS["state_labels.csv"]) == "channel,value,label,source,revision,locator"
    assert ",".join(COLUMNS["units.csv"]) == "channel,unit,scale,label,source,revision,locator"
    assert ",".join(COLUMNS["channel_roles.csv"]) == (
        "class,role,signal,scope,shown_as,unit,scale,press_value,label,basis,source,revision,"
        "locator"
    )
    assert ",".join(COLUMNS["card_sections.csv"]) == (
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source"
    )
    assert ",".join(COLUMNS["chains.csv"]) == (
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source"
    )
    assert ",".join(COLUMNS["symbols.csv"]) == "class,devices,symbol,lane,source"
    assert ",".join(COLUMNS["card_recipes.csv"]) == (
        "class,under,devices,recipe,description,channels_cols,source"
    )
    assert ",".join(COLUMNS["no_cards.csv"]) == (
        "class,under,devices,owner_relation,owner_direction,owner_to,reason,source,when,attribute"
    )
    assert ",".join(COLUMNS["fronts.csv"]) == (
        "recipe,face,group,order,connector,kind,count,channel,far_relation,far_direction,"
        "far_port,label,source,roles,apart"
    )
    assert OPTIONAL_FILES == frozenset(
        {
            "pictures.csv",
            "excluded.csv",
            "names.csv",
            "metrics.csv",
            "calibration_kinds.csv",
            "calibrations.csv",
            "calibration_curves.csv",
            "probe_pictures.csv",
            "state_labels.csv",
            "units.csv",
            "channel_roles.csv",
            "card_recipes.csv",
            "card_sections.csv",
            "chains.csv",
            "symbols.csv",
            "fronts.csv",
            "no_cards.csv",
        }
    )


def test_the_fixture_tables_read(panel_tables: Path) -> None:
    tables = read_tables(panel_tables)
    assert len(tables.positions) == 13
    assert len(tables.record_types) == 6
    assert len(tables.pairs) == 3
    assert len(tables.health) == 17
    assert len(tables.excluded) == 3
    assert len(tables.pictures) == 2
    rack = next(pose for pose in tables.positions if pose.thing == "XR/XR01/RK1")
    assert rack.s_m is None
    assert rack.az_rad == pytest.approx(1.570796)
    assert rack.locator == "sheet 1, row 2"
    pump = next(pose for pose in tables.positions if pose.thing == "XR01:IP1")
    assert pump.ax_rad is None
    assert pump.derived_assumes is None
    inferred = next(pair for pair in tables.pairs if pair.setpoint == "XR01-PS1:On-Cmd")
    assert inferred.basis == "inferred"
    assert inferred.revision is None
    assert {rule.value for rule in tables.health} >= {"0", "1", "!=0"}


def test_a_missing_optional_file_is_empty(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    for name in OPTIONAL_FILES:
        (directory / name).unlink(missing_ok=True)
    tables = read_tables(directory)
    assert tables.pictures == ()
    assert tables.excluded == ()
    assert tables.names == ()
    assert tables.calibrations == () and tables.calibration_kinds == ()


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
        ("health.csv", "XR01-IP1:State-Sts,0,green,screen,stated,s,r,l", "state"),
        ("health.csv", "XR01-IP1:State-Sts,,bad,screen,stated,s,r,l", "value"),
        ("health.csv", "XR01-IP1:State-Sts,0,bad,screen,stated,s,r", "fields"),
        ("health.csv", "XR01-IP1:State-Sts,0,bad,screen,stated,s,r,l,extra", "fields"),
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
        ("pairs.csv", "XR01-QF1:Current-SP,XR01-QD1:Current-RB,hardware,stated,s,r,l"),
        ("record_types.csv", "XR01-QF1:Current-SP,ao,setpoint,stated,s,r,l"),
        (
            "positions.csv",
            "XR01:QF1,XR-RING,1,2,3,0,0,0,1,0,0,,,stated,d,v,s,l",
        ),
        ("names.csv", "XR01:QF1,QF,controls,stated,d,v,s,l"),  # one name per thing
    ],
)
def test_a_second_row_for_one_key_is_refused(tmp_path: Path, name: str, row: str) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / name).read_text(encoding="utf-8").splitlines()
    _edit(directory, name, [*lines, row])
    with pytest.raises(PanelTableError, match=rf"line {len(lines) + 1}\b.*line 2\b"):
        read_tables(directory)


def test_a_picture_part_may_be_of_any_kind_the_facility_names(tmp_path: Path) -> None:
    from tests.facility.panels._fixture import copy_tables

    tables = copy_tables(tmp_path / "panels")
    with (tables / "pictures.csv").open("a") as rows:
        rows.write("Quadrupole,end,probe 1,a_kind_of_this_facility,1.0,2.0,0.0,,,,\n")
    assert read_tables(tables).pictures[-1].kind == "a_kind_of_this_facility"


def test_a_short_name_row_reads_with_its_citation(panel_tables: Path) -> None:
    (first, *_rest) = read_tables(panel_tables).names
    assert (first.thing, first.short_name, first.scheme, first.basis) == (
        "XR01:QF1",
        "F1",
        "controls",
        "stated",
    )
    assert (first.document, first.version, first.locator) == (
        "XR-NAMES-001",
        "A.1",
        "sheet 1 row 2",
    )


def test_a_short_name_must_be_stated(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    _edit(directory, "names.csv", [_header("names.csv"), "XR01:QF1,,controls,stated,d,v,s,l"])
    with pytest.raises(PanelTableError, match="short_name is empty"):
        read_tables(directory)


def test_the_calibration_tables_read(panel_tables: Path) -> None:
    tables = read_tables(panel_tables)
    kinds = {(kind.kind, kind.cls): kind for kind in tables.calibration_kinds}
    assert kinds["integrated_gradient", "Quadrupole"].tolerance == ("rel", pytest.approx(0.002))
    assert kinds["centre", "Quadrupole"].tolerance == ("abs", pytest.approx(5.0))
    assert kinds["strength_scale", "Quadrupole"].tolerance is None
    assert kinds["current_offset", "Quadrupole"].held_in == "current_setpoint"
    assert kinds["centre", "Dipole"].under == "XR/XR02"
    assert kinds["centre", "Quadrupole"].under is None
    bench = next(c for c in tables.calibrations if c.id == "ig/QF1/bench")
    assert bench.supersedes == "ig/QF/design" and bench.basis == "bench"
    assert [(p.x, p.y) for p in tables.calibration_curves] == [(0.0, -0.02376), (73.33, -4.9198)]


_CLAIM = "x,integrated_gradient,XR01:QF1,1.0,T,,bench,,2024-01-01,D,A,Released,,accepted"


@pytest.mark.parametrize(
    ("name", "row", "says"),
    [
        ("calibrations.csv", _CLAIM.replace("integrated_gradient", "nope"), "no stated kind"),
        ("calibrations.csv", _CLAIM.replace(",,accepted", ",ghost,accepted"), "no claim"),
        ("calibrations.csv", _CLAIM.replace("bench", "guess"), "basis is 'guess'"),
        ("calibrations.csv", _CLAIM.replace("accepted", "maybe"), "standing is 'maybe'"),
        ("calibration_curves.csv", "x,1,2", "no claim whose value is a curve"),
        ("calibration_kinds.csv", "k,K,Quadrupole,blob,,bench,,,,1,", "shape is 'blob'"),
        ("calibration_kinds.csv", "k,K,Quadrupole,scalar,,bench,0.01,,,1,", "rel:<fraction>"),
        ("calibration_kinds.csv", "k,K,Quadrupole,scalar,,bench,rel:,,,1,", "rel:<fraction>"),
        ("calibration_kinds.csv", "k,K,Quadrupole,scalar,,bench,abs:x,,,1,", "rel:<fraction>"),
        ("calibration_kinds.csv", "k,K,Quadrupole,scalar,,bench,abs:-1,,,1,", "rel:<fraction>"),
        ("calibration_kinds.csv", "k,K,Quadrupole,curve,A,bench,,,,1,", "x;y"),
        ("calibration_kinds.csv", "k,K,Quadrupole,curve,A;T;s,bench,,,,1,", "x;y"),
        ("calibration_kinds.csv", "k,K,Quadrupole,scalar,A;T,bench,,,,1,", "only a curve"),
        ("calibration_kinds.csv", "centre,C,Quadrupole,tuple,um,survey,,,,9,", "already has"),
    ],
)
def test_a_calibration_row_that_does_not_hold_is_refused(tmp_path, name, row, says) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / name).read_text().splitlines()
    if name == "calibration_curves.csv":
        claims = (directory / "calibrations.csv").read_text().splitlines()
        _edit(directory, "calibrations.csv", [*claims, _CLAIM])
    _edit(directory, name, [*lines, row])
    with pytest.raises(PanelTableError, match=says) as raised:
        read_tables(directory)
    assert raised.value.file == name and raised.value.line == len(lines) + 1


def test_a_calibration_id_must_be_unique(tmp_path) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / "calibrations.csv").read_text().splitlines()
    _edit(directory, "calibrations.csv", [*lines, lines[1]])
    with pytest.raises(PanelTableError, match="already has a row"):
        read_tables(directory)


def test_a_kind_id_may_name_kinds_of_several_classes(panel_tables: Path) -> None:
    kinds = read_tables(panel_tables).calibration_kinds
    assert [k.cls for k in kinds if k.kind == "centre"] == ["Quadrupole", "Dipole"]


def test_the_under_column_may_be_left_out_of_the_header(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / "calibration_kinds.csv").read_text().splitlines()
    _edit(directory, "calibration_kinds.csv", [line.rsplit(",", 1)[0] for line in lines])
    kinds = read_tables(directory).calibration_kinds
    assert len(kinds) == len(lines) - 1 and all(k.under is None for k in kinds)


def test_a_metric_may_state_its_decimals_and_nothing_else_there(tmp_path) -> None:
    from osprey.facility.panels.tables import PanelTableError

    tables = copy_tables(tmp_path / "ok")
    path = tables / "metrics.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0] + ",precision", lines[1] + ",2"]) + "\n")
    (metric,) = read_tables(tables).metrics
    assert metric.precision == 2
    path.write_text("\n".join([lines[0] + ",precision", lines[1] + ",1.5"]) + "\n")
    with pytest.raises(PanelTableError, match="precision"):
        read_tables(tables)


def test_state_labels_read_one_word_per_channel_and_value(tmp_path) -> None:
    from osprey.facility.panels.tables import STATE_LABELS

    tables = copy_tables(tmp_path / "panels")
    assert read_tables(tables).state_labels == ()  # absent: no words
    header = "channel,value,label,source,revision,locator\n"
    (tables / STATE_LABELS).write_text(header + "X:A,0,Closed,S,1,l\nX:A,1,Open,S,1,l\n")
    assert [(r.value, r.label) for r in read_tables(tables).state_labels] == [
        ("0", "Closed"),
        ("1", "Open"),
    ]
    (tables / STATE_LABELS).write_text(header + "X:A,1,Open,S,1,l\nX:A,1,Shut,S,1,l\n")
    with pytest.raises(PanelTableError, match="already has a row"):
        read_tables(tables)


def test_the_graph_finds_a_state_word_for_a_value_however_it_is_written(tmp_path) -> None:
    from osprey.facility.panels.model import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "state_labels.csv").write_text(
        "channel,value,label,source,revision,locator\nXR01-PS1:On-Cmd,1,On,S,1,l\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    for value in (1, 1.0, "1", " 1 "):
        assert graph.state_label("XR01-PS1:On-Cmd", value) == "On"
    assert graph.state_label("XR01-PS1:On-Cmd", 0) is None


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ("Pump,start,*:Go,device,action,,,1,,stated,S,1,l", "names its press in label"),
        ("Pump,p,*:IP{n}:P,device,value,,,,,stated,S,1,l", "come from a walked relation"),
        ("Pump,p,*:P,device,value,,0,,,stated,S,1,l", "scale is 0"),
        ("Pump,p,*:P,beside,value,,,,,stated,S,1,l", "scope is 'beside'"),
    ],
)
def test_a_channel_role_row_that_cannot_be_drawn_is_refused(tmp_path, row: str, why: str):
    from osprey.facility.panels.tables import CHANNEL_ROLES

    tables = copy_tables(tmp_path / "panels")
    (tables / CHANNEL_ROLES).write_text(",".join(COLUMNS[CHANNEL_ROLES]) + "\n" + row + "\n")
    with pytest.raises(PanelTableError, match=re.escape(why)):
        read_tables(tables)


def test_a_card_key_number_names_its_class_and_shows_a_value_or_a_word(tmp_path):
    tables = copy_tables(tmp_path / "panels")
    path = tables / "metrics.csv"
    base = path.read_text()
    path.write_text(base + "card,,1,Pressure,word,pressure,Pump,,,S\n")
    assert [m.measure for m in read_tables(tables).metrics if m.view == "card"] == ["word"]
    path.write_text(base + "card,,1,Pressure,rms,pressure,Pump,,,S\n")
    with pytest.raises(PanelTableError, match="one of value, word, count_true, max, sum"):
        read_tables(tables)
    path.write_text(base + "card,,1,Pressure,value,pressure,,,,S\n")
    with pytest.raises(PanelTableError, match="names its class"):
        read_tables(tables)
    path.write_text(base + "place,,9,Pressure,word,pressure,Pump,,,S\n")
    with pytest.raises(PanelTableError, match="only a card's key number is a word"):
        read_tables(tables)


@pytest.mark.parametrize(
    "far",
    ["powered_by,", ",in"],
)
def test_a_front_row_with_one_of_far_relation_and_far_direction_is_refused(
    tmp_path: Path, far: str
) -> None:
    directory = copy_tables(tmp_path / "panels")
    (directory / "fronts.csv").write_text(
        _header("fronts.csv")
        + "\nsupply,front,outputs,1,Out1,power,,,powered_by,in,,,S,,"
        + f"\nsupply,front,outputs,2,Out2,power,,,{far},,,S,,\n"
    )
    with pytest.raises(PanelTableError, match=r"fronts\.csv line 3\b.*both given or both empty"):
        read_tables(directory)


def test_a_front_row_with_neither_far_relation_nor_far_direction_is_read(tmp_path: Path) -> None:
    directory = copy_tables(tmp_path / "panels")
    (directory / "fronts.csv").write_text(
        _header("fronts.csv") + "\nsupply,front,inputs,1,In1,io,,,,,,,S,,\n"
    )
    (row,) = read_tables(directory).fronts
    assert (row.far_relation, row.far_direction) == (None, None)
