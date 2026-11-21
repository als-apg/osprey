"""The probe picture: a pickup's cross-section on its card, from ``probe_pictures.csv``."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.facility.panels.tables import COLUMNS, PanelTableError
from tests.facility.panels._fixture import (
    PANELS_DIR,
    copy_tables,
    graph_with_bpms,
    read_fixture_tables,
)

PROBE = "probe_pictures.csv"


@pytest.fixture(scope="module")
def bpm_graph(tmp_path_factory):
    return graph_with_bpms(tmp_path_factory.mktemp("bpm"))


def _edit(directory: Path, rows: list[str]) -> None:
    with (directory / PROBE).open("a") as handle:
        handle.writelines(f"{row}\n" for row in rows)


# -- the table ---------------------------------------------------------------------


def test_the_probe_table_reads() -> None:
    assert ",".join(COLUMNS[PROBE]) == (
        "class,element,shape,label,angle_deg,radius_mm,size_mm,signal,signal_y,min,max,"
        "range_mm,count,document,version,state,locator,under"
    )
    rows = read_fixture_tables(PANELS_DIR).probe_pictures
    anchor = next(r for r in rows if r.element == "A")
    assert (anchor.cls, anchor.shape, anchor.angle_deg, anchor.size_mm) == ("BPM", "anchor", 45, 8)
    marker = next(r for r in rows if r.shape == "marker")
    assert (marker.signal, marker.signal_y, marker.range_mm) == (
        "position_x_readback", "position_y_readback", 2
    )  # fmt: skip
    assert [r.under for r in rows if r.shape == "bore"] == [None, "XR/XR02"]


_ROW = "BPM,{element},{shape},{label},{angle},{radius},{size},{signal},{signal_y},,,{range},,,,,,{under}"


def _row(**fields: str) -> str:
    base = dict.fromkeys(
        ("element", "shape", "label", "angle", "radius", "size", "signal", "signal_y", "range"), ""
    )
    return _ROW.format(**{**base, "under": "", **fields})


@pytest.mark.parametrize(
    ("row", "says"),
    [
        (_row(element="e", shape="blob"), "shape is 'blob'"),
        (_row(element="E", shape="anchor", radius="14", size="8"), "angle_deg"),
        (_row(element="m2", shape="marker", signal="a", signal_y="b", under="XR/XR01"), "range_mm"),
        (_row(element="bore2", shape="bore"), "radius_mm"),
        (_row(element="r", shape="readout", label="sum"), "signal"),
        (_row(element="p", shape="plot"), "signal"),
        (_row(element="bore2", shape="bore", radius="9"), "one bore"),
        (_row(element="A", shape="anchor", angle="1", radius="1", size="1"), "already has"),
        (_row(element="turn by turn", shape="plot", label="x", signal="s"), "already has"),
    ],
)
def test_a_probe_row_that_does_not_hold_is_refused(tmp_path, row, says) -> None:
    directory = copy_tables(tmp_path / "panels")
    lines = (directory / PROBE).read_text().splitlines()
    _edit(directory, [row])
    with pytest.raises(PanelTableError, match=says) as raised:
        read_fixture_tables(directory)
    assert raised.value.file == PROBE and raised.value.line == len(lines) + 1


def test_a_place_may_have_its_own_bore(tmp_path) -> None:
    directory = copy_tables(tmp_path / "panels")
    _edit(directory, [_row(element="ring", shape="bore", radius="9", under="XR/XR01")])
    assert len([r for r in read_fixture_tables(directory).probe_pictures if r.shape == "bore"]) == 3


# -- the graph ---------------------------------------------------------------------


def test_the_most_specific_under_wins_element_by_element(bpm_graph) -> None:
    here = {r.element: r for r in bpm_graph.probe("XR01:BPM1")}
    there = {r.element: r for r in bpm_graph.probe("XR02:BPM1")}
    assert (here["bore"].radius_mm, here["bore"].size_mm) == (14.2, None)
    assert (there["bore"].radius_mm, there["bore"].size_mm) == (30, 16)
    assert [r.element for r in bpm_graph.probe("XR02:BPM1")] == [
        r.element for r in bpm_graph.probe("XR01:BPM1")
    ]  # the override keeps the element's place
    assert len(bpm_graph.probe("XR01:BPM1")) == 9
    assert bpm_graph.probe("XR01:QF1") == ()


def test_a_signal_is_found_at_the_port_then_the_device_then_what_reads_it_out(bpm_graph):
    # amplitude_a: stated at the port device, on the pickup and on the chassis
    assert bpm_graph.probe_channel("XR01:BPM1", "amplitude_a") == "XR01:BPM1-E:P1:amplitude_a"
    assert bpm_graph.probe_channel("XR01:BPM1", "tbt_x") == "XR01:BPM1:tbt_x"  # the pickup's own
    assert bpm_graph.probe_channel("XR01:BPM1", "sum_readback") == "XR01:BPM1-E:sum_readback"
    assert bpm_graph.probe_channel("XR01:BPM1", "nothing") is None
    assert bpm_graph.signal_channels("XR01:BPM1", "amplitude_a") == ("XR01:BPM1-E:P1:amplitude_a",)


def test_a_port_that_names_no_device_is_passed_over(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import add_bpm, write_fixture_facility

    def change(document) -> None:
        add_bpm(document, "XR01:BPM1", "XR/XR01")
        document["relations"][-1]["object"]["port"] = "input 3"

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", change), PANELS_DIR)
    assert graph.probe_channel("XR01:BPM1", "amplitude_a") == "XR01:BPM1:amplitude_a"


def test_a_value_held_in_the_control_system_is_found_the_same_way(tmp_path) -> None:
    from osprey.facility.panels.describe import calibration_row

    tables = copy_tables(tmp_path / "panels")
    with (tables / "calibration_kinds.csv").open("a") as handle:
        handle.write("gains,Gains,BPM,scalar,1,beam,,,amplitude_a,1,\n")
    graph = graph_with_bpms(tmp_path, tables)
    row = calibration_row(graph, "XR01:BPM1", graph.calibration_kind("gains", "BPM"))
    assert row.channels == ("XR01:BPM1-E:P1:amplitude_a",)


def test_check_notes_a_probe_row_whose_signal_no_channel_states(tmp_path) -> None:
    from osprey.facility.panels.emit import check, write

    tables = copy_tables(tmp_path / "panels")
    with (tables / PROBE).open("a") as handle:
        handle.write("BPM,ghost,readout,ghost,,,,ghost_readback,,,,,,,,,,\n")
    graph = graph_with_bpms(tmp_path, tables)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:BPM1")
    found = [p for p in check(graph, out) if p.kind == "unbound_probe"]
    assert len(found) == 1 and not found[0].failing
    assert "ghost_readback" in found[0].detail and "XR01:BPM1" in found[0].detail
