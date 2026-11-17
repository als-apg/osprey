"""The probe picture: a pickup's cross-section on its card, from ``probe_pictures.csv``."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.facility.panels.tables import COLUMNS, PanelTableError, read_tables
from tests.facility.panels._fixture import PANELS_DIR, copy_tables, graph_with_bpms

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
    rows = read_tables(PANELS_DIR).probe_pictures
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
        read_tables(directory)
    assert raised.value.file == PROBE and raised.value.line == len(lines) + 1


def test_a_place_may_have_its_own_bore(tmp_path) -> None:
    directory = copy_tables(tmp_path / "panels")
    _edit(directory, [_row(element="ring", shape="bore", radius="9", under="XR/XR01")])
    assert len([r for r in read_tables(directory).probe_pictures if r.shape == "bore"]) == 3


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


def test_a_signal_is_found_on_the_device_or_what_reads_it_out(bpm_graph) -> None:
    assert bpm_graph.probe_channel("XR01:BPM1", "amplitude_a") == "XR01:BPM1:amplitude_a"
    assert bpm_graph.probe_channel("XR01:BPM1", "sum_readback") == "XR01:BPM1-E:sum_readback"
    assert bpm_graph.probe_channel("XR01:BPM1", "nothing") is None
