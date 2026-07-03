"""Unit tests for the migrated Phoebus Data Browser ``.plt`` generator.

Structural-parity coverage for ``create_plt_from_config`` — the XML shape must
match the retired ``phoebus_launch`` server's serializer (axis, per-PV
color/trace/ring_size/request/<archive>, time span), and the archiver binding
must be facility-neutral: no default archiver URL, and no ALS-specific string
anywhere in the migrated source.
"""

from pathlib import Path

import defusedxml.ElementTree as ET

from osprey.mcp_server.phoebus import plt_generator
from osprey.mcp_server.phoebus.models import (
    LineStyle,
    PlotConfig,
    PointType,
    PVConfig,
    TimeRange,
    TraceType,
)


def _minimal_config(**overrides) -> PlotConfig:
    defaults = {
        "title": "Test Plot",
        "pvs": [PVConfig(name="SR:DCCT")],
        "time_range": TimeRange(start="-24 hours", end="now"),
    }
    defaults.update(overrides)
    return PlotConfig(**defaults)


# ── facility-neutral hygiene ────────────────────────────────────────────────
def test_no_facility_specific_strings_in_migrated_source():
    """grep-clean: the migrated files must not bake in an ALS-specific host."""
    for path in (
        Path(plt_generator.__file__),
        Path(plt_generator.__file__).with_name("models.py"),
    ):
        text = path.read_text()
        assert "controls-web.als" not in text
        assert "cagw-alsdmz" not in text
        assert "/home/als" not in text


def test_default_archiver_url_is_none_no_archive_block(tmp_path):
    """With no archiver_url supplied, PVs are emitted without <archive> — no
    facility default is baked into the generator."""
    plt_path = plt_generator.create_plt_from_config(_minimal_config(), workspace_dir=tmp_path)
    content = Path(plt_path).read_text()
    assert "<archive>" not in content
    assert "archappl" not in content


def test_explicit_archiver_url_binds_into_archive_block(tmp_path):
    plt_path = plt_generator.create_plt_from_config(
        _minimal_config(),
        workspace_dir=tmp_path,
        archiver_url="pbraw://example-archiver.test/archappl_retrieve",
    )
    root = ET.fromstring(Path(plt_path).read_text())
    archive = root.find("./pvlist/pv/archive")
    assert archive is not None
    assert archive.find("name").text == "archappl"
    assert archive.find("url").text == "pbraw://example-archiver.test/archappl_retrieve"
    assert archive.find("key").text == "1"


# ── structural parity with the old serializer ──────────────────────────────
def test_root_and_title(tmp_path):
    plt_path = plt_generator.create_plt_from_config(
        _minimal_config(title="My Plot"), workspace_dir=tmp_path
    )
    root = ET.fromstring(Path(plt_path).read_text())
    assert root.tag == "databrowser"
    assert root.find("title").text == "My Plot"


def test_time_span_from_time_range(tmp_path):
    plt_path = plt_generator.create_plt_from_config(
        _minimal_config(time_range=TimeRange(start="-2 hours", end="now")),
        workspace_dir=tmp_path,
    )
    root = ET.fromstring(Path(plt_path).read_text())
    assert root.find("start").text == "-2 hours"
    assert root.find("end").text == "now"


def test_default_time_span_is_last_24_hours_when_no_time_range(tmp_path):
    config = PlotConfig(title="No range", pvs=[PVConfig(name="SR:DCCT")], time_range=None)
    plt_path = plt_generator.create_plt_from_config(config, workspace_dir=tmp_path)
    root = ET.fromstring(Path(plt_path).read_text())
    # Both start/end are formatted datetimes (not relative strings) when no
    # time_range is given — parity with the old default-24h fallback.
    assert root.find("start").text.count("-") >= 2
    assert root.find("end").text.endswith(".999")


def test_single_axis_block_with_name_and_grid(tmp_path):
    plt_path = plt_generator.create_plt_from_config(
        _minimal_config(axis_name="Beam Current", show_grid=False, log_scale=True),
        workspace_dir=tmp_path,
    )
    root = ET.fromstring(Path(plt_path).read_text())
    axes = root.findall("./axes/axis")
    assert len(axes) == 1
    axis = axes[0]
    assert axis.find("name").text == "Beam Current"
    assert axis.find("grid").text == "false"
    assert axis.find("log_scale").text == "true"


def test_per_pv_styling_and_ring_size_request(tmp_path):
    pv = PVConfig(
        name="SR:BPM:X",
        display_name="Horizontal Position",
        color_red=200,
        color_green=0,
        color_blue=0,
        trace_type=TraceType.AREA,
        line_style=LineStyle.DASH,
        line_width=3,
        point_type=PointType.CIRCLE,
        point_size=4,
        axis=1,
    )
    config = _minimal_config(pvs=[pv])
    plt_path = plt_generator.create_plt_from_config(config, workspace_dir=tmp_path)
    root = ET.fromstring(Path(plt_path).read_text())
    pv_el = root.find("./pvlist/pv")
    assert pv_el.find("display_name").text == "Horizontal Position"
    assert pv_el.find("name").text == "SR:BPM:X"
    assert pv_el.find("axis").text == "1"
    assert pv_el.find("color/red").text == "200"
    assert pv_el.find("trace_type").text == "AREA"
    assert pv_el.find("line_style").text == "DASH"
    assert pv_el.find("linewidth").text == "3"
    assert pv_el.find("point_type").text == "CIRCLE"
    assert pv_el.find("point_size").text == "4"
    assert pv_el.find("ring_size").text == "5000"
    assert pv_el.find("request").text == "OPTIMIZED"


def test_multiple_pvs_each_get_a_pv_element(tmp_path):
    config = _minimal_config(
        pvs=[PVConfig(name="SR:A"), PVConfig(name="SR:B"), PVConfig(name="SR:C")]
    )
    plt_path = plt_generator.create_plt_from_config(config, workspace_dir=tmp_path)
    root = ET.fromstring(Path(plt_path).read_text())
    pv_els = root.findall("./pvlist/pv")
    assert [p.find("name").text for p in pv_els] == ["SR:A", "SR:B", "SR:C"]


def test_sanitize_xml_text_escapes_and_normalizes():
    assert plt_generator.sanitize_xml_text("30°C & rising") == "30degC &amp; rising"
    assert plt_generator.sanitize_xml_text("") == ""


def test_file_written_under_workspace_dir_with_plt_suffix(tmp_path):
    plt_path = plt_generator.create_plt_from_config(_minimal_config(), workspace_dir=tmp_path)
    p = Path(plt_path)
    assert p.exists()
    assert p.suffix == ".plt"
    assert p.parent == tmp_path
