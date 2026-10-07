"""A plot spec is drawn as a dated strip chart ending at its entry's instant."""

from __future__ import annotations

import io
import struct
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import pytest
from PIL import Image

from osprey.simulation.machine import parse_plot_spec
from osprey.simulation.plots import DPI, FIGSIZE, draw_plot_spec, render_plot_spec

ZONE = ZoneInfo("America/Los_Angeles")
#: 10:00 local on a day whose UTC date is the same, so a UTC-read axis would
#: shift every tick by seven hours and show it.
END = datetime(2026, 9, 28, 10, 0, tzinfo=ZONE)


def _spec(**overrides):
    hours = [168.0 - i for i in range(169)]
    raw = {
        "filename": "orbit_rms.png",
        "title": "SR orbit RMS",
        "ylabel": "µm",
        "hours_before": hours,
        "series": [
            {"label": "X", "values": [10.0 + (i % 5) * 0.1 for i in range(169)]},
            {"label": "Y", "values": [9.0 + (i % 3) * 0.1 for i in range(169)]},
        ],
        "ylim": [0, 20],
    }
    raw.update(overrides)
    return parse_plot_spec(raw)


def _chunk_types(data: bytes) -> set[bytes]:
    chunks, offset = set(), 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunks.add(data[offset + 4 : offset + 8])
        offset += 12 + length
    return chunks


def test_the_time_axis_spans_the_week_up_to_the_entry_in_its_zone():
    fig = draw_plot_spec(_spec(), END)
    ax = fig.axes[0]
    left, right = (mdates.num2date(x, tz=ZONE) for x in ax.get_xlim())
    assert right == END
    assert left == END - timedelta(days=7)

    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels, "the time axis carries no tick labels"
    # Daily ticks at the zone's midnights, labelled with the date, never hours.
    assert labels == [f"Sep {day}" for day in range(22, 29)]


def test_the_picture_carries_only_the_specs_words():
    fig = draw_plot_spec(_spec(), END)
    ax = fig.axes[0]
    assert ax.get_title() == "SR orbit RMS"
    assert ax.get_ylabel() == "µm"
    assert ax.get_xlabel() == ""
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["X", "Y"]
    assert len(ax.texts) == 0, "no annotations on a strip chart"
    assert ax.get_ylim() == (0.0, 20.0)


def test_lines_follow_the_default_colour_cycle():
    import matplotlib

    cycle = matplotlib.rcParamsDefault["axes.prop_cycle"].by_key()["color"]
    fig = draw_plot_spec(_spec(), END)
    assert [line.get_color() for line in fig.axes[0].get_lines()] == cycle[:2]


def test_rendering_is_deterministic_and_clean():
    first = render_plot_spec(_spec(), END)
    assert render_plot_spec(_spec(), END) == first
    with Image.open(io.BytesIO(first)) as image:
        assert image.format == "PNG"
        assert image.mode == "P"
        assert image.size == (int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI))
    assert not {b"tEXt", b"iTXt", b"zTXt", b"tIME"} & _chunk_types(first)
    assert len(first) < 80 * 1024


def test_the_same_spec_at_another_instant_is_another_picture():
    """The dates are drawn, so moving the entry moves the picture."""
    assert render_plot_spec(_spec(), END) != render_plot_spec(_spec(), END + timedelta(days=30))


def test_the_rendered_style_ignores_ambient_rc_settings():
    import matplotlib

    clean = render_plot_spec(_spec(), END)
    with matplotlib.rc_context({"lines.linewidth": 4.0, "axes.facecolor": "black"}):
        assert render_plot_spec(_spec(), END) == clean


def test_a_naive_end_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        render_plot_spec(_spec(), END.replace(tzinfo=None))


def test_a_utc_end_reads_in_utc():
    end = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
    left, right = (
        mdates.num2date(x, tz=UTC) for x in draw_plot_spec(_spec(), end).axes[0].get_xlim()
    )
    assert (left, right) == (end - timedelta(days=7), end)
