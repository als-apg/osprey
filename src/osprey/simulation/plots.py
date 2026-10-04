"""Draw a scenario plot spec as the PNG an operator would export from a strip chart.

A :class:`~osprey.simulation.machine.PlotSpec` carries its time axis as hours
before the instant its logbook entry is written. :func:`render_plot_spec` places
that axis at a real instant, so the picture shows the dates of the entry it is
attached to, read in that instant's own time zone.

Every picture follows one style: the spec's terse title, y label and series
labels, matplotlib's default colour cycle, a legend, a light grid, date ticks on
the time axis, and nothing else drawn on the axes. :func:`figure_png` is the one
encoder for every demo picture: a fixed pixel size, a palette PNG with no text
chunks, so the same inputs always give the same bytes.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from osprey.simulation.machine import PlotSpec

#: Figure size in inches; at :data:`DPI` every picture is 800 x 450 pixels.
FIGSIZE = (8.0, 4.5)
DPI = 100

#: Colours kept by the palette re-encode: enough for antialiased lines and text.
_PALETTE_COLORS = 64

#: Tick label formats by locator level (years, months, days, hours, minutes,
#: seconds): a week-long axis reads "Sep 25", "Sep 26", ...
_TICK_FORMATS = ["%Y", "%b", "%b %d", "%H:%M", "%H:%M", "%S.%f"]
_ZERO_FORMATS = ["", "%Y", "%b", "%b %d", "%H:%M", "%H:%M"]
_OFFSET_FORMATS = ["", "%Y", "%Y", "%Y", "%Y %b %d", "%Y %b %d %H:%M"]


def render_plot_spec(spec: PlotSpec, end: datetime) -> bytes:
    """Draw ``spec`` with its last point at ``end`` and return the PNG bytes.

    Args:
        spec: The plot spec, as the bundle loader validated it.
        end: The instant ``hours_before == 0`` stands for; must be timezone
            aware. Tick labels are read in ``end.tzinfo``.

    Returns:
        A palette PNG of :data:`FIGSIZE` at :data:`DPI` with no text chunks.

    Raises:
        ValueError: If ``end`` is naive.
    """
    from matplotlib import style

    # Ticks and their labels are laid out at save time, so the pinned style
    # must cover the save as well as the drawing.
    with style.context("default"):
        return figure_png(draw_plot_spec(spec, end))


def draw_plot_spec(spec: PlotSpec, end: datetime) -> Figure:
    """Draw ``spec`` on a detached figure under the current matplotlib style.

    :func:`render_plot_spec` is the caller that pins the style; this half is
    separate so the drawn axes can be inspected.

    Raises:
        ValueError: If ``end`` is naive.
    """
    if end.tzinfo is None:
        raise ValueError("a plot spec is drawn against a timezone-aware end instant")

    import matplotlib.dates as mdates

    zone = end.tzinfo
    times = [end - timedelta(hours=hours) for hours in spec.hours_before]
    fig, ax = _new_figure()
    for series in spec.series:
        ax.plot(times, series.values, lw=1.0, label=series.label)
    locator = mdates.AutoDateLocator(tz=zone)
    formatter = mdates.ConciseDateFormatter(
        locator,
        tz=zone,
        formats=_TICK_FORMATS,
        zero_formats=_ZERO_FORMATS,
        offset_formats=_OFFSET_FORMATS,
    )
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(times[0], times[-1])
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.set_title(spec.title)
    ax.set_ylabel(spec.ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _new_figure():
    """A detached Agg figure and its one axes; no pyplot state is touched."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def figure_png(fig: Figure) -> bytes:
    """Encode ``fig`` as a palette PNG with no text chunks, at :data:`DPI`."""
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, metadata={"Software": None})
    buffer.seek(0)
    with Image.open(buffer) as image:
        flat = image.convert("RGB").quantize(
            colors=_PALETTE_COLORS, method=Image.Quantize.MEDIANCUT
        )
    out = io.BytesIO()
    flat.save(out, format="PNG", optimize=True)
    return out.getvalue()
