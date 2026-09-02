"""Regenerate the shipped example artifact (``interactive-plot.html``).

The gallery seeds this file into an empty workspace (see
:mod:`osprey.interfaces.artifacts.example_artifact`). It is committed as
bytes so a deployment never needs plotly or numpy to show it; run this
module from a dev checkout when the figure itself should change::

    uv run python -m osprey.interfaces.artifacts.examples.make_interactive_plot

Two synthetic panels: 24 h of stored beam current with a dump and refill,
and one turn of horizontal orbit with a local bump. The random generator is
seeded so the output is reproducible. No colors or fonts are set anywhere
in the figure: the gallery re-themes every Plotly page from the design
tokens, and a figure that names none follows the theme's palette.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUTPUT = Path(__file__).with_name("interactive-plot.html")


def build_figure():  # noqa: ANN201 — plotly is a dev-only import here
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rng = np.random.default_rng(7)
    t = np.arange(0, 24 * 60, 1.0)  # minutes
    current = 500 - 4.0 * np.mod(t, 2.0) / 2.0 + rng.normal(0, 0.15, t.size)  # top-up sawtooth
    dump = (t > 14 * 60 + 30) & (t < 14 * 60 + 52)
    current[dump] = 0
    ramp = (t >= 14 * 60 + 52) & (t < 14 * 60 + 70)
    current[ramp] = np.linspace(0, 500, ramp.sum())
    s = np.linspace(0, 196.8, 240)
    x = 0.03 * np.sin(2 * np.pi * 14.28 * s / 196.8) + rng.normal(0, 0.006, s.size)
    bump = (s > 60) & (s < 80)
    x[bump] += 0.12 * np.sin(np.pi * (s[bump] - 60) / 20)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        subplot_titles=("Stored beam current, last 24 h", "Horizontal orbit, one turn"),
    )
    fig.add_trace(
        go.Scatter(
            x=t / 60,
            y=current,
            mode="lines",
            name="DCCT",
            line={"width": 1.2},
            hovertemplate="%{y:.1f} mA<extra>DCCT</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=s,
            y=x,
            mode="lines",
            name="BPM x",
            line={"width": 1.2},
            hovertemplate="%{y:.3f} mm<extra>BPM x</extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=s[bump],
            y=x[bump],
            mode="lines",
            name="local bump",
            line={"width": 3},
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=14.65,
        y=250,
        text="dump and refill",
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        ax=70,
        ay=-28,
        font={"size": 11},
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=70,
        y=0.155,
        text="local bump",
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        ax=55,
        ay=-24,
        font={"size": 11},
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="time (h)", row=1, col=1)
    fig.update_yaxes(title_text="mA", row=1, col=1, rangemode="tozero")
    fig.update_xaxes(title_text="s (m)", row=2, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    for a in fig.layout.annotations[:2]:
        a.font = {"size": 12}
    fig.update_layout(
        title={
            "text": "Example plot · synthetic data, shipped with this preset",
            "x": 0.02,
            "font": {"size": 14},
        },
        legend={
            "orientation": "h",
            "y": -0.12,
            "x": 0,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 11},
            "entrywidth": 120,
            "entrywidthmode": "pixels",
        },
        hovermode="x unified",
        font={"size": 11},
        margin={"t": 70, "r": 24, "b": 80, "l": 56},
    )
    return fig


def main() -> None:
    # Same shape serialize_object() gives a saved Plotly figure (no bundled
    # plotly.js; the gallery injects its vendored copy), with a fixed div id
    # so two runs of this script produce identical bytes.
    html = build_figure().to_html(
        include_plotlyjs=False, full_html=True, div_id="osprey-example-plot"
    )
    # Trailing newline: the repo's end-of-file check wants one, and the bytes
    # in the store must equal the committed file.
    OUTPUT.write_text(html + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUTPUT} ({len(html.encode()) + 1} bytes)\n")


if __name__ == "__main__":
    main()
