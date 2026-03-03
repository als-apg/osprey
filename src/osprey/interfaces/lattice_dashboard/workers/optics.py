"""Optics worker — beta functions and dispersion.

Computes Twiss parameters and produces a two-panel Plotly figure:
top panel = beta_x / beta_y, bottom panel = eta_x (dispersion).
Overlays baseline traces (dashed) when a baseline exists.
"""

from __future__ import annotations

import at
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_state,
    parse_args,
    save_figure,
)


def compute_optics(ring: at.Lattice) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (s_pos, beta_x, beta_y, eta_x)."""
    refpts = range(len(ring) + 1)
    _, _, ld = at.get_optics(ring, refpts=refpts, get_chrom=True)
    s_pos = ring.get_s_pos(refpts)
    return s_pos, ld.beta[:, 0], ld.beta[:, 1], ld.dispersion[:, 0]


def build_figure(
    s_pos: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    eta_x: np.ndarray,
    baseline_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Beta Functions", "Horizontal Dispersion"),
    )

    # Current traces (solid)
    fig.add_trace(
        go.Scatter(
            x=s_pos,
            y=beta_x,
            name="\u03b2<sub>x</sub>",
            line={"color": "rgb(31,119,180)", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=s_pos,
            y=beta_y,
            name="\u03b2<sub>y</sub>",
            line={"color": "rgb(255,127,14)", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=s_pos,
            y=eta_x,
            name="\u03b7<sub>x</sub>",
            line={"color": "rgb(44,160,44)", "width": 1.5},
        ),
        row=2,
        col=1,
    )

    # Baseline traces (dashed)
    if baseline_data is not None:
        bs, bbx, bby, bex = baseline_data
        fig.add_trace(
            go.Scatter(
                x=bs,
                y=bbx,
                name="\u03b2<sub>x</sub> baseline",
                line={"color": "rgb(31,119,180)", "width": 1, "dash": "dash"},
                opacity=0.5,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bs,
                y=bby,
                name="\u03b2<sub>y</sub> baseline",
                line={"color": "rgb(255,127,14)", "width": 1, "dash": "dash"},
                opacity=0.5,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bs,
                y=bex,
                name="\u03b7<sub>x</sub> baseline",
                line={"color": "rgb(44,160,44)", "width": 1, "dash": "dash"},
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="\u03b2 [m]", row=1, col=1, gridcolor="lightgray")
    fig.update_yaxes(title_text="\u03b7 [m]", row=2, col=1, gridcolor="lightgray")
    fig.update_xaxes(title_text="s [m]", row=2, col=1, gridcolor="lightgray")
    fig.update_layout(
        title="Optics",
        height=500,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    s_pos, beta_x, beta_y, eta_x = compute_optics(ring)

    baseline_data = None
    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        baseline_data = compute_optics(baseline_ring)

    fig = build_figure(s_pos, beta_x, beta_y, eta_x, baseline_data)
    save_figure(fig, output_path)


if __name__ == "__main__":
    main()
