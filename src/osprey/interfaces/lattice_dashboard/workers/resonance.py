"""Resonance diagram worker — tune diagram with resonance lines.

Computes resonance lines m*nu_x + n*nu_y = p for orders 1-5 and
marks the working point.  Overlays baseline working point when available.
"""

from __future__ import annotations

import at
import plotly.graph_objects as go

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_state,
    parse_args,
    save_data,
)

ORDER_COLORS = {1: "red", 2: "orange", 3: "green", 4: "blue", 5: "purple"}


def _add_resonance_lines(
    fig: go.Figure,
    tune_range_x: tuple[float, float],
    tune_range_y: tuple[float, float],
) -> None:
    """Add resonance lines for orders 1-5 in the visible tune range."""
    for order in range(1, 6):
        for m in range(-order, order + 1):
            n_abs = order - abs(m)
            for n in [n_abs, -n_abs] if n_abs != 0 else [0]:
                if m == 0 and n == 0:
                    continue
                for p in range(-100, 101):
                    pts: list[tuple[float, float]] = []
                    if n != 0:
                        for nx_edge in tune_range_x:
                            ny_val = (p - m * nx_edge) / n
                            if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                                pts.append((nx_edge, ny_val))
                    if m != 0:
                        for ny_edge in tune_range_y:
                            nx_val = (p - n * ny_edge) / m
                            if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                                pts.append((nx_val, ny_edge))
                    if n == 0 and m != 0:
                        nx_val = p / m
                        if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                            pts.append((nx_val, tune_range_y[0]))
                            pts.append((nx_val, tune_range_y[1]))
                    if m == 0 and n != 0:
                        ny_val = p / n
                        if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                            pts.append((tune_range_x[0], ny_val))
                            pts.append((tune_range_x[1], ny_val))

                    if len(pts) >= 2:
                        pts = sorted(set(pts))
                        fig.add_trace(
                            go.Scatter(
                                x=[pts[0][0], pts[-1][0]],
                                y=[pts[0][1], pts[-1][1]],
                                mode="lines",
                                line={
                                    "color": ORDER_COLORS[order],
                                    "width": max(3 - order * 0.4, 0.5),
                                },
                                opacity=0.3,
                                name=f"Order {order}",
                                legendgroup=f"order{order}",
                                showlegend=False,
                                hovertemplate=(
                                    f"{m}\u00b7\u03bd<sub>x</sub>"
                                    f" + {n}\u00b7\u03bd<sub>y</sub>"
                                    f" = {p}<extra></extra>"
                                ),
                            )
                        )

    # Legend entries
    for order, color in ORDER_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": max(3 - order * 0.4, 0.5)},
                opacity=0.3,
                name=f"Order {order}",
                legendgroup=f"order{order}",
                showlegend=True,
            )
        )


def build_figure(
    nux: float,
    nuy: float,
    baseline_nux: float | None = None,
    baseline_nuy: float | None = None,
) -> go.Figure:
    tune_range_x = (nux - 0.1, nux + 0.1)
    tune_range_y = (nuy - 0.1, nuy + 0.1)

    fig = go.Figure()
    _add_resonance_lines(fig, tune_range_x, tune_range_y)

    # Baseline working point (dashed ring)
    if baseline_nux is not None and baseline_nuy is not None:
        fig.add_trace(
            go.Scatter(
                x=[baseline_nux],
                y=[baseline_nuy],
                mode="markers",
                marker={"size": 12, "color": "gray", "symbol": "star", "opacity": 0.5},
                name="Baseline",
                hovertemplate=(
                    f"\u03bd<sub>x</sub> = {baseline_nux:.4f}<br>"
                    f"\u03bd<sub>y</sub> = {baseline_nuy:.4f}"
                    "<extra>Baseline</extra>"
                ),
            )
        )

    # Current working point
    fig.add_trace(
        go.Scatter(
            x=[nux],
            y=[nuy],
            mode="markers",
            marker={"size": 14, "color": "black", "symbol": "star"},
            name="Working point",
            hovertemplate=(
                f"\u03bd<sub>x</sub> = {nux:.4f}<br>\u03bd<sub>y</sub> = {nuy:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Resonance Diagram",
        xaxis_title="\u03bd<sub>x</sub>",
        yaxis_title="\u03bd<sub>y</sub>",
        height=500,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        xaxis={"range": list(tune_range_x), "gridcolor": "lightgray"},
        yaxis={"range": list(tune_range_y), "gridcolor": "lightgray"},
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    tunes = at.get_tune(ring)
    nux, nuy = float(tunes[0]), float(tunes[1])

    raw: dict = {
        "nux": nux,
        "nuy": nuy,
        "baseline_nux": None,
        "baseline_nuy": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bt = at.get_tune(baseline_ring)
        raw["baseline_nux"] = float(bt[0])
        raw["baseline_nuy"] = float(bt[1])

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
