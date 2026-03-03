"""Chromaticity scan worker — tune vs. momentum deviation.

Sweeps dp/p from -3% to +3% and computes tunes at each point.
The slope gives the chromaticity (xi_x, xi_y).
"""

from __future__ import annotations

import at
import numpy as np
import plotly.graph_objects as go

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_settings,
    load_state,
    parse_args,
    save_data,
)


def compute_chromaticity(
    ring: at.Lattice,
    dp_min: float = -0.03,
    dp_max: float = 0.03,
    n_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep dp/p and return (dp_values, nux_values, nuy_values)."""
    dp_values = np.linspace(dp_min, dp_max, n_steps)
    nux_vals = np.zeros_like(dp_values)
    nuy_vals = np.zeros_like(dp_values)

    for i, dp in enumerate(dp_values):
        try:
            _, rd, _ = at.get_optics(ring, dp=dp)
            nux_vals[i] = rd.tune[0]
            nuy_vals[i] = rd.tune[1]
        except Exception:
            nux_vals[i] = np.nan
            nuy_vals[i] = np.nan

    return dp_values, nux_vals, nuy_vals


def build_figure(
    dp: np.ndarray,
    nux: np.ndarray,
    nuy: np.ndarray,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    fig = go.Figure()

    dp_pct = dp * 100  # Convert to percent

    # Baseline (dashed)
    if baseline is not None:
        bdp, bnux, bnuy = baseline
        bdp_pct = bdp * 100
        fig.add_trace(
            go.Scatter(
                x=bdp_pct,
                y=bnux,
                name="\u03bd<sub>x</sub> baseline",
                line={"color": "rgb(31,119,180)", "width": 1, "dash": "dash"},
                opacity=0.5,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=bdp_pct,
                y=bnuy,
                name="\u03bd<sub>y</sub> baseline",
                line={"color": "rgb(255,127,14)", "width": 1, "dash": "dash"},
                opacity=0.5,
            ),
        )

    # Current (solid)
    fig.add_trace(
        go.Scatter(
            x=dp_pct,
            y=nux,
            name="\u03bd<sub>x</sub>",
            line={"color": "rgb(31,119,180)", "width": 2},
            mode="lines+markers",
            marker={"size": 3},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=dp_pct,
            y=nuy,
            name="\u03bd<sub>y</sub>",
            line={"color": "rgb(255,127,14)", "width": 2},
            mode="lines+markers",
            marker={"size": 3},
        ),
    )

    # Fit linear chromaticity
    title = "Chromaticity"
    valid = np.isfinite(nux) & np.isfinite(nuy)
    if np.sum(valid) > 2:
        px = np.polyfit(dp[valid], nux[valid], 1)
        py = np.polyfit(dp[valid], nuy[valid], 1)
        xi_x, xi_y = px[0], py[0]
        title = f"Chromaticity: \u03be<sub>x</sub>={xi_x:.2f}, \u03be<sub>y</sub>={xi_y:.2f}"

    fig.update_layout(
        title=title,
        xaxis_title="\u03b4p/p [%]",
        yaxis_title="\u03bd",
        height=400,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        showlegend=True,
        xaxis={"gridcolor": "lightgray"},
        yaxis={"gridcolor": "lightgray"},
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    settings = load_settings(state, "chromaticity")
    dp_min = settings["dp_min_pct"] / 100.0
    dp_max = settings["dp_max_pct"] / 100.0
    n_steps = settings["n_steps"]

    dp, nux, nuy = compute_chromaticity(ring, dp_min=dp_min, dp_max=dp_max, n_steps=n_steps)

    raw: dict = {
        "dp": dp.tolist(),
        "nux": nux.tolist(),
        "nuy": nuy.tolist(),
        "baseline": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bdp, bnux, bnuy = compute_chromaticity(
            baseline_ring, dp_min=dp_min, dp_max=dp_max, n_steps=n_steps
        )
        raw["baseline"] = {
            "dp": bdp.tolist(),
            "nux": bnux.tolist(),
            "nuy": bnuy.tolist(),
        }

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
