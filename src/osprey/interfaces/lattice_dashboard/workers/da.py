"""Dynamic aperture worker — DA boundary via particle tracking.

Binary-searches the DA boundary at multiple angles, then plots the
(x, y) boundary with area annotation.  Overlays baseline DA when
available.
"""

from __future__ import annotations

import at
import numpy as np
import plotly.graph_objects as go

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_state,
    parse_args,
    save_data,
)


def find_da_at_angle(
    ring: at.Lattice,
    angle_rad: float,
    nturns: int,
    amp_min: float,
    amp_max: float,
    n_bisect: int,
) -> float:
    """Binary search for the dynamic aperture at a given angle."""
    lo, hi = amp_min, amp_max
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        rin = np.zeros(6)
        rin[0] = mid * np.cos(angle_rad)
        rin[2] = mid * np.sin(angle_rad)

        rout = ring.track(rin, nturns=nturns)
        survived = np.all(np.isfinite(rout))

        if survived:
            lo = mid
        else:
            hi = mid
    return lo


def compute_da(
    ring: at.Lattice,
    nturns: int = 512,
    n_angles: int = 19,
    amp_max: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute DA boundary and return (da_x_full, da_y_full, da_amplitudes, area_mm2)."""
    angles_deg = np.linspace(0, 90, n_angles)
    angles_rad = np.radians(angles_deg)

    amp_min = 0.0001
    n_bisect = 15

    da_amplitudes = np.zeros(n_angles)
    da_x = np.zeros(n_angles)
    da_y = np.zeros(n_angles)

    for i, angle in enumerate(angles_rad):
        da_amp = find_da_at_angle(ring, angle, nturns, amp_min, amp_max, n_bisect)
        da_amplitudes[i] = da_amp
        da_x[i] = da_amp * np.cos(angle)
        da_y[i] = da_amp * np.sin(angle)

    # Close the boundary by mirroring
    da_x_full = np.concatenate([da_x, -da_x[::-1], da_x[:1]])
    da_y_full = np.concatenate([da_y, da_y[::-1], da_y[:1]])

    # Compute area (shoelace formula)
    area = 0.5 * abs(np.sum(da_x_full[:-1] * da_y_full[1:] - da_x_full[1:] * da_y_full[:-1]))
    area_mm2 = area * 1e6

    return da_x_full, da_y_full, da_amplitudes, area_mm2


def build_figure(
    da_x: np.ndarray,
    da_y: np.ndarray,
    area_mm2: float,
    nturns: int = 512,
    baseline: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Baseline DA (dashed)
    if baseline is not None:
        bx, by, barea = baseline
        fig.add_trace(
            go.Scatter(
                x=bx * 1000,
                y=by * 1000,
                mode="lines",
                line={"color": "gray", "width": 1.5, "dash": "dash"},
                fill="toself",
                fillcolor="rgba(128,128,128,0.08)",
                name=f"Baseline DA ({barea:.0f} mm\u00b2)",
                hovertemplate="x = %{x:.2f} mm<br>y = %{y:.2f} mm<extra>Baseline</extra>",
            )
        )

    # Current DA
    fig.add_trace(
        go.Scatter(
            x=da_x * 1000,
            y=da_y * 1000,
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(31,119,180,0.15)",
            line={"color": "rgb(31,119,180)", "width": 2},
            marker={"size": 3},
            name=f"DA ({area_mm2:.0f} mm\u00b2)",
            hovertemplate="x = %{x:.2f} mm<br>y = %{y:.2f} mm<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Dynamic Aperture ({nturns} turns)",
        xaxis_title="x [mm]",
        yaxis_title="y [mm]",
        height=500,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        xaxis={"gridcolor": "lightgray", "zeroline": True, "zerolinecolor": "gray"},
        yaxis={
            "gridcolor": "lightgray",
            "zeroline": True,
            "zerolinecolor": "gray",
            "scaleanchor": "x",
        },
        annotations=[
            {
                "text": f"Area = {area_mm2:.1f} mm\u00b2",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.98,
                "showarrow": False,
                "font": {"size": 13},
            }
        ],
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    da_x, da_y, _, area_mm2 = compute_da(ring)

    raw: dict = {
        "da_x": da_x.tolist(),
        "da_y": da_y.tolist(),
        "area_mm2": float(area_mm2),
        "nturns": 512,
        "baseline": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bda_x, bda_y, _, barea = compute_da(baseline_ring)
        raw["baseline"] = {
            "da_x": bda_x.tolist(),
            "da_y": bda_y.tolist(),
            "area_mm2": float(barea),
        }

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
