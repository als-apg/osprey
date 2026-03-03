"""Frequency map analysis worker — diffusion rate in tune space.

Tracks particles on a (x, y) grid, extracts tunes via FFT from first
and second halves, computes diffusion rate as a chaos indicator.
Plots in tune space with resonance line overlay.
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


def get_tune_fft(turn_data: np.ndarray) -> float:
    """Extract tune from turn-by-turn data via FFT."""
    n = len(turn_data)
    data = turn_data - np.mean(turn_data)
    data *= np.hanning(n)
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n)
    peak_idx = np.argmax(spectrum[1:]) + 1
    return float(freqs[peak_idx])


def compute_fma(
    ring: at.Lattice,
    n_half: int = 256,
    nx: int = 12,
    ny: int = 12,
    x_max: float = 0.003,
    y_max: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frequency map.

    Returns (nux_map, nuy_map, diffusion) as 2D arrays.
    NaN entries indicate lost particles.
    """
    x_vals = np.linspace(0.0005, x_max, nx)
    y_vals = np.linspace(0.0005, y_max, ny)

    nux_map = np.full((ny, nx), np.nan)
    nuy_map = np.full((ny, nx), np.nan)
    diffusion = np.full((ny, nx), np.nan)

    for iy, y0 in enumerate(y_vals):
        for ix, x0 in enumerate(x_vals):
            rin = np.zeros(6)
            rin[0] = x0
            rin[2] = y0

            try:
                rout = ring.track(rin, nturns=2 * n_half, refpts=[0])
                if not np.all(np.isfinite(rout)):
                    continue

                x_tbt = rout[0, 0, :]
                y_tbt = rout[2, 0, :]

                nux1 = get_tune_fft(x_tbt[:n_half])
                nuy1 = get_tune_fft(y_tbt[:n_half])
                nux2 = get_tune_fft(x_tbt[n_half : 2 * n_half])
                nuy2 = get_tune_fft(y_tbt[n_half : 2 * n_half])

                nux_map[iy, ix] = 0.5 * (nux1 + nux2)
                nuy_map[iy, ix] = 0.5 * (nuy1 + nuy2)

                dnux = nux2 - nux1
                dnuy = nuy2 - nuy1
                d = np.sqrt(dnux**2 + dnuy**2)
                diffusion[iy, ix] = np.log10(d) if d > 0 else -15.0
            except Exception:
                continue

    return nux_map, nuy_map, diffusion


def _add_resonance_overlay(
    fig: go.Figure,
    nux_range: tuple[float, float],
    nuy_range: tuple[float, float],
) -> None:
    """Add light resonance lines for context."""
    for order in range(1, 5):
        for m in range(-order, order + 1):
            n_abs = order - abs(m)
            for n in [n_abs, -n_abs] if n_abs != 0 else [0]:
                if m == 0 and n == 0:
                    continue
                for p in range(-100, 101):
                    pts: list[tuple[float, float]] = []
                    if n != 0:
                        for nx_edge in nux_range:
                            ny_val = (p - m * nx_edge) / n
                            if nuy_range[0] <= ny_val <= nuy_range[1]:
                                pts.append((nx_edge, ny_val))
                    if m != 0:
                        for ny_edge in nuy_range:
                            nx_val = (p - n * ny_edge) / m
                            if nux_range[0] <= nx_val <= nux_range[1]:
                                pts.append((nx_val, ny_edge))
                    if len(pts) >= 2:
                        pts = sorted(set(pts))
                        fig.add_trace(
                            go.Scatter(
                                x=[pts[0][0], pts[-1][0]],
                                y=[pts[0][1], pts[-1][1]],
                                mode="lines",
                                line={"color": "rgba(128,128,128,0.2)", "width": 0.5},
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )


def build_figure(
    nux_map: np.ndarray,
    nuy_map: np.ndarray,
    diffusion: np.ndarray,
    design_tune: tuple[float, float] | None = None,
    baseline_tune: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure()

    valid = np.isfinite(diffusion.ravel())
    n_survived = int(np.sum(valid))
    n_total = diffusion.size

    # FMA scatter
    fig.add_trace(
        go.Scatter(
            x=nux_map.ravel()[valid],
            y=nuy_map.ravel()[valid],
            mode="markers",
            marker={
                "size": 7,
                "color": diffusion.ravel()[valid],
                "colorscale": "Viridis_r",
                "colorbar": {"title": "log\u2081\u2080(\u0394\u03bd)"},
                "cmin": -12,
                "cmax": -2,
            },
            name="FMA",
            hovertemplate=(
                "\u03bd<sub>x</sub> = %{x:.5f}<br>"
                "\u03bd<sub>y</sub> = %{y:.5f}<br>"
                "log\u2081\u2080(\u0394\u03bd) = %{marker.color:.1f}<extra></extra>"
            ),
        )
    )

    # Resonance line overlay
    if np.any(valid):
        nux_range = (float(np.nanmin(nux_map)) - 0.005, float(np.nanmax(nux_map)) + 0.005)
        nuy_range = (float(np.nanmin(nuy_map)) - 0.005, float(np.nanmax(nuy_map)) + 0.005)
        _add_resonance_overlay(fig, nux_range, nuy_range)

    # Baseline design tune
    if baseline_tune is not None:
        fig.add_trace(
            go.Scatter(
                x=[baseline_tune[0]],
                y=[baseline_tune[1]],
                mode="markers",
                marker={"size": 10, "color": "gray", "symbol": "star", "opacity": 0.5},
                name="Baseline tune",
            )
        )

    # Design tune marker
    if design_tune is not None:
        fig.add_trace(
            go.Scatter(
                x=[design_tune[0]],
                y=[design_tune[1]],
                mode="markers",
                marker={"size": 10, "color": "red", "symbol": "star"},
                name="Design tune",
            )
        )

    fig.update_layout(
        title=f"Frequency Map ({n_survived}/{n_total} survived)",
        xaxis_title="\u03bd<sub>x</sub>",
        yaxis_title="\u03bd<sub>y</sub>",
        height=500,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        xaxis={"gridcolor": "lightgray"},
        yaxis={"gridcolor": "lightgray", "scaleanchor": "x"},
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    nux_map, nuy_map, diffusion = compute_fma(ring)

    tunes = at.get_tune(ring)
    design_tune = [float(tunes[0]), float(tunes[1])]

    raw: dict = {
        "nux_map": nux_map.tolist(),
        "nuy_map": nuy_map.tolist(),
        "diffusion": diffusion.tolist(),
        "design_tune": design_tune,
        "baseline_tune": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bt = at.get_tune(baseline_ring)
        raw["baseline_tune"] = [float(bt[0]), float(bt[1])]

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
