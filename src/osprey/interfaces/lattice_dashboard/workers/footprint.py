"""Tune footprint worker — amplitude-dependent tune shift with diffusion.

Tracks particles at various amplitudes to extract the tune footprint
in (nu_x, nu_y) space.  Computes diffusion rate (from first/second
half of tracking) as a chaos indicator, colored in the plot.  This
merges the former FMA panel's diffusion coloring with the footprint scan.
"""

from __future__ import annotations

import at
import numpy as np
import plotly.graph_objects as go

from osprey.interfaces.lattice_dashboard.workers._base import (
    add_resonance_overlay,
    load_baseline_ring,
    load_ring,
    load_settings,
    load_state,
    parse_args,
    save_data,
    unpack_tracking,
)


def get_tune_fft(turn_data: np.ndarray, pad_factor: int = 16) -> float:
    """Extract tune from turn-by-turn data via zero-padded FFT.

    Zero-padding interpolates between raw FFT bins, improving frequency
    resolution from 1/N to 1/(N*pad_factor) without needing more turns.
    """
    n = len(turn_data)
    data = turn_data - np.mean(turn_data)
    data *= np.hanning(n)
    n_fft = n * pad_factor
    spectrum = np.abs(np.fft.rfft(data, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft)
    peak_idx = np.argmax(spectrum[1:]) + 1
    return float(freqs[peak_idx])


def compute_footprint(
    ring: at.Lattice,
    n_amp: int = 10,
    x_max: float = 0.003,
    y_max: float = 0.001,
    n_half: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute tune footprint with diffusion rate at various amplitudes.

    Tracks each particle for 2*n_half turns, extracts tunes from each
    half independently, and computes diffusion as log10(sqrt(dnux^2 + dnuy^2)).

    Returns (nux_arr, nuy_arr, amplitudes, diffusion) for surviving particles.
    """
    x_vals = np.linspace(0.0005, x_max, n_amp)
    y_vals = np.linspace(0.0005, y_max, n_amp)

    nux_list, nuy_list, amp_list, diff_list = [], [], [], []

    for x0 in x_vals:
        for y0 in y_vals:
            rin = np.zeros(6)
            rin[0] = x0
            rin[2] = y0

            try:
                result = ring.track(rin, nturns=2 * n_half, refpts=[0])
                rout = unpack_tracking(result)
                if not np.all(np.isfinite(rout)):
                    continue

                x_tbt = rout[0, 0, :]
                y_tbt = rout[2, 0, :]

                # Tunes from each half
                nux1 = get_tune_fft(x_tbt[:n_half])
                nuy1 = get_tune_fft(y_tbt[:n_half])
                nux2 = get_tune_fft(x_tbt[n_half : 2 * n_half])
                nuy2 = get_tune_fft(y_tbt[n_half : 2 * n_half])

                # Averaged tune
                nux_list.append(0.5 * (nux1 + nux2))
                nuy_list.append(0.5 * (nuy1 + nuy2))
                amp_list.append(np.sqrt(x0**2 + y0**2) * 1000)  # mm

                # Diffusion rate
                dnux = nux2 - nux1
                dnuy = nuy2 - nuy1
                d = np.sqrt(dnux**2 + dnuy**2)
                diff_list.append(np.log10(d) if d > 0 else -15.0)
            except Exception:
                continue

    return (
        np.array(nux_list),
        np.array(nuy_list),
        np.array(amp_list),
        np.array(diff_list),
    )


def build_figure(
    nux: np.ndarray,
    nuy: np.ndarray,
    amps: np.ndarray,
    diffusion: np.ndarray,
    design_tune: tuple[float, float] | None = None,
    baseline_tune: tuple[float, float] | None = None,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    n_total: int = 100,
) -> go.Figure:
    fig = go.Figure()

    n_survived = len(nux)

    # Baseline footprint (faded)
    if baseline is not None:
        bnux, bnuy, bamps = baseline
        fig.add_trace(
            go.Scatter(
                x=bnux,
                y=bnuy,
                mode="markers",
                marker={
                    "size": 6,
                    "color": bamps,
                    "colorscale": "Greys",
                    "opacity": 0.3,
                    "symbol": "circle-open",
                },
                name="Baseline footprint",
                hovertemplate=(
                    "\u03bd<sub>x</sub> = %{x:.5f}<br>"
                    "\u03bd<sub>y</sub> = %{y:.5f}<extra>Baseline</extra>"
                ),
            )
        )

    # Current footprint — colored by diffusion rate
    fig.add_trace(
        go.Scatter(
            x=nux,
            y=nuy,
            mode="markers",
            marker={
                "size": 7,
                "color": diffusion,
                "colorscale": "Viridis_r",
                "colorbar": {"title": "log\u2081\u2080(\u0394\u03bd)"},
                "cmin": -12,
                "cmax": -2,
            },
            name="Footprint",
            customdata=np.column_stack([amps]) if len(amps) > 0 else None,
            hovertemplate=(
                "\u03bd<sub>x</sub> = %{x:.5f}<br>"
                "\u03bd<sub>y</sub> = %{y:.5f}<br>"
                "log\u2081\u2080(\u0394\u03bd) = %{marker.color:.1f}<br>"
                "Amp = %{customdata[0]:.1f} mm<extra></extra>"
            ),
        )
    )

    # Resonance line overlay
    if n_survived > 0:
        nux_range = (float(np.min(nux)) - 0.005, float(np.max(nux)) + 0.005)
        nuy_range = (float(np.min(nuy)) - 0.005, float(np.max(nuy)) + 0.005)
        add_resonance_overlay(fig, nux_range, nuy_range)

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
        title=f"Tune Footprint ({n_survived}/{n_total} survived)",
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
    settings = load_settings(state, "footprint")
    n_amp = settings["n_amp"]
    x_max = settings["x_max_mm"] / 1000.0
    y_max = settings["y_max_mm"] / 1000.0
    n_half = settings["n_half"]

    nux, nuy, amps, diffusion = compute_footprint(
        ring, n_amp=n_amp, x_max=x_max, y_max=y_max, n_half=n_half
    )

    tunes = at.get_tune(ring)
    design_tune = [float(tunes[0]), float(tunes[1])]

    raw: dict = {
        "nux": nux.tolist(),
        "nuy": nuy.tolist(),
        "amps": amps.tolist(),
        "diffusion": diffusion.tolist(),
        "design_tune": design_tune,
        "n_amp": n_amp,
        "baseline": None,
        "baseline_tune": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bnux, bnuy, bamps, _ = compute_footprint(
            baseline_ring, n_amp=n_amp, x_max=x_max, y_max=y_max, n_half=n_half
        )
        raw["baseline"] = {
            "nux": bnux.tolist(),
            "nuy": bnuy.tolist(),
            "amps": bamps.tolist(),
        }
        bt = at.get_tune(baseline_ring)
        raw["baseline_tune"] = [float(bt[0]), float(bt[1])]

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
