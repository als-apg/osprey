"""Tune footprint worker — amplitude-dependent tune shift.

Tracks particles at various amplitudes to extract the tune footprint
in (nu_x, nu_y) space. Shows how tunes shift with betatron amplitude,
which indicates nonlinear resonance proximity.
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


def compute_footprint(
    ring: at.Lattice,
    n_amp: int = 10,
    x_max: float = 0.003,
    y_max: float = 0.001,
    nturns: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tune footprint at various amplitudes.

    Returns (nux_arr, nuy_arr, amplitudes) for surviving particles.
    """
    x_vals = np.linspace(0.0005, x_max, n_amp)
    y_vals = np.linspace(0.0005, y_max, n_amp)

    nux_list, nuy_list, amp_list = [], [], []

    for x0 in x_vals:
        for y0 in y_vals:
            rin = np.zeros(6)
            rin[0] = x0
            rin[2] = y0

            try:
                rout = ring.track(rin, nturns=nturns, refpts=[0])
                if not np.all(np.isfinite(rout)):
                    continue

                x_tbt = rout[0, 0, :]
                y_tbt = rout[2, 0, :]

                nux = get_tune_fft(x_tbt)
                nuy = get_tune_fft(y_tbt)

                nux_list.append(nux)
                nuy_list.append(nuy)
                amp_list.append(np.sqrt(x0**2 + y0**2) * 1000)  # mm
            except Exception:
                continue

    return np.array(nux_list), np.array(nuy_list), np.array(amp_list)


def build_figure(
    nux: np.ndarray,
    nuy: np.ndarray,
    amps: np.ndarray,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Baseline footprint
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

    # Current footprint
    fig.add_trace(
        go.Scatter(
            x=nux,
            y=nuy,
            mode="markers",
            marker={
                "size": 7,
                "color": amps,
                "colorscale": "Viridis",
                "colorbar": {"title": "Amp [mm]"},
            },
            name="Footprint",
            hovertemplate=(
                "\u03bd<sub>x</sub> = %{x:.5f}<br>"
                "\u03bd<sub>y</sub> = %{y:.5f}<br>"
                "Amp = %{marker.color:.1f} mm<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Tune Footprint",
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
    nux, nuy, amps = compute_footprint(ring)

    raw: dict = {
        "nux": nux.tolist(),
        "nuy": nuy.tolist(),
        "amps": amps.tolist(),
        "baseline": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bnux, bnuy, bamps = compute_footprint(baseline_ring)
        raw["baseline"] = {
            "nux": bnux.tolist(),
            "nuy": bnuy.tolist(),
            "amps": bamps.tolist(),
        }

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
