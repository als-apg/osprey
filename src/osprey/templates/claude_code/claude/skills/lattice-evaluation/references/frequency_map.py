"""Frequency map analysis with diffusion rate visualization.

Demonstrates:
- Setting up a (x, y) grid of initial conditions
- Tracking particles for 2N turns and extracting turn-by-turn data
- Computing tunes via FFT from first and second halves
- Computing diffusion rate as a chaos indicator
- Plotting in tune space with resonance line overlay
"""

import at
import numpy as np
import plotly.graph_objects as go

# --- Load lattice ---
ring = at.load_m("machine_data/als.m")

print("Frequency Map Analysis")
print("=" * 50)

# --- Parameters ---
n_half = 256  # turns per half => 512 total
nx, ny = 15, 15  # grid size
x_max = 0.015  # 15 mm
y_max = 0.008  # 8 mm

# Grid of initial conditions (avoid zero — use small offset)
x_vals = np.linspace(0.0005, x_max, nx)
y_vals = np.linspace(0.0005, y_max, ny)

# Get working point for resonance line context
tunes_ref = at.get_tune(ring)
print(f"  Working point: nu_x = {tunes_ref[0]:.4f}, nu_y = {tunes_ref[1]:.4f}")
print(f"  Grid: {nx} x {ny} = {nx * ny} particles")
print(f"  Turns: 2 x {n_half} = {2 * n_half}")


def get_tune_fft(turn_data):
    """Extract tune from turn-by-turn data via FFT."""
    n = len(turn_data)
    # Remove mean and apply Hanning window
    data = turn_data - np.mean(turn_data)
    data *= np.hanning(n)
    spectrum = np.abs(np.fft.rfft(data))
    # Find peak (skip DC)
    freqs = np.fft.rfftfreq(n)
    peak_idx = np.argmax(spectrum[1:]) + 1
    return freqs[peak_idx]


# --- Track and compute tunes ---
nux_map = np.full((ny, nx), np.nan)
nuy_map = np.full((ny, nx), np.nan)
diffusion = np.full((ny, nx), np.nan)

n_survived = 0
n_total = nx * ny

for iy, y0 in enumerate(y_vals):
    for ix, x0 in enumerate(x_vals):
        rin = np.zeros(6)
        rin[0] = x0
        rin[2] = y0

        # Track for 2*n_half turns, get turn-by-turn at observation point
        rout = ring.track(rin, nturns=2 * n_half, refpts=[0])

        # rout shape: (6, 1, nturns+1) at refpts[0]
        # Check survival
        if not np.all(np.isfinite(rout)):
            continue

        n_survived += 1

        # Extract x and y turn-by-turn data
        x_tbt = rout[0, 0, :]
        y_tbt = rout[2, 0, :]

        # First half tunes
        nux1 = get_tune_fft(x_tbt[:n_half])
        nuy1 = get_tune_fft(y_tbt[:n_half])

        # Second half tunes
        nux2 = get_tune_fft(x_tbt[n_half : 2 * n_half])
        nuy2 = get_tune_fft(y_tbt[n_half : 2 * n_half])

        # Store average tune
        nux_map[iy, ix] = 0.5 * (nux1 + nux2)
        nuy_map[iy, ix] = 0.5 * (nuy1 + nuy2)

        # Diffusion rate (chaos indicator)
        dnux = nux2 - nux1
        dnuy = nuy2 - nuy1
        d = np.sqrt(dnux**2 + dnuy**2)
        diffusion[iy, ix] = np.log10(d) if d > 0 else -15.0

print(f"\n  Survived: {n_survived}/{n_total} particles")
print(f"  Diffusion range: [{np.nanmin(diffusion):.1f}, {np.nanmax(diffusion):.1f}]")

# --- Plot in tune space ---
fig = go.Figure()

# FMA scatter colored by diffusion
valid = np.isfinite(diffusion.ravel())
fig.add_trace(
    go.Scatter(
        x=nux_map.ravel()[valid],
        y=nuy_map.ravel()[valid],
        mode="markers",
        marker=dict(
            size=8,
            color=diffusion.ravel()[valid],
            colorscale="Viridis_r",
            colorbar=dict(title="log₁₀(Δν)"),
            cmin=-12,
            cmax=-2,
        ),
        name="FMA",
        hovertemplate=(
            "ν<sub>x</sub> = %{x:.5f}<br>"
            "ν<sub>y</sub> = %{y:.5f}<br>"
            "log₁₀(Δν) = %{marker.color:.1f}"
            "<extra></extra>"
        ),
    )
)

# Overlay key resonance lines (orders 1-4)
nux_range = (np.nanmin(nux_map) - 0.005, np.nanmax(nux_map) + 0.005)
nuy_range = (np.nanmin(nuy_map) - 0.005, np.nanmax(nuy_map) + 0.005)

for order in range(1, 5):
    for m in range(-order, order + 1):
        n = order - abs(m)
        for n_sign in [n, -n] if n != 0 else [0]:
            if m == 0 and n_sign == 0:
                continue
            for p in range(-100, 101):
                pts = []
                if n_sign != 0:
                    for nx_edge in nux_range:
                        ny_val = (p - m * nx_edge) / n_sign
                        if nuy_range[0] <= ny_val <= nuy_range[1]:
                            pts.append((nx_edge, ny_val))
                if m != 0:
                    for ny_edge in nuy_range:
                        nx_val = (p - n_sign * ny_edge) / m
                        if nux_range[0] <= nx_val <= nux_range[1]:
                            pts.append((nx_val, ny_edge))
                if len(pts) >= 2:
                    pts = list(set(pts))
                    pts.sort()
                    fig.add_trace(
                        go.Scatter(
                            x=[pts[0][0], pts[-1][0]],
                            y=[pts[0][1], pts[-1][1]],
                            mode="lines",
                            line=dict(color="rgba(128,128,128,0.3)", width=0.5),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

# Working point marker
fig.add_trace(
    go.Scatter(
        x=[tunes_ref[0]],
        y=[tunes_ref[1]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="star"),
        name="Design tune",
    )
)

fig.update_layout(
    title=f"ALS Frequency Map Analysis ({2 * n_half} turns)",
    xaxis_title="ν<sub>x</sub> (fractional)",
    yaxis_title="ν<sub>y</sub> (fractional)",
    height=700,
    width=700,
    template="plotly_white",
    xaxis=dict(gridcolor="lightgray"),
    yaxis=dict(gridcolor="lightgray", scaleanchor="x"),
)

# --- Save ---
save_artifact(
    fig,
    "ALS Frequency Map",
    f"FMA with {n_survived} particles, {2 * n_half} turns, diffusion coloring",
)

# Alternative: fig.write_html("_agent_data/artifacts/frequency_map.html")
print("\nFrequency map saved.")
