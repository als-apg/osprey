"""Compute dynamic aperture boundary via particle tracking.

Demonstrates:
- Setting up a tracking grid at multiple angles
- Binary-searching the DA boundary using ring.track()
- Plotting the DA boundary in (x, y) space with Plotly
- Reporting DA area and min/max amplitudes
"""

import at
import numpy as np
import plotly.graph_objects as go

# --- Load lattice ---
ring = at.load_m("machine_data/als.m")

print("Dynamic Aperture Computation")
print("=" * 50)

# --- Tracking parameters ---
nturns = 512
n_angles = 19  # 0 to 90 degrees in 5-degree steps
angles_deg = np.linspace(0, 90, n_angles)
angles_rad = np.radians(angles_deg)

# Amplitude search range [m]
amp_min = 0.0001  # 0.1 mm
amp_max = 0.030   # 30 mm
n_bisect = 15     # binary search iterations


def find_da_at_angle(ring, angle_rad, nturns, amp_min, amp_max, n_bisect):
    """Binary search for the dynamic aperture at a given angle."""
    lo, hi = amp_min, amp_max
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        # Initial condition: (x, x', y, y', dp, ct)
        rin = np.zeros(6)
        rin[0] = mid * np.cos(angle_rad)  # x
        rin[2] = mid * np.sin(angle_rad)  # y

        rout = ring.track(rin, nturns=nturns)
        # rout shape: (6, 1, nturns) — check if particle survived
        survived = np.all(np.isfinite(rout))

        if survived:
            lo = mid
        else:
            hi = mid
    return lo


# --- Scan angles ---
da_amplitudes = np.zeros(n_angles)
da_x = np.zeros(n_angles)
da_y = np.zeros(n_angles)

for i, angle in enumerate(angles_rad):
    da_amp = find_da_at_angle(ring, angle, nturns, amp_min, amp_max, n_bisect)
    da_amplitudes[i] = da_amp
    da_x[i] = da_amp * np.cos(angle)
    da_y[i] = da_amp * np.sin(angle)
    print(f"  Angle {angles_deg[i]:5.1f}°: DA = {da_amp * 1000:.2f} mm")

# Close the boundary by mirroring
da_x_full = np.concatenate([da_x, -da_x[::-1], da_x[:1]])
da_y_full = np.concatenate([da_y, da_y[::-1], da_y[:1]])

# --- Compute DA area (shoelace formula) ---
area = 0.5 * abs(
    np.sum(da_x_full[:-1] * da_y_full[1:] - da_x_full[1:] * da_y_full[:-1])
)
area_mm2 = area * 1e6

print(f"\n  DA area:     {area_mm2:.1f} mm²")
print(f"  Min DA:      {np.min(da_amplitudes) * 1000:.2f} mm")
print(f"  Max DA:      {np.max(da_amplitudes) * 1000:.2f} mm")

# --- Plot ---
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=da_x_full * 1000,
        y=da_y_full * 1000,
        mode="lines+markers",
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgb(31, 119, 180)", width=2),
        marker=dict(size=4),
        name=f"DA boundary ({nturns} turns)",
        hovertemplate="x = %{x:.2f} mm<br>y = %{y:.2f} mm<extra></extra>",
    )
)

fig.update_layout(
    title=f"ALS Dynamic Aperture ({nturns} turns)",
    xaxis_title="x [mm]",
    yaxis_title="y [mm]",
    height=600,
    width=700,
    template="plotly_white",
    xaxis=dict(gridcolor="lightgray", zeroline=True, zerolinecolor="gray"),
    yaxis=dict(
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="gray",
        scaleanchor="x",
    ),
    annotations=[
        dict(
            text=f"DA area = {area_mm2:.1f} mm²",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=14),
        )
    ],
)

# --- Save ---
save_artifact(fig, "ALS Dynamic Aperture", f"DA boundary, {nturns} turns, area={area_mm2:.1f} mm²")

# Alternative: fig.write_html("osprey-workspace/artifacts/dynamic_aperture.html")
print("\nDynamic aperture plot saved.")
