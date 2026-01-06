"""
Aligned Multiple Plots - Example Script

This script demonstrates how to create MULTIPLE DIFFERENT plot types
in a SINGLE combined figure with proper alignment.

IMPORTANT: This example shows DIFFERENT PARAMETER TYPES in separate subplots
because they have DIFFERENT UNITS and DIFFERENT SCALES:
- Quadrupole currents [A] - linear scale
- Beam current [mA] - linear scale
- Vacuum pressure [Torr] - LOG scale (semilogy)

If plotting SIMILAR channels with SAME UNITS (e.g., 20 BPM positions all in [mm]),
see time_series_basic.py instead - plot them on SAME axes with different colors!

Common Problem:
- User asks: "Plot time series and correlation matrix"
- Code generates two separate plots with different widths
- They look misaligned and awkward

Solution:
- Create ONE combined figure with both plot types
- Use GridSpec for flexible layout
- Everything is guaranteed to be aligned

Key practices demonstrated:
- Combining different plot types (time series + heatmap) in one figure
- Using GridSpec for flexible subplot layout
- Separate subplots for DIFFERENT parameter types (different units/scales)
- Consistent styling across different plot types
- Single output file with everything aligned
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Generate example time series data for multiple channels
np.random.seed(42)
n_hours = 48
times = [datetime.now() - timedelta(hours=int(n_hours - h)) for h in range(n_hours)]

# Simulate correlated control system channels
base_trend = np.sin(2 * np.pi * np.arange(n_hours) / 24)  # Daily pattern
quadrupole_q1 = 150 + 5 * base_trend + np.random.normal(0, 0.5, n_hours)
quadrupole_q2 = 145 + 4.5 * base_trend + np.random.normal(0, 0.6, n_hours)
beam_current = 500 - 10 * base_trend + np.random.normal(0, 2, n_hours)
vacuum_pressure = 1e-9 + 5e-11 * np.abs(base_trend) + np.random.normal(0, 1e-11, n_hours)

# Channel names and data matrix
channels = ["Q1 Current", "Q2 Current", "Beam Current", "Vacuum"]
data_matrix = np.column_stack([quadrupole_q1, quadrupole_q2, beam_current, vacuum_pressure])

# =============================================================================
# Create Combined Figure with GridSpec
# =============================================================================
# Key: GridSpec allows flexible layout with different plot types vertically stacked

fig = plt.figure(figsize=(12, 14))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)

# =============================================================================
# Section 1: Time Series Subplots
# =============================================================================
gs_ts = gs[0].subgridspec(4, 1, hspace=0.1)  # Nested grid for time series
axes_ts = [fig.add_subplot(gs_ts[i]) for i in range(4)]

# Plot each channel
axes_ts[0].plot(times, quadrupole_q1, "o-", linewidth=2, markersize=3, color="#0077BB")
axes_ts[0].set_ylabel("Q1 Current [A]", fontsize=10)
axes_ts[0].grid(True, alpha=0.3, linestyle="--")
axes_ts[0].set_title("Time Series - Last 48 Hours", fontsize=12, fontweight="bold", pad=10)

axes_ts[1].plot(times, quadrupole_q2, "o-", linewidth=2, markersize=3, color="#009988")
axes_ts[1].set_ylabel("Q2 Current [A]", fontsize=10)
axes_ts[1].grid(True, alpha=0.3, linestyle="--")

axes_ts[2].plot(times, beam_current, "o-", linewidth=2, markersize=3, color="#EE7733")
axes_ts[2].set_ylabel("Beam Current [mA]", fontsize=10)
axes_ts[2].grid(True, alpha=0.3, linestyle="--")

axes_ts[3].semilogy(times, vacuum_pressure, "o-", linewidth=2, markersize=3, color="#CC3311")
axes_ts[3].set_ylabel("Vacuum [Torr]", fontsize=10)
axes_ts[3].set_xlabel("Time", fontsize=11)
axes_ts[3].grid(True, alpha=0.3, linestyle="--", which="both")

# Format x-axis (only on bottom subplot)
axes_ts[3].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
axes_ts[3].xaxis.set_major_locator(mdates.HourLocator(interval=8))
plt.setp(axes_ts[3].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Hide x-tick labels on upper subplots
for ax in axes_ts[:-1]:
    ax.set_xticklabels([])

# =============================================================================
# Section 2: Correlation Matrix
# =============================================================================
ax_corr = fig.add_subplot(gs[1])

# Calculate and plot correlation matrix
correlation_matrix = np.corrcoef(data_matrix.T)
im = ax_corr.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

# Add colorbar
cbar = plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
cbar.set_label("Correlation Coefficient", fontsize=11, fontweight="bold")

# Set ticks and labels
ax_corr.set_xticks(np.arange(len(channels)))
ax_corr.set_yticks(np.arange(len(channels)))
ax_corr.set_xticklabels(channels, rotation=45, ha="right")
ax_corr.set_yticklabels(channels)

# Add correlation values as text overlay
for i in range(len(channels)):
    for j in range(len(channels)):
        color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
        ax_corr.text(
            j,
            i,
            f"{correlation_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color=color,
            fontsize=10,
            fontweight="bold",
        )

ax_corr.set_title("Correlation Matrix", fontsize=12, fontweight="bold", pad=10)

# =============================================================================
# Finalize and Save
# =============================================================================
fig.suptitle("Multi-Channel Analysis Report", fontsize=14, fontweight="bold", y=0.995)

# Save to figures/ subdirectory (framework will find it automatically)
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
plot_path = figures_dir / "aligned_multi_analysis.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Store results
results = {
    "plot_path": str(plot_path),  # Convert Path to string for JSON serialization
    "channels_analyzed": channels,
    "time_range_hours": n_hours,
    "strongest_correlation": {
        "channels": ["Q1 Current", "Q2 Current"],
        "coefficient": float(correlation_matrix[0, 1]),
    },
    "description": "Combined figure with time series and correlation matrix, properly aligned using GridSpec",
}
