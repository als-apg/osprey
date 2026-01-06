"""
Multi-Subplot Plot - Example Script

This script demonstrates best practices for creating multiple subplots
with shared axes and consistent formatting.

Key practices demonstrated:
- Multiple subplots with shared x-axis
- Consistent y-axis labels with units
- Individual subplot titles
- Color-blind friendly color palette
- Overall figure title
- Proper spacing between subplots
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

# Generate example data for multiple channels
hours = np.arange(24)
time_labels = [(datetime.now() - timedelta(hours=int(24 - h))).strftime("%H:%M") for h in hours]

# Simulate different control system parameters
beam_current = 500 * np.exp(-hours / 50) + np.random.normal(0, 5, 24)
vacuum_pressure = 1e-9 + np.random.normal(0, 5e-11, 24)
magnet_current = 200 + np.random.normal(0, 2, 24)

# Create figure with multiple subplots (shared x-axis)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Color-blind friendly palette
colors = ["#0077BB", "#EE7733", "#009988"]

# Plot 1: Beam Current
axes[0].plot(hours, beam_current, "o-", color=colors[0], linewidth=2, markersize=4)
axes[0].set_ylabel("Beam Current [mA]", fontsize=11)
axes[0].set_title("Storage Ring Beam Current", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3, linestyle="--")

# Plot 2: Vacuum Pressure (log scale for pressure)
axes[1].semilogy(hours, vacuum_pressure, "s-", color=colors[1], linewidth=2, markersize=4)
axes[1].set_ylabel("Pressure [Torr]", fontsize=11)
axes[1].set_title("Vacuum System Pressure", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3, linestyle="--", which="both")

# Plot 3: Magnet Current
axes[2].plot(hours, magnet_current, "^-", color=colors[2], linewidth=2, markersize=4)
axes[2].set_ylabel("Current [A]", fontsize=11)
axes[2].set_title("Dipole Magnet Current", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Time [hours ago]", fontsize=11)
axes[2].grid(True, alpha=0.3, linestyle="--")

# Set x-axis ticks (only on bottom plot since x-axis is shared)
axes[2].set_xticks(hours[::4])
axes[2].set_xticklabels(time_labels[::4], rotation=45, ha="right")

# Overall figure title
fig.suptitle("Control System Multi-Parameter Monitor", fontsize=14, fontweight="bold", y=0.995)

# Adjust spacing between subplots
plt.tight_layout()

# Save to figures/ subdirectory (framework will find it automatically)
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
plot_path = figures_dir / "multi_parameter_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# Store results
results = {
    "plot_path": str(plot_path),  # Convert Path to string for JSON serialization
    "parameters_plotted": ["beam_current", "vacuum_pressure", "magnet_current"],
    "time_range_hours": 24,
    "description": "Multi-subplot plot showing three control system parameters over 24 hours",
}
