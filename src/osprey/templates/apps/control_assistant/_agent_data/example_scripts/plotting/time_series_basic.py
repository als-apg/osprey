"""
Basic Time Series Plot - Example Script

This script demonstrates fundamental best practices for creating
clean, readable time series plots for control system data.

IMPORTANT: When plotting MULTIPLE SIMILAR channels (same units, comparable scales):
→ Plot them ALL on the SAME axes with different colors
→ DO NOT create separate subplots for each channel

Use separate subplots ONLY when channels have different units or require different scales.

Key practices demonstrated:
- Multiple similar traces on same axes (color-coded)
- Clear axis labels with units
- Proper datetime formatting
- Grid for readability
- Legend placement
- Color-blind friendly palette
- tight_layout() for clean spacing
- Saving plot to results dictionary
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Generate example time series data for multiple BPM positions
# All channels have same units [mm] → plot on SAME axes
current_time = datetime.now()
times = [current_time - timedelta(hours=i) for i in range(24, 0, -1)]

# Simulate 5 BPM horizontal positions (all in millimeters)
np.random.seed(42)
bpm_positions = {
    "BPM01": 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(24) / 12) + np.random.normal(0, 0.02, 24),
    "BPM02": 0.3
    + 0.1 * np.sin(2 * np.pi * np.arange(24) / 12 + 0.5)
    + np.random.normal(0, 0.02, 24),
    "BPM03": -0.2
    + 0.1 * np.sin(2 * np.pi * np.arange(24) / 12 + 1.0)
    + np.random.normal(0, 0.02, 24),
    "BPM04": 0.1
    + 0.1 * np.sin(2 * np.pi * np.arange(24) / 12 + 1.5)
    + np.random.normal(0, 0.02, 24),
    "BPM05": -0.4
    + 0.1 * np.sin(2 * np.pi * np.arange(24) / 12 + 2.0)
    + np.random.normal(0, 0.02, 24),
}

# Color-blind friendly palette (sufficient for many channels)
colors = ["#0077BB", "#009988", "#EE7733", "#CC3311", "#33BBEE"]

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(12, 6))

# Plot ALL similar channels on SAME axes (not separate subplots!)
for idx, (channel_name, position_data) in enumerate(bpm_positions.items()):
    ax.plot(
        times,
        position_data,
        "o-",
        linewidth=2,
        markersize=3,
        label=channel_name,
        color=colors[idx],
        alpha=0.8,
    )

# Configure x-axis for datetime
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
plt.xticks(rotation=45, ha="right")

# Add clear labels with units in brackets
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Horizontal Position [mm]", fontsize=12)
ax.set_title("BPM Horizontal Positions - Last 24 Hours", fontsize=14, fontweight="bold")

# Add grid for easier reading (subtle)
ax.grid(True, alpha=0.3, linestyle="--")

# Add legend (outside plot area to avoid obscuring data)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9, fontsize=10)

# Use tight_layout to prevent label cutoff
plt.tight_layout()

# Save plot to figures/ subdirectory (framework will find it automatically)
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
plot_path = figures_dir / "bpm_positions_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

# Close figure to free memory (framework already captured the file)
plt.close(fig)

# Calculate statistics for all channels
all_positions = np.concatenate([pos for pos in bpm_positions.values()])

# Store results for framework
results = {
    "plot_path": str(plot_path),  # Convert Path to string for JSON serialization
    "channels_plotted": list(bpm_positions.keys()),
    "num_channels": len(bpm_positions),
    "mean_position": float(np.mean(all_positions)),
    "std_position": float(np.std(all_positions)),
    "description": f"Time series plot of {len(bpm_positions)} BPM positions over 24 hours (all on same axes)",
}
