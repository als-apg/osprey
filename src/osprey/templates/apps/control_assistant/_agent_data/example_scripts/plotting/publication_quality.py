"""
Publication-Quality Plot - Example Script

This script demonstrates ALL best practices for creating
publication-ready plots with professional formatting.

Key practices demonstrated:
- High DPI (300) for print quality
- Proper font sizes for readability
- Color-blind friendly palette
- Error bars for uncertainty
- Statistical annotations
- Professional styling
- Multiple data series
- Comprehensive legend
- Axis range optimization
- Professional color scheme
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality plot parameters
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)

# Generate example data with uncertainties
hours = np.linspace(0, 24, 100)
times = [datetime.now() - timedelta(hours=24 - h) for h in hours]

# Simulate quadrupole magnet currents (theoretical vs actual)
theoretical = 150 * np.ones_like(hours)
actual = 150 + 2 * np.sin(2 * np.pi * hours / 12) + np.random.normal(0, 0.5, len(hours))
uncertainty = 0.3 * np.ones_like(hours)

# Sample points for error bars (every 10th point to avoid clutter)
sample_indices = np.arange(0, len(hours), 10)

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(12, 7))

# Plot theoretical setpoint (dashed line)
ax.plot(
    times,
    theoretical,
    "--",
    linewidth=2.5,
    color="#555555",
    label="Setpoint (150 A)",
    alpha=0.8,
    zorder=1,
)

# Plot actual measured values with error bars
ax.errorbar(
    [times[i] for i in sample_indices],
    [actual[i] for i in sample_indices],
    yerr=[uncertainty[i] for i in sample_indices],
    fmt="o",
    color="#0077BB",
    ecolor="#0077BB",
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    markersize=6,
    label="Measured ± 0.3 A",
    alpha=0.9,
    zorder=3,
)

# Plot smooth curve through actual data
ax.plot(times, actual, "-", linewidth=2, color="#0077BB", alpha=0.6, zorder=2)

# Configure axes
ax.set_xlabel("Time", fontsize=12, fontweight="bold")
ax.set_ylabel("Magnet Current [A]", fontsize=12, fontweight="bold")
ax.set_title(
    "Focusing Quadrupole Q4F2 - Current Stability Analysis", fontsize=14, fontweight="bold", pad=15
)

# Format time axis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Set appropriate y-axis range (centered on mean with margin)
mean_current = np.mean(actual)
y_margin = 5
ax.set_ylim(mean_current - y_margin, mean_current + y_margin)

# Add professional grid
ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, which="major")
ax.grid(True, alpha=0.1, linestyle=":", linewidth=0.5, which="minor")
ax.minorticks_on()

# Add statistical annotations
textstr = "\n".join(
    [
        f"Mean: {mean_current:.2f} A",
        f"Std Dev: {np.std(actual):.3f} A",
        f"Max Deviation: {np.max(np.abs(actual - theoretical)):.3f} A",
    ]
)
props = {
    "boxstyle": "round",
    "facecolor": "wheat",
    "alpha": 0.8,
    "edgecolor": "black",
    "linewidth": 1,
}
ax.text(
    0.02,
    0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=props,
    family="monospace",
)

# Add legend with professional styling
ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fancybox=False, shadow=False)

# Ensure tight layout for publication
plt.tight_layout()

# Save to figures/ subdirectory (framework will find it automatically)
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
plot_path = figures_dir / "publication_quality_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)

# Store comprehensive results
results = {
    "plot_path": str(plot_path),  # Convert Path to string for JSON serialization
    "mean_current": float(mean_current),
    "std_deviation": float(np.std(actual)),
    "max_deviation": float(np.max(np.abs(actual - theoretical))),
    "setpoint": 150.0,
    "measurement_uncertainty": 0.3,
    "time_range_hours": 24,
    "num_samples": len(actual),
    "description": "Publication-quality plot with error bars and statistical analysis",
}
