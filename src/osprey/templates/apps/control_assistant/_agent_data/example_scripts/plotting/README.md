# Plotting Example Scripts

This directory contains example Python scripts that demonstrate **best practices for creating high-quality plots** in control system applications.

## Purpose

When using the **Claude Code generator**, these examples are automatically provided to Claude during the code generation process. Claude reads these scripts to learn your preferred plotting style and conventions, then applies those patterns when generating code for your requests.

## Quick Decision Guide

**Should I use separate subplots or plot on same axes?**

```
Do all channels have the SAME units and comparable scales?
│
├─ YES (e.g., 10 BPM positions all in [mm])
│  └─ ⭐ Plot ALL on SAME axes with different colors
│     → See: time_series_basic.py
│
└─ NO (e.g., current [A], pressure [Torr], position [mm])
   └─ Use SEPARATE subplots for each parameter type
      → See: multi_subplot.py or aligned_multiple_plots.py
```

## Example Scripts

### 1. `time_series_basic.py` - ⭐ Multiple Similar Channels on Same Plot
**Best for:** Plotting MULTIPLE SIMILAR channels (same units, comparable scales)

**Key Pattern:** When you have many similar measurements (e.g., 20 BPM positions all in [mm]),
plot them ALL on the SAME axes with different colors. DO NOT create separate subplots for each.

**Demonstrates:**
- Multiple similar traces on same axes (color-coded)
- Clear axis labels with units in brackets `[unit]`
- Proper datetime formatting on x-axis
- Subtle grid (alpha=0.3) for readability
- Color-blind friendly palette for multiple traces
- Legend positioned outside plot area
- `tight_layout()` to prevent label cutoff
- Saving plot path to `results` dictionary

### 2. `multi_subplot.py` - Multiple Parameter Comparison
**Best for:** Comparing DIFFERENT types of measurements (different units/scales)

**When to use separate subplots:**
- Channels have different units (e.g., current [A] vs. pressure [Torr])
- Data requires different scales (linear vs. log scale)
- Comparing fundamentally different parameter types

**When NOT to use (plot on same axes instead):**
- Multiple similar channels with same units (e.g., 20 BPM positions all in [mm])
- Data ranges are comparable and can share a y-axis
- See `time_series_basic.py` for multiple similar traces

**Demonstrates:**
- Shared x-axis across subplots
- Color-blind friendly palette
- Individual subplot titles
- Consistent labeling
- Log scale for pressure data
- Overall figure title

### 3. `publication_quality.py` - Professional Publication Plots
**Best for:** Plots for papers, reports, or presentations

**Demonstrates:**
- High DPI (300) for print quality
- Error bars with uncertainty
- Statistical annotations (mean, std dev, max deviation)
- Professional styling with appropriate fonts
- Minor grid lines
- Comprehensive legend
- Detailed results dictionary

### 4. `aligned_multiple_plots.py` - ⭐ Aligned Multiple Plot Types
**Best for:** Creating multiple DIFFERENT plot types that look aligned when viewed together

**Common Problem Solved:**
- User asks: "Plot time series and correlation matrix"
- Code generates two separate plots with different widths
- They look misaligned and awkward when viewed together

**Demonstrates:**
- **Consistent figure width** across different plot types
- Time series + correlation matrix example
- Two approaches:
  - **Separate files:** Same width for perfect alignment
  - **Combined file:** Guaranteed alignment in single image
- Proper subplot configuration
- GridSpec for flexible layout
- Multiple plot paths in results dictionary

## How It Works

When you ask the assistant to create a plot, the Claude Code generator:

1. **Scans** these example scripts to identify relevant patterns
2. **Plans** how to apply these patterns to your specific request
3. **Generates** code following the best practices shown here

## Adding Your Own Examples

You can add your own example scripts to customize Claude's code generation:

```bash
# Add a new example
cd _agent_data/example_scripts/plotting/
# Create your_example.py with clear comments explaining the patterns
```

**Tips for good example scripts:**
- Use clear, descriptive variable names
- Add comments explaining WHY you do things, not just WHAT
- Include complete working examples
- Store results in the `results` dictionary
- Save figures to `figures/` subdirectory (not `/tmp/`)
- Close figures after saving: `plt.close(fig)` (good for memory)

## Best Practices Summary

All examples follow these core principles:

✓ **Labels with units:** Always use `[unit]` notation (e.g., "Current [mA]")
✓ **High DPI:** Save plots with `dpi=300` for quality
✓ **Tight layout:** Use `plt.tight_layout()` before saving
✓ **Grid:** Add subtle grid with `alpha=0.3` for readability
✓ **Colors:** Use color-blind friendly palettes
✓ **Save location:** Save to `figures/` subdirectory (framework auto-discovers)
✓ **Memory:** Close figures after saving with `plt.close(fig)`
✓ **Results:** Store plot path and key metrics in `results` dictionary

## Framework Integration

The `results` dictionary is how your code communicates back to the framework:

```python
from pathlib import Path

# Create figures directory in execution folder
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)
plot_path = figures_dir / 'my_plot.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

results = {
    'plot_path': str(plot_path),       # Convert Path to string for JSON
    'mean_value': 123.45,               # Optional: key metrics
    'description': 'Plot description'   # Optional: what was plotted
}
```

**How Figure Discovery Works:**

The framework has TWO ways to find your plots:
1. **File Scanner** (recommended): Scans execution folder for PNG/JPG files
   - Saves to `figures/` subdirectory keeps things organized
   - Works even after figures are closed
   - No special code needed beyond saving the file

2. **Auto-capture** (legacy): Captures open matplotlib figures
   - Only works if figures are still open (not closed)
   - Saves to `figures/` with auto-generated names
   - Less reliable than explicit file saving

**Best Practice:** Save figures explicitly to `figures/` subdirectory as shown above.

The framework automatically:
- Discovers plots in execution folder (any PNG/JPG/SVG)
- Displays them to the user
- Extracts and presents metrics from `results`
- Saves the executed script for future reference

---

**Note:** These examples are only included when you select the **Claude Code generator** during project creation. If you're using the basic generator, you can safely ignore this directory.
