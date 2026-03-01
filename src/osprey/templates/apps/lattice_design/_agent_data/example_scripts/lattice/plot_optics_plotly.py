"""Create an interactive Plotly optics plot saved as an HTML artifact.

Demonstrates:
- Computing full optics (beta functions and dispersion)
- Building a two-panel Plotly figure with shared x-axis
- Saving interactive HTML to the workspace artifacts directory
"""

import at
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load lattice and compute optics ---
ring = at.load_m("machine_data/als.m")
refpts = range(len(ring) + 1)
ld0, ringdata, ld = at.get_optics(ring, refpts=refpts, get_chrom=True)

s_pos = ring.get_s_pos(refpts)
beta_x = ld.beta[:, 0]
beta_y = ld.beta[:, 1]
eta_x = ld.dispersion[:, 0]

# --- Build two-panel figure ---
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Beta Functions", "Horizontal Dispersion"),
)

# Top panel: beta functions
fig.add_trace(
    go.Scatter(x=s_pos, y=beta_x, name="\u03b2<sub>x</sub>",
               line=dict(color="blue", width=1.5)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=s_pos, y=beta_y, name="\u03b2<sub>y</sub>",
               line=dict(color="red", width=1.5)),
    row=1, col=1,
)

# Bottom panel: horizontal dispersion
fig.add_trace(
    go.Scatter(x=s_pos, y=eta_x, name="\u03b7<sub>x</sub>",
               line=dict(color="green", width=1.5)),
    row=2, col=1,
)

# --- Layout ---
fig.update_yaxes(title_text="\u03b2 [m]", row=1, col=1, gridcolor="lightgray")
fig.update_yaxes(title_text="\u03b7 [m]", row=2, col=1, gridcolor="lightgray")
fig.update_xaxes(title_text="s [m]", row=2, col=1, gridcolor="lightgray")
fig.update_layout(
    title="ALS Optics",
    height=600,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# --- Save as interactive HTML artifact ---
output_path = "osprey-workspace/artifacts/optics.html"
fig.write_html(output_path)
print(f"Interactive optics plot saved to {output_path}")
