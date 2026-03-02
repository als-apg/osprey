"""Plot the working point on a tune diagram with resonance lines.

Demonstrates:
- Getting fractional tunes with at.get_tune()
- Computing resonance lines m*nu_x + n*nu_y = p for orders 1-5
- Building a Plotly figure with color-coded resonance lines
- Marking the working point and optional before/after comparison
"""

import at
import plotly.graph_objects as go

# --- Load lattice and get working point ---
ring = at.load_m("machine_data/als.m")
tunes = at.get_tune(ring)
nux, nuy = tunes[0], tunes[1]

print("Resonance Diagram")
print("=" * 50)
print(f"  Working point: nu_x = {nux:.4f}, nu_y = {nuy:.4f}")

# --- Compute resonance lines ---
# Resonance condition: m*nu_x + n*nu_y = p (integers)
# Order = |m| + |n|; lower orders are more dangerous
tune_range_x = (nux - 0.1, nux + 0.1)
tune_range_y = (nuy - 0.1, nuy + 0.1)

order_colors = {
    1: "red",
    2: "orange",
    3: "green",
    4: "blue",
    5: "purple",
}

fig = go.Figure()

for order in range(1, 6):
    for m in range(-order, order + 1):
        n = order - abs(m)
        for n_sign in ([n, -n] if n != 0 else [0]):
            if m == 0 and n_sign == 0:
                continue
            # Find integer p values that put the line in the visible region
            for p in range(-100, 101):
                # Line: m*nu_x + n_sign*nu_y = p
                # Compute endpoints at the edges of the tune range
                pts = []
                if n_sign != 0:
                    for nx_edge in tune_range_x:
                        ny_val = (p - m * nx_edge) / n_sign
                        if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                            pts.append((nx_edge, ny_val))
                if m != 0:
                    for ny_edge in tune_range_y:
                        nx_val = (p - n_sign * ny_edge) / m
                        if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                            pts.append((nx_val, ny_edge))
                if n_sign == 0 and m != 0:
                    nx_val = p / m
                    if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                        pts.append((nx_val, tune_range_y[0]))
                        pts.append((nx_val, tune_range_y[1]))
                if m == 0 and n_sign != 0:
                    ny_val = p / n_sign
                    if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                        pts.append((tune_range_x[0], ny_val))
                        pts.append((tune_range_x[1], ny_val))

                if len(pts) >= 2:
                    # Deduplicate and sort
                    pts = list(set(pts))
                    pts.sort()
                    fig.add_trace(
                        go.Scatter(
                            x=[pts[0][0], pts[-1][0]],
                            y=[pts[0][1], pts[-1][1]],
                            mode="lines",
                            line=dict(
                                color=order_colors[order],
                                width=max(3 - order * 0.4, 0.5),
                            ),
                            name=f"Order {order}",
                            legendgroup=f"order{order}",
                            showlegend=False,
                            hovertemplate=(
                                f"{m}·ν<sub>x</sub> + {n_sign}·ν<sub>y</sub> = {p}"
                                "<extra></extra>"
                            ),
                        )
                    )

# Add one legend entry per order
for order, color in order_colors.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=color, width=max(3 - order * 0.4, 0.5)),
            name=f"Order {order}",
            legendgroup=f"order{order}",
            showlegend=True,
        )
    )

# --- Mark working point ---
fig.add_trace(
    go.Scatter(
        x=[nux],
        y=[nuy],
        mode="markers",
        marker=dict(size=12, color="black", symbol="star"),
        name="Working point",
        hovertemplate=f"ν<sub>x</sub> = {nux:.4f}<br>ν<sub>y</sub> = {nuy:.4f}<extra></extra>",
    )
)

# --- Layout ---
fig.update_layout(
    title="ALS Resonance Diagram",
    xaxis_title="ν<sub>x</sub> (fractional)",
    yaxis_title="ν<sub>y</sub> (fractional)",
    height=700,
    width=700,
    template="plotly_white",
    xaxis=dict(range=list(tune_range_x), gridcolor="lightgray"),
    yaxis=dict(range=list(tune_range_y), gridcolor="lightgray", scaleanchor="x"),
)

# --- Save ---
save_artifact(fig, "ALS Resonance Diagram", "Working point with resonance lines orders 1-5")

# Alternative: fig.write_html("osprey-workspace/artifacts/resonance_diagram.html")
print("\nResonance diagram saved.")
