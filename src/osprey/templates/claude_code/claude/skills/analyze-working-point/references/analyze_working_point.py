"""Quantitative working point analysis with physics-based resonance classification.

Demonstrates:
- Computing proximity to all resonances m*nu_x + n*nu_y = p (orders 1-5)
- Classifying resonances by physics type (not just order)
- Printing a ranked danger table
- Building an annotated Plotly figure highlighting the most threatening resonances
"""

import math

import at
import plotly.graph_objects as go

# ── Category priority (lower = more dangerous) ───────────────────────
CATEGORY_PRIORITY = {
    "CRITICAL": 0,
    "WARNING": 1,
    "DANGEROUS": 2,
    "LOW": 3,
}

CATEGORY_COLORS = {
    "CRITICAL": "red",
    "WARNING": "orange",
    "DANGEROUS": "green",
    "LOW": "steelblue",
}


def classify_resonance(m: int, n: int) -> tuple[str, str]:
    """Classify a resonance by its physics type.

    Returns (category, type_label) based on the resonance coefficients.
    The classification follows Wiedemann Ch15-16:
    - Order 1: integer/half-integer → always CRITICAL
    - Order 2 with same-sign coefficients: sum resonance → CRITICAL
    - Order 2 with opposite-sign coefficients: difference resonance → WARNING
    - Order 3: sextupole-driven → DANGEROUS
    - Order 4+: higher-order → LOW
    """
    order = abs(m) + abs(n)

    if order == 1:
        return "CRITICAL", "Integer/half-integer"
    if order == 2:
        # Sum resonance: both coefficients have the same sign
        # (or one is zero, which is a half-integer of a single plane)
        if m == 0 or n == 0:
            return "CRITICAL", "Half-integer"
        if m * n > 0:
            return "CRITICAL", "Sum coupling"
        return "WARNING", "Difference coupling"
    if order == 3:
        return "DANGEROUS", "Sextupole-driven"
    return "LOW", f"Order {order}"


def format_condition(m: int, n: int, p: int) -> str:
    """Format a resonance condition as a readable string like '2·νx + νy = 37'."""
    parts = []
    for coeff, label in [(m, "νx"), (n, "νy")]:
        if coeff == 0:
            continue
        if coeff == 1:
            term = label
        elif coeff == -1:
            term = f"-{label}"
        else:
            term = f"{coeff}·{label}"
        parts.append(term)

    lhs = parts[0] if len(parts) == 1 else " + ".join(parts)
    # Clean up "+ -" to "- "
    lhs = lhs.replace("+ -", "- ")
    return f"{lhs} = {p}"


# ── Load lattice and get working point ────────────────────────────────
ring = at.load_m("machine_data/als.m")
tunes = at.get_tune(ring)
nux, nuy = tunes[0], tunes[1]

print("Working Point Analysis")
print("=" * 60)
print(f"  Working point: νx = {nux:.4f}, νy = {nuy:.4f}")
print()

# ── Compute proximity to all resonances ──────────────────────────────
PROXIMITY_THRESHOLD = 0.05
nearby = []

for order in range(1, 6):
    for m in range(-order, order + 1):
        n_abs = order - abs(m)
        for n in ([n_abs, -n_abs] if n_abs != 0 else [0]):
            if m == 0 and n == 0:
                continue
            # Find integer p values near m*nux + n*nuy
            resonance_val = m * nux + n * nuy
            p_center = round(resonance_val)
            for p in range(p_center - 1, p_center + 2):
                norm = math.sqrt(m * m + n * n)
                d = abs(m * nux + n * nuy - p) / norm
                if d < PROXIMITY_THRESHOLD:
                    category, type_label = classify_resonance(m, n)
                    nearby.append({
                        "m": m,
                        "n": n,
                        "p": p,
                        "order": order,
                        "distance": d,
                        "category": category,
                        "type_label": type_label,
                        "condition": format_condition(m, n, p),
                    })

# ── Deduplicate: (m, n, p) ≡ (-m, -n, -p) ───────────────────────────
seen = set()
unique = []
for r in nearby:
    key = (r["m"], r["n"], r["p"])
    neg_key = (-r["m"], -r["n"], -r["p"])
    # Canonical form: pick the one with positive first nonzero coefficient
    if key in seen or neg_key in seen:
        continue
    seen.add(key)
    unique.append(r)

# ── Sort: category priority first, then distance ─────────────────────
unique.sort(key=lambda r: (CATEGORY_PRIORITY[r["category"]], r["distance"]))

# ── Print danger table ───────────────────────────────────────────────
top_n = min(10, len(unique))
print(f"Top {top_n} nearest resonances:")
print(f"{'Rank':<6}{'Resonance':<22}{'Type':<26}{'Category':<12}{'Distance':<10}")
print("─" * 76)
for i, r in enumerate(unique[:top_n], 1):
    print(
        f"{i:<6}{r['condition']:<22}{r['type_label']:<26}"
        f"{r['category']:<12}{r['distance']:.4f}"
    )
print()

# ── Build annotated Plotly figure ─────────────────────────────────────
tune_range_x = (nux - 0.1, nux + 0.1)
tune_range_y = (nuy - 0.1, nuy + 0.1)

order_colors = {1: "red", 2: "orange", 3: "green", 4: "blue", 5: "purple"}
fig = go.Figure()

# Background resonance lines (semi-transparent)
for order in range(1, 6):
    for m in range(-order, order + 1):
        n_abs = order - abs(m)
        for n in ([n_abs, -n_abs] if n_abs != 0 else [0]):
            if m == 0 and n == 0:
                continue
            for p in range(-100, 101):
                pts = []
                if n != 0:
                    for nx_edge in tune_range_x:
                        ny_val = (p - m * nx_edge) / n
                        if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                            pts.append((nx_edge, ny_val))
                if m != 0:
                    for ny_edge in tune_range_y:
                        nx_val = (p - n * ny_edge) / m
                        if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                            pts.append((nx_val, ny_edge))
                if n == 0 and m != 0:
                    nx_val = p / m
                    if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                        pts.append((nx_val, tune_range_y[0]))
                        pts.append((nx_val, tune_range_y[1]))
                if m == 0 and n != 0:
                    ny_val = p / n
                    if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                        pts.append((tune_range_x[0], ny_val))
                        pts.append((tune_range_x[1], ny_val))

                if len(pts) >= 2:
                    pts = sorted(set(pts))
                    fig.add_trace(
                        go.Scatter(
                            x=[pts[0][0], pts[-1][0]],
                            y=[pts[0][1], pts[-1][1]],
                            mode="lines",
                            line=dict(
                                color=order_colors[order],
                                width=max(3 - order * 0.4, 0.5),
                            ),
                            opacity=0.25,
                            name=f"Order {order}",
                            legendgroup=f"order{order}",
                            showlegend=False,
                            hovertemplate=(
                                f"{m}·ν<sub>x</sub> + {n}·ν<sub>y</sub> = {p}"
                                "<extra></extra>"
                            ),
                        )
                    )

# Legend entries for background orders
for order, color in order_colors.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=color, width=max(3 - order * 0.4, 0.5)),
            opacity=0.25,
            name=f"Order {order}",
            legendgroup=f"order{order}",
            showlegend=True,
        )
    )

# ── Highlight top 5 closest resonances ────────────────────────────────
top_highlight = min(5, len(unique))
for i, r in enumerate(unique[:top_highlight]):
    m, n, p = r["m"], r["n"], r["p"]
    color = CATEGORY_COLORS[r["category"]]

    # Compute line endpoints in the visible region
    pts = []
    if n != 0:
        for nx_edge in tune_range_x:
            ny_val = (p - m * nx_edge) / n
            if tune_range_y[0] <= ny_val <= tune_range_y[1]:
                pts.append((nx_edge, ny_val))
    if m != 0:
        for ny_edge in tune_range_y:
            nx_val = (p - n * ny_edge) / m
            if tune_range_x[0] <= nx_val <= tune_range_x[1]:
                pts.append((nx_val, ny_edge))
    if n == 0 and m != 0:
        nx_val = p / m
        if tune_range_x[0] <= nx_val <= tune_range_x[1]:
            pts.append((nx_val, tune_range_y[0]))
            pts.append((nx_val, tune_range_y[1]))
    if m == 0 and n != 0:
        ny_val = p / n
        if tune_range_y[0] <= ny_val <= tune_range_y[1]:
            pts.append((tune_range_x[0], ny_val))
            pts.append((tune_range_x[1], ny_val))

    if len(pts) >= 2:
        pts = sorted(set(pts))
        fig.add_trace(
            go.Scatter(
                x=[pts[0][0], pts[-1][0]],
                y=[pts[0][1], pts[-1][1]],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"#{i + 1} {r['condition']}",
                legendgroup=f"highlight{i}",
                showlegend=True,
                hovertemplate=(
                    f"#{i + 1} {r['condition']}<br>"
                    f"{r['type_label']} ({r['category']})<br>"
                    f"Distance: {r['distance']:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        # Find the point on the line closest to the working point for annotation
        x1, y1 = pts[0]
        x2, y2 = pts[-1]
        dx, dy = x2 - x1, y2 - y1
        if dx * dx + dy * dy > 0:
            t = ((nux - x1) * dx + (nuy - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            ax_pt = x1 + t * dx
            ay_pt = y1 + t * dy
        else:
            ax_pt, ay_pt = x1, y1

        fig.add_annotation(
            x=ax_pt,
            y=ay_pt,
            text=f"#{i + 1} {r['condition']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
        )

# ── Working point marker ─────────────────────────────────────────────
fig.add_trace(
    go.Scatter(
        x=[nux],
        y=[nuy],
        mode="markers",
        marker=dict(size=14, color="black", symbol="star"),
        name="Working point",
        hovertemplate=(
            f"ν<sub>x</sub> = {nux:.4f}<br>ν<sub>y</sub> = {nuy:.4f}<extra></extra>"
        ),
    )
)

# ── Layout ────────────────────────────────────────────────────────────
fig.update_layout(
    title="Working Point Analysis — Resonance Proximity",
    xaxis_title="ν<sub>x</sub> (fractional)",
    yaxis_title="ν<sub>y</sub> (fractional)",
    height=700,
    width=800,
    template="plotly_white",
    xaxis=dict(range=list(tune_range_x), gridcolor="lightgray"),
    yaxis=dict(range=list(tune_range_y), gridcolor="lightgray", scaleanchor="x"),
)

# ── Save ──────────────────────────────────────────────────────────────
save_artifact(
    fig,
    "Working Point Analysis",
    "Resonance proximity analysis with physics-based classification",
)
print("Working point analysis diagram saved.")

# ── Before/After Comparison (uncomment after /fit-tune) ──────────────
# To compare before and after a tune change:
#
# # Store initial results
# nux_before, nuy_before = nux, nuy
# danger_before = unique[:10]
#
# # After /fit-tune, reload and re-analyze
# ring_after = at.load_m("machine_data/als.m")  # or use modified ring
# tunes_after = at.get_tune(ring_after)
# nux_after, nuy_after = tunes_after[0], tunes_after[1]
#
# # Add "after" working point to the figure
# fig.add_trace(
#     go.Scatter(
#         x=[nux_after],
#         y=[nuy_after],
#         mode="markers",
#         marker=dict(
#             size=14, color="gold", symbol="star",
#             line=dict(color="black", width=1.5),
#         ),
#         name="New working point",
#     )
# )
