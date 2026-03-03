"""Local momentum aperture worker — dp acceptance vs s-position.

Computes the momentum acceptance at reference points around one sector
of the ring using pyAT's ``get_momentum_acceptance``.  Overlays a
lattice element strip showing magnet locations and types.
"""

from __future__ import annotations

from typing import Any

import at
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_state,
    parse_args,
    save_data,
)


def compute_lma(
    ring: at.Lattice,
    n_refpts: int = 100,
    nturns: int = 512,
    dp_max: float = 0.05,
    sector_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local momentum acceptance over one sector.

    Returns (s_pos, dp_plus, dp_minus) arrays.
    dp_plus/dp_minus are positive/negative momentum acceptance at each refpt.
    """
    circumference = float(ring.get_s_pos(len(ring))[0])
    if sector_length is None:
        sector_length = circumference

    # Select reference points within one sector
    s_all = ring.get_s_pos(range(len(ring) + 1))
    refpts = [i for i in range(len(ring)) if s_all[i] < sector_length]

    # Subsample if too many
    if len(refpts) > n_refpts:
        indices = np.linspace(0, len(refpts) - 1, n_refpts, dtype=int)
        refpts = [refpts[i] for i in indices]

    resolution = dp_max / 50

    try:
        dp_plus, dp_minus = at.get_momentum_acceptance(
            ring,
            resolution,
            dp_max,
            nturns=nturns,
            refpts=refpts,
        )
    except Exception:
        # Fallback: return empty arrays
        return np.array([]), np.array([]), np.array([])

    s_pos = np.array([float(s_all[i]) for i in refpts])

    return s_pos, np.array(dp_plus), np.array(dp_minus)


def extract_lattice_elements(
    ring: at.Lattice,
    sector_length: float,
) -> list[dict[str, Any]]:
    """Extract magnet elements for one sector as a list of dicts.

    Each dict has: s_start, s_end, type, name, strength.
    """
    s_all = ring.get_s_pos(range(len(ring) + 1))
    elements = []

    for i, elem in enumerate(ring):
        s_start = float(s_all[i])
        if s_start >= sector_length:
            break
        s_end = float(s_all[i + 1])
        length = s_end - s_start
        if length < 1e-6:
            continue

        fam = getattr(elem, "FamName", "")

        # Classify element type
        h_val = getattr(elem, "H", None)
        if h_val is None:
            poly_b = getattr(elem, "PolynomB", None)
            if poly_b is not None and len(poly_b) >= 3:
                h_val = poly_b[2]

        k_val = getattr(elem, "K", None)
        bend_angle = getattr(elem, "BendingAngle", None)

        if bend_angle is not None and float(bend_angle) != 0.0:
            elements.append(
                {
                    "s_start": s_start,
                    "s_end": min(s_end, sector_length),
                    "type": "dipole",
                    "name": fam,
                    "strength": float(bend_angle),
                }
            )
        elif h_val is not None and float(h_val) != 0.0:
            elements.append(
                {
                    "s_start": s_start,
                    "s_end": min(s_end, sector_length),
                    "type": "sextupole",
                    "name": fam,
                    "strength": float(h_val),
                }
            )
        elif k_val is not None and float(k_val) != 0.0:
            elements.append(
                {
                    "s_start": s_start,
                    "s_end": min(s_end, sector_length),
                    "type": "quadrupole",
                    "name": fam,
                    "strength": float(k_val),
                }
            )

    return elements


def build_figure(
    s_pos: np.ndarray,
    dp_plus: np.ndarray,
    dp_minus: np.ndarray,
    lattice_elements: list[dict[str, Any]],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    """Build LMA figure with lattice strip overlay."""
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.12, 0.88],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ── Lattice strip (top row) ────────────────────────────
    colors = {
        "dipole": "rgba(100,149,237,0.7)",  # cornflower blue
        "quadrupole": "rgba(220,60,60,0.7)",  # red (focusing)
        "quad_defoc": "rgba(70,130,180,0.7)",  # steel blue (defocusing)
        "sextupole": "rgba(50,180,80,0.7)",  # green
    }

    # Find max strength per type for normalization
    max_strength = {"dipole": 1.0, "quadrupole": 1.0, "sextupole": 1.0}
    for elem in lattice_elements:
        t = elem["type"]
        s = abs(elem["strength"])
        if s > max_strength.get(t, 0):
            max_strength[t] = s

    for elem in lattice_elements:
        t = elem["type"]
        s = elem["strength"]
        h_norm = min(abs(s) / max_strength[t], 1.0) if max_strength[t] > 0 else 0.5

        if t == "quadrupole":
            color = colors["quadrupole"] if s > 0 else colors["quad_defoc"]
            y_base = 0.0
            y_top = h_norm if s > 0 else -h_norm
        elif t == "dipole":
            color = colors["dipole"]
            y_base = 0.0
            y_top = 0.5
        else:
            color = colors["sextupole"]
            y_base = 0.0
            y_top = h_norm if s > 0 else -h_norm

        fig.add_shape(
            type="rect",
            x0=elem["s_start"],
            x1=elem["s_end"],
            y0=y_base,
            y1=y_top,
            fillcolor=color,
            line={"width": 0},
            row=1,
            col=1,
        )

    fig.update_yaxes(
        range=[-1.1, 1.1],
        showticklabels=False,
        showgrid=False,
        zeroline=True,
        zerolinecolor="rgba(128,128,128,0.3)",
        row=1,
        col=1,
    )

    # ── LMA plot (bottom row) ──────────────────────────────

    # Baseline (dashed)
    if baseline is not None:
        bs, bdp_p, bdp_m = baseline
        fig.add_trace(
            go.Scatter(
                x=bs,
                y=bdp_p * 100,
                mode="lines",
                line={"color": "gray", "width": 1, "dash": "dash"},
                name="Baseline dp+",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bs,
                y=bdp_m * 100,
                mode="lines",
                line={"color": "gray", "width": 1, "dash": "dash"},
                name="Baseline dp-",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # Positive acceptance (fill to zero)
    fig.add_trace(
        go.Scatter(
            x=s_pos,
            y=dp_plus * 100,
            mode="lines",
            line={"color": "rgb(31,119,180)", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.15)",
            name="dp+",
            hovertemplate="s = %{x:.1f} m<br>dp+ = %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Negative acceptance (fill to zero)
    fig.add_trace(
        go.Scatter(
            x=s_pos,
            y=-np.abs(dp_minus) * 100,
            mode="lines",
            line={"color": "rgb(255,127,14)", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(255,127,14,0.15)",
            name="dp-",
            hovertemplate="s = %{x:.1f} m<br>dp- = %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="dp [%]",
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="gray",
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="s [m]", gridcolor="lightgray", row=2, col=1)

    fig.update_layout(
        title="Local Momentum Aperture (1 sector)",
        height=500,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        showlegend=False,
    )
    return fig


def main() -> None:
    state_path, output_path = parse_args()
    state = load_state(state_path)

    ring = load_ring(state)
    circumference = float(ring.get_s_pos(len(ring))[0])
    periodicity = state.get("summary", {}).get("periodicity", 1)
    sector_length = circumference / periodicity

    s_pos, dp_plus, dp_minus = compute_lma(ring, sector_length=sector_length)
    lattice_elements = extract_lattice_elements(ring, sector_length)

    raw: dict = {
        "s_pos": s_pos.tolist(),
        "dp_plus": dp_plus.tolist(),
        "dp_minus": dp_minus.tolist(),
        "lattice_elements": lattice_elements,
        "baseline": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bs, bdp_p, bdp_m = compute_lma(baseline_ring, sector_length=sector_length)
        raw["baseline"] = {
            "s_pos": bs.tolist(),
            "dp_plus": bdp_p.tolist(),
            "dp_minus": bdp_m.tolist(),
        }

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
