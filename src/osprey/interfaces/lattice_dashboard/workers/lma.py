"""Local momentum aperture worker — dp acceptance vs s-position.

Computes the momentum acceptance at reference points around one sector
of the ring using a bisection search (matching the DA algorithm).
Overlays a lattice element strip showing magnet locations and types.
"""

from __future__ import annotations

from typing import Any

import at
import numpy as np
import plotly.graph_objects as go

from osprey.interfaces.lattice_dashboard.workers._base import (
    load_baseline_ring,
    load_ring,
    load_settings,
    load_state,
    parse_args,
    save_data,
    unpack_tracking,
)


def find_ma_at_refpt(
    ring: at.Lattice,
    refpt: int,
    nturns: int,
    dp_max: float,
    n_bisect: int,
) -> tuple[float, float]:
    """Binary search for momentum acceptance at a reference point.

    Returns (dp_plus, dp_minus) — both positive values representing
    the positive and negative momentum acceptance.
    """
    rotated = ring.rotate(refpt)

    def search(dp_sign: float) -> float:
        lo, hi = 0.0, dp_max
        for _ in range(n_bisect):
            mid = (lo + hi) / 2.0
            rin = np.zeros(6)
            rin[4] = dp_sign * mid
            result = rotated.track(rin, nturns=nturns)
            rout = unpack_tracking(result)
            try:
                survived = np.all(np.isfinite(rout))
            except (ValueError, TypeError):
                survived = False
            if survived:
                lo = mid
            else:
                hi = mid
        return lo

    dp_plus = search(+1.0)
    dp_minus = search(-1.0)
    return dp_plus, dp_minus


def compute_lma(
    ring: at.Lattice,
    n_refpts: int = 100,
    nturns: int = 512,
    dp_max: float = 0.05,
    n_bisect: int = 15,
    sector_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local momentum acceptance over one sector via bisection."""
    circumference = float(ring.get_s_pos(len(ring))[0])
    if sector_length is None:
        sector_length = circumference

    s_all = ring.get_s_pos(range(len(ring) + 1))
    refpts = [i for i in range(len(ring)) if s_all[i] < sector_length]

    if len(refpts) > n_refpts:
        indices = np.linspace(0, len(refpts) - 1, n_refpts, dtype=int)
        refpts = [refpts[i] for i in indices]

    dp_plus = np.zeros(len(refpts))
    dp_minus = np.zeros(len(refpts))
    for i, refpt in enumerate(refpts):
        dp_plus[i], dp_minus[i] = find_ma_at_refpt(ring, refpt, nturns, dp_max, n_bisect)

    s_pos = np.array([float(s_all[r]) for r in refpts])
    return s_pos, dp_plus, dp_minus


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
    n_sectors: int = 1,
) -> go.Figure:
    """Build LMA figure with lattice elements overlaid on the zero line."""
    fig = go.Figure()

    # ── Element scale from dp data ─────────────────────────
    avg_dp = float(np.mean(np.concatenate([dp_plus, np.abs(dp_minus)]))) * 100  # in %
    elem_scale = avg_dp * 0.20  # 20% of average dp height

    # ── Lattice elements on zero line ──────────────────────
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
            y_top = h_norm * elem_scale if s > 0 else -h_norm * elem_scale
        elif t == "dipole":
            color = colors["dipole"]
            y_top = 0.5 * elem_scale
        else:
            color = colors["sextupole"]
            y_top = h_norm * elem_scale if s > 0 else -h_norm * elem_scale

        fig.add_shape(
            type="rect",
            x0=elem["s_start"],
            x1=elem["s_end"],
            y0=0,
            y1=y_top,
            fillcolor=color,
            line={"width": 0},
        )

    # ── LMA traces ─────────────────────────────────────────

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
    )

    fig.update_yaxes(
        title_text="dp [%]", gridcolor="lightgray", zeroline=True, zerolinecolor="gray"
    )
    fig.update_xaxes(title_text="s [m]", gridcolor="lightgray")

    fig.update_layout(
        title=f"Local Momentum Aperture ({n_sectors} sector{'s' if n_sectors > 1 else ''})",
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
    settings = load_settings(state, "lma")
    nturns = settings["nturns"]
    n_refpts = settings["n_refpts"]
    dp_max = settings["dp_max_pct"] / 100.0
    n_bisect = settings["n_bisect"]

    circumference = float(ring.get_s_pos(len(ring))[0])
    periodicity = state.get("summary", {}).get("periodicity", 1)
    n_sectors_setting = settings["n_sectors"]
    n_sectors = n_sectors_setting if n_sectors_setting is not None else periodicity
    sector_length = circumference / n_sectors

    s_pos, dp_plus, dp_minus = compute_lma(
        ring,
        n_refpts=n_refpts,
        nturns=nturns,
        dp_max=dp_max,
        n_bisect=n_bisect,
        sector_length=sector_length,
    )
    lattice_elements = extract_lattice_elements(ring, sector_length)

    raw: dict = {
        "s_pos": s_pos.tolist(),
        "dp_plus": dp_plus.tolist(),
        "dp_minus": dp_minus.tolist(),
        "lattice_elements": lattice_elements,
        "n_sectors": n_sectors,
        "baseline": None,
    }

    baseline_ring = load_baseline_ring(state_path, state)
    if baseline_ring is not None:
        bs, bdp_p, bdp_m = compute_lma(
            baseline_ring,
            n_refpts=n_refpts,
            nturns=nturns,
            dp_max=dp_max,
            n_bisect=n_bisect,
            sector_length=sector_length,
        )
        raw["baseline"] = {
            "s_pos": bs.tolist(),
            "dp_plus": bdp_p.tolist(),
            "dp_minus": bdp_m.tolist(),
        }

    save_data(raw, output_path)


if __name__ == "__main__":
    main()
