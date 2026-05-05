"""Shared helpers for lattice dashboard workers.

Each worker is invoked as::

    python -m osprey.interfaces.lattice_dashboard.workers.<name> \\
        <state_path> <output_path>

This module provides the common boilerplate: argument parsing, ring
loading (with overrides), baseline ring loading, and data saving.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import at
import numpy as np
import plotly.graph_objects as go


def load_settings(state: dict[str, Any], group: str) -> dict[str, Any]:
    """Load settings for a worker group, merged with defaults.

    Workers call this as ``settings = load_settings(state, "da")``
    to get a complete settings dict even if state.json is missing keys.
    """
    from osprey.interfaces.lattice_dashboard.state import DEFAULT_SETTINGS

    defaults = DEFAULT_SETTINGS.get(group, {})
    saved = state.get("settings", {}).get(group, {})
    merged = dict(defaults)
    merged.update({k: v for k, v in saved.items() if k in defaults})
    return merged


def unpack_tracking(result: Any) -> np.ndarray:
    """Extract ndarray from ring.track() return value.

    Handles both old API (returns ndarray) and new API (returns tuple).
    Squeezes single-particle dimension for 1-particle tracking.
    """
    if isinstance(result, tuple):
        data = result[0]
    else:
        data = result
    # Squeeze nparticles dim: (6, nrefpts, 1, nturns) → (6, nrefpts, nturns)
    if data.ndim == 4 and data.shape[2] == 1:
        data = data[:, :, 0, :]
    return data


def parse_args() -> tuple[Path, Path]:
    """Parse CLI args: state_path, output_path."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <state_path> <output_path>", file=sys.stderr)
        sys.exit(1)
    return Path(sys.argv[1]), Path(sys.argv[2])


def load_state(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_text())


def load_ring(state: dict[str, Any]) -> at.Lattice:
    """Load pyAT ring with parameter overrides applied."""
    lattice_path = state["base_lattice"]
    ring = at.load_lattice(lattice_path)
    overrides = state.get("overrides", {})
    families = state.get("families", {})

    for fam_name, value in overrides.items():
        param = families.get(fam_name, {}).get("param", "K")
        for elem in ring:
            if getattr(elem, "FamName", None) == fam_name:
                setattr(elem, param, value)
    return ring


def load_baseline_ring(state_path: Path, state: dict[str, Any]) -> at.Lattice | None:
    """Load baseline ring if baseline.json exists alongside state.json."""
    baseline_path = state_path.parent / "baseline.json"
    if not baseline_path.exists():
        return None

    baseline = json.loads(baseline_path.read_text())
    baseline_overrides = baseline.get("overrides", {})

    lattice_path = state["base_lattice"]
    ring = at.load_lattice(lattice_path)
    families = state.get("families", {})

    for fam_name, value in baseline_overrides.items():
        param = families.get(fam_name, {}).get("param", "K")
        for elem in ring:
            if getattr(elem, "FamName", None) == fam_name:
                setattr(elem, param, value)
    return ring


def _numpy_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_data(data: dict[str, Any], output_path: Path) -> None:
    """Save raw physics data as plain JSON (no Plotly, no bdata)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, default=_numpy_default))


def figure_to_dict(fig: Any) -> dict[str, Any]:
    """Convert a Plotly figure to a JSON-safe dict (no numpy types)."""
    return json.loads(json.dumps(fig.to_dict(), default=_numpy_default))


def add_resonance_overlay(
    fig: go.Figure,
    nux_range: tuple[float, float],
    nuy_range: tuple[float, float],
) -> None:
    """Add light resonance lines to a tune-space figure.

    Draws lines m*nux + n*nuy = p for orders 1-4 as faint gray
    lines, giving context for resonance proximity without visual clutter.
    """
    for order in range(1, 5):
        for m in range(-order, order + 1):
            n_abs = order - abs(m)
            for n in [n_abs, -n_abs] if n_abs != 0 else [0]:
                if m == 0 and n == 0:
                    continue
                for p in range(-100, 101):
                    pts: list[tuple[float, float]] = []
                    if n != 0:
                        for nx_edge in nux_range:
                            ny_val = (p - m * nx_edge) / n
                            if nuy_range[0] <= ny_val <= nuy_range[1]:
                                pts.append((nx_edge, ny_val))
                    if m != 0:
                        for ny_edge in nuy_range:
                            nx_val = (p - n * ny_edge) / m
                            if nux_range[0] <= nx_val <= nux_range[1]:
                                pts.append((nx_val, ny_edge))
                    if len(pts) >= 2:
                        pts = sorted(set(pts))
                        fig.add_trace(
                            go.Scatter(
                                x=[pts[0][0], pts[-1][0]],
                                y=[pts[0][1], pts[-1][1]],
                                mode="lines",
                                line={"color": "rgba(128,128,128,0.2)", "width": 0.5},
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
