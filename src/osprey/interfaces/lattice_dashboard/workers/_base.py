"""Shared helpers for lattice dashboard workers.

Each worker is invoked as::

    python -m osprey.interfaces.lattice_dashboard.workers.<name> \\
        <job_path> <output_path>

The job file is the launch's immutable input: the deck, the engine's prepared
settings, the families' parameters, the overrides, the baseline overrides,
the figure's settings group, the figure's key and the job id. A worker reads
nothing else, and writes ``{key, job_id, deck_sha256, data}`` to the output.
This module provides the common boilerplate: argument parsing, job loading,
ring loading (with overrides), baseline ring loading, and data saving.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import at
import numpy as np
import plotly.graph_objects as go


def load_settings(job: dict[str, Any], group: str) -> dict[str, Any]:
    """Return the job's settings group, merged over the group's defaults.

    Workers call this as ``settings = load_settings(job, "da")`` to get a
    complete settings dict even when the job carries fewer keys.
    """
    from osprey.interfaces.lattice_dashboard.state import DEFAULT_SETTINGS

    defaults = DEFAULT_SETTINGS.get(group, {})
    saved = job.get("settings") or {}
    merged = dict(defaults)
    merged.update({k: v for k, v in saved.items() if k in defaults})
    return merged


def unpack_tracking(result: Any) -> np.ndarray:
    """Extract ndarray from ring.track() return value.

    Handles both old API (returns ndarray) and new API (returns tuple).
    Squeezes single-particle dimension for 1-particle tracking.
    """
    data: np.ndarray = result[0] if isinstance(result, tuple) else result
    # Squeeze nparticles dim: (6, nrefpts, 1, nturns) → (6, nrefpts, nturns)
    if data.ndim == 4 and data.shape[2] == 1:
        data = data[:, :, 0, :]
    return data


def parse_args() -> tuple[Path, Path]:
    """Parse CLI args: job_path, output_path."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <job_path> <output_path>", file=sys.stderr)
        sys.exit(1)
    return Path(sys.argv[1]), Path(sys.argv[2])


def load_job(job_path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(job_path.read_text()))


def _lattice_with(job: dict[str, Any], overrides: dict[str, float]) -> at.Lattice:
    ring = at.load_lattice(job["deck"])
    families = job.get("families", {})
    for fam_name, value in overrides.items():
        param = families.get(fam_name, "K")
        for elem in ring:
            if getattr(elem, "FamName", None) == fam_name:
                setattr(elem, param, value)
    return ring


def load_ring(job: dict[str, Any]) -> at.Lattice:
    """Load the job's deck with its overrides applied."""
    return _lattice_with(job, job.get("overrides") or {})


def load_baseline_ring(job: dict[str, Any]) -> at.Lattice | None:
    """Load the job's deck with its baseline overrides applied, or None with no baseline."""
    baseline_overrides = job.get("baseline_overrides")
    if baseline_overrides is None:
        return None
    return _lattice_with(job, baseline_overrides)


def prepared_twiss_in(job: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Return a ``single_pass`` job's prepared ``twiss_in`` as arrays, else None."""
    from osprey.interfaces.lattice_dashboard.state import SINGLE_PASS

    prepared = job.get("prepared") or {}
    if prepared.get("solve") != SINGLE_PASS or prepared.get("twiss_in") is None:
        return None
    return {key: np.asarray(values, dtype=float) for key, values in prepared["twiss_in"].items()}


def _numpy_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_data(job: dict[str, Any], data: dict[str, Any], output_path: Path) -> None:
    """Save raw physics data as plain JSON (no Plotly, no bdata) under the job's key.

    The file is written through a temporary file, so no reader sees half of it.
    """
    payload = {
        "key": job.get("key"),
        "job_id": job.get("job_id"),
        "deck_sha256": job.get("deck_sha256"),
        "data": data,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, default=_numpy_default))
    tmp.replace(output_path)


def figure_to_dict(fig: Any) -> dict[str, Any]:
    """Convert a Plotly figure to a JSON-safe dict (no numpy types)."""
    return cast(dict[str, Any], json.loads(json.dumps(fig.to_dict(), default=_numpy_default)))


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
