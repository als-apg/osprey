"""Shared helpers for lattice dashboard workers.

Each worker is invoked as::

    python -m osprey.interfaces.lattice_dashboard.workers.<name> \\
        <state_path> <output_path>

This module provides the common boilerplate: argument parsing, ring
loading (with overrides), baseline ring loading, and figure saving.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import at


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
    ring = at.load_m(lattice_path)
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
    ring = at.load_m(lattice_path)
    families = state.get("families", {})

    for fam_name, value in baseline_overrides.items():
        param = families.get(fam_name, {}).get("param", "K")
        for elem in ring:
            if getattr(elem, "FamName", None) == fam_name:
                setattr(elem, param, value)
    return ring


def save_figure(fig: Any, output_path: Path) -> None:
    """Save a Plotly figure as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fig.to_json())
