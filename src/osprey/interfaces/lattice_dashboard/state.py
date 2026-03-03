"""Lattice state manager — JSON-backed state for the dashboard.

Manages ``osprey-workspace/lattice/state.json`` with thread-safe
load/save operations.  Workers and the dashboard server coordinate
through this file.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.lattice_dashboard.state")

FAST_FIGURES = ("optics", "resonance", "chromaticity", "footprint")
VERIFICATION_FIGURES = ("da", "fma")
ALL_FIGURES = FAST_FIGURES + VERIFICATION_FIGURES


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_figure_status() -> dict[str, dict[str, Any]]:
    return {name: {"status": "idle", "updated": None, "error": None} for name in ALL_FIGURES}


class LatticeState:
    """Thread-safe JSON state manager for the lattice dashboard.

    Args:
        state_dir: Directory for state.json / baseline.json.
            Typically ``osprey-workspace/lattice/``.
    """

    def __init__(self, state_dir: Path) -> None:
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._dir / "state.json"
        self._baseline_path = self._dir / "baseline.json"
        self._lock = threading.Lock()

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def figures_dir(self) -> Path:
        d = self._dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Load / Save ───────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text())
        return self._empty_state()

    def save(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._save_unlocked(state)

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        self._state_path.write_text(json.dumps(state, indent=2, default=str))

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "base_lattice": None,
            "overrides": {},
            "summary": {},
            "families": {},
            "figures": _default_figure_status(),
            "baseline": None,
        }

    # ── Initialize ────────────────────────────────────────────

    def initialize(self, lattice_path: str) -> dict[str, Any]:
        """Load a lattice file, discover magnet families, compute summary.

        This performs the heavy pyAT import inside the call so the
        import cost is only paid when actually initializing.
        """
        import at
        import numpy as np

        ring = at.load_m(lattice_path)
        ld0, rd, ld = at.get_optics(ring, get_chrom=True)

        tunes = [float(rd.tune[0]), float(rd.tune[1])]
        chrom = [float(rd.chromaticity[0]), float(rd.chromaticity[1])]
        energy_gev = float(ring.energy) / 1e9
        circumference = float(ring.get_s_pos(len(ring))[0])

        # Discover magnet families
        families: dict[str, dict[str, Any]] = {}
        for elem in ring:
            fam = getattr(elem, "FamName", None)
            if fam is None:
                continue
            if fam in families:
                families[fam]["count"] += 1
                continue

            k_val = getattr(elem, "K", None)
            if k_val is not None:
                families[fam] = {
                    "type": "quadrupole",
                    "param": "K",
                    "value": float(k_val),
                    "count": 1,
                    "range": [-5.0, 5.0],
                }
                continue

            h_val = getattr(elem, "H", None)
            if h_val is None:
                poly_b = getattr(elem, "PolynomB", None)
                if poly_b is not None and len(poly_b) >= 3:
                    h_val = poly_b[2]
            if h_val is not None and float(h_val) != 0.0:
                families[fam] = {
                    "type": "sextupole",
                    "param": "H",
                    "value": float(h_val),
                    "count": 1,
                    "range": [-200.0, 200.0],
                }

        summary = {
            "energy_gev": energy_gev,
            "circumference_m": circumference,
            "tunes": tunes,
            "chromaticity": chrom,
            "num_elements": len(ring),
            "beta_max": [float(np.max(ld.beta[:, 0])), float(np.max(ld.beta[:, 1]))],
        }

        state = {
            "base_lattice": str(lattice_path),
            "overrides": {},
            "summary": summary,
            "families": families,
            "figures": _default_figure_status(),
            "baseline": None,
        }

        self.save(state)

        # Auto-set baseline
        self.set_baseline()

        return state

    # ── Parameter overrides ───────────────────────────────────

    def set_param(self, family: str, value: float) -> dict[str, Any]:
        """Set a magnet family parameter override and mark fast figures stale."""
        with self._lock:
            state = self._load_unlocked()
            state["overrides"][family] = value
            # Mark fast figures as stale
            for fig_name in FAST_FIGURES:
                if state["figures"][fig_name]["status"] == "ready":
                    state["figures"][fig_name]["status"] = "stale"
            # Mark verification figures as stale too
            for fig_name in VERIFICATION_FIGURES:
                if state["figures"][fig_name]["status"] == "ready":
                    state["figures"][fig_name]["status"] = "stale"
            self._save_unlocked(state)
            return state

    # ── Ring loader (with overrides) ──────────────────────────

    def get_ring(self) -> Any:
        """Load the pyAT ring with current overrides applied."""
        import at

        state = self.load()
        lattice_path = state.get("base_lattice")
        if not lattice_path:
            raise ValueError("No lattice loaded — call initialize() first")

        ring = at.load_m(lattice_path)
        overrides = state.get("overrides", {})

        for fam_name, value in overrides.items():
            fam_info = state["families"].get(fam_name, {})
            param = fam_info.get("param", "K")
            for elem in ring:
                if getattr(elem, "FamName", None) == fam_name:
                    setattr(elem, param, value)

        return ring

    # ── Figure status ─────────────────────────────────────────

    def mark_computing(self, figure: str) -> None:
        with self._lock:
            state = self._load_unlocked()
            state["figures"][figure] = {
                "status": "computing",
                "updated": _now_iso(),
                "error": None,
            }
            self._save_unlocked(state)

    def mark_ready(self, figure: str, summary_updates: dict[str, Any] | None = None) -> None:
        with self._lock:
            state = self._load_unlocked()
            state["figures"][figure] = {
                "status": "ready",
                "updated": _now_iso(),
                "error": None,
            }
            if summary_updates:
                state["summary"].update(summary_updates)
            self._save_unlocked(state)

    def mark_error(self, figure: str, error: str) -> None:
        with self._lock:
            state = self._load_unlocked()
            state["figures"][figure] = {
                "status": "error",
                "updated": _now_iso(),
                "error": error,
            }
            self._save_unlocked(state)

    # ── Baseline ──────────────────────────────────────────────

    def set_baseline(self) -> dict[str, Any]:
        """Snapshot current state as the comparison baseline."""
        with self._lock:
            state = self._load_unlocked()
            baseline = {
                "summary": dict(state.get("summary", {})),
                "overrides": dict(state.get("overrides", {})),
                "set_at": _now_iso(),
            }
            state["baseline"] = baseline
            self._save_unlocked(state)
            # Also write a separate baseline.json for workers
            self._baseline_path.write_text(json.dumps(baseline, indent=2, default=str))
            return baseline

    def get_baseline(self) -> dict[str, Any] | None:
        if self._baseline_path.exists():
            return json.loads(self._baseline_path.read_text())
        return None

    def clear_baseline(self) -> None:
        with self._lock:
            state = self._load_unlocked()
            state["baseline"] = None
            self._save_unlocked(state)
        if self._baseline_path.exists():
            self._baseline_path.unlink()
