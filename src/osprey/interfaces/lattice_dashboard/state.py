"""Lattice state — the selection, the what-if inputs and the figure store.

Everything lives under ``<agent_data>/lattice/``::

    selection.json                       the model the operator picked (shared)
    figures/shared/<figure>/<key>.json   figures of the unmodified deck (shared)
    sessions/<session>/whatif.json       overrides, baseline and settings
    sessions/<session>/jobs/<job>.json   one immutable input file per launch
    sessions/<session>/figures/<figure>/<key>.json   what-if figures

A figure is stored under the content key of its inputs (:func:`figure_key`):
the deck, the engine's prepared settings, the figure's settings group, the
overrides and the baseline overrides. A figure is shown only for the key of
the inputs on screen, so a switch, an override or a settings change needs no
file to be deleted: the new key simply has no file until its worker writes
one. A figure with no override and no baseline override lands in the shared
store, every other one in its session's.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from osprey.simulation.engines.pyat import Prepared

logger = logging.getLogger("osprey.lattice_dashboard.state")

FAST_FIGURES = ("optics", "resonance", "chromaticity", "footprint")
VERIFICATION_FIGURES = ("da", "lma")
ALL_FIGURES = FAST_FIGURES + VERIFICATION_FIGURES

#: The solve a model without ``settings.pyat.solve`` uses.
PERIODIC = "periodic"

#: The solve whose optics start from ``settings.pyat.twiss_in``.
SINGLE_PASS = "single_pass"

#: The only figure a ``single_pass`` model draws: it has no tune, so no figure
#: built on tunes, chromaticity or turn-by-turn tracking applies to it.
OPTICS_ONLY = ("optics",)

#: The refusal a figure route gives a figure the selected model cannot draw.
SINGLE_PASS_UNAVAILABLE = "not available for a single-pass model"

#: The settings group each figure's worker reads; None for a figure with none.
FIGURE_SETTINGS: dict[str, str | None] = {
    "optics": None,
    "resonance": None,
    "chromaticity": "chromaticity",
    "footprint": "footprint",
    "da": "da",
    "lma": "lma",
}

#: The session every request belongs to.
DEFAULT_SESSION = "default"

#: The scope of the figures every session shares.
SHARED_SCOPE = "shared"

#: How many keys of one figure a scope keeps; older ones are pruned on write.
KEPT_KEYS = 8

SelectionStatus = Literal["loading", "ready", "failed", "none"]

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "da": {
        "nturns": 512,
        "n_angles": 19,
        "amp_max_mm": 30.0,
        "n_bisect": 15,
    },
    "lma": {
        "nturns": 512,
        "n_refpts": 15,
        "dp_max_pct": 5.0,
        "n_sectors": None,
        "n_bisect": 15,
    },
    "chromaticity": {
        "dp_min_pct": -3.0,
        "dp_max_pct": 3.0,
        "n_steps": 25,
    },
    "footprint": {
        "n_amp": 10,
        "x_max_mm": 3.0,
        "y_max_mm": 1.0,
        "n_half": 256,
    },
}

_VALIDATION_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "da": {
        "nturns": (64, 8192),
        "n_angles": (5, 72),
        "amp_max_mm": (1.0, 100.0),
        "n_bisect": (5, 30),
    },
    "lma": {
        "nturns": (64, 8192),
        "n_refpts": (10, 500),
        "dp_max_pct": (0.5, 20.0),
        "n_sectors": (1, 100),
        "n_bisect": (5, 30),
    },
    "chromaticity": {
        "dp_min_pct": (-20.0, 0.0),
        "dp_max_pct": (0.0, 20.0),
        "n_steps": (5, 200),
    },
    "footprint": {
        "n_amp": (3, 30),
        "x_max_mm": (0.1, 50.0),
        "y_max_mm": (0.1, 50.0),
        "n_half": (32, 2048),
    },
}


@dataclass(frozen=True)
class Capabilities:
    """The figures a selection draws.

    Attributes:
        figures: Every figure it draws.
        fast_figures: The ones a refresh recomputes.
        verify: Whether the verification figures apply.
    """

    figures: tuple[str, ...] = ()
    fast_figures: tuple[str, ...] = ()
    verify: bool = False


def capabilities_for(solve: str) -> Capabilities:
    """Return the capabilities of a model with *solve*."""
    if solve == SINGLE_PASS:
        return Capabilities(figures=OPTICS_ONLY, fast_figures=OPTICS_ONLY, verify=False)
    return Capabilities(figures=ALL_FIGURES, fast_figures=FAST_FIGURES, verify=True)


@dataclass(frozen=True, eq=False)
class Selection:
    """The model the dashboard draws, as one resolve found it.

    Attributes:
        model: The model's name, or None when nothing is selected.
        status: ``loading`` while a resolve runs, ``ready`` once the deck is
            loaded, ``failed`` when the engine or the deck stopped it,
            ``none`` when the build offers no such model.
        error: Why the selection failed, or None.
        deck: The deck copy the figures are computed from.
        deck_sha256: The deck's digest.
        prepared: The model's settings as the pyAT engine prepared them.
        capabilities: The figures the model draws.
        families: The deck's magnet families, by name.
        summary: The deck's own numbers: energy, circumference, periodicity
            and element count.
    """

    model: str | None = None
    status: SelectionStatus = "none"
    error: str | None = None
    deck: Path | None = None
    deck_sha256: str | None = None
    prepared: Prepared | None = None
    capabilities: Capabilities = field(default_factory=Capabilities)
    families: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether figures can be computed for this selection."""
        return self.status == "ready" and self.prepared is not None and self.deck is not None


def describe_deck(deck: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deck's magnet families and its own summary numbers.

    Loads the deck through pyAT, so it is called off the event loop.

    Returns:
        ``(families, summary)``: each family's type, parameter, value, count
        and slider range, and the deck's energy, circumference, periodicity
        and element count.
    """
    import at

    ring = at.load_lattice(str(deck))
    sect_count = sum(
        1
        for elem in ring
        if getattr(elem, "FamName", "").startswith("SECT")
        and getattr(elem, "FamName", "")[4:].isdigit()
    )
    periodicity = sect_count if sect_count > 1 else int(getattr(ring, "periodicity", 1))

    families: dict[str, dict[str, Any]] = {}
    for elem in ring:
        fam = getattr(elem, "FamName", None)
        if fam is None:
            continue
        if fam in families:
            families[fam]["count"] += 1
            continue

        # Check H (sextupole) BEFORE K (quadrupole), since sextupoles
        # also have K attribute (returns their quadrupole component = 0)
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

    summary: dict[str, Any] = {
        "energy_gev": float(ring.energy) / 1e9,
        "circumference_m": float(ring.get_s_pos(len(ring))[0]),
        "periodicity": periodicity,
        "num_elements": len(ring),
    }
    return families, summary


def prepared_lists(prepared: Prepared) -> dict[str, Any]:
    """Return *prepared* as plain JSON values, every array a list of floats."""
    twiss_in = (
        None
        if prepared.twiss_in is None
        else {key: [float(v) for v in values] for key, values in prepared.twiss_in.items()}
    )
    return {
        "solve": prepared.solve,
        "twiss_in": twiss_in,
        "rest_mass_gev": float(prepared.rest_mass_gev),
        "length_m": float(prepared.length_m),
    }


def figure_key(
    figure: str,
    *,
    deck_sha256: str,
    prepared: dict[str, Any],
    settings: dict[str, Any] | None,
    overrides: dict[str, float],
    baseline_overrides: dict[str, float] | None,
) -> str:
    """Return the content key of one figure's inputs.

    The key is the SHA-256 of the canonical JSON of the inputs; it names no
    session, so a figure of the same inputs has the same key everywhere.
    """
    inputs = {
        "figure": figure,
        "deck_sha256": deck_sha256,
        "prepared": prepared,
        "settings": settings,
        "overrides": dict(sorted(overrides.items())),
        "baseline_overrides": (
            None if baseline_overrides is None else dict(sorted(baseline_overrides.items()))
        ),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_setting(group: str, key: str, value: Any) -> Any:
    """Coerce type and clamp to valid range."""
    default = DEFAULT_SETTINGS.get(group, {}).get(key)

    # Handle nullable n_sectors
    if group == "lma" and key == "n_sectors":
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        lo, hi = _VALIDATION_RANGES["lma"]["n_sectors"]
        return max(int(lo), min(int(hi), value))

    # Type coercion based on default type
    if isinstance(default, int):
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            return default
    elif isinstance(default, float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

    # Range clamping
    ranges = _VALIDATION_RANGES.get(group, {}).get(key)
    if ranges is not None:
        lo, hi = ranges
        value = max(lo, min(hi, value))
        if isinstance(default, int):
            value = int(value)

    return value


def _merged_settings(saved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = copy.deepcopy(DEFAULT_SETTINGS)
    for group, defaults in merged.items():
        for key in defaults:
            if key in saved.get(group, {}):
                defaults[key] = saved[group][key]
    return merged


def _write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* through a temporary file, so no reader sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class LatticeState:
    """The dashboard's selection, one session's what-if inputs, and the figure store.

    Called on the event loop only. Constructing it writes no file.

    Args:
        root: The lattice directory, ``<agent_data>/lattice/``.
        session: The session whose what-if inputs this state holds.
    """

    def __init__(self, root: Path, session: str = DEFAULT_SESSION) -> None:
        self._root = Path(root)
        self._session = session
        #: The current selection; replaced by each resolve.
        self.selection = Selection(status="loading")
        self._decks: dict[Path, tuple[tuple[int, int], str | None]] = {}

    # ── Paths ─────────────────────────────────────────────

    @property
    def session_dir(self) -> Path:
        return self._root / "sessions" / self._session

    @property
    def jobs_dir(self) -> Path:
        return self.session_dir / "jobs"

    def _scope_dir(self, scope: str) -> Path:
        if scope == SHARED_SCOPE:
            return self._root / "figures" / SHARED_SCOPE
        return self.session_dir / "figures"

    # ── Selection ─────────────────────────────────────────

    def wanted(self) -> str | None:
        """Return the model the operator picked, or None for the build's default."""
        record = _read_json(self._root / "selection.json")
        model = record.get("model") if isinstance(record, dict) else None
        return str(model) if model else None

    def last_model(self) -> str | None:
        """Return the model the what-if inputs belong to, or None."""
        model = self._whatif()["model"]
        return str(model) if model else None

    def set_wanted(self, model: str) -> None:
        """Record *model* as the operator's pick."""
        _write_json(self._root / "selection.json", {"model": model})

    def adopt(self, selection: Selection, *, reset: bool) -> bool:
        """Make *selection* current.

        The what-if inputs belong to one model and deck: when *selection* is
        another, or *reset* is set, its overrides are dropped and the baseline
        is set to the unmodified deck. Settings are kept.

        Returns:
            True when the what-if inputs were reset for a ready selection.
        """
        self.selection = selection
        if not selection.ready:
            return False
        whatif = self._whatif()
        same = (
            whatif.get("model") == selection.model
            and whatif.get("deck_sha256") == selection.deck_sha256
        )
        if same and not reset:
            return False
        whatif.update(
            model=selection.model, deck_sha256=selection.deck_sha256, overrides={}, baseline=None
        )
        self._save_whatif(whatif)
        self.set_baseline()
        return True

    # ── What-if inputs ────────────────────────────────────

    def _whatif(self) -> dict[str, Any]:
        record = _read_json(self.session_dir / "whatif.json")
        whatif: dict[str, Any] = record if isinstance(record, dict) else {}
        whatif.setdefault("model", None)
        whatif.setdefault("deck_sha256", None)
        whatif.setdefault("overrides", {})
        whatif.setdefault("baseline", None)
        whatif.setdefault("settings", {})
        return whatif

    def _save_whatif(self, whatif: dict[str, Any]) -> None:
        _write_json(self.session_dir / "whatif.json", whatif)

    @property
    def overrides(self) -> dict[str, float]:
        return cast(dict[str, float], self._whatif()["overrides"])

    def set_param(self, family: str, value: float) -> None:
        """Set a magnet family parameter override."""
        whatif = self._whatif()
        whatif["overrides"][family] = value
        self._save_whatif(whatif)

    def get_settings(self) -> dict[str, Any]:
        """Return current settings merged with defaults for missing keys."""
        return _merged_settings(self._whatif()["settings"])

    def update_settings(self, new_settings: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge setting updates and validate them."""
        whatif = self._whatif()
        settings = _merged_settings(whatif["settings"])
        for group, values in new_settings.items():
            if group not in DEFAULT_SETTINGS or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key in DEFAULT_SETTINGS[group]:
                    settings[group][key] = _validate_setting(group, key, value)
        whatif["settings"] = settings
        self._save_whatif(whatif)
        return settings

    def reset_settings(self) -> dict[str, Any]:
        """Reset all settings to defaults."""
        whatif = self._whatif()
        whatif["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        self._save_whatif(whatif)
        return cast(dict[str, Any], whatif["settings"])

    def get_baseline(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self._whatif()["baseline"])

    def set_baseline(self) -> dict[str, Any]:
        """Snapshot the current overrides and summary as the comparison baseline."""
        whatif = self._whatif()
        baseline = {
            "summary": self.summary(),
            "overrides": dict(whatif["overrides"]),
            "set_at": _now_iso(),
        }
        whatif["baseline"] = baseline
        self._save_whatif(whatif)
        return baseline

    def clear_baseline(self) -> None:
        whatif = self._whatif()
        whatif["baseline"] = None
        self._save_whatif(whatif)

    # ── Figure keys and the store ─────────────────────────

    def _inputs(self, name: str) -> dict[str, Any] | None:
        """Return figure *name*'s key inputs, or None with no ready selection."""
        selection = self.selection
        if not selection.ready or selection.prepared is None or selection.deck_sha256 is None:
            return None
        whatif = self._whatif()
        group = FIGURE_SETTINGS[name]
        baseline = whatif["baseline"]
        return {
            "deck_sha256": selection.deck_sha256,
            "prepared": prepared_lists(selection.prepared),
            "settings": None if group is None else _merged_settings(whatif["settings"])[group],
            "overrides": dict(whatif["overrides"]),
            "baseline_overrides": None if baseline is None else dict(baseline["overrides"]),
        }

    def figure_key(self, name: str) -> str | None:
        """Return the key of figure *name* for the inputs on screen, or None."""
        inputs = self._inputs(name)
        return None if inputs is None else figure_key(name, **inputs)

    def _scope(self) -> str:
        whatif = self._whatif()
        baseline = whatif["baseline"]
        if whatif["overrides"] or (baseline is not None and baseline["overrides"]):
            return self._session
        return SHARED_SCOPE

    def figure_path(self, name: str, key: str) -> Path:
        """Return where figure *name* of *key* is stored for the inputs on screen."""
        return self._scope_dir(self._scope()) / name / f"{key}.json"

    def read_figure(self, name: str) -> dict[str, Any] | None:
        """Return figure *name*'s stored ``{key, job_id, data}`` for the inputs on screen."""
        key = self.figure_key(name)
        if key is None:
            return None
        payload = _read_json(self.figure_path(name, key))
        return payload if isinstance(payload, dict) and payload.get("key") == key else None

    def has_other_key(self, name: str, key: str) -> bool:
        """Whether figure *name* of the selected deck is stored under a key other than *key*."""
        for scope in (SHARED_SCOPE, self._session):
            directory = self._scope_dir(scope) / name
            for path in directory.glob("*.json") if directory.is_dir() else ():
                if path.stem != key and self._deck_of(path) == self.selection.deck_sha256:
                    return True
        return False

    def _deck_of(self, path: Path) -> str | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._decks.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        payload = _read_json(path)
        deck = payload.get("deck_sha256") if isinstance(payload, dict) else None
        self._decks[path] = (signature, deck)
        return deck

    def prune(self, name: str, kept: int = KEPT_KEYS) -> None:
        """Keep only the *kept* newest keys of figure *name* in each scope."""
        for scope in (SHARED_SCOPE, self._session):
            directory = self._scope_dir(scope) / name
            if not directory.is_dir():
                continue
            files = sorted(
                directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
            )
            for path in files[kept:]:
                path.unlink(missing_ok=True)
                self._decks.pop(path, None)

    def job_spec(self, name: str) -> dict[str, Any] | None:
        """Return the inputs a worker of figure *name* reads, or None with no ready selection."""
        inputs = self._inputs(name)
        selection = self.selection
        if inputs is None or selection.deck is None:
            return None
        return {
            "figure": name,
            "key": figure_key(name, **inputs),
            "deck": str(selection.deck),
            "periodicity": selection.summary.get("periodicity", 1),
            "families": {fam: info["param"] for fam, info in sorted(selection.families.items())},
            **inputs,
        }

    def write_job(self, spec: dict[str, Any], job_id: int) -> Path:
        """Write one launch's immutable job file and return its path."""
        path = self.jobs_dir / f"{job_id}.json"
        _write_json(path, {**spec, "job_id": job_id})
        return path

    # ── Summary ───────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return the deck's summary with the current optics figure's numbers.

        Tunes, chromaticity and beta maxima come from the optics figure of
        the inputs on screen and are absent until it exists.
        """
        summary = dict(self.selection.summary)
        optics = self.read_figure("optics")
        data = optics.get("data") if optics else None
        updates = data.get("summary_updates") if isinstance(data, dict) else None
        if isinstance(updates, dict):
            summary.update(updates)
        return summary
