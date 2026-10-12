"""The one source of every number the beam-loss comparison benchmark plants.

The benchmark compares a recent partial beam loss, read out of the archive,
with an older one that exists only as a plot on a logbook entry. The recent
loss is two ``step`` events in ``beam-dip-recent/scenario.json``; the older one
is this module's picture: a lifetime study with top-up off, so the current
decays and is refilled by hand every :data:`REFILL_PERIOD_H`, and one refill
cycle carries a loss of :data:`DIP_DEPTH_MA` that stays until the next refill.

The axis spans :data:`YLIM` so the loss is a small feature on a tall axis; the
title, y-label and series label name the channel and its unit and nothing
else. ``hours_before`` counts back from the entry's own time, as every plot
spec does (see :func:`osprey.simulation.plots.render_plot_spec`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).parent
RECENT_BUNDLE = "beam-dip-recent"
LOGBOOK_BUNDLE = "beam-loss-decay-run"
TARGET_ENTRY_ID = "BLC-DECAY-001"

DCCT_CHANNEL = "SR:DIAG:DCCT:01:CURRENT:RB"

#: Stored current right after a refill, and the recent dip the archive holds.
#: The recent dip is the larger of the two losses, so an agent that assumes the
#: logged event must be the bigger one, or reads the plot as larger than it is,
#: gets the comparison backwards.
BASELINE_MA = 500.0
RECENT_DIP_MA = 60.0

#: The decay-mode sawtooth: current lost between refills, and how often they come.
#: Shallow on purpose: the loss can be sized from the step, from trough to
#: trough, or from the nominal current down to the dip's bottom, and those
#: readings differ by one tooth, so the tooth stays small beside the loss.
DECAY_PER_FILL_MA = 5.0
REFILL_PERIOD_H = 0.75

#: The picture: its span, and the loss only it records.
WINDOW_H = 6.0
DIP_DEPTH_MA = 42.0
DIP_HOURS_BEFORE = 2.1
#: The loss lasts until the next refill: the dip ends at elapsed 4.5 h, a
#: multiple of the refill period, so no refill falls inside it.
DIP_DURATION_H = 0.6
YLIM = (0.0, 600.0)

#: Which loss a correct answer names as the larger: ``"recent"`` (the archive's
#: dip) or ``"logged"`` (the plot's), derived so the two depths decide it.
LARGER_LOSS = "recent" if RECENT_DIP_MA > DIP_DEPTH_MA else "logged"

STEP_MIN = 1.0
NOISE_MA = 0.3
SEED = 20261012

FILENAME = "dcct_decay_run.png"
SERIES_LABEL = "DCCT"


def hours_axis() -> list[float]:
    """Hours before the entry, one sample per :data:`STEP_MIN`, ending at exactly 0."""
    count = int(round(WINDOW_H * 60.0 / STEP_MIN))
    hours = [round(WINDOW_H - i * STEP_MIN / 60.0, 6) for i in range(count)]
    return [*hours, 0.0]


def _sawtooth(hours_before: float) -> float:
    elapsed = WINDOW_H - hours_before
    phase = (elapsed % REFILL_PERIOD_H) / REFILL_PERIOD_H
    return BASELINE_MA - DECAY_PER_FILL_MA * phase


def _dip(hours_before: float) -> float:
    inside = DIP_HOURS_BEFORE - DIP_DURATION_H < hours_before <= DIP_HOURS_BEFORE
    return DIP_DEPTH_MA if inside else 0.0


def build_plot_spec() -> dict[str, Any]:
    """The plot spec mapping, in the shape ``parse_plot_spec`` validates."""
    hours = hours_axis()
    noise = np.random.default_rng(SEED).normal(0.0, NOISE_MA, size=len(hours))
    values = [
        round(_sawtooth(h) - _dip(h) + float(n), 3) for h, n in zip(hours, noise, strict=True)
    ]
    return {
        "filename": FILENAME,
        "title": DCCT_CHANNEL,
        "ylabel": "mA",
        "hours_before": hours,
        "series": [{"label": SERIES_LABEL, "values": values}],
        "ylim": list(YLIM),
    }


def write_plot_spec(bundle_dir: Path) -> Path:
    """Write the spec where the bundle's ``logbook.json`` points; return the path."""
    path = bundle_dir / "plots" / "dcct_decay_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_plot_spec(), indent=1), encoding="utf-8")
    return path


def install_bundles(scenarios_dir: Path) -> None:
    """Copy both bundles into a deployment's ``data/simulation/scenarios/`` and draw the spec."""
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    for name in (RECENT_BUNDLE, LOGBOOK_BUNDLE):
        shutil.copytree(FIXTURE_DIR / name, scenarios_dir / name, dirs_exist_ok=True)
    write_plot_spec(scenarios_dir / LOGBOOK_BUNDLE)
