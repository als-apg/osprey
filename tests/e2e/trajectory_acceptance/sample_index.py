"""The calibrated injection windows, placed on the sample index a panel plots.

The injection-window tests freeze the calibrated physics as turns after the
trigger (1-based). The Raw BPM plots show a turn series by sample index
("Index of turn"), the pre-trigger samples included, so turn ``t`` sits at
sample ``trigger_sample + t - 1``. Every acceptance consumer that states a
threshold on a plotted series -- the walk's rubric and the deployed e2e --
reads it here, from the trigger sample the facility declares and the windows'
own constants, and never restates a number.

``python -m tests.e2e.trajectory_acceptance.sample_index`` prints
:func:`thresholds` as JSON for consumers outside this repository.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.e2e.trajectory_acceptance import shape
from tests.simulation import test_pyat_txt_injection_windows as windows

#: The facility the acceptance stack is rendered from.
EXAMPLE_FACILITY = (
    Path(__file__).resolve().parents[3] / "src" / "osprey" / "templates" / "facilities" / "example"
)

#: The model the walk and the deployed e2e inject into.
MODEL = windows.MODEL


def trigger_sample(facility_dir: Path | str = EXAMPLE_FACILITY, model: str = MODEL) -> int:
    """The sample every turn-by-turn buffer of ``model`` receives the injected shot at.

    Read from the model's trajectory declaration; raises ``KeyError`` for a
    model the facility does not hold and ``ValueError`` when it declares none.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.views.pyaml import trajectory_slots

    document = build_facility(Path(facility_dir), project_name="acceptance")
    declared = trajectory_slots(document, model)["trigger_sample"]
    if declared is None:
        raise ValueError(f"model {model!r} declares no trigger sample")
    return int(declared)


def plotted_index(turn: int, trigger: int) -> int:
    """The plotted sample holding ``turn`` after the trigger (1-based): turn 1 is the trigger sample."""
    if trigger < 0:
        raise ValueError(f"trigger sample {trigger} is negative")
    if turn < 1:
        raise ValueError(f"turn {turn} precedes the first turn after the trigger")
    return trigger + turn - 1


def thresholds(trigger: int | None = None) -> dict[str, Any]:
    """Every calibrated threshold, with each window turn placed at its plotted index.

    ``trigger`` defaults to the example facility's declared trigger sample.
    Shares are fractions of a monitor's first-turn sum; the stored threshold is
    a pre-trigger mean over the post-injection peak.
    """
    if trigger is None:
        trigger = trigger_sample()

    def at(turn: int) -> int:
        return plotted_index(turn, trigger)

    return {
        "model": MODEL,
        "trigger_sample": trigger,
        "n_default": windows.N_DEFAULT,
        "turns_tracked": windows.RF_ON_TURNS,
        "n_shots_max": shape.N_SHOTS_MAX,
        "pre_trigger_indices": [0, trigger - 1],
        "first_turn_index": at(1),
        "last_index": at(windows.RF_ON_TURNS),
        "rf_off": {
            "high": {
                "turn": windows.RF_OFF_HIGH_TURN,
                "index": at(windows.RF_OFF_HIGH_TURN),
                "share_above": windows.RF_OFF_HIGH_SURVIVAL,
            },
            "low": {
                "turn": windows.RF_OFF_LOW_TURN,
                "index": at(windows.RF_OFF_LOW_TURN),
                "share_below": windows.RF_OFF_LOW_SURVIVAL,
            },
            "decoherence": {
                "early_turns": list(windows.EARLY_TURNS),
                "late_turns": list(windows.LATE_TURNS),
                "early_indices": [at(turn) for turn in windows.EARLY_TURNS],
                "late_indices": [at(turn) for turn in windows.LATE_TURNS],
                "ratio_max": windows.DECOHERENCE_MAX,
            },
        },
        "rf_on": {
            "first_index": at(1),
            "last_turn": windows.RF_ON_TURNS,
            "last_index": at(windows.RF_ON_TURNS),
            "share_at_least": windows.RF_ON_SURVIVAL_MIN,
        },
        "stored": {
            "share": windows.STORED_SHARE,
            "margin": windows.STORED_SHARE_MARGIN,
            "threshold": windows.STORED_SHARE_THRESHOLD,
        },
    }


def share_at(series: Sequence[float], trigger: int, turn: int) -> float:
    """A monitor's sum at ``turn`` over its sum at the first turn after the trigger."""
    index = plotted_index(turn, trigger)
    if index >= len(series):
        raise IndexError(f"turn {turn} lies past the {len(series)} samples of the series")
    return float(series[index]) / float(series[plotted_index(1, trigger)])


def stored_ratio(series: Sequence[float], trigger: int) -> float:
    """The mean sum before the trigger over the peak sum after it."""
    pre = [float(value) for value in series[:trigger]]
    post = [float(value) for value in series[trigger:]]
    if not pre or not post:
        raise ValueError("a stored ratio needs pre-trigger and post-injection samples")
    return (sum(pre) / len(pre)) / max(post)


if __name__ == "__main__":
    print(json.dumps(thresholds(), indent=2))
