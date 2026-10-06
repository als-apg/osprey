"""The shipped control_assistant machine model under the nominal scenario.

The nominal machine shows no SR07 vacuum spike, even at the time of day a
scenario would place one.
"""

from datetime import UTC, datetime, timedelta

import numpy as np

GAUGE = "SR:VAC:GAUGE:SR{:02d}:PRESSURE:RB"


def _window(center: datetime, minutes: int = 10, step_s: int = 1) -> list[datetime]:
    """Return per-second timestamps for a window centered on ``center``."""
    start = center - timedelta(minutes=minutes / 2)
    return [start + timedelta(seconds=i) for i in range(minutes * 60 // step_s)]


def _yesterday_event() -> datetime:
    """Yesterday 14:32:08 in the facility zone (UTC in tests) — the daily ``at_time``
    anchor fires on any past date. Building the window tz-aware in the facility zone
    keeps the contract independent of the deploy host's ``$TZ``."""
    day = datetime.now(UTC) - timedelta(days=1)
    return day.replace(hour=14, minute=32, second=8, microsecond=0)


def test_nominal_scenario_has_no_event(engine_factory):
    """The nominal scenario shows no SR07 spike even at 14:32."""
    engine = engine_factory("nominal")
    ts = _window(_yesterday_event())
    sr07 = np.array(engine.synthesize_series(GAUGE.format(7), ts))
    assert sr07.max() < 1.0e-7, f"nominal SR07 peak {sr07.max():.3e}, contract < 1e-7"
