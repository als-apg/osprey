"""Unit tests for the substrate PV-list parser (``devices/_specs_from_env.py``).

Covers the parse + de-duplication behavior directly (the app-wiring tests in
``test_epics_substrate_scanner_wiring.py`` exercise the same code through
``app.py``, but not the collision path). Guarded by ``importorskip`` because
``_specs_from_env`` imports ``devices/epics.py``, which imports ophyd-async: a
core dependency, so this normally runs, but the guard skips cleanly in a slimmed
install where ophyd-async is absent, mirroring the wiring test.
"""

from __future__ import annotations

from collections import UserDict

import pytest

pytest.importorskip("ophyd_async")

from osprey.services.bluesky_bridge.devices._specs_from_env import specs_from_env  # noqa: E402


def test_duplicate_motor_name_drops_the_later_entry() -> None:
    """Two motor entries with the same device name keep only the first."""
    motors, _ = specs_from_env({"BLUESKY_EPICS_MOTORS": "m=A:SP|A:RB,m=B:SP|B:RB"})
    assert [s.name for s in motors] == ["m"]
    assert motors[0].setpoint_pv == "A:SP"  # first wins, not the shadowing second


def test_detector_name_colliding_with_a_motor_is_dropped() -> None:
    """A detector reusing a motor's device name is dropped (motors resolve first);
    device names become event-data column keys, so a collision must not stand."""
    motors, detectors = specs_from_env(
        {
            "BLUESKY_EPICS_MOTORS": "dev=A:SP|A:RB",
            "BLUESKY_EPICS_DETECTORS": "dev=B:RB,other=C:RB",
        }
    )
    assert [s.name for s in motors] == ["dev"]
    assert [s.name for s in detectors] == ["other"]  # 'dev' detector dropped


def test_distinct_names_are_all_kept() -> None:
    """No collision → every parsed device survives (dedup is not over-eager)."""
    motors, detectors = specs_from_env(
        {
            "BLUESKY_EPICS_MOTORS": "m1=A:SP,m2=B:SP",
            "BLUESKY_EPICS_DETECTORS": "d1=C:RB,d2=D:RB",
        }
    )
    assert [s.name for s in motors] == ["m1", "m2"]
    assert [s.name for s in detectors] == ["d1", "d2"]


def test_accepts_any_mapping_not_only_dict() -> None:
    """``specs_from_env`` is typed ``Mapping[str, str]`` so it accepts
    ``os.environ`` (an ``os._Environ``, not a ``dict``). A ``UserDict`` stands
    in for any non-``dict`` mapping here."""
    env: UserDict = UserDict({"BLUESKY_EPICS_MOTORS": "m=A:SP"})
    motors, detectors = specs_from_env(env)
    assert [s.name for s in motors] == ["m"]
    assert detectors == []
