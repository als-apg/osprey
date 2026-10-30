"""A real lane-model corrector write moves a real downstream BPM, through the
real connector, against the real container.

Uses the lane's bounded corrector, the lane model's first monitor in the plane
that corrector steers and its first monitor in the other plane
(``conftest.LANE``) -- no other test in this suite writes that corrector,
so its lattice state is exclusively owned here for the life of the session
container.

The container serves the lane model through the pyat engine:
a corrector write sets the element's kick and the served BPM reading is the
closed-orbit solve at that monitor. Sextupoles sit at their deck strengths, so
the response is linear to feed-down accuracy (relative deviation well inside
LINEAR_REL_TOL at these amplitudes) and antisymmetric about zero current, not
merely "moves in the same direction as the raw local kick", which is wrong for
a periodic lattice. This asserts the antisymmetric-linear shape directly: +I
and -I give opposite shifts, 2x I doubles the shift, and the shift is nonzero
at the steered-plane monitor -- never same-sign or magnitude alone.

The shared container starts in the suite scenario
(``conftest.SUITE_SCENARIOS``), which serves its monitors without their
declared noise and drift, so a reading is the solved orbit and nothing else,
and every cycle reproduces the first.

Sweep currents are fractions of the band ``limits.yaml`` gives this corrector
(``conftest.LANE.band``), so they stay inside it on any facility: the view carries it as the setpoint's
``value_range`` and the runner clamps an out-of-band write into it, so a write
above the band would read back as its upper end and corrupt the linearity
measurement rather than fail loudly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.va._collection import assert_module_collected
from tests.va.e2e import conftest as e2e_conftest

LANE = e2e_conftest.LANE
CORRECTOR_SP = e2e_conftest.BOUNDED_SETPOINT
STEERED_PLANE = LANE.plane
OTHER_PLANE = "y" if STEERED_PLANE == "x" else "x"
STEERED_BPM = LANE.monitor_x if STEERED_PLANE == "x" else LANE.monitor_y
OTHER_BPM = LANE.monitor_y if STEERED_PLANE == "x" else LANE.monitor_x

#: Half the band's narrower side: every sweep point below lies inside the band.
_HALF = min(abs(LANE.band[0]), abs(LANE.band[1]))
#: The sweep, in order: zero, +I, -I and +2I with I a quarter of ``_HALF``.
SWEEP = (
    ("zero", 0.0),
    ("plus", _HALF / 4),
    ("minus", -_HALF / 4),
    ("double", _HALF / 2),
)

SETTLE_BOUND_S = 1.0
NONZERO_FLOOR_M = 1e-5  # 10 microns -- comfortably below the documented mm-scale shift
LINEAR_REL_TOL = 0.02  # 2%: sextupole feed-down at these amplitudes stays far inside it
#: Two solves at one corrector setting agree to this, in metres: the readings
#: carry no motion, so only the solver's own rounding separates them.
REPRODUCE_ABS_M = 1e-12

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 2


async def _write_current(connector, value: float) -> None:
    # No `confirm` kwarg: this channel has no confirm entry, so it resolves to the
    # fleet default (true), and on this call that means the EPICS put waits for the
    # control system's acknowledgement (`caput(wait=confirm)`) and the write is then
    # confirmed by re-reading the setpoint -- which is what this sweep wants before
    # it reads the paired BPM. The waiting is not unconditional: a channel
    # configured `confirm: false` would put without waiting.
    result = await connector.write_channel(CORRECTOR_SP, value)
    assert result.outcome == "confirmed", (
        f"write {CORRECTOR_SP}={value} was {result.outcome}: {result.error_message or result.notes}"
    )


async def _read_bpm(connector, address: str, *, retries: int = 10, delay: float = 0.1) -> float:
    """Poll for up to SETTLE_BOUND_S for the BPM readback (should already be
    current by the time write_channel() returns -- the write's own runner pass
    solves the orbit and publishes the readings before put-completion -- but a
    short poll absorbs any CA propagation latency to this separate PV)."""
    last = None
    for _ in range(retries):
        last = (await connector.read_channel(address)).value
        await asyncio.sleep(delay)
    return last


async def _measure_one_cycle(connector) -> dict[str, float]:
    """Write each ``SWEEP`` current in turn and return each state's monitor
    readings, (steered plane, other plane).

    All amplitudes are inside the corrector's committed band -- see the module
    docstring for why exceeding it silently clamps instead of failing.
    """
    readings: dict[str, tuple[float, float]] = {}
    for label, current in SWEEP:
        await _write_current(connector, current)
        steered = await _read_bpm(connector, STEERED_BPM, retries=3, delay=0.05)
        other = await _read_bpm(connector, OTHER_BPM, retries=1, delay=0.0)
        readings[label] = (steered, other)
    return readings


class TestOrbitResponse:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("va_container", "va_switch_on")
    async def test_corrector_write_moves_paired_bpm_antisymmetric_linear(self):
        with e2e_conftest.patched_config(**{"control_system.writes": "allowed"}):
            connector = await e2e_conftest.connect_va()

            # N=5 repeats: the solve is deterministic and the monitors carry
            # no motion, so every cycle reproduces the first.
            cycles = []
            start = time.monotonic()
            for _ in range(5):
                cycles.append(await _measure_one_cycle(connector))
            elapsed = time.monotonic() - start

            # Reset the corrector back to zero so this device's state doesn't
            # leak into anything that inspects it after this test.
            await _write_current(connector, 0.0)

        for i, readings in enumerate(cycles):
            for label, (steered, other) in readings.items():
                first_steered, first_other = cycles[0][label]
                assert steered == pytest.approx(first_steered, abs=REPRODUCE_ABS_M), (
                    f"cycle {i}: {STEERED_PLANE} at {label} read {steered}, "
                    f"cycle 0 read {first_steered}"
                )
                assert other == pytest.approx(first_other, abs=REPRODUCE_ABS_M), (
                    f"cycle {i}: {OTHER_PLANE} at {label} read {other}, cycle 0 read {first_other}"
                )
            s0, o0 = readings["zero"]
            s_plus, o_plus = readings["plus"]
            s_minus, _ = readings["minus"]
            s_double, _ = readings["double"]

            delta_plus = s_plus - s0
            delta_minus = s_minus - s0
            delta_double = s_double - s0

            assert abs(delta_plus) > NONZERO_FLOOR_M, (
                f"cycle {i}: +I produced a negligible {STEERED_PLANE} shift ({delta_plus} m) -- "
                "corrector write is not reaching the physics bridge"
            )
            # Antisymmetric: +I and -I give opposite shifts, not naive same-sign.
            assert delta_plus == pytest.approx(-delta_minus, rel=LINEAR_REL_TOL), (
                f"cycle {i}: +I shift ({delta_plus}) is not antisymmetric with "
                f"-I shift ({delta_minus})"
            )
            # Linear: 2x the current doubles the shift.
            assert delta_double == pytest.approx(2 * delta_plus, rel=LINEAR_REL_TOL), (
                f"cycle {i}: +2I shift ({delta_double}) is not double the +I shift ({delta_plus})"
            )
            # Plane decoupling: the steered plane moves, the other stays put.
            assert o_plus == pytest.approx(o0, abs=1e-6), (
                f"cycle {i}: a corrector steering {STEERED_PLANE} moved {OTHER_PLANE} "
                f"({o0} -> {o_plus}); expected the planes to stay decoupled"
            )

        assert elapsed < 5 * SETTLE_BOUND_S * 4 + 30, (
            f"orbit-response cycles took implausibly long; 5 cycles x 4 writes took {elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
