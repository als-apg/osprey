"""A real SR corrector write moves a real downstream BPM, through the real
connector, against the real container.

Uses device 01 (``SR:MAG:HCM:01`` <-> ``SR:DIAG:BPM:01``) -- no other test in
this suite touches device 01's correctors, so its lattice state is exclusively
owned here for the life of the session container.

The container serves the demo facility's ``SR`` model through the pyat engine:
a corrector write sets the element's kick and the served BPM reading is the
closed-orbit solve at that monitor. Sextupoles sit at their deck strengths, so
the response is linear to feed-down accuracy (relative deviation well inside
LINEAR_REL_TOL at these amplitudes) and antisymmetric about zero current, not
merely "moves in the same direction as the raw local kick", which is wrong for
a periodic lattice. This asserts the antisymmetric-linear shape directly: +I
and -I give opposite shifts, 2x I doubles the shift, and the shift is nonzero
at the paired BPM -- never same-sign or magnitude alone.

The shared container serves its monitors without their declared noise and
drift (``conftest.stage_demo_data_dir``), so a reading is the solved orbit and
nothing else, and every cycle reproduces the first.

Sweep currents are a small fraction of the +/-12 A band ``limits.yaml`` gives
this corrector. The band's ends are milliradian kicks, where feed-down bends
the response by several percent, so the sweep stays at microradian kicks, in
the linear regime this test asserts. The view carries the band as the
setpoint's ``value_range`` and the runner clamps an out-of-band write into it,
so a 20 A write would read back as 12 A and corrupt the linearity measurement
rather than fail loudly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.va.e2e import conftest as e2e_conftest

CORRECTOR_SP = "SR:MAG:HCM:01:CURRENT:SP"
BPM_X = "SR:DIAG:BPM:01:POSITION:X"
BPM_Y = "SR:DIAG:BPM:01:POSITION:Y"  # HCM steers x only; y should stay ~put (plane decoupling)

#: The sweep's base current, in amperes: a kick of a few microradians, where
#: the antisymmetry and doubling residuals are a tenth of LINEAR_REL_TOL.
SWEEP_CURRENT_A = 0.05

SETTLE_BOUND_S = 1.0
NONZERO_FLOOR_M = 1e-5  # 10 microns -- a quarter of the shift the smaller sweep current gives
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
    """Write 0, +I, -I, +2I in turn and return each state's BPM01 (x, y).

    All amplitudes are far inside the committed +/-12 A band -- see the module
    docstring for why exceeding it silently clamps instead of failing, and
    for why the sweep stays this small.
    """
    readings: dict[str, tuple[float, float]] = {}
    for label, current in (
        ("zero", 0.0),
        ("plus", SWEEP_CURRENT_A),
        ("minus", -SWEEP_CURRENT_A),
        ("double", 2 * SWEEP_CURRENT_A),
    ):
        await _write_current(connector, current)
        x = await _read_bpm(connector, BPM_X, retries=3, delay=0.05)
        y = await _read_bpm(connector, BPM_Y, retries=1, delay=0.0)
        readings[label] = (x, y)
    return readings


class TestOrbitResponse:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("va_container")
    async def test_corrector_write_moves_paired_bpm_antisymmetric_linear(self):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
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
            for label, (x, y) in readings.items():
                first_x, first_y = cycles[0][label]
                assert x == pytest.approx(first_x, abs=REPRODUCE_ABS_M), (
                    f"cycle {i}: BPM01 x at {label} read {x}, cycle 0 read {first_x}"
                )
                assert y == pytest.approx(first_y, abs=REPRODUCE_ABS_M), (
                    f"cycle {i}: BPM01 y at {label} read {y}, cycle 0 read {first_y}"
                )
            x0, y0 = readings["zero"]
            x_plus, y_plus = readings["plus"]
            x_minus, _ = readings["minus"]
            x_double, _ = readings["double"]

            delta_plus = x_plus - x0
            delta_minus = x_minus - x0
            delta_double = x_double - x0

            assert abs(delta_plus) > NONZERO_FLOOR_M, (
                f"cycle {i}: +I produced a negligible BPM01 x shift ({delta_plus} m) -- "
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
            # Plane decoupling: HCM steers x only -- y stays put at BPM01.
            assert y_plus == pytest.approx(y0, abs=1e-6), (
                f"cycle {i}: HCM write moved BPM01 y ({y0} -> {y_plus}); "
                "expected x/y plane decoupling to hold"
            )

        assert elapsed < 5 * SETTLE_BOUND_S * 4 + 30, (
            f"orbit-response cycles took implausibly long; 5 cycles x 4 writes took {elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_orbit_response.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
