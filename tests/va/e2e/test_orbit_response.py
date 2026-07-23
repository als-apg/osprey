"""SC3 acceptance: a real SR corrector write moves a real downstream BPM,
through the real connector, against the real container.

Uses device 01 (``SR:MAG:HCM:01`` <-> ``SR:DIAG:BPM:01``) -- no other test in
this suite touches device 01's correctors, so its lattice state is exclusively
owned here for the life of the session container.

The physics bridge (osprey.services.virtual_accelerator.ioc.physics_bridge) applies
a linear kick-angle map and AT's find_orbit4 closed-orbit solve on the real
ALS-U AR ring; sextupoles sit at their baked strengths, so the response is
linear to feed-down accuracy (relative deviation well inside LINEAR_REL_TOL
at these amplitudes) and antisymmetric about zero current, not merely "moves
in the same direction as the raw local kick" (that naive assumption is wrong
for a periodic ring; see findings #8). This asserts the antisymmetric-linear
shape directly: +I and -I give opposite shifts, 2x I doubles the shift, and
the shift is nonzero at the paired BPM -- never same-sign or magnitude alone.

Sweep currents stay inside the committed +/-12 A corrector band: the records
layer enforces channel_limits.json bands as DRVL/DRVH, so an out-of-band
write (e.g. 20 A) is silently clamped at the record and would corrupt the
linearity measurement rather than fail loudly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.va.e2e import conftest as e2e_conftest

CORRECTOR_SP = "SR:MAG:HCM:01:CURRENT:SP"
BPM_X = "SR:DIAG:BPM:01:POSITION:X"
BPM_Y = "SR:DIAG:BPM:01:POSITION:Y"  # HCM steers x only; y should stay ~put (plane decoupling)

SETTLE_BOUND_S = 1.0
NONZERO_FLOOR_M = 1e-5  # 10 microns -- comfortably below the documented mm-scale shift
LINEAR_REL_TOL = 0.02  # 2%: exact in theory (linear lattice), generous for cross-process fp noise


async def _write_current(connector, value: float) -> None:
    result = await connector.write_channel(CORRECTOR_SP, value, verification_level="callback")
    assert result.success, f"write {CORRECTOR_SP}={value} failed: {result.error_message}"


async def _read_bpm(connector, address: str, *, retries: int = 10, delay: float = 0.1) -> float:
    """Poll for up to SETTLE_BOUND_S for the BPM readback (should already be
    current by the time write_channel() returns -- the physics recompute is
    synchronous in the record's write handler -- but a short poll absorbs any
    CA propagation latency to this separate PV)."""
    last = None
    for _ in range(retries):
        last = (await connector.read_channel(address)).value
        await asyncio.sleep(delay)
    return last


async def _measure_one_cycle(connector) -> dict[str, float]:
    """Write 0, +5, -5, +10 A in turn and return each state's BPM01 (x, y).

    All amplitudes are inside the committed +/-12 A band -- see the module
    docstring for why exceeding it silently clamps instead of failing.
    """
    readings: dict[str, tuple[float, float]] = {}
    for label, current in (("zero", 0.0), ("plus5", 5.0), ("minus5", -5.0), ("plus10", 10.0)):
        await _write_current(connector, current)
        x = await _read_bpm(connector, BPM_X, retries=3, delay=0.05)
        y = await _read_bpm(connector, BPM_Y, retries=1, delay=0.0)
        readings[label] = (x, y)
    return readings


class TestOrbitResponse:
    @pytest.mark.asyncio
    async def test_corrector_write_moves_paired_bpm_antisymmetric_linear(self, va_container):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
            connector = await e2e_conftest.connect_va()

            # N=5 repeats for determinism -- the physics recompute is exactly
            # deterministic (no noise applied to pyat-coupled BPM readbacks),
            # so every cycle should reproduce the same deltas.
            cycles = []
            start = time.monotonic()
            for _ in range(5):
                cycles.append(await _measure_one_cycle(connector))
            elapsed = time.monotonic() - start

            # Reset the corrector back to zero so this device's state doesn't
            # leak into anything that inspects it after this test.
            await _write_current(connector, 0.0)

        for i, readings in enumerate(cycles):
            x0, y0 = readings["zero"]
            x_p5, y_p5 = readings["plus5"]
            x_m5, _ = readings["minus5"]
            x_p10, _ = readings["plus10"]

            delta_p5 = x_p5 - x0
            delta_m5 = x_m5 - x0
            delta_p10 = x_p10 - x0

            assert abs(delta_p5) > NONZERO_FLOOR_M, (
                f"cycle {i}: +5A produced a negligible BPM01 x shift ({delta_p5} m) -- "
                "corrector write is not reaching the physics bridge"
            )
            # Antisymmetric: +I and -I give opposite shifts, not naive same-sign.
            assert delta_p5 == pytest.approx(-delta_m5, rel=LINEAR_REL_TOL), (
                f"cycle {i}: +5A shift ({delta_p5}) is not antisymmetric with "
                f"-5A shift ({delta_m5})"
            )
            # Linear: 2x the current doubles the shift.
            assert delta_p10 == pytest.approx(2 * delta_p5, rel=LINEAR_REL_TOL), (
                f"cycle {i}: +10A shift ({delta_p10}) is not double the +5A shift ({delta_p5})"
            )
            # Plane decoupling: HCM steers x only -- y stays put at BPM01.
            assert y_p5 == pytest.approx(y0, abs=1e-6), (
                f"cycle {i}: HCM write moved BPM01 y ({y0} -> {y_p5}); "
                "expected x/y plane decoupling to hold"
            )

        assert elapsed < 5 * SETTLE_BOUND_S * 4 + 30, (
            f"orbit-response cycles took implausibly long; 5 cycles x 4 writes took {elapsed:.1f}s"
        )
