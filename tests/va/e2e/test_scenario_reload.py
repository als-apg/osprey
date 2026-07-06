"""SC8 acceptance: ``osprey sim apply`` reaches the live container and the
IOC's poll-driven reload picks up the new scenario within its documented
bound.

A note on scope, established by reading the actual container wiring (not
assumed): the literal task language describes writing a VAC/RF channel via
``channel_write`` and then observing that write "reset" by a scenario switch.
That specific mechanism doesn't exist for VAC telemetry as built --
``docker/virtual-accelerator/manifest/classify.py`` puts read-only VAC/RF
telemetry (gauges, temperatures, ion-pump current) in the ``static-noisy``
partition, which ``ioc/records.py`` builds as plain CA *In*-type records --
there is no Out record to ``channel_write`` in the first place, so nothing
written there can be "reset" by a later apply. The only VAC/RF fields that
are genuinely writable over CA (``ion-pump VOLTAGE``, RF ``VOLTAGE``/
``FREQUENCY``/``TUNER``) are classified ``sp-echo`` -- wired to their
readback by a plain value copy in ``ioc/records.py``, with no tie to
``SimulationEngine`` at all, so they are *not* reset by a scenario switch
either (confirmed below, as a real regression check, not an assumption).

What the built system actually does, and what this test verifies instead:
``ioc/engine_source.py`` polls ``active_scenarios`` (1s interval) and drives
every *static-noisy* channel from ``SimulationEngine.read()``, which composes
whichever scenario's ``overrides`` are currently active. None of the shipped
example scenarios (nominal/rf-thermal/vacuum-burst) carry an ``overrides``
block -- only archiver history events -- so this test's ``conftest.py`` adds
a synthetic ``va-e2e-burst`` scenario with a real override to exercise the
reload path end-to-end: apply it, observe the override live over CA within
the poll bound, apply ``nominal`` again, and observe the value reset back to
baseline.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.va.e2e import conftest as e2e_conftest

# A genuinely writable VAC channel (sp-echo -- ion-pump voltage setpoint),
# used to demonstrate that a plain channel_write's value is NOT touched by a
# scenario switch (sp-echo has no tie to SimulationEngine at all).
VAC_WRITABLE_SP = "SR:VAC:ION-PUMP:01:VOLTAGE:SP"
VAC_WRITABLE_RB = "SR:VAC:ION-PUMP:01:VOLTAGE:RB"
SESSION_WRITE_VALUE = 437.5

POLL_INTERVAL_S = 1.0
RELOAD_WAIT_BOUND_S = 3.0  # generous margin over the documented ~1-2s poll bound
BASELINE_TORR = 5e-8
BURST_THRESHOLD_TORR = 1e-6  # well above baseline+noise, well below the burst override


async def _wait_until(connector, address: str, predicate, *, bound_s: float) -> float:
    """Poll ``address`` until ``predicate(value)`` is true or ``bound_s`` elapses.

    Returns the last-read value (whether or not the predicate was ever met,
    so a failing assertion can report what was actually observed).
    """
    deadline = time.monotonic() + bound_s
    value = (await connector.read_channel(address)).value
    while time.monotonic() < deadline:
        value = (await connector.read_channel(address)).value
        if predicate(value):
            return value
        await asyncio.sleep(0.2)
    return value


class TestScenarioReload:
    @pytest.mark.asyncio
    async def test_scenario_switch_reloads_static_noisy_channel(self, va_container):
        project = va_container

        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
            connector = await e2e_conftest.connect_va()

            # Establish a genuine channel_write to a writable VAC channel
            # BEFORE the scenario dance below, so its persistence across the
            # switch is a real regression check, not a hope.
            result = await connector.write_channel(
                VAC_WRITABLE_SP, SESSION_WRITE_VALUE, verification_level="callback"
            )
            assert result.success, f"setup write failed: {result.error_message}"
            written_rb = (await connector.read_channel(VAC_WRITABLE_RB)).value
            assert written_rb == pytest.approx(SESSION_WRITE_VALUE)

            # Baseline: nominal is already active (conftest seeds active_scenarios
            # with it), but re-assert explicitly so this test doesn't depend on
            # fixture ordering.
            applied = project.sim_apply("nominal")
            assert applied.returncode == 0, applied.stdout + applied.stderr
            baseline = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v < BURST_THRESHOLD_TORR,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert baseline < BURST_THRESHOLD_TORR, (
                f"{e2e_conftest.BURST_CHANNEL} baseline read {baseline}, expected near "
                f"{BASELINE_TORR} Torr"
            )

            # Apply the synthetic burst scenario: its override should become
            # visible over CA within the documented poll bound.
            applied = project.sim_apply(e2e_conftest.BURST_SCENARIO_NAME)
            assert applied.returncode == 0, applied.stdout + applied.stderr
            burst_value = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v > BURST_THRESHOLD_TORR,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert burst_value > BURST_THRESHOLD_TORR, (
                f"{e2e_conftest.BURST_CHANNEL} never reflected the "
                f"'{e2e_conftest.BURST_SCENARIO_NAME}' override "
                f"(last read {burst_value}) within {RELOAD_WAIT_BOUND_S}s"
            )

            # The channel_write from setup must be completely unaffected by
            # the scenario switch: sp-echo has no tie to SimulationEngine.
            unaffected_rb = (await connector.read_channel(VAC_WRITABLE_RB)).value
            assert unaffected_rb == pytest.approx(SESSION_WRITE_VALUE), (
                f"{VAC_WRITABLE_RB} changed from {SESSION_WRITE_VALUE} to {unaffected_rb} "
                "after a scenario switch -- sp-echo channels are not scenario-tied"
            )

            # Apply nominal again: the override must reset back to baseline.
            applied = project.sim_apply("nominal")
            assert applied.returncode == 0, applied.stdout + applied.stderr
            reset_value = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v < BURST_THRESHOLD_TORR,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert reset_value < BURST_THRESHOLD_TORR, (
                f"{e2e_conftest.BURST_CHANNEL} never reset to baseline after re-applying "
                f"'nominal' (last read {reset_value}) within {RELOAD_WAIT_BOUND_S}s"
            )

            # And the sp-echo write is STILL untouched by this second switch.
            still_unaffected_rb = (await connector.read_channel(VAC_WRITABLE_RB)).value
            assert still_unaffected_rb == pytest.approx(SESSION_WRITE_VALUE)
