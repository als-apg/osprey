"""``osprey sim apply`` reaches the live container, which serves the new scenarios.

The container serves the simulator view through one composite, which re-reads
the ``active_scenarios`` file ``osprey sim apply`` writes whenever that file
changes -- on the runner's next pass after it sees the change -- and
rebuilds at the newly active set: each scenario's ``overrides`` become the
start state, and every session write is dropped. A texture setpoint written
over Channel Access therefore returns to its seed on a switch, and its echoed
readback with it.

None of the shipped scenarios carries a live ``overrides`` entry for a
texture reading, so this suite's ``conftest.py`` adds the synthetic
``va-e2e-burst`` scenario, which overrides one vacuum gauge to a value far from
its seed. The test applies it, sees the override over Channel Access, applies
``nominal`` again, and sees the gauge back at its seed, with a session write to
an ion-pump voltage setpoint dropped by each switch.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.va.e2e import conftest as e2e_conftest

#: A writable texture setpoint and the readback its writes echo into. Its seed
#: is :data:`VAC_SEED_V`; the readback carries the seed's noise.
VAC_WRITABLE_SP = "SR:VAC:ION-PUMP:02:VOLTAGE:SP"
VAC_WRITABLE_RB = "SR:VAC:ION-PUMP:02:VOLTAGE:RB"
VAC_SEED_V = 5000.0
#: The readback's seed noise, one standard deviation.
VAC_RB_NOISE_V = 50.0
#: Forty standard deviations of readback noise below the seed, so a readback
#: that kept the session write cannot pass for one back at its seed.
SESSION_WRITE_VALUE = 3000.0
#: A readback within this many standard deviations of the seed is at its seed.
NOISE_BOUND_SIGMA = 6.0

#: How long an apply may take to reach the container. The composite sees the
#: new state on the runner pass after the file changes, a tick at most; a
#: desktop container runtime's file sharing can report the changed file to the
#: container seconds after the host wrote it.
RELOAD_WAIT_BOUND_S = 10.0
#: The burst gauge's seed and a threshold far above its seed noise and far
#: below the burst override.
BASELINE_PRESSURE = 5e-8
BURST_THRESHOLD = 1e-6

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 2


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


async def _write_session_value(connector) -> None:
    """Write :data:`SESSION_WRITE_VALUE` and see it echoed into the readback."""
    # No `confirm` kwarg: the fleet default confirms by re-reading the setpoint.
    result = await connector.write_channel(VAC_WRITABLE_SP, SESSION_WRITE_VALUE)
    assert result.outcome == "confirmed", (
        f"setup write was {result.outcome}: {result.error_message or result.notes}"
    )
    echoed = (await connector.read_channel(VAC_WRITABLE_RB)).value
    assert abs(echoed - SESSION_WRITE_VALUE) <= NOISE_BOUND_SIGMA * VAC_RB_NOISE_V


async def _assert_session_write_dropped(connector, switch: str) -> None:
    """The switch returned the written setpoint, and its readback, to the seed."""
    setpoint = await _wait_until(
        connector,
        VAC_WRITABLE_SP,
        lambda v: v == VAC_SEED_V,
        bound_s=RELOAD_WAIT_BOUND_S,
    )
    assert setpoint == VAC_SEED_V, (
        f"{VAC_WRITABLE_SP} read {setpoint} after applying {switch!r}; a scenario switch "
        f"drops session writes, so it reads its seed {VAC_SEED_V}"
    )
    readback = (await connector.read_channel(VAC_WRITABLE_RB)).value
    assert abs(readback - VAC_SEED_V) <= NOISE_BOUND_SIGMA * VAC_RB_NOISE_V, (
        f"{VAC_WRITABLE_RB} read {readback} after applying {switch!r}, not its seed "
        f"{VAC_SEED_V} within its noise"
    )


class TestScenarioReload:
    @pytest.mark.asyncio
    async def test_scenario_switch_reloads_overrides_and_drops_session_writes(self, va_container):
        project = va_container

        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
            connector = await e2e_conftest.connect_va()

            # Baseline: nominal is already active (conftest seeds active_scenarios
            # with it), but re-assert explicitly so this test doesn't depend on
            # fixture ordering.
            applied = project.sim_apply("nominal")
            assert applied.returncode == 0, applied.stdout + applied.stderr
            baseline = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v < BURST_THRESHOLD,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert baseline < BURST_THRESHOLD, (
                f"{e2e_conftest.BURST_CHANNEL} baseline read {baseline}, expected near "
                f"{BASELINE_PRESSURE}"
            )
            await _assert_session_write_dropped(connector, "nominal")

            # A session write, then the synthetic burst scenario: its override
            # is served within two ticks, and the write is gone.
            await _write_session_value(connector)
            applied = project.sim_apply(e2e_conftest.BURST_SCENARIO_NAME)
            assert applied.returncode == 0, applied.stdout + applied.stderr
            burst_value = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v > BURST_THRESHOLD,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert burst_value > BURST_THRESHOLD, (
                f"{e2e_conftest.BURST_CHANNEL} never reflected the "
                f"'{e2e_conftest.BURST_SCENARIO_NAME}' override "
                f"(last read {burst_value}) within {RELOAD_WAIT_BOUND_S}s"
            )
            await _assert_session_write_dropped(connector, e2e_conftest.BURST_SCENARIO_NAME)

            # Another session write, then nominal again: the override is gone
            # and so is the write.
            await _write_session_value(connector)
            applied = project.sim_apply("nominal")
            assert applied.returncode == 0, applied.stdout + applied.stderr
            reset_value = await _wait_until(
                connector,
                e2e_conftest.BURST_CHANNEL,
                lambda v: v < BURST_THRESHOLD,
                bound_s=RELOAD_WAIT_BOUND_S,
            )
            assert reset_value < BURST_THRESHOLD, (
                f"{e2e_conftest.BURST_CHANNEL} never reset to baseline after re-applying "
                f"'nominal' (last read {reset_value}) within {RELOAD_WAIT_BOUND_S}s"
            )
            await _assert_session_write_dropped(connector, "nominal")


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_scenario_reload.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
