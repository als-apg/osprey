"""A scenario fault ``osprey sim apply`` turns on reaches a live reading within two ticks.

The container's composite re-reads the ``active_scenarios`` file on every
runner pass and rebuilds at the newly active set when the file has changed;
the next pass serves the rebuilt model. So once the container can see the
changed file, a fault the new set turns on is served within two runner ticks.

The clock starts when the file the host wrote reads the same inside the
container: how long a desktop container runtime's file sharing takes to show
a host write to the container is the runtime's, not the virtual accelerator's.

The directory conftest's kick scenario puts a nonzero vertical closed orbit
at :data:`~tests.va.e2e.conftest.KICK_MONITOR`, and the session container
serves its monitors without their declared motion, so the reading there is
the solved orbit exactly. Adding the shipped ``bpm-polarity`` scenario, which
inverts that monitor's polarity, turns the reading ``r`` into ``-r``.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any

import pytest

from osprey_connectors.simulation import DEFAULT_TICK_S
from tests.va.e2e import conftest as e2e_conftest

#: The session container is booted without ``VA_POLL_INTERVAL_S``, so its
#: runner ticks at the default.
TICK_S = DEFAULT_TICK_S
#: The shipped scenario that inverts the kick monitor's polarity.
POLARITY_SCENARIO = "bpm-polarity"
#: How long an apply may take to become visible inside the container: a
#: desktop runtime's file sharing, not a property of the virtual accelerator.
VISIBLE_BOUND_S = 30.0
#: How long the container may take to serve a scenario set this module only
#: prepares, rather than times.
SETTLE_BOUND_S = 30.0
#: The interval between polls while a reading is timed.
POLL_S = 0.05

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 2


def _active_inside_container() -> str:
    """The ``active_scenarios`` file as the session container reads it."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            e2e_conftest.CONTAINER_NAME,
            "cat",
            "/state/simulation/active_scenarios",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _apply(project: e2e_conftest.VaProject, *scenarios: str) -> None:
    """``osprey sim apply`` the scenarios, then wait until the container sees the file."""
    applied = project.sim_apply(*scenarios)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    written = (project.state_dir / "active_scenarios").read_text(encoding="utf-8")
    deadline = time.monotonic() + VISIBLE_BOUND_S
    while _active_inside_container() != written:
        assert time.monotonic() < deadline, (
            f"the container never saw the active_scenarios file the host wrote ({written!r})"
        )
        time.sleep(POLL_S)


async def _read(connector: Any) -> float:
    return float((await connector.read_channel(e2e_conftest.KICK_MONITOR)).value)


async def _settle(connector: Any, predicate: Any) -> float:
    """Poll the kick monitor until ``predicate`` holds; return the reading."""
    deadline = time.monotonic() + SETTLE_BOUND_S
    value = await _read(connector)
    while not predicate(value):
        assert time.monotonic() < deadline, (
            f"{e2e_conftest.KICK_MONITOR} never settled (last read {value})"
        )
        await asyncio.sleep(POLL_S)
        value = await _read(connector)
    return value


@pytest.mark.asyncio
async def test_sim_apply_flips_the_monitor_sign_within_two_ticks(
    va_container: e2e_conftest.VaProject,
) -> None:
    project = va_container
    with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
        connector = await e2e_conftest.connect_va()
        try:
            _apply(project, e2e_conftest.KICK_SCENARIO_NAME)
            before = await _settle(connector, lambda value: value != 0.0)

            _apply(project, e2e_conftest.KICK_SCENARIO_NAME, POLARITY_SCENARIO)
            visible = time.monotonic()
            deadline = visible + 2 * TICK_S
            issued = visible
            after = await _read(connector)
            while after != -before:
                await asyncio.sleep(POLL_S)
                if time.monotonic() > deadline:
                    break
                issued = time.monotonic()
                after = await _read(connector)
        finally:
            _apply(project, "nominal")
            await _settle(connector, lambda value: value == 0.0)

    assert before != 0.0
    assert after == -before, (
        f"{e2e_conftest.KICK_MONITOR} read {after}, not {-before}, by "
        f"{issued - visible:.2f} s after the container saw {POLARITY_SCENARIO!r}; "
        f"the bound is two ticks, {2 * TICK_S} s, plus one read"
    )


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_sim_apply_va.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
