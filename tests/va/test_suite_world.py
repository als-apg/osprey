"""The names the VA live-container suite writes down, vouched against the facility.

The suite selects its lane model and that model's channels from the preset's
view (``conftest.LANE``). Three kinds of name stay written, each once: the
addresses the suite's own scenarios and operating points author (its scenario
world, in the directory conftest), the bench IOC's records
(``tests/fixtures/bench_ioc.py``), and the second model a lane tests on
purpose. Each is checked here against the example view, the built facility
document and ``docker/bench-ioc/bench.db``, without Docker.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from osprey_connectors.simulation.envelope import _noise_sigma
from tests.e2e._facility_select import monitor_addresses, second_models
from tests.fixtures import bench_ioc as bench
from tests.va.e2e import conftest as e2e_conftest
from tests.va.e2e.test_orbit_response import SWEEP

BENCH_DB = e2e_conftest.REPO_ROOT / "docker" / "bench-ioc" / "bench.db"

#: Every record name ``bench_ioc`` gives the bench IOC.
BENCH_RECORDS = (
    bench.PROBE_CHANNEL,
    bench.ALARM_CHANNEL,
    bench.SHARED_BPM_X,
    bench.WRITABLE_SP,
    bench.PROTECTED_SP,
    bench.MODE_STATE,
    bench.SCALED_AMPLITUDE,
    bench.BENCH_ONLY_AMPLITUDE,
)

VIEW = e2e_conftest.PRESET_VIEW
LANE = e2e_conftest.LANE


@pytest.fixture(scope="module")
def limits() -> dict[str, Any]:
    """The limits database a build renders from the preset's limits.yaml."""
    database: dict[str, Any] = json.loads(e2e_conftest.LIMITS_DB_PATH.read_text(encoding="utf-8"))
    return database


def _model_entry(name: str) -> dict[str, Any]:
    return next(entry for entry in e2e_conftest.PRESET_DOCUMENT["models"] if entry["name"] == name)


def _kinds(name: str) -> list[str]:
    return list((_model_entry(name).get("measurement") or {}).get("kinds") or ())


def test_lane_selection_is_the_lane_model_s_bounded_corrector(limits: dict[str, Any]) -> None:
    binding = VIEW.binding(LANE.setpoint)
    record = limits[LANE.setpoint]

    assert binding is not None and binding.model == LANE.model
    assert binding.role == "setpoint" and binding.plane is not None
    assert LANE.band == tuple(VIEW.channel(LANE.setpoint).value_range or ())
    assert LANE.band == (record["min_value"], record["max_value"])
    assert VIEW.channel(LANE.setpoint).pair == LANE.readback != LANE.setpoint
    assert LANE.monitor_x in monitor_addresses(VIEW, plane="x", model=LANE.model)
    assert LANE.monitor_y in monitor_addresses(VIEW, plane="y", model=LANE.model)
    assert LANE.status_address == VIEW.status_addresses()[LANE.model]
    assert VIEW.channel(LANE.locked_setpoint).role == "setpoint"
    assert limits[LANE.locked_setpoint]["writable"] is False
    assert e2e_conftest.READINESS_ADDRESS == LANE.readback
    assert e2e_conftest.BOUNDED_SETPOINT == LANE.setpoint
    assert e2e_conftest.LANE_MODEL == LANE.model
    assert LANE.plane == binding.plane
    assert LANE.plane in {"x", "y"}
    low, high = LANE.band
    for label, current in SWEEP:
        assert low <= current <= high, f"sweep point {label} ({current}) leaves {LANE.band}"


def test_the_scenario_world_names_the_kinds_it_claims(limits: dict[str, Any]) -> None:
    lane = e2e_conftest.LANE_MODEL

    kick = VIEW.binding(e2e_conftest.KICK_CORRECTOR)
    assert kick is not None
    assert (kick.model, kick.role, kick.plane) == (lane, "setpoint", "y")
    assert e2e_conftest.KICK_MONITOR in monitor_addresses(VIEW, plane="y", model=lane)
    assert e2e_conftest.SEEDED_READOUT_BPM in monitor_addresses(VIEW, plane="x", model=lane)

    quadrupole = VIEW.binding(e2e_conftest.UNSTABLE_QUADRUPOLE)
    assert quadrupole is not None
    assert (quadrupole.model, quadrupole.role, quadrupole.plane) == (lane, "setpoint", None)
    on = VIEW.channel(e2e_conftest.UNSTABLE_QUADRUPOLE).on
    assert on is not None and "device" in on
    devices = {device["id"]: device for device in e2e_conftest.PRESET_DOCUMENT["devices"]}
    assert devices[on["device"]]["class"] == "Quadrupole"

    burst = VIEW.channel(e2e_conftest.BURST_CHANNEL)
    burst_seed = VIEW.seed(e2e_conftest.BURST_CHANNEL)
    assert burst.role != "setpoint" and VIEW.binding(e2e_conftest.BURST_CHANNEL) is None
    assert burst_seed is not None
    nominal = float(burst_seed["nominal"])
    sigma = _noise_sigma(burst_seed.get("noise"), nominal)
    assert sigma > 0.0
    assert e2e_conftest.BURST_VALUE - nominal > 100 * sigma

    echo = VIEW.channel(e2e_conftest.TEXTURE_ECHO_SETPOINT)
    assert echo.role == "setpoint" and echo.writable
    assert VIEW.binding(e2e_conftest.TEXTURE_ECHO_SETPOINT) is None
    assert echo.pair == e2e_conftest.TEXTURE_ECHO_READBACK != e2e_conftest.TEXTURE_ECHO_SETPOINT
    if echo.value_range is not None:
        low, high = echo.value_range
        assert low <= e2e_conftest.TEXTURE_PLANTED_VALUE <= high

    lanes_slots = {e2e_conftest.corrector_at_slot(slot).setpoint_address for slot in range(3)}
    wired = {
        address: binding
        for address in e2e_conftest.OPERATING_POINT
        if (binding := VIEW.binding(address)) is not None
    }
    assert sorted((binding.model, binding.role, binding.plane) for binding in wired.values()) == [
        (lane, "setpoint", "x"),
        (lane, "setpoint", "y"),
    ]
    assert not wired.keys() & lanes_slots
    (texture,) = set(e2e_conftest.OPERATING_POINT) - wired.keys()
    texture_channel = VIEW.channel(texture)
    texture_seed = VIEW.seed(texture)
    assert texture_channel.role == "setpoint" and texture_channel.writable
    assert texture_channel.pair is not None and texture_channel.pair != texture
    assert texture not in limits
    assert texture_seed is not None
    assert e2e_conftest.OPERATING_POINT[texture] != texture_seed["nominal"]

    for address in e2e_conftest.CHROMATICITY:
        output = VIEW.binding(address)
        assert output is not None
        assert (output.model, output.role) == (lane, "output")


def test_the_second_model_is_a_served_line_without_dispersion() -> None:
    second = e2e_conftest.SECOND_MODEL

    assert second in VIEW.served()
    assert second != e2e_conftest.LANE_MODEL
    assert "orm" in _kinds(second) and "dispersion" not in _kinds(second)
    assert all(VIEW.seed(binding.address) is None for binding in VIEW.bindings(model=second))
    assert second_models(VIEW, e2e_conftest.PRESET_DOCUMENT) == (second,)


def test_the_bench_world_names_bench_records() -> None:
    records = set(re.findall(r'record\(\s*\w+\s*,\s*"([^"]+)"\s*\)', BENCH_DB.read_text()))
    channels = set(VIEW.channels())

    assert records
    assert set(BENCH_RECORDS) <= records
    assert VIEW.channel(bench.PROBE_CHANNEL).role == "readback"
    assert VIEW.channel(bench.WRITABLE_SP).role == "setpoint"
    assert VIEW.channel(bench.PROTECTED_SP).role == "setpoint"
    shared = VIEW.binding(bench.SHARED_BPM_X)
    assert shared is not None and shared.role == "monitor"
    writable = VIEW.channel(bench.WRITABLE_SP)
    assert writable.writable and writable.value_range is not None
    low, high = writable.value_range
    assert low <= bench.VA_WRITE_VALUE <= high
    assert bench.VA_ONLY_CHANNEL in channels and bench.VA_ONLY_CHANNEL not in records
    assert bench.BENCH_ONLY_AMPLITUDE in records and bench.BENCH_ONLY_AMPLITUDE not in channels
