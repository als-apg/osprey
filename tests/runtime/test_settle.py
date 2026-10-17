"""``osprey.runtime.settle`` waits for each written setpoint's readback, as the render states it.

Each case stands in a throwaway render: ``build/config.yml`` under the working
directory, with the facility file beside it at ``build/facility.json``. The
runtime's connector is a fake served through ``osprey.runtime._get_connector``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import osprey.runtime
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey_connectors.control_system.base import ChannelMetadata, ChannelValue
from osprey_connectors.settle import SettleTimeoutError
from osprey_connectors.simulation.envelope import MOTION_SIGMAS
from osprey_connectors.workspace import reset_config_cache

LIVE = {"type": "epics", "connector": {"epics": {}}}
SIMULATOR = {"type": "virtual_accelerator", "connector": {"virtual_accelerator": {}}}

#: A readback seed whose motion band is ``MOTION_SIGMAS x 0.001``.
MOVING = {"nominal": 2.0, "noise": {"absolute": 0.001}}


def _render(root: Path, control_system: dict[str, Any]) -> None:
    (root / "build").mkdir(parents=True, exist_ok=True)
    (root / "build" / "config.yml").write_text(
        yaml.safe_dump({"control_system": control_system}), encoding="utf-8"
    )
    reset_config_cache()


def _plant(root: Path, *channels: dict[str, Any]) -> Path:
    path = root / "build" / "facility.json"
    path.write_text(json.dumps({"channels": list(channels)}), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A render naming a live machine, the working directory, with no target stamp.

    ``CONFIG_FILE`` names the render's config, as the executor sets it for a sandbox.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    monkeypatch.chdir(root)
    for name in ("OSPREY_CONFIG", ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CONFIG_FILE", str(root / "build" / "config.yml"))
    _render(root, LIVE)
    yield root
    reset_config_cache()


class Readbacks:
    """A connector whose readings are a dict; every read is logged."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.reads: list[str] = []

    async def read_channel(self, address: str, timeout: float | None = None) -> ChannelValue:
        del timeout
        self.reads.append(address)
        return ChannelValue(
            value=self.values.get(address), timestamp=datetime.now(UTC), metadata=ChannelMetadata()
        )


@pytest.fixture
def served() -> Iterator[Readbacks]:
    machine = Readbacks({})
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = machine
        yield machine


def test_the_readback_read_is_the_setpoints_pair(repo: Path, served: Readbacks) -> None:
    _plant(
        repo, {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.05}}
    )
    served.values["Q1:RB"] = 2.04

    osprey.runtime.settle({"Q1:SP": 2.0})

    assert served.reads == ["Q1:RB"]


def test_a_setpoint_with_no_pair_is_its_own_readback(repo: Path, served: Readbacks) -> None:
    _plant(repo, {"id": "Q1:SP", "role": "setpoint", "tolerance": {"absolute": 0.05}})
    served.values["Q1:SP"] = 2.0

    osprey.runtime.settle({"Q1:SP": 2.0})

    assert served.reads == ["Q1:SP"]


def test_a_relative_tolerance_is_taken_of_the_demand(repo: Path, served: Readbacks) -> None:
    _plant(
        repo, {"id": "RF:SP", "role": "setpoint", "pair": "RF:RB", "tolerance": {"relative": 1e-3}}
    )
    served.values["RF:RB"] = 499.6

    osprey.runtime.settle({"RF:SP": 500.0})

    with pytest.raises(SettleTimeoutError):
        osprey.runtime.settle({"RF:SP": 5.0}, timeout_s=0.0)


def test_an_untoleranced_setpoint_on_a_live_machine_is_not_read(
    repo: Path, served: Readbacks
) -> None:
    _plant(
        repo,
        {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB"},
        {"id": "Q1:RB", "role": "readback", "simulation": MOVING},
    )

    osprey.runtime.settle({"Q1:SP": 2.0})

    assert served.reads == []


def test_on_the_simulator_an_untoleranced_setpoint_settles_within_its_readbacks_motion(
    repo: Path, served: Readbacks
) -> None:
    _render(repo, SIMULATOR)
    _plant(
        repo,
        {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB"},
        {"id": "Q1:RB", "role": "readback", "simulation": MOVING},
    )
    band = MOTION_SIGMAS * 0.001
    served.values["Q1:RB"] = 2.0 + 0.9 * band

    osprey.runtime.settle({"Q1:SP": 2.0})

    served.values["Q1:RB"] = 2.0 + 1.1 * band
    with pytest.raises(SettleTimeoutError) as info:
        osprey.runtime.settle({"Q1:SP": 2.0}, timeout_s=0.0)
    assert info.value.landing.bound == pytest.approx(band)


def test_the_target_stamp_decides_whether_the_machine_is_the_simulator(
    repo: Path, served: Readbacks, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render(repo, {"type": "epics", "connector": {"epics": {}, "virtual_accelerator": {}}})
    _plant(
        repo,
        {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB"},
        {"id": "Q1:RB", "role": "readback", "simulation": MOVING},
    )
    served.values["Q1:RB"] = 2.0

    osprey.runtime.settle({"Q1:SP": 2.0})
    assert served.reads == [], "the live machine has no motion band"

    monkeypatch.setenv(ENV_CONTROL_TARGET, "va")
    osprey.runtime.settle({"Q1:SP": 2.0})
    assert served.reads == ["Q1:RB"]


def test_a_stuck_readback_fails_closed_naming_its_bound(repo: Path, served: Readbacks) -> None:
    _plant(
        repo, {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.05}}
    )
    served.values["Q1:RB"] = 1.5

    with pytest.raises(SettleTimeoutError) as info:
        osprey.runtime.settle({"Q1:SP": 2.0}, timeout_s=0.1)

    message = str(info.value)
    assert "Q1:RB" in message and "± 0.05" in message and "(last read: 1.5)" in message


def test_the_default_wait_is_the_settle_timeout(repo: Path, served: Readbacks) -> None:
    _plant(
        repo, {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.05}}
    )
    served.values["Q1:RB"] = 1.5
    waits: list[float] = []

    async def no_wait(connector: Any, landings: Any, *, timeout_s: float) -> None:
        del connector, landings
        waits.append(timeout_s)

    with patch("osprey_connectors.settle.settle", no_wait):
        osprey.runtime.settle({"Q1:SP": 2.0})

    assert waits == [osprey.runtime.SETTLE_TIMEOUT_S]


@pytest.mark.parametrize("address", ["Q9:SP", "Q1:RB"], ids=["unknown", "readback"])
def test_an_address_that_is_no_setpoint_is_refused(
    repo: Path, served: Readbacks, address: str
) -> None:
    path = _plant(
        repo,
        {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.05}},
        {"id": "Q1:RB", "role": "readback"},
    )

    with pytest.raises(ValueError, match=f"{address} is no setpoint in {path}"):
        osprey.runtime.settle({address: 2.0})
    assert served.reads == []


def test_a_render_without_a_facility_file_is_refused(repo: Path, served: Readbacks) -> None:
    path = repo / "build" / "facility.json"
    assert not path.exists()

    with pytest.raises(FileNotFoundError) as info:
        osprey.runtime.settle({"Q1:SP": 2.0})

    assert str(path) in str(info.value) and "osprey build" in str(info.value)
    assert served.reads == []


def test_a_rewritten_facility_file_is_read_again(repo: Path, served: Readbacks) -> None:
    path = _plant(
        repo, {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.1}}
    )
    served.values["Q1:RB"] = 2.15
    with pytest.raises(SettleTimeoutError):
        osprey.runtime.settle({"Q1:SP": 2.0}, timeout_s=0.0)

    _plant(
        repo, {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "tolerance": {"absolute": 0.2}}
    )
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    osprey.runtime.settle({"Q1:SP": 2.0}, timeout_s=0.0)


def test_settle_is_part_of_the_runtime_api() -> None:
    assert "settle" in osprey.runtime.__all__
