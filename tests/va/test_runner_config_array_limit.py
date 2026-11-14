"""The runner configuration refuses an array channel larger than its array byte limit.

A waveform channel is served as an array of doubles, ``prod(shape)`` elements
of 8 bytes each. A channel whose array exceeds the configuration's
``max_array_bytes`` would be served truncated, so ``apply_safety`` refuses
the view instead, naming the channel, its size and the limit. Channels within
the limit, and scalar channels, are configured as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.serving.runner_config import apply_safety
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    SCHEMAS,
    VARIABLES_FILE,
    SimulatorView,
)
from tests._simulator_view import view_channel

#: The array byte limit the tests configure: 100 doubles.
LIMIT = 800


def _view(tmp_path: Path, channels: list[dict[str, Any]]) -> SimulatorView:
    """A texture-only view serving ``channels``."""
    variables = {
        "schema": SCHEMAS[VARIABLES_FILE],
        "code": "T",
        "models": [
            {
                "name": "texture",
                "engine": "texture",
                "served": True,
                "settings": {},
                "deck": None,
                "wiring": [],
            }
        ],
        "channels": channels,
    }
    addresses = {
        "schema": SCHEMAS[ADDRESSES_FILE],
        "channels": sorted(channel["address"] for channel in channels),
        "status": [],
    }
    directory = tmp_path / "view"
    directory.mkdir()
    (directory / ADDRESSES_FILE).write_text(json.dumps(addresses), encoding="utf-8")
    (directory / VARIABLES_FILE).write_text(json.dumps(variables), encoding="utf-8")
    return SimulatorView.open(directory)


def _config(addresses: list[str], limit: int | str = LIMIT) -> dict[str, Any]:
    """A configuration in ``Runner.generate_config``'s shape, its limit a string as there."""
    return {
        "description": "",
        "prefix": "",
        "max_array_bytes": str(limit),
        "variables": {
            address: {"name": address, "pv": address, "mode": "rw"} for address in addresses
        },
    }


def test_an_array_over_the_limit_is_refused_with_its_size_and_the_limit(
    tmp_path: Path,
) -> None:
    view = _view(
        tmp_path,
        [
            view_channel("T:SCALAR"),
            view_channel("T:TBT:X", value_type="waveform", shape=[101]),
        ],
    )

    with pytest.raises(ValueError, match="T:TBT:X") as refused:
        apply_safety(_config(["T:SCALAR", "T:TBT:X"]), view)

    message = str(refused.value)
    assert "808" in message
    assert str(LIMIT) in message


def test_an_array_exactly_at_the_limit_is_accepted(tmp_path: Path) -> None:
    view = _view(tmp_path, [view_channel("T:TBT:X", value_type="waveform", shape=[100])])

    config = apply_safety(_config(["T:TBT:X"]), view)

    assert config["variables"]["T:TBT:X"]["mode"] == "ro"
    assert config["max_array_bytes"] == str(LIMIT)


def test_a_multi_dimensional_array_counts_every_element(tmp_path: Path) -> None:
    view = _view(tmp_path, [view_channel("T:IMAGE", value_type="waveform", shape=[10, 11])])

    with pytest.raises(ValueError, match=r"T:IMAGE.*880.*800"):
        apply_safety(_config(["T:IMAGE"]), view)


def test_scalar_channels_are_configured_as_before_under_any_limit(tmp_path: Path) -> None:
    view = _view(
        tmp_path,
        [
            view_channel("T:A"),
            view_channel("T:N", value_type="int"),
            view_channel("T:SP", role="setpoint", writable=True, value_range=[0.0, 1.0]),
        ],
    )

    config = apply_safety(_config(["T:A", "T:N", "T:SP"], limit=1), view)

    assert config["variables"]["T:A"] == {
        "name": "T:A",
        "pv": "T:A",
        "mode": "ro",
        "precision": 4,
    }
    assert config["variables"]["T:N"] == {"name": "T:N", "pv": "T:N", "mode": "ro"}
    assert config["variables"]["T:SP"]["mode"] == "rw"
    assert config["variables"]["T:SP"]["value_range"] == [0.0, 1.0]


def test_an_address_the_view_does_not_name_is_not_measured(tmp_path: Path) -> None:
    view = _view(tmp_path, [view_channel("T:A")])

    config = apply_safety(_config(["T:A", "T:STATUS"], limit=1), view)

    assert config["variables"]["T:STATUS"]["mode"] == "ro"
