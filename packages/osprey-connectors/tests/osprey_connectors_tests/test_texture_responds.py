"""A status that answers its command for a stated time, and a readback its command moves.

A seed's ``responds: {to, for_s, value}`` holds ``value`` for ``for_s`` seconds
after each live write to ``to``; a channel's ``started_by`` holds a setpoint's
write until its command is written. The view reader refuses either key when it
names no command the texture can serve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.texture import TextureModel
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    SCHEMAS,
    SEEDS_FILE,
    VARIABLES_FILE,
    SimulatorView,
    ViewSchemaError,
)

T0 = 1_760_000_000.0
FOR_S = 2.0

MOVE = "T:STAGE:MOVE:SP"
STATUS = "T:STAGE:STATUS"
POSITION_SP = "T:STAGE:POSITION:SP"
POSITION_RB = "T:STAGE:POSITION:RB"
HEAT_SP = "T:HEAT:SP"
HEAT_RB = "T:HEAT:RB"
INTERLOCK = "T:INTERLOCK"
FURTHER = "T:STAGE:MOVE:ALT"
PHYSICS_SP = "T:PHYSICS:SP"
PHYSICS_RB = "T:PHYSICS:RB"
FURTHER_RB = "T:PHYSICS:RB:ALT"


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": fields.pop("pair", address if role == "setpoint" else None),
        "value_type": fields.pop("value_type", "float"),
        "unit": None,
        "description": None,
        "writable": fields.pop("writable", role == "setpoint"),
        "value_range": None,
        "owner": fields.pop("owner", "texture"),
        "on": None,
        **fields,
    }


def _documents() -> tuple[dict[str, Any], dict[str, Any]]:
    """A stage whose move command starts its position and times its status."""
    channels = [
        _channel(MOVE, role="setpoint", value_type="int"),
        _channel(STATUS, value_type="enum", options=["DONE", "MOVING", "FAULT"]),
        _channel(POSITION_SP, role="setpoint", pair=POSITION_RB, started_by=MOVE),
        _channel(POSITION_RB),
        _channel(HEAT_SP, role="setpoint", pair=HEAT_RB),
        _channel(HEAT_RB),
        _channel(INTERLOCK, role="setpoint", value_type="enum", options=["OK", "TRIPPED"]),
    ]
    variables = {"code": "T", "models": [], "channels": channels}
    seeds = {
        "seeds": {
            STATUS: {
                "nominal": "DONE",
                "responds": {"to": MOVE, "for_s": FOR_S, "value": "MOVING"},
                "gate": {"inputs": {INTERLOCK: "TRIPPED"}, "gated": "FAULT"},
            },
            POSITION_SP: {"nominal": 1.0},
            HEAT_SP: {"nominal": 20.0},
            INTERLOCK: {"nominal": "OK"},
        }
    }
    return variables, seeds


def _clocked() -> tuple[TextureModel, list[float]]:
    variables, seeds = _documents()
    clock = [T0]
    return TextureModel(variables, seeds, clock=lambda: clock[0]), clock


# ---------------------------------------------------------------------------
# The view reader
# ---------------------------------------------------------------------------


def _write_view(tmp_path: Path, variables: dict[str, Any], seeds: dict[str, Any]) -> SimulatorView:
    documents = {
        ADDRESSES_FILE: {
            "channels": sorted(channel["address"] for channel in variables["channels"]),
            "status": [],
        },
        VARIABLES_FILE: variables,
        SEEDS_FILE: seeds,
    }
    for name, document in documents.items():
        (tmp_path / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return SimulatorView.open(tmp_path)


def _with_further_address(variables: dict[str, Any]) -> None:
    """Serve ``MOVE`` again as ``FURTHER``, a further address of a physics record."""
    variables["models"] = [
        {
            "name": "M",
            "engine": "stub",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [
                {
                    "id": "w1",
                    "address": PHYSICS_SP,
                    "also": [FURTHER],
                    "direction": "write",
                    "engine_role": "setpoint",
                    "plane": None,
                    "refresh": "pass",
                }
            ],
        }
    ]
    variables["channels"] += [
        _channel(PHYSICS_SP, role="setpoint", owner="M"),
        _channel(FURTHER, role="setpoint", owner="M"),
    ]


def _no_to(_variables: dict[str, Any], seeds: dict[str, Any]) -> str:
    del seeds["seeds"][STATUS]["responds"]["to"]
    return SEEDS_FILE


def _no_time(_variables: dict[str, Any], seeds: dict[str, Any]) -> str:
    seeds["seeds"][STATUS]["responds"]["for_s"] = 0
    return SEEDS_FILE


def _to_a_readback(_variables: dict[str, Any], seeds: dict[str, Any]) -> str:
    seeds["seeds"][STATUS]["responds"]["to"] = POSITION_RB
    return SEEDS_FILE


def _to_a_further_address(variables: dict[str, Any], seeds: dict[str, Any]) -> str:
    _with_further_address(variables)
    seeds["seeds"][STATUS]["responds"]["to"] = FURTHER
    return SEEDS_FILE


def _on_a_followed_readback(_variables: dict[str, Any], seeds: dict[str, Any]) -> str:
    seeds["seeds"][HEAT_RB] = {"responds": {"to": MOVE, "for_s": FOR_S, "value": 0.0}}
    return SEEDS_FILE


def _on_a_followed_readback_named_by_its_further_address(
    variables: dict[str, Any], seeds: dict[str, Any]
) -> str:
    """``HEAT_SP`` follows a physics readback by its further address; its seed responds."""
    variables["models"] = [
        {
            "name": "M",
            "engine": "stub",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [
                {
                    "id": "w1",
                    "address": PHYSICS_RB,
                    "also": [FURTHER_RB],
                    "direction": "read",
                    "engine_role": "readback",
                    "plane": None,
                    "refresh": "pass",
                }
            ],
        }
    ]
    variables["channels"] += [
        _channel(PHYSICS_RB, owner="M"),
        _channel(FURTHER_RB, owner="M"),
    ]
    variables["channels"][4]["pair"] = FURTHER_RB
    seeds["seeds"][PHYSICS_RB] = {"responds": {"to": MOVE, "for_s": FOR_S, "value": 0.0}}
    return SEEDS_FILE


def _started_by_a_readback(variables: dict[str, Any], _seeds: dict[str, Any]) -> str:
    variables["channels"][2]["started_by"] = STATUS
    return VARIABLES_FILE


def _started_by_a_further_address(variables: dict[str, Any], _seeds: dict[str, Any]) -> str:
    _with_further_address(variables)
    variables["channels"][2]["started_by"] = FURTHER
    return VARIABLES_FILE


@pytest.mark.parametrize(
    "break_view",
    [
        _no_to,
        _no_time,
        _to_a_readback,
        _to_a_further_address,
        _on_a_followed_readback,
        _on_a_followed_readback_named_by_its_further_address,
        _started_by_a_readback,
        _started_by_a_further_address,
    ],
    ids=lambda case: case.__name__.lstrip("_"),
)
def test_a_view_with_a_malformed_responds_is_refused(tmp_path: Path, break_view: Any) -> None:
    variables, seeds = _documents()
    name = break_view(variables, seeds)
    view = _write_view(tmp_path, variables, seeds)

    with pytest.raises(ViewSchemaError, match="rebuild with osprey build"):
        view.document(name)


def test_a_view_without_the_new_keys_reads_as_before(tmp_path: Path) -> None:
    variables, seeds = _documents()
    del seeds["seeds"][STATUS]["responds"]
    del variables["channels"][2]["started_by"]
    view = _write_view(tmp_path, variables, seeds)

    assert view.document(SEEDS_FILE)["seeds"][STATUS]["nominal"] == "DONE"
    assert view.channel(POSITION_SP).pair == POSITION_RB


def test_a_view_with_both_keys_reads(tmp_path: Path) -> None:
    variables, seeds = _documents()
    view = _write_view(tmp_path, variables, seeds)

    assert view.document(SEEDS_FILE)["seeds"][STATUS]["responds"]["to"] == MOVE
    assert view.document(VARIABLES_FILE)["channels"][2]["started_by"] == MOVE
