"""The demo's ``seeds.yaml``, derived from its machine file and the channel taxonomy.

A seed says how a channel behaves when simulated. The demo writes one for
every channel no model wires, and for every wired readback that moves or is
clamped:

* a channel in the machine file takes its ``value`` as ``nominal`` (a bool's
  0/1 as its ``FALSE``/``TRUE`` label) unless a model wires it; its relative
  ``noise`` scaled by ``|value|`` and its absolute ``noise_abs``, combined in
  quadrature, as ``noise``; its ``texture`` as ``drift``; and, on a readback,
  its ``min``/``max`` as ``clamp`` (a setpoint's range is its limits record's);
* a channel whose machine-file ``expr`` is one channel minus another takes
  that difference as ``linear``;
* a channel absent from the machine file takes the taxonomy's base value as
  ``nominal`` (a bool: ``TRUE`` where the base is non-zero), and a float
  readback also takes the taxonomy's ``noise_sigma`` at :data:`NOISE_LEVEL`.

Motion and ``clamp`` apply to float readbacks only. Seeds are sorted by
address.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_records() -> ModuleType:
    """``_records.py`` beside this file, under the name the generator loads it by."""
    name = "facility_demo__records"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("_records.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_records = _load_records()

MACHINE = f"{_records.fp._CA_DATA}/simulation/machine.json"

#: The mock connector's relative noise level, which the demo preset leaves at
#: its default.
NOISE_LEVEL = 0.01

#: The labels of a bool channel with no ``options``, false first.
BOOL_LABELS = ("FALSE", "TRUE")

#: The one texture kind the machine file uses: a slow sinusoidal wander.
WANDER = "wander"

#: A machine-file expression the seed states as ``linear``: one channel minus another.
_DIFFERENCE = re.compile(r"^ch\('(?P<plus>[^']+)'\) - ch\('(?P<minus>[^']+)'\)$")


def _noise(address: str, entry: dict[str, Any]) -> float:
    relative = float(entry.get("noise", 0.0))
    absolute = float(entry.get("noise_abs", 0.0))
    if relative and "value" not in entry:
        raise _records.RecordsError(f"{address}: relative noise on a channel with no value")
    return math.hypot(abs(float(entry.get("value", 0.0))) * relative, absolute)


def _drift(address: str, texture: dict[str, Any]) -> dict[str, float]:
    if texture.get("kind") != WANDER:
        raise _records.RecordsError(
            f"{address}: texture kind {texture.get('kind')} is not {WANDER}"
        )
    return {"amplitude": float(texture["amplitude"]), "period_s": float(texture["period_s"])}


def _linear(address: str, expr: str) -> dict[str, float]:
    match = _DIFFERENCE.match(expr)
    if match is None:
        raise _records.RecordsError(f"{address}: expression {expr!r} is not a difference")
    return {match["plus"]: 1.0, match["minus"]: -1.0}


def _nominal(address: str, value: Any, value_type: str) -> Any:
    if value_type == "bool":
        if value not in (0, 1):
            raise _records.RecordsError(f"{address}: bool value {value!r} is not 0 or 1")
        return BOOL_LABELS[int(value)]
    return float(value)


def _machine_seed(
    address: str, entry: dict[str, Any], channel: dict[str, Any], wired: bool
) -> dict[str, Any]:
    value_type = channel.get("value_type", "float")
    role = channel.get("role", "readback")
    seed: dict[str, Any] = {}
    if "expr" in entry:
        seed["linear"] = _linear(address, entry["expr"])
    elif not wired:
        if "value" not in entry:
            raise _records.RecordsError(f"{address}: no value and no expression")
        seed["nominal"] = _nominal(address, entry["value"], value_type)
    noise = _noise(address, entry)
    if value_type != "float" or role != "readback":
        if noise or "texture" in entry:
            raise _records.RecordsError(f"{address}: motion on a {value_type} {role}")
        return seed
    if noise:
        seed["noise"] = noise
    if "texture" in entry:
        seed["drift"] = _drift(address, entry["texture"])
    if "min" in entry or "max" in entry:
        seed["clamp"] = [entry.get("min"), entry.get("max")]
    return seed


def _procedural_seed(address: str, channel: dict[str, Any]) -> dict[str, Any]:
    from osprey_connectors.channel_taxonomy import classify_channel

    kind = classify_channel(address)
    value_type = channel.get("value_type", "float")
    if value_type == "bool":
        return {"nominal": BOOL_LABELS[kind.base_value != 0.0]}
    seed: dict[str, Any] = {"nominal": float(kind.base_value)}
    if value_type == "float" and channel.get("role", "readback") == "readback":
        noise = float(kind.noise_sigma(kind.base_value, NOISE_LEVEL))
        if noise:
            seed["noise"] = noise
    return seed


def build_seeds(
    channels: list[dict[str, Any]], models: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """``seeds.yaml``: address -> seed, sorted by address.

    Args:
        channels: The channel records.
        models: The ``models.yaml`` models; an address any of them wires takes
            no ``nominal``, and a wired setpoint takes no seed.

    Returns:
        The seeds, one per channel that has any.

    Raises:
        RecordsError: A machine-file channel is not a channel record, carries
            motion off a float readback, a non-wander texture, a bool value
            other than 0 or 1, an expression other than a difference, or, unwired,
            neither a value nor an expression.
    """
    by_id = {channel["id"]: channel for channel in channels}
    machine: dict[str, dict[str, Any]] = _records._json(MACHINE)["channels"]
    unknown = sorted(set(machine) - set(by_id))
    if unknown:
        raise _records.RecordsError(f"{MACHINE}: channels with no record: {unknown[:5]}")
    wired = {record["address"] for model in models for record in model["wiring"]}
    seeds: dict[str, dict[str, Any]] = {}
    for address in sorted(by_id):
        channel = by_id[address]
        if address in machine:
            seed = _machine_seed(address, machine[address], channel, address in wired)
        elif address in wired:
            seed = {}
        else:
            seed = _procedural_seed(address, channel)
        if seed:
            seeds[address] = seed
    return seeds
