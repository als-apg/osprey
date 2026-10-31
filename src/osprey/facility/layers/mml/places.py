"""The mml layer's places: each imported device's place, from its export's ``DeviceList``.

A model whose mapping answers ``places`` (:class:`~osprey.facility.layers.mml.mapping.PlaceRules`)
is a place, ``<model>``; a model whose ``places`` states ``sectors`` also has
one place per sector under it, ``<model>/<token>``, written from that list
alone, whether or not a device sits there. A model without ``places`` gets no
place, and neither do its devices.

Each device of such a model takes the first of these that applies:

1. one of its families is in the model's ``model_wide``: the model;
2. one of its families has a ``sector_override``: that sector;
3. its ``DeviceList`` rows name more than one sector: the model;
4. its rows name one sector: that sector;
5. it has no ``DeviceList`` row: the model.

A device's address is never where its place comes from: a series supply is one
address that many magnet devices bind, each on its own row, so each magnet is
placed by its own row. Two refusals keep a wrong sector out of the records,
both reported as problems of the mapping so the import stops before anything
is written:

* a row of a device placed by rule 3 or 4 names a sector beyond the model's
  list;
* a device placed by rule 4 is bound alone to an address that begins with
  another sector's token. A ``sector_override`` or ``model_wide`` answer for
  one of its families settles it.

The build records such a place as ``place_from: imported``
(:mod:`~osprey.facility.compute`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from osprey.facility.layers.mml.mapping import Model, Problem

__all__ = ["Placed", "Slot", "place_devices"]


@dataclass(frozen=True)
class Slot:
    """One device slot of one family: its system, family, device id and ``DeviceList`` row."""

    system: str
    family: str
    device: str
    row: tuple[int, ...] | None


@dataclass
class Placed:
    """The place records a set of models carries, each device's place, and the refusals."""

    places: list[dict[str, Any]] = field(default_factory=list)
    devices: dict[str, str] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)


@dataclass
class _Device:
    system: str
    families: list[str] = field(default_factory=list)
    sectors: list[int] = field(default_factory=list)


def place_devices(
    models: Mapping[str, Model],
    slots: Iterable[Slot],
    bound: Mapping[str, Iterable[str]],
) -> Placed:
    """Place every device of the models that answer ``places``.

    Args:
        models: The imported models, keyed by raw system token.
        slots: Every device slot of the import, in import order.
        bound: The addresses each device is bound to alone (its channels' ``on``).

    Returns:
        The place records sorted by id, ``{device id: place id}``, and every
        refusal in device order.
    """
    placed = Placed()
    records: dict[str, dict[str, Any]] = {}
    for model in models.values():
        rules = model.places
        if rules is None:
            continue
        records[model.name] = {"id": model.name}
        if rules.sectors is not None:
            for name in rules.sectors.names().values():
                place = f"{model.name}/{name}"
                records[place] = {"id": place}
    placed.places = [records[key] for key in sorted(records)]

    devices: dict[str, _Device] = {}
    for slot in slots:
        model = models.get(slot.system)
        if model is None or model.places is None:
            continue
        device = devices.setdefault(slot.device, _Device(slot.system))
        if slot.family not in device.families:
            device.families.append(slot.family)
        if slot.row:
            device.sectors.append(slot.row[0])
    for device_id, device in devices.items():
        model = models[device.system]
        place, problems = _place(device_id, device, model, list(bound.get(device_id, ())))
        placed.devices[device_id] = place
        placed.problems.extend(problems)
    return placed


def _place(
    device_id: str, device: _Device, model: Model, addresses: list[str]
) -> tuple[str, list[Problem]]:
    """One device's place by the five rules, and what refuses it."""
    rules = model.places
    assert rules is not None
    sectors = rules.sectors
    if sectors is None:
        return model.name, []
    path = f"models.{model.raw}.places"
    if any(family in rules.model_wide for family in device.families):
        return model.name, []
    overrides = sorted(
        {rules.sector_override[f] for f in device.families if f in rules.sector_override}
    )
    if len(overrides) > 1:
        shown = ", ".join(str(sector) for sector in overrides)
        return model.name, [
            Problem(f"{path}.sector_override", f"{device_id}'s families answer sectors {shown}")
        ]
    if overrides:
        return f"{model.name}/{sectors.name(overrides[0])}", []
    distinct = sorted(set(device.sectors))
    beyond = [sector for sector in distinct if not 1 <= sector <= sectors.count]
    if beyond:
        shown = ", ".join(str(sector) for sector in beyond)
        return model.name, [
            Problem(
                f"{path}.sectors",
                f"{device_id} has DeviceList sector {shown} beyond {model.raw}'s "
                f"{sectors.count}; answer it with sector_override or model_wide",
            )
        ]
    if len(distinct) != 1:
        return model.name, []
    sector = distinct[0]
    named = sectors.name(sector)
    problems = []
    for address in addresses:
        stated = _address_sector(address, sectors.names())
        if stated is not None and stated != sector:
            problems.append(
                Problem(
                    path,
                    f"{device_id} is in DeviceList sector {sector} ({named}) but is bound alone "
                    f"to {address}, an address of {sectors.name(stated)}; answer it with "
                    "sector_override or model_wide",
                )
            )
    return f"{model.name}/{named}", problems


def _address_sector(address: str, names: Mapping[int, str]) -> int | None:
    """The sector whose token the address begins with, the longest token winning."""
    found = [(len(name), sector) for sector, name in names.items() if address.startswith(name)]
    return max(found)[1] if found else None
