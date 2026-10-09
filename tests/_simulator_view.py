"""Stage the scenarios of a render's simulator view, as the build writes them."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.facility.sources import read_yaml
from osprey.facility.views import view_bytes
from osprey.facility.views.simulator import SCENARIOS_DIR, SCENARIOS_FILE, SCENARIOS_SCHEMA
from osprey_connectors.simulation.view import SCHEMAS


def write_scenarios_view(
    render: Path,
    scenarios: Mapping[str, Mapping[str, Any]],
    files: Path | None = None,
) -> Path:
    """Write ``<render>/data/simulator/scenarios.json`` listing ``scenarios``.

    Args:
        render: The render's root.
        scenarios: Scenario name -> its blocks (``description``, ``archiver``,
            ``logbook``, ...), carried verbatim.
        files: A facility ``scenarios/`` directory whose ``<name>/``
            subdirectories, holding the files the entries attach, are copied
            beside the view as the build copies them.

    Returns:
        The file written.
    """
    document = {
        "schema": SCENARIOS_SCHEMA,
        "scenarios": [{"name": name, **dict(scenarios[name])} for name in sorted(scenarios)],
    }
    path = render / "data" / "simulator" / SCENARIOS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(view_bytes(document))
    if files is not None:
        for name in scenarios:
            if (files / name).is_dir():
                shutil.copytree(files / name, path.parent / SCENARIOS_DIR / name)
    return path


def facility_scenarios(directory: Path) -> dict[str, dict[str, Any]]:
    """The scenarios of a facility's ``scenarios/`` directory, by file stem."""
    return {
        path.stem: read_yaml(path.read_text(encoding="utf-8")) or {}
        for path in sorted(directory.glob("*.yaml"))
    }


#: The fields of a :func:`write_texture_view` channel that go to ``seeds.json``.
_SEED_FIELDS = ("nominal", "noise", "drift", "clamp")


def write_texture_view(
    render: Path,
    channels: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    models: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write a whole simulator view, every document carrying its schema.

    Args:
        render: The render's root.
        channels: Address -> its ``value_type`` (``float`` when absent),
            ``options``, ``role`` (``readback`` when absent) and ``owner``
            (``texture`` when absent), plus the seed fields ``nominal``,
            ``noise``, ``drift`` and ``clamp``.
        scenarios: Scenario name -> its blocks, as :func:`write_scenarios_view`
            takes them.
        models: Served physics models, each a ``variables.json`` model record
            (``name``, ``engine``, ``settings``, ``deck`` and ``wiring``, every
            wiring record carrying its ``role``, ``plane`` and ``refresh``);
            each is listed as served with its status address.

    Returns:
        The view directory, ``<render>/data/simulator``.
    """
    records = []
    seeds: dict[str, dict[str, Any]] = {}
    for address in sorted(channels):
        spec = dict(channels[address])
        role = spec.get("role", "readback")
        records.append(
            {
                "address": address,
                "role": role,
                "pair": address if role == "setpoint" else None,
                "value_type": spec.get("value_type", "float"),
                "unit": None,
                "description": None,
                "writable": role == "setpoint",
                "value_range": None,
                "owner": spec.get("owner", "texture"),
                "on": None,
                **({"options": spec["options"]} if "options" in spec else {}),
            }
        )
        seed = {field: spec[field] for field in _SEED_FIELDS if field in spec}
        if seed:
            seeds[address] = seed
    physics = [{"served": True, **dict(model)} for model in models]
    names = sorted(str(model["name"]) for model in physics)
    view = render / "data" / "simulator"
    documents = {
        "served_models.json": {"models": [*names, "texture"]},
        "addresses.json": {
            "channels": sorted(channels),
            "status": [f"T:SIM:{name}:STATUS" for name in names],
        },
        "variables.json": {"code": "T", "models": physics, "channels": records},
        "seeds.json": {"seeds": seeds},
    }
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_bytes(view_bytes({"schema": SCHEMAS[name], **document}))
    write_scenarios_view(render, {"nominal": {}, **(scenarios or {})})
    return view
