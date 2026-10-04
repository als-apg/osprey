"""The simulator view: the models a render serves, their addresses and decks.

Written into ``<render>/data/simulator/``::

    served_models.json   {schema: osprey.facility.served_models/1, models: [...]}
    addresses.json       {schema: osprey.facility.addresses/1, channels: [...], status: [...]}
    decks/<model>.json   a byte copy of each deck-bearing model's deck

``models`` lists the served physics models sorted by name, then ``texture``;
readers building physics children or selectors skip engine ``texture``.
``channels`` is every channel address of the facility file, sorted; ``status``
is ``<code>:SIM:<model>:STATUS`` for each served physics model and appears in
no other view. A deck is copied for every model that names one, served or not,
so a render's deck set does not depend on ``simulation.models``.

``simulator_wiring`` gives one model's wiring entries: each wired address with
its element (or slices), engine block and calibration, plus the channel facts
the build filled in (direction, unit, default, value_range).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.build import FacilityDocument
from osprey.facility.views import ViewInputs, view_bytes

__all__ = [
    "ADDRESSES_FILE",
    "ADDRESSES_SCHEMA",
    "DECKS_DIR",
    "SERVED_MODELS_FILE",
    "SERVED_MODELS_SCHEMA",
    "simulator_wiring",
    "status_address",
    "write_simulator_view",
]

SERVED_MODELS_FILE = "served_models.json"
SERVED_MODELS_SCHEMA = "osprey.facility.served_models/1"
ADDRESSES_FILE = "addresses.json"
ADDRESSES_SCHEMA = "osprey.facility.addresses/1"
DECKS_DIR = "decks"

#: The keys a wiring entry may carry, in emission order.
_WIRING_ENTRY_KEYS = (
    "id",
    "address",
    "element",
    "slices",
    "engine",
    "calibration",
    "direction",
    "unit",
    "default",
    "value_range",
)


def simulator_wiring(facility: FacilityDocument, model: str) -> list[dict[str, Any]]:
    """The wiring entries of one model, in the facility file's record order.

    Each entry holds the keys of ``_WIRING_ENTRY_KEYS`` the wiring record
    carries and no other; a key the record lacks is absent from its entry.

    Args:
        facility: The in-memory facility file.
        model: The model's name.

    Returns:
        One entry per wiring record of the model; empty for a model without
        wiring. The entries share no objects with ``facility``.

    Raises:
        KeyError: ``facility`` has no model named ``model``.
    """
    for entry in facility.get("models", []):
        if entry["name"] == model:
            return [
                {key: copy.deepcopy(record[key]) for key in _WIRING_ENTRY_KEYS if key in record}
                for record in entry.get("wiring", [])
            ]
    raise KeyError(f"the facility file has no model {model!r}")


def status_address(code: str, model: str) -> str:
    """The status address the simulator serves for one physics model.

    Args:
        code: The facility's identity ``code``.
        model: The model's name.

    Returns:
        ``<code>:SIM:<model>:STATUS``.
    """
    return f"{code}:SIM:{model}:STATUS"


def write_simulator_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the simulator view into ``root``.

    Args:
        root: The view's directory, ``<render>/data/simulator``.
        inputs: The render's view inputs.

    Returns:
        The files written, sorted.
    """
    doc = inputs.doc
    code = str(doc["identity"]["code"])
    physics = [name for name in inputs.served if name != TEXTURE]
    documents = {
        SERVED_MODELS_FILE: {"schema": SERVED_MODELS_SCHEMA, "models": list(inputs.served)},
        ADDRESSES_FILE: {
            "schema": ADDRESSES_SCHEMA,
            "channels": sorted(str(channel["id"]) for channel in doc.get("channels", [])),
            "status": [status_address(code, name) for name in physics],
        },
    }

    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, document in documents.items():
        target = root / name
        target.write_bytes(view_bytes(document))
        written.append(target)

    decks = root / DECKS_DIR
    for model in doc.get("models", []):
        deck = model.get("deck")
        if deck is None:
            continue
        decks.mkdir(exist_ok=True)
        target = decks / f"{model['name']}.json"
        target.write_bytes((inputs.facility_dir / deck).read_bytes())
        written.append(target)
    return sorted(written)
