"""The archive composite of a view that serves a model with no deck."""

from __future__ import annotations

import json
from pathlib import Path

from osprey_connectors.config import facility_timezone
from osprey_connectors.simulation.archive import build
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.records import (
    AddressesDocument,
    ScenariosDocument,
    SeedsDocument,
    ServedModelsDocument,
    VariablesDocument,
)
from osprey_connectors.simulation.state import ActiveState

T0 = 1_760_000_000.0


def _view(path: Path) -> Path:
    """A texture channel held at 5.0 beside a served model ``D`` that names no deck."""
    channel = {
        "address": "T:RB",
        "role": "readback",
        "pair": None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": "texture",
        "on": None,
    }
    documents = {
        ServedModelsDocument: {"models": ["D", "texture"]},
        AddressesDocument: {"channels": ["T:RB"], "status": ["T:SIM:D:STATUS"]},
        VariablesDocument: {
            "code": "T",
            "models": [
                {
                    "name": "D",
                    "engine": "unregistered",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
            ],
            "channels": [channel],
        },
        SeedsDocument: {"seeds": {"T:RB": {"nominal": 5.0}}},
        ScenariosDocument: {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir()
    for kind, document in documents.items():
        (view / kind.FILE).write_text(
            json.dumps({"schema": kind.SCHEMA, **document}), encoding="utf-8"
        )
    return view


def test_a_deckless_model_does_not_stop_the_archive_of_the_other_channels(
    tmp_path: Path,
) -> None:
    view = _view(tmp_path)
    times = [T0 + second for second in range(5)]

    archive = build(view, ActiveState(), tz=facility_timezone())
    archived = archive.series("T:RB", times)

    assert Composite(view, model_log=False).status("D") == "no deck"
    assert archived == [5.0] * len(times)
