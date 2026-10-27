"""A synthetic render's simulator view, and the wait for the dashboard to resolve over it.

:func:`write_render` writes the view's documents, ``variables.json`` listing the
models, and a pyAT JSON deck (:func:`fodo_deck`) per deck-bearing model;
:func:`settle` reads the app's state once no resolve is loading.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    SCHEMAS,
    SEEDS_FILE,
    SERVED_MODELS_FILE,
    TEXTURE,
    VARIABLES_FILE,
)

at = pytest.importorskip("at")


def fodo_deck(kf: float = 1.0) -> at.Lattice:
    d = at.Drift("DR", 0.5)
    qf = at.Quadrupole("QF", 0.2, kf)
    qd = at.Quadrupole("QD", 0.2, -kf)
    b = at.Dipole("BM", 0.5, np.pi / 8)
    return at.Lattice([qf, d, b, d, qd, d, b, d] * 2, name="FODO", energy=2e9)


def write_render(root, *, served, models):
    """Write a render's simulator view.

    ``models`` is ``{name: settings.pyat or None}`` (None = no deck); a model is
    served when ``served`` names it, and the texture model is listed too.
    """
    view = root / "data" / "simulator"
    (view / "decks").mkdir(parents=True, exist_ok=True)
    records = [
        {
            "name": "texture",
            "engine": "texture",
            "served": "texture" in served,
            "settings": {},
            "deck": None,
        }
    ]
    for name, pyat in models.items():
        deck = None
        if pyat is not None:
            deck = f"decks/{name}.json"
            fodo_deck().save(str(view / deck))
        records.append(
            {
                "name": name,
                "engine": "pyat",
                "served": name in served,
                "settings": {"pyat": pyat} if pyat is not None else {},
                "deck": deck,
            }
        )
    documents = {
        SERVED_MODELS_FILE: {
            "models": [name for name in served if name in models or name == TEXTURE]
        },
        ADDRESSES_FILE: {"channels": [], "status": []},
        VARIABLES_FILE: {"code": "T", "models": records, "channels": []},
        SEEDS_FILE: {"seeds": {}},
    }
    for name, document in documents.items():
        (view / name).write_text(json.dumps({"schema": SCHEMAS[name], **document}))
    return root


def settle(client) -> dict:
    """Return the state once no resolve is loading."""
    deadline = time.monotonic() + 20
    while True:
        state = client.get("/api/state").json()
        if state["selection"]["status"] != "loading":
            return state
        assert time.monotonic() < deadline, "the selection never resolved"
        time.sleep(0.02)
