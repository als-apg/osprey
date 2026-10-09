"""The lattice models a render's simulator view offers the dashboard.

A render carries the simulator view under ``data/simulator/``:
``variables.json`` lists every model with its engine, its settings, whether the
render serves it, and the path of its deck copy under ``decks/``. The dashboard
switches between the models that have a deck; engine ``texture`` is never one
of them. Each model's settings are checked against its deck by the pyAT
engine's ``prepare``, so the dashboard solves a deck exactly as the simulator
does, and a model the engine stops on is listed with the stop's text. The
served models come first, in the view's order, then the others by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.simulation.engines.pyat import Prepared

#: The simulator view's directory under a render root.
SIMULATOR_VIEW_DIR = Path("data") / "simulator"

#: Shown when the render carries no simulator view.
NO_VIEW_TEXT = "no simulator view in this build"

#: Shown when the view serves no deck-bearing model.
NO_SERVED_MODEL_TEXT = "no lattice model is served"


@dataclass(frozen=True)
class DashboardModel:
    """One model the dashboard can load.

    Attributes:
        name: The model's name in the facility file.
        served: Whether the render serves it.
        deck: The deck copy under the simulator view.
        prepared: The model's ``pyat`` settings checked against its deck, or
            None when the engine stops on them.
        error: The engine's stop, or None when the settings are sound.
    """

    name: str
    served: bool
    deck: Path
    prepared: Prepared | None
    error: str | None = None

    @property
    def solve(self) -> str | None:
        """The model's solve, or None when its settings are not sound."""
        return None if self.prepared is None else self.prepared.solve


@dataclass(frozen=True)
class ModelCatalog:
    """The models of one render, or the absence of a simulator view.

    Attributes:
        has_view: Whether the render carries ``variables.json``.
        models: The switchable models, served first.
    """

    has_view: bool
    models: tuple[DashboardModel, ...] = ()

    def get(self, name: str) -> DashboardModel | None:
        """Return the model called *name*, or None."""
        return next((model for model in self.models if model.name == name), None)

    def default(self) -> DashboardModel | None:
        """Return the first served model, or None when none is served."""
        return next((model for model in self.models if model.served), None)

    def notice(self) -> str | None:
        """Return the banner text for a render the dashboard cannot load from."""
        if not self.has_view:
            return NO_VIEW_TEXT
        if self.default() is None:
            return NO_SERVED_MODEL_TEXT
        return None


def catalog_sources(render_root: Path) -> tuple[Path, ...]:
    """Return the files a catalog of *render_root* is read from.

    Returns:
        ``data/simulator/variables.json``, alone.
    """
    from osprey.facility.views.simulator import VARIABLES_FILE

    return (Path(render_root) / SIMULATOR_VIEW_DIR / VARIABLES_FILE,)


def _model(view: Path, record: dict[str, Any]) -> DashboardModel:
    from osprey.facility.errors import FacilityBuildError
    from osprey.simulation.engines import pyat

    name = str(record["name"])
    deck = view / str(record["deck"])
    try:
        prepared: Prepared | None = pyat.prepare(deck, record.get("settings"), model=name)
        error = None
    except FacilityBuildError as exc:
        prepared, error = None, str(exc)
    return DashboardModel(
        name=name, served=bool(record.get("served")), deck=deck, prepared=prepared, error=error
    )


def read_catalog(render_root: Path | None) -> ModelCatalog:
    """Read the switchable models of the render at *render_root*.

    Loads each model's deck through the pyAT engine, so it is called off the
    request path.

    Args:
        render_root: The render's root directory, or None for no render.

    Returns:
        The catalog; ``has_view`` is False when *render_root* is None or holds
        no ``data/simulator/variables.json``.
    """
    from osprey.facility import TEXTURE

    if render_root is None:
        return ModelCatalog(has_view=False)
    (variables_path,) = catalog_sources(render_root)
    if not variables_path.is_file():
        return ModelCatalog(has_view=False)

    view = variables_path.parent
    found = [
        _model(view, record)
        for record in json.loads(variables_path.read_text()).get("models", [])
        if record.get("engine") != TEXTURE and record.get("deck") and record.get("name")
    ]
    served = [model for model in found if model.served]
    unserved = sorted((model for model in found if not model.served), key=lambda m: m.name)
    return ModelCatalog(has_view=True, models=tuple(served + unserved))
