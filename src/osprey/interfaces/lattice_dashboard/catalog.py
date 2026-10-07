"""The lattice models a render's simulator view offers the dashboard.

A render carries ``facility.json`` at its root and the simulator view under
``data/simulator/``: ``served_models.json`` names the served models and
``decks/<model>.json`` holds a copy of each deck-bearing model's deck. The
dashboard switches between the facility file's models that have such a copy;
engine ``texture`` is never one of them. A model is ``served`` when
``served_models.json`` lists it, and the served ones come first, in that
file's order, then the others by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.interfaces.lattice_dashboard.state import PERIODIC

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
        served: Whether ``served_models.json`` lists it.
        solve: ``periodic`` or ``single_pass``.
        twiss_in: The model's ``settings.pyat.twiss_in``, or None.
        deck: The deck copy under the simulator view.
    """

    name: str
    served: bool
    solve: str
    twiss_in: dict[str, Any] | None
    deck: Path


@dataclass(frozen=True)
class ModelCatalog:
    """The models of one render, or the absence of a simulator view.

    Attributes:
        has_view: Whether the render carries ``served_models.json``.
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


def _pyat_settings(record: dict[str, Any]) -> dict[str, Any]:
    settings = record.get("settings")
    block = settings.get("pyat") if isinstance(settings, dict) else None
    return block if isinstance(block, dict) else {}


def catalog_sources(render_root: Path) -> tuple[Path, Path]:
    """Return the two files a catalog of *render_root* is read from.

    Returns:
        ``data/simulator/served_models.json`` and ``facility.json``.
    """
    from osprey.facility import FACILITY_FILE
    from osprey.facility.views.simulator import SERVED_MODELS_FILE

    root = Path(render_root)
    return root / SIMULATOR_VIEW_DIR / SERVED_MODELS_FILE, root / FACILITY_FILE


def read_catalog(render_root: Path | None) -> ModelCatalog:
    """Read the switchable models of the render at *render_root*.

    Args:
        render_root: The render's root directory, or None for no render.

    Returns:
        The catalog; ``has_view`` is False when *render_root* is None or holds
        no ``data/simulator/served_models.json``.
    """
    from osprey.facility import TEXTURE
    from osprey.facility.views.simulator import DECKS_DIR

    if render_root is None:
        return ModelCatalog(has_view=False)
    served_path, facility_path = catalog_sources(render_root)
    view = served_path.parent
    if not served_path.is_file():
        return ModelCatalog(has_view=False)

    served_names = [str(name) for name in json.loads(served_path.read_text()).get("models", [])]
    records = (
        json.loads(facility_path.read_text()).get("models", []) if facility_path.is_file() else []
    )

    found: dict[str, DashboardModel] = {}
    for record in records:
        name = str(record.get("name", ""))
        deck = view / DECKS_DIR / f"{name}.json"
        if not name or name == TEXTURE or not deck.is_file():
            continue
        block = _pyat_settings(record)
        twiss_in = block.get("twiss_in")
        found[name] = DashboardModel(
            name=name,
            served=name in served_names,
            solve=str(block.get("solve", PERIODIC)),
            twiss_in=twiss_in if isinstance(twiss_in, dict) else None,
            deck=deck,
        )

    served = [found[name] for name in served_names if name in found]
    unserved = sorted((model for model in found.values() if not model.served), key=lambda m: m.name)
    return ModelCatalog(has_view=True, models=tuple(served + unserved))
