"""The deployment's choices: which decks, which engine computes each, which analyses run.

Read from the application's config (``apps.lattice.config`` in the profile)::

    deadline_s: 300            # one analysis's bound, unless the analysis names its own
    max_workers: 2             # analyses computed at once
    precompute: false          # true computes every offered analysis at startup
    engines:                   # the facility's own engines, beside pyat and elegant
      aps-elegant:
        command: [python, -m, aps_lattice.engine]
        analyses: [optics, da, lma]     # omit: every analysis
        version: "3"                    # change it to drop the engine's cached results
    models:
      - name: booster
        label: Booster
        deck: decks/booster.lte         # absolute, or relative to the deployment's root
        engine: pyat
        engine_settings: {bend_pass: ExactSectorBendPass, dp_step: 1.0e-5}
        energies_gev: [0.425, 3.0, 6.0]
        analyses: {lma: {enabled: false}}   # this model's overrides of the block below
    analyses:
      da: {settings: {nturns: 128, n_angles: 9, n_bisect: 8}, deadline_s: 1200}

With no ``models`` the deployment's simulator decks are offered, computed by pyAT.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lattice_app.catalog import ANALYSES

#: One analysis's bound in seconds when the config names none.
DEFAULT_DEADLINE_S = 300.0
#: Analyses computed at once when the config names no ``max_workers``.
DEFAULT_MAX_WORKERS = 2


@dataclass(frozen=True)
class Engine:
    """What computes an analysis: a command run as ``<command> <job.json> <out.json>``.

    Attributes:
        name: The engine's name in ``models[].engine``.
        command: The command's leading arguments.
        analyses: The analyses it computes; ``None`` for every one.
        version: Part of every cached result's key: a new version recomputes.
    """

    name: str
    command: tuple[str, ...]
    analyses: frozenset[str] | None = None
    version: str = "1"


#: The engines the application carries.
BUILTIN_ENGINES: dict[str, Engine] = {
    "pyat": Engine("pyat", (sys.executable, "-m", "lattice_app.engines.pyat"), version="2"),
    "elegant": Engine(
        "elegant",
        (sys.executable, "-m", "lattice_app.engines.elegant"),
        frozenset({"optics", "chromaticity", "da", "lma"}),
    ),
}


@dataclass(frozen=True)
class Setup:
    """How one analysis runs on one model."""

    enabled: bool
    settings: Mapping[str, Any]
    deadline_s: float


@dataclass(frozen=True)
class Model:
    """One deck the panel offers.

    Attributes:
        name: The model's key in routes and tools.
        label: What the panel calls it.
        deck: The deck file.
        engine: What computes its analyses.
        engine_settings: Handed to the engine unread.
        energies_gev: The energies the panel offers, the first the default; empty when
            the deck states its own.
        setups: Each analysis's setup on this model.
    """

    name: str
    label: str
    deck: Path
    engine: Engine
    engine_settings: Mapping[str, Any]
    energies_gev: tuple[float, ...]
    setups: Mapping[str, Setup]

    def offered(self) -> list[str]:
        """The analyses this model runs: enabled, and computed by its engine."""
        supported = self.engine.analyses
        return [
            name
            for name, setup in self.setups.items()
            if setup.enabled and (supported is None or name in supported)
        ]


@dataclass(frozen=True)
class Settings:
    """The application's config, checked."""

    models: tuple[Model, ...]
    max_workers: int = DEFAULT_MAX_WORKERS
    precompute: bool = False

    def model(self, name: str) -> Model | None:
        """The model named *name*, or ``None``."""
        return next((model for model in self.models if model.name == name), None)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} is not a mapping")
    return value


def _seconds(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} is not a positive number of seconds")
    return float(value)


def _engines(raw: Any) -> dict[str, Engine]:
    engines = dict(BUILTIN_ENGINES)
    for name, entry in _mapping(raw, "engines").items():
        where = f"engines.{name}"
        entry = _mapping(entry, where)
        command = entry.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{where}.command is not a list of arguments")
        analyses = entry.get("analyses")
        if analyses is not None:
            if unknown := sorted(set(analyses) - set(ANALYSES)):
                raise ValueError(f"{where}.analyses names no analysis {', '.join(unknown)}")
            analyses = frozenset(analyses)
        engines[str(name)] = Engine(
            str(name),
            tuple(str(part) for part in command),
            analyses,
            str(entry.get("version", "1")),
        )
    return engines


def _setups(
    raw: Any, where: str, base: Mapping[str, Setup], default_deadline_s: float
) -> dict[str, Setup]:
    """*base* with the ``analyses`` block *raw* applied over it, key by key."""
    block = _mapping(raw, where)
    if unknown := sorted(set(block) - set(ANALYSES)):
        raise ValueError(f"{where} names no analysis {', '.join(unknown)}")
    setups: dict[str, Setup] = {}
    for name, analysis in ANALYSES.items():
        entry = _mapping(block.get(name), f"{where}.{name}")
        before = base.get(name) or Setup(True, dict(analysis.settings), default_deadline_s)
        overrides = _mapping(entry.get("settings"), f"{where}.{name}.settings")
        deadline = entry.get("deadline_s")
        setups[name] = Setup(
            enabled=bool(entry.get("enabled", before.enabled)),
            settings={**before.settings, **overrides},
            deadline_s=(
                before.deadline_s
                if deadline is None
                else _seconds(deadline, f"{where}.{name}.deadline_s")
            ),
        )
    return setups


def _model(
    entry: Any, index: int, root: Path, engines: Mapping[str, Engine], base: Mapping[str, Setup]
) -> Model:
    where = f"models[{index}]"
    entry = _mapping(entry, where)
    name, deck = entry.get("name"), entry.get("deck")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}.name is missing")
    if not isinstance(deck, str) or not deck.strip():
        raise ValueError(f"{where}.deck is missing")
    engine_name = str(entry.get("engine", "pyat"))
    if engine_name not in engines:
        raise ValueError(
            f"{where}.engine names no engine {engine_name}; the engines: {', '.join(engines)}"
        )
    energies = entry.get("energies_gev") or []
    if not isinstance(energies, list) or any(
        isinstance(energy, bool) or not isinstance(energy, (int, float)) or energy <= 0
        for energy in energies
    ):
        raise ValueError(f"{where}.energies_gev is not a list of positive energies")
    return Model(
        name=name,
        label=str(entry.get("label") or name),
        deck=(root / deck).resolve(),
        engine=engines[engine_name],
        engine_settings=dict(_mapping(entry.get("engine_settings"), f"{where}.engine_settings")),
        energies_gev=tuple(float(energy) for energy in energies),
        setups=_setups(entry.get("analyses"), f"{where}.analyses", base, DEFAULT_DEADLINE_S),
    )


def simulator_models(base: Mapping[str, Setup]) -> tuple[Model, ...]:
    """The deployment's simulator decks as models, computed by pyAT; none without a render."""
    try:
        from osprey_connectors.simulation.view import SimulatorView
        from osprey_connectors.workspace import resolve_config_path

        view = SimulatorView.of_render(Path(resolve_config_path()).parent)
        found = [model for model in view.physics_models() if model.deck is not None]
    except Exception:  # no render, no view, an unreadable one: no models to offer
        return ()
    return tuple(
        Model(
            name=model.name,
            label=model.name,
            deck=Path(model.deck),
            engine=BUILTIN_ENGINES["pyat"],
            engine_settings={},
            energies_gev=(),
            setups=base,
        )
        for model in found
        if model.engine == "pyat" and model.deck is not None
    )


def load_settings(config: Mapping[str, Any], root: Path | None = None) -> Settings:
    """The application's config as settings.

    Args:
        config: The application's config, as the deployment rendered it.
        root: What a relative deck path is relative to; the working directory when
            omitted, which in the application's container is the deployment's root.

    Raises:
        ValueError: The config names something the application cannot run.
    """
    root = Path.cwd() if root is None else root
    deadline = config.get("deadline_s")
    default_deadline_s = (
        DEFAULT_DEADLINE_S if deadline is None else _seconds(deadline, "deadline_s")
    )
    base = _setups(config.get("analyses"), "analyses", {}, default_deadline_s)
    engines = _engines(config.get("engines"))
    listed = config.get("models")
    if listed is None:
        models = simulator_models(base)
    elif not isinstance(listed, list):
        raise ValueError("models is not a list")
    else:
        models = tuple(
            _model(entry, index, root, engines, base) for index, entry in enumerate(listed)
        )
        names = [model.name for model in models]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"models names {', '.join(repeated)} more than once")
    workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("max_workers is not a positive whole number")
    return Settings(models=models, max_workers=workers, precompute=bool(config.get("precompute")))
