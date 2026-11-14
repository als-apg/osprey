"""The engine ``fixture_ring``: the shipped pyat engine, plus the records it fires as commands.

Reached through the ``osprey.simulation.engines`` group as a deployment's own
engine would be. A wiring record whose engine block names a ``fire`` is the
engine's own: a written record it describes with the ``command`` mark and
starts at zero. Every other record, the deck and the settings go to the
``pyat`` engine, which the plug-in reaches through its entry point too; the
settings block is the engine's own under its name, ``fixture_ring``, plus the
``turns`` it tracks. The engine is not time-driven, and a scenario fault on a
record it fires is refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import metadata
from typing import Any

#: The engine's name, the key of its settings block.
NAME = "fixture_ring"

#: The engine block key that makes a record one the engine fires.
FIRE = "fire"

#: The settings key the engine reads beyond pyat's own.
TURNS = "turns"


def _pyat() -> Any:
    return metadata.entry_points(group="osprey.simulation.engines")["pyat"].load()


def _value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _fired(record: Any) -> bool:
    block = _value(record, "engine")
    return isinstance(block, Mapping) and FIRE in block


def _pyat_settings(settings: Any) -> dict[str, Any]:
    block = dict((settings or {}).get(NAME) or {})
    block.pop(TURNS, None)
    return {"pyat": block}


def locate(deck: Any, element: str, *, model: str | None = None) -> tuple[float, float]:
    """An element's entrance and length, as pyat finds them."""
    located: tuple[float, float] = _pyat().locate(deck, element, model=model)
    return located


def prepare(deck: Any, settings: Any, *, model: str | None = None) -> Any:
    """The deck checked against the pyat part of the settings."""
    return _pyat().prepare(deck, _pyat_settings(settings), model=model)


def start_values(
    deck: Any,
    wiring: Iterable[Any],
    settings: Any,
    *,
    readbacks: Mapping[str, str | None] | None = None,
    value_types: Mapping[str, str] | None = None,
    shapes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """pyat's start values, and zero for every record the engine fires."""
    records = list(wiring)
    fired = {str(_value(record, "address")) for record in records if _fired(record)}
    values: dict[str, Any] = dict(
        _pyat().start_values(
            deck,
            [record for record in records if not _fired(record)],
            _pyat_settings(settings),
            readbacks={a: s for a, s in (readbacks or {}).items() if a not in fired},
            value_types={a: t for a, t in (value_types or {}).items() if a not in fired},
            shapes={a: s for a, s in (shapes or {}).items() if a not in fired},
        )
    )
    return dict(sorted({**values, **dict.fromkeys(fired, 0.0)}.items()))


def describe(wiring_record: Any) -> Mapping[str, Any]:
    """A fired record is a setpoint with the ``command`` mark; pyat describes the rest."""
    if _fired(wiring_record):
        return {"engine_role": "setpoint", "plane": None, "refresh": "pass", "marks": ["command"]}
    return _pyat().describe(wiring_record)


def build(
    model: str,
    wiring: Any,
    deck: Any,
    settings: Any,
    active: Mapping[str, Any] | None = None,
) -> Any:
    """The pyat model over the records pyat drives; the fired records are not served."""
    return _pyat().build(
        model,
        [record for record in wiring if not _fired(record)],
        deck,
        _pyat_settings(settings),
        active or {},
    )


def polynomial_kicks(deck: Any, elements: Iterable[str]) -> str:
    """The deck with the named kicks as dipole polynomials, as pyat writes it."""
    kicked: str = _pyat().polynomial_kicks(deck, elements)
    return kicked


def fault_variables(wiring: Iterable[Any], settings: Any) -> dict[str, Any]:
    """No faults of its own; refuses a ``turns`` setting that is not a positive whole number."""
    del wiring
    turns = ((settings or {}).get(NAME) or {}).get(TURNS)
    if turns is not None and (isinstance(turns, bool) or not isinstance(turns, int) or turns < 1):
        raise ValueError(f"settings key {NAME}.{TURNS} is {turns!r}, not a positive whole number")
    return {}


def scenario_stops(wiring: Iterable[Any], faults: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Refuse a scenario fault on a record the engine fires."""
    fired = {str(_value(record, "address")) for record in wiring if _fired(record)}
    return [
        (
            str(fault),
            f"{fault} is a command the engine fires; a scenario holds no command",
            f"drop {fault} from the scenario's faults",
        )
        for fault in faults
        if str(fault) in fired
    ]
