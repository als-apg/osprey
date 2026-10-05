"""Data-driven simulation engine.

Provides :class:`SimulationEngine`, which loads a machine description
(``machine.json``) and serves channel reads/writes plus synthesized archiver
time-series.
"""

from osprey_connectors.simulation.engine import (
    SimReading,
    SimulationEngine,
    engine_serves,
)
from osprey_connectors.simulation.expressions import ExpressionError
from osprey_connectors.simulation.machine import DEFAULT_SCENARIO

__all__ = [
    "DEFAULT_SCENARIO",
    "ExpressionError",
    "SimReading",
    "SimulationEngine",
    "engine_serves",
]
