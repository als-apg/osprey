"""Data-driven simulation engine for OSPREY mock connectors.

Provides :class:`SimulationEngine`, which loads a machine description
(``machine.json``) and serves channel reads/writes plus synthesized archiver
time-series to the mock control-system and archiver connectors.
"""

from osprey.simulation.engine import (
    DEFAULT_SCENARIO,
    SimReading,
    SimulationEngine,
    engine_from_connector_config,
)
from osprey.simulation.expressions import ExpressionError

__all__ = [
    "DEFAULT_SCENARIO",
    "ExpressionError",
    "SimReading",
    "SimulationEngine",
    "engine_from_connector_config",
]
