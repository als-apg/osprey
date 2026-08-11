"""Data-driven simulation engine for OSPREY mock connectors.

Provides :class:`SimulationEngine`, which loads a machine description
(``machine.json``) and serves channel reads/writes plus synthesized archiver
time-series to the mock control-system and archiver connectors.
"""

from osprey_connectors.simulation import (
    DEFAULT_SCENARIO,
    ExpressionError,
    SimReading,
    SimulationEngine,
    engine_from_connector_config,
    engine_serves,
)

__all__ = [
    "DEFAULT_SCENARIO",
    "ExpressionError",
    "SimReading",
    "SimulationEngine",
    "engine_from_connector_config",
    "engine_serves",
]
