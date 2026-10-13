"""Data-driven simulation engine for OSPREY mock connectors.

Provides :class:`SimulationEngine`, which loads a machine description
(``machine.json``) and serves channel reads/writes plus synthesized archiver
time-series to the mock control-system and archiver connectors, together with
the machine model and the deploy-time archive seed.
"""

from osprey_connectors.simulation._archiver_seed import (
    MANIFEST_ID,
    SeedKnobs,
    SeedState,
    compare_fingerprint,
    oldest_sample,
    seed_base,
    seed_fingerprint,
    tier_expiry,
)
from osprey_connectors.simulation._engine import (
    ACTIVE_SCENARIOS_FILENAME,
    SimReading,
    SimulationEngine,
    engine_from_connector_config,
    engine_serves,
    resolve_active_scenarios,
    resolve_simulation_file,
    resolve_state_dir,
)
from osprey_connectors.simulation._expressions import (
    ExpressionError,
)
from osprey_connectors.simulation._machine import (
    DEFAULT_SCENARIO,
    BpmErrorSpec,
    PlotSpec,
    Scenario,
    ScenarioLogEntry,
    load_narratives,
    parse_machine,
    read_machine_json,
)
from osprey_connectors.simulation._series import (
    anchored_instant,
    daily_occurrences,
)

__all__ = [
    "ACTIVE_SCENARIOS_FILENAME",
    "BpmErrorSpec",
    "DEFAULT_SCENARIO",
    "ExpressionError",
    "MANIFEST_ID",
    "PlotSpec",
    "Scenario",
    "ScenarioLogEntry",
    "SeedKnobs",
    "SeedState",
    "SimReading",
    "SimulationEngine",
    "anchored_instant",
    "compare_fingerprint",
    "daily_occurrences",
    "engine_from_connector_config",
    "engine_serves",
    "load_narratives",
    "oldest_sample",
    "parse_machine",
    "read_machine_json",
    "resolve_active_scenarios",
    "resolve_simulation_file",
    "resolve_state_dir",
    "seed_base",
    "seed_fingerprint",
    "tier_expiry",
]
