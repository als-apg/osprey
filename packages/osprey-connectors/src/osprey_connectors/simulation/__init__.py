"""The simulated machine behind OSPREY's mock connectors.

The package root holds only what every reader needs without loading a model:
value coercion and the active-scenario state helpers. It imports neither
numpy nor lume; the engine, machine and expression names are imported from
their own modules (:mod:`.engine`, :mod:`.machine`, :mod:`.expressions`).
"""

from osprey_connectors.simulation.state import (
    ACTIVE_SCENARIOS_FILENAME,
    OVERLAP_EVENT,
    Overlap,
    format_overlap_record,
    overlap_record,
    resolve_active_scenarios,
    validate_composition,
)
from osprey_connectors.simulation.values import coerce

__all__ = [
    "ACTIVE_SCENARIOS_FILENAME",
    "OVERLAP_EVENT",
    "Overlap",
    "coerce",
    "format_overlap_record",
    "overlap_record",
    "resolve_active_scenarios",
    "validate_composition",
]
