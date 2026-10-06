"""The files of the simulator view the package reads.

The module imports only the standard library, so a reader that must not load
a model takes the names from here.
"""

from __future__ import annotations

__all__ = [
    "ADDRESSES_FILE",
    "SCENARIOS_FILE",
    "SEEDS_FILE",
    "SERVED_MODELS_FILE",
    "VARIABLES_FILE",
]

#: The files of the simulator view the composite reads.
SERVED_MODELS_FILE = "served_models.json"
ADDRESSES_FILE = "addresses.json"
VARIABLES_FILE = "variables.json"
SEEDS_FILE = "seeds.json"
SCENARIOS_FILE = "scenarios.json"
