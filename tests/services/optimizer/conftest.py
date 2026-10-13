"""The engine suite needs the ``optimizer`` extra (xopt, torch, polars).

The main unit lanes install only ``dev``; there the engine tests are left out
of collection, and the environment tests beside them (pydantic and YAML only)
still run. The ``optimizer`` CI job installs the extra and runs everything.
"""

import importlib.util

# The directory itself, not a glob under it: its conftest imports xopt too.
collect_ignore = (
    [] if all(importlib.util.find_spec(m) for m in ("xopt", "torch", "polars")) else ["engine"]
)
