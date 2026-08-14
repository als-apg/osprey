"""Import-stable shim; holds no plan definitions.

No plan may be hardcoded here as a module-level dict:
`plan_loader.get_facility_plans()` is the sole plan registry, sourced by
scanning `plans_core/` (and any preset/facility/session directory layer) for
`PLAN_METADATA`/`build_plan` files — see `plan_loader.py`. The shipped tier
(`grid_scan`, `orm`) lives as files under `plans_core/`.

This module has no content of its own; it stays importable for callers that
`import ...plans` before reading the registry.
"""

from __future__ import annotations
