"""Retained as an import-stable shim after the single-registry migration.

The v1 hardcoded plan set (`count`/`scan`/`grid_scan`/`orm`) that used to live
here as a module-level dict has been deleted: `plan_loader.get_facility_plans()`
is now the sole plan registry, sourced by scanning `plans_core/` (and any
preset/facility/session directory layer) for `PLAN_METADATA`/`build_plan`
files — see `plan_loader.py`. `grid_scan` and `orm` now live as shipped-tier
files under `plans_core/`; `count`/`scan` were dropped rather than ported.

This module has no remaining content of its own; it stays importable because
some callers still `import ...plans` before checking whether it still
carries the (now absent) hardcoded plan dict.
"""

from __future__ import annotations
