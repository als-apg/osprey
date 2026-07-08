"""The facility plan-injection contract: loads a facility's plans and devices
from a config-pointed module path, external to this package.

A facility plan module is any ``.py`` file exposing:

- ``PLANS: dict[str, PlanSpec]`` — additional (or overriding) scan plans.
- ``get_devices() -> dict[str, Any]`` — builds and returns the device mapping
  those plans' ``PlanSpec.plan`` callables resolve device names against.

This module is deliberately free of bluesky/ophyd/tiled imports — it loads
arbitrary facility modules *by path* (``importlib.util``, not a dotted import,
since a facility module typically lives outside this installed package), and
only the loaded module itself needs bluesky. That keeps `plan_loader.py`
importable in any bridge process regardless of whether the `bluesky-bridge`
extra is installed, so facility-injected plans and devices work even in a
bluesky-less test/contract environment (see ``plans.py``'s built-in registry,
which does need bluesky and degrades separately in ``app.py``'s `/plans`
route).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan_types import PlanSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.plan_loader")

# Set per bridge instance by the framework server definition (mirrors
# BLUESKY_BRIDGE_URL/BLUESKY_PROMOTE_TOKEN); wins outright over config.yml.
_MODULE_PATH_ENV = "BLUESKY_PLAN_MODULE"


@dataclass
class FacilityPlans:
    """A facility's injected plans and devices, as loaded from its plan module."""

    plans: dict[str, PlanSpec[Any]] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)


def _resolve_plan_module_path() -> str | None:
    """Resolve the facility plan module's filesystem path.

    Resolution order:

    1. ``BLUESKY_PLAN_MODULE`` env var (a filesystem path) — set by the
       framework server definition per bridge instance; wins outright.
    2. ``scan.plan_module`` in config.yml (local/dev convenience).
    3. ``None`` — no facility plans are injected; the bridge serves built-ins
       only.
    """
    path = os.environ.get(_MODULE_PATH_ENV)
    if path:
        return path

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    value = config.get("scan", {}).get("plan_module")
    return str(value) if value else None


def _load_module_from_path(path: Path) -> Any:
    """Import a `.py` file at ``path`` as a standalone module, by path (not name)."""
    if not path.is_file():
        raise FileNotFoundError(f"facility plan module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_bluesky_bridge_facility_plans_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load facility plan module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module can reference itself during import
    # (dataclasses, pickling, submodule imports all resolve the module by name
    # via sys.modules). Remove it again if exec fails, so a half-initialized
    # module never lingers to shadow a later, fixed load of the same path.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def load_facility_plans(module_path: str | None = None) -> FacilityPlans:
    """Load a facility's ``PLANS``/``get_devices()`` from a config-pointed module path.

    ``module_path`` overrides resolution (mainly for tests); production
    callers leave it unset so it's resolved via env/config.yml (see
    ``_resolve_plan_module_path``). Returns an empty `FacilityPlans` when no
    path is configured — the bridge then serves only the built-in plan set.

    A path that *is* configured but names a missing, unloadable module, or one
    missing the ``PLANS``/``get_devices`` contract, raises rather than
    silently falling back — a misconfigured facility injection should fail
    loudly, not masquerade as "no facility plans".
    """
    path_str = module_path if module_path is not None else _resolve_plan_module_path()
    if not path_str:
        return FacilityPlans()

    module = _load_module_from_path(Path(path_str))

    plans = getattr(module, "PLANS", None)
    if plans is None:
        raise AttributeError(f"facility plan module {path_str!r} does not define PLANS")
    get_devices = getattr(module, "get_devices", None)
    if get_devices is None:
        raise AttributeError(f"facility plan module {path_str!r} does not define get_devices()")

    devices = get_devices()
    logger.info(
        "plan_loader: loaded %d facility plan(s) and %d device(s) from %s",
        len(plans),
        len(devices),
        path_str,
    )
    _warn_if_shadowing_builtins(plans)
    return FacilityPlans(plans=dict(plans), devices=dict(devices))


def _warn_if_shadowing_builtins(facility_plans: dict[str, PlanSpec[Any]]) -> None:
    """Log once, at load time, if a facility plan overrides a built-in of the same name.

    Silent shadowing here would be a surprising way for an operator to lose a
    built-in plan (e.g. `count`) to a same-named facility plan without any
    trace in the logs. Guarded/lazy import of `plans.py` (which needs
    bluesky) so this check itself never forces `plan_loader.py` to depend on
    bluesky — absent bluesky, there's no built-in set to shadow anyway.
    """
    try:
        from .plans import BUILTIN_PLANS
    except ImportError:
        return
    shadowed = sorted(set(BUILTIN_PLANS) & set(facility_plans))
    if shadowed:
        logger.warning(
            "plan_loader: facility plan(s) %s override built-in plan(s) of the same name",
            shadowed,
        )


# ---------------------------------------------------------------------------
# Module-level singleton: loaded (and cached) once per bridge process, since
# device construction may be expensive (e.g. real EPICS connections).
# ---------------------------------------------------------------------------
_facility_plans: FacilityPlans | None = None


def get_facility_plans() -> FacilityPlans:
    """The bridge process's facility-injected plans/devices, loading them on first use."""
    global _facility_plans
    if _facility_plans is None:
        _facility_plans = load_facility_plans()
    return _facility_plans


def reset_facility_plans() -> None:
    """Clear the cached `FacilityPlans` singleton (for testing)."""
    global _facility_plans
    _facility_plans = None
