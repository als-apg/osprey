"""Loads scan plans from a layered directory catalog plus the legacy facility
plan-injection contract, and merges both into one trust-resolved plan set.

Two kinds of plan source, both scanned into the same fail-closed registry:

- **Directory layers** — ordered ``(directory, provenance)`` pairs, each
  scanned for ``.py`` files exposing ``PLAN_METADATA``/``build_plan``
  (optionally ``PARAMS``). Device-agnostic: these files never define
  ``get_devices()``; their plans resolve device names against the bridge's
  own device map at launch.
- **The legacy single-module contract** — a single ``.py`` file exposing
  ``PLANS: dict[str, PlanSpec]`` and ``get_devices()``, resolved from
  ``BLUESKY_PLAN_MODULE`` (env) or ``scan.plan_module`` (config.yml). This
  predates the layered model and is folded in as a one-entry ``facility``-tier
  layer — it is the only source of injected devices.

Layer sources and their provenance-tier mapping:

1. ``shipped`` — the in-image core dir bundled with this package
   (``plans_core/``, alongside this module). May not exist yet in a given
   install; an absent/empty directory is not an error.
2. ``preset`` — directories listed in ``scan.plan_dirs`` in config.yml.
   Config-shipped and versioned with a deployment's config bundle: lower
   operator trust than a per-instance runtime override.
3. ``facility`` — directories listed in ``BLUESKY_PLAN_DIRS`` (env,
   ``os.pathsep``-separated), set per bridge instance at launch. This mirrors
   the legacy contract's existing precedent that an env override outranks
   config (``BLUESKY_PLAN_MODULE`` wins over ``scan.plan_module``). The
   legacy single-module contract itself is also pinned to ``facility`` —
   it is scanned *first*, so a lower-trust directory layer (``shipped`` or
   ``preset``) can never silently reclaim a name it already owns; only an
   equal- or higher-trust directory layer can.

Trust order (ascending): ``shipped < preset < facility < session <
unreviewed``. A same-name collision across layers is resolved fail-closed:
a strictly-higher-trust incoming plan overrides (warns); a strictly-lower-
trust incoming plan is rejected outright (errors, keeps the existing
registration); an equal-trust collision lets the later-scanned definition win
(warns) — same-tier directories are all operator-controlled, so there is no
principled tie-breaker beyond scan order.

Deliberately free of bluesky/ophyd/tiled imports — this module only execs
plan files and reads pydantic metadata; only a loaded module itself needs
bluesky. That keeps `plan_loader.py` importable in any bridge process
regardless of whether the `bluesky-bridge` extra is installed (see
``plans.py``'s built-in registry, which does need bluesky and degrades
separately in ``app.py``'s `/plans` route).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .plan_metadata import parse_plan_metadata
from .plan_types import PlanSpec, Provenance

logger = logging.getLogger("osprey.services.bluesky_bridge.plan_loader")

# Set per bridge instance by the framework server definition (mirrors
# BLUESKY_BRIDGE_URL/BLUESKY_PROMOTE_TOKEN); wins outright over config.yml.
_MODULE_PATH_ENV = "BLUESKY_PLAN_MODULE"

# os.pathsep-separated list of directory-layer dirs, set per bridge instance.
_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"

# The in-image core plan directory shipped with this package (task 1.5
# populates it; scanned even if absent/empty).
_SHIPPED_PLANS_DIR = Path(__file__).parent / "plans_core"

_TRUST_ORDER: dict[Provenance, int] = {
    "shipped": 0,
    "preset": 1,
    "facility": 2,
    "session": 3,
    "unreviewed": 4,
}


class _EmptyParams(BaseModel):
    """Default parameter schema for a directory-layer plan file with no ``PARAMS``.

    Phase-2 session plans may not ship a schema; a permissive empty model
    keeps the loader forward-compatible rather than rejecting such a file.
    """


class PlanLoaderError(ValueError):
    """A directory-layer plan file fails its load contract (missing/uncallable
    ``build_plan``, malformed ``PARAMS``). Always caught and quarantined by
    the scanner — never escapes this module."""


@dataclass
class FacilityPlans:
    """The bridge's merged plans and devices, aggregated across every layer."""

    plans: dict[str, PlanSpec[Any]] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)


def _resolve_plan_module_path() -> str | None:
    """Resolve the legacy facility plan module's filesystem path.

    Resolution order:

    1. ``BLUESKY_PLAN_MODULE`` env var (a filesystem path) — set by the
       framework server definition per bridge instance; wins outright.
    2. ``scan.plan_module`` in config.yml (local/dev convenience).
    3. ``None`` — no legacy facility module is injected.
    """
    path = os.environ.get(_MODULE_PATH_ENV)
    if path:
        return path

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    value = config.get("scan", {}).get("plan_module")
    return str(value) if value else None


def _resolve_plan_dir_layers() -> list[tuple[Path, Provenance]]:
    """Ordered ``(directory, provenance)`` layers for the directory scan, ascending trust.

    See the module docstring for the full source-to-tier mapping rationale.
    """
    layers: list[tuple[Path, Provenance]] = [(_SHIPPED_PLANS_DIR, "shipped")]

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    preset_dirs = config.get("scan", {}).get("plan_dirs") or []
    if isinstance(preset_dirs, str):
        preset_dirs = [preset_dirs]
    layers.extend((Path(d), "preset") for d in preset_dirs)

    env_value = os.environ.get(_PLAN_DIRS_ENV)
    if env_value:
        layers.extend((Path(d), "facility") for d in env_value.split(os.pathsep) if d)

    return layers


def _load_module_from_path(path: Path) -> Any:
    """Import a `.py` file at ``path`` as a standalone module, by path (not name).

    Used for the legacy single-module contract, keyed by the file's stem —
    a colliding stem here is a genuine misconfiguration (there is only ever
    one legacy module per bridge instance), unlike directory-layer files
    (see `_load_plan_module_from_path`).
    """
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


def _load_plan_module_from_path(path: Path) -> Any:
    """Import a directory-layer `.py` file under a unique, full-path-derived
    `sys.modules` name.

    Unlike the legacy single-module loader (keyed by stem), directory layers
    commonly reuse stems across layers (e.g. a shipped and a facility-tier
    `orm.py`) — hashing the resolved absolute path into the synthetic name
    keeps every file's `sys.modules` entry distinct regardless of filename
    collisions. Registered before `exec_module` and popped on failure,
    mirroring `_load_module_from_path`.
    """
    abspath = path.resolve()
    name = f"_osprey_bridge_plan_{hashlib.sha1(str(abspath).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load plan file: {abspath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _iter_plan_files(directory: Path) -> Iterator[Path]:
    """Yield a layer directory's plan files, sorted for deterministic scan order.

    Tolerates a missing/non-directory path (not an error — see the module
    docstring). Skips `__init__.py` and any other dunder-named module.
    """
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.py")):
        stem = path.stem
        if stem.startswith("__") and stem.endswith("__"):
            continue
        yield path


def _register_plan(
    registry: dict[str, tuple[Provenance, str, PlanSpec[Any]]],
    name: str,
    provenance: Provenance,
    source: str,
    spec: PlanSpec[Any],
) -> None:
    """Register ``spec`` under ``name``, resolving a same-name collision by trust tier.

    Fail-closed: a lower-trust incoming plan never overrides a higher-trust
    registration — it is rejected outright and the existing registration
    stays. A strictly-higher-trust incoming plan overrides (warns). An
    equal-trust collision lets the later-scanned definition win (warns) —
    same-tier directories are all operator-controlled.
    """
    existing = registry.get(name)
    if existing is None:
        registry[name] = (provenance, source, spec)
        return

    existing_provenance, existing_source, _ = existing
    incoming_rank = _TRUST_ORDER[provenance]
    existing_rank = _TRUST_ORDER[existing_provenance]

    if incoming_rank > existing_rank:
        logger.warning(
            "plan_loader: higher-trust plan %r from %s (%s) overrides lower-trust "
            "registration from %s (%s)",
            name,
            source,
            provenance,
            existing_source,
            existing_provenance,
        )
        registry[name] = (provenance, source, spec)
    elif incoming_rank < existing_rank:
        logger.error(
            "plan_loader: rejecting lower-trust plan %r from %s (%s) — a higher-trust "
            "registration from %s (%s) already owns this name",
            name,
            source,
            provenance,
            existing_source,
            existing_provenance,
        )
    else:
        logger.warning(
            "plan_loader: plan %r redefined at equal trust (%s) — %s overrides %s",
            name,
            provenance,
            source,
            existing_source,
        )
        registry[name] = (provenance, source, spec)


def _load_plan_file(
    path: Path,
    provenance: Provenance,
    registry: dict[str, tuple[Provenance, str, PlanSpec[Any]]],
) -> None:
    """Load, validate, and register one directory-layer plan file.

    Never raises: any failure (syntax error, missing/invalid `PLAN_METADATA`,
    missing `build_plan`, malformed `PARAMS`) is logged as a warning and the
    file is quarantined — skipped, without aborting the rest of the scan.

    Catches `SystemExit` alongside `Exception` (it isn't an `Exception`
    subclass) so a plan file that calls `sys.exit()` at import time is
    quarantined like any other bad file rather than aborting the whole
    directory scan — today's shipped/preset/facility tiers are all
    operator-trusted, but this same scan path is where task 2.4's untrusted
    `session`/`unreviewed` tier will land, where one authored `sys.exit()`
    would otherwise be a sibling-plan discovery denial-of-service. Deliberately
    NOT widened to bare `BaseException`: a genuine `KeyboardInterrupt` or
    `GeneratorExit` must still propagate — an operator's Ctrl-C is not a
    plan-file failure.
    """
    try:
        module = _load_plan_module_from_path(path)
        meta = parse_plan_metadata(module, source=str(path))
        build_plan = getattr(module, "build_plan", None)
        if not callable(build_plan):
            raise PlanLoaderError(f"{path}: missing required build_plan callable")
        params_attr = getattr(module, "PARAMS", None)
        if params_attr is None:
            params_schema: type[BaseModel] = _EmptyParams
        elif isinstance(params_attr, type) and issubclass(params_attr, BaseModel):
            params_schema = params_attr
        else:
            raise PlanLoaderError(
                f"{path}: PARAMS must be a pydantic BaseModel subclass, got {params_attr!r}"
            )
        spec = PlanSpec(
            name=meta.name,
            plan=build_plan,
            schema=params_schema,
            description=meta.description,
            metadata=meta,
            provenance=provenance,
        )
    except (Exception, SystemExit) as exc:
        logger.warning("plan_loader: quarantining %s (%s): %s", path, provenance, exc)
        return
    _register_plan(registry, meta.name, provenance, str(path), spec)


def load_facility_plans(module_path: str | None = None) -> FacilityPlans:
    """Load the legacy facility ``PLANS``/``get_devices()`` from a config-pointed module path.

    ``module_path`` overrides resolution (mainly for tests); production
    callers leave it unset so it's resolved via env/config.yml (see
    ``_resolve_plan_module_path``). Returns an empty `FacilityPlans` when no
    path is configured — no legacy plans/devices are injected.

    A path that *is* configured but names a missing, unloadable module, or one
    missing the ``PLANS``/``get_devices`` contract, raises rather than
    silently falling back — a misconfigured facility injection should fail
    loudly, not masquerade as "no facility plans".

    Every returned plan's ``provenance`` is normalized to ``"facility"``
    regardless of what the module itself set (or left at its `PlanSpec`
    default) — provenance is loader-assigned, never self-declared.
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

    normalized_plans = {
        name: (spec if spec.provenance == "facility" else replace(spec, provenance="facility"))
        for name, spec in plans.items()
    }
    return FacilityPlans(plans=normalized_plans, devices=dict(devices))


def _warn_if_shadowing_builtins(facility_plans: dict[str, PlanSpec[Any]]) -> None:
    """Log once, at load time, if a legacy facility plan overrides a built-in of the same name.

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


def _load_all_layers(module_path: str | None) -> FacilityPlans:
    """Merge the legacy single-module contract with every directory layer.

    The legacy module (if configured) is registered *first*, as a one-entry
    ``facility``-tier layer — so a lower-trust directory layer (``shipped``,
    ``preset``) can never silently reclaim a name it already owns; only an
    equal- or higher-trust directory layer can (see `_register_plan`).
    Directory layers are then scanned in ascending trust order. Devices come
    only from the legacy module's `get_devices()` — directory layers are
    device-agnostic.
    """
    registry: dict[str, tuple[Provenance, str, PlanSpec[Any]]] = {}

    legacy = load_facility_plans(module_path)
    legacy_source = module_path or _resolve_plan_module_path() or "<unconfigured>"
    for name, spec in legacy.plans.items():
        _register_plan(registry, name, "facility", str(legacy_source), spec)

    for directory, provenance in _resolve_plan_dir_layers():
        for path in _iter_plan_files(directory):
            _load_plan_file(path, provenance, registry)

    plans = {name: spec for name, (_, _, spec) in registry.items()}
    return FacilityPlans(plans=plans, devices=dict(legacy.devices))


# ---------------------------------------------------------------------------
# Module-level singleton: loaded (and cached) once per bridge process, since
# device construction may be expensive (e.g. real EPICS connections).
# ---------------------------------------------------------------------------
_facility_plans: FacilityPlans | None = None


def get_facility_plans() -> FacilityPlans:
    """The bridge process's merged plans/devices, loading them on first use.

    Aggregates the legacy single-module contract with every directory layer
    (shipped/preset/facility) into one trust-resolved plan set — see
    `_load_all_layers`.
    """
    global _facility_plans
    if _facility_plans is None:
        _facility_plans = _load_all_layers(None)
    return _facility_plans


def reset_facility_plans() -> None:
    """Clear the cached `FacilityPlans` singleton (for testing)."""
    global _facility_plans
    _facility_plans = None
