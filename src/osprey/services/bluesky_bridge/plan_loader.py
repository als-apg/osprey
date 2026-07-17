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
  ``BLUESKY_PLAN_MODULE`` (env) or ``bluesky.plan_module`` (config.yml). This
  predates the layered model and is folded in as a one-entry ``facility``-tier
  layer — it is the only source of injected devices.

Layer sources and their provenance-tier mapping:

1. ``shipped`` — the in-image core dir bundled with this package
   (``plans_core/``, alongside this module). May not exist yet in a given
   install; an absent/empty directory is not an error.
2. ``preset`` — directories listed in ``bluesky.plan_dirs`` in config.yml.
   Config-shipped and versioned with a deployment's config bundle: lower
   operator trust than a per-instance runtime override.
3. ``facility`` — directories listed in ``BLUESKY_PLAN_DIRS`` (env,
   ``os.pathsep``-separated), set per bridge instance at launch. This mirrors
   the legacy contract's existing precedent that an env override outranks
   config (``BLUESKY_PLAN_MODULE`` wins over ``bluesky.plan_module``). The
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

4. ``session`` — the bridge-owned, writable directory resolved by
   ``session_dir.resolve_session_plan_dir()``. Unlike every layer above, this
   one is agent-authored, not operator-supplied — so a ``session`` (or a
   future ``unreviewed``) file is subject to an additional LOAD-TIME gate
   (``_load_plan_file``) before it is ever ``exec_module``'d: its current
   on-disk content is hashed with ``plan_validation.hash_plan_body`` and
   checked against ``validation_record.validation_records`` for a passing
   record. No record, no exec — the file is skipped like any other
   quarantined file, never registered, never launchable. This is the
   feature's primary enforcement point: it runs on every scan (see
   ``get_facility_plans``), so an edit that invalidates a previously-passing
   file's hash re-quarantines it on the very next scan. Higher-trust tiers
   carry no such gate — they are exec'd on discovery unconditionally, as
   they always have been.

Deliberately free of bluesky/ophyd/tiled imports — this module only execs
plan files and reads pydantic metadata; only a loaded module itself needs
bluesky. That keeps `plan_loader.py` importable in any bridge process
regardless of whether the `bluesky-bridge` extra is installed.
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
from .plan_validation import hash_plan_body
from .session_dir import resolve_session_plan_dir
from .validation_record import validation_records

logger = logging.getLogger("osprey.services.bluesky_bridge.plan_loader")

# Set per bridge instance by the framework server definition (mirrors
# BLUESKY_BRIDGE_URL/BLUESKY_LAUNCH_TOKEN); wins outright over config.yml.
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
    2. ``bluesky.plan_module`` in config.yml (local/dev convenience).
    3. ``None`` — no legacy facility module is injected.
    """
    path = os.environ.get(_MODULE_PATH_ENV)
    if path:
        return path

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    value = config.get("bluesky", {}).get("plan_module")
    return str(value) if value else None


def _resolve_plan_dir_layers() -> list[tuple[Path, Provenance]]:
    """Ordered ``(directory, provenance)`` layers for the directory scan, ascending trust.

    See the module docstring for the full source-to-tier mapping rationale.
    The ``session`` layer is appended last (highest trust so far — see
    ``_TRUST_ORDER``) and resolves to the bridge-owned, writable directory
    from ``session_dir.resolve_session_plan_dir()``; unlike every layer
    above it, files landing there are gated at load time (see
    ``_load_plan_file``) rather than trusted on discovery.
    """
    layers: list[tuple[Path, Provenance]] = [(_SHIPPED_PLANS_DIR, "shipped")]

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    preset_dirs = config.get("bluesky", {}).get("plan_dirs") or []
    if isinstance(preset_dirs, str):
        preset_dirs = [preset_dirs]
    layers.extend((Path(d), "preset") for d in preset_dirs)

    env_value = os.environ.get(_PLAN_DIRS_ENV)
    if env_value:
        layers.extend((Path(d), "facility") for d in env_value.split(os.pathsep) if d)

    layers.append((resolve_session_plan_dir(), "session"))

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
    operator-trusted, but this same scan path is where the untrusted
    `session`/`unreviewed` tier lands, where one authored `sys.exit()`
    would otherwise be a sibling-plan discovery denial-of-service. Deliberately
    NOT widened to bare `BaseException`: a genuine `KeyboardInterrupt` or
    `GeneratorExit` must still propagate — an operator's Ctrl-C is not a
    plan-file failure.

    For `provenance in ("session", "unreviewed")` only, a LOAD-TIME gate runs
    first: the file's current content is hashed with
    `plan_validation.hash_plan_body` and checked against
    `validation_record.validation_records` for a passing record. No record,
    no `exec_module` — the file is skipped exactly like a quarantined file
    (never registered, never launchable). Higher-trust tiers (`shipped`,
    `preset`, `facility`) carry no such gate; they are `exec_module`'d on
    discovery unconditionally, as they always have been. Gating strictly on
    provenance (not on "no metadata"/"no record" alone) matters: built-ins and
    the shipped exemplars carry no validation record either, and a broader
    gate would wrongly quarantine them too.
    """
    try:
        if provenance in ("session", "unreviewed"):
            content = path.read_text(encoding="utf-8")
            content_hash = hash_plan_body(content)
            if not validation_records.has_passing_record(content_hash):
                logger.info(
                    "plan_loader: skipping unvalidated %s-tier plan file %s "
                    "(content hash %s has no passing validation record)",
                    provenance,
                    path,
                    content_hash,
                )
                return
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

    normalized_plans = {
        name: (spec if spec.provenance == "facility" else replace(spec, provenance="facility"))
        for name, spec in plans.items()
    }
    return FacilityPlans(plans=normalized_plans, devices=dict(devices))


@dataclass
class _StartupLayers:
    """The registry and devices built from every *startup* layer: the legacy
    single-module contract plus the `shipped`/`preset`/`facility` directory
    layers. Cached once per process (see `_startup_layers`) since device
    construction may be expensive (e.g. real EPICS connections) — unlike the
    `session` layer, these are all operator-supplied and don't change without
    a bridge restart, so there is no correctness reason to re-scan them."""

    registry: dict[str, tuple[Provenance, str, PlanSpec[Any]]] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)


def _load_startup_layers(module_path: str | None) -> _StartupLayers:
    """Merge the legacy single-module contract with every startup directory layer.

    The legacy module (if configured) is registered *first*, as a one-entry
    ``facility``-tier layer — so a lower-trust directory layer (``shipped``,
    ``preset``) can never silently reclaim a name it already owns; only an
    equal- or higher-trust directory layer can (see `_register_plan`).
    Directory layers are then scanned in ascending trust order, *excluding*
    the ``session`` layer — that one is deliberately left out of this
    cached, once-per-process build and re-scanned fresh on every
    `get_facility_plans()` call instead (see `_session_layer_signature`).
    Devices come only from the legacy module's `get_devices()` — directory
    layers are device-agnostic.
    """
    registry: dict[str, tuple[Provenance, str, PlanSpec[Any]]] = {}

    legacy = load_facility_plans(module_path)
    legacy_source = module_path or _resolve_plan_module_path() or "<unconfigured>"
    for name, spec in legacy.plans.items():
        _register_plan(registry, name, "facility", str(legacy_source), spec)

    for directory, provenance in _resolve_plan_dir_layers():
        if provenance == "session":
            continue
        for path in _iter_plan_files(directory):
            _load_plan_file(path, provenance, registry)

    return _StartupLayers(registry=registry, devices=dict(legacy.devices))


# ---------------------------------------------------------------------------
# Module-level caches. `_startup_layers` is built once per process (see
# `_StartupLayers`) — the expensive part (e.g. real device construction).
# `_merged_plans` caches only the *previous return value*: every call still
# re-scans and re-gates the session layer from scratch (see
# `get_facility_plans`), but when that fresh scan produces a result equal to
# what was last returned (the common case — an unauthored or unchanged
# session directory), the old object is handed back instead of a new one, so
# repeat callers that compare by identity (or just want a stable reference)
# see one. A signature/mtime-based skip was deliberately rejected: a file's
# mtime doesn't change when a *validation record* is added after the fact
# (task 2.3's validate route only touches `validation_record.py`, never the
# file), so caching on file staleness alone would keep serving a stale
# rejection after a plan actually became valid — exactly the live-authoring
# case this layer exists for. Re-gating on every call is the correctness
# requirement; the equality check below is purely a reference-stability nicety.
# ---------------------------------------------------------------------------
_startup_layers: _StartupLayers | None = None
_merged_plans: FacilityPlans | None = None


def get_facility_plans() -> FacilityPlans:
    """The bridge process's merged plans/devices.

    Aggregates the legacy single-module contract with every directory layer
    (`shipped`/`preset`/`facility`/`session`) into one trust-resolved plan
    set. The startup layers are loaded once and cached (see
    `_load_startup_layers`) — but the `session` layer is a live authoring
    surface (task 2.3's `POST /plans/session` writes into it between
    requests, and its validation status can change without the file itself
    changing), so it is fully re-scanned and re-gated (see `_load_plan_file`)
    on *every* call, merged fresh over the cached startup registry. A newly
    written and validated session plan therefore appears on the very next
    call, with no bridge restart and no explicit cache invalidation.
    """
    global _startup_layers, _merged_plans

    if _startup_layers is None:
        _startup_layers = _load_startup_layers(None)

    registry = dict(_startup_layers.registry)
    for path in _iter_plan_files(resolve_session_plan_dir()):
        _load_plan_file(path, "session", registry)

    plans = {name: spec for name, (_, _, spec) in registry.items()}
    merged = FacilityPlans(plans=plans, devices=dict(_startup_layers.devices))

    if _merged_plans is not None and merged == _merged_plans:
        return _merged_plans
    _merged_plans = merged
    return _merged_plans


def reset_facility_plans() -> None:
    """Clear every cached layer (startup and last-merged result) — for testing."""
    global _startup_layers, _merged_plans
    _startup_layers = None
    _merged_plans = None
