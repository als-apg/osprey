"""Apply simulation scenarios: make telemetry and logbook live, deterministically.

:func:`apply_scenarios` is the one entry point that composes a set of
self-contained scenario bundles and makes everything live at once. It computes a
single apply-time anchor T0 and uses it for both the simulator state (so
``at_offset`` telemetry anchors against it) and logbook timestamp resolution, so
the narrative the agent searches always matches the telemetry it reads, against
one clock. For simulation-backed projects it purges and reseeds the ARIEL
logbook from the active scenarios' own entries.

Build never calls this (it must not require a running Postgres); seeding happens
on demand via ``osprey sim apply``.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.connectors.types import MOCK
from osprey.simulation.engine import SimulationEngine
from osprey.simulation.machine import DEFAULT_SCENARIO, parse_machine
from osprey.utils.config import get_facility_timezone, load_config
from osprey.utils.logger import get_logger
from osprey.utils.relative_time import resolve_relative_timestamp

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.simulation.machine import BpmErrorSpec, Scenario, ScenarioLogEntry

logger = get_logger("simulation_apply")


def resolve_simulation_file(config: dict, project_dir: Path) -> tuple[Path | None, str, str, str]:
    """Resolve the simulation-model file for the active control-system type.

    Looks up ``control_system.connector.<type>.simulation_file`` for the active
    ``control_system.type`` (defaulting to ``mock`` when unset). Non-mock types
    fall back to ``connector.mock.simulation_file`` when their own key is unset;
    for the mock type itself this fallback is a no-op (it's the same key it
    already tried), so mock resolution is unaffected by the fallback.

    Shared by :func:`apply_scenarios` and the ``sim`` CLI so the two call sites
    agree on exactly which config keys back a simulation-backed project.

    Returns:
        A 4-tuple ``(path, active_type, type_key, mock_key)``. ``path`` is the
        resolved file path (made absolute against ``project_dir`` if relative),
        or ``None`` if neither key had a value. ``type_key``/``mock_key`` are
        the dotted config paths that were tried, for error messages.
    """
    control_system = config.get("control_system", {})
    active_type = control_system.get("type", MOCK)
    connector = control_system.get("connector", {})

    type_key = f"control_system.connector.{active_type}.simulation_file"
    mock_key = "control_system.connector.mock.simulation_file"

    sim_file = connector.get(active_type, {}).get("simulation_file")
    if not sim_file and active_type != MOCK:
        sim_file = connector.get(MOCK, {}).get("simulation_file")

    if not sim_file:
        return None, active_type, type_key, mock_key

    machine_path = Path(sim_file)
    if not machine_path.is_absolute():
        machine_path = Path(project_dir) / machine_path
    return machine_path, active_type, type_key, mock_key


def _run_coro(make_coro: Callable[[], Coroutine]):
    """Run an async coroutine to completion from this sync function.

    ``apply_scenarios`` is a sync API (the CLI calls it directly), but it is also
    invoked from inside a running event loop (the async scenario e2e tests call
    it during setup). ``asyncio.run`` is illegal from a running loop, so when one
    is already active we run the coroutine in a fresh thread that has none. The
    thunk defers coroutine creation until we know which thread will run it (a
    coroutine must be created and awaited on the same loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())  # no loop in this thread — safe
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


@dataclass
class ApplyResult:
    """Outcome of :func:`apply_scenarios`."""

    active: tuple[str, ...]
    logbook_seeded: int
    purged: bool


def apply_scenarios(
    project_dir: Path | str,
    names: Sequence[str],
    *,
    seed_logbook: bool = True,
    now: datetime | None = None,
) -> ApplyResult:
    """Compose and activate scenarios for a built project; optionally seed its logbook.

    Args:
        project_dir: Root of the built project (holds ``config.yml`` and
            ``data/simulation/``).
        names: Scenario names to activate (``nominal`` is always implicit).
        seed_logbook: When True (and the project has an ``ariel`` config),
            purge and reseed the ARIEL logbook from the active scenarios'
            entries so the narrative matches the telemetry.
        now: Apply-time anchor T0 (injectable for tests). Defaults to the
            current time in the facility timezone, so seeded logbook entries
            resolve their time-of-day on the same clock as the telemetry.

    Returns:
        :class:`ApplyResult` with the resolved active set and seed/purge status.

    Raises:
        ValueError: If the project is not simulation-backed, a scenario name is
            unknown, or the requested set does not compose (channel collision).
    """
    project_dir = Path(project_dir)
    config = load_config(str(project_dir / "config.yml"))

    machine_path, active_type, type_key, mock_key = resolve_simulation_file(config, project_dir)
    if machine_path is None:
        if active_type == MOCK:
            raise ValueError(
                f"Project {project_dir} has no mock 'simulation_file' configured; "
                f"`sim apply` only applies to simulation-backed projects (guards a real DB)."
            )
        raise ValueError(
            f"Project {project_dir} has no simulation_file configured for "
            f"control_system.type '{active_type}' (tried {type_key} and {mock_key}); "
            f"`sim apply` only applies to simulation-backed projects (guards a real DB)."
        )
    engine = SimulationEngine.from_file(machine_path)

    # Default anchor in the FACILITY zone (not UTC): the anchor's tzinfo is the
    # zone each seeded logbook entry's relative time-of-day resolves into, and it
    # must match where the simulation engine places the telemetry it narrates
    # (daily ``at_time`` events are facility-local). A UTC default silently shifts
    # the narrative hours away from its archiver evidence on a non-UTC facility.
    t0 = now or datetime.now(get_facility_timezone())
    # set_active_scenarios validates composition and raises on collisions/unknowns.
    active = engine.set_active_scenarios(names, anchor=t0)
    logger.info(f"Activated scenarios {list(active)!r} with anchor {t0.isoformat()}")

    seeded = 0
    purged = False
    if seed_logbook:
        ariel_config = config.get("ariel")
        if ariel_config:
            entries = [_to_enhanced_entry(e, t0) for e in engine.active_logbook()]
            seeded, purged = _run_coro(lambda: _seed_logbook(ariel_config, entries))
            logger.info(f"Seeded {seeded} logbook entries (logbook purged and reseeded)")
        else:
            logger.info("No 'ariel' config in project; skipped logbook seeding")

    return ApplyResult(active=active, logbook_seeded=seeded, purged=purged)


# BpmErrorSpec field -> VA_BPM_ERRORS sub-field(s) it fans out to, at the
# entrypoint's per-transverse-plane granularity (see
# `virtual_accelerator/entrypoint.py::_BPM_ERROR_FIELD_BOUNDS`). A scenario
# author states one isotropic value per BPM; the render step applies it to
# both planes. `roll` has no axis split on either side, so it maps 1:1.
_BPM_ERROR_AXIS_FIELDS: dict[str, tuple[str, ...]] = {
    "offset": ("offset_x", "offset_y"),
    "gain": ("gain_x", "gain_y"),
    "polarity": ("polarity_x", "polarity_y"),
    "roll": ("roll",),
    "noise": ("noise_x", "noise_y"),
}
# Identity value per BpmErrorSpec field -- mirrors PhysicsBridge's own
# `_IDENTITY_BPM_ERROR` defaults, so an unset field never renders.
_BPM_ERROR_IDENTITY: dict[str, float] = {
    "offset": 0.0,
    "gain": 1.0,
    "polarity": 1,
    "roll": 0.0,
    "noise": 0.0,
}
# Emission order within one device's field list, matching the entrypoint's own
# `_BPM_ERROR_FIELD_BOUNDS` ordering -- deterministic, readable .env output.
_BPM_ERROR_FIELD_ORDER = (
    "offset_x",
    "offset_y",
    "gain_x",
    "gain_y",
    "polarity_x",
    "polarity_y",
    "roll",
    "noise_x",
    "noise_y",
)


def render_scenario_physics_env(
    project_dir: Path | str,
    names: Sequence[str],
    *,
    env_path: Path | None = None,
) -> dict[str, str]:
    """Resolve the active scenario's ``physics`` fault into VA_* env vars in ``.env``.

    The deploy-time counterpart to :func:`apply_scenarios`'s telemetry/logbook
    half (FR5). A scenario's optional ``physics`` block (see
    :class:`~osprey.simulation.machine.PhysicsFault`) is deploy-time-only -- a
    physics fault applies once at VA container boot, and hot-swapping it needs
    a restart, unlike ``overrides``/``archiver`` -- so it is rendered here into
    the project ``.env`` as ``VA_BPM_ERRORS``/
    ``VA_CORR_GAIN``, the exact env vars
    ``virtual_accelerator/entrypoint.py`` parses, rather than applied live.
    Call this before ``deploy up`` so the VA container picks up the rendered
    values at boot.

    Args:
        project_dir: Root of the built project (holds ``config.yml`` and
            ``data/simulation/``).
        names: Scenario names to activate (``nominal`` is always implicit),
            resolved the same nominal-first, deduped way
            :meth:`~osprey.simulation.engine.SimulationEngine.set_active_scenarios`
            resolves them.
        env_path: ``.env`` path to write into (defaults to
            ``project_dir/.env``, injectable for tests).

    Returns:
        The ``VA_*`` vars written. Empty if no active scenario declares a
        ``physics`` block -- backward compatible: a project whose ``.env``
        never had a rendered fault gets no ``.env`` write at all. Every call
        reconciles the full ``VA_BPM_ERRORS``/
        ``VA_CORR_GAIN`` block to exactly the active set, so switching to a
        scenario with no (or a different) ``physics`` block clears a prior
        render's stale values rather than leaving them to leak into the next
        VA boot.

    Raises:
        ValueError: If the project is not simulation-backed (mirrors
            :func:`apply_scenarios`), a requested scenario name is unknown, or
            two active scenarios declare a physics fault on the same device.
    """
    project_dir = Path(project_dir)
    config = load_config(str(project_dir / "config.yml"))

    machine_path, active_type, type_key, mock_key = resolve_simulation_file(config, project_dir)
    if machine_path is None:
        if active_type == MOCK:
            raise ValueError(
                f"Project {project_dir} has no mock 'simulation_file' configured; "
                f"physics-fault rendering only applies to simulation-backed projects."
            )
        raise ValueError(
            f"Project {project_dir} has no simulation_file configured for "
            f"control_system.type '{active_type}' (tried {type_key} and {mock_key}); "
            f"physics-fault rendering only applies to simulation-backed projects."
        )
    with open(machine_path) as f:
        machine = json.load(f)
    model = parse_machine(machine, machine_path)

    resolved: list[str] = [DEFAULT_SCENARIO]
    for name in names:
        if name != DEFAULT_SCENARIO and name not in resolved:
            resolved.append(name)
    unknown = [n for n in resolved if n not in model.scenarios]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown!r}; available: {sorted(model.scenarios)}")

    rendered = _render_physics_vars(model.scenarios, resolved)
    if env_path is None:
        env_path = project_dir / ".env"
    _write_physics_env(env_path, rendered)
    return rendered


def _render_physics_vars(scenarios: dict[str, Scenario], active: list[str]) -> dict[str, str]:
    """Merge the active scenarios' ``physics`` blocks and render them to VA_* strings.

    Active scenarios must declare *disjoint* devices per physics field,
    mirroring ``SimulationEngine.validate_composition``'s disjointness rule
    for ``overrides``/``archiver`` -- a device faulted by two active scenarios
    at once would compose order-dependently and silently wrong.
    """
    corrector_gain: dict[str, float] = {}
    bpm_errors: dict[str, BpmErrorSpec] = {}
    owner: dict[tuple[str, str], str] = {}  # (field, device) -> owning scenario name

    def claim(field: str, device: str, name: str) -> None:
        key = (field, device)
        prior = owner.get(key)
        if prior is not None and prior != name:
            raise ValueError(
                f"physics.{field}[{device!r}] is declared by both {prior!r} and {name!r}; "
                f"active scenarios must declare disjoint physics-fault devices"
            )
        owner[key] = name

    for name in active:
        physics = scenarios[name].physics
        if physics is None:
            continue
        for device, factor in physics.corrector_gain.items():
            claim("corrector_gain", device, name)
            corrector_gain[device] = factor
        for device, spec in physics.bpm_errors.items():
            claim("bpm_errors", device, name)
            bpm_errors[device] = spec

    # Guard on the rendered string being non-empty, not the source dict: an
    # all-identity BpmErrorSpec (every field at its default) renders "" even
    # though its device is present in `bpm_errors`, and that empty string must
    # not become a `VA_BPM_ERRORS=` line -- "empty" must mean "nothing to
    # render" all the way through, matching the docstring's "empty if no
    # active scenario declares a physics block" contract.
    rendered: dict[str, str] = {}
    corrector_gain_str = _render_device_value_map(corrector_gain)
    if corrector_gain_str:
        rendered["VA_CORR_GAIN"] = corrector_gain_str
    bpm_errors_str = _render_bpm_errors(bpm_errors)
    if bpm_errors_str:
        rendered["VA_BPM_ERRORS"] = bpm_errors_str
    return rendered


def _render_device_value_map(values: dict[str, float]) -> str:
    """Render ``{device: value}`` as the `VA_STUCK_SETPOINTS`-shaped ``"DEVICE=value,..."``."""
    return ",".join(f"{device}={value}" for device, value in sorted(values.items()))


def _render_bpm_errors(specs: dict[str, BpmErrorSpec]) -> str:
    """Render ``{device: BpmErrorSpec}`` as ``"DEVICE:field=value[,field=value...];..."``.

    Only non-identity fields are emitted, mirroring ``PhysicsBridge``'s own
    sparse-override idiom ("fault dicts... only need to name the fields they
    perturb"). An isotropic scenario-authored value fans out to both
    transverse-plane fields the entrypoint parses (``offset`` ->
    ``offset_x``/``offset_y``, etc.); ``roll`` has no axis split on either side.
    """
    parts: list[str] = []
    for device, spec in sorted(specs.items()):
        fields = _bpm_error_env_fields(spec)
        if not fields:
            continue
        field_str = ",".join(
            f"{key}={fields[key]}" for key in _BPM_ERROR_FIELD_ORDER if key in fields
        )
        parts.append(f"{device}:{field_str}")
    return ";".join(parts)


def _bpm_error_env_fields(spec: BpmErrorSpec) -> dict[str, float]:
    """Expand one BPM's isotropic error spec into its non-identity env fields."""
    fields: dict[str, float] = {}
    for attr, axis_fields in _BPM_ERROR_AXIS_FIELDS.items():
        value = getattr(spec, attr)
        if value == _BPM_ERROR_IDENTITY[attr]:
            continue
        for env_field in axis_fields:
            fields[env_field] = float(value)
    return fields


# The full set of keys `_write_physics_env` owns -- reconciled on every call
# (set if rendered, removed if not), never left stale from a prior scenario.
_PHYSICS_ENV_VARS = ("VA_BPM_ERRORS", "VA_CORR_GAIN")


def _write_physics_env(env_path: Path, rendered: dict[str, str]) -> None:
    """Reconcile the physics-fault block in ``.env`` to exactly ``rendered``.

    Unlike ``_ensure_service_tokens``'s append-only idiom (an existing token is
    a deliberate value, never overwritten), a scenario's physics vars ARE the
    single source of truth for "what physics fault is active": this function
    owns all of ``_PHYSICS_ENV_VARS`` unconditionally, replacing an existing
    line for a key ``rendered`` sets and removing one it doesn't, so switching
    the active scenario never leaves a stale fault from a previous scenario
    alongside (or instead of) the new one. Every other line (comments,
    unrelated vars) is left untouched. A no-op (no write at all) when there is
    nothing to render and no ``.env`` yet exists to clean up.
    """
    if not rendered and not env_path.is_file():
        return

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in _PHYSICS_ENV_VARS:
                continue  # dropped here; re-added below if still active
        kept.append(line)
    while kept and kept[-1] == "":
        kept.pop()

    if rendered:
        if kept:
            kept.append("")
        kept.append("# Scenario physics fault (osprey sim apply / deploy up)")
        kept.extend(f"{k}={rendered[k]}" for k in _PHYSICS_ENV_VARS if k in rendered)

    text = "\n".join(kept) + ("\n" if kept else "")
    env_path.write_text(text, encoding="utf-8")
    os.chmod(env_path, 0o600)


def _to_enhanced_entry(entry: ScenarioLogEntry, now: datetime) -> EnhancedLogbookEntry:
    """Convert a bundle :class:`ScenarioLogEntry` to an ``EnhancedLogbookEntry``.

    Mirrors ``GenericJSONAdapter._convert_entry`` field mapping so seeded entries
    are indistinguishable from ingested ones: ``raw_text`` is title + body, and
    title/tags/categories/loto_tag plus any ``extra`` ride in ``metadata``.
    """
    timestamp = resolve_relative_timestamp(entry.when, now)
    if entry.title and entry.text:
        raw_text = f"{entry.title}\n\n{entry.text}"
    else:
        raw_text = entry.title or entry.text

    metadata: dict = {}
    if entry.title:
        metadata["title"] = entry.title
    if entry.tags:
        metadata["tags"] = list(entry.tags)
    if entry.categories:
        metadata["categories"] = list(entry.categories)
    if entry.loto_tag:
        metadata["loto_tag"] = entry.loto_tag
    metadata.update(entry.extra)

    return {
        "entry_id": entry.entry_id,
        "source_system": "Simulation",
        "timestamp": timestamp,
        "author": entry.author,
        "raw_text": raw_text,
        "attachments": [],
        "metadata": metadata,
        "created_at": now,
        "updated_at": now,
    }


async def _seed_logbook(
    ariel_config: dict, entries: list[EnhancedLogbookEntry]
) -> tuple[int, bool]:
    """Migrate, purge, then seed the ARIEL logbook. Returns (seeded, purged).

    Migrate first so the schema exists before the purge truncates it; purge so
    the seeded narrative is the only narrative (no stale incident bleed-through).
    """
    from osprey.services.ariel_search.cli_operations import (
        execute_purge,
        run_migrate,
        seed_logbook_entries,
    )

    await run_migrate(ariel_config)
    await execute_purge(ariel_config, embeddings_only=False)
    seeded = await seed_logbook_entries(ariel_config, entries)
    return seeded, True
