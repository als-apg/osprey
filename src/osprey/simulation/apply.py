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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.connectors.types import MOCK
from osprey.simulation.engine import SimulationEngine
from osprey.utils.config import get_facility_timezone, load_config
from osprey.utils.logger import get_logger
from osprey.utils.relative_time import resolve_relative_timestamp

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.simulation.machine import ScenarioLogEntry

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
