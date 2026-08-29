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
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.connectors.types import MOCK
from osprey.port_layout import default_port, resolve_port_base
from osprey.simulation.engine import (
    ACTIVE_SCENARIO_FILENAME,
    ACTIVE_SCENARIOS_FILENAME,
    SimulationEngine,
    resolve_active_scenarios,
    resolve_state_dir,
)
from osprey.simulation.machine import parse_machine
from osprey.utils.config import get_facility_timezone, load_config
from osprey.utils.logger import get_logger
from osprey.utils.relative_time import resolve_relative_timestamp

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
    from zoneinfo import ZoneInfo

    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.simulation.archiver_seed import SeedKnobs
    from osprey.simulation.machine import BpmErrorSpec, Scenario, ScenarioLogEntry

logger = get_logger("simulation_apply")


def _config_file(project_dir: Path) -> Path:
    """The ``config.yml`` belonging to *project_dir*, wherever the split put it.

    A deployment repo keeps its render under ``build/``, so the config sits at
    ``<repo>/build/config.yml`` while everything else this module resolves — the
    ``data/simulation/`` model, the mutable state under ``var/agent_data/`` —
    anchors at the repo root. A container's project directory *is* the render
    and holds ``config.yml`` at its own root. One directory still identifies the
    deployment either way; only the config moved, so only the config lookup
    needs to know.
    """
    from osprey.utils.workspace import rendered_config_path

    rendered = rendered_config_path(project_dir)
    return rendered if rendered.is_file() else project_dir / "config.yml"


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


def _require_simulation_file(config: dict, project_dir: Path, scope: str) -> Path:
    """Resolve the simulation-model file, or raise the not-simulation-backed error.

    Both entry points into a built project -- :func:`apply_scenarios` and
    :func:`compute_scenario_physics_env` -- refuse the same way on the same two
    branches (the mock type, whose one key is simply unset, versus a non-mock
    type, whose own key and the mock fallback were both tried). ``scope`` is the
    trailing clause naming what is refusing, so each caller keeps its own wording.
    """
    machine_path, active_type, type_key, mock_key = resolve_simulation_file(config, project_dir)
    if machine_path is None:
        if active_type == MOCK:
            raise ValueError(
                f"Project {project_dir} has no mock 'simulation_file' configured; {scope}"
            )
        raise ValueError(
            f"Project {project_dir} has no simulation_file configured for "
            f"control_system.type '{active_type}' (tried {type_key} and {mock_key}); {scope}"
        )
    return machine_path


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
    archiver: ArchiverSeedResult | None = None
    """What the archive rewrite did, or ``None`` when it was skipped."""


@dataclass
class ArchiverSeedResult:
    """What one archive rewrite changed.

    ``skipped`` names why nothing happened when nothing did — a project with no
    stored archive, an unreachable store, or an explicit opt-out. It is a
    reported outcome rather than a silent one: "the scenario is active but its
    history is not" is precisely the divergence this feature exists to remove,
    so it must never be something a caller has to infer from zero counts.
    """

    channels: tuple[str, ...] = ()
    restored: int = 0
    """Documents put back under retention, outside every live event window.

    A window a previous set touched, or the calm stretch between two of this
    set's own bumps: recomputed to what the active set says they hold, and
    re-stamped with their tier's ordinary lifetime so they age again."""

    updated: int = 0
    """Documents inside a live event window this set actually changed.

    Counted per document that was written, not per document visited: a re-apply
    of the set already in force changes nothing and reports zero, which is the
    honest answer to "how many archived samples did this rewrite?"
    """

    inserted: int = 0
    """Dense samples added between coarse ones inside an event window."""

    removed: int = 0
    """Previously inserted dense samples deleted during the restore."""

    uncovered: int = 0
    """Dense samples an event window called for that the store cannot carry.

    A window may reach across a stretch the archive has no samples in at all —
    a recorder outage, or history that expired. Filling it would be inventing
    coverage that never existed, so those grid points are left out and counted
    here instead of silently disappearing.
    """

    skipped: str | None = None

    def describe(self) -> str:
        """A one-line summary for the CLI."""
        if self.skipped:
            return f"archive not rewritten: {self.skipped}"
        line = (
            f"rewrote {self.updated:,} archived samples across {len(self.channels)} channel(s) "
            f"(+{self.inserted:,} dense, restored {self.restored:,}, removed {self.removed:,})"
        )
        if self.uncovered:
            line += f"; left {self.uncovered:,} uncovered instant(s) empty"
        return line


def apply_scenarios(
    project_dir: Path | str,
    names: Sequence[str],
    *,
    seed_logbook: bool = True,
    seed_archive: bool = True,
    now: datetime | None = None,
) -> ApplyResult:
    """Compose and activate scenarios for a built project; optionally seed its logbook.

    Args:
        project_dir: The deployment repo root — it anchors the ``data/simulation/``
            model and the scenario state under ``var/agent_data/simulation/``,
            and its render supplies ``config.yml`` (see :func:`_config_file`).
        names: Scenario names to activate (``nominal`` is always implicit).
        seed_logbook: When True (and the project has an ``ariel`` config),
            purge and reseed the ARIEL logbook from the active scenarios'
            entries so the narrative matches the telemetry.
        seed_archive: When True (and the project has a stored archive), rewrite
            the event windows of that archive so its history matches the
            telemetry too. A project whose history is synthesized at read time
            has nothing to rewrite and is unaffected either way.
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
    config = load_config(str(_config_file(project_dir)))

    machine_path = _require_simulation_file(
        config,
        project_dir,
        "`sim apply` only applies to simulation-backed projects (guards a real DB).",
    )
    engine = SimulationEngine.from_file(
        machine_path, state_dir=resolve_state_dir(config, project_dir)
    )

    # Default anchor in the FACILITY zone (not UTC): the anchor's tzinfo is the
    # zone each seeded logbook entry's relative time-of-day resolves into, and it
    # must match where the simulation engine places the telemetry it narrates
    # (daily ``at_time`` events are facility-local). A UTC default silently shifts
    # the narrative hours away from its archiver evidence on a non-UTC facility.
    t0 = now or datetime.now(get_facility_timezone())
    # set_active_scenarios validates composition and raises on collisions/unknowns.
    # Not announced here. Both callers already say it: `osprey sim apply` echoes
    # `✓ Active scenarios: …`, and the deploy-time reseed closes with a step
    # naming the same scenarios. The anchor is internal -- the reseed reuses the
    # persisted one precisely so nothing slides.
    active = engine.set_active_scenarios(names, anchor=t0)

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

    # After activation, never before: the rewrite synthesizes from the composed
    # event scripts the engine now holds (see :func:`seed_archiver`).
    archiver = None
    if seed_archive:
        archiver = seed_archiver(project_dir, config, engine, machine_path, list(names), t0)

    return ApplyResult(active=active, logbook_seeded=seeded, purged=purged, archiver=archiver)


# ---------------------------------------------------------------------------
# The archive rewrite
# ---------------------------------------------------------------------------

#: Rendered-config subtree holding the store's connection keys.
ARCHIVER_CONFIG_PREFIX = "mongodb_archiver"

#: The TTL field, restated as a plain name so this module can build queries and
#: updates without importing the seeder at module scope (it pulls in numpy, and
#: this module is reachable from config-loading paths that must stay lean).
EXPIRE_FIELD_NAME = "expireAt"

#: Marks a document this rewrite inserted to densify a coarse stretch. Sample
#: documents are read with an explicit projection (``date`` plus the requested
#: channels), so an extra field is invisible to every archiver query — and it is
#: the only way a later restore can tell an inserted sample from a seeded one it
#: must keep.
DENSIFIED_FIELD = "osprey_densified"

#: Documents per round trip. An event window on a month-deep archive is tens of
#: thousands of documents, and one unbounded request carrying all of them would
#: cross the server's 16 MB command limit long before it became merely slow.
_WRITE_CHUNK = 1000

#: How many sigmas either side of a spike count as "inside" its window. A
#: Gaussian is never exactly zero, so the window has to be cut somewhere; four
#: sigmas leaves under 0.01% of the bump outside it, which is far below the
#: channel's own noise and therefore invisible in the restored history.
_SPIKE_WINDOW_SIGMAS = 4.0


def active_logbook_entries(config: dict, project_dir: Path) -> list[EnhancedLogbookEntry]:
    """The logbook entries the project's ALREADY-active scenarios narrate.

    Read back from the project's own state rather than composed from an argument:
    the caller here (a deploy) is not activating anything, it is writing down what
    the running world already says it is. The anchor comes from the same state, so
    the entries land where the telemetry that accompanies them already is — a
    fresh anchor would slide the narrative to a T0 nobody asked for.

    Args:
        config: The project's loaded ``config.yml``.
        project_dir: Root of the built project.

    Returns:
        The entries, or ``[]`` when the project is not simulation-backed — a
        project with no machine model narrates nothing, which is a normal
        configuration and not a fault.
    """
    machine_path, _, _, _ = resolve_simulation_file(config, project_dir)
    if machine_path is None or not machine_path.is_file():
        return []

    engine = SimulationEngine.from_file(
        machine_path, state_dir=resolve_state_dir(config, project_dir)
    )
    anchor = persisted_scenario_anchor(config, project_dir) or datetime.now(get_facility_timezone())
    return [_to_enhanced_entry(entry, anchor) for entry in engine.active_logbook()]


async def _export_qmd_mirror(ariel_config: dict) -> None:
    """Write the markdown mirror for entries that were just seeded.

    Seeding upserts rows straight into the logbook and skips the enhancement
    passes, so nothing writes the mirror the qmd sidecar indexes. Without this
    pass a seeded deployment answers ``keyword`` searches fine while ``hybrid``
    searches an index that was never built — the failure mode is an empty
    result set, not an error, so it reads as "the logbook has nothing on that".

    The rebuild variant is deliberate: seeding is preceded by a purge in one
    caller and gated on an empty logbook in the other, and a full rebuild is
    the only pass that also clears mirrored files for entries that are gone.

    A deployment without the ``qmd_export`` module has no mirror to keep, and
    :func:`~osprey.services.ariel_search.cli_operations.run_qmd_resync` makes
    this a no-op there.

    Args:
        ariel_config: ARIEL config section with its DSN already resolved.
    """
    from osprey.services.ariel_search.cli_operations import run_qmd_resync

    await run_qmd_resync(ariel_config, rebuild=True)


def seed_active_logbook(config: dict, project_dir: Path, ariel_config: dict) -> int:
    """Write the active narrative into a logbook that has none. Returns entries seeded.

    The counterpart of :func:`seed_archiver` for the other half of a simulated
    world: a deployment whose archive is full while its logbook is empty documents
    a machine nobody can read about. Called by the deploy, which is why it is
    strictly additive where :func:`apply_scenarios`' own seeding purges first — an
    operator asking for a scenario is asking for that narrative and no other, but
    a deploy is asking for the stack to come up and has no licence to delete
    entries anyone wrote.

    So it writes only into an EMPTY logbook. A logbook with anything in it is left
    exactly as it is; the operator's route to a clean reseed remains
    ``osprey sim apply``.

    Args:
        config: The project's loaded ``config.yml``.
        project_dir: Root of the built project.
        ariel_config: ARIEL config section with its DSN already resolved.

    Returns:
        The number of entries seeded; ``0`` when the project narrates none, or
        when the logbook already holds entries.
    """
    entries = active_logbook_entries(config, project_dir)
    if not entries:
        return 0

    async def _seed_if_empty() -> int:
        from osprey.services.ariel_search import cli_operations

        if await cli_operations.logbook_entry_count(ariel_config) > 0:
            return 0
        seeded = await cli_operations.seed_logbook_entries(ariel_config, entries)
        await _export_qmd_mirror(ariel_config)
        return seeded

    return _run_coro(_seed_if_empty)


def archiver_store_config(config: dict, project_dir: Path) -> dict | None:
    """Connection parameters for the project's stored archive, if it has one.

    The password is read from ``<project_dir>/.env`` **by name**, never from the
    ambient environment and never from a ``.env`` in the current directory. A
    scenario apply is routinely run from somewhere else — a test's temporary
    directory, a benchmark harness, another project — and picking up whatever
    ``MONGO_ROOT_PASSWORD`` happened to be exported would either fail
    confusingly or, far worse, rewrite a different deployment's archive.

    Args:
        config: The project's loaded ``config.yml``.
        project_dir: Root of the built project; supplies the ``.env``.

    Returns:
        The parameters, or ``None`` when the project declares no MongoDB
        archive — a project whose history is synthesized at read time has
        nothing to rewrite, which is a normal configuration, not a fault.
    """
    archiver = config.get("archiver") or {}
    store = archiver.get(ARCHIVER_CONFIG_PREFIX)
    if not isinstance(store, dict) or not store.get("host"):
        return None

    from osprey.utils.dotenv import parse_dotenv_file

    env_path = Path(project_dir) / ".env"
    env = parse_dotenv_file(env_path) if env_path.is_file() else {}
    password_env = str(store.get("password_env") or "MONGO_ROOT_PASSWORD")

    return {
        "host": store["host"],
        # No ``port`` key means the store is the one this deployment publishes,
        # so its host port is the ``mongo`` slot of THIS config's block — never
        # the layout's default base, which would rewrite another deployment's
        # archive on a host running two.
        "port": int(store.get("port", default_port("mongo", base=resolve_port_base(config)))),
        "database": str(store.get("name") or "osprey_archiver"),
        "collection": str(store.get("collection") or "pv_history"),
        "auth_database": str(store.get("auth") or "admin"),
        "username": str(store.get("username") or "osprey"),
        "password": env.get(password_env),
        "password_env": password_env,
        "timeout_s": int(store.get("timeout", 5)),
    }


def persisted_scenario_anchor(config: dict, project_dir: Path) -> datetime | None:
    """The apply-time anchor T0 the project's scenario state already records.

    A deployment that has to *re-apply* its active set — a reseed after a knob
    change, say — must anchor that re-apply where the running world already is.
    Calling :func:`apply_scenarios` with ``now=None`` would mint a fresh anchor
    instead, silently sliding the whole timeline: the live VA's ``at_offset``
    events, the seeded logbook and the archive's event windows would all move to
    a T0 nobody asked for, as a side effect of a deploy that was supposed to
    rebuild the store and change nothing else.

    Args:
        config: The project's loaded ``config.yml``.
        project_dir: Root of the built project.

    Returns:
        The anchor as a timezone-aware datetime, or ``None`` when no set has
        been activated yet or the state file records no ``anchor=`` line — in
        which case there is no established timeline to preserve and a fresh
        anchor is the right answer.
    """
    state_dir = resolve_state_dir(config, project_dir)
    for name in (ACTIVE_SCENARIOS_FILENAME, ACTIVE_SCENARIO_FILENAME):
        path = state_dir / name
        if not path.is_file():
            continue
        # The engine's own parser, not a second one: the anchor line's format
        # (and its naive-value timezone rule) is the engine's to define, and a
        # copy here would be free to drift from the file the engine actually
        # reads. Private only because nothing outside the engine needed it
        # before.
        _names, anchor_epoch = SimulationEngine._parse_state(  # noqa: SLF001
            path.read_text(encoding="utf-8")
        )
        if anchor_epoch is not None:
            return datetime.fromtimestamp(anchor_epoch, UTC)
        return None
    return None


def _require_pymongo() -> None:
    """Fail with the fix rather than with an ImportError nobody can act on."""
    try:
        import pymongo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Rewriting the archive needs pymongo, which is not installed. "
            "It is a core dependency, so this environment is incomplete. "
            "Reinstall it with: pip install --upgrade osprey-framework"
        ) from exc


@contextmanager
def archiver_collection(store: dict):
    """Open the archive collection named by :func:`archiver_store_config`."""
    _require_pymongo()
    from pymongo import MongoClient

    client: Any = MongoClient(
        host=store["host"],
        port=store["port"],
        username=store["username"],
        password=store["password"],
        authSource=store["auth_database"],
        serverSelectionTimeoutMS=store["timeout_s"] * 1000,
    )
    try:
        yield client[store["database"]][store["collection"]]
    finally:
        client.close()


def active_archiver_events(machine_path: Path, names: Sequence[str]) -> dict[str, list[dict]]:
    """The composed archiver event scripts of an active set, by channel.

    Read straight from the machine model rather than from a live engine: this
    is the same route :func:`compute_scenario_physics_env` takes, and it keeps
    the rewrite decidable before anything is activated — so the CLI can tell a
    user what is about to change while an abort still leaves the project
    untouched.
    """
    with open(machine_path) as handle:
        model = parse_machine(json.load(handle), machine_path)

    resolved = resolve_active_scenarios(names)
    unknown = [name for name in resolved if name not in model.scenarios]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown!r}; available: {sorted(model.scenarios)}")

    events: dict[str, list[dict]] = {}
    for name in resolved:
        for pv, script in model.scenarios[name].archiver.items():
            events.setdefault(pv, []).extend(script)
    return events


def _refuse_window_fraction(event: Mapping[str, Any]) -> None:
    """Refuse an event positioned by window fraction, by name."""
    if "at" in event:
        raise ValueError(
            f"Archiver event {event!r} is positioned by window fraction ('at'), which "
            f"has no place in stored history — a fraction names a position in the "
            f"reader's window, not an instant. Use 'at_offset' (seconds relative to "
            f"the activation anchor) instead."
        )


def _require_events(events: Sequence[dict]) -> None:
    """Refuse an empty script rather than answer with an inverted window."""
    if not events:
        raise ValueError(
            "An empty archiver event script names no window: a channel with nothing "
            "to write has no stretch of history to recompute, protect or record. "
            "Skip the channel instead of asking for its window."
        )


def event_window(
    events: Sequence[dict], anchor: float, horizon_start: float
) -> tuple[float, float]:
    """The absolute span one channel's events can affect, in epoch seconds.

    This is the span whose *values* have to be recomputed — every instant whose
    sample the script can change. A step or a ramp changes every sample from its
    position onward, so its span runs to the end of the archive; a daily
    ``at_time`` event recurs on every date the archive covers, so its span is
    the whole archive. Both are measured against ``anchor`` — the apply-time T0
    written into the scenario state file, not this process's wall clock (a
    deploy's post-reseed re-apply passes the anchor the running world is already
    on; see :func:`seed_archiver`).

    It is deliberately *not* the span to densify or to protect from retention.
    A daily event's span is a month long but its evidence is a handful of
    minutes a day; treating the two as one would de-expire an entire deployment
    and insert dense samples across every quiet hour between the bumps. Ask
    :func:`event_subwindows` for those.

    Args:
        events: One channel's event script; must be non-empty.
        anchor: T0 in epoch seconds.
        horizon_start: Oldest instant the archive covers; clamps a span whose
            event is anchored further back than the archive reaches.

    Returns:
        ``(start, end)`` in epoch seconds. ``end`` is ``anchor`` for anything
        persistent.

    Raises:
        ValueError: If the script is empty, or an event is positioned by window
            *fraction*. A fraction names a place in whatever window a reader
            happens to ask for, which is not a place in stored history at all —
            there is no honest timestamp to write it at.
    """
    _require_events(events)
    start = anchor
    end = horizon_start
    for event in events:
        _refuse_window_fraction(event)
        if "at_time" in event:
            # A daily time-of-day recurs on every date the archive covers, so
            # the span its values reach over is the archive.
            return horizon_start, anchor
        at = anchor + float(event["at_offset"])
        if event["shape"] == "spike":
            width = float(event["width"]) * _SPIKE_WINDOW_SIGMAS
            start, end = min(start, at - width), max(end, at + width)
        else:
            start, end = min(start, at), anchor
    return max(start, horizon_start), min(end, anchor)


def event_subwindows(
    events: Sequence[dict],
    anchor: float,
    horizon_start: float,
    *,
    tz: ZoneInfo | None = None,
) -> list[tuple[float, float]]:
    """The stretches one channel's events are actually *evidence* over.

    One entry per occurrence rather than one per script: a spike contributes its
    own bump, a daily spike contributes one bump per date the archive covers,
    and a step or a ramp — which genuinely does change every later sample —
    contributes the stretch from its position to the anchor. Overlapping
    stretches are merged, so the result is disjoint and ascending.

    These are the stretches the rewrite densifies and lifts retention from, and
    keeping them separate from :func:`event_window`'s single span is what stops
    a three-spike script from protecting the calm day between its bumps and a
    daily event from protecting the whole deployment.

    Args:
        events: One channel's event script; must be non-empty.
        anchor: T0 in epoch seconds.
        horizon_start: Oldest instant the archive covers. Occurrences are
            clamped to ``[horizon_start, anchor]``, and one that falls entirely
            outside it contributes nothing.
        tz: Timezone daily ``at_time`` occurrences are placed in. Defaults to
            the facility zone, matching where the engine places them.

    Returns:
        Disjoint ``(start, end)`` pairs, ascending. Empty when every occurrence
        falls outside the archive.

    Raises:
        ValueError: As :func:`event_window`.
    """
    _require_events(events)
    windows: list[tuple[float, float]] = []
    for event in events:
        _refuse_window_fraction(event)
        for at in _event_instants(event, anchor, horizon_start, tz):
            if event["shape"] == "spike":
                half = float(event["width"]) * _SPIKE_WINDOW_SIGMAS
                windows.append((at - half, at + half))
            else:
                windows.append((at, anchor))
    clamped = [
        (max(start, horizon_start), min(end, anchor))
        for start, end in windows
        if end >= horizon_start and start <= anchor
    ]
    return _merge_intervals(clamped)


def _event_instants(
    event: Mapping[str, Any], anchor: float, horizon_start: float, tz: ZoneInfo | None
) -> list[float]:
    """Every instant one event fires at, inside ``[horizon_start, anchor]``."""
    if "at_time" in event:
        from osprey.simulation.series import daily_occurrences

        if tz is None:
            tz = get_facility_timezone()
        return daily_occurrences(str(event["at_time"]), _np_array([horizon_start, anchor]), tz)
    return [anchor + float(event["at_offset"])]


def _merge_intervals(windows: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Overlapping or touching spans collapsed into disjoint ascending ones."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside(windows: Sequence[tuple[float, float]] | None, moment: float) -> bool:
    """Whether an instant falls in any of these spans."""
    return any(start <= moment <= end for start, end in windows or ())


def seed_archiver(
    project_dir: Path,
    config: dict,
    engine: SimulationEngine,
    machine_path: Path,
    names: Sequence[str],
    anchor: datetime,
) -> ArchiverSeedResult:
    """Rewrite the stored archive so it tells the newly active set's story.

    Call this *after* the set has been activated: the engine's composed event
    scripts are what the rewrite synthesizes from, and one recompute under the
    new set is both the restore and the apply.

    That collapse is the design, and it is worth being explicit about. A
    previous set's events left their marks on some windows, and the naive
    reading of "restore, then apply" is two passes. But every value in this
    archive is a pure function of ``(channel, timestamp)`` *under the active
    set* — so recomputing the union of the old touched windows and the new
    event windows, once, under the set now in force, lands every document on
    its correct final value in one pass. A channel the new set does not touch
    recomputes to its bare base, which is exactly what restoring it means. Two
    passes would do the same arithmetic twice and leave a window in an
    intermediate state in between.

    What is genuinely two-sided is *expiry*, because it does not follow from the
    values, and it is decided one document at a time: a document whose own
    timestamp falls inside a live event window has its expiry removed, so
    retention can never prune the evidence a scenario is about to be diagnosed
    from, and every other document in the recomputed span — including the whole
    stretch that only the previous set touched — gets its tier's stamp back.
    Dense samples this rewrite inserted are likewise deleted and re-inserted
    rather than recomputed — they exist only where an event needs resolution the
    coarse grid cannot give.

    The order of the writes is a durability contract rather than an efficiency
    one. The touched-window ledger goes down *first*, naming the union of the
    old and the new windows, and is narrowed to the new ones only once the
    rewrite has returned: a process that dies part way through has already
    written event values into windows the narrowed ledger would not mention, and
    a window no ledger names is one no later apply ever comes back for.

    Args:
        project_dir: Root of the built project; supplies ``.env`` and the store.
        config: The project's loaded ``config.yml``.
        engine: The engine, already activated on the new set.
        machine_path: The machine model, for reading the composed event scripts.
        names: The requested scenario names.
        anchor: The apply-time anchor T0 — the *same* instant that was written
            into the scenario state file. It has to be: the engine resolves
            ``at_offset`` against the state file, so a window computed against
            any other clock would describe a stretch of history the engine is
            not writing its events into. One clock for the telemetry, the
            narrative and the archive is the whole invariant.

    Returns:
        What changed, or a result carrying ``skipped`` when the project has no
        stored archive to rewrite.

    Raises:
        RuntimeError: If pymongo is missing, or the store is configured but its
            password is not in the project's ``.env``.
        ValueError: If an active scenario positions an archiver event by window
            fraction, which stored history cannot represent.
    """
    from osprey.simulation.archiver_seed import MANIFEST_ID, SeedKnobs

    store = archiver_store_config(config, project_dir)
    if store is None:
        return ArchiverSeedResult(skipped="project declares no MongoDB archive")
    if not store["password"]:
        raise RuntimeError(_missing_password_message(project_dir, store))

    _require_pymongo()
    knobs = SeedKnobs.from_config(config)
    # An empty script is not a window: it names a channel the scenario mentions
    # and then leaves alone, and asking for its span would invert one.
    events = {
        pv: script for pv, script in active_archiver_events(machine_path, names).items() if script
    }
    tz = get_facility_timezone()

    with archiver_collection(store) as collection:
        manifest = collection.find_one({"_id": MANIFEST_ID})
        if manifest is None:
            return ArchiverSeedResult(
                skipped="the archive has not been seeded yet — run 'osprey up'"
            )

        anchor_s = anchor.timestamp()
        horizon_start = _archive_start(collection, manifest, anchor_s, knobs)
        previous = _ledger_windows(manifest)
        current = {
            pv: event_window(script, anchor_s, horizon_start) for pv, script in events.items()
        }
        live = {
            pv: event_subwindows(script, anchor_s, horizon_start, tz=tz)
            for pv, script in events.items()
        }
        spans = _union_spans(previous, current)

        result = ArchiverSeedResult(channels=tuple(sorted(current)))

        # The ledger goes down first, and it names the *union* — every window
        # this run is about to disturb, not the ones it means to leave marked.
        # A process that dies mid-rewrite has already written event values into
        # windows the new ledger would not mention, and a window no ledger names
        # is a window no later apply ever comes back for: the marks would be
        # permanent and unsignalled. A superset costs the next apply one extra
        # recompute over a stretch that is already at its base value.
        #
        # Written whenever there is anything at all to disturb, including when
        # the union happens to equal the new windows — which is exactly the case
        # on a store no scenario has touched yet. Skipping it there because "the
        # ledger would say the same thing at the end anyway" would leave the
        # *first* eventful apply on every fresh deployment writing marks under
        # an empty ledger, which is the one crash nothing could recover from.
        #
        # Any guard here must be phrased against what is already *on disk*,
        # never against what this run intends to leave behind; the unconditional
        # form is the one that cannot be got wrong, and it costs one small
        # manifest write per eventful apply.
        if spans:
            _write_ledger(collection, spans, anchor_s)

        # Dense inserts are keyed to the windows that needed them; a window that
        # is no longer an event window must not keep resolution nothing asked
        # for, so they go before anything is recomputed.
        regions = _merge_intervals(list(spans.values()))
        result.removed = _drop_densified(collection, regions)

        result.updated, result.restored = _rewrite_documents(collection, engine, knobs, spans, live)
        result.inserted, result.uncovered = _densify(collection, engine, live, knobs)

        _write_ledger(collection, current, anchor_s)

    if result.uncovered:
        logger.warning(
            f"{result.uncovered} dense sample(s) an event window called for were left out: "
            f"the archive has no coverage there to densify."
        )
    # As above: `osprey sim apply` echoes `✓ Archive rewritten: <describe()>`
    # itself, and on a deploy the reseed's closing step carries the same counts.
    return result


def _missing_password_message(project_dir: Path, store: dict) -> str:
    """Why an archive rewrite cannot proceed without the project's own password."""
    return (
        f"The archive is configured but {store['password_env']} is not in "
        f"{project_dir / '.env'}, so its history cannot be rewritten and would "
        f"contradict the scenario now active. Run 'osprey up' to mint it."
    )


def preflight_archive_rewrite(
    project_dir: Path,
    config: dict,
    machine_path: Path,
    names: Sequence[str],
) -> dict | None:
    """Decide the archive rewrite *before* anything has been activated.

    :func:`active_archiver_events` is readable straight from the machine model,
    with no engine and no store, precisely so this is possible — and this is the
    caller that uses it. Every refusal :func:`seed_archiver` can raise before it
    touches a document (an event positioned by window fraction, a store whose
    password the project's ``.env`` does not carry) is raised here instead,
    while the scenario is still inactive and the logbook still intact.

    Reaching those refusals from inside the rewrite would leave the world half
    applied: telemetry live, narrative reseeded, history untouched and
    contradicting both — which is the exact divergence this feature removes.

    Args:
        project_dir: Root of the built project.
        config: The project's loaded ``config.yml``.
        machine_path: The machine model.
        names: The requested scenario names.

    Returns:
        The store parameters, or ``None`` when the project declares no stored
        archive and there is nothing to rewrite.

    Raises:
        ValueError: If a scenario name is unknown, or an active scenario
            positions an archiver event by window fraction.
        RuntimeError: If the store is configured but its password is not in the
            project's ``.env``.
    """
    for script in active_archiver_events(machine_path, names).values():
        for event in script:
            _refuse_window_fraction(event)

    store = archiver_store_config(config, project_dir)
    if store is not None and not store["password"]:
        raise RuntimeError(_missing_password_message(project_dir, store))
    return store


def _archive_start(collection, manifest: dict, anchor_s: float, knobs: SeedKnobs) -> float:
    """The oldest instant this archive actually covers.

    Read from the store rather than computed from the anchor: retention has been
    pruning the far end since the day it was seeded, and a horizon derived from
    arithmetic would claim coverage that expired days ago. Falls back to the
    manifest's seed instant less the retention span when the store is somehow
    empty of dated samples, which is the coverage a fresh seed would have.

    Windows are clamped to this so an event anchored further back than the
    archive reaches writes into the history that exists instead of describing
    history that does not.
    """
    from osprey.simulation.archiver_seed import oldest_sample

    oldest = oldest_sample(collection)
    if oldest is not None:
        return float(oldest.timestamp())
    seeded_at = manifest.get("seeded_at")
    if isinstance(seeded_at, datetime):
        aware = seeded_at if seeded_at.tzinfo is not None else seeded_at.replace(tzinfo=UTC)
        return float(aware.timestamp()) - knobs.retention_s
    return anchor_s - knobs.retention_s


def _ledger_windows(manifest: dict) -> dict[str, tuple[float, float]]:
    """The windows a previous apply reported touching, by channel."""
    windows: dict[str, tuple[float, float]] = {}
    for entry in manifest.get("touched_windows") or []:
        try:
            start, end = float(entry["start"]), float(entry["end"])
            for pv in entry["channels"]:
                windows[str(pv)] = (start, end)
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Ignoring an unreadable touched-window ledger entry: {entry!r}")
    return windows


def _ledger_entries(windows: dict[str, tuple[float, float]]) -> list[dict]:
    """The ledger to store, grouping channels that share a window."""
    grouped: dict[tuple[float, float], list[str]] = {}
    for pv, span in windows.items():
        grouped.setdefault(span, []).append(pv)
    return [
        {"start": start, "end": end, "channels": sorted(channels)}
        for (start, end), channels in sorted(grouped.items())
    ]


def _write_ledger(collection, windows: dict[str, tuple[float, float]], anchor_s: float) -> None:
    """Record the windows a later apply has to recompute over."""
    from osprey.simulation.archiver_seed import MANIFEST_ID

    collection.update_one(
        {"_id": MANIFEST_ID},
        {"$set": {"touched_windows": _ledger_entries(windows), "touched_anchor": anchor_s}},
    )


def _union_spans(
    previous: dict[str, tuple[float, float]],
    current: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """The span each channel needs recomputing over, old and new together.

    A channel in both maps needs the union of its two spans: the old marks have
    to be recomputed away and the new ones written, and one span covering both
    does each document exactly once.
    """
    spans: dict[str, tuple[float, float]] = {}
    for pv in set(previous) | set(current):
        old, new = previous.get(pv), current.get(pv)
        if old and new:
            spans[pv] = (min(old[0], new[0]), max(old[1], new[1]))
        else:
            spans[pv] = old or new  # type: ignore[assignment]
    return spans


def _region_documents(collection, channels: Sequence[str], start: float, end: float) -> list[dict]:
    """Existing documents in a span, projected to ``date`` and these channels."""
    projection = {"date": 1, EXPIRE_FIELD_NAME: 1, **dict.fromkeys(channels, 1)}
    cursor = collection.find(
        {
            "date": {
                "$gte": datetime.fromtimestamp(start, UTC),
                "$lte": datetime.fromtimestamp(end, UTC),
            }
        },
        projection,
    ).sort("date", 1)
    return list(cursor)


def _stamps(documents: Sequence[dict]) -> list[datetime]:
    """The instants a set of documents was sampled at, UTC-aware."""
    return [
        doc["date"] if doc["date"].tzinfo is not None else doc["date"].replace(tzinfo=UTC)
        for doc in documents
    ]


def _match_stored_type(existing, value):
    """Coerce a recomputed value to the type the stored one already had.

    The store is its own schema authority here. The base seed wrote each channel
    in the type the machine serves it as — a flag as a boolean, an enumeration
    as an integer — and a rewrite that put a float on a flag channel would leave
    that channel's history changing type partway through, which reads as a
    different instrument rather than a different value.
    """
    if isinstance(existing, bool):
        return bool(value)
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, str):
        return str(value)
    return float(value)


def _rewrite_documents(
    collection,
    engine: SimulationEngine,
    knobs: SeedKnobs,
    spans: dict[str, tuple[float, float]],
    live: dict[str, list[tuple[float, float]]],
) -> tuple[int, int]:
    """Recompute every document the old and the new set between them reach.

    One pass over the union of the spans, and every document visited exactly
    once. Values come from the engine at each document's *own* timestamp, so the
    rewrite lands on exactly what a query for that instant would synthesize —
    the same property the base seed relies on, applied to a narrower window.

    Expiry is decided **per document**, from whether that document's own
    timestamp falls inside a live event window of a channel it actually carries.
    Inside one, the expiry is removed outright: retention must never prune the
    stretch of history a scenario exists to be diagnosed from. Outside one, the
    document goes back under its tier's ordinary lifetime, so a window a
    previous set protected starts ageing again the moment it stops being
    evidence. Deciding this once per *region* instead — the union of the old and
    the new window is a region, and it is live at one end and dead at the other
    — would leave the whole old window unexpiring for good: the new ledger does
    not name it, so no later apply would ever come back for it.

    Returns:
        ``(updated, restored)`` — documents this pass actually wrote, split by
        whether they now carry the new set's story or were put back under
        retention. A document whose values and expiry are already correct is not
        rewritten and not counted, so re-applying the set in force reports zero.
    """
    from osprey.simulation.archiver_seed import tier_expiry

    if not spans:
        return 0, 0
    channels = sorted(spans)
    start = min(span[0] for span in spans.values())
    end = max(span[1] for span in spans.values())
    documents = _region_documents(collection, channels, start, end)
    if not documents:
        return 0, 0

    stamps = _stamps(documents)
    epochs = [stamp.timestamp() for stamp in stamps]
    values = _recomputed_values(engine, spans, channels, stamps, epochs)
    expiry = tier_expiry(knobs, _np_array(epochs))

    from pymongo import UpdateOne

    operations = []
    updated = restored = 0
    for index, document in enumerate(documents):
        protected = False
        changed: dict[str, Any] = {}
        for pv in channels:
            column = values.get(pv)
            if column is None or index not in column or pv not in document:
                continue
            value = _match_stored_type(document[pv], column[index])
            if document[pv] != value:
                changed[pv] = value
            # Per channel *and* per timestamp — do not simplify to a test against
            # the merged live windows of every channel. The two are behaviourally
            # identical on this schema, because a base-seeded document carries
            # every channel and so is "inside" the merged windows exactly when it
            # is inside its own channel's; no test here can tell them apart. What
            # makes them different is the sparse documents: a dense insert
            # carries only the channels live at its instant, and a channel-blind
            # test would protect it on a neighbour's evidence rather than its own.
            protected = protected or _inside(live.get(pv), epochs[index])
        update = _expiry_update(document, protected, float(expiry[index]))
        if changed:
            update.setdefault("$set", {}).update(changed)
        if not update:
            continue
        operations.append(UpdateOne({"_id": document["_id"]}, update))
        if protected:
            updated += 1
        else:
            restored += 1

    for batch in _chunked(operations, _WRITE_CHUNK):
        collection.bulk_write(batch, ordered=False)
    return updated, restored


def _recomputed_values(
    engine: SimulationEngine,
    spans: dict[str, tuple[float, float]],
    channels: Sequence[str],
    stamps: Sequence[datetime],
    epochs: Sequence[float],
) -> dict[str, dict[int, float]]:
    """Each channel's recomputed values, by document index, over its own span.

    Synthesized over the channel's own span rather than over the union of every
    channel's: a daily event's occurrences are placed inside the window it is
    asked for, so widening that window to accommodate an unrelated channel would
    make one channel's history depend on which other channels the active set
    happens to touch.
    """
    values: dict[str, dict[int, float]] = {}
    for pv in channels:
        low, high = spans[pv]
        indices = [index for index, epoch in enumerate(epochs) if low <= epoch <= high]
        if not indices:
            continue
        series = engine.synthesize_series(pv, [stamps[index] for index in indices])
        values[pv] = dict(zip(indices, series, strict=True))
    return values


def _expiry_update(document: Mapping[str, Any], protected: bool, expiry_s: float) -> dict:
    """The expiry half of one document's update, empty when it is already right."""
    held = document.get(EXPIRE_FIELD_NAME)
    if protected:
        return {"$unset": {EXPIRE_FIELD_NAME: ""}} if held is not None else {}
    wanted = datetime.fromtimestamp(expiry_s, UTC)
    if isinstance(held, datetime):
        aware = held if held.tzinfo is not None else held.replace(tzinfo=UTC)
        # The store keeps datetimes to the millisecond, so an exact compare
        # would report a change on every re-apply of an unchanged window.
        if abs((aware - wanted).total_seconds()) < 0.001:
            return {}
    return {"$set": {EXPIRE_FIELD_NAME: wanted}}


def _densify(
    collection,
    engine: SimulationEngine,
    live: dict[str, list[tuple[float, float]]],
    knobs: SeedKnobs,
) -> tuple[int, int]:
    """Add dense samples between the coarse ones inside the live event windows.

    Most of the archive is a month deep at a coarse cadence, and a spike whose
    width is comparable to that cadence would be represented by one or two
    points — a shape no operator could read and no correlation could find. The
    inserts carry *only* the channels live at each instant: the schema allows a
    sparse document, so a dense sample of the channels an event touches costs a
    fraction of a full one, and every other channel keeps the coarse history it
    genuinely has rather than being given invented resolution.

    Densification runs once over the merged live windows, not once per channel,
    so an instant two channels share gets one document carrying both. Filling
    per channel would let whichever ran first claim the timestamp and leave the
    other with a hole — and the order it ran in would be a set-iteration
    accident, so the archive would come out differently run to run.

    Each insert is marked (:data:`DENSIFIED_FIELD`) and left unexpiring, so a
    later rewrite can tell them from seeded samples and take them away again
    when the window stops being an event window.

    Returns:
        ``(inserted, uncovered)`` — samples written, and grid points left empty
        because the store has no coverage there to densify.
    """
    if not live:
        return 0, 0
    intervals = _merge_intervals([window for windows in live.values() for window in windows])
    if not intervals:
        return 0, 0

    types = _stored_types(collection, sorted(live))
    inserted = uncovered = 0
    for start, end in intervals:
        added, missed = _densify_interval(collection, engine, live, knobs, types, start, end)
        inserted += added
        uncovered += missed
    return inserted, uncovered


def _densify_interval(
    collection,
    engine: SimulationEngine,
    live: dict[str, list[tuple[float, float]]],
    knobs: SeedKnobs,
    types: Mapping[str, Any],
    start: float,
    end: float,
) -> tuple[int, int]:
    """Fill one live window's coarse stretches, and only its coarse stretches.

    A grid point is filled only when it lies strictly between two stored samples
    no further apart than the coarse cadence — that is what "between the coarse
    ones" means, and it is the difference between adding resolution to history
    the store has and manufacturing history it does not. A recorder outage, a
    stretch that expired, or a window reaching past the newest sample all leave
    gaps wider than the coarse cadence, and inventing a month of samples across
    one of them would be the same defect this feature exists to remove, told in
    the opposite direction.
    """
    reach = float(knobs.tail_cadence_sec)
    grid = [
        moment
        for moment in _aligned_epochs(start, end, knobs.hot_cadence_sec)
        if moment % knobs.tail_cadence_sec != 0
    ]
    if not grid:
        return 0, 0

    # One coarse cadence of margin either side, so a grid point at the very edge
    # of the window can still see the stored sample that brackets it.
    neighbours = sorted(
        stamp.timestamp()
        for stamp in _stamps(_region_documents(collection, [], start - reach, end + reach))
    )
    held = set(neighbours)

    wanted: dict[float, tuple[str, ...]] = {}
    uncovered = 0
    for moment in grid:
        if moment in held:
            continue
        channels = tuple(pv for pv in sorted(live) if _inside(live[pv], moment))
        if not channels:
            continue
        if not _bracketed(neighbours, moment, reach):
            uncovered += 1
            continue
        wanted[moment] = channels

    if not wanted:
        return 0, uncovered

    documents = _dense_documents(engine, types, wanted)
    for batch in _chunked(documents, _WRITE_CHUNK):
        collection.insert_many(batch, ordered=False)
    return len(documents), uncovered


def _bracketed(neighbours: Sequence[float], moment: float, reach: float) -> bool:
    """Whether two stored samples no more than ``reach`` apart straddle ``moment``."""
    import bisect

    index = bisect.bisect_left(neighbours, moment)
    if index == 0 or index >= len(neighbours):
        return False
    return neighbours[index] - neighbours[index - 1] <= reach


def _dense_documents(
    engine: SimulationEngine,
    types: Mapping[str, Any],
    wanted: Mapping[float, tuple[str, ...]],
) -> list[dict]:
    """The documents to insert: one per instant, carrying the channels live at it."""
    moments = sorted(wanted)
    stamps = [datetime.fromtimestamp(moment, UTC) for moment in moments]
    columns: dict[str, dict[int, float]] = {}
    for pv in sorted({pv for channels in wanted.values() for pv in channels}):
        indices = [index for index, moment in enumerate(moments) if pv in wanted[moment]]
        series = engine.synthesize_series(pv, [stamps[index] for index in indices])
        columns[pv] = dict(zip(indices, series, strict=True))

    documents = []
    for index, moment in enumerate(moments):
        document: dict = {"date": stamps[index], DENSIFIED_FIELD: True}
        for pv in wanted[moment]:
            # Same coercion the updates use: a flag channel seeded as a boolean
            # must not acquire a float beside it half way through its history.
            document[pv] = _match_stored_type(types.get(pv), columns[pv][index])
        documents.append(document)
    return documents


def _stored_types(collection, channels: Sequence[str]) -> dict[str, Any]:
    """One stored value per channel, as the type authority for inserts.

    The store is its own schema authority (see :func:`_match_stored_type`), and
    an insert has no document of its own to read that from — so it reads one
    from the channel's existing history instead.
    """
    samples: dict[str, Any] = {}
    for pv in channels:
        document = collection.find_one({pv: {"$exists": True}}, {pv: 1})
        if document is not None:
            samples[pv] = document.get(pv)
    return samples


def _chunked(items: Sequence[Any], size: int) -> Iterator[list]:
    """Split a batch of writes into round trips the server will accept."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _aligned_epochs(start: float, end: float, cadence: int) -> list[float]:
    """Whole multiples of ``cadence`` since the epoch inside ``[start, end]``.

    The same alignment the base seed's grid uses, so an inserted sample lands on
    a timestamp the seeded grid would have used had the window been dense —
    never half a cadence away from one, which would leave two samples where the
    archive should hold one.
    """
    first = math.ceil(start / cadence) * cadence
    return [float(moment) for moment in range(int(first), int(end) + 1, cadence)]


def _np_array(values):
    """Local numpy import: this module is reachable from config-loading paths."""
    import numpy as np

    return np.asarray(values, dtype=float)


def _drop_densified(collection, windows: Sequence[tuple[float, float]]) -> int:
    """Delete the dense samples a previous rewrite inserted in these windows."""
    removed = 0
    for start, end in windows:
        outcome = collection.delete_many(
            {
                DENSIFIED_FIELD: True,
                "date": {
                    "$gte": datetime.fromtimestamp(start, UTC),
                    "$lte": datetime.fromtimestamp(end, UTC),
                },
            }
        )
        removed += outcome.deleted_count
    return removed


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


def compute_scenario_physics_env(
    project_dir: Path | str,
    names: Sequence[str],
) -> dict[str, str]:
    """Resolve the active scenarios' ``physics`` faults into VA_* env vars -- pure.

    The compute/validate half of :func:`render_scenario_physics_env`: it reads
    the project's config and machine description, resolves the active set, and
    renders the ``VA_*`` values, but has *no* filesystem effect. Every way this
    step can fail -- non-simulation-backed project, unknown scenario name, two
    active scenarios faulting the same device -- raises here, before anything
    is written, so a caller that validates first (``osprey sim apply``) can
    abort with zero writes anywhere (FR1).

    Args:
        project_dir: The deployment repo root — it anchors the
            ``data/simulation/`` model, and its render supplies ``config.yml``
            (see :func:`_config_file`).
        names: Scenario names to activate (``nominal`` is always implicit),
            resolved the same nominal-first, deduped way
            :meth:`~osprey.simulation.engine.SimulationEngine.set_active_scenarios`
            resolves them.

    Returns:
        The ``VA_*`` vars the active set calls for, empty if no active scenario
        declares a ``physics`` block. Hand this to
        :func:`write_scenario_physics_env` to make it live.

    Raises:
        ValueError: If the project is not simulation-backed (mirrors
            :func:`apply_scenarios`), a requested scenario name is unknown, or
            two active scenarios declare a physics fault on the same device.
    """
    project_dir = Path(project_dir)
    config = load_config(str(_config_file(project_dir)))

    machine_path = _require_simulation_file(
        config,
        project_dir,
        "physics-fault rendering only applies to simulation-backed projects.",
    )
    with open(machine_path) as f:
        machine = json.load(f)
    model = parse_machine(machine, machine_path)

    resolved = resolve_active_scenarios(names)
    unknown = [n for n in resolved if n not in model.scenarios]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown!r}; available: {sorted(model.scenarios)}")

    return _render_physics_vars(model.scenarios, resolved)


def write_scenario_physics_env(
    project_dir: Path | str,
    rendered: dict[str, str],
    *,
    env_path: Path | None = None,
) -> bool:
    """Write :func:`compute_scenario_physics_env`'s result into the repo's ``.env``.

    The write half of :func:`render_scenario_physics_env`, callable on its own
    so a caller can put every prompt and validation ahead of the first
    filesystem effect.

    Args:
        project_dir: The deployment repo root — it supplies the default
            ``.env``, which is the file ``osprey up``'s compose reads as
            ``--env-file``. Pointing this at the render writes the faults into
            a file nothing interpolates, and the VA boots fault-free.
        rendered: The ``VA_*`` vars to reconcile the ``.env`` to, as returned
            by :func:`compute_scenario_physics_env`.
        env_path: ``.env`` path to write into (defaults to
            ``project_dir/.env``, injectable for tests).

    Returns:
        Whether the ``.env``'s physics block actually *changed* -- rendering a
        new fault, or clearing a prior render's stale one, both count; a
        rewrite that reproduces the existing content byte for byte does not.
        Callers use this to decide whether the running VA is now out of date
        with the file and needs an ``osprey up`` (FR2).
    """
    if env_path is None:
        env_path = Path(project_dir) / ".env"
    return _write_physics_env(env_path, rendered)


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
    the repo's ``.env`` as ``VA_BPM_ERRORS``/
    ``VA_CORR_GAIN``, the exact env vars
    ``virtual_accelerator/entrypoint.py`` parses, rather than applied live.
    Call this before ``osprey up`` so the VA container picks up the rendered
    values at boot.

    Composes :func:`compute_scenario_physics_env` and
    :func:`write_scenario_physics_env` back to back; call those two directly
    instead when something has to happen between validating and writing.

    Args:
        project_dir: The deployment repo root — it anchors the build-owned
            ``data/simulation/`` model, the scenario state under the agent-data
            root (``agent_data.base_dir``), and the ``.env`` written here; its
            render supplies ``config.yml``.
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
    rendered = compute_scenario_physics_env(project_dir, names)
    write_scenario_physics_env(project_dir, rendered, env_path=env_path)
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
# The block's header line, owned and reconciled exactly like the keys under it:
# dropped on the way in and re-emitted only alongside a rendered value, so a
# re-render reproduces the file byte for byte instead of stacking a fresh
# header each time (which would make every rewrite look like a change).
_PHYSICS_ENV_HEADER = "# Scenario physics fault (osprey sim apply / osprey up)"


def _write_physics_env(env_path: Path, rendered: dict[str, str]) -> bool:
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

    Returns whether the *physics block* changed -- the vars this function owns,
    before versus after -- not whether bytes moved. The rewrite is
    unconditional, so "a write happened" is not the signal a caller wants; nor
    is a whole-file comparison, which would report a change for incidental
    normalization (a hand-edited ``.env`` with no final newline, say) and make
    ``osprey sim apply`` announce a physics change that never happened.
    Re-applying the same scenario reports False; clearing a stale ``VA_*`` line
    reports True even though ``rendered`` is empty.
    """
    if not rendered and not env_path.is_file():
        return False

    before = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    lines = before.splitlines()
    kept: list[str] = []
    before_block: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped == _PHYSICS_ENV_HEADER:
            continue  # dropped here; re-added below if still active
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)  # values contain '=' (DEVICE=factor)
            key = key.strip()
            if key in _PHYSICS_ENV_VARS:
                before_block[key] = value.strip()
                continue  # dropped here; re-added below if still active
        kept.append(line)
    while kept and kept[-1] == "":
        kept.pop()

    if rendered:
        if kept:
            kept.append("")
        kept.append(_PHYSICS_ENV_HEADER)
        kept.extend(f"{k}={rendered[k]}" for k in _PHYSICS_ENV_VARS if k in rendered)

    text = "\n".join(kept) + ("\n" if kept else "")
    env_path.write_text(text, encoding="utf-8")
    os.chmod(env_path, 0o600)
    return before_block != rendered


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
    await _export_qmd_mirror(ariel_config)
    return seeded, True
