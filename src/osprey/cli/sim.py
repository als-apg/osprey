"""Simulation scenario CLI commands.

Thin CLI wrappers over the simulation engine and
:func:`osprey.simulation.apply.apply_scenarios`.

These commands find their deployment the way every repo-scoped verb does —
walk up from the working directory to the nearest ``profile.yml``, or start
from ``--repo`` — so they work from any subdirectory of the repo rather than
only from its root. Three directories come out of that one decision and they
are genuinely different files:

- the render (``build/``) holds ``config.yml``, the simulator view
  ``data/simulator/`` that ``list`` and ``status`` read, and the build-owned
  ``data/simulation/`` model ``apply`` loads;
- the repo root anchors ``var/agent_data/simulation/``, where the mutable
  active-scenario state lives, because a scenario switch has to survive
  ``osprey build`` wiping the render;
- the repo root also holds the single ``.env`` a scenario's ``physics`` block
  is rendered into, for the virtual accelerator to read at its next boot.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from osprey.cli import output
from osprey.utils.config import load_config
from osprey.utils.logger import get_logger
from osprey_connectors.types import CONTROL_TARGETS

from .repo_resolver import find_repo_root, repo_option

logger = get_logger("sim")


def _parse_now(now_iso: str) -> datetime:
    """Parse an ISO-8601 ``--now`` anchor into an aware datetime.

    A naive value takes the facility timezone — the same zone
    :func:`osprey.simulation.apply.apply_scenarios` resolves seeded logbook
    time-of-day into — so a bare ``2024-03-18T12:00:00`` freezes the narrative
    on the facility clock rather than silently landing in UTC.
    """
    try:
        anchor = datetime.fromisoformat(now_iso)
    except ValueError:
        output.fail(
            f"The --now value {now_iso!r} is not valid ISO-8601",
            "An instant looks like 2024-03-18T12:00:00.",
        )
        raise SystemExit(1) from None
    if anchor.tzinfo is None:
        from osprey.utils.config import get_facility_timezone

        anchor = anchor.replace(tzinfo=get_facility_timezone())
    return anchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_deployment(repo: Path | None) -> tuple[Path, dict]:
    """Return ``(repo_root, config)`` for the deployment being acted on.

    The repo root is what every path here anchors on, because it is what
    ``project_root`` means in the rendered config — the value the mock
    connectors resolve their own model and state against at runtime. Anchoring
    the CLI anywhere else would let ``sim apply`` write an active-scenario file
    that the running connectors never read.

    Exits with a clear message when the repo carries no render, the one state
    these commands cannot work in: ``config.yml`` is what names the simulation
    model, the state directory, and the ARIEL logbook.

    The render is also anchored as this process's config for the rest of the
    command. These verbs run wholly on the host, so nothing injects
    ``CONFIG_FILE`` the way compose does for a container, and the working
    directory is a repo root that holds no ``config.yml`` — every unqualified
    lookup a scenario apply makes would otherwise answer from defaults. The
    facility timezone is the one that bites: it decides what wall-clock a
    seeded logbook entry lands on, and the fallback is a plausible-looking UTC
    rather than an error. Unwound with the click context, so the anchor cannot
    outlive the verb (see :func:`~osprey.utils.config.config_anchored_at`).

    Raises:
        RepoNotFoundError: When no ``profile.yml`` encloses the search start.
    """
    from osprey.utils.config import config_anchored_at
    from osprey.utils.workspace import BUILD_DIR_NAME, rendered_config_path

    repo_root = find_repo_root(repo)
    config_path = rendered_config_path(repo_root)
    if not config_path.is_file():
        output.fail(
            f"No build found at {repo_root / BUILD_DIR_NAME}",
            "The scenarios are created by the build.",
            "run 'osprey build' first",
        )
        raise SystemExit(1)
    click.get_current_context().with_resource(config_anchored_at(config_path))
    return repo_root, load_config(str(config_path))


def _require_simulator_view(repo_root: Path) -> Path:
    """The simulator view of the repo's render, ``<render>/data/simulator``.

    Exits with a clear message when the render carries none.
    """
    from osprey.facility.views.simulator import SCENARIOS_FILE
    from osprey.utils.workspace import rendered_config_path

    view = rendered_config_path(repo_root).parent / "data" / "simulator"
    if not (view / SCENARIOS_FILE).is_file():
        output.fail(
            f"No simulator view in {view}",
            "The simulator view is written by the build.",
            "run 'osprey build' first",
        )
        raise SystemExit(1)
    return view


def _read_view_file(view: Path, name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((view / name).read_text(encoding="utf-8"))
    return document


def _served_physics_models(view: Path) -> list[str]:
    """The view's served models whose engine is not ``texture``, sorted by name."""
    from osprey.facility import TEXTURE
    from osprey.facility.views.simulator import SERVED_MODELS_FILE, VARIABLES_FILE

    engines = {
        str(record["name"]): record.get("engine")
        for record in _read_view_file(view, VARIABLES_FILE)["models"]
    }
    return sorted(
        str(name)
        for name in _read_view_file(view, SERVED_MODELS_FILE)["models"]
        if str(name) in engines and engines[str(name)] != TEXTURE
    )


async def _model_statuses(section: dict, target: str | None, models: list[str]) -> dict[str, str]:
    """Each model's status, read from a mock connector built for ``target``."""
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.simulation import model_status

    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(
        section, control_target=target
    )
    try:
        return {model: model_status(connector, model) for model in models}
    finally:
        await connector.disconnect()


def _overlap_records(log: Path) -> list[dict[str, Any]]:
    """The overlap records a model log holds, in file order; none when it is absent."""
    from osprey_connectors.simulation import OVERLAP_EVENT

    try:
        text = log.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("event") == OVERLAP_EVENT:
            records.append(record)
    return records


def _echo_physics_notice(config: dict, rendered: dict[str, str]) -> None:
    """Tell the user a changed physics fault needs a container recreate.

    Called only when the ``.env``'s physics block actually changed -- in either
    direction, since a *cleared* fault is still live in the running container
    until it is recreated.

    Gated on the virtual accelerator being deployed rather than on
    ``control_system.type``: the reference preset's default shape is mock-type
    *with* the VA deployed and bridge-driven, and it is the container, not the
    connector, that consumes these vars at boot. A project that deploys no VA
    has nothing reading them, so the notice stays silent there.
    """
    if "virtual_accelerator" not in (config.get("deployed_services") or []):
        return
    if rendered:
        output.report("Physics fault written to .env: " + ", ".join(sorted(rendered)) + ".")
    else:
        output.report("Cleared the previous scenario's physics fault from .env.")
    output.note("The virtual accelerator reads this only when its container is created.")
    output.note(
        "Run 'osprey up' to recreate it. Restarting the container reuses the old environment."
    )


def _confirm_archive_rewrite(store: dict) -> None:
    """Warn before overwriting stored history, and let the user back out.

    The archive is *stored data*, and the rewrite is not additive: windows a
    previous scenario marked go back to base, and on a virtual-accelerator
    deployment the affected windows may hold samples a recorder took from the
    running machine. That is the documented behaviour — one timeline, and the
    active scenario owns its event windows — but it is not something to do to
    someone's data without saying so first.

    The caller passes the store the preflight already resolved, and skips this
    entirely when the project has none: there is nothing to lose and nothing to
    decide, and a prompt about a store that does not exist trains people to hit
    enter.
    """
    output.report(
        f"This will REWRITE the scenario's event windows in the stored archive "
        f"({store['host']}:{store['port']}/{store['database']}.{store['collection']}), "
        f"restoring any windows a previous scenario touched."
    )
    click.confirm("Continue?", abort=True)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("sim")
def sim_group() -> None:
    """Simulation scenario commands.

    List, inspect, and apply the self-contained scenario bundles that drive the
    mock control system and mock archiver. Applying a set composes their
    telemetry overlays and seeds their logbook entries into ARIEL.
    """


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@sim_group.command("list")
@repo_option
def list_command(repo: Path | None) -> None:
    """List available scenarios (the active set is marked with *)."""
    from osprey.facility.views.simulator import SCENARIOS_FILE
    from osprey_connectors.simulation import (
        ACTIVE_SCENARIOS_FILENAME,
        parse_active_state,
        resolve_active_scenarios,
    )
    from osprey_connectors.workspace import resolve_simulation_state_dir

    repo_root, config = _resolve_deployment(repo)
    view = _require_simulator_view(repo_root)
    state = resolve_simulation_state_dir(config, repo_root) / ACTIVE_SCENARIOS_FILENAME
    names = parse_active_state(state.read_text(encoding="utf-8"))[0] if state.is_file() else []
    active = set(resolve_active_scenarios(names))
    for scenario in _read_view_file(view, SCENARIOS_FILE)["scenarios"]:
        name = str(scenario["name"])
        marker = "*" if name in active else " "
        output.report(f"{marker} {name}  (logbook: {'yes' if scenario.get('logbook') else 'no'})")
        if description := scenario.get("description"):
            # A second step in, on top of the one `note` already applies: the
            # marker column means the name itself does not start at column 0,
            # so a description one step in would line up under the marker
            # rather than under the name it describes.
            output.note(f"  {description}")


@sim_group.command("status")
@repo_option
@click.option(
    "--target",
    type=click.Choice(CONTROL_TARGETS),
    default=None,
    help="The control target to report on. Defaults to the deployment's own.",
)
def status_command(repo: Path | None, target: str | None) -> None:
    """Show each served physics model's status and the log it writes."""
    from osprey_connectors.types import MOCK, resolve_control_system_type, resolve_target

    repo_root, config = _resolve_deployment(repo)
    section = config.get("control_system") or {}
    if target is None:
        connector_type = resolve_control_system_type(section)
    else:
        try:
            connector_type = resolve_target(section, target)
        except ValueError as exc:
            output.fail(f"The {target} target is not configured", str(exc))
            raise SystemExit(1) from None
    if connector_type != MOCK:
        output.fail(
            f"sim status: {connector_type} targets do not report model status yet", mark=False
        )
        raise SystemExit(1)

    models = _served_physics_models(_require_simulator_view(repo_root))
    statuses = asyncio.run(_model_statuses(section, target, models))
    for model in models:
        output.report(f"{model}: {statuses[model]}")

    from osprey_connectors.simulation import format_overlap_record
    from osprey_connectors.simulation.composite import log_dir

    logs = log_dir()
    if logs is None:
        return
    for model in models:
        log = logs / f"{model}.log"
        output.report(f"log: {log}")
        for record in _overlap_records(log):
            output.report(format_overlap_record(model, record))


@sim_group.command("apply")
@repo_option
@click.argument("names", nargs=-1, required=True)
@click.option("--no-seed", is_flag=True, help="Change telemetry only; touch no stored data.")
@click.option("--no-seed-logbook", is_flag=True, help="Leave the logbook database untouched.")
@click.option("--no-seed-archiver", is_flag=True, help="Leave the stored archive untouched.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompts.")
@click.option(
    "--now",
    "now_iso",
    default=None,
    envvar="OSPREY_SIM_NOW",
    metavar="ISO8601",
    help=(
        "Freeze the apply-time anchor T0 to an ISO-8601 instant "
        "(e.g. 2024-03-18T12:00:00) so seeded logbook dates are reproducible. "
        "A naive value takes the facility timezone. Defaults to wall-clock now. "
        "Falls back to the OSPREY_SIM_NOW environment variable."
    ),
)
def apply_command(
    repo: Path | None,
    names: tuple[str, ...],
    no_seed: bool,
    no_seed_logbook: bool,
    no_seed_archiver: bool,
    yes: bool,
    now_iso: str | None,
) -> None:
    """Activate scenarios NAMES and seed their stored data.

    Active scenarios must touch disjoint channel sets. Seeding purges and
    reseeds the ARIEL logbook, and rewrites the affected windows of the stored
    archive, so the narrative and the history both match the active telemetry.
    Use --no-seed-logbook or --no-seed-archiver to leave one of them alone, or
    --no-seed for both.

    A scenario's physics block is rendered into the deployment's .env for the
    virtual accelerator to pick up at its next container boot.
    """
    from osprey.simulation.apply import (
        _require_view_scenarios,
        apply_scenarios,
        compute_scenario_physics_env,
        preflight_archive_rewrite,
        write_scenario_physics_env,
    )
    from osprey.simulation.engine import resolve_active_scenarios
    from osprey_connectors.simulation.engine import resolve_simulation_file
    from osprey_connectors.simulation.state import scenario_targets, validate_composition

    seed_logbook = not (no_seed or no_seed_logbook)
    seed_archive = not (no_seed or no_seed_archiver)
    repo_root, config = _resolve_deployment(repo)
    # After the deployment resolves, never before: a naive --now is stamped with
    # the facility timezone, and that zone is only knowable once this repo's
    # render is the config being read.
    now = _parse_now(now_iso) if now_iso else None
    ariel_config = config.get("ariel")

    # Validate pure, write last: every check that can reject the requested set
    # runs here, ahead of the purge prompt and of the first write, so a
    # collision or an aborted prompt leaves the project completely untouched.
    # A project with no simulation file has neither a composition to judge nor
    # physics to render -- apply_scenarios below raises the canonical
    # "not simulation-backed" error for it, which the handler turns into exit 1.
    machine_path, *_ = resolve_simulation_file(config, repo_root)
    physics: dict[str, str] | None = None
    store: dict | None = None
    if machine_path is not None:
        # The set is judged on the build's simulator view, by the rule the
        # serving composite applies, so the command refuses exactly the sets the
        # simulator would refuse to serve.
        try:
            scenarios = _require_view_scenarios(repo_root)
            overlaps = validate_composition(
                {name: scenario_targets(scenario) for name, scenario in scenarios.items()},
                resolve_active_scenarios(names),
            )
        except ValueError as exc:
            output.fail("Cannot activate these scenarios", str(exc))
            raise SystemExit(1) from None
        if overlaps:
            output.fail("Cannot activate these scenarios", "; ".join(map(str, overlaps)))
            raise SystemExit(1)
        try:
            physics = compute_scenario_physics_env(repo_root, list(names))
        except ValueError as exc:
            output.fail("Cannot activate these scenarios", str(exc))
            raise SystemExit(1) from None

        # The archive rewrite's own refusals belong here too, not inside it: a
        # store whose password the project's .env does not carry, or an event
        # positioned by window fraction, would otherwise be discovered after
        # the scenario is live and the logbook reseeded -- leaving telemetry
        # and narrative saying one thing and the untouched history another.
        if seed_archive:
            try:
                store = preflight_archive_rewrite(repo_root, config, list(names))
            except (ValueError, RuntimeError) as exc:
                output.fail("The stored archive cannot be rewritten", str(exc))
                raise SystemExit(1) from None

    if seed_logbook and not yes and ariel_config:
        from osprey.services.ariel_search.cli_operations import get_purge_info

        try:
            info = asyncio.run(get_purge_info(ariel_config))
        except Exception:
            info = None  # DB unreachable — apply will surface the error below
        if info is not None:
            output.report(
                f"This will PURGE {info.entry_count} existing logbook "
                f"entr{'y' if info.entry_count == 1 else 'ies'} and reseed from the "
                f"active scenarios."
            )
            click.confirm("Continue?", abort=True)

    if seed_archive and not yes and store is not None:
        _confirm_archive_rewrite(store)

    # Past the last abort point: write the physics vars, then say so immediately.
    # Emitting the notice here rather than after apply_scenarios means a failed
    # logbook seed can never swallow it.
    if physics is not None and write_scenario_physics_env(repo_root, physics):
        _echo_physics_notice(config, physics)

    try:
        result = apply_scenarios(
            repo_root,
            list(names),
            seed_logbook=seed_logbook,
            seed_archive=seed_archive,
            now=now,
        )
    except (ValueError, RuntimeError) as exc:
        output.fail("The scenarios could not be applied", str(exc))
        raise SystemExit(1) from None
    except Exception as exc:
        msg = str(exc)
        if "connect" in msg.lower():
            output.fail(
                "Cannot connect to the ARIEL database",
                None,
                "start it with 'osprey up', or pass --no-seed",
            )
            raise SystemExit(1) from None
        raise

    output.report("✓ Active scenarios: " + ", ".join(result.active))
    if not seed_logbook:
        output.note("(logbook unchanged)")
    elif result.logbook_seeded:
        output.report(f"✓ Seeded {result.logbook_seeded} logbook entries (purged and reseeded).")
    elif ariel_config is None:
        output.note("(no ARIEL configured, so the logbook was not seeded)")

    if not seed_archive:
        output.note("(stored archive unchanged)")
    elif result.archiver is not None and not result.archiver.skipped:
        output.report(f"✓ Archive rewritten: {result.archiver.describe()}")
    elif result.archiver is not None:
        output.note(f"({result.archiver.skipped})")
