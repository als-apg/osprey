"""Simulation scenario CLI commands.

Thin CLI wrappers over the simulation engine and
:func:`osprey.simulation.apply.apply_scenarios`. Run from within a built
project (the project root is the current working directory).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import click

from osprey.utils.config import load_config
from osprey.utils.logger import get_logger

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
        click.echo(f"Error: --now value {now_iso!r} is not valid ISO-8601.", err=True)
        raise SystemExit(1) from None
    if anchor.tzinfo is None:
        from osprey.utils.config import get_facility_timezone

        anchor = anchor.replace(tzinfo=get_facility_timezone())
    return anchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_project_engine():
    """Return (project_dir, config, engine) for the project in the CWD.

    Exits with a clear message if the project is not simulation-backed.
    """
    from osprey.connectors.types import MOCK
    from osprey.simulation.apply import resolve_simulation_file
    from osprey.simulation.engine import SimulationEngine, resolve_state_dir

    project_dir = Path.cwd()
    config_path = project_dir / "config.yml"
    if not config_path.is_file():
        click.echo(f"Error: no config.yml in {project_dir}.", err=True)
        click.echo("Run this from a built project's root directory.", err=True)
        raise SystemExit(1)

    config = load_config(str(config_path))
    machine_path, active_type, type_key, mock_key = resolve_simulation_file(config, project_dir)
    if machine_path is None:
        if active_type == MOCK:
            click.echo("Error: no mock 'simulation_file' configured in config.yml.", err=True)
        else:
            click.echo(
                f"Error: no simulation_file configured for control_system.type "
                f"'{active_type}' (tried {type_key} and {mock_key}).",
                err=True,
            )
        click.echo("This project does not use the simulation engine.", err=True)
        raise SystemExit(1)
    engine = SimulationEngine.from_file(
        machine_path, state_dir=resolve_state_dir(config, project_dir)
    )
    return project_dir, config, engine


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
        click.echo("! Physics fault rendered to .env: " + ", ".join(sorted(rendered)) + ".")
    else:
        click.echo("! Cleared the previous scenario's physics fault from .env.")
    click.echo("  The virtual accelerator reads this only at container boot — run")
    click.echo("  'osprey deploy up' to recreate it. A plain 'docker restart' keeps")
    click.echo("  the old environment.")


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
    click.echo(
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
def list_command() -> None:
    """List available scenarios (active set marked with ``*``)."""
    _, _, engine = _load_project_engine()
    active = set(engine.active_scenarios())
    for name, description in engine.list_scenarios().items():
        has_log = len(engine.scenario_logbook(name)) > 0
        marker = "*" if name in active else " "
        click.echo(f"{marker} {name}  (logbook: {'yes' if has_log else 'no'})")
        if description:
            click.echo(f"    {description}")


@sim_group.command("status")
def status_command() -> None:
    """Show the currently active scenario set."""
    _, _, engine = _load_project_engine()
    active = engine.active_scenarios()
    click.echo("Active scenarios: " + ", ".join(active))
    logbook = engine.active_logbook()
    click.echo(f"Composed logbook entries: {len(logbook)}")


@sim_group.command("apply")
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

    A scenario's physics block is rendered into the project .env for the
    virtual accelerator to pick up at its next container boot.
    """
    from osprey.simulation.apply import (
        apply_scenarios,
        compute_scenario_physics_env,
        preflight_archive_rewrite,
        resolve_simulation_file,
        write_scenario_physics_env,
    )
    from osprey.simulation.engine import resolve_active_scenarios

    now = _parse_now(now_iso) if now_iso else None
    seed_logbook = not (no_seed or no_seed_logbook)
    seed_archive = not (no_seed or no_seed_archiver)
    project_dir = Path.cwd()
    config = load_config(str(project_dir / "config.yml"))
    ariel_config = config.get("ariel")

    # Validate pure, write last: every check that can reject the requested set
    # runs here, ahead of the purge prompt and of the first write, so a
    # collision or an aborted prompt leaves the project completely untouched.
    # A project with no simulation file has neither an engine to validate nor
    # physics to render -- apply_scenarios below raises the canonical
    # "not simulation-backed" error for it, which the handler turns into exit 1.
    machine_path, *_ = resolve_simulation_file(config, project_dir)
    physics: dict[str, str] | None = None
    store: dict | None = None
    if machine_path is not None:
        from osprey.simulation.engine import SimulationEngine, resolve_state_dir

        engine = SimulationEngine.from_file(
            machine_path, state_dir=resolve_state_dir(config, project_dir)
        )
        # validate_composition RETURNS its problems (unknown names, channel
        # collisions) rather than raising; an empty list is the only "OK".
        problems = engine.validate_composition(resolve_active_scenarios(names))
        if problems:
            click.echo("Error: cannot activate scenarios: " + "; ".join(problems), err=True)
            raise SystemExit(1)
        try:
            physics = compute_scenario_physics_env(project_dir, list(names))
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from None

        # The archive rewrite's own refusals belong here too, not inside it: a
        # store whose password the project's .env does not carry, or an event
        # positioned by window fraction, would otherwise be discovered after
        # the scenario is live and the logbook reseeded -- leaving telemetry
        # and narrative saying one thing and the untouched history another.
        if seed_archive:
            try:
                store = preflight_archive_rewrite(project_dir, config, machine_path, list(names))
            except (ValueError, RuntimeError) as exc:
                click.echo(f"Error: {exc}", err=True)
                raise SystemExit(1) from None

    if seed_logbook and not yes and ariel_config:
        from osprey.services.ariel_search.cli_operations import get_purge_info

        try:
            info = asyncio.run(get_purge_info(ariel_config))
        except Exception:
            info = None  # DB unreachable — apply will surface the error below
        if info is not None:
            click.echo(
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
    if physics is not None and write_scenario_physics_env(project_dir, physics):
        _echo_physics_notice(config, physics)

    try:
        result = apply_scenarios(
            project_dir,
            list(names),
            seed_logbook=seed_logbook,
            seed_archive=seed_archive,
            now=now,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None
    except Exception as exc:
        msg = str(exc)
        if "connect" in msg.lower():
            click.echo("Error: cannot connect to the ARIEL database.", err=True)
            click.echo("Start it with 'osprey deploy up', or pass --no-seed.", err=True)
            raise SystemExit(1) from None
        raise

    click.echo("✓ Active scenarios: " + ", ".join(result.active))
    if not seed_logbook:
        click.echo("  (logbook unchanged)")
    elif result.logbook_seeded:
        click.echo(f"✓ Seeded {result.logbook_seeded} logbook entries (purged and reseeded).")
    elif ariel_config is None:
        click.echo("  (no ARIEL config — logbook not seeded)")

    if not seed_archive:
        click.echo("  (stored archive unchanged)")
    elif result.archiver is not None and not result.archiver.skipped:
        click.echo(f"✓ Archive rewritten: {result.archiver.describe()}")
    elif result.archiver is not None:
        click.echo(f"  ({result.archiver.skipped})")
