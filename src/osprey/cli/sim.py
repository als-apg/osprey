"""Simulation scenario CLI commands.

Thin CLI wrappers over the simulation engine and
:func:`osprey.simulation.apply.apply_scenarios`. Run from within a built
project (the project root is the current working directory).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from osprey.utils.config import load_config
from osprey.utils.logger import get_logger

logger = get_logger("sim")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_project_engine():
    """Return (project_dir, config, engine) for the project in the CWD.

    Exits with a clear message if the project is not simulation-backed.
    """
    from osprey.connectors.types import MOCK
    from osprey.simulation.apply import resolve_simulation_file
    from osprey.simulation.engine import SimulationEngine

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
    return project_dir, config, SimulationEngine.from_file(machine_path)


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
@click.option("--no-seed", is_flag=True, help="Change telemetry only; do not touch the logbook DB.")
@click.option("--yes", "-y", is_flag=True, help="Skip the purge confirmation prompt.")
def apply_command(names: tuple[str, ...], no_seed: bool, yes: bool) -> None:
    """Compose and activate scenarios NAMES, seeding their logbook (unless --no-seed).

    Active scenarios must touch disjoint channel sets. Seeding purges and
    reseeds the ARIEL logbook so the narrative matches the active telemetry.
    """
    from osprey.simulation.apply import apply_scenarios

    project_dir = Path.cwd()
    config = load_config(str(project_dir / "config.yml"))
    ariel_config = config.get("ariel")

    if not no_seed and not yes and ariel_config:
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

    try:
        result = apply_scenarios(project_dir, list(names), seed_logbook=not no_seed)
    except ValueError as exc:
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
    if no_seed:
        click.echo("  (telemetry only — logbook unchanged)")
    elif result.logbook_seeded:
        click.echo(f"✓ Seeded {result.logbook_seeded} logbook entries (purged and reseeded).")
    elif ariel_config is None:
        click.echo("  (no ARIEL config — logbook not seeded)")
