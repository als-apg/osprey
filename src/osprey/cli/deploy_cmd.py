"""Service deployment CLI command wrapping osprey.deployment.container_manager."""

import os

import click

from osprey.cli.styles import Styles, console
from osprey.deployment.container_manager import (
    clean_deployment,
    deploy_down,
    deploy_restart,
    deploy_up,
    prepare_compose_files,
    rebuild_deployment,
    show_status,
)

from .project_utils import resolve_config_path, resolve_project_path


@click.command()
@click.argument(
    "action",
    type=click.Choice(
        [
            "up",
            "down",
            "restart",
            "status",
            "build",
            "clean",
            "rebuild",
            "decommission",
            "prune",
            "nuke",
            "seed",
        ]
    ),
)
@click.argument("user", required=False)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="config.yml",
    help="Configuration file (default: config.yml in project directory)",
)
@click.option(
    "--detached",
    "-d",
    is_flag=True,
    help="Run services in detached mode (for up, restart, rebuild)",
)
@click.option(
    "--dev",
    is_flag=True,
    help="Development mode: copy local osprey package to containers instead of using PyPI version. Use this when testing local osprey changes.",
)
@click.option(
    "--expose",
    is_flag=True,
    help="Expose services to all network interfaces (0.0.0.0). WARNING: This exposes services to the network! Only use with proper authentication configured.",
)
@click.option(
    "--archive",
    is_flag=True,
    help="Archive a user's workspace before removing it (decommission/prune only).",
)
@click.option(
    "--purge",
    is_flag=True,
    help="Permanently delete a user's workspace without archiving it (decommission/prune only).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Assume yes to confirmation prompts (decommission/prune/nuke).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without making changes (prune only).",
)
def deploy(
    action: str,
    user: str | None,
    project: str,
    config: str,
    detached: bool,
    dev: bool,
    expose: bool,
    archive: bool,
    purge: bool,
    yes: bool,
    dry_run: bool,
):
    """Manage Docker/Podman services for Osprey projects.

    This command wraps the existing container management functionality,
    providing control over service deployment, status, and cleanup, plus
    per-user web-terminal lifecycle management.

    Actions:

    \b
      up            - Start all configured services
      down          - Stop all services
      restart       - Restart all services
      status        - Show service status
      build         - Build/prepare compose files without starting services
      clean         - Remove containers and volumes (WARNING: destructive)
      rebuild       - Clean, rebuild, and restart services
      decommission  - Remove a single user's web-terminal workspace (requires USER)
      prune         - Remove workspaces for users no longer in the user index
      nuke          - Tear down the entire multi-user web-terminal stack (WARNING: destructive)
      seed          - (Re)seed web-terminal workspaces from the user index; optional
                      USER targets one user, omit to reseed all

    The services to deploy are defined in your config.yml under
    the 'deployed_services' key.

    Lifecycle flags:

    \b
      --archive     Archive a user's workspace before removing it
                     (decommission/prune only; mutually exclusive with --purge)
      --purge       Permanently delete a user's workspace, no archive
                     (decommission/prune only; mutually exclusive with --archive)
      --yes, -y     Assume yes to confirmation prompts (decommission/prune/nuke)
      --dry-run     Show what would happen without making changes (prune only)

    Examples:

    \b
      # Start services in current directory
      $ osprey deploy up

      # Start services in specific project
      $ osprey deploy up --project ~/projects/my-agent

      # Start in background (detached mode)
      $ osprey deploy up -d

      # Start with local osprey for development/testing
      $ osprey deploy up --dev

      # Stop services
      $ osprey deploy down

      # Check status
      $ osprey deploy status

      # Build compose files without starting services
      $ osprey deploy build

      # Use environment variable
      $ export OSPREY_PROJECT=~/projects/my-agent
      $ osprey deploy up

      # Use custom config
      $ osprey deploy up --config my-config.yml

      # Clean everything (removes data!)
      $ osprey deploy clean

      # Rebuild with local osprey for development
      $ osprey deploy rebuild --dev

      # Remove a single user's workspace, archiving it first
      $ osprey deploy decommission alice --archive

      # Preview which stale user workspaces would be pruned
      $ osprey deploy prune --dry-run

      # Prune stale user workspaces, purging without archiving
      $ osprey deploy prune --purge --yes

      # Tear down the entire multi-user stack
      $ osprey deploy nuke --yes

      # Reseed a single user's workspace from the user index
      $ osprey deploy seed alice

      # Reseed every user's workspace from the user index
      $ osprey deploy seed
    """
    # Argument validation happens BEFORE any project/config resolution or
    # lazy import so it is exercised even when the lifecycle modules (or a
    # project directory) don't exist yet.
    if action == "decommission" and not user:
        raise click.UsageError(
            f"'{action}' requires a USER argument, e.g. osprey deploy {action} alice"
        )

    if archive and purge:
        raise click.UsageError("--archive and --purge are mutually exclusive.")

    if (archive or purge) and action not in ("decommission", "prune"):
        flag = "--archive" if archive else "--purge"
        raise click.UsageError(
            f"{flag} is only valid for 'decommission' and 'prune', not '{action}'."
        )

    if dry_run and action != "prune":
        raise click.UsageError(f"--dry-run is only valid for 'prune', not '{action}'.")

    # A stray USER on an action that doesn't consume it is a typo, not a no-op:
    # only decommission/seed take a target user (the require-USER guard above
    # covers the missing-user case for those two).
    if user and action not in ("decommission", "seed"):
        raise click.UsageError(f"'{action}' does not take a USER argument (got {user!r}).")

    if action != "status":
        console.print(f"Service management: [bold]{action}[/bold]")

    try:
        # Resolve project directory and chdir into it so that all
        # CWD-relative operations (template loading, .env lookup,
        # build/ output) resolve against the project root.
        project_dir = resolve_project_path(project)
        os.chdir(project_dir)

        # Resolve config path from project and config args
        config_path = resolve_config_path(project, config)

        # Validate config file exists with helpful error message
        from pathlib import Path

        config_file = Path(config_path)
        if not config_file.exists():
            console.print(
                f"\n✗ Configuration file not found: [accent]{config_path}[/accent]",
                style=Styles.ERROR,
            )
            console.print("\nHint: Are you in a project directory?", style=Styles.WARNING)
            console.print(f"   Current directory: [dim]{Path.cwd()}[/dim]\n")

            # Look for nearby project directories with config.yml
            # Exclude common non-project directories
            excluded_dirs = {
                "docs",
                "tests",
                "test",
                "build",
                "dist",
                "venv",
                ".venv",
                "node_modules",
                ".git",
                "__pycache__",
                "src",
                "lib",
            }
            nearby_projects = []
            try:
                for item in Path.cwd().iterdir():
                    if (
                        item.is_dir()
                        and item.name not in excluded_dirs
                        and not item.name.startswith(".")
                        and (item / "config.yml").exists()
                    ):
                        nearby_projects.append(item.name)
            except PermissionError:
                pass  # Skip if can't read directory

            if nearby_projects:
                console.print("   Found project(s) in current directory:", style=Styles.WARNING)
                for proj in nearby_projects[:5]:  # Limit to 5 suggestions
                    console.print(
                        f"     • [command]cd {proj} && osprey deploy {action}[/command] or: "
                    )
                    console.print(
                        f"       [command]osprey deploy {action} --project {proj}[/command]"
                    )
            else:
                console.print("   Try:", style=Styles.WARNING)
                console.print("     • Navigate to your project directory first")
                console.print(
                    "     • Use [command]--project[/command] flag to specify project location"
                )

            console.print("\n   Or use interactive menu: [command]osprey[/command]\n")
            raise click.Abort()

        if action == "up":
            deploy_up(config_path, detached=detached, dev_mode=dev, expose_network=expose)

        elif action == "down":
            deploy_down(config_path, dev_mode=dev)

        elif action == "restart":
            deploy_restart(config_path, detached=detached, expose_network=expose)

        elif action == "status":
            show_status(config_path, console=console, styles=Styles)

        elif action == "build":
            # Just prepare compose files without starting services
            console.print("Building compose files...")
            _, compose_files = prepare_compose_files(
                config_path, dev_mode=dev, expose_network=expose
            )
            console.print("\n✓ Compose files built successfully:")
            for compose_file in compose_files:
                console.print(f"  • {compose_file}")

        elif action == "clean":
            # clean_deployment expects compose_files list, so prepare them first
            _, compose_files = prepare_compose_files(
                config_path, dev_mode=dev, expose_network=expose
            )
            clean_deployment(compose_files)

        elif action == "rebuild":
            rebuild_deployment(config_path, detached=detached, dev_mode=dev, expose_network=expose)

        elif action == "decommission":
            from osprey.deployment.web_terminals.lifecycle import decommission_user

            decommission_user(config_path, user, archive=archive, purge=purge, assume_yes=yes)

        elif action == "prune":
            from osprey.deployment.web_terminals.lifecycle import prune_users

            prune_users(config_path, dry_run=dry_run, archive=archive, purge=purge, assume_yes=yes)

        elif action == "nuke":
            from osprey.deployment.web_terminals.lifecycle import nuke_stack

            nuke_stack(config_path, assume_yes=yes)

        elif action == "seed":
            from osprey.deployment.web_terminals.seeding import seed_web_terminals

            seed_web_terminals(config_path, user)

    except KeyboardInterrupt:
        console.print("\n!  Operation cancelled by user", style=Styles.WARNING)
        raise click.Abort() from None
    except Exception as e:
        console.print(f"✗ Deployment failed: {e}", style=Styles.ERROR)
        # Show more details in verbose mode
        if os.environ.get("DEBUG"):
            import traceback

            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from None


if __name__ == "__main__":
    deploy()
