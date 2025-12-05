"""Capability removal commands for Osprey Framework.

This module provides the 'osprey remove' command group for removing
Osprey capabilities and their associated files from projects.
"""

from pathlib import Path

import click

from .styles import Messages, Styles, console


def is_project_initialized() -> bool:
    """Check if we're in an osprey project directory.

    Returns:
        True if config.yml exists in current directory
    """
    return (Path.cwd() / "config.yml").exists()


def find_capability_file(capability_name: str) -> Path | None:
    """Find the capability file.

    Args:
        capability_name: Name of the capability

    Returns:
        Path to capability file or None if not found
    """
    try:
        from osprey.generators.registry_updater import find_registry_file

        # Try to find from registry location
        registry_path = find_registry_file()
        if registry_path:
            capabilities_dir = registry_path.parent / "capabilities"
            capability_file = capabilities_dir / f"{capability_name}.py"
            if capability_file.exists():
                return capability_file
    except Exception:
        pass

    # Fallback: check simple relative path
    fallback_path = Path(f"capabilities/{capability_name}.py")
    if fallback_path.exists():
        return fallback_path

    return None


@click.group()
def remove():
    """Remove Osprey capabilities and components from your project.

    This command group provides utilities for safely removing capabilities
    that were previously generated or added to your project.

    Available commands:

    \b
      - capability: Remove a capability and all its associated files

    Examples:

    \b
      # Remove a capability interactively
      $ osprey remove capability --name weather_demo

      # Force removal without confirmation
      $ osprey remove capability --name weather_demo --force
    """
    pass


@remove.command()
@click.option(
    "--name",
    "-n",
    "capability_name",
    required=True,
    help="Name of the capability to remove (e.g., weather_demo, slack_mcp)",
)
@click.option("--force", "-f", is_flag=True, help="Force removal without confirmation prompt")
@click.option("--quiet", "-q", is_flag=True, help="Reduce output verbosity")
def capability(capability_name: str, force: bool, quiet: bool):
    """Remove capability from project.

    Removes a capability and all its associated components:

    \b
      - Capability registration from registry.py
      - Context class registration from registry.py
      - Model configuration from config.yml (if present)
      - Capability file

    Backups are automatically created before any modifications.

    Examples:

    \b
      # Remove capability interactively (recommended)
      $ osprey remove capability --name weather_demo

      # Force removal without confirmation
      $ osprey remove capability --name weather_demo --force

      # Quiet mode with forced removal
      $ osprey remove capability -n slack_mcp -f -q
    """
    # Check if we're in a project directory
    if not is_project_initialized():
        console.print(f"\n{Messages.error('Not in an Osprey project directory')}")
        console.print()
        console.print("  This command requires an Osprey project with [accent]config.yml[/accent]")
        console.print()
        console.print("  [bold]Navigate to your project:[/bold]")
        console.print("    " + Messages.command("cd my-project"))
        console.print(
            "    " + Messages.command(f"osprey remove capability --name {capability_name}")
        )
        console.print()
        raise click.Abort()

    if not quiet:
        console.print(
            f"\n🗑️  [{Styles.HEADER}]Removing Capability: {capability_name}[/{Styles.HEADER}]\n"
        )

    try:
        from osprey.generators.config_updater import (
            find_config_file,
            has_capability_react_model,
            remove_capability_react_from_config,
        )
        from osprey.generators.registry_updater import (
            find_registry_file,
            is_already_registered,
            remove_from_registry,
        )

        # Step 1: Discover what exists
        if not quiet:
            console.print(f"[{Styles.HEADER}]Scanning for components...[/{Styles.HEADER}]")

        registry_path = find_registry_file()
        config_path = find_config_file()
        capability_file = find_capability_file(capability_name)

        # Check what we found
        has_registry = registry_path and is_already_registered(registry_path, capability_name)
        has_config = config_path and has_capability_react_model(config_path, capability_name)
        has_file = capability_file is not None

        if not quiet:
            if has_registry:
                console.print(
                    f"  ✓ Registry entries: [{Styles.VALUE}]{registry_path}[/{Styles.VALUE}]"
                )
            else:
                console.print(f"  [{Styles.DIM}]✗ Registry entries: not found[/{Styles.DIM}]")

            if has_config:
                model_key = f"{capability_name}_react"
                console.print(
                    f"  ✓ Config model: [{Styles.VALUE}]{model_key}[/{Styles.VALUE}] in {config_path}"
                )
            else:
                console.print(f"  [{Styles.DIM}]✗ Config model: not found[/{Styles.DIM}]")

            if has_file:
                console.print(
                    f"  ✓ Capability file: [{Styles.VALUE}]{capability_file}[/{Styles.VALUE}]"
                )
            else:
                console.print(f"  [{Styles.DIM}]✗ Capability file: not found[/{Styles.DIM}]")

        # Nothing to remove?
        if not has_registry and not has_config and not has_file:
            console.print(
                f"\n{Messages.warning(f'No components found for capability: {capability_name}')}"
            )
            console.print(
                f"\n  [{Styles.DIM}]Nothing to remove. Capability may already be removed or never existed.[/{Styles.DIM}]"
            )
            return

        # Step 2: Show preview of what will be removed
        if not quiet:
            console.print(f"\n[{Styles.HEADER}]Preview of changes:[/{Styles.HEADER}]")

        # Get previews
        registry_preview = ""
        config_preview = ""

        if has_registry:
            _, registry_preview, _ = remove_from_registry(registry_path, capability_name)
            if not quiet:
                console.print(registry_preview)

        if has_config:
            _, config_preview, _ = remove_capability_react_from_config(config_path, capability_name)
            if not quiet:
                console.print(config_preview)

        if has_file:
            # Get file size
            file_size = capability_file.stat().st_size
            if not quiet:
                console.print("\n[bold]Capability File:[/bold]")
                console.print(f"[red]- DELETE:[/red] {capability_file} ({file_size:,} bytes)")

        # Step 3: Confirm with user
        if not force:
            try:
                import questionary

                from .styles import get_questionary_style

                console.print()
                confirmed = questionary.confirm(
                    "Proceed with removal? (backups will be created)",
                    default=True,
                    style=get_questionary_style(),
                ).ask()

                if not confirmed:
                    console.print(f"\n{Messages.info('Removal cancelled')}")
                    return

            except ImportError:
                # questionary not available, require --force flag
                console.print(
                    f"\n{Messages.warning('Interactive mode requires questionary package')}"
                )
                console.print("  Install with: pip install questionary")
                console.print("  Or use --force flag to proceed without confirmation")
                raise click.Abort() from None

        # Step 4: Perform removals
        console.print()
        if not quiet:
            console.print(f"[{Styles.HEADER}]Removing components...[/{Styles.HEADER}]")

        # Remove from registry
        if has_registry:
            # Create backup
            backup_path = registry_path.with_suffix(".py.bak")
            backup_path.write_text(registry_path.read_text())
            if not quiet:
                console.print(f"  ✓ Created backup: [{Styles.DIM}]{backup_path}[/{Styles.DIM}]")

            # Remove
            new_content, _, _ = remove_from_registry(registry_path, capability_name)
            registry_path.write_text(new_content)
            console.print(f"  {Messages.success(f'Removed from {registry_path.name}')}")

        # Remove from config
        if has_config:
            # Create backup
            backup_path = config_path.with_suffix(".yml.bak")
            backup_path.write_text(config_path.read_text())
            if not quiet:
                console.print(f"  ✓ Created backup: [{Styles.DIM}]{backup_path}[/{Styles.DIM}]")

            # Remove
            new_content, _, _ = remove_capability_react_from_config(config_path, capability_name)
            config_path.write_text(new_content)
            console.print(f"  {Messages.success(f'Removed from {config_path.name}')}")

        # Delete file
        if has_file:
            capability_file.unlink()
            console.print(f"  {Messages.success(f'Deleted: {capability_file}')}")

        # Success!
        console.print("\n" + "=" * 70)
        console.print(
            f"[{Styles.BOLD_SUCCESS}]✅ Capability '{capability_name}' successfully removed![/{Styles.BOLD_SUCCESS}]"
        )
        console.print("=" * 70 + "\n")

        if has_registry or has_config:
            console.print(
                f"[{Styles.DIM}]Backups created in case you need to restore:[/{Styles.DIM}]"
            )
            if has_registry:
                console.print(f"  • {registry_path.with_suffix('.py.bak')}")
            if has_config:
                console.print(f"  • {config_path.with_suffix('.yml.bak')}")
            console.print()

    except KeyboardInterrupt:
        console.print(f"\n{Messages.warning('Removal cancelled by user')}")
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n{Messages.error(f'Removal failed: {e}')}")
        if not quiet:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


if __name__ == "__main__":
    remove()
