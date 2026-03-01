"""Registry display for the Osprey CLI (osprey registry)."""

from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from osprey.cli.styles import Messages, Styles, ThemeConfig, console
from osprey.registry import get_registry


def display_registry_contents(verbose: bool = False):
    """Display the contents of the current registry.

    Args:
        verbose: Whether to display verbose information (descriptions, etc.)
    """
    try:
        from osprey.utils.log_filter import quiet_logger

        # Get registry (initialize if needed) - suppress initialization logs
        with quiet_logger(["registry", "CONFIG"]):
            registry = get_registry()
            if not registry._initialized:
                console.print("\n[dim]Initializing registry...[/dim]")
                registry.initialize()

        # Get registry stats
        stats = registry.get_stats()

        # Display header
        console.print()
        console.print(
            Panel(
                Text("Registry Contents", style=Styles.HEADER),
                border_style=ThemeConfig.get_border_style(),
                expand=False,
            )
        )
        console.print()

        # Display summary
        console.print(f"[{Styles.HEADER}]Registry Summary[/{Styles.HEADER}]")
        console.print(f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Services: {stats['services']}")
        console.print()

        # Display services
        if stats["service_names"]:
            _display_services_table(registry, verbose)

        # Display providers
        providers = registry.list_providers()
        if providers:
            _display_providers_table(registry, providers, verbose)

        console.print()

    except Exception as e:
        console.print(Messages.error(f"Error displaying registry: {e}"))
        if verbose:
            import traceback

            traceback.print_exc()
        return False

    return True


def _display_services_table(registry, verbose: bool):
    """Display services in a formatted table."""
    console.print(f"[{Styles.HEADER}]Services[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Type", style=Styles.VALUE)

    stats = registry.get_stats()
    for name in sorted(stats["service_names"]):
        service = registry.get_service(name)
        service_type = type(service).__name__ if service else "Unknown"
        table.add_row(name, service_type)

    console.print(table)
    console.print()


def _display_providers_table(registry, providers: list, verbose: bool):
    """Display providers in a formatted table."""
    console.print(f"[{Styles.HEADER}]AI Providers[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Available", style=Styles.VALUE)

    if verbose:
        table.add_column("Description", style=Styles.DIM)

    for provider_name in sorted(providers):
        provider_class = registry.get_provider(provider_name)

        if provider_class:
            # Try to get metadata from the class
            available = "✓" if provider_class else "✗"

            if verbose and hasattr(provider_class, "description"):
                description = getattr(provider_class, "description", "")
                table.add_row(provider_name, available, description)
            else:
                table.add_row(provider_name, available)
        else:
            table.add_row(provider_name, "✗")

    console.print(table)
    console.print()


def handle_registry_action(project_path: Path | None = None, verbose: bool = False):
    """Handle registry display action from interactive menu.

    Args:
        project_path: Optional project directory path (defaults to current directory)
        verbose: Whether to show verbose output
    """
    import os

    # Save and optionally change directory
    original_dir = None
    if project_path:
        original_dir = Path.cwd()

        try:
            os.chdir(project_path)
        except (OSError, PermissionError) as e:
            console.print(f"\n{Messages.error(f'Cannot change to project directory: {e}')}")
            input("\nPress ENTER to continue...")
            return

    try:
        # Display registry contents
        display_registry_contents(verbose=verbose)

    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        if verbose:
            import traceback

            traceback.print_exc()
    finally:
        # Restore original directory
        if original_dir:
            try:
                os.chdir(original_dir)
            except (OSError, PermissionError) as e:
                console.print(f"\n{Messages.warning(f'Could not restore directory: {e}')}")

    input("\nPress ENTER to continue...")
