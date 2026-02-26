"""Registry Display Command for Osprey CLI.

This module provides functionality to display the current registry contents,
showing all registered components including capabilities, nodes, context classes,
data sources, services, and providers. It provides a formatted view of what
components are available in the current project.

Key Features:
    - Display all registered components by category
    - Show component metadata (descriptions, requirements, etc.)
    - Rich formatted output with sections and tables
    - Support for verbose and compact display modes

Architecture:
    - Uses RegistryManager to access component data
    - Rich library for beautiful terminal formatting
    - Categorized component display with metadata
"""

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
        console.print(
            f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Capabilities: {stats['capabilities']}"
        )
        console.print(f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Nodes: {stats['nodes']}")
        console.print(
            f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Context Classes: {stats['context_classes']}"
        )
        console.print(
            f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Data Sources: {stats['data_sources']}"
        )
        console.print(f"  [{Styles.ACCENT}]•[/{Styles.ACCENT}] Services: {stats['services']}")
        console.print()

        # Display capabilities
        if stats["capability_names"]:
            _display_capabilities_table(registry, verbose)

        # Display infrastructure nodes (filtered to exclude capability nodes)
        if stats["node_names"]:
            _display_nodes_table(registry, verbose)

        # Display context classes (verbose only - redundant with Capabilities "Provides")
        if verbose and stats["context_types"]:
            _display_context_classes_table(registry, verbose)

        # Display data sources
        if stats["data_source_names"]:
            _display_data_sources_table(registry, verbose)

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


def _display_capabilities_table(registry, verbose: bool):
    """Display capabilities in a formatted table."""
    console.print(f"[{Styles.HEADER}]Capabilities[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Provides", style=Styles.VALUE)
    table.add_column("Requires", style=Styles.DIM)

    if verbose:
        table.add_column("Description", style=Styles.DIM)

    capabilities = registry.get_all_capabilities()
    for cap in sorted(capabilities, key=lambda c: c.name):
        # Handle both strings and tuples in provides/requires
        provides = (
            ", ".join(str(p) if isinstance(p, tuple) else p for p in cap.provides)
            if cap.provides
            else "-"
        )
        requires = (
            ", ".join(str(r) if isinstance(r, tuple) else r for r in cap.requires)
            if cap.requires
            else "-"
        )

        if verbose and hasattr(cap, "description"):
            table.add_row(cap.name, provides, requires, cap.description or "")
        else:
            table.add_row(cap.name, provides, requires)

    console.print(table)
    console.print()


def _display_nodes_table(registry, verbose: bool):
    """Display infrastructure nodes in a formatted table.

    Filters out capability nodes to avoid duplication with the Capabilities table,
    showing only framework infrastructure nodes (classifier, orchestrator, router, etc.).
    """
    console.print(f"[{Styles.HEADER}]Infrastructure Nodes[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Type", style=Styles.VALUE)

    # Get capability names to filter them out
    capability_names = {cap.name for cap in registry.get_all_capabilities()}

    # Only show infrastructure nodes (non-capability nodes)
    nodes = registry.get_all_nodes()
    infrastructure_nodes = {
        name: node for name, node in nodes.items() if name not in capability_names
    }

    for name, node in sorted(infrastructure_nodes.items()):
        node_type = type(node).__name__ if node else "Unknown"
        table.add_row(name, node_type)

    console.print(table)
    console.print()


def _display_context_classes_table(registry, verbose: bool):
    """Display context classes in a formatted table."""
    console.print(f"[{Styles.HEADER}]Context Classes[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Context Type", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Class Name", style=Styles.VALUE)

    context_classes = registry.get_all_context_classes()
    for context_type, context_class in sorted(context_classes.items()):
        class_name = context_class.__name__ if context_class else "Unknown"
        table.add_row(context_type, class_name)

    console.print(table)
    console.print()


def _display_data_sources_table(registry, verbose: bool):
    """Display data sources in a formatted table."""
    console.print(f"[{Styles.HEADER}]Data Sources[/{Styles.HEADER}]\n")

    table = Table(
        show_header=True, header_style=Styles.HEADER, border_style=Styles.DIM, expand=False
    )

    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Type", style=Styles.VALUE)

    stats = registry.get_stats()
    for name in sorted(stats["data_source_names"]):
        ds = registry.get_data_source(name)
        ds_type = type(ds).__name__ if ds else "Unknown"
        table.add_row(name, ds_type)

    console.print(table)
    console.print()


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
