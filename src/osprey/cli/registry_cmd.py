"""Registry display for the Osprey CLI (osprey registry)."""

from rich.text import Text

from osprey.cli import output
from osprey.cli.styles import Styles, data_table, panel
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
            if not registry.get_stats()["initialized"]:
                output.note("Initializing registry...")
            # Idempotent -- self-guards when already initialized.
            registry.initialize()

        # Get registry stats
        stats = registry.get_stats()

        # Display header
        output.report("")
        output.table(panel(Text("Registry Contents", style=Styles.HEADER), expand=False))
        output.report("")

        # Display summary
        output.section("Registry Summary", {"Services": stats["services"]})
        output.report("")

        # Display services
        if stats["service_names"]:
            _display_services_table(registry, verbose)

        # Display providers
        providers = registry.list_providers()
        if providers:
            _display_providers_table(registry, providers, verbose)

        output.report("")

    except Exception as e:
        output.fail("Could not display the registry", str(e))
        if verbose:
            import traceback

            traceback.print_exc()
        return False

    return True


def _display_services_table(registry, verbose: bool):
    """Display services in a formatted table."""
    output.report("Services", style=Styles.HEADER)
    output.report("")

    table = data_table(expand=False)
    table.add_column("Name", style=Styles.ACCENT, no_wrap=True)
    table.add_column("Type", style=Styles.VALUE)

    stats = registry.get_stats()
    for name in sorted(stats["service_names"]):
        service = registry.get_service(name)
        service_type = type(service).__name__ if service else "Unknown"
        table.add_row(name, service_type)

    output.table(table)
    output.report("")


def _display_providers_table(registry, providers: list, verbose: bool):
    """Display providers in a formatted table."""
    output.report("AI Providers", style=Styles.HEADER)
    output.report("")

    table = data_table(expand=False)
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

    output.table(table)
    output.report("")
