"""Configuration management commands.

This module provides the 'osprey config' command group for managing project
configuration. All configuration-related operations are unified under this
namespace following industry standard CLI patterns (git config, docker config, etc.).

Commands:
    - config show: Display current project configuration
    - config export: Export framework default configuration
    - config set-control-system: Switch between Mock/EPICS/Tango connectors
    - config set-epics-gateway: Configure EPICS gateway settings
"""

import sys
from pathlib import Path

import click
import yaml
from jinja2 import Template
from rich.syntax import Syntax

from osprey.cli.styles import Styles, console


@click.group(name="config", invoke_without_command=True)
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)"
)
@click.pass_context
def config(ctx, project):
    """Manage project configuration.

    Configuration commands for viewing, exporting, and modifying project settings.
    All commands work with the project's config.yml file.

    If no subcommand is provided, launches interactive configuration menu.

    Note: Most subcommands require a project directory. Only 'export' works without a project.

    Examples:

    \b
      # Launch interactive config menu (requires project)
      osprey config

      # Display current configuration (requires project)
      osprey config show

      # Export framework defaults (works anywhere)
      osprey config export -o defaults.yml

      # Switch to EPICS control system (requires project)
      osprey config set-control-system epics

      # Configure EPICS gateway (requires project)
      osprey config set-epics-gateway --facility als
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided - launch interactive menu
        # This requires a project directory
        try:
            from .interactive_menu import handle_config_action
            from .project_utils import resolve_config_path, resolve_project_path

            # Check if we're in a project directory
            try:
                project_path = resolve_project_path(project)
                config_path_str = resolve_config_path(project)
                config_path = Path(config_path_str)

                if not config_path.exists():
                    console.print(
                        "❌ No Osprey project found in current directory",
                        style=Styles.ERROR
                    )
                    console.print(
                        f"   Looking for: {config_path}",
                        style=Styles.DIM
                    )
                    console.print(
                        "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                        style=Styles.DIM
                    )
                    console.print(
                        "   Or run from a project directory",
                        style=Styles.DIM
                    )
                    sys.exit(1)

            except Exception:
                console.print(
                    "❌ No Osprey project found",
                    style=Styles.ERROR
                )
                console.print(
                    "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                    style=Styles.DIM
                )
                console.print(
                    "   Or run from a project directory containing config.yml",
                    style=Styles.DIM
                )
                sys.exit(1)

            # Launch interactive menu (shared implementation)
            handle_config_action(project_path)

        except KeyboardInterrupt:
            console.print("\n⚠️  Operation cancelled", style=Styles.WARNING)
            sys.exit(0)
        except SystemExit:
            raise  # Re-raise sys.exit() calls
        except Exception as e:
            console.print(f"❌ Failed to launch config menu: {e}", style=Styles.ERROR)
            import os
            if os.environ.get("DEBUG"):
                import traceback
                console.print(traceback.format_exc(), style=Styles.DIM)
            sys.exit(1)


@config.command(name="show")
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)"
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format (default: yaml)"
)
def show(project: str, format: str):
    """Display current project configuration.

    Shows the active configuration for the current project with syntax highlighting.
    Useful for debugging and understanding current settings.

    Requires: Must be run from a project directory containing config.yml

    Examples:

    \b
      # Show current project's config
      osprey config show

      # Show specific project's config
      osprey config show --project ~/my-agent

      # Export as JSON
      osprey config show --format json
    """
    try:
        from .project_utils import resolve_config_path

        # Resolve config path (returns string)
        try:
            config_path_str = resolve_config_path(project)
            config_path = Path(config_path_str)
        except Exception as e:
            console.print(
                "❌ No Osprey project found",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            console.print(
                "   Or run from a project directory containing config.yml",
                style=Styles.DIM
            )
            raise click.Abort()

        if not config_path.exists():
            console.print(
                f"❌ Configuration file not found: {config_path}",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            raise click.Abort()

        # Load configuration
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Format output
        if format == "yaml":
            output_str = yaml.dump(
                config_data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
        else:  # json
            import json
            output_str = json.dumps(config_data, indent=2, ensure_ascii=False)

        # Display with syntax highlighting
        console.print(f"\n[bold]Configuration:[/bold] {config_path}\n")
        syntax = Syntax(
            output_str,
            format,
            theme="monokai",
            line_numbers=False,
            word_wrap=True
        )
        console.print(syntax)

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled", style=Styles.WARNING)
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Failed to show configuration: {e}", style=Styles.ERROR)
        raise click.Abort()


@config.command(name="export")
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file (default: print to console)"
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format (default: yaml)"
)
def export(output: str, format: str):
    """Export framework default configuration template.

    Displays the Osprey framework's default configuration template that is used
    when creating new projects with 'osprey init'. This is useful for:

    \b
      - Understanding available configuration options
      - Seeing default values for models, services, etc.
      - Debugging configuration issues
      - Creating custom configurations

    Examples:

    \b
      # Display to console with syntax highlighting
      osprey config export

      # Save to file
      osprey config export -o defaults.yml

      # Export as JSON
      osprey config export --format json -o defaults.json
    """
    try:
        # Load osprey's configuration template
        template_path = Path(__file__).parent.parent / "templates" / "project" / "config.yml.j2"

        if not template_path.exists():
            console.print(
                "❌ Could not locate Osprey configuration template.",
                style=Styles.ERROR
            )
            console.print(
                f"   Expected at: {template_path}",
                style=Styles.DIM
            )
            raise click.Abort()

        # Read and render the template with example values
        with open(template_path) as f:
            template_content = f.read()

        template = Template(template_content)
        rendered_config = template.render(
            project_name="example_project",
            package_name="example_project",
            project_root="/path/to/example_project",
            hostname="localhost",
            default_provider="cborg",
            default_model="anthropic/claude-haiku"
        )

        # Parse the rendered config as YAML
        config_data = yaml.safe_load(rendered_config)

        # Format output based on requested format
        if format == "yaml":
            output_str = yaml.dump(
                config_data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
        else:  # json
            import json
            output_str = json.dumps(config_data, indent=2, ensure_ascii=False)

        # Output to file or console
        if output:
            output_path = Path(output)
            output_path.write_text(output_str)
            console.print(
                f"✅ Configuration exported to: [bold]{output_path}[/bold]"
            )
        else:
            # Print to console with syntax highlighting
            console.print("\n[bold]Osprey Framework Default Configuration:[/bold]\n")
            syntax = Syntax(
                output_str,
                format,
                theme="monokai",
                line_numbers=False,
                word_wrap=True
            )
            console.print(syntax)
            console.print(
                "\n[dim]💡 Tip: Save to file with --output flag[/dim]"
            )

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled", style=Styles.WARNING)
        raise click.Abort()
    except Exception as e:
        console.print(f"❌ Failed to export configuration: {e}", style=Styles.ERROR)
        import os
        if os.environ.get("DEBUG"):
            import traceback
            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort()


@config.command(name="set-control-system")
@click.argument(
    "system_type",
    type=click.Choice(["mock", "epics", "tango", "labview"], case_sensitive=False)
)
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)"
)
def set_control_system(system_type: str, project: str):
    """Switch control system connector type.

    Changes the control_system.type setting in config.yml. This determines which
    connector is used at runtime for control system operations.

    Note: Pattern detection is control-system-agnostic (same for all types).
    This setting only affects which connector is loaded at runtime.

    Requires: Must be run from a project directory containing config.yml

    Examples:

    \b
      # Switch to mock mode (development)
      osprey config set-control-system mock

      # Switch to EPICS (production)
      osprey config set-control-system epics

      # Switch to Tango
      osprey config set-control-system tango
    """
    try:
        from .project_utils import resolve_config_path
        from osprey.generators.config_updater import update_control_system_type

        try:
            config_path_str = resolve_config_path(project)
            config_path = Path(config_path_str)
        except Exception:
            console.print(
                "❌ No Osprey project found",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            console.print(
                "   Or run from a project directory containing config.yml",
                style=Styles.DIM
            )
            raise click.Abort()

        if not config_path.exists():
            console.print(
                f"❌ Configuration file not found: {config_path}",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            raise click.Abort()

        # Update configuration
        update_control_system_type(config_path, system_type.lower())

        console.print(
            f"✅ Control system type updated to: [bold]{system_type}[/bold]"
        )
        console.print(
            f"   Configuration: {config_path}",
            style=Styles.DIM
        )

    except Exception as e:
        console.print(f"❌ Failed to update control system: {e}", style=Styles.ERROR)
        raise click.Abort()


@config.command(name="set-epics-gateway")
@click.option(
    "--facility",
    type=click.Choice(["als", "aps", "custom"], case_sensitive=False),
    help="Facility preset (als, aps, or custom for manual entry)"
)
@click.option(
    "--address",
    help="Gateway address (for custom facility)"
)
@click.option(
    "--port",
    type=int,
    help="Gateway port (for custom facility)"
)
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)"
)
def set_epics_gateway(facility: str, address: str, port: int, project: str):
    """Configure EPICS gateway settings.

    Sets the EPICS gateway address and port in config.yml. Can use facility presets
    (ALS, APS) or specify custom gateway settings.

    Requires: Must be run from a project directory containing config.yml

    Examples:

    \b
      # Use ALS gateway preset
      osprey config set-epics-gateway --facility als

      # Use APS gateway preset
      osprey config set-epics-gateway --facility aps

      # Set custom gateway
      osprey config set-epics-gateway --facility custom \\
          --address gateway.example.com --port 5064
    """
    try:
        from .project_utils import resolve_config_path
        from osprey.generators.config_updater import update_epics_gateway

        try:
            config_path_str = resolve_config_path(project)
            config_path = Path(config_path_str)
        except Exception:
            console.print(
                "❌ No Osprey project found",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            console.print(
                "   Or run from a project directory containing config.yml",
                style=Styles.DIM
            )
            raise click.Abort()

        if not config_path.exists():
            console.print(
                f"❌ Configuration file not found: {config_path}",
                style=Styles.ERROR
            )
            console.print(
                "\n💡 Create a new project with: [bold cyan]osprey init my-project[/bold cyan]",
                style=Styles.DIM
            )
            raise click.Abort()

        if facility == "custom" and (not address or not port):
            console.print(
                "❌ Custom facility requires --address and --port",
                style=Styles.ERROR
            )
            raise click.Abort()

        # Update configuration
        update_epics_gateway(config_path, facility, address, port)

        console.print(
            f"✅ EPICS gateway updated"
        )
        console.print(
            f"   Configuration: {config_path}",
            style=Styles.DIM
        )

    except Exception as e:
        console.print(f"❌ Failed to update EPICS gateway: {e}", style=Styles.ERROR)
        raise click.Abort()


if __name__ == "__main__":
    config()

