"""Project action handlers for the interactive menu.

This module contains handlers for deploy, health, tasks, export,
and configuration management actions invoked from the TUI menu.
"""

import os
import socket
from pathlib import Path

from osprey.cli.styles import (
    Messages,
    console,
    get_questionary_style,
)
from osprey.connectors import types

try:
    import questionary
    from questionary import Choice
except ImportError:
    questionary = None
    Choice = None


custom_style = get_questionary_style()


def handle_project_selection(project_path: Path):
    """Handle selection of a discovered project from subdirectory.

    Shows project-specific menu in a loop until user chooses to go back.

    Args:
        project_path: Path to the selected project directory
    """
    from osprey.cli.init_wizard import run_interactive_init
    from osprey.cli.interactive_menu import get_project_info, get_project_menu_choices
    from osprey.cli.menu_display import handle_help_action, show_banner

    project_name = project_path.name
    project_info = get_project_info(project_path / "config.yml")

    # Loop to keep showing project menu after actions complete
    while True:
        console.clear()
        show_banner(context="interactive")

        console.print(f"\n{Messages.header('Selected Project:')} {project_name}")
        console.print(f"[dim]Location: {Messages.path(str(project_path))}[/dim]")

        if project_info:
            console.print(
                f"[dim]Provider: {project_info.get('provider', 'unknown')} | "
                f"Model: {project_info.get('model', 'unknown')}[/dim]\n"
            )

        # Use centralized project menu choices (with 'back' action)
        action = questionary.select(
            "Select command:",
            choices=get_project_menu_choices(exit_action="back"),
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            return  # Exit the loop and return to main menu

        # Execute the selected action with the project path
        if action == "deploy":
            handle_deploy_action(project_path=project_path)
        elif action == "health":
            handle_health_action(project_path=project_path)
        elif action == "config":
            handle_config_action(project_path=project_path)
        elif action == "registry":
            from osprey.cli.registry_cmd import handle_registry_action

            handle_registry_action(project_path=project_path)
        elif action == "tasks":
            handle_tasks_action()
        elif action == "init_interactive":
            # Save current directory before init flow
            original_dir = Path.cwd()
            try:
                # Init can run from anywhere, but we restore directory after
                next_action = run_interactive_init()
                if next_action == "exit":
                    # User chose to exit after init, return to main menu instead
                    return
            finally:
                try:
                    os.chdir(original_dir)
                except (OSError, PermissionError):
                    pass
        elif action == "help":
            handle_help_action()

        # After action completes, loop continues and shows project menu again


def handle_deploy_action(project_path: Path | None = None):
    """Manage deployment services menu.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    from osprey.cli.menu_display import show_deploy_help

    # Loop to allow returning to menu after help
    while True:
        action = questionary.select(
            "Select deployment action:",
            choices=[
                Choice("[^] up      - Start all services", value="up"),
                Choice("[v] down    - Stop all services", value="down"),
                Choice("[i] status  - Show service status", value="status"),
                Choice("[*] restart - Restart all services", value="restart"),
                Choice("[+] build   - Build/prepare compose files only", value="build"),
                Choice("[R] rebuild - Clean, rebuild, and restart services", value="rebuild"),
                Choice(
                    "[X] clean   - Remove containers and volumes (WARNING: destructive)",
                    value="clean",
                ),
                Choice("─" * 60, value=None, disabled=True),
                Choice("[?] help    - Detailed descriptions and usage guide", value="show_help"),
                Choice("[<] back    - Back to main menu", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            return

        if action == "show_help":
            show_deploy_help()
            continue  # Return to menu after help

        # Action selected - break out of menu loop and execute
        import subprocess

        # Determine config path
        if project_path:
            config_path = str(project_path / "config.yml")
            # Save and change directory
            original_dir = Path.cwd()

            try:
                os.chdir(project_path)
            except (OSError, PermissionError) as e:
                console.print(f"\n{Messages.error(f'Cannot change to project directory: {e}')}")
                input("\nPress ENTER to continue...")
                continue  # Return to menu
        else:
            config_path = "config.yml"
            original_dir = None

        try:
            # Confirm destructive operations
            if action == "clean":
                console.print("\n[bold red]⚠️  WARNING: Destructive Operation[/bold red]")
                console.print("\n[warning]This will permanently delete:[/warning]")
                console.print("  • All containers for this project")
                console.print("  • All volumes (including databases and stored data)")
                console.print("  • All networks created by compose")
                console.print("  • Container images built for this project")
                console.print("\n[dim]This action cannot be undone![/dim]\n")

                confirm = questionary.confirm(
                    "Are you sure you want to proceed?", default=False, style=custom_style
                ).ask()

                if not confirm:
                    console.print(f"\n{Messages.warning('Operation cancelled')}")
                    input("\nPress ENTER to continue...")
                    if original_dir:
                        try:
                            os.chdir(original_dir)
                        except (OSError, PermissionError):
                            pass
                    continue  # Return to menu

            elif action == "rebuild":
                console.print("\n[bold yellow]⚠️  Rebuild Operation[/bold yellow]")
                console.print("\n[warning]This will:[/warning]")
                console.print("  • Stop and remove all containers")
                console.print("  • Delete all volumes (data will be lost)")
                console.print("  • Remove container images")
                console.print("  • Rebuild everything from scratch")
                console.print("  • Start services again")
                console.print("\n[dim]Any data stored in volumes will be lost![/dim]\n")

                confirm = questionary.confirm(
                    "Proceed with rebuild?", default=False, style=custom_style
                ).ask()

                if not confirm:
                    console.print(f"\n{Messages.warning('Rebuild cancelled')}")
                    input("\nPress ENTER to continue...")
                    if original_dir:
                        try:
                            os.chdir(original_dir)
                        except (OSError, PermissionError):
                            pass
                    continue  # Return to menu

            # Build the osprey deploy command
            # Use 'osprey' command directly to avoid module import warnings
            cmd = ["osprey", "deploy", action]

            if action in ["up", "restart", "rebuild"]:
                cmd.append("-d")  # Run in detached mode

            cmd.extend(["--config", config_path])

            if action == "up":
                console.print("\n[bold]Starting services...[/bold]")
            elif action == "down":
                console.print("\n[bold]Stopping services...[/bold]")
            elif action == "restart":
                console.print("\n[bold]Restarting services...[/bold]")
            elif action == "build":
                console.print("\n[bold]Building compose files...[/bold]")
            elif action == "rebuild":
                console.print("\n[bold]Rebuilding services (clean + build + start)...[/bold]")
            elif action == "clean":
                console.print("\n[bold red]⚠️  Cleaning deployment...[/bold red]")
            # Note: 'status' action doesn't print a header here because show_status() prints its own

            try:
                # Run subprocess with timeout (5 minutes for deploy operations)
                # Set environment to suppress config/registry warnings in subprocess
                env = os.environ.copy()
                env["OSPREY_QUIET"] = "1"  # Signal to suppress non-critical warnings
                result = subprocess.run(cmd, cwd=project_path or Path.cwd(), timeout=300, env=env)
            except subprocess.TimeoutExpired:
                console.print(f"\n{Messages.error('Command timed out after 5 minutes')}")
                console.print(
                    Messages.warning("The operation took too long. Check your container runtime.")
                )
                input("\nPress ENTER to continue...")
                if original_dir:
                    try:
                        os.chdir(original_dir)
                    except (OSError, PermissionError):
                        pass
                continue  # Return to menu

            if result.returncode == 0:
                if action == "up":
                    console.print(f"\n{Messages.success('Services started')}")
                elif action == "down":
                    console.print(f"\n{Messages.success('Services stopped')}")
                elif action == "restart":
                    console.print(f"\n{Messages.success('Services restarted')}")
                elif action == "build":
                    console.print(f"\n{Messages.success('Compose files built')}")
                elif action == "rebuild":
                    console.print(f"\n{Messages.success('Services rebuilt and started')}")
                elif action == "clean":
                    console.print(f"\n{Messages.success('Deployment cleaned')}")
            else:
                console.print(
                    f"\n{Messages.warning(f'Command exited with code {result.returncode}')}"
                )

        except Exception as e:
            console.print(f"\n{Messages.error(str(e))}")
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
        break  # Exit loop after action completes


def handle_health_action(project_path: Path | None = None):
    """Run health check.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    # Save and optionally change directory
    if project_path:
        original_dir = Path.cwd()

        try:
            os.chdir(project_path)
        except (OSError, PermissionError) as e:
            console.print(f"\n{Messages.error(f'Cannot change to project directory: {e}')}")
            input("\nPress ENTER to continue...")
            return
    else:
        original_dir = None

    try:
        from osprey.cli.health_cmd import HealthChecker
        from osprey.utils.log_filter import quiet_logger

        # Create and run health checker (full mode by default)
        # Suppress config/registry initialization messages
        with quiet_logger(["registry", "CONFIG"]):
            checker = HealthChecker(verbose=False, full=True)
            success = checker.check_all()

        if success:
            console.print(f"\n{Messages.success('Health check completed successfully')}")
        else:
            console.print(f"\n{Messages.warning('Health check completed with warnings')}")

    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
    finally:
        # Restore original directory
        if original_dir:
            try:
                os.chdir(original_dir)
            except (OSError, PermissionError) as e:
                console.print(f"\n{Messages.warning(f'Could not restore directory: {e}')}")

    input("\nPress ENTER to continue...")


def handle_tasks_action():
    """Handle tasks browsing action from interactive menu.

    Launches the interactive task browser for selecting and managing tasks.
    """
    from osprey.cli.tasks_cmd import interactive_task_browser

    interactive_task_browser()


def handle_export_action(project_path: Path | None = None):
    """Show configuration export.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    try:
        from pathlib import Path

        import yaml
        from jinja2 import Template
        from rich.syntax import Syntax

        # If project_path provided, show that project's config
        if project_path:
            config_path = project_path / "config.yml"

            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)

                output_str = yaml.dump(
                    config_data, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

                console.print(f"\n[bold]Configuration for {project_path.name}:[/bold]\n")
                syntax = Syntax(
                    output_str, "yaml", theme="monokai", line_numbers=False, word_wrap=True
                )
                console.print(syntax)
            else:
                console.print(f"{Messages.error(f'No config.yml found in {project_path}')}")
        else:
            # Load framework's configuration template
            template_path = Path(__file__).parent.parent / "templates" / "project" / "config.yml.j2"

            if not template_path.exists():
                console.print(Messages.error("Could not locate framework configuration template."))
                console.print(f"[dim]Expected at: {Messages.path(str(template_path))}[/dim]")
            else:
                # Read and render the template with example values
                with open(template_path) as f:
                    template_content = f.read()

                template = Template(template_content)
                rendered_config = template.render(
                    project_name="example_project",
                    package_name="example_project",
                    project_root="/path/to/example_project",
                    hostname="localhost",
                    default_provider="anthropic",
                    default_model="anthropic/claude-haiku",
                )

                # Parse the rendered config as YAML
                config_data = yaml.safe_load(rendered_config)

                # Format as YAML
                output_str = yaml.dump(
                    config_data, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

                # Print to console with syntax highlighting
                console.print("\n[bold]Osprey Default Configuration:[/bold]\n")
                syntax = Syntax(
                    output_str, "yaml", theme="monokai", line_numbers=False, word_wrap=True
                )
                console.print(syntax)
                console.print(
                    f"\n[dim]Tip: Use {Messages.command('osprey config export --output file.yml')} to save to file[/dim]"
                )

    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")

    input("\nPress ENTER to continue...")


def show_config_menu() -> str | None:
    """Show config submenu.

    Returns:
        Selected config action, or None if user cancels/goes back
    """
    if not questionary:
        return None

    console.print(f"\n{Messages.header('Configuration')}")
    console.print("[dim]Manage project configuration settings[/dim]\n")

    return questionary.select(
        "What would you like to do?",
        choices=[
            Choice(
                "[→] set-control-system - Switch between Mock/EPICS connectors",
                value="set_control_system",
            ),
            Choice(
                "[→] set-epics-gateway  - Configure EPICS gateway (APS, ALS, custom)",
                value="set_epics_gateway",
            ),
            Choice(
                "[→] set-models         - Change AI provider and models",
                value="set_models",
            ),
            Choice("[→] show               - Display current configuration", value="show"),
            Choice(
                "[→] export             - Export framework default configuration", value="export"
            ),
            Choice("─" * 60, value=None, disabled=True),
            Choice("[←] back               - Return to main menu", value="back"),
        ],
        style=custom_style,
    ).ask()


def handle_config_action(project_path: Path | None = None) -> None:
    """Handle config menu and its subcommands."""
    while True:
        action = show_config_menu()

        if action is None or action == "back":
            return  # Return to main menu

        if action == "show":
            handle_export_action(project_path)
            input("\nPress ENTER to continue...")
        elif action == "export":
            # Export framework defaults (works from anywhere)
            import click

            from osprey.cli.config_cmd import export as export_cmd

            try:
                ctx = click.Context(export_cmd)
                ctx.invoke(export_cmd, output=None, format="yaml")
            except click.Abort:
                pass
            input("\nPress ENTER to continue...")
        elif action == "set_control_system":
            handle_set_control_system(project_path)
        elif action == "set_epics_gateway":
            handle_set_epics_gateway(project_path)
        elif action == "set_models":
            handle_set_models(project_path)


def handle_set_control_system(project_path: Path | None = None) -> None:
    """Handle interactive control system type configuration."""
    from osprey.utils.config_writer import (
        find_config_file,
        get_control_system_type,
        set_control_system_type,
    )

    console.clear()
    console.print(f"\n{Messages.header('Configure Control System')}\n")

    # Find config file
    if project_path:
        config_path = project_path / "config.yml"
    else:
        config_path = find_config_file()

    if not config_path or not config_path.exists():
        console.print(f"{Messages.error('No config.yml found in current directory')}")
        input("\nPress ENTER to continue...")
        return

    # Show current configuration
    current_type = get_control_system_type(config_path)
    current_archiver = get_control_system_type(config_path, key="archiver.type")

    console.print(f"[dim]Current control system: {current_type or 'mock'}[/dim]")
    console.print(f"[dim]Current archiver: {current_archiver or 'mock_archiver'}[/dim]\n")

    # Show choices
    choices = [
        Choice("Mock - Tutorial/Development mode (safe, no hardware)", value=types.MOCK),
        Choice("EPICS - Production mode (connects to real control system)", value=types.EPICS),
        Choice("─" * 60, value=None, disabled=True),
        Choice("[←] Back - Return to config menu", value="back"),
    ]

    control_type = questionary.select(
        "Select control system type:", choices=choices, style=custom_style
    ).ask()

    if control_type is None or control_type == "back":
        return

    # Ask about archiver too
    if control_type == types.EPICS:
        console.print("\n[bold]Archiver Configuration[/bold]\n")
        archiver_type = questionary.select(
            "Also switch archiver to EPICS?",
            choices=[
                Choice("Yes - Use EPICS Archiver Appliance", value=types.EPICS_ARCHIVER),
                Choice("No - Keep mock archiver", value=types.MOCK_ARCHIVER),
            ],
            style=custom_style,
        ).ask()
    else:
        archiver_type = types.MOCK_ARCHIVER

    # Update configuration
    new_content, preview = set_control_system_type(config_path, control_type, archiver_type)

    # Show preview
    console.print("\n" + preview)

    # Confirm
    if questionary.confirm(
        "\nUpdate config.yml with this configuration?", default=True, style=custom_style
    ).ask():
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        config_path.write_text(new_content, encoding="utf-8")
        console.print(f"\n{Messages.success('✓ Control system configuration updated!')}")

        if control_type == types.EPICS:
            console.print("\n[dim]💡 Next steps:[/dim]")
            console.print("[dim]   1. Configure EPICS gateway: config → set-epics-gateway[/dim]")
            console.print("[dim]   2. Verify EPICS connection settings[/dim]")
        else:
            console.print("\n[dim]You're now in Mock mode - safe for development and testing[/dim]")
    else:
        console.print(f"\n{Messages.warning('✗ Configuration not changed')}")

    input("\nPress ENTER to continue...")


def _check_simulation_ioc_running(host: str = "localhost", port: int = 5064) -> bool:
    """Check if a simulation IOC is running on the specified port.

    Args:
        host: Host address to check
        port: Port number to check

    Returns:
        True if port is open and accepting connections, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result == 0
    except OSError:
        return False


def handle_set_epics_gateway(project_path: Path | None = None) -> None:
    """Handle interactive EPICS gateway configuration."""
    from osprey.templates.data import FACILITY_PRESETS
    from osprey.utils.config_writer import (
        find_config_file,
        get_control_system_type,
        get_facility_from_gateway_config,
        set_control_system_type,
        set_epics_gateway_config,
    )

    console.clear()
    console.print(f"\n{Messages.header('Configure EPICS Gateway')}\n")

    # Find config file
    if project_path:
        config_path = project_path / "config.yml"
    else:
        config_path = find_config_file()

    if not config_path or not config_path.exists():
        console.print(f"{Messages.error('No config.yml found in current directory')}")
        input("\nPress ENTER to continue...")
        return

    # Show current configuration
    current_facility = get_facility_from_gateway_config(config_path)
    if current_facility:
        console.print(f"[dim]Current configuration: {current_facility}[/dim]\n")
    else:
        console.print("[dim]Current configuration: Default (Mock mode)[/dim]\n")

    # Show facility choices
    choices = []
    for facility_id, preset in FACILITY_PRESETS.items():
        display_name = f"{preset['name']} - {preset['description']}"
        choices.append(Choice(display_name, value=facility_id))

    choices.extend(
        [
            Choice("Custom - Manual configuration", value="custom"),
            Choice("─" * 60, value=None, disabled=True),
            Choice("[←] Back - Return to config menu", value="back"),
        ]
    )

    facility = questionary.select(
        "Select EPICS facility:", choices=choices, style=custom_style
    ).ask()

    if facility is None or facility == "back":
        return

    if facility == "custom":
        # Interactive prompts for custom gateway
        console.print("\n[bold]Custom EPICS Gateway Configuration[/bold]\n")

        read_address = questionary.text(
            "Read gateway address:", default="your-gateway.facility.edu"
        ).ask()

        if not read_address:
            return

        read_port = questionary.text("Read gateway port:", default="5064").ask()

        write_address = questionary.text(
            "Write gateway address (or same as read):", default=read_address
        ).ask()

        write_port = questionary.text("Write gateway port:", default="5084").ask()

        use_name_server = questionary.confirm(
            "Use name server mode? (for SSH tunnels)", default=False
        ).ask()

        custom_config = {
            "read_only": {
                "address": read_address,
                "port": int(read_port),
                "use_name_server": use_name_server,
            },
            "write_access": {
                "address": write_address,
                "port": int(write_port),
                "use_name_server": use_name_server,
            },
        }

        new_content, preview = set_epics_gateway_config(config_path, "custom", custom_config)
    else:
        # Use preset
        new_content, preview = set_epics_gateway_config(config_path, facility)

        # Check if simulation IOC is running when using simulation preset
        if facility == "simulation":
            preset = FACILITY_PRESETS[facility]
            host = preset["gateways"]["read_only"]["address"]
            port = preset["gateways"]["read_only"]["port"]

            if not _check_simulation_ioc_running(host, port):
                console.print(f"\n{Messages.warning(f'⚠ No IOC detected on {host}:{port}')}")
                console.print(
                    "\n[dim]To start the simulation IOC:[/dim]"
                    "\n[dim]  1. Generate IOC: osprey generate soft-ioc[/dim]"
                    "\n[dim]  2. Run IOC: python generated_iocs/<ioc_name>_ioc.py[/dim]"
                )

    # Show preview
    console.print("\n" + preview)

    # Confirm
    if questionary.confirm(
        "\nUpdate config.yml with this configuration?", default=True, style=custom_style
    ).ask():
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        config_path.write_text(new_content, encoding="utf-8")
        console.print(f"\n{Messages.success('✓ EPICS gateway configuration updated!')}")

        # Check if mode is still 'mock' and offer to switch
        current_type = get_control_system_type(config_path)
        if current_type in (None, types.MOCK):
            # None means missing config key, treat same as mock
            if questionary.confirm(
                "\nYour control system is set to 'mock' mode. Switch to 'epics' to use this "
                "gateway?",
                default=True,
                style=custom_style,
            ).ask():
                type_content, _ = set_control_system_type(config_path, types.EPICS)
                config_path.write_text(type_content, encoding="utf-8")
                console.print(f"{Messages.success('✓ Switched to epics mode!')}")
            else:
                console.print(
                    "\n[dim]Note: Gateway configured but mode is still 'mock'. "
                    "Use 'set-control-system' to switch when ready.[/dim]"
                )
        elif current_type == types.EPICS:
            console.print("[dim]Control system already set to 'epics' mode.[/dim]")
        else:
            # Other types like 'tango', 'labview' - don't auto-switch
            console.print(
                f"[dim]Note: Control system is set to '{current_type}'. "
                "This gateway config applies when using 'epics' mode.[/dim]"
            )
    else:
        console.print(f"\n{Messages.warning('✗ Configuration not changed')}")

    input("\nPress ENTER to continue...")


def handle_set_models(project_path: Path | None = None) -> None:
    """Handle interactive model configuration."""
    from osprey.cli.interactive_menu import (
        get_provider_metadata,
        select_model,
        select_provider,
    )
    from osprey.utils.config_writer import (
        find_config_file,
        get_all_model_configs,
        update_all_models,
    )

    console.clear()
    console.print(f"\n{Messages.header('Configure AI Models')}\n")

    # Find config file
    if project_path:
        config_path = project_path / "config.yml"
    else:
        config_path = find_config_file()

    if not config_path or not config_path.exists():
        console.print(f"{Messages.error('No config.yml found in current directory')}")
        input("\nPress ENTER to continue...")
        return

    # Show current configuration
    current_models = get_all_model_configs(config_path)
    if current_models:
        # Show first model as example of current config
        first_model = next(iter(current_models.values()))
        current_provider = first_model.get("provider", "unknown")
        current_model_id = first_model.get("model_id", "unknown")
        console.print(f"[dim]Current: {current_provider}/{current_model_id}[/dim]")
        console.print(f"[dim]Configured models: {len(current_models)}[/dim]\n")
    else:
        console.print("[dim]No model configuration found[/dim]\n")

    # Get provider metadata
    console.print("[dim]Loading available providers and models...[/dim]")
    providers = get_provider_metadata()

    if not providers:
        console.print(f"\n{Messages.error('Could not load provider information')}")
        input("\nPress ENTER to continue...")
        return

    # Provider selection
    console.print("\n[bold]Step 1: Select AI Provider[/bold]\n")
    provider = select_provider(providers)
    if provider is None:
        return

    # Model selection
    console.print(f"\n[bold]Step 2: Select Model for {provider}[/bold]\n")
    model_id = select_model(provider, providers)
    if model_id is None:
        return

    # Generate update and preview
    try:
        new_content, preview = update_all_models(config_path, provider, model_id)
    except Exception as e:
        console.print(f"\n{Messages.error(f'Failed to generate update: {e}')}")
        input("\nPress ENTER to continue...")
        return

    # Show preview
    console.print("\n" + preview)

    # Confirm
    if questionary.confirm(
        "\nUpdate all models in config.yml?", default=True, style=custom_style
    ).ask():
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        config_path.write_text(new_content, encoding="utf-8")
        console.print(f"\n{Messages.success('✓ Model configuration updated!')}")
        console.print(
            f"\n[dim]All {len(current_models)} model(s) now use: {provider}/{model_id}[/dim]"
        )
    else:
        console.print(f"\n{Messages.warning('✗ Configuration not changed')}")

    input("\nPress ENTER to continue...")
