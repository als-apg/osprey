"""Menu display helpers: banners, ASCII art, and help screens.

Pure display functions extracted from interactive_menu.py.
These have no business logic dependencies — only Rich/console output.
"""

from osprey.cli.styles import (
    Messages,
    Styles,
    ThemeConfig,
    console,
)


def show_banner(context: str = "interactive", config_path: str | None = None):
    """Display the unified osprey banner with ASCII art.

    Args:
        context: Display context - "interactive", "chat", or "welcome"
        config_path: Optional path to config file for custom banner
    """
    from pathlib import Path

    from rich.text import Text

    from osprey.utils.config import get_config_value
    from osprey.utils.log_filter import quiet_logger

    # Get version number
    try:
        from osprey import __version__

        version_str = f"v{__version__}"
    except (ImportError, AttributeError):
        version_str = ""

    console.print()

    # Try to load custom banner if in a project directory
    banner_text = None

    try:
        # Check if config exists before trying to load
        # Suppress config loading messages in interactive menu
        with quiet_logger(["registry", "CONFIG"]):
            if config_path:
                banner_text = get_config_value("cli.banner", None, config_path)
            elif (Path.cwd() / "config.yml").exists():
                banner_text = get_config_value("cli.banner", None)
    except Exception:
        pass  # Fallback to default - CLI should always work

    # Default banner if not configured
    if banner_text is None:
        banner_text = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║                                                           ║
    ║    ░█████╗░░██████╗██████╗░██████╗░███████╗██╗░░░██╗      ║
    ║    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝╚██╗░██╔╝      ║
    ║    ██║░░██║╚█████╗░██████╔╝██████╔╝█████╗░░░╚████╔╝░      ║
    ║    ██║░░██║░╚═══██╗██╔═══╝░██╔══██╗██╔══╝░░░░╚██╔╝░░      ║
    ║    ╚█████╔╝██████╔╝██║░░░░░██║░░██║███████╗░░░██║░░░      ║
    ║    ░╚════╝░╚═════╝░╚═╝░░░░░╚═╝░░╚═╝╚══════╝░░░╚═╝░░░      ║
    ║                                                           ║
    ║                                                           ║
    ║      Command Line Interface for the Osprey Framework      ║
    ╚═══════════════════════════════════════════════════════════╝
        """

    console.print(Text(banner_text, style=ThemeConfig.get_banner_style()))

    # Show version if available
    if version_str:
        console.print(f"    [{Styles.DIM}]{version_str}[/{Styles.DIM}]")

    # Context-specific subtitle
    if context == "interactive":
        console.print(f"    [{Styles.HEADER}]Interactive Menu System[/{Styles.HEADER}]")
        console.print(
            f"    [{Styles.DIM}]Use arrow keys to navigate • Press Ctrl+C to exit[/{Styles.DIM}]"
        )
    elif context == "chat":
        msg = Messages.info("💡 Type 'bye' or 'end' to exit")
        console.print(f"    {msg}")
        console.print(
            f"    [{Styles.ACCENT}]⚡ Use slash commands (/) for quick actions - try /help[/{Styles.ACCENT}]"
        )

    console.print()


def show_success_art():
    """Display success ASCII art."""
    art = """
    ┌───────────────────┐
    │   ✓  SUCCESS  ✓   │
    └───────────────────┘
    """
    console.print(art, style=Styles.BOLD_SUCCESS)


def show_deploy_help():
    """Display detailed help for deployment options."""
    console.clear()
    show_banner(context="interactive")

    console.print(f"\n{Messages.header('Deployment Services - Help')}\n")

    console.print(f"[{Styles.HEADER}][^] up - Start all services[/{Styles.HEADER}]")
    console.print()
    console.print("  • Builds and starts all containers defined in docker-compose.yml")
    console.print("  • Creates volumes and networks as needed")
    console.print("  • Runs services in detached mode (background)")
    console.print(
        f"  • [{Styles.DIM}]Use this to start your web terminal and other services[/{Styles.DIM}]"
    )
    console.print()

    console.print(f"[{Styles.HEADER}][v] down - Stop all services[/{Styles.HEADER}]")
    console.print()
    console.print("  • Stops and removes all running containers")
    console.print("  • Preserves volumes (data persists)")
    console.print("  • Removes networks created by compose")
    console.print(f"  • [{Styles.DIM}]Safe operation - your data remains intact[/{Styles.DIM}]")
    console.print()

    console.print(f"[{Styles.HEADER}][i] status - Show service status[/{Styles.HEADER}]")
    console.print()
    console.print("  • Lists all containers for this project")
    console.print("  • Shows running state, ports, and health status")
    console.print("  • Displays resource usage if available")
    console.print(
        f"  • [{Styles.DIM}]Use this to verify services are running correctly[/{Styles.DIM}]"
    )
    console.print()

    console.print(f"[{Styles.HEADER}][*] restart - Restart all services[/{Styles.HEADER}]")
    console.print()
    console.print("  • Stops and restarts all containers")
    console.print("  • Applies configuration changes")
    console.print("  • Preserves volumes and data")
    console.print(
        f"  • [{Styles.DIM}]Use after modifying docker-compose.yml or environment variables[/{Styles.DIM}]"
    )
    console.print()

    console.print(
        f"[{Styles.HEADER}][+] build - Build/prepare compose files only[/{Styles.HEADER}]"
    )
    console.print()
    console.print("  • Generates docker-compose.yml from templates")
    console.print("  • Does not start any containers")
    console.print("  • Validates compose file structure")
    console.print(
        f"  • [{Styles.DIM}]Use to inspect generated configuration before deployment[/{Styles.DIM}]"
    )
    console.print()

    console.print(
        f"[{Styles.HEADER}][R] rebuild - Clean, rebuild, and restart services[/{Styles.HEADER}]"
    )
    console.print()
    console.print("  • Stops and removes all containers and volumes")
    console.print("  • Removes container images")
    console.print("  • Rebuilds everything from scratch")
    console.print("  • Starts services with fresh state")
    console.print(
        f"  • [{Styles.WARNING}]⚠️  Warning: All data in volumes will be lost[/{Styles.WARNING}]"
    )
    console.print()

    console.print(f"[{Styles.HEADER}][X] clean - Remove containers and volumes[/{Styles.HEADER}]")
    console.print()
    console.print("  • Permanently deletes all containers")
    console.print("  • Permanently deletes all volumes and data")
    console.print("  • Removes networks and images")
    console.print(f"  • [{Styles.WARNING}]⚠️  Destructive: Cannot be undone![/{Styles.WARNING}]")
    console.print()

    input("Press ENTER to continue...")


def handle_help_action_root():
    """Show help for root menu (no project detected)."""
    console.clear()
    show_banner(context="interactive")

    console.print(f"\n{Messages.header('Getting Started - Help')}\n")

    # Select existing project
    console.print(f"[{Styles.HEADER}][→] Select a project[/{Styles.HEADER}]")
    console.print()
    console.print("  • Navigate into an existing Osprey project in a subdirectory")
    console.print("  • Opens the project menu with full access to all commands")
    console.print("  • Use chat, deploy services, generate capabilities, etc.")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Working with an existing agent project[/{Styles.DIM}]"
    )
    console.print()

    # Create new project
    console.print(f"[{Styles.HEADER}][+] Create new project (interactive)[/{Styles.HEADER}]")
    console.print()
    console.print("  • Guided wizard to create a new Osprey project from scratch")
    console.print("  • Choose template: minimal, weather example, or control assistant")
    console.print("  • Select AI provider (Anthropic, OpenAI, etc.) and model")
    console.print("  • Configure API keys securely with interactive prompts")
    console.print("  • Generates complete project structure ready to use")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Starting your first agent or creating a new use case[/{Styles.DIM}]"
    )
    console.print()

    # Workflow
    console.print(f"[{Styles.HEADER}]Typical Workflow:[/{Styles.HEADER}]")
    console.print()
    console.print("  1. Create a new project (or select existing)")
    console.print("  2. Navigate into the project directory")
    console.print("  3. Use the project menu to:")
    console.print("     • Chat with your agent")
    console.print("     • Deploy web interfaces")
    console.print("     • Generate new capabilities")
    console.print("     • Monitor health and configuration")
    console.print()

    input("Press ENTER to continue...")


def handle_help_action():
    """Show help for project menu options."""
    console.clear()
    show_banner(context="interactive")

    console.print(f"\n{Messages.header('Project Menu - Help')}\n")

    # chat
    # deploy
    console.print(f"[{Styles.HEADER}][>] deploy - Manage services (web UIs)[/{Styles.HEADER}]")
    console.print()
    console.print("  • Start, stop, and manage containerized services")
    console.print("  • Launch web terminal and other interfaces")
    console.print("  • View service status and logs")
    console.print(f"  • [{Styles.DIM}]Perfect for: Production deployments[/{Styles.DIM}]")
    console.print()

    # health
    console.print(f"[{Styles.HEADER}][>] health - Run system health check[/{Styles.HEADER}]")
    console.print()
    console.print("  • Verifies your Osprey installation")
    console.print("  • Tests API connectivity to your LLM provider")
    console.print("  • Checks capabilities and registry configuration")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Troubleshooting, validating setup after changes[/{Styles.DIM}]"
    )
    console.print()

    # generate
    console.print(f"[{Styles.HEADER}][>] generate - Generate components[/{Styles.HEADER}]")
    console.print()
    console.print("  • Create capabilities from MCP servers or natural language")
    console.print("  • Generate demo MCP servers for testing")
    console.print("  • Create Claude Code generator configurations")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Extending your agent with new capabilities[/{Styles.DIM}]"
    )
    console.print()

    # config
    console.print(f"[{Styles.HEADER}][>] config - Show configuration[/{Styles.HEADER}]")
    console.print()
    console.print("  • View your current project configuration")
    console.print("  • See provider, model, and capability settings")
    console.print("  • Verify registry and execution configuration")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Understanding your project setup, debugging config issues[/{Styles.DIM}]"
    )
    console.print()

    # registry
    console.print(f"[{Styles.HEADER}][>] registry - Show registry contents[/{Styles.HEADER}]")
    console.print()
    console.print("  • View all registered capabilities, providers, and tools")
    console.print("  • See what your agent has access to")
    console.print("  • Inspect capability metadata and parameters")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Understanding available features, debugging capabilities[/{Styles.DIM}]"
    )
    console.print()

    # init
    console.print(f"[{Styles.HEADER}][+] init - Create new project[/{Styles.HEADER}]")
    console.print()
    console.print("  • Guided wizard to create a new Osprey project")
    console.print("  • Choose template, provider, model, and configure API keys")
    console.print("  • Generates complete project structure ready to use")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Starting a new agent project from scratch[/{Styles.DIM}]"
    )
    console.print()

    input("Press ENTER to continue...")
