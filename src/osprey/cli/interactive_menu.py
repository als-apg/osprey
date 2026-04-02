"""Interactive Terminal UI (TUI) orchestrator for Osprey Framework CLI.

This module provides the top-level navigation loop and shared utilities
for the interactive menu system. It delegates to:
- menu_display: Banner, ASCII art, help screens
- init_wizard: Interactive project creation flow
- project_actions: Deploy, health, config, and other action handlers

The interactive menu is optional - users can still use direct commands like:
    osprey init my-project
    osprey web
    osprey deploy up
"""

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Re-export from init_wizard (backward compatibility)
from osprey.cli.init_wizard import (  # noqa: F401
    configure_api_key,
    get_api_key_name,
    get_default_name_for_template,
    run_interactive_init,
    run_interactive_profile_wizard,
    select_channel_finder_mode,
    select_template,
    show_api_key_help,
    show_manual_config_instructions,
    write_env_file,
)

# Re-export from menu_display (backward compatibility)
from osprey.cli.menu_display import (  # noqa: F401
    handle_help_action,
    handle_help_action_root,
    show_banner,
    show_deploy_help,
    show_success_art,
)

# Re-export from project_actions (backward compatibility)
from osprey.cli.project_actions import (  # noqa: F401
    _check_simulation_ioc_running,
    handle_config_action,
    handle_deploy_action,
    handle_export_action,
    handle_health_action,
    handle_project_selection,
    handle_set_control_system,
    handle_set_epics_gateway,
    handle_set_models,
    show_config_menu,
)
from osprey.cli.styles import (
    Messages,
    console,
    get_questionary_style,
)
from osprey.deployment.runtime_helper import get_runtime_command

try:
    import questionary
    from questionary import Choice
except ImportError:
    questionary = None
    Choice = None


# --- Console And Styling ---

custom_style = get_questionary_style()


# --- Project Detection ---


def is_project_initialized() -> bool:
    """Check if we're in an osprey project directory.

    Returns:
        True if config.yml exists in current directory
    """
    return (Path.cwd() / "config.yml").exists()


def get_project_info(config_path: Path | None = None) -> dict[str, Any]:
    """Load and parse config.yml for project metadata.

    Args:
        config_path: Optional path to config.yml (defaults to current directory)

    Returns:
        Dictionary with project information (provider, model, etc.)
        Returns empty dict if no project found or error parsing
    """
    if config_path is None:
        config_path = Path.cwd() / "config.yml"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate config.yml structure after parsing
        if config is None:
            if os.environ.get("DEBUG"):
                console.print(f"[dim]Warning: Empty config.yml at {config_path}[/dim]")
            return {}

        if not isinstance(config, dict):
            if os.environ.get("DEBUG"):
                console.print(
                    f"[dim]Warning: Invalid config.yml structure (not a dict) at {config_path}[/dim]"
                )
            return {}

        # Extract relevant information with safe defaults
        info = {
            "project_root": config.get("project_root", str(config_path.parent)),
            "registry_path": config.get("registry_path", ""),
        }

        # Extract provider and model from models.orchestrator section
        models_config = config.get("models", {})
        if not isinstance(models_config, dict):
            models_config = {}

        orchestrator = models_config.get("orchestrator", {})
        if not isinstance(orchestrator, dict):
            orchestrator = {}

        if orchestrator:
            info["provider"] = orchestrator.get("provider", "unknown")
            info["model"] = orchestrator.get("model_id", "unknown")

        return info

    except yaml.YAMLError as e:
        console.print(Messages.warning(f"Invalid YAML in config.yml: {e}"))
        return {}
    except UnicodeDecodeError as e:
        console.print(Messages.warning(f"Encoding error in config.yml: {e}"))
        return {}
    except Exception as e:
        console.print(Messages.warning(f"Could not parse config.yml: {e}"))
        return {}


def discover_nearby_projects(max_dirs: int = 50, max_time_ms: int = 100) -> list[tuple[str, Path]]:
    """Discover osprey projects in immediate subdirectories.

    This performs a SHALLOW, non-recursive search (1 level deep only) for
    config.yml files in subdirectories of the current working directory.

    Performance safeguards:
    - Only checks immediate subdirectories (not recursive)
    - Stops after checking max_dirs subdirectories
    - Has timeout protection (max_time_ms)
    - Ignores hidden directories and common non-project directories

    Args:
        max_dirs: Maximum number of subdirectories to check (default: 50)
        max_time_ms: Maximum time to spend searching in milliseconds (default: 100)

    Returns:
        List of tuples: (project_name, project_path)
        Sorted alphabetically by project name

    Examples:
        >>> discover_nearby_projects()
        [('my-agent', Path('/current/dir/my-agent')),
         ('weather-app', Path('/current/dir/weather-app'))]
    """
    import time

    projects = []
    start_time = time.time()
    max_time_seconds = max_time_ms / 1000.0

    # Directories to ignore (common non-project directories)
    ignore_dirs = {
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        "build",
        "dist",
        ".egg-info",
        "site-packages",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "docs",
        "_agent_data",
        ".cache",
    }

    try:
        cwd = Path.cwd()
        checked_count = 0

        # Get all immediate subdirectories
        subdirs = []
        try:
            for item in cwd.iterdir():
                # Check timeout
                if time.time() - start_time > max_time_seconds:
                    if os.environ.get("DEBUG"):
                        console.print(f"[dim]Project discovery timeout after {max_time_ms}ms[/dim]")
                    break

                # Only check directories
                if not item.is_dir():
                    continue

                # Skip hidden directories (start with .)
                if item.name.startswith("."):
                    continue

                # Skip common non-project directories
                if item.name in ignore_dirs:
                    continue

                subdirs.append(item)

        except (PermissionError, OSError) as e:
            # Skip directories we can't read
            if os.environ.get("DEBUG"):
                console.print(f"[dim]Warning: Could not read directory: {e}[/dim]")
            return projects

        # Sort subdirectories alphabetically for consistent ordering
        subdirs.sort(key=lambda p: p.name.lower())

        # Check each subdirectory for config.yml
        for subdir in subdirs:
            # Check limits
            if checked_count >= max_dirs:
                if os.environ.get("DEBUG"):
                    console.print(
                        f"[dim]Project discovery stopped after checking {max_dirs} directories[/dim]"
                    )
                break

            if time.time() - start_time > max_time_seconds:
                if os.environ.get("DEBUG"):
                    console.print(f"[dim]Project discovery timeout after {max_time_ms}ms[/dim]")
                break

            try:
                config_file = subdir / "config.yml"

                if config_file.exists() and config_file.is_file():
                    # Found a project!
                    projects.append((subdir.name, subdir))

            except (PermissionError, OSError):
                # Skip directories we can't access
                pass

            checked_count += 1

    except Exception as e:
        # Fail gracefully - return whatever we found so far
        if os.environ.get("DEBUG"):
            console.print(f"[dim]Warning during project discovery: {e}[/dim]")

    # Return sorted list
    return sorted(projects, key=lambda x: x[0].lower())


# Cache for provider metadata (loaded once per TUI session)
_provider_cache: dict[str, dict[str, Any]] | None = None


def get_provider_metadata() -> dict[str, dict[str, Any]]:
    """Get provider information from osprey registry.

    Loads providers directly from the osprey registry configuration
    without requiring a project config.yml. This reads the osprey's
    provider registrations and introspects provider class attributes
    for metadata (single source of truth).

    This approach works whether or not you're in a project directory,
    making it perfect for the TUI init flow.

    Results are cached for the TUI session to avoid repeated registry loading.

    Returns:
        Dictionary mapping provider names to their metadata:
        {
            'anthropic': {
                'name': 'anthropic',
                'description': 'Anthropic (Claude models)',
                'requires_key': True,
                'requires_base_url': False,
                'models': ['claude-sonnet-4-5', ...],
                'default_model': 'claude-sonnet-4-5',
                'health_check_model': 'claude-haiku-4-5'
            },
            ...
        }
    """
    global _provider_cache

    # Return cached data if available
    if _provider_cache is not None:
        return _provider_cache

    import importlib

    try:
        # Import osprey registry provider directly (no config.yml needed!)
        from osprey.registry.builtins import FrameworkRegistryProvider

        # Get osprey registry config (doesn't require project config)
        framework_registry = FrameworkRegistryProvider()
        config = framework_registry.get_registry_config()

        providers = {}

        # Load each provider registration from osprey config
        for provider_reg in config.providers:
            try:
                # Import the provider module
                module = importlib.import_module(provider_reg.module_path)

                # Get the provider class
                provider_class = getattr(module, provider_reg.class_name)

                # Extract metadata from class attributes (single source of truth)
                providers[provider_class.name] = {
                    "name": provider_class.name,
                    "description": provider_class.description,
                    "requires_key": provider_class.requires_api_key,
                    "requires_base_url": provider_class.requires_base_url,
                    "models": provider_class.available_models,
                    "default_model": provider_class.default_model_id,
                    "health_check_model": provider_class.health_check_model_id,
                    "api_key_url": provider_class.api_key_url,
                    "api_key_instructions": provider_class.api_key_instructions,
                    "api_key_note": provider_class.api_key_note,
                }
            except Exception as e:
                # Skip providers that fail to load, but log for debugging
                if os.environ.get("DEBUG"):
                    console.print(
                        f"[dim]Warning: Could not load provider {provider_reg.class_name}: {e}[/dim]"
                    )
                continue

        if not providers:
            console.print(Messages.warning("No providers could be loaded from osprey registry"))

        # Cache the result for future calls
        _provider_cache = providers
        return providers

    except Exception as e:
        # This should rarely happen - osprey registry should always be available
        console.print(Messages.error(f"Could not load providers from osprey registry: {e}"))
        console.print(
            Messages.warning(
                "The TUI requires access to provider information to initialize projects."
            )
        )
        if os.environ.get("DEBUG"):
            import traceback

            traceback.print_exc()

        # Return empty dict but don't cache failures
        return {}



# --- Main Menu ---


def get_project_menu_choices(exit_action: str = "exit") -> list[Choice]:
    """Get standard project menu choices.

    This is the single source of truth for project menu options,
    used by both the main menu (when in a project) and the project
    selection submenu (when navigating from parent directory).

    Args:
        exit_action: Either 'exit' (for main menu) or 'back' (for submenu)

    Returns:
        List of Choice objects for the project menu
    """
    choices = [
        Choice("[>] deploy      - Manage services (web UIs)", value="deploy"),
        Choice("[>] health      - Run system health check", value="health"),
        Choice("[>] config      - Configuration settings", value="config"),
        Choice("[>] registry    - Show registry contents", value="registry"),
        Choice("─" * 60, value=None, disabled=True),
        Choice("[+] init        - Create new project", value="init_interactive"),
        Choice("[?] help        - Show all commands", value="help"),
    ]

    # Add context-appropriate exit/back option
    if exit_action == "back":
        choices.append(Choice("[<] back        - Return to main menu", value="back"))
    else:
        choices.append(Choice("[x] exit        - Exit CLI", value="exit"))

    return choices


def show_main_menu() -> str | None:
    """Show context-aware main menu.

    Returns:
        Selected action string, or None if user cancels
    """
    if not questionary:
        console.print(Messages.error("questionary package not installed."))
        console.print(f"Install with: {Messages.command('uv pip install questionary')}")
        return None

    if not is_project_initialized():
        # No project in current directory - discover nearby projects
        console.print("\n[dim]No project detected in current directory[/dim]")

        # Quick shallow search for projects in subdirectories
        nearby_projects = discover_nearby_projects()

        # Build menu choices
        choices = []

        # If we found nearby projects, add them to the menu
        if nearby_projects:
            console.print(f"[dim]Found {len(nearby_projects)} project(s) in subdirectories[/dim]\n")

            for project_name, project_path in nearby_projects:
                # Get project info for display
                project_info = get_project_info(project_path / "config.yml")

                if project_info and "provider" in project_info:
                    display = f"[→] {project_name:20} ({project_info['provider']} / {project_info.get('model', 'unknown')[:20]})"
                else:
                    display = f"[→] {project_name:20} (osprey project)"

                # Value is tuple so we can distinguish from other actions
                choices.append(Choice(display, value=("select_project", project_path)))

            # Add separator
            choices.append(Choice("─" * 60, value=None, disabled=True))

        # Standard menu options
        choices.extend(
            [
                Choice("[+] Create new project (interactive)", value="init_interactive"),
                Choice("[?] Help", value="help"),
                Choice("[x] Exit", value="exit"),
            ]
        )

        return questionary.select(
            "What would you like to do?", choices=choices, style=custom_style
        ).ask()
    else:
        # Project menu
        project_info = get_project_info()
        project_name = Path.cwd().name

        console.print(f"\n{Messages.header('Project:')} {project_name}")
        if project_info:
            console.print(
                f"[dim]Provider: {project_info.get('provider', 'unknown')} | "
                f"Model: {project_info.get('model', 'unknown')}[/dim]"
            )

        # Use centralized project menu choices (with 'exit' action)
        return questionary.select(
            "Select command:",
            choices=get_project_menu_choices(exit_action="exit"),
            style=custom_style,
        ).ask()


# --- Directory Safety Checks ---


def check_directory_has_active_mounts(directory: Path) -> tuple[bool, list[str]]:
    """Check if a directory has active Docker/Podman volume mounts.

    This helps prevent accidentally deleting directories that contain running
    services with active volume mounts, which can lead to corrupted containers.

    Args:
        directory: Directory path to check

    Returns:
        Tuple of (has_mounts, mount_details)
        - has_mounts: True if active mounts detected
        - mount_details: List of mount descriptions

    Examples:
        >>> has_mounts, details = check_directory_has_active_mounts(Path("my-project"))
        >>> if has_mounts:
        ...     print(f"Active mounts: {details}")
    """
    import json
    import subprocess

    mount_details = []

    # Normalize the directory path
    dir_str = str(directory.resolve())

    # Determine which container runtime to use
    try:
        runtime_cmd = get_runtime_command()
        runtime = runtime_cmd[0]  # 'docker' or 'podman'
    except RuntimeError:
        # No runtime available
        return False, []

    # Check for container mounts using detected runtime
    try:
        result = subprocess.run(
            [runtime, "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=1
        )

        if result.returncode == 0:
            containers = result.stdout.strip().split("\n")
            containers = [c for c in containers if c]  # Remove empty strings

            for container in containers:
                # Inspect each container for mounts
                inspect_result = subprocess.run(
                    [runtime, "inspect", "--format", "{{json .Mounts}}", container],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if inspect_result.returncode == 0:
                    try:
                        mounts = json.loads(inspect_result.stdout)
                        for mount in mounts:
                            source = mount.get("Source", "")
                            if dir_str in source or source.startswith(dir_str):
                                mount_details.append(f"Container '{container}' has mount: {source}")
                    except json.JSONDecodeError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Podman also not available - assume no mounts
        pass

    return len(mount_details) > 0, mount_details


# --- Provider And Model Selection ---


def select_provider(providers: dict[str, dict[str, Any]]) -> str | None:
    """Interactive provider selection.

    Args:
        providers: Provider metadata dictionary

    Returns:
        Selected provider name, or None if cancelled
    """
    # Validate providers dict before selection menus (fail gracefully if empty)
    if not providers:
        console.print(f"\n{Messages.error('No providers available')}")
        console.print(Messages.warning("Osprey could not load any AI providers."))
        console.print(
            f"[dim]Check that osprey is properly installed: {Messages.command('uv sync --all-extras')}[/dim]\n"
        )
        return None

    choices = []
    for key, p in sorted(providers.items()):
        try:
            # Validate provider metadata structure
            if not isinstance(p, dict):
                continue
            if "name" not in p or "description" not in p:
                if os.environ.get("DEBUG"):
                    console.print(f"[dim]Warning: Provider {key} missing required metadata[/dim]")
                continue

            # Description comes directly from provider class attribute
            key_info = " [requires API key]" if p.get("requires_key", True) else " [no API key]"
            display = f"{p['name']:12} - {p['description']}{key_info}"
            choices.append(Choice(display, value=key))
        except Exception as e:
            if os.environ.get("DEBUG"):
                console.print(f"[dim]Warning: Error processing provider {key}: {e}[/dim]")
            continue

    if not choices:
        console.print(f"\n{Messages.error('No valid providers found')}")
        console.print(f"{Messages.warning('All providers failed validation.')}\n")
        return None

    return questionary.select(
        "Select default AI provider:",
        choices=choices,
        style=custom_style,
        instruction="(This sets default provider in config.yml)",
    ).ask()


def select_model(provider: str, providers: dict[str, dict[str, Any]]) -> str | None:
    """Interactive model selection for chosen provider.

    Args:
        provider: Provider name
        providers: Provider metadata dictionary

    Returns:
        Selected model ID, or None if cancelled
    """
    provider_info = providers[provider]

    choices = [Choice(model, value=model) for model in provider_info["models"]]

    default = provider_info.get("default_model")

    return questionary.select(
        f"Select default model for {provider}:",
        choices=choices,
        style=custom_style,
        default=default if default in provider_info["models"] else None,
    ).ask()


# --- Navigation ---


def navigation_loop():
    """Main navigation loop."""
    while True:
        console.clear()
        show_banner(context="interactive")

        action = show_main_menu()

        if action is None or action == "exit":
            console.print("\n[accent]👋 Goodbye![/accent]\n")
            break

        # Handle tuple actions (project selection)
        if isinstance(action, tuple):
            action_type, action_data = action

            if action_type == "select_project":
                project_path = action_data
                handle_project_selection(project_path)
                continue

        # Handle string actions (standard commands)
        if action == "init_interactive":
            next_action = run_interactive_init()
            if next_action == "exit":
                break
        elif action == "deploy":
            handle_deploy_action()
        elif action == "health":
            handle_health_action()
        elif action == "config":
            handle_config_action()
        elif action == "registry":
            from osprey.cli.registry_cmd import handle_registry_action

            handle_registry_action()
        elif action == "help":
            # Show contextual help based on whether we're in a project or not
            if is_project_initialized():
                handle_help_action()
            else:
                handle_help_action_root()


# --- Entry Point ---


def launch_tui():
    """Entry point for TUI mode."""
    # Check dependencies
    if not questionary:
        console.print(Messages.error("Missing required dependency 'questionary'"))
        console.print("\nInstall with:")
        console.print(f"  {Messages.command('uv pip install questionary')}")
        console.print("\nOr install full osprey dependencies:")
        console.print(f"  {Messages.command('uv sync --all-extras')}\n")
        sys.exit(1)

    try:
        navigation_loop()
    except KeyboardInterrupt:
        console.print("\n\n[accent]👋 Goodbye![/accent]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n{Messages.error(f'Unexpected error: {e}')}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
