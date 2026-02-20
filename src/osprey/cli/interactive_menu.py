"""Interactive Terminal UI (TUI) for Osprey Framework CLI.

This module provides the interactive menu system that launches when the user
runs 'osprey' with no arguments. It provides a context-aware interface that
adapts based on whether the user is in a project directory.

Key Features:
- Context-aware main menu (different for "no project" vs "existing project")
- Interactive init flow with template, provider, and model selection
- Automatic environment variable detection (API keys from shell)
- Secure API key configuration with password input
- Smooth transitions between menu and direct commands
- Integration with existing Click commands (no duplication)

Architecture:
- Uses questionary for interactive prompts with custom styling
- Uses rich for terminal output and formatting
- Integrates with existing TemplateManager for project creation
- Integrates with registry system for provider metadata
- Calls underlying functions directly (not Click commands)

The TUI is optional - users can still use direct commands like:
    osprey init my-project
    osprey chat
    osprey deploy up

"""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# Import centralized styles
from osprey.cli.styles import (
    Messages,
    Styles,
    ThemeConfig,
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


# ============================================================================
# CONSOLE AND STYLING
# ============================================================================

# Use centralized questionary style
custom_style = get_questionary_style()


# ============================================================================
# BANNER AND BRANDING
# ============================================================================


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


# ============================================================================
# PROJECT DETECTION
# ============================================================================


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


# ============================================================================
# PROVIDER METADATA (from Registry)
# ============================================================================

# Cache for provider metadata (loaded once per TUI session)
_provider_cache: dict[str, dict[str, Any]] | None = None

# Cache for code generator metadata (loaded once per TUI session)
_code_generator_cache: dict[str, dict[str, Any]] | None = None


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
        from osprey.registry.registry import FrameworkRegistryProvider

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


def get_code_generator_metadata() -> dict[str, dict[str, Any]]:
    """Get code generator information from osprey registry.

    Loads generators directly from the osprey registry configuration,
    excluding mock generators (for testing only). This reads the
    framework's code generator registrations without requiring a
    project config.yml.

    Results are cached for the TUI session to avoid repeated registry loading.

    Returns:
        Dictionary mapping generator names to their metadata:
        {
            'basic': {
                'name': 'basic',
                'description': 'Simple single-pass LLM code generator',
                'available': True
            },
            'claude_code': {
                'name': 'claude_code',
                'description': 'Claude Code SDK-based generator...',
                'available': False,  # If dependencies not installed
                'optional_dependencies': ['claude-agent-sdk']
            },
            ...
        }
    """
    global _code_generator_cache

    # Return cached data if available
    if _code_generator_cache is not None:
        return _code_generator_cache

    import importlib

    try:
        # Import osprey registry provider directly (no config.yml needed!)
        from osprey.registry.registry import FrameworkRegistryProvider

        # Get osprey registry config (doesn't require project config)
        framework_registry = FrameworkRegistryProvider()
        config = framework_registry.get_registry_config()

        generators = {}

        # Load each code generator registration from osprey config
        for gen_reg in config.code_generators:
            # Skip mock generators (for testing only)
            if gen_reg.name == "mock":
                if os.environ.get("DEBUG"):
                    console.print("[dim]Skipping mock generator (testing only)[/dim]")
                continue

            try:
                # Try to import the generator to check availability
                module = importlib.import_module(gen_reg.module_path)
                _ = getattr(module, gen_reg.class_name)  # Check class exists

                # Generator is available
                generators[gen_reg.name] = {
                    "name": gen_reg.name,
                    "description": gen_reg.description,
                    "available": True,
                    "optional_dependencies": (
                        gen_reg.optional_dependencies
                        if hasattr(gen_reg, "optional_dependencies")
                        else []
                    ),
                }

            except ImportError as e:
                # Generator not available (missing dependencies)
                if hasattr(gen_reg, "optional_dependencies") and gen_reg.optional_dependencies:
                    # Optional dependency not installed - include but mark unavailable
                    generators[gen_reg.name] = {
                        "name": gen_reg.name,
                        "description": gen_reg.description,
                        "available": False,
                        "optional_dependencies": gen_reg.optional_dependencies,
                        "import_error": str(e),
                    }
                    if os.environ.get("DEBUG"):
                        console.print(f"[dim]Generator '{gen_reg.name}' unavailable: {e}[/dim]")
                else:
                    # Required dependency missing - this is an error
                    if os.environ.get("DEBUG"):
                        console.print(
                            f"[dim]Warning: Could not load generator {gen_reg.name}: {e}[/dim]"
                        )
                    continue

            except Exception as e:
                # Other errors - skip this generator
                if os.environ.get("DEBUG"):
                    console.print(
                        f"[dim]Warning: Could not load generator {gen_reg.name}: {e}[/dim]"
                    )
                continue

        if not generators:
            console.print(
                Messages.warning("No code generators could be loaded from osprey registry")
            )

        # Cache the result for future calls
        _code_generator_cache = generators
        return generators

    except Exception as e:
        # This should rarely happen - osprey registry should always be available
        console.print(Messages.error(f"Could not load code generators from osprey registry: {e}"))
        console.print(Messages.warning("The TUI requires access to code generator information."))
        if os.environ.get("DEBUG"):
            import traceback

            traceback.print_exc()

        # Return empty dict but don't cache failures
        return {}


# ============================================================================
# MAIN MENU
# ============================================================================


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
        Choice("[>] chat        - Start CLI conversation", value="chat"),
        Choice("[>] chat (tui)  - Start TUI conversation (experimental)", value="chat-tui"),
        Choice("[>] deploy      - Manage services (web UIs)", value="deploy"),
        Choice("[>] health      - Run system health check", value="health"),
        Choice("[>] generate    - Generate components", value="generate"),
        Choice("[>] config      - Configuration settings", value="config"),
        Choice("[>] registry    - Show registry contents", value="registry"),
        Choice("[>] tasks       - Browse AI assistant tasks", value="tasks"),
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
        console.print(f"Install with: {Messages.command('pip install questionary')}")
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
                Choice("[>] Browse AI assistant tasks", value="tasks"),
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


# ============================================================================
# DIRECTORY SAFETY CHECKS
# ============================================================================


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


# ============================================================================
# TEMPLATE SELECTION
# ============================================================================


def select_template(templates: list[str]) -> str | None:
    """Interactive template selection.

    Args:
        templates: List of available template names

    Returns:
        Selected template name, or None if cancelled
    """
    # Template descriptions (could also come from template metadata)
    descriptions = {
        "minimal": "Empty project structure with TODO placeholders",
        "hello_world_weather": "Single capability weather example (tutorial)",
        "control_assistant": "Control system integration with channel finder (production-grade)",
    }

    choices = []
    for template in templates:
        desc = descriptions.get(template, "No description available")
        display = f"{template:22} - {desc}"
        choices.append(Choice(display, value=template))

    return questionary.select("Select project template:", choices=choices, style=custom_style).ask()


def get_default_name_for_template(template: str) -> str:
    """Get a sensible default project name for the template.

    Args:
        template: Template name

    Returns:
        Default project name suggestion
    """
    defaults = {
        "minimal": "my-agent",
        "hello_world_weather": "weather-agent",
        "control_assistant": "my-control-assistant",
    }
    return defaults.get(template, "my-project")


def select_channel_finder_mode() -> str | None:
    """Interactive channel finder mode selection for control_assistant template.

    Returns:
        Selected mode ('in_context', 'hierarchical', 'middle_layer', 'all'), or None if cancelled
    """
    console.print("[dim]Select the channel finding approach for your control system:[/dim]\n")

    choices = [
        Choice(
            "in_context       - Semantic search (flat database, best for <200 channels)",
            value="in_context",
        ),
        Choice(
            "hierarchical     - Pattern navigation (builds channel address from naming rules, scalable)",
            value="hierarchical",
        ),
        Choice(
            "middle_layer     - Functional exploration (retrieves channel address by function, scalable)",
            value="middle_layer",
        ),
        Choice(
            "all              - Include all three pipelines (maximum flexibility, comparison)",
            value="all",
        ),
    ]

    return questionary.select("Channel finder mode:", choices=choices, style=custom_style).ask()


def select_code_generator(generators: dict[str, dict[str, Any]]) -> str | None:
    """Interactive code generator selection.

    Shows all available code generators from the registry, with clear indication
    of which ones are available vs. require additional dependencies.

    Args:
        generators: Code generator metadata dictionary from get_code_generator_metadata()

    Returns:
        Selected generator name, or None if cancelled
    """
    if not generators:
        console.print(f"\n{Messages.error('No code generators available')}")
        console.print(Messages.warning("Osprey could not load any code generators."))
        console.print(
            f"[dim]Check that osprey is properly installed: {Messages.command('pip install -e .[all]')}[/dim]\n"
        )
        return None

    console.print("[dim]Select the code generation strategy for Python execution:[/dim]\n")

    choices = []
    default_choice = None

    # Sort generators: available first, then unavailable
    sorted_generators = sorted(
        generators.items(), key=lambda x: (not x[1].get("available", False), x[0])
    )

    for gen_name, gen_info in sorted_generators:
        is_available = gen_info.get("available", False)
        description = gen_info.get("description", "No description available")

        if is_available:
            # Available generator
            display = f"{gen_name:15} - {description}"
            choices.append(Choice(display, value=gen_name))

            # Set basic as default if available
            if gen_name == "basic" and default_choice is None:
                default_choice = gen_name

        else:
            # Unavailable generator (missing optional dependencies)
            deps = gen_info.get("optional_dependencies", [])
            deps_str = ", ".join(deps) if deps else "unknown dependencies"
            display = f"{gen_name:15} - [dim]{description} (requires: {deps_str})[/dim]"
            choices.append(Choice(display, value=gen_name, disabled=True))

    if not any(not c.disabled for c in choices if hasattr(c, "disabled")):
        console.print(f"\n{Messages.error('No available code generators found')}")
        console.print(f"{Messages.warning('All generators require additional dependencies.')}\n")
        return None

    return questionary.select(
        "Code generator:", choices=choices, style=custom_style, default=default_choice
    ).ask()


# ============================================================================
# PROVIDER AND MODEL SELECTION
# ============================================================================


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
            f"[dim]Check that osprey is properly installed: {Messages.command('pip install -e .[all]')}[/dim]\n"
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


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================


def get_api_key_name(provider: str) -> str | None:
    """Get environment variable name for provider API key.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai')

    Returns:
        Environment variable name, or None if provider doesn't need API key
    """
    key_names = {
        "cborg": "CBORG_API_KEY",
        "amsc": "AMSC_API_KEY",
        "stanford": "STANFORD_API_KEY",
        "argo": "ARGO_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,  # Ollama doesn't need API key
    }
    return key_names.get(provider, f"{provider.upper()}_API_KEY")


def configure_api_key(
    provider: str, project_path: Path, providers: dict[str, dict[str, Any]]
) -> bool:
    """Configure API key for the selected provider.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai')
        project_path: Path to project directory
        providers: Provider metadata dictionary

    Returns:
        True if API key configured successfully, False otherwise
    """
    console.print(f"\n{Messages.header('API Key Configuration')}\n")

    # Get key name
    key_name = get_api_key_name(provider)

    if not key_name:
        console.print(Messages.success(f"Provider '{provider}' does not require an API key"))
        return True

    console.print(f"Provider: [accent]{provider}[/accent]")
    console.print(f"Required: [accent]{key_name}[/accent]\n")

    # Check if already detected from environment
    from osprey.cli.templates import TemplateManager

    manager = TemplateManager()
    detected_env = manager._detect_environment_variables()

    if key_name in detected_env:
        console.print(Messages.success("API key already detected from environment"))
        console.print(f"[dim]Value: {detected_env[key_name][:10]}...[/dim]\n")

        use_detected = questionary.confirm(
            "Use detected API key?",
            default=True,
            style=custom_style,
        ).ask()

        if use_detected:
            write_env_file(project_path, key_name, detected_env[key_name])
            return True

    # Give user options
    action = questionary.select(
        "How would you like to configure the API key?",
        choices=[
            Choice("[#] Paste API key now (secure input)", value="paste"),
            Choice("[-] Configure later (edit .env manually)", value="later"),
            Choice("[?] Where do I get an API key?", value="help"),
        ],
        style=custom_style,
    ).ask()

    if action == "help":
        show_api_key_help(provider)
        return configure_api_key(provider, project_path, providers)  # Ask again

    elif action == "paste":
        console.print(f"\n[dim]Enter your {key_name} (input will be hidden)[/dim]")

        api_key = questionary.password(
            f"{key_name}:",
            style=custom_style,
        ).ask()

        if api_key and len(api_key.strip()) > 0:
            write_env_file(project_path, key_name, api_key.strip())
            console.print(f"\n{Messages.success(f'{key_name} configured securely')}\n")
            return True
        else:
            console.print(f"\n{Messages.warning('No API key provided')}\n")
            return False

    elif action == "later":
        show_manual_config_instructions(provider, key_name, project_path)
        return False

    return False


def write_env_file(project_path: Path, key_name: str, api_key: str):
    """Write API key to .env file with proper permissions.

    Args:
        project_path: Path to project directory
        key_name: Environment variable name
        api_key: API key value
    """
    from dotenv import set_key

    env_file = project_path / ".env"

    # Copy from .env.example if doesn't exist
    if not env_file.exists():
        env_example = project_path / ".env.example"
        if env_example.exists():
            shutil.copy(env_example, env_file)
        else:
            env_file.touch()

    # Set the key
    set_key(str(env_file), key_name, api_key)

    # Set permissions to 600 (owner read/write only)
    os.chmod(env_file, 0o600)

    console.print("  [success]✓[/success] Wrote {key_name} to .env")
    console.print("  [success]✓[/success] Set file permissions to 600")


def show_api_key_help(provider: str):
    """Show provider-specific instructions for getting API keys.

    Reads metadata from provider class to ensure single source of truth.

    Args:
        provider: Provider name
    """
    console.print()

    # Try to get provider metadata from cached registry data
    try:
        providers = get_provider_metadata()
        provider_data = providers.get(provider)

        if not provider_data:
            # Fallback for unknown providers
            console.print(f"[dim]Check {provider} documentation for API key instructions[/dim]\n")
            input("Press ENTER to continue...")
            return

        # Display provider-specific instructions from metadata
        provider_display = provider_data.get("description") or provider.title()
        console.print(f"[bold]Getting a {provider_display} API Key:[/bold]")

        # Show URL if available
        api_key_url = provider_data.get("api_key_url")
        if api_key_url:
            console.print(f"  1. Visit: {api_key_url}")
            step_offset = 2
        else:
            step_offset = 1

        # Show instructions
        api_key_instructions = provider_data.get("api_key_instructions", [])
        if api_key_instructions:
            for i, instruction in enumerate(api_key_instructions, start=step_offset):
                console.print(f"  {i}. {instruction}")
            console.print()  # Extra line after instructions

        # Show note if available
        api_key_note = provider_data.get("api_key_note")
        if api_key_note:
            console.print(f"[dim]Note: {api_key_note}[/dim]\n")

    except Exception as e:
        # Fallback in case of any errors
        console.print(f"[dim]Check {provider} documentation for API key instructions[/dim]")
        console.print(f"[dim](Error loading provider info: {e})[/dim]\n")

    input("Press ENTER to continue...")


def show_manual_config_instructions(provider: str, key_name: str, project_path: Path):
    """Show instructions for manual API key configuration.

    Args:
        provider: Provider name
        key_name: Environment variable name
        project_path: Path to project directory
    """
    console.print(f"\n{Messages.info('API key not configured')}")
    console.print("\n[bold]To configure manually:[/bold]")
    console.print(f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}")
    console.print(f"  2. Copy template: {Messages.command('cp .env.example .env')}")
    console.print(f"  3. Edit .env and set {key_name}")
    console.print(f"  4. Set permissions: {Messages.command('chmod 600 .env')}\n")


# ============================================================================
# INTERACTIVE INIT FLOW
# ============================================================================


def run_interactive_init() -> str:
    """Interactive init flow with provider/model selection.

    Returns:
        Navigation action ('menu', 'exit', 'chat', etc.)
    """
    console.clear()
    show_banner(context="interactive")
    console.print(f"\n{Messages.header('Create New Project')}\n")

    # Get dynamic data with loading indicator
    from osprey.cli.templates import TemplateManager

    manager = TemplateManager()

    try:
        # Show spinner while loading
        with console.status(
            "[dim]Loading templates, providers, and code generators...[/dim]", spinner="dots"
        ):
            templates = manager.list_app_templates()
            providers = get_provider_metadata()
            code_generators = get_code_generator_metadata()
    except Exception as e:
        console.print(f"[error]✗ Error loading templates/providers/generators:[/error] {e}")
        input("\nPress ENTER to continue...")
        return "menu"

    # 1. Template selection
    console.print("[bold]Step 1: Select Template[/bold]\n")
    template = select_template(templates)
    if template is None:
        return "menu"

    # 2. Project name
    console.print("\n[bold]Step 2: Project Name[/bold]\n")
    project_name = questionary.text(
        "Project name:",
        default=get_default_name_for_template(template),
        style=custom_style,
    ).ask()

    if not project_name:
        return "menu"

    # 2b. Channel finder mode (only for control_assistant template)
    channel_finder_mode = None
    if template == "control_assistant":
        console.print("\n[bold]Step 3: Channel Finder Configuration[/bold]\n")
        channel_finder_mode = select_channel_finder_mode()
        if channel_finder_mode is None:
            return "menu"

    # 2c. Control capabilities selection (native framework capabilities)
    control_capabilities = None
    if template == "control_assistant":
        console.print("\n[bold]Step 4: Control System Capabilities[/bold]\n")
        console.print(
            "[dim]The framework provides these native capabilities (all enabled by default):[/dim]\n"
        )

        all_caps = [
            Choice(
                "channel_finding     - Find control system channels by description",
                value="channel_finding",
                checked=True,
            ),
            Choice(
                "channel_read        - Read current channel values",
                value="channel_read",
                checked=True,
            ),
            Choice(
                "channel_write       - Write values to channels (with safety controls)",
                value="channel_write",
                checked=True,
            ),
            Choice(
                "archiver_retrieval  - Query historical time-series data",
                value="archiver_retrieval",
                checked=True,
            ),
        ]

        selected = questionary.checkbox(
            "Control capabilities:",
            choices=all_caps,
            style=custom_style,
        ).ask()

        if selected is None:
            return "menu"

        # Validate: channel_finding required if any others are selected
        if selected and "channel_finding" not in selected:
            other_caps = [c for c in selected if c != "channel_finding"]
            if other_caps:
                console.print(
                    f"\n{Messages.warning('channel_finding is required when using: ' + ', '.join(other_caps))}"
                )
                selected.insert(0, "channel_finding")
                console.print(f"{Messages.info('Automatically included channel_finding')}\n")

        control_capabilities = selected if selected else None

    # 2d. Code generator selection (for templates that use Python execution)
    # Skip for hello_world_weather (simple example), include for control_assistant
    code_generator = None
    if template == "control_assistant":
        step_num = 5  # After channel finder + capabilities
        console.print(f"\n[bold]Step {step_num}: Code Generator[/bold]\n")
        code_generator = select_code_generator(code_generators)
        if code_generator is None:
            return "menu"

    # Check if project directory already exists (before other configuration steps)
    project_path = Path.cwd() / project_name
    if project_path.exists():
        msg = Messages.warning(f"Directory '{project_path}' already exists.")
        console.print(f"\n{msg}\n")

        # Check if directory exists immediately before deletion (safety check) and check for active Docker/Podman mounts before allowing deletion
        has_mounts, mount_details = check_directory_has_active_mounts(project_path)

        if has_mounts:
            console.print(
                f"{Messages.error('⚠️  DANGER: This directory has active container mounts!')}"
            )
            console.print(
                f"{Messages.warning('The following containers are using this directory:')}\n"
            )
            for detail in mount_details:
                console.print(f"  • {detail}")
            console.print("\n[bold]You MUST stop containers before deleting this directory:[/bold]")
            console.print(f"  {Messages.command(f'cd {project_name} && osprey deploy down')}\n")

            proceed_anyway = questionary.confirm(
                "⚠️  Delete anyway? (This may corrupt running containers!)",
                default=False,
                style=custom_style,
            ).ask()

            if not proceed_anyway:
                console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
                console.print(
                    f"[dim]Tip: Stop containers first with {Messages.command('osprey deploy down')}[/dim]"
                )
                input("\nPress ENTER to continue...")
                return "menu"

        action = questionary.select(
            "What would you like to do?",
            choices=[
                Choice(
                    "[!] Override - Delete existing directory and create new project",
                    value="override",
                ),
                Choice("[*] Rename - Choose a different project name", value="rename"),
                Choice("[-] Abort - Return to main menu", value="abort"),
            ],
            style=custom_style,
        ).ask()

        if action == "abort" or action is None:
            console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
            input("\nPress ENTER to continue...")
            return "menu"
        elif action == "rename":
            # Go back to project name input
            console.print("\n[bold]Choose a different project name:[/bold]\n")
            new_project_name = questionary.text(
                "Project name:",
                default=f"{project_name}-2",
                style=custom_style,
            ).ask()

            if not new_project_name:
                return "menu"

            project_name = new_project_name
            project_path = Path.cwd() / project_name

            # Check again if new name exists
            if project_path.exists():
                msg = Messages.warning(f"Directory '{project_path}' also exists.")
                console.print(f"\n{msg}")
                override = questionary.confirm(
                    "Override existing directory?",
                    default=False,
                    style=custom_style,
                ).ask()

                if not override:
                    console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
                    input("\nPress ENTER to continue...")
                    return "menu"

                # Delete existing directory
                console.print("\n[dim]Removing existing directory...[/dim]")

                # Check directory exists immediately before deletion (TOCTOU protection)
                if not project_path.exists():
                    console.print(
                        Messages.warning("Directory was already deleted by another process")
                    )
                else:
                    try:
                        shutil.rmtree(project_path)
                        console.print(f"  {Messages.success('Removed existing directory')}")
                    except PermissionError as e:
                        console.print(f"\n{Messages.error(f'Permission denied: {e}')}")
                        console.print(
                            Messages.warning(
                                "Try running with appropriate permissions or stop any running processes"
                            )
                        )
                        input("\nPress ENTER to continue...")
                        return "menu"
                    except OSError as e:
                        console.print(f"\n{Messages.error(f'Could not delete directory: {e}')}")
                        input("\nPress ENTER to continue...")
                        return "menu"
        elif action == "override":
            # Delete existing directory
            console.print("\n[dim]Removing existing directory...[/dim]")

            # Check directory exists immediately before deletion (TOCTOU protection)
            if not project_path.exists():
                console.print(Messages.warning("Directory was already deleted by another process"))
            else:
                try:
                    shutil.rmtree(project_path)
                    console.print(f"  {Messages.success('Removed existing directory')}")
                except PermissionError as e:
                    console.print(f"\n{Messages.error(f'Permission denied: {e}')}")
                    console.print(
                        Messages.warning(
                            "Try running with appropriate permissions or stop any running processes"
                        )
                    )
                    input("\nPress ENTER to continue...")
                    return "menu"
                except OSError as e:
                    console.print(f"\n{Messages.error(f'Could not delete directory: {e}')}")
                    input("\nPress ENTER to continue...")
                    return "menu"

    # 3. Registry style (step number adjusts based on previous steps)
    if template == "control_assistant":
        step_num = 6  # After template, name, channel_finder, capabilities, code_generator
    else:
        step_num = 3  # After template, name
    console.print(f"\n[bold]Step {step_num}: Registry Style[/bold]\n")

    registry_style = questionary.select(
        "Select registry style:",
        choices=[
            Choice("extend     - Extends framework defaults (recommended)", value="extend"),
            Choice("standalone - Complete explicit registry (advanced)", value="standalone"),
        ],
        style=custom_style,
        instruction="(extend mode is recommended for most projects)",
    ).ask()

    if registry_style is None:
        return "menu"

    # 4. Provider selection (step number adjusts)
    if template == "control_assistant":
        step_num = 7  # After template, name, channel_finder, capabilities, code_generator, registry
    else:
        step_num = 4  # After template, name, registry
    console.print(f"\n[bold]Step {step_num}: AI Provider[/bold]\n")
    provider = select_provider(providers)
    if provider is None:
        return "menu"

    # 5. Model selection (step number adjusts)
    if template == "control_assistant":
        step_num = 8  # After template, name, channel_finder, capabilities, code_generator, registry, provider
    else:
        step_num = 5  # After template, name, registry, provider
    console.print(f"\n[bold]Step {step_num}: Model Selection[/bold]\n")
    model = select_model(provider, providers)
    if model is None:
        return "menu"

    # Summary
    console.print(f"\n{Messages.header('Configuration Summary:')}")
    console.print(f"  Project:       [value]{project_name}[/value]")
    console.print(f"  Template:      [value]{template}[/value]")
    if channel_finder_mode:
        console.print(f"  Pipeline:      [value]{channel_finder_mode}[/value]")
    if control_capabilities is not None:
        caps_str = ", ".join(control_capabilities) if control_capabilities else "none"
        console.print(f"  Capabilities:  [value]{caps_str}[/value]")
    if code_generator:
        console.print(f"  Code Gen:      [value]{code_generator}[/value]")
    console.print(f"  Registry:      [value]{registry_style}[/value]")
    console.print(f"  Provider:      [value]{provider}[/value]")
    console.print(f"  Model:         [value]{model}[/value]\n")

    # Confirm
    proceed = questionary.confirm(
        "Create project with these settings?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
        input("\nPress ENTER to continue...")
        return "menu"

    # Create project
    console.print("\n[bold]Creating project...[/bold]\n")

    try:
        # Note: force=True because we already handled directory deletion if user chose override
        # Build context dict with optional channel_finder_mode and code_generator
        context = {"default_provider": provider, "default_model": model}
        if channel_finder_mode:
            context["channel_finder_mode"] = channel_finder_mode
        if control_capabilities is not None:
            context["control_capabilities"] = control_capabilities
        if code_generator:
            context["code_generator"] = code_generator

        project_path = manager.create_project(
            project_name=project_name,
            output_dir=Path.cwd(),
            template_name=template,
            registry_style=registry_style,
            context=context,
            force=True,
        )

        # Generate manifest for migration support
        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=template,
            registry_style=registry_style,
            context=context,
        )

        msg = Messages.success("Project created at:")
        path = Messages.path(str(project_path))
        console.print(f"\n{msg} {path}\n")

        # Check if API keys were detected and .env was created
        detected_env = manager._detect_environment_variables()
        api_keys = [
            "CBORG_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "STANFORD_API_KEY",
        ]
        has_api_keys = any(key in detected_env for key in api_keys)

        if has_api_keys:
            env_file = project_path / ".env"
            if env_file.exists():
                console.print(Messages.success("Created .env with detected API keys"))
                detected_keys = [key for key in api_keys if key in detected_env]
                console.print(f"[dim]  Detected: {', '.join(detected_keys)}[/dim]\n")

        # API key configuration
        if providers[provider]["requires_key"]:
            api_configured = configure_api_key(provider, project_path, providers)
        else:
            api_configured = True

        # Success summary
        show_success_art()
        console.print(Messages.success("Project created successfully!") + "\n")

        # Offer to launch chat immediately
        if api_configured:
            console.print("[bold]What would you like to do next?[/bold]\n")

            next_action = questionary.select(
                "Select action:",
                choices=[
                    Choice("[>] Start chat in this project now", value="chat"),
                    Choice("[<] Return to main menu", value="menu"),
                    Choice("[x] Exit and show next steps", value="exit"),
                ],
                style=custom_style,
            ).ask()

            if next_action == "chat":
                console.print(f"\n[dim]Launching chat for project: {project_path.name}[/dim]\n")
                handle_chat_action(project_path=project_path)
                return "menu"
            elif next_action == "exit":
                # Show next steps like the direct init command
                console.print("\n[bold]Next steps:[/bold]")
                console.print(
                    f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}"
                )
                console.print("  2. # .env already configured with API key")
                console.print(f"  3. Start chatting: {Messages.command('osprey chat')}")
                console.print(f"  4. Start services: {Messages.command('osprey deploy up')}")
                console.print()
                return "exit"
        else:
            console.print("[bold]Next steps:[/bold]")
            console.print(
                f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}"
            )
            console.print(
                f"  2. Configure API key: {Messages.command('cp .env.example .env')} (then edit)"
            )
            console.print(f"  3. Start chatting: {Messages.command('osprey chat')}")
            console.print(f"  4. Start services: {Messages.command('osprey deploy up')}")

            console.print("\n[dim]Press ENTER to continue...[/dim]")
            input()

        return "menu"

    except ValueError as e:
        # This should not happen anymore since we check directory existence above
        # But catch it just in case
        console.print(f"\n[error]✗ Error creating project:[/error] {e}")
        input("\nPress ENTER to continue...")
        return "menu"
    except Exception as e:
        console.print(f"\n[error]✗ Unexpected error creating project:[/error] {e}")
        if os.environ.get("DEBUG"):
            import traceback

            traceback.print_exc()
        input("\nPress ENTER to continue...")
        return "menu"


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def handle_project_selection(project_path: Path):
    """Handle selection of a discovered project from subdirectory.

    Shows project-specific menu in a loop until user chooses to go back.

    Args:
        project_path: Path to the selected project directory
    """
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
        if action == "chat":
            handle_chat_action(project_path=project_path)
        elif action == "chat-tui":
            handle_chat_tui_action(project_path=project_path)
        elif action == "deploy":
            handle_deploy_action(project_path=project_path)
        elif action == "health":
            handle_health_action(project_path=project_path)
        elif action == "generate":
            # Generate needs to run in the project directory
            original_dir = Path.cwd()
            try:
                os.chdir(project_path)
                handle_generate_action()
            finally:
                try:
                    os.chdir(original_dir)
                except (OSError, PermissionError):
                    pass
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


def handle_chat_action(project_path: Path | None = None):
    """Start chat interface - calls underlying function directly.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    try:
        from osprey.interfaces.cli.direct_conversation import run_cli
    except ImportError as e:
        console.print(f"\n{Messages.error('Import Error: Could not load chat interface')}")
        console.print(f"[dim]{e}[/dim]")
        input("\nPress ENTER to continue...")
        return
    except Exception as e:
        # Handle pydantic compatibility issues and other import errors
        error_msg = str(e)
        console.print(f"\n{Messages.error(f'Dependency Error: {error_msg}')}\n")

        if "TypedDict" in error_msg and "Python < 3.12" in error_msg:
            console.print(
                f"{Messages.warning('This appears to be a pydantic/Python version compatibility issue.')}\n"
            )
            console.print("[bold]Possible solutions:[/bold]")
            console.print("  1. Upgrade typing_extensions:")
            console.print(f"     {Messages.command('pip install --upgrade typing-extensions')}\n")
            console.print("  2. Upgrade pydantic:")
            console.print(
                f"     {Messages.command('pip install --upgrade pydantic pydantic-core')}\n"
            )
            console.print("  3. Or upgrade to Python 3.12+\n")
        else:
            console.print(Messages.warning("There was an error loading the chat dependencies."))
            console.print(
                f"[dim]Try reinstalling osprey dependencies: {Messages.command('pip install -e .[all]')}[/dim]\n"
            )

        if os.environ.get("DEBUG"):
            console.print("\n[dim]Full traceback:[/dim]")
            import traceback

            traceback.print_exc()

        input("\nPress ENTER to continue...")
        return

    console.print(f"\n{Messages.header('Starting Osprey CLI interface...')}")
    console.print("   [dim]Press Ctrl+C to exit[/dim]\n")

    try:
        if project_path:
            # When launching chat in a specific project, we need to:
            # 1. Reset the global registry to prevent state leakage between projects
            # 2. Set CONFIG_FILE env var so config loading works
            # 3. Change to project directory so relative paths work
            config_path = str(project_path / "config.yml")

            # Save original state
            original_dir = Path.cwd()
            original_config_env = os.environ.get("CONFIG_FILE")

            try:
                # Reset the global registry to ensure we load the correct project configuration
                # This prevents capability leakage when switching between projects in the same session
                from osprey.registry import reset_registry

                reset_registry()

                # Set up environment for the project
                os.environ["CONFIG_FILE"] = config_path

                # Add exception handling around os.chdir() operations
                try:
                    os.chdir(project_path)
                except (OSError, PermissionError) as e:
                    console.print(f"\n{Messages.error(f'Cannot change to project directory: {e}')}")
                    return

                # Run chat
                asyncio.run(run_cli(config_path=config_path))

            finally:
                # Restore original state
                try:
                    os.chdir(original_dir)
                except (OSError, PermissionError) as e:
                    # If we can't restore, at least warn the user
                    console.print(
                        f"\n{Messages.warning(f'Could not restore original directory: {e}')}"
                    )
                    console.print(
                        f"[dim]Current directory may have changed. Original was: {original_dir}[/dim]"
                    )

                if original_config_env is not None:
                    os.environ["CONFIG_FILE"] = original_config_env
                elif "CONFIG_FILE" in os.environ:
                    del os.environ["CONFIG_FILE"]
        else:
            # Default behavior - run in current directory
            # Reset the global registry to ensure we load the correct project configuration
            from osprey.registry import reset_registry

            reset_registry()

            # Set CONFIG_FILE for subprocess execution (critical for Python executor)
            config_path = str(Path.cwd() / "config.yml")
            os.environ["CONFIG_FILE"] = config_path
            asyncio.run(run_cli(config_path=config_path))

    except KeyboardInterrupt:
        console.print(f"\n\n{Messages.warning('Chat session ended.')}")
        # No pause needed - user intentionally exited with Ctrl+C
    except Exception as e:
        console.print(f"\n{Messages.error(f'Chat error: {e}')}")
        if os.environ.get("DEBUG"):
            import traceback

            traceback.print_exc()
        input("\nPress ENTER to continue...")

    # Return to menu (with pause only for actual errors)


def handle_chat_tui_action(project_path: Path | None = None):
    """Start TUI chat interface (experimental).

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    try:
        from osprey.interfaces.tui import run_tui
    except ImportError as e:
        console.print(f"\n{Messages.error('TUI not available')}")
        console.print(f"[dim]{e}[/dim]\n")
        console.print("The TUI requires the 'textual' package which is not installed.")
        console.print("Install with: [bold cyan]pip install osprey-framework\\[tui][/bold cyan]\n")
        input("\nPress ENTER to continue...")
        return
    except Exception as e:
        error_msg = str(e)
        console.print(f"\n{Messages.error(f'Dependency Error: {error_msg}')}\n")
        console.print(Messages.warning("There was an error loading the TUI dependencies."))
        console.print(
            "Try reinstalling: [bold cyan]pip install osprey-framework\\[tui][/bold cyan]\n"
        )

        if os.environ.get("DEBUG"):
            console.print("\n[dim]Full traceback:[/dim]")
            import traceback

            traceback.print_exc()

        input("\nPress ENTER to continue...")
        return

    console.print(f"\n{Messages.header('Starting Osprey TUI interface (experimental)...')}")
    console.print("   [dim]Press 'q' or double Ctrl+C to exit[/dim]\n")

    try:
        if project_path:
            # When launching TUI in a specific project
            config_path = str(project_path / "config.yml")

            # Save original state
            original_dir = Path.cwd()
            original_config_env = os.environ.get("CONFIG_FILE")

            try:
                # Reset the global registry
                from osprey.registry import reset_registry

                reset_registry()

                # Set up environment for the project
                os.environ["CONFIG_FILE"] = config_path

                try:
                    os.chdir(project_path)
                except (OSError, PermissionError) as e:
                    console.print(f"\n{Messages.error(f'Cannot change to project directory: {e}')}")
                    return

                # Run TUI
                asyncio.run(run_tui(config_path=config_path))

            finally:
                # Restore original state
                try:
                    os.chdir(original_dir)
                except (OSError, PermissionError) as e:
                    console.print(
                        f"\n{Messages.warning(f'Could not restore original directory: {e}')}"
                    )
                    console.print(
                        f"[dim]Current directory may have changed. Original was: {original_dir}[/dim]"
                    )

                if original_config_env is not None:
                    os.environ["CONFIG_FILE"] = original_config_env
                elif "CONFIG_FILE" in os.environ:
                    del os.environ["CONFIG_FILE"]
        else:
            # Default behavior - run in current directory
            from osprey.registry import reset_registry

            reset_registry()

            config_path = str(Path.cwd() / "config.yml")
            os.environ["CONFIG_FILE"] = config_path
            asyncio.run(run_tui(config_path=config_path))

    except KeyboardInterrupt:
        console.print(f"\n\n{Messages.warning('TUI session ended.')}")
    except Exception as e:
        console.print(f"\n{Messages.error(f'TUI error: {e}')}")
        if os.environ.get("DEBUG"):
            import traceback

            traceback.print_exc()
        input("\nPress ENTER to continue...")

    # Return to menu (with pause only for actual errors)


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
        f"  • [{Styles.DIM}]Use this to start your web UI services (Open WebUI, Jupyter, etc.)[/{Styles.DIM}]"
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


def handle_deploy_action(project_path: Path | None = None):
    """Manage deployment services menu.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
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


def handle_workflows_action():
    """Handle workflows export action from interactive menu.

    Exports AI workflow markdown files to a local directory for easy
    access by AI coding assistants.
    """
    from osprey.cli.workflows_cmd import get_workflows_source_path

    console.print(f"\n{Messages.header('Export AI Workflow Files')}")
    console.print(
        f"[{Styles.DIM}]These markdown files guide AI coding assistants through common tasks[/{Styles.DIM}]\n"
    )

    # Check if workflows are available
    source = get_workflows_source_path()
    if not source or not source.exists():
        console.print(Messages.error("Workflow files not found in installed package"))
        console.print(
            f"[{Styles.DIM}]This might indicate a packaging issue. "
            f"Try reinstalling: pip install --force-reinstall osprey-framework[/{Styles.DIM}]"
        )
        input("\nPress ENTER to continue...")
        return

    # Simple: export to default location in current directory
    target = Path.cwd() / "osprey-workflows"

    # Check if already exists
    if target.exists():
        console.print(f"{Messages.info('Target directory already exists:')} {target}\n")
        overwrite = questionary.confirm(
            "Overwrite existing files?", default=False, style=custom_style
        ).ask()

        if not overwrite:
            console.print(f"[{Styles.DIM}]Export cancelled[/{Styles.DIM}]")
            input("\nPress ENTER to continue...")
            return

    try:
        # Call the export function programmatically
        import shutil

        # Create target directory
        target.mkdir(parents=True, exist_ok=True)

        # Copy workflow files
        console.print(f"{Messages.header('Exporting workflows to:')} {target}\n")

        copied = 0
        for wf_file in source.iterdir():
            if wf_file.suffix == ".md":
                dest_file = target / wf_file.name
                shutil.copy2(wf_file, dest_file)
                console.print(f"  [{Styles.SUCCESS}]✓[/{Styles.SUCCESS}] {wf_file.name}")
                copied += 1

        console.print(f"\n{Messages.success(f'✓ Exported {copied} workflow files')}")

        # Usage instructions
        console.print(f"\n{Messages.header('Usage in AI coding assistants:')}")
        console.print(
            f"  {Messages.command('@src/osprey/workflows/testing-workflow.md What type of test?')}"
        )
        console.print(
            f"  {Messages.command('@src/osprey/workflows/pre-merge-cleanup.md Scan changes')}"
        )
        console.print(
            f"\n[{Styles.DIM}]Learn more: https://als-apg.github.io/osprey/contributing/03_ai-assisted-development.html[/{Styles.DIM}]"
        )

    except Exception as e:
        console.print(f"\n{Messages.error(f'Export failed: {e}')}")

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
                    default_provider="cborg",
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
    console.print(f"[{Styles.HEADER}][>] chat - Start CLI conversation[/{Styles.HEADER}]")
    console.print()
    console.print("  • Opens an interactive chat session with your AI agent")
    console.print("  • Use natural language to query data, execute code, or control systems")
    console.print("  • Supports slash commands (type /help in chat for details)")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Testing your agent, exploring capabilities, debugging[/{Styles.DIM}]"
    )
    console.print()

    # chat-tui
    console.print(
        f"[{Styles.HEADER}][>] chat (tui) - Start TUI conversation (experimental)[/{Styles.HEADER}]"
    )
    console.print()
    console.print("  • Full-screen terminal interface built with Textual")
    console.print("  • Real-time streaming with step-by-step visualization")
    console.print("  • Theme support, command palette (Ctrl+P), slash commands")
    console.print("  • Requires: pip install osprey-framework\\[tui]")
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Visual debugging, monitoring agent reasoning[/{Styles.DIM}]"
    )
    console.print()

    # deploy
    console.print(f"[{Styles.HEADER}][>] deploy - Manage services (web UIs)[/{Styles.HEADER}]")
    console.print()
    console.print("  • Start, stop, and manage containerized services")
    console.print("  • Launch web interfaces (Open WebUI, Jupyter notebooks)")
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


# ============================================================================
# GENERATE MENU
# ============================================================================


def show_generate_help():
    """Display detailed help for all generate options."""
    console.clear()
    show_banner(context="interactive")

    console.print(f"\n{Messages.header('Generate Components - Help')}\n")

    # capability option
    console.print(
        f"[{Styles.HEADER}][→] capability - From MCP server or natural language[/{Styles.HEADER}]"
    )
    console.print()
    console.print(f"  [{Styles.ACCENT}]From MCP Server:[/{Styles.ACCENT}]")
    console.print("  • Connects to a running MCP (Model Context Protocol) server")
    console.print("  • Introspects available tools and resources")
    console.print("  • Generates production-ready capability with ReAct agent")
    console.print("  • Includes automatic retry logic and error handling")
    console.print(
        f"  • [{Styles.DIM}]Best for: Integrating external services (weather, databases, APIs)[/{Styles.DIM}]"
    )
    console.print()
    console.print(f"  [{Styles.ACCENT}]From Natural Language Prompt:[/{Styles.ACCENT}]")
    console.print("  • Describe what the capability should do in plain English")
    console.print("  • LLM generates a skeleton capability class")
    console.print("  • You implement the actual logic (marked with TODO comments)")
    console.print("  • Includes type hints and docstrings")
    console.print(f"  • [{Styles.DIM}]Best for: Custom domain-specific capabilities[/{Styles.DIM}]")
    console.print()

    # mcp-server option
    console.print(
        f"[{Styles.HEADER}][→] mcp-server - Demo MCP server for testing[/{Styles.HEADER}]"
    )
    console.print()
    console.print("  • Creates demo weather MCP server for testing and learning")
    console.print("  • FastMCP-based HTTP server")
    console.print("  • Includes example tools (get_weather, get_forecast)")
    console.print(
        f"  • Ready to run: just [{Styles.VALUE}]pip install fastmcp && python server.py[/{Styles.VALUE}]"
    )
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Testing capability generation, learning MCP protocol[/{Styles.DIM}]"
    )
    console.print()

    # claude-config option
    console.print(
        f"[{Styles.HEADER}][→] claude-config - Claude Code generator configuration[/{Styles.HEADER}]"
    )
    console.print()
    console.print(
        f"  • Generates [{Styles.VALUE}]claude_generator_config.yml[/{Styles.VALUE}] for Claude Code SDK"
    )
    console.print("  • Configures code generation profiles (fast vs robust)")
    console.print("  • Sets up agentic code execution with Claude")
    console.print("  • Includes tool definitions and execution limits")
    console.print(
        f"  • [{Styles.DIM}]Required for: Using claude_code generator in Python execution[/{Styles.DIM}]"
    )
    console.print()

    # soft-ioc option
    console.print(
        f"[{Styles.HEADER}][→] soft-ioc - Simulated control system for development[/{Styles.HEADER}]"
    )
    console.print()
    console.print("  • Generates caproto-based EPICS soft IOC from channel database")
    console.print("  • Uses channel database from your channel_finder config")
    console.print("  • Auto-detects PV types and access modes from naming conventions")
    console.print("  • Supports mock_style (simulated values) or passthrough backends")
    console.print(
        f"  • Ready to run: just [{Styles.VALUE}]pip install caproto && python <ioc>.py[/{Styles.VALUE}]"
    )
    console.print(
        f"  • [{Styles.DIM}]Perfect for: Testing without real hardware, development mode[/{Styles.DIM}]"
    )
    console.print()

    input("Press ENTER to continue...")


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


def show_generate_menu() -> str | None:
    """Show generate submenu.

    Returns:
        Selected generation type, or None if user cancels/goes back
    """
    if not questionary:
        return None

    console.print(f"\n{Messages.header('Generate Components')}")
    console.print("[dim]Generate capabilities, servers, and configurations[/dim]\n")

    return questionary.select(
        "What would you like to generate?",
        choices=[
            Choice(
                "[→] capability     - From MCP server or natural language",
                value="generate_capability",
            ),
            Choice("[→] mcp-server     - Demo MCP server for testing", value="generate_mcp_server"),
            Choice(
                "[→] claude-config  - Claude Code generator configuration",
                value="generate_claude_config",
            ),
            Choice(
                "[→] soft-ioc       - Simulated control system for development",
                value="generate_soft_ioc",
            ),
            Choice("─" * 60, value=None, disabled=True),
            Choice("[?] help           - Detailed descriptions and usage guide", value="show_help"),
            Choice("[←] back           - Return to main menu", value="back"),
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
    from osprey.generators.config_updater import (
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
        Choice("Mock - Tutorial/Development mode (safe, no hardware)", value="mock"),
        Choice("EPICS - Production mode (connects to real control system)", value="epics"),
        Choice("─" * 60, value=None, disabled=True),
        Choice("[←] Back - Return to config menu", value="back"),
    ]

    control_type = questionary.select(
        "Select control system type:", choices=choices, style=custom_style
    ).ask()

    if control_type is None or control_type == "back":
        return

    # Ask about archiver too
    if control_type == "epics":
        console.print("\n[bold]Archiver Configuration[/bold]\n")
        archiver_type = questionary.select(
            "Also switch archiver to EPICS?",
            choices=[
                Choice("Yes - Use EPICS Archiver Appliance", value="epics_archiver"),
                Choice("No - Keep mock archiver", value="mock_archiver"),
            ],
            style=custom_style,
        ).ask()
    else:
        archiver_type = "mock_archiver"

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

        if control_type == "epics":
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
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result == 0
    except OSError:
        return False


def handle_set_epics_gateway(project_path: Path | None = None) -> None:
    """Handle interactive EPICS gateway configuration."""
    from osprey.generators.config_updater import (
        find_config_file,
        get_control_system_type,
        get_facility_from_gateway_config,
        set_control_system_type,
        set_epics_gateway_config,
    )
    from osprey.templates.data import FACILITY_PRESETS

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
        if current_type in (None, "mock"):
            # None means missing config key, treat same as mock
            if questionary.confirm(
                "\nYour control system is set to 'mock' mode. Switch to 'epics' to use this "
                "gateway?",
                default=True,
                style=custom_style,
            ).ask():
                type_content, _ = set_control_system_type(config_path, "epics")
                config_path.write_text(type_content, encoding="utf-8")
                console.print(f"{Messages.success('✓ Switched to epics mode!')}")
            else:
                console.print(
                    "\n[dim]Note: Gateway configured but mode is still 'mock'. "
                    "Use 'set-control-system' to switch when ready.[/dim]"
                )
        elif current_type == "epics":
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
    from osprey.generators.config_updater import (
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


def handle_generate_action():
    """Handle generate menu and its subcommands."""
    while True:
        action = show_generate_menu()

        if action is None or action == "back":
            return  # Return to main menu

        if action == "generate_capability":
            handle_generate_capability()
        elif action == "generate_mcp_server":
            handle_generate_mcp_server()
        elif action == "generate_claude_config":
            handle_generate_claude_config()
        elif action == "generate_soft_ioc":
            handle_generate_soft_ioc()
        elif action == "show_help":
            show_generate_help()
            # Loop continues - returns to generate menu after help


def handle_generate_capability():
    """Handle interactive capability generation."""
    console.clear()
    console.print(f"\n{Messages.header('Generate Capability')}\n")

    # Ask which mode
    mode = questionary.select(
        "How would you like to generate the capability?",
        choices=[
            Choice("[→] From MCP Server - Production-ready with ReAct agent", value="mcp"),
            Choice("[→] From Prompt - Skeleton with guides (you implement)", value="prompt"),
            Choice("[←] Back", value="back"),
        ],
        style=custom_style,
    ).ask()

    if mode is None or mode == "back":
        return

    try:
        from osprey.cli.generate_cmd import _generate_from_mcp, _generate_from_prompt

        if mode == "mcp":
            # Check if port 3001 is reachable (default MCP server port)
            default_url = "simulated"
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)  # 500ms timeout
                result = sock.connect_ex(("localhost", 3001))
                sock.close()
                if result == 0:
                    # Port 3001 is open, likely our demo MCP server
                    default_url = "http://localhost:3001"
            except Exception:
                # If check fails, just use simulated
                pass

            # Get MCP URL
            mcp_url = questionary.text(
                "MCP server URL (or 'simulated' for demo):", default=default_url, style=custom_style
            ).ask()

            if not mcp_url:
                return

            # Get capability name
            capability_name = questionary.text(
                "Capability name:", default="weather_mcp", style=custom_style
            ).ask()

            if not capability_name:
                return

            # Generate
            _generate_from_mcp(
                mcp_url=mcp_url,
                capability_name=capability_name,
                server_name=None,
                output_file=None,
                provider=None,
                model_id=None,
                quiet=False,
            )

        elif mode == "prompt":
            # Get prompt
            prompt = questionary.text(
                "Describe what the capability should do:", multiline=True, style=custom_style
            ).ask()

            if not prompt:
                return

            # Get capability name (optional)
            capability_name = questionary.text(
                "Capability name (leave empty for LLM suggestion):", default="", style=custom_style
            ).ask()

            # Generate
            _generate_from_prompt(
                prompt=prompt,
                capability_name=capability_name if capability_name else None,
                output_file=None,
                provider=None,
                model_id=None,
                quiet=False,
            )

        console.print()
        input("Press ENTER to continue...")

    except RuntimeError as e:
        # RuntimeError with clear message - show it directly
        error_msg = str(e)
        if error_msg.startswith("\n"):
            console.print(error_msg)
        else:
            console.print(f"\n{Messages.error('Generation failed: ')}")
            console.print(f"\n{error_msg}")
        console.print()
        input("Press ENTER to continue...")
    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        console.print()
        input("Press ENTER to continue...")


def handle_generate_mcp_server():
    """Handle interactive MCP server generation."""
    console.clear()
    console.print(f"\n{Messages.header('Generate MCP Server')}\n")
    console.print("[dim]Creates a demo weather MCP server for testing[/dim]\n")

    # Get server name
    server_name = questionary.text("Server name:", default="demo_mcp", style=custom_style).ask()

    if not server_name:
        return

    # Get port
    port_str = questionary.text("Port:", default="3001", style=custom_style).ask()

    if not port_str:
        return

    try:
        port = int(port_str)
    except ValueError:
        console.print(f"\n{Messages.error('Invalid port number')}")
        input("Press ENTER to continue...")
        return

    try:
        from pathlib import Path

        from osprey.cli.generate_cmd import get_server_template

        output_file = f"{server_name}_server.py"
        output_path = Path(output_file)

        write_mcp_server_file = get_server_template()
        output_path = write_mcp_server_file(
            output_path=output_path, server_name=server_name, port=port
        )

        console.print(f"\n{Messages.success(f'Server generated: {output_path}')}\n")

        # Check if fastmcp is installed
        fastmcp_installed = False
        try:
            import importlib.util

            if importlib.util.find_spec("fastmcp") is not None:
                fastmcp_installed = True
        except Exception:
            pass

        # Always show the instructions upfront
        console.print(f"[{Styles.HEADER}]Usage Instructions:[/{Styles.HEADER}]")
        if not fastmcp_installed:
            console.print(
                f"  1. Install dependencies: [{Styles.ACCENT}]pip install fastmcp[/{Styles.ACCENT}]"
            )
            console.print(
                f"  2. Run the server: [{Styles.ACCENT}]python {output_path}[/{Styles.ACCENT}]"
            )
            console.print(
                f"  3. Generate capability: [{Styles.ACCENT}]osprey generate capability --from-mcp http://localhost:{port} -n {server_name}[/{Styles.ACCENT}]"
            )
        else:
            console.print(f"  • Server file: [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]")
            console.print(
                f"  • Server URL: [{Styles.VALUE}]http://localhost:{port}[/{Styles.VALUE}]"
            )
            console.print(
                f"  • Manual start: [{Styles.ACCENT}]python {output_path}[/{Styles.ACCENT}]"
            )
            console.print(
                f"  • Create capability: [{Styles.ACCENT}]osprey generate capability --from-mcp http://localhost:{port} -n {server_name}[/{Styles.ACCENT}]"
            )
        console.print()

        # Offer to launch the server immediately with three options
        if fastmcp_installed:
            choices = [
                Choice("[▶] Launch in background - Detached, returns to menu", value="background"),
                Choice(
                    "[▶] Launch in this terminal - Interactive, see output live", value="foreground"
                ),
                Choice("[←] Back to menu - Launch manually later", value="back"),
            ]
        else:
            choices = [
                Choice(
                    "[!] Cannot launch - fastmcp not installed (pip install fastmcp)",
                    value="install",
                    disabled=True,
                ),
                Choice(
                    "[←] Back to menu - Install fastmcp and launch manually later", value="back"
                ),
            ]

        next_action = questionary.select(
            "How would you like to launch the server?",
            choices=choices,
            style=custom_style,
        ).ask()

        if next_action == "background" and fastmcp_installed:
            # Launch the server in the background (detached)
            console.print("\n[dim]Starting MCP server in background...[/dim]")

            import subprocess

            try:
                # Start the server process in the background
                process = subprocess.Popen(
                    [sys.executable, str(output_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,  # Detach from current session
                )

                # Give it a moment to start
                import time

                time.sleep(1)

                # Check if it's still running
                if process.poll() is None:
                    console.print(f"\n{Messages.success('✓ Server started in background!')}")
                    console.print(f"[dim]  PID: {process.pid}[/dim]")
                    console.print(f"[dim]  URL: http://localhost:{port}[/dim]")
                    console.print(f"\n[{Styles.ACCENT}]To stop the server:[/{Styles.ACCENT}]")
                    console.print(f"  kill {process.pid}")
                    console.print(
                        f"\n[{Styles.ACCENT}]Next step - Generate a capability:[/{Styles.ACCENT}]"
                    )
                    console.print(
                        f"  osprey generate capability --from-mcp http://localhost:{port} -n {server_name}"
                    )
                else:
                    # Process died immediately, show error
                    _, stderr = process.communicate(timeout=1)
                    console.print(f"\n{Messages.error('Server failed to start')}")
                    if stderr:
                        console.print(f"[dim]{stderr.decode()[:500]}[/dim]")
            except Exception as e:
                console.print(f"\n{Messages.error(f'Failed to launch server: {e}')}")

            console.print()
            input("Press ENTER to continue...")

        elif next_action == "foreground" and fastmcp_installed:
            # Launch the server in the foreground (this terminal)
            console.print(f"\n{Messages.success('Starting MCP server in this terminal...')}")
            console.print(f"[dim]Server URL: http://localhost:{port}[/dim]")
            console.print("[dim]Press Ctrl+C to stop the server and return to menu[/dim]\n")

            import subprocess

            try:
                # Start the server in foreground (user can see output)
                process = subprocess.run([sys.executable, str(output_path)], cwd=Path.cwd())
            except KeyboardInterrupt:
                console.print(f"\n\n{Messages.warning('Server stopped by user')}")
            except Exception as e:
                console.print(f"\n{Messages.error(f'Server error: {e}')}")

            console.print()
            input("Press ENTER to continue...")
        # If 'back', just continue to menu

    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        console.print()
        input("Press ENTER to continue...")


def handle_generate_claude_config():
    """Handle interactive Claude config generation."""
    console.clear()
    console.print(f"\n{Messages.header('Generate Claude Code Configuration')}\n")
    console.print("[dim]Creates claude_generator_config.yml with sensible defaults[/dim]\n")

    # Check if file exists
    from pathlib import Path

    output_path = Path("claude_generator_config.yml")

    if output_path.exists():
        overwrite = questionary.confirm(
            f"{output_path} already exists. Overwrite?", default=False, style=custom_style
        ).ask()

        if not overwrite:
            return

    try:
        import yaml

        from osprey.cli.templates import TemplateManager

        # Try to detect provider from config.yml if available
        default_provider = "anthropic"
        config_path = Path.cwd() / "config.yml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if config and "llm" in config:
                        default_provider = config["llm"].get("default_provider", "anthropic")
                        console.print(f"[dim]Detected provider: {default_provider}[/dim]\n")
            except Exception:
                pass

        # Create template context
        ctx = {"default_provider": default_provider, "default_model": "claude-haiku-4-5-20251001"}

        # Render template
        template_manager = TemplateManager()
        template_manager.render_template(
            "apps/control_assistant/claude_generator_config.yml.j2", ctx, output_path
        )

        console.print(f"\n{Messages.success(f'Configuration generated: {output_path}')}\n")

        # Show next steps
        console.print(f"[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
        console.print(f"  1. Review: [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]")
        console.print()
        console.print("  2. Enable in [accent]config.yml[/accent]:")
        console.print()
        console.print("     [dim]execution:[/dim]")
        console.print('     [dim]  code_generator: "claude_code"[/dim]')
        console.print("     [dim]  generators:[/dim]")
        console.print("     [dim]    claude_code:[/dim]")
        console.print('     [dim]      profile: "fast"  # or "robust"[/dim]')
        console.print(f'     [dim]      claude_config_path: "{output_path.name}"[/dim]')
        console.print()
        console.print("  3. Set API key in [accent].env[/accent]:")
        if default_provider == "cborg":
            console.print("     [dim]CBORG_API_KEY=your-key-here[/dim]")
        else:
            console.print("     [dim]ANTHROPIC_API_KEY=your-key-here[/dim]")
        console.print()
        input("Press ENTER to continue...")

    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print()
        input("Press ENTER to continue...")


def handle_generate_soft_ioc():
    """Handle interactive soft IOC generation."""
    console.clear()
    console.print(f"\n{Messages.header('Generate Soft IOC')}\n")
    console.print("[dim]Creates a simulated control system from your channel database[/dim]\n")

    from pathlib import Path

    import click

    from osprey.cli.generate_cmd import soft_ioc

    # Check for config.yml
    config_path = Path.cwd() / "config.yml"
    if not config_path.exists():
        console.print(f"\n{Messages.error('No config.yml found in current directory')}")
        console.print()
        console.print("  Run [accent]osprey init[/accent] to create a project first.")
        console.print()
        input("Press ENTER to continue...")
        return

    # Ask about options
    use_init = questionary.confirm(
        "Run interactive setup wizard?",
        default=True,
        style=custom_style,
    ).ask()

    if use_init is None:
        return

    dry_run = questionary.confirm(
        "Dry-run mode (preview without writing files)?",
        default=False,
        style=custom_style,
    ).ask()

    if dry_run is None:
        return

    try:
        # Build CLI args
        args = []
        if use_init:
            args.append("--init")
        if dry_run:
            args.append("--dry-run")

        # Invoke the command
        ctx = click.Context(soft_ioc)
        ctx.invoke(soft_ioc, config_path=None, output_file=None, dry_run=dry_run, init=use_init)

    except click.Abort:
        console.print(f"\n{Messages.info('Generation cancelled by user.')}")
    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    console.print()
    input("Press ENTER to continue...")


# ============================================================================
# NAVIGATION LOOP
# ============================================================================


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
        elif action == "chat":
            handle_chat_action()
        elif action == "chat-tui":
            handle_chat_tui_action()
        elif action == "deploy":
            handle_deploy_action()
        elif action == "health":
            handle_health_action()
        elif action == "generate":
            handle_generate_action()
        elif action == "config":
            handle_config_action()
        elif action == "registry":
            from osprey.cli.registry_cmd import handle_registry_action

            handle_registry_action()
        elif action == "tasks":
            handle_tasks_action()
        elif action == "help":
            # Show contextual help based on whether we're in a project or not
            if is_project_initialized():
                handle_help_action()
            else:
                handle_help_action_root()


# ============================================================================
# ENTRY POINT
# ============================================================================


def launch_tui():
    """Entry point for TUI mode."""
    # Check dependencies
    if not questionary:
        console.print(Messages.error("Missing required dependency 'questionary'"))
        console.print("\nInstall with:")
        console.print(f"  {Messages.command('pip install questionary')}")
        console.print("\nOr install full osprey dependencies:")
        console.print(f"  {Messages.command('pip install -e .[all]')}\n")
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
