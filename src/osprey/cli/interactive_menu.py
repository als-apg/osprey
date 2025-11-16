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
from rich.markdown import Markdown
from rich.panel import Panel

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

    console.print()

    # Try to load custom banner if in a project directory
    banner_text = None

    try:
        # Check if config exists before trying to load
        # Suppress config loading messages in interactive menu
        with quiet_logger(['REGISTRY', 'CONFIG']):
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

    # Context-specific subtitle
    if context == "interactive":
        console.print(f"    [{Styles.HEADER}]Interactive Menu System[/{Styles.HEADER}]")
        console.print(f"    [{Styles.DIM}]Use arrow keys to navigate • Press Ctrl+C to exit[/{Styles.DIM}]")
    elif context == "chat":
        msg = Messages.info("💡 Type 'bye' or 'end' to exit")
        console.print(f"    {msg}")
        console.print(f"    [{Styles.ACCENT}]⚡ Use slash commands (/) for quick actions - try /help[/{Styles.ACCENT}]")

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
    return (Path.cwd() / 'config.yml').exists()


def get_project_info(config_path: Path | None = None) -> dict[str, Any]:
    """Load and parse config.yml for project metadata.

    Args:
        config_path: Optional path to config.yml (defaults to current directory)

    Returns:
        Dictionary with project information (provider, model, etc.)
        Returns empty dict if no project found or error parsing
    """
    if config_path is None:
        config_path = Path.cwd() / 'config.yml'

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate config.yml structure after parsing
        if config is None:
            if os.environ.get('DEBUG'):
                console.print(f"[dim]Warning: Empty config.yml at {config_path}[/dim]")
            return {}

        if not isinstance(config, dict):
            if os.environ.get('DEBUG'):
                console.print(f"[dim]Warning: Invalid config.yml structure (not a dict) at {config_path}[/dim]")
            return {}

        # Extract relevant information with safe defaults
        info = {
            'project_root': config.get('project_root', str(config_path.parent)),
            'registry_path': config.get('registry_path', ''),
        }

        # Extract provider and model from models.orchestrator section
        models_config = config.get('models', {})
        if not isinstance(models_config, dict):
            models_config = {}

        orchestrator = models_config.get('orchestrator', {})
        if not isinstance(orchestrator, dict):
            orchestrator = {}

        if orchestrator:
            info['provider'] = orchestrator.get('provider', 'unknown')
            info['model'] = orchestrator.get('model_id', 'unknown')

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
        'node_modules', 'venv', '.venv', 'env', '.env',
        '__pycache__', '.git', '.svn', '.hg',
        'build', 'dist', '.egg-info', 'site-packages',
        '.pytest_cache', '.mypy_cache', '.tox',
        'docs', '_agent_data', '.cache'
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
                    if os.environ.get('DEBUG'):
                        console.print(f"[dim]Project discovery timeout after {max_time_ms}ms[/dim]")
                    break

                # Only check directories
                if not item.is_dir():
                    continue

                # Skip hidden directories (start with .)
                if item.name.startswith('.'):
                    continue

                # Skip common non-project directories
                if item.name in ignore_dirs:
                    continue

                subdirs.append(item)

        except (PermissionError, OSError) as e:
            # Skip directories we can't read
            if os.environ.get('DEBUG'):
                console.print(f"[dim]Warning: Could not read directory: {e}[/dim]")
            return projects

        # Sort subdirectories alphabetically for consistent ordering
        subdirs.sort(key=lambda p: p.name.lower())

        # Check each subdirectory for config.yml
        for subdir in subdirs:
            # Check limits
            if checked_count >= max_dirs:
                if os.environ.get('DEBUG'):
                    console.print(f"[dim]Project discovery stopped after checking {max_dirs} directories[/dim]")
                break

            if time.time() - start_time > max_time_seconds:
                if os.environ.get('DEBUG'):
                    console.print(f"[dim]Project discovery timeout after {max_time_ms}ms[/dim]")
                break

            try:
                config_file = subdir / 'config.yml'

                if config_file.exists() and config_file.is_file():
                    # Found a project!
                    projects.append((subdir.name, subdir))

            except (PermissionError, OSError):
                # Skip directories we can't access
                pass

            checked_count += 1

    except Exception as e:
        # Fail gracefully - return whatever we found so far
        if os.environ.get('DEBUG'):
            console.print(f"[dim]Warning during project discovery: {e}[/dim]")

    # Return sorted list
    return sorted(projects, key=lambda x: x[0].lower())


# ============================================================================
# PROVIDER METADATA (from Registry)
# ============================================================================

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
                    'name': provider_class.name,
                    'description': provider_class.description,
                    'requires_key': provider_class.requires_api_key,
                    'requires_base_url': provider_class.requires_base_url,
                    'models': provider_class.available_models,
                    'default_model': provider_class.default_model_id,
                    'health_check_model': provider_class.health_check_model_id,
                    'api_key_url': provider_class.api_key_url,
                    'api_key_instructions': provider_class.api_key_instructions,
                    'api_key_note': provider_class.api_key_note
                }
            except Exception as e:
                # Skip providers that fail to load, but log for debugging
                if os.environ.get('DEBUG'):
                    console.print(f"[dim]Warning: Could not load provider {provider_reg.class_name}: {e}[/dim]")
                continue

        if not providers:
            console.print(Messages.warning("No providers could be loaded from osprey registry"))

        # Cache the result for future calls
        _provider_cache = providers
        return providers

    except Exception as e:
        # This should rarely happen - osprey registry should always be available
        console.print(Messages.error(f"Could not load providers from osprey registry: {e}"))
        console.print(Messages.warning("The TUI requires access to provider information to initialize projects."))
        if os.environ.get('DEBUG'):
            import traceback
            traceback.print_exc()

        # Return empty dict but don't cache failures
        return {}


# ============================================================================
# MAIN MENU
# ============================================================================

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
                project_info = get_project_info(project_path / 'config.yml')

                if project_info and 'provider' in project_info:
                    display = f"[→] {project_name:20} ({project_info['provider']} / {project_info.get('model', 'unknown')[:20]})"
                else:
                    display = f"[→] {project_name:20} (osprey project)"

                # Value is tuple so we can distinguish from other actions
                choices.append(Choice(display, value=('select_project', project_path)))

            # Add separator
            choices.append(Choice("─" * 60, value=None, disabled=True))

        # Standard menu options
        choices.extend([
            Choice("[+] Create new project (interactive)", value='init_interactive'),
            Choice("[?] Help", value='help'),
            Choice("[x] Exit", value='exit')
        ])

        return questionary.select(
            "What would you like to do?",
            choices=choices,
            style=custom_style,
            instruction="(Use arrow keys to navigate)"
        ).ask()
    else:
        # Project menu
        project_info = get_project_info()
        project_name = Path.cwd().name

        console.print(f"\n{Messages.header('Project:')} {project_name}")
        if project_info:
            console.print(f"[dim]Provider: {project_info.get('provider', 'unknown')} | "
                         f"Model: {project_info.get('model', 'unknown')}[/dim]")

        return questionary.select(
            "Select command:",
            choices=[
                Choice("[>] chat        - Start CLI conversation", value='chat'),
                Choice("[>] deploy      - Manage services (web UIs)", value='deploy'),
                Choice("[>] health      - Run system health check", value='health'),
                Choice("[>] config      - Show configuration", value='config'),
                Choice("[>] registry    - Show registry contents", value='registry'),
                Choice("[+] init        - Create new project", value='init_interactive'),
                Choice("[?] help        - Show all commands", value='help'),
                Choice("[x] exit        - Exit CLI", value='exit')
            ],
            style=custom_style,
            instruction="(Use arrow keys to navigate)"
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
            [runtime, "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            containers = result.stdout.strip().split('\n')
            containers = [c for c in containers if c]  # Remove empty strings

            for container in containers:
                # Inspect each container for mounts
                inspect_result = subprocess.run(
                    [runtime, "inspect", "--format", "{{json .Mounts}}", container],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if inspect_result.returncode == 0:
                    try:
                        mounts = json.loads(inspect_result.stdout)
                        for mount in mounts:
                            source = mount.get('Source', '')
                            if dir_str in source or source.startswith(dir_str):
                                mount_details.append(
                                    f"Container '{container}' has mount: {source}"
                                )
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
        'minimal': 'Empty project structure with TODO placeholders',
        'hello_world_weather': 'Single capability weather example (tutorial)',
        'control_assistant': 'Control system integration with channel finder (production-grade)'
    }

    choices = []
    for template in templates:
        desc = descriptions.get(template, 'No description available')
        display = f"{template:22} - {desc}"
        choices.append(Choice(display, value=template))

    return questionary.select(
        "Select project template:",
        choices=choices,
        style=custom_style,
        instruction="(Use arrow keys to navigate)"
    ).ask()


def get_default_name_for_template(template: str) -> str:
    """Get a sensible default project name for the template.

    Args:
        template: Template name

    Returns:
        Default project name suggestion
    """
    defaults = {
        'minimal': 'my-agent',
        'hello_world_weather': 'weather-agent',
        'control_assistant': 'my-control-assistant'
    }
    return defaults.get(template, 'my-project')


def select_channel_finder_mode() -> str | None:
    """Interactive channel finder mode selection for control_assistant template.

    Returns:
        Selected mode ('in_context', 'hierarchical', 'both'), or None if cancelled
    """
    console.print("[dim]Select the channel finding approach for your control system:[/dim]\n")

    choices = [
        Choice(
            "in_context       - Semantic search (best for few hundred channels, faster)",
            value='in_context'
        ),
        Choice(
            "hierarchical     - Structured navigation (best for >1,000 channels, scalable)",
            value='hierarchical'
        ),
        Choice(
            "both             - Include both pipelines (maximum flexibility, comparison)",
            value='both'
        ),
    ]

    return questionary.select(
        "Channel finder mode:",
        choices=choices,
        style=custom_style,
        instruction="(Use arrow keys to navigate)"
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
        console.print(f"[dim]Check that osprey is properly installed: {Messages.command('pip install -e .[all]')}[/dim]\n")
        return None

    choices = []
    for key, p in sorted(providers.items()):
        try:
            # Validate provider metadata structure
            if not isinstance(p, dict):
                continue
            if 'name' not in p or 'description' not in p:
                if os.environ.get('DEBUG'):
                    console.print(f"[dim]Warning: Provider {key} missing required metadata[/dim]")
                continue

            # Description comes directly from provider class attribute
            key_info = " [requires API key]" if p.get('requires_key', True) else " [no API key]"
            display = f"{p['name']:12} - {p['description']}{key_info}"
            choices.append(Choice(display, value=key))
        except Exception as e:
            if os.environ.get('DEBUG'):
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
        instruction="(This sets default provider in config.yml)"
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

    choices = [
        Choice(model, value=model)
        for model in provider_info['models']
    ]

    default = provider_info.get('default_model')

    return questionary.select(
        f"Select default model for {provider}:",
        choices=choices,
        style=custom_style,
        default=default if default in provider_info['models'] else None
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
        'cborg': 'CBORG_API_KEY',
        'stanford': 'STANFORD_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'ollama': None  # Ollama doesn't need API key
    }
    return key_names.get(provider, f'{provider.upper()}_API_KEY')


def configure_api_key(provider: str, project_path: Path,
                     providers: dict[str, dict[str, Any]]) -> bool:
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
            Choice("[#] Paste API key now (secure input)", value='paste'),
            Choice("[-] Configure later (edit .env manually)", value='later'),
            Choice("[?] Where do I get an API key?", value='help'),
        ],
        style=custom_style,
    ).ask()

    if action == 'help':
        show_api_key_help(provider)
        return configure_api_key(provider, project_path, providers)  # Ask again

    elif action == 'paste':
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

    elif action == 'later':
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
        provider_display = provider_data.get('description') or provider.title()
        console.print(f"[bold]Getting a {provider_display} API Key:[/bold]")

        # Show URL if available
        api_key_url = provider_data.get('api_key_url')
        if api_key_url:
            console.print(f"  1. Visit: {api_key_url}")
            step_offset = 2
        else:
            step_offset = 1

        # Show instructions
        api_key_instructions = provider_data.get('api_key_instructions', [])
        if api_key_instructions:
            for i, instruction in enumerate(api_key_instructions, start=step_offset):
                console.print(f"  {i}. {instruction}")
            console.print()  # Extra line after instructions

        # Show note if available
        api_key_note = provider_data.get('api_key_note')
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
        with console.status("[dim]Loading templates and providers...[/dim]", spinner="dots"):
            templates = manager.list_app_templates()
            providers = get_provider_metadata()
    except Exception as e:
        console.print(f"[error]✗ Error loading templates/providers:[/error] {e}")
        input("\nPress ENTER to continue...")
        return 'menu'

    # 1. Template selection
    console.print("[bold]Step 1: Select Template[/bold]\n")
    template = select_template(templates)
    if template is None:
        return 'menu'

    # 2. Project name
    console.print("\n[bold]Step 2: Project Name[/bold]\n")
    project_name = questionary.text(
        "Project name:",
        default=get_default_name_for_template(template),
        style=custom_style,
    ).ask()

    if not project_name:
        return 'menu'

    # 2b. Channel finder mode (only for control_assistant template)
    channel_finder_mode = None
    if template == 'control_assistant':
        console.print("\n[bold]Step 3: Channel Finder Configuration[/bold]\n")
        channel_finder_mode = select_channel_finder_mode()
        if channel_finder_mode is None:
            return 'menu'

    # Check if project directory already exists (before other configuration steps)
    project_path = Path.cwd() / project_name
    if project_path.exists():
        msg = Messages.warning(f"Directory '{project_path}' already exists.")
        console.print(f"\n{msg}\n")

        # Check if directory exists immediately before deletion (safety check) and check for active Docker/Podman mounts before allowing deletion
        has_mounts, mount_details = check_directory_has_active_mounts(project_path)

        if has_mounts:
            console.print(f"{Messages.error('⚠️  DANGER: This directory has active container mounts!')}")
            console.print(f"{Messages.warning('The following containers are using this directory:')}\n")
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
                console.print(f"[dim]Tip: Stop containers first with {Messages.command('osprey deploy down')}[/dim]")
                input("\nPress ENTER to continue...")
                return 'menu'

        action = questionary.select(
            "What would you like to do?",
            choices=[
                Choice("[!] Override - Delete existing directory and create new project", value='override'),
                Choice("[*] Rename - Choose a different project name", value='rename'),
                Choice("[-] Abort - Return to main menu", value='abort'),
            ],
            style=custom_style,
        ).ask()

        if action == 'abort' or action is None:
            console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
            input("\nPress ENTER to continue...")
            return 'menu'
        elif action == 'rename':
            # Go back to project name input
            console.print("\n[bold]Choose a different project name:[/bold]\n")
            new_project_name = questionary.text(
                "Project name:",
                default=f"{project_name}-2",
                style=custom_style,
            ).ask()

            if not new_project_name:
                return 'menu'

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
                    return 'menu'

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
                        console.print(Messages.warning("Try running with appropriate permissions or stop any running processes"))
                        input("\nPress ENTER to continue...")
                        return 'menu'
                    except OSError as e:
                        console.print(f"\n{Messages.error(f'Could not delete directory: {e}')}")
                        input("\nPress ENTER to continue...")
                        return 'menu'
        elif action == 'override':
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
                    console.print(Messages.warning("Try running with appropriate permissions or stop any running processes"))
                    input("\nPress ENTER to continue...")
                    return 'menu'
                except OSError as e:
                    console.print(f"\n{Messages.error(f'Could not delete directory: {e}')}")
                    input("\nPress ENTER to continue...")
                    return 'menu'

    # 3. Registry style (step number adjusts if channel finder mode was shown)
    step_num = 4 if template == 'control_assistant' else 3
    console.print(f"\n[bold]Step {step_num}: Registry Style[/bold]\n")

    registry_style = questionary.select(
        "Select registry style:",
        choices=[
            Choice("extend     - Extends framework defaults (recommended)", value='extend'),
            Choice("standalone - Complete explicit registry (advanced)", value='standalone'),
        ],
        style=custom_style,
        instruction="(extend mode is recommended for most projects)"
    ).ask()

    if registry_style is None:
        return 'menu'

    # 4. Provider selection (step number adjusts)
    step_num = 5 if template == 'control_assistant' else 4
    console.print(f"\n[bold]Step {step_num}: AI Provider[/bold]\n")
    provider = select_provider(providers)
    if provider is None:
        return 'menu'

    # 5. Model selection (step number adjusts)
    step_num = 6 if template == 'control_assistant' else 5
    console.print(f"\n[bold]Step {step_num}: Model Selection[/bold]\n")
    model = select_model(provider, providers)
    if model is None:
        return 'menu'

    # Summary
    console.print(f"\n{Messages.header('Configuration Summary:')}")
    console.print(f"  Project:  [value]{project_name}[/value]")
    console.print(f"  Template: [value]{template}[/value]")
    if channel_finder_mode:
        console.print(f"  Pipeline: [value]{channel_finder_mode}[/value]")
    console.print(f"  Registry: [value]{registry_style}[/value]")
    console.print(f"  Provider: [value]{provider}[/value]")
    console.print(f"  Model:    [value]{model}[/value]\n")

    # Confirm
    proceed = questionary.confirm(
        "Create project with these settings?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        console.print(f"\n{Messages.warning('✗ Project creation cancelled')}")
        input("\nPress ENTER to continue...")
        return 'menu'

    # Create project
    console.print("\n[bold]Creating project...[/bold]\n")

    try:
        # Note: force=True because we already handled directory deletion if user chose override
        # Build context dict with optional channel_finder_mode
        context = {
            'default_provider': provider,
            'default_model': model
        }
        if channel_finder_mode:
            context['channel_finder_mode'] = channel_finder_mode

        project_path = manager.create_project(
            project_name=project_name,
            output_dir=Path.cwd(),
            template_name=template,
            registry_style=registry_style,
            context=context,
            force=True
        )

        msg = Messages.success('Project created at:')
        path = Messages.path(str(project_path))
        console.print(f"\n{msg} {path}\n")

        # Check if API keys were detected and .env was created
        detected_env = manager._detect_environment_variables()
        api_keys = ['CBORG_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
        has_api_keys = any(key in detected_env for key in api_keys)

        if has_api_keys:
            env_file = project_path / ".env"
            if env_file.exists():
                console.print(Messages.success("Created .env with detected API keys"))
                detected_keys = [key for key in api_keys if key in detected_env]
                console.print(f"[dim]  Detected: {', '.join(detected_keys)}[/dim]\n")

        # API key configuration
        if providers[provider]['requires_key']:
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
                    Choice("[>] Start chat in this project now", value='chat'),
                    Choice("[<] Return to main menu", value='menu'),
                    Choice("[x] Exit and show next steps", value='exit'),
                ],
                style=custom_style,
            ).ask()

            if next_action == 'chat':
                console.print(f"\n[dim]Launching chat for project: {project_path.name}[/dim]\n")
                handle_chat_action(project_path=project_path)
                return 'menu'
            elif next_action == 'exit':
                # Show next steps like the direct init command
                console.print("\n[bold]Next steps:[/bold]")
                console.print(f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}")
                console.print("  2. # .env already configured with API key")
                console.print(f"  3. Start chatting: {Messages.command('osprey chat')}")
                console.print(f"  4. Start services: {Messages.command('osprey deploy up')}")
                console.print()
                return 'exit'
        else:
            console.print("[bold]Next steps:[/bold]")
            console.print(f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}")
            console.print(f"  2. Configure API key: {Messages.command('cp .env.example .env')} (then edit)")
            console.print(f"  3. Start chatting: {Messages.command('osprey chat')}")
            console.print(f"  4. Start services: {Messages.command('osprey deploy up')}")

            console.print("\n[dim]Press ENTER to continue...[/dim]")
            input()

        return 'menu'

    except ValueError as e:
        # This should not happen anymore since we check directory existence above
        # But catch it just in case
        console.print(f"\n[error]✗ Error creating project:[/error] {e}")
        input("\nPress ENTER to continue...")
        return 'menu'
    except Exception as e:
        console.print(f"\n[error]✗ Unexpected error creating project:[/error] {e}")
        if os.environ.get('DEBUG'):
            import traceback
            traceback.print_exc()
        input("\nPress ENTER to continue...")
        return 'menu'


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
    project_info = get_project_info(project_path / 'config.yml')

    # Loop to keep showing project menu after actions complete
    while True:
        console.clear()
        show_banner(context="interactive")

        console.print(f"\n{Messages.header('Selected Project:')} {project_name}")
        console.print(f"[dim]Location: {Messages.path(str(project_path))}[/dim]")

        if project_info:
            console.print(f"[dim]Provider: {project_info.get('provider', 'unknown')} | "
                         f"Model: {project_info.get('model', 'unknown')}[/dim]\n")

        # Offer actions for the selected project
        action = questionary.select(
            "What would you like to do with this project?",
            choices=[
                Choice("[>] chat        - Start CLI conversation", value='chat'),
                Choice("[>] deploy      - Manage services (web UIs)", value='deploy'),
                Choice("[>] health      - Run system health check", value='health'),
                Choice("[>] config      - Show configuration", value='config'),
                Choice("[>] registry    - Show registry contents", value='registry'),
                Choice("[<] back        - Return to main menu", value='back'),
            ],
            style=custom_style,
            instruction="(Use arrow keys to navigate)"
        ).ask()

        if action == 'back' or action is None:
            return  # Exit the loop and return to main menu

        # Execute the selected action with the project path
        if action == 'chat':
            handle_chat_action(project_path=project_path)
        elif action == 'deploy':
            handle_deploy_action(project_path=project_path)
        elif action == 'health':
            handle_health_action(project_path=project_path)
        elif action == 'config':
            handle_export_action(project_path=project_path)
        elif action == 'registry':
            from osprey.cli.registry_cmd import handle_registry_action
            handle_registry_action(project_path=project_path)

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
            console.print(f"{Messages.warning('This appears to be a pydantic/Python version compatibility issue.')}\n")
            console.print("[bold]Possible solutions:[/bold]")
            console.print("  1. Upgrade typing_extensions:")
            console.print(f"     {Messages.command('pip install --upgrade typing-extensions')}\n")
            console.print("  2. Upgrade pydantic:")
            console.print(f"     {Messages.command('pip install --upgrade pydantic pydantic-core')}\n")
            console.print("  3. Check pydantic-ai compatibility:")
            console.print(f"     {Messages.command('pip install --upgrade pydantic-ai')}\n")
            console.print("  4. Or upgrade to Python 3.12+\n")
        else:
            console.print(Messages.warning("There was an error loading the chat dependencies."))
            console.print(f"[dim]Try reinstalling osprey dependencies: {Messages.command('pip install -e .[all]')}[/dim]\n")

        if os.environ.get('DEBUG'):
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
            # 1. Set CONFIG_FILE env var so config loading works
            # 2. Change to project directory so relative paths work
            config_path = str(project_path / "config.yml")

            # Save original state
            original_dir = Path.cwd()
            original_config_env = os.environ.get('CONFIG_FILE')

            try:
                # Set up environment for the project
                os.environ['CONFIG_FILE'] = config_path

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
                    console.print(f"\n{Messages.warning(f'Could not restore original directory: {e}')}")
                    console.print(f"[dim]Current directory may have changed. Original was: {original_dir}[/dim]")

                if original_config_env is not None:
                    os.environ['CONFIG_FILE'] = original_config_env
                elif 'CONFIG_FILE' in os.environ:
                    del os.environ['CONFIG_FILE']
        else:
            # Default behavior - run in current directory
            # Set CONFIG_FILE for subprocess execution (critical for Python executor)
            config_path = str(Path.cwd() / "config.yml")
            os.environ['CONFIG_FILE'] = config_path
            asyncio.run(run_cli(config_path=config_path))

    except KeyboardInterrupt:
        console.print(f"\n\n{Messages.warning('Chat session ended.')}")
        # No pause needed - user intentionally exited with Ctrl+C
    except Exception as e:
        console.print(f"\n{Messages.error(f'Chat error: {e}')}")
        if os.environ.get('DEBUG'):
            import traceback
            traceback.print_exc()
        input("\nPress ENTER to continue...")

    # Return to menu (with pause only for actual errors)


def handle_deploy_action(project_path: Path | None = None):
    """Manage deployment services menu.

    Args:
        project_path: Optional project directory path (defaults to current directory)
    """
    action = questionary.select(
        "Select deployment action:",
        choices=[
            Choice("[^] up      - Start all services", value='up'),
            Choice("[v] down    - Stop all services", value='down'),
            Choice("[i] status  - Show service status", value='status'),
            Choice("[*] restart - Restart all services", value='restart'),
            Choice("[+] build   - Build/prepare compose files only", value='build'),
            Choice("[R] rebuild - Clean, rebuild, and restart services", value='rebuild'),
            Choice("[X] clean   - Remove containers and volumes (WARNING: destructive)", value='clean'),
            Choice("[<] back    - Back to main menu", value='back'),
        ],
        style=custom_style,
    ).ask()

    if action == 'back' or action is None:
        return

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
            return
    else:
        config_path = "config.yml"
        original_dir = None

    try:
        # Confirm destructive operations
        if action == 'clean':
            console.print("\n[bold red]⚠️  WARNING: Destructive Operation[/bold red]")
            console.print("\n[warning]This will permanently delete:[/warning]")
            console.print("  • All containers for this project")
            console.print("  • All volumes (including databases and stored data)")
            console.print("  • All networks created by compose")
            console.print("  • Container images built for this project")
            console.print("\n[dim]This action cannot be undone![/dim]\n")

            confirm = questionary.confirm(
                "Are you sure you want to proceed?",
                default=False,
                style=custom_style
            ).ask()

            if not confirm:
                console.print(f"\n{Messages.warning('Operation cancelled')}")
                input("\nPress ENTER to continue...")
                if original_dir:
                    try:
                        os.chdir(original_dir)
                    except (OSError, PermissionError):
                        pass
                return

        elif action == 'rebuild':
            console.print("\n[bold yellow]⚠️  Rebuild Operation[/bold yellow]")
            console.print("\n[warning]This will:[/warning]")
            console.print("  • Stop and remove all containers")
            console.print("  • Delete all volumes (data will be lost)")
            console.print("  • Remove container images")
            console.print("  • Rebuild everything from scratch")
            console.print("  • Start services again")
            console.print("\n[dim]Any data stored in volumes will be lost![/dim]\n")

            confirm = questionary.confirm(
                "Proceed with rebuild?",
                default=False,
                style=custom_style
            ).ask()

            if not confirm:
                console.print(f"\n{Messages.warning('Rebuild cancelled')}")
                input("\nPress ENTER to continue...")
                if original_dir:
                    try:
                        os.chdir(original_dir)
                    except (OSError, PermissionError):
                        pass
                return

        # Build the osprey deploy command
        # Use 'osprey' command directly to avoid module import warnings
        cmd = ["osprey", "deploy", action]

        if action in ['up', 'restart', 'rebuild']:
            cmd.append("-d")  # Run in detached mode

        cmd.extend(["--config", config_path])

        if action == 'up':
            console.print("\n[bold]Starting services...[/bold]")
        elif action == 'down':
            console.print("\n[bold]Stopping services...[/bold]")
        elif action == 'restart':
            console.print("\n[bold]Restarting services...[/bold]")
        elif action == 'build':
            console.print("\n[bold]Building compose files...[/bold]")
        elif action == 'rebuild':
            console.print("\n[bold]Rebuilding services (clean + build + start)...[/bold]")
        elif action == 'clean':
            console.print("\n[bold red]⚠️  Cleaning deployment...[/bold red]")
        # Note: 'status' action doesn't print a header here because show_status() prints its own

        try:
            # Run subprocess with timeout (5 minutes for deploy operations)
            # Set environment to suppress config/registry warnings in subprocess
            env = os.environ.copy()
            env['OSPREY_QUIET'] = '1'  # Signal to suppress non-critical warnings
            result = subprocess.run(cmd, cwd=project_path or Path.cwd(), timeout=300, env=env)
        except subprocess.TimeoutExpired:
            console.print(f"\n{Messages.error('Command timed out after 5 minutes')}")
            console.print(Messages.warning("The operation took too long. Check your container runtime."))
            input("\nPress ENTER to continue...")
            if original_dir:
                try:
                    os.chdir(original_dir)
                except (OSError, PermissionError):
                    pass
            return

        if result.returncode == 0:
            if action == 'up':
                console.print(f"\n{Messages.success('Services started')}")
            elif action == 'down':
                console.print(f"\n{Messages.success('Services stopped')}")
            elif action == 'restart':
                console.print(f"\n{Messages.success('Services restarted')}")
            elif action == 'build':
                console.print(f"\n{Messages.success('Compose files built')}")
            elif action == 'rebuild':
                console.print(f"\n{Messages.success('Services rebuilt and started')}")
            elif action == 'clean':
                console.print(f"\n{Messages.success('Deployment cleaned')}")
        else:
            console.print(f"\n{Messages.warning(f'Command exited with code {result.returncode}')}")

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
        with quiet_logger(['REGISTRY', 'CONFIG']):
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
                    config_data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )

                console.print(f"\n[bold]Configuration for {project_path.name}:[/bold]\n")
                syntax = Syntax(
                    output_str,
                    "yaml",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True
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
                    default_model="anthropic/claude-haiku"
                )

                # Parse the rendered config as YAML
                config_data = yaml.safe_load(rendered_config)

                # Format as YAML
                output_str = yaml.dump(
                    config_data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )

                # Print to console with syntax highlighting
                console.print("\n[bold]Osprey Default Configuration:[/bold]\n")
                syntax = Syntax(
                    output_str,
                    "yaml",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True
                )
                console.print(syntax)
                console.print(f"\n[dim]Tip: Use {Messages.command('osprey export-config --output file.yml')} to save to file[/dim]")

    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")

    input("\nPress ENTER to continue...")


def handle_help_action():
    """Show CLI help."""
    help_text = """
# Osprey Framework CLI

## Main Commands

- `osprey` - Launch interactive menu (you are here!)
- `osprey init <project>` - Create new project
- `osprey chat` - Start CLI conversation
- `osprey deploy up|down|status` - Manage services
- `osprey health` - Run system health check
- `osprey export-config` - View configuration

## Examples

Create a new project:
    osprey init my-agent

Start a conversation (from project directory):
    osprey chat

Deploy web services:
    osprey deploy up

## Documentation

- Documentation: https://als-apg.github.io/osprey
- GitHub: https://github.com/als-apg/osprey
- Paper: https://arxiv.org/abs/2508.15066

"""
    console.print()
    console.print(Panel(
        Markdown(help_text),
        title="[header]Osprey Help[/header]",
        border_style=ThemeConfig.get_border_style(),
        width=80
    ))
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

        if action is None or action == 'exit':
            console.print("\n[accent]👋 Goodbye![/accent]\n")
            break

        # Handle tuple actions (project selection)
        if isinstance(action, tuple):
            action_type, action_data = action

            if action_type == 'select_project':
                project_path = action_data
                handle_project_selection(project_path)
                continue

        # Handle string actions (standard commands)
        if action == 'init_interactive':
            next_action = run_interactive_init()
            if next_action == 'exit':
                break
        elif action == 'chat':
            handle_chat_action()
        elif action == 'deploy':
            handle_deploy_action()
        elif action == 'health':
            handle_health_action()
        elif action == 'config':
            handle_export_action()
        elif action == 'registry':
            from osprey.cli.registry_cmd import handle_registry_action
            handle_registry_action()
        elif action == 'help':
            handle_help_action()


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

