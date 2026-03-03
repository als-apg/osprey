"""Interactive init wizard for creating new Osprey projects.

This module contains the step-by-step project creation flow,
including template selection, channel finder configuration,
code generator selection, provider/model selection, and
API key configuration.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from osprey.cli.styles import (
    Messages,
    console,
    get_questionary_style,
)

try:
    import questionary
    from questionary import Choice
except ImportError:
    questionary = None
    Choice = None


custom_style = get_questionary_style()


def select_template(templates: list[str]) -> str | None:
    """Interactive template selection.

    Args:
        templates: List of available template names

    Returns:
        Selected template name, or None if cancelled
    """
    # Template descriptions (could also come from template metadata)
    descriptions = {
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
            "all              - Include both pipelines (maximum flexibility, comparison)",
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
            f"[dim]Check that osprey is properly installed: {Messages.command('uv sync --all-extras')}[/dim]\n"
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


def get_api_key_name(provider: str) -> str | None:
    """Get environment variable name for provider API key.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai')

    Returns:
        Environment variable name, or None if provider doesn't need API key
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    if provider in PROVIDER_API_KEYS:
        return PROVIDER_API_KEYS[provider]
    return f"{provider.upper()}_API_KEY"


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
    from osprey.cli.templates.scaffolding import detect_environment_variables

    detected_env = detect_environment_variables()

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
    from osprey.cli.interactive_menu import get_provider_metadata

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


def run_interactive_init() -> str:
    """Interactive init flow with provider/model selection.

    Returns:
        Navigation action ('menu', 'exit', 'chat', etc.)
    """
    from osprey.cli.interactive_menu import (
        check_directory_has_active_mounts,
        get_code_generator_metadata,
        get_provider_metadata,
        select_model,
        select_provider,
    )
    from osprey.cli.menu_display import show_banner, show_success_art

    console.clear()
    show_banner(context="interactive")
    console.print(f"\n{Messages.header('Create New Project')}\n")

    # Get dynamic data with loading indicator
    from osprey.cli.templates.manager import TemplateManager

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

    # 3. Provider selection (step number adjusts)
    if template == "control_assistant":
        step_num = 6  # After template, name, channel_finder, capabilities, code_generator
    else:
        step_num = 3  # After template, name
    console.print(f"\n[bold]Step {step_num}: AI Provider[/bold]\n")
    provider = select_provider(providers)
    if provider is None:
        return "menu"

    # 4. Model selection (step number adjusts)
    if template == "control_assistant":
        step_num = 7  # After template, name, channel_finder, capabilities, code_generator, provider
    else:
        step_num = 4  # After template, name, provider
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
    console.print("  Mode:          [value]Claude Code[/value]")
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
        context = {
            "default_provider": provider,
            "default_model": model,
            "claude_code_only": True,
        }
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
            registry_style="extend",
            context=context,
            force=True,
        )

        # Generate manifest for migration support
        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=template,
            registry_style="extend",
            context=context,
        )

        msg = Messages.success("Project created at:")
        path = Messages.path(str(project_path))
        console.print(f"\n{msg} {path}\n")

        # Check if API keys were detected and .env was created
        from osprey.cli.templates.scaffolding import detect_environment_variables
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        detected_env = detect_environment_variables()
        api_key_names = {v for v in PROVIDER_API_KEYS.values() if v is not None}
        has_api_keys = any(key in detected_env for key in api_key_names)

        if has_api_keys:
            env_file = project_path / ".env"
            if env_file.exists():
                console.print(Messages.success("Created .env with detected API keys"))
                detected_keys = [key for key in api_key_names if key in detected_env]
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
                    Choice("[<] Return to main menu", value="menu"),
                    Choice("[x] Exit and show next steps", value="exit"),
                ],
                style=custom_style,
            ).ask()

            if next_action == "exit":
                # Show next steps like the direct init command
                console.print("\n[bold]Next steps:[/bold]")
                console.print(
                    f"  1. Navigate to project: {Messages.command(f'cd {project_path.name}')}"
                )
                console.print("  2. # .env already configured with API key")
                console.print(f"  3. Launch web terminal: {Messages.command('osprey web')}")
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
            console.print(f"  3. Launch web terminal: {Messages.command('osprey web')}")
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
