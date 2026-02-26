"""Code generation commands for Osprey Framework.

This module provides the 'osprey generate' command group for creating
Osprey components from various sources.
"""

import asyncio
import re
from pathlib import Path

import click

from .styles import Messages, Styles, console

# =============================================================================
# Project Detection (same logic as interactive_menu.py)
# =============================================================================


def is_project_initialized() -> bool:
    """Check if we're in an osprey project directory.

    Returns:
        True if config.yml exists in current directory
    """
    return (Path.cwd() / "config.yml").exists()


# Lazy imports for heavy dependencies
def get_mcp_generator():
    """Lazy import of MCP generator."""
    from osprey.generators import MCPCapabilityGenerator

    return MCPCapabilityGenerator


def get_prompt_generator():
    """Lazy import of Prompt generator."""
    from osprey.generators import PromptCapabilityGenerator

    return PromptCapabilityGenerator


def get_server_template():
    """Lazy import of MCP server template."""
    from osprey.generators.mcp_server_template import write_mcp_server_file

    return write_mcp_server_file


def initialize_registry():
    """Lazy import and initialize registry.

    Must be called from within a project directory (config.yml present).
    """
    from osprey.registry import initialize_registry
    from osprey.utils.log_filter import quiet_logger

    # Initialize registry (quiet mode to suppress logs)
    with quiet_logger(["registry", "CONFIG"]):
        initialize_registry()

    return True


def _determine_capabilities_path(capability_name: str) -> Path:
    """Determine the proper capabilities directory based on project structure.

    Tries to find the capabilities directory from the registry path.
    Falls back to ./capabilities/ if registry not found.

    Args:
        capability_name: Name of the capability

    Returns:
        Path to the capability file
    """
    try:
        from osprey.generators.registry_updater import find_registry_file

        registry_path = find_registry_file()
        if registry_path:
            # Capabilities should be in the same directory as registry
            # e.g., if registry is at src/my_project/registry.py
            # capabilities should be at src/my_project/capabilities/
            capabilities_dir = registry_path.parent / "capabilities"

            # Create the directory if it doesn't exist
            capabilities_dir.mkdir(parents=True, exist_ok=True)

            return capabilities_dir / f"{capability_name}.py"
    except Exception:
        # If anything fails, fall back to simple path
        pass

    # Fallback: use simple relative path
    return Path(f"capabilities/{capability_name}.py")


@click.group()
def generate():
    """Generate Osprey components from various sources.

    This command group provides code generation utilities for creating
    Osprey capabilities, demo servers, and other components.

    Available generators:

    \b
      - capability: Generate Osprey capability from MCP server or natural language prompt
      - mcp-server: Generate demo MCP server for testing
      - claude-config: Generate Claude Code generator configuration file

    Examples:

    \b
      # Generate capability from MCP server
      $ osprey generate capability --from-mcp http://localhost:3001 --name slack_mcp

      # Generate capability from natural language prompt
      $ osprey generate capability --from-prompt "This capability fetches weather data" --name weather

      # Generate capability in simulated mode (no server needed)
      $ osprey generate capability --from-mcp simulated --name weather_mcp

      # Generate a demo MCP server
      $ osprey generate mcp-server --name my_server --output ./server.py

      # Generate Claude Code configuration
      $ osprey generate claude-config
    """
    pass


@generate.command()
@click.option(
    "--from-mcp",
    "mcp_url",
    default=None,
    help='MCP server URL (e.g., http://localhost:3001) or "simulated" for demo mode',
)
@click.option(
    "--from-prompt",
    "prompt",
    default=None,
    help="Natural language description of what the capability should do",
)
@click.option(
    "--name",
    "-n",
    "capability_name",
    default=None,
    help="Name for the generated capability (e.g., slack_mcp, weather_mcp). Auto-generated if using --from-prompt.",
)
@click.option(
    "--server-name",
    default=None,
    help="Human-readable server name (only for --from-mcp, default: derived from capability name)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default=None,
    help="Output file path (default: ./capabilities/<name>.py)",
)
@click.option(
    "--provider", default=None, help="LLM provider override (e.g., anthropic, openai, cborg, amsc)"
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Model ID override (e.g., claude-sonnet-4-20250514, gpt-4o)",
)
@click.option("--quiet", "-q", is_flag=True, help="Reduce output verbosity")
def capability(
    mcp_url: str,
    prompt: str,
    capability_name: str,
    server_name: str,
    output_file: str,
    provider: str,
    model_id: str,
    quiet: bool,
):
    """Generate Osprey capability from MCP server or natural language prompt.

    This command supports two generation modes:

    \b
      1. FROM MCP SERVER (--from-mcp):
         Creates a production-ready capability that connects to an MCP server
         with ReAct agent execution, complete business logic.

      2. FROM PROMPT (--from-prompt):
         Creates a capability skeleton with classifier/orchestrator guides
         but PLACEHOLDER business logic that you must implement.

    Generated components (both modes):

    \b
      - Capability class structure
      - Classifier guide (when to activate)
      - Orchestrator guide (how to plan steps)
      - Context class structure
      - Error handling structure
      - Registry registration snippet

    Examples:

    \b
      # Generate from real MCP server
      $ osprey generate capability --from-mcp http://localhost:3001 -n slack_mcp

      # Generate from natural language prompt (name auto-suggested)
      $ osprey generate capability --from-prompt "Fetches weather data from an API"

      # Generate from prompt with explicit name
      $ osprey generate capability --from-prompt "Query database for user info" -n db_query

      # Simulated mode (no server needed, uses weather demo tools)
      $ osprey generate capability --from-mcp simulated -n weather_mcp

      # Custom output location
      $ osprey generate capability --from-mcp http://localhost:3001 \\
          -n slack_mcp --output ./my_app/capabilities/slack.py

      # Override LLM provider/model for guide generation
      $ osprey generate capability --from-prompt "Send emails via SMTP" \\
          --provider anthropic --model claude-sonnet-4-20250514

    After generation:

    \b
      1. Review and customize the generated code
      2. If using --from-prompt: IMPLEMENT the execute() method
      3. Customize the context class based on your data structure
      4. Add to your project's registry.py
      5. Test with: osprey chat
    """
    # Validate: Must provide exactly one of --from-mcp or --from-prompt
    if (mcp_url is None) == (prompt is None):
        console.print(
            f"\n{Messages.error('Must specify exactly one of --from-mcp or --from-prompt')}"
        )
        console.print()
        console.print("  [bold]Examples:[/bold]")
        console.print(
            "    "
            + Messages.command(
                "osprey generate capability --from-mcp http://localhost:3001 -n slack"
            )
        )
        console.print(
            "    "
            + Messages.command('osprey generate capability --from-prompt "Fetch weather data"')
        )
        console.print()
        raise click.Abort()

    # Determine generation mode
    from_prompt_mode = prompt is not None

    # For --from-prompt, name is optional (will be suggested by LLM)
    if from_prompt_mode:
        if not capability_name and not quiet:
            console.print(
                f"\n[{Styles.DIM}]ℹ️  No --name provided. LLM will suggest a capability name.[/{Styles.DIM}]"
            )
    else:
        # For --from-mcp, name is required
        if not capability_name:
            console.print(f"\n{Messages.error('--name is required when using --from-mcp')}")
            console.print()
            console.print("  [bold]Example:[/bold]")
            console.print(
                "    "
                + Messages.command(
                    "osprey generate capability --from-mcp http://localhost:3001 -n slack_mcp"
                )
            )
            console.print()
            raise click.Abort()

    # Check if we're in a project directory
    if not is_project_initialized():
        console.print(f"\n{Messages.error('Not in an Osprey project directory')}")
        console.print()
        console.print("  This command requires an Osprey project with [accent]config.yml[/accent]")
        console.print()
        console.print("  [bold]To create a new project:[/bold]")
        console.print("    " + Messages.command("osprey init my-project"))
        console.print()
        console.print("  [bold]Or navigate to an existing project:[/bold]")
        console.print("    " + Messages.command("cd my-project"))
        console.print("    " + Messages.command("osprey generate capability ..."))
        console.print()
        raise click.Abort()

    if from_prompt_mode:
        # FROM PROMPT mode
        _generate_from_prompt(
            prompt=prompt,
            capability_name=capability_name,
            output_file=output_file,
            provider=provider,
            model_id=model_id,
            quiet=quiet,
        )
    else:
        # FROM MCP mode
        _generate_from_mcp(
            mcp_url=mcp_url,
            capability_name=capability_name,
            server_name=server_name,
            output_file=output_file,
            provider=provider,
            model_id=model_id,
            quiet=quiet,
        )


def _generate_from_mcp(
    mcp_url: str,
    capability_name: str,
    server_name: str,
    output_file: str,
    provider: str,
    model_id: str,
    quiet: bool,
):
    """Generate capability from MCP server."""
    # Derive server name if not provided
    if not server_name:
        # Convert capability_name to title case (e.g., slack_mcp -> Slack Mcp)
        server_name = capability_name.replace("_", " ").title().replace(" ", "")

    # Derive output file if not provided
    if not output_file:
        # Try to determine the proper capabilities directory from registry
        output_path = _determine_capabilities_path(capability_name)
    else:
        output_path = Path(output_file)

    # Check if simulated mode
    simulated = mcp_url.lower() == "simulated"
    if simulated:
        mcp_url = None

    console.print("\n🎨 [header]Generating MCP Capability[/header]\n")
    console.print(
        f"  [{Styles.LABEL}]Capability:[/{Styles.LABEL}] [{Styles.VALUE}]{capability_name}[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Server:[/{Styles.LABEL}] [{Styles.VALUE}]{server_name}[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Mode:[/{Styles.LABEL}] [{Styles.VALUE}]{'Simulated' if simulated else f'Real MCP ({mcp_url})'}[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Output:[/{Styles.LABEL}] [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]\n"
    )

    try:
        # Initialize registry (required for LLM providers)
        if not quiet:
            with console.status("[dim]Initializing registry...[/dim]"):
                initialize_registry()
            console.print(f"  {Messages.success('Registry initialized')}")
        else:
            initialize_registry()

        # Create generator
        MCPCapabilityGenerator = get_mcp_generator()
        generator = MCPCapabilityGenerator(
            capability_name=capability_name,
            server_name=server_name,
            verbose=not quiet,
            provider=provider,
            model_id=model_id,
        )

        # Run async generation
        asyncio.run(
            _generate_capability_async(
                generator=generator,
                mcp_url=mcp_url,
                simulated=simulated,
                output_path=output_path,
                quiet=quiet,
            )
        )

    except KeyboardInterrupt:
        console.print(f"\n{Messages.warning('Generation cancelled by user')}")
        raise click.Abort() from None
    except RuntimeError as e:
        # RuntimeError with clear message - don't show traceback
        error_msg = str(e)
        if error_msg.startswith("\n"):
            # Error already formatted with newlines
            console.print(error_msg)
        else:
            console.print(f"\n{Messages.error('Generation failed: ')}")
            console.print(f"\n{error_msg}")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        if not quiet:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


def _generate_from_prompt(
    prompt: str,
    capability_name: str | None,
    output_file: str,
    provider: str,
    model_id: str,
    quiet: bool,
):
    """Generate capability from natural language prompt."""
    console.print("\n🎨 [header]Generating Capability from Prompt[/header]\n")
    console.print(
        f"  [{Styles.LABEL}]Prompt:[/{Styles.LABEL}] [{Styles.VALUE}]{prompt[:80]}{'...' if len(prompt) > 80 else ''}[/{Styles.VALUE}]"
    )
    if capability_name:
        console.print(
            f"  [{Styles.LABEL}]Name:[/{Styles.LABEL}] [{Styles.VALUE}]{capability_name}[/{Styles.VALUE}]"
        )
    else:
        console.print(
            f"  [{Styles.LABEL}]Name:[/{Styles.LABEL}] [{Styles.DIM}](will be suggested by LLM)[/{Styles.DIM}]"
        )
    console.print()

    try:
        # Initialize registry (required for LLM providers)
        if not quiet:
            with console.status("[dim]Initializing registry...[/dim]"):
                initialize_registry()
            console.print(f"  {Messages.success('Registry initialized')}")
        else:
            initialize_registry()

        # Create generator
        PromptCapabilityGenerator = get_prompt_generator()
        generator = PromptCapabilityGenerator(
            prompt=prompt,
            capability_name=capability_name,
            verbose=not quiet,
            provider=provider,
            model_id=model_id,
        )

        # Run async generation
        asyncio.run(
            _generate_from_prompt_async(generator=generator, output_file=output_file, quiet=quiet)
        )

    except KeyboardInterrupt:
        console.print(f"\n{Messages.warning('Generation cancelled by user')}")
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        if not quiet:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


async def _generate_from_prompt_async(generator, output_file: str, quiet: bool):
    """Async helper for prompt-based capability generation."""
    # Step 1: Generate metadata (name suggestions)
    if not quiet:
        console.print(f"\n🧠 [{Styles.HEADER}]Step 1: Analyzing prompt...[/{Styles.HEADER}]")

    with console.status("[dim]Generating metadata with LLM...[/dim]"):
        metadata = await generator.generate_metadata()

    # Use the suggested name if user didn't provide one
    capability_name = generator.capability_name or metadata.capability_name_suggestion
    generator.capability_name = capability_name  # Update generator with final name

    console.print(f"  {Messages.success(f'Capability name: {capability_name}')}")
    console.print(f"  {Messages.success(f'Context type: {metadata.context_type_suggestion}')}")

    # Determine output path
    if not output_file:
        output_path = _determine_capabilities_path(capability_name)
    else:
        output_path = Path(output_file)

    console.print(
        f"  [{Styles.LABEL}]Output:[/{Styles.LABEL}] [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]"
    )

    # Step 2: Generate guides
    if not quiet:
        console.print(
            f"\n🤖 [{Styles.HEADER}]Step 2: Generating classifier and orchestrator guides...[/{Styles.HEADER}]"
        )

    with console.status("[dim]Analyzing requirements and generating examples...[/dim]"):
        classifier_analysis, orchestrator_analysis = await generator.generate_guides()

    num_examples = len(classifier_analysis.positive_examples) + len(
        classifier_analysis.negative_examples
    )
    console.print("  " + Messages.success(f"Generated {num_examples} classifier examples"))
    console.print(
        "  "
        + Messages.success(
            f"Generated {len(orchestrator_analysis.example_steps)} orchestrator examples"
        )
    )

    # Step 3: Generate code
    if not quiet:
        console.print(
            f"\n📝 [{Styles.HEADER}]Step 3: Generating capability code...[/{Styles.HEADER}]"
        )

    with console.status("[dim]Creating capability skeleton...[/dim]"):
        code = generator.generate_capability_code(classifier_analysis, orchestrator_analysis)

    console.print("  " + Messages.success("Code generated"))

    # Step 4: Write file
    if not quiet:
        console.print(f"\n💾 [{Styles.HEADER}]Step 4: Writing output file...[/{Styles.HEADER}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 encoding explicitly to support Unicode characters on Windows
    output_path.write_text(code, encoding="utf-8")

    console.print(f"  {Messages.success(f'Written: {output_path} ({len(code):,} bytes)')}")

    # Success summary
    console.print("\n" + "=" * 70)
    console.print(
        f"[{Styles.BOLD_SUCCESS}]✅ SUCCESS! Capability Skeleton Generated[/{Styles.BOLD_SUCCESS}]"
    )
    console.print("=" * 70 + "\n")

    console.print(f"[{Styles.HEADER}]What was created:[/{Styles.HEADER}]")
    console.print(f"  ✓ Capability skeleton: {capability_name}")
    console.print(f"  ✓ Classifier guide with {num_examples} examples")
    console.print(
        f"  ✓ Orchestrator guide with {len(orchestrator_analysis.example_steps)} examples"
    )
    console.print("  ✓ Context class structure")
    console.print("  ✓ Error handling structure")
    console.print("  ✓ Registry registration snippet")

    console.print(
        f"\n[{Styles.WARNING}]⚠️  IMPORTANT: This is a SKELETON with PLACEHOLDER business logic![/{Styles.WARNING}]"
    )
    console.print()

    console.print(f"[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
    console.print(f"  1. Review: [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]")
    console.print("  2. [bold]IMPLEMENT the execute() method[/bold] with actual business logic")
    console.print("  3. Customize the context class to match your data structure")
    console.print("  4. Update provides/requires fields based on dependencies")
    console.print("  5. Add to your registry.py (see snippet at bottom of file)")
    console.print(f"  6. Test with: [{Styles.ACCENT}]osprey chat[/{Styles.ACCENT}]")
    console.print()

    # Offer to add to registry automatically
    if not quiet:
        await _offer_registry_integration(generator, classifier_analysis, orchestrator_analysis)


async def _offer_config_integration(generator):
    """Offer to automatically add mcp_react model to config.yml.

    Args:
        generator: MCPCapabilityGenerator instance
    """
    try:
        import questionary

        from osprey.generators.config_updater import (
            add_capability_react_to_config,
            find_config_file,
            get_config_preview,
            get_orchestrator_model_config,
            has_capability_react_model,
        )

        from .styles import get_questionary_style

        capability_name = generator.capability_name

        # Find config file
        config_path = find_config_file()
        if not config_path:
            return

        # Check if already configured
        if has_capability_react_model(config_path, capability_name):
            model_key = f"{capability_name}_react"
            console.print(
                f"\n[{Styles.DIM}]ℹ️  {model_key} model already configured in config.yml[/{Styles.DIM}]"
            )
            return

        # Get orchestrator config as template
        template_config = get_orchestrator_model_config(config_path)

        # Ask user
        console.print()
        console.print(f"[{Styles.HEADER}]Config Integration:[/{Styles.HEADER}]")
        console.print(f"  Found config: [{Styles.VALUE}]{config_path}[/{Styles.VALUE}]")
        console.print()

        model_key = f"{capability_name}_react"
        add_to_config = await questionary.confirm(
            f"Add {model_key} model configuration to config.yml?",
            default=True,
            style=get_questionary_style(),
        ).ask_async()

        if not add_to_config:
            console.print("\n  " + Messages.info("Skipped. You can add manually if needed."))
            return

        # Show preview
        console.print(f"\n[{Styles.HEADER}]Preview of changes:[/{Styles.HEADER}]")
        preview = get_config_preview(capability_name, template_config)
        console.print(preview)

        # Confirm
        confirm = await questionary.confirm(
            "Apply these changes to config.yml?", default=True, style=get_questionary_style()
        ).ask_async()

        if confirm:
            # Backup first
            backup_path = config_path.with_suffix(".yml.bak")
            backup_path.write_text(config_path.read_text())

            # Write new content
            new_content, _ = add_capability_react_to_config(
                config_path, capability_name, template_config
            )
            config_path.write_text(new_content)

            console.print(f"\n  {Messages.success(f'Updated {config_path}')}")
            console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")
            console.print()
            console.print(
                "  "
                + Messages.info(
                    f"{model_key} model configured! {capability_name} will use this model."
                )
            )
        else:
            console.print(
                "\n  " + Messages.info("Changes not applied. Add manually to config.yml if needed.")
            )

    except ImportError:
        # questionary not available
        pass
    except Exception as e:
        # Escape the error message to prevent Rich markup interpretation
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        console.print(
            f"\n[{Styles.WARNING}]⚠️  Could not update config: {error_msg}[/{Styles.WARNING}]"
        )
        console.print(
            "  " + Styles.DIM + "Add mcp_react model manually to config.yml." + f"[/{Styles.DIM}]"
        )


async def _offer_registry_integration(generator, classifier_analysis, orchestrator_analysis):
    """Offer to automatically add capability to registry.

    Args:
        generator: BaseCapabilityGenerator instance (MCP or Prompt)
        classifier_analysis: Generated classifier analysis
        orchestrator_analysis: Generated orchestrator analysis
    """
    try:
        import questionary

        from osprey.generators.registry_updater import (
            add_to_registry,
            find_registry_file,
            is_already_registered,
        )

        from .styles import get_questionary_style

        # Find registry file from config
        registry_path = find_registry_file()
        if not registry_path:
            console.print(
                f"\n[{Styles.DIM}]Note: Could not find registry.py from config. "
                "Add capability manually.[/{Styles.DIM}]"
            )
            return

        # Check if already registered
        if is_already_registered(registry_path, generator.capability_name):
            console.print(
                f"\n[{Styles.WARNING}]⚠️  Capability '{generator.capability_name}' "
                f"is already registered in {registry_path}[/{Styles.WARNING}]"
            )
            return

        # Ask user if they want to add it
        console.print()
        console.print(f"[{Styles.HEADER}]Registry Integration:[/{Styles.HEADER}]")
        console.print(f"  Found registry: [{Styles.VALUE}]{registry_path}[/{Styles.VALUE}]")
        console.print()

        add_to_reg = await questionary.confirm(
            "Add this capability to your registry automatically?",
            default=True,
            style=get_questionary_style(),
        ).ask_async()

        if not add_to_reg:
            console.print(
                "\n  "
                + Messages.info(
                    "Skipped registry update. "
                    "Add manually using snippet at bottom of generated file."
                )
            )
            return

        # Generate class name and context type from generator
        # Support both MCP and Prompt generators
        class_name = generator._to_class_name(generator.capability_name)

        # Determine context class name and type based on generator type
        if hasattr(generator, "server_name"):
            # MCP generator
            context_class_name = generator._to_class_name(
                generator.capability_name, suffix="ResultsContext"
            )
            context_type = f"{generator.server_name.upper()}_RESULTS"
            description = f"{generator.server_name} operations via MCP server"
        else:
            # Prompt generator - use metadata if available
            context_class_name = generator._to_class_name(
                generator.capability_name, suffix="Context"
            )
            if hasattr(generator, "metadata") and generator.metadata:
                context_type = generator.metadata.context_type_suggestion
                description = generator.metadata.description
            else:
                context_type = f"{generator.capability_name.upper()}_RESULTS"
                description = f"{generator.capability_name} capability"

        # Add to registry
        new_content, preview = add_to_registry(
            registry_path,
            generator.capability_name,
            class_name,
            context_type,
            context_class_name,
            description,
        )

        # Show preview
        console.print(f"\n[{Styles.HEADER}]Preview of changes:[/{Styles.HEADER}]")
        console.print(preview)

        # Confirm
        confirm = await questionary.confirm(
            "Apply these changes to registry.py?", default=True, style=get_questionary_style()
        ).ask_async()

        if confirm:
            # Backup first
            backup_path = registry_path.with_suffix(".py.bak")
            # Use UTF-8 encoding for backup
            backup_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

            # Write new content
            # Use UTF-8 encoding explicitly to support Unicode characters on Windows
            registry_path.write_text(new_content, encoding="utf-8")

            console.print(f"\n  {Messages.success(f'Updated {registry_path}')}")
            console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")
            console.print()
            console.print(
                "  " + Messages.info("Capability is now registered! Test with: osprey chat")
            )
        else:
            console.print("\n  " + Messages.info("Changes not applied. Add manually if needed."))

    except ImportError:
        # questionary not available
        console.print(
            f"\n[{Styles.DIM}]Note: Install questionary for interactive registry updates[/{Styles.DIM}]"
        )
    except Exception as e:
        # Escape the error message to prevent Rich markup interpretation
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        console.print(
            f"\n[{Styles.WARNING}]⚠️  Could not update registry: {error_msg}[/{Styles.WARNING}]"
        )
        console.print(
            "  "
            + Styles.DIM
            + "Add capability manually using snippet at bottom of generated file."
            + f"[/{Styles.DIM}]"
        )


async def _generate_capability_async(
    generator, mcp_url: str, simulated: bool, output_path: Path, quiet: bool
):
    """Async helper for capability generation."""
    # Step 1: Discover tools
    if not quiet:
        console.print(f"\n📡 [{Styles.HEADER}]Step 1: Discovering MCP tools...[/{Styles.HEADER}]")

    with console.status(
        "[dim]Connecting to MCP server...[/dim]"
        if not simulated
        else "[dim]Loading simulated tools...[/dim]"
    ):
        tools = await generator.discover_tools(mcp_url=mcp_url, simulated=simulated)

    if not tools:
        console.print(f"  {Messages.error('No tools found')}")
        raise click.Abort()

    console.print(f"  {Messages.success(f'Found {len(tools)} tools')}")
    if not quiet:
        for tool in tools[:5]:
            console.print(f"    [{Styles.DIM}]•[/{Styles.DIM}] {tool['name']}")
        if len(tools) > 5:
            console.print(f"    [{Styles.DIM}]... and {len(tools) - 5} more[/{Styles.DIM}]")

    # Step 2: Generate guides
    if not quiet:
        console.print(
            f"\n🤖 [{Styles.HEADER}]Step 2: Generating guides with LLM...[/{Styles.HEADER}]"
        )

    with console.status("[dim]Analyzing tools and generating examples...[/dim]"):
        classifier_analysis, orchestrator_analysis = await generator.generate_guides()

    num_examples = len(classifier_analysis.positive_examples) + len(
        classifier_analysis.negative_examples
    )
    console.print("  " + Messages.success(f"Generated {num_examples} classifier examples"))
    console.print(
        "  "
        + Messages.success(
            f"Generated {len(orchestrator_analysis.example_steps)} orchestrator examples"
        )
    )

    # Step 3: Generate code
    if not quiet:
        console.print(
            f"\n📝 [{Styles.HEADER}]Step 3: Generating capability code...[/{Styles.HEADER}]"
        )

    with console.status("[dim]Creating capability class...[/dim]"):
        code = generator.generate_capability_code(classifier_analysis, orchestrator_analysis)

    console.print("  " + Messages.success("Code generated"))

    # Step 4: Write file
    if not quiet:
        console.print(f"\n💾 [{Styles.HEADER}]Step 4: Writing output file...[/{Styles.HEADER}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 encoding explicitly to support Unicode characters on Windows
    output_path.write_text(code, encoding="utf-8")

    console.print(f"  {Messages.success(f'Written: {output_path} ({len(code):,} bytes)')}")

    # Success summary
    console.print("\n" + "=" * 70)
    console.print(
        f"[{Styles.BOLD_SUCCESS}]✅ SUCCESS! MCP Capability Generated[/{Styles.BOLD_SUCCESS}]"
    )
    console.print("=" * 70 + "\n")

    console.print(f"[{Styles.HEADER}]What was created:[/{Styles.HEADER}]")
    console.print(f"  ✓ Capability class: {generator.capability_name}")
    console.print("  ✓ MCP client integration")
    console.print(f"  ✓ Classifier guide with {num_examples} examples")
    console.print(
        f"  ✓ Orchestrator guide with {len(orchestrator_analysis.example_steps)} examples"
    )
    console.print("  ✓ Context class for results")
    console.print("  ✓ Error handling")
    console.print("  ✓ Registry registration snippet")

    console.print(f"\n[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
    console.print(f"  1. Review: [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]")
    console.print("  2. Customize the context class based on your data structure")
    console.print("  3. Add to your registry.py (see snippet at bottom of file)")
    console.print(f"  4. Test with: [{Styles.ACCENT}]osprey chat[/{Styles.ACCENT}]")
    console.print()

    # Offer to add to registry automatically
    if not quiet:
        await _offer_registry_integration(generator, classifier_analysis, orchestrator_analysis)

        # Offer to add mcp_react model to config
        await _offer_config_integration(generator)


@generate.command(name="mcp-server")
@click.option(
    "--name",
    "-n",
    "server_name",
    default="demo_mcp",
    help="Name for the server (default: demo_mcp)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default=None,
    help="Output file path (default: ./<name>_server.py)",
)
@click.option("--port", "-p", default=3001, type=int, help="Server port (default: 3001)")
def mcp_server(server_name: str, output_file: str, port: int):
    """Generate demo MCP server for testing.

    Creates a demo MCP server with weather tools that you can run locally
    to test Osprey's MCP capability generation. The server uses FastMCP
    for simple, Pythonic MCP server implementation.

    Included tools:

    \b
      - get_current_weather: Get current weather conditions
      - get_forecast: Get weather forecast for upcoming days
      - get_weather_alerts: Get active weather alerts and warnings

    Examples:

    \b
      # Generate weather demo server
      $ osprey generate mcp-server

      # Generate on custom port
      $ osprey generate mcp-server --port 3002

      # Custom output location
      $ osprey generate mcp-server --name my_server --output ./servers/mcp.py

    After generation:

    \b
      1. Install FastMCP: pip install fastmcp
      2. Run the server: python <output_file>
      3. Generate capability: osprey generate capability --from-mcp http://localhost:<port>
    """
    # Derive output file if not provided
    if not output_file:
        output_file = f"{server_name}_server.py"

    output_path = Path(output_file)

    console.print("\n🚀 [header]Generating MCP Server[/header]\n")
    console.print(
        f"  [{Styles.LABEL}]Server:[/{Styles.LABEL}] [{Styles.VALUE}]{server_name}[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Tools:[/{Styles.LABEL}] [{Styles.VALUE}]weather (demo)[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Port:[/{Styles.LABEL}] [{Styles.VALUE}]{port}[/{Styles.VALUE}]"
    )
    console.print(
        f"  [{Styles.LABEL}]Output:[/{Styles.LABEL}] [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]\n"
    )

    try:
        with console.status("[dim]Generating server code...[/dim]"):
            write_mcp_server_file = get_server_template()
            output_path = write_mcp_server_file(
                output_path=output_path, server_name=server_name, port=port
            )

        console.print(f"  {Messages.success(f'Server generated: {output_path}')}\n")

        console.print(f"[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
        console.print(
            f"  1. Install dependencies: [{Styles.ACCENT}]pip install fastmcp[/{Styles.ACCENT}]"
        )
        console.print(
            f"  2. Run the server: [{Styles.ACCENT}]python {output_path}[/{Styles.ACCENT}]"
        )
        console.print(
            f"  3. Generate capability: [{Styles.ACCENT}]osprey generate capability --from-mcp http://localhost:{port} -n {server_name}[/{Styles.ACCENT}]"
        )
        console.print()

    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


@generate.command(name="claude-config")
@click.option(
    "--output",
    "-o",
    "output_file",
    default="claude_generator_config.yml",
    help="Output file path (default: ./claude_generator_config.yml)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file if it exists")
def claude_config(output_file: str, force: bool):
    """Generate Claude Code generator configuration file.

    Creates a claude_generator_config.yml file with sensible defaults for
    the Claude Code generator. This file is required when using the
    'claude_code' code generator in your config.yml.

    The generated configuration includes:

    \b
      - API configuration (Anthropic or CBORG)
      - Phase definitions (scan, plan, implement, generate)
      - Pre-configured profiles (fast, robust)
      - Codebase guidance for example scripts
      - Security settings and documentation

    Examples:

    \b
      # Generate with default settings
      $ osprey generate claude-config

      # Custom output location
      $ osprey generate claude-config --output ./configs/claude.yml

      # Overwrite existing file
      $ osprey generate claude-config --force

    After generation:

    \b
      1. Review the generated configuration
      2. Update config.yml to use 'claude_code' generator
      3. Set API key: ANTHROPIC_API_KEY or CBORG_API_KEY
      4. Optionally add example scripts to _agent_data/example_scripts/
    """
    import yaml

    from .templates import TemplateManager

    output_path = Path(output_file)

    # Check if file exists
    if output_path.exists() and not force:
        console.print(f"\n{Messages.error(f'File already exists: {output_path}')}")
        console.print()
        console.print("  Use [accent]--force[/accent] to overwrite:")
        console.print(f"    {Messages.command('osprey generate claude-config --force')}")
        console.print()
        raise click.Abort()

    console.print("\n⚙️  [header]Generating Claude Code Configuration[/header]\n")
    console.print(
        f"  [{Styles.LABEL}]Output:[/{Styles.LABEL}] [{Styles.VALUE}]{output_path}[/{Styles.VALUE}]"
    )

    try:
        # Try to detect provider from config.yml if available
        default_provider = "anthropic"
        default_model = "claude-haiku-4-5-20251001"

        config_path = Path.cwd() / "config.yml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if config and "llm" in config:
                        default_provider = config["llm"].get("default_provider", "anthropic")
                        console.print(
                            f"  [{Styles.LABEL}]Detected provider:[/{Styles.LABEL}] [{Styles.VALUE}]{default_provider}[/{Styles.VALUE}]"
                        )
            except Exception:
                pass  # Best-effort provider detection; not critical for generation
        else:
            console.print(f"  [{Styles.DIM}]No config.yml found - using defaults[/{Styles.DIM}]")

        # Create template context
        ctx = {"default_provider": default_provider, "default_model": default_model}

        # Render template
        console.print(f"\n  [{Styles.DIM}]Rendering template...[/{Styles.DIM}]")

        template_manager = TemplateManager()
        template_manager.render_template(
            "apps/control_assistant/claude_generator_config.yml.j2", ctx, output_path
        )

        console.print(f"  {Messages.success(f'Configuration generated: {output_path}')}\n")

        # Success summary
        console.print(f"[{Styles.HEADER}]What was created:[/{Styles.HEADER}]")
        console.print("  ✓ API configuration (ready for " + default_provider + ")")
        console.print("  ✓ Phase definitions (scan, plan, implement, generate)")
        console.print("  ✓ Pre-configured profiles (fast, robust)")
        console.print("  ✓ Codebase guidance configuration")
        console.print("  ✓ Security and documentation")

        console.print(f"\n[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
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
        console.print("  4. (Optional) Add example scripts:")
        console.print("     [dim]mkdir -p _agent_data/example_scripts/plotting[/dim]")
        console.print("     [dim]# Add your example Python files...[/dim]")
        console.print()

    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


# =============================================================================
# Soft IOC Generator
# =============================================================================


def _get_channel_database_from_config(config_path: Path) -> str | None:
    """Get channel database path from channel_finder section in config.yml.

    Reads the active pipeline mode and extracts the database path from the
    corresponding pipeline configuration.

    Args:
        config_path: Path to config.yml

    Returns:
        Channel database path string, or None if not found
    """
    import yaml

    if not config_path.exists():
        return None

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    channel_finder = config.get("channel_finder", {})
    if not channel_finder:
        return None

    # Get the active pipeline mode
    pipeline_mode = channel_finder.get("pipeline_mode", "hierarchical")

    # Get the database path from the active pipeline config
    pipelines = channel_finder.get("pipelines", {})
    pipeline_config = pipelines.get(pipeline_mode, {})
    database_config = pipeline_config.get("database", {})

    return database_config.get("path")


def _generate_simulation_yaml_preview(sim_config: dict) -> str:
    """Generate YAML preview for simulation section.

    Args:
        sim_config: Simulation configuration dict

    Returns:
        Formatted YAML string for preview
    """
    lines = ["simulation:"]

    # Channel database
    lines.append(f'  channel_database: "{sim_config["channel_database"]}"')

    # IOC settings
    lines.append("  ioc:")
    lines.append(f'    name: "{sim_config["ioc"]["name"]}"')
    lines.append(f"    port: {sim_config['ioc']['port']}")
    lines.append(f'    output_dir: "{sim_config["ioc"]["output_dir"]}"')

    # Base backend (single dict, no dash)
    base = sim_config.get("base", {"type": "mock_style"})
    lines.append("  base:")
    base_type = base.get("type", "mock_style")
    lines.append(f'    type: "{base_type}"')
    if base_type == "mock_style":
        lines.append(f"    noise_level: {base.get('noise_level', 0.01)}")
        lines.append(f"    update_rate: {base.get('update_rate', 10.0)}")
    elif base.get("module_path"):
        lines.append(f'    module_path: "{base["module_path"]}"')
        lines.append(f'    class_name: "{base["class_name"]}"')
        if base.get("params"):
            lines.append("    params:")
            for k, v in base["params"].items():
                lines.append(f"      {k}: {repr(v)}")

    # Overlays (list with dashes)
    overlays = sim_config.get("overlays", [])
    if overlays:
        lines.append("  overlays:")
        for overlay in overlays:
            if overlay.get("module_path"):
                lines.append(f'    - module_path: "{overlay["module_path"]}"')
                lines.append(f'      class_name: "{overlay["class_name"]}"')
                if overlay.get("params"):
                    lines.append("      params:")
                    for k, v in overlay["params"].items():
                        lines.append(f"        {k}: {repr(v)}")
            elif overlay.get("type"):
                lines.append(f'    - type: "{overlay["type"]}"')

    return "\n".join(lines)


def _write_simulation_config(config_path: Path, sim_config: dict) -> Path:
    """Write simulation section to config.yml with backup.

    Uses comment-preserving YAML to maintain formatting and comments.
    Adds a section header when creating a new simulation section.

    Args:
        config_path: Path to config.yml
        sim_config: Simulation configuration dict

    Returns:
        Path to backup file
    """
    from osprey.generators.config_updater import update_yaml_file

    backup_path = update_yaml_file(
        config_path,
        {"simulation": sim_config},
        create_backup=True,
        section_comments={"simulation": "SIMULATION CONFIGURATION"},
    )
    return backup_path


async def _offer_simulation_config_setup(
    config_path: Path,
    force_init: bool = False,
    dry_run: bool = False,
) -> dict | None:
    """Offer interactive setup for simulation config.

    Args:
        config_path: Path to config.yml
        force_init: If True, always offer setup even if section exists
        dry_run: If True, show preview but don't write config

    Returns:
        Simulation config dict if created/previewed, None if skipped
    """
    try:
        import questionary

        from .styles import get_questionary_style
    except ImportError:
        console.print(
            f"\n[{Styles.WARNING}]questionary not installed. "
            f"Add simulation section manually to config.yml[/{Styles.WARNING}]"
        )
        return None

    console.print(f"\n[{Styles.HEADER}]Simulation Config Setup[/{Styles.HEADER}]\n")

    # 1. Get channel database from channel_finder config section
    channel_database = _get_channel_database_from_config(config_path)

    if channel_database:
        console.print(
            f"  [{Styles.LABEL}]Using channel database from config:[/{Styles.LABEL}] "
            f"[{Styles.VALUE}]{channel_database}[/{Styles.VALUE}]"
        )
        console.print()

        # Confirm or allow override
        use_detected = await questionary.confirm(
            f"Use this channel database? ({channel_database})",
            default=True,
            style=get_questionary_style(),
        ).ask_async()

        if use_detected is None:
            console.print(f"\n{Messages.warning('Setup cancelled')}")
            return None

        if not use_detected:
            channel_database = await questionary.path(
                "Enter channel database path:",
                style=get_questionary_style(),
            ).ask_async()
            if not channel_database:
                # Offer empty IOC instead of cancelling
                create_empty = await questionary.confirm(
                    "No channel database provided. Create empty IOC (heartbeat only)?",
                    default=True,
                    style=get_questionary_style(),
                ).ask_async()
                if not create_empty:
                    console.print(f"\n{Messages.warning('Setup cancelled')}")
                    return None
                channel_database = None  # Explicitly None for empty IOC
    else:
        # No channel_finder config - prompt for path
        console.print(
            f"  [{Styles.DIM}]No channel database found in channel_finder config[/{Styles.DIM}]"
        )
        console.print()
        channel_database = await questionary.path(
            "Enter channel database path:",
            style=get_questionary_style(),
        ).ask_async()
        if not channel_database:
            # Offer empty IOC instead of cancelling
            create_empty = await questionary.confirm(
                "No channel database provided. Create empty IOC (heartbeat only)?",
                default=True,
                style=get_questionary_style(),
            ).ask_async()
            if not create_empty:
                console.print(f"\n{Messages.warning('Setup cancelled')}")
                return None
            channel_database = None  # Explicitly None for empty IOC

    # 2. IOC Name - derive default from project name or config
    default_ioc_name = Path.cwd().name.replace("-", "_").lower() + "_sim"
    ioc_name = await questionary.text(
        "IOC name:",
        default=default_ioc_name,
        style=get_questionary_style(),
    ).ask_async()

    if not ioc_name:
        console.print(f"\n{Messages.warning('Setup cancelled')}")
        return None

    # 3. Port
    port_str = await questionary.text(
        "EPICS CA port:",
        default="5064",
        style=get_questionary_style(),
    ).ask_async()

    if not port_str:
        console.print(f"\n{Messages.warning('Setup cancelled')}")
        return None

    try:
        port = int(port_str)
    except ValueError:
        console.print(f"\n{Messages.error('Invalid port number')}")
        return None

    # 4. Output directory
    output_dir = await questionary.text(
        "Output directory:",
        default="generated_iocs/",
        style=get_questionary_style(),
    ).ask_async()

    if output_dir is None:
        console.print(f"\n{Messages.warning('Setup cancelled')}")
        return None

    # 5. Backend type
    backend_type = await questionary.select(
        "Backend type:",
        choices=[
            "mock_style (recommended - archiver-style simulation)",
            "passthrough (no simulation)",
            "custom (your own physics backend)",
        ],
        style=get_questionary_style(),
    ).ask_async()

    if not backend_type:
        console.print(f"\n{Messages.warning('Setup cancelled')}")
        return None

    if "mock_style" in backend_type:
        backend_type = "mock_style"
    elif "passthrough" in backend_type:
        backend_type = "passthrough"
    else:
        backend_type = "custom"

    # 6. Backend settings
    noise_level = 0.01
    update_rate = 10.0
    module_path = None
    class_name = None
    params = {}

    if backend_type == "mock_style":
        noise_str = await questionary.text(
            "Noise level (0.01 = 1%):",
            default="0.01",
            style=get_questionary_style(),
        ).ask_async()

        if noise_str:
            try:
                noise_level = float(noise_str)
            except ValueError:
                noise_level = 0.01

        rate_str = await questionary.text(
            "Update rate (Hz):",
            default="10.0",
            style=get_questionary_style(),
        ).ask_async()

        if rate_str:
            try:
                update_rate = float(rate_str)
            except ValueError:
                update_rate = 10.0

    elif backend_type == "custom":
        console.print()
        console.print(
            f"  [{Styles.DIM}]Custom backends must implement initialize(), on_write(), and step() methods."
        )
        console.print(f"  See: osprey.generators.backend_protocol.SimulationBackend[/{Styles.DIM}]")
        console.print()

        module_path = await questionary.text(
            "Module path (e.g., my_project.simulation.backend):",
            style=get_questionary_style(),
        ).ask_async()

        if not module_path:
            console.print(f"\n{Messages.warning('Setup cancelled')}")
            return None

        class_name = await questionary.text(
            "Class name (e.g., PyATBackend):",
            style=get_questionary_style(),
        ).ask_async()

        if not class_name:
            console.print(f"\n{Messages.warning('Setup cancelled')}")
            return None

        rate_str = await questionary.text(
            "Update rate (Hz):",
            default="10.0",
            style=get_questionary_style(),
        ).ask_async()

        if rate_str:
            try:
                update_rate = float(rate_str)
            except ValueError:
                update_rate = 10.0

        # Ask about constructor params
        add_params = await questionary.confirm(
            "Add constructor parameters to your backend?",
            default=False,
            style=get_questionary_style(),
        ).ask_async()

        if add_params:
            console.print(
                f"  [{Styles.DIM}]Enter params as 'key: value' (one per line, empty line to finish):[/{Styles.DIM}]"  # noqa: E501
            )
            while True:
                param_line = await questionary.text(
                    "  param:",
                    style=get_questionary_style(),
                ).ask_async()

                if not param_line or not param_line.strip():
                    break

                if ":" in param_line:
                    key, value = param_line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to parse as Python literal
                    try:
                        import ast

                        params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Keep as string
                        params[key] = value

    # Build base backend config
    base_config = {
        "type": backend_type,
        "noise_level": noise_level,
        "update_rate": update_rate,
    }

    # For custom backend as base, add module fields
    if backend_type == "custom":
        base_config["module_path"] = module_path
        base_config["class_name"] = class_name
        if params:
            base_config["params"] = params

    # Build config dict using base + overlays format
    sim_config = {
        "channel_database": channel_database,
        "ioc": {
            "name": ioc_name,
            "port": port,
            "output_dir": output_dir,
        },
        "base": base_config,
        "overlays": [],  # Empty by default, user can add later
    }

    # Show preview
    console.print(f"\n[{Styles.HEADER}]Preview of simulation config:[/{Styles.HEADER}]")
    preview = _generate_simulation_yaml_preview(sim_config)
    console.print(f"[{Styles.DIM}]{preview}[/{Styles.DIM}]")

    if dry_run:
        console.print(f"\n[{Styles.DIM}]Dry-run mode: config not written[/{Styles.DIM}]")
        return sim_config

    # Confirm and write
    confirm = await questionary.confirm(
        "Write this configuration to config.yml?",
        default=True,
        style=get_questionary_style(),
    ).ask_async()

    if not confirm:
        console.print(f"\n{Messages.info('Configuration not written')}")
        return None

    try:
        backup_path = _write_simulation_config(config_path, sim_config)
        console.print(f"\n  {Messages.success(f'Updated {config_path}')}")
        console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")
        return sim_config
    except Exception as e:
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        console.print(f"\n{Messages.error(f'Failed to write config: {error_msg}')}")
        return None


def _validate_custom_backend_config(backend_config: dict) -> None:
    """Validate custom backend configuration structure.

    Note: This only validates configuration structure, not whether the module
    can be imported. Module validation happens at runtime when the IOC starts,
    allowing backends to be changed without regenerating the IOC.

    Args:
        backend_config: A single backend configuration dict

    Raises:
        click.ClickException: If custom backend config is invalid
    """
    # Check if this is a custom backend (type=custom or has class_name without type)
    is_custom = backend_config.get("type") == "custom"
    has_class_name = backend_config.get("class_name") is not None

    if not is_custom and not has_class_name:
        return  # Built-in backend type, no validation needed

    # For backends with class_name, require module_path
    if has_class_name and not backend_config.get("module_path"):
        raise click.ClickException(
            "Backend with class_name requires 'module_path'.\n"
            "Example: module_path: 'my_project.simulation.pyat_backend'"
        )

    # For type=custom, require both module_path and class_name
    if is_custom:
        if not backend_config.get("module_path"):
            raise click.ClickException(
                "Custom backend requires 'module_path'.\n"
                "Example: module_path: 'my_project.simulation.pyat_backend'"
            )
        if not backend_config.get("class_name"):
            raise click.ClickException(
                "Custom backend requires 'class_name'.\nExample: class_name: 'PyATBackend'"
            )

    # Note: We no longer validate that the module can be imported at generation time.
    # Backend validation happens at runtime, allowing changes without regeneration.


def _load_simulation_config(config_path: str | None = None) -> dict:
    """Load simulation configuration from config.yml.

    Args:
        config_path: Optional path to config file (default: config.yml in cwd)

    Returns:
        Simulation configuration dict

    Raises:
        click.ClickException: If config file not found or missing simulation section
    """
    import yaml

    path = Path(config_path) if config_path else Path.cwd() / "config.yml"

    if not path.exists():
        raise click.ClickException(
            f"Config file not found: {path}\nRun 'osprey init' to create one."
        )

    with open(path) as f:
        config = yaml.safe_load(f)

    if "simulation" not in config:
        raise click.ClickException(
            f"No 'simulation' section in {path}.\nAdd simulation configuration to generate soft IOCs."
        )

    sim_config = config["simulation"]

    # Apply ioc defaults
    ioc_defaults = {
        "name": "soft_ioc",
        "port": 5064,
        "output_dir": "generated_iocs/",
    }
    if "ioc" not in sim_config:
        sim_config["ioc"] = ioc_defaults
    else:
        for k, v in ioc_defaults.items():
            sim_config["ioc"].setdefault(k, v)

    # Parse base + overlays config structure
    # New format: base (single dict) + overlays (list)
    # Default base to mock_style if not specified
    if "base" not in sim_config:
        sim_config["base"] = {"type": "mock_style", "noise_level": 0.01, "update_rate": 10.0}

    base = sim_config["base"]

    # Apply defaults to base backend
    if base.get("type") == "mock_style" or ("type" not in base and "class_name" not in base):
        base.setdefault("type", "mock_style")
        base.setdefault("noise_level", 0.01)
        base.setdefault("update_rate", 10.0)
    if base.get("type") == "custom" or base.get("class_name"):
        base.setdefault("params", {})

    # Validate base backend configuration
    _validate_custom_backend_config(base)

    # Default overlays to empty list if not specified
    if "overlays" not in sim_config:
        sim_config["overlays"] = []

    overlays = sim_config["overlays"]

    # Apply defaults and validate each overlay
    for overlay in overlays:
        if overlay.get("type") == "custom" or overlay.get("class_name"):
            overlay.setdefault("params", {})
        _validate_custom_backend_config(overlay)

    return sim_config


def _load_pairings(pairings_file: str | None) -> dict[str, str]:
    """Load SP->RB pairings from JSON file.

    Args:
        pairings_file: Path to pairings JSON file, or None

    Returns:
        Dict mapping setpoint PV names to readback PV names.
        Empty dict if no file provided.

    Raises:
        click.ClickException: If file not found or invalid format
    """
    if not pairings_file:
        return {}

    import json

    path = Path(pairings_file)
    if not path.exists():
        raise click.ClickException(f"Pairings file not found: {path}")

    with open(path) as f:
        pairings = json.load(f)

    # Pairings file is just a simple dict: {"SP_name": "RB_name", ...}
    if not isinstance(pairings, dict):
        raise click.ClickException(
            f"Invalid pairings file format. Expected a JSON dict, got {type(pairings).__name__}"
        )

    return pairings


def _validate_pairings(pairings: dict[str, str], channels: list[dict]) -> dict[str, str]:
    """Validate pairings against channel database, warn and skip invalid ones.

    Args:
        pairings: Raw pairings dict from JSON file
        channels: List of PV definitions from database

    Returns:
        Validated pairings dict (invalid entries removed)
    """
    # Build set of valid PV names for fast lookup
    valid_pv_names = {ch["name"] for ch in channels}

    validated = {}
    for sp_name, rb_name in pairings.items():
        # Check if SP exists
        if sp_name not in valid_pv_names:
            console.print(
                f"[yellow]Warning:[/yellow] Pairing '{sp_name}' → '{rb_name}' skipped: "
                f"'{sp_name}' not in channel database"
            )
            continue

        # Check if RB exists
        if rb_name not in valid_pv_names:
            console.print(
                f"[yellow]Warning:[/yellow] Pairing '{sp_name}' → '{rb_name}' skipped: "
                f"'{rb_name}' not in channel database"
            )
            continue

        validated[sp_name] = rb_name

    skipped = len(pairings) - len(validated)
    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} invalid pairing(s)[/yellow]")

    return validated


def _load_channels_from_database(db_path: str | Path, db_type: str | None = None) -> list[dict]:
    """Load ALL channels from any Osprey channel database.

    Uses Channel Finder's database classes to support all 4 built-in database types
    with a single code path.

    Args:
        db_path: Path to the database file (string or Path object)
        db_type: Optional database type override. If None, auto-detects.
                 Values: 'flat', 'template', 'hierarchical', 'middle_layer'

    Returns:
        List of PV definitions for caproto generation

    Raises:
        click.ClickException: If database has duplicate PV names
        ValueError: If unknown database type specified
    """
    from osprey.generators.soft_ioc_template import sanitize_pv_name

    # Ensure db_path is a Path object
    db_path = Path(db_path)

    if not db_path.exists():
        raise click.ClickException(f"Channel database not found: {db_path}")

    # Import Channel Finder database classes from native service location
    from osprey.services.channel_finder.databases import (
        FlatChannelDatabase,
        HierarchicalChannelDatabase,
        MiddleLayerDatabase,
        TemplateChannelDatabase,
    )

    # Auto-detect database type if not specified
    if db_type is None:
        db_type = _detect_database_type(db_path)

    # Get database class (4 built-in types only for v1)
    # Note: 'legacy' is an alias for 'flat' for backward compatibility
    database_classes = {
        "flat": FlatChannelDatabase,
        "legacy": FlatChannelDatabase,
        "template": TemplateChannelDatabase,
        "hierarchical": HierarchicalChannelDatabase,
        "middle_layer": MiddleLayerDatabase,
    }

    if db_type not in database_classes:
        available = list(database_classes.keys())
        raise click.ClickException(
            f"Unknown database type: '{db_type}'. Available types: {', '.join(available)}"
        )

    # Load database using the appropriate class
    db_class = database_classes[db_type]
    database = db_class(str(db_path))

    # Get ALL channels using the unified interface
    # Note: Template databases return already-expanded channels
    all_channels = database.get_all_channels()

    # Check for duplicate PV names (error condition)
    seen_names: dict[str, int] = {}  # name -> first occurrence index
    for idx, channel in enumerate(all_channels):
        pv_name = channel.get("channel", channel.get("address", ""))
        if pv_name in seen_names:
            raise click.ClickException(
                f"Duplicate PV name '{pv_name}' found in channel database.\n"
                f"  First occurrence: index {seen_names[pv_name]}\n"
                f"  Duplicate: index {idx}\n"
                f"Please remove duplicates before generating IOC."
            )
        seen_names[pv_name] = idx

    # Convert to PV definitions for caproto
    pvs = []
    for channel in all_channels:
        pv_name = channel.get("channel", channel.get("address", ""))
        desc = channel.get("description", "")

        # Infer PV type and access mode
        is_readonly = _is_readonly_channel(pv_name, desc)
        pv_type = _infer_pv_type_from_channel(pv_name, desc, channel)

        pvs.append(
            {
                "name": pv_name,
                "python_name": sanitize_pv_name(pv_name),  # For Python attribute
                "type": pv_type,
                "description": desc[:100] if desc else "",
                "read_only": is_readonly,
                "address": channel.get("address", pv_name),
                **_get_pv_defaults(pv_type, channel),
            }
        )

    return pvs


def _detect_database_type(db_path: Path) -> str:
    """Auto-detect database type from file structure.

    Detection priority (first match wins):
    1. hierarchical: Has 'hierarchy' + 'tree' or 'hierarchy_definition' + 'tree'
    2. middle_layer: Has nested dicts with 'ChannelNames' keys (MML export format)
    3. template: Has 'channels' array with at least one entry having 'template': true
    4. flat: Default fallback (simple list or 'channels' array without templates)

    Args:
        db_path: Path to database file

    Returns:
        Database type string: 'hierarchical', 'template', 'middle_layer', or 'flat'
    """
    import json

    with open(db_path) as f:
        db = json.load(f)

    # Hierarchical: has 'hierarchy' section with tree
    if "hierarchy" in db and "tree" in db:
        return "hierarchical"

    # Legacy hierarchical format: has 'hierarchy_definition' and 'tree'
    if "hierarchy_definition" in db and "tree" in db:
        return "hierarchical"

    # Middle Layer: has system keys with nested families containing ChannelNames
    # Check this before 'channels' check since MML format is more specific
    if isinstance(db, dict):
        for key, value in db.items():
            if key.startswith("_"):  # Skip metadata keys
                continue
            if isinstance(value, dict):
                for _family_key, family_value in value.items():
                    if isinstance(family_value, dict):
                        # Check for MML structure: families have fields with ChannelNames
                        for _field_key, field_value in family_value.items():
                            if isinstance(field_value, dict) and "ChannelNames" in field_value:
                                return "middle_layer"

    # Template: has 'channels' array with template entries
    if "channels" in db:
        # Check if any entries are templates (have 'template': true)
        has_templates = any(entry.get("template", False) for entry in db["channels"])
        return "template" if has_templates else "flat"

    # Default to flat (handles simple list format)
    return "flat"


def _is_readonly_channel(pv_name: str, description: str) -> bool:
    """Determine if a channel is read-only based on naming conventions.

    Args:
        pv_name: Channel/PV name
        description: Channel description

    Returns:
        True if channel should be read-only
    """
    name_upper = pv_name.upper()
    desc_lower = description.lower() if description else ""

    # Readback indicators (read-only)
    readonly_patterns = ["RB", "READBACK", "MONITOR", "STATUS", "STATE", "IMAGE", "WAVEFORM"]
    if any(pattern in name_upper for pattern in readonly_patterns):
        return True

    # Description-based detection
    if "read-only" in desc_lower or "readback" in desc_lower or "read back" in desc_lower:
        return True

    # Setpoint indicators (read-write)
    writable_patterns = ["SP", "SETPOINT", "SET", "CMD", "COMMAND", "MOTOR"]
    if any(pattern in name_upper for pattern in writable_patterns):
        return False

    # Default: read-only (safer for testing)
    return True


def _infer_pv_type_from_channel(pv_name: str, description: str, channel: dict) -> str:
    """Infer caproto PV type from channel information.

    Args:
        pv_name: Channel/PV name
        description: Channel description
        channel: Full channel dictionary (may contain 'DataType', 'Units', etc.)

    Returns:
        PV type string: 'float', 'int', 'enum', 'string', 'float_array', 'int_array'
    """
    name_upper = pv_name.upper()
    desc_lower = description.lower() if description else ""

    # Check for MML metadata (from middle_layer database)
    if "DataType" in channel:
        data_type = channel["DataType"]
        if "Scalar" in str(data_type):
            return "float"
        if "Integer" in str(data_type):
            return "int"

    # Status and boolean indicators -> enum
    enum_patterns = [
        "STATUS",
        "STATE",
        "READY",
        "FAULT",
        "VALID",
        "ALARM",
        "CONNECTED",
        "INTERLOCK",
        "OPEN",
        "CLOSED",
        "ENABLED",
    ]
    if any(pattern in name_upper for pattern in enum_patterns):
        return "enum"

    # Waveforms/arrays/images
    if "waveform" in desc_lower or "image" in desc_lower or "array" in desc_lower:
        return "float_array"
    if "IMAGE" in name_upper or "WAVEFORM" in name_upper:
        return "float_array"

    # String-like values
    if "name" in desc_lower or "string" in desc_lower or "message" in desc_lower:
        return "string"

    # Measurement values -> float (most common)
    return "float"


def _get_pv_defaults(pv_type: str, channel: dict) -> dict:
    """Get default parameters for PV type, incorporating any channel metadata.

    Args:
        pv_type: PV type string
        channel: Channel dictionary (may contain units, limits, enum_strings, etc.)

    Returns:
        Dict of default parameters for the PV type
    """
    # Extract metadata if present (from MML exports or channel database)
    units = channel.get("Units", channel.get("HWUnits", ""))
    if isinstance(units, list):
        units = units[0] if units else ""

    # Check for custom enum_strings in channel metadata
    custom_enum_strings = channel.get("enum_strings")

    defaults = {
        "float": {
            "units": str(units),
            "precision": 4,
            "high_alarm": 100.0,
            "low_alarm": 0.0,
        },
        "int": {
            "units": str(units),
        },
        "enum": {
            # Use custom enum_strings from channel metadata if provided,
            # otherwise default to binary states
            "enum_strings": custom_enum_strings or ["Off", "On"],
        },
        "string": {
            "max_length": 256,
        },
        "float_array": {
            "count": 128,
        },
        "int_array": {
            "count": 128,
        },
    }
    return defaults.get(pv_type, {})


def _get_output_path(sim_config: dict) -> Path:
    """Determine output file path from config.

    Args:
        sim_config: Simulation configuration dict

    Returns:
        Path object for output file
    """
    output_dir = Path(sim_config["ioc"]["output_dir"])
    ioc_name = sim_config["ioc"]["name"]
    return output_dir / f"{ioc_name}_ioc.py"


def _write_ioc_file(output_path: Path, ioc_code: str) -> None:
    """Write generated IOC code to file with backup.

    Creates output directory if needed. Backs up existing file.

    Args:
        output_path: Path to output file
        ioc_code: Generated Python code
    """
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file if present
    if output_path.exists():
        backup_path = output_path.with_suffix(".py.bak")
        backup_path.write_text(output_path.read_text())
        console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")

    # Write new file
    output_path.write_text(ioc_code, encoding="utf-8")
    output_path.chmod(0o755)  # Make executable
    console.print(f"  {Messages.success(f'Generated: {output_path}')}")


def _show_dry_run_summary(sim_config: dict, channels: list[dict], pairings: dict[str, str]) -> None:
    """Display dry-run summary without writing files.

    Args:
        sim_config: Simulation configuration
        channels: List of PV definitions
        pairings: Validated SP->RB pairings
    """
    from rich.table import Table

    console.print(f"\n[{Styles.HEADER}]Dry Run Summary[/{Styles.HEADER}]\n")

    # Configuration table
    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("IOC Name", sim_config["ioc"]["name"])
    table.add_row("Output", str(_get_output_path(sim_config)))
    table.add_row("PV Count", str(len(channels)))
    table.add_row("SP/RB Pairings", str(len(pairings)))
    # Get backend info from base + overlays
    base = sim_config.get("base", {"type": "mock_style"})
    overlays = sim_config.get("overlays", [])
    base_type = base.get("type") or base.get("class_name", "custom")
    backend_parts = [base_type]
    if base.get("noise_level") is not None:
        backend_parts.append(f"noise={base['noise_level']}")
    if base.get("update_rate") is not None:
        backend_parts.append(f"rate={base['update_rate']} Hz")
    base_info = (
        f"{backend_parts[0]} ({', '.join(backend_parts[1:])})"
        if len(backend_parts) > 1
        else backend_parts[0]
    )
    table.add_row("Base", base_info)
    if overlays:
        overlay_names = [o.get("class_name") or o.get("type", "unknown") for o in overlays]
        table.add_row("Overlays", ", ".join(overlay_names))
    table.add_row("Port", str(sim_config["ioc"]["port"]))

    console.print(table)

    # Sample PVs
    console.print(f"\n[{Styles.HEADER}]Sample PVs:[/{Styles.HEADER}]")
    for pv in channels[:5]:
        access = "read-only" if pv.get("read_only") else "writable"
        console.print(f"  - {pv['name']} ({pv['type']}, {access})")
    if len(channels) > 5:
        console.print(f"  [{Styles.DIM}]... and {len(channels) - 5} more[/{Styles.DIM}]")

    console.print(f"\n[{Styles.DIM}]No files written (dry-run mode)[/{Styles.DIM}]")


async def _offer_control_system_config_update(ioc_name: str, port: int):
    """Offer to update config.yml to connect to soft IOC.

    Args:
        ioc_name: Name of the IOC
        port: EPICS CA port
    """
    try:
        import questionary
        import yaml

        from .styles import get_questionary_style

        # Find config file
        config_path = Path.cwd() / "config.yml"
        if not config_path.exists():
            _print_manual_config_instructions(port)
            return

        # Ask user with styled prompt
        console.print()
        console.print(f"[{Styles.HEADER}]Config Integration:[/{Styles.HEADER}]")
        console.print(f"  Found config: [{Styles.VALUE}]{config_path}[/{Styles.VALUE}]")
        console.print()

        update_config = await questionary.confirm(
            "Update config.yml to connect to soft IOC for testing?",
            default=True,
            style=get_questionary_style(),
        ).ask_async()

        if not update_config:
            console.print("\n  " + Messages.info("Skipped. You can configure manually if needed."))
            _print_manual_config_instructions(port)
            return

        # Load config to check structure
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check if control_system section exists
        if "control_system" not in config:
            console.print(
                f"\n[{Styles.WARNING}]⚠️  No 'control_system' section in config.yml[/{Styles.WARNING}]"
            )
            _print_manual_config_instructions(port)
            return

        # Show preview of changes
        console.print(f"\n[{Styles.HEADER}]Preview of changes:[/{Styles.HEADER}]")
        console.print("  control_system:")
        console.print("    type: mock → epics")
        console.print("    connector.epics.gateways.read_only:")
        console.print("      address: ... → localhost")
        console.print(f"      port: ... → {port}")

        # Confirm
        confirm = await questionary.confirm(
            "Apply these changes to config.yml?",
            default=True,
            style=get_questionary_style(),
        ).ask_async()

        if not confirm:
            console.print(
                "\n  " + Messages.info("Changes not applied. Configure manually if needed.")
            )
            return

        # Use comment-preserving YAML update
        from osprey.generators.config_updater import update_yaml_file

        backup_path = update_yaml_file(
            config_path,
            {
                "control_system.type": "epics",
                "control_system.connector.epics.gateways.read_only.address": "localhost",
                "control_system.connector.epics.gateways.read_only.port": port,
            },
            create_backup=True,
        )

        console.print(f"\n  {Messages.success(f'Updated {config_path}')}")
        console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")
        console.print()
        console.print("  " + Messages.info(f"To restore: cp {backup_path} {config_path}"))

    except ImportError:
        # questionary not available
        _print_manual_config_instructions(port)
    except Exception as e:
        # Escape the error message to prevent Rich markup interpretation
        error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
        console.print(
            f"\n[{Styles.WARNING}]⚠️  Could not update config: {error_msg}[/{Styles.WARNING}]"
        )
        _print_manual_config_instructions(port)


def _print_manual_config_instructions(port: int):
    """Print manual configuration instructions.

    Args:
        port: EPICS CA port
    """
    console.print(f"\n[{Styles.HEADER}]Manual Configuration:[/{Styles.HEADER}]")
    console.print("  To connect to the soft IOC, update config.yml:")
    console.print()
    console.print("    control_system:")
    console.print("      type: epics")
    console.print("      connector:")
    console.print("        epics:")
    console.print("          gateways:")
    console.print("            read_only:")
    console.print("              address: localhost")
    console.print(f"              port: {port}")
    console.print()


@generate.command(name="soft-ioc")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Config file path (default: config.yml)",
)
@click.option("--output", "-o", "output_file", help="Override output file path")
@click.option("--dry-run", is_flag=True, help="Preview without writing files")
@click.option(
    "--init",
    is_flag=True,
    help="Force interactive setup (overwrites existing simulation config if confirmed)",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Limit to first N PVs from database (useful for large databases)",
)
@click.option(
    "--filter-pattern",
    type=str,
    default=None,
    help="Regex pattern to filter PV names (e.g., 'QUAD.*' or '.*:SP$')",
)
@click.option(
    "--pv-count-warning-threshold",
    type=click.IntRange(min=1),
    default=1000,
    help="Warn if PV count exceeds this threshold (default: 1000)",
)
def soft_ioc(
    config_path: str | None,
    output_file: str | None,
    dry_run: bool,
    init: bool,
    limit: int | None,
    filter_pattern: str | None,
    pv_count_warning_threshold: int,
):
    """Generate Python soft IOC for EPICS testing.

    Creates a pure Python EPICS soft IOC using caproto. All settings are read
    from the 'simulation' section in config.yml.

    If no simulation section exists, or if --init is passed, an interactive
    setup wizard will help create the configuration.

    Large PV Databases:

    \b
      For large databases (10^4+ PVs), consider using --limit or --filter-pattern
      to generate a subset IOC for testing. The tool will warn you if the PV count
      exceeds the threshold (default: 1000).

    Examples:

    \b
      # Generate using config.yml settings
      $ osprey generate soft-ioc

      # Interactive setup for new simulation config
      $ osprey generate soft-ioc --init

      # Preview setup without writing config
      $ osprey generate soft-ioc --init --dry-run

      # Preview generation without writing IOC file
      $ osprey generate soft-ioc --dry-run

      # Override output location
      $ osprey generate soft-ioc --output /tmp/test_ioc.py

      # Limit to first 500 PVs (useful for large databases)
      $ osprey generate soft-ioc --limit 500

      # Filter PVs by regex pattern
      $ osprey generate soft-ioc --filter-pattern "QUAD.*"

      # Combine limit and filter (filter first, then limit)
      $ osprey generate soft-ioc --filter-pattern ".*:SP$" --limit 100

    Setup (manual):

    \b
      1. Add simulation section to config.yml
      2. Create pairings.json (optional, for SP/RB tracking)
      3. Generate: osprey generate soft-ioc
      4. Install caproto: pip install caproto numpy
      5. Run: python generated_iocs/<ioc_name>_ioc.py
    """
    import yaml

    from osprey.generators.soft_ioc_template import generate_soft_ioc

    console.print(f"\n[{Styles.HEADER}]🔧 Generating Soft IOC[/{Styles.HEADER}]\n")

    try:
        # Determine config file path
        cfg_path = Path(config_path) if config_path else Path.cwd() / "config.yml"

        # Check if config file exists
        if not cfg_path.exists():
            console.print(f"{Messages.error(f'Config file not found: {cfg_path}')}")
            console.print()
            console.print(
                "  Run [accent]osprey init[/accent] to create a project, then run this command."
            )
            console.print()
            raise click.Abort()

        # Check if simulation section exists
        with open(cfg_path, encoding="utf-8") as f:
            existing_config = yaml.safe_load(f) or {}
        has_simulation = "simulation" in existing_config

        # Handle --init flag or missing simulation section
        sim_config = None
        if init:
            # --init flag: Force interactive setup
            if has_simulation and not dry_run:
                console.print(
                    f"[{Styles.WARNING}]Existing simulation config found. "
                    f"Interactive setup will overwrite it if confirmed.[/{Styles.WARNING}]"
                )

            sim_config = asyncio.run(
                _offer_simulation_config_setup(cfg_path, force_init=True, dry_run=dry_run)
            )
            if sim_config is None:
                # User cancelled or setup failed
                raise click.Abort()

            if dry_run:
                # In --init --dry-run mode, we only show the config preview
                console.print(f"\n[{Styles.DIM}]Dry-run mode: no files written[/{Styles.DIM}]")
                return

        elif not has_simulation:
            # No simulation section - offer to create one
            console.print(
                f"[{Styles.WARNING}]No 'simulation' section found in {cfg_path}[/{Styles.WARNING}]"
            )
            console.print()

            try:
                import questionary

                from .styles import get_questionary_style

                create_config = questionary.confirm(
                    "Would you like to create a simulation configuration interactively?",
                    default=True,
                    style=get_questionary_style(),
                ).ask()

                if create_config:
                    sim_config = asyncio.run(
                        _offer_simulation_config_setup(cfg_path, force_init=False, dry_run=dry_run)
                    )
                    if sim_config is None:
                        raise click.Abort()

                    if dry_run:
                        console.print(
                            f"\n[{Styles.DIM}]Dry-run mode: no files written[/{Styles.DIM}]"
                        )
                        return
                else:
                    console.print()
                    console.print(
                        f"[{Styles.HEADER}]To add manually, add this to {cfg_path}:[/{Styles.HEADER}]"
                    )
                    console.print()
                    console.print(f"[{Styles.DIM}]simulation:")
                    console.print('  channel_database: "path/to/channel_database.json"')
                    console.print("  ioc:")
                    console.print('    name: "my_ioc"')
                    console.print("    port: 5064")
                    console.print('    output_dir: "generated_iocs/"')
                    console.print("  base:")
                    console.print('    type: "mock_style"')
                    console.print(f"    noise_level: 0.01[/{Styles.DIM}]")
                    console.print()
                    raise click.Abort()
            except ImportError:
                console.print("  Add a 'simulation' section to config.yml to generate soft IOCs.")
                console.print("  See: osprey generate soft-ioc --help")
                console.print()
                raise click.Abort() from None

        # Load configuration (either from setup or from file)
        if sim_config is None:
            with console.status("[dim]Loading configuration...[/dim]"):
                sim_config = _load_simulation_config(str(cfg_path))

        console.print(
            f"  [{Styles.LABEL}]IOC Name:[/{Styles.LABEL}] "
            f"[{Styles.VALUE}]{sim_config['ioc']['name']}[/{Styles.VALUE}]"
        )
        db_path = sim_config.get("channel_database")
        db_display = db_path if db_path else "none (empty IOC)"
        console.print(
            f"  [{Styles.LABEL}]Database:[/{Styles.LABEL}] [{Styles.VALUE}]{db_display}[/{Styles.VALUE}]"
        )
        # Display backend info (base + overlays)
        base = sim_config.get("base", {"type": "mock_style"})
        overlays = sim_config.get("overlays", [])
        base_info = base.get("type") or base.get("class_name", "custom")
        console.print(
            f"  [{Styles.LABEL}]Base:[/{Styles.LABEL}] [{Styles.VALUE}]{base_info}[/{Styles.VALUE}]"
        )
        if overlays:
            overlay_names = [o.get("class_name") or o.get("type", "unknown") for o in overlays]
            console.print(
                f"  [{Styles.LABEL}]Overlays:[/{Styles.LABEL}] "
                f"[{Styles.VALUE}]{', '.join(overlay_names)}[/{Styles.VALUE}]"
            )
        console.print()

        # Load channels from database (or use empty list for empty IOC)
        if db_path:
            with console.status("[dim]Loading channel database...[/dim]"):
                channels = _load_channels_from_database(
                    db_path,
                    db_type=sim_config.get("channel_database_type"),
                )

            original_count = len(channels)
            msg = f"Loaded {original_count} channels from database"
            console.print(f"  {Messages.success(msg)}")

            # Apply filter pattern if specified
            if filter_pattern:
                try:
                    pattern = re.compile(filter_pattern)
                    channels = [ch for ch in channels if pattern.search(ch["name"])]
                    msg = f"Applied filter pattern: {filter_pattern}"
                    console.print(f"  {Messages.info(msg)}")
                    msg = f"Filtered to {len(channels)} channels (from {original_count})"
                    console.print(f"  {Messages.info(msg)}")
                except re.error as e:
                    msg = f"Invalid regex pattern: {e}"
                    console.print(f"\n{Messages.error(msg)}")
                    raise click.Abort() from e

            # Apply limit if specified
            if limit is not None:
                pre_limit_count = len(channels)
                channels = channels[:limit]
                if pre_limit_count > limit:
                    msg = f"Limited to first {limit} channels (from {pre_limit_count})"
                    console.print(f"  {Messages.info(msg)}")

            # Warn if PV count is large (after filtering/limiting)
            if len(channels) > pv_count_warning_threshold:
                w = Styles.WARNING
                d = Styles.DIM
                c = Styles.COMMAND
                count = len(channels)
                console.print()
                console.print(f"  [{w}]⚠️  Warning: Large PV database detected![/{w}]")
                console.print(f"  [{w}]Generating IOC with {count} PVs may take a while.[/{w}]")
                console.print(
                    f"  [{w}]The generated IOC may suffer"
                    f" performance issues on limited"
                    f" resources.[/{w}]"
                )
                console.print()
                console.print(
                    f"  [{d}]Consider using --limit or --filter-pattern to reduce PV count:[/{d}]"
                )
                console.print(f"    [{c}]osprey generate soft-ioc --limit 500[/{c}]")
                console.print(f'    [{c}]osprey generate soft-ioc --filter-pattern "QUAD.*"[/{c}]')
                console.print()
        else:
            # No channel database - confirm empty IOC creation
            try:
                import questionary

                from .styles import get_questionary_style

                create_empty = questionary.confirm(
                    "No channel database configured. Create empty IOC (heartbeat only)?",
                    default=True,
                    style=get_questionary_style(),
                ).ask()

                if not create_empty:
                    raise click.Abort()
            except ImportError:
                console.print(f"\n{Messages.error('No channel database configured')}")
                raise click.Abort() from None

            channels = []
            console.print(f"  {Messages.info('Creating empty IOC (heartbeat only)')}")

        # Load and validate pairings (warns and skips invalid entries)
        raw_pairings = _load_pairings(sim_config.get("pairings_file"))
        pairings = _validate_pairings(raw_pairings, channels)

        if pairings:
            console.print(f"  {Messages.success(f'Loaded {len(pairings)} SP/RB pairings')}")

        if dry_run:
            _show_dry_run_summary(sim_config, channels, pairings)
            return

        # Generate IOC code
        with console.status("[dim]Generating IOC code...[/dim]"):
            ioc_code = generate_soft_ioc(
                config=sim_config,
                channels=channels,
                pairings=pairings,
            )

        console.print(f"  {Messages.success('Generated IOC code')}")
        console.print()

        # Write output file (with backup if exists)
        output_path = Path(output_file) if output_file else _get_output_path(sim_config)
        _write_ioc_file(output_path, ioc_code)

        # Success summary
        console.print()
        console.print("=" * 60)
        console.print(
            f"[{Styles.BOLD_SUCCESS}]✅ Soft IOC Generated Successfully[/{Styles.BOLD_SUCCESS}]"
        )
        console.print("=" * 60)
        console.print()

        console.print(f"[{Styles.HEADER}]Next Steps:[/{Styles.HEADER}]")
        console.print("  1. Install dependencies:")
        console.print(f"     [{Styles.COMMAND}]pip install caproto numpy[/{Styles.COMMAND}]")
        console.print()
        console.print("  2. Start the IOC:")
        console.print(f"     [{Styles.COMMAND}]python {output_path}[/{Styles.COMMAND}]")
        console.print()
        console.print("  3. Test with:")
        console.print(f"     [{Styles.COMMAND}]caget SIM:HEARTBEAT[/{Styles.COMMAND}]")
        console.print()

        # Offer config integration
        asyncio.run(
            _offer_control_system_config_update(
                ioc_name=sim_config["ioc"]["name"], port=sim_config["ioc"]["port"]
            )
        )

    except click.ClickException:
        raise
    except KeyboardInterrupt:
        console.print(f"\n{Messages.warning('Generation cancelled by user')}")
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n{Messages.error(f'Generation failed: {e}')}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort() from e


if __name__ == "__main__":
    generate()
