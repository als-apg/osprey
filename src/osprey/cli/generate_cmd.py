"""Code generation commands for Osprey Framework.

This module provides the 'osprey generate' command group for creating
Osprey components from various sources.
"""

import asyncio
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
    with quiet_logger(["REGISTRY", "CONFIG"]):
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
    "--provider", default=None, help="LLM provider override (e.g., anthropic, openai, cborg)"
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
    output_path.write_text(code)

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
                "\n  "
                + Messages.info("Changes not applied. " "Add manually to config.yml if needed.")
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
            backup_path.write_text(registry_path.read_text())

            # Write new content
            registry_path.write_text(new_content)

            console.print(f"\n  {Messages.success(f'Updated {registry_path}')}")
            console.print(f"  [{Styles.DIM}]Backup saved to: {backup_path}[/{Styles.DIM}]")
            console.print()
            console.print(
                "  " + Messages.info("Capability is now registered! " "Test with: osprey chat")
            )
        else:
            console.print("\n  " + Messages.info("Changes not applied. " "Add manually if needed."))

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
    output_path.write_text(code)

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
        console.print(f"    {Messages.command(f'osprey generate claude-config --force')}")
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
                pass
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
        console.print(f"  3. Set API key in [accent].env[/accent]:")
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


if __name__ == "__main__":
    generate()
