"""
Command Category Implementations for Osprey Framework

This module provides the implementation of all built-in command categories for the
centralized slash command system. Each category contains specialized command handlers
that integrate with specific framework subsystems and provide rich user experiences
across different interface types.

Command Categories:
    - CLI Commands: Interface control and user experience (help, clear, exit)
    - Agent Control Commands: Agent behavior and execution control (planning, approval)
    - Service Commands: Framework service management (status, logs, metrics)

The category system enables organized command discovery, context-aware execution,
and extensible patterns for application-specific command extensions. Each category
provides specialized handlers with appropriate error handling and user feedback.
"""

from typing import Any

from rich.panel import Panel
from rich.table import Table

from osprey.cli.styles import Styles
from osprey.cli.styles import console as themed_console

from .types import Command, CommandCategory, CommandContext, CommandExecutionError, CommandResult


def register_cli_commands(registry) -> None:
    """Register CLI interface commands for user experience and interface control.

    Registers essential CLI commands that provide user interface control, help systems,
    and session management. These commands are designed for interactive terminal
    interfaces and provide rich formatted output using the Rich library.

    Registered Commands:
        /help [command]: Display available commands or detailed help for specific command
        /exit, /quit: Exit the current CLI session
        /clear: Clear the terminal screen

    :param registry: Command registry instance for command registration
    :type registry: CommandRegistry

    .. note::
       CLI commands are interface-specific and may not be available in
       non-interactive contexts like OpenWebUI or API interfaces.

    Examples:
        Command usage in CLI::

            /help              # Show all available commands
            /help planning     # Show detailed help for planning command
            /clear             # Clear terminal screen
            /exit              # Exit CLI session
    """

    def help_handler(args: str, context: CommandContext) -> CommandResult:
        """Show available commands or help for specific command."""
        console = context.console or themed_console

        if args.strip():
            # Show help for specific command
            cmd = registry.get_command(args.strip())
            if cmd:
                panel_content = f"[bold]/{cmd.name}[/bold]\n\n"
                panel_content += f"Category: {cmd.category.value}\n"
                panel_content += f"Syntax: {cmd.syntax}\n\n"
                panel_content += cmd.help_text

                if cmd.aliases:
                    panel_content += (
                        f"\n\nAliases: {', '.join([f'/{alias}' for alias in cmd.aliases])}"
                    )

                panel = Panel(
                    panel_content, title="Command Help", border_style=Styles.BORDER_ACCENT
                )
                console.print(panel)
            else:
                console.print(f"❌ Unknown command: /{args.strip()}", style=Styles.ERROR)
        else:
            # Show all commands organized by category
            commands_by_category = {}
            for cmd in registry.get_all_commands():
                if cmd.is_valid_for_interface(context.interface_type):
                    if cmd.category not in commands_by_category:
                        commands_by_category[cmd.category] = []
                    commands_by_category[cmd.category].append(cmd)

            for category, commands in commands_by_category.items():
                table = Table(title=f"{category.value.title()} Commands", show_header=True)
                table.add_column("Command", style=Styles.ACCENT, width=20)
                table.add_column("Description", style=Styles.PRIMARY)

                for cmd in sorted(commands, key=lambda x: x.name):
                    aliases_text = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                    table.add_row(f"/{cmd.name}{aliases_text}", cmd.description)

                console.print(table)
                console.print()

            console.print("💡 Use /help <command> for detailed help", style=Styles.DIM)

        return CommandResult.HANDLED

    def clear_handler(args: str, context: CommandContext) -> CommandResult:
        """Clear the screen."""
        from prompt_toolkit.shortcuts import clear

        clear()
        return CommandResult.HANDLED

    def exit_handler(args: str, context: CommandContext) -> CommandResult:
        """Exit the CLI."""
        console = context.console or themed_console
        console.print("👋 Goodbye!", style=Styles.WARNING)
        return CommandResult.EXIT

    def config_handler(args: str, context: CommandContext) -> CommandResult:
        """Show current configuration."""
        console = context.console or themed_console

        if context.config:
            # Handle both direct config and LangGraph base_config structure
            config = context.config.get("configurable", context.config)

            # SESSION INFORMATION
            session_info = []
            if "thread_id" in config:
                thread_display = (
                    config["thread_id"][:16] + "..."
                    if len(config["thread_id"]) > 16
                    else config["thread_id"]
                )
                session_info.append(f"Thread ID: {thread_display}")
            if "session_id" in config and config["session_id"]:
                session_display = (
                    str(config["session_id"])[:16] + "..."
                    if len(str(config["session_id"])) > 16
                    else str(config["session_id"])
                )
                session_info.append(f"Session ID: {session_display}")
            if "interface_context" in config:
                session_info.append(f"Interface: {config['interface_context']}")
            if "user_id" in config and config["user_id"]:
                session_info.append(f"User ID: {config['user_id']}")
            if "chat_id" in config and config["chat_id"]:
                session_info.append(f"Chat ID: {config['chat_id']}")

            # MODEL CONFIGURATION
            model_info = []
            if "model_configs" in config:
                models = config["model_configs"]
                for role, model_config in models.items():
                    model_name = model_config.get("model_id", "Unknown")
                    provider = model_config.get("provider", "Unknown")
                    max_tokens = model_config.get("max_tokens", "N/A")
                    model_info.append(f"  {role}:")
                    model_info.append(f"    Model: {model_name}")
                    model_info.append(f"    Provider: {provider}")
                    model_info.append(f"    Max Tokens: {max_tokens}")

            # PROVIDER CONFIGURATION
            provider_info = []
            if "provider_configs" in config:
                providers = config["provider_configs"]
                for name, provider_config in providers.items():
                    base_url = provider_config.get("base_url", "N/A")
                    timeout = provider_config.get("timeout", "N/A")
                    provider_info.append(f"  {name}: {base_url} (timeout: {timeout}s)")

            # EXECUTION LIMITS
            execution_info = []
            if "execution_limits" in config:
                limits = config["execution_limits"]
                execution_info.append(f"Max Steps: {limits.get('max_steps', 'N/A')}")
                execution_info.append(
                    f"Max Reclassifications: {limits.get('max_reclassifications', 'N/A')}"
                )
                execution_info.append(
                    f"Max Planning Attempts: {limits.get('max_planning_attempts', 'N/A')}"
                )
                execution_info.append(f"Max Step Retries: {limits.get('max_step_retries', 'N/A')}")
                execution_info.append(
                    f"Max Execution Time: {limits.get('max_execution_time_seconds', 'N/A')}s"
                )
                execution_info.append(
                    f"Max Concurrent Classifications: {limits.get('max_concurrent_classifications', 'N/A')}"
                )

            # Check recursion_limit in base config
            if "recursion_limit" in context.config:
                execution_info.append(f"Graph Recursion Limit: {context.config['recursion_limit']}")

            # AGENT CONTROL
            agent_info = []
            if "agent_control_defaults" in config:
                agent_control = config["agent_control_defaults"]

                # Planning and bypass modes
                planning = agent_control.get("planning_mode_enabled", False)
                agent_info.append(f"Planning Mode: {'✅ Enabled' if planning else '❌ Disabled'}")

                task_bypass = agent_control.get("task_extraction_bypass_enabled", False)
                agent_info.append(
                    f"Task Extraction Bypass: {'✅ Enabled' if task_bypass else '❌ Disabled'}"
                )

                caps_bypass = agent_control.get("capability_selection_bypass_enabled", False)
                agent_info.append(
                    f"Capability Selection Bypass: {'✅ Enabled' if caps_bypass else '❌ Disabled'}"
                )

                # EPICS control
                epics_writes = agent_control.get("epics_writes_enabled", False)
                agent_info.append(
                    f"EPICS Writes: {'✅ Enabled' if epics_writes else '❌ Disabled'}"
                )

                # Approval settings
                approval_global = agent_control.get("approval_global_mode", "N/A")
                agent_info.append(f"Approval Global Mode: {approval_global}")

                python_approval = agent_control.get("python_execution_approval_enabled", False)
                agent_info.append(
                    f"Python Approval: {'✅ Enabled' if python_approval else '❌ Disabled'}"
                )

                python_approval_mode = agent_control.get("python_execution_approval_mode", "N/A")
                agent_info.append(f"Python Approval Mode: {python_approval_mode}")

                memory_approval = agent_control.get("memory_approval_enabled", False)
                agent_info.append(
                    f"Memory Approval: {'✅ Enabled' if memory_approval else '❌ Disabled'}"
                )

            # PYTHON EXECUTOR
            python_info = []
            if "python_executor" in config:
                py_config = config["python_executor"]
                if py_config:
                    jupyter_url = py_config.get("jupyter_url", "N/A")
                    python_info.append(f"Jupyter URL: {jupyter_url}")
                    execution_mode = py_config.get("execution_mode", "N/A")
                    python_info.append(f"Execution Mode: {execution_mode}")

            # SERVICES
            service_info = []
            if "service_configs" in config:
                services = config["service_configs"]
                for service_name in list(services.keys())[:5]:  # Show first 5 services
                    service_info.append(f"  {service_name}")

            # DEVELOPMENT SETTINGS
            dev_info = []
            if "development" in config:
                dev = config["development"]
                if dev:
                    debug = dev.get("debug", False)
                    dev_info.append(f"Debug Mode: {'✅ Enabled' if debug else '❌ Disabled'}")

                    prompts = dev.get("prompts", {})
                    if prompts:
                        print_all = prompts.get("print_all", False)
                        dev_info.append(
                            f"Print Prompts: {'✅ Enabled' if print_all else '❌ Disabled'}"
                        )

                    raise_raw = dev.get("raise_raw_errors", False)
                    dev_info.append(
                        f"Raise Raw Errors: {'✅ Enabled' if raise_raw else '❌ Disabled'}"
                    )

            # PROJECT INFO
            project_info = []
            if "project_root" in config and config["project_root"]:
                project_info.append(f"Root: {config['project_root']}")
            if "current_application" in config and config["current_application"]:
                project_info.append(f"Application: {config['current_application']}")
            if "registry_path" in config and config["registry_path"]:
                project_info.append(f"Registry: {config['registry_path']}")

            # Build output
            output_parts = []

            if session_info:
                output_parts.append("[bold]Session:[/bold]\n" + "\n".join(session_info))

            if model_info:
                output_parts.append("[bold]Models:[/bold]\n" + "\n".join(model_info))

            if provider_info:
                output_parts.append("[bold]Providers:[/bold]\n" + "\n".join(provider_info))

            if execution_info:
                output_parts.append("[bold]Execution Limits:[/bold]\n" + "\n".join(execution_info))

            if agent_info:
                output_parts.append("[bold]Agent Control:[/bold]\n" + "\n".join(agent_info))

            if python_info:
                output_parts.append("[bold]Python Executor:[/bold]\n" + "\n".join(python_info))

            if service_info:
                output_parts.append("[bold]Services:[/bold]\n" + "\n".join(service_info))

            if dev_info:
                output_parts.append("[bold]Development:[/bold]\n" + "\n".join(dev_info))

            if project_info:
                output_parts.append("[bold]Project:[/bold]\n" + "\n".join(project_info))

            if output_parts:
                panel = Panel(
                    "\n\n".join(output_parts),
                    title="Framework Configuration",
                    border_style=Styles.SUCCESS,
                    padding=(1, 2),
                )
                console.print(panel)
            else:
                console.print(
                    "📋 Configuration loaded but no details available", style=Styles.WARNING
                )
        else:
            console.print("❌ No configuration available", style=Styles.ERROR)

        return CommandResult.HANDLED

    def status_handler(args: str, context: CommandContext) -> CommandResult:
        """Show comprehensive system status using Osprey health check."""
        console = context.console or themed_console

        try:
            # Import and run the health checker with full diagnostics
            from osprey.cli.health_cmd import HealthChecker

            console.print("🔍 Running comprehensive system health check...", style=Styles.INFO)
            console.print()

            # Create health checker with full diagnostics enabled
            checker = HealthChecker(verbose=True, full=True)

            # Run all health checks (this already displays the results)
            checker.check_all()

            # Add session-specific information
            console.print()
            session_info = []

            if context.session_id:
                session_display = (
                    context.session_id[:8] + "..."
                    if len(context.session_id) > 8
                    else context.session_id
                )
                session_info.append(f"Session ID: {session_display}")

            if context.gateway:
                session_info.append("Gateway: ✅ Connected")
            else:
                session_info.append("Gateway: ❌ Not connected")

            if context.agent_state:
                # Show more detailed agent state information
                state_info = "Agent State: ✅ Available"

                # Add some key state details if available
                try:
                    if isinstance(context.agent_state, dict):
                        # Count messages if available
                        if "messages" in context.agent_state:
                            msg_count = len(context.agent_state["messages"])
                            state_info += f" ({msg_count} messages)"

                        # Show execution status if available
                        if "execution_step_results" in context.agent_state:
                            step_count = len(context.agent_state["execution_step_results"])
                            if step_count > 0:
                                state_info += f", {step_count} execution steps"
                except Exception:
                    pass  # Keep basic info if detailed parsing fails

                session_info.append(state_info)
            else:
                session_info.append("Agent State: ❌ Not available")

            if session_info:
                panel = Panel(
                    "\n".join(session_info),
                    title=f"Current Session ({context.interface_type})",
                    border_style=Styles.BORDER_DIM,
                )
                console.print(panel)

        except Exception as e:
            console.print(f"❌ Error running health check: {e}", style=Styles.ERROR)
            console.print("💡 Try running 'osprey health --full' directly", style=Styles.DIM)

        return CommandResult.HANDLED

    # Register CLI commands
    registry.register(
        Command(
            name="help",
            category=CommandCategory.CLI,
            description="Show available commands or help for a specific command",
            handler=help_handler,
            aliases=["h", "?"],
            help_text="Show available commands or help for a specific command.\n\nUsage:\n  /help          - Show all commands\n  /help <cmd>    - Show help for specific command",
            interface_restrictions=["cli"],
        )
    )

    registry.register(
        Command(
            name="clear",
            category=CommandCategory.CLI,
            description="Clear the terminal screen",
            handler=clear_handler,
            aliases=["cls", "c"],
            help_text="Clear the terminal screen.",
            interface_restrictions=["cli"],
        )
    )

    registry.register(
        Command(
            name="exit",
            category=CommandCategory.CLI,
            description="Exit the CLI interface",
            handler=exit_handler,
            aliases=["quit", "bye", "q"],
            help_text="Exit the CLI interface.",
            interface_restrictions=["cli"],
        )
    )

    registry.register(
        Command(
            name="config",
            category=CommandCategory.CLI,
            description="Show current framework configuration",
            handler=config_handler,
            help_text="Display the current framework configuration including LLM settings and capabilities.",
        )
    )

    registry.register(
        Command(
            name="status",
            category=CommandCategory.CLI,
            description="Run comprehensive system health check and show status",
            handler=status_handler,
            help_text="Run a full framework health check including configuration validation, API connectivity, container status, and session information. Equivalent to 'osprey health --full'.",
        )
    )


def register_agent_control_commands(registry) -> None:
    """Register agent control commands for behavior and execution management.

    Registers commands that control agent execution behavior, planning modes,
    approval workflows, and performance optimizations. These commands modify
    the agent control state and affect how the framework processes requests
    and executes capabilities.

    Registered Commands:
        /planning:on|off: Enable or disable planning mode for execution coordination
        /approval:enabled|disabled|selective: Control human approval workflows
        /task:on|off: Control task extraction bypass for performance
        /caps:on|off: Control capability selection bypass for performance

    :param registry: Command registry instance for command registration
    :type registry: CommandRegistry

    .. note::
       Agent control commands return state change dictionaries instead of
       CommandResult values. These changes are applied to the agent control state.

    .. warning::
       Agent control changes affect framework behavior and should be used
       carefully in production environments.

    Examples:
        Agent control usage::

            /planning:on           # Enable planning mode
            /approval:selective    # Enable selective approval
            /task:off             # Bypass task extraction for performance
            /caps:off             # Bypass capability selection for performance
    """

    def planning_handler(args: str, context: CommandContext) -> dict[str, Any]:
        """Control planning mode."""
        if args in ["on", "enabled", "true"] or args == "":
            return {"planning_mode_enabled": True}
        elif args in ["off", "disabled", "false"]:
            return {"planning_mode_enabled": False}
        else:
            raise CommandExecutionError(
                f"Invalid option '{args}' for /planning", "planning", "Use 'on' or 'off'"
            )

    def approval_handler(args: str, context: CommandContext) -> dict[str, Any]:
        """Control approval workflows."""
        if args in ["on", "enabled", "true"] or args == "":
            return {"approval_mode": "enabled"}
        elif args in ["off", "disabled", "false"]:
            return {"approval_mode": "disabled"}
        elif args == "selective":
            return {"approval_mode": "selective"}
        else:
            raise CommandExecutionError(
                f"Invalid option '{args}' for /approval",
                "approval",
                "Use 'on', 'off', or 'selective'",
            )

    def task_handler(args: str, context: CommandContext) -> dict[str, Any]:
        """Control task extraction bypass."""
        if args in ["off", "disabled", "false"]:
            return {"task_extraction_bypass_enabled": True}
        elif args in ["on", "enabled", "true"]:
            return {"task_extraction_bypass_enabled": False}
        else:
            raise CommandExecutionError(
                f"Invalid option '{args}' for /task", "task", "Use 'on' or 'off'"
            )

    def caps_handler(args: str, context: CommandContext) -> dict[str, Any]:
        """Control capability selection bypass."""
        if args in ["off", "disabled", "false"]:
            return {"capability_selection_bypass_enabled": True}
        elif args in ["on", "enabled", "true"]:
            return {"capability_selection_bypass_enabled": False}
        else:
            raise CommandExecutionError(
                f"Invalid option '{args}' for /caps", "caps", "Use 'on' or 'off'"
            )

    # Register agent control commands
    registry.register(
        Command(
            name="planning",
            category=CommandCategory.AGENT_CONTROL,
            description="Enable/disable planning mode",
            handler=planning_handler,
            valid_options=["on", "off", "enabled", "disabled", "true", "false"],
            help_text="Control planning mode for the agent.\n\nOptions:\n  on/enabled/true  - Enable planning\n  off/disabled/false - Disable planning",
        )
    )

    registry.register(
        Command(
            name="approval",
            category=CommandCategory.AGENT_CONTROL,
            description="Control approval workflows",
            handler=approval_handler,
            valid_options=["on", "off", "selective", "enabled", "disabled", "true", "false"],
            help_text="Control approval workflows.\n\nOptions:\n  on/enabled - Enable all approvals\n  off/disabled - Disable approvals\n  selective - Selective approval mode",
        )
    )

    registry.register(
        Command(
            name="task",
            category=CommandCategory.AGENT_CONTROL,
            description="Control task extraction bypass",
            handler=task_handler,
            valid_options=["on", "off", "enabled", "disabled", "true", "false"],
            help_text="Control task extraction bypass for performance.\n\nOptions:\n  on/enabled - Use task extraction (default)\n  off/disabled - Bypass task extraction (use full context)",
        )
    )

    registry.register(
        Command(
            name="caps",
            category=CommandCategory.AGENT_CONTROL,
            description="Control capability selection bypass",
            handler=caps_handler,
            aliases=["capabilities"],
            valid_options=["on", "off", "enabled", "disabled", "true", "false"],
            help_text="Control capability selection bypass.\n\nOptions:\n  on/enabled - Use capability selection (default)\n  off/disabled - Bypass selection (activate all capabilities)",
        )
    )


def register_service_commands(registry) -> None:
    """Register service-specific commands."""

    def logs_handler(args: str, context: CommandContext) -> CommandResult:
        """Handle log viewer command."""
        if not context.service_instance:
            console = context.console or themed_console
            console.print("❌ Log viewer not available in this context", style=Styles.ERROR)
            return CommandResult.HANDLED

        # Delegate to service-specific log handling
        if hasattr(context.service_instance, "_handle_log_command"):
            # This would be called by the service
            return CommandResult.HANDLED
        else:
            console = context.console or themed_console
            console.print("❌ Log viewer not implemented", style=Styles.ERROR)
            return CommandResult.HANDLED

    registry.register(
        Command(
            name="logs",
            category=CommandCategory.SERVICE,
            description="View service logs",
            handler=logs_handler,
            help_text="View and filter service logs.\n\nUsage varies by service implementation.",
            interface_restrictions=["openwebui"],
        )
    )
