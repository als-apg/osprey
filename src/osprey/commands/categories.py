"""Built-in command handlers grouped by category (CLI, agent control, service)."""

from typing import Any

from rich.panel import Panel
from rich.table import Table

from osprey.cli.styles import Styles
from osprey.cli.styles import console as themed_console

from .types import Command, CommandCategory, CommandContext, CommandExecutionError, CommandResult


def register_cli_commands(registry) -> None:
    """Register /help, /exit, /clear, /config, and /status commands."""

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

    def exit_handler(args: str, context: CommandContext) -> CommandResult | dict[str, Any]:
        """Exit direct chat mode or the CLI session."""
        console = context.console or themed_console

        if context.agent_state:
            session_state = context.agent_state.get("session_state", {})
            if session_state.get("direct_chat_capability"):
                capability_name = session_state["direct_chat_capability"]
                console.print(
                    f"✓ Exited direct chat with [bold]{capability_name}[/bold]",
                    style=Styles.SUCCESS,
                )
                console.print("  Returning to normal mode\n", style=Styles.DIM)

                transition_message = {
                    "role": "system",
                    "content": f"[End of direct chat session with '{capability_name}'. "
                    f"The messages above were from a specialized mode. "
                    f"New user messages should be processed as fresh requests.]",
                }

                return {
                    "session_state": {
                        "direct_chat_capability": None,
                        "last_direct_chat_result": None,
                    },
                    "messages": [transition_message],
                }

        console.print("👋 Goodbye!", style=Styles.WARNING)
        return CommandResult.EXIT

    def config_handler(args: str, context: CommandContext) -> CommandResult:
        """Show current configuration."""
        console = context.console or themed_console

        if context.config:
            # Unwrap "configurable" wrapper if present.
            config = context.config.get("configurable", context.config)

            # Session
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

            # Models
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

            # Providers
            provider_info = []
            if "provider_configs" in config:
                providers = config["provider_configs"]
                for name, provider_config in providers.items():
                    base_url = provider_config.get("base_url", "N/A")
                    timeout = provider_config.get("timeout", "N/A")
                    provider_info.append(f"  {name}: {base_url} (timeout: {timeout}s)")

            # Execution limits
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

            # Agent control
            agent_info = []
            if "agent_control_defaults" in config:
                agent_control = config["agent_control_defaults"]

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

                epics_writes = agent_control.get("epics_writes_enabled", False)
                agent_info.append(
                    f"EPICS Writes: {'✅ Enabled' if epics_writes else '❌ Disabled'}"
                )

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

            # Python executor
            python_info = []
            if "python_executor" in config:
                py_config = config["python_executor"]
                if py_config:
                    jupyter_url = py_config.get("jupyter_url", "N/A")
                    python_info.append(f"Jupyter URL: {jupyter_url}")
                    execution_mode = py_config.get("execution_mode", "N/A")
                    python_info.append(f"Execution Mode: {execution_mode}")

            # Services
            service_info = []
            if "service_configs" in config:
                services = config["service_configs"]
                for service_name in list(services.keys())[:5]:  # Show first 5 services
                    service_info.append(f"  {service_name}")

            # Development settings
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

            # Project
            project_info = []
            if "project_root" in config and config["project_root"]:
                project_info.append(f"Root: {config['project_root']}")
            if "current_application" in config and config["current_application"]:
                project_info.append(f"Application: {config['current_application']}")
            if "registry_path" in config and config["registry_path"]:
                project_info.append(f"Registry: {config['registry_path']}")

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
            from osprey.cli.health_cmd import HealthChecker

            console.print("🔍 Running comprehensive system health check...", style=Styles.INFO)
            console.print()

            checker = HealthChecker(verbose=True, full=True)
            checker.check_all()

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
                state_info = "Agent State: ✅ Available"

                if isinstance(context.agent_state, dict):
                    if "messages" in context.agent_state:
                        msg_count = len(context.agent_state["messages"])
                        state_info += f" ({msg_count} messages)"

                    if "execution_step_results" in context.agent_state:
                        step_count = len(context.agent_state["execution_step_results"])
                        if step_count > 0:
                            state_info += f", {step_count} execution steps"

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
            description="Exit direct chat mode or CLI interface",
            handler=exit_handler,
            aliases=["quit", "bye", "q"],
            help_text="Exit direct chat mode (returns to normal mode) or exit the CLI interface.",
            gateway_handled=True,
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
    """Register /planning, /approval, /task, /caps, and /chat commands."""

    def chat_mode_handler(args: str, context: CommandContext) -> CommandResult | dict[str, Any]:
        """Enter direct chat mode with a capability's ReAct agent."""
        from osprey.registry import get_registry

        console = context.console or themed_console
        reg = get_registry()

        if not args.strip():
            direct_chat_capable = []
            for cap_instance in reg.get_all_capabilities():
                if getattr(cap_instance, "direct_chat_enabled", False):
                    direct_chat_capable.append(
                        {
                            "name": getattr(cap_instance, "name", "unknown"),
                            "description": getattr(cap_instance, "description", "N/A"),
                        }
                    )

            if not direct_chat_capable:
                console.print("❌ No capabilities support direct chat mode", style=Styles.ERROR)
                console.print(
                    "💡 Enable direct chat by setting direct_chat_enabled = True on a capability",
                    style=Styles.DIM,
                )
                return CommandResult.HANDLED

            table = Table(title="Available Direct Chat Capabilities", show_header=True)
            table.add_column("Capability", style=Styles.ACCENT)
            table.add_column("Description", style=Styles.PRIMARY)

            for cap in direct_chat_capable:
                table.add_row(cap["name"], cap["description"])

            console.print(table)
            console.print("\n💡 Use /chat:<capability_name> to start", style=Styles.DIM)
            return CommandResult.HANDLED

        capability_name = args.strip()

        cap_instance = reg.get_capability(capability_name)
        if cap_instance is None:
            console.print(f"❌ Unknown capability: {capability_name}", style=Styles.ERROR)
            console.print("💡 Use /chat to list available capabilities", style=Styles.DIM)
            return CommandResult.HANDLED

        if not getattr(cap_instance, "direct_chat_enabled", False):
            console.print(
                f"❌ Capability '{capability_name}' does not support direct chat mode",
                style=Styles.ERROR,
            )
            console.print(
                "💡 Only capabilities with direct_chat_enabled = True support direct chat",
                style=Styles.DIM,
            )
            return CommandResult.HANDLED

        console.print(
            f"✓ Entering direct chat with [bold]{capability_name}[/bold]",
            style=Styles.SUCCESS,
        )
        console.print("  Type /exit to return to normal mode", style=Styles.DIM)
        if capability_name != "state_manager":
            console.print("  💡 Say 'save that as <key>' to store results\n", style=Styles.DIM)
        else:
            console.print()  # Just add newline for state_manager

        return {
            "session_state": {
                "direct_chat_capability": capability_name,
            }
        }

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

    registry.register(
        Command(
            name="planning",
            category=CommandCategory.AGENT_CONTROL,
            description="Enable/disable planning mode",
            handler=planning_handler,
            valid_options=["on", "off", "enabled", "disabled", "true", "false"],
            help_text="Control planning mode for the agent.\n\nOptions:\n  on/enabled/true  - Enable planning\n  off/disabled/false - Disable planning",
            gateway_handled=True,
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
            gateway_handled=True,
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
            gateway_handled=True,
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
            gateway_handled=True,
        )
    )

    registry.register(
        Command(
            name="chat",
            category=CommandCategory.AGENT_CONTROL,
            description="Enter direct chat mode with a capability",
            handler=chat_mode_handler,
            help_text="""Enter direct chat mode for conversational interaction with a capability's ReAct agent.

Usage:
  /chat                    - List available capabilities
  /chat:<capability_name>  - Enter direct chat mode

In direct chat mode:
  - Your messages go directly to the capability's ReAct agent
  - No task extraction, classification, or orchestration
  - Say "save that as <key>" to store results in context
  - Use /exit to return to normal mode

Examples:
  /chat:weather_mcp       # Chat with weather MCP capability
  /chat:slack_mcp         # Chat with Slack MCP capability

  # Inside chat mode:
  > What's the weather in Tokyo?
  [Agent responds]
  > Save that as tokyo_weather
  [Saved to context]""",
            syntax="/chat[:<capability_name>]",
            gateway_handled=True,
        )
    )
