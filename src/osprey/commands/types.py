"""
Type Definitions for Centralized Slash Command System

This module provides the foundational type definitions for the Osprey Framework's
unified slash command system. The type system enables type-safe command registration,
execution, and context management across all framework interfaces.

Architecture:
    - CommandCategory: Hierarchical command organization for discovery and validation
    - CommandResult: Standardized execution outcomes for interface coordination
    - CommandContext: Rich execution context with interface-specific data
    - Command: Complete command specification with metadata and handlers
    - CommandHandler: Protocol for type-safe command implementation

The type system supports extensible command categories, context-aware execution,
and seamless integration with CLI, OpenWebUI, and custom interfaces through
a unified command registry and execution model.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class CommandCategory(Enum):
    """Categories of slash commands for organization and interface validation.

    Command categories provide hierarchical organization and enable interface-specific
    command filtering. Each category represents a distinct functional domain with
    specific execution contexts and validation requirements.

    Categories:
        CLI: Interface and user experience commands (help, clear, exit, history)
        AGENT_CONTROL: Agent behavior and execution control (planning, approval, debug)
        SERVICE: Framework service management (status, logs, metrics, restart)
        CUSTOM: Application-specific and user-defined extensions

    .. note::
       Categories are used for command discovery, autocompletion filtering,
       and interface-specific command availability validation.
    """

    CLI = "cli"  # Interface/UI commands (help, clear, exit)
    AGENT_CONTROL = "agent"  # Agent behavior control (planning, approval, debug)
    SERVICE = "service"  # Service-specific commands (logs, metrics)
    CUSTOM = "custom"  # User-defined custom commands


class CommandResult(Enum):
    """Standardized result types for command execution flow control.

    Command results enable interfaces to coordinate execution flow and handle
    different command outcomes appropriately. Results provide clear semantics
    for command processing continuation, state changes, and interface control.

    Results:
        CONTINUE: Command processed, continue with normal message flow
        HANDLED: Command fully processed, stop further message processing
        EXIT: Request interface termination (CLI exit, session end)
        AGENT_STATE_CHANGED: Agent control state modified, may affect execution

    .. note::
       Interfaces should handle each result type appropriately based on their
       execution model and user experience requirements.
    """

    CONTINUE = "continue"  # Continue normal processing
    HANDLED = "handled"  # Command was handled, stop processing
    EXIT = "exit"  # Exit the current interface
    AGENT_STATE_CHANGED = "agent_state_changed"  # Agent control state was modified


@dataclass
class CommandContext:
    """Execution context information available to command handlers.

    CommandContext provides rich contextual information to command handlers,
    enabling context-aware command execution with access to interface state,
    agent configuration, and service instances. The context supports multiple
    interface types while maintaining type safety and extensibility.

    Context Categories:
        Interface Context: Interface type, user identification, session management
        CLI Context: CLI instance access, console output, terminal control
        Agent Context: Current agent state, gateway access, configuration
        Service Context: Service instance access, deployment information
        Extension Context: Custom context data for application-specific commands

    :param interface_type: Interface identifier ("cli", "openwebui", "api", etc.)
    :type interface_type: str
    :param user_id: User identifier for multi-user interfaces
    :type user_id: Optional[str]
    :param session_id: Session identifier for state management
    :type session_id: Optional[str]
    :param cli_instance: CLI interface instance for direct access
    :type cli_instance: Optional[Any]
    :param console: Rich console instance for formatted output
    :type console: Optional[Any]
    :param agent_state: Current agent state for state-aware commands
    :type agent_state: Optional[Dict[str, Any]]
    :param gateway: Gateway instance for message processing
    :type gateway: Optional[Any]
    :param config: Framework configuration for service access
    :type config: Optional[Dict[str, Any]]
    :param service_instance: Service instance for service-specific commands
    :type service_instance: Optional[Any]
    :param extra: Additional context data for custom extensions
    :type extra: Dict[str, Any]

    .. note::
       Context fields are populated based on the executing interface and
       command category. Not all fields are available in all contexts.

    Examples:
        CLI command context::

            context = CommandContext(
                interface_type="cli",
                cli_instance=cli,
                console=rich_console,
                agent_state=current_state
            )

        Service command context::

            context = CommandContext(
                interface_type="api",
                service_instance=jupyter_service,
                config=framework_config
            )
    """

    # Interface context
    interface_type: str = "unknown"  # "cli", "openwebui", "api", etc.
    user_id: str | None = None
    session_id: str | None = None

    # CLI-specific context
    cli_instance: Any | None = None
    console: Any | None = None

    # Agent context
    agent_state: dict[str, Any] | None = None
    gateway: Any | None = None
    config: dict[str, Any] | None = None

    # Service context
    service_instance: Any | None = None

    # Additional context data
    extra: dict[str, Any] = field(default_factory=dict)


class CommandHandler(Protocol):
    """Protocol for type-safe command handler function signatures.

    CommandHandler defines the interface contract for all command handler
    functions, ensuring consistent signatures and return types across
    the command system. Handlers receive parsed arguments and execution
    context, returning standardized results or state changes.

    Handler Contract:
        - Accept string arguments and CommandContext
        - Return CommandResult or Dict[str, Any] for state changes
        - Handle errors gracefully with appropriate user feedback
        - Support both synchronous and asynchronous execution

    .. note::
       Agent control commands may return dictionaries representing
       state changes instead of CommandResult values.
    """

    async def __call__(self, args: str, context: CommandContext) -> CommandResult | dict[str, Any]:
        """Execute the command.

        Args:
            args: Command arguments as string
            context: Execution context

        Returns:
            CommandResult or dict of state changes for agent control commands
        """
        ...


@dataclass
class Command:
    """Complete specification for a slash command with metadata and execution constraints.

    The Command class provides a comprehensive definition for slash commands in the
    Osprey Framework. It includes execution handlers, validation constraints,
    interface restrictions, and rich metadata for help generation and autocompletion.

    Command Lifecycle:

    1. **Definition**: Command created with required fields and optional metadata
    2. **Registration**: Command registered in CommandRegistry with validation
    3. **Discovery**: Command discovered through registry lookup and filtering
    4. **Execution**: Command executed with context and argument validation
    5. **Completion**: Command provides autocompletion suggestions

    **Core Components:**

    - **Required Fields**: name, category, description, handler for basic functionality
    - **Metadata**: aliases, help_text, syntax for user experience and documentation
    - **Constraints**: requires_args, valid_options, interface_restrictions for validation
    - **Display**: hidden, deprecated flags for command visibility and lifecycle

    :param name: Unique command identifier used for registration and execution
    :type name: str
    :param category: Command category for organization and filtering
    :type category: CommandCategory
    :param description: Brief description for help text and documentation
    :type description: str
    :param handler: Function or callable that executes the command logic
    :type handler: CommandHandler
    :param aliases: Alternative names for the command (shortcuts, compatibility)
    :type aliases: List[str]
    :param help_text: Detailed help text with usage examples and options
    :type help_text: Optional[str]
    :param syntax: Command syntax pattern for help display and validation
    :type syntax: Optional[str]
    :param requires_args: Whether the command requires arguments to execute
    :type requires_args: bool
    :param valid_options: List of valid argument values for validation
    :type valid_options: Optional[List[str]]
    :param interface_restrictions: List of interfaces where command is available
    :type interface_restrictions: Optional[List[str]]
    :param hidden: Whether command is hidden from help and autocompletion
    :type hidden: bool
    :param deprecated: Whether command is deprecated (shown with warning)
    :type deprecated: bool

    Examples:
        Simple command definition::

            command = Command(
                name="status",
                category=CommandCategory.SERVICE,
                description="Show service status",
                handler=status_handler
            )

        Command with validation and help::

            command = Command(
                name="planning",
                category=CommandCategory.AGENT_CONTROL,
                description="Control planning mode",
                handler=planning_handler,
                aliases=["plan"],
                valid_options=["on", "off", "enabled", "disabled"],
                help_text="Enable or disable planning mode.\\n\\nOptions:\\n  on/enabled - Enable planning\\n  off/disabled - Disable planning",
                interface_restrictions=["cli", "openwebui"]
            )

        Hidden administrative command::

            command = Command(
                name="internal_debug",
                category=CommandCategory.CUSTOM,
                description="Internal debugging command",
                handler=debug_handler,
                hidden=True,
                requires_args=True
            )
    """

    name: str
    category: CommandCategory
    description: str
    handler: CommandHandler

    # Command metadata
    aliases: list[str] = field(default_factory=list)
    help_text: str | None = None
    syntax: str | None = None

    # Execution constraints
    requires_args: bool = False
    valid_options: list[str] | None = None
    interface_restrictions: list[str] | None = None  # ["cli", "openwebui"]

    # Display properties
    hidden: bool = False
    deprecated: bool = False

    def __post_init__(self):
        """Initialize computed fields and auto-generate missing metadata.

        Automatically generates help_text from description if not provided,
        and creates syntax patterns based on valid_options and requires_args
        settings for consistent command documentation and validation.
        """
        if self.help_text is None:
            self.help_text = self.description

        if self.syntax is None:
            if self.valid_options:
                options = "|".join(self.valid_options)
                self.syntax = (
                    f"/{self.name}:{options}" if self.requires_args else f"/{self.name}[:{options}]"
                )
            else:
                self.syntax = f"/{self.name}:<value>" if self.requires_args else f"/{self.name}"

    def is_valid_for_interface(self, interface_type: str) -> bool:
        """Check if command is valid for the given interface type.

        Validates whether the command can be executed in the specified interface
        based on interface_restrictions. Commands without restrictions are
        available in all interfaces.

        :param interface_type: Interface identifier ("cli", "openwebui", "api", etc.)
        :type interface_type: str
        :return: True if command is available in the interface, False otherwise
        :rtype: bool

        Examples:
            Check CLI availability::

                if command.is_valid_for_interface("cli"):
                    # Command can be used in CLI
                    pass
        """
        if self.interface_restrictions is None:
            return True
        return interface_type in self.interface_restrictions

    def validate_option(self, option: str | None) -> bool:
        """Validate command arguments against defined constraints.

        Performs validation of command arguments based on requires_args and
        valid_options settings. Used by the command registry to ensure
        proper command usage before execution.

        :param option: Command argument to validate (None if no argument provided)
        :type option: Optional[str]
        :return: True if argument is valid, False if validation fails
        :rtype: bool

        Validation Rules:
            - If requires_args=True, option cannot be None
            - If valid_options is set, option must be in the list
            - If no constraints, any option (including None) is valid

        Examples:
            Validate planning command::

                # Command with valid_options=["on", "off"]
                command.validate_option("on")    # True
                command.validate_option("invalid")  # False
                command.validate_option(None)    # True (optional)
        """
        if self.requires_args and option is None:
            return False
        if self.valid_options and option is not None:
            return option in self.valid_options
        return True


@dataclass
class ParsedCommand:
    """Result of parsing a command line."""

    command_name: str
    option: str | None = None
    remaining_text: str = ""
    is_valid: bool = True
    error_message: str | None = None


class CommandExecutionError(Exception):
    """Exception raised during command execution."""

    def __init__(self, message: str, command_name: str, suggestion: str | None = None):
        super().__init__(message)
        self.command_name = command_name
        self.suggestion = suggestion
