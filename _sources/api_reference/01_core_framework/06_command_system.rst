Command System
==============

.. currentmodule:: osprey.commands

The centralized command system provides unified slash command processing across all interfaces (CLI, OpenWebUI, etc.) with extensible command categories and rich autocompletion.

This system enables consistent command handling, context-aware execution, and seamless integration across different framework interfaces through a unified registry and execution model.

Core Components
---------------

Command Registry
~~~~~~~~~~~~~~~~

.. autoclass:: CommandRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Command Types
~~~~~~~~~~~~~

.. autoclass:: Command
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CommandContext
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CommandResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CommandCategory
   :members:
   :undoc-members:
   :show-inheritance:

Registry Functions
------------------

.. autofunction:: get_command_registry

.. autofunction:: register_command

.. autofunction:: execute_command

.. autofunction:: parse_command_line

Command Categories
------------------

.. autofunction:: register_cli_commands

.. autofunction:: register_agent_control_commands

.. autofunction:: register_service_commands

Usage Examples
--------------

Basic Command Execution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.commands import get_command_registry, CommandContext

   # Get the global registry
   registry = get_command_registry()

   # Create execution context
   context = CommandContext(
       interface_type="cli",
       console=console
   )

   # Execute a command
   result = await registry.execute("/help", context)

Custom Command Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from osprey.commands import Command, CommandCategory, CommandResult

   def my_handler(args: str, context: CommandContext) -> CommandResult:
       context.console.print(f"Custom command executed with args: {args}")
       return CommandResult.HANDLED

   # Register custom command
   registry = get_command_registry()
   registry.register(Command(
       name="custom",
       category=CommandCategory.CUSTOM,
       description="My custom command",
       handler=my_handler,
       help_text="Execute custom functionality"
   ))

Interface Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CLI interface integration
   context = CommandContext(
       interface_type="cli",
       cli_instance=cli,
       console=rich_console
   )

   # OpenWebUI interface integration
   context = CommandContext(
       interface_type="openwebui",
       user_id="user123",
       session_id="session456"
   )

   # Execute commands with appropriate context
   result = await registry.execute(user_input, context)

.. seealso::

   :doc:`../../developer-guides/02_quick-start-patterns/00_cli-reference`
       CLI usage and available slash commands

   :doc:`../../developer-guides/04_infrastructure-components/01_gateway-architecture`
       Gateway integration with command system

   :doc:`../../developer-guides/03_core-framework-systems/01_state-management-architecture`
       Agent state management for command execution
