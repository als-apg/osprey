"""Framework prompt system for domain-agnostic prompt management and dependency injection.

This module provides the core infrastructure for the framework's prompt system,
enabling clean separation between generic framework functionality and domain-specific
prompt customization. Applications can provide specialized prompts while maintaining
full compatibility with framework orchestration, task extraction, and response
generation systems.

The prompt system follows a dependency injection pattern where framework components
request prompt builders through abstract interfaces, and the registry system provides
application-specific implementations at runtime. This achieves true dependency
inversion - the framework depends on abstractions, not concrete implementations.

Key Components:
    - **FrameworkPromptBuilder**: Abstract base class for building modular prompts
    - **get_framework_prompts**: Primary access function for framework infrastructure
    - **FrameworkPromptProvider**: Provider interface for application-specific prompts
    - **FrameworkPromptLoader**: Global registry for prompt provider management

Architecture Benefits:
    - **Domain Agnostic**: Framework remains generic while supporting specialized prompts
    - **No Circular Dependencies**: Clean separation through dependency injection
    - **Flexible Composition**: Modular prompt building with optional components
    - **Development Support**: Integrated debugging and prompt inspection tools

.. note::
   Applications typically register their prompt providers during initialization,
   and framework components access them through get_framework_prompts() without
   knowing which specific application is providing the prompts.

.. warning::
   Prompt providers must be registered before framework components attempt to
   access prompts, or ValueError exceptions will be raised with clear diagnostics.

Examples:
    Framework infrastructure accessing prompts::

        # In orchestration_node.py
        from osprey.prompts import get_framework_prompts

        prompt_provider = get_framework_prompts()
        orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()
        system_prompt = orchestrator_builder.get_planning_instructions(
            capabilities=active_capabilities,
            context_manager=context_manager
        )

    Application registering custom prompts::

        # In application initialization
        from osprey.prompts.loader import register_framework_prompt_provider
        from applications.als_assistant.framework_prompts import ALSPromptProvider

        register_framework_prompt_provider("als_assistant", ALSPromptProvider())

.. seealso::
   :doc:`/developer-guides/03_core-framework-systems/04_prompt-customization` : Complete guide for customizing prompts
   :class:`osprey.prompts.defaults.DefaultPromptProvider` : Framework default implementations
   :class:`applications.als_assistant.framework_prompts.ALSPromptProvider` : Example application customization
"""

from .base import FrameworkPromptBuilder
from .loader import get_framework_prompts

__all__ = ["FrameworkPromptBuilder", "get_framework_prompts"]
