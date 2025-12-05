"""
Framework prompt loader system.

This file implements the registration and dependency injection system for framework prompts.
It solves the architectural challenge of allowing framework infrastructure to use domain-specific
prompts without creating circular dependencies. Applications register their prompt providers here,
and framework components can request prompts through the abstract interface without knowing which
specific application is providing them. This achieves true dependency inversion - the framework
depends on abstractions, not concrete implementations.
"""

from .base import FrameworkPromptBuilder


class FrameworkPromptProvider:
    """Abstract provider interface for framework prompt builders with dependency injection support.

    This class defines the contract that applications must implement to provide
    domain-specific prompt builders to the framework infrastructure. It enables
    clean separation of concerns where the framework handles orchestration, task
    extraction, and response generation logic while applications provide the
    domain-specific prompts and terminology.

    The provider pattern solves the architectural challenge of allowing framework
    infrastructure to use application-specific prompts without creating circular
    dependencies. Framework components request prompt builders through this
    abstract interface, and the registry system injects the appropriate
    application-specific implementations at runtime.

    Prompt builders are organized into two categories:

    **Infrastructure Prompts**: Used by core framework components like
    orchestration, task extraction, and response generation. These prompts
    control the fundamental behavior of the agent system.

    **Framework Capability Prompts**: Used by built-in framework capabilities
    like memory extraction, time parsing, and Python execution. These prompts
    can be customized to use domain-specific terminology and examples.

    :raises NotImplementedError: All methods must be implemented by concrete providers

    .. note::
       Applications typically inherit from this class and override only the
       prompt builders they want to customize, using framework defaults for
       the rest through composition patterns.

    .. warning::
       All methods in this interface must be implemented. Use framework defaults
       or delegation patterns if you don't need custom behavior for specific prompts.

    Examples:
        Basic application-specific provider::

            class ALSPromptProvider(FrameworkPromptProvider):
                def __init__(self):
                    # Use custom builders for key infrastructure prompts
                    self._orchestrator = ALSOrchestratorPromptBuilder()
                    self._task_extraction = ALSTaskExtractionPromptBuilder()

                    # Use framework defaults for others
                    from osprey.prompts.defaults import DefaultPromptProvider
                    self._defaults = DefaultPromptProvider()

                def get_orchestrator_prompt_builder(self):
                    return self._orchestrator

                def get_task_extraction_prompt_builder(self):
                    return self._task_extraction

                def get_classification_prompt_builder(self):
                    # Delegate to framework default
                    return self._defaults.get_classification_prompt_builder()

        Registration in application registry::

            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    application_name="als_assistant",
                    module_path="applications.als_assistant.framework_prompts",
                    description="ALS-specific framework prompt provider",
                    prompt_builders={
                        "orchestrator": "ALSOrchestratorPromptBuilder",
                        "task_extraction": "ALSTaskExtractionPromptBuilder"
                        # Others use framework defaults
                    }
                )
            ]

    .. seealso::
       :class:`FrameworkPromptBuilder` : Base class for individual prompt builders
       :class:`FrameworkPromptLoader` : Global loader for provider management
       :func:`register_framework_prompt_provider` : Provider registration function
       :doc:`/developer-guides/03_core-framework-systems/04_prompt-customization` : Complete customization guide
    """

    # =================================================================
    # Infrastructure prompts (used by framework infrastructure)
    # =================================================================

    def get_orchestrator_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for execution planning and orchestration.

        This prompt builder is used by the orchestration node to create detailed
        execution plans that break down user requests into specific, actionable
        steps. The orchestrator prompt is critical for the agent's ability to
        coordinate multiple capabilities and manage complex workflows.

        :return: Orchestrator prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           The orchestrator prompt heavily influences the agent's planning
           capabilities and should include domain-specific planning patterns
           and capability integration guidance.

        .. seealso::
           :class:`OrchestrationNode` : Infrastructure component that uses this prompt
           :class:`OrchestratorGuide` : Planning guidance structure
        """
        raise NotImplementedError

    def get_task_extraction_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for converting conversations into actionable tasks.

        This prompt builder is used by the task extraction node to analyze
        conversation history and extract clear, actionable tasks. It's responsible
        for understanding user intent and creating focused task descriptions that
        can be effectively planned and executed.

        :return: Task extraction prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Task extraction prompts should include domain-specific terminology
           and examples to improve understanding of specialized requests.

        .. seealso::
           :class:`TaskExtractionNode` : Infrastructure component that uses this prompt
           :class:`ExtractedTask` : Output structure for extracted tasks
        """
        raise NotImplementedError

    def get_response_generation_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for generating final user responses.

        This prompt builder is used by the response generation system to create
        coherent, helpful responses based on execution results. It handles the
        synthesis of multiple execution outputs into user-friendly responses
        with appropriate formatting and context.

        :return: Response generation prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Response generation prompts should reflect the application's
           communication style and include domain-specific formatting guidelines.

        .. seealso::
           :class:`RespondCapability` : Component that uses this prompt
           :func:`_get_base_system_prompt` : Response prompt composition
        """
        raise NotImplementedError

    def get_classification_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for task and capability classification.

        This prompt builder is used by the classification system to determine
        which capabilities are needed for specific tasks. It enables intelligent
        routing of tasks to appropriate capabilities based on content analysis
        and domain-specific patterns.

        :return: Classification prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Classification prompts should include clear criteria for capability
           selection and domain-specific task categorization patterns.

        .. seealso::
           :func:`select_capabilities` : Task classification function
           :class:`TaskClassifierGuide` : Classification guidance structure
        """
        raise NotImplementedError

    def get_error_analysis_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for error analysis and recovery guidance.

        This prompt builder is used by the error handling system to analyze
        failures and provide meaningful explanations to users. It focuses on
        translating technical errors into understandable explanations with
        appropriate recovery suggestions.

        :return: Error analysis prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Error analysis prompts should provide domain-specific context for
           common failure modes and recovery patterns.

        .. seealso::
           :class:`ErrorNode` : Infrastructure component for error handling
           :class:`ErrorClassification` : Error categorization system
        """
        raise NotImplementedError

    def get_clarification_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for generating clarifying questions.

        This prompt builder is used by the clarification system to generate
        targeted questions when user requests are ambiguous or incomplete.
        It helps gather the specific information needed to provide accurate
        and helpful responses.

        :return: Clarification prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Clarification prompts should include domain-specific question
           patterns and common ambiguity resolution strategies.

        .. seealso::
           :class:`ClarifyCapability` : Component that uses this prompt
           :class:`ClarifyingQuestionsResponse` : Output structure for questions
        """
        raise NotImplementedError

    # =================================================================
    # Framework capability prompts (used by framework capabilities)
    # =================================================================

    def get_memory_extraction_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for extracting memorable information from conversations.

        This prompt builder is used by the memory extraction capability to
        identify and structure information that should be preserved for future
        reference. It focuses on extracting user preferences, important facts,
        and contextual information that enhances future interactions.

        :return: Memory extraction prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Memory extraction prompts should include domain-specific patterns
           for identifying important information and user preferences.

        .. seealso::
           :class:`MemoryCapability` : Framework capability that uses this prompt
           :class:`MemoryExtractionResult` : Output structure for extracted memories
        """
        raise NotImplementedError

    def get_time_range_parsing_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for parsing natural language time expressions.

        This prompt builder is used by the time range parsing capability to
        convert natural language time expressions into structured datetime
        ranges. It handles relative expressions, absolute dates, and
        domain-specific time references.

        :return: Time range parsing prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Time parsing prompts should include domain-specific time patterns
           and examples relevant to the application's temporal context.

        .. seealso::
           :class:`TimeRangeParsingCapability` : Framework capability that uses this prompt
           :class:`TimeRange` : Output structure for parsed time ranges
        """
        raise NotImplementedError

    def get_python_prompt_builder(self) -> FrameworkPromptBuilder:
        """Provide prompt builder for Python code generation and execution.

        This prompt builder is used by the Python execution capability to
        generate and execute Python code for data analysis, calculations,
        and other computational tasks. It includes safety guidelines and
        domain-specific code patterns.

        :return: Python capability prompt builder instance
        :rtype: FrameworkPromptBuilder
        :raises NotImplementedError: Must be implemented by concrete providers

        .. note::
           Python prompts should include domain-specific libraries, patterns,
           and safety constraints appropriate for the application context.

        .. seealso::
           :class:`PythonCapability` : Framework capability that uses this prompt
           :class:`PythonExecutionService` : Code execution infrastructure
        """
        raise NotImplementedError


class FrameworkPromptLoader:
    """Global registry and dependency injection system for framework prompt providers.

    This class manages the registration and retrieval of application-specific
    prompt providers, enabling the framework infrastructure to access domain-specific
    prompts without creating circular dependencies. It implements a service locator
    pattern with fail-fast error handling and clear diagnostics.

    The loader maintains a registry of prompt providers keyed by application name,
    with automatic default provider selection and explicit provider override
    capabilities. It's designed to be used as a singleton through the global
    module-level functions.

    :param _providers: Registry of application prompt providers
    :type _providers: Dict[str, FrameworkPromptProvider]
    :param _default_provider: Name of the default provider application
    :type _default_provider: Optional[str]

    .. note::
       This class is typically accessed through the module-level functions
       rather than instantiated directly. The global instance handles all
       framework prompt provider management.

    .. warning::
       Provider registration must occur during application initialization
       before any framework components attempt to access prompts.

    Examples:
        Typical usage through module functions::

            # Registration (usually in application initialization)
            register_framework_prompt_provider("als_assistant", ALSPromptProvider())

            # Access (from framework infrastructure)
            provider = get_framework_prompts()
            orchestrator_builder = provider.get_orchestrator_prompt_builder()

        Direct usage for testing or specialized cases::

            loader = FrameworkPromptLoader()
            loader.register_provider("test_app", TestPromptProvider())
            provider = loader.get_provider("test_app")

    .. seealso::
       :func:`get_framework_prompts` : Primary access function
       :func:`register_framework_prompt_provider` : Provider registration
       :class:`FrameworkPromptProvider` : Provider interface
    """

    def __init__(self):
        """Initialize empty prompt provider registry.

        Creates a new loader instance with empty provider registry and no
        default provider. The first registered provider automatically becomes
        the default unless explicitly overridden.
        """
        self._providers: dict[str, FrameworkPromptProvider] = {}
        self._default_provider: str | None = None

    def register_provider(self, application_name: str, provider: FrameworkPromptProvider):
        """Register a prompt provider for an application with automatic default selection.

        Adds the provider to the registry and automatically sets it as the default
        if no default provider is currently configured. This enables simple
        single-application setups while supporting multi-application scenarios.

        :param application_name: Unique identifier for the application
        :type application_name: str
        :param provider: Prompt provider implementation for the application
        :type provider: FrameworkPromptProvider

        .. note::
           The first registered provider automatically becomes the default.
           Use set_default_provider() to change the default selection.

        Examples:
            Basic registration::

                loader.register_provider("als_assistant", ALSPromptProvider())
                # als_assistant becomes default if first registration

            Multiple application registration::

                loader.register_provider("als_assistant", ALSPromptProvider())
                loader.register_provider("wind_turbine", WindTurbinePromptProvider())
                # als_assistant remains default

        .. seealso::
           :meth:`set_default_provider` : Explicit default provider selection
           :meth:`get_provider` : Provider retrieval
        """
        self._providers[application_name] = provider
        if self._default_provider is None:
            self._default_provider = application_name

    def set_default_provider(self, application_name: str):
        """Set the default prompt provider with validation.

        Changes the default provider to the specified application, with
        validation to ensure the provider is already registered. The default
        provider is used when no specific application is requested.

        :param application_name: Name of registered application to use as default
        :type application_name: str
        :raises ValueError: If the specified provider is not registered

        Examples:
            Setting default after multiple registrations::

                loader.register_provider("als_assistant", ALSPromptProvider())
                loader.register_provider("wind_turbine", WindTurbinePromptProvider())
                loader.set_default_provider("wind_turbine")
                # wind_turbine is now default instead of als_assistant

        .. seealso::
           :meth:`register_provider` : Provider registration with auto-default
           :meth:`get_provider` : Provider retrieval using defaults
        """
        if application_name not in self._providers:
            raise ValueError(f"Provider '{application_name}' not registered")
        self._default_provider = application_name

    def get_provider(self, application_name: str | None = None) -> FrameworkPromptProvider:
        """Retrieve prompt provider with fail-fast error handling and clear diagnostics.

        Returns the prompt provider for the specified application, or the default
        provider if no application is specified. Provides comprehensive error
        messages with available alternatives when providers are not found.

        :param application_name: Specific application name, or None for default
        :type application_name: Optional[str]
        :return: Prompt provider implementation for the application
        :rtype: FrameworkPromptProvider
        :raises ValueError: If no default provider is configured
        :raises ValueError: If specified provider is not found

        .. note::
           This method uses fail-fast error handling to catch configuration
           issues early in the application lifecycle with clear error messages.

        Examples:
            Using default provider::

                provider = loader.get_provider()  # Uses default
                orchestrator = provider.get_orchestrator_prompt_builder()

            Using specific provider::

                provider = loader.get_provider("als_assistant")
                task_extractor = provider.get_task_extraction_prompt_builder()

            Error handling::

                try:
                    provider = loader.get_provider("nonexistent")
                except ValueError as e:
                    print(f"Provider error: {e}")
                    # Error includes list of available providers

        .. seealso::
           :meth:`register_provider` : Provider registration
           :meth:`set_default_provider` : Default provider management
        """
        if application_name is None:
            application_name = self._default_provider

        if application_name is None:
            raise ValueError("No default prompt provider configured")

        if application_name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"Prompt provider '{application_name}' not found. Available: {available}"
            )

        return self._providers[application_name]


# Global prompt loader instance - singleton for framework-wide prompt provider management
_prompt_loader = FrameworkPromptLoader()


def get_framework_prompts(application_name: str | None = None) -> FrameworkPromptProvider:
    """Access the framework prompt provider system with optional application targeting.

    This is the primary entry point for framework components to access prompt
    builders. It provides a clean interface to the global prompt provider registry,
    enabling dependency injection of application-specific prompts without
    circular dependencies.

    The function supports both default provider access (for single-application
    deployments) and explicit provider selection (for multi-application scenarios).
    It integrates with the global prompt loader to provide consistent access
    patterns across the framework.

    :param application_name: Specific application provider to use, or None for default
    :type application_name: Optional[str]
    :return: Prompt provider implementation for the specified or default application
    :rtype: FrameworkPromptProvider
    :raises ValueError: If no providers are registered or specified provider not found

    .. note::
       This function is the recommended way to access prompt providers from
       framework infrastructure components. It provides consistent error handling
       and integrates with the application registry system.

    Examples:
        Framework infrastructure usage::

            # In orchestration_node.py
            prompt_provider = get_framework_prompts()
            orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()
            system_prompt = orchestrator_builder.get_system_instructions(
                capabilities=active_capabilities,
                context_manager=context_manager
            )

        Multi-application deployment::

            # Use specific application's prompts
            als_provider = get_framework_prompts("als_assistant")
            wind_provider = get_framework_prompts("wind_turbine")

        Error handling pattern::

            try:
                provider = get_framework_prompts()
            except ValueError as e:
                logger.error(f"Prompt provider not configured: {e}")
                # Fallback to framework defaults or fail gracefully

    .. seealso::
       :func:`register_framework_prompt_provider` : Provider registration
       :class:`FrameworkPromptProvider` : Provider interface definition
       :class:`FrameworkPromptLoader` : Underlying registry implementation
    """
    return _prompt_loader.get_provider(application_name)


def register_framework_prompt_provider(application_name: str, provider: FrameworkPromptProvider):
    """Register an application-specific prompt provider in the global registry.

    This function integrates application prompt providers with the framework's
    dependency injection system. It's typically called during application
    initialization to make domain-specific prompts available to framework
    infrastructure components.

    The registration enables the framework to use application-specific terminology,
    examples, and patterns while maintaining clean architectural separation.
    The first registered provider automatically becomes the default for
    single-application deployments.

    :param application_name: Unique identifier for the application
    :type application_name: str
    :param provider: Implementation of the prompt provider interface
    :type provider: FrameworkPromptProvider

    .. note::
       This function should be called during application initialization,
       typically from the application's registry configuration or startup code.

    .. warning::
       Provider registration must occur before any framework components
       attempt to access prompts, or ValueError exceptions will be raised.

    Examples:
        Application initialization::

            # In als_assistant/__init__.py or registry setup
            from applications.als_assistant.framework_prompts import ALSPromptProvider

            register_framework_prompt_provider(
                "als_assistant",
                ALSPromptProvider()
            )

        Multiple application setup::

            register_framework_prompt_provider("als_assistant", ALSPromptProvider())
            register_framework_prompt_provider("wind_turbine", WindTurbinePromptProvider())
            # First registration (als_assistant) becomes default

        Testing setup::

            # In test fixtures
            test_provider = MockPromptProvider()
            register_framework_prompt_provider("test", test_provider)

    .. seealso::
       :func:`get_framework_prompts` : Provider access function
       :func:`set_default_framework_prompt_provider` : Default provider management
       :class:`FrameworkPromptProviderRegistration` : Registry metadata structure
    """
    _prompt_loader.register_provider(application_name, provider)


def set_default_framework_prompt_provider(application_name: str):
    """Configure the default prompt provider for framework components.

    This function changes which application's prompt provider is used when
    framework components don't specify a particular application. It's useful
    in multi-application deployments where you want to control which
    application's prompts are used by default.

    The default provider is used by get_framework_prompts() when no specific
    application name is provided, enabling clean single-application usage
    patterns while supporting multi-application flexibility.

    :param application_name: Name of registered application to use as default
    :type application_name: str
    :raises ValueError: If the specified application is not registered

    .. note::
       The first registered provider automatically becomes the default.
       This function is only needed to change the default selection.

    Examples:
        Changing default in multi-application setup::

            # Register multiple providers
            register_framework_prompt_provider("als_assistant", ALSPromptProvider())
            register_framework_prompt_provider("wind_turbine", WindTurbinePromptProvider())

            # als_assistant is default (first registered)
            # Change to wind_turbine as default
            set_default_framework_prompt_provider("wind_turbine")

            # Now get_framework_prompts() returns wind_turbine provider

        Runtime provider switching::

            # Switch defaults based on configuration or context
            if config.get("primary_application") == "wind_turbine":
                set_default_framework_prompt_provider("wind_turbine")
            else:
                set_default_framework_prompt_provider("als_assistant")

    .. seealso::
       :func:`register_framework_prompt_provider` : Provider registration
       :func:`get_framework_prompts` : Provider access with default selection
       :class:`FrameworkPromptLoader` : Underlying registry implementation
    """
    _prompt_loader.set_default_provider(application_name)
