"""
Base classes for framework prompt builders.

This file defines the abstract interfaces that application layers must implement to provide
domain-specific prompts to the framework infrastructure. This enables clean separation between
the generic framework (which handles orchestration, task extraction, etc.) and application-specific
domain knowledge (like ALS accelerator terminology). Applications register concrete implementations
of these builders, allowing the framework to remain domain-agnostic while still generating
contextually appropriate prompts.
"""

import os
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from osprey.base import TaskClassifierGuide
from osprey.utils.config import get_agent_dir, get_config_value
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.base import BaseExample, OrchestratorGuide
logger = get_logger("osprey")


class FrameworkPromptBuilder(ABC):
    """Abstract base class for building domain-agnostic framework prompts with flexible composition.

    This class provides the foundational architecture for the framework's prompt system,
    enabling clean separation between generic framework infrastructure and domain-specific
    prompt customization. Applications can inherit from this class to provide specialized
    prompts while maintaining compatibility with the framework's orchestration, task
    extraction, and response generation systems.

    The prompt building system follows a modular composition pattern where each component
    (role, task, instructions, examples, context) can be independently customized or
    omitted based on the specific needs of the prompt type. This flexibility allows
    everything from simple single-purpose prompts to complex multi-stage prompts with
    dynamic context injection.

    :raises NotImplementedError: If required abstract methods are not implemented

    .. note::
       All prompts automatically integrate with the framework's debug system for
       development visibility. Set development.prompts.show_all=true in config
       to see generated prompts in logs.

    .. warning::
       Subclasses must implement get_role_definition() and get_instructions().
       These are the minimum requirements for a functional prompt.

    Examples:
        Basic prompt builder implementation::

            class CustomPromptBuilder(FrameworkPromptBuilder):
                def get_role_definition(self) -> str:
                    return "You are a data analysis specialist."

                def get_instructions(self) -> str:
                    return "Analyze the provided data and extract key insights."

        Advanced prompt with dynamic context::

            class ContextualPromptBuilder(FrameworkPromptBuilder):
                def get_role_definition(self) -> str:
                    return "You are an ALS accelerator operations expert."

                def get_instructions(self) -> str:
                    return "Provide technical analysis using ALS-specific terminology."

                def _get_dynamic_context(self, **kwargs) -> Optional[str]:
                    if 'device_status' in kwargs:
                        return f"Current device status: {kwargs['device_status']}"
                    return None

    .. seealso::
       :class:`FrameworkPromptProvider` : Provider interface for registering prompt builders
       :func:`get_framework_prompts` : Access registered prompt builders
       :func:`debug_print_prompt` : Standalone debug function for prompt inspection
       :doc:`/developer-guides/03_core-framework-systems/04_prompt-customization` : Guide for customizing framework prompts
    """

    @abstractmethod
    def get_role_definition(self) -> str:
        """Define the AI agent's role and primary identity for the prompt.

        This method provides the foundational "You are..." statement that establishes
        the agent's expertise, perspective, and behavioral context. This is always
        required and forms the first component of every generated prompt.

        :return: Role definition string that establishes agent identity and expertise
        :rtype: str

        Examples:
            Generic framework role::

                return "You are an expert execution planner for the assistant system."

            Domain-specific role::

                return "You are an ALS accelerator operations specialist with expertise in beam diagnostics."
        """
        pass

    def get_task_definition(self) -> str | None:
        """Define the specific task or objective for the prompt.

        This method provides an explicit task statement when the role definition
        alone is insufficient to establish the prompt's purpose. Can be omitted
        if the task is embedded within the role definition or instructions.

        :return: Task definition string or None if task is embedded elsewhere
        :rtype: Optional[str]

        Examples:
            Explicit task definition::

                return "TASK: Create a detailed execution plan for the user's request."

            No separate task (embedded in role)::

                return None
        """
        return None

    @abstractmethod
    def get_instructions(self) -> str:
        """Provide detailed instructions for how the agent should perform its task.

        This method contains the core operational guidance that tells the agent
        exactly how to approach and execute its assigned task. This is always
        required and typically contains the most detailed content of the prompt.

        :return: Comprehensive instructions for task execution
        :rtype: str

        Examples:
            Structured instructions with guidelines::

                return textwrap.dedent('''
                    Follow these steps:
                    1. Analyze the user's request for key requirements
                    2. Identify necessary data sources and dependencies
                    3. Create a step-by-step execution plan

                    Guidelines:
                    - Be specific and actionable
                    - Consider error handling scenarios
                    - Optimize for efficiency
                    ''')
        """
        pass

    def _get_examples(self, **kwargs) -> list["BaseExample"] | None:
        """Provide few-shot examples to guide agent behavior and output format.

        This method can return static examples or generate dynamic examples based
        on the provided context. Examples are particularly valuable for structured
        output tasks, complex reasoning patterns, or domain-specific formatting.

        This is an internal method called only by get_system_instructions().
        Subclasses can override this to provide custom examples.

        :param kwargs: Context parameters for dynamic example generation
        :type kwargs: dict
        :return: List of examples or None if no examples needed
        :rtype: Optional[List[BaseExample]]

        Examples:
            Static examples::

                def _get_examples(self, **kwargs):
                    return [
                        TaskExtractionExample(
                            input="Show me yesterday's beam current data",
                            output="Extract beam current measurements from yesterday"
                        )
                    ]

            Dynamic examples based on context::

                def _get_examples(self, **kwargs):
                    if kwargs.get('task_type') == 'data_analysis':
                        return self._get_analysis_examples()
                    return self._get_general_examples()
        """
        return None

    def _get_dynamic_context(self, **kwargs) -> str | None:
        """Inject runtime context information into the prompt.

        This method allows prompts to incorporate dynamic information such as
        current system state, user preferences, or execution context. The context
        is appended to the prompt after all other components.

        This is an internal method called only by get_system_instructions().
        Subclasses can override this to provide custom dynamic context.

        :param kwargs: Runtime context data for prompt customization
        :type kwargs: dict
        :return: Dynamic context string or None if no context needed
        :rtype: Optional[str]

        Examples:
            System status context::

                def _get_dynamic_context(self, **kwargs):
                    if 'system_status' in kwargs:
                        return f"Current system status: {kwargs['system_status']}"
                    return None

            User preferences context::

                def _get_dynamic_context(self, **kwargs):
                    prefs = kwargs.get('user_preferences', {})
                    if prefs:
                        return f"User preferences: {prefs}"
                    return None
        """
        return None

    def get_system_instructions(self, **context) -> str:
        """Compose and return complete system instructions for agent/LLM configuration.

        This method orchestrates the prompt building process by combining all prompt
        components in the correct order: role, task, instructions, examples, and
        dynamic context. It handles optional components gracefully and automatically
        integrates with the framework's debug system for development visibility.

        The composition follows this structure:
        1. Role definition (always present)
        2. Task definition (optional)
        3. Instructions (always present)
        4. Examples (optional, can be static or dynamic)
        5. Dynamic context (optional)

        :param context: Runtime context data passed to dynamic methods
        :type context: dict
        :return: Complete system prompt ready for LLM consumption
        :rtype: str

        .. note::
           This method automatically calls debug_print_prompt() if debug output
           is enabled in the configuration. No manual debug calls are needed.

        Examples:
            Basic usage in framework infrastructure::

                prompt_builder = get_framework_prompts().get_orchestrator_prompt_builder()
                system_prompt = prompt_builder.get_system_instructions(
                    capabilities=active_capabilities,
                    context_manager=context_manager
                )

            With dynamic context injection::

                system_prompt = builder.get_system_instructions(
                    user_preferences={'format': 'detailed'},
                    system_status='operational',
                    available_data_sources=['archiver', 'logbook']
                )

        .. seealso::
           :meth:`debug_print_prompt` : Debug output for prompt development
           :meth:`_format_examples` : Custom example formatting override
        """
        sections = []

        # Role (always present)
        sections.append(self.get_role_definition())

        # Task (optional)
        task = self.get_task_definition()
        if task:
            sections.append(task)

        # Instructions (always present)
        instructions = self.get_instructions()
        sections.append(instructions)

        # Examples (optional, can be static or dynamic)
        examples = self._get_examples(**context)
        if examples:
            formatted_examples = self._format_examples(examples)
            sections.append(f"EXAMPLES:\n{formatted_examples}")

        # Dynamic context (optional)
        dynamic_context = self._get_dynamic_context(**context)
        if dynamic_context:
            sections.append(dynamic_context)

        final_prompt = "\n\n".join(sections)

        # Debug: Print system instructions if enabled (automatic for all framework prompts)
        self.debug_print_prompt(final_prompt)

        return final_prompt

    def debug_print_prompt(self, prompt: str, name: str | None = None) -> None:
        """Output prompt content for debugging and development visibility.

        This method integrates with the framework's development configuration to
        provide optional prompt debugging through console output and file saving.
        It's automatically called by get_system_instructions() but can also be
        used manually for specialized prompts or intermediate prompt stages.

        The debug output includes metadata such as timestamp, builder class,
        and configuration settings, making it easy to trace prompt generation
        during development and troubleshooting.

        :param prompt: The complete prompt text to output for debugging
        :type prompt: str
        :param name: Custom name for the prompt in debug output. If not provided,
                    uses class-based default naming
        :type name: Optional[str]

        .. note::
           Debug output is controlled by development.prompts configuration:
           - show_all: Enable console output with detailed formatting
           - print_all: Enable file output to prompts directory
           - latest_only: Control file naming (latest.md vs timestamped)

        Examples:
            Automatic usage (called by get_system_instructions)::

                # Debug output happens automatically
                system_prompt = builder.get_system_instructions()

            Manual usage for specialized prompts::

                classification_prompt = "Classify this task..."
                builder.debug_print_prompt(classification_prompt, "task_classification")

        .. seealso::
           :func:`debug_print_prompt` : Standalone debug function
           :meth:`_get_default_prompt_name` : Default naming logic
        """
        prompt_name = name or self._get_default_prompt_name()
        debug_print_prompt(prompt, prompt_name, self.__class__.__name__)

    def _get_default_prompt_name(self) -> str:
        """Generate default name for prompt debugging based on class metadata.

        This method provides consistent naming for debug output by using either
        the PROMPT_TYPE class attribute (if defined) or deriving a name from
        the class name by removing 'PromptBuilder' suffix and converting to lowercase.

        :return: Default prompt name for debug output
        :rtype: str

        Examples:
            With PROMPT_TYPE attribute::

                class CustomPromptBuilder(FrameworkPromptBuilder):
                    PROMPT_TYPE = "custom_analysis"
                # Returns: "custom_analysis"

            Without PROMPT_TYPE attribute::

                class DataAnalysisPromptBuilder(FrameworkPromptBuilder):
                    pass
                # Returns: "dataanalysis"
        """
        return getattr(
            self, "PROMPT_TYPE", self.__class__.__name__.replace("PromptBuilder", "").lower()
        )

    def _format_examples(self, examples: list["BaseExample"]) -> str:
        """Format example objects into prompt-ready text representation.

        This method converts a list of BaseExample objects into a formatted string
        suitable for inclusion in the final prompt. The default implementation
        calls format_for_prompt() on each example and joins them with newlines.

        This is an internal method called only by get_system_instructions().
        Subclasses can override this method to provide custom formatting.

        :param examples: List of example objects to format
        :type examples: List[BaseExample]
        :return: Formatted examples string ready for prompt inclusion
        :rtype: str

        Examples:
            Default formatting::

                examples = [TaskExample(input="...", output="...")]
                formatted = self._format_examples(examples)
                # Returns: "Input: ...\nOutput: ...\n"

            Custom formatting override::

                def _format_examples(self, examples):
                    formatted = []
                    for i, ex in enumerate(examples, 1):
                        formatted.append(f"Example {i}:\n{ex.format_for_prompt()}")
                    return "\n\n".join(formatted)

        .. seealso::
           :class:`BaseExample` : Base class for prompt examples
           :meth:`BaseExample.format_for_prompt` : Individual example formatting
        """
        return "\n".join(ex.format_for_prompt() for ex in examples)

    def get_orchestrator_guide(self) -> Optional["OrchestratorGuide"]:
        """Provide orchestrator planning guidance for capability-specific prompts.

        This method allows prompt builders to provide structured guidance to the
        orchestrator about how to plan and execute tasks related to their specific
        capability or domain. The guide includes detailed instructions, rich examples,
        and priority settings for capability coordination.

        :return: Orchestrator guide object or None if no guidance needed
        :rtype: Optional[OrchestratorGuide]

        .. note::
           This method is primarily used by capability-specific prompt builders
           rather than infrastructure prompt builders.

        Examples:
            Capability-specific orchestrator guidance::

                def get_orchestrator_guide(self):
                    return OrchestratorGuide(
                        instructions="Use for data analysis when user requests statistical analysis or data validation",
                        examples=[
                            OrchestratorExample(
                                step=PlannedStep(
                                    context_key="analysis_results",
                                    capability="data_analysis",
                                    task_objective="Analyze provided data for trends and anomalies",
                                    success_criteria="Statistical analysis complete",
                                    expected_output="ANALYSIS_RESULTS",
                                    inputs=[{"DATA_SET": "numerical_data"}]
                                ),
                                scenario_description="When user requests data analysis or trend identification",
                                context_requirements={"DATA_SET": "Numerical data for analysis"}
                            )
                        ],
                        priority=10
                    )

        .. seealso::
           :class:`OrchestratorGuide` : Structure for orchestrator guidance
           :class:`PlannedStep` : Structure for planned steps
           :meth:`get_classifier_guide` : Related classifier guidance
        """
        return None

    def get_classifier_guide(self) -> Optional["TaskClassifierGuide"]:
        """Provide task classification guidance for capability-specific prompts.

        This method allows prompt builders to provide structured guidance to the
        task classifier about when and how their associated capability should be
        selected for different types of tasks. This helps ensure accurate capability
        routing in multi-capability scenarios.

        :return: Classifier guide object or None if no guidance needed
        :rtype: Optional[TaskClassifierGuide]

        .. note::
           This method is primarily used by capability-specific prompt builders
           to improve task routing accuracy.

        Examples:
            Capability-specific classification guidance::

                def get_classifier_guide(self):
                    return TaskClassifierGuide(
                        instructions="Determine if the task involves data analysis or trend identification requests",
                        examples=[
                            ClassifierExample(
                                query="Analyze the beam current trends",
                                result=True,
                                reason="Request for data analysis requiring trend identification"
                            ),
                            ClassifierExample(
                                query="What time is it?",
                                result=False,
                                reason="Simple information request, not data analysis"
                            )
                        ],
                        actions_if_true=ClassifierActions()
                    )

        .. seealso::
           :class:`TaskClassifierGuide` : Structure for classification guidance
           :class:`ClassifierExample` : Structure for classifier examples
           :meth:`get_orchestrator_guide` : Related orchestrator guidance
        """
        return None


def debug_print_prompt(prompt: str, name: str, builder_class: str | None = None) -> None:
    """Output prompt content for debugging across the entire framework.

    This standalone function provides prompt debugging capabilities that can be
    used anywhere in the framework, not just within prompt builders. It supports
    both console output with detailed formatting and file output with metadata
    headers, making it invaluable for development and troubleshooting.

    The function integrates with the development configuration system to provide
    flexible debugging options. It handles configuration errors gracefully to
    ensure debugging never breaks main functionality.

    :param prompt: The complete prompt text to output for debugging
    :type prompt: str
    :param name: Descriptive name for the prompt, used in console output and filename
    :type name: str
    :param builder_class: Optional builder class name for metadata and tracing
    :type builder_class: Optional[str]

    .. note::
       Debug output is controlled by development.prompts configuration:

       - show_all: Enable detailed console output with separators
       - print_all: Enable file output to configured prompts directory
       - latest_only: Use latest.md filenames vs timestamped files

    .. warning::
       This function silently handles configuration errors to prevent debugging
       from breaking main application functionality. Check logs for debug issues.

    Examples:
        Basic debugging from any framework component::

            debug_print_prompt(
                "You are a data analyst...",
                "custom_analysis",
                "CustomAnalysisBuilder"
            )

        Infrastructure component debugging::

            classification_prompt = build_classification_prompt(task)
            debug_print_prompt(
                classification_prompt,
                "task_classification",
                "OrchestrationNode"
            )

        Development workflow integration::

            # Enable in config.yml:
            # development:
            #   prompts:
            #     show_all: true
            #     print_all: true
            #     latest_only: false

    .. seealso::
       :meth:`FrameworkPromptBuilder.debug_print_prompt` : Builder-specific debug method
       :func:`get_config_value` : Configuration system integration
       :func:`get_agent_dir` : Directory management for file output
    """
    try:
        development_config = get_config_value("development", {})
        prompts_config = development_config.get("prompts", {})

        # Console output
        if prompts_config.get("show_all", False):
            builder_info = f" ({builder_class})" if builder_class else ""

            # Print with clear separators for readability
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 DEBUG PROMPT: {name}{builder_info}")
            logger.info(f"{'='*80}")
            logger.info(prompt)
            logger.info(f"{'='*80}\n")

            # Also log at debug level
            logger.debug(f"Generated prompt: {name}{builder_info}")

        # File output
        if prompts_config.get("print_all", False):
            prompts_dir = get_agent_dir("prompts_dir")
            os.makedirs(prompts_dir, exist_ok=True)

            # Determine filename based on latest_only flag
            latest_only = prompts_config.get("latest_only", True)
            if latest_only:
                filename = f"{name}_latest.md"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.md"

            prompt_file_path = os.path.join(prompts_dir, filename)

            # Create header with metadata
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = textwrap.dedent(
                f"""
                # PROMPT METADATA
                # Generated: {timestamp_str}
                # Name: {name}
                # Builder: {builder_class or 'Unknown'}
                # File: {prompt_file_path}
                # Latest Only: {latest_only}
                """
            ).strip()

            with open(prompt_file_path, "w") as f:
                f.write(header + "\n\n\n" + prompt)
            logger.debug(f"Prompt written to {prompt_file_path}")

    except Exception as e:
        # Silently continue if config is not available or there's an error
        # This ensures the debug function never breaks the main functionality
        logger.warning(f"Error displaying prompt: {e}")
        pass
