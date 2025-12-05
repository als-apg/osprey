"""Prompt-based Capability Generator for Osprey Framework.

Generates capability scaffolding from natural language descriptions.
Creates classifier and orchestrator guides, with placeholder business logic.
"""

from .base_generator import BaseCapabilityGenerator
from .models import CapabilityMetadata, ClassifierAnalysis, OrchestratorAnalysis

# =============================================================================
# Prompt Capability Generator
# =============================================================================


class PromptCapabilityGenerator(BaseCapabilityGenerator):
    """Generate capability scaffolding from natural language prompt."""

    def __init__(
        self,
        prompt: str,
        capability_name: str | None = None,
        verbose: bool = False,
        provider: str | None = None,
        model_id: str | None = None,
    ):
        """Initialize generator.

        Args:
            prompt: Natural language description of what the capability should do
            capability_name: Optional name for the capability (will be suggested by LLM if not provided)
            verbose: Whether to print progress messages
            provider: Optional LLM provider override
            model_id: Optional model ID override
        """
        # Initialize with empty name if not provided (will be set from metadata)
        super().__init__(capability_name or "", verbose, provider, model_id)
        self.prompt = prompt
        self.metadata: CapabilityMetadata | None = None

    async def generate_metadata(self) -> CapabilityMetadata:
        """Generate capability metadata from prompt.

        Returns:
            Metadata with suggested names and description

        Raises:
            RuntimeError: If generation fails
        """
        if self.verbose:
            print("\n🤖 Analyzing prompt and generating metadata...")

        metadata_prompt = f"""You are an expert at analyzing capability requirements and generating appropriate names and metadata.

A user wants to create an Osprey capability with the following description:

{self.prompt}

Your task: Analyze this description and generate appropriate metadata.

Generate:
- A snake_case capability name (e.g., 'weather_data', 'slack_messages', 'database_query')
- A brief, clear description (1-2 sentences)
- A suggested UPPER_SNAKE_CASE context type for the output (e.g., 'WEATHER_DATA', 'SLACK_MESSAGES', 'QUERY_RESULTS')

Keep names concise but descriptive. Follow Python naming conventions.
Output as JSON matching the CapabilityMetadata schema.
"""

        metadata = await self._call_llm(metadata_prompt, CapabilityMetadata)
        self.metadata = metadata

        # Update capability_name if it was not provided
        if not self.capability_name:
            self.capability_name = metadata.capability_name_suggestion

        if self.verbose:
            print(f"✓ Suggested name: {metadata.capability_name_suggestion}")
            print(f"✓ Context type: {metadata.context_type_suggestion}")

        return metadata

    async def generate_guides(self) -> tuple[ClassifierAnalysis, OrchestratorAnalysis]:
        """Generate classifier and orchestrator guides using LLM.

        Uses the configured orchestrator model (or overrides if specified)
        to analyze the prompt and generate activation guides.

        Returns:
            Tuple of (classifier_analysis, orchestrator_analysis)

        Raises:
            RuntimeError: If generation fails
        """
        if self.verbose:
            print("\n🤖 Generating classifier and orchestrator guides...")

        # Use capability name from metadata if available
        capability_name = self.capability_name or (
            self.metadata.capability_name_suggestion if self.metadata else "unnamed_capability"
        )

        # Generate classifier analysis
        classifier_prompt = f"""You are an expert at analyzing capability requirements and generating task classification rules.

A user wants to create a capability with the following description:

{self.prompt}

The capability will be called: {capability_name}

Your task: Analyze this description and generate a comprehensive classifier guide.

Generate:
- Clear activation criteria (when should this capability be triggered?)
- Key terms/patterns that indicate this capability is needed
- 5-7 realistic positive examples (queries that SHOULD activate) with reasoning
- 3-4 realistic negative examples (queries that SHOULD NOT activate) with reasoning
- Edge cases to watch for

Make the examples natural and varied - think about real users asking questions.
Consider different phrasings and ways users might express the same intent.
Output as JSON matching the ClassifierAnalysis schema.
"""

        classifier_analysis = await self._call_llm(classifier_prompt, ClassifierAnalysis)

        # Generate orchestrator analysis
        orchestrator_prompt = f"""You are an expert at high-level task planning for capability orchestration.

A user wants to create a capability with the following description:

{self.prompt}

The capability will be called: {capability_name}

Your task: Generate a SIMPLE orchestrator guide for HIGH-LEVEL planning only.

The orchestrator should know:
- When to invoke the {capability_name} capability (what types of user requests)
- How to formulate clear task_objective descriptions
- General patterns for using this capability

Generate:
- Clear "when to use" guidance
- 3-5 example scenarios showing WHAT users might ask for (not implementation details)
- Each example should have a clear task_objective that describes the goal
- **IMPORTANT**: Each example MUST have a descriptive context_key that captures the essence of the step
  - Good: "weather_sf_current", "slack_channel_messages", "database_user_query"
  - Bad: "result_1", "data", "output"
  - The context_key should be specific, descriptive, and use snake_case
- Common sequences or patterns for this capability
- Important notes about formulating good task objectives

Focus on WHAT to accomplish, not HOW to implement it.
Output as JSON matching the OrchestratorAnalysis schema.
"""

        orchestrator_analysis = await self._call_llm(orchestrator_prompt, OrchestratorAnalysis)

        if self.verbose:
            print("✓ Guides generated")

        return classifier_analysis, orchestrator_analysis

    def generate_capability_code(
        self, classifier_analysis: ClassifierAnalysis, orchestrator_analysis: OrchestratorAnalysis
    ) -> str:
        """Generate capability Python code with placeholder business logic.

        Args:
            classifier_analysis: Classifier guide analysis from LLM
            orchestrator_analysis: Orchestrator guide analysis from LLM

        Returns:
            Complete Python source code for the capability
        """
        timestamp = self._get_timestamp()

        # Use metadata if available, otherwise use provided capability_name or generate from prompt
        if self.metadata:
            capability_name = self.capability_name or self.metadata.capability_name_suggestion
            description = self.metadata.description
            context_type = self.metadata.context_type_suggestion
        else:
            capability_name = self.capability_name or "unnamed_capability"
            description = "TODO: Add description"
            context_type = "TODO_CONTEXT_TYPE"

        class_name = self._to_class_name(capability_name)
        context_class_name = self._to_class_name(capability_name, suffix="Context")

        # Build examples using base class methods
        classifier_examples_code = self._build_classifier_examples_code(classifier_analysis)
        orchestrator_examples_code = self._build_orchestrator_examples_code(
            orchestrator_analysis, context_type
        )

        code = f'''"""
{description}

Auto-generated capability scaffolding for Osprey Framework.
Generated: {timestamp}

IMPORTANT: This is a SKELETON/PLACEHOLDER capability generated from a prompt.
You MUST implement the actual business logic before using this capability.

What was generated:
- Classifier guide (when to activate this capability) ✓
- Orchestrator guide (how to plan steps) ✓
- Capability class structure with placeholder execute() method
- Context class structure
- Error handling structure

What you NEED to implement:
- [ ] The execute() method with actual business logic
- [ ] Customize the context class to match your actual data structure
- [ ] Update provides/requires fields based on actual dependencies
- [ ] Implement proper error handling
- [ ] Add any necessary imports and dependencies

Original prompt:
{self.prompt}

Generated by: osprey generate capability --from-prompt
"""

from __future__ import annotations
from typing import Dict, Any, Optional, ClassVar, TYPE_CHECKING
import textwrap

if TYPE_CHECKING:
    from osprey.state import AgentState

# Framework imports
from osprey.base.decorators import capability_node
from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.planning import PlannedStep
from osprey.base.examples import OrchestratorGuide, OrchestratorExample, TaskClassifierGuide, ClassifierExample, ClassifierActions
from osprey.context import CapabilityContext
from osprey.state import StateManager


# =============================================================================
# Context Class - CUSTOMIZE THIS FOR YOUR USE CASE
# =============================================================================

class {context_class_name}(CapabilityContext):
    """
    Context for {capability_name} results.

    TODO: Customize this class based on your actual data structure.
    This is a placeholder - define the actual fields your capability will produce.

    📚 Documentation: For detailed guidance on creating context classes, see:
    https://als-apg.github.io/osprey/developer-guides/03_core-framework-systems/02_context-management-system.html
    """

    CONTEXT_TYPE: ClassVar[str] = "{context_type}"
    CONTEXT_CATEGORY: ClassVar[str] = "TODO"  # Choose: EXTERNAL_DATA, COMPUTED_STATE, USER_INPUT, SYSTEM_STATE, PLANNING, EPHEMERAL

    # TODO: Define your actual fields here
    # Example fields:
    # data: Dict[str, Any]
    # status: str
    # metadata: Dict[str, Any]

    def get_access_details(self, key: str) -> Dict[str, Any]:
        """Tell the LLM how to access this context data.

        TODO: Customize this to describe how to access your actual data fields.
        """
        return {{
            "description": "TODO: Describe what this context contains",
            "data_structure": "TODO: Describe the structure",
            "access_pattern": f"context.{{self.CONTEXT_TYPE}}.{{key}}.YOUR_FIELD_NAME",
            "available_fields": "TODO: List available fields",
        }}

    def get_summary(self) -> Dict[str, Any]:
        """Format data for human display.

        TODO: Customize this to format your actual data for display.
        """
        return {{
            "type": "{capability_name} results",
            "data": "TODO: Add your actual data here",
        }}


# =============================================================================
# Error Classes
# =============================================================================

class {class_name}Error(Exception):
    """Base error for {capability_name} operations."""
    pass


# =============================================================================
# Capability Implementation
# =============================================================================

@capability_node
class {class_name}(BaseCapability):
    """
    {description}

    TODO: Implement the actual business logic in the execute() method.
    """

    name = "{capability_name}"
    description = "{description}"

    # TODO: Update these based on what your capability actually provides and requires
    provides = ["{context_type}"]
    requires = []  # TODO: Add any context types this capability depends on

    async def execute(self) -> Dict[str, Any]:
        """Execute {capability_name} capability.

        TODO: IMPLEMENT THE ACTUAL BUSINESS LOGIC HERE.

        This is a placeholder. You need to:
        1. Extract task_objective and any inputs using self.get_task_objective()
        2. Extract required contexts using self.get_required_contexts()
        3. Perform the actual work
        4. Create a proper context object with results
        5. Store it in state using self.store_output_context()

        Returns:
            State updates dictionary
        """
        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Extract task objective (with automatic fallback to current task)
        task_objective = self.get_task_objective()

        logger.status(f"Processing {{task_objective}}...")

        try:
            # TODO: IMPLEMENT YOUR ACTUAL BUSINESS LOGIC HERE
            # This is just a placeholder structure

            # Example placeholder implementation:
            # 1. Do some work based on task_objective
            result_data = {{
                "status": "placeholder",
                "message": "TODO: Implement actual logic",
                "task": task_objective
            }}

            # 2. Create context with results
            context = {context_class_name}(
                # TODO: Fill in with actual data
                # data=result_data,
                # status="success",
                # etc.
            )

            # 3. Store in state using new pattern and return updates
            logger.status(f"{capability_name} complete")
            return self.store_output_context(context)

        except Exception as e:
            error_msg = f"{capability_name} failed: {{str(e)}}"
            logger.error(error_msg)
            raise {class_name}Error(error_msg) from e

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify {capability_name} errors.

        TODO: Customize error classification based on your actual error types.
        """
        if isinstance(exc, {class_name}Error):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"{capability_name} operation failed: {{str(exc)}}",
                metadata={{
                    "technical_details": str(exc),
                    "replanning_reason": f"{capability_name} execution failed"
                }}
            )
        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected {capability_name} error: {{exc}}",
                metadata={{
                    "technical_details": str(exc),
                    "safety_abort_reason": "Unhandled error"
                }}
            )

    def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
        """Classifier guide: When should this capability be activated?"""
        return TaskClassifierGuide(
            instructions=textwrap.dedent("""
                {classifier_analysis.activation_criteria}

                Activate if the query involves:
                {chr(10).join('- ' + kw for kw in classifier_analysis.keywords[:10])}

                Edge cases to consider:
                {chr(10).join('- ' + case for case in classifier_analysis.edge_cases[:5])}
            """).strip(),
            examples=[
{classifier_examples_code}
            ],
            actions_if_true=ClassifierActions()
        )

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
        """Orchestrator guide: How should steps be planned?"""
        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "{capability_name}" steps:**
                {orchestrator_analysis.when_to_use}

                **Step Structure:**
                - context_key: Descriptive identifier (e.g., "{capability_name}_something_specific")
                - capability: "{capability_name}"
                - task_objective: Clear description of WHAT the user wants
                - expected_output: "{context_type}"

                **Common Patterns:**
                {chr(10).join('- ' + pattern for pattern in orchestrator_analysis.common_sequences[:5])}

                **Important Notes:**
                {chr(10).join('- ' + note for note in orchestrator_analysis.important_notes[:5])}

                **Output:** {context_type}
                Contains results from the {capability_name} capability.
            """).strip(),
            examples=[
{orchestrator_examples_code}
            ],
            priority=2
        )


# =============================================================================
# Registry Registration
# =============================================================================
"""
Add this to your registry.py:

from osprey.registry import RegistryConfigProvider, extend_framework_registry
from osprey.registry.base import CapabilityRegistration, ContextClassRegistration

class MyAppRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="{capability_name}",
                    module_path="your_app.capabilities.{capability_name}",
                    class_name="{class_name}",
                    provides=["{context_type}"],
                    requires=[]  # TODO: Update based on actual dependencies
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="{context_type}",
                    module_path="your_app.capabilities.{capability_name}",
                    class_name="{context_class_name}"
                ),
            ]
        )
"""
'''

        return code
