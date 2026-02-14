"""
Memory Extraction Prompt Builder - Application-agnostic prompts for memory extraction
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Imports for orchestrator and classifier guides
from osprey.base import (
    BaseExample,
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    PlannedStep,
    TaskClassifierGuide,
)
from osprey.registry import get_registry
from osprey.state import ChatHistoryFormatter, MessageUtils

from ..base import FrameworkPromptBuilder


class MemoryContentExtraction(BaseModel):
    """Structured output model for memory content extraction."""

    content: str = Field(
        description="The content that should be saved to memory, or empty string if no content identified"
    )
    found: bool = Field(
        description="True if content to save was identified in the user message, False otherwise"
    )
    explanation: str = Field(description="Brief explanation of what content was extracted and why")


@dataclass
class MemoryExtractionExample(BaseExample):
    """Example for memory extraction prompt."""

    def __init__(self, messages: list[BaseMessage], expected_output: MemoryContentExtraction):
        self.messages = messages
        self.expected_output = expected_output

    def format_for_prompt(self) -> str:
        """Format this example for inclusion in prompts."""
        # Format chat history using native message formatter
        chat_formatted = ChatHistoryFormatter.format_for_llm(self.messages)

        return textwrap.dedent(
            f"""
            **Chat History:**
            {textwrap.indent(chat_formatted, "  ")}

            **Expected Output:**
            {{
                "content": "{self.expected_output.content}",
                "found": {str(self.expected_output.found).lower()},
                "explanation": "{self.expected_output.explanation}"
            }}
            """
        ).strip()


class DefaultMemoryExtractionPromptBuilder(FrameworkPromptBuilder):
    """Framework prompt builder for memory extraction.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change the task statement       | ``get_task()``                    |
    +---------------------------------+----------------------------------------------+
    | Change extraction instructions  | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Replace default examples        | ``load_examples()``                          |
    +---------------------------------+----------------------------------------------+
    | Supply different examples       | ``get_examples(**kwargs)``                   |
    +---------------------------------+----------------------------------------------+
    | Change example formatting       | ``format_examples(examples)``                |
    +---------------------------------+----------------------------------------------+
    | Change response format section  | ``build_dynamic_context(**kwargs)``           |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "memory_extraction"

    def __init__(self):
        super().__init__()
        self.examples = []
        self.load_examples()

    def load_examples(self):
        """Load memory extraction examples with native LangGraph messages."""

        # Explicit save instruction with quoted content
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        'Please save this finding: "Database performance degrades significantly when connection pool exceeds 50 connections - optimal range is 20-30 connections"'
                    )
                ],
                expected_output=MemoryContentExtraction(
                    content="Database performance degrades significantly when connection pool exceeds 50 connections - optimal range is 20-30 connections",
                    found=True,
                    explanation="User explicitly requested to save a specific finding with quantitative thresholds",
                ),
            )
        )

        # Technical insight with specific parameters
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        "I've been analyzing the server logs and found a pattern"
                    ),
                    MessageUtils.create_assistant_message("What pattern did you identify?"),
                    MessageUtils.create_user_message(
                        "Remember that API response times increase by 40% when memory usage exceeds 85% - this happens during peak hours between 2-4 PM"
                    ),
                ],
                expected_output=MemoryContentExtraction(
                    content="API response times increase by 40% when memory usage exceeds 85% - this happens during peak hours between 2-4 PM",
                    found=True,
                    explanation="Technical insight with specific performance metrics and timing patterns",
                ),
            )
        )

        # Procedural discovery with workflow details
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        "Store this procedure for future reference: Code deployments work best when done after 6 PM, with database migrations run first, then application restart, followed by cache clearing - allow 15 minutes between each step"
                    )
                ],
                expected_output=MemoryContentExtraction(
                    content="Code deployments work best when done after 6 PM, with database migrations run first, then application restart, followed by cache clearing - allow 15 minutes between each step",
                    found=True,
                    explanation="Detailed procedural workflow with timing and sequencing requirements",
                ),
            )
        )

        # Configuration insight with mixed save/don't save instructions
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        "Today's incident was caused by a timeout issue, but don't save that. However, do remember that we found the root cause: default timeout of 30 seconds is too short for large file uploads - increase to 120 seconds for files over 100MB"
                    )
                ],
                expected_output=MemoryContentExtraction(
                    content="default timeout of 30 seconds is too short for large file uploads - increase to 120 seconds for files over 100MB",
                    found=True,
                    explanation="User explicitly requested to save specific configuration finding while excluding incident details",
                ),
            )
        )

        # Negative example - routine status check (should not save)
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        "What's the current system status and how are the servers performing today?"
                    ),
                    MessageUtils.create_assistant_message(
                        "All systems are running normally with 99.2% uptime. Server load is within normal parameters."
                    ),
                    MessageUtils.create_user_message(
                        "Thanks, everything looks good for today's operations"
                    ),
                ],
                expected_output=MemoryContentExtraction(
                    content="",
                    found=False,
                    explanation="This is routine operational status checking and acknowledgment, not content intended for permanent memory",
                ),
            )
        )

        # Negative example - procedural question (should not save)
        self.examples.append(
            MemoryExtractionExample(
                messages=[
                    MessageUtils.create_user_message(
                        "How do I configure the load balancer to handle SSL termination?"
                    )
                ],
                expected_output=MemoryContentExtraction(
                    content="",
                    found=False,
                    explanation="This is a procedural question about system configuration, not content to be saved to memory",
                ),
            )
        )

    def get_role(self) -> str:
        """Get the role definition."""
        return "You are an expert content extraction assistant. Your task is to identify and extract content that a user wants to save to their memory from their message."

    def get_task(self) -> str:
        """Get the task definition."""
        return (
            "TASK: Extract content from the user's message that they want to save/store/remember."
        )

    def get_instructions(self) -> str:
        """Get the memory extraction instructions."""
        return textwrap.dedent(
            """
            INSTRUCTIONS:
            1. Analyze the user message to identify content they explicitly want to save
            2. Look for patterns like:
               - "save this:" followed by content
               - "remember that [content]"
               - "store [content] in my memory"
               - "add to memory: [content]"
               - Content in quotes that should be saved
               - Explicit instructions to save specific information
            3. Extract the ACTUAL CONTENT to be saved, not the instruction itself
            4. If no clear content is identified for saving, set found=false
            5. Provide a brief explanation of your decision

            CRITICAL REQUIREMENTS:
            - Only extract content that is clearly intended for saving/storage
            - Do not extract questions, commands, or conversational text
            - Remove quotes and prefixes like "save this:", "remember that", etc.
            - Extract the pure content to be remembered
            - Be conservative - if unclear, set found=false
            """
        ).strip()

    def get_examples(self, **kwargs) -> list[MemoryExtractionExample]:
        """Get generic memory extraction examples."""
        return self.examples

    def format_examples(self, examples: list[MemoryExtractionExample]) -> str:
        """Format multiple MemoryExtractionExample objects for inclusion in prompts."""
        return MemoryExtractionExample.join(examples, add_numbering=True)

    def build_dynamic_context(self, **kwargs) -> str:
        """Build the response format section."""
        return textwrap.dedent(
            """
            RESPOND WITH VALID JSON IN THIS EXACT FORMAT:
            {
                "content": "the exact content to save (or empty string if none found)",
                "found": true/false,
                "explanation": "brief explanation of your decision"
            }

            You will be provided with a chat history. Extract the content to save from the user message in that chat history.
            """
        ).strip()

    def get_orchestrator_guide(self) -> Any | None:
        """Create generic orchestrator guide for memory operations."""
        registry = get_registry()

        # Define structured examples
        save_memory_example = OrchestratorExample(
            step=PlannedStep(
                context_key="memory_save",
                capability="memory",
                task_objective="Save the important finding about data correlation to memory",
                expected_output=registry.context_types.MEMORY_CONTEXT,
                success_criteria="Memory entry saved successfully",
                inputs=[],
            ),
            scenario_description="Saving important information to user memory",
            notes=f"Content is persisted to memory file and provided as {registry.context_types.MEMORY_CONTEXT} context for response confirmation.",
        )

        show_memory_example = OrchestratorExample(
            step=PlannedStep(
                context_key="memory_display",
                capability="memory",
                task_objective="Show all my saved memory entries",
                expected_output=registry.context_types.MEMORY_CONTEXT,
                success_criteria="Memory content displayed to user",
                inputs=[],
            ),
            scenario_description="Displaying stored memory content",
            notes=f"Retrieves memory content as {registry.context_types.MEMORY_CONTEXT}. Typically followed by respond step to present results to user.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(
                f"""
                **When to plan "memory" steps:**
                - When the user explicitly asks to save, store, or remember something for later
                - When the user asks to show, display, or view their saved memory
                - When the user explicitly mentions memory operations

                **IMPORTANT**: This capability has a VERY STRICT classifier. Only use when users
                explicitly mention memory-related operations. Do NOT use for general information
                storage or context management.

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "memory_save", "memory_display")
                - task_objective: The specific memory operation to perform

                **Output: {registry.context_types.MEMORY_CONTEXT}**
                - Save operations: Contains saved content for response confirmation
                - Retrieve operations: Contains stored memory content for use by respond step
                - Available to downstream steps via context system

                Only plan this step when users explicitly request memory operations.
                """
            ),
            examples=[save_memory_example, show_memory_example],
            priority=10,  # Later in the prompt ordering since it's specialized
        )

    def get_classifier_guide(self) -> Any | None:
        """Create generic classifier guide for memory operations."""
        # Create generic memory-specific examples
        memory_examples = [
            ClassifierExample(
                query="Save this finding to my memory: database performance correlates with connection pool size",
                result=True,
                reason="Direct memory save request with specific content to preserve",
            ),
            ClassifierExample(
                query="Remember that cache optimization works best 15 minutes after restart",
                result=True,
                reason="Memory save request using 'remember' keyword with procedural knowledge",
            ),
            ClassifierExample(
                query="What do I have saved in my memory about performance tuning?",
                result=True,
                reason="Memory retrieval request asking to show stored information",
            ),
            ClassifierExample(
                query="Show me my saved notes",
                result=True,
                reason="Memory retrieval request for displaying saved content",
            ),
            ClassifierExample(
                query="Store this configuration procedure: set timeout at 120 seconds",
                result=True,
                reason="Explicit store request with specific technical procedure to save",
            ),
        ]

        return TaskClassifierGuide(
            instructions="Determine if the task involves saving content to memory or retrieving content from memory.",
            examples=memory_examples,
            actions_if_true=ClassifierActions(),
        )

    def get_memory_classification_prompt(self) -> str:
        """Get prompt for classifying memory operations as SAVE or RETRIEVE.

        Returns the system prompt used by LLM to classify user tasks into
        memory operation types. Includes automatic debug printing.

        Returns:
            System prompt string for memory operation classification
        """
        prompt = textwrap.dedent(
            """
            You are a memory operation classifier. Analyze the user's task and determine if they want to:
            - SAVE: Store new information to memory (e.g. user asks to save, store, remember, record, add, append)
            - RETRIEVE: Show existing memories (e.g. user asks to show, display, view, retrieve, see, list)

            Focus on the user's intent, not just keyword matching. Context matters.
            """
        ).strip()

        # Automatic debug printing for framework helper prompts
        self.debug_print_prompt(prompt, "memory_operation_classification")

        return prompt
