"""
Clarify Capability

This capability asks users specific questions when their queries are ambiguous
or missing critical information needed for execution. It helps refine user
intent before proceeding with technical operations.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from osprey.base import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.models import get_chat_completion
from osprey.prompts.loader import get_framework_prompts
from osprey.utils.config import get_model_config
from osprey.utils.logger import get_logger

logger = logging.getLogger(__name__)


# --- Pydantic Model for Clarifying Questions ---


class ClarifyingQuestionsResponse(BaseModel):
    """Structured response for clarifying questions.

    Contains the reason for clarification and specific questions to ask
    the user to gather missing information.

    :param reason: Brief explanation of why clarification is needed
    :type reason: str
    :param questions: List of specific, targeted questions to clarify the user's request
    :type questions: List[str]
    :param missing_info: List of types of missing information
    :type missing_info: List[str]
    """

    reason: str = Field(description="Brief explanation of why clarification is needed")
    questions: list[str] = Field(
        description="List of specific, targeted questions to clarify the user's request",
        min_length=1,
        max_length=4,
    )
    missing_info: list[str] = Field(
        description="List of types of missing information (e.g., 'time_range', 'system_specification')",
        default=[],
    )


# --- Convention-Based Capability Definition ---


@capability_node
class ClarifyCapability(BaseCapability):
    """Ask user for clarification when queries are ambiguous.

    Communication capability that generates targeted questions to clarify
    user intent when requests lack sufficient detail or context.
    """

    name = "clarify"
    description = (
        "Ask specific questions when user queries are ambiguous or missing critical details"
    )
    provides = (
        []
    )  # Communication capability - no context output (questions go to user via chat history)
    requires = []  # Can work with any execution context

    async def execute(self) -> dict[str, Any]:
        """Generate specific questions to ask user based on missing information.

        :return: State update with clarification response
        :rtype: Dict[str, Any]
        """
        state = self._state

        # Explicit logger retrieval - professional practice
        logger = get_logger("clarify")

        # Extract task objective using helper method
        task_objective = self.get_task_objective(default="unknown")

        logger.info(f"Clarification task objective: {task_objective}")

        try:
            logger.info("Starting clarification generation")

            # Use get_stream_writer() for pure LangGraph streaming
            streaming = get_stream_writer()

            if streaming:
                streaming(
                    {
                        "event_type": "status",
                        "message": "Analyzing query for clarification...",
                        "progress": 0.2,
                    }
                )

            # Generate clarifying questions using PydanticAI
            # Run sync function in thread pool to avoid blocking event loop for streaming
            questions_response = await asyncio.to_thread(
                _generate_clarifying_questions, state, task_objective
            )

            if streaming:
                streaming(
                    {
                        "event_type": "status",
                        "message": "Generating clarification questions...",
                        "progress": 0.6,
                    }
                )

            # Format questions for user interaction
            formatted_questions = _format_questions_for_user(questions_response)

            if streaming:
                streaming(
                    {
                        "event_type": "status",
                        "message": "Clarification ready",
                        "progress": 1.0,
                        "complete": True,
                    }
                )

            logger.info(f"Generated {len(questions_response.questions)} clarifying questions")

            # Return clarifying questions using native LangGraph pattern
            return {"messages": [AIMessage(content=formatted_questions)]}

        except Exception as e:
            logger.error(f"Error generating clarifying questions: {e}")
            # Let the decorator handle error classification
            raise

    # Optional: Add error classification if needed
    @staticmethod
    def classify_error(exc: Exception, context: dict):
        """Clarify error classification."""

        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Failed to generate clarifying questions: {str(exc)}",
            metadata={"technical_details": str(exc)},
        )

    def _create_orchestrator_guide(self):
        """Get orchestrator snippet from prompt builder.

        :return: Orchestrator prompt snippet for this capability
        """

        prompt_provider = get_framework_prompts()  # Registry will determine the right provider
        clarification_builder = prompt_provider.get_clarification_prompt_builder()

        return clarification_builder.get_orchestrator_guide()

    def _create_classifier_guide(self):
        """Get classifier config from prompt builder.

        :return: Classifier configuration for this capability
        """

        prompt_provider = get_framework_prompts()  # Registry will determine the right provider
        clarification_builder = prompt_provider.get_clarification_prompt_builder()

        return clarification_builder.get_classifier_guide()


# --- Helper Functions ---


def _generate_clarifying_questions(state, task_objective: str) -> ClarifyingQuestionsResponse:
    """Generate specific clarifying questions using PydanticAI.

    :param state: Current agent state
    :param task_objective: The orchestrator's clarification instruction
    :type task_objective: str
    :return: Structured clarifying questions response
    :rtype: ClarifyingQuestionsResponse
    """

    # Get clarification prompt builder from framework
    prompt_provider = get_framework_prompts()
    clarification_builder = prompt_provider.get_clarification_prompt_builder()

    # Use prompt builder's get_system_instructions() which handles all composition
    # This extracts runtime data (user_query, chat_history, context) from state
    # and composes the complete prompt with PRIMARY TASK prioritization
    message = clarification_builder.get_system_instructions(state, task_objective)

    response_config = get_model_config("response")
    result = get_chat_completion(
        message=message, model_config=response_config, output_model=ClarifyingQuestionsResponse
    )

    return result


def _format_questions_for_user(questions_response: ClarifyingQuestionsResponse) -> str:
    """Format clarifying questions for direct user interaction.

    :param questions_response: The structured questions response
    :type questions_response: ClarifyingQuestionsResponse
    :return: Formatted questions string for user
    :rtype: str
    """
    questions_text = "I need some clarification:\n\n"
    for i, question in enumerate(questions_response.questions, 1):
        questions_text += f"{i}. {question}\n"
    return questions_text
