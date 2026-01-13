"""ReAct Agent for Machine State Identification.

This module provides a ReAct agent that assesses machine readiness for optimization.
The agent uses tools to:
1. Read reference documentation about machine ready criteria
2. Read current channel values from the control system
3. Determine if the machine is READY, NOT_READY, or UNKNOWN

The agent adapts based on available resources:
- Reference files can be mock (for testing) or real (from configured path)
- Channel access uses the existing control system connector (mock or real via config)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from osprey.models.langchain import get_langchain_model
from osprey.utils.logger import get_logger

from ..models import MachineState
from .tools import create_channel_access_tools, create_reference_file_tools

logger = get_logger("xopt_optimizer")


# =============================================================================
# STRUCTURED OUTPUT MODEL
# =============================================================================


class MachineStateAssessment(BaseModel):
    """Structured output for machine state assessment.

    This model is used with LangChain's `with_structured_output` for the
    final assessment after the agent has gathered information.
    """

    state: MachineState = Field(
        description="The assessed machine state: 'ready', 'not_ready', or 'unknown'"
    )
    reasoning: str = Field(
        description="Explanation of why this state was determined, including key observations"
    )
    channels_checked: list[str] = Field(
        default_factory=list,
        description="List of channel names that were checked during assessment",
    )
    key_observations: dict[str, Any] = Field(
        default_factory=dict,
        description="Key observations from channel readings and reference docs",
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

STATE_IDENTIFICATION_PROMPT = """You are a machine state assessment agent for accelerator optimization.

Your task is to determine if the machine is ready for optimization by:
1. Reading reference documentation to understand the ready criteria
2. Checking current channel values against those criteria
3. Providing a clear assessment with reasoning

## Your Workflow

1. **Read Documentation First**: Use `list_reference_files` to see available docs, then
   `read_reference_file` to understand the machine ready criteria.

2. **Check Channel Values**: Based on what you learn from the docs, use `read_channel_values`
   to check the relevant channels. Pass channel names as comma-separated values.

3. **Make Assessment**: Based on the criteria and current values, determine:
   - **READY**: All criteria are met, machine can proceed with optimization
   - **NOT_READY**: One or more criteria are not met, optimization should not proceed
   - **UNKNOWN**: Unable to determine state (missing data, conflicting info, etc.)

## Important Guidelines

- Always read the reference docs first to understand what criteria to check
- Check all relevant channels mentioned in the documentation
- Be conservative: if unsure, report UNKNOWN rather than guessing
- Include specific channel values in your reasoning
- List all channels you checked in your response

## Response Format

After gathering information, provide your assessment with:
- The machine state (ready/not_ready/unknown)
- Clear reasoning explaining your decision
- List of channels you checked
- Key observations from your investigation
"""


# =============================================================================
# AGENT CLASS
# =============================================================================


class StateIdentificationAgent:
    """ReAct agent for assessing machine readiness for optimization.

    This agent:
    1. Reads reference documentation about machine ready criteria
    2. Checks current channel values from the control system
    3. Determines if the machine is READY, NOT_READY, or UNKNOWN

    The tools adapt based on configuration:
    - mock_files=True: Uses hardcoded mock reference data
    - mock_files=False: Reads real files from reference_path
    - Channel access uses ConnectorFactory (mock or real via control_system.type config)
    """

    def __init__(
        self,
        reference_path: str | None = None,
        mock_files: bool = False,
        model_config: dict[str, Any] | None = None,
    ):
        """Initialize the state identification agent.

        Args:
            reference_path: Path to reference documentation directory.
                Ignored if mock_files=True.
            mock_files: If True, use mock reference file data for testing.
            model_config: Configuration for the LLM model to use.
        """
        self.reference_path = reference_path
        self.mock_files = mock_files
        self.model_config = model_config
        self._agent = None

    def _get_tools(self) -> list[Any]:
        """Get tools for the agent.

        Returns:
            List of LangChain tools for file reading and channel access
        """
        tools = []

        # Reference file tools (mock or real)
        tools.extend(
            create_reference_file_tools(
                reference_path=self.reference_path,
                mock_mode=self.mock_files,
            )
        )

        # Channel access tools (uses existing mock connector via config)
        tools.extend(create_channel_access_tools())

        return tools

    def _get_model(self):
        """Get the LangChain model for the agent.

        Returns:
            LangChain BaseChatModel instance

        Raises:
            ValueError: If no model_config is available
        """
        if self.model_config:
            return get_langchain_model(model_config=self.model_config)

        raise ValueError(
            "No model_config provided to StateIdentificationAgent. "
            "Ensure xopt_optimizer.state_identification.model_config_name is set in config.yml "
            "or that 'orchestrator' model is configured as fallback."
        )

    def _get_agent(self):
        """Get or create the ReAct agent.

        Returns:
            Compiled ReAct agent graph
        """
        if self._agent is None:
            model = self._get_model()
            tools = self._get_tools()

            self._agent = create_react_agent(
                model=model,
                tools=tools,
            )

        return self._agent

    async def assess_state(
        self,
        objective: str,
        additional_context: dict[str, Any] | None = None,
    ) -> tuple[MachineState, dict[str, Any]]:
        """Assess machine readiness for optimization.

        Args:
            objective: The optimization objective (provides context for assessment)
            additional_context: Optional additional context

        Returns:
            Tuple of (MachineState, details dict with reasoning and observations)

        Raises:
            ValueError: If assessment fails
        """
        agent = self._get_agent()

        # Build the user message
        user_message = f"""Assess whether the machine is ready for optimization.

**Optimization Objective:** {objective}

Please:
1. First read the reference documentation to understand the machine ready criteria
2. Then check the relevant channel values
3. Provide your assessment of the machine state

Remember to check ALL relevant criteria before making your assessment.
"""

        if additional_context:
            user_message += f"\n**Additional Context:** {additional_context}"

        logger.info("Starting state identification agent...")

        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=STATE_IDENTIFICATION_PROMPT),
                        HumanMessage(content=user_message),
                    ]
                }
            )

            # Extract the final response
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("Agent did not produce any output")

            # Get the last message content
            last_message = messages[-1]
            content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

            # Parse the assessment from the response
            assessment = self._parse_assessment(content)

            logger.info(f"State assessment complete: {assessment['state'].value}")
            return assessment["state"], {
                "reasoning": assessment["reasoning"],
                "channels_checked": assessment.get("channels_checked", []),
                "key_observations": assessment.get("key_observations", {}),
                "raw_response": content,
            }

        except Exception as e:
            logger.error(f"State identification agent failed: {e}")
            raise ValueError(f"State assessment failed: {e}") from e

    def _parse_assessment(self, content: str) -> dict[str, Any]:
        """Parse machine state assessment from agent response.

        Args:
            content: The agent's response text

        Returns:
            Dict with state, reasoning, channels_checked, and key_observations
        """
        content_lower = content.lower()

        # Determine state from response
        if "not_ready" in content_lower or "not ready" in content_lower:
            state = MachineState.NOT_READY
        elif "unknown" in content_lower and (
            "state: unknown" in content_lower
            or "state is unknown" in content_lower
            or "cannot determine" in content_lower
            or "unable to determine" in content_lower
        ):
            state = MachineState.UNKNOWN
        elif "ready" in content_lower:
            # Check it's not "not ready"
            if "not ready" not in content_lower and "not_ready" not in content_lower:
                state = MachineState.READY
            else:
                state = MachineState.NOT_READY
        else:
            # Default to unknown if we can't parse
            logger.warning("Could not parse state from response, defaulting to UNKNOWN")
            state = MachineState.UNKNOWN

        # Extract channel names mentioned in the response
        channels_checked = []
        # Look for common channel patterns
        import re

        channel_pattern = r"[A-Z][A-Z0-9_:]+:[A-Z0-9_:]+"
        channels_checked = list(set(re.findall(channel_pattern, content)))

        return {
            "state": state,
            "reasoning": content,
            "channels_checked": channels_checked,
            "key_observations": {},
        }


def create_state_identification_agent(
    reference_path: str | None = None,
    mock_files: bool = False,
    model_config: dict[str, Any] | None = None,
) -> StateIdentificationAgent:
    """Factory function to create a state identification agent.

    Args:
        reference_path: Path to reference documentation directory
        mock_files: If True, use mock reference file data
        model_config: Optional model configuration

    Returns:
        Configured StateIdentificationAgent instance
    """
    return StateIdentificationAgent(
        reference_path=reference_path,
        mock_files=mock_files,
        model_config=model_config,
    )
