"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.state import AgentState


# ===================================================================
# Registry Mock for Testing
# ===================================================================

@pytest.fixture(scope="session", autouse=True)
def mock_registry_session():
    """Mock the registry for the entire test session to avoid config requirement.
    
    This allows importing capabilities without needing a config.yml file.
    """
    import sys
    from unittest.mock import MagicMock, patch
    
    # Create a mock registry
    mock_reg = MagicMock()
    mock_reg.context_types = MagicMock()
    mock_reg.services = MagicMock()
    
    # Patch get_registry before any imports
    with patch("osprey.registry.manager.get_registry", return_value=mock_reg):
        yield mock_reg

# ===================================================================
# Test State Factory
# ===================================================================

def create_test_state(
    user_message: str = "test query",
    task_objective: str = "test objective",
    conversation_history: list[tuple[str, str]] | None = None,
    capability: str = "test_capability",
    context_key: str = "test_context",
    final_objective: str = "Test objective",
    success_criteria: str = "Task completed successfully",
    **overrides
) -> AgentState:
    """Factory function to create test states with minimal boilerplate.

    This is a global test utility that can be used across all test modules
    to create AgentState objects with sensible defaults, reducing test
    verbosity and maintenance burden.

    Args:
        user_message: The user's message (used if no conversation_history provided)
        task_objective: The task objective for the planned step
        conversation_history: List of (role, content) tuples for multi-turn conversations
                            role should be 'user' or 'ai'
        capability: Capability name for the planned step
        context_key: Context key for the planned step
        final_objective: The execution plan's final objective
        success_criteria: Success criteria for the planned step
        **overrides: Any additional state fields to override

    Returns:
        Complete AgentState with sensible defaults for testing

    Examples:
        Simple single-message state::

            state = create_test_state(
                user_message="whats the weather?",
                task_objective="Ask for location"
            )

        Multi-turn conversation::

            state = create_test_state(
                conversation_history=[
                    ("user", "I need data"),
                    ("ai", "What kind of data?"),
                    ("user", "beam current")
                ],
                task_objective="Ask for time range"
            )

        With capability-specific settings::

            state = create_test_state(
                user_message="fetch data",
                capability="data_retrieval",
                context_key="fetch_pv_data",
                task_depends_on_chat_history=True
            )
    """
    # Build messages from conversation history or single message
    messages = []
    if conversation_history:
        for role, content in conversation_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    else:
        messages = [HumanMessage(content=user_message)]

    # Create execution plan
    execution_plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key=context_key,
                capability=capability,
                task_objective=task_objective,
                success_criteria=success_criteria,
                expected_output=None,
                inputs=[]
            )
        ],
        final_objective=final_objective
    )

    # Create state with defaults
    state: AgentState = {
        'messages': messages,
        'planning_execution_plan': execution_plan,
        'planning_current_step_index': 0,
        'capability_context_data': {},
        'agent_control': {},
        'status_updates': [],
        'progress_events': [],
        'task_current_task': 'Test task',
        'task_depends_on_chat_history': False,
        'task_depends_on_user_memory': False,
        'task_custom_message': None,
        'planning_active_capabilities': [capability],
        'execution_step_results': {},
        'execution_last_result': None,
        'execution_pending_approvals': {},
        'execution_start_time': None,
        'execution_total_time': None,
        'approval_approved': None,
        'approved_payload': None,
        'control_reclassification_reason': None,
        'control_reclassification_count': 0,
        'control_plans_created_count': 1,
        'control_current_step_retry_count': 0,
        'control_retry_count': 0,
        'control_has_error': False,
        'control_error_info': None,
        'control_last_error': None,
        'control_max_retries': 3,
        'control_is_killed': False,
        'control_kill_reason': None,
        'control_is_awaiting_validation': False,
        'control_validation_context': None,
        'control_validation_timestamp': None,
        'ui_captured_notebooks': [],
        'ui_captured_figures': [],
        'ui_launchable_commands': [],
        'ui_agent_context': None,
        'runtime_checkpoint_metadata': None,
        'runtime_info': None,
    }

    # Apply any overrides
    state.update(overrides)

    return state


# ===================================================================
# Prompt Testing Helpers
# ===================================================================

class PromptTestHelpers:
    """Helper methods for testing prompt structure and content."""

    @staticmethod
    def extract_section(prompt: str, section_header: str) -> str:
        """Extract a specific section from the prompt by its header.

        Args:
            prompt: The full prompt text
            section_header: The header marking the start of the section

        Returns:
            The extracted section content (without the header)
        """
        lines = prompt.split('\n')
        section_lines = []
        in_section = False

        for line in lines:
            if section_header in line:
                in_section = True
                continue
            if in_section:
                # Stop at next all-caps header with colon
                if line.strip() and line.strip().replace(' ', '').replace("'", '').isupper() and ':' in line:
                    break
                section_lines.append(line)

        return '\n'.join(section_lines).strip()

    @staticmethod
    def get_section_positions(prompt: str, *section_headers: str) -> dict[str, int]:
        """Get the positions of multiple section headers in the prompt.

        Args:
            prompt: The full prompt text
            *section_headers: Variable number of section headers to find

        Returns:
            Dictionary mapping section headers to their positions (-1 if not found)
        """
        positions = {}
        for header in section_headers:
            try:
                positions[header] = prompt.index(header)
            except ValueError:
                positions[header] = -1
        return positions


# ===================================================================
# Pytest Fixtures
# ===================================================================

@pytest.fixture
def test_state():
    """Fixture providing a basic test state for quick testing."""
    return create_test_state()


@pytest.fixture
def prompt_helpers():
    """Fixture providing prompt testing helper methods."""
    return PromptTestHelpers

