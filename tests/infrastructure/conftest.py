"""Shared fixtures for infrastructure node tests."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from osprey.state import AgentState


@pytest.fixture
def base_state() -> AgentState:
    """Create a basic agent state for testing."""
    return {
        "messages": [HumanMessage(content="Test query")],
        "control_routing_count": 0,
        "task_current_task": "Test task",
        "planning_active_capabilities": ["test_capability"],
    }


@pytest.fixture
def state_with_task() -> AgentState:
    """Create agent state with task information."""
    return {
        "messages": [
            HumanMessage(content="What is the weather?"),
            AIMessage(content="I'll help you with that."),
        ],
        "task_current_task": "Get current weather information",
        "task_depends_on_chat_history": False,
        "task_depends_on_user_memory": False,
        "control_routing_count": 1,
    }


@pytest.fixture
def state_with_plan() -> AgentState:
    """Create agent state with execution plan."""
    return {
        "messages": [HumanMessage(content="Test query")],
        "task_current_task": "Test task",
        "planning_active_capabilities": ["python", "memory"],
        "planning_execution_plan": {
            "steps": [
                {
                    "step_index": 0,
                    "capability": "python",
                    "task_objective": "Run calculation",
                    "reasoning": "Need to compute result",
                },
                {
                    "step_index": 1,
                    "capability": "respond",
                    "task_objective": "Provide answer",
                    "reasoning": "Return result to user",
                },
            ],
            "source": "llm_based",
        },
        "control_step_index": 0,
        "control_routing_count": 2,
    }


@pytest.fixture
def state_with_error() -> AgentState:
    """Create agent state with error information."""
    return {
        "messages": [HumanMessage(content="Test query")],
        "task_current_task": "Test task",
        "control_has_error": True,
        "control_error_info": {
            "original_error": "Test error occurred",
            "capability_name": "test_capability",
            "node_name": "test_node",
            "execution_time": 1.5,
        },
        "control_routing_count": 3,
    }


@pytest.fixture
def mock_step() -> dict[str, Any]:
    """Create a mock execution step."""
    return {
        "step_index": 0,
        "capability": "test_capability",
        "task_objective": "Test objective",
        "reasoning": "Test reasoning",
    }
