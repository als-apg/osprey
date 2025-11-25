"""Shared fixtures for capability tests."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Configure pytest to mock registry before any imports."""
    # Create mock registry
    mock_reg = MagicMock()
    mock_reg.context_types = MagicMock()
    mock_reg.services = MagicMock()

    # Mock the get_registry function at module level
    import osprey.registry.manager

    osprey.registry.manager._registry = mock_reg
    osprey.registry.manager.get_registry = lambda: mock_reg


from osprey.state import AgentState


@pytest.fixture
def mock_state() -> AgentState:
    """Create a mock agent state with complete execution plan."""
    return {
        "messages": [],
        "planning_execution_steps": [
            {
                "step_index": 0,
                "capability": "test_capability",
                "context_key": "test_key_001",
                "task_objective": "Test task objective",
                "reasoning": "Test reasoning",
                "inputs": [],
            }
        ],
        "planning_current_step_index": 0,
        "capability_context_data": {},
        "context_data": {},
        "execution_step_results": {},
        "input_output": {
            "user_query": "Test user query",
        },
        "config": {
            "user_id": "test_user",
        },
        "control_routing_count": 0,
    }


@pytest.fixture
def mock_step() -> dict[str, Any]:
    """Create a mock execution step."""
    return {
        "step_index": 0,
        "capability": "test_capability",
        "context_key": "test_key_001",
        "task_objective": "Test task objective",
        "reasoning": "Test reasoning",
        "inputs": [],
    }


@pytest.fixture
def mock_registry(monkeypatch):
    """Mock the registry for testing."""
    mock_reg = MagicMock()
    mock_reg.context_types.MEMORY_CONTEXT = "MEMORY_CONTEXT"
    mock_reg.context_types.PYTHON_RESULTS = "PYTHON_RESULTS"
    mock_reg.context_types.TIME_RANGE = "TIME_RANGE"

    monkeypatch.setattr("osprey.registry.get_registry", lambda: mock_reg)
    return mock_reg


@pytest.fixture
def mock_state_manager(monkeypatch):
    """Mock StateManager for testing."""
    mock_sm = MagicMock()
    mock_sm.store_context.return_value = {"context_data": {}}
    mock_sm.register_figure.return_value = {}
    mock_sm.register_notebook.return_value = {}

    monkeypatch.setattr("osprey.state.StateManager.store_context", mock_sm.store_context)
    monkeypatch.setattr("osprey.state.StateManager.register_figure", mock_sm.register_figure)
    monkeypatch.setattr("osprey.state.StateManager.register_notebook", mock_sm.register_notebook)

    return mock_sm
