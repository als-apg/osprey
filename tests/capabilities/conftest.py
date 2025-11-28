"""Shared fixtures for capability tests.

NOTE: This conftest.py mocks the registry for capability unit tests.
Do NOT run these tests together with e2e tests - run them separately:
  - Unit tests: pytest tests/ --ignore=tests/e2e
  - E2E tests:   pytest tests/e2e/
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey.state import AgentState


def pytest_configure(config):
    """Configure pytest to mock registry before any imports.

    This mock is ONLY for capability unit tests.
    E2E tests should be run separately: pytest tests/e2e/
    """
    # Create mock registry for capability unit tests
    mock_reg = MagicMock()
    mock_reg.context_types = MagicMock()
    mock_reg.services = MagicMock()

    # Mock registry methods to avoid isinstance() errors
    mock_reg._registries = {}  # Empty dict to bypass validation
    mock_reg.is_valid_context_type = MagicMock(return_value=True)
    mock_reg.get_context_class = MagicMock(return_value=None)  # Return None to skip type validation
    mock_reg.get_all_context_types = MagicMock(return_value=[])

    # Mock the get_registry function at module level
    import osprey.registry.manager

    osprey.registry.manager._registry = mock_reg
    # Accept optional keyword arguments for compatibility
    osprey.registry.manager.get_registry = lambda **kwargs: mock_reg




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
