"""
Integration test for channel_write capability with approval workflow.

This test exercises the real execution path (Graph → channel_write → Approval)
to catch bugs in the capability's interaction with the approval system.

The bug being tested:
- Capability tried to call self._get_verification_config() when building
  analysis_details for approval, but this method doesn't exist on the capability
  (it exists on connectors, which haven't been created yet at that point)

Test strategy:
- Use actual my-control-assistant project (like e2e tests)
- Initialize full framework with proper registry
- Create state with channel_write task and invoke through graph (bypasses orchestrator)
- Mock only LLMs to avoid real API calls
- Test fails if _get_verification_config bug is present
- Test passes if approval interrupt is triggered (bug is fixed)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# Override the conftest's autouse fixture that mocks the registry
# We need the REAL registry for integration tests
# ============================================================================
@pytest.fixture(autouse=True, scope="function")
def mock_registry_for_capability_tests():
    """Override the conftest's autouse fixture - we need the real registry for integration tests.

    The conftest.py in this directory has an autouse fixture that mocks the registry.
    That's great for unit tests, but for integration tests we need the real thing.
    This override makes the fixture do nothing (no mocking).
    """
    # Do nothing - let the real registry be used
    yield


@pytest.fixture
def my_control_assistant_project():
    """Get path to my-control-assistant project."""
    project_path = Path(__file__).parent.parent.parent / "my-control-assistant"
    if not project_path.exists():
        pytest.skip("my-control-assistant project not available")
    return project_path


@pytest.fixture
def my_control_assistant_config(my_control_assistant_project):
    """Get path to my-control-assistant config with approval enabled."""
    config_path = my_control_assistant_project / "config.yml"
    if not config_path.exists():
        pytest.skip("my-control-assistant config not found")

    # Store original CONFIG_FILE and working directory
    original_config = os.environ.get("CONFIG_FILE")
    original_cwd = os.getcwd()

    # Set CONFIG_FILE to my-control-assistant
    os.environ["CONFIG_FILE"] = str(config_path)

    # Change to my-control-assistant directory (needed for relative paths in config)
    os.chdir(my_control_assistant_project)

    # Don't reset registry here - let the autouse fixture handle it
    # Just clear config caches so it loads the new CONFIG_FILE
    import osprey.approval.approval_manager as approval_module
    from osprey.utils import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()
    approval_module._approval_manager = None

    yield config_path

    # Restore original state
    os.chdir(original_cwd)
    if original_config is None:
        if "CONFIG_FILE" in os.environ:
            del os.environ["CONFIG_FILE"]
    else:
        os.environ["CONFIG_FILE"] = original_config


@pytest.mark.asyncio
async def test_channel_write_approval_workflow_catches_verification_config_bug(
    my_control_assistant_project, my_control_assistant_config
):
    """
    Test that channel_write capability's approval workflow catches the _get_verification_config bug.

    BUG: Line ~507 in channel_write.py tries to call:
        self._get_verification_config(op.channel_address, op.value)[0]

    But _get_verification_config() exists on CONNECTORS, not CAPABILITIES.
    This causes AttributeError when approval is triggered.

    This test FAILS if the buggy code is present (correctly detecting the bug).
    Once the bug is fixed, this test should PASS.

    Strategy: Create initial state with channel_write task, invoke through graph
    (needed for approval interrupt context), bypassing gateway/orchestrator LLM routing.
    """
    from osprey.base.planning import ExecutionPlan, PlannedStep
    from osprey.capabilities.channel_write import (
        WriteOperation,
        WriteOperationsOutput,
    )
    from osprey.graph import create_graph
    from osprey.registry import get_registry, initialize_registry
    from osprey.utils.config import get_config_value, get_full_configuration
    from tests.conftest import create_test_state

    # Initialize registry FIRST
    initialize_registry(config_path=str(my_control_assistant_config))
    registry = get_registry()

    # Load config
    configurable = get_full_configuration(str(my_control_assistant_config)).copy()
    configurable.update(
        {
            "user_id": "test_user",
            "thread_id": "test_channel_write_approval",
            "chat_id": "test_chat",
            "session_id": "test_session",
            "interface_context": "test",
        }
    )

    # Create graph with checkpointer (needed for approval system)
    checkpointer = MemorySaver()
    graph = create_graph(registry, checkpointer=checkpointer)

    # Set up graph config
    recursion_limit = get_config_value("execution_limits.graph_recursion_limit")
    graph_config = {"configurable": configurable, "recursion_limit": recursion_limit}

    def mock_chat_completion(model_config=None, message=None, output_model=None, **kwargs):
        """Mock LLM calls - return proper structured outputs."""
        if output_model is not None:
            # Check if this is the WriteOperationsOutput model
            if (
                hasattr(output_model, "__name__")
                and "WriteOperationsOutput" in output_model.__name__
            ):
                # Return valid write operation for "Set first quadrupole to 200"
                return WriteOperationsOutput(
                    write_operations=[
                        WriteOperation(
                            channel_address="MAG:QF[QF01]:CURRENT:SP",
                            value=200.0,
                            units="A",
                            notes="Setting quadrupole magnet current",
                        )
                    ],
                    found=True,
                )
        # For other LLM calls without output_model, return simple text
        return "Mock LLM response"

    # Patch get_config_value to enable writes
    original_get_config_value = get_config_value

    def mock_config_value(key, default=None):
        if key == "control_system.writes_enabled":
            return True  # Enable writes for this test
        return original_get_config_value(key, default)

    # Create initial state with channel_write capability
    user_request = "Set the first quadrupole magnet focusing to 200"

    # Create state - note we need to customize the execution plan inputs
    planned_step = PlannedStep(
        context_key="test_channel_write_001",
        capability="channel_write",
        task_objective=user_request,
        success_criteria="Task completed successfully",
        expected_output=None,
        inputs=[
            {"CHANNEL_ADDRESSES": "channel_addresses_001"}
        ],  # Tell it where to find CHANNEL_ADDRESSES
    )

    execution_plan = ExecutionPlan(steps=[planned_step], final_objective="Test objective")

    initial_state = create_test_state(
        user_message=user_request,
        task_objective=user_request,
        capability="channel_write",
        context_key="test_channel_write_001",
    )

    # Override with our custom execution plan that has proper inputs
    initial_state["planning_execution_plan"] = execution_plan

    # Add CHANNEL_ADDRESSES context that channel_write needs
    initial_state["capability_context_data"] = {
        "CHANNEL_ADDRESSES": {
            "channel_addresses_001": {
                "channels": ["MAG:QF[QF01]:CURRENT:SP"],
                "original_query": "first quadrupole magnet focusing",
            }
        }
    }

    # Patch and execute through graph (needed for approval interrupt context)
    with patch(
        "osprey.capabilities.channel_write.get_chat_completion",
        side_effect=mock_chat_completion,
    ):
        with patch("osprey.utils.config.get_config_value", side_effect=mock_config_value):
            final_state = None
            approval_interrupt_occurred = False

            try:
                # Invoke through graph (approval system needs graph context)
                final_state = await graph.ainvoke(initial_state, config=graph_config)

            except AttributeError as e:
                # Direct AttributeError - the bug manifested
                if "_get_verification_config" in str(e):
                    pytest.fail(f"Bug detected: {str(e)}")
                else:
                    raise

            except Exception as e:
                # Check if it's an approval interrupt (expected if bug is fixed)
                if "interrupt" in str(type(e).__name__).lower():
                    # Good! Approval was triggered, meaning we got past the buggy code
                    # This means the bug is FIXED
                    approval_interrupt_occurred = True
                else:
                    # Unexpected error
                    raise

            # Check if capability caught the error and put it in the state
            if final_state and not approval_interrupt_occurred:
                control_error_info = final_state.get("control_error_info")
                if control_error_info:
                    # Extract error details from various possible locations
                    error_details = ""

                    # Check classification.metadata.technical_details
                    if "classification" in control_error_info:
                        classification = control_error_info["classification"]
                        if hasattr(classification, "metadata"):
                            error_details = classification.metadata.get("technical_details", "")
                        elif isinstance(classification, dict):
                            metadata = classification.get("metadata", {})
                            error_details = metadata.get("technical_details", "")

                    # Also check original_error and user_message fields
                    if not error_details:
                        error_details = control_error_info.get("original_error", "")
                    if not error_details:
                        error_details = control_error_info.get("user_message", "")

                    if "_get_verification_config" in str(error_details):
                        pytest.fail(
                            f"Bug detected in channel_write capability: "
                            f"Trying to call _get_verification_config() on capability before connector exists. "
                            f"Error: {error_details}"
                        )

            # If we got here and approval_interrupt_occurred, the bug is fixed!
            if approval_interrupt_occurred:
                # Test passes - approval system was reached successfully
                pass
