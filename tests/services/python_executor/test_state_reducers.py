"""Unit Tests for Python Executor State Reducers.

This module tests the custom state reducers used in PythonExecutionState,
particularly the preserve_once_set reducer that prevents critical fields
from being lost during LangGraph state updates and checkpoint resumption.
"""

import pytest

from osprey.services.python_executor.models import (
    PythonExecutionRequest,
    preserve_once_set,
)


class TestPreserveOnceSetReducer:
    """Test the preserve_once_set reducer for state field protection."""

    def test_preserves_existing_value(self):
        """Test that existing value is preserved when new value provided."""
        # Create a sample request
        request = PythonExecutionRequest(
            user_query="Test query",
            task_objective="Test objective",
            execution_folder_name="test_folder",
        )

        # Simulate state update where existing value should be preserved
        result = preserve_once_set(existing=request, new=None)

        # Existing value should be returned
        assert result == request
        assert result is request  # Same object reference

    def test_accepts_new_value_when_no_existing(self):
        """Test that new value is accepted when no existing value."""
        # Create a sample request
        request = PythonExecutionRequest(
            user_query="Test query",
            task_objective="Test objective",
            execution_folder_name="test_folder",
        )

        # Simulate initial state creation
        result = preserve_once_set(existing=None, new=request)

        # New value should be returned
        assert result == request
        assert result is request

    def test_preserves_over_new_value(self):
        """Test that existing value is preserved even when new value differs."""
        # Create two different requests
        request1 = PythonExecutionRequest(
            user_query="First query",
            task_objective="First objective",
            execution_folder_name="folder1",
        )
        request2 = PythonExecutionRequest(
            user_query="Second query",
            task_objective="Second objective",
            execution_folder_name="folder2",
        )

        # Simulate state update attempting to replace value
        result = preserve_once_set(existing=request1, new=request2)

        # Original value should be preserved
        assert result == request1
        assert result is request1
        assert result != request2

    def test_handles_none_both_sides(self):
        """Test that None is returned when both values are None."""
        result = preserve_once_set(existing=None, new=None)
        assert result is None

    def test_checkpoint_resume_scenario(self):
        """Test the exact scenario that occurs during checkpoint resumption.

        This simulates what happens when LangGraph resumes from a checkpoint
        with Command(resume={"approved": True}):
        1. Checkpoint has full state including request
        2. Resume payload doesn't include request (only {"approved": True})
        3. Reducer should preserve the request from checkpoint
        """
        # Original state from checkpoint
        checkpoint_request = PythonExecutionRequest(
            user_query="Write to EPICS PV",
            task_objective="Test EPICS write approval",
            execution_folder_name="test_approval",
        )

        # Resume payload doesn't include request field (implicitly None)
        resume_request = None

        # Reducer should preserve checkpoint value
        result = preserve_once_set(existing=checkpoint_request, new=resume_request)

        assert result == checkpoint_request
        assert result is checkpoint_request
        assert result.user_query == "Write to EPICS PV"

    def test_works_with_arbitrary_values(self):
        """Test that reducer works with any type of value, not just requests."""
        # Test with strings
        assert preserve_once_set("existing", "new") == "existing"
        assert preserve_once_set(None, "new") == "new"

        # Test with integers
        assert preserve_once_set(42, 100) == 42
        assert preserve_once_set(None, 100) == 100

        # Test with dicts
        dict1 = {"key": "value1"}
        dict2 = {"key": "value2"}
        assert preserve_once_set(dict1, dict2) == dict1
        assert preserve_once_set(None, dict2) == dict2


class TestStatePreservationIntegration:
    """Integration tests for state field preservation in actual workflows."""

    def test_request_field_has_reducer_annotation(self):
        """Verify that PythonExecutionState.request has the preserve_once_set reducer."""
        from typing import get_args, get_origin, get_type_hints

        from osprey.services.python_executor.models import PythonExecutionState

        # Get type hints for PythonExecutionState
        hints = get_type_hints(PythonExecutionState, include_extras=True)

        # Check that 'request' field exists
        assert "request" in hints, "PythonExecutionState should have 'request' field"

        # Get the type annotation for 'request'
        request_annotation = hints["request"]

        # Check if it's an Annotated type
        origin = get_origin(request_annotation)
        if origin is not None:
            # It's an Annotated type, get the metadata
            args = get_args(request_annotation)
            assert len(args) >= 2, "Annotated should have type and metadata"

            # The second argument should be the reducer function
            metadata = args[1:]
            assert (
                preserve_once_set in metadata
            ), "request field should have preserve_once_set reducer in metadata"

    def test_state_creation_includes_request(self):
        """Test that _create_internal_state includes request field.

        This verifies that the state initialization creates the proper structure
        that will be protected by the preserve_once_set reducer.
        """
        from osprey.services.python_executor.models import PythonExecutionState

        # Create a sample request
        request = PythonExecutionRequest(
            user_query="Test query",
            task_objective="Test objective",
            execution_folder_name="test_folder",
        )

        # Create state manually (mimicking what _create_internal_state does)
        state = PythonExecutionState(
            request=request,
            capability_context_data=None,
            generation_attempt=0,
            error_chain=[],
            current_stage="generation",
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            generated_code=None,
            analysis_result=None,
            analysis_failed=None,
            execution_failed=None,
            execution_result=None,
            execution_folder=None,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
        )

        # Verify request is in state
        assert "request" in state
        assert state["request"] == request

        # Simulate a state update that doesn't include request (like Command.resume)
        # The reducer would preserve the request field
        preserved_request = preserve_once_set(
            existing=state["request"], new=None  # What comes from Command.resume
        )

        assert preserved_request == request
        assert preserved_request is request
