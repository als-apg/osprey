"""Unit tests for executor node safety-critical error handling.

While the full execution node requires complex infrastructure (see integration/e2e tests),
the safety-violation-prevents-retry logic in the exception handler is pure decision-making
that can and should be unit tested to lock in safety guarantees.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.python_executor.execution.node import create_executor_node


class TestExecutionNodeDocumentation:
    """Document that execution node is tested via integration/e2e tests."""

    def test_execution_node_is_integration_tested(self):
        """Execution node is comprehensively tested in integration/e2e suites."""
        # The execution node requires complex infrastructure (containers, Python env,
        # file systems) and is fully tested via:
        #
        # Integration tests:
        # - tests/integration/test_python_executor_service.py::TestBasicWorkflow
        # - tests/integration/test_python_executor_service.py::TestExecutionMethods
        # - tests/integration/test_python_executor_service.py::TestErrorHandling
        #
        # E2E tests:
        # - tests/e2e/test_code_generator_workflows.py (433 lines)
        # - tests/e2e/test_runtime_limits.py (635 lines)
        #
        # These cover:
        # - Local and container execution methods
        # - Error handling and retry logic
        # - Runtime utilities and safety mechanisms
        # - Complete generation → analysis → execution workflows
        assert True, "See integration/e2e tests for execution node coverage"


def _make_mock_state(*, retries=3, error_chain=None):
    """Create a minimal mock state dict for the executor node."""
    mock_request = MagicMock()
    mock_request.retries = retries
    mock_request.execution_folder_name = "test_exec"

    return {
        "generated_code": "results = {'x': 1}",
        "request": mock_request,
        "generation_attempt": 1,
        "error_chain": error_chain or [],
        "execution_folder": None,
    }


class TestSafetyViolationPreventsRetry:
    """Tests that ChannelLimitsViolationError sets is_failed=True (I-3).

    These tests mock the executor to raise specific exceptions, then verify
    the node's error-handling logic correctly classifies safety violations
    as non-retryable.
    """

    @pytest.mark.asyncio
    async def test_channel_limits_violation_sets_is_failed(self):
        """ChannelLimitsViolationError in exception message marks execution as permanently failed."""
        from osprey.services.python_executor.exceptions import ChannelLimitsViolationError

        executor_node = create_executor_node()
        state = _make_mock_state(retries=3)

        # Create the actual exception that would be raised
        safety_error = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=150.0,
            violation_type="MAX_EXCEEDED",
            violation_reason="Value 150.0 above maximum 100.0",
            min_value=0.0,
            max_value=100.0,
        )

        with (
            patch(
                "osprey.services.python_executor.execution.node._create_execution_folder"
            ) as mock_folder,
            patch(
                "osprey.services.python_executor.execution.node._get_execution_method",
                return_value="local",
            ),
            patch("osprey.services.python_executor.execution.node._get_execution_mode_from_state"),
            patch(
                "osprey.services.python_executor.execution.node._create_error_notebook",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("osprey.utils.config.get_full_configuration", return_value={}),
            patch(
                "osprey.services.python_executor.execution.node.LocalCodeExecutor"
            ) as MockExecutor,
        ):
            # Set up execution folder mock
            mock_exec_folder = MagicMock()
            mock_exec_folder.folder_path = MagicMock()
            mock_exec_folder.context_file_path = MagicMock()
            mock_exec_folder.context_file_path.exists.return_value = True
            mock_folder.return_value = mock_exec_folder

            # Make executor raise the safety error
            mock_instance = MockExecutor.return_value
            mock_instance.execute_code = AsyncMock(side_effect=safety_error)

            result = await executor_node(state)

        assert result["is_failed"] is True
        assert "Channel limits violation" in result["failure_reason"]
        assert result["is_successful"] is False
        assert result["execution_failed"] is True

    @pytest.mark.asyncio
    async def test_channel_limits_violation_uppercase_marker_sets_is_failed(self):
        """Exception with 'CHANNEL LIMITS VIOLATION' also marks as permanently failed."""
        executor_node = create_executor_node()
        state = _make_mock_state(retries=3)

        # Some error paths may produce the uppercase marker
        error = RuntimeError("CHANNEL LIMITS VIOLATION: TEST:PV value 150.0 exceeds max 100.0")

        with (
            patch(
                "osprey.services.python_executor.execution.node._create_execution_folder"
            ) as mock_folder,
            patch(
                "osprey.services.python_executor.execution.node._get_execution_method",
                return_value="local",
            ),
            patch("osprey.services.python_executor.execution.node._get_execution_mode_from_state"),
            patch(
                "osprey.services.python_executor.execution.node._create_error_notebook",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("osprey.utils.config.get_full_configuration", return_value={}),
            patch(
                "osprey.services.python_executor.execution.node.LocalCodeExecutor"
            ) as MockExecutor,
        ):
            mock_exec_folder = MagicMock()
            mock_exec_folder.folder_path = MagicMock()
            mock_exec_folder.context_file_path = MagicMock()
            mock_exec_folder.context_file_path.exists.return_value = True
            mock_folder.return_value = mock_exec_folder

            mock_instance = MockExecutor.return_value
            mock_instance.execute_code = AsyncMock(side_effect=error)

            result = await executor_node(state)

        assert result["is_failed"] is True
        assert "Channel limits violation" in result["failure_reason"]

    @pytest.mark.asyncio
    async def test_non_safety_error_with_retries_remaining_is_not_failed(self):
        """A normal runtime error with retries remaining does NOT set is_failed=True."""
        executor_node = create_executor_node()
        # retries=3, error_chain=[] → 1 error after this, still below 3
        state = _make_mock_state(retries=3, error_chain=[])

        error = RuntimeError("Some regular execution error")

        with (
            patch(
                "osprey.services.python_executor.execution.node._create_execution_folder"
            ) as mock_folder,
            patch(
                "osprey.services.python_executor.execution.node._get_execution_method",
                return_value="local",
            ),
            patch("osprey.services.python_executor.execution.node._get_execution_mode_from_state"),
            patch(
                "osprey.services.python_executor.execution.node._create_error_notebook",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("osprey.utils.config.get_full_configuration", return_value={}),
            patch(
                "osprey.services.python_executor.execution.node.LocalCodeExecutor"
            ) as MockExecutor,
        ):
            mock_exec_folder = MagicMock()
            mock_exec_folder.folder_path = MagicMock()
            mock_exec_folder.context_file_path = MagicMock()
            mock_exec_folder.context_file_path.exists.return_value = True
            mock_folder.return_value = mock_exec_folder

            mock_instance = MockExecutor.return_value
            mock_instance.execute_code = AsyncMock(side_effect=error)

            result = await executor_node(state)

        # Should NOT be permanently failed — retries still available
        assert result["is_failed"] is False
        assert result["failure_reason"] is None
        assert result["execution_failed"] is True
        assert result["is_successful"] is False

    @pytest.mark.asyncio
    async def test_non_safety_error_at_retry_limit_is_failed(self):
        """A normal error that exhausts retries DOES set is_failed=True."""
        executor_node = create_executor_node()
        # retries=2, error_chain has 1 existing error → after this one, len=2 >= max_retries=2
        existing_error = MagicMock()
        state = _make_mock_state(retries=2, error_chain=[existing_error])

        error = RuntimeError("Another execution error")

        with (
            patch(
                "osprey.services.python_executor.execution.node._create_execution_folder"
            ) as mock_folder,
            patch(
                "osprey.services.python_executor.execution.node._get_execution_method",
                return_value="local",
            ),
            patch("osprey.services.python_executor.execution.node._get_execution_mode_from_state"),
            patch(
                "osprey.services.python_executor.execution.node._create_error_notebook",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("osprey.utils.config.get_full_configuration", return_value={}),
            patch(
                "osprey.services.python_executor.execution.node.LocalCodeExecutor"
            ) as MockExecutor,
        ):
            mock_exec_folder = MagicMock()
            mock_exec_folder.folder_path = MagicMock()
            mock_exec_folder.context_file_path = MagicMock()
            mock_exec_folder.context_file_path.exists.return_value = True
            mock_folder.return_value = mock_exec_folder

            mock_instance = MockExecutor.return_value
            mock_instance.execute_code = AsyncMock(side_effect=error)

            result = await executor_node(state)

        assert result["is_failed"] is True
        assert "failed after 2 attempts" in result["failure_reason"]
