"""Integration tests for Gateway approval detection with full workflow."""

from unittest.mock import MagicMock, patch

import pytest

from osprey.infrastructure.gateway import Gateway


@pytest.fixture
def mock_graph():
    """Create a mock compiled graph."""
    graph = MagicMock()

    # Mock state with pending interrupt
    interrupt_state = MagicMock()
    interrupt_state.values = {"messages": []}
    interrupt = MagicMock()
    interrupt.value = {
        "user_message": "Test approval request",
        "resume_payload": {"approval_type": "test_approval", "test_data": "test_value"},
    }
    interrupt_state.interrupts = [interrupt]

    # Mock state without interrupts
    normal_state = MagicMock()
    normal_state.values = {"messages": []}
    normal_state.interrupts = []

    return graph, interrupt_state, normal_state


@pytest.fixture
def gateway():
    """Create a Gateway instance for testing."""
    return Gateway()


class TestApprovalWorkflowIntegration:
    """Integration tests for approval workflow with Gateway."""

    @pytest.mark.asyncio
    async def test_explicit_yes_creates_resume_command(self, gateway, mock_graph):
        """Test that explicit 'yes' creates correct resume command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process explicit "yes" response
        result = await gateway.process_message("yes", graph, config)

        # Should create resume command with approval
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is True
        assert result.resume_command.update["approved_payload"] is not None
        assert result.resume_command.update["approved_payload"]["approval_type"] == "test_approval"
        assert result.approval_detected is True
        assert result.is_interrupt_resume is True

    @pytest.mark.asyncio
    async def test_explicit_yes_with_punctuation_creates_resume_command(self, gateway, mock_graph):
        """Test that 'yes!' creates correct resume command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process explicit "yes!" response
        result = await gateway.process_message("yes!", graph, config)

        # Should create resume command with approval
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is True
        assert result.resume_command.update["approved_payload"] is not None
        assert result.approval_detected is True

    @pytest.mark.asyncio
    async def test_explicit_no_creates_rejection_command(self, gateway, mock_graph):
        """Test that explicit 'no' creates correct rejection command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process explicit "no" response
        result = await gateway.process_message("no", graph, config)

        # Should create resume command with rejection
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is False
        assert result.resume_command.update["approved_payload"] is None
        assert result.approval_detected is True
        assert result.is_interrupt_resume is True

    @pytest.mark.asyncio
    async def test_explicit_no_with_punctuation_creates_rejection_command(
        self, gateway, mock_graph
    ):
        """Test that 'no.' creates correct rejection command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process explicit "no." response
        result = await gateway.process_message("no.", graph, config)

        # Should create resume command with rejection
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is False
        assert result.resume_command.update["approved_payload"] is None
        assert result.approval_detected is True

    @pytest.mark.asyncio
    async def test_okay_creates_approval_command(self, gateway, mock_graph):
        """Test that 'okay' creates correct approval command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process "okay" response
        result = await gateway.process_message("okay", graph, config)

        # Should create resume command with approval
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is True
        assert result.resume_command.update["approved_payload"] is not None

    @pytest.mark.asyncio
    async def test_cancel_creates_rejection_command(self, gateway, mock_graph):
        """Test that 'cancel' creates correct rejection command."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        # Process "cancel" response
        result = await gateway.process_message("cancel", graph, config)

        # Should create resume command with rejection
        assert result.resume_command is not None
        assert result.resume_command.update["approval_approved"] is False
        assert result.resume_command.update["approved_payload"] is None

    @pytest.mark.asyncio
    async def test_complex_response_uses_llm(self, gateway, mock_graph):
        """Test that complex responses fall back to LLM for approval detection."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            with patch("osprey.infrastructure.gateway.get_model_config") as mock_config:
                # Setup mocks
                mock_config.return_value = {"model": "test-model"}
                mock_llm.return_value = MagicMock(approved=True)

                # Process complex response
                result = await gateway.process_message(
                    "I think this looks good, let's proceed", graph, config
                )

                # Should call LLM for complex response
                mock_llm.assert_called_once()

                # Should create resume command based on LLM result
                assert result.resume_command is not None
                assert result.resume_command.update["approval_approved"] is True

    @pytest.mark.asyncio
    async def test_no_interrupt_processes_as_normal_message(self, gateway, mock_graph):
        """Test that messages without interrupts are processed normally."""
        graph, _, normal_state = mock_graph
        graph.get_state.return_value = normal_state

        config = {"thread_id": "test"}

        # Process normal message
        result = await gateway.process_message("What is the weather?", graph, config)

        # Should create agent state for normal processing
        assert result.agent_state is not None
        assert result.resume_command is None
        assert result.approval_detected is False
        assert result.is_interrupt_resume is False

    @pytest.mark.asyncio
    async def test_case_insensitive_approval(self, gateway, mock_graph):
        """Test that approval detection is case-insensitive."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        test_cases = ["YES", "Yes", "yEs", "OK", "Ok", "OKAY"]

        for test_input in test_cases:
            result = await gateway.process_message(test_input, graph, config)

            # All should be approved
            assert result.resume_command is not None, f"Failed for '{test_input}'"
            assert (
                result.resume_command.update["approval_approved"] is True
            ), f"Not approved for '{test_input}'"

    @pytest.mark.asyncio
    async def test_case_insensitive_rejection(self, gateway, mock_graph):
        """Test that rejection detection is case-insensitive."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        test_cases = ["NO", "No", "nO", "CANCEL", "Cancel"]

        for test_input in test_cases:
            result = await gateway.process_message(test_input, graph, config)

            # All should be rejected
            assert result.resume_command is not None, f"Failed for '{test_input}'"
            assert (
                result.resume_command.update["approval_approved"] is False
            ), f"Not rejected for '{test_input}'"

    @pytest.mark.asyncio
    async def test_whitespace_handling(self, gateway, mock_graph):
        """Test that whitespace doesn't affect approval detection."""
        graph, interrupt_state, _ = mock_graph
        graph.get_state.return_value = interrupt_state

        config = {"thread_id": "test"}

        test_cases = ["  yes  ", "\tyes\t", "\nyes\n", "  yes!  "]

        for test_input in test_cases:
            result = await gateway.process_message(test_input, graph, config)

            # All should be approved
            assert result.resume_command is not None, f"Failed for '{test_input}'"
            assert (
                result.resume_command.update["approval_approved"] is True
            ), f"Not approved for '{test_input}'"
