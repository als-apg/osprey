"""Tests for validate_single_step() extracted from orchestration_node.

Tests validate individual step validation logic that is shared between
the plan-first and reactive orchestrators.
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.base.errors import InvalidContextKeyError
from osprey.base.planning import PlannedStep
from osprey.infrastructure.orchestration_node import (
    _validate_and_fix_execution_plan,
    validate_single_step,
)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    # Ensure all common log methods exist (including custom ones from osprey)
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.success = MagicMock()
    logger.key_info = MagicMock()
    return logger


@pytest.fixture
def mock_registry():
    """Create a mock registry with test capabilities."""
    registry = MagicMock()
    # Default: all capabilities exist
    registry.get_node.return_value = True
    registry.get_stats.return_value = {
        "capability_names": ["channel_finding", "channel_read", "respond"]
    }
    return registry


class TestValidSingleStep:
    """Test validate_single_step with valid capabilities."""

    def test_valid_capability(self, mock_logger, mock_registry):
        """Valid capability passes validation."""
        step = PlannedStep(
            context_key="test_step",
            capability="channel_finding",
            task_objective="Find channels",
        )
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            # Should not raise
            validate_single_step(step, state, mock_logger)

    def test_no_capability_specified(self, mock_logger, mock_registry):
        """Step with empty capability logs warning but doesn't raise."""
        step = PlannedStep(
            context_key="test_step",
            capability="",
            task_objective="Do something",
        )
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            validate_single_step(step, state, mock_logger)
            mock_logger.warning.assert_called()


class TestHallucinatedCapability:
    """Test validate_single_step rejects unknown capabilities."""

    def test_hallucinated_capability_raises(self, mock_logger, mock_registry):
        """Non-existent capability raises ValueError."""
        mock_registry.get_node.return_value = None

        step = PlannedStep(
            context_key="test_step",
            capability="nonexistent_capability",
            task_objective="Do something impossible",
        )
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            with pytest.raises(ValueError, match="nonexistent_capability"):
                validate_single_step(step, state, mock_logger)


class TestContextKeyValidation:
    """Test input context key reference validation."""

    def test_valid_context_ref_from_existing_state(self, mock_logger, mock_registry):
        """Step referencing existing context key passes."""
        step = PlannedStep(
            context_key="read_result",
            capability="channel_read",
            task_objective="Read channels",
            inputs=[{"CHANNEL_ADDRESSES": "found_channels"}],
        )
        state = {
            "capability_context_data": {
                "CHANNEL_ADDRESSES": {"found_channels": {"pvs": ["SR:C01"]}}
            }
        }

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            # Should not raise
            validate_single_step(step, state, mock_logger)

    def test_invalid_context_ref_raises(self, mock_logger, mock_registry):
        """Step referencing nonexistent context key raises InvalidContextKeyError."""
        step = PlannedStep(
            context_key="read_result",
            capability="channel_read",
            task_objective="Read channels",
            inputs=[{"CHANNEL_ADDRESSES": "nonexistent_key"}],
        )
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            with pytest.raises(InvalidContextKeyError, match="nonexistent_key"):
                validate_single_step(step, state, mock_logger)

    def test_no_requires_no_inputs(self, mock_logger, mock_registry):
        """Step with no inputs passes validation."""
        step = PlannedStep(
            context_key="test_step",
            capability="channel_finding",
            task_objective="Find channels",
            inputs=[],
        )
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            validate_single_step(step, state, mock_logger)

    def test_with_available_keys_map(self, mock_logger, mock_registry):
        """Test with explicitly provided available_keys map."""
        step = PlannedStep(
            context_key="read_result",
            capability="channel_read",
            task_objective="Read channels",
            inputs=[{"CHANNEL_ADDRESSES": "step1_channels"}],
        )
        state = {"capability_context_data": {}}
        available_keys = {"step1_channels": (1, "channel_finding")}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            # step_index=2 means step1_channels (at index 1) is valid
            validate_single_step(
                step, state, mock_logger, available_keys=available_keys, step_index=2
            )


class TestFullValidatorRegression:
    """Ensure the refactored _validate_and_fix_execution_plan still works."""

    def test_full_validator_still_catches_hallucinated(self, mock_logger, mock_registry):
        """Full validator raises ValueError for hallucinated capabilities."""
        mock_registry.get_node.return_value = None

        plan = {
            "steps": [
                PlannedStep(
                    context_key="s1",
                    capability="fake_cap",
                    task_objective="Do something",
                )
            ]
        }
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            with pytest.raises(ValueError, match="hallucinated"):
                _validate_and_fix_execution_plan(plan, "test task", state, mock_logger)

    def test_full_validator_adds_respond_step(self, mock_logger, mock_registry):
        """Full validator adds respond step if plan doesn't end with one."""
        plan = {
            "steps": [
                PlannedStep(
                    context_key="s1",
                    capability="channel_finding",
                    task_objective="Find channels",
                    inputs=[],
                )
            ]
        }
        state = {"capability_context_data": {}}

        with patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ):
            result = _validate_and_fix_execution_plan(plan, "test task", state, mock_logger)
            steps = result.get("steps", [])
            assert len(steps) == 2
            assert steps[-1]["capability"] == "respond"
