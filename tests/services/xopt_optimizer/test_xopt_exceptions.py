"""Unit Tests for XOpt Optimizer Exceptions.

This module tests the exception hierarchy for the XOpt optimizer service.
"""

from osprey.services.xopt_optimizer.exceptions import (
    ConfigGenerationError,
    ConfigurationError,
    ErrorCategory,
    MachineStateAssessmentError,
    MaxIterationsExceededError,
    XOptExecutionError,
    XOptExecutorException,
)


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_error_categories_exist(self):
        """All expected error categories should exist."""
        assert ErrorCategory.MACHINE_STATE.value == "machine_state"
        assert ErrorCategory.CONFIG_GENERATION.value == "config_generation"
        assert ErrorCategory.EXECUTION.value == "execution"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.WORKFLOW.value == "workflow"


class TestXOptExecutorException:
    """Test base exception class."""

    def test_base_exception_creation(self):
        """Base exception should be creatable with message."""
        exc = XOptExecutorException("Test error message")
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.category == ErrorCategory.WORKFLOW  # Default

    def test_base_exception_with_category(self):
        """Base exception should accept custom category."""
        exc = XOptExecutorException("Test error", category=ErrorCategory.MACHINE_STATE)
        assert exc.category == ErrorCategory.MACHINE_STATE

    def test_is_retriable(self):
        """is_retriable should return True for retriable categories."""
        machine_exc = XOptExecutorException("Test", category=ErrorCategory.MACHINE_STATE)
        config_exc = XOptExecutorException("Test", category=ErrorCategory.CONFIG_GENERATION)
        workflow_exc = XOptExecutorException("Test", category=ErrorCategory.WORKFLOW)

        assert machine_exc.is_retriable() is True
        assert config_exc.is_retriable() is True
        assert workflow_exc.is_retriable() is False

    def test_should_retry_config_generation(self):
        """should_retry_config_generation should return True for config gen errors."""
        config_exc = XOptExecutorException("Test", category=ErrorCategory.CONFIG_GENERATION)
        other_exc = XOptExecutorException("Test", category=ErrorCategory.EXECUTION)

        assert config_exc.should_retry_config_generation() is True
        assert other_exc.should_retry_config_generation() is False


class TestMachineStateAssessmentError:
    """Test MachineStateAssessmentError."""

    def test_creation(self):
        """Should be creatable with message and details."""
        exc = MachineStateAssessmentError(
            "Machine not ready",
            assessment_details={"reason": "No beam"},
        )
        assert exc.message == "Machine not ready"
        assert exc.category == ErrorCategory.MACHINE_STATE
        assert exc.assessment_details == {"reason": "No beam"}

    def test_is_retriable(self):
        """Machine state errors should be retriable."""
        exc = MachineStateAssessmentError("Test")
        assert exc.is_retriable() is True


class TestConfigGenerationError:
    """Test ConfigGenerationError."""

    def test_creation(self):
        """Should be creatable with message and config details."""
        exc = ConfigGenerationError(
            "Invalid config",
            generated_config={"bad": "config"},
            validation_errors=["Missing field X"],
        )
        assert exc.message == "Invalid config"
        assert exc.category == ErrorCategory.CONFIG_GENERATION
        assert exc.generated_config == {"bad": "config"}
        assert exc.validation_errors == ["Missing field X"]

    def test_should_retry_config_generation(self):
        """Config generation errors should trigger retry."""
        exc = ConfigGenerationError("Test")
        assert exc.should_retry_config_generation() is True


class TestXOptExecutionError:
    """Test XOptExecutionError."""

    def test_creation(self):
        """Should be creatable with message and execution details."""
        exc = XOptExecutionError(
            "XOpt failed",
            config_used={"algorithm": "random"},
            xopt_error="Runtime error",
        )
        assert exc.message == "XOpt failed"
        assert exc.category == ErrorCategory.EXECUTION
        assert exc.config_used == {"algorithm": "random"}
        assert exc.xopt_error == "Runtime error"

    def test_not_retriable(self):
        """Execution errors should not be retriable."""
        exc = XOptExecutionError("Test")
        assert exc.is_retriable() is False


class TestMaxIterationsExceededError:
    """Test MaxIterationsExceededError."""

    def test_creation(self):
        """Should be creatable with message and iteration count."""
        exc = MaxIterationsExceededError(
            "Max iterations reached",
            iterations_completed=5,
        )
        assert exc.message == "Max iterations reached"
        assert exc.category == ErrorCategory.WORKFLOW
        assert exc.iterations_completed == 5


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_creation(self):
        """Should be creatable with message and config key."""
        exc = ConfigurationError(
            "Invalid config",
            config_key="xopt_optimizer.max_iterations",
        )
        assert exc.message == "Invalid config"
        assert exc.category == ErrorCategory.CONFIGURATION
        assert exc.config_key == "xopt_optimizer.max_iterations"
