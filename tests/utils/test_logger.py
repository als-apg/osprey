"""
Tests for unified logging and streaming system.

Tests cover:
- Logger without streaming context (module-level usage)
- Backward compatibility with old API
"""

import pytest

from osprey.utils.logger import ComponentLogger, get_logger


class TestComponentLoggerBasic:
    """Test basic ComponentLogger functionality without streaming."""

    def test_logger_creation_without_state(self):
        """Test that logger can be created without state (module-level usage)."""
        logger = get_logger("test_component")
        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "test_component"
        assert logger._state is None

    def test_logger_creation_with_custom_params(self):
        """Test custom logger creation with explicit parameters."""
        logger = get_logger(name="custom_logger", color="blue")
        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "custom_logger"
        assert logger.color == "blue"

    def test_basic_logging_methods(self):
        """Test that basic logging methods work without crashing."""
        logger = get_logger("test_component")

        # These should all work without streaming
        logger.info("Info message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.success("Success message")
        logger.key_info("Key info message")
        logger.timing("Timing message")
        logger.approval("Approval message")
        logger.resume("Resume message")

    def test_status_without_streaming_context(self):
        """Test that status() works gracefully without streaming context."""
        logger = get_logger("test_component")

        # Should not raise even though no stream writer available
        logger.status("Test status message")


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_critical_and_exception_methods(self):
        """Test that critical and exception methods still work."""
        logger = get_logger("test_component")

        # These methods should still work
        logger.critical("Critical message")
        logger.exception("Exception message")

    def test_logger_properties(self):
        """Test that logger properties are accessible."""
        logger = get_logger("test_component")

        assert logger.name == "test_component"
        assert isinstance(logger.level, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
