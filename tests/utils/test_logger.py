"""
Tests for unified logging and streaming system.

Tests cover:
- Logger without streaming context (module-level usage)
- Backward compatibility with old API
"""

import pytest

from osprey_connectors.logger import ComponentLogger, get_logger


class TestComponentLoggerBasic:
    """Test basic ComponentLogger functionality without streaming."""

    def test_logger_creation(self):
        """A logger is looked up by its name alone (module-level usage)."""
        logger = get_logger("test_component")
        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "test_component"

    def test_name_is_positional_only(self):
        """``get_logger`` takes the name positionally and nothing else."""
        with pytest.raises(TypeError):
            get_logger(name="custom_logger")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            get_logger("custom_logger", color="blue")  # type: ignore[call-arg]

    @pytest.mark.parametrize("retired", ["status", "success", "timing", "approval", "resume"])
    def test_retired_intent_methods_are_gone(self, retired):
        """The retired intents log through ``info`` now."""
        assert not hasattr(get_logger("test_component"), retired)

    def test_basic_logging_methods(self):
        """Test that basic logging methods work without crashing."""
        logger = get_logger("test_component")

        # These should all work without streaming
        logger.info("Info message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.key_info("Key info message")


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
