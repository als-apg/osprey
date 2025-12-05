"""Tests for logging filter utilities.

These tests verify the quiet_logger context manager and related filtering
functionality that is used throughout the codebase to suppress verbose output.
"""

import logging
from io import StringIO

import pytest

from osprey.utils.log_filter import LoggerFilter, quiet_logger


class TestQuietLogger:
    """Test the quiet_logger context manager."""

    def test_suppresses_info_messages(self, caplog):
        """Test that INFO messages are suppressed within context."""
        logger = logging.getLogger("TEST_LOGGER")

        with caplog.at_level(logging.INFO):
            # Outside context - should appear
            logger.info("This should appear")

            # Inside context - should be suppressed
            with quiet_logger("TEST_LOGGER"):
                logger.info("This should be suppressed")

            # After context - should appear again
            logger.info("This should also appear")

        # Check that only 2 messages appeared (not 3)
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages) == 2
        assert "This should appear" in info_messages[0]
        assert "This should also appear" in info_messages[1]
        assert "This should be suppressed" not in str(info_messages)

    def test_warnings_still_show_through(self, caplog):
        """Test that WARNING and ERROR messages still appear."""
        logger = logging.getLogger("TEST_LOGGER")

        with caplog.at_level(logging.WARNING):
            with quiet_logger("TEST_LOGGER"):
                logger.warning("Warning should show")
                logger.error("Error should show")

        assert len(caplog.records) == 2
        assert "Warning should show" in caplog.text
        assert "Error should show" in caplog.text

    def test_multiple_loggers_suppressed(self, caplog):
        """Test suppressing multiple loggers at once."""
        logger1 = logging.getLogger("LOGGER1")
        logger2 = logging.getLogger("LOGGER2")

        with caplog.at_level(logging.INFO):
            with quiet_logger(["LOGGER1", "LOGGER2"]):
                logger1.info("Suppressed 1")
                logger2.info("Suppressed 2")

        # Both should be suppressed
        assert len(caplog.records) == 0

    def test_filter_is_removed_after_context(self, caplog):
        """Test that filter is properly removed when exiting context."""
        logger = logging.getLogger("TEST_LOGGER")

        # Test that suppression works and then stops working
        with caplog.at_level(logging.INFO):
            logger.info("Before context")

            with quiet_logger("TEST_LOGGER"):
                logger.info("Inside context - suppressed")

            logger.info("After context")

        # Should have 2 messages (before and after, not during)
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages) == 2

    def test_filter_removed_even_on_exception(self, caplog):
        """Test that filter is removed even if exception occurs."""
        logger = logging.getLogger("TEST_LOGGER")

        # Test that filter cleanup happens even on exception
        with caplog.at_level(logging.INFO):
            logger.info("Before exception")

            try:
                with quiet_logger("TEST_LOGGER"):
                    logger.info("During context - suppressed")
                    raise ValueError("Test exception")
            except ValueError:
                pass

            logger.info("After exception")

        # Should have 2 messages (before and after exception context)
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages) == 2


class TestLoggerFilter:
    """Test the LoggerFilter class directly."""

    def test_filters_specific_logger_only(self, caplog):
        """Test that filter only affects specified logger."""
        target_logger = logging.getLogger("TARGET")
        other_logger = logging.getLogger("OTHER")

        log_filter = LoggerFilter(logger_names=["TARGET"])
        target_logger.addFilter(log_filter)

        with caplog.at_level(logging.INFO):
            target_logger.info("Filtered")
            other_logger.info("Not filtered")

        messages = [r.message for r in caplog.records]
        assert "Not filtered" in messages
        assert "Filtered" not in messages

        target_logger.removeFilter(log_filter)

    def test_message_pattern_filtering(self, caplog):
        """Test filtering by message pattern."""
        logger = logging.getLogger("PATTERN_TEST")

        log_filter = LoggerFilter(
            logger_names=["PATTERN_TEST"],
            message_patterns=[r"Configured.*registry", r"Added.*path"],
        )
        logger.addFilter(log_filter)

        with caplog.at_level(logging.INFO):
            logger.info("Configured the registry")  # Should be filtered
            logger.info("Added to sys.path")  # Should be filtered
            logger.info("Normal message")  # Should appear

        messages = [r.message for r in caplog.records]
        assert len(messages) == 1
        assert "Normal message" in messages[0]

        logger.removeFilter(log_filter)
