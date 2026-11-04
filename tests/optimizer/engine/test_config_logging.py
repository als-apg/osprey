"""Tests for config/logging.py — setup_logging and configure_library_loggers."""

from __future__ import annotations

import logging
import sys

import pytest

from osprey_optimizer.config.logging import configure_library_loggers, setup_logging

# The global logging mutations these tests trigger are contained by the autouse
# ``restore_logging_state`` fixture in conftest.py.


class TestSetupLogging:
    def test_replaces_handlers_and_sets_level(self) -> None:
        root = logging.getLogger()
        dummy = logging.StreamHandler()
        root.addHandler(dummy)

        setup_logging(level=logging.DEBUG)

        assert dummy not in root.handlers
        assert root.level == logging.DEBUG
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        assert stream_handlers[-1].stream is sys.stdout


class TestConfigureLibraryLoggers:
    @pytest.mark.parametrize(
        "level,expected_xopt_level",
        [
            (logging.INFO, logging.WARNING),
            (logging.DEBUG, logging.DEBUG),
        ],
        ids=["info_suppresses_xopt", "debug_allows_xopt"],
    )
    def test_xopt_level_follows_base_level(self, level: int, expected_xopt_level: int) -> None:
        configure_library_loggers(level=level)
        assert logging.getLogger("xopt").level == expected_xopt_level
        # urllib3 is always pinned to WARNING regardless of base level.
        assert logging.getLogger("urllib3").level == logging.WARNING
