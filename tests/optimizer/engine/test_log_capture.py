"""Tests for utils/log_capture.py — LogCaptureHandler and capture_logs."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from osprey_optimizer.utils.log_capture import LogCaptureHandler, capture_logs


def _make_record(msg: str) -> logging.LogRecord:
    """Create a simple INFO LogRecord with the given message."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestLogCaptureHandler:
    def test_get_logs_accumulates_until_cleared(self) -> None:
        handler = LogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        for msg in ("one", "two", "three"):
            handler.emit(_make_record(msg))
        assert handler.get_logs() == "one\ntwo\nthree\n"

        handler.clear()
        assert handler.get_logs() == ""

    def test_emit_exception_calls_handle_error(self) -> None:
        handler = LogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = _make_record("ok")
        with (
            patch.object(handler, "format", side_effect=RuntimeError("boom")),
            patch.object(handler, "handleError") as mock_handle,
        ):
            handler.emit(record)
            mock_handle.assert_called_once_with(record)


class TestCaptureLogs:
    @pytest.mark.parametrize(
        "exit_via_exception", [False, True], ids=["normal_exit", "exception_exit"]
    )
    def test_handler_added_on_enter_and_removed_on_exit(self, exit_via_exception: bool) -> None:
        """The handler joins the root logger on enter and leaves it on exit,
        whether the block exits normally or via an exception."""
        root = logging.getLogger()
        raises = (
            pytest.raises(ValueError, match="deliberate") if exit_via_exception else nullcontext()
        )
        with raises, capture_logs() as handler:
            assert handler in root.handlers
            if exit_via_exception:
                raise ValueError("deliberate")
        assert handler not in root.handlers
