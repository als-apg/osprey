"""Utility for capturing logging output."""

import contextlib
import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures log messages to a StringIO buffer.

    This handler is designed to work in multiprocessing contexts.
    It uses a list-based buffer instead of StringIO with locks to avoid multiprocessing
    lock issues.
    """

    def __init__(self) -> None:
        """Initialize the log capture handler."""
        super().__init__()
        self._messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the buffer.

        Args:
            record: Log record to emit
        """
        try:
            msg = self.format(record)
            # Append to list (thread-safe in CPython due to GIL, and list.append is atomic)
            self._messages.append(msg)
        except Exception:
            self.handleError(record)

    def get_logs(self) -> str:
        """Get all captured logs.

        Returns:
            String containing all captured log messages
        """
        return "\n".join(self._messages) + ("\n" if self._messages else "")

    def clear(self) -> None:
        """Clear the log buffer."""
        self._messages.clear()


@contextlib.contextmanager
def capture_logs() -> Iterator[LogCaptureHandler]:
    """Attach a ``LogCaptureHandler`` to the root logger for the duration of the block.

    Yields:
        LogCaptureHandler collecting everything logged inside the block.
    """
    root_logger = logging.getLogger()
    handler = LogCaptureHandler()
    handler.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    logger.debug("capture_logs: started capture, %d handlers on root", len(root_logger.handlers))
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)
