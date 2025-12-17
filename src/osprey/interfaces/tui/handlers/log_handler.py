"""Log handler for routing Python log records to TUI event queue."""

import asyncio
import logging


class QueueLogHandler(logging.Handler):
    """Routes Python log records to TUI event queue.

    Extracts ALL metadata from ComponentLogger's extra dict:
    - raw_message, log_type (for LOG section)
    - task, capabilities, steps, phase (for block lifecycle)
    """

    # Fields to extract from LogRecord extra dict
    EXTRA_FIELDS = [
        "task",
        "capabilities",
        "capability_names",
        "steps",
        "phase",
        "step_num",
        "step_name",
        "llm_prompt",
        "llm_response",
    ]

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Initialize the handler.

        Args:
            queue: The asyncio queue to send events to.
            loop: The event loop for thread-safe queue operations.
        """
        super().__init__()
        self.queue = queue
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the TUI event queue.

        Args:
            record: The log record to emit.
        """
        # Extract raw message from extra (set by ComponentLogger)
        raw_msg = getattr(record, "raw_message", None)
        log_type = getattr(record, "log_type", record.levelname.lower())

        # Skip if no raw message (not from ComponentLogger)
        if raw_msg is None:
            return

        event = {
            "event_type": "log",  # ALL TUI events are "log" type
            "level": log_type,
            "message": raw_msg,
            "component": record.name,
        }

        # Extract ALL streaming data fields
        for key in self.EXTRA_FIELDS:
            val = getattr(record, key, None)
            if val is not None:
                event[key] = val

        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            pass  # Event loop closed
