"""Render log records off the thread that logs them.

The Osprey ``RichHandler`` draws a record inside ``logging``'s ``emit``, on
whichever thread logged it. In a server that thread is the event loop, and a
record that is expensive to draw — a syntax-highlighted traceback takes seconds
— stops every request for as long as it draws.
:func:`rendering_logs_off_thread` puts a queue between the two for as long as a
server runs: the logging call enqueues and returns, and one listener thread
draws the records in the order they were logged.
"""

from __future__ import annotations

import logging
import queue
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from typing import TYPE_CHECKING

from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["rendering_logs_off_thread"]


class _RecordQueueHandler(QueueHandler):
    """Hands each record to the queue as it is.

    The stock handler formats a record before it enqueues it, which folds the
    traceback into the message so the record can cross a process boundary. The
    listener here is a thread of the same process, and the renderer is what
    decides how an exception is drawn.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


@contextmanager
def rendering_logs_off_thread() -> Iterator[None]:
    """Draw the root logger's Rich output on a listener thread while the block runs.

    Every ``RichHandler`` on the root logger is taken off it and served from a
    queue instead, keeping its own level and filters. Other handlers — a host
    application's, a test's capture — stay where they are and stay synchronous.
    Leaving the block, however it is left, drains the queue before the handlers
    go back on the root logger, so the last records of a server are not lost.

    With no ``RichHandler`` on the root logger this does nothing.

    Yields:
        None, with the queue in place.
    """
    root = logging.getLogger()
    before = list(root.handlers)
    renderers = [handler for handler in before if isinstance(handler, RichHandler)]
    if not renderers:
        yield
        return

    relay = _RecordQueueHandler(queue.SimpleQueue())
    listener = QueueListener(relay.queue, *renderers, respect_handler_level=True)
    for handler in renderers:
        root.removeHandler(handler)
    root.addHandler(relay)
    listener.start()
    try:
        yield
    finally:
        root.removeHandler(relay)
        # Stopping joins the listener behind everything already queued.
        listener.stop()
        # Back in their old places, around whatever was added meanwhile.
        added = [handler for handler in root.handlers if handler not in before]
        root.handlers[:] = before + added
