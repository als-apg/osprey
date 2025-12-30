"""
Base Worker Thread for Osprey PyQt GUI

This module provides a base class for background worker threads with common
functionality like async loop management and error handling.
"""

import asyncio

from PyQt5.QtCore import QThread, pyqtSignal

from osprey.utils.logger import get_logger

logger = get_logger("base_worker")


class BaseWorker(QThread):
    """
    Base class for background worker threads.

    Provides common functionality:
    - Async event loop management
    - Standardized error handling
    - Signal definitions for common events

    Subclasses should override the execute() method to implement
    their specific processing logic.

    Signals:
        error_occurred: Emitted when an error occurs (error_message)
        processing_complete: Emitted when processing completes successfully
    """

    error_occurred = pyqtSignal(str)
    processing_complete = pyqtSignal()

    def __init__(self):
        """Initialize the base worker."""
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._should_stop = False

    def run(self):
        """
        Execute the worker in a background thread.

        This method sets up the async event loop, calls execute(),
        and handles cleanup. Subclasses should NOT override this method.
        """
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Execute the worker's main logic
            self.execute()

            # Emit completion signal if not stopped
            if not self._should_stop:
                self.processing_complete.emit()

        except Exception as e:
            logger.exception(f"Error in {self.__class__.__name__}: {e}")
            self.error_occurred.emit(str(e))
        finally:
            # Clean up event loop
            if self._loop:
                try:
                    # Cancel any pending tasks
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()

                    # Run loop one more time to handle cancellations
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception as e:
                    logger.warning(f"Error during loop cleanup: {e}")
                finally:
                    self._loop.close()
                    self._loop = None

    def execute(self):
        """
        Execute the worker's main logic.

        Subclasses MUST override this method to implement their
        specific processing logic. This method runs in the worker thread
        with an active async event loop available via self._loop.

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def run_async(self, coro):
        """
        Run an async coroutine in the worker's event loop.

        Args:
            coro: Coroutine to execute

        Returns:
            Result from the coroutine

        Raises:
            RuntimeError: If called before event loop is initialized
        """
        if not self._loop:
            raise RuntimeError("Event loop not initialized. Call from execute() only.")

        return self._loop.run_until_complete(coro)

    def stop(self):
        """
        Request the worker to stop processing.

        This sets a flag that subclasses can check to gracefully
        terminate their processing.
        """
        self._should_stop = True
        logger.debug(f"{self.__class__.__name__} stop requested")

    def should_stop(self) -> bool:
        """
        Check if the worker should stop processing.

        Returns:
            True if stop was requested, False otherwise
        """
        return self._should_stop

    def handle_error(self, error: Exception, context: str = ""):
        """
        Handle an error with standardized logging and signaling.

        Args:
            error: The exception that occurred
            context: Optional context string for better error messages
        """
        error_msg = f"{context}: {error}" if context else str(error)
        logger.exception(f"Error in {self.__class__.__name__} - {error_msg}")
        self.error_occurred.emit(error_msg)
