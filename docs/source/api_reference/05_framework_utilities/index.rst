===================
Framework Utilities
===================

Supporting systems for advanced usage and development tooling.

LLM Completion Interface
========================

Multi-provider LLM completions via LiteLLM for structured generation and direct completions.

.. currentmodule:: osprey.models

.. autofunction:: get_chat_completion

Developer Tools
===============

Unified logging system with automatic LangGraph streaming support for framework development.

Logging and Streaming
---------------------

The framework provides a unified logging API that automatically handles both CLI output
and web UI streaming. Use ``logger.status()`` for high-level updates that should appear
in both interfaces, and standard logging methods (``info()``, ``debug()``) for detailed
CLI-only output.

**Recommended Pattern:**

.. code-block:: python

   # In capabilities - automatic streaming
   logger = self.get_logger()
   logger.status("Creating execution plan...")  # Logs + streams
   logger.info("Active capabilities: [...]")   # Logs only

   # In other nodes with state
   logger = get_logger("orchestrator", state=state)
   logger.status("Processing...")  # Logs + streams

.. currentmodule:: osprey.utils.logger

.. autofunction:: get_logger

.. autoclass:: ComponentLogger
   :members:
   :show-inheritance:

Legacy Streaming API (Deprecated)
----------------------------------

.. deprecated:: 0.9.2
   The separate streaming API is deprecated in favor of the unified logging system.
   Use :meth:`osprey.base.capability.BaseCapability.get_logger` in capabilities or
   :func:`get_logger` with ``state`` parameter for automatic streaming support.

For backward compatibility only. New code should use the unified logging system above.

.. currentmodule:: osprey.utils.streaming

.. autofunction:: get_streamer

.. autoclass:: StreamWriter
   :members:
   :show-inheritance:

.. seealso::

   :doc:`../../developer-guides/01_understanding-the-framework/04_orchestration-architecture`
       Development utilities integration patterns and configuration conventions
