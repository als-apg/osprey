================
Python Execution
================

Python code execution service with approval integration and flexible deployment options.

.. note::
   For implementation tutorials and usage examples, see :doc:`../../../developer-guides/05_production-systems/03_python-execution-service/index`.

Request and Response Models
===========================

.. currentmodule:: osprey.services.python_executor.models

.. autoclass:: PythonExecutionRequest
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: PythonServiceResult
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: PythonExecutionSuccess
   :members:
   :show-inheritance:
   :no-index:

State Management
================

.. currentmodule:: osprey.services.python_executor.models

.. autoclass:: PythonExecutionContext
   :members:
   :show-inheritance:
   :no-index:

Configuration Models
====================

.. currentmodule:: osprey.services.python_executor.config

.. autoclass:: PythonExecutorConfig
   :members:
   :show-inheritance:
   :no-index:

.. currentmodule:: osprey.services.python_executor.execution.control

.. autoclass:: ExecutionControlConfig
   :members:
   :show-inheritance:
   :no-index:

Notebook Management
===================

.. currentmodule:: osprey.services.python_executor.models

.. autoclass:: NotebookAttempt
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: NotebookType
   :members:
   :show-inheritance:
   :no-index:

Exceptions
==========

.. currentmodule:: osprey.services.python_executor.exceptions

.. autoclass:: PythonExecutorException
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: CodeRuntimeError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ContainerConnectivityError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ExecutionTimeoutError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ChannelLimitsViolationError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ErrorCategory
   :members:
   :show-inheritance:
   :no-index:

Serialization Utilities
=======================

.. currentmodule:: osprey.services.python_executor.services

.. autofunction:: make_json_serializable

.. autofunction:: serialize_results_to_file

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/03_python-execution-service/index`
       Complete implementation guide and examples

   :class:`osprey.capabilities.python.PythonCapability`
       Capability interface that uses this service
