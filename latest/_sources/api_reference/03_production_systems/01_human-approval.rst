==============
Human Approval
==============

LangGraph-native approval system for production-ready human-in-the-loop workflows with configurable security policies.

.. note::
   For implementation guides and examples, see :doc:`../../../developer-guides/05_production-systems/01_human-approval-workflows`.

.. currentmodule:: osprey.approval

Configuration Management
========================

.. autoclass:: ApprovalManager
   :members:
   :show-inheritance:

.. autoclass:: GlobalApprovalConfig
   :members:
   :show-inheritance:

Configuration Models
====================

.. autoclass:: PythonExecutionApprovalConfig
   :members:
   :show-inheritance:

.. autoclass:: MemoryApprovalConfig
   :members:
   :show-inheritance:

.. autoclass:: ApprovalMode
   :members:
   :show-inheritance:

Business Logic Evaluators
==========================

.. autoclass:: PythonExecutionApprovalEvaluator
   :members:
   :show-inheritance:

.. autoclass:: MemoryApprovalEvaluator
   :members:
   :show-inheritance:

.. autoclass:: ApprovalDecision
   :members:
   :show-inheritance:

Approval System Functions
=========================

.. autofunction:: create_approval_type

.. autofunction:: create_code_approval_interrupt

.. autofunction:: create_plan_approval_interrupt

.. autofunction:: create_memory_approval_interrupt

.. autofunction:: get_approval_resume_data

Utility Functions
=================

.. autofunction:: get_approval_manager

.. seealso::

   :doc:`../../../developer-guides/05_production-systems/01_human-approval-workflows`
       Complete implementation guide and examples

   :class:`osprey.services.python_executor.PythonExecutorService`
       Service that integrates with approval system
