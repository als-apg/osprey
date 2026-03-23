=====================
Recovery Coordination
=====================

Router-based recovery strategies and infrastructure error coordination.

.. currentmodule:: osprey

The framework implements intelligent recovery coordination through a centralized router system that automatically determines recovery strategies based on error classification. The system coordinates between retry mechanisms, orchestrator replanning, and graceful termination.

Router Recovery System
======================

Router Conditional Edge
-----------------------

.. autofunction:: osprey.infrastructure.router_node.router_conditional_edge

   Central recovery coordination implementing the complete recovery strategy.

   **Recovery Flow:**

   1. **Manual Retry Handling** (checked first):
      - RETRIABLE errors: retry with backoff if attempts remain
      - REPLANNING errors: route to orchestrator if planning attempts remain
      - RECLASSIFICATION errors: route to classifier if reclassification attempts remain
      - CRITICAL errors: route to error node immediately

   2. **Normal Routing Logic**:
      - Check execution state and route to next capability or termination

RouterNode Infrastructure
-------------------------

.. autoclass:: osprey.infrastructure.router_node.RouterNode
   :members:
   :show-inheritance:

   Infrastructure node that coordinates routing decisions and state management.

Orchestrator Replanning
=======================

OrchestrationNode
-----------------

.. automethod:: osprey.infrastructure.orchestration_node.OrchestrationNode.execute

   Handles replanning when REPLANNING errors are encountered:

   - **Plan Creation**: Generate new execution plan based on current task
   - **Capability Validation**: Ensure all planned capabilities exist
   - **State Updates**: Clear error state and increment plan counter
   - **Limits**: Respect maximum planning attempts to prevent infinite loops

Classifier Reclassification
===========================

ClassificationNode
-------------------

.. automethod:: osprey.infrastructure.classification_node.ClassificationNode.execute

   Handles reclassification when RECLASSIFICATION errors are encountered:

   - **Context Awareness**: Uses previous failure context for improved classification
   - **Capability Reselection**: Analyzes available capabilities with failure context
   - **State Updates**: Clear reclassification flags and increment counters
   - **Limits**: Respect maximum reclassification attempts (configurable via ``max_reclassifications``)
   - **Fresh Analysis**: Resets planning state for completely new capability selection

Error Response Generation
=========================

ErrorNode System
----------------

.. autoclass:: osprey.infrastructure.error_node.ErrorNode
   :members:
   :show-inheritance:

   Provides comprehensive error handling with LLM-powered analysis and structured error reports.

Infrastructure Error Classification
===================================

Classification Node
-------------------

.. automethod:: osprey.infrastructure.classification_node.ClassificationNode.classify_error

   Built-in error classification for classifier operations with LLM-aware retry logic.

.. automethod:: osprey.infrastructure.classification_node.ClassificationNode.get_retry_policy

   Custom retry policy for LLM-based classification operations.

.. seealso::

   :doc:`01_classification_system`
       Error classification and severity management

   :doc:`02_exception_reference`
       Complete exception hierarchy with inheritance structure

   :doc:`../02_infrastructure/05_execution-control`
       **ErrorNode**, **RouterNode**, and complete execution control flow
