=====================
Classification System
=====================

Core error classification and severity management system.

.. currentmodule:: osprey.base.errors

The classification system provides the foundation for intelligent error handling by enabling automatic recovery strategy selection based on error severity and context. This system integrates seamlessly with both capability execution and infrastructure operations.

Error Severity Levels
=====================

ErrorSeverity
-------------

.. autoclass:: ErrorSeverity
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of error severity levels with recovery strategies:

   - **RETRIABLE**: Retry execution with exponential backoff
   - **REPLANNING**: Route to orchestrator for new execution plan
   - **RECLASSIFICATION**: Route to classifier for new capability selection
   - **CRITICAL**: Graceful termination with user notification
   - **FATAL**: Immediate system termination

   .. rubric:: Usage Pattern

   .. code-block:: python

      if isinstance(exc, ConnectionError):
          return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)
      elif isinstance(exc, AuthenticationError):
          return ErrorClassification(
              severity=ErrorSeverity.CRITICAL,
              metadata={"safety_abort_reason": "Authentication failed"}
          )
      elif isinstance(exc, CapabilityMismatchError):
          return ErrorClassification(
              severity=ErrorSeverity.RECLASSIFICATION,
              user_message="Required capability not available",
              metadata={"reclassification_reason": "Capability mismatch detected"}
          )

Classification Results
======================

ErrorClassification
-------------------

.. autoclass:: ErrorClassification
   :members:
   :show-inheritance:

   Structured error analysis result that determines recovery strategy.

   .. rubric:: Basic Usage Pattern

   .. code-block:: python

      classification = ErrorClassification(
          severity=ErrorSeverity.RETRIABLE,
          user_message="Network connection timeout, retrying...",
          metadata={"technical_details": "HTTP request timeout after 30 seconds"}
      )

   .. rubric:: Advanced Usage with Rich Metadata

   .. code-block:: python

      classification = ErrorClassification(
          severity=ErrorSeverity.CRITICAL,
          user_message="Service validation failed",
          metadata={
              "technical_details": "Authentication service returned 403",
              "safety_abort_reason": "Security validation failed",

              "retry_after": 30,
              "error_code": "AUTH_FAILED"
          }
      )

ExecutionError
--------------

.. autoclass:: ExecutionError
   :members:
   :show-inheritance:

   Comprehensive error container with recovery coordination support.

   .. rubric:: Usage Pattern

   .. code-block:: python

    error = ExecutionError(
         severity=ErrorSeverity.RETRIABLE,
         message="Database connection failed",
         capability_name="database_query",
         metadata={"technical_details": "PostgreSQL connection timeout after 30 seconds"}
     )

Classification Methods
======================

Base Capability Classification
------------------------------

.. automethod:: osprey.base.capability.BaseCapability.classify_error

   Domain-specific error classification for capabilities. Override this method to provide sophisticated error handling based on specific failure modes.

   .. rubric:: Classification Strategy

   .. code-block:: python

      @staticmethod
      def classify_error(exc: Exception, context: dict) -> ErrorClassification:
          if isinstance(exc, ConnectionError):
              return ErrorClassification(
                  severity=ErrorSeverity.RETRIABLE,
                  user_message="Network issue detected, retrying...",
                  metadata={"technical_details": str(exc)}
              )
              return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected error: {exc}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Unhandled capability error: {exc}",
                    "suggestions": ["Check system logs", "Contact support if issue persists"]
                }
          )

Infrastructure Node Classification
----------------------------------

.. automethod:: osprey.base.nodes.BaseInfrastructureNode.classify_error

   Conservative error classification for infrastructure nodes. Infrastructure nodes handle system-critical functions, so failures typically require immediate attention.

   .. rubric:: Conservative Strategy

   .. code-block:: python

      @staticmethod
      def classify_error(exc: Exception, context: dict) -> ErrorClassification:
          # Infrastructure defaults to critical for fast failure
          return ErrorClassification(
             severity=ErrorSeverity.CRITICAL,
             user_message=f"Infrastructure error: {exc}",
             metadata={"technical_details": str(exc)}
         )

Retry Policy Configuration
==========================

.. automethod:: osprey.base.capability.BaseCapability.get_retry_policy

   Retry policy configuration for failure recovery strategies.

   **Default Policy:**

   .. code-block:: python

      {
          "max_attempts": 3,        # Total attempts including initial
          "delay_seconds": 0.5,     # Base delay before first retry
          "backoff_factor": 1.5     # Exponential backoff multiplier
      }

.. automethod:: osprey.base.nodes.BaseInfrastructureNode.get_retry_policy

   Conservative retry policy for infrastructure operations.

   **Infrastructure Policy:**

   .. code-block:: python

      {
          "max_attempts": 2,        # Fast failure for infrastructure
          "delay_seconds": 0.2,     # Quick retry attempt
          "backoff_factor": 1.0     # No backoff
      }

Integration Pattern
===================

Basic Error Handling
--------------------

.. code-block:: python

   try:
       result = await capability.execute(state)
   except Exception as exc:
       # Classify error for recovery strategy
       classification = capability.classify_error(exc, context)

       if classification.severity == ErrorSeverity.RETRIABLE:
           # Handle with retry policy
           policy = capability.get_retry_policy()
           await retry_with_backoff(capability, state, policy)
       elif classification.severity == ErrorSeverity.REPLANNING:
           # Route to orchestrator for new execution plan
           return "orchestrator"
       elif classification.severity == ErrorSeverity.RECLASSIFICATION:
           # Route to classifier for new capability selection
           return "classifier"
       elif classification.severity == ErrorSeverity.CRITICAL:
           # End execution with clear error message
           raise ExecutionError(
              severity=ErrorSeverity.CRITICAL,
              message=classification.user_message,
              metadata=classification.metadata
          )

Primary Error Context: Metadata Field
=====================================

The ``metadata`` field is the **primary mechanism** for providing structured error context in ``ErrorClassification``.

**Suggested Metadata Keys:**

- ``technical_details``: Detailed technical information (replaces old technical_details field)
- ``safety_abort_reason``: Explanation for critical/fatal errors requiring immediate termination
- ``replanning_reason``: Explanation for errors requiring new execution plan generation
- ``reclassification_reason``: Explanation for errors requiring new capability selection
- ``suggestions``: List of actionable recovery steps for users
- ``error_code``: Machine-readable error identifier for programmatic handling
- ``retry_after``: Suggested delay before retry attempts (in seconds)

**Advanced Usage Patterns:**

1. **Structured Technical Details**: Replace simple strings with nested objects
2. **Contextual Information**: Include relevant state and execution context
3. **Recovery Guidance**: Provide specific, actionable recovery steps
4. **System Integration**: Enable programmatic error handling and monitoring

.. code-block:: python

   # Example: Comprehensive metadata usage
   return ErrorClassification(
       severity=ErrorSeverity.REPLANNING,
       user_message="Data query scope too broad for current system limits",
       metadata={
           "technical_details": f"Query returned {result_count} results, limit is {max_limit}",
           "replanning_reason": "Query scope exceeded system performance limits",
           "suggestions": [
               "Reduce time range to last 24 hours",
               "Specify fewer measurement types",
               "Use data filtering parameters"
           ],
           "error_code": "QUERY_SCOPE_EXCEEDED",
           "retry_after": 10,
           "query_metrics": {
               "result_count": result_count,
               "max_limit": max_limit,
               "query_duration": query_time
           }
       }
   )

.. seealso::

   :doc:`02_exception_reference`
       Complete catalog of Osprey exceptions

   :doc:`03_recovery_coordination`
       Router coordination and recovery strategies
