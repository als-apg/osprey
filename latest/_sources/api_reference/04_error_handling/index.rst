===============
Error Handling
===============

.. toctree::
   :maxdepth: 2
   :caption: Error Handling Components
   :hidden:

   01_classification_system
   02_exception_reference
   03_recovery_coordination

.. dropdown:: What You'll Find Here
   :color: primary
   :icon: book

   **Comprehensive error handling system for user-friendly agentic operations:**

   - **Classification System** - Four-tier severity levels (RETRIABLE, REPLANNING, CRITICAL, FATAL) with domain-specific error analysis and recovery strategy coordination
   - **Exception Hierarchy** - Complete catalog of Osprey Framework exceptions including base errors, Python executor categories, memory operations, and time parsing with inheritance structure
   - **Recovery Coordination** - Router-based automatic recovery with exponential backoff, orchestrator replanning, LLM-powered error responses, and graceful degradation patterns
   - **Production Integration** - Network-aware classification, data validation triggers, custom retry policies, and category-based handling for real-world deployment scenarios

   **Prerequisites:** Basic understanding of framework capabilities and infrastructure components

   **Target Audience:** Framework users, capability developers, scientists running experiments, infrastructure maintainers

Sophisticated error handling and recovery system designed for production-grade agentic systems. The Osprey Framework implements a comprehensive three-layer error management architecture that provides intelligent error classification, automatic recovery coordination, and graceful degradation patterns.

.. currentmodule:: osprey

Architecture Overview
=====================

The Osprey Framework implements **Manual Retry Coordination** with intelligent recovery strategies designed for domain experts who need fast debugging cycles without extensive stack traces. While traditional error traces remain valuable for developers, agentic systems are often used by scientists and researchers who need immediate, actionable feedback when experiments fail:

**Traditional Approach:**

.. code-block:: text

   Error → Generic Retry → Fail → Cryptic User Notification

**Intelligent Recovery Approach:**

.. code-block:: text

   Error → Classification → Targeted Recovery → Escalation → LLM-Powered Response

**Benefits:** User-friendly error messages, automatic recovery without intervention, rapid experiment iteration.

The Three Layers
================

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 🎯 Classification System
      :link: 01_classification_system
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      **Intelligent Error Analysis**

      Severity-based classification with sophisticated recovery strategy selection and context-aware analysis.

   .. grid-item-card:: 📋 Exception Hierarchy
      :link: 02_exception_reference
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      **Comprehensive Error Catalog**

      Structured exception classes with domain-specific recovery hints and detailed categorization.

   .. grid-item-card:: 🔄 Recovery Coordination
      :link: 03_recovery_coordination
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      **Automated Recovery Strategies**

      Router-based recovery coordination with retry policies, replanning, and graceful termination.

Recovery Strategy Integration
=============================

The system coordinates recovery through a unified strategy hierarchy:

.. tab-set::

   .. tab-item:: Error Classification

      How errors are analyzed and categorized:

      .. code-block:: python

         # Domain-specific error classification
         @staticmethod
         def classify_error(exc: Exception, context: dict) -> ErrorClassification:
             if isinstance(exc, (ConnectionError, TimeoutError)):
                return ErrorClassification(
                    severity=ErrorSeverity.RETRIABLE,
                    user_message="Network issue detected, retrying...",
                    metadata={"technical_details": str(exc)}
                )
             elif isinstance(exc, KeyError) and "context" in str(exc):
                                 return ErrorClassification(
                    severity=ErrorSeverity.REPLANNING,
                    user_message="Required data not available, trying different approach",
                    metadata={
                        "technical_details": f"Missing context data: {str(exc)}",
                        "replanning_reason": f"Missing required context: {exc}",
                        "suggestions": ["Verify data dependencies", "Check previous steps"]
                    }
                )
             # Default from BaseCapability implementation
             capability_name = context.get('capability', 'unknown_capability')
                         return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unhandled error in {capability_name}: {exc}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Unhandled error in {capability_name}: {exc}",
                    "suggestions": ["Check capability logs", "Contact support team"]
                }
            )

   .. tab-item:: Recovery Coordination

      Router-based automatic recovery strategies:

      .. code-block:: python

         import time

         # Router coordinates all recovery strategies
         if error_classification.severity == ErrorSeverity.RETRIABLE:
             if retry_count < max_retries:
                 # Calculate delay with backoff for this retry attempt
                 actual_delay = delay_seconds * (backoff_factor ** (retry_count - 1)) if retry_count > 0 else 0

                 # Apply delay if this is a retry (not the first attempt)
                 if retry_count > 0 and actual_delay > 0:
                     time.sleep(actual_delay)  # Simple sleep for now, could be async

                 # Increment retry count in state before routing back
                 state['control_retry_count'] = retry_count + 1
                 return capability_name  # Retry same capability
             else:
                 return "error"  # Retries exhausted → ErrorNode

         elif error_classification.severity == ErrorSeverity.REPLANNING:
             # Check how many plans have been created by orchestrator
             current_plans_created = state.get('control_plans_created_count', 0)

             # Get max planning attempts from execution limits config
             limits = get_execution_limits()
             max_planning_attempts = limits.get('max_planning_attempts', 2)

             if current_plans_created < max_planning_attempts:
                 return "orchestrator"  # Create new execution plan
             else:
                 return "error"  # Planning attempts exhausted → ErrorNode

         elif error_classification.severity == ErrorSeverity.CRITICAL:
             return "error"  # Immediate termination → ErrorNode

   .. tab-item:: Production Patterns

      Real-world error handling patterns:

      .. code-block:: python

         # LLM-aware retry policy for infrastructure operations
         @staticmethod
         def get_retry_policy() -> Dict[str, Any]:
             return {
                 "max_attempts": 4,        # More attempts for LLM operations
                 "delay_seconds": 2.0,     # Longer initial delay
                 "backoff_factor": 2.0     # Aggressive backoff for rate limiting
             }

         # Category-based Python executor error handling
         try:
             result = await executor.execute_code(code)
         except PythonExecutorException as e:
             if e.should_retry_execution():
                 # Infrastructure error - retry same code
                 logger.info("Infrastructure issue, retrying execution...")
                 await retry_execution_with_backoff(code)
             elif e.should_retry_code_generation():
                 # Code error - regenerate and retry
                 logger.info("Code issue, regenerating with feedback...")
                 improved_code = await regenerate_with_feedback(str(e))
                 await execute_code(improved_code)
             else:
                 # Workflow error - requires intervention
                 logger.error(f"Execution failed: {e.message}")
                 await notify_user(f"Execution failed: {e.message}")

.. dropdown:: 🚀 Next Steps

   Master error handling implementation and integration patterns:

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🎯 Start with Classification
         :link: 01_classification_system
         :link-type: doc
         :class-header: bg-primary text-white
         :class-body: text-center
         :shadow: md

         Error severity levels, classification results, and recovery strategy coordination

      .. grid-item-card:: 📋 Explore Exceptions
         :link: 02_exception_reference
         :link-type: doc
         :class-header: bg-success text-white
         :class-body: text-center
         :shadow: md

         Complete exception hierarchy with usage patterns and domain-specific handling

   .. grid:: 1 1 2 2
      :gutter: 3
      :class-container: guides-section-grid

      .. grid-item-card:: 🔄 Master Recovery
         :link: 03_recovery_coordination
         :link-type: doc
         :class-header: bg-info text-white
         :class-body: text-center
         :shadow: md

         Router coordination, retry policies, replanning triggers, and recovery examples

      .. grid-item-card:: 🏗️ Integration Patterns
         :link: ../../developer-guides/04_infrastructure-components/06_error-handling-infrastructure
         :link-type: doc
         :class-header: bg-warning text-white
         :class-body: text-center
         :shadow: md

         Implementation examples, custom error handling, and production deployment patterns

.. note::
   The framework uses manual retry coordination rather than LangGraph's native retry policies to ensure consistent behavior and sophisticated error classification across all components.
