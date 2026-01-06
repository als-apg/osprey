==============
Human Approval
==============

**What you'll build:** Human approval workflows for production agent deployments with LangGraph-native interrupts and configurable security policies

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Implementing :class:`ApprovalManager` and policy configuration patterns
   - Using ``create_code_approval_interrupt()`` and ``get_approval_resume_data()`` functions
   - Configuring approval evaluators with :class:`PythonExecutionApprovalEvaluator`
   - LangGraph-native interrupt integration with ``interrupt()`` function
   - Security analysis patterns and domain-specific approval rules

   **Prerequisites:** Understanding of :doc:`../03_core-framework-systems/01_state-management-architecture` (AgentState) and LangGraph interrupts

   **Time Investment:** 30-45 minutes for complete understanding

Overview
========

The Human Approval system provides comprehensive approval workflows designed for high-stakes environments where human oversight is required for critical operations. The system integrates with LangGraph's interrupt mechanism to provide secure, resumable approval workflows with configurable policy management.

**Key Features:**

- **LangGraph-Native Interrupts**: Seamless workflow suspension using ``interrupt()`` function
- **Configurable Policies**: Domain-specific approval rules with multiple security modes
- **Security-First Design**: Fail-secure defaults with comprehensive validation
- **Rich Context**: Detailed approval information with safety assessments and code analysis
- **Resumable Workflows**: Checkpoint-based execution resumption after human approval

Architecture
============

The approval system implements a three-layer architecture:

**1. Configuration Layer (ApprovalManager)**
   Type-safe policy configuration with global and capability-specific settings

**2. Business Logic Layer (Evaluators)**
   Domain-specific approval decision logic with security analysis

**3. Integration Layer (Approval Functions)**
   LangGraph interrupt creation and state management

This separation ensures security policies can be modified without changing business logic, and new approval types can be added without framework modifications.

Configuration
=============

Configure your approval system in ``config.yml`` with global modes and capability-specific settings:

.. code-block:: yaml

   # Global approval configuration
   approval:
     global_mode: "selective"  # disabled, selective, all_capabilities
   capabilities:
     python_execution:
       enabled: true
       mode: "control_writes"  # disabled, all_code, control_writes
     memory:
       enabled: false

**Configuration Modes:**

- ``disabled``: No approval required (development only)
- ``selective``: Use capability-specific settings (recommended for production)
- ``all_capabilities``: Force approval for all operations (maximum security)

.. note::
   **Current Implementation Status**

   The human approval system is currently implemented for the following capabilities:

   - **Python Execution Capability**: Supports approval for all code execution or selective approval for EPICS write operations
   - **Memory Capability**: Supports approval for memory operations and data persistence

   Additional capabilities will be equipped with approval workflows in future releases based on operational requirements and security assessments.


**Python Execution Modes:**

- ``disabled``: No approval required for Python code
- ``all_code``: Approve all Python code execution
- ``control_writes``: Approve only code that writes to control systems (EPICS, Tango, etc.)

.. note::
   The old mode name ``epics_writes`` is deprecated but still supported for backward compatibility.
   It is automatically mapped to ``control_writes``.

Implementation Patterns
=======================

Basic Approval Integration
--------------------------

Integrate approval workflows into capabilities using the framework's approval functions:

.. code-block:: python

   from osprey.base import BaseCapability, capability_node
   from osprey.state import AgentState
   from osprey.context import ContextManager
   from osprey.approval import (
       create_code_approval_interrupt,
       get_approval_resume_data,
       get_python_execution_evaluator
   )
   from langgraph.types import interrupt

   @capability_node
   class PythonExecutionCapability(BaseCapability):
       """Python execution with human approval workflows."""

       async def execute(self, state: AgentState, context: ContextManager) -> dict:
           # Check for approval resume first
           has_resume, resume_payload = get_approval_resume_data(state, "python_executor")

           if has_resume and resume_payload:
               # Resume from approval - execute approved code
               approved_code = resume_payload['code']
               return await self._execute_code(approved_code)

           # Fresh execution - generate code and check approval
           generated_code = await self._generate_python_code(state, context)

           # Evaluate approval requirement
           evaluator = get_python_execution_evaluator()
           has_control_writes = self._analyze_for_control_writes(generated_code)

           decision = evaluator.evaluate(
               has_control_writes=has_control_writes,
               has_control_reads=False
           )

           if decision.needs_approval:
               # Create approval interrupt with rich context
               analysis_details = {
                   'safety_level': 'medium' if has_control_writes else 'low',
                   'operations_detected': ['control system writes'] if has_control_writes else [],
                   'risk_assessment': decision.reasoning
               }

               safety_concerns = []
               if has_control_writes:
                   safety_concerns.append("Code modifies control system setpoints")

               interrupt_data = create_code_approval_interrupt(
                   code=generated_code,
                   analysis_details=analysis_details,
                   execution_mode='write_access' if has_control_writes else 'readonly',
                   safety_concerns=safety_concerns
               )

               # Pause execution for human approval
               interrupt(interrupt_data)

           else:
               # No approval needed - execute directly
               return await self._execute_code(generated_code)

Approval Response Handling
--------------------------

Handle approval responses through LangGraph checkpoints:

.. code-block:: python

   def _handle_approval_response(self, state: AgentState) -> dict:
       """Handle approval response after workflow resumption."""
       has_resume, resume_payload = get_approval_resume_data(state, "python_executor")

       if not has_resume:
           return {"error": "No approval data found after resume"}

       approved = resume_payload.get('approved', False)

       if approved:
           approved_code = resume_payload['code']
           return self._execute_code(approved_code)
       else:
           return {
               "success": False,
               "message": "Code execution cancelled by user approval",
               "rejection_reason": resume_payload.get('rejection_reason', 'User declined')
           }

Security Analysis Integration
-----------------------------

Implement domain-specific security analysis:

.. code-block:: python

   def _analyze_for_control_writes(self, code: str) -> bool:
       """Detect control system write operations in code.

       Uses the pattern_detection module which handles both legacy EPICS
       patterns (caput, PV.put) and the unified osprey.runtime API
       (write_channel, write_channels).
       """
       from osprey.services.python_executor.analysis.pattern_detection import (
           detect_control_system_operations
       )
       result = detect_control_system_operations(code)
       return result['has_writes']

   def _assess_safety_level(self, security_analysis: dict) -> str:
       """Assess overall safety level based on detected operations."""
       if security_analysis.get('has_control_writes'):
           return 'high'
       elif security_analysis.get('has_file_operations'):
           return 'medium'
       else:
           return 'low'

Advanced Patterns
=================

Multi-Stage Approval
--------------------

For complex operations requiring multiple approval stages:

.. code-block:: python

   # Plan approval followed by execution approval
   async def multi_stage_approval(self, state: AgentState) -> dict:
       # Stage 1: Plan approval
       plan_interrupt = create_plan_approval_interrupt(
           plan=execution_plan,
           task_description="Data analysis workflow"
       )
       interrupt(plan_interrupt)

       # Stage 2: Code approval (after plan approval)
       code_interrupt = create_code_approval_interrupt(
           code=generated_code,
           analysis_details=analysis,
           execution_mode='readonly',
           safety_concerns=[]
       )
       interrupt(code_interrupt)

Conditional Approval
--------------------

Different approval requirements based on context:

.. code-block:: python

   def get_approval_mode(self, context: ContextManager) -> str:
       """Determine approval mode based on context."""
       user_role = context.get_user_context().get('role', 'user')
       time_of_day = datetime.now().hour

       if user_role == 'operator' and 9 <= time_of_day <= 17:
           return 'reduced_approval'
       else:
           return 'full_approval'

Channel Write Approval
======================

Require human approval before executing control system channel writes.

**Use Case:** Production environments where direct hardware writes require operator approval before execution.

**Pattern:**

.. code-block:: python

   from osprey.approval import create_channel_write_approval_interrupt
   from osprey.approval import get_approval_resume_data
   from langgraph.types import interrupt

   async def execute_channel_write(self, state: AgentState) -> dict:
       """Execute channel writes with approval workflow."""

       # Check if we're resuming from an approval interrupt
       has_approval_resume, approved_payload = get_approval_resume_data(
           state,
           "channel_write"
       )

       if has_approval_resume:
           if not approved_payload:
               raise ChannelWriteAccessError("Write operation cancelled by user")
           # User approved - continue with execution
           logger.info("Resuming approved write operation")
       else:
           # First time execution - check if approval is needed
           approval_evaluator = get_python_execution_evaluator()
           decision = approval_evaluator.evaluate(has_control_writes=True)

           if decision.needs_approval:
               # Create approval interrupt
               interrupt_data = create_channel_write_approval_interrupt(
                   operations=write_operations,  # List of WriteOperation objects
                   analysis_details={
                       'operation_count': len(write_operations),
                       'safety_level': 'high',
                   },
                   safety_concerns=[
                       f"Direct hardware write: {op.channel_address} = {op.value}"
                       for op in write_operations
                   ],
                   step_objective="Set beam current to target value"
               )

               # Pause for approval
               interrupt(interrupt_data)

       # Execute writes (after approval or if no approval needed)
       results = await self._execute_writes(write_operations)
       return {"write_results": results}

**Configuration:**

.. code-block:: yaml

   approval:
     capabilities:
       python_execution:
         mode: "control_writes"  # Triggers approval for control system writes

**Interrupt Data Structure:**

The channel write approval interrupt includes:

- **operations**: List of pending write operations with channel addresses and values
- **analysis_details**: Safety analysis and operation metadata
- **safety_concerns**: Human-readable list of safety concerns
- **step_objective**: Description of what the write operation accomplishes

See the Control Assistant template for a complete implementation example.

Testing and Validation
======================

Test your approval workflows with different security scenarios:

.. code-block:: python

   async def test_approval_workflows():
       """Test approval workflows with different security scenarios."""

       # Test 1: Safe code (no approval required)
       safe_code = "print('Hello, world!')"
       result = await capability.execute(state, context)
       assert result['success'] == True

       # Test 2: EPICS writes (approval required)
       epics_code = "caput('BEAM:CURRENT', 150.0)"
       # This should trigger approval interrupt

       # Test 3: Approval resumption
       # Simulate user approval and test resumption

       # Test 4: Approval rejection
       # Simulate user rejection and test error handling

Troubleshooting
===============

**Common Issues:**

**Issue**: Approval interrupts not pausing execution
   - **Cause**: Missing LangGraph checkpointer configuration
   - **Solution**: Ensure your graph is compiled with a checkpointer

**Issue**: Approval data lost after resumption
   - **Cause**: State not properly preserved across checkpoints
   - **Solution**: Verify approval data is stored in AgentState, not local variables

**Issue**: Multiple approval prompts for same operation
   - **Cause**: Not clearing approval state after processing
   - **Solution**: Call ``clear_approval_state()`` after successful resumption

**Issue**: Approval evaluator not respecting configuration
   - **Cause**: ApprovalManager not properly initialized
   - **Solution**: Verify approval configuration is present in config.yml

**Debugging Approval Workflows:**

.. code-block:: python

   # Enable detailed approval logging
   import logging
   logging.getLogger("osprey.approval").setLevel(logging.DEBUG)

   # Check approval configuration
   from osprey.approval import get_approval_manager
   manager = get_approval_manager()
   config_summary = manager.get_config_summary()
   print(f"Approval config: {config_summary}")

   # Verify approval evaluator behavior
   evaluator = get_python_execution_evaluator()
   decision = evaluator.evaluate(has_control_writes=True, has_control_reads=False)
   print(f"Approval decision: {decision}")

.. seealso::

   :doc:`02_data-source-integration`
       Integrate approval with data source providers

   :doc:`03_python-execution-service/index`
       Advanced Python execution with approval

   :doc:`../../api_reference/03_production_systems/01_human-approval`
       Complete approval system API

   :doc:`../../api_reference/01_core_framework/02_state_and_context`
       AgentState and approval data management