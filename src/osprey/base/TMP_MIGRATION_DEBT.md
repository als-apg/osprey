# Migration Debt Catalogue - base/ + capabilities/

Generated: 2026-02-26

## DEAD (zero live callers — safe to remove)

### /Users/thellert/LBL/ML/osprey/src/osprey/base/nodes.py:1-223 Category: dead-subsystem
**What:** Entire `BaseInfrastructureNode` class. Base class for LangGraph infrastructure nodes. No subclass implementations exist. `src/osprey/infrastructure/` is empty.
**Evidence:** grep returns only self-references. No class inherits from it. Zero callers of any method.
**Action:** Delete entire file `src/osprey/base/nodes.py`

### /Users/thellert/LBL/ML/osprey/src/osprey/base/examples.py:1-226 Category: dead-subsystem
**What:** Entire file: `BaseExample`, `OrchestratorExample`, `ClassifierExample`, `ClassifierActions`, `TaskClassifierGuide`, `OrchestratorGuide`. Powered LangGraph orchestrator/classifier prompt construction.
**Evidence:** Zero imports from outside `base/`. `OrchestratorGuide`/`TaskClassifierGuide` only in docstring type annotations, never instantiated.
**Action:** Delete entire file `src/osprey/base/examples.py`

### /Users/thellert/LBL/ML/osprey/src/osprey/base/planning.py:1-112 Category: dead-subsystem
**What:** `PlannedStep`, `ExecutionPlan` TypedDicts, `save_execution_plan_to_file`, `load_execution_plan_from_file`. LangGraph orchestrator plan format.
**Evidence:** Only referenced within `base/` itself (dead files). Zero external callers or tests.
**Action:** Delete entire file `src/osprey/base/planning.py`

### /Users/thellert/LBL/ML/osprey/src/osprey/base/results.py:1-72 Category: dead-subsystem
**What:** `ExecutionResult`, `ExecutionRecord`, `CapabilityMatch`. LangGraph execution tracking.
**Evidence:** Zero external imports. Different from `mcp_server/python_executor/executor.py:ExecutionResult`.
**Action:** Delete entire file `src/osprey/base/results.py`

### /Users/thellert/LBL/ML/osprey/src/osprey/base/errors.py:142-175 Category: dead-subsystem
**What:** `ReclassificationRequiredError` and `InvalidContextKeyError`. Raised by deleted LangGraph orchestrator.
**Evidence:** Zero imports, zero raises, zero catches anywhere.
**Action:** Delete both exception classes

### /Users/thellert/LBL/ML/osprey/src/osprey/base/errors.py:25-26 Category: dead-field
**What:** `ErrorSeverity.REPLANNING` and `ErrorSeverity.RECLASSIFICATION` enum values.
**Evidence:** Only appear in definition and one docstring example. No capability returns these. No LangGraph replanning loop exists.
**Action:** Delete both enum values

### /Users/thellert/LBL/ML/osprey/src/osprey/base/errors.py:53-82 Category: dead-subsystem
**What:** `ErrorClassification.format_for_llm()` method. Formatted errors for LangGraph replanning prompts.
**Evidence:** Zero callers anywhere.
**Action:** Delete method

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:19-52 Category: dead-subsystem
**What:** Module-level `slash_command()` function. Read slash commands from LangGraph agent state via Gateway.
**Evidence:** Zero external imports. `_capability_slash_commands` key was populated by deleted Gateway.
**Action:** Delete function

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:651-689 Category: dead-subsystem
**What:** `BaseCapability.slash_command()` instance method. Same as above but instance-level.
**Evidence:** Zero callers. Gateway deleted. Also delete `tests/base/test_slash_commands.py`.
**Action:** Delete method + test file

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:751-797 Category: dead-subsystem
**What:** `_create_orchestrator_guide()`, `_create_classifier_guide()`, `orchestrator_guide` property, `classifier_guide` property.
**Evidence:** Zero callers outside `capability.py`. Zero overrides in any capability. Guide types themselves are dead.
**Action:** Delete all four methods/properties

### /Users/thellert/LBL/ML/osprey/src/osprey/base/__init__.py:6-11 Category: orphaned-import
**What:** `ExecutionError` export. Zero imports from outside `base/`.
**Evidence:** Only consumer is dead `results.py`.
**Action:** Remove export + delete `ExecutionError` class from `errors.py`

---

## REFACTOR (alive but needs cleanup)

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:254-258 Category: orphaned-import
**What:** `get_required_contexts()` imports deleted `ContextManager`. Will crash at runtime.
**Risk:** ImportError if any capability calls `self.get_required_contexts()`.
**Suggested fix:** Delete method or update import.

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:549-594 Category: orphaned-import
**What:** `store_output_contexts()` imports deleted `StateManager`. Will crash at runtime.
**Risk:** ImportError if called.
**Suggested fix:** Delete method or reimplement.

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:641 Category: langgraph-reference
**What:** Logger docstring mentions "TUI" as downstream client. TUI is deleted.
**Suggested fix:** Update docstring.

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:86-121 Category: stale-compat
**What:** Docstring and error messages reference `@capability_node` decorator which doesn't exist.
**Suggested fix:** Remove decorator references or recreate simplified version.

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:351-492 Category: stale-compat
**What:** `get_parameters()`, `get_task_objective()`, `get_step_inputs()` depend on `self._step` (never set).
**Suggested fix:** Delete if unused in new architecture.

### /Users/thellert/LBL/ML/osprey/src/osprey/capabilities/*.py (all 4) Category: stale-compat
**What:** All `execute()` methods use LangGraph signature `(state: dict, **kwargs)` returning `{"messages": ...}`.
**Suggested fix:** Simplify to match abstract method signature or remove entirely.

---

## UNCERTAIN (need human decision)

### BaseCapability role question
**What:** Serves two roles: (1) metadata carrier (works), (2) execution framework (broken). Strip to metadata-only?
**Why uncertain:** Architectural decision needed.

### ErrorSeverity/classify_error() relevance
**What:** `classify_error()` implemented by all 4 capabilities but never called by any framework code.
**Why uncertain:** May be called via MCP hooks not visible in grep, or truly dead.

### ErrorClassification
**What:** Returned by `classify_error()`. If that's dead, this is too.
**Why uncertain:** Same dependency.
