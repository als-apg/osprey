# Migration Debt Catalogue - context/ + runtime/

Generated: 2026-02-26

## DEAD (zero live callers — safe to remove)

None found. All four files have live callers in production code.

---

## REFACTOR (alive but needs cleanup)

### /Users/thellert/LBL/ML/osprey/src/osprey/context/base.py:79-104 Category: stale-compat

**What:** `get_summary()` contains a 25-line backwards-compatibility shim that detects subclasses implementing the deprecated `get_human_summary()` method and proxies calls to it with a deprecation warning.

**Evidence:** Grepping the entire codebase for `def get_human_summary` returns zero results. No subclass anywhere implements it. The migration is fully complete.

**Risk:** None. No code calls the old name.

**Suggested fix:** Delete the `if "get_human_summary" in self.__class__.__dict__:` branch entirely, keeping only the `raise NotImplementedError`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/context/loader.py:1-6 Category: langgraph-reference

**What:** Module docstring says "Used by the generated code wrapper to provide context access in subprocess execution." Class docstring says "Replaces the deleted ContextManager for simple read-only context access." Both reference the old architecture.

**Evidence:** Module is alive (called by python_executor service). Only docstrings are stale.

**Risk:** None — cosmetic.

**Suggested fix:** Update docstrings to remove old architecture framing.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/runtime/__init__.py:72-73 Category: langgraph-reference

**What:** `configure_from_context` docstring says `context: ContextManager instance from load_context()`. ContextManager class has been deleted. Actual argument is `_DictNamespace`.

**Evidence:** `context_manager.py` does not exist. Function works correctly; only docstring is wrong.

**Risk:** None — cosmetic.

**Suggested fix:** Fix docstring type reference.

---

## UNCERTAIN (need human decision)

### /Users/thellert/LBL/ML/osprey/src/osprey/base/capability.py:254,470 Category: orphaned-import

**What:** `BaseCapability` contains two imports of the deleted `ContextManager`:
- Line 254: `from osprey.context.context_manager import ContextManager`
- Line 470: `from osprey.context import ContextManager`

These are inside lazy-loaded methods that WILL crash at runtime if called.

**Why uncertain:** These methods are part of the old capability execution pipeline. If `BaseCapability.execute()` is dead, these methods should be removed. If not, `ContextManager` needs restoring.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/context/base.py (CapabilityContext class) Category: dead-field

**What:** `CapabilityContext` was designed for the old LangGraph orchestration flow. Its serialization machinery (model_dump, model_validate, get_access_details, get_summary) exists primarily to serve the old orchestration pipeline. The MCP tool wrappers call connectors directly without going through capabilities or context objects.

**Why uncertain:** HIGH risk if removed — 4 native capabilities, template system, and registry all reference it. Need human decision on whether the capability pipeline is considered dead.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/context/base.py:44-46 Category: dead-field

**What:** `CONTEXT_TYPE` and `CONTEXT_CATEGORY` ClassVar constants support old capability-to-capability data flow.

**Why uncertain:** May still be part of public API for user template-generated code.
