==================================
Migration Guide: Instance Methods
==================================

.. admonition:: 📦 Version Information
   :class: tip

   This guide is for upgrading capabilities from **v0.9.1 and earlier** to **v0.9.2+**.

   - **v0.9.1 and earlier:** Static method pattern with manual state management
   - **v0.9.2+:** Instance method pattern with automatic helper methods

This guide helps you migrate capabilities from the legacy static method pattern (v0.9.1 and earlier) to the new recommended instance method pattern (v0.9.2+).

Backward Compatibility
======================

.. important::

   **The framework maintains full backward compatibility.** Existing static method capabilities continue to work without changes.

**You should migrate when:**

- Writing new capabilities (always use new pattern)
- Major refactoring of existing capabilities
- Experiencing pattern-related confusion
- Wanting to reduce boilerplate

**You can defer migration when:**

- Capability works well and is rarely modified
- Team unfamiliar with new pattern
- Testing/validation resources limited

Side-by-Side Comparison
========================

Minimal Example
---------------

**Legacy (Static Method):**

.. code-block:: python

   @capability_node
   class WeatherCapability(BaseCapability):
       name = "weather"
       provides = ["WEATHER"]

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           task = StateManager.get_current_task(state)
           step = StateManager.get_current_step(state)

           weather = fetch_weather(task)
           context = WeatherContext(data=weather)

           return StateManager.store_context(
               state,
               registry.context_types.WEATHER,
               step.get("context_key"),
               context
           )

**New (Instance Method):**

.. code-block:: python

   @capability_node
   class WeatherCapability(BaseCapability):
       name = "weather"
       provides = ["WEATHER"]

       async def execute(self) -> Dict[str, Any]:
           task = self.get_task_objective()

           weather = fetch_weather(task)
           context = WeatherContext(data=weather)

           return self.store_output_context(context)

**Changes:**

1. Remove ``@staticmethod`` decorator
2. Change ``state: AgentState, **kwargs`` → ``self``
3. Replace ``StateManager.get_current_task(state)`` → ``self.get_task_objective()``
4. Replace ``StateManager.store_context(...)`` → ``self.store_output_context(...)``
5. Remove ``step`` extraction (handled internally)

**Impact:** 55% code reduction, 100% boilerplate elimination

With Dependencies
-----------------

**Legacy (Static Method):**

.. code-block:: python

   @capability_node
   class AnalysisCapability(BaseCapability):
       name = "analysis"
       provides = ["RESULTS"]
       requires = []  # Not used in static pattern

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           step = StateManager.get_current_step(state)
           context_manager = ContextManager(state)

           # Manual extraction
           contexts = context_manager.extract_from_step(
               step, state,
               constraints=[
                   (registry.context_types.INPUT_DATA, "single"),
                   (registry.context_types.TIME_RANGE, "single")
               ],
               constraint_mode="hard"
           )

           input_data = contexts[registry.context_types.INPUT_DATA]
           time_range = contexts[registry.context_types.TIME_RANGE]

           results = analyze(input_data, time_range)
           context = ResultsContext(data=results)

           return StateManager.store_context(
               state,
               registry.context_types.RESULTS,
               step.get("context_key"),
               context
           )

**New (Instance Method):**

.. code-block:: python

   @capability_node
   class AnalysisCapability(BaseCapability):
       name = "analysis"
       provides = ["RESULTS"]
       requires = [
           ("INPUT_DATA", "single"),
           ("TIME_RANGE", "single")
       ]

       async def execute(self) -> Dict[str, Any]:
           # Automatic extraction with tuple unpacking
           input_data, time_range = self.get_required_contexts()

           results = analyze(input_data, time_range)
           context = ResultsContext(data=results)

           return self.store_output_context(context)

**Changes:**

1. Populate ``requires`` field with context dependencies
2. Use cardinality constraints in tuple format
3. Replace manual ``ContextManager.extract_from_step()`` → ``self.get_required_contexts()``
4. Use tuple unpacking for cleaner code
5. All extraction boilerplate eliminated

**Impact:** 60% code reduction

Migration Checklist
===================

For each capability to migrate:

1. **Signature:**

   - [ ] Remove ``@staticmethod`` decorator
   - [ ] Change parameters: ``state: AgentState, **kwargs`` → ``self``
   - [ ] Update return type annotation if needed

2. **Dependencies:**

   - [ ] Move context requirements to ``requires`` field
   - [ ] Add cardinality constraints where appropriate
   - [ ] Remove manual ``ContextManager`` instantiation
   - [ ] Replace ``extract_from_step()`` → ``self.get_required_contexts()``

3. **State Access:**

   - [ ] Replace ``StateManager.get_current_step(state)`` → use helpers or ``self._step``
   - [ ] Replace ``StateManager.get_current_task(state)`` → ``self.get_task_objective()``
   - [ ] Replace ``step.get('parameters')`` → ``self.get_parameters()``

4. **Context Storage:**

   - [ ] Replace ``StateManager.store_context(...)`` → ``self.store_output_context(...)``
   - [ ] Remove ``step.get("context_key")`` references
   - [ ] Remove registry type lookups for storage

5. **Testing:**

   - [ ] Update unit tests to use instance method
   - [ ] Verify context extraction works correctly
   - [ ] Test error cases and edge conditions
   - [ ] Confirm integration tests pass

6. **Documentation:**

   - [ ] Update capability docstring
   - [ ] Update orchestrator guide examples
   - [ ] Update classifier guide examples

Common Migration Issues
=======================

Issue 1: "RuntimeError: requires self._state"
----------------------------------------------

**Symptom:**

.. code-block:: python

   RuntimeError: MyCapability.get_required_contexts() requires self._state
   to be injected by @capability_node decorator.

**Cause:** Calling helper methods outside of ``execute()`` context.

**Solution:** Only call helper methods from within ``execute()``:

.. code-block:: python

   # ❌ Wrong
   class MyCapability(BaseCapability):
       def __init__(self):
           super().__init__()
           self.task = self.get_task_objective()  # ❌ Too early!

   # ✅ Correct
   class MyCapability(BaseCapability):
       async def execute(self) -> Dict[str, Any]:
           task = self.get_task_objective()  # ✅ In execute context

Issue 2: "ValueError: not enough values to unpack"
---------------------------------------------------

**Symptom:**

.. code-block:: python

   ValueError: not enough values to unpack (expected 2, got 1)

**Cause:** Tuple unpacking with ``constraint_mode="soft"`` when not all contexts found.

**Solution:** Use dict access with soft mode:

.. code-block:: python

   # ❌ Wrong with soft mode
   a, b = self.get_required_contexts(constraint_mode="soft")  # May fail

   # ✅ Correct with soft mode
   contexts = self.get_required_contexts(constraint_mode="soft")
   a = contexts.get("CONTEXT_A")
   b = contexts.get("CONTEXT_B")

Issue 3: Cardinality validation errors
---------------------------------------

**Symptom:**

.. code-block:: python

   ValueError: Expected single CONTEXT_TYPE but got list with 3 items

**Cause:** Mismatch between orchestrator plan and cardinality constraint.

**Solution:** Review orchestrator guide and planning logic:

.. code-block:: python

   # If orchestrator sometimes provides lists, don't use "single"
   requires = [("DATA", "single")]  # ❌ Too strict
   requires = ["DATA"]              # ✅ Flexible

Testing After Migration
========================

Unit Testing Pattern
--------------------

.. code-block:: python

   import pytest
   from osprey.state import StateManager

   @pytest.mark.asyncio
   async def test_my_capability():
       # Create capability instance
       capability = MyCapability()

       # Create test state
       state = StateManager.create_fresh_state("Test task")

       # Manually inject state (simulates decorator)
       capability._state = state
       capability._step = {
           'context_key': 'test_key',
           'task_objective': 'Test task',
           'parameters': {'timeout': 30}
       }

       # Execute
       result = await capability.execute()

       # Verify
       assert "capability_context_data" in result
       assert result["capability_context_data"]["MY_CONTEXT"]["test_key"]

Integration Testing
-------------------

Integration tests don't change - the decorator handles injection automatically in real execution.

Gradual Migration Strategy
===========================

**Phase 1:** New capabilities only (2-4 weeks)

- All new capabilities use instance pattern
- Team trains on new pattern
- Templates and examples updated

**Phase 2:** Active development (1-2 months)

- Migrate capabilities under active development
- Update capabilities when bugs fixed
- Defer stable, working capabilities

**Phase 3:** Comprehensive (Optional)

- Migrate remaining capabilities for consistency
- Update all documentation
- Deprecation notices for static pattern (future)

**Timeline:** Expect 2-6 months for complete migration depending on codebase size.

Quick Reference
===============

Helper Methods
--------------

**get_required_contexts()**

.. code-block:: python

   # Dict access
   contexts = self.get_required_contexts()
   data = contexts["INPUT_DATA"]

   # Tuple unpacking (matches requires order)
   data, time_range = self.get_required_contexts()

   # Soft mode (optional contexts)
   contexts = self.get_required_contexts(constraint_mode="soft")

**get_parameters()**

.. code-block:: python

   params = self.get_parameters()
   timeout = params.get('timeout', 30)

**get_task_objective()**

.. code-block:: python

   task = self.get_task_objective()
   # With custom default
   task = self.get_task_objective(default="unknown task")

**store_output_context()**

.. code-block:: python

   context = ResultContext(data=result)
   return self.store_output_context(context)

**store_output_contexts()**

.. code-block:: python

   return self.store_output_contexts(context1, context2, context3)

Next Steps
==========

1. **Review updated tutorials:**

   - :doc:`02_quick-start-patterns/01_building-your-first-capability`
   - :doc:`02_quick-start-patterns/02_state-and-context-essentials`

2. **Study helper method documentation:**

   - :doc:`../api_reference/01_core_framework/01_base_components`

3. **Start with one capability:**

   - Choose a simple, low-risk capability
   - Follow the migration checklist
   - Test thoroughly before proceeding

4. **Share knowledge:**

   - Document team-specific migration patterns
   - Hold training sessions on new pattern
   - Update internal documentation

See Also
========

- :doc:`02_quick-start-patterns/02_state-and-context-essentials` - Context management patterns
- :doc:`03_core-framework-systems/02_context-management-system` - Detailed context system guide
- :doc:`../api_reference/01_core_framework/01_base_components` - Complete API reference

