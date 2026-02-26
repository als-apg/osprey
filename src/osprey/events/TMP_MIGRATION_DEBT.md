# Migration Debt Catalogue - events/ + approval/

Scan date: 2026-02-26
Scanner: Claude Code migration debt scanner
Branch: feature/claude-code

---

## DEAD (zero live callers -- safe to remove)

### /Users/thellert/LBL/ML/osprey/src/osprey/events/parser.py:1-142 Category: Dead module (LangGraph event consumer)
**What:** Entire `parser.py` module -- `parse_event()`, `is_osprey_event()`, `EVENT_CLASSES` dict. These reconstruct typed events from serialized dicts received via LangGraph `graph.astream(stream_mode="custom")`. The docstrings explicitly reference LangGraph streaming.
**Evidence:** Grep for `parse_event(`, `is_osprey_event(`, `EVENT_CLASSES` across all `.py` files outside `events/` returns zero hits. No tests exist for this module. All references are self-referential (within `events/__init__.py`, `events/parser.py` docstrings).
**Action:** Delete `parser.py`. Remove re-exports from `events/__init__.py` (lines 47-51, 112-114).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/emitter.py:38-73 Category: Dead streaming infrastructure
**What:** `register_fallback_handler()` and `clear_fallback_handlers()` functions, plus the global `_fallback_handlers` list. These were the mechanism for TUI/CLI/Web interfaces to subscribe to the event stream. All those consumers (Textual TUI, prompt_toolkit CLI, Flask web) have been removed.
**Evidence:** Grep for `register_fallback_handler` and `clear_fallback_handlers` across all `.py` files outside `events/` returns zero hits. The only references are docstring examples and re-exports within `events/` itself.
**Action:** Delete `register_fallback_handler()`, `clear_fallback_handlers()`, and `_fallback_handlers`. Remove re-exports from `events/__init__.py` (lines 42-43, 109-110). Once removed, `EventEmitter.emit()` becomes effectively a no-op (emits to nothing), which surfaces the question of whether `EventEmitter` itself should remain (see UNCERTAIN section).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:112-154 Category: Dead event types (LangGraph orchestration)
**What:** `TaskExtractedEvent`, `CapabilitiesSelectedEvent`, `PlanCreatedEvent` -- these three event types modeled LangGraph orchestration phases (task extraction -> capability classification -> plan creation). In the new architecture, Claude Code IS the orchestrator; there is no task extraction or capability classification phase.
**Evidence:** Grep for each class name outside `events/` returns zero construction or usage hits. They appear only in `events/parser.py` (itself dead), `events/types.py` (definitions), and `events/__init__.py` (re-exports).
**Action:** Delete these three dataclasses from `types.py`. Remove from `OspreyEvent` union type. Remove from `__init__.py` exports.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:71-103 Category: Dead event types (LangGraph phase lifecycle)
**What:** `PhaseStartEvent` and `PhaseCompleteEvent` -- modeled phase transitions in the LangGraph execution pipeline (task_extraction, classification, planning, execution, response).
**Evidence:** Outside `events/`, these are only referenced in `utils/logger.py` in docstring examples (lines 29-30, 170, 177-178) -- they are never actually constructed or emitted by any live code path. The `emit_event()` method that would emit them has zero callers (see next entry). No tests exist.
**Action:** Delete these two dataclasses. Remove from `OspreyEvent` union. Remove from `__init__.py`. Clean up docstring examples in `utils/logger.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:162-196 Category: Dead event types (LangGraph capability lifecycle)
**What:** `CapabilityStartEvent` and `CapabilityCompleteEvent` -- modeled capability execution within the LangGraph pipeline.
**Evidence:** Outside `events/`, only referenced in `utils/logger.py` docstrings (line 170, non-constructed). Zero construction instances found. The pattern `CapabilityStartEvent(` returns hits only in `events/` definitions, `events/parser.py` (dead), and `events/__init__.py` docstring examples.
**Action:** Delete these two dataclasses. Remove from `OspreyEvent` union. Remove from `__init__.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:262-290 Category: Dead event types (never constructed)
**What:** `ToolUseEvent` and `ToolResultEvent` -- designed for MCP/Claude Code tool tracking but never actually emitted.
**Evidence:** Grep for `ToolUseEvent(` and `ToolResultEvent(` outside `events/types.py` returns zero construction hits. These were defined speculatively but never wired into any tool invocation path.
**Action:** Delete these two dataclasses. Remove from `OspreyEvent` union. Remove from `__init__.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:294-348 Category: Dead event types (TUI-specific)
**What:** `CodeGeneratedEvent`, `CodeExecutedEvent`, `CodeGenerationStartEvent` -- these were designed specifically for the Textual TUI to manage code generation widgets (creating/finalizing CollapsibleCodeMessage widgets, handling retry attempts). The TUI is gone.
**Evidence:** Grep for each `*Event(` constructor pattern outside `events/` returns zero hits. The docstrings explicitly reference "TUI" (e.g., "Signals TUI to finalize widget", "enable TUI to create separate widgets").
**Action:** Delete these three dataclasses. Remove from `OspreyEvent` union. Remove from `__init__.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:356-382 Category: Dead event types (LangGraph approval flow)
**What:** `ApprovalRequiredEvent` and `ApprovalReceivedEvent` -- modeled approval interrupts within the LangGraph execution graph. In the new architecture, approval is handled by Claude Code's PreToolUse hooks and the `allowedTools` configuration, not by emitting events into a graph.
**Evidence:** Grep for `ApprovalRequiredEvent(` and `ApprovalReceivedEvent(` outside `events/` returns zero hits. These were consumed by TUI/CLI event handlers that no longer exist.
**Action:** Delete these two dataclasses. Remove from `OspreyEvent` union. Remove from `__init__.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:391-428 Category: Dead event types (LangGraph result summary)
**What:** `ResultEvent` -- represented the final execution result of a LangGraph pipeline run, including total cost, duration, and capabilities used. In the new architecture, Claude Code manages its own conversation lifecycle.
**Evidence:** Grep for `ResultEvent(` outside `events/` returns zero hits. Only appears in `events/parser.py` (dead) and `events/__init__.py` (re-export).
**Action:** Delete this dataclass. Remove from `OspreyEvent` union. Remove from `__init__.py`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/utils/logger.py:167-242 Category: Dead logger methods (no callers)
**What:** Three methods on `ComponentLogger`: `emit_event()` (line 167), `emit_llm_request()` (line 189), `emit_llm_response()` (line 213). These were used by LangGraph infrastructure nodes to emit structured events into the stream. None have any callers.
**Evidence:** Grep for `emit_event(`, `emit_llm_request(`, `emit_llm_response(` in all `.py` files outside `utils/logger.py` returns zero hits (both src/ and tests/).
**Action:** Delete these three methods. Remove the lazy imports of `PhaseStartEvent`, `LLMRequestEvent`, `LLMResponseEvent` from logger.py. NOTE: This is in `utils/logger.py`, not in `events/` or `approval/`, but is directly caused by dead event types.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/approval/approval_manager.py:402-441 Category: Dead convenience function
**What:** Module-level `get_memory_evaluator()` convenience function. It wraps `get_approval_manager().get_memory_evaluator()` but has zero callers anywhere in the codebase.
**Evidence:** Grep for `get_memory_evaluator` outside `approval/` returns zero hits. The memory approval evaluator was designed for a memory service that either does not exist or has been migrated to use different approval patterns.
**Action:** Delete `get_memory_evaluator()` function (lines 402-441). Consider also deleting `ApprovalManager.get_memory_evaluator()` and `ApprovalManager.get_memory_config()` if the entire memory approval subsystem is dead (see UNCERTAIN section).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:27-36 Category: Unexported base class
**What:** `BaseEvent` base dataclass -- exported from `__init__.py` but never imported or used by any code outside `events/`. All subclasses that ARE still used (StatusEvent, ErrorEvent) inherit from it, so it stays alive as their parent.
**Evidence:** Grep for `BaseEvent` outside `events/` returns zero hits. External code only uses concrete event subclasses.
**Action:** Keep the class (it is the parent of alive subclasses), but remove from `__all__` in `__init__.py` since it has no external consumers.

---

## REFACTOR (alive but needs cleanup)

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:1-19 Category: Stale docstring (LangGraph references)
**What:** Module docstring references "transport via LangGraph streaming" (line 4) and mentions "router" as a component example (line 18). Both concepts are from the dead architecture.
**Evidence:** Lines 4 and 18 contain direct LangGraph/Router references in docstrings.
**Risk:** None (cosmetic).
**Suggested fix:** Update module docstring to describe the event system as used for structured logging and inter-component signaling. Remove "router" example, replace with live component names.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/emitter.py:1-24 Category: Stale docstring (TUI/CLI/Web references)
**What:** Module docstring references "TUI, CLI, Web" consumers (line 7) and "TUI/UI that needs events" (line 17). The `register_fallback_handler` docstring (line 41) also mentions "TUI, CLI, Web".
**Evidence:** Lines 7, 17, 41 reference dead interfaces.
**Risk:** None (cosmetic).
**Suggested fix:** If handler registration is removed (per DEAD section), update docstring to reflect that EventEmitter is used for structured logging only.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/__init__.py:1-37 Category: Stale module docstring (LangGraph streaming)
**What:** Package docstring describes an architecture with "Interface handlers (TUI, CLI, Web) use pattern matching for clean routing" (line 11), and shows `parse_event` and `CapabilityStartEvent` pattern matching examples (lines 20-29) that reference dead consumers.
**Evidence:** Lines 11, 15-36 describe the dead TUI/CLI/Web consumption pattern.
**Risk:** None (cosmetic).
**Suggested fix:** Rewrite docstring to reflect current usage: structured logging via `ComponentLogger` and approval configuration events.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:205-253 Category: Alive but vestigial fields
**What:** `LLMRequestEvent` and `LLMResponseEvent` contain fields designed for TUI display: `full_prompt` (line 223), `full_response` (line 248), `key` (lines 224, 253). The `key` field was for accumulating multiple prompts/responses in TUI widgets. These events are only constructed in `utils/logger.py` methods `emit_llm_request()` and `emit_llm_response()` which themselves have zero callers (see DEAD section above). If those logger methods are removed, these event types become fully dead.
**Evidence:** Docstrings explicitly say "for TUI display" (lines 215, 241). The `key` field docstring says "for accumulating multiple prompts/responses" which was TUI-specific behavior.
**Risk:** Low. These events are technically alive (they are still defined and importable) but have no live emitters.
**Suggested fix:** If `emit_llm_request`/`emit_llm_response` are removed from logger (per DEAD section), also delete these event types entirely. If kept, remove the TUI-specific fields (`full_prompt`, `full_response`, `key`).

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/emitter.py:76-142 Category: Alive but effectively no-op
**What:** `EventEmitter` class. Currently alive -- it is instantiated by `ComponentLogger` (logger.py:91), `approval_manager.py` (lines 135, 335, 346), `config_models.py` (lines 214, 351, 478), `evaluators.py` (line 237), and `planning.py` (lines 78, 102). However, its `emit()` method dispatches events to `_fallback_handlers` which has zero registered handlers (since `register_fallback_handler` has no callers). Every `emit()` call is effectively a no-op.
**Evidence:** All live callers construct `EventEmitter` and call `.emit()`, but no handler is ever registered to receive those events. The entire emit pipeline is a no-op.
**Risk:** Medium. Removing EventEmitter would require updating all callers (logger, approval, planning) to remove emit calls. The emitter creates no side effects currently, so it is wasted CPU (serialization + handler loop with empty list).
**Suggested fix:** Either (a) wire handlers into the new architecture (e.g., web terminal SSE feed) to make events useful again, or (b) remove all `EventEmitter` usage from approval/ and planning.py since the events go nowhere. For logger.py, the `_emit_stream_event` method (which emits StatusEvent/ErrorEvent) is also effectively a no-op but is the core of the ComponentLogger API, so keep the method but simplify it.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/approval/approval_manager.py:127-170 Category: Alive but noisy no-op emits
**What:** `ApprovalManager.__init__` emits StatusEvent (lines 138-144, 153-159) and ErrorEvent (lines 163-169) via EventEmitter during initialization. Since no handlers are registered, these are no-ops.
**Evidence:** All `emitter.emit(...)` calls in `approval_manager.py`, `config_models.py`, and `evaluators.py` dispatch to zero handlers.
**Risk:** Low. No functional impact but adds unnecessary object construction overhead on every approval manager init.
**Suggested fix:** Replace EventEmitter usage with direct `logger.info()` / `logger.error()` calls. The ComponentLogger already has the emit infrastructure and also provides console output.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/approval/config_models.py:113-116 Category: Deprecated enum value
**What:** `ApprovalMode.EPICS_WRITES` is explicitly marked deprecated (docstring line 104-105) in favor of `CONTROL_WRITES`. A backward-compat mapping exists in `from_dict()` (lines 243-258).
**Evidence:** Line 115 comment says "Deprecated - kept for backward compatibility". Line 247 emits deprecation warning.
**Risk:** Low. The backward compat code works correctly.
**Suggested fix:** Set a deadline for removal. After one release cycle, delete `EPICS_WRITES` enum member and the mapping code in `from_dict()`.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/events/types.py:434-455 Category: Bloated union type
**What:** `OspreyEvent` union type includes 18 event types, most of which are dead (only `StatusEvent` and `ErrorEvent` have live emitters via `ComponentLogger`). The union type is used for type hints in `EventEmitter.emit()` and `ComponentLogger.emit_event()`.
**Evidence:** After removing dead event types per DEAD section, the union should shrink to 2-4 types.
**Risk:** None (type-level only).
**Suggested fix:** After deleting dead event types, update the union to include only the remaining alive types.

---

## UNCERTAIN (need human decision)

### /Users/thellert/LBL/ML/osprey/src/osprey/events/ (entire package) Category: Architecture decision
**What:** The entire events/ package was designed as the streaming event system for LangGraph orchestration. In the new Claude Code architecture, most of its components are dead. The only surviving consumers are:
1. `StatusEvent` -- used heavily via `ComponentLogger._emit_stream_event()` (logger.py:150)
2. `ErrorEvent` -- used via `ComponentLogger._log_to_stdlib_and_stream()` (logger.py:127)
3. `EventEmitter` -- used by ComponentLogger and approval/ (but emits to zero handlers)
**Why uncertain:** The event system could be valuable if wired into the new web terminal (FastAPI + SSE), providing real-time status streaming to the browser. Alternatively, it could be completely replaced with standard Python logging, since `ComponentLogger` already wraps stdlib logging. The decision depends on whether real-time event streaming to the web terminal is a desired feature. If yes, keep and refactor. If no, replace event emission with plain logging and delete the package.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/approval/ (memory subsystem) Category: Feature scope decision
**What:** `MemoryApprovalEvaluator`, `MemoryApprovalConfig`, `ApprovalManager.get_memory_evaluator()`, `ApprovalManager.get_memory_config()`, `get_memory_evaluator()` convenience function. The entire memory approval subsystem has zero external callers.
**Why uncertain:** The memory approval subsystem might be intended for future use with a memory service that hasn't been migrated yet, or it might be dead code from the old architecture. The Python execution approval subsystem IS alive (called from `policy_analyzer.py`), so the approval package itself should stay, but the memory portion needs a human decision on whether a memory service that uses approval is planned.

---

### /Users/thellert/LBL/ML/osprey/src/osprey/utils/logger.py:42-46 Category: Dead constant from old architecture
**What:** `TASK_PREPARATION_STEPS` dict (lines 44-46) -- a "hard-coded step mapping for task preparation phases" explicitly noted as "Moved from deprecated streaming.py module". It maps `"task_extraction"` to step/total_steps/phase metadata. This was used for the old LangGraph phase system.
**Evidence:** Referenced only in `ComponentLogger.__init__` (lines 95-96) to set `self._stream_step_info`, which is used by `_emit_stream_event` to add phase/step metadata to StatusEvents. Since no handlers receive these events, the metadata goes nowhere.
**Why uncertain:** This is in `utils/logger.py`, not in `events/` or `approval/`, so it is slightly out of scan scope. Including it because it is directly coupled to the dead event phase system. If the event handler system is removed, this constant and the `_stream_step_info` field become dead.
