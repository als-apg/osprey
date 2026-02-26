# Migration Debt Catalogue - services/ariel_search/

Scanned 44 Python files across `database/`, `search/`, `ingestion/`, `enhancement/`, `agent/`, `prompts/`.

No imports from dead modules (`osprey.graph`, `osprey.infrastructure`, `osprey.state`, `osprey.interfaces.tui`, `osprey.interfaces.cli`, `osprey.commands`). No OpenWebUI mentions. No Flask references. No prompt_toolkit references. No TODOs/FIXMEs/HACKs.

---

## DEAD

### 1. `ARIELSearchRequest.capability_context_data` (models.py:228-237)

**Field**: `capability_context_data: dict[str, Any]` on `ARIELSearchRequest`
**Docstring**: "Context from main graph state" -- the "main graph" is LangGraph's `StateGraph`, which no longer exists.

This field is:
- Defined on the request dataclass (line 237)
- Never read by any code in the entire codebase (zero attribute accesses of `.capability_context_data` on any `ARIELSearchRequest` instance)
- Never set by any production caller (CLI, MCP tools, web routes -- none pass it)
- Only referenced in one unit test (`tests/services/ariel_search/test_models.py:57-64`) that asserts the field exists

**Verdict**: DEAD. Pure LangGraph graph-state artifact. The field and its docstring reference "main graph state" which was `AgentState` in the old LangGraph orchestration. Remove the field and update the test.

---

## REFACTOR

### 1. Stale LangGraph/LangChain references in docstrings and comments (5 locations)

These are documentation-only issues -- the code itself is clean, but the comments reference the old architecture:

| File | Line | Text | Issue |
|------|------|------|-------|
| `agent/__init__.py` | 4 | "LangGraph. This is a transitional implementation that preserves the AgentExecutor / AgentResult interface." | Calls itself "transitional" from LangGraph. It IS the implementation now -- not transitional. |
| `agent/executor.py` | 4 | "no LangGraph or LangChain required" | Defensive "we don't use LangGraph" note; stale after migration is complete. |
| `agent/executor.py` | 7 | "same as the previous LangGraph-based agent" | References the old agent that no longer exists. |
| `search/base.py` | 5-6 | "to build LangChain tools automatically" | References LangChain tool wrapping; the agent now uses OpenAI-format tool definitions directly. |
| `search/base.py` | 88 | "Tool name for LangChain (e.g. ...)" | Attribute docstring references LangChain. |
| `search/sql_query.py` | 8 | "that's for the LangChain agent executor" | References LangChain agent executor. |

**Verdict**: REFACTOR. Update all 6 comment/docstring sites to remove LangGraph/LangChain references. The code functions correctly; only the prose is outdated.

### 2. `ARIELSearchRequest.capability_context_data` docstring (models.py:228)

The docstring says "Context from main graph state". Even if the field is kept as a passthrough for future use, the description must not reference a graph that no longer exists.

**Verdict**: If the field is kept rather than removed, the docstring must be updated.

---

## UNCERTAIN

None found.

---

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| DEAD     | 1 field | Low -- unused field, no runtime impact |
| REFACTOR | 6 comment sites | Low -- documentation-only, no code changes needed |
| UNCERTAIN | 0 | -- |

The `ariel_search` service is **clean**. It was already migrated away from LangGraph in the `agent/executor.py` rewrite (self-contained ReAct loop using `osprey.models.completion`). All 44 files are alive with callers from CLI (`cli/ariel.py`), MCP tools (`interfaces/ariel/mcp/`), REST routes (`interfaces/ariel/`), the registry system, and extensive test coverage. No code imports from any dead module. The only migration debt is one vestigial field and a handful of stale comments.
