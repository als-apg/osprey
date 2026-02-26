# Migration Debt Catalogue - connectors/

**Scanned:** 10 files in `src/osprey/connectors/`
**Date:** 2026-02-26
**Scanner:** Migration debt scanner (LangGraph -> Claude Code architecture)

## Summary

The connectors package is **ALIVE and central** to the new architecture. It is used
by `MCPRegistry`, `osprey.runtime`, and `RegistryManager`. There are no LangGraph
imports, no dead-module imports, no OpenWebUI references. The package is clean of
the old orchestration architecture.

The debt items found are:
- 6 DEAD items (deprecated backward-compat shims with zero live callers)
- 5 REFACTOR items (stale references, duplicated code, deprecated config paths)
- 0 UNCERTAIN items

---

## DEAD (zero live callers -- safe to remove)

### [control_system/base.py:247-262] Category: deprecated-method-shim

**What:** `ControlSystemConnector.read_pv()` -- deprecated method alias for `read_channel()`.
Emits `DeprecationWarning` and delegates to `read_channel()`.

**Evidence:** Grep for `\.read_pv\(` across all `.py` files returns only:
- The definition itself (base.py:247)
- Docstring examples in `epics_connector.py:50,66` and `mock_connector.py:52` (not actual calls)
- Test names using `read_pv` as part of identifier (`test_read_pv_accepts_any_name`) but
  these tests call `connector.read_channel()`, not `read_pv()`.

Zero live callers in `src/osprey/` outside the definition. Backward compat policy is zero.

**Action:** Delete `read_pv()` method from `ControlSystemConnector`. Update the 3 docstring examples
in `epics_connector.py` and `mock_connector.py` that still reference `read_pv`.

---

### [control_system/base.py:264-286] Category: deprecated-method-shim

**What:** `ControlSystemConnector.write_pv()` -- deprecated method alias for `write_channel()`.

**Evidence:** Grep for `\.write_pv\(` across all `.py` files returns zero results outside
the definition itself. No callers in src/ or tests/.

**Action:** Delete `write_pv()` method.

---

### [control_system/base.py:288-304] Category: deprecated-method-shim

**What:** `ControlSystemConnector.read_multiple_pvs()` -- deprecated method alias
for `read_multiple_channels()`.

**Evidence:** Grep for `\.read_multiple_pvs\(` returns zero results outside the definition.

**Action:** Delete `read_multiple_pvs()` method.

---

### [control_system/base.py:306-320] Category: deprecated-method-shim

**What:** `ControlSystemConnector.validate_pv()` -- deprecated method alias
for `validate_channel()`.

**Evidence:** Grep for `\.validate_pv\(` returns zero results outside the definition
(test `test_validate_pv_always_true` uses `validate_channel()` despite its name).

**Action:** Delete `validate_pv()` method.

---

### [control_system/base.py:327-345] Category: deprecated-class-alias

**What:** `PVMetadata(ChannelMetadata)` -- deprecated subclass alias. Emits
`DeprecationWarning` on construction.

**Evidence:** Grep for `PVMetadata` returns only:
- Definition in `base.py:327`
- Re-export in `control_system/__init__.py:8,21`
- No imports or usages anywhere in `src/osprey/` or `tests/` outside those locations.
- Referenced in migration assist tasks (`v0.9.6.yml`) and `CHANGELOG.md` as documentation only.

**Action:** Delete `PVMetadata` class. Remove from `control_system/__init__.py` exports.

---

### [control_system/base.py:348-366] Category: deprecated-class-alias

**What:** `PVValue(ChannelValue)` -- deprecated subclass alias. Emits
`DeprecationWarning` on construction.

**Evidence:** Grep for `PVValue` returns only:
- Definition in `base.py:348`
- Re-export in `control_system/__init__.py:9,20`
- No imports or usages anywhere in `src/osprey/` or `tests/` outside those locations.
- Referenced in migration assist tasks and `CHANGELOG.md` as documentation only.

**Action:** Delete `PVValue` class. Remove from `control_system/__init__.py` exports.

---

## REFACTOR (alive but needs cleanup)

### [control_system/epics_connector.py:50,66 & mock_connector.py:52] Category: stale-docstring

**What:** Docstring examples still show `await connector.read_pv('BEAM:CURRENT')` instead
of `read_channel()`. These are the class-level docstring examples in `EPICSConnector` and
`MockConnector`.

**Evidence:** `read_pv()` is deprecated with zero callers (see DEAD section above).
The docstrings mislead new developers into using the deprecated API.

**Risk:** Low. Cosmetic only, but confusing.

**Suggested fix:** Replace `read_pv` with `read_channel` in all three docstring examples.

---

### [control_system/mock_connector.py:70,77-111] Category: deprecated-config-path

**What:** `MockConnector.connect()` supports three deprecated config locations:
1. `config.control_system.connector.mock.enable_writes` (line 77-85)
2. `execution_control.epics.writes_enabled` (line 96-104)
3. The local `enable_writes` parameter itself (line 70)

The current canonical path is `control_system.writes_enabled`. The backward compat
policy is zero, so these fallbacks are dead weight.

**Evidence:** Grep for `enable_writes` shows tests still pass the local parameter directly
(e.g., `test_mock_connector.py:19`, `test_auto_verification.py:22`). These tests would
need updating. The `execution_control.epics.writes_enabled` path is also referenced in
`services/python_executor/execution/control.py:274` and `utils/config.py:290` -- so
removing it from `MockConnector` alone would be inconsistent; a coordinated cleanup
is needed.

**Risk:** Medium. Tests depend on the deprecated `enable_writes` parameter.

**Suggested fix:** Phase 1: Update tests to use patched config instead of local
`enable_writes`. Phase 2: Remove all three deprecated paths from `MockConnector.connect()`.
Coordinate with the same cleanup in `control.py` and `config.py`.

---

### [control_system/mock_connector.py:166-214 & epics_connector.py:233-281] Category: code-duplication

**What:** `_get_verification_config()` is duplicated verbatim between `MockConnector` and
`EPICSConnector`. Both contain identical logic: check limits validator, fall back to
global config, fall back to hardcoded defaults.

**Evidence:** Diff of lines 166-214 in `mock_connector.py` vs lines 233-281 in
`epics_connector.py` shows identical logic, identical comments, identical variable names.

**Risk:** Low. Maintenance burden -- a fix in one must be replicated in the other.

**Suggested fix:** Extract `_get_verification_config()` into `ControlSystemConnector`
base class (or a mixin), since both subclasses use the exact same implementation.

---

### [all files] Category: stale-issue-reference

**What:** Every file in the connectors package references "Issue #18 - Control System
Abstraction (Layer 2)" in its module docstring. This issue is long-closed and the
references provide no actionable information.

**Evidence:** 8 occurrences across all 8 non-`__pycache__` files:
- `__init__.py:9`, `factory.py:8`, `control_system/base.py:7`,
  `control_system/epics_connector.py:7`, `control_system/mock_connector.py:7`,
  `archiver/base.py:7`, `archiver/epics_archiver_connector.py:7`,
  `archiver/mock_archiver_connector.py:7`

**Risk:** None. Cosmetic only.

**Suggested fix:** Remove "Related to Issue #18" lines from all module docstrings.

---

### [control_system/__init__.py:8-9,19-21] Category: deprecated-re-exports

**What:** `control_system/__init__.py` re-exports `PVMetadata` and `PVValue` with
comments marking them as "Deprecated alias". Once the classes are deleted (see DEAD
section), these exports must also be removed.

**Evidence:** These exports exist solely to support the deprecated aliases.

**Risk:** None if coordinated with the DEAD removals above.

**Suggested fix:** Remove the `PVMetadata` and `PVValue` lines from `__init__.py`
imports and `__all__` when deleting the classes.

---

## UNCERTAIN (need human decision)

None found.

---

## Architecture Health Summary

| Aspect | Status |
|--------|--------|
| LangGraph imports | None found |
| Dead module imports (graph/, infrastructure/, state/, tui/, cli/, commands/) | None found |
| OpenWebUI references | None found |
| Stale TODOs/FIXMEs | None found |
| Active callers (MCP server, runtime, registry) | Healthy -- ConnectorFactory used by MCPRegistry, runtime, RegistryManager |
| Test coverage | Good -- dedicated test files for each connector and the factory |
