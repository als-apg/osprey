---
workflow: update-documentation
category: documentation
applies_when: [after_code_change, before_commit, api_change, behavior_change]
estimated_time: 15-30 minutes
ai_ready: true
related: [docstrings, comments, pre-merge-cleanup]
---

# Documentation Update Workflow - Keeping Docs in Sync with Code

Keep documentation synchronized with code changes - docstrings, CHANGELOG, and examples.

This document provides a comprehensive workflow for identifying and updating documentation when code changes occur. It ensures documentation remains professional, accurate, and synchronized with the codebase.

**🎯 DEFAULT SCOPE: This workflow analyzes UNCOMMITTED changes (not yet pushed to git) by default.** This catches documentation needs before code is committed.

## 🤖 AI Quick Start

**For AI assistants (recommended):**

```
@src/osprey/assist/tasks/update-documentation/instructions.md

I have uncommitted changes. Please:
1. Analyze what changed using git diff
2. Determine documentation impact using the decision tree
3. Update only the necessary documentation (apply proportionality)
4. Verify the documentation builds
```

**For specific analyses:**
- `"Check if my changes need documentation updates"`
- `"I modified [specific file], what docs need updating?"`
- `"Guide me through updating docs for a breaking API change"`

## ⚡ Quick Start

**For immediate documentation update analysis:**

```bash
# 1. See what files changed
git status --porcelain
git diff --name-only

# 2. Analyze the changes
git diff

# 3. For each modified public function/class, update:
#    - Docstrings (see DOCSTRINGS.md)
#    - Inline comments (see COMMENTS.md)
#    - API reference documentation
#    - Examples in user guides
#    - CHANGELOG.md

# 4. Verify documentation builds
cd docs && make clean html && make linkcheck

# 5. Check pre-commit checklist (see Section 7 below)
```

**Critical Checkpoints:**
- ✅ Did I update docstrings for all modified public functions?
- ✅ Did I update all examples that use the changed code?
- ✅ Did I update CHANGELOG.md?
- ✅ Did I check for edge cases listed in Section 4?
- ✅ Does documentation build without warnings?

### 🌳 Quick Decision Tree: Do I Need to Update Docs?

```
START: What changed?
│
├─❓ Is it a private function/class (starts with _)?
│  └─ NO DOCS NEEDED (maybe internal comment)
│
├─❓ Is it pure refactoring with identical behavior?
│  └─ NO DOCS NEEDED (maybe CHANGELOG note)
│
├─❓ Does it fix a bug that restores documented behavior?
│  └─ MINIMAL: CHANGELOG only
│
├─❓ Did I add a new parameter with a default value?
│  └─ LOW: Update docstring + CHANGELOG
│
├─❓ Did I change behavior of a public function?
│  └─ MODERATE: Docstring + Examples + CHANGELOG
│
├─❓ Did I add a new public feature/capability?
│  └─ SIGNIFICANT: Full docs + Examples
│
└─❓ Is it a breaking change?
   └─ COMPREHENSIVE: Everything + Migration guide
```

**Rule of Thumb**: If a user wouldn't notice or care about the change, don't create extensive documentation.

### 📋 One-Page Quick Reference

**Print or bookmark this section for daily use:**

| What Changed | Minimum Docs Required | Check These Locations |
|--------------|----------------------|----------------------|
| Private function (`_name`) | None | Maybe internal comment |
| Pure refactoring | None or CHANGELOG note | - |
| Bug fix (restores docs) | CHANGELOG | - |
| New param (with default) | Docstring + CHANGELOG | Examples that use it |
| Behavior change (public) | Docstring + Examples + CHANGELOG | All usage examples |
| New feature/class | Docstring + API ref + CHANGELOG | Getting Started if major |
| Breaking change | Everything + Migration guide | **All** examples + guides |
| Type hint change | Docstring + Examples | Type checking examples |
| Config option added | Docstring + CHANGELOG | `config.yml` examples |
| Import path changed | All import examples + CHANGELOG | README, tutorials, guides |
| Error message changed | Error handling docs | Troubleshooting guides |
| Decorator modified | All decorator examples | Developer guides |
| CLI output changed | CLI docs + Examples | User guides |

**Critical checks before commit:**
- [ ] Docstrings updated? (`grep -r "def modified_function"`)
- [ ] CHANGELOG entry added?
- [ ] Examples still work? (`cd docs && make html`)
- [ ] Links valid? (`make linkcheck`)
- [ ] Cross-references correct?

For complete guidance, read the full document below.

---

## 📑 Table of Contents

1. [Purpose and Philosophy](#purpose-and-philosophy)
2. [Step-by-Step Workflow](#step-by-step-workflow)
   - Step 1: Identify Modified Files
   - Step 2: Analyze Each Changed File
   - Step 3: Map Changes to Documentation Locations
   - Step 4: Update Documentation Systematically
   - Step 5: Quality Assurance
3. [Change Analysis Guidelines](#change-analysis-guidelines)
   - Public vs Private
   - Behavior Changes
   - Signature Changes
   - Error Handling Changes
4. [Edge Cases and Gotchas](#edge-cases-and-gotchas) ⚠️ **CRITICAL SECTION**
5. [Hidden Documentation Locations](#hidden-documentation-locations)
6. [Documentation by File Type](#documentation-by-file-type)
7. [Pre-Commit Checklist](#pre-commit-checklist)
8. [Cross-File Impact Analysis](#cross-file-impact-analysis)
9. [Critical Warning Signs](#critical-warning-signs)
10. [Summary: The Golden Rule](#summary-the-golden-rule)

## 🎯 Purpose and Philosophy

### Why Documentation Synchronization Matters

**Documentation is a contract with users.** Outdated documentation erodes trust, causes integration failures, and wastes developer time. Every code change that affects public interfaces, behavior, or usage patterns requires corresponding documentation updates.

**The principle**: Documentation updates are not optional follow-up work—they are an integral part of the feature implementation. A feature is not complete until its documentation is updated.

### Scope of This Workflow

This workflow focuses on:
- **Uncommitted changes** (by default) - Changes not yet committed to git
- **Recently committed changes** (optional) - Changes in recent commits
- **Modified functions/classes** - Analyzing what changed and why
- **Documentation impact** - Determining what docs need updates
- **Quality assurance** - Ensuring consistency and completeness

### **⚖️ The Proportionality Principle**

**Documentation effort should match the impact of changes.**

| Change Type | Documentation Needed | Effort Level |
|-------------|---------------------|--------------|
| **Internal refactoring (no behavior change)** | None (maybe CHANGELOG note) | 🟢 Minimal |
| **Bug fix (restores documented behavior)** | CHANGELOG only | 🟢 Minimal |
| **Minor parameter addition (with default)** | Docstring + CHANGELOG | 🟡 Low |
| **Behavior change to public function** | Docstring + Examples + CHANGELOG | 🟠 Moderate |
| **New public feature/class** | Full docs + Examples + Guide | 🔴 Significant |
| **Breaking change** | Everything + Migration guide | 🔴🔴 Comprehensive |

**Key Rule**: If users won't notice the change, documentation updates should be minimal.

**When in doubt, ask**: *"Would a user of this function/class need to know about this change?"*
- **No** → CHANGELOG entry only (or skip if truly internal)
- **Yes, but it's backward compatible** → Update docstring and examples that directly use it
- **Yes, and it breaks compatibility** → Full documentation update including migration guide

---

## 📋 Step-by-Step Workflow

### **Step 1: Identify Modified Files**

First, determine what has changed in the codebase.

#### **For Uncommitted Changes (Default)**

```bash
# Check git status for modified files
git status --porcelain

# Get detailed diff of changes
git diff

# For specific file analysis
git diff src/osprey/registry/manager.py
```

#### **For Recent Commits**

```bash
# See changes in last N commits
git log -n 5 --oneline
git diff HEAD~5..HEAD

# Changes since specific version
git diff v0.8.2..HEAD
```

#### **For Staged but Uncommitted Changes**

```bash
# View staged changes
git diff --cached
```

### **Step 2: Analyze Each Changed File**

For each modified file, determine the nature and scope of changes.

#### **Classification of Changes**

**Public API Changes (High Priority for Documentation)**
- New functions or classes
- Modified function signatures (parameters added/removed/reordered)
- Changed parameter types or defaults
- Modified return types or structures
- New exceptions raised
- Changed behavior of public methods
- New public attributes or properties

**Internal Implementation Changes (Lower Priority)**
- Refactoring without behavior changes
- Performance optimizations that don't affect usage
- Bug fixes that restore documented behavior
- Internal helper function changes
- Private method modifications

**Configuration and Setup Changes**
- New configuration options
- Changed default settings
- Modified environment variables
- Dependency updates
- Installation procedure changes

**Breaking Changes (Critical Priority)**
- Removed functions or classes
- Changed function signatures incompatibly
- Modified expected input/output formats
- Changed error handling behavior
- Deprecated features

### **Step 3: Map Changes to Documentation Locations**

For each change, identify all documentation locations that need updates.

#### **Documentation Hierarchy**

```
Changes can affect:
├── Source Code Docstrings
│   ├── Module docstrings (__init__.py files)
│   ├── Class docstrings
│   ├── Function/method docstrings
│   └── Inline comments (see COMMENTS.md)
├── API Reference Documentation (docs/source/api_reference/*.rst)
│   ├── Auto-generated from docstrings
│   ├── Manual overrides and examples
│   └── Cross-references between modules
├── User Guides (docs/source/getting-started/, developer-guides/)
│   ├── Tutorials and walkthroughs
│   ├── Concept explanations
│   └── Usage examples
├── Example Applications (docs/source/example-applications/)
│   └── Complete working examples
├── Release Documentation
│   ├── CHANGELOG.md
│   ├── RELEASE_NOTES.md
│   └── Migration guides
└── Project Documentation
    ├── README.md
    └── Configuration examples
```

#### **Mapping Rules** (Apply Proportionally)

| Change Type | **Minimum Required** | Optional/If Widely Used |
|-------------|---------------------|------------------------|
| **New public function/class** | Docstring + CHANGELOG | API reference + Usage example if significant |
| **Modified function signature (backward compatible)** | Docstring + CHANGELOG | Examples if commonly referenced |
| **Modified function signature (breaking)** | Docstring + All examples using it + CHANGELOG + RELEASE_NOTES | Migration guide if complex |
| **New major feature/capability** | Docstring + API reference + Getting Started + CHANGELOG | Developer Guide + Example app + README |
| **Bug fix (behavior change)** | Docstring update if needed + CHANGELOG | Examples if they relied on old behavior |
| **Bug fix (restoring documented behavior)** | CHANGELOG only | Nothing else needed |
| **New configuration option** | Config docstring + CHANGELOG | Getting Started if commonly needed |
| **Deprecation** | Deprecation warning in docstring + CHANGELOG | RELEASE_NOTES + Migration guide |
| **Performance improvement** | CHANGELOG only | Docstring note if usage patterns change |
| **Internal refactoring only** | Nothing | CHANGELOG if it fixes issues |

**Remember**: Not every change needs comprehensive documentation. Focus on user impact.

### **Step 4: Update Documentation Systematically**

Follow this order to ensure completeness:

#### **Phase 1: Source Code Documentation**
- Update docstrings (follow DOCSTRINGS.md): signatures, parameters, returns, exceptions, examples
- Update inline comments (follow COMMENTS.md): add for new logic, remove outdated

#### **Phase 2: API Reference**
- Rebuild API docs: `cd docs && make clean html`
- Check `docs/source/api_reference/*.rst` for manual updates needed
- Update examples and cross-references

#### **Phase 3: User-Facing Guides**
- Update Getting Started guides if installation/configuration/quick start changed
- Update Developer guides for new patterns or workflows
- Update Example applications to use new APIs correctly

#### **Phase 4: Release Documentation**
- Add CHANGELOG.md entry (Added/Changed/Fixed/Deprecated/Removed)
- Update RELEASE_NOTES.md for significant changes
- Update README.md for major features

### **Step 5: Quality Assurance**

**Essential checks:**
```bash
cd docs && make linkcheck  # Check for broken references
grep -r "old_function_name" docs/source/  # Find outdated names
make html  # Verify build without warnings
```

**Verify:**
- [ ] Function/class names consistent across all docs
- [ ] All `:func:`, `:class:`, `:mod:` references valid
- [ ] Examples tested and working
- [ ] Import statements consistent

---

## 🔍 Change Analysis Guidelines

### **Determining Documentation Impact**

For each modified function/class, ask these questions:

#### **Public vs. Private**

```python
# Public - REQUIRES documentation
def process_data(input: str) -> dict:
    """Public API function."""

class DataProcessor:
    """Public class."""

# Private - usually NO documentation update needed
def _internal_helper(data):
    """Internal implementation detail."""

class _PrivateProcessor:
    """Internal use only."""
```

**Rule**: If it's in `__all__`, doesn't start with `_`, or is documented in API reference, it's public and needs documentation.

#### **Behavior Changes**

**Question**: "Would existing code using this function behave differently?"

```python
# BEFORE
def calculate_score(data):
    """Calculate score (0-100)."""
    return sum(data) / len(data)

# AFTER - Behavior changed!
def calculate_score(data):
    """Calculate score (0-100)."""
    return min(100, sum(data) / len(data))  # Now capped at 100
```

**Action**: Update docstring to document capping behavior. Update CHANGELOG. Check all examples.

#### **Signature Changes**

**Question**: "Would existing function calls still work?"

```python
# BEFORE
def register_provider(provider_class, name):
    """Register a provider."""

# AFTER - Breaking change!
def register_provider(provider_class, name, enable_caching=False):
    """Register a provider with optional caching."""
```

**Action**:
- Update docstring with new parameter
- Update all examples using this function
- Add to CHANGELOG under "Changed"
- If breaking: Add migration guide

#### **Error Handling Changes**

**Question**: "Are different exceptions raised now?"

```python
# BEFORE
def load_config(path):
    """Load configuration."""
    # Raises: FileNotFoundError

# AFTER - New exception type!
def load_config(path):
    """Load configuration."""
    # Raises: ConfigurationError (wraps FileNotFoundError)
```

**Action**: Update docstring `:raises:` section. Update error handling examples.

---

## ⚠️ Edge Cases and Gotchas

> **🎯 Quick Scan**: This section covers 16 commonly overlooked scenarios. Most critical: **Type hint changes**, **Default value changes**, **Async/await conversions**, **State schema changes**, and **`__all__` exports changes**. Skim the headers and read sections relevant to your changes.

### **Type Hint Changes**

```python
# Type hints are part of the API contract
# BEFORE
def process(data: str) -> dict:

# AFTER - Breaking change for type checkers!
def process(data: str) -> Optional[dict]:  # Can now return None!
```

**Action**: Document new return behavior. Update examples showing None handling.

### **Default Value Changes**

```python
# BEFORE
def configure(timeout=30):
    """Configure with timeout in seconds."""

# AFTER - Behavior change!
def configure(timeout=60):  # Different default!
    """Configure with timeout in seconds."""
```

**Action**: Document new default. Add to CHANGELOG. Check if any examples relied on old default.

### **Dependency on Other Changed Functions**

If function A calls function B, and B changed behavior:

```python
# If internal_process() changed behavior
def public_api():
    """Public API function."""
    result = internal_process()  # This might behave differently now
    return result
```

**Action**: Check if public_api's documented behavior is still accurate.

### **Module Reorganization**

```python
# BEFORE
from osprey.registry.manager import RegistryManager

# AFTER - Import path changed!
from osprey.registry import RegistryManager  # Moved to __init__.py
```

**Action**: Update all import examples. Add backward compatibility note if maintained.

### **Decorator Changes**

```python
# BEFORE
@server.tool()
async def read_channel(name: str) -> float:
    return await connector.read(name)

# AFTER - New parameter!
@server.tool()
async def read_channel(name: str, timeout: float = 5.0) -> float:
    return await connector.read(name, timeout=timeout)
```

**Action**:
- Update all examples showing tool usage
- Document new parameter in tool documentation
- Add to CHANGELOG
- Check if existing code breaks or needs migration

### **Class Attribute Changes**

```python
# BEFORE
class MockConnector(BaseConnector):
    name = "mock"

# AFTER - New required attribute!
class MockConnector(BaseConnector):
    name = "mock"
    version = "1.0.0"  # Now required
```

**Action**:
- Update class docstring
- Update all examples
- Update developer guides showing class structure
- Add migration guide if breaking

### **Async/Await Pattern Changes**

```python
# BEFORE - Synchronous
def execute(state):
    return process(state)

# AFTER - Now async!
async def execute(state):
    return await process(state)
```

**Action**:
- Update function docstring with async behavior
- Update ALL examples to use `await`
- Update developer guides about async patterns
- Add to CHANGELOG as breaking change
- Create migration guide

### **State Schema Changes**

```python
# BEFORE
state = {
    "messages": [...],
    "context": {...}
}

# AFTER - Nested structure!
state = {
    "messages": [...],
    "execution_context": {
        "context": {...},
        "metadata": {...}  # New field
    }
}
```

**Action**:
- Update state management documentation thoroughly
- Update ALL examples accessing state
- Update developer guides
- Breaking change - migration guide essential

### **__all__ Exports Changes**

```python
# BEFORE
__all__ = ["RegistryManager", "register_provider"]

# AFTER - Removed public API!
__all__ = ["RegistryManager"]  # register_provider removed
```

**Action**:
- Critical breaking change
- Update all import examples
- Remove from API reference
- Create migration guide
- Add to CHANGELOG under "Removed"

### **File Path or Directory Structure Changes**

```python
# BEFORE
from osprey.utils.helpers import format_data

# AFTER - File moved!
from osprey.core.formatting import format_data
```

**Action**:
- Update all import examples throughout docs
- Update API reference structure
- Add backward compatibility imports if possible
- Document in migration guide

### **Initialization Sequence Changes**

```python
# BEFORE
manager = RegistryManager()
manager.discover_providers()

# AFTER - Auto-initialization!
manager = RegistryManager()  # Now auto-discovers
```

**Action**:
- Update getting started tutorials
- Update all initialization examples
- Document new behavior in class docstring
- Note in CHANGELOG under "Changed"

---

## 🔍 Hidden Documentation Locations

**These locations are often overlooked but must be checked:**

### **In-Code Documentation**
- **CLI help text** - `@click.option()` descriptions, command help strings
- **Error messages** - User-visible exception messages
- **Log messages** - Especially INFO and WARNING level logs users might see
- **Validation messages** - Pydantic model validation errors
- **Type hints** - Part of the public API contract

### **Configuration Files**
- **env.example** - Environment variable descriptions and examples
- **config.yml examples** - Comments explaining configuration options
- **Template files (.j2)** - Comments in Jinja2 templates
- **pyproject.toml** - Project metadata, dependencies, entry points

### **Example Projects**
- **weather-agent/** - Complete example that must stay working
- **services/** - Docker compose and service configurations
- **Example notebooks** - If any Jupyter notebooks exist

### **Meta Documentation**
- **README.md** - Installation, quick start, features list
- **CONTRIBUTING.md** - Development setup instructions
- **Migration guides** - Version upgrade instructions
- **Troubleshooting guides** - Common error solutions

### **Auto-Generated Documentation**
- **API reference RST files** - May have manual additions
- **Type stub files (.pyi)** - If they exist
- **OpenAPI/Schema files** - API specifications

---

## 📚 Documentation by File Type

### **Registry Changes** (`src/osprey/registry/`)

**Check These Docs:**
- `docs/source/api_reference/registry.rst`
- `docs/source/developer-guides/registry-system.rst`
- `docs/source/developer-guides/provider-registration.rst`
- Examples in `docs/source/getting-started/`

**Common Updates:**
- Provider registration patterns
- Discovery mechanisms
- Configuration options
- Error handling

### **Capability Changes** (`src/osprey/capabilities/`)

**Check These Docs:**
- `docs/source/api_reference/capabilities.rst`
- `docs/source/developer-guides/creating-capabilities.rst`
- `docs/source/example-applications/`
- Getting started tutorials

**Common Updates:**
- Capability interfaces
- State management patterns
- Execution workflows
- Integration examples

### **State Management** (`src/osprey/state/`)

**Check These Docs:**
- `docs/source/api_reference/state.rst`
- `docs/source/developer-guides/state-management.rst`
- All tutorials that show state usage

**Common Updates:**
- State structure
- State update patterns
- Field descriptions
- Serialization behavior

### **CLI Changes** (`src/osprey/cli/`)

**Check These Docs:**
- `docs/source/getting-started/command-line-interface.rst`
- `docs/source/developer-guides/cli-commands.rst`
- README.md (CLI examples)
- Installation guides

**Common Updates:**
- Command syntax
- Options and flags
- Configuration file formats
- Example commands

### **Configuration Changes** (`config.yml` related)

**Check These Docs:**
- `docs/source/getting-started/configuration.rst`
- All example `config.yml` files
- `env.example` file
- Configuration schema docs

**Common Updates:**
- New configuration keys
- Changed default values
- Deprecated options
- Configuration validation rules

---

## ✅ Pre-Commit Checklist

Before committing code changes, verify documentation is complete:

### **Code-Level Documentation**
- [ ] All modified public functions have updated docstrings
- [ ] All new functions have complete docstrings (see DOCSTRINGS.md)
- [ ] Docstrings follow Sphinx format correctly
- [ ] Examples in docstrings are correct and runnable
- [ ] Inline comments follow COMMENTS.md guidelines
- [ ] No historical/migration comments (see COMMENTS.md anti-patterns)

### **API Reference**
- [ ] API reference RST files reviewed for affected modules
- [ ] Manual API documentation examples updated
- [ ] Cross-references checked and updated
- [ ] Sphinx build completes without warnings: `make html`

### **User Guides**
- [ ] Getting Started guides updated if workflow changed
- [ ] Developer guides updated for new patterns
- [ ] Example applications still work with changes
- [ ] Tutorials tested and verified

### **Release Documentation**
- [ ] CHANGELOG.md updated with all changes
- [ ] Changes categorized correctly (Added/Changed/Fixed/Deprecated/Removed)
- [ ] Breaking changes clearly marked
- [ ] RELEASE_NOTES.md updated if significant change

### **Quality Assurance**
- [ ] No references to old function/class names
- [ ] All code examples use current API
- [ ] Links checked: `make linkcheck`
- [ ] Terminology consistent across all docs
- [ ] Version references appropriate

### **Build Verification**
```bash
# Run these commands before committing
cd docs
make clean
make html
make linkcheck

# Check for common issues
grep -r "TODO" docs/source/
grep -r "FIXME" docs/source/
grep -r "XXX" docs/source/
```

---

## 🔗 Cross-File Impact Analysis

**Changes in one file can cascade to affect multiple other files and their documentation.**

### **Dependency Mapping**

When you change a file, check what depends on it:

```bash
# Find all files that import from the changed module
grep -r "from osprey.registry.manager import" src/

# Find all references to a specific class
grep -r "RegistryManager" src/ docs/

# Find all uses of a changed function
grep -r "register_provider(" src/ docs/
```

### **Common Cascade Patterns**

**Base Class Changes → All Subclasses**
```python
# If you change BaseConnector or BaseProvider interface
# → All connector/provider implementations must be checked
# → All connector/provider documentation must be reviewed
```

**Configuration Schema Changes → MCP Servers**
```python
# If config.yml schema changes
# → Every MCP server reading config is affected
# → Every example showing config is affected
# → Configuration docs need comprehensive updates
```

**Registry Changes → Discovery Patterns**
```python
# If provider discovery logic changes
# → All provider implementations may be affected
# → Getting started guides need review
# → Example applications need testing
```

**Configuration Schema Changes → All Configs**
```python
# If config.yml schema changes
# → All example config files need updates
# → Configuration documentation needs updates
# → env.example may need updates
# → Default value documentation needs updates
```

### **Checklist for Cascading Changes**

- [ ] **Identify all subclasses** of modified base classes
- [ ] **Find all imports** of modified functions/classes
- [ ] **Check inheritance chains** for affected methods
- [ ] **Verify interface contracts** haven't changed unexpectedly
- [ ] **Test example applications** to ensure they still work
- [ ] **Review integration points** with external systems
- [ ] **Check factory functions** that create modified objects
- [ ] **Verify serialization** if data structures changed

---

## 🚨 Critical Warning Signs

Watch for these situations that often cause documentation issues:

### **Warning Sign 1: Import Changes**

If imports change anywhere, search all documentation:

```bash
grep -r "from osprey.old_module import" docs/
```

### **Warning Sign 2: Configuration Schema Changes**

If config structure changes, check:
- All example config files
- Configuration documentation
- Environment variable documentation
- Default value documentation
- Validation error messages

### **Warning Sign 3: Error Message Changes**

If error messages or exceptions change:
- Update error handling examples
- Update troubleshooting guides
- Check logged examples match new messages
- Update integration tests checking error messages

### **Warning Sign 4: Dependency Version Updates**

If `pyproject.toml` dependencies change:
- Update installation documentation
- Update requirements in README
- Note any new system requirements
- Check for API changes in updated dependencies
- Update Docker base images if needed

### **Warning Sign 5: State Structure Changes**

If MCP tool schemas or response formats change:
- **CRITICAL**: This affects Claude Code integration
- Review ALL tutorials and examples
- Update MCP server documentation completely
- Check every tool that returns structured data
- Verify Claude Code can still parse responses

### **Warning Sign 6: Base Class Modifications**

If BaseConnector, BaseProvider, or other base classes change:
- Check ALL implementations
- Update developer guides for creating new implementations
- Verify backward compatibility
- Test all existing capabilities/providers

---

## 📖 Summary: The Golden Rule

> **💎 "If you changed code that users interact with, you must update documentation that describes that interaction."**
>
> **Corollary**: If users won't notice the change, keep documentation updates minimal or skip them entirely.

### **The Complete Documentation Update Process**

When in doubt, follow this systematic approach:

1. **Identify Changes**
   - Run `git diff` to see exactly what changed
   - Use the automated detection script for comprehensive analysis
   - List all modified files and their purposes

2. **Analyze Impact**
   - Classify each change (Public API, Internal, Breaking, etc.)
   - Identify all edge cases from this document
   - Map changes to documentation locations
   - Check for cascading effects on dependent code

3. **Update Systematically**
   - **Phase 1**: Source code (docstrings + comments)
   - **Phase 2**: API reference documentation
   - **Phase 3**: User guides and tutorials
   - **Phase 4**: Release documentation (CHANGELOG, etc.)

4. **Check Hidden Locations**
   - CLI help text
   - Error messages
   - Log messages
   - Configuration examples
   - Template files

5. **Verify Quality**
   - Build documentation: `cd docs && make clean html`
   - Check links: `make linkcheck`
   - Test code examples
   - Verify cross-references
   - Check for orphaned references to old names

6. **Final Validation**
   - Run pre-commit checklist
   - Review with fresh eyes
   - Consider having another developer review
   - Test example applications

### **Common Mistakes to Avoid**

- ❌ **Updating code but not examples** - Examples break, users get confused
- ❌ **Forgetting CHANGELOG** - Users don't know what changed
- ❌ **Not searching for all references** - Inconsistent documentation
- ❌ **Skipping cross-references** - Links break, navigation fails
- ❌ **Not testing examples** - Examples that don't work are worse than none
- ❌ **Not checking cascading effects** - Base class changes affect all subclasses

### **Final Sanity Check Before Committing**

Before you commit, ask yourself:

1. **User Impact**: *"Would users notice this change?"* → If no, minimal/no docs needed
2. **Breaking Change**: *"Does this break existing code?"* → If yes, migration guide required
3. **Proportionality**: *"Am I over-documenting?"* → Match documentation effort to change impact
4. **Consistency**: *"Is this documented like similar changes?"* → Follow existing patterns

**Remember**: Every hour on documentation saves ten hours of support and debugging. Good enough is good enough—focus on major features and breaking changes.

---

## 🔗 Quick Navigation

- **[↑ Back to Quick Start](#quick-start)** - Fast decision tree
- **[↑ One-Page Reference](#one-page-quick-reference)** - Bookmark this
- **[↑ Pre-Commit Checklist](#pre-commit-checklist)** - Before you commit
- **[↑ Edge Cases](#edge-cases-and-gotchas)** - Common gotchas
- **[Related: Docstrings Guide](docstrings.md)** - Writing docstrings
- **[Related: Comments Guide](comments.md)** - Inline comments
- **[Related: Pre-Merge Cleanup](pre-merge-cleanup.md)** - Final verification

---

**This workflow document itself follows these principles** - it provides comprehensive, structured guidance while emphasizing proportionality and avoiding documentation busywork.

**Document Version**: 2.0 (December 2024)
**Maintained By**: Osprey Core Team
