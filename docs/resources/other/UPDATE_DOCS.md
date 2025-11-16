# Documentation Update Workflow - Keeping Docs in Sync with Code

This document provides a comprehensive workflow for identifying and updating documentation when code changes occur. It ensures documentation remains 100% professional, accurate, and synchronized with the codebase.

**🎯 DEFAULT SCOPE: This workflow analyzes UNCOMMITTED changes (not yet pushed to git) by default.** This catches documentation needs before code is committed.

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

# 5. Check pre-commit checklist (Section 7 below)
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

For complete guidance, read the full document below.

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
8. [Automated Detection Script](#automated-detection-script)
9. [Common Scenarios and Solutions](#common-scenarios-and-solutions)
10. [Cross-File Impact Analysis](#cross-file-impact-analysis)
11. [Critical Warning Signs](#critical-warning-signs)
12. [Summary: The Golden Rule](#summary-the-golden-rule)

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

1. **Update Docstrings First**
   - Follow DOCSTRINGS.md guidelines
   - Update function signatures in docstrings
   - Revise parameter descriptions
   - Update return value documentation
   - Add or modify exception documentation
   - Update examples in docstrings

2. **Update Inline Comments**
   - Follow COMMENTS.md guidelines
   - Add comments for complex new logic
   - Remove outdated comments
   - Update comments that reference changed behavior

#### **Phase 2: API Reference**

3. **Check Auto-Generated API Docs**
   ```bash
   # Rebuild API docs to see changes
   cd docs
   make clean html
   
   # Check for warnings about missing documentation
   make html 2>&1 | grep -i warning
   ```

4. **Update Manual API Documentation**
   - Check `docs/source/api_reference/*.rst` files
   - Update any manual overrides
   - Add examples for new functionality
   - Update cross-references

#### **Phase 3: User-Facing Guides**

5. **Update Getting Started Guides**
   - Installation changes → `installation.rst`
   - Configuration changes → `configuration.rst`
   - Quick start changes → `hello-world-tutorial.rst`

6. **Update Developer Guides**
   - New patterns → Relevant pattern guide
   - Changed workflows → Workflow documentation
   - Architecture changes → Architecture guides

7. **Update Example Applications**
   - Ensure examples use new APIs correctly
   - Update example code if needed
   - Verify examples still run

#### **Phase 4: Release Documentation**

8. **Update CHANGELOG.md**
   ```markdown
   ## [Unreleased]
   
   ### Added
   - New capability: `DataAnalysisCapability` for automated data analysis
   - New parameter `enable_caching` in `RegistryManager.register_provider()`
   
   ### Changed
   - `process_message()` now returns structured response dict instead of string
   - Configuration format updated to support nested capability settings
   
   ### Fixed
   - Fixed race condition in concurrent capability execution
   
   ### Deprecated
   - `old_process_function()` - Use `new_process_function()` instead
   
   ### Removed
   - Support for deprecated `legacy_config_format`
   ```

9. **Update RELEASE_NOTES.md** (for significant changes)
   - Add user-facing descriptions
   - Include migration instructions
   - Highlight breaking changes

10. **Update README.md** (for major features)
    - Update feature list
    - Add new examples
    - Update badges if needed

### **Step 5: Quality Assurance**

Perform these checks before considering documentation complete.

#### **Consistency Checks**

```bash
# Check for broken references in RST files
cd docs
make linkcheck

# Search for old function names that should be updated
grep -r "old_function_name" docs/source/

# Check for version-specific references that need updating
grep -r "v0.8" docs/source/ | grep -i "new in"

# Verify all code examples use consistent imports
grep -r "^from osprey" docs/source/ | sort | uniq
```

#### **Terminology Consistency**

- **Function name references**: Ensure all docs use the current function name
- **Parameter names**: Check all references use correct parameter names
- **Class names**: Verify class name consistency across all docs
- **Concept terminology**: Ensure consistent use of framework terms (e.g., "capability" vs "node")

#### **Example Validation**

```python
# Extract and test code examples
cd docs
python -c "
import doctest
import sys
# Test docstring examples
result = doctest.testmod(sys.modules['osprey.registry.manager'])
print(f'Examples tested: {result.attempted}, Failed: {result.failed}')
"
```

#### **Cross-Reference Verification**

- [ ] All `:func:` references point to existing functions
- [ ] All `:class:` references point to existing classes
- [ ] All `:mod:` references point to existing modules
- [ ] All `.. seealso::` links are valid
- [ ] Internal documentation links work

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

### **Edge Cases and Gotchas**

#### **Type Hint Changes**

```python
# Type hints are part of the API contract
# BEFORE
def process(data: str) -> dict:

# AFTER - Breaking change for type checkers!
def process(data: str) -> Optional[dict]:  # Can now return None!
```

**Action**: Document new return behavior. Update examples showing None handling.

#### **Default Value Changes**

```python
# BEFORE
def configure(timeout=30):
    """Configure with timeout in seconds."""

# AFTER - Behavior change!
def configure(timeout=60):  # Different default!
    """Configure with timeout in seconds."""
```

**Action**: Document new default. Add to CHANGELOG. Check if any examples relied on old default.

#### **Dependency on Other Changed Functions**

If function A calls function B, and B changed behavior:

```python
# If internal_process() changed behavior
def public_api():
    """Public API function."""
    result = internal_process()  # This might behave differently now
    return result
```

**Action**: Check if public_api's documented behavior is still accurate.

#### **Module Reorganization**

```python
# BEFORE
from osprey.registry.manager import RegistryManager

# AFTER - Import path changed!
from osprey.registry import RegistryManager  # Moved to __init__.py
```

**Action**: Update all import examples. Add backward compatibility note if maintained.

#### **Decorator Changes**

```python
# BEFORE
@capability_node
class MyCapability(BaseCapability):
    pass

# AFTER - New decorator parameter!
@capability_node(auto_register=True)  # New parameter added
class MyCapability(BaseCapability):
    pass
```

**Action**: 
- Update all examples showing decorator usage
- Document new parameter in decorator documentation
- Add to CHANGELOG
- Check if existing code breaks or needs migration

#### **Class Attribute Changes**

```python
# BEFORE
class MyCapability(BaseCapability):
    name = "my_capability"
    
# AFTER - New required attribute!
class MyCapability(BaseCapability):
    name = "my_capability"
    version = "1.0.0"  # Now required
```

**Action**: 
- Update class docstring
- Update all examples
- Update developer guides showing class structure
- Add migration guide if breaking

#### **Constants and Configuration Defaults**

```python
# BEFORE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# AFTER - Values changed!
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60
```

**Action**: 
- Document in module docstring
- Update configuration documentation
- Check all examples that reference these values
- Add to CHANGELOG explaining rationale

#### **Async/Await Pattern Changes**

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

#### **Property to Method Changes (or vice versa)**

```python
# BEFORE - Property access
class Config:
    @property
    def settings(self):
        return self._settings

# AFTER - Method call required!
class Config:
    def get_settings(self):
        return self._settings
```

**Action**: 
- Update docstring
- Search for all `config.settings` and change to `config.get_settings()`
- Breaking change - add migration guide
- Update all examples

#### **Error Message or Exception Text Changes**

```python
# BEFORE
raise ValueError("Invalid configuration")

# AFTER - More specific message
raise ValueError("Invalid configuration: missing required field 'api_key'")
```

**Action**: 
- Update error handling examples if they check message content
- Update troubleshooting guides with new error messages
- Update integration tests that assert on error messages

#### **Logging Output Changes**

```python
# BEFORE
logger.info("Processing started")

# AFTER
logger.info(f"Processing started for capability: {capability_name}")
```

**Action**: 
- Update any documentation that shows log output
- Update troubleshooting guides
- If log format changed, document new format

#### **Template File Changes** (.j2 files)

```python
# Changes to docker-compose.yml.j2 or other templates
```

**Action**: 
- Update deployment documentation
- Update getting started guides that reference templates
- Check if example projects need template updates
- Document new template variables

#### **CLI Output Format Changes**

```bash
# BEFORE
osprey deploy
> Deploying application...

# AFTER
osprey deploy
> [2025-01-15 10:30:00] Deploying application...
```

**Action**: 
- Update CLI documentation with new output format
- Update troubleshooting guides
- Update any scripts that parse CLI output

#### **State Schema Changes**

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

#### **__all__ Exports Changes**

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

#### **File Path or Directory Structure Changes**

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

#### **Initialization Sequence Changes**

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
# If you change BaseCapability interface
# → All capability implementations must be checked
# → All capability documentation must be reviewed
```

**State Model Changes → Everything**
```python
# If AgentState structure changes
# → Every capability that accesses state is affected
# → Every example showing state is affected
# → State management docs need comprehensive updates
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

If AgentState or related models change:
- **CRITICAL**: This affects almost everything
- Review ALL tutorials and examples
- Update state management documentation completely
- Check every capability that accesses state
- Verify serialization/deserialization still works

### **Warning Sign 6: Base Class Modifications**

If BaseCapability, BaseProvider, or other base classes change:
- Check ALL implementations
- Update developer guides for creating new implementations
- Verify backward compatibility
- Test all existing capabilities/providers

## 📖 Summary: The Golden Rule

**"If you changed code that users interact with, you must update documentation that describes that interaction."**

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

### **Documentation Quality Principles**

**Completeness**
- Every public API change documented
- All parameters and return values explained
- Examples provided for complex operations
- Error conditions documented

**Consistency**
- Terminology used uniformly across all docs
- Import paths match actual code
- Examples follow current best practices
- Version references are accurate

**Clarity**
- Clear, concise language
- Examples show realistic use cases
- Complex concepts explained with context
- Assumptions stated explicitly

**Maintainability**
- No historical/migration comments in code (see COMMENTS.md)
- Docstrings follow DOCSTRINGS.md guidelines
- Comments follow COMMENTS.md guidelines
- Documentation structure is logical and navigable

### **When Documentation Is NOT Needed** ✅

**To avoid over-documentation and busywork, SKIP documentation updates for:**

**1. Private/Internal Code**
- Functions starting with `_` (e.g., `_internal_helper()`)
- Classes starting with `_` (e.g., `_PrivateCache`)
- Changes to test files
- Internal utility functions not exposed in `__all__`

**2. No-Behavior-Change Refactoring**
```python
# BEFORE
def calculate_total(items):
    sum = 0
    for item in items:
        sum += item.price
    return sum

# AFTER - Refactored but identical behavior
def calculate_total(items):
    return sum(item.price for item in items)
```
**Documentation needed**: None (maybe CHANGELOG: "Refactored calculation logic")

**3. Bug Fixes That Restore Documented Behavior**
```python
# The docstring says it raises ValueError, but code returned None
# Fix: Now correctly raises ValueError as documented
```
**Documentation needed**: CHANGELOG only ("Fixed error handling")

**4. Code Quality Changes**
- Formatting changes (black, isort, etc.)
- Adding type hints to already-typed code
- Renaming internal variables
- Adding internal comments
- Fixing typos in internal comments

**5. Performance Optimizations (If Usage Unchanged)**
```python
# Changed from O(n²) to O(n) algorithm internally
# But function signature and behavior identical
```
**Documentation needed**: CHANGELOG only ("Improved performance of process_data")

**6. Logging/Debugging Changes**
- Adding debug-level log statements
- Improving internal error messages users don't see
- Adding internal tracing/metrics

**Examples of Changes That DON'T Need Documentation:**
- ❌ Reformatted code with black
- ❌ Fixed internal variable name from `tmp` to `temporary_result`
- ❌ Added `# type: ignore` comment
- ❌ Optimized internal loop (same output)
- ❌ Changed `_helper_function()` implementation
- ❌ Added unit test for existing functionality
- ❌ Fixed typo in internal comment
- ❌ Reordered internal imports

**When Uncertain**: Ask yourself, "Would this change affect what a user types or what they receive back?" If no, skip extensive documentation.

### **Common Mistakes to Avoid**

1. ❌ **Updating code but not examples** - Examples break, users get confused
2. ❌ **Forgetting CHANGELOG** - Users don't know what changed
3. ❌ **Not searching for all references** - Inconsistent documentation
4. ❌ **Assuming obvious changes** - What's obvious to you isn't to users
5. ❌ **Skipping cross-references** - Links break, navigation fails
6. ❌ **Not testing examples** - Examples that don't work are worse than none
7. ❌ **Historical comments in code** - Confuses new users about current system
8. ❌ **Not checking cascading effects** - Base class changes affect all subclasses

### **Documentation as Part of Definition of Done**

A feature is **NOT complete** until:
- [ ] All public APIs have complete docstrings
- [ ] Inline comments explain non-obvious logic
- [ ] API reference is updated
- [ ] User guides reflect new functionality
- [ ] Examples are updated and tested
- [ ] CHANGELOG is updated
- [ ] Breaking changes have migration guides
- [ ] Documentation builds without warnings
- [ ] All links are valid
- [ ] Cross-references work correctly

### **Final Thoughts**

Documentation is not an afterthought—it's an **essential part of professional software development**. 

**For Users**: Good documentation means they can successfully use your software without asking for help.

**For Maintainers**: Good documentation means you can understand your own code six months later.

**For Contributors**: Good documentation means they can contribute without fear of breaking things.

**For the Project**: Good documentation is the difference between a professional tool and an abandoned experiment.

Taking the time to update documentation properly:
- ✅ Prevents user confusion and frustration
- ✅ Reduces support burden
- ✅ Maintains professional reputation
- ✅ Enables community contribution
- ✅ Facilitates project growth
- ✅ Serves as a contract for API stability

**Remember**: Every hour spent on documentation saves ten hours of support, debugging, and explaining the same things repeatedly.

### **Final Sanity Check Before Committing**

Before you commit your changes, ask yourself:

1. **Proportionality Check**
   - *"Am I creating more documentation than the change warrants?"*
   - Small change = small doc update (or none)
   - Big change = comprehensive docs

2. **User Impact Check**
   - *"Would users actually notice this change?"*
   - If no → Minimal or no documentation
   - If yes → Document it properly

3. **Breaking Change Check**
   - *"Does this break existing user code?"*
   - If no → You're probably fine with docstring + CHANGELOG
   - If yes → You need migration guidance

4. **Consistency Check**
   - *"Are similar changes documented consistently?"*
   - Check existing documentation style
   - Match the existing level of detail

**Good enough is good enough.** Perfect documentation for a trivial change is wasted effort. Focus your energy where it matters: major features, breaking changes, and things users will actually interact with.

---

**This workflow document itself follows these principles** - it provides comprehensive, structured guidance while emphasizing proportionality and avoiding documentation busywork.

