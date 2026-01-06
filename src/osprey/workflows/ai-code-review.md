---
workflow: ai-code-review
category: code-quality
applies_when: [after_ai_generation, before_commit, refactoring]
estimated_time: 30-60 minutes (often 2x the generation time)
ai_ready: true
related: [pre-merge-cleanup, commit-organization, testing-workflow]
---

# AI Code Review and Refactoring Workflow

**Purpose**: Critical review and cleanup of AI-generated code to identify redundancy, unused functions, and API inconsistencies.

**Reality Check**: AI assistants are excellent at generating new code, but often create redundant functions, inconsistent APIs, and unnecessary complexity. This workflow helps you systematically identify and clean up these issues before committing.

**Starting Point**: You have uncommitted changes from an AI-assisted implementation session.

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant (Cursor/Copilot):**

```
Following @docs/workflows/ai-code-review.md, perform a systematic code review of my uncommitted changes.

PHASE 1 - Statistical Overview:
- Total lines added/removed
- Files changed (new vs modified)
- Functions/classes added
- Complexity indicators (nested loops, long functions >50 lines)

PHASE 2 - Critical Issues Analysis:

1. REDUNDANCY:
   - Duplicate/similar functions (>70% code similarity)
   - Repeated logic blocks (same pattern 3+ times)
   - Cross-module overlaps (same logic in different files)
   - Redundant parameters or return values

2. UNUSED CODE:
   - Functions defined but never called
   - Unused parameters (suggest running: ruff check --select ARG)
   - Unused imports (suggest running: ruff check --select F401)
   - Classes never instantiated

3. API CONSISTENCY:
   - Naming patterns (snake_case for functions, PascalCase for classes)
   - Parameter ordering across similar functions
   - Return type consistency (dict vs object, None vs raise)
   - Breaking changes without migration path

4. TYPE SAFETY:
   - Missing type hints on public functions
   - Inconsistent type syntax (List[] vs list[], Optional vs |)
   - Overuse of Any type
   - Missing return type hints

5. ERROR HANDLING:
   - Inconsistent patterns (raise vs return None)
   - Bare except clauses
   - Missing input validation
   - Improper exception types

6. OVER-ENGINEERING:
   - Unnecessary abstractions (ABC with single implementation)
   - Premature optimization (caching, custom data structures)
   - Overly complex solutions to simple problems

For each issue found, provide:
- Category + Priority (CRITICAL/HIGH/MEDIUM/LOW)
- File path and line numbers
- Code snippet showing the problem
- Impact analysis (breaking changes, performance, maintainability)
- Concrete fix with code example
- Estimated time to fix

PHASE 3 - Action Plan:
- Summary statistics (issues by category and priority)
- Prioritized action plan (critical first)
- Quick wins vs deep refactors
- Total estimated cleanup time
```

**After the AI review**, work through the detailed analysis steps below for manual verification.

**Related workflows**: [pre-merge-cleanup.md](pre-merge-cleanup.md), [commit-organization.md](commit-organization.md)

---

## ⚠️ Why This Matters

### The AI Generation Problem

AI coding assistants have a bias toward **additive** changes:
- ✅ Excellent at: Creating new functions, adding features, generating boilerplate
- ❌ Poor at: Removing redundancy, simplifying, maintaining consistency

**Common issues in AI-generated code:**
- Functions that solve the same problem with slight variations
- Overly generic abstractions "for future extensibility"
- Inconsistent APIs (especially when generating across multiple sessions)
- Leftover experimental code that wasn't cleaned up
- Over-engineered solutions to simple problems

### Time Investment Reality

```
AI Generation:    1 hour  (fast, creative, generates lots of code)
Critical Review:  2 hours (slower, requires judgment, removes/refactors)
Total Time:       3 hours (review often takes 2x generation time)
```

**This is normal and expected.** The review step is where you add professional quality to AI-generated code.

---

## Priority Levels

```
CRITICAL   → API breaks, security issues, major redundancy (must fix now)
HIGH       → Unused code, inconsistency, moderate redundancy (fix before merge)
MEDIUM     → Over-engineering, minor inconsistencies (fix if time permits)
LOW        → Style preferences, micro-optimizations (defer or skip)
```

---

## 1. Redundancy Analysis (CRITICAL/HIGH)

**Goal**: Find duplicate or overlapping functionality.

### 1.1 Duplicate Functions

**Scan for similar function names:**
```bash
# List all new function definitions
git diff main...HEAD | grep -E "^\+def " | sed 's/^\+def //' | cut -d'(' -f1 | sort

# Find functions with similar prefixes (potential duplicates)
git diff main...HEAD | grep -E "^\+def " | sed 's/^\+def //' | cut -d'(' -f1 | \
  awk -F'_' '{if (NF > 1) print $1"_*"}' | sort | uniq -c | sort -rn | \
  awk '$1 > 2 {print "⚠ Multiple functions with prefix: " $2 " (count: " $1 ")"}'

# More reliable: Use ruff to detect similar code
ruff check --select SIM src/ tests/  # Simplification suggestions
```

**Manual check**: Do any function names suggest similar purposes?
- `get_X` / `fetch_X` / `retrieve_X` → likely redundant
- `process_X` / `handle_X` / `execute_X` → possibly redundant
- `validate_X` / `check_X` / `verify_X` → possibly redundant

**Action**: Compare function bodies:
```bash
# Example: compare two suspicious functions
git diff main...HEAD | grep -A 20 "^\+def get_data"
git diff main...HEAD | grep -A 20 "^\+def fetch_data"
```

**If >70% similar** → Merge into single function with parameters for variation.

### 1.2 Duplicate Logic Blocks

**Scan for repeated code patterns:**
```bash
# Find common code patterns
git diff main...HEAD | grep "^\+" | sort | uniq -c | sort -rn | head -20
```

**Manual inspection**: Look for repeated validation, error handling, or data transformation code.

**Refactoring rule**: Extract to helper if >70% similar and used 3+ times.

### 1.3 Overlapping Functionality Across Modules

**Cross-file redundancy scan:**
```bash
# Find files modified/added
git diff main...HEAD --name-only --diff-filter=AM | grep "\.py$"

# For each file, extract function signatures
git diff main...HEAD --name-only --diff-filter=AM | grep "\.py$" | while read file; do
  echo "=== $file ==="
  git diff main...HEAD -- "$file" | grep -E "^\+def " | sed 's/^\+def /  /'
done
```

**Manual review**: Look for functional overlap between modules:
- Database access in multiple files → centralize in `db.py`
- API calls scattered across modules → centralize in `api_client.py`
- Config parsing in multiple places → centralize in `config.py`
- Similar data transformations → centralize in `utils.py` or `transforms.py`

**Decision matrix:**

| Scenario | Action |
|----------|--------|
| Identical logic in 2+ modules | Move to shared utility module |
| Similar but specialized logic | Extract common core to shared module, keep specializations |
| Accidental duplication | Remove, import from existing location |
| Intentional separation (e.g., different domains) | Keep separate, add comment explaining why |

---

## 2. Unused Code Detection (HIGH)

**Goal**: Find and remove code that's defined but never used.

### 2.1 Unused Functions

**Automated scan with ruff (recommended):**
```bash
# Most reliable: Use ruff to find unused functions
ruff check --select ARG,F841 src/ tests/

# Look for functions that might be unused (manual review needed)
# Note: This is less reliable than ruff but can catch some cases
git diff main...HEAD | grep -E "^\+def [a-zA-Z_]" | \
  sed 's/^\+def //' | cut -d'(' -f1 | sort -u > /tmp/new_functions.txt

echo "=== Checking for potentially unused functions ==="
while read func; do
  # Count how many times function appears in codebase
  # (definition + uses)
  count=$(git grep -w "$func" -- "*.py" | grep -c "$func(")

  # If only found once, likely just the definition (unused)
  if [ "$count" -le 1 ]; then
    echo "⚠ Potentially unused: $func"
    git grep -n "def $func(" -- "*.py" | head -1
  fi
done < /tmp/new_functions.txt

# Alternative: Use vulture for dead code detection
# pip install vulture
# vulture src/ --min-confidence 80
```

**Manual verification** (automated scan has false positives):
- Check if function is part of public API (in `__all__`)
- Check if used in tests (search `tests/` directory)
- Check if used in docs/examples
- Check if called via `getattr()` or other dynamic mechanisms

**Decision tree:**

```
Is function used?
├─ No, and no planned use → DELETE
├─ No, but part of public API → KEEP (but add tests)
├─ No, but will be used soon → ADD TODO with issue link or DELETE
└─ Yes → KEEP
```

### 2.2 Unused Parameters & Imports

**Scan for unused code:**
```bash
ruff check --select ARG src/ tests/  # Unused parameters
ruff check --select F401 src/ tests/  # Unused imports
ruff check --select F401 --fix src/ tests/  # Auto-fix imports
```

**When to keep unused parameters:**
- Required by interface/protocol (add comment)
- Part of callback signature (use `_unused` prefix)

---

## 3. API Consistency Analysis (CRITICAL)

**Goal**: Ensure consistent and intuitive APIs across the codebase.

### 3.1 Naming Consistency

**Scan for inconsistent naming patterns:**
```bash
# Check for mixed naming conventions
echo "=== Checking function naming ==="
git diff main...HEAD | grep -E "^\+def " | \
  sed 's/^\+def //' | cut -d'(' -f1 | \
  grep -E "[A-Z]" && echo "⚠ Found camelCase/PascalCase in function names"

echo "=== Checking class naming ==="
git diff main...HEAD | grep -E "^\+class " | \
  sed 's/^\+class //' | cut -d'(' -f1 | cut -d':' -f1 | \
  grep -E "^[a-z]" && echo "⚠ Found snake_case in class names"
```

**Python conventions:**
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

**Consistency within similar functions:**
```bash
# Find functions with similar prefixes
git diff main...HEAD | grep -E "^\+def " | sed 's/^\+def //' | cut -d'(' -f1 | \
  cut -d'_' -f1 | sort | uniq -c | sort -rn
```

**Check**: If you have `get_`, `fetch_`, and `retrieve_` prefixes, pick ONE and standardize.

**Naming checklist:**
- [ ] All function names use consistent verbs for similar operations
- [ ] All boolean functions start with `is_`, `has_`, or `can_`
- [ ] All functions returning counts start with `count_` or `get_*_count`
- [ ] Private functions use `_` prefix consistently

### 3.2 Parameter Consistency

**Scan function signatures:**
```bash
git diff main...HEAD | grep -E "^\+def " | head -20
```

**Consistency rules:**
1. **Identifiers first**: `user_id`, `email`, etc.
2. **Primary data next**: `name`, `data`, etc.
3. **Options last**: `config`, `options`, flags
4. **Optional parameters**: Always use defaults, always last

Ensure similar functions (create/update/delete) use same parameter ordering.

### 3.3 Return Type Consistency

**Check return patterns:**
```bash
git diff main...HEAD | grep -E "^\+def .* -> " | sed 's/.*-> //' | sort | uniq -c | sort -rn
```

**Common fixes:**
- Use modern type syntax: `dict` not `Dict`, `list` not `List`
- Similar functions should handle "not found" consistently (all return `None` OR all raise)
- Use consistent types within a module (all `dict` or all custom objects)

### 3.4 Breaking Changes

**Critical check for changes to existing APIs:**
```bash
# Find modified function signatures
git diff main...HEAD | grep -E "^-def |^\+def " | grep -B1 "^\+def " | grep "^-def "
```

**If you modified existing APIs, verify:**
- [ ] Change is backward compatible, OR
- [ ] Change is documented in CHANGELOG as **BREAKING**, AND
- [ ] Migration path is provided, AND
- [ ] All call sites in codebase are updated

**Example of proper breaking change:**
```markdown
### Changed
- **BREAKING**: `process_data(data)` now requires `config` parameter
  - Migration: Add `config={}` for default behavior
  - Reason: Required to support new configuration options
  - Updated all internal call sites
  - External users: Update calls before upgrading
```

---

## 4. Type Safety Review (HIGH)

**Goal**: Ensure type hints are complete, consistent, and accurate.

### 4.1 Missing Type Hints

**Python best practice**: All public functions should have type hints.

**Scan for functions without return type hints:**
```bash
# Find public functions without return type hints
git diff main...HEAD | grep -E "^\+def [a-z_]+" | grep -v " -> " | head -20

# Count functions with vs without hints
echo "Functions with type hints:"
git diff main...HEAD | grep -E "^\+def " | grep -c " -> "
echo "Functions without type hints:"
git diff main...HEAD | grep -E "^\+def " | grep -vc " -> "
```

**Rules:**
- ✅ All public functions must have complete type hints
- ✅ All public methods must have complete type hints
- ⚠️ Private functions (`_prefix`) should have hints if complex
- ⚠️ Test functions may omit hints (but useful for clarity)


### 4.2 Inconsistent Type Hint Syntax

**Python 3.9+ best practices** (PEP 585, PEP 604):
- ✅ `list[str]` instead of `List[str]`
- ✅ `dict[str, int]` instead of `Dict[str, int]`
- ✅ `tuple[int, ...]` instead of `Tuple[int, ...]`
- ✅ `str | None` instead of `Optional[str]` or `Union[str, None]`

**Check for old-style type hints:**
```bash
# Find old typing imports
git diff main...HEAD | grep -E "from typing import (List|Dict|Tuple|Optional|Union)"

# Find usage of old-style hints
echo "=== Old-style type hints found ==="
git diff main...HEAD | grep -E "List\[|Dict\[|Tuple\[" | grep "^\+" | head -10
git diff main...HEAD | grep -E "Optional\[|Union\[" | grep "^\+" | head -10
```

**When to keep `typing` module:** `Any`, `TypeVar`, `Protocol`, `Callable`, `Literal`

### 4.3 Overuse of `Any` Type

**The `Any` type defeats type checking.** Use sparingly.

**Find Any types:**
```bash
# Find Any usage
git diff main...HEAD | grep -E ": Any|-> Any|\[Any\]" | grep "^\+"
```

**When `Any` is acceptable:**
- ✅ Truly dynamic data (e.g., JSON parsing results)
- ✅ Third-party library without type stubs
- ✅ Complex recursive types (temporary, add TODO)
- ✅ Gradual typing migration (temporary)

**When to fix `Any`:**
- ❌ Internal functions with known types
- ❌ Data models and schemas
- ❌ Public API functions
- ❌ Simple dictionary structures


### 4.4 Type Checking Validation

**Run mypy to verify type correctness:**
```bash
# Check type hints are correct
mypy src/osprey --ignore-missing-imports

# Check for common type issues
mypy src/osprey --strict --ignore-missing-imports 2>&1 | grep "error:" | head -20
```

**Common type errors AI introduces:**
- Returning wrong type from function
- Missing `None` in return type
- Incompatible types in conditionals
- Type narrowing issues with `isinstance()`

---

## 5. Error Handling Consistency (HIGH)

**Goal**: Ensure predictable, consistent error handling patterns.

### 5.1 Exception Pattern Consistency

**Check for inconsistent error patterns:**
```bash
# Find functions that return None for errors
git diff main...HEAD | grep -E "-> .* \| None" | wc -l

# Find functions that raise exceptions
git diff main...HEAD | grep -E "raise (ValueError|TypeError|KeyError|RuntimeError)" | wc -l

# These should follow consistent patterns!
```

**Consistency rules for Osprey:**

| Scenario | Pattern | Example |
|----------|---------|---------|
| **Input validation fails** | Raise `ValueError` or `TypeError` | `raise ValueError("PV name cannot be empty")` |
| **Resource not found** | Return `None` | `return None  # User not found` |
| **External service fails** | Raise specific exception | `raise ConnectionError("LLM API unavailable")` |
| **Configuration error** | Raise `ValueError` at startup | `raise ValueError("Invalid config: missing API key")` |
| **Internal logic error** | Raise `RuntimeError` | `raise RuntimeError("Invalid state: ...")` |


### 5.2 Bare Except Clauses (CRITICAL)

**Bare `except:` clauses are dangerous** - they catch everything including `KeyboardInterrupt` and `SystemExit`.

**Find problematic exception handling:**
```bash
# Find bare except clauses (CRITICAL issue)
git diff main...HEAD | grep -n "except:$"

# Find overly broad except clauses
git diff main...HEAD | grep -n "except Exception"

# Should use specific exceptions instead!
```

**Fix**: Use specific exceptions like `(ValueError, KeyError)`, not bare `except:` or `except Exception`.

### 5.3 Missing Input Validation

**AI often forgets to validate inputs.** Check all public functions.

**Scan for functions without validation:**
```bash
# Find public functions that take parameters
git diff main...HEAD | grep -E "^\+def [a-z_]+\([^)]+\)" | head -20

# Manually review: Do they validate inputs?
```

**Validation checklist:**
- [ ] Required parameters not None/empty
- [ ] Numeric parameters in valid range
- [ ] File paths exist
- [ ] URLs well-formed

### 5.4 Exception Context Loss

**Preserve exception context** with `raise ... from e` for debugging.

**Check for context loss:**
```bash
# Find raises without 'from' clause
git diff main...HEAD | grep "raise " | grep -v " from " | grep "^\+"
```

**Always use `raise ... from e`** to preserve exception chain for debugging.

---

## 6. Over-Engineering Detection (MEDIUM)

**Goal**: Identify unnecessary complexity introduced by AI.

**Red flags to scan for:**
```bash
git diff main...HEAD | grep -c "from abc import ABC"  # ABCs with 1 implementation
git diff main...HEAD | grep -i "cache\|lru_cache"     # Premature caching
git diff main...HEAD | grep -E "config\[|getenv\("    # Over-configuration
```

**Common issues:**
- **Unnecessary abstractions**: ABC with only 1 implementation → Simplify to function/class
- **Premature optimization**: Caching without profiling → Remove until proven needed
- **Configuration bloat**: Config for things that never change → Use sensible defaults

**Decision**: Keep abstraction only if 3+ implementations exist and are used.

---

## 7. Architecture Review (HIGH)

**Quick checks:**
```bash
# List new files
git diff main...HEAD --name-only --diff-filter=A | grep "\.py$"

# Check imports
git diff main...HEAD | grep -E "^\+(import |from .* import )" | sed 's/^\+//' | sort -u
```

**Verify:**
- [ ] New files in appropriate modules (utils, services, capabilities, etc.)
- [ ] No circular imports
- [ ] Dependencies flow high-level → low-level
- [ ] Functions have single responsibility
- [ ] External dependencies added to `pyproject.toml`

---

## 8. Systematic Review Workflow

**Step-by-step process to work through your uncommitted changes:**

### Step 1: Statistical Overview (5 min)

```bash
echo "=== Change Statistics ==="
echo "Files changed: $(git diff main...HEAD --name-only | wc -l)"
echo "Lines added: $(git diff main...HEAD --numstat | awk '{sum+=$1} END {print sum}')"
echo "Lines removed: $(git diff main...HEAD --numstat | awk '{sum+=$2} END {print sum}')"
echo ""
echo "New functions: $(git diff main...HEAD | grep -c '^+def ')"
echo "New classes: $(git diff main...HEAD | grep -c '^+class ')"
echo "New files: $(git diff main...HEAD --name-only --diff-filter=A | wc -l)"
echo ""
echo "Modified files:"
git diff main...HEAD --name-only | head -10
```

**Reality check**: If you added 1000+ lines but only needed one feature, something is wrong.

### Step 2: Quick AI Scan (10 min)

Use the AI prompt from the top of this document. Review the AI's findings.

### Step 3: Manual Verification (30-45 min)

Work through sections 1-7 above, focusing on HIGH and CRITICAL items.

### Step 4: Refactoring (20-60 min)

Make changes based on your findings. Prioritize:
1. Delete unused code (quick wins)
2. Fix API inconsistencies (prevents future pain)
3. Consolidate redundancy (improves maintainability)
4. Simplify over-engineering (reduces technical debt)

### Step 5: Validation (15 min)

```bash
# Run tests
pytest tests/ --ignore=tests/e2e -v

# Check lints and formatting
ruff check src/ tests/
black --check src/ tests/

# Verify no unused code
ruff check --select F401,ARG,F841 src/

# Check type hints
mypy src/osprey --ignore-missing-imports

# Verify no critical issues
echo "Checking for critical issues..."
git diff main...HEAD | grep "except:$" && echo "⚠ CRITICAL: Bare except found!" || echo "✓ No bare excepts"

# Check docs still build
cd docs && make html
```

---

## 9. Decision Framework

When you find an issue, use this framework to decide what to do:

### Delete vs. Keep Decision Tree

```
Is the code used?
├─ Not used anywhere
│  ├─ Part of public API? → Keep but add tests and docs
│  └─ Internal code? → DELETE
├─ Used in one place
│  ├─ Complex logic? → Keep as function
│  └─ Simple logic? → Inline at call site
└─ Used in 2+ places → Keep and enhance
```

### Refactor vs. Rewrite Decision Tree

```
Is the code problematic?
├─ Wrong → Rewrite (fix logic)
├─ Inconsistent with codebase → Refactor (align with patterns)
├─ Over-engineered → Simplify (remove complexity)
├─ Under-engineered → Enhance (add error handling, tests)
└─ Just different style → Keep (style is subjective)
```

### Fix Now vs. Fix Later Decision Tree

```
What's the impact?
├─ Security issue → Fix now (BLOCKER)
├─ API break → Fix now (CRITICAL)
├─ Causes errors → Fix now (CRITICAL)
├─ Unused code → Fix now (HIGH) - easy to remove
├─ Inconsistency → Fix now if easy, defer if complex (HIGH/MEDIUM)
├─ Over-engineering → Fix now if simple, defer if risky (MEDIUM)
└─ Style/preference → Defer (LOW)
```

---

## 10. Common Anti-Patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| **"Just in Case" parameters** | Unused params for "flexibility" | Remove until actually needed |
| **Partial implementation** | TODOs and unfinished methods | Complete or remove |
| **Abstraction ladder** | 4+ layers of trivial wrappers | Collapse to 2 layers max |
| **Swiss Army function** | One function with many modes | Split into focused functions |
| **Speculative generics** | ABC with 1 implementation | Remove abstraction |

---

## 11. Checklist

### Before Starting Review
- [ ] All changes are from current session (not mixing old and new work)
- [ ] Tests pass in current state
- [ ] You have 20-40 minutes for focused review

### Redundancy (CRITICAL/HIGH)
- [ ] No duplicate or near-duplicate functions
- [ ] No repeated logic blocks (>70% similar)
- [ ] No overlapping functionality across modules
- [ ] Common patterns extracted to shared utilities

### Unused Code (HIGH)
- [ ] All defined functions are called somewhere
- [ ] All function parameters are used (or documented why not)
- [ ] All imports are used
- [ ] All classes are instantiated

### API Consistency (CRITICAL)
- [ ] Naming follows project conventions (snake_case, PascalCase, etc.)
- [ ] Similar functions have consistent parameter ordering
- [ ] Return types are consistent across similar functions
- [ ] Breaking changes are documented with migration path

### Type Safety (HIGH)
- [ ] All public functions have complete type hints
- [ ] Type hints use modern syntax (list[] not List[], | not Union)
- [ ] Any type is used sparingly and documented
- [ ] mypy passes with no errors

### Error Handling (HIGH)
- [ ] Exception patterns are consistent (raise vs return None)
- [ ] No bare except clauses
- [ ] Input validation on all public functions
- [ ] Exception context preserved with "from e"

### Over-Engineering (MEDIUM)
- [ ] No unnecessary abstractions (ABC with single impl)
- [ ] No premature optimizations (caching, custom data structures)
- [ ] Configuration is minimal and necessary
- [ ] Simple problems have simple solutions

### Architecture (HIGH)
- [ ] New code is in appropriate modules
- [ ] No circular dependencies
- [ ] Dependencies flow in correct direction (high-level → low-level)
- [ ] Separation of concerns maintained

### After Refactoring
- [ ] Tests still pass
- [ ] Lints pass
- [ ] Docs still build
- [ ] CHANGELOG updated
- [ ] Git diff is smaller than before review

---


---

## 12. Time Management

**Realistic time allocation for different change sizes:**

| Change Size | Generation | Review | Refactoring | Total |
|-------------|------------|--------|-------------|-------|
| Small (< 200 lines) | 15 min | 15 min | 15 min | 45 min |
| Medium (200-500 lines) | 30 min | 30 min | 30 min | 90 min |
| Large (500-1000 lines) | 60 min | 60 min | 60 min | 3 hours |
| X-Large (1000+ lines) | 2 hours | 2 hours | 2 hours | 6 hours |

**Tips:**
- **Don't skip review to save time** - Technical debt costs more later
- **Review same day as generation** - Context is fresh
- **Take breaks** - Critical thinking requires focus
- **Use AI for mechanical tasks** - But verify its suggestions

---

## See Also

- [pre-merge-cleanup.md](pre-merge-cleanup.md) - Final checks before committing
- [commit-organization.md](commit-organization.md) - Organizing refactored code into commits
- [testing-workflow.md](testing-workflow.md) - Writing tests for refactored code
- [docstrings.md](docstrings.md) - Documenting your cleaned-up APIs

---

**Remember**: AI is your creative partner for generating code, but YOU are the senior engineer who ensures quality, consistency, and maintainability. This review step is where you add professional polish to AI-generated code.

**Rule of Thumb**: If you can't explain why every function exists, it probably shouldn't.


