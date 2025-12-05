# Pre-Merge Cleanup Guide

**Purpose**: Systematic detection of loose ends before merging feature branches.

**Principle**: If a diff needs extensive explanation, it's incomplete.

---

## Priority Levels

```
BLOCKER  → Merge immediately fails (debug code, secrets, broken tests)
CRITICAL → Required for merge (CHANGELOG, docstrings, type hints)
HIGH     → Must verify manually (TODOs, refactoring completion, coverage)
MEDIUM   → Clean but not blocking (formatting, docs warnings, orphans)
```

---

## Quick Scan (3 minutes)

Run this first to catch 90% of issues:

```bash
BASE="${1:-main}"

echo "=== BLOCKERS ==="
git diff $BASE...HEAD | grep -E "^\+.*(print\(|pdb\.|breakpoint\(|console\.log)" && echo "⚠ Debug code found"
git diff $BASE...HEAD | grep -E "^\+.*# *(def |class |import )" && echo "⚠ Commented code found"
git diff $BASE...HEAD | grep -iE "^\+.*(password|api_key|token).*=.*[\"']" | grep -v getenv && echo "⚠ Hardcoded secrets found"
pytest tests/ --ignore=tests/e2e -x --tb=short -q || echo "⚠ Tests failing"

echo -e "\n=== CRITICAL ==="
git diff $BASE...HEAD --name-only | grep -q CHANGELOG.md || echo "⚠ CHANGELOG.md not modified"
git diff $BASE...HEAD | grep -E "^\+def [a-z_]|^\+class [A-Z]" | wc -l | xargs echo "New functions/classes:"
git diff $BASE...HEAD | grep -E "^\+def [a-z_].*->" | wc -l | xargs echo "  With return type hints:"

echo -e "\n=== HIGH ==="
git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK)" | xargs echo "New TODOs:"
git diff $BASE...HEAD | grep -cE "^\+.*os\.(getenv|environ)" | xargs echo "New env vars:"
git diff $BASE...HEAD --name-only --diff-filter=D | wc -l | xargs echo "Deleted files:"

echo -e "\n=== MEDIUM ==="
git diff $BASE...HEAD --name-only | grep -E "\.(tmp|bak|swp|orig)$" && echo "⚠ Temp files tracked"
git status --short | grep "^?" && echo "⚠ Untracked files in working tree"
```

**Action**: If any BLOCKER or test failures → stop and fix. Otherwise proceed to detailed scans.

---

## Detailed Scans by Category

### 1. Debug Artifacts (BLOCKER)

**Scan:**
```bash
git diff $BASE...HEAD | grep -nE "^\+.*(print\(|pdb|breakpoint|console\.log|debugger)"
git diff $BASE...HEAD | grep -nE "^\+.*logger\.(info|warning|error)" | grep -v "# "
```

**Remove:**
- `print()`, `pprint()`, `console.log()` — unless in CLI/test output
- `pdb.set_trace()`, `breakpoint()` — always
- Debug-level logs with sensitive data
- `import pdb` when not used elsewhere

**Keep:**
- `logger.debug()` — acceptable for development traces
- `print()` in `if __name__ == "__main__"` blocks
- Test assertion messages

---

### 2. Commented Code (BLOCKER)

**Scan:**
```bash
git diff $BASE...HEAD | grep -E "^\+.*# *(def |class |import |return |if |for )"
```

**Remove:**
- Entire commented functions/classes
- Commented imports
- Commented logic blocks > 3 lines

**Keep:**
- `# TODO: Future enhancement (issue #NNN)` — with issue reference
- `# NOTE: Business logic per requirement X` — explains *why*, not *what*
- Documentation examples in docstrings/comments

---

### 3. Hardcoded Secrets (BLOCKER)

**Scan:**
```bash
git diff $BASE...HEAD | grep -iE "(password|secret|api_key|token|bearer)" | grep -E "(=|:)" | grep -v "getenv\|environ\|param"
```

**Any match** → Stop immediately. Move to env vars + add to `env.example`:
```bash
# Description of what this key does
API_KEY=your_key_here  # Example or placeholder
```

---

### 4. CHANGELOG (CRITICAL)

**Check:**
```bash
git diff $BASE...HEAD CHANGELOG.md | grep -A 5 "## \[Unreleased\]"
```

**Required**: Entry under `## [Unreleased]` in one of:
```markdown
### Added      # New features, CLI commands, capabilities
### Changed    # Behavior changes, API breaks, deprecations
### Fixed      # Bug fixes (include issue # if exists)
### Removed    # Deleted features
```

**Breaking change format:**
```markdown
### Changed
- **BREAKING**: `old_func(x)` now requires `param` argument
  - Migration: Add `param="default"` to existing calls
  - Reason: [brief justification]
```

---

### 5. Docstrings + Type Hints (CRITICAL)

**Scan for new public APIs:**
```bash
# Find new functions/classes without leading underscore
git diff $BASE...HEAD | grep -E "^\+def [a-z_]+\(|^\+class [A-Z]" | grep -v "^+    " | grep -v "^+def _"
```

**Each needs:**
```python
def func(arg: str, opt: int = 0) -> dict:
    """One-line summary ending with period.

    Longer description if needed. Explain purpose, not implementation.

    Args:
        arg: What it represents (not "the arg parameter")
        opt: What it does. Defaults to 0.

    Returns:
        Dict with keys: 'result', 'status'

    Raises:
        ValueError: If arg is empty
        RuntimeError: If operation fails
    """
```

**Missing:** Run ruff or mypy to find:
```bash
ruff check --select ANN src/  # Missing type annotations
```

**See**: `DOCSTRINGS.md` for full spec.

---

### 6. TODOs and FIXMEs (HIGH)

**Scan:**
```bash
git diff $BASE...HEAD | grep -nE "^\+.*(TODO|FIXME|HACK|XXX)" | sed 's/^/  /'
```

**Decision tree:**

| Fix Time | Action |
|----------|--------|
| < 10 min | Fix now, remove comment |
| Blocks feature | Must fix before merge |
| Future work | Add issue → `# TODO: Description (issue #NNN)` |
| Obsolete | Delete |

**Valid:**
```python
# TODO: Add retry logic for network failures (issue #456)
# FIXME: Refactor to use new API after v2.0 release (issue #457)
```

**Invalid:**
```python
# TODO: fix this
# FIXME: make better
# HACK: temporary workaround  # (no issue link)
```

---

### 7. Refactoring Completion (HIGH)

If you renamed/moved/deleted anything, verify no references remain:

```bash
# Example: renamed old_function → new_function
OLD="old_function"
git grep -n "$OLD" src/ tests/ docs/ | grep -v CHANGELOG | grep -v "migration"
# Should return ZERO results
```

**Checklist for renames:**
- [ ] All call sites updated (grep search)
- [ ] All imports updated
- [ ] `__all__` exports in `__init__.py` updated
- [ ] Test function names updated (`test_old_*` → `test_new_*`)
- [ ] Docstring cross-references updated (`:func:`, `:class:`)
- [ ] Documentation pages updated
- [ ] CHANGELOG entry in `### Changed` or `### Removed`

**For deleted files:**
```bash
git diff $BASE...HEAD --name-only --diff-filter=D | while read f; do
  module=$(echo "$f" | sed 's|/|.|g; s|\.py$||; s|^src/||')
  echo "Checking: $module"
  git grep -l "$module" src/ docs/ tests/ | grep -v __pycache__ | grep -v CHANGELOG
done
```

---

### 8. Config Synchronization (HIGH)

**New environment variables:**
```bash
git diff $BASE...HEAD | grep -oE 'getenv\("[^"]+"|environ\["[^"]+' | \
  sed 's/getenv("//; s/environ\["//' | tr -d '"' | sort -u > /tmp/new_env_vars.txt

while read var; do
  grep -q "^$var=" env.example || echo "Missing in env.example: $var"
done < /tmp/new_env_vars.txt
```

**New dependencies:**
```bash
git diff $BASE...HEAD | grep -E "^\+(import [a-z_]+|from [a-z_]+ import)" | \
  grep -v "^\+    " | sed 's/^\+//' | cut -d' ' -f2 | cut -d. -f1 | sort -u
```

Cross-check each against `pyproject.toml` dependencies. If missing:
```toml
dependencies = [
    "new-package>=1.2.3",  # Brief reason
]
```

**New config options in code:**
```bash
git diff $BASE...HEAD | grep -E "config\[|config\.get\(" | grep -oE '["'\''][^"'\'']+["'\'']' | sort -u
```

Verify each exists in template configs (e.g., `config.yml.j2`).

---

### 9. Test Coverage (HIGH)

**Missing test files:**
```bash
git diff $BASE...HEAD --name-only --diff-filter=A | grep "^src/.*\.py$" | while read f; do
  base=$(basename "$f" .py)
  test_file=$(find tests/ -name "test_${base}.py" -o -name "test_*${base}*.py" | head -1)
  [ -z "$test_file" ] && echo "No test file for: $f"
done
```

**Coverage check:**
```bash
pytest tests/ --ignore=tests/e2e --cov=src/osprey --cov-report=term-missing:skip-covered \
  | grep "^src/" | awk '$4 < 80 {print}'
```

**Minimum per new code:**
- New function → 1 happy path + 2 edge cases + 1 error case
- Modified function → 1 regression test
- New class → init + primary methods + error handling

---

### 10. Import Cleanup (MEDIUM)

**Unused imports:**
```bash
ruff check --select F401 src/ tests/  # Unused imports
ruff check --select I src/ tests/     # Import sorting
```

**Auto-fix:**
```bash
ruff check --select F401,I --fix src/ tests/
```

**Stray debug imports:**
```bash
git diff $BASE...HEAD | grep -E "^\+import (pdb|pprint|traceback)" | grep -v "^\+    "
```

---

### 11. Orphaned References (MEDIUM)

**Deleted modules still imported:**
```bash
git diff $BASE...HEAD --diff-filter=D --name-only | grep "\.py$" | while read f; do
  module=$(echo "$f" | sed 's|^src/||; s|/__init__\.py$||; s|\.py$||; s|/|.|g')
  refs=$(git grep -l "import.*$module\|from $module" src/ tests/ 2>/dev/null | wc -l)
  [ $refs -gt 0 ] && echo "Orphaned import: $module (found in $refs files)"
done
```

**Deleted test fixtures still referenced:**
```bash
git diff $BASE...HEAD --diff-filter=D | grep -E "^\-@pytest\.fixture|^\-def .+\(.*fixture" | \
  sed 's/.*def //; s/(.*//; s/@pytest.fixture.*//; s/^-//' | while read fixture; do
    git grep -q "$fixture" tests/ && echo "Deleted fixture still used: $fixture"
done
```

---

### 12. Documentation (MEDIUM)

**Build check:**
```bash
cd docs
make clean
make html 2>&1 | tee /tmp/docs_build.log
grep -iE "(warning|error)" /tmp/docs_build.log | grep -v "WARNING: html_static_path"

make linkcheck 2>&1 | grep -E "\(broken\|redirect\)" | head -10
```

**New modules documented:**
```bash
git diff $BASE...HEAD --name-only --diff-filter=A | grep "^src/.*\.py$" | while read f; do
  module=$(basename "$f" .py)
  grep -rq "$module" docs/source/ || echo "Module not in docs: $f"
done
```

**Common fixes:**
- `undefined label` → Fix `:ref:\`target\`` links
- `unknown document` → Update `.. toctree::` directive
- `broken link` → Update URL or remove

---

### 13. File Hygiene (MEDIUM)

**Tracked temp files:**
```bash
git diff $BASE...HEAD --name-only | grep -E "\.(tmp|bak|swp|orig|log|cache)$"
git ls-files | grep -E "tmp_.*\.py|.*\.bak$|.*\.orig$|\.DS_Store" | grep -v tests/fixtures/
```

**Untracked files that should be committed or ignored:**
```bash
git status --short | grep "^??" | awk '{print $2}'
```

**Large files accidentally added:**
```bash
git diff $BASE...HEAD --name-only | while read f; do
  [ -f "$f" ] && size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
  [ "$size" -gt 1000000 ] && echo "Large file ($(($size/1024))KB): $f"
done
```

---

### 14. Code Formatting (MEDIUM)

**Check:**
```bash
black --check src/ tests/ || echo "⚠ Black formatting needed"
isort --check src/ tests/ || echo "⚠ isort needed"
ruff check src/ tests/ || echo "⚠ Ruff errors found"
```

**Auto-fix all:**
```bash
black src/ tests/
isort src/ tests/
ruff check --fix src/ tests/
```

---

## Automation Script

Save as `scripts/premerge_check.sh`:

```bash
#!/bin/bash
set -eo pipefail
BASE="${1:-main}"
ERRORS=0

echo "🔍 Pre-merge scan against $BASE"
echo "========================================"

# BLOCKER checks
echo -e "\n=== BLOCKERS ==="
if git diff $BASE...HEAD | grep -qE "^\+.*(print\(|pdb\.|breakpoint\()"; then
  echo "✗ Debug code found"
  git diff $BASE...HEAD | grep -nE "^\+.*(print\(|pdb\.|breakpoint\()" | head -3
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No debug code"
fi

if git diff $BASE...HEAD | grep -qE "^\+.*# *(def |class )"; then
  echo "✗ Commented code found"
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No commented code"
fi

if git diff $BASE...HEAD | grep -iqE "^\+.*(password|api_key).*=.*[\"']" | grep -qv getenv; then
  echo "✗ Possible hardcoded secrets"
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No obvious secrets"
fi

if ! pytest tests/ --ignore=tests/e2e -x --tb=no -q >/dev/null 2>&1; then
  echo "✗ Tests failing"
  ERRORS=$((ERRORS + 1))
else
  echo "✓ Tests pass"
fi

# CRITICAL checks
echo -e "\n=== CRITICAL ==="
if git diff $BASE...HEAD --name-only | grep -q CHANGELOG.md; then
  echo "✓ CHANGELOG updated"
else
  echo "✗ CHANGELOG not updated"
  ERRORS=$((ERRORS + 1))
fi

new_funcs=$(git diff $BASE...HEAD | grep -cE "^\+def [a-z_]|^\+class [A-Z]" || echo 0)
typed_funcs=$(git diff $BASE...HEAD | grep -cE "^\+def [a-z_].*->" || echo 0)
if [ "$new_funcs" -gt 0 ]; then
  echo "  New functions/classes: $new_funcs"
  echo "  With type hints: $typed_funcs"
  if [ "$typed_funcs" -lt "$new_funcs" ]; then
    echo "⚠ Some functions missing return type hints"
  fi
fi

# HIGH checks
echo -e "\n=== HIGH ==="
todos=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME)" || echo 0)
linked=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME).*issue #[0-9]+" || echo 0)
unlinked=$((todos - linked))
if [ $unlinked -gt 0 ]; then
  echo "⚠ $unlinked TODOs without issue links"
  ERRORS=$((ERRORS + 1))
else
  echo "✓ All TODOs linked (count: $todos)"
fi

deleted=$(git diff $BASE...HEAD --name-only --diff-filter=D | wc -l | xargs)
if [ "$deleted" -gt 0 ]; then
  echo "  $deleted files deleted - verify no orphaned references"
fi

# MEDIUM checks
echo -e "\n=== MEDIUM ==="
if black --check src/ tests/ >/dev/null 2>&1; then
  echo "✓ Black formatted"
else
  echo "⚠ Black formatting needed"
fi

if ruff check src/ tests/ --quiet >/dev/null 2>&1; then
  echo "✓ Ruff clean"
else
  echo "⚠ Ruff issues found"
fi

# Summary
echo -e "\n========================================"
if [ $ERRORS -eq 0 ]; then
  echo "✅ Automated checks passed"
  echo ""
  echo "Manual verification needed:"
  echo "  • Docstrings complete (see section 5)"
  echo "  • Refactorings complete (see section 7)"
  echo "  • Test coverage adequate (see section 9)"
  echo "  • Config files synced (see section 8)"
  exit 0
else
  echo "❌ Found $ERRORS blocking issues"
  echo ""
  echo "See: docs/resources/other/PRE_MERGE_CLEANUP.md"
  exit 1
fi
```

**Usage:**
```bash
chmod +x scripts/premerge_check.sh
./scripts/premerge_check.sh main  # or origin/main
```

---

## Final Checklist

Before requesting merge:

### BLOCKER
- [ ] No debug code: `print()`, `pdb`, `breakpoint()`
- [ ] No commented code blocks
- [ ] No hardcoded secrets
- [ ] All tests pass: `pytest tests/ --ignore=tests/e2e -v`

### CRITICAL
- [ ] CHANGELOG.md has entry under `## [Unreleased]`
- [ ] New public functions have docstrings
- [ ] New public functions have type hints (args + return)

### HIGH
- [ ] All TODOs either fixed or link to issues
- [ ] Refactorings complete: `git grep OLD_NAME` returns zero
- [ ] New env vars added to `env.example`
- [ ] New dependencies in `pyproject.toml`
- [ ] Test coverage ≥80% for new code

### MEDIUM
- [ ] No orphaned imports/references
- [ ] Documentation builds clean: `cd docs && make html`
- [ ] Code formatted: `black src/ tests/ && ruff check`
- [ ] No tracked temp files (`.tmp`, `.bak`, etc.)

---

## Common Edge Cases

| Scenario | Keep | Remove |
|----------|------|--------|
| Logging | `logger.debug()`, `logger.info()` in prod code | `print()` except CLI/tests |
| TODOs | `# TODO: X (issue #123)` | `# TODO: fix`, `# FIXME: later` |
| Comments | `# Why: Business rule X` | `# what: loops over items` |
| Conditionals | `if sys.platform == "win32":` | `if False:`, `if 0:` |
| Compatibility | `OldClass = NewClass  # Deprecated v1.0` | Commented-out old classes |
| Imports | Production dependencies | `import pdb`, `import pprint` not in `__main__` |
| Test files | `tests/fixtures/*.tmp` (intentional) | `src/tmp_debug.py` |

**Rule**: Keep = documented purpose or production use. Remove = debug/experiment artifact.

---

## See Also

- `COMMIT_ORGANIZATION.md` — Organizing atomic commits
- `DOCSTRINGS.md` — Docstring specification
- `RELEASE_WORKFLOW.md` — Release preparation
- `COMMENTS.md` — When and how to comment code
