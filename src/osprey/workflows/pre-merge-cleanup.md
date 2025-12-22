---
workflow: pre-merge-cleanup
category: code-quality
applies_when: [before_commit, pre_review, before_merge]
estimated_time: 10-15 minutes
ai_ready: true
related: [commit-organization, update-documentation]
---

# Pre-Merge Cleanup Guide

**Purpose**: Systematic detection of loose ends before merging feature branches.

**Principle**: If a diff needs extensive explanation, it's incomplete.

## 🤖 AI Quick Start

**Paste this prompt to your AI assistant (Cursor/Copilot):**

```
I'm ready to commit my changes. Following @docs/workflows/pre-merge-cleanup.md,
scan my uncommitted changes for:

1. BLOCKERS:
   - Debug code (print, pdb, breakpoint, console.log)
   - Commented-out code blocks
   - Hardcoded secrets (passwords, API keys, tokens)
   - Failing tests

2. CRITICAL:
   - Missing CHANGELOG.md entry
   - New functions without docstrings
   - New functions without type hints

3. HIGH PRIORITY:
   - TODOs without issue links
   - New environment variables not in env.example
   - Orphaned imports or references

For each issue found, show me:
- Category (BLOCKER/CRITICAL/HIGH)
- File path and line number
- Specific code snippet
- Recommended fix

Then generate a checklist for me to work through.
```

**After the AI scan**, work through the detailed checklists in the sections below.

**Related workflows**: [commit-organization.md](commit-organization.md), [update-documentation.md](update-documentation.md)

---

## Priority Levels

```
BLOCKER  → Merge immediately fails (debug code, secrets, broken tests)
CRITICAL → Required for merge (CHANGELOG, docstrings, type hints)
HIGH     → Must verify manually (TODOs, refactoring completion, coverage)
MEDIUM   → Clean but not blocking (formatting, docs warnings, orphans)
```

---

## Quick Reference Table

| Check | Priority | Command | Expected Result |
|-------|----------|---------|----------------|
| Debug code | BLOCKER | `git diff main...HEAD \| grep -E "^\+.*print\("` | No matches |
| Tests pass | BLOCKER | `pytest tests/ --ignore=tests/e2e` | All pass |
| CHANGELOG | CRITICAL | `git diff main...HEAD CHANGELOG.md` | Has entries |
| Type hints | CRITICAL | `ruff check --select ANN src/` | No errors |
| TODOs linked | HIGH | `git diff main...HEAD \| grep "TODO"` | All have issue # |
| Coverage | HIGH | `pytest --cov=src/osprey` | ≥80% |
| Formatting | MEDIUM | `black --check src/` | Already formatted |
| Imports | MEDIUM | `ruff check --select F401,I src/` | No unused |

---

## Quick Scan (3 minutes)

Run this first to catch 90% of issues:

```bash
BASE="${1:-main}"

echo "=== BLOCKERS ==="
git diff $BASE...HEAD | grep -E "^\+.*(print\(|pdb\.|breakpoint\(|console\.log)" && echo "⚠ Debug code found" || true
git diff $BASE...HEAD | grep -E "^\+.*# *(def |class |import )" && echo "⚠ Commented code found" || true
git diff $BASE...HEAD | grep -iE "^\+.*(password|api_key|token).*=.*[\"']" | grep -v -E "(getenv|environ\[)" && echo "⚠ Hardcoded secrets found" || true
pytest tests/ --ignore=tests/e2e -x --tb=short -q || echo "⚠ Tests failing"

echo -e "\n=== CRITICAL ==="
git diff $BASE...HEAD --name-only | grep -q CHANGELOG.md && echo "✓ CHANGELOG updated" || echo "⚠ CHANGELOG.md not modified"
new_funcs=$(git diff $BASE...HEAD | grep -cE "^\+def [a-z_]|^\+class [A-Z]" || echo 0)
typed_funcs=$(git diff $BASE...HEAD | grep -cE "^\+def [a-z_].*->" || echo 0)
echo "New functions/classes: $new_funcs"
echo "  With return type hints: $typed_funcs"

echo -e "\n=== HIGH ==="
todos=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK)" || echo 0)
echo "New TODOs: $todos"
env_vars=$(git diff $BASE...HEAD | grep -cE "^\+.*os\.(getenv|environ)" || echo 0)
echo "New env vars: $env_vars"
deleted=$(git diff $BASE...HEAD --name-only --diff-filter=D | wc -l | tr -d ' ')
echo "Deleted files: $deleted"

echo -e "\n=== MEDIUM ==="
git diff $BASE...HEAD --name-only | grep -E "\.(tmp|bak|swp|orig)$" && echo "⚠ Temp files tracked" || true
git status --short | grep "^?" && echo "⚠ Untracked files in working tree" || true
```

**Action**: If any BLOCKER warnings or test failures → stop and fix. Otherwise proceed to detailed scans.

**Tip**: Save this as `scripts/quick_scan.sh` for easy reuse.

---

## Detailed Scans by Category

### 1. Debug Artifacts (BLOCKER)

**Scan:**
```bash
# Find debug statements in source code (excluding tests and __main__)
git diff $BASE...HEAD -- 'src/*.py' | grep -nE "^\+.*(print\(|pdb|breakpoint|console\.log|debugger)" | grep -v "if __name__" || true

# Check for debug imports still present
git diff $BASE...HEAD | grep -nE "^\+import (pdb|pprint|traceback)" | grep -v "^\+    " || true

# Find potentially sensitive log statements (review manually)
git diff $BASE...HEAD | grep -nE "^\+.*logger\.(info|warning|error)" | grep -E "(password|token|key|secret)" || true
```

**Remove:**
- `print()`, `pprint()`, `console.log()` — unless in CLI/test output
- `pdb.set_trace()`, `breakpoint()`, `debugger;` — always remove
- Logging statements with sensitive data (passwords, tokens, API keys)
- `import pdb` when not used elsewhere in the file

**Keep:**
- `logger.debug()` — acceptable for development traces
- `logger.info()` — acceptable for normal operation logging
- `print()` in `if __name__ == "__main__"` blocks
- `print()` in test files for debugging test failures
- Test assertion messages with descriptive output

**Examples:**

❌ **Remove these:**
```python
print(f"User data: {user}")  # Debug statement
result = calculate()
# breakpoint()  # Even commented out
import pdb; pdb.set_trace()  # Leftover debug
logger.info(f"API key: {api_key}")  # Leaking secrets
```

✅ **Keep these:**
```python
logger.debug(f"Processing {len(items)} items")  # Development trace
logger.info("Service started successfully")  # Operational log

if __name__ == "__main__":
    print(f"Result: {main()}")  # CLI output

def test_feature():
    print(f"Debug: state={state}")  # Test debugging
    assert result == expected, f"Expected {expected}, got {result}"
```

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
# Check for hardcoded secrets (excluding proper env var usage)
git diff $BASE...HEAD | grep -iE "^\+.*(password|secret|api_key|token|bearer).*=.*[\"'][^\"']*[\"']" | \
  grep -v -E "(getenv|environ\[|param|arg|description|example|placeholder|test)" || true

# Check for URLs with embedded credentials
git diff $BASE...HEAD | grep -E "^\+.*https?://[^:]+:[^@]+@" || true

# Check for common secret patterns
git diff $BASE...HEAD | grep -iE "^\+.*(private_key|client_secret|aws_secret|ssh_key).*=" || true
```

**Any match** → Stop immediately and remediate:

1. **Move to environment variables:**
```python
# ❌ Bad
api_key = "sk-1234567890abcdef"
db_password = "mysecretpassword"

# ✅ Good
api_key = os.getenv("API_KEY")
db_password = os.getenv("DB_PASSWORD")
```

2. **Add to `env.example` with placeholder:**
```bash
# OpenAI API key for LLM integration
API_KEY=sk-placeholder-get-from-platform

# Database credentials
DB_PASSWORD=secure-password-here
```

3. **Update documentation** if config changed

4. **If secret was committed:** Rotate the secret immediately and consider using `git-filter-repo` to remove from history

**Note**: Even in tests, use fixtures or environment variables rather than hardcoded secrets.

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

**See**: [docstrings.md](docstrings.md) for full spec.

---

### 6. TODOs and FIXMEs (HIGH)

**Scan:**
```bash
# Find all new TODOs/FIXMEs/HACKs
git diff $BASE...HEAD | grep -nE "^\+.*(TODO|FIXME|HACK|XXX)" | sed 's/^/  /'

# Check how many are linked to issues
total=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX)" || echo 0)
linked=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX).*(issue #[0-9]+|https://)" || echo 0)
echo "Total: $total, Linked: $linked, Unlinked: $((total - linked))"
```

**Decision tree:**

| Situation | Estimated Fix Time | Action | Example |
|-----------|-------------------|--------|---------|
| Simple fix | < 10 minutes | **Fix now**, remove TODO | Typo, missing validation |
| Blocks current feature | Any | **Must fix before merge** | Feature incomplete without it |
| Future enhancement | > 30 minutes | **Create issue**, link in TODO | Optimization, nice-to-have |
| Technical debt | Varies | **Create issue**, link in TODO | Refactoring needed |
| Obsolete/done | N/A | **Delete comment** | Old TODO no longer relevant |

**Valid TODOs** (with issue links):
```python
# TODO: Add retry logic with exponential backoff (issue #456)
#   Current implementation fails immediately on network errors

# FIXME: Refactor to use new API after v2.0 release (issue #457)
#   See: https://github.com/org/repo/issues/457

# HACK: Workaround for upstream bug in library v1.2.3 (issue #458)
#   Remove when fixed: https://github.com/upstream/lib/issues/789
```

**Invalid TODOs** (too vague, no links):
```python
# TODO: fix this
# FIXME: make better
# HACK: temporary workaround  # (no explanation or issue)
# XXX: check this later  # (what? why?)
```

**Best practices:**
- Be specific: `# TODO: Add retry logic with exponential backoff (issue #456)`
- Link to context: issue number or URL
- Explain why not done now
- Choose marker: `TODO` (future work), `FIXME` (known issue), `HACK` (workaround), `XXX` (critical)

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
# Extract all getenv/environ calls from new code
git diff $BASE...HEAD | grep -oE 'getenv\("[^"]+"|environ\["[^"]+' | \
  sed 's/getenv("//; s/environ\["//' | tr -d '"' | sort -u > /tmp/new_env_vars.txt

# Check which are missing from env.example
if [ -f env.example ]; then
  while read var; do
    grep -q "^${var}=" env.example || echo "Missing in env.example: $var"
  done < /tmp/new_env_vars.txt
else
  echo "⚠ env.example not found"
fi
```

**Required actions for each new env var:**
1. Add to `env.example` with description and placeholder
2. Document in README or docs if user-facing
3. Add to deployment templates/CI config if needed

**Example `env.example` entry:**
```bash
# OpenAI API key for LLM-based features
# Get from: https://platform.openai.com/api-keys
# Required: Yes (for LLM features), No (for other features)
OPENAI_API_KEY=sk-placeholder-paste-your-key-here

# Database connection string
# Format: postgresql://user:pass@host:port/dbname
# Default: Uses SQLite if not provided
DATABASE_URL=postgresql://localhost:5432/osprey
```

**New dependencies:**
- Cross-check new imports against `pyproject.toml`
- Add with version constraints: `"requests>=2.31.0"  # Brief reason`
- Include in appropriate section: `dependencies`, `test`, or `dev`

---

### 9. Test Coverage (HIGH)

**Missing test files:**
```bash
# Check for new source files without corresponding test files
git diff $BASE...HEAD --name-only --diff-filter=A | grep "^src/.*\.py$" | while read f; do
  base=$(basename "$f" .py)
  # Look for test file with matching name
  test_file=$(find tests/ -name "test_${base}.py" 2>/dev/null | head -1)
  if [ -z "$test_file" ]; then
    # Only warn if it's not __init__.py and not a private module
    if [[ ! "$f" =~ __init__\.py$ ]] && [[ ! "$base" =~ ^_.* ]]; then
      echo "No test file for: $f (expected: tests/.../test_${base}.py)"
    fi
  fi
done
```

**Coverage check:**
```bash
# Overall coverage
pytest tests/ --ignore=tests/e2e \
  --cov=src/osprey \
  --cov-report=term-missing:skip-covered \
  --cov-report=html

# Files below threshold
pytest tests/ --ignore=tests/e2e --cov=src/osprey --cov-report=term-missing:skip-covered | \
  grep "^src/" | \
  awk '$4 < 80 {print "Low coverage:", $1, $4"%"}'

# Open detailed HTML report
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
```

**Coverage for changed files only:**
```bash
# Get list of modified Python files
changed_files=$(git diff $BASE...HEAD --name-only | grep "^src/.*\.py$" | tr '\n' ',' | sed 's/,$//')

# Run coverage on just those files
if [ -n "$changed_files" ]; then
  pytest tests/ --ignore=tests/e2e \
    --cov=src/osprey \
    --cov-report=term-missing \
    | grep -E "$(echo $changed_files | tr ',' '|')"
fi
```

**Minimum test requirements:**

| Code Change | Required Tests | Example |
|-------------|----------------|---------|
| New function | • 1 happy path<br>• 2-3 edge cases<br>• 1 error case | `test_func_success()`,<br>`test_func_empty_input()`,<br>`test_func_invalid_type()` |
| Modified function | • 1 regression test<br>• Tests for new behavior | `test_func_handles_new_param()` |
| New class | • `__init__` variations<br>• Public methods<br>• Error handling | `test_class_init()`,<br>`test_class_main_method()`,<br>`test_class_invalid_state()` |
| Bug fix | • Test that reproduces bug<br>• Test that verifies fix | `test_issue_123_edge_case()` |

**Use `# pragma: no cover` for:**
- `if __name__ == "__main__":` blocks
- Impossible/defensive error cases
- Platform-specific code you can't test
- Type checking blocks: `if TYPE_CHECKING:`

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

**Check for deleted items still referenced:**

```bash
# Deleted modules
git diff $BASE...HEAD --diff-filter=D --name-only | grep "\.py$" | while read f; do
  module=$(echo "$f" | sed 's|^src/||; s|/__init__\.py$||; s|\.py$||; s|/|.|g')
  git grep -q "import.*\b$module\b\|from $module" src/ tests/ && echo "⚠ Orphaned: $module"
done

# Deleted functions/classes
git diff $BASE...HEAD --diff-filter=D | grep -E "^\-(def |class )" | \
  sed 's/.*\(def\|class\) //; s/(.*//; s/:.*//; s/^-//' | sort -u | while read name; do
    git grep -q "\b$name\b" src/ tests/ && echo "⚠ Orphaned: $name"
done
```

**For renames, verify old name is gone:**
```bash
OLD="old_function"  # Replace with actual name
git grep -n "$OLD" src/ tests/ docs/ | grep -v CHANGELOG
# Should return ZERO results
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

### 15. Pre-commit Hooks (RECOMMENDED)

Automate many of these checks using pre-commit hooks:

**Setup:**
```bash
pip install pre-commit
```

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements  # Catches pdb, breakpoint, etc.

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Install hooks:**
```bash
pre-commit install
```

**Run manually:**
```bash
pre-commit run --all-files  # Check all files
pre-commit run --files path/to/file.py  # Check specific file
```

**Skip hooks (when necessary):**
```bash
git commit --no-verify  # Skip all hooks (use sparingly!)
SKIP=black,ruff git commit  # Skip specific hooks
```

**Benefits:**
- ✅ Automatic formatting on every commit
- ✅ Catches debug statements before commit
- ✅ Validates YAML/JSON syntax
- ✅ Prevents large files from being added
- ✅ Fast feedback loop (< 1 second for most checks)

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
  echo "See: docs/workflows/pre-merge-cleanup.md"
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

Before requesting merge, verify all items:

### BLOCKER (Must Fix)
- [ ] **No debug code**: Run `git diff main...HEAD -- 'src/*.py' | grep -E "print\(|pdb|breakpoint"`
  - Result: No matches (or only in `__main__` blocks)
- [ ] **No commented code**: Run `git diff main...HEAD | grep "# *(def |class |import )"`
  - Result: No matches
- [ ] **No hardcoded secrets**: Run `git diff main...HEAD | grep -iE "(password|api_key|token).*=.*[\"']"`
  - Result: Only `getenv()` or `environ[]` usage
- [ ] **All tests pass**: Run `pytest tests/ --ignore=tests/e2e -v`
  - Result: All tests pass (green)

### CRITICAL (Required for Merge)
- [ ] **CHANGELOG.md updated**: Run `git diff main...HEAD CHANGELOG.md`
  - Has entry under `## [Unreleased]` in appropriate section (Added/Changed/Fixed/Removed)
  - Breaking changes marked with `**BREAKING**:`
- [ ] **Docstrings complete**: Check all new public functions/classes
  - One-line summary + detailed description
  - Args, Returns, and Raises sections present
  - See [docstrings.md](docstrings.md) for format
- [ ] **Type hints present**: Run `ruff check --select ANN src/`
  - All function arguments have types
  - All functions have return type (or `-> None`)

### HIGH (Manually Verify)
- [ ] **TODOs have issue links**: Run `git diff main...HEAD | grep "TODO"`
  - All have format: `# TODO: Description (issue #NNN)`
  - Quick fixes completed and TODO removed
- [ ] **Refactorings complete**: If renamed/moved code
  - `git grep OLD_NAME src/ tests/ docs/` returns zero results
  - CHANGELOG has migration notes if breaking
- [ ] **Config synchronized**:
  - [ ] New env vars in `env.example` with descriptions
  - [ ] New dependencies in `pyproject.toml` with versions
  - [ ] New config keys in template config files
- [ ] **Test coverage ≥80%**: Run `pytest --cov=src/osprey --cov-report=term`
  - New code has tests (happy path + edge cases + errors)
  - Regression tests for bug fixes

### MEDIUM (Quality Checks)
- [ ] **No orphaned imports**: Run `ruff check --select F401,I src/ tests/`
  - No unused imports
  - Imports properly sorted
- [ ] **Documentation builds**: Run `cd docs && make clean && make html`
  - No errors (warnings OK except critical ones)
  - New modules documented if public API
- [ ] **Code formatted**: Run `black --check src/ tests/`
  - Already formatted (or run `black src/ tests/`)
- [ ] **Ruff clean**: Run `ruff check src/ tests/`
  - No linting errors (or run `ruff check --fix`)
- [ ] **No temp files**: Run `git status --short`
  - No `*.tmp`, `*.bak`, `*.swp`, `.DS_Store` in tracked files

### RECOMMENDED (Best Practices)
- [ ] **Pre-commit hooks**: Run `pre-commit install` and `pre-commit run --all-files`
- [ ] **AI code review**: If AI-generated code, follow [ai-code-review.md](ai-code-review.md)
- [ ] **Commit organization**: Follow [commit-organization.md](commit-organization.md)
  - Atomic commits (one logical change per commit)
  - Good commit messages (imperative, descriptive)

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

## Troubleshooting

**"Tests pass locally but fail in CI"**
- Check Python version: `python --version`
- Install test dependencies: `pip install -e ".[test]"`
- Run in random order: `pytest tests/ --randomly`
- Check for uncommitted files affecting tests

**"Grep returns no results"**
- Often expected (no issues found)
- Use `|| true` suffix to prevent script failure

**"Script shows different results"**
- Check base branch: `./scripts/premerge_check.sh origin/main`
- Stash uncommitted changes: `git stash`
- Note: `...` uses merge base, `..` uses direct comparison

**"False positives in secret detection"**
- Add inline comment: `token = "fake"  # nosec: test fixture`
- Update scan pattern to exclude test/example patterns

**"Coverage below 80%"**
- View HTML report: `pytest --cov=src/osprey --cov-report=html`
- Focus on new public APIs and error handling
- Use `# pragma: no cover` for impossible states

**"Formatting conflicts"**
- Run formatters in order: `black` first, then `ruff check --fix`
- Check `pyproject.toml` for custom config

---

## See Also

- [ai-code-review.md](ai-code-review.md) — Critical review and refactoring of AI-generated code
- [commit-organization.md](commit-organization.md) — Organizing atomic commits
- [docstrings.md](docstrings.md) — Docstring specification
- [release-workflow.md](release-workflow.md) — Release preparation
- [comments.md](comments.md) — When and how to comment code
