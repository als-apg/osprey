#!/bin/bash
set -eo pipefail
BASE="${1:-main}"
ERRORS=0
WARNINGS=0

echo "🔍 Pre-merge scan against $BASE"
echo "========================================"

# BLOCKER checks
echo -e "\n=== BLOCKERS ==="

# Debug code (exclude __main__ blocks which are legitimate)
debug_in_src=$(git diff $BASE...HEAD -- 'src/*.py' | grep -E "^\+.*(print\(|pdb\.|breakpoint\()" | grep -v "if __name__" || true)
if [ -n "$debug_in_src" ]; then
  echo "✗ Debug code found in src/"
  echo "$debug_in_src" | head -3
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No debug code in src/"
fi

# Commented code (more comprehensive patterns, exclude doc comments)
commented=$(git diff $BASE...HEAD | grep -E "^\+.*# *(def |class |import |return |if |for |while )" | grep -v "^\+.*#.*:" || true)
if [ -n "$commented" ]; then
  echo "✗ Possible commented code found"
  echo "$commented" | head -3
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No obvious commented code"
fi

# Hardcoded secrets (fixed pipe - now correctly filters out getenv/environ)
secrets=$(git diff $BASE...HEAD | grep -iE "^\+.*(password|api_key|secret|token).*=.*[\"'][^\"']*[\"']" | grep -v -E "(getenv|environ\[)" || true)
if [ -n "$secrets" ]; then
  echo "✗ Possible hardcoded secrets"
  echo "$secrets" | head -3
  ERRORS=$((ERRORS + 1))
else
  echo "✓ No obvious secrets"
fi

if ! uv run pytest tests/ --ignore=tests/e2e -x --tb=no -q >/dev/null 2>&1; then
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

# Type hints (fixed: only count functions, not classes; exclude methods)
new_funcs=$(git diff $BASE...HEAD | grep -E "^\+def [a-z_]" | grep -v "^\+    " | wc -l || echo 0)
typed_funcs=$(git diff $BASE...HEAD | grep -E "^\+def [a-z_][^(]*\([^)]*\) *->" | grep -v "^\+    " | wc -l || echo 0)
if [ "$new_funcs" -gt 0 ]; then
  echo "  New top-level functions: $new_funcs"
  echo "  With return type hints: $typed_funcs"
  if [ "$typed_funcs" -lt "$new_funcs" ]; then
    echo "⚠ Some functions missing return type hints (not blocking)"
    WARNINGS=$((WARNINGS + 1))
  fi
fi

# HIGH checks
echo -e "\n=== HIGH ==="
todos=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX)" || echo 0)
linked=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX).*(issue #[0-9]+|https://)" || echo 0)
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
if uv run ruff format --check src/ tests/ >/dev/null 2>&1; then
  echo "✓ Ruff formatted"
else
  echo "⚠ Ruff formatting needed"
  WARNINGS=$((WARNINGS + 1))
fi

if uv run ruff check src/ tests/ --quiet >/dev/null 2>&1; then
  echo "✓ Ruff clean"
else
  echo "⚠ Ruff issues found"
  WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo -e "\n========================================"
if [ $ERRORS -eq 0 ]; then
  echo "✅ Automated checks passed ($WARNINGS warnings)"
  echo ""
  echo "Manual verification recommended:"
  echo "  • Review with: @src/osprey/assist/tasks/ai-code-review/instructions.md (if AI-generated)"
  echo "  • Pre-merge cleanup: @src/osprey/assist/tasks/pre-merge-cleanup/instructions.md"
  echo "  • Check docstrings, test coverage, and config sync"
  exit 0
else
  echo "❌ Found $ERRORS blocking issues ($WARNINGS warnings)"
  echo ""
  echo "See: src/osprey/assist/tasks/pre-merge-cleanup/instructions.md for detailed guidance"
  exit 1
fi
