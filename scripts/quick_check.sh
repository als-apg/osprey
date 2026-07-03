#!/bin/bash
# Quick pre-commit checks - Run before every commit (< 30 seconds)
# This catches most common issues without running the full test suite

set -e

echo "🚀 Quick pre-commit checks..."
echo "================================"

# Auto-fix formatting issues
echo "→ Auto-fixing code style..."
uv run ruff check src/ tests/ --fix --quiet || true
uv run ruff format src/ tests/ --quiet

# Prune stale bytecode. After deleting a package's .py files, git leaves the
# now-empty dir behind if untracked __pycache__/*.pyc remain — and Python imports
# any non-empty dir as a PEP 420 namespace package, so "package removed" tests
# fail locally only. Drop the caches, then the dirs they kept alive.
echo "→ Pruning stale bytecode caches..."
find src/osprey -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find src/osprey -type d -empty -delete 2>/dev/null || true

# The behavioral/visual Playwright suites (test_panels_browser.py, test_behavioral.py,
# and the visual suite) are @pytest.mark.slow and excluded below to keep this check's
# documented ~30s contract — they run in ci_check.sh and CI instead, chromium or not.
echo "ℹ️  Browser-based theming tests are slow-marked; they run in ci_check.sh and CI, not in this fast pre-commit check."

# Run fast tests only (stop on first failure for speed)
echo "→ Running fast unit tests..."
uv run pytest tests/ --ignore=tests/e2e -m "not slow" -x --tb=line -q

echo ""
echo "✅ Quick checks passed! Safe to commit."
echo ""
echo "💡 Tip: Run './scripts/ci_check.sh' before pushing for full validation"
