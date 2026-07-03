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

# The behavioral/visual Playwright suites under tests/ skip cleanly when chromium
# isn't installed. A silent skip in the pytest summary below is easy to miss —
# run the exact same chromium-launch check those suites' chromium_browser fixture
# uses, and say so loudly instead.
echo "→ Checking chromium availability for browser-based theming tests..."
if uv run python - <<'PYEOF' >/dev/null 2>&1
from playwright.sync_api import sync_playwright

pw = sync_playwright().start()
try:
    browser = pw.chromium.launch(headless=True)
    browser.close()
finally:
    pw.stop()
PYEOF
then
    echo "✅ Chromium available — browser-based theming tests will run for real"
else
    echo "⚠️  Browser-based theming tests NOT verified on this platform (no chromium) — CI enforces them"
fi

# Run fast tests only (stop on first failure for speed)
echo "→ Running fast unit tests..."
uv run pytest tests/ --ignore=tests/e2e -x --tb=line -q

echo ""
echo "✅ Quick checks passed! Safe to commit."
echo ""
echo "💡 Tip: Run './scripts/ci_check.sh' before pushing for full validation"
