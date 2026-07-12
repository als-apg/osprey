#!/bin/bash
# Full CI checks - Replicates GitHub Actions workflow locally
# Run this before pushing to catch issues early and save CI minutes

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

echo "🔍 Running full CI checks locally..."
echo "===================================="
echo ""

# Track failures
FAILED_CHECKS=()

# 1. Lint checks (matches .github/workflows/ci.yml lint job)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 1/4: Linting"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ Running ruff (linting)..."
if ! uv run ruff check src/ tests/ --output-format=github; then
    FAILED_CHECKS+=("ruff-linting")
    echo "❌ Ruff linting failed"
else
    echo "✅ Ruff linting passed"
fi
echo ""

echo "→ Running ruff (formatting)..."
if ! uv run ruff format --check src/ tests/; then
    FAILED_CHECKS+=("ruff-formatting")
    echo "❌ Ruff formatting failed"
    echo "💡 Run 'ruff format src/ tests/' to fix"
else
    echo "✅ Ruff formatting passed"
fi
echo ""

echo "→ Running mypy (type checking)..."
if ! uv run mypy src/ --no-error-summary; then
    echo "⚠️  Mypy found type issues (not blocking)"
else
    echo "✅ Mypy passed"
fi
echo ""

# 2. Tests (matches .github/workflows/ci.yml test job)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Step 2/4: Unit Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Prune stale bytecode that can resurrect a deleted package as a PEP 420
# namespace package locally (a dir left holding only __pycache__ still imports).
# Drop the caches, then the now-empty dirs they kept alive. CI is unaffected —
# a fresh checkout has no __pycache__ — but this keeps local runs matching CI.
echo "→ Pruning stale bytecode caches..."
find src/osprey -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find src/osprey -type d -empty -delete 2>/dev/null || true
echo ""

# The behavioral/visual Playwright suites (tests/interfaces/web_terminal/test_panels_browser.py,
# tests/interfaces/design_system/test_behavioral.py) skip cleanly when chromium isn't installed.
# A silent skip in the pytest summary below is easy to miss — run the exact same
# chromium-launch check those suites' chromium_browser fixture uses, and say so loudly.
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
echo ""

echo "→ Running pytest with coverage..."
if ! uv run pytest tests/ --ignore=tests/e2e -v --tb=short --cov=src/osprey --cov-report=xml --cov-report=term; then
    FAILED_CHECKS+=("pytest")
    echo "❌ Tests failed"
else
    echo "✅ Tests passed"
fi
echo ""

# Frontend JS checks (matches .github/workflows/ci.yml frontend-js job).
# Optional locally: like the chromium check above, we *notice* when the Node
# toolchain is absent rather than hard-failing — CI enforces it on every push.
echo "→ Checking Node/npm availability for frontend JS checks..."
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "✅ Node/npm available — running frontend JS checks"
    echo ""
    echo "→ Installing JS dependencies (npm ci)..."
    if ! npm ci; then
        FAILED_CHECKS+=("npm-ci")
        echo "❌ npm ci failed"
    else
        echo "✅ JS dependencies installed"
        echo ""
        echo "→ Running typecheck (npm run typecheck)..."
        if ! npm run typecheck; then
            FAILED_CHECKS+=("js-typecheck")
            echo "❌ JS typecheck failed"
        else
            echo "✅ JS typecheck passed"
        fi
        echo ""
        echo "→ Running lint (npm run lint)..."
        if ! npm run lint; then
            FAILED_CHECKS+=("js-lint")
            echo "❌ JS lint failed"
        else
            echo "✅ JS lint passed"
        fi
        echo ""
        echo "→ Running JS tests (npm run test:js)..."
        if ! npm run test:js; then
            FAILED_CHECKS+=("js-tests")
            echo "❌ JS tests failed"
        else
            echo "✅ JS tests passed"
        fi
    fi
else
    echo "⚠️  Frontend JS checks NOT verified on this platform (no node/npm) — CI enforces them"
fi
echo ""

# 3. Documentation build (matches .github/workflows/ci.yml docs job)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Step 3/4: Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ Building documentation..."
cd docs
if ! make clean > /dev/null 2>&1 && make html; then
    FAILED_CHECKS+=("docs-build")
    echo "❌ Documentation build failed"
else
    echo "✅ Documentation build passed"
fi

echo ""
echo "→ Checking for broken links..."
if ! make linkcheck 2>&1 | grep -q "build succeeded"; then
    echo "⚠️  Link check found issues (not blocking)"
else
    echo "✅ Link check passed"
fi
cd ..
echo ""

# 4. Package build (matches .github/workflows/ci.yml package job)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 4/4: Package Build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ Building package..."
if ! uv build --quiet; then
    FAILED_CHECKS+=("package-build")
    echo "❌ Package build failed"
else
    echo "✅ Package build passed"
fi
echo ""

echo "→ Checking package with twine..."
if ! uvx twine check dist/* 2>&1 | grep -q "PASSED"; then
    FAILED_CHECKS+=("twine-check")
    echo "❌ Twine check failed"
else
    echo "✅ Twine check passed"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo "✅ All CI checks passed locally!"
    echo ""
    echo "🚀 Your code is ready to push. CI should pass on GitHub."
    echo ""
    exit 0
else
    echo "❌ ${#FAILED_CHECKS[@]} check(s) failed:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "   - $check"
    done
    echo ""
    echo "Please fix the issues above before pushing."
    echo ""
    echo "💡 Tips:"
    echo "   - Run 'uv run ruff format src/ tests/' to fix formatting"
    echo "   - Run 'uv run ruff check src/ tests/ --fix' to auto-fix linting"
    echo "   - Check test output above for specific failures"
    echo ""
    exit 1
fi
