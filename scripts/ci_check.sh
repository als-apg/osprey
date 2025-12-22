#!/bin/bash
# Full CI checks - Replicates GitHub Actions workflow locally
# Run this before pushing to catch issues early and save CI minutes

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Continuing with system Python..."
fi

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
if ! ruff check src/ tests/ --output-format=github; then
    FAILED_CHECKS+=("ruff-linting")
    echo "❌ Ruff linting failed"
else
    echo "✅ Ruff linting passed"
fi
echo ""

echo "→ Running ruff (formatting)..."
if ! ruff format --check src/ tests/; then
    FAILED_CHECKS+=("ruff-formatting")
    echo "❌ Ruff formatting failed"
    echo "💡 Run 'ruff format src/ tests/' to fix"
else
    echo "✅ Ruff formatting passed"
fi
echo ""

echo "→ Running mypy (type checking)..."
if ! mypy src/ --no-error-summary; then
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

echo "→ Running pytest with coverage..."
if ! pytest tests/ --ignore=tests/e2e -v --tb=short --cov=src/osprey --cov-report=xml --cov-report=term; then
    FAILED_CHECKS+=("pytest")
    echo "❌ Tests failed"
else
    echo "✅ Tests passed"
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
if ! python -m build --quiet; then
    FAILED_CHECKS+=("package-build")
    echo "❌ Package build failed"
else
    echo "✅ Package build passed"
fi
echo ""

echo "→ Checking package with twine..."
if ! twine check dist/* 2>&1 | grep -q "PASSED"; then
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
    echo "   - Run 'ruff format src/ tests/' to fix formatting"
    echo "   - Run 'ruff check src/ tests/ --fix' to auto-fix linting"
    echo "   - Check test output above for specific failures"
    echo ""
    exit 1
fi


