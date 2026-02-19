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

# Run fast tests only (stop on first failure for speed)
echo "→ Running fast unit tests..."
uv run pytest tests/ --ignore=tests/e2e -m "not langgraph" -x --tb=line -q

echo ""
echo "✅ Quick checks passed! Safe to commit."
echo ""
echo "💡 Tip: Run './scripts/ci_check.sh' before pushing for full validation"
