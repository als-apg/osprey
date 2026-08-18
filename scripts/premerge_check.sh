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

# --maxfail=1 (same option -x aliases): fast-fail under xdist is approximate —
# the other workers still finish their in-flight tests, so a short-circuited
# run can report more than one failure.
# -m "not pty": the real-terminal suite times its own repaints and cannot share
# the machine with four workers. This script is the fast pre-merge sweep, so it
# does not re-run the suite serially; ci_check.sh does, and so does CI.
if ! uv run pytest tests/ --ignore=tests/e2e -m "not pty" -n 4 --dist loadgroup --maxfail=1 --tb=no -q >/dev/null 2>&1; then
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
# The `|| true` is inside the group so a no-match grep (exit 1) can't abort the
# substitution under `set -eo pipefail`; `xargs` strips the padding BSD `wc -l`
# emits, leaving a bare integer for the comparisons below.
new_funcs=$({ git diff $BASE...HEAD | grep -E "^\+def [a-z_]" | grep -v "^\+    " || true; } | wc -l | xargs)
typed_funcs=$({ git diff $BASE...HEAD | grep -E "^\+def [a-z_][^(]*\([^)]*\) *->" | grep -v "^\+    " || true; } | wc -l | xargs)
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
todos=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX)" || true)
linked=$(git diff $BASE...HEAD | grep -cE "^\+.*(TODO|FIXME|HACK|XXX).*(issue #[0-9]+|https://)" || true)
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

# MANUAL container checks
#
# tests/e2e/ is excluded from every automated lane in this script and in CI's
# `test` job (`--ignore=tests/e2e`): those suites build real images and deploy
# real containers, which is minutes-to-tens-of-minutes of wall clock and needs
# a Docker daemon. So they are a HUMAN pre-merge step, and the only place a
# human reliably looks before opening a PR is right here.
#
# Advisory, never blocking, and only printed when the diff actually touches the
# scan stack -- a checklist that appears on every unrelated PR is a checklist
# people learn to skip.
echo -e "\n=== MANUAL (containers) ==="
scan_stack_touched=$(git diff $BASE...HEAD --name-only | grep -E \
  "^(src/osprey/services/bluesky_bridge/|src/osprey/interfaces/bluesky_web/|src/osprey/templates/services/bluesky|src/osprey/mcp_server/bluesky/|tests/e2e/test_bluesky|tests/e2e/test_tiled_roundtrip|tests/e2e/test_grid_scan_roundtrip|tests/e2e/_orm_stack|tests/e2e/_queue_drive)" || true)
if [ -n "$scan_stack_touched" ]; then
  echo "⚠ This diff touches the Bluesky scan stack — run the container e2e by hand"
  echo "  (needs Docker; none of these run in this script). In this order:"
  echo ""
  echo "  # 1. the whole queue stack: capability, arming, serial drain, session"
  echo "  #    plans, abort, bridge restart, mock flip, network isolation."
  echo "  #    Runs on CI in the bluesky-queue-e2e lane; run it here to iterate"
  echo "  #    locally before the PR rather than to cover a gap."
  echo "  uv run pytest tests/e2e/test_bluesky_queue_e2e.py -v"
  echo ""
  echo "  # 2. minimum deploy (bridge + RE manager + Redis, browse-only)"
  echo "  uv run pytest tests/e2e/test_bluesky_deploy.py -v"
  echo ""
  echo "  # 3. Tiled durability across a bridge restart (acceptance criterion 11)"
  echo "  uv run pytest tests/e2e/test_tiled_roundtrip.py -v"
  echo ""
  echo "  # 4. the sidecar hop: a scan driven entirely through the panels relay"
  echo "  uv run pytest tests/e2e/test_bluesky_web_deploy.py -v"
  echo ""
  echo "  # 5. layered plan catalog served by a deployed container (browse-only)"
  echo "  uv run pytest tests/e2e/test_bluesky_catalog_e2e.py -v"
  echo ""
  echo "  # 6. authoring sandbox: an unvalidated plan reaches no hardware"
  echo "  uv run pytest tests/e2e/test_bluesky_sandbox_escape_e2e.py -v"
  echo ""
  echo "  # 7. grid_scan over the real VA. Runs on CI in the orm-roundtrip-e2e"
  echo "  #    lane, sequentially after test_orm_roundtrip.py; run it here to"
  echo "  #    iterate locally before the PR rather than to cover a gap."
  echo "  uv run pytest tests/e2e/test_grid_scan_roundtrip.py -v"
  echo ""
  echo "  Never 'pytest -m e2e' — the marker selection breaks registry isolation."
  echo "  Each module tears its own stack down via 'osprey down'."
  WARNINGS=$((WARNINGS + 1))
else
  echo "✓ No scan-stack changes — container e2e not required for this diff"
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
