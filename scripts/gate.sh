#!/bin/bash
# The merge gate: lint, types, the parallel sweep, each serial step, JS tests.
#
#   scripts/gate.sh full                    the whole unit suite
#   scripts/gate.sh fast <path or node id>… the same steps over the named tests
#
# Stops at the first failing step and exits with its code. Every pytest line
# below is pinned by tests/deployment/test_ci_workflow_wiring.py to the lane
# table there: the sweep's marker expression is exactly the unit lane's, and
# each marker the sweep deselects has a serial line of its own in both modes.
# Nothing a caller passes reaches a -m.

set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    echo "usage: scripts/gate.sh full" >&2
    echo "       scripts/gate.sh fast <test path or node id>..." >&2
    exit 2
}

mode="${1:-}"
case "$mode" in
    full)
        [ "$#" -eq 1 ] || usage
        shift
        ;;
    fast)
        [ "$#" -ge 2 ] || usage
        shift
        ;;
    *)
        usage
        ;;
esac

step() {
    echo "gate: $1"
}

# A serial step in fast mode may find none of its marker among the named tests;
# pytest's exit 5 (no tests collected) then means there was nothing to run.
# In full mode the step names its own directory, so exit 5 there is a failure.
serial() {
    local code=0
    "$@" || code=$?
    if [ "$code" -eq 5 ] && [ "$mode" = fast ]; then
        return 0
    fi
    return "$code"
}

step "ruff check"
uv run ruff check .

step "ruff format"
uv run ruff format --check .

step "mypy"
uv run python scripts/mypy_gate.py

# The sweep deselects every marker whose tests cannot share the machine with
# xdist workers; each of those markers runs alone in a serial step below.
# test_scale.py is a wall-clock guard held to an unloaded workstation.
step "pytest sweep"
if [ "$mode" = full ]; then
    uv run python scripts/run_bounded.py 5400 -- uv run pytest tests/ --ignore=tests/e2e --ignore=tests/services/channel_finder/graph_index/test_scale.py -m "not pty and not wallclock" -n auto --dist loadgroup -q --tb=short --timeout=600 --timeout-method=signal
else
    uv run python scripts/run_bounded.py 1800 -- uv run pytest "$@" --ignore=tests/services/channel_finder/graph_index/test_scale.py -m "not pty and not wallclock" -n auto --dist loadgroup -q --tb=short --timeout=600 --timeout-method=signal
fi

# The real-terminal suite asserts on repaint cadence, so it runs with no
# workers beside it.
step "pytest pty (serial)"
if [ "$mode" = full ]; then
    serial uv run python scripts/run_bounded.py 600 -- uv run pytest tests/pty -m pty -q --tb=short
else
    serial uv run python scripts/run_bounded.py 600 -- uv run pytest "$@" -m pty -q --tb=short
fi

# A wall-clock budget timed beside loaded workers measures the workers.
step "pytest wallclock (serial)"
if [ "$mode" = full ]; then
    serial uv run python scripts/run_bounded.py 900 -- uv run pytest tests/integration -m wallclock -q --tb=short
else
    serial uv run python scripts/run_bounded.py 900 -- uv run pytest "$@" -m wallclock -q --tb=short
fi

step "js tests"
npm run test:js
