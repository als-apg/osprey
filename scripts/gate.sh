#!/bin/bash
# The merge gate: every CI step that needs no secret or Docker.
#
#   scripts/gate.sh full [--base <ref>]                     the whole suite
#   scripts/gate.sh fast [--base <ref>] <test path or id>…  the same steps over the named tests
#
# Every step is a CI step that needs no secret or Docker, and each is pinned by
# tests/deployment/test_ci_workflow_wiring.py to its ci.yml line: the lint and
# type checks, the changelog and plugin-version checks against <ref> (default
# origin/main), the pytest lanes, the JS checks and tests, and the docs build.
# The sweep's marker expression is exactly the unit lane's, and each marker the
# sweep deselects has a serial line of its own in both modes. Nothing a caller
# passes reaches a -m. Fast mode sends each named test to its own runner:
# Python files, node ids and directories to pytest, *.test.js and *.test.mjs
# files to vitest. The gate never fetches: <ref> must already resolve.
#
# Stops at the first failing step and exits with its code.

set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    echo "usage: scripts/gate.sh full [--base <ref>]" >&2
    echo "       scripts/gate.sh fast [--base <ref>] <test path or node id>..." >&2
    exit 2
}

mode="${1:-}"
case "$mode" in
    full | fast) shift ;;
    *) usage ;;
esac

base=origin/main
if [ "${1:-}" = --base ]; then
    [ "$#" -ge 2 ] || usage
    base="$2"
    shift 2
fi

case "$mode" in
    full) [ "$#" -eq 0 ] || usage ;;
    fast) [ "$#" -ge 1 ] || usage ;;
esac

if ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
    echo "gate: base $base does not resolve; fetch it or pass --base" >&2
    exit 2
fi

# The suffixes vitest takes are exactly vitest.config.js's include set. A
# directory goes to pytest only: vitest reads its arguments as file filters.
py=()
js=()
if [ "$mode" = fast ]; then
    for arg in "$@"; do
        path="${arg%%::*}"
        case "$path" in
            *.test.js | *.test.mjs) js+=("$arg") ;;
            *.py) py+=("$arg") ;;
            *)
                if [ -d "$path" ]; then
                    py+=("$arg")
                else
                    echo "gate: $arg belongs to no test runner" >&2
                    exit 2
                fi
                ;;
        esac
    done
    set -- ${py[@]+"${py[@]}"}
fi
named_python_tests=$#

step() {
    echo "gate: $1"
}

# True when fast mode named no Python test, so a pytest step has nothing to run.
no_python_test() {
    if [ "$mode" = fast ] && [ "$named_python_tests" -eq 0 ]; then
        echo "gate: $1 (no Python test named)"
        return 0
    fi
    return 1
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

# Any `uv run` re-locks a stale lockfile, so the lockfile is checked first.
step "lockfile"
uv lock --check

step "ruff (linting)"
uv run ruff check .

step "ruff (formatting)"
uv run ruff format --check .

step "config keys"
uv run python scripts/check_config_keys.py --back-test

step "changelog fragments"
python3 scripts/changelog_fragments.py check --base "$base"

step "plugin version"
python3 scripts/plugin_version.py check --base "$base"

step "mypy"
uv run python scripts/mypy_gate.py

# The sweep deselects every marker whose tests cannot share the machine with
# xdist workers; each of those markers runs alone in a serial step below.
# test_scale.py is a wall-clock guard held to an unloaded workstation.
if ! no_python_test "pytest sweep"; then
    step "pytest sweep"
    if [ "$mode" = full ]; then
        uv run python scripts/run_bounded.py 5400 -- uv run pytest tests/ --ignore=tests/e2e --ignore=tests/services/channel_finder/graph_index/test_scale.py -m "not pty and not wallclock" -n auto --dist loadgroup -q --tb=short --timeout=600 --timeout-method=signal
    else
        uv run python scripts/run_bounded.py 1800 -- uv run pytest "$@" --ignore=tests/services/channel_finder/graph_index/test_scale.py -m "not pty and not wallclock" -n auto --dist loadgroup -q --tb=short --timeout=600 --timeout-method=signal
    fi
fi

# The real-terminal suite asserts on repaint cadence, so it runs with no
# workers beside it.
if ! no_python_test "pytest pty (serial)"; then
    step "pytest pty (serial)"
    if [ "$mode" = full ]; then
        serial uv run python scripts/run_bounded.py 600 -- uv run pytest tests/pty -m pty -q --tb=short
    else
        serial uv run python scripts/run_bounded.py 600 -- uv run pytest "$@" -m pty -q --tb=short
    fi
fi

# A wall-clock budget timed beside loaded workers measures the workers.
if ! no_python_test "pytest wallclock (serial)"; then
    step "pytest wallclock (serial)"
    if [ "$mode" = full ]; then
        serial uv run python scripts/run_bounded.py 900 -- uv run pytest tests/integration -m wallclock -q --tb=short
    else
        serial uv run python scripts/run_bounded.py 900 -- uv run pytest "$@" -m wallclock -q --tb=short
    fi
fi

step "js typecheck"
npm run typecheck

step "js lint"
npm run lint

if [ "$mode" = full ]; then
    step "js tests"
    npm run test:js
elif [ "${#js[@]}" -eq 0 ]; then
    echo "gate: js tests (no JS test named)"
else
    step "js tests"
    npm run test:js -- "${js[@]}"
fi

# The docs build needs graphviz for its diagrams; a missing `dot` fails the
# gate, as a missing package would. The build installs the docs extra itself.
step "docs"
command -v dot >/dev/null || {
    echo "gate: the docs build needs graphviz (dot) on PATH" >&2
    exit 1
}
(cd docs && uv run --extra docs make html SPHINXOPTS="-W --keep-going")
