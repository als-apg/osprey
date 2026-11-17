#!/usr/bin/env bash
# Fetch the pinned neosemantics (n10s) jar and hand its path to the job's later
# steps as OSPREY_TEST_N10S_JAR.
#
# Fetched by the workflow, not by the test fixture: a download failure in the
# fixture would SKIP the graph paradigm, while here it reds the step. The URL is
# the one tests/_graphdb_container.py pins.
#
# Usage: scripts/ci/fetch_n10s_jar.sh
#   RUNNER_TEMP  where the jar lands (default: $TMPDIR, then /tmp).
#   GITHUB_ENV   receives OSPREY_TEST_N10S_JAR; outside Actions it is printed as an export.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
url="$(uv run python -c 'from tests._graphdb_container import N10S_JAR_URL; print(N10S_JAR_URL)')"
jar="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/$(basename "$url")"
rc=0
curl -fsSL --retry 3 --retry-all-errors --max-time 300 -o "$jar" "$url" || rc=$?
if [ "$rc" -ne 0 ]; then
  # A download outage is not a test failure: say so, and what to do about it.
  echo "::error title=n10s jar download failed::curl exit $rc fetching $url (3 retries)." \
    "The graph suites need this jar, so the job stops here rather than skip them." \
    "Re-run the job; the jar is a github.com release download, so if it keeps failing" \
    "check https://www.githubstatus.com." >&2
  exit "$rc"
fi
ls -l "$jar"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "OSPREY_TEST_N10S_JAR=$jar" >> "$GITHUB_ENV"
else
  echo "export OSPREY_TEST_N10S_JAR=$jar"
fi
