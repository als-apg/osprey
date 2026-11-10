#!/usr/bin/env bash
# Remove what each named compose project may have left on the runner: its
# containers and networks, its agent record and that record's ClickHouse store,
# and every volume under the project's own name prefix. Exact names, never a prune.
#
# ci.yml's "Clean up any stranded stack resources" steps run it with `if: always()`:
# a deploy's own teardown does not run when the job is killed, and a self-hosted,
# reused or `act` runner keeps the stack. Every removal is `|| true`, so this
# exits 0 whatever is (or is not) there and can never fail the job.
#
# Usage: scripts/ci/cleanup_stacks.sh PROJECT [PROJECT ...]
for project in "$@"; do
  docker compose -p "$project" down || true
  docker rm -f "${project}-agent-record" || true
  docker rm -f "${project}-agent-record-clickhouse" || true
  docker volume ls -q --filter "name=^${project}_" \
    | xargs -r docker volume rm -f || true
done
exit 0
