#!/usr/bin/env bash
# Remove what each named compose project may have left on the runner: its
# containers and networks, its agent record and that record's ClickHouse store,
# and every volume under the project's own name prefix. Exact names or a
# project's own prefix, never a prune.
#
# ci.yml's stranded-resource cleanup steps run it with `if: always()`: a
# deploy's own teardown does not run when the job is killed, and a self-hosted,
# reused or `act` runner keeps the stack. Every removal is `|| true`, so this
# exits 0 whatever is (or is not) there and can never fail the job.
#
# Usage: [ENGINE=podman] scripts/ci/cleanup_stacks.sh [--sweep] [--image REF]... PROJECT...
#   ENGINE       the container CLI to run (default: docker).
#   --sweep      also remove every container and volume named `PROJECT-*`, which
#                no compose project owns (a test's own `run` helpers).
#   --image REF  after the projects, remove the images `ENGINE images REF` lists:
#                an exact `name:tag` or a glob such as `name-*`.
engine="${ENGINE:-docker}"
sweep=false
images=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep) sweep=true; shift ;;
    --image) images+=("${2:-}"); shift $(($# > 1 ? 2 : 1)) ;;
    *) break ;;
  esac
done

for project in "$@"; do
  "$engine" compose -p "$project" down || true
  "$engine" rm -f "${project}-agent-record" || true
  "$engine" rm -f "${project}-agent-record-clickhouse" || true
  prefixes=("${project}_")
  if [[ $sweep == true ]]; then
    "$engine" ps -aq --filter "name=^${project}-" | xargs -r "$engine" rm -f || true
    prefixes+=("${project}-")
  fi
  for prefix in "${prefixes[@]}"; do
    "$engine" volume ls -q --filter "name=^${prefix}" \
      | xargs -r "$engine" volume rm -f || true
  done
done

# Last: a surviving container pins its image.
for image in "${images[@]}"; do
  [[ -n $image ]] || continue
  "$engine" images --format '{{.Repository}}:{{.Tag}}' "$image" \
    | xargs -r "$engine" rmi -f || true
done
exit 0
