#!/usr/bin/env bash
# Build the images a full benchmark run needs and never builds itself, and record
# them against this checkout's revision. A full cell refuses to start until they
# match (check_e2e_coverage.py --check-board-images). The build commands are the
# "board_images" entries of matrix_e2e_config.json.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
CONFIG="$REPO/scripts/benchmark/matrix_e2e_config.json"
cd "$REPO"

"$PY" -c 'import json, sys; [print(e["build"]) for e in json.load(open(sys.argv[1]))["board_images"]]' "$CONFIG" |
  while IFS= read -r build; do
    echo ">> $build" >&2
    eval "$build" </dev/null
  done

"$PY" scripts/benchmark/check_e2e_coverage.py --repo-root "$REPO" --config "$CONFIG" \
  --record-board-images
echo ">> board images recorded" >&2
