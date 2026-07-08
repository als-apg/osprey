#!/usr/bin/env bash
# Launch the Virtual Accelerator container for tutorial/interactive use.
#
# Usage:
#   scripts/va/run_va.sh [DATA_DIR]
#
# DATA_DIR is your project's data/simulation/ DIRECTORY (never a single
# file -- see docker/virtual-accelerator/README.md for why). Defaults to the
# packaged control_assistant preset's own copy, so this runs with zero
# arguments out of the box; point it at a real project's data/simulation to
# use that project's channel_limits.json-scoped scenarios instead.
#
# Builds the image if it doesn't already exist (set OSPREY_VA_REBUILD=1 to
# force a rebuild, e.g. after editing docker/virtual-accelerator/**). Runs in
# the foreground -- Ctrl-C (or `docker stop`) shuts the IOC down cleanly.
#
# After it reports ready, point a project at it with:
#   control_system:
#     type: virtual_accelerator
#     facility: simulation   # the "Local Simulation" gateway preset, localhost:5064
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VA_DIR="${WORKTREE_ROOT}/docker/virtual-accelerator"
IMAGE="osprey-va-full:latest"
CONTAINER="osprey-va-tutorial"
CA_PORT="5064"

DEFAULT_DATA_DIR="${WORKTREE_ROOT}/src/osprey/templates/apps/control_assistant/data/simulation"
DATA_DIR_ARG="${1:-${DEFAULT_DATA_DIR}}"
if [[ ! -d "${DATA_DIR_ARG}" ]]; then
    echo "FATAL: ${DATA_DIR_ARG} is not a directory." >&2
    echo "DATA_DIR must be a project's data/simulation/ directory." >&2
    exit 1
fi
DATA_DIR="$(cd "${DATA_DIR_ARG}" && pwd)"
if [[ ! -f "${DATA_DIR}/machine.json" ]]; then
    echo "FATAL: no machine.json under ${DATA_DIR}" >&2
    echo "DATA_DIR must be a project's data/simulation/ directory." >&2
    exit 1
fi

RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1 && \
       podman run --rm --name "osprey-va-tutorial-healthcheck-$$" busybox:latest true >/dev/null 2>&1; then
        RUNTIME="podman"
    elif command -v docker >/dev/null 2>&1; then
        RUNTIME="docker"
    else
        echo "FATAL: neither a working podman nor docker found on PATH" >&2
        exit 1
    fi
fi
echo "Using container runtime: ${RUNTIME}"

# Single cleanup/trap for both the staging dir (only created on a rebuild --
# see below) and the container, so neither leaks regardless of which path
# runs or where the script exits.
STAGING_DIR=""
cleanup() {
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    [[ -n "${STAGING_DIR}" ]] && rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT INT TERM

if [[ "${OSPREY_VA_REBUILD:-0}" == "1" ]] || ! "${RUNTIME}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-va-full-build.XXXXXX")"

    echo "--- Staging minimal build context at ${STAGING_DIR} ---"
    cp "${WORKTREE_ROOT}/pyproject.toml" "${WORKTREE_ROOT}/README.md" "${STAGING_DIR}/"
    mkdir -p "${STAGING_DIR}/src" "${STAGING_DIR}/docker"
    cp -R "${WORKTREE_ROOT}/src/." "${STAGING_DIR}/src/"
    cp -R "${VA_DIR}" "${STAGING_DIR}/docker/virtual-accelerator"
    find "${STAGING_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

    echo "--- Building ${IMAGE} (linux/amd64) ---"
    "${RUNTIME}" build --platform linux/amd64 --build-arg TARGET_PLATFORM=linux/amd64 \
        -t "${IMAGE}" -f "${STAGING_DIR}/docker/virtual-accelerator/Containerfile" "${STAGING_DIR}"
else
    echo "Reusing existing image ${IMAGE} (set OSPREY_VA_REBUILD=1 to force a rebuild)"
fi

"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "--- Serving CA on localhost:${CA_PORT} (name-server mode); data dir: ${DATA_DIR} ---"
echo "--- On the host, connect with: ---"
echo "    export EPICS_CA_NAME_SERVERS=localhost:${CA_PORT}"
echo "    export EPICS_CA_AUTO_ADDR_LIST=NO"
echo "--- Ctrl-C stops the container. ---"

"${RUNTIME}" run --rm --name "${CONTAINER}" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -v "${DATA_DIR}:/data/simulation:ro" \
    "${IMAGE}"
